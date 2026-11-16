"""Normalization Canonicalizer Pass.

Fuses low-level normalization expansions into canonical ops:
- GroupNorm (Reshape -> InstanceNorm -> Reshape -> Mul -> Add) -> GroupNormalization
- RMSNorm (ReduceL2 -> Clip -> Expand -> Div -> Mul -> Mul OR Pow(2)->ReduceMean->Sqrt->Div->Mul) -> RMSNormalization
- LayerNorm (Mean/Var expansion) -> LayerNormalization
- SkipNorms (Add residual + Norm) -> SkipSimplifiedLayerNormalization / SkipLayerNormalization
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper

from neuron_toolkit.compiler.base import CompilerContext, Pass

log = logging.getLogger("neuron_toolkit.compiler.norm_canonicalizer")


class NormCanonicalizer(Pass):
    """Canonicalizes normalization layers into standard ONNX or custom FunctionProtos."""

    @property
    def name(self) -> str:
        return "norm_canonicalizer"

    @property
    def category(self) -> str:
        return "canonicalization"

    def run(self, model: onnx.ModelProto, context: CompilerContext) -> onnx.ModelProto:
        nodes_before = len(model.graph.node)

        if context.should_run("groupnorm"):
            model = self.apply_groupnorm_fusion(model, mode=context.mode)
        if context.should_run("layernorm"):
            model = self.apply_layernorm_fusion(model, mode=context.mode)
        if context.should_run("rmsnorm"):
            model = self.apply_rmsnorm_fusion(model, mode=context.mode)
        if context.should_run("skip_norm") or context.should_run("skip_rmsnorm") or context.should_run("skip_layernorm"):
            model = self.apply_skip_norm_fusion(model, mode=context.mode)

        nodes_after = len(model.graph.node)
        context.record_pass(self.name, nodes_before, nodes_after)
        return model

    # -------------------------------------------------------------------------
    # GroupNorm Fusion
    # -------------------------------------------------------------------------
    def apply_groupnorm_fusion(self, model: onnx.ModelProto, mode: str = "function_proto") -> onnx.ModelProto:
        """Fuses 9-node GroupNorm pattern into unified GroupNorm op."""
        graph = model.graph
        node_by_output: dict[str, onnx.NodeProto] = {
            o: n for n in graph.node for o in n.output
        }
        consumer_map: dict[str, list[onnx.NodeProto]] = {}
        for n in graph.node:
            for inp in n.input:
                consumer_map.setdefault(inp, []).append(n)

        inits_dict = {
            i.name: numpy_helper.to_array(i) for i in graph.initializer
        }
        nodes_to_remove: set[str] = set()
        new_gn_nodes: list[onnx.NodeProto] = []
        gn_count = 0

        for inn in graph.node:
            if inn.op_type != "InstanceNormalization":
                continue
            reshape_expand = node_by_output.get(inn.input[0])
            if not (reshape_expand and reshape_expand.op_type == "Reshape"):
                continue
            in_cons = consumer_map.get(inn.output[0], [])
            if len(in_cons) != 1 or in_cons[0].op_type != "Reshape":
                continue
            reshape_restore = in_cons[0]
            r_cons = consumer_map.get(reshape_restore.output[0], [])
            if len(r_cons) != 1 or r_cons[0].op_type != "Mul":
                continue
            mul_node = r_cons[0]
            m_cons = consumer_map.get(mul_node.output[0], [])
            if len(m_cons) != 1 or m_cons[0].op_type != "Add":
                continue
            add_node = m_cons[0]

            x_3d = reshape_expand.input[0]
            final_out = add_node.output[0]
            scale_name = [
                i for i in mul_node.input if i != reshape_restore.output[0]
            ][0]
            bias_name = [
                i for i in add_node.input if i != mul_node.output[0]
            ][0]

            # Convert scale and bias inits to 1D
            if scale_name in inits_dict:
                scale_1d = inits_dict[scale_name].flatten()
                scale_1d_name = f"{scale_name}_1d"
                graph.initializer.append(
                    numpy_helper.from_array(scale_1d, name=scale_1d_name)
                )
            else:
                scale_1d_name = scale_name

            if bias_name in inits_dict:
                bias_1d = inits_dict[bias_name].flatten()
                bias_1d_name = f"{bias_name}_1d"
                graph.initializer.append(
                    numpy_helper.from_array(bias_1d, name=bias_1d_name)
                )
            else:
                bias_1d_name = bias_name

            # Extract num_groups from reshape_expand shape
            num_groups = 1
            shape_inp_name = reshape_expand.input[1]
            if shape_inp_name in inits_dict:
                sh_arr = inits_dict[shape_inp_name]
                if len(sh_arr) >= 2 and sh_arr[1] > 0:
                    num_groups = int(sh_arr[1])
            elif shape_inp_name in node_by_output:
                sh_node = node_by_output[shape_inp_name]
                if sh_node.op_type == "Constant":
                    for a in sh_node.attribute:
                        if a.name == "value":
                            sh_arr = numpy_helper.to_array(a.t)
                            if len(sh_arr) >= 2 and sh_arr[1] > 0:
                                num_groups = int(sh_arr[1])

            if mode == "modern_onnx":
                gn_node = helper.make_node(
                    "GroupNormalization",
                    inputs=[x_3d, scale_1d_name, bias_1d_name],
                    outputs=[final_out],
                    name=f"groupnorm_{gn_count}",
                    num_groups=num_groups,
                    epsilon=1e-5,
                    doc_string="Standard ONNX GroupNormalization op",
                )
            else:
                gn_node = helper.make_node(
                    "GroupNorm",
                    inputs=[x_3d, scale_1d_name, bias_1d_name],
                    outputs=[final_out],
                    name=f"groupnorm_{gn_count}",
                    domain="custom.pytorch",
                    num_groups=num_groups,
                    epsilon=1e-5,
                    doc_string="Fused 1D GroupNorm / InstanceNorm adapter",
                )
            new_gn_nodes.append(gn_node)
            gn_count += 1

            for n in [reshape_expand, inn, reshape_restore, mul_node, add_node]:
                nodes_to_remove.add(n.name or n.output[0])

            shape_in = reshape_restore.input[1]
            if shape_in in node_by_output and node_by_output[shape_in].op_type == "Shape":
                nodes_to_remove.add(node_by_output[shape_in].name or shape_in)
            for c_inp in inn.input[1:]:
                if c_inp in node_by_output and node_by_output[c_inp].op_type == "Constant":
                    nodes_to_remove.add(node_by_output[c_inp].name or c_inp)
            exp_c = reshape_expand.input[1]
            if exp_c in node_by_output and node_by_output[exp_c].op_type == "Constant":
                nodes_to_remove.add(node_by_output[exp_c].name or exp_c)

        if not new_gn_nodes:
            return model

        kept = [
            n
            for n in graph.node
            if (n.name or (n.output[0] if n.output else "")) not in nodes_to_remove
        ]
        kept.extend(new_gn_nodes)
        del graph.node[:]
        graph.node.extend(kept)

        custom_domain = "custom.pytorch"
        if not any(op.domain == custom_domain for op in model.opset_import):
            model.opset_import.append(helper.make_opsetid(custom_domain, 1))

        if mode == "function_proto":
            self._attach_groupnorm_function_def(model)

        log.info("GroupNorm Fusion: Fused %d GroupNorm instances", gn_count)
        return model

    def _attach_groupnorm_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for GroupNorm."""
        func_name = "GroupNorm"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", 14)
        in_node = helper.make_node(
            "InstanceNormalization",
            inputs=["x", "scale", "bias"],
            outputs=["y"],
            epsilon=1e-5,
        )
        gn_func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["x", "scale", "bias"],
            outputs=["y"],
            nodes=[in_node],
            opset_imports=[opset_standard],
            doc_string="GroupNorm normalization function",
        )
        model.functions.append(gn_func)

    # -------------------------------------------------------------------------
    # RMSNorm Fusion (ReduceL2 and MeanSquare formulations)
    # -------------------------------------------------------------------------
    def apply_rmsnorm_fusion(self, model: onnx.ModelProto, mode: str = "function_proto") -> onnx.ModelProto:
        """Fuse RMSNorm expansions (ReduceL2 or MeanSq) into RMSNorm."""
        graph = model.graph
        node_by_output: dict[str, onnx.NodeProto] = {
            out: n for n in graph.node for out in n.output
        }
        nodes_to_remove: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        rms_count = 0

        # Pattern 1: ReduceL2 formulation
        for n in graph.node:
            if n.op_type != "ReduceL2":
                continue
            x = n.input[0]
            clips = [c for c in graph.node if n.output[0] in c.input and c.op_type == "Clip"]
            if not clips:
                continue
            clip = clips[0]
            expands = [c for c in graph.node if clip.output[0] in c.input and c.op_type == "Expand"]
            expand = expands[0] if expands else None
            div_input = expand.output[0] if expand is not None else clip.output[0]
            divs = [c for c in graph.node if div_input in c.input and c.op_type == "Div"]
            if not divs:
                continue
            div = divs[0]
            muls1 = [c for c in graph.node if div.output[0] in c.input and c.op_type == "Mul"]
            if not muls1:
                continue
            mul1 = muls1[0]
            muls2 = [c for c in graph.node if mul1.output[0] in c.input and c.op_type == "Mul"]
            if not muls2:
                continue
            mul2 = muls2[0]

            gamma = mul2.input[1] if mul2.input[0] == mul1.output[0] else mul2.input[0]
            nodes_to_remove.update({n.name, clip.name, div.name, mul1.name, mul2.name})
            if expand is not None:
                nodes_to_remove.add(expand.name)

            # Check if expand's shape input is an isolated Shape node
            if expand is not None and len(expand.input) > 1 and expand.input[1] in node_by_output:
                sh_node = node_by_output[expand.input[1]]
                if sh_node.op_type == "Shape":
                    sh_consumers = [c for c in graph.node if sh_node.output[0] in c.input]
                    if len(sh_consumers) == 1:
                        nodes_to_remove.add(sh_node.name)

            if mode == "modern_onnx":
                rmsnorm_node = helper.make_node(
                    op_type="RMSNormalization",
                    inputs=[x, gamma],
                    outputs=[mul2.output[0]],
                    name=f"{n.name}_fused_rmsnorm",
                    axis=-1,
                    epsilon=1e-12,
                    doc_string="Standard ONNX RMSNormalization op",
                )
            else:
                rmsnorm_node = helper.make_node(
                    op_type="RMSNorm",
                    inputs=[x, gamma],
                    outputs=[mul2.output[0]],
                    name=f"{n.name}_fused_rmsnorm",
                    domain="custom.pytorch",
                    epsilon=1e-12,
                    doc_string="Fused RMSNorm op",
                )
            new_nodes.append(rmsnorm_node)
            rms_count += 1

        # Pattern 2: Pow(2) -> ReduceMean -> Add(eps) -> Sqrt -> Div(x, ...) -> Mul(..., gamma)
        for pow_node in graph.node:
            if pow_node.op_type != "Pow" or (pow_node.name in nodes_to_remove):
                continue
            x = pow_node.input[0]
            # Guard: If x is produced by Sub(inp, mean), this is LayerNorm variance, not RMSNorm
            if x in node_by_output and node_by_output[x].op_type == "Sub":
                sub_node = node_by_output[x]
                if any(inp in node_by_output and node_by_output[inp].op_type == "ReduceMean" for inp in sub_node.input):
                    continue
            rm_candidates = [c for c in graph.node if pow_node.output[0] in c.input and c.op_type == "ReduceMean"]
            if not rm_candidates:
                continue
            rm_node = rm_candidates[0]
            add_candidates = [c for c in graph.node if rm_node.output[0] in c.input and c.op_type == "Add"]
            if not add_candidates:
                continue
            add_node = add_candidates[0]
            sqrt_candidates = [c for c in graph.node if add_node.output[0] in c.input and c.op_type == "Sqrt"]
            if not sqrt_candidates:
                continue
            sqrt_node = sqrt_candidates[0]
            div_candidates = [c for c in graph.node if sqrt_node.output[0] in c.input and c.op_type == "Div" and c.input[0] == x]
            if not div_candidates:
                continue
            div_node = div_candidates[0]
            mul_candidates = [c for c in graph.node if div_node.output[0] in c.input and c.op_type == "Mul"]
            if not mul_candidates:
                continue
            mul_node = mul_candidates[0]
            gamma = mul_node.input[1] if mul_node.input[0] == div_node.output[0] else mul_node.input[0]

            nodes_to_remove.update({pow_node.name, rm_node.name, add_node.name, sqrt_node.name, div_node.name, mul_node.name})
            if mode == "modern_onnx":
                rmsnorm_node = helper.make_node(
                    op_type="RMSNormalization",
                    inputs=[x, gamma],
                    outputs=[mul_node.output[0]],
                    name=f"rmsnorm_meansq_{rms_count}",
                    axis=-1,
                    epsilon=1e-5,
                )
            else:
                rmsnorm_node = helper.make_node(
                    op_type="RMSNorm",
                    inputs=[x, gamma],
                    outputs=[mul_node.output[0]],
                    name=f"rmsnorm_meansq_{rms_count}",
                    domain="custom.pytorch",
                    epsilon=1e-5,
                )
            new_nodes.append(rmsnorm_node)
            rms_count += 1

        if not new_nodes:
            return model

        surviving = [n for n in graph.node if n.name not in nodes_to_remove] + new_nodes
        del graph.node[:]
        graph.node.extend(surviving)

        custom_domain = "custom.pytorch"
        if not any(op.domain == custom_domain for op in model.opset_import):
            model.opset_import.append(helper.make_opsetid(custom_domain, 1))

        if mode == "function_proto":
            self._attach_rmsnorm_function_def(model)

        log.info("RMSNorm Fusion: Fused %d RMSNorm instances", rms_count)
        return model

    def _attach_rmsnorm_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for RMSNorm."""
        func_name = "RMSNorm"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", 14)
        r_node = helper.make_node("ReduceL2", inputs=["x"], outputs=["norm"], axes=[-1], keepdims=1)
        const_eps = helper.make_node("Constant", inputs=[], outputs=["eps"], value=helper.make_tensor("eps", onnx.TensorProto.FLOAT, [], [1e-12]))
        c_node = helper.make_node("Clip", inputs=["norm", "eps"], outputs=["clipped_norm"])
        d_node = helper.make_node("Div", inputs=["x", "clipped_norm"], outputs=["normed_x"])
        m_node = helper.make_node("Mul", inputs=["normed_x", "gamma"], outputs=["y"])
        rms_func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["x", "gamma"],
            outputs=["y"],
            nodes=[r_node, const_eps, c_node, d_node, m_node],
            opset_imports=[opset_standard],
            doc_string="RMSNorm root mean square normalization function",
        )
        model.functions.append(rms_func)

    # -------------------------------------------------------------------------
    # LayerNorm Mean/Var Fusion
    # -------------------------------------------------------------------------
    def apply_layernorm_fusion(self, model: onnx.ModelProto, mode: str = "function_proto") -> onnx.ModelProto:
        """Fuses ReduceMean -> Sub -> Pow(2) -> ReduceMean -> Sqrt -> Div -> Mul -> Add into LayerNormalization."""
        graph = model.graph
        consumer_map: dict[str, list[onnx.NodeProto]] = {}
        for n in graph.node:
            for inp in n.input:
                consumer_map.setdefault(inp, []).append(n)

        nodes_to_remove: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        count = 0

        for rm_mean in graph.node:
            if rm_mean.op_type != "ReduceMean":
                continue
            x = rm_mean.input[0]
            mean_out = rm_mean.output[0]

            sub_candidates = [c for c in consumer_map.get(mean_out, []) if c.op_type == "Sub" and c.input[0] == x]
            if not sub_candidates:
                continue
            sub_node = sub_candidates[0]
            x_minus_mean = sub_node.output[0]

            pow_candidates = [
                c for c in consumer_map.get(x_minus_mean, [])
                if c.op_type == "Pow"
                or (c.op_type == "Mul" and len(c.input) == 2 and c.input[0] == x_minus_mean and c.input[1] == x_minus_mean)
            ]
            if not pow_candidates:
                continue
            pow_node = pow_candidates[0]

            rm_var_candidates = [c for c in consumer_map.get(pow_node.output[0], []) if c.op_type == "ReduceMean"]
            if not rm_var_candidates:
                continue
            rm_var_node = rm_var_candidates[0]

            add_candidates = [c for c in consumer_map.get(rm_var_node.output[0], []) if c.op_type == "Add"]
            if not add_candidates:
                continue
            add_node = add_candidates[0]

            sqrt_candidates = [c for c in consumer_map.get(add_node.output[0], []) if c.op_type == "Sqrt"]
            if not sqrt_candidates:
                continue
            sqrt_node = sqrt_candidates[0]

            div_candidates = [c for c in consumer_map.get(sqrt_node.output[0], []) if c.op_type == "Div" and c.input[0] == x_minus_mean]
            if not div_candidates:
                continue
            div_node = div_candidates[0]

            mul_candidates = [c for c in consumer_map.get(div_node.output[0], []) if c.op_type == "Mul"]
            if not mul_candidates:
                continue
            mul_node = mul_candidates[0]
            scale = mul_node.input[1] if mul_node.input[0] == div_node.output[0] else mul_node.input[0]

            add_bias_candidates = [c for c in consumer_map.get(mul_node.output[0], []) if c.op_type == "Add"]
            if not add_bias_candidates:
                continue
            add_bias_node = add_bias_candidates[0]
            bias = add_bias_node.input[1] if add_bias_node.input[0] == mul_node.output[0] else add_bias_node.input[0]
            final_out = add_bias_node.output[0]

            nodes_to_remove.update({
                rm_mean.name or rm_mean.output[0],
                sub_node.name or sub_node.output[0],
                pow_node.name or pow_node.output[0],
                rm_var_node.name or rm_var_node.output[0],
                add_node.name or add_node.output[0],
                sqrt_node.name or sqrt_node.output[0],
                div_node.name or div_node.output[0],
                mul_node.name or mul_node.output[0],
                add_bias_node.name or add_bias_node.output[0],
            })

            ln_node = helper.make_node(
                "LayerNormalization",
                inputs=[x, scale, bias],
                outputs=[final_out],
                name=f"layernorm_{count}",
                axis=-1,
                epsilon=1e-5,
            )
            new_nodes.append(ln_node)
            count += 1

        if not new_nodes:
            return model

        kept = [n for n in graph.node if (n.name or (n.output[0] if n.output else "")) not in nodes_to_remove]
        kept.extend(new_nodes)
        del graph.node[:]
        graph.node.extend(kept)

        req_domain = "com.microsoft" if mode == "modern_onnx" else "custom.pytorch"
        if not any(op.domain == req_domain for op in model.opset_import):
            model.opset_import.append(helper.make_opsetid(req_domain, 1))

        return model

    # -------------------------------------------------------------------------
    # SkipNorms (Add + Norm) Fusion
    # -------------------------------------------------------------------------
    def apply_skip_norm_fusion(self, model: onnx.ModelProto, mode: str = "function_proto") -> onnx.ModelProto:
        """Fuses Add(x, residual) followed by RMSNorm / LayerNorm into fused SkipNorm."""
        graph = model.graph
        node_by_output = {out: n for n in graph.node for out in n.output}
        consumer_map: dict[str, list[onnx.NodeProto]] = {}
        for n in graph.node:
            for inp in n.input:
                consumer_map.setdefault(inp, []).append(n)
        graph_outputs = {o.name for o in graph.output}

        nodes_to_remove: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        count = 0

        for n in graph.node:
            if n.op_type in ("RMSNorm", "RMSNormalization"):
                x_in = n.input[0]
                if x_in in node_by_output and node_by_output[x_in].op_type == "Add":
                    add_node = node_by_output[x_in]
                    # Only fuse if Add is consumed solely by this norm and not an output
                    if len(consumer_map.get(add_node.output[0], [])) == 1 and add_node.output[0] not in graph_outputs:
                        res0, res1 = add_node.input[0], add_node.input[1]
                        gamma = n.input[1]
                        out = n.output[0]
                        nodes_to_remove.update({add_node.name or add_node.output[0], n.name or n.output[0]})
                        skip_node = helper.make_node(
                            "SkipSimplifiedLayerNormalization",
                            inputs=[res0, res1, gamma],
                            outputs=[out],
                            name=f"skip_rmsnorm_{count}",
                            domain="com.microsoft" if mode == "modern_onnx" else "custom.pytorch",
                        )
                        new_nodes.append(skip_node)
                        count += 1
            elif n.op_type in ("LayerNorm", "LayerNormalization"):
                x_in = n.input[0]
                if x_in in node_by_output and node_by_output[x_in].op_type == "Add":
                    add_node = node_by_output[x_in]
                    if len(consumer_map.get(add_node.output[0], [])) == 1 and add_node.output[0] not in graph_outputs:
                        res0, res1 = add_node.input[0], add_node.input[1]
                        scale = n.input[1]
                        bias = n.input[2] if len(n.input) > 2 else ""
                        out = n.output[0]
                        nodes_to_remove.update({add_node.name or add_node.output[0], n.name or n.output[0]})
                        skip_node = helper.make_node(
                            "SkipLayerNormalization",
                            inputs=[res0, res1, scale, bias] if bias else [res0, res1, scale],
                            outputs=[out],
                            name=f"skip_layernorm_{count}",
                            domain="com.microsoft" if mode == "modern_onnx" else "custom.pytorch",
                        )
                        new_nodes.append(skip_node)
                        count += 1

        if not new_nodes:
            return model

        kept = [n for n in graph.node if (n.name or (n.output[0] if n.output else "")) not in nodes_to_remove]
        kept.extend(new_nodes)
        del graph.node[:]
        graph.node.extend(kept)

        req_domain = "com.microsoft" if mode == "modern_onnx" else "custom.pytorch"
        if not any(op.domain == req_domain for op in model.opset_import):
            model.opset_import.append(helper.make_opsetid(req_domain, 1))

        return model
