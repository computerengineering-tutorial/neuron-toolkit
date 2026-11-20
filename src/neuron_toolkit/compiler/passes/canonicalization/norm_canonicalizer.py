"""Normalization Canonicalizer Pass.

Fuses low-level normalization expansions into canonical ops:
- GroupNorm (Reshape -> InstanceNorm -> Reshape -> Mul -> Add) -> GroupNormalization
- RMSNorm (ReduceL2 -> Clip -> Expand -> Div -> Mul -> Mul OR Pow(2)->ReduceMean->Sqrt->Div->Mul) -> RMSNormalization
- LayerNorm (Mean/Var expansion) -> LayerNormalization
- SkipNorms (Add residual + Norm) -> SkipSimplifiedLayerNormalization / SkipLayerNormalization
"""

from __future__ import annotations

import logging

import numpy as np
import onnx
from onnx import helper, numpy_helper

from neuron_toolkit.compiler.base import CompilerContext, Pass, LATEST_OPSET

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
        if (
            context.should_run("skip_norm")
            or context.should_run("skip_rmsnorm")
            or context.should_run("skip_layernorm")
        ):
            model = self.apply_skip_norm_fusion(model, mode=context.mode)

        nodes_after = len(model.graph.node)
        context.record_pass(self.name, nodes_before, nodes_after)
        return model

    # -------------------------------------------------------------------------
    # GroupNorm Fusion
    # -------------------------------------------------------------------------
    def apply_groupnorm_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuses 9-node GroupNorm pattern into unified GroupNorm op."""
        graph = model.graph
        node_by_output: dict[str, onnx.NodeProto] = {
            o: n for n in graph.node for o in n.output
        }
        consumer_map: dict[str, list[onnx.NodeProto]] = {}
        for n in graph.node:
            for inp in n.input:
                consumer_map.setdefault(inp, []).append(n)

        inits_dict = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
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
            scale_name = [i for i in mul_node.input if i != reshape_restore.output[0]][
                0
            ]
            bias_name = [i for i in add_node.input if i != mul_node.output[0]][0]

            # Convert scale and bias inits to 1D
            if scale_name in inits_dict:
                scale_1d_name = f"{scale_name}_1d"
                if scale_1d_name not in inits_dict:
                    scale_1d = inits_dict[scale_name].flatten()
                    inits_dict[scale_1d_name] = scale_1d
                    graph.initializer.append(
                        numpy_helper.from_array(scale_1d, name=scale_1d_name)
                    )
            else:
                scale_1d_name = scale_name

            if bias_name in inits_dict:
                bias_1d_name = f"{bias_name}_1d"
                if bias_1d_name not in inits_dict:
                    bias_1d = inits_dict[bias_name].flatten()
                    inits_dict[bias_1d_name] = bias_1d
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
            if (
                shape_in in node_by_output
                and node_by_output[shape_in].op_type == "Shape"
            ):
                nodes_to_remove.add(node_by_output[shape_in].name or shape_in)
            for c_inp in inn.input[1:]:
                if (
                    c_inp in node_by_output
                    and node_by_output[c_inp].op_type == "Constant"
                ):
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

        if mode == "modern_onnx":
            for op in model.opset_import:
                if op.domain in ("", "ai.onnx") and op.version < 21:
                    op.version = 21
        else:
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
        opset_standard = helper.make_opsetid("", LATEST_OPSET)
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
    def apply_rmsnorm_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
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
            clips = [
                c for c in graph.node if n.output[0] in c.input and c.op_type == "Clip"
            ]
            if not clips:
                continue
            clip = clips[0]
            expands = [
                c
                for c in graph.node
                if clip.output[0] in c.input and c.op_type == "Expand"
            ]
            expand = expands[0] if expands else None
            div_input = expand.output[0] if expand is not None else clip.output[0]
            divs = [
                c for c in graph.node if div_input in c.input and c.op_type == "Div"
            ]
            if not divs:
                continue
            div = divs[0]
            muls1 = [
                c for c in graph.node if div.output[0] in c.input and c.op_type == "Mul"
            ]
            if not muls1:
                continue
            mul1 = muls1[0]
            muls2 = [
                c
                for c in graph.node
                if mul1.output[0] in c.input and c.op_type == "Mul"
            ]
            if not muls2:
                continue
            mul2 = muls2[0]

            gamma = mul2.input[1] if mul2.input[0] == mul1.output[0] else mul2.input[0]
            nodes_to_remove.update({n.name, clip.name, div.name, mul1.name, mul2.name})
            if expand is not None:
                nodes_to_remove.add(expand.name)

            # Check if expand's shape input is an isolated Shape node
            if (
                expand is not None
                and len(expand.input) > 1
                and expand.input[1] in node_by_output
            ):
                sh_node = node_by_output[expand.input[1]]
                if sh_node.op_type == "Shape":
                    sh_consumers = [
                        c for c in graph.node if sh_node.output[0] in c.input
                    ]
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
                if any(
                    inp in node_by_output
                    and node_by_output[inp].op_type == "ReduceMean"
                    for inp in sub_node.input
                ):
                    continue
            rm_candidates = [
                c
                for c in graph.node
                if pow_node.output[0] in c.input and c.op_type == "ReduceMean"
            ]
            if not rm_candidates:
                continue
            rm_node = rm_candidates[0]
            add_candidates = [
                c
                for c in graph.node
                if rm_node.output[0] in c.input and c.op_type == "Add"
            ]
            if not add_candidates:
                continue
            add_node = add_candidates[0]
            sqrt_candidates = [
                c
                for c in graph.node
                if add_node.output[0] in c.input and c.op_type == "Sqrt"
            ]
            if not sqrt_candidates:
                continue
            sqrt_node = sqrt_candidates[0]
            div_candidates = [
                c
                for c in graph.node
                if sqrt_node.output[0] in c.input
                and c.op_type == "Div"
                and c.input[0] == x
            ]
            if div_candidates:
                div_node = div_candidates[0]
                mul_candidates = [
                    c
                    for c in graph.node
                    if div_node.output[0] in c.input and c.op_type == "Mul"
                ]
                if not mul_candidates:
                    continue
                mul_node = mul_candidates[0]
                gamma = (
                    mul_node.input[1]
                    if mul_node.input[0] == div_node.output[0]
                    else mul_node.input[0]
                )
                nodes_to_remove.update(
                    {
                        pow_node.name,
                        rm_node.name,
                        add_node.name,
                        sqrt_node.name,
                        div_node.name,
                        mul_node.name,
                    }
                )
            else:
                # Reciprocal variant: Sqrt -> Reciprocal -> Mul(x, rsqrt) -> Mul(..., gamma)
                recip_candidates = [
                    c
                    for c in graph.node
                    if sqrt_node.output[0] in c.input and c.op_type == "Reciprocal"
                ]
                if not recip_candidates:
                    continue
                recip_node = recip_candidates[0]
                mul1_candidates = [
                    c
                    for c in graph.node
                    if recip_node.output[0] in c.input
                    and c.op_type == "Mul"
                    and x in c.input
                ]
                if not mul1_candidates:
                    continue
                mul1_node = mul1_candidates[0]
                mul2_candidates = [
                    c
                    for c in graph.node
                    if mul1_node.output[0] in c.input and c.op_type == "Mul"
                ]
                if not mul2_candidates:
                    continue
                mul_node = mul2_candidates[0]
                gamma = (
                    mul_node.input[1]
                    if mul_node.input[0] == mul1_node.output[0]
                    else mul_node.input[0]
                )
                nodes_to_remove.update(
                    {
                        pow_node.name,
                        rm_node.name,
                        add_node.name,
                        sqrt_node.name,
                        recip_node.name,
                        mul1_node.name,
                        mul_node.name,
                    }
                )
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

        if mode == "modern_onnx":
            for op in model.opset_import:
                if op.domain in ("", "ai.onnx") and op.version < 23:
                    op.version = 23
        else:
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
        opset_standard = helper.make_opsetid("", LATEST_OPSET)
        r_node = helper.make_node(
            "ReduceL2", inputs=["x"], outputs=["norm"], axes=[-1], keepdims=1
        )
        const_eps = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["eps"],
            value=helper.make_tensor("eps", onnx.TensorProto.FLOAT, [], [1e-12]),
        )
        c_node = helper.make_node(
            "Clip", inputs=["norm", "eps"], outputs=["clipped_norm"]
        )
        d_node = helper.make_node(
            "Div", inputs=["x", "clipped_norm"], outputs=["normed_x"]
        )
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
    def apply_layernorm_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuses ReduceMean -> Sub -> Pow(2) -> ReduceMean -> Sqrt -> Div -> Mul -> Add into LayerNormalization."""
        graph = model.graph
        use_standard_ln = True
        ln_domain = ""

        init_map: dict[str, np.ndarray] = {
            i.name: numpy_helper.to_array(i) for i in graph.initializer
        }
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

            sub_candidates = [
                c
                for c in consumer_map.get(mean_out, [])
                if c.op_type == "Sub" and c.input[0] == x
            ]
            if not sub_candidates:
                continue

            # Identify sub_for_var (feeding Pow or Mul(diff, diff))
            sub_for_var = None
            pow_node = None
            for sc in sub_candidates:
                pow_cands = [
                    c
                    for c in consumer_map.get(sc.output[0], [])
                    if c.op_type == "Pow"
                    or (
                        c.op_type == "Mul"
                        and len(c.input) == 2
                        and c.input[0] == sc.output[0]
                        and c.input[1] == sc.output[0]
                    )
                ]
                if pow_cands:
                    sub_for_var = sc
                    pow_node = pow_cands[0]
                    break

            if not sub_for_var or not pow_node:
                continue

            rm_var_candidates = [
                c
                for c in consumer_map.get(pow_node.output[0], [])
                if c.op_type == "ReduceMean"
            ]
            if not rm_var_candidates:
                continue
            rm_var_node = rm_var_candidates[0]

            add_candidates = [
                c
                for c in consumer_map.get(rm_var_node.output[0], [])
                if c.op_type == "Add"
            ]
            if not add_candidates:
                continue
            add_node = add_candidates[0]

            sqrt_candidates = [
                c
                for c in consumer_map.get(add_node.output[0], [])
                if c.op_type == "Sqrt"
            ]
            if not sqrt_candidates:
                continue
            sqrt_node = sqrt_candidates[0]

            intermediate_nodes: list[onnx.NodeProto] = []
            norm_out_node: onnx.NodeProto | None = None
            sub_for_num: onnx.NodeProto | None = None

            # Path A: Standard Sqrt -> Div
            div_cands = [
                c
                for c in consumer_map.get(sqrt_node.output[0], [])
                if c.op_type == "Div"
                and any(c.input[0] == sc.output[0] for sc in sub_candidates)
            ]
            if div_cands:
                div_node = div_cands[0]
                sub_for_num = next(
                    sc for sc in sub_candidates if sc.output[0] == div_node.input[0]
                )
                norm_out_node = div_node
                intermediate_nodes.append(div_node)
            else:
                # Path B: Decomposed Sqrt -> Reciprocal -> Mul
                recip_cands = [
                    c
                    for c in consumer_map.get(sqrt_node.output[0], [])
                    if c.op_type == "Reciprocal"
                ]
                if recip_cands:
                    recip_node = recip_cands[0]
                    mul_cands = [
                        c
                        for c in consumer_map.get(recip_node.output[0], [])
                        if c.op_type == "Mul"
                        and any(sc.output[0] in c.input for sc in sub_candidates)
                    ]
                    if mul_cands:
                        mul_norm_node = mul_cands[0]
                        sub_for_num = next(
                            sc
                            for sc in sub_candidates
                            if sc.output[0] in mul_norm_node.input
                        )
                        norm_out_node = mul_norm_node
                        intermediate_nodes.extend([recip_node, mul_norm_node])

            if not norm_out_node or not sub_for_num:
                continue

            curr_out = norm_out_node.output[0]
            res_cands = [
                c
                for c in consumer_map.get(curr_out, [])
                if c.op_type == "Reshape"
            ]
            reshape_node: onnx.NodeProto | None = None
            if res_cands and len(consumer_map.get(curr_out, [])) == 1:
                reshape_node = res_cands[0]
                curr_out = reshape_node.output[0]

            mul_candidates = [
                c
                for c in consumer_map.get(curr_out, [])
                if c.op_type == "Mul"
            ]
            if not mul_candidates:
                continue
            mul_node = mul_candidates[0]
            scale = (
                mul_node.input[1]
                if mul_node.input[0] == curr_out
                else mul_node.input[0]
            )

            add_bias_candidates = [
                c
                for c in consumer_map.get(mul_node.output[0], [])
                if c.op_type == "Add"
            ]
            add_bias_node = add_bias_candidates[0] if add_bias_candidates else None
            if add_bias_node:
                bias = (
                    add_bias_node.input[1]
                    if add_bias_node.input[0] == mul_node.output[0]
                    else add_bias_node.input[0]
                )
                final_out = add_bias_node.output[0]
            else:
                bias = None
                final_out = mul_node.output[0]

            eps_name = (
                add_node.input[1]
                if add_node.input[0] == rm_var_node.output[0]
                else add_node.input[0]
            )
            eps_val = (
                float(init_map[eps_name]) if eps_name in init_map else 1e-5
            )

            to_drop = [
                rm_mean,
                sub_for_var,
                sub_for_num,
                pow_node,
                rm_var_node,
                add_node,
                sqrt_node,
                mul_node,
            ] + intermediate_nodes
            if add_bias_node is not None:
                to_drop.append(add_bias_node)

            graph_outputs = {o.name for o in graph.output}
            to_drop_ids = {id(n) for n in to_drop}
            exit_node = add_bias_node if add_bias_node is not None else mul_node
            for n in to_drop:
                if n is exit_node:
                    nodes_to_remove.add(n.name or n.output[0])
                    continue
                has_external = False
                for out in n.output:
                    if out in graph_outputs:
                        has_external = True
                        break
                    consumers = consumer_map.get(out, [])
                    if any(id(c) not in to_drop_ids for c in consumers):
                        has_external = True
                        break
                if not has_external:
                    nodes_to_remove.add(n.name or n.output[0])

            if not use_standard_ln and bias is None:
                scale_arr = init_map.get(scale)
                if scale_arr is not None:
                    zero_b_name = f"{scale}_zero_bias"
                    if zero_b_name not in init_map:
                        zero_arr = np.zeros(scale_arr.shape, dtype=scale_arr.dtype)
                        init_map[zero_b_name] = zero_arr
                        graph.initializer.append(
                            numpy_helper.from_array(zero_arr, name=zero_b_name)
                        )
                    bias = zero_b_name

            ln_inputs = [x, scale, bias] if bias is not None else [x, scale]

            if reshape_node is not None:
                ln_out = f"layernorm_out_{count}"
                ln_node = helper.make_node(
                    "LayerNormalization",
                    inputs=ln_inputs,
                    outputs=[ln_out],
                    name=f"layernorm_{count}",
                    domain=ln_domain,
                    axis=-1,
                    epsilon=eps_val,
                )
                new_nodes.append(ln_node)
                nodes_to_remove.add(reshape_node.name or reshape_node.output[0])
                new_res = helper.make_node(
                    "Reshape",
                    inputs=[ln_out, reshape_node.input[1]],
                    outputs=[final_out],
                    name=f"layernorm_reshape_{count}",
                )
                new_nodes.append(new_res)
            else:
                ln_node = helper.make_node(
                    "LayerNormalization",
                    inputs=ln_inputs,
                    outputs=[final_out],
                    name=f"layernorm_{count}",
                    domain=ln_domain,
                    axis=-1,
                    epsilon=eps_val,
                )
                new_nodes.append(ln_node)

            count += 1

        if not new_nodes:
            return model

        kept = [
            n
            for n in graph.node
            if (n.name or (n.output[0] if n.output else "")) not in nodes_to_remove
        ]
        kept.extend(new_nodes)
        del graph.node[:]
        graph.node.extend(kept)
        from neuron_toolkit.runtime.runner import toposort_graph
        sorted_nodes = toposort_graph(model)
        del graph.node[:]
        graph.node.extend(sorted_nodes)

        if use_standard_ln:
            for op in model.opset_import:
                if op.domain in ("", "ai.onnx") and op.version < 17:
                    op.version = 17
        else:
            req_domain = "custom.pytorch"
            if not any(op.domain == req_domain for op in model.opset_import):
                model.opset_import.append(helper.make_opsetid(req_domain, 1))
            if mode == "function_proto":
                self._attach_layernorm_function_def(model)

        return model

    # -------------------------------------------------------------------------
    # SkipNorms (Add + Norm) Fusion
    # -------------------------------------------------------------------------
    def apply_skip_norm_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
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
                    if (
                        len(consumer_map.get(add_node.output[0], [])) == 1
                        and add_node.output[0] not in graph_outputs
                    ):
                        res0, res1 = add_node.input[0], add_node.input[1]
                        gamma = n.input[1] if len(n.input) > 1 else ""
                        out = n.output[0]
                        nodes_to_remove.update(
                            {add_node.name or add_node.output[0], n.name or n.output[0]}
                        )
                        skip_inputs = [res0, res1, gamma] if gamma else [res0, res1]
                        skip_node = helper.make_node(
                            "SkipSimplifiedLayerNormalization",
                            inputs=skip_inputs,
                            outputs=[out],
                            name=f"skip_rmsnorm_{count}",
                            domain="com.microsoft"
                            if mode == "modern_onnx"
                            else "custom.pytorch",
                        )
                        new_nodes.append(skip_node)
                        count += 1
            elif n.op_type in ("LayerNorm", "LayerNormalization"):
                x_in = n.input[0]
                if x_in in node_by_output and node_by_output[x_in].op_type == "Add":
                    add_node = node_by_output[x_in]
                    if (
                        len(consumer_map.get(add_node.output[0], [])) == 1
                        and add_node.output[0] not in graph_outputs
                    ):
                        res0, res1 = add_node.input[0], add_node.input[1]
                        scale = n.input[1] if len(n.input) > 1 else ""
                        bias = n.input[2] if len(n.input) > 2 else ""
                        out = n.output[0]
                        nodes_to_remove.update(
                            {add_node.name or add_node.output[0], n.name or n.output[0]}
                        )
                        skip_inputs = [res0, res1]
                        if scale:
                            skip_inputs.append(scale)
                            if bias:
                                skip_inputs.append(bias)
                        skip_node = helper.make_node(
                            "SkipLayerNormalization",
                            inputs=skip_inputs,
                            outputs=[out],
                            name=f"skip_layernorm_{count}",
                            domain="com.microsoft"
                            if mode == "modern_onnx"
                            else "custom.pytorch",
                        )
                        new_nodes.append(skip_node)
                        count += 1

        if not new_nodes:
            return model

        kept = [
            n
            for n in graph.node
            if (n.name or (n.output[0] if n.output else "")) not in nodes_to_remove
        ]
        kept.extend(new_nodes)
        del graph.node[:]
        graph.node.extend(kept)
        from neuron_toolkit.runtime.runner import toposort_graph
        sorted_nodes = toposort_graph(model)
        del graph.node[:]
        graph.node.extend(sorted_nodes)

        req_domain = "com.microsoft" if mode == "modern_onnx" else "custom.pytorch"
        if not any(op.domain == req_domain for op in model.opset_import):
            model.opset_import.append(helper.make_opsetid(req_domain, 1))

        if mode == "function_proto":
            self._attach_skip_norm_function_def(model)

        return model

    def _attach_layernorm_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for LayerNormalization."""
        func_name = "LayerNormalization"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", LATEST_OPSET)
        rm_m = helper.make_node(
            "ReduceMean", inputs=["x"], outputs=["mean_out"], axes=[-1], keepdims=1
        )
        sub = helper.make_node("Sub", inputs=["x", "mean_out"], outputs=["sub_out"])
        pow_n = helper.make_node("Pow", inputs=["sub_out", "two"], outputs=["pow_out"])
        rm_v = helper.make_node(
            "ReduceMean", inputs=["pow_out"], outputs=["var_out"], axes=[-1], keepdims=1
        )
        add1 = helper.make_node("Add", inputs=["var_out", "eps"], outputs=["var_eps"])
        sqrt = helper.make_node("Sqrt", inputs=["var_eps"], outputs=["std_out"])
        div = helper.make_node(
            "Div", inputs=["sub_out", "std_out"], outputs=["norm_out"]
        )
        mul = helper.make_node(
            "Mul", inputs=["norm_out", "scale"], outputs=["scaled_out"]
        )
        add2 = helper.make_node("Add", inputs=["scaled_out", "bias"], outputs=["y"])
        two_init = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["two"],
            value=helper.make_tensor("two", onnx.TensorProto.FLOAT, [], [2.0]),
        )
        eps_init = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["eps"],
            value=helper.make_tensor("eps", onnx.TensorProto.FLOAT, [], [1e-5]),
        )
        ln_func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["x", "scale", "bias"],
            outputs=["y"],
            nodes=[
                two_init,
                eps_init,
                rm_m,
                sub,
                pow_n,
                rm_v,
                add1,
                sqrt,
                div,
                mul,
                add2,
            ],
            opset_imports=[opset_standard],
            doc_string="LayerNormalization custom function",
        )
        model.functions.append(ln_func)

    def _attach_skip_norm_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for SkipNorm variants."""
        domain = "custom.pytorch"
        opset_standard = helper.make_opsetid("", LATEST_OPSET)
        opset_custom = helper.make_opsetid("custom.pytorch", 1)

        if not any(
            f.name == "SkipSimplifiedLayerNormalization" and f.domain == domain
            for f in model.functions
        ):
            add_n = helper.make_node("Add", inputs=["x1", "x2"], outputs=["res"])
            norm_n = helper.make_node(
                "RMSNorm", inputs=["res", "gamma"], outputs=["y"], domain=domain
            )
            func_rms = helper.make_function(
                domain=domain,
                fname="SkipSimplifiedLayerNormalization",
                inputs=["x1", "x2", "gamma"],
                outputs=["y"],
                nodes=[add_n, norm_n],
                opset_imports=[opset_standard, opset_custom],
                doc_string="SkipSimplifiedLayerNormalization function",
            )
            model.functions.append(func_rms)
            self._attach_rmsnorm_function_def(model)

        if not any(
            f.name == "SkipLayerNormalization" and f.domain == domain
            for f in model.functions
        ):
            add_n = helper.make_node("Add", inputs=["x1", "x2"], outputs=["res"])
            ln_n = helper.make_node(
                "LayerNormalization",
                inputs=["res", "scale", "bias"],
                outputs=["y"],
                domain=domain,
            )
            func_ln = helper.make_function(
                domain=domain,
                fname="SkipLayerNormalization",
                inputs=["x1", "x2", "scale", "bias"],
                outputs=["y"],
                nodes=[add_n, ln_n],
                opset_imports=[opset_standard, opset_custom],
                doc_string="SkipLayerNormalization function",
            )
            model.functions.append(func_ln)
            self._attach_layernorm_function_def(model)
