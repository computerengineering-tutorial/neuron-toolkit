"""Attention & RoPE Macro-Block Synthesizer Pass.

Synthesizes high-level attention macro-blocks:
- RoPE (Rotary Position Embedding rewiring & frequency table binding) -> RotaryEmbedding
- GatedUnit (Sigmoid -> Mul attention gating) -> GatedUnit
- SoftmaxMatMul (Softmax -> MatMul attention score contraction) -> SoftmaxMatMul
- ScaledDotProductAttention (QK transpose, scale, softmax, V contraction) -> ScaledDotProductAttention
- SplitQKV (Packed QKV decomposition) -> SplitQKV
- MergeHeads (Multi-head transpose & reshape) -> MergeHeads
Attaches formal FunctionProto definitions for Netron expandable hierarchical visualization.
"""

from __future__ import annotations

import collections
import logging
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper

from neuron_toolkit.compiler.base import CompilerContext, Pass
from neuron_toolkit.compiler.passes.canonicalization.shape_canonicalizer import (
    ShapeCanonicalizer,
)

log = logging.getLogger("neuron_toolkit.compiler.attention_synthesizer")


class AttentionSynthesizer(Pass):
    """Synthesizes attention and rotary position embedding blocks."""

    @property
    def name(self) -> str:
        return "attention_synthesizer"

    @property
    def category(self) -> str:
        return "synthesis"

    def run(self, model: onnx.ModelProto, context: CompilerContext) -> onnx.ModelProto:
        nodes_before = len(model.graph.node)

        if context.should_run("rope"):
            model = self.apply_rope_fusion(model, mode=context.mode)
        if context.should_run("gated_unit"):
            model = self.apply_gated_unit_fusion(model, mode=context.mode)
        if context.should_run("softmax_matmul"):
            model = self.apply_softmax_matmul_fusion(model, mode=context.mode)
        if context.should_run("sdpa") or context.should_run("attention"):
            model = self.apply_scaled_dot_product_attention_fusion(model, mode=context.mode)
        if context.should_run("merge_heads"):
            model = self.apply_merge_heads_fusion(model, mode=context.mode)
        if context.should_run("split_qkv"):
            model = self.apply_split_qkv_fusion(model, mode=context.mode)
        if context.should_run("gated_attention") or context.should_run("attention"):
            model = self.apply_gated_attention_fusion(model, mode=context.mode)

        nodes_after = len(model.graph.node)
        context.record_pass(self.name, nodes_before, nodes_after)
        return model

    # -------------------------------------------------------------------------
    # RoPE Fusion
    # -------------------------------------------------------------------------
    def apply_rope_fusion(self, model: onnx.ModelProto, mode: str = "function_proto") -> onnx.ModelProto:
        """Fuse 12-node Rotary Position Embedding expansions into RotaryEmbedding."""
        graph = model.graph
        node_by_output: dict[str, onnx.NodeProto] = {
            out: n for n in graph.node for out in n.output
        }
        nodes_to_remove: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        rope_count = 0

        for n in graph.node:
            if n.op_type == "Add" and len(n.input) == 2:
                in0 = node_by_output.get(n.input[0])
                in1 = node_by_output.get(n.input[1])
                if in0 and in1 and in0.op_type == "Mul" and in1.op_type == "Mul":
                    m_cos, m_sin = None, None
                    for mul_node in (in0, in1):
                        for inp in mul_node.input:
                            p = node_by_output.get(inp)
                            if p and p.op_type == "Cos":
                                m_cos = (mul_node, inp)
                            elif p and p.op_type == "Sin":
                                m_sin = (mul_node, inp)

                    if m_cos and m_sin:
                        mul_c, cos_tensor = m_cos
                        mul_s, sin_tensor = m_sin
                        x_slice = mul_c.input[0] if mul_c.input[1] == cos_tensor else mul_c.input[1]
                        rot_tensor = mul_s.input[0] if mul_s.input[1] == sin_tensor else mul_s.input[1]

                        subgraph_nodes = {n.name, mul_c.name, mul_s.name}
                        curr = node_by_output.get(rot_tensor)
                        if curr and curr.op_type == "Reshape":
                            subgraph_nodes.add(curr.name)
                            concat_n = node_by_output.get(curr.input[0])
                            if concat_n and concat_n.op_type == "Concat":
                                subgraph_nodes.add(concat_n.name)
                                for u_inp in concat_n.input:
                                    u_n = node_by_output.get(u_inp)
                                    if u_n and u_n.op_type == "Unsqueeze":
                                        subgraph_nodes.add(u_n.name)
                                        p_n = node_by_output.get(u_n.input[0])
                                        if p_n:
                                            subgraph_nodes.add(p_n.name)
                                            if p_n.op_type == "Neg":
                                                sq = node_by_output.get(p_n.input[0])
                                                if sq and sq.op_type == "Squeeze":
                                                    subgraph_nodes.add(sq.name)
                                                    sp = node_by_output.get(sq.input[0])
                                                    if sp and sp.op_type == "Split":
                                                        subgraph_nodes.add(sp.name)
                                            elif p_n.op_type == "Squeeze":
                                                sp = node_by_output.get(p_n.input[0])
                                                if sp and sp.op_type == "Split":
                                                    subgraph_nodes.add(sp.name)

                        nodes_to_remove.update(subgraph_nodes)

                        cos_n = node_by_output.get(cos_tensor)
                        sin_n = node_by_output.get(sin_tensor)
                        if cos_n and sin_n and cos_n.op_type == "Cos" and sin_n.op_type == "Sin" and cos_n.input == sin_n.input:
                            theta_tensor = cos_n.input[0]
                            base_theta = self._find_base_frequency_tensor(theta_tensor, node_by_output)
                            rope_inputs = [x_slice, base_theta]
                            nodes_to_remove.update({cos_n.name, sin_n.name})
                        else:
                            rope_inputs = [x_slice, cos_tensor, sin_tensor]

                        rope_node = helper.make_node(
                            op_type="RotaryEmbedding",
                            inputs=rope_inputs,
                            outputs=[n.output[0]],
                            name=f"{n.name}_fused_rope",
                            domain="custom.pytorch",
                        )
                        new_nodes.append(rope_node)
                        rope_count += 1

        if not new_nodes:
            return model

        rewrites: dict[str, str] = {}
        for r_node in new_nodes:
            r_out = r_node.output[0]
            concats = [c for c in graph.node if c.op_type == "Concat" and r_out in c.input]
            if concats:
                concat = concats[0]
                rewrites[concat.output[0]] = r_out
                nodes_to_remove.add(concat.name)

            x_in = r_node.input[0]
            p = node_by_output.get(x_in)
            if p and p.op_type == "Slice":
                r_node.input[0] = p.input[0]
                nodes_to_remove.add(p.name)

        def canonical(name: str) -> str:
            while name in rewrites:
                name = rewrites[name]
            return name

        surviving = []
        for n in graph.node:
            if n.name in nodes_to_remove:
                continue
            for idx, inp in enumerate(n.input):
                n.input[idx] = canonical(inp)
            surviving.append(n)
        surviving.extend(new_nodes)

        del graph.node[:]
        graph.node.extend(surviving)

        if mode == "function_proto":
            self._attach_rope_function_def(model)

        log.info("RoPE Fusion: Fused %d RotaryEmbedding instances", rope_count)
        return ShapeCanonicalizer().apply_dead_code_elimination(model)

    def _attach_rope_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for RotaryEmbedding."""
        func_name = "RotaryEmbedding"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", 14)
        cos_node = helper.make_node("Cos", inputs=["theta"], outputs=["cos"])
        sin_node = helper.make_node("Sin", inputs=["theta"], outputs=["sin"])
        m_cos = helper.make_node("Mul", inputs=["x", "cos"], outputs=["q_cos"])
        m_sin = helper.make_node("Mul", inputs=["x", "sin"], outputs=["q_sin"])
        add_node = helper.make_node("Add", inputs=["q_cos", "q_sin"], outputs=["y"])
        rope_func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["x", "theta"],
            outputs=["y"],
            nodes=[cos_node, sin_node, m_cos, m_sin, add_node],
            opset_imports=[opset_standard],
            doc_string="Rotary position embedding function",
        )
        model.functions.append(rope_func)

    def _find_base_frequency_tensor(
        self,
        tensor_name: str,
        node_by_output: dict[str, onnx.NodeProto],
    ) -> str:
        """Trace back through low-level slice/reshape nodes to find the root frequency tensor."""
        curr = tensor_name
        visited = set()
        while curr in node_by_output and curr not in visited:
            visited.add(curr)
            n = node_by_output[curr]
            if n.op_type == "ScatterND":
                return n.input[0] if n.input else curr
            if n.op_type == "Slice":
                curr = n.input[0]
            elif n.op_type in ("Reshape", "Expand", "Unsqueeze", "Cast"):
                curr = n.input[0]
            elif n.op_type == "Einsum":
                scatter_nodes = [node for node in node_by_output.values() if node.op_type == "ScatterND" and node.input]
                if scatter_nodes:
                    return scatter_nodes[0].input[0]
                return curr
            else:
                break
        return curr

    # -------------------------------------------------------------------------
    # GatedUnit Fusion
    # -------------------------------------------------------------------------
    def apply_gated_unit_fusion(self, model: onnx.ModelProto, mode: str = "function_proto") -> onnx.ModelProto:
        """Fuse Sigmoid -> Mul gating into GatedUnit."""
        graph = model.graph
        node_by_output: dict[str, onnx.NodeProto] = {
            out: n for n in graph.node for out in n.output
        }
        consumer_map: dict[str, list[onnx.NodeProto]] = {}
        for node in graph.node:
            for inp in node.input:
                consumer_map.setdefault(inp, []).append(node)
        graph_outputs = {out.name for out in graph.output}

        nodes_to_remove: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        gate_count = 0

        for n in graph.node:
            if n.op_type == "Mul" and len(n.input) == 2:
                sig_in = [inp for inp in n.input if inp in node_by_output and node_by_output[inp].op_type == "Sigmoid"]
                if sig_in:
                    sig_node = node_by_output[sig_in[0]]
                    other_inp = [inp for inp in n.input if inp != sig_node.output[0]][0]
                    gate_in = sig_node.input[0]

                    nodes_to_remove.add(n.name)
                    # Only remove sig_node if no other node consumes its output
                    if len(consumer_map.get(sig_node.output[0], [])) <= 1 and sig_node.output[0] not in graph_outputs:
                        nodes_to_remove.add(sig_node.name)

                    gated_node = helper.make_node(
                        op_type="GatedUnit",
                        inputs=[other_inp, gate_in],
                        outputs=[n.output[0]],
                        name=f"{n.name}_fused_gated",
                        domain="custom.pytorch",
                    )
                    new_nodes.append(gated_node)
                    gate_count += 1

        if not new_nodes:
            return model

        surviving = [n for n in graph.node if n.name not in nodes_to_remove] + new_nodes
        del graph.node[:]
        graph.node.extend(surviving)

        if mode == "function_proto":
            self._attach_gated_unit_function_def(model)

        log.info("GatedUnit Fusion: Fused %d GatedUnit instances", gate_count)
        return ShapeCanonicalizer().apply_dead_code_elimination(model)

    def _find_base_gate_tensor(
        self,
        gate_in: str,
        node_by_output: dict[str, onnx.NodeProto],
    ) -> str:
        """Trace back through gate reshape/transpose pipeline to the root linear projection output."""
        curr = gate_in
        visited = set()
        while curr in node_by_output and curr not in visited:
            visited.add(curr)
            n = node_by_output[curr]
            if n.op_type in ("Reshape", "Transpose"):
                curr = n.input[0]
            else:
                break
        return curr

    def _attach_gated_unit_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for GatedUnit."""
        func_name = "GatedUnit"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", 14)
        t_node = helper.make_node("Transpose", inputs=["gate"], outputs=["gate_t"], perm=[0, 2, 1])
        axes_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["axes_val"],
            value=helper.make_tensor("axes_val", onnx.TensorProto.INT64, [1], [-1]),
        )
        u_node = helper.make_node("Unsqueeze", inputs=["gate_t", "axes_val"], outputs=["gate_4d"])
        sig_node = helper.make_node("Sigmoid", inputs=["gate_4d"], outputs=["sig_out"])
        mul_node = helper.make_node("Mul", inputs=["x", "sig_out"], outputs=["y"])
        gate_func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["x", "gate"],
            outputs=["y"],
            nodes=[t_node, axes_node, u_node, sig_node, mul_node],
            opset_imports=[opset_standard],
            doc_string="Gated unit attention projection function",
        )
        model.functions.append(gate_func)

    # -------------------------------------------------------------------------
    # SoftmaxMatMul Fusion
    # -------------------------------------------------------------------------
    def apply_softmax_matmul_fusion(self, model: onnx.ModelProto, mode: str = "function_proto") -> onnx.ModelProto:
        """Fuse Softmax -> MatMul into SoftmaxMatMul."""
        graph = model.graph
        node_by_output: dict[str, onnx.NodeProto] = {
            out: n for n in graph.node for out in n.output
        }
        consumer_map: dict[str, list[onnx.NodeProto]] = {}
        for node in graph.node:
            for inp in node.input:
                consumer_map.setdefault(inp, []).append(node)
        graph_outputs = {out.name for out in graph.output}

        nodes_to_remove: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        sm_count = 0

        for n in graph.node:
            if n.op_type == "MatMul" and len(n.input) == 2:
                in0 = node_by_output.get(n.input[0])
                if in0 and in0.op_type == "Softmax":
                    scores = in0.input[0]
                    v = n.input[1]
                    axis = -1
                    for a in in0.attribute:
                        if a.name == "axis":
                            axis = a.i
                    nodes_to_remove.add(n.name)
                    if len(consumer_map.get(in0.output[0], [])) <= 1 and in0.output[0] not in graph_outputs:
                        nodes_to_remove.add(in0.name)
                    sm_node = helper.make_node(
                        op_type="SoftmaxMatMul",
                        inputs=[scores, v],
                        outputs=[n.output[0]],
                        name=f"{n.name}_fused_sm_matmul",
                        domain="custom.pytorch",
                        axis=axis,
                    )
                    new_nodes.append(sm_node)
                    sm_count += 1

        if not new_nodes:
            return model

        surviving = [n for n in graph.node if n.name not in nodes_to_remove] + new_nodes
        del graph.node[:]
        graph.node.extend(surviving)

        if mode == "function_proto":
            self._attach_softmax_matmul_function_def(model)

        log.info("SoftmaxMatMul Fusion: Fused %d SoftmaxMatMul instances", sm_count)
        return model

    def _attach_softmax_matmul_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for SoftmaxMatMul."""
        func_name = "SoftmaxMatMul"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", 14)
        sm_node = helper.make_node("Softmax", inputs=["scores"], outputs=["probs"], axis=-1)
        mm_node = helper.make_node("MatMul", inputs=["probs", "v"], outputs=["y"])
        func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["scores", "v"],
            outputs=["y"],
            nodes=[sm_node, mm_node],
            opset_imports=[opset_standard],
            doc_string="Softmax-MatMul attention function",
        )
        model.functions.append(func)

    @staticmethod
    def _handle_k_transpose(
        tr_node: onnx.NodeProto,
        k_raw_input: str,
        nodes_to_drop: set[str],
    ) -> str:
        """Handle transposition on K input, retaining head-permutation if not identity."""
        perm_attrs = [a for a in tr_node.attribute if a.name == "perm"]
        if perm_attrs and len(perm_attrs[0].ints) >= 2:
            p = list(perm_attrs[0].ints)
            new_perm = p[:-2] + [p[-1], p[-2]]
            if new_perm == list(range(len(p))):
                nodes_to_drop.add(tr_node.name)
                return k_raw_input
            else:
                perm_attrs[0].ints[:] = new_perm
                if tr_node.input[0] != k_raw_input:
                    tr_node.input[0] = k_raw_input
                return tr_node.output[0]
        else:
            nodes_to_drop.add(tr_node.name)
            return k_raw_input

    # -------------------------------------------------------------------------
    # ScaledDotProductAttention Fusion
    # -------------------------------------------------------------------------
    def apply_scaled_dot_product_attention_fusion(self, model: onnx.ModelProto, mode: str = "function_proto") -> onnx.ModelProto:
        """Fuse attention scaling, transposition, dot-product, and softmax into ScaledDotProductAttention."""
        graph = model.graph
        node_by_output: dict[str, onnx.NodeProto] = {
            out: n for n in graph.node for out in n.output
        }
        sm_nodes = [n for n in graph.node if n.op_type == "SoftmaxMatMul"]
        new_nodes: list[onnx.NodeProto] = []
        nodes_to_drop: set[str] = set()

        inits = {i.name for i in graph.initializer}
        for sm in sm_nodes:
            scores_inp, v_inp = sm.input[0], sm.input[1]
            mm = node_by_output.get(scores_inp)
            scale_node_after_mm = None
            if mm and mm.op_type in ("Div", "Mul") and len(mm.input) >= 2:
                parent_mm = node_by_output.get(mm.input[0])
                if parent_mm and parent_mm.op_type == "MatMul":
                    scale_operand = mm.input[1]
                    is_scalar = False
                    if scale_operand in inits:
                        init_tensor = next(i for i in graph.initializer if i.name == scale_operand)
                        if len(init_tensor.dims) == 0 or (len(init_tensor.dims) == 1 and init_tensor.dims[0] == 1):
                            is_scalar = True
                    else:
                        cand_node = node_by_output.get(scale_operand)
                        if cand_node and cand_node.op_type in ("Constant", "Pow", "Sqrt", "Div"):
                            is_scalar = True
                    if is_scalar:
                        scale_node_after_mm = mm
                        mm = parent_mm

            if not mm or mm.op_type != "MatMul":
                continue

            if scale_node_after_mm is not None:
                nodes_to_drop.add(scale_node_after_mm.name)

            q_cand = node_by_output.get(mm.input[0])
            k_cand = node_by_output.get(mm.input[1])

            # Resolve Q input
            has_scale_mul = (scale_node_after_mm is not None)
            if q_cand and q_cand.op_type in ("Mul", "Div"):
                has_scale_mul = True
                nodes_to_drop.add(q_cand.name)
                if q_cand.op_type == "Div":
                    q_inp = q_cand.input[0]
                else:
                    # Check which input is scale
                    in1_node = node_by_output.get(q_cand.input[1])
                    in0_node = node_by_output.get(q_cand.input[0])
                    if q_cand.input[0] in inits or (in0_node and in0_node.op_type in ("Pow", "Constant")):
                        q_inp = q_cand.input[1]
                    else:
                        q_inp = q_cand.input[0]
            else:
                q_inp = mm.input[0]
                # Check if scale Mul/Div occurred before Transpose / Reshape
                curr = q_cand
                while curr and curr.op_type in ("Transpose", "Reshape") and len(curr.input) >= 1:
                    parent = node_by_output.get(curr.input[0])
                    if parent and parent.op_type in ("Mul", "Div"):
                        has_scale_mul = True
                        if parent.op_type == "Div":
                            curr.input[0] = parent.input[0]
                        else:
                            in0_p = node_by_output.get(parent.input[0])
                            scale_cand = parent.input[0] if (parent.input[0] in inits or (in0_p and in0_p.op_type in ("Pow", "Constant"))) else parent.input[1]
                            val_cand = parent.input[1] if scale_cand == parent.input[0] else parent.input[0]
                            curr.input[0] = val_cand
                        nodes_to_drop.add(parent.name)
                        break
                    curr = parent

            # Resolve K input
            if k_cand and k_cand.op_type == "Transpose":
                k_sub = node_by_output.get(k_cand.input[0])
                if k_sub and k_sub.op_type in ("Mul", "Div"):
                    nodes_to_drop.add(k_sub.name)
                    if k_sub.op_type == "Div":
                        k_raw = k_sub.input[0]
                    else:
                        in0_k = node_by_output.get(k_sub.input[0])
                        k_raw = k_sub.input[1] if (k_sub.input[0] in inits or (in0_k and in0_k.op_type in ("Pow", "Constant"))) else k_sub.input[0]
                    k_inp = self._handle_k_transpose(k_cand, k_raw, nodes_to_drop)
                else:
                    k_inp = self._handle_k_transpose(k_cand, k_cand.input[0], nodes_to_drop)
            elif k_cand and k_cand.op_type in ("Mul", "Div"):
                nodes_to_drop.add(k_cand.name)
                if k_cand.op_type == "Div":
                    k_inp_cand = k_cand.input[0]
                else:
                    in0_k = node_by_output.get(k_cand.input[0])
                    k_inp_cand = k_cand.input[1] if (k_cand.input[0] in inits or (in0_k and in0_k.op_type in ("Pow", "Constant"))) else k_cand.input[0]
                k_sub = node_by_output.get(k_inp_cand)
                if k_sub and k_sub.op_type == "Transpose":
                    k_inp = self._handle_k_transpose(k_sub, k_sub.input[0], nodes_to_drop)
                else:
                    k_inp = k_inp_cand
            else:
                k_inp = mm.input[1]

            cast_q = node_by_output.get(q_inp)
            if cast_q and cast_q.op_type == "Cast":
                q_inp = cast_q.input[0]
            cast_k = node_by_output.get(k_inp)
            if cast_k and cast_k.op_type == "Cast":
                k_inp = cast_k.input[0]
            cast_v = node_by_output.get(v_inp)
            if cast_v and cast_v.op_type == "Cast":
                v_inp = cast_v.input[0]

            sdpa = helper.make_node(
                "ScaledDotProductAttention",
                inputs=[q_inp, k_inp, v_inp],
                outputs=list(sm.output),
                name=f"{sm.name}_sdpa",
                domain="custom.pytorch",
            )
            new_nodes.append(sdpa)
            nodes_to_drop.add(sm.name)
            nodes_to_drop.add(mm.name)

        if not new_nodes:
            return model

        kept = [n for n in graph.node if n.name not in nodes_to_drop]
        kept.extend(new_nodes)
        del graph.node[:]
        graph.node.extend(kept)

        if mode == "function_proto":
            self._attach_sdpa_function_def(model)

        log.info("ScaledDotProductAttention Fusion: Fused %d attention blocks", len(new_nodes))
        return ShapeCanonicalizer().apply_dead_code_elimination(model)

    def _attach_sdpa_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for ScaledDotProductAttention."""
        func_name = "ScaledDotProductAttention"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", 14)
        k_trans = helper.make_node("Transpose", inputs=["K"], outputs=["KT"], perm=[0, 1, 3, 2])
        mm1 = helper.make_node("MatMul", inputs=["Q", "KT"], outputs=["Scores"])
        sm = helper.make_node("Softmax", inputs=["Scores"], outputs=["Probs"], axis=-1)
        mm2 = helper.make_node("MatMul", inputs=["Probs", "V"], outputs=["Y"])
        func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            nodes=[k_trans, mm1, sm, mm2],
            opset_imports=[opset_standard],
            doc_string="Scaled Dot-Product Attention function",
        )
        model.functions.append(func)

    # -------------------------------------------------------------------------
    # SplitQKV Fusion
    # -------------------------------------------------------------------------
    def apply_split_qkv_fusion(self, model: onnx.ModelProto, mode: str = "function_proto") -> onnx.ModelProto:
        """Fuse Reshape -> Transpose(perm=[2, 0, 3, 1, 4]) -> Unbind into SplitQKV."""
        graph = model.graph
        node_by_output: dict[str, onnx.NodeProto] = {
            out: n for n in graph.node for out in n.output
        }
        new_nodes: list[onnx.NodeProto] = []
        nodes_to_drop: set[str] = set()

        unbind_nodes = [n for n in graph.node if n.op_type == "Unbind"]
        for unbind in unbind_nodes:
            trans = node_by_output.get(unbind.input[0])
            if not trans or trans.op_type != "Transpose":
                continue
            perm = next((list(a.ints) for a in trans.attribute if a.name == "perm"), [])
            if perm != [2, 0, 3, 1, 4]:
                continue
            reshape = node_by_output.get(trans.input[0])
            if not reshape or reshape.op_type != "Reshape":
                continue

            concat = node_by_output.get(reshape.input[1])
            num_heads = 4
            if concat and concat.op_type == "Concat" and len(concat.input) >= 4:
                h_node = node_by_output.get(concat.input[3])
                if h_node and h_node.op_type == "Constant":
                    try:
                        num_heads = int(numpy_helper.to_array(h_node.attribute[0].t).item())
                    except Exception:
                        pass

            qkv_inp = reshape.input[0]
            sqkv = helper.make_node(
                "SplitQKV",
                inputs=[qkv_inp],
                outputs=list(unbind.output),
                name=f"{unbind.name}_split_qkv",
                domain="custom.pytorch",
                num_heads=num_heads,
            )
            new_nodes.append(sqkv)
            nodes_to_drop.add(unbind.name)
            nodes_to_drop.add(trans.name)
            nodes_to_drop.add(reshape.name)

        if not new_nodes:
            return model

        kept = [n for n in graph.node if n.name not in nodes_to_drop]
        kept.extend(new_nodes)
        del graph.node[:]
        graph.node.extend(kept)

        if mode == "function_proto":
            self._attach_split_qkv_function_def(model)

        log.info("SplitQKV Fusion: Fused %d SplitQKV instances", len(new_nodes))
        return ShapeCanonicalizer().apply_dead_code_elimination(model)

    def _attach_split_qkv_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for SplitQKV."""
        func_name = "SplitQKV"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", 14)
        s = helper.make_node("Split", inputs=["QKV"], outputs=["Q", "K", "V"], axis=0)
        func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["QKV"],
            outputs=["Q", "K", "V"],
            nodes=[s],
            opset_imports=[opset_standard],
            doc_string="Split QKV projections into Q, K, V",
        )
        model.functions.append(func)

    # -------------------------------------------------------------------------
    # MergeHeads Fusion
    # -------------------------------------------------------------------------
    def apply_merge_heads_fusion(self, model: onnx.ModelProto, mode: str = "function_proto") -> onnx.ModelProto:
        """Fuse Transpose(perm=[0, 2, 1, 3]) and Reshape after GatedUnit into MergeHeads."""
        graph = model.graph
        consumer_map: dict[str, list[onnx.NodeProto]] = collections.defaultdict(list)
        for n in graph.node:
            for inp in n.input:
                consumer_map[inp].append(n)

        new_nodes: list[onnx.NodeProto] = []
        nodes_to_drop: set[str] = set()

        gated_units = [n for n in graph.node if n.op_type == "GatedUnit"]
        for gu in gated_units:
            users = consumer_map.get(gu.output[0], [])
            trans = [u for u in users if u.op_type == "Transpose"]
            if not trans:
                continue
            t_node = trans[0]
            perm = next((list(a.ints) for a in t_node.attribute if a.name == "perm"), [])
            if perm != [0, 2, 1, 3]:
                continue
            reshapes = [u for u in consumer_map.get(t_node.output[0], []) if u.op_type == "Reshape"]
            if not reshapes:
                continue
            r_node = reshapes[0]
            mh = helper.make_node(
                "MergeHeads",
                inputs=[gu.output[0]],
                outputs=list(r_node.output),
                name=f"{r_node.name}_merge_heads",
                domain="custom.pytorch",
            )
            new_nodes.append(mh)
            nodes_to_drop.add(t_node.name)
            nodes_to_drop.add(r_node.name)

        if not new_nodes:
            return model

        kept = [n for n in graph.node if n.name not in nodes_to_drop]
        kept.extend(new_nodes)
        del graph.node[:]
        graph.node.extend(kept)

        if mode == "function_proto":
            self._attach_merge_heads_function_def(model)

        log.info("MergeHeads Fusion: Fused %d MergeHeads instances", len(new_nodes))
        return ShapeCanonicalizer().apply_dead_code_elimination(model)

    def _attach_merge_heads_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for MergeHeads."""
        func_name = "MergeHeads"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", 14)
        t = helper.make_node("Transpose", inputs=["X"], outputs=["Y"], perm=[0, 2, 1, 3])
        func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["X"],
            outputs=["Y"],
            nodes=[t],
            opset_imports=[opset_standard],
            doc_string="Merge attention heads function",
        )
        model.functions.append(func)

    # -------------------------------------------------------------------------
    # Gated Multi-Head Attention Fusion
    # -------------------------------------------------------------------------
    def apply_gated_attention_fusion(self, model: onnx.ModelProto, mode: str = "function_proto") -> onnx.ModelProto:
        """Fuse full attention subgraphs (Norm -> QKV/Gates -> RoPE -> SDPA -> GatedUnit -> MergeHeads -> Out) into GatedAttention."""
        graph = model.graph
        node_by_output: dict[str, onnx.NodeProto] = {
            out: n for n in graph.node for out in n.output if out
        }
        consumer_map: dict[str, list[onnx.NodeProto]] = collections.defaultdict(list)
        for n in graph.node:
            for inp in n.input:
                if inp:
                    consumer_map[inp].append(n)

        new_nodes: list[onnx.NodeProto] = []
        nodes_to_drop: set[str] = set()

        for sdpa in graph.node:
            if sdpa.op_type != "ScaledDotProductAttention" or len(sdpa.input) < 3:
                continue

            # 1. Q, K, V sources
            q_node = node_by_output.get(sdpa.input[0])
            k_node = node_by_output.get(sdpa.input[1])
            v_tensor = sdpa.input[2]

            if not q_node or q_node.op_type != "RotaryEmbedding":
                continue
            if not k_node or k_node.op_type != "RotaryEmbedding":
                continue

            q_in = q_node.input[0]
            k_in = k_node.input[0]
            theta_name = q_node.input[1]

            # 2. SplitQKV
            split_qkv = node_by_output.get(q_in)
            if not split_qkv or split_qkv.op_type != "SplitQKV" or len(split_qkv.output) < 3:
                continue
            if split_qkv.output[0] != q_in or split_qkv.output[1] != k_in or split_qkv.output[2] != v_tensor:
                continue

            # 3. to_qkv
            to_qkv = node_by_output.get(split_qkv.input[0])
            if not to_qkv or to_qkv.op_type not in ("Linear", "MatMul"):
                continue

            # 4. Norm (RMSNorm or LayerNorm)
            norm = node_by_output.get(to_qkv.input[0])
            if not norm or norm.op_type not in ("RMSNorm", "RMSNormalization", "LayerNormalization"):
                continue

            # 5. Downstream: GatedUnit
            sdpa_users = consumer_map.get(sdpa.output[0], [])
            if len(sdpa_users) != 1 or sdpa_users[0].op_type != "GatedUnit":
                continue
            gated_unit = sdpa_users[0]

            # to_gates
            to_gates = node_by_output.get(gated_unit.input[1])
            if not to_gates or to_gates.op_type not in ("Linear", "MatMul"):
                continue
            if to_gates.input[0] != norm.output[0]:
                continue

            # 6. MergeHeads
            gu_users = consumer_map.get(gated_unit.output[0], [])
            if len(gu_users) != 1 or gu_users[0].op_type != "MergeHeads":
                continue
            merge_heads = gu_users[0]

            # 7. to_out
            mh_users = consumer_map.get(merge_heads.output[0], [])
            if len(mh_users) != 1 or mh_users[0].op_type not in ("Linear", "MatMul"):
                continue
            to_out = mh_users[0]

            # All 10 operations confirmed!
            x = norm.input[0]
            norm_gamma = norm.input[1] if len(norm.input) > 1 else ""

            qkv_w = to_qkv.input[1]
            qkv_b = to_qkv.input[2] if len(to_qkv.input) > 2 else ""

            gates_w = to_gates.input[1]
            gates_b = to_gates.input[2] if len(to_gates.input) > 2 else ""

            out_w = to_out.input[1]
            out_b = to_out.input[2] if len(to_out.input) > 2 else ""

            final_out = to_out.output[0]

            attn_inputs = [x, norm_gamma, qkv_w, qkv_b, gates_w, gates_b, theta_name, out_w, out_b]
            while attn_inputs and attn_inputs[-1] == "":
                attn_inputs.pop()

            gated_attn_node = helper.make_node(
                "GatedAttention",
                inputs=attn_inputs,
                outputs=[final_out],
                name=f"{norm.name}_fused_gated_attention",
                domain="custom.pytorch",
                doc_string="Fused Gated Multi-Head Attention block with RoPE",
            )
            for a in norm.attribute:
                if a.name == "epsilon":
                    gated_attn_node.attribute.append(a)
            for a in split_qkv.attribute:
                if a.name == "num_heads":
                    gated_attn_node.attribute.append(a)

            new_nodes.append(gated_attn_node)
            nodes_to_drop.update({
                norm.name,
                to_qkv.name,
                split_qkv.name,
                q_node.name,
                k_node.name,
                sdpa.name,
                to_gates.name,
                gated_unit.name,
                merge_heads.name,
                to_out.name,
            })

        if not new_nodes:
            return model

        kept = [n for n in graph.node if n.name not in nodes_to_drop]
        kept.extend(new_nodes)
        del graph.node[:]
        graph.node.extend(kept)

        custom_domain = "custom.pytorch"
        if not any(op.domain == custom_domain for op in model.opset_import):
            model.opset_import.append(helper.make_opsetid(custom_domain, 1))

        if mode == "function_proto":
            self._attach_gated_attention_function_def(model)

        log.info("GatedAttention Fusion: Fused %d GatedAttention instances", len(new_nodes))
        return ShapeCanonicalizer().apply_dead_code_elimination(model)

    def _attach_gated_attention_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for GatedAttention."""
        func_name = "GatedAttention"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", 14)
        opset_custom = helper.make_opsetid("custom.pytorch", 1)

        norm = helper.make_node("RMSNorm", inputs=["X", "norm_gamma"], outputs=["X_norm"], domain="custom.pytorch", epsilon=1e-12)
        to_qkv = helper.make_node("Linear", inputs=["X_norm", "qkv_w"], outputs=["QKV"], domain="custom.pytorch")
        to_gates = helper.make_node("Linear", inputs=["X_norm", "gates_w", "gates_b"], outputs=["Gates"], domain="custom.pytorch")
        split_qkv = helper.make_node("SplitQKV", inputs=["QKV"], outputs=["Q", "K", "V"], domain="custom.pytorch")
        rope_q = helper.make_node("RotaryEmbedding", inputs=["Q", "theta"], outputs=["Q_rot"], domain="custom.pytorch")
        rope_k = helper.make_node("RotaryEmbedding", inputs=["K", "theta"], outputs=["K_rot"], domain="custom.pytorch")
        sdpa = helper.make_node("ScaledDotProductAttention", inputs=["Q_rot", "K_rot", "V"], outputs=["AttnScores"], domain="custom.pytorch")
        gated = helper.make_node("GatedUnit", inputs=["AttnScores", "Gates"], outputs=["GatedScores"], domain="custom.pytorch")
        merge = helper.make_node("MergeHeads", inputs=["GatedScores"], outputs=["Merged"], domain="custom.pytorch")
        to_out = helper.make_node("Linear", inputs=["Merged", "out_w"], outputs=["Y"], domain="custom.pytorch")

        func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["X", "norm_gamma", "qkv_w", "qkv_b", "gates_w", "gates_b", "theta", "out_w", "out_b"],
            outputs=["Y"],
            nodes=[norm, to_qkv, to_gates, split_qkv, rope_q, rope_k, sdpa, gated, merge, to_out],
            opset_imports=[opset_standard, opset_custom],
            doc_string="Gated Multi-Head Attention block with RoPE and GatedUnit",
        )
        model.functions.append(func)
