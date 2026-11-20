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

import onnx
from onnx import helper, numpy_helper

from neuron_toolkit.compiler.architectures.fingerprint import (
    build_monomorphized_op_type,
    compute_subgraph_fingerprint,
    ensure_specialized_function_proto,
)
from neuron_toolkit.compiler.base import CompilerContext, Pass, LATEST_OPSET
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
        if context.should_run("gemma2_softcap") or context.should_run("softcap"):
            model = self.apply_gemma2_softcapping_fusion(model, mode=context.mode)
        if context.should_run("deepseek_mla") or context.should_run("mla"):
            model = self.apply_deepseek_mla_fusion(model, mode=context.mode)
        if context.should_run("gated_unit"):
            model = self.apply_gated_unit_fusion(model, mode=context.mode)
        if context.should_run("softmax_matmul"):
            model = self.apply_softmax_matmul_fusion(model, mode=context.mode)
        if context.should_run("sdpa") or context.should_run("attention"):
            model = self.apply_scaled_dot_product_attention_fusion(
                model, mode=context.mode
            )
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
    def apply_rope_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuse 12-node Rotary Position Embedding expansions into RotaryEmbedding."""
        graph = model.graph
        node_by_output: dict[str, onnx.NodeProto] = {
            out: n for n in graph.node for out in n.output
        }
        nodes_to_remove: set[int] = set()
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
                        x_slice = (
                            mul_c.input[0]
                            if mul_c.input[1] == cos_tensor
                            else mul_c.input[1]
                        )
                        rot_tensor = (
                            mul_s.input[0]
                            if mul_s.input[1] == sin_tensor
                            else mul_s.input[1]
                        )

                        subgraph_nodes = {id(n), id(mul_c), id(mul_s)}
                        curr = node_by_output.get(rot_tensor)
                        if curr and curr.op_type == "Reshape":
                            subgraph_nodes.add(id(curr))
                            concat_n = node_by_output.get(curr.input[0])
                            if concat_n and concat_n.op_type == "Concat":
                                subgraph_nodes.add(id(concat_n))
                                for u_inp in concat_n.input:
                                    u_n = node_by_output.get(u_inp)
                                    if u_n and u_n.op_type == "Unsqueeze":
                                        subgraph_nodes.add(id(u_n))
                                        p_n = node_by_output.get(u_n.input[0])
                                        if p_n:
                                            subgraph_nodes.add(id(p_n))
                                            if p_n.op_type == "Neg":
                                                sq = node_by_output.get(p_n.input[0])
                                                if sq and sq.op_type == "Squeeze":
                                                    subgraph_nodes.add(id(sq))
                                                    sp = node_by_output.get(sq.input[0])
                                                    if sp and sp.op_type == "Split":
                                                        subgraph_nodes.add(id(sp))
                                            elif p_n.op_type == "Squeeze":
                                                sp = node_by_output.get(p_n.input[0])
                                                if sp and sp.op_type == "Split":
                                                    subgraph_nodes.add(id(sp))

                        nodes_to_remove.update(subgraph_nodes)

                        cos_n = node_by_output.get(cos_tensor)
                        sin_n = node_by_output.get(sin_tensor)
                        if (
                            cos_n
                            and sin_n
                            and cos_n.op_type == "Cos"
                            and sin_n.op_type == "Sin"
                            and cos_n.input == sin_n.input
                        ):
                            theta_tensor = cos_n.input[0]
                            base_theta = self._find_base_frequency_tensor(
                                theta_tensor, node_by_output
                            )
                            rope_inputs = [x_slice, base_theta]
                            nodes_to_remove.update({id(cos_n), id(sin_n)})
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
                    else:
                        # Check HuggingFace rotate_half RoPE:
                        # One Mul has rx = Concat([Neg(x2), x1]) and sin
                        # The other Mul has x and cos
                        rot_n, mul_rx, mul_x, sin_t, cos_t = None, None, None, None, None
                        for m_cand, m_other in ((in0, in1), (in1, in0)):
                            for inp in m_cand.input:
                                cand_rot = node_by_output.get(inp)
                                if cand_rot and cand_rot.op_type == "Concat" and len(cand_rot.input) == 2:
                                    p_neg = node_by_output.get(cand_rot.input[0])
                                    if p_neg and p_neg.op_type == "Neg":
                                        rot_n = cand_rot
                                        mul_rx = m_cand
                                        mul_x = m_other
                                        sin_t = m_cand.input[1] if m_cand.input[0] == inp else m_cand.input[0]
                                        break
                            if rot_n:
                                break

                        if rot_n and mul_x and mul_rx:
                            neg_n = node_by_output.get(rot_n.input[0])
                            x2_cand = node_by_output.get(neg_n.input[0]) if neg_n else None
                            x1_cand = node_by_output.get(rot_n.input[1])
                            x_orig = None
                            if x2_cand and x1_cand:
                                if x2_cand.input and x1_cand.input and x2_cand.input[0] == x1_cand.input[0]:
                                    x_orig = x2_cand.input[0]
                            if not x_orig and x2_cand and x2_cand.op_type == "Split":
                                x_orig = x2_cand.input[0]

                            cos_t = mul_x.input[1] if (x_orig and mul_x.input[0] == x_orig) else (
                                mul_x.input[0] if (x_orig and mul_x.input[1] == x_orig) else mul_x.input[1]
                            )
                            x_in = x_orig or (mul_x.input[0] if mul_x.input[1] == cos_t else mul_x.input[1])
                            nodes_to_remove.update({id(n), id(mul_rx), id(mul_x), id(rot_n)})
                            if neg_n:
                                nodes_to_remove.add(id(neg_n))
                            if x1_cand:
                                nodes_to_remove.add(id(x1_cand))
                            if x2_cand:
                                nodes_to_remove.add(id(x2_cand))

                            rope_node = helper.make_node(
                                op_type="RotaryEmbedding",
                                inputs=[x_in, cos_t, sin_t],
                                outputs=[n.output[0]],
                                name=f"{n.name}_fused_rope",
                                domain="custom.pytorch",
                                interleaved=0,
                            )
                            new_nodes.append(rope_node)
                            rope_count += 1

        if not new_nodes:
            return model

        rewrites: dict[str, str] = {}
        for r_node in new_nodes:
            r_out = r_node.output[0]
            concats = [
                c for c in graph.node if c.op_type == "Concat" and r_out in c.input
            ]
            if concats:
                concat = concats[0]
                rewrites[concat.output[0]] = r_out
                nodes_to_remove.add(id(concat))

            x_in = r_node.input[0]
            p = node_by_output.get(x_in)
            if p and p.op_type == "Slice":
                r_node.input[0] = p.input[0]
                nodes_to_remove.add(id(p))

        def canonical(name: str) -> str:
            while name in rewrites:
                name = rewrites[name]
            return name

        surviving = []
        for n in graph.node:
            if id(n) in nodes_to_remove:
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
        opset_standard = helper.make_opsetid("", LATEST_OPSET)
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
            if n.op_type == "Slice" or n.op_type in (
                "Reshape",
                "Expand",
                "Unsqueeze",
                "Cast",
            ):
                curr = n.input[0]
            elif n.op_type == "Einsum":
                scatter_nodes = [
                    node
                    for node in node_by_output.values()
                    if node.op_type == "ScatterND" and node.input
                ]
                if scatter_nodes:
                    return scatter_nodes[0].input[0]
                return curr
            else:
                break
        return curr

    # -------------------------------------------------------------------------
    # GatedUnit Fusion
    # -------------------------------------------------------------------------
    def apply_gated_unit_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
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

        nodes_to_remove: set[int] = set()
        replacements: dict[int, onnx.NodeProto] = {}
        gate_count = 0

        for n in graph.node:
            if n.op_type == "Mul" and len(n.input) == 2:
                sig_in = [
                    inp
                    for inp in n.input
                    if inp in node_by_output
                    and node_by_output[inp].op_type
                    in ("Sigmoid", "SiLU", "Gelu", "Tanh", "Relu")
                ]
                if sig_in:
                    sig_node = node_by_output[sig_in[0]]
                    act_name = sig_node.op_type
                    other_inps = [inp for inp in n.input if inp != sig_node.output[0]]
                    if not other_inps:
                        # Inputs are identical (e.g. squaring x * x); not a gating pair
                        continue
                    other_inp = other_inps[0]
                    gate_in = sig_node.input[0]

                    nodes_to_remove.add(id(n))
                    # Only remove sig_node if no other node consumes its output
                    if (
                        len(consumer_map.get(sig_node.output[0], [])) <= 1
                        and sig_node.output[0] not in graph_outputs
                    ):
                        nodes_to_remove.add(id(sig_node))

                    fp = compute_subgraph_fingerprint([sig_node, n])
                    specialized_op = build_monomorphized_op_type(
                        "GatedUnit", act_name, fp
                    )

                    node_name = n.name or f"gated_unit_{gate_count}"
                    gated_node = helper.make_node(
                        op_type=specialized_op,
                        inputs=[other_inp, gate_in],
                        outputs=[n.output[0]],
                        name=f"{node_name}_fused_gated",
                        domain="custom.pytorch",
                        activation=act_name.lower(),
                    )
                    replacements[id(n)] = gated_node
                    gate_count += 1

                    if mode == "function_proto":
                        self._attach_specialized_gated_unit_function(
                            model, specialized_op, act_name, fp
                        )

        if not replacements:
            return model

        surviving = []
        for n in graph.node:
            if id(n) in replacements:
                surviving.append(replacements[id(n)])
            elif id(n) not in nodes_to_remove:
                surviving.append(n)
        del graph.node[:]
        graph.node.extend(surviving)

        log.info(
            "GatedUnit Fusion: Fused %d GatedUnit instances with polymorphic fingerprints",
            gate_count,
        )
        return ShapeCanonicalizer().apply_dead_code_elimination(model)

    def _attach_specialized_gated_unit_function(
        self,
        model: onnx.ModelProto,
        fname: str,
        act_name: str,
        fingerprint: str,
    ) -> None:
        """Attach formal, content-hashed ONNX FunctionProto definition for GatedUnit with exact inner activation."""
        if any(
            f.name == fname and f.domain == "custom.pytorch" for f in model.functions
        ):
            return
        opset_standard = helper.make_opsetid("", LATEST_OPSET)
        sig_node = helper.make_node(
            act_name,
            inputs=["gate"],
            outputs=["sig_out"],
            domain="" if act_name in ("Relu", "Tanh", "Sigmoid") else "custom.pytorch",
        )
        mul_node = helper.make_node("Mul", inputs=["x", "sig_out"], outputs=["y"])

        ensure_specialized_function_proto(
            model=model,
            domain="custom.pytorch",
            fname=fname,
            formal_inputs=["x", "gate"],
            formal_outputs=["y"],
            body_nodes=[sig_node, mul_node],
            opset_imports=[opset_standard],
            doc_string=f"Content-hashed polymorphic GatedUnit ({act_name}) [{fingerprint}]",
        )

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
        opset_standard = helper.make_opsetid("", LATEST_OPSET)
        t_node = helper.make_node(
            "Transpose", inputs=["gate"], outputs=["gate_t"], perm=[0, 2, 1]
        )
        axes_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["axes_val"],
            value=helper.make_tensor("axes_val", onnx.TensorProto.INT64, [1], [-1]),
        )
        u_node = helper.make_node(
            "Unsqueeze", inputs=["gate_t", "axes_val"], outputs=["gate_4d"]
        )
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
    def apply_softmax_matmul_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
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

        nodes_to_remove: set[int] = set()
        replacements: dict[int, onnx.NodeProto] = {}
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
                    nodes_to_remove.add(id(n))
                    if (
                        len(consumer_map.get(in0.output[0], [])) <= 1
                        and in0.output[0] not in graph_outputs
                    ):
                        nodes_to_remove.add(id(in0))
                    node_name = n.name or f"sm_matmul_{sm_count}"
                    sm_node = helper.make_node(
                        op_type="SoftmaxMatMul",
                        inputs=[scores, v],
                        outputs=[n.output[0]],
                        name=f"{node_name}_fused_sm_matmul",
                        domain="custom.pytorch",
                        axis=axis,
                    )
                    replacements[id(n)] = sm_node
                    sm_count += 1

        if not replacements:
            return model

        surviving = []
        for n in graph.node:
            if id(n) in replacements:
                surviving.append(replacements[id(n)])
            elif id(n) not in nodes_to_remove:
                surviving.append(n)
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
        opset_standard = helper.make_opsetid("", LATEST_OPSET)
        sm_node = helper.make_node(
            "Softmax", inputs=["scores"], outputs=["probs"], axis=-1
        )
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
        nodes_to_drop: set[int],
    ) -> str:
        """Handle transposition on K input, retaining head-permutation if not identity."""
        perm_attrs = [a for a in tr_node.attribute if a.name == "perm"]
        if perm_attrs and len(perm_attrs[0].ints) >= 2:
            p = list(perm_attrs[0].ints)
            new_perm = p[:-2] + [p[-1], p[-2]]
            if new_perm == list(range(len(p))):
                nodes_to_drop.add(id(tr_node))
                return k_raw_input
            perm_attrs[0].ints[:] = new_perm
            if tr_node.input[0] != k_raw_input:
                tr_node.input[0] = k_raw_input
            return tr_node.output[0]
        nodes_to_drop.add(id(tr_node))
        return k_raw_input

    # -------------------------------------------------------------------------
    # ScaledDotProductAttention Fusion
    # -------------------------------------------------------------------------
    def apply_scaled_dot_product_attention_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuse attention scaling, transposition, dot-product, and softmax into ScaledDotProductAttention."""
        graph = model.graph
        node_by_output: dict[str, onnx.NodeProto] = {
            out: n for n in graph.node for out in n.output
        }
        sm_nodes = [n for n in graph.node if n.op_type == "SoftmaxMatMul"]
        replacements: dict[int, onnx.NodeProto] = {}
        nodes_to_drop: set[int] = set()

        inits = {i.name for i in graph.initializer}
        init_map: dict[str, np.ndarray] = {
            i.name: numpy_helper.to_array(i) for i in graph.initializer
        }
        for sm in sm_nodes:
            scores_inp, v_inp = sm.input[0], sm.input[1]
            mm = node_by_output.get(scores_inp)
            scale_node_after_mm = None
            extra_attn_inputs: list[str] = []

            # Peel off Where (mask), Add (ALiBi / relative bias), Mul/Div (scaling)
            unwrapped = True
            while mm and unwrapped:
                unwrapped = False
                if mm.op_type == "Where" and len(mm.input) >= 2:
                    mask_cond = mm.input[0]
                    cand_scores = node_by_output.get(mm.input[1])
                    if cand_scores and cand_scores.op_type in ("MatMul", "Div", "Mul", "Add"):
                        extra_attn_inputs.append(mask_cond)
                        nodes_to_drop.add(id(mm))
                        mm = cand_scores
                        unwrapped = True
                        continue
                if mm.op_type == "Add" and len(mm.input) == 2:
                    cand0 = node_by_output.get(mm.input[0])
                    cand1 = node_by_output.get(mm.input[1])
                    if cand0 and cand0.op_type in ("MatMul", "Div", "Mul"):
                        extra_attn_inputs.append(mm.input[1])
                        nodes_to_drop.add(id(mm))
                        mm = cand0
                        unwrapped = True
                        continue
                    if cand1 and cand1.op_type in ("MatMul", "Div", "Mul"):
                        extra_attn_inputs.append(mm.input[0])
                        nodes_to_drop.add(id(mm))
                        mm = cand1
                        unwrapped = True
                        continue
                if mm.op_type in ("Div", "Mul") and len(mm.input) >= 2:
                    parent_mm = node_by_output.get(mm.input[0])
                    if parent_mm and parent_mm.op_type == "MatMul":
                        scale_op_name = mm.input[1]
                        if scale_op_name in init_map and init_map[scale_op_name].size > 1:
                            pass
                        else:
                            scale_node_after_mm = mm
                            mm = parent_mm
                            unwrapped = True
                            continue

            if not mm or mm.op_type != "MatMul":
                continue

            if scale_node_after_mm is not None:
                nodes_to_drop.add(id(scale_node_after_mm))

            q_cand = node_by_output.get(mm.input[0])
            k_cand = node_by_output.get(mm.input[1])

            # Resolve Q input
            has_scale_mul = scale_node_after_mm is not None
            if q_cand and q_cand.op_type in ("Mul", "Div"):
                has_scale_mul = True
                nodes_to_drop.add(id(q_cand))
                if q_cand.op_type == "Div":
                    q_inp = q_cand.input[0]
                else:
                    # Check which input is scale
                    in1_node = node_by_output.get(q_cand.input[1])
                    in0_node = node_by_output.get(q_cand.input[0])
                    if q_cand.input[0] in inits or (
                        in0_node and in0_node.op_type in ("Pow", "Constant")
                    ):
                        q_inp = q_cand.input[1]
                    else:
                        q_inp = q_cand.input[0]
            else:
                q_inp = mm.input[0]
                # Check if scale Mul/Div occurred before Transpose / Reshape
                curr = q_cand
                while (
                    curr
                    and curr.op_type in ("Transpose", "Reshape")
                    and len(curr.input) >= 1
                ):
                    parent = node_by_output.get(curr.input[0])
                    if parent and parent.op_type in ("Mul", "Div"):
                        has_scale_mul = True
                        if parent.op_type == "Div":
                            curr.input[0] = parent.input[0]
                        else:
                            in0_p = node_by_output.get(parent.input[0])
                            scale_cand = (
                                parent.input[0]
                                if (
                                    parent.input[0] in inits
                                    or (in0_p and in0_p.op_type in ("Pow", "Constant"))
                                )
                                else parent.input[1]
                            )
                            val_cand = (
                                parent.input[1]
                                if scale_cand == parent.input[0]
                                else parent.input[0]
                            )
                            curr.input[0] = val_cand
                        nodes_to_drop.add(id(parent))
                        break
                    curr = parent

            # Resolve K input
            if k_cand and k_cand.op_type == "Transpose":
                k_sub = node_by_output.get(k_cand.input[0])
                if k_sub and k_sub.op_type in ("Mul", "Div"):
                    nodes_to_drop.add(id(k_sub))
                    if k_sub.op_type == "Div":
                        k_raw = k_sub.input[0]
                    else:
                        in0_k = node_by_output.get(k_sub.input[0])
                        k_raw = (
                            k_sub.input[1]
                            if (
                                k_sub.input[0] in inits
                                or (in0_k and in0_k.op_type in ("Pow", "Constant"))
                            )
                            else k_sub.input[0]
                        )
                    k_inp = self._handle_k_transpose(k_cand, k_raw, nodes_to_drop)
                else:
                    k_inp = self._handle_k_transpose(
                        k_cand, k_cand.input[0], nodes_to_drop
                    )
            elif k_cand and k_cand.op_type in ("Mul", "Div"):
                nodes_to_drop.add(id(k_cand))
                if k_cand.op_type == "Div":
                    k_inp_cand = k_cand.input[0]
                else:
                    in0_k = node_by_output.get(k_cand.input[0])
                    k_inp_cand = (
                        k_cand.input[1]
                        if (
                            k_cand.input[0] in inits
                            or (in0_k and in0_k.op_type in ("Pow", "Constant"))
                        )
                        else k_cand.input[0]
                    )
                k_sub = node_by_output.get(k_inp_cand)
                if k_sub and k_sub.op_type == "Transpose":
                    k_inp = self._handle_k_transpose(
                        k_sub, k_sub.input[0], nodes_to_drop
                    )
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
                inputs=[q_inp, k_inp, v_inp] + extra_attn_inputs,
                outputs=list(sm.output),
                name=f"{sm.name}_sdpa",
                domain="custom.pytorch",
            )
            replacements[id(sm)] = sdpa
            nodes_to_drop.add(id(sm))
            nodes_to_drop.add(id(mm))

        if not replacements:
            return model

        kept = []
        for n in graph.node:
            if id(n) in replacements:
                kept.append(replacements[id(n)])
            elif id(n) not in nodes_to_drop:
                kept.append(n)
        del graph.node[:]
        graph.node.extend(kept)

        if mode == "function_proto":
            self._attach_sdpa_function_def(model)

        log.info(
            "ScaledDotProductAttention Fusion: Fused %d attention blocks",
            len(replacements),
        )
        return ShapeCanonicalizer().apply_dead_code_elimination(model)

    def _attach_sdpa_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for ScaledDotProductAttention."""
        func_name = "ScaledDotProductAttention"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", LATEST_OPSET)
        k_trans = helper.make_node(
            "Transpose", inputs=["K"], outputs=["KT"], perm=[0, 1, 3, 2]
        )
        mm1 = helper.make_node("MatMul", inputs=["Q", "KT"], outputs=["RawScores"])
        scale_val = helper.make_tensor(
            "scale_val", onnx.TensorProto.FLOAT, [1], [0.125]
        )
        c_scale = helper.make_node(
            "Constant", inputs=[], outputs=["scale_const"], value=scale_val
        )
        mul_scale = helper.make_node(
            "Mul", inputs=["RawScores", "scale_const"], outputs=["Scores"]
        )
        sm = helper.make_node("Softmax", inputs=["Scores"], outputs=["Probs"], axis=-1)
        mm2 = helper.make_node("MatMul", inputs=["Probs", "V"], outputs=["Y"])
        func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            nodes=[k_trans, mm1, c_scale, mul_scale, sm, mm2],
            opset_imports=[opset_standard],
            doc_string="Scaled Dot-Product Attention function",
        )
        model.functions.append(func)

    # -------------------------------------------------------------------------
    # SplitQKV Fusion
    # -------------------------------------------------------------------------
    def apply_split_qkv_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuse Reshape -> Transpose(perm=[2, 0, 3, 1, 4]) -> Unbind into SplitQKV."""
        graph = model.graph
        node_by_output: dict[str, onnx.NodeProto] = {
            out: n for n in graph.node for out in n.output
        }
        replacements: dict[int, onnx.NodeProto] = {}
        nodes_to_drop: set[int] = set()

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
                        num_heads = int(
                            numpy_helper.to_array(h_node.attribute[0].t).item()
                        )
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
            replacements[id(unbind)] = sqkv
            nodes_to_drop.add(id(unbind))
            nodes_to_drop.add(id(trans))
            nodes_to_drop.add(id(reshape))

        if not replacements:
            return model

        kept = []
        for n in graph.node:
            if id(n) in replacements:
                kept.append(replacements[id(n)])
            elif id(n) not in nodes_to_drop:
                kept.append(n)
        del graph.node[:]
        graph.node.extend(kept)

        if mode == "function_proto":
            self._attach_split_qkv_function_def(model)

        log.info("SplitQKV Fusion: Fused %d SplitQKV instances", len(replacements))
        return ShapeCanonicalizer().apply_dead_code_elimination(model)

    def _attach_split_qkv_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for SplitQKV."""
        func_name = "SplitQKV"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", LATEST_OPSET)
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
    def apply_merge_heads_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuse Transpose(perm=[0, 2, 1, 3]) and Reshape after GatedUnit into MergeHeads."""
        graph = model.graph
        consumer_map: dict[str, list[onnx.NodeProto]] = collections.defaultdict(list)
        for n in graph.node:
            for inp in n.input:
                consumer_map[inp].append(n)

        replacements: dict[int, onnx.NodeProto] = {}
        nodes_to_drop: set[int] = set()

        gated_units = [
            n
            for n in graph.node
            if n.op_type.startswith("GatedUnit")
            or n.op_type.startswith("ScaledDotProductAttention")
        ]
        for gu in gated_units:
            users = consumer_map.get(gu.output[0], [])
            trans = [u for u in users if u.op_type == "Transpose"]
            if not trans:
                continue
            t_node = trans[0]
            perm = next(
                (list(a.ints) for a in t_node.attribute if a.name == "perm"), []
            )
            if perm != [0, 2, 1, 3]:
                continue
            reshapes = [
                u
                for u in consumer_map.get(t_node.output[0], [])
                if u.op_type == "Reshape"
            ]
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
            replacements[id(r_node)] = mh
            nodes_to_drop.add(id(t_node))
            nodes_to_drop.add(id(r_node))

        if not replacements:
            return model

        kept = []
        for n in graph.node:
            if id(n) in replacements:
                kept.append(replacements[id(n)])
            elif id(n) not in nodes_to_drop:
                kept.append(n)
        del graph.node[:]
        graph.node.extend(kept)

        if mode == "function_proto":
            self._attach_merge_heads_function_def(model)

        log.info("MergeHeads Fusion: Fused %d MergeHeads instances", len(replacements))
        return ShapeCanonicalizer().apply_dead_code_elimination(model)

    def _attach_merge_heads_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for MergeHeads."""
        func_name = "MergeHeads"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", LATEST_OPSET)
        t = helper.make_node(
            "Transpose", inputs=["X"], outputs=["Y"], perm=[0, 2, 1, 3]
        )
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
    def apply_gated_attention_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
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
            if (
                not split_qkv
                or split_qkv.op_type != "SplitQKV"
                or len(split_qkv.output) < 3
            ):
                continue
            if (
                split_qkv.output[0] != q_in
                or split_qkv.output[1] != k_in
                or split_qkv.output[2] != v_tensor
            ):
                continue

            # 3. to_qkv
            to_qkv = node_by_output.get(split_qkv.input[0])
            if not to_qkv or to_qkv.op_type not in ("Linear", "MatMul"):
                continue

            # 4. Norm (RMSNorm or LayerNorm)
            norm = node_by_output.get(to_qkv.input[0])
            if not norm or norm.op_type not in (
                "RMSNorm",
                "RMSNormalization",
                "LayerNormalization",
            ):
                continue

            # 5. Downstream: GatedUnit
            sdpa_users = consumer_map.get(sdpa.output[0], [])
            if len(sdpa_users) != 1 or not sdpa_users[0].op_type.startswith(
                "GatedUnit"
            ):
                continue
            gated_unit = sdpa_users[0]
            gate_act = (
                gated_unit.op_type.split("_")[1]
                if "_" in gated_unit.op_type
                else "Sigmoid"
            )

            # to_gates: trace upstream through intermediate reshape / transpose / unsqueeze / rearrange
            gate_producer = node_by_output.get(gated_unit.input[1])
            intermediate_gates_nodes: list[onnx.NodeProto] = []
            curr = gate_producer
            while (
                curr
                and curr.op_type not in ("Linear", "MatMul")
                and curr.op_type
                in (
                    "Reshape",
                    "Transpose",
                    "AxialRearrange",
                    "Unsqueeze",
                    "Squeeze",
                )
            ):
                intermediate_gates_nodes.append(curr)
                curr = (
                    node_by_output.get(curr.input[0])
                    if curr.input
                    else None
                )

            to_gates = curr
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

            constituent_nodes = [
                norm,
                to_qkv,
                split_qkv,
                q_node,
                k_node,
                sdpa,
                to_gates,
                gated_unit,
                merge_heads,
                to_out,
            ]
            fp = compute_subgraph_fingerprint(constituent_nodes)
            specialized_attn_op = build_monomorphized_op_type(
                "GatedAttention", gate_act, fp
            )

            attn_inputs = [
                x,
                norm_gamma,
                qkv_w,
                qkv_b,
                gates_w,
                gates_b,
                theta_name,
                out_w,
                out_b,
            ]
            while attn_inputs and attn_inputs[-1] == "":
                attn_inputs.pop()

            gated_attn_node = helper.make_node(
                specialized_attn_op,
                inputs=attn_inputs,
                outputs=[final_out],
                name=f"{norm.name}_fused_gated_attention",
                domain="custom.pytorch",
                doc_string=f"Fused polymorphic Gated Multi-Head Attention ({gate_act}) block",
            )
            for a in norm.attribute:
                if a.name == "epsilon":
                    gated_attn_node.attribute.append(a)
            for a in split_qkv.attribute:
                if a.name == "num_heads":
                    gated_attn_node.attribute.append(a)

            new_nodes.append(gated_attn_node)
            nodes_to_drop.update(
                {
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
                }
            )
            for ign in intermediate_gates_nodes:
                nodes_to_drop.add(ign.name)

            if mode == "function_proto":
                self._attach_specialized_gated_attention_function(
                    model=model,
                    fname=specialized_attn_op,
                    gate_unit_op=gated_unit.op_type,
                    norm_op=norm.op_type,
                    fingerprint=fp,
                    num_inputs=len(attn_inputs),
                )

        if not new_nodes:
            return model

        kept = [n for n in graph.node if n.name not in nodes_to_drop]
        kept.extend(new_nodes)
        del graph.node[:]
        graph.node.extend(kept)

        custom_domain = "custom.pytorch"
        if not any(op.domain == custom_domain for op in model.opset_import):
            model.opset_import.append(helper.make_opsetid(custom_domain, 1))

        log.info(
            "GatedAttention Fusion: Fused %d GatedAttention instances with polymorphic fingerprints",
            len(new_nodes),
        )
        return ShapeCanonicalizer().apply_dead_code_elimination(model)

    def _attach_specialized_gated_attention_function(
        self,
        model: onnx.ModelProto,
        fname: str,
        gate_unit_op: str,
        norm_op: str,
        fingerprint: str,
        num_inputs: int = 9,
    ) -> None:
        """Attach formal, content-hashed ONNX FunctionProto definition for GatedAttention."""
        if any(
            f.name == fname and f.domain == "custom.pytorch" for f in model.functions
        ):
            return
        opset_standard = helper.make_opsetid("", LATEST_OPSET)
        opset_custom = helper.make_opsetid("custom.pytorch", 1)

        norm = helper.make_node(
            norm_op,
            inputs=["X", "norm_gamma"],
            outputs=["X_norm"],
            domain="custom.pytorch",
            epsilon=1e-12,
        )
        all_formal = [
            "X",
            "norm_gamma",
            "qkv_w",
            "qkv_b",
            "gates_w",
            "gates_b",
            "theta",
            "out_w",
            "out_b",
        ]
        formal_inputs = all_formal[:num_inputs]

        qkv_inputs = (
            ["X_norm", "qkv_w", "qkv_b"]
            if "qkv_b" in formal_inputs
            else ["X_norm", "qkv_w"]
        )
        to_qkv = helper.make_node(
            "Linear",
            inputs=qkv_inputs,
            outputs=["QKV"],
            domain="custom.pytorch",
        )
        gates_inputs = (
            ["X_norm", "gates_w", "gates_b"]
            if "gates_b" in formal_inputs
            else ["X_norm", "gates_w"]
        )
        to_gates = helper.make_node(
            "Linear",
            inputs=gates_inputs,
            outputs=["Gates"],
            domain="custom.pytorch",
        )
        split_qkv = helper.make_node(
            "SplitQKV", inputs=["QKV"], outputs=["Q", "K", "V"], domain="custom.pytorch"
        )
        rope_q = helper.make_node(
            "RotaryEmbedding",
            inputs=["Q", "theta"],
            outputs=["Q_rot"],
            domain="custom.pytorch",
        )
        rope_k = helper.make_node(
            "RotaryEmbedding",
            inputs=["K", "theta"],
            outputs=["K_rot"],
            domain="custom.pytorch",
        )
        sdpa = helper.make_node(
            "ScaledDotProductAttention",
            inputs=["Q_rot", "K_rot", "V"],
            outputs=["AttnScores"],
            domain="custom.pytorch",
        )
        gated = helper.make_node(
            gate_unit_op,
            inputs=["AttnScores", "Gates"],
            outputs=["GatedScores"],
            domain="custom.pytorch",
        )
        merge = helper.make_node(
            "MergeHeads",
            inputs=["GatedScores"],
            outputs=["Merged"],
            domain="custom.pytorch",
        )
        out_inputs = (
            ["Merged", "out_w", "out_b"] if "out_b" in formal_inputs else ["Merged", "out_w"]
        )
        to_out = helper.make_node(
            "Linear", inputs=out_inputs, outputs=["Y"], domain="custom.pytorch"
        )

        ensure_specialized_function_proto(
            model=model,
            domain="custom.pytorch",
            fname=fname,
            formal_inputs=formal_inputs,
            formal_outputs=["Y"],
            body_nodes=[
                norm,
                to_qkv,
                to_gates,
                split_qkv,
                rope_q,
                rope_k,
                sdpa,
                gated,
                merge,
                to_out,
            ],
            opset_imports=[opset_standard, opset_custom],
            doc_string=f"Content-hashed polymorphic GatedAttention block [{fingerprint}]",
        )

    # -------------------------------------------------------------------------
    # Gemma-2 Logits Soft-Capping Fusion
    # -------------------------------------------------------------------------
    def apply_gemma2_softcapping_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuse Gemma-2 style Logits Soft-Capping: cap * tanh(scores / cap) into LogitsSoftCap."""
        graph = model.graph
        node_by_output: dict[str, onnx.NodeProto] = {
            out: n for n in graph.node for out in n.output
        }
        init_map = {
            init.name: numpy_helper.to_array(init) for init in graph.initializer
        }

        nodes_to_drop: set[str] = set()
        replacements: dict[str, onnx.NodeProto] = {}

        for n in graph.node:
            if n.op_type != "Mul" or len(n.input) < 2:
                continue

            tanh_cand = None
            cap_val = None
            for i in (0, 1):
                inp, other = n.input[i], n.input[1 - i]
                parent = node_by_output.get(inp)
                if (
                    parent
                    and parent.op_type == "Tanh"
                    and len(parent.input) >= 1
                ):
                    tanh_cand = parent
                    if other in init_map:
                        arr = init_map[other]
                        if arr.size == 1:
                            cap_val = float(arr.item())
                    else:
                        c_node = node_by_output.get(other)
                        if c_node and c_node.op_type == "Constant":
                            for a in c_node.attribute:
                                if a.name == "value":
                                    try:
                                        arr = numpy_helper.to_array(a.t)
                                        if arr.size == 1:
                                            cap_val = float(arr.item())
                                    except Exception:
                                        pass
                    break

            if tanh_cand is None or cap_val is None:
                continue

            div_cand = node_by_output.get(tanh_cand.input[0])
            if not div_cand or len(div_cand.input) < 2:
                continue

            scores_input = None
            if div_cand.op_type == "Div":
                divisor = div_cand.input[1]
                div_val = None
                if divisor in init_map:
                    arr = init_map[divisor]
                    if arr.size == 1:
                        div_val = float(arr.item())
                else:
                    c_node = node_by_output.get(divisor)
                    if c_node and c_node.op_type == "Constant":
                        for a in c_node.attribute:
                            if a.name == "value":
                                try:
                                    arr = numpy_helper.to_array(a.t)
                                    if arr.size == 1:
                                        div_val = float(arr.item())
                                except Exception:
                                    pass
                if div_val is not None and abs(div_val - cap_val) < 1e-3:
                    scores_input = div_cand.input[0]

            elif div_cand.op_type == "Mul":
                for j in (0, 1):
                    sinp, sother = div_cand.input[j], div_cand.input[1 - j]
                    inv_val = None
                    if sother in init_map:
                        arr = init_map[sother]
                        if arr.size == 1:
                            inv_val = float(arr.item())
                    if (
                        inv_val is not None
                        and abs(inv_val - (1.0 / cap_val)) < 1e-3
                    ):
                        scores_input = sinp
                        break

            if scores_input is None:
                continue

            out_name = n.output[0]
            fused_node = helper.make_node(
                "LogitsSoftCap",
                inputs=[scores_input],
                outputs=[out_name],
                name=f"{n.name or out_name}_softcap",
                domain="custom.pytorch",
                softcap=float(cap_val),
            )
            replacements[out_name] = fused_node
            nodes_to_drop.add(n.name or out_name)
            nodes_to_drop.add(tanh_cand.name or tanh_cand.output[0])
            nodes_to_drop.add(div_cand.name or div_cand.output[0])

        if not replacements:
            return model

        new_nodes = []
        for n in graph.node:
            out = n.output[0] if n.output else ""
            if out in replacements:
                new_nodes.append(replacements[out])
            elif (n.name or out) not in nodes_to_drop:
                new_nodes.append(n)

        del graph.node[:]
        graph.node.extend(new_nodes)

        if mode == "function_proto":
            self._attach_logits_softcap_function_def(model)

        log.info(
            "Gemma-2 SoftCapping Fusion: Fused %d soft-capping blocks",
            len(replacements),
        )
        return ShapeCanonicalizer().apply_dead_code_elimination(model)

    def _attach_logits_softcap_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for LogitsSoftCap."""
        func_name = "LogitsSoftCap"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", LATEST_OPSET)
        c_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["cap_val"],
            value=helper.make_tensor(
                "cap_val", onnx.TensorProto.FLOAT, [1], [50.0]
            ),
        )
        div_node = helper.make_node(
            "Div", inputs=["x", "cap_val"], outputs=["div_out"]
        )
        tanh_node = helper.make_node(
            "Tanh", inputs=["div_out"], outputs=["tanh_out"]
        )
        mul_node = helper.make_node(
            "Mul", inputs=["tanh_out", "cap_val"], outputs=["y"]
        )
        func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["x"],
            outputs=["y"],
            nodes=[c_node, div_node, tanh_node, mul_node],
            opset_imports=[opset_standard],
            doc_string="Gemma-2 style logits soft-capping function: cap * tanh(x / cap)",
        )
        model.functions.append(func)

    # -------------------------------------------------------------------------
    # DeepSeek MLA (Multi-Head Latent Attention) Fusion
    # -------------------------------------------------------------------------
    def apply_deepseek_mla_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Synthesize DeepSeek MLA low-rank KV latent projection and expansion."""
        graph = model.graph
        consumer_map: dict[str, list[onnx.NodeProto]] = {}
        for n in graph.node:
            for inp in n.input:
                consumer_map.setdefault(inp, []).append(n)

        linear_ops = ("Gemm", "MatMul", "Linear")
        replacements: dict[str, onnx.NodeProto] = {}
        nodes_to_drop: set[str] = set()

        for n in graph.node:
            if n.op_type not in linear_ops or len(n.output) != 1:
                continue

            c_kv = n.output[0]
            consumers = consumer_map.get(c_kv, [])
            up_proj_consumers = [c for c in consumers if c.op_type in linear_ops]
            if len(up_proj_consumers) == 2:
                up_k, up_v = up_proj_consumers[0], up_proj_consumers[1]
                out_k = up_k.output[0]
                out_v = up_v.output[0]

                mla_kv_node = helper.make_node(
                    "MultiHeadLatentAttention_KVProjection",
                    inputs=[n.input[0], n.input[1], up_k.input[1], up_v.input[1]],
                    outputs=[out_k, out_v],
                    name=f"{n.name or c_kv}_mla_kv",
                    domain="custom.pytorch",
                )
                replacements[out_k] = mla_kv_node
                nodes_to_drop.add(n.name or c_kv)
                nodes_to_drop.add(up_k.name or out_k)
                nodes_to_drop.add(up_v.name or out_v)

        if not replacements:
            return model

        new_nodes = []
        for n in graph.node:
            out = n.output[0] if n.output else ""
            if out in replacements:
                new_nodes.append(replacements[out])
            elif (n.name or out) not in nodes_to_drop:
                new_nodes.append(n)

        del graph.node[:]
        graph.node.extend(new_nodes)

        log.info(
            "DeepSeek MLA Fusion: Synthesized %d MLA low-rank KV blocks",
            len(replacements),
        )
        return ShapeCanonicalizer().apply_dead_code_elimination(model)
