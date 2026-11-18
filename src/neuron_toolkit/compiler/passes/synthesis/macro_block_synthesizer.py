"""Macro architecture block synthesis passes."""

from __future__ import annotations

import logging

import onnx
from onnx import helper

from neuron_toolkit.compiler.architectures.fingerprint import (
    build_monomorphized_op_type,
    compute_subgraph_fingerprint,
    ensure_specialized_function_proto,
)
from neuron_toolkit.compiler.base import CompilerPass
from neuron_toolkit.compiler.passes.canonicalization.shape_canonicalizer import (
    ShapeCanonicalizer,
)

log = logging.getLogger(__name__)


class MacroBlockSynthesizer(CompilerPass):
    """Synthesizes high-level macro architectural blocks (ResMLP, SGU, SE, Highway, MHA, ConvNeXt)."""

    @property
    def name(self) -> str:
        return "macro_block_synthesizer"

    def run(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Run macro block synthesis passes sequentially."""
        model = self.apply_deep_mlp_norm_act_fusion(model, mode=mode)
        model = self.apply_spatial_gating_unit_fusion(model, mode=mode)
        model = self.apply_highway_block_fusion(model, mode=mode)
        model = self.apply_squeeze_excitation_fusion(model, mode=mode)
        model = self.apply_mha_composite_fusion(model, mode=mode)
        return model

    # -------------------------------------------------------------------------
    # 1. DeepMLP_Norm_Act Fusion: LayerNorm / RMSNorm -> Linear -> Gelu
    # -------------------------------------------------------------------------
    def apply_deep_mlp_norm_act_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuse Norm -> Linear -> Gelu into DeepMLP_Norm_Act."""
        graph = model.graph
        node_by_output: dict[str, onnx.NodeProto] = {
            out: n for n in graph.node for out in n.output
        }
        consumer_map: dict[str, list[onnx.NodeProto]] = {}
        for n in graph.node:
            for inp in n.input:
                consumer_map.setdefault(inp, []).append(n)
        graph_outputs = {out.name for out in graph.output}

        nodes_to_drop: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        count = 0

        for n in graph.node:
            if n.op_type in ("Gelu", "FastGelu", "Relu", "SiLU") and len(n.input) >= 1:
                lin_node = node_by_output.get(n.input[0])
                if not lin_node or lin_node.op_type not in ("Linear", "Gemm", "MatMul"):
                    continue

                # Check if lin_node consumed a Norm
                norm_node = node_by_output.get(lin_node.input[0])
                if not norm_node or norm_node.op_type not in (
                    "LayerNormalization",
                    "RMSNorm",
                    "SimplifiedLayerNormalization",
                ):
                    continue

                # Ensure linear is only consumed by this activation
                if (
                    len(consumer_map.get(lin_node.output[0], [])) > 1
                    or lin_node.output[0] in graph_outputs
                ):
                    continue

                act_name = n.op_type
                nodes_to_drop.update({n.name, lin_node.name})
                if (
                    len(consumer_map.get(norm_node.output[0], [])) <= 1
                    and norm_node.output[0] not in graph_outputs
                ):
                    nodes_to_drop.add(norm_node.name)
                    inputs = list(norm_node.input) + list(lin_node.input[1:])
                    has_norm = 1
                    sub_nodes = [norm_node, lin_node, n]
                else:
                    inputs = [norm_node.output[0]] + list(lin_node.input[1:])
                    has_norm = 0
                    sub_nodes = [lin_node, n]

                fp = compute_subgraph_fingerprint(sub_nodes)
                specialized_op = build_monomorphized_op_type("DeepMLP", act_name, fp)

                fused = helper.make_node(
                    specialized_op,
                    inputs=inputs,
                    outputs=[n.output[0]],
                    name=f"{n.name}_fused_mlp_norm_act",
                    domain="custom.pytorch",
                    activation=act_name,
                    has_norm=has_norm,
                )
                new_nodes.append(fused)
                count += 1

                if mode == "function_proto":
                    self._attach_specialized_deep_mlp_function(
                        model=model,
                        fname=specialized_op,
                        act_name=act_name,
                        norm_op=norm_node.op_type if has_norm else None,
                        has_norm=has_norm,
                        fingerprint=fp,
                        num_inputs=len(inputs),
                    )

        if not new_nodes:
            return model

        kept = [n for n in graph.node if n.name not in nodes_to_drop] + new_nodes
        del graph.node[:]
        graph.node.extend(kept)

        log.info(
            "DeepMLP_Norm_Act Fusion: Fused %d blocks with polymorphic fingerprints",
            count,
        )
        return ShapeCanonicalizer().apply_dead_code_elimination(model)

    def _attach_specialized_deep_mlp_function(
        self,
        model: onnx.ModelProto,
        fname: str,
        act_name: str,
        norm_op: str | None,
        has_norm: int,
        fingerprint: str,
        num_inputs: int = 5,
    ) -> None:
        """Attach formal, content-hashed ONNX FunctionProto definition with exact inner activation and norm."""
        if any(
            f.name == fname and f.domain == "custom.pytorch" for f in model.functions
        ):
            return

        opset_standard = helper.make_opsetid("", 14)
        if has_norm and norm_op:
            if num_inputs == 3:
                ln_node = helper.make_node(
                    norm_op, inputs=["x", "gamma"], outputs=["norm_out"]
                )
                lin_node = helper.make_node(
                    "MatMul", inputs=["norm_out", "w"], outputs=["lin_out"]
                )
                act_node = helper.make_node(act_name, inputs=["lin_out"], outputs=["y"])
                body_nodes = [ln_node, lin_node, act_node]
                formal_inputs = ["x", "gamma", "w"]
            elif num_inputs == 4:
                ln_node = helper.make_node(
                    norm_op, inputs=["x", "gamma"], outputs=["norm_out"]
                )
                lin_node = helper.make_node(
                    "MatMul", inputs=["norm_out", "w"], outputs=["mm_out"]
                )
                add_node = helper.make_node(
                    "Add", inputs=["mm_out", "b"], outputs=["lin_out"]
                )
                act_node = helper.make_node(act_name, inputs=["lin_out"], outputs=["y"])
                body_nodes = [ln_node, lin_node, add_node, act_node]
                formal_inputs = ["x", "gamma", "w", "b"]
            else:
                ln_node = helper.make_node(
                    norm_op, inputs=["x", "gamma", "beta"], outputs=["norm_out"]
                )
                lin_node = helper.make_node(
                    "MatMul", inputs=["norm_out", "w"], outputs=["mm_out"]
                )
                add_node = helper.make_node(
                    "Add", inputs=["mm_out", "b"], outputs=["lin_out"]
                )
                act_node = helper.make_node(act_name, inputs=["lin_out"], outputs=["y"])
                body_nodes = [ln_node, lin_node, add_node, act_node]
                formal_inputs = ["x", "gamma", "beta", "w", "b"]
        elif num_inputs == 2:
            lin_node = helper.make_node(
                "MatMul", inputs=["x", "w"], outputs=["lin_out"]
            )
            act_node = helper.make_node(act_name, inputs=["lin_out"], outputs=["y"])
            body_nodes = [lin_node, act_node]
            formal_inputs = ["x", "w"]
        else:
            lin_node = helper.make_node("MatMul", inputs=["x", "w"], outputs=["mm_out"])
            add_node = helper.make_node(
                "Add", inputs=["mm_out", "b"], outputs=["lin_out"]
            )
            act_node = helper.make_node(act_name, inputs=["lin_out"], outputs=["y"])
            body_nodes = [lin_node, add_node, act_node]
            formal_inputs = ["x", "w", "b"]

        ensure_specialized_function_proto(
            model=model,
            domain="custom.pytorch",
            fname=fname,
            formal_inputs=formal_inputs,
            formal_outputs=["y"],
            body_nodes=body_nodes,
            opset_imports=[opset_standard],
            doc_string=f"Content-hashed polymorphic DeepMLP ({act_name}) block [{fingerprint}]",
        )

    # -------------------------------------------------------------------------
    # 2. Spatial Gating Unit (gMLP SGU) Fusion
    # -------------------------------------------------------------------------
    def apply_spatial_gating_unit_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuse Split -> Norm -> Linear -> Mul into SpatialGatingUnit."""
        graph = model.graph
        node_by_output: dict[str, onnx.NodeProto] = {
            out: n for n in graph.node for out in n.output
        }
        nodes_to_drop: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        count = 0

        for n in graph.node:
            if n.op_type == "Mul" and len(n.input) == 2:
                in0_node = node_by_output.get(n.input[0])
                in1_node = node_by_output.get(n.input[1])
                lin_cand = None
                other_inp = None
                if in0_node and in0_node.op_type in ("Linear", "MatMul"):
                    lin_cand = in0_node
                    other_inp = n.input[1]
                elif in1_node and in1_node.op_type in ("Linear", "MatMul"):
                    lin_cand = in1_node
                    other_inp = n.input[0]

                if lin_cand:
                    norm_cand = node_by_output.get(lin_cand.input[0])
                    # Check if norm_cand and other_inp trace to the same Split node
                    split_cand = None
                    if norm_cand and norm_cand.op_type in (
                        "LayerNormalization",
                        "RMSNorm",
                        "SimplifiedLayerNormalization",
                    ):
                        split_cand = node_by_output.get(norm_cand.input[0])
                    else:
                        split_cand = node_by_output.get(lin_cand.input[0])

                    other_split = node_by_output.get(other_inp)
                    if (
                        split_cand
                        and other_split
                        and (
                            split_cand.name == other_split.name
                            or split_cand == other_split
                        )
                        and split_cand.op_type in ("Split", "SplitQKV")
                    ):
                        nodes_to_drop.update({n.name, lin_cand.name, split_cand.name})
                        norm_inputs = []
                        if norm_cand:
                            nodes_to_drop.add(norm_cand.name)
                            norm_inputs = list(norm_cand.input[1:])

                        sgu_node = helper.make_node(
                            "SpatialGatingUnit",
                            inputs=[split_cand.input[0]]
                            + norm_inputs
                            + list(lin_cand.input[1:]),
                            outputs=[n.output[0]],
                            name=f"{n.name}_fused_sgu",
                            domain="custom.pytorch",
                        )
                        new_nodes.append(sgu_node)
                        count += 1

        if not new_nodes:
            return model

        kept = [n for n in graph.node if n.name not in nodes_to_drop] + new_nodes
        del graph.node[:]
        graph.node.extend(kept)

        if mode == "function_proto":
            self._attach_sgu_function_def(model)

        log.info("SpatialGatingUnit Fusion: Fused %d blocks", count)
        return ShapeCanonicalizer().apply_dead_code_elimination(model)

    def _attach_sgu_function_def(self, model: onnx.ModelProto) -> None:
        func_name = "SpatialGatingUnit"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", 14)

        sgu_nodes = [
            n
            for n in model.graph.node
            if n.op_type == func_name and (n.domain == domain or not n.domain)
        ]
        max_inputs = max([len(n.input) for n in sgu_nodes], default=2)

        sp_node = helper.make_node("Split", inputs=["x"], outputs=["u", "v"], axis=-1)
        if max_inputs == 2:
            formal_inputs = ["x", "w"]
            ln_node = helper.make_node(
                "LayerNormalization", inputs=["v"], outputs=["v_norm"]
            )
            mm_node = helper.make_node(
                "MatMul", inputs=["v_norm", "w"], outputs=["gate"]
            )
            body_nodes = [sp_node, ln_node, mm_node]
        elif max_inputs == 3:
            formal_inputs = ["x", "gamma", "w"]
            ln_node = helper.make_node(
                "LayerNormalization", inputs=["v", "gamma"], outputs=["v_norm"]
            )
            mm_node = helper.make_node(
                "MatMul", inputs=["v_norm", "w"], outputs=["gate"]
            )
            body_nodes = [sp_node, ln_node, mm_node]
        elif max_inputs == 4:
            formal_inputs = ["x", "gamma", "w", "b"]
            ln_node = helper.make_node(
                "LayerNormalization", inputs=["v", "gamma"], outputs=["v_norm"]
            )
            mm_node = helper.make_node("MatMul", inputs=["v_norm", "w"], outputs=["mm"])
            add_node = helper.make_node("Add", inputs=["mm", "b"], outputs=["gate"])
            body_nodes = [sp_node, ln_node, mm_node, add_node]
        else:
            formal_inputs = ["x", "gamma", "beta", "w", "b"]
            ln_node = helper.make_node(
                "LayerNormalization", inputs=["v", "gamma", "beta"], outputs=["v_norm"]
            )
            mm_node = helper.make_node("MatMul", inputs=["v_norm", "w"], outputs=["mm"])
            add_node = helper.make_node("Add", inputs=["mm", "b"], outputs=["gate"])
            body_nodes = [sp_node, ln_node, mm_node, add_node]

        mul_node = helper.make_node("Mul", inputs=["u", "gate"], outputs=["y"])
        body_nodes.append(mul_node)

        func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=formal_inputs,
            outputs=["y"],
            nodes=body_nodes,
            opset_imports=[opset_standard],
            doc_string="Spatial Gating Unit (gMLP)",
        )
        model.functions.append(func)

    # -------------------------------------------------------------------------
    # 3. Highway Block Fusion
    # -------------------------------------------------------------------------
    def apply_highway_block_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuse Transform Gate + Carry Gate + Addition into HighwayBlock."""
        graph = model.graph
        node_by_output: dict[str, onnx.NodeProto] = {
            out: n for n in graph.node for out in n.output
        }
        nodes_to_drop: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        count = 0

        for n in graph.node:
            if n.op_type == "Add" and len(n.input) == 2:
                # Check if one input is GatedUnit or Mul (h * t) and the other is Mul (x * (1 - t))
                m1 = node_by_output.get(n.input[0])
                m2 = node_by_output.get(n.input[1])
                if not m1 or not m2:
                    continue

                # Case: GatedUnit + Mul(x, 1 - t)
                gated_cand = (
                    m1
                    if m1.op_type == "GatedUnit"
                    else (m2 if m2.op_type == "GatedUnit" else None)
                )
                carry_mul = (
                    m2 if gated_cand == m1 else (m1 if gated_cand == m2 else None)
                )

                if gated_cand and carry_mul and carry_mul.op_type == "Mul":
                    sub_cand = node_by_output.get(
                        carry_mul.input[0]
                    ) or node_by_output.get(carry_mul.input[1])
                    if sub_cand and sub_cand.op_type == "Sub":
                        carry_x = (
                            carry_mul.input[1]
                            if sub_cand.output[0] == carry_mul.input[0]
                            else carry_mul.input[0]
                        )
                        nodes_to_drop.update(
                            {n.name, gated_cand.name, carry_mul.name, sub_cand.name}
                        )
                        hw_node = helper.make_node(
                            "HighwayBlock",
                            inputs=[gated_cand.input[0], gated_cand.input[1], carry_x],
                            outputs=[n.output[0]],
                            name=f"{n.name}_fused_highway",
                            domain="custom.pytorch",
                        )
                        new_nodes.append(hw_node)
                        count += 1

        if not new_nodes:
            return model

        kept = [n for n in graph.node if n.name not in nodes_to_drop] + new_nodes
        del graph.node[:]
        graph.node.extend(kept)

        if mode == "function_proto":
            self._attach_highway_function_def(model)

        log.info("HighwayBlock Fusion: Fused %d blocks", count)
        return ShapeCanonicalizer().apply_dead_code_elimination(model)

    def _attach_highway_function_def(self, model: onnx.ModelProto) -> None:
        func_name = "HighwayBlock"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", 14)
        sig_node = helper.make_node("Sigmoid", inputs=["gate_in"], outputs=["t"])
        c_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["one"],
            value=helper.make_tensor("one", onnx.TensorProto.FLOAT, [], [1.0]),
        )
        sub_node = helper.make_node("Sub", inputs=["one", "t"], outputs=["carry"])
        m1_node = helper.make_node("Mul", inputs=["h", "t"], outputs=["out_h"])
        m2_node = helper.make_node("Mul", inputs=["x", "carry"], outputs=["out_c"])
        add_node = helper.make_node("Add", inputs=["out_h", "out_c"], outputs=["y"])
        func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["h", "gate_in", "x"],
            outputs=["y"],
            nodes=[sig_node, c_node, sub_node, m1_node, m2_node, add_node],
            opset_imports=[opset_standard],
            doc_string="Highway Transform & Carry Block",
        )
        model.functions.append(func)

    # -------------------------------------------------------------------------
    # 4. Squeeze-and-Excitation (SE) Fusion
    # -------------------------------------------------------------------------
    def apply_squeeze_excitation_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuse Global Pooling -> Linear/Conv -> ReLU -> Linear/Conv -> Sigmoid -> Mul into SqueezeExcitation."""
        graph = model.graph
        node_by_output: dict[str, onnx.NodeProto] = {
            out: n for n in graph.node for out in n.output
        }
        nodes_to_drop: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        count = 0

        for n in graph.node:
            if n.op_type == "Mul" and len(n.input) == 2:
                sig_cand = node_by_output.get(n.input[0]) or node_by_output.get(
                    n.input[1]
                )
                other_inp = (
                    n.input[1]
                    if sig_cand and sig_cand.output[0] == n.input[0]
                    else n.input[0]
                )

                if sig_cand and sig_cand.op_type in ("Sigmoid", "HardSigmoid"):
                    fc2_cand = node_by_output.get(sig_cand.input[0])
                    if fc2_cand and fc2_cand.op_type in ("Linear", "Conv", "MatMul"):
                        act_cand = node_by_output.get(fc2_cand.input[0])
                        if act_cand and act_cand.op_type in (
                            "Relu",
                            "HardSwish",
                            "SiLU",
                        ):
                            fc1_cand = node_by_output.get(act_cand.input[0])
                            if fc1_cand and fc1_cand.op_type in (
                                "Linear",
                                "Conv",
                                "MatMul",
                            ):
                                pool_cand = node_by_output.get(fc1_cand.input[0])
                                if pool_cand and pool_cand.op_type in (
                                    "GlobalAveragePool",
                                    "ReduceMean",
                                ):
                                    # Ensure pool_cand takes other_inp as its original input
                                    if pool_cand.input[0] == other_inp:
                                        nodes_to_drop.update(
                                            {
                                                n.name,
                                                sig_cand.name,
                                                fc2_cand.name,
                                                act_cand.name,
                                                fc1_cand.name,
                                                pool_cand.name,
                                            }
                                        )
                                        se_node = helper.make_node(
                                            "SqueezeExcitation",
                                            inputs=[other_inp]
                                            + list(fc1_cand.input[1:])
                                            + list(fc2_cand.input[1:]),
                                            outputs=[n.output[0]],
                                            name=f"{n.name}_fused_se",
                                            domain="custom.pytorch",
                                        )
                                        new_nodes.append(se_node)
                                        count += 1

        if not new_nodes:
            return model

        kept = [n for n in graph.node if n.name not in nodes_to_drop] + new_nodes
        del graph.node[:]
        graph.node.extend(kept)

        if mode == "function_proto":
            self._attach_se_function_def(model)

        log.info("SqueezeExcitation Fusion: Fused %d blocks", count)
        return ShapeCanonicalizer().apply_dead_code_elimination(model)

    def _attach_se_function_def(self, model: onnx.ModelProto) -> None:
        func_name = "SqueezeExcitation"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", 14)
        pool = helper.make_node("GlobalAveragePool", inputs=["x"], outputs=["pooled"])
        fc1_mm = helper.make_node("MatMul", inputs=["pooled", "w1"], outputs=["fc1_mm"])
        fc1 = helper.make_node("Add", inputs=["fc1_mm", "b1"], outputs=["fc1_out"])
        relu = helper.make_node("Relu", inputs=["fc1_out"], outputs=["act_out"])
        fc2_mm = helper.make_node(
            "MatMul", inputs=["act_out", "w2"], outputs=["fc2_mm"]
        )
        fc2 = helper.make_node("Add", inputs=["fc2_mm", "b2"], outputs=["fc2_out"])
        sig = helper.make_node("Sigmoid", inputs=["fc2_out"], outputs=["scale"])
        mul = helper.make_node("Mul", inputs=["x", "scale"], outputs=["y"])
        func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["x", "w1", "b1", "w2", "b2"],
            outputs=["y"],
            nodes=[pool, fc1_mm, fc1, relu, fc2_mm, fc2, sig, mul],
            opset_imports=[opset_standard],
            doc_string="Squeeze-and-Excitation channel gating",
        )
        model.functions.append(func)

    # -------------------------------------------------------------------------
    # 5. Composite MultiHeadAttention Fusion
    # -------------------------------------------------------------------------
    def apply_mha_composite_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuse SplitQKV -> SDPA -> MergeHeads -> Linear out into MultiHeadAttention."""
        graph = model.graph
        node_by_output: dict[str, onnx.NodeProto] = {
            out: n for n in graph.node for out in n.output
        }
        nodes_to_drop: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        count = 0

        for n in graph.node:
            if n.op_type == "Linear" and len(n.input) >= 2:
                merge_cand = node_by_output.get(n.input[0])
                if merge_cand and merge_cand.op_type == "MergeHeads":
                    sdpa_cand = node_by_output.get(merge_cand.input[0])
                    if sdpa_cand and sdpa_cand.op_type in (
                        "ScaledDotProductAttention",
                        "Attention",
                    ):
                        q_node = node_by_output.get(sdpa_cand.input[0])
                        # If Q came from SplitQKV
                        split_cand = None
                        if q_node and q_node.op_type == "SplitQKV":
                            split_cand = q_node

                        if split_cand:
                            nodes_to_drop.update(
                                {
                                    n.name,
                                    merge_cand.name,
                                    sdpa_cand.name,
                                    split_cand.name,
                                }
                            )
                            mha_node = helper.make_node(
                                "MultiHeadAttention",
                                inputs=[split_cand.input[0]] + list(n.input[1:]),
                                outputs=[n.output[0]],
                                name=f"{n.name}_fused_mha_composite",
                                domain="custom.pytorch",
                            )
                            new_nodes.append(mha_node)
                            count += 1

        if not new_nodes:
            return model

        kept = [n for n in graph.node if n.name not in nodes_to_drop] + new_nodes
        del graph.node[:]
        graph.node.extend(kept)

        if mode == "function_proto":
            self._attach_mha_composite_function_def(model)

        log.info("MultiHeadAttention Composite Fusion: Fused %d blocks", count)
        return ShapeCanonicalizer().apply_dead_code_elimination(model)

    def _attach_mha_composite_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for MultiHeadAttention."""
        func_name = "MultiHeadAttention"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", 14)
        opset_custom = helper.make_opsetid("custom.pytorch", 1)

        sp = helper.make_node(
            "SplitQKV", inputs=["x"], outputs=["q", "k", "v"], domain=domain
        )
        sdpa = helper.make_node(
            "ScaledDotProductAttention",
            inputs=["q", "k", "v"],
            outputs=["attn_out"],
            domain=domain,
        )
        mh = helper.make_node(
            "MergeHeads", inputs=["attn_out"], outputs=["merged"], domain=domain
        )
        lin = helper.make_node(
            "Linear", inputs=["merged", "w_out"], outputs=["y"], domain=domain
        )

        func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["x", "w_out"],
            outputs=["y"],
            nodes=[sp, sdpa, mh, lin],
            opset_imports=[opset_standard, opset_custom],
            doc_string="Multi-Head Attention composite block",
        )
        model.functions.append(func)
