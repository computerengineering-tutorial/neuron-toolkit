"""Declarative JSON-Driven Macro Architecture Block Synthesis Passes.

Loads declarative JSON pattern rules from `neuron_toolkit/rules/definitions/` and
uses the `Pattern` DSL and `PatternDetector` pattern-matching engine to detect and
fuse complex macro-architectural motifs:
1. DeepMLP_Norm_Act (RMSNorm/LayerNorm -> Linear -> Gelu/Relu/SiLU)
2. SpatialGatingUnit (Split -> Norm -> Linear -> Mul)
3. HighwayBlock (Transform branch & Carry gate)
4. SqueezeExcitation (GAP -> Linear -> ReLU -> Linear -> Sigmoid -> Mul)
5. ResidualBlock (Pre-Activation ResNet: BN -> Conv -> ReLU -> BN -> Conv + Residual -> ReLU)
6. AudioLogMelCompressor (Clip -> Div -> Log -> Div -> Mul -> Add -> Clip)
7. MultiHeadAttention (Linear -> SplitQKV -> SDPA -> MergeHeads -> Linear)
8. GatedMultiHeadAttention (RMSNorm -> GAU -> MergeHeads -> Linear)
9. TransformerBlock (Attention + Residual Add + FeedForward + Residual Add)
10. PyTorch TransformerEncoderLayer (scoped layer regex fusion)
11. MultiFieldEmbedding (cascading multi-token table Gather + Add tree)

Attaches formal ONNX FunctionProto definitions for Netron expandable hierarchical visualization.
"""

from __future__ import annotations

import collections
import copy
import json
import logging
import re
from pathlib import Path

import onnx
from onnx import helper

from neuron_toolkit.compiler.architectures.fingerprint import (
    build_monomorphized_op_type,
    compute_subgraph_fingerprint,
    ensure_specialized_function_proto,
)
from neuron_toolkit.compiler.base import CompilerPass, LATEST_OPSET
from neuron_toolkit.compiler.passes.canonicalization.shape_canonicalizer import (
    ShapeCanonicalizer,
)
from neuron_toolkit.pattern.detector import PatternDetector
from neuron_toolkit.pattern.dsl import Pattern
from neuron_toolkit.rules.schema import JSONRule

log = logging.getLogger(__name__)


def _ensure_custom_domain(model: onnx.ModelProto, domain: str = "custom.pytorch") -> None:
    """Ensure custom domain exists in opset_import."""
    if not any(op.domain == domain for op in model.opset_import):
        model.opset_import.append(helper.make_opsetid(domain, 1))


def _replace_nodes_topologically(
    graph: onnx.GraphProto,
    nodes_to_drop: set[str | int],
    new_nodes: list[onnx.NodeProto],
) -> None:
    """Replace dropped nodes in-place at the topological exit positions."""
    clean_names = {x for x in nodes_to_drop if isinstance(x, str) and x}
    drop_ids = {x for x in nodes_to_drop if isinstance(x, int)}
    new_node_map = {out: n for n in new_nodes for out in n.output if out}
    new_graph_nodes = []
    for n in graph.node:
        is_dropped = False
        if n.name and n.name in clean_names:
            is_dropped = True
        elif id(n) in drop_ids:
            is_dropped = True

        if is_dropped:
            for out in n.output:
                if out in new_node_map:
                    new_node = new_node_map.pop(out)
                    new_graph_nodes.append(new_node)
                    for other_out in new_node.output:
                        new_node_map.pop(other_out, None)
                    break
        else:
            new_graph_nodes.append(n)
    new_graph_nodes.extend(new_node_map.values())
    del graph.node[:]
    graph.node.extend(new_graph_nodes)


def _toposort_nodes(nodes: list[onnx.NodeProto]) -> list[onnx.NodeProto]:
    """Sort an arbitrary list of nodes topologically according to internal tensor dependencies."""
    node_map = {id(n): n for n in nodes}
    out_to_node_id = {out: id(n) for n in nodes for out in n.output if out}
    dep_graph = collections.defaultdict(list)
    in_degree = collections.defaultdict(int)
    for n in nodes:
        nid = id(n)
        dep_count = 0
        for inp in n.input:
            if inp and inp in out_to_node_id:
                dep_graph[out_to_node_id[inp]].append(nid)
                dep_count += 1
        in_degree[nid] = dep_count
    queue = collections.deque([nid for nid in node_map if in_degree[nid] == 0])
    sorted_nodes: list[onnx.NodeProto] = []
    while queue:
        nid = queue.popleft()
        sorted_nodes.append(node_map[nid])
        for nxt in dep_graph[nid]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)
    if len(sorted_nodes) < len(nodes):
        seen = {id(n) for n in sorted_nodes}
        for n in nodes:
            if id(n) not in seen:
                sorted_nodes.append(n)
    return sorted_nodes


def _canonicalize_subgraph_nodes(
    sub_nodes: list[onnx.NodeProto],
    external_inputs: list[str],
    external_outputs: list[str],
) -> tuple[list[onnx.NodeProto], list[str], list[str]]:
    """Canonicalize internal tensor names of a subgraph to ensure clean FunctionProto export."""
    sub_nodes = _toposort_nodes(sub_nodes)
    input_map = {inp: f"in_{i}" for i, inp in enumerate(external_inputs)}
    output_map = {out: f"out_{j}" for j, out in enumerate(external_outputs)}
    internal_map: dict[str, str] = {}
    t_idx = 0
    for sn in sub_nodes:
        for out in sn.output:
            if out and out not in output_map and out not in internal_map:
                internal_map[out] = f"t_{t_idx}"
                t_idx += 1
    tensor_map = {**input_map, **output_map, **internal_map}
    canon_nodes: list[onnx.NodeProto] = []
    for sn in sub_nodes:
        n_copy = copy.deepcopy(sn)
        n_copy.input[:] = [tensor_map.get(inp, inp) for inp in sn.input]
        n_copy.output[:] = [tensor_map.get(out, out) for out in sn.output]
        canon_nodes.append(n_copy)
    return (
        canon_nodes,
        [input_map[inp] for inp in external_inputs],
        [output_map[out] for out in external_outputs],
    )


class MacroBlockSynthesizer(CompilerPass):
    """Synthesizes high-level macro architectural blocks using declarative JSON rules and pattern matching."""

    _cached_rules: dict[str, JSONRule] = {}

    @property
    def name(self) -> str:
        return "macro_block_synthesizer"

    @classmethod
    def load_rule(cls, rule_id: str) -> JSONRule:
        """Load a declarative JSON rule from definitions."""
        if rule_id in cls._cached_rules:
            return cls._cached_rules[rule_id]
        definitions_dir = Path(__file__).resolve().parents[3] / "rules" / "definitions"
        rule_path = definitions_dir / f"{rule_id}.json"
        if not rule_path.exists():
            msg = f"Rule definition file not found: {rule_path}"
            raise FileNotFoundError(msg)
        with open(rule_path, encoding="utf-8") as f:
            data = json.load(f)
        rule = JSONRule.from_dict(data)
        cls._cached_rules[rule_id] = rule
        return rule

    def run(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Run macro block synthesis passes sequentially."""
        model = self.apply_deep_mlp_norm_act_fusion(model, mode=mode)
        model = self.apply_spatial_gating_unit_fusion(model, mode=mode)
        model = self.apply_highway_block_fusion(model, mode=mode)
        model = self.apply_squeeze_excitation_fusion(model, mode=mode)
        model = self.apply_preactivation_resnet_fusion(model, mode=mode)
        model = self.apply_audio_logmel_compressor_fusion(model, mode=mode)
        model = self.apply_mha_composite_fusion(model, mode=mode)
        model = self.apply_grouped_query_attention_fusion(model, mode=mode)
        model = self.apply_parallel_transformer_block_fusion(model, mode=mode)
        model = self.apply_moe_fusion(model, mode=mode)
        model = self.apply_differential_attention_fusion(model, mode=mode)
        model = self.apply_mamba_ssm_fusion(model, mode=mode)
        model = self.apply_shifted_window_attention_fusion(model, mode=mode)
        model = self.apply_vit_patch_embedding_fusion(model, mode=mode)
        model = self.apply_deepseek_mla_macro_fusion(model, mode=mode)
        model = self.apply_multi_field_embedding_fusion(model, mode=mode)
        model = self.apply_attention_key_padding_mask_fusion(model, mode=mode)
        model = self.apply_dynamic_quant_linear_fusion(model, mode=mode)
        model = self.apply_transformer_encoder_layer_fusion(model, mode=mode)
        model = self.apply_sequence_classifier_head_fusion(model, mode=mode)
        model = self.apply_multi_task_audio_head_fusion(model, mode=mode)
        return model

    # -------------------------------------------------------------------------
    # 1. DeepMLP_Norm_Act Fusion: LayerNorm / RMSNorm -> Linear -> Gelu
    # -------------------------------------------------------------------------
    def apply_deep_mlp_norm_act_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuse Norm -> Linear -> Gelu into DeepMLP_Norm_Act using pattern matching."""
        rule = self.load_rule("deep_mlp_norm_act_fusion")
        graph = model.graph
        graph_outputs = {out.name for out in graph.output}
        consumer_map: dict[str, list[onnx.NodeProto]] = collections.defaultdict(list)
        for n in graph.node:
            for inp in n.input:
                consumer_map[inp].append(n)

        norm_pat = Pattern.any_of(
            Pattern.op("RMSNorm"),
            Pattern.op("RMSNormalization"),
            Pattern.op("LayerNormalization"),
            Pattern.op("SimplifiedLayerNormalization"),
        ).capture("norm")

        lin_pat = Pattern.any_of(
            Pattern.op("Linear", norm_pat),
            Pattern.op("Gemm", norm_pat),
            Pattern.op("MatMul", norm_pat),
        ).capture("linear")

        act_pat = Pattern.any_of(
            Pattern.op("Gelu", lin_pat),
            Pattern.op("FastGelu", lin_pat),
            Pattern.op("Relu", lin_pat),
            Pattern.op("SiLU", lin_pat),
        ).capture("act")

        detector = PatternDetector(model)
        matches = detector.find_all(act_pat)
        if not matches:
            return model

        _ensure_custom_domain(model, rule.action.domain)
        nodes_to_drop: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        count = 0

        for match in matches:
            act_node = match.bindings.get("act")
            lin_node = match.bindings.get("linear")
            norm_node = match.bindings.get("norm")
            if not act_node or not lin_node or not norm_node:
                continue

            if lin_node.name in nodes_to_drop or act_node.name in nodes_to_drop:
                continue

            # Ensure linear output is consumed only by this activation
            if (
                len(consumer_map.get(lin_node.output[0], [])) > 1
                or lin_node.output[0] in graph_outputs
            ):
                continue

            act_name = act_node.op_type
            nodes_to_drop.update({act_node.name, lin_node.name})

            if (
                len(consumer_map.get(norm_node.output[0], [])) <= 1
                and norm_node.output[0] not in graph_outputs
            ):
                nodes_to_drop.add(norm_node.name)
                inputs = list(norm_node.input) + list(lin_node.input[1:])
                has_norm = 1
                sub_nodes = [norm_node, lin_node, act_node]
            else:
                inputs = list(lin_node.input)
                has_norm = 0
                sub_nodes = [lin_node, act_node]

            fp = compute_subgraph_fingerprint(sub_nodes)
            specialized_op = build_monomorphized_op_type(
                rule.action.op_type, act_name, fp
            )

            new_node = helper.make_node(
                specialized_op,
                inputs=inputs,
                outputs=[act_node.output[0]],
                name=f"{act_node.name}_fused_{rule.rule_id}",
                domain=rule.action.domain,
                activation=act_name,
                has_norm=has_norm,
                doc_string=rule.description,
            )
            new_nodes.append(new_node)
            count += 1

            if mode == "function_proto":
                formal_inputs = ["x", "norm_w", "w"] if has_norm else ["x", "w"]
                formal_outputs = ["y"]
                domain_act = "custom.pytorch" if act_name in ("Gelu", "SiLU", "FastGelu") else ""
                if has_norm:
                    formal_norm = helper.make_node(norm_node.op_type, inputs=["x", "norm_w"], outputs=["norm_out"])
                    formal_lin = helper.make_node("Linear", inputs=["norm_out", "w"], outputs=["lin_out"], domain="custom.pytorch")
                    formal_act = helper.make_node(act_name, inputs=["lin_out"], outputs=["y"], domain=domain_act)
                    formal_body = [formal_norm, formal_lin, formal_act]
                else:
                    formal_lin = helper.make_node("Linear", inputs=["x", "w"], outputs=["lin_out"], domain="custom.pytorch")
                    formal_act = helper.make_node(act_name, inputs=["lin_out"], outputs=["y"], domain=domain_act)
                    formal_body = [formal_lin, formal_act]
                ensure_specialized_function_proto(
                    model=model,
                    domain=rule.action.domain,
                    fname=specialized_op,
                    formal_inputs=formal_inputs,
                    formal_outputs=formal_outputs,
                    body_nodes=formal_body,
                    doc_string=f"Specialized DeepMLP Norm Act block: {rule.description}",
                )

        if not new_nodes:
            return model

        _replace_nodes_topologically(graph, nodes_to_drop, new_nodes)
        log.info("DeepMLP Norm Act Fusion: Fused %d instances", count)
        return ShapeCanonicalizer().apply_dead_code_elimination(model)

    # -------------------------------------------------------------------------
    # 2. Spatial Gating Unit (SGU) Fusion
    # -------------------------------------------------------------------------
    def apply_spatial_gating_unit_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuse Split -> Norm -> Linear -> Mul into SpatialGatingUnit."""
        rule = self.load_rule("spatial_gating_unit_fusion")
        graph = model.graph

        norm_pat = Pattern.any_of(
            Pattern.op("RMSNorm"),
            Pattern.op("LayerNormalization"),
        ).capture("norm")

        proj_pat = Pattern.any_of(
            Pattern.op("Linear", norm_pat),
            Pattern.op("MatMul", norm_pat),
            Pattern.op("Gemm", norm_pat),
        ).capture("proj")

        mul_pat = Pattern.op("Mul", proj_pat, Pattern.any()).capture("mul")

        detector = PatternDetector(model)
        matches = detector.find_all(mul_pat)
        if not matches:
            return model

        _ensure_custom_domain(model, rule.action.domain)
        nodes_to_drop: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        count = 0

        for match in matches:
            mul_node = match.bindings.get("mul")
            proj_node = match.bindings.get("proj")
            norm_node = match.bindings.get("norm")
            if not mul_node or not proj_node or not norm_node:
                continue

            if mul_node.name in nodes_to_drop or proj_node.name in nodes_to_drop:
                continue

            branch_a = (
                mul_node.input[1]
                if mul_node.input[0] == proj_node.output[0]
                else mul_node.input[0]
            )

            count += 1
            nodes_to_drop.update({mul_node.name, proj_node.name, norm_node.name})

            sgu_node = helper.make_node(
                rule.action.op_type,
                inputs=[branch_a, norm_node.input[0]] + list(proj_node.input[1:]),
                outputs=[mul_node.output[0]],
                name=f"{mul_node.name}_fused_{rule.rule_id}",
                domain=rule.action.domain,
                doc_string=rule.description,
            )
            new_nodes.append(sgu_node)

        if not new_nodes:
            return model

        _replace_nodes_topologically(graph, nodes_to_drop, new_nodes)
        if mode == "function_proto":
            self._attach_sgu_function_def(model)

        log.info("SpatialGatingUnit Fusion: Fused %d instances", count)
        return ShapeCanonicalizer().apply_dead_code_elimination(model)

    def _attach_sgu_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for SpatialGatingUnit."""
        func_name = "SpatialGatingUnit"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", LATEST_OPSET)
        opset_custom = helper.make_opsetid("custom.pytorch", 1)

        norm = helper.make_node("RMSNorm", inputs=["u"], outputs=["norm_out"], domain=domain)
        proj = helper.make_node("Linear", inputs=["norm_out", "w"], outputs=["proj_out"], domain=domain)
        mul = helper.make_node("Mul", inputs=["v", "proj_out"], outputs=["y"])

        func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["v", "u", "w"],
            outputs=["y"],
            nodes=[norm, proj, mul],
            opset_imports=[opset_standard, opset_custom],
            doc_string="Formal Spatial Gating Unit (v * Linear(RMSNorm(u)))",
        )
        model.functions.append(func)

    # -------------------------------------------------------------------------
    # 3. Highway Block Fusion
    # -------------------------------------------------------------------------
    def apply_highway_block_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuse Highway transformation branch and carry gate into HighwayBlock."""
        rule = self.load_rule("highway_block_fusion")
        graph = model.graph

        x = Pattern.any().capture("x")
        lin_h = Pattern.any_of(
            Pattern.op("Linear", x), Pattern.op("Gemm", x), Pattern.op("MatMul", x)
        ).capture("lin_h")
        act_h = Pattern.any_of(
            Pattern.op("Relu", lin_h), Pattern.op("Gelu", lin_h), Pattern.op("Tanh", lin_h)
        ).capture("act_h")
        lin_t = Pattern.any_of(
            Pattern.op("Linear", x), Pattern.op("Gemm", x), Pattern.op("MatMul", x)
        ).capture("lin_t")
        sig_t = Pattern.op("Sigmoid", lin_t).capture("sig_t")
        mul_h = (act_h * sig_t).capture("mul_h")
        one_minus_t = Pattern.const(1.0) - sig_t
        mul_x = (x * one_minus_t).capture("mul_x")
        add_out = (mul_h + mul_x).capture("add_out")

        detector = PatternDetector(model)
        matches = detector.find_all(add_out)
        if not matches:
            # Fallback: check pre-fused GatedUnit + carry branch pattern
            node_by_output = {out: n for n in graph.node for out in n.output if out}
            nodes_to_drop = set()
            new_nodes = []
            count = 0
            for n in graph.node:
                if n.op_type == "Add" and len(n.input) == 2:
                    m1 = node_by_output.get(n.input[0])
                    m2 = node_by_output.get(n.input[1])
                    if not m1 or not m2:
                        continue
                    gated_cand = m1 if m1.op_type.startswith("GatedUnit") else (m2 if m2.op_type.startswith("GatedUnit") else None)
                    carry_mul = m2 if gated_cand == m1 else (m1 if gated_cand == m2 else None)
                    if gated_cand and carry_mul and carry_mul.op_type == "Mul":
                        sub_cand = node_by_output.get(carry_mul.input[0]) or node_by_output.get(carry_mul.input[1])
                        x_tensor = carry_mul.input[1] if sub_cand == node_by_output.get(carry_mul.input[0]) else carry_mul.input[0]
                        if sub_cand and sub_cand.op_type == "Sub":
                            sig_cand = node_by_output.get(sub_cand.input[1]) or node_by_output.get(sub_cand.input[0])
                            nodes_to_drop.update({n.name, gated_cand.name, carry_mul.name, sub_cand.name})
                            if sig_cand:
                                nodes_to_drop.add(sig_cand.name)
                            hw_inputs = [gated_cand.input[0], gated_cand.input[1], x_tensor]
                            _ensure_custom_domain(model, "custom.pytorch")
                            hw_node = helper.make_node(
                                "HighwayBlock",
                                inputs=hw_inputs,
                                outputs=[n.output[0]],
                                name=f"{n.name}_fused_highway",
                                domain="custom.pytorch",
                                activation="Sigmoid",
                            )
                            new_nodes.append(hw_node)
                            count += 1
            if not new_nodes:
                return model
            _replace_nodes_topologically(graph, nodes_to_drop, new_nodes)
            if mode == "function_proto":
                self._attach_highway_function_def(model, formal_inputs=["h", "gate_in", "x"])
            log.info("HighwayBlock Fusion: Fused %d instances", count)
            return ShapeCanonicalizer().apply_dead_code_elimination(model)

        _ensure_custom_domain(model, rule.action.domain)
        nodes_to_drop: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        count = 0

        for match in matches:
            add_node = match.bindings.get("add_out")
            act_node = match.bindings.get("act_h")
            sig_node = match.bindings.get("sig_t")
            lin_h_node = match.bindings.get("lin_h")
            lin_t_node = match.bindings.get("lin_t")
            x_node = match.bindings.get("x")
            if not add_node or not act_node or not sig_node:
                continue

            count += 1
            for n_m in (act_node, sig_node, lin_h_node, lin_t_node, add_node):
                if n_m and hasattr(n_m, "name"):
                    nodes_to_drop.add(n_m.name)

            hw_inputs = [x_node.output[0] if hasattr(x_node, "output") else str(x_node)]
            if lin_h_node and len(lin_h_node.input) > 1:
                hw_inputs.append(lin_h_node.input[1])
            if lin_t_node and len(lin_t_node.input) > 1:
                hw_inputs.append(lin_t_node.input[1])

            hw_node = helper.make_node(
                rule.action.op_type,
                inputs=hw_inputs,
                outputs=[add_node.output[0]],
                name=f"{add_node.name}_fused_{rule.rule_id}",
                domain=rule.action.domain,
                activation=act_node.op_type,
                doc_string=rule.description,
            )
            new_nodes.append(hw_node)

        if not new_nodes:
            return model

        _replace_nodes_topologically(graph, nodes_to_drop, new_nodes)
        if mode == "function_proto":
            self._attach_highway_function_def(model)

        log.info("HighwayBlock Fusion: Fused %d instances", count)
        return ShapeCanonicalizer().apply_dead_code_elimination(model)

    def _attach_highway_function_def(
        self, model: onnx.ModelProto, formal_inputs: list[str] | None = None
    ) -> None:
        """Attach formal ONNX FunctionProto definition for HighwayBlock."""
        func_name = "HighwayBlock"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", LATEST_OPSET)
        opset_custom = helper.make_opsetid("custom.pytorch", 1)

        if formal_inputs and formal_inputs == ["h", "gate_in", "x"]:
            sig_node = helper.make_node("Sigmoid", inputs=["gate_in"], outputs=["t"])
            c_node = helper.make_node(
                "Constant",
                inputs=[],
                outputs=["one"],
                value=helper.make_tensor("one", onnx.TensorProto.FLOAT, [1], [1.0]),
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
        else:
            lh = helper.make_node("Linear", inputs=["x", "wh"], outputs=["h_raw"], domain=domain)
            act = helper.make_node("Relu", inputs=["h_raw"], outputs=["h"])
            lt = helper.make_node("Linear", inputs=["x", "wt"], outputs=["t_raw"], domain=domain)
            sig = helper.make_node("Sigmoid", inputs=["t_raw"], outputs=["t"])
            c1 = helper.make_node("Constant", inputs=[], outputs=["one"], value_float=1.0)
            sub = helper.make_node("Sub", inputs=["one", "t"], outputs=["omt"])
            m1 = helper.make_node("Mul", inputs=["h", "t"], outputs=["term1"])
            m2 = helper.make_node("Mul", inputs=["x", "omt"], outputs=["term2"])
            add = helper.make_node("Add", inputs=["term1", "term2"], outputs=["y"])

            func = helper.make_function(
                domain=domain,
                fname=func_name,
                inputs=["x", "wh", "wt"],
                outputs=["y"],
                nodes=[lh, act, lt, sig, c1, sub, m1, m2, add],
                opset_imports=[opset_standard, opset_custom],
                doc_string="Formal Highway Block (H * T + x * (1 - T))",
            )
        model.functions.append(func)

    # -------------------------------------------------------------------------
    # 4. Squeeze-and-Excitation (SE) Block Fusion
    # -------------------------------------------------------------------------
    def apply_squeeze_excitation_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuse GAP -> FC1 -> Relu -> FC2 -> Sigmoid -> Mul into SqueezeExcitation."""
        rule = self.load_rule("squeeze_excitation_fusion")
        graph = model.graph

        x = Pattern.any().capture("x")
        gap = Pattern.any_of(
            Pattern.op("GlobalAveragePool", x),
            Pattern.op("ReduceMean", x),
        ).capture("gap")
        fc1 = Pattern.any_of(
            Pattern.op("Conv", gap),
            Pattern.op("Linear", gap),
            Pattern.op("Gemm", gap),
        ).capture("fc1")
        act = Pattern.op("Relu", fc1).capture("act")
        fc2 = Pattern.any_of(
            Pattern.op("Conv", act),
            Pattern.op("Linear", act),
            Pattern.op("Gemm", act),
        ).capture("fc2")
        sig = Pattern.op("Sigmoid", fc2).capture("sig")
        se_out = (x * sig).capture("se_out")

        detector = PatternDetector(model)
        matches = detector.find_all(se_out)
        if not matches:
            return model

        _ensure_custom_domain(model, rule.action.domain)
        nodes_to_drop: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        count = 0

        for match in matches:
            mul_node = match.bindings.get("se_out")
            fc1_node = match.bindings.get("fc1")
            act_node = match.bindings.get("act")
            fc2_node = match.bindings.get("fc2")
            sig_node = match.bindings.get("sig")
            gap_node = match.bindings.get("gap")
            x_node = match.bindings.get("x")
            if not mul_node or not fc1_node or not fc2_node:
                continue

            count += 1
            for n_m in (gap_node, fc1_node, act_node, fc2_node, sig_node, mul_node):
                if n_m and hasattr(n_m, "name"):
                    nodes_to_drop.add(n_m.name)

            se_inputs = [x_node.output[0] if hasattr(x_node, "output") else str(x_node)]
            for fc in (fc1_node, fc2_node):
                if len(fc.input) > 1:
                    se_inputs.append(fc.input[1])

            se_node = helper.make_node(
                rule.action.op_type,
                inputs=se_inputs,
                outputs=[mul_node.output[0]],
                name=f"{mul_node.name}_fused_{rule.rule_id}",
                domain=rule.action.domain,
                doc_string=rule.description,
            )
            new_nodes.append(se_node)

        if not new_nodes:
            return model

        _replace_nodes_topologically(graph, nodes_to_drop, new_nodes)
        if mode == "function_proto":
            self._attach_se_function_def(model)

        log.info("SqueezeExcitation Fusion: Fused %d instances", count)
        return ShapeCanonicalizer().apply_dead_code_elimination(model)

    def _attach_se_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for SqueezeExcitation."""
        func_name = "SqueezeExcitation"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", LATEST_OPSET)
        opset_custom = helper.make_opsetid("custom.pytorch", 1)

        gap = helper.make_node("GlobalAveragePool", inputs=["x"], outputs=["pooled"])
        fc1 = helper.make_node("Linear", inputs=["pooled", "w1"], outputs=["fc1_out"], domain=domain)
        act = helper.make_node("Relu", inputs=["fc1_out"], outputs=["act_out"])
        fc2 = helper.make_node("Linear", inputs=["act_out", "w2"], outputs=["fc2_out"], domain=domain)
        sig = helper.make_node("Sigmoid", inputs=["fc2_out"], outputs=["scale"])
        mul = helper.make_node("Mul", inputs=["x", "scale"], outputs=["y"])

        func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["x", "w1", "w2"],
            outputs=["y"],
            nodes=[gap, fc1, act, fc2, sig, mul],
            opset_imports=[opset_standard, opset_custom],
            doc_string="Formal Squeeze-and-Excitation Channel Attention Block",
        )
        model.functions.append(func)

    # -------------------------------------------------------------------------
    # 5. Pre-Activation ResNet Residual Block Fusion
    # -------------------------------------------------------------------------
    def apply_preactivation_resnet_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuse Pre-Activation ResNet blocks: BN -> Conv -> ReLU -> BN -> Conv + Residual -> ReLU."""
        rule = self.load_rule("preactivation_resnet_fusion")
        graph = model.graph

        x = Pattern.any().capture("x")
        bn1 = Pattern.op("BatchNormalization", x).capture("bn1")
        conv1 = Pattern.op("Conv", bn1).capture("conv1")
        relu1 = Pattern.op("Relu", conv1).capture("relu1")
        bn2 = Pattern.op("BatchNormalization", relu1).capture("bn2")
        conv2 = Pattern.op("Conv", bn2).capture("conv2")
        add = (conv2 + x).capture("add")
        relu2 = Pattern.op("Relu", add).capture("relu2")

        detector = PatternDetector(model)
        matches = detector.find_all(relu2)
        if not matches:
            return model

        _ensure_custom_domain(model, rule.action.domain)
        nodes_to_drop: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        count = 0

        for match in matches:
            relu2_node = match.bindings.get("relu2")
            add_node = match.bindings.get("add")
            conv1_node = match.bindings.get("conv1")
            relu1_node = match.bindings.get("relu1")
            bn1_node = match.bindings.get("bn1")
            bn2_node = match.bindings.get("bn2")
            conv2_node = match.bindings.get("conv2")
            x_val = match.bindings.get("x")
            if not relu2_node or not conv1_node or not conv2_node:
                continue

            count += 1
            # Drop ONLY internal block nodes, NOT the upstream producer 'x'
            sub_nodes = [bn1_node, conv1_node, relu1_node, bn2_node, conv2_node, add_node, relu2_node]
            for sn in sub_nodes:
                if sn and hasattr(sn, "name"):
                    nodes_to_drop.add(sn.name)

            internal_producers = {out for sn in sub_nodes for out in sn.output if out}
            external_inputs: list[str] = []
            for sn in sub_nodes:
                for inp in sn.input:
                    if inp and inp not in internal_producers and inp not in external_inputs:
                        external_inputs.append(inp)
            external_outputs = [relu2_node.output[0]]

            fp = compute_subgraph_fingerprint(sub_nodes, external_inputs, external_outputs)
            op_type = build_monomorphized_op_type(rule.action.op_type, fingerprint=fp)

            if mode == "function_proto":
                canon_nodes, f_in, f_out = _canonicalize_subgraph_nodes(
                    sub_nodes, external_inputs, external_outputs
                )
                ensure_specialized_function_proto(
                    model,
                    domain=rule.action.domain,
                    fname=op_type,
                    formal_inputs=f_in,
                    formal_outputs=f_out,
                    body_nodes=canon_nodes,
                    doc_string=f"Formal Pre-Activation ResNet Residual Block ({op_type})",
                )

            res_node = helper.make_node(
                op_type,
                inputs=external_inputs,
                outputs=external_outputs,
                name=f"{relu2_node.name}_fused_{rule.rule_id}",
                domain=rule.action.domain,
                doc_string=rule.description,
            )
            new_nodes.append(res_node)

        if not new_nodes:
            return model

        _replace_nodes_topologically(graph, nodes_to_drop, new_nodes)
        log.info("PreActivation ResNet Fusion: Fused %d ResidualBlock instances", count)
        return ShapeCanonicalizer().apply_dead_code_elimination(model)

    def _attach_residual_block_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for ResidualBlock."""
        func_name = "ResidualBlock"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", LATEST_OPSET)
        opset_custom = helper.make_opsetid("custom.pytorch", 1)

        bn1 = helper.make_node("BatchNormalization", inputs=["x", "s1", "b1", "m1", "v1"], outputs=["bn1_out"])
        conv1 = helper.make_node("Conv", inputs=["bn1_out", "w1"], outputs=["c1_out"])
        relu1 = helper.make_node("Relu", inputs=["c1_out"], outputs=["r1_out"])
        bn2 = helper.make_node("BatchNormalization", inputs=["r1_out", "s2", "b2", "m2", "v2"], outputs=["bn2_out"])
        conv2 = helper.make_node("Conv", inputs=["bn2_out", "w2"], outputs=["c2_out"])
        add = helper.make_node("Add", inputs=["c2_out", "x"], outputs=["add_out"])
        relu2 = helper.make_node("Relu", inputs=["add_out"], outputs=["y"])

        func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["x", "w1", "w2", "s1", "b1", "m1", "v1", "s2", "b2", "m2", "v2"],
            outputs=["y"],
            nodes=[bn1, conv1, relu1, bn2, conv2, add, relu2],
            opset_imports=[opset_standard, opset_custom],
            doc_string="Formal Pre-Activation ResNet Residual Block",
        )
        model.functions.append(func)

    # -------------------------------------------------------------------------
    # 6. Audio LogMel Dynamic Range Compressor Fusion
    # -------------------------------------------------------------------------
    def apply_audio_logmel_compressor_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuse Clip -> Div -> Log -> Div -> Mul -> Add -> Clip into AudioLogMelCompressor."""
        rule = self.load_rule("audio_logmel_compressor_fusion")
        graph = model.graph

        c1 = Pattern.op("Clip").capture("c1")
        d1 = Pattern.op("Div", c1).capture("d1")
        l1 = Pattern.op("Log", d1).capture("l1")
        d2 = Pattern.op("Div", l1).capture("d2")
        m1 = Pattern.op("Mul", d2).capture("m1")
        a1 = Pattern.op("Add", m1).capture("a1")
        c2 = Pattern.op("Clip", a1).capture("c2")

        detector = PatternDetector(model)
        matches = detector.find_all(c2)
        if not matches:
            return model

        _ensure_custom_domain(model, rule.action.domain)
        nodes_to_drop: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        count = 0

        for match in matches:
            c2_node = match.bindings.get("c2")
            a1_node = match.bindings.get("a1")
            m1_node = match.bindings.get("m1")
            d2_node = match.bindings.get("d2")
            l1_node = match.bindings.get("l1")
            d1_node = match.bindings.get("d1")
            c1_node = match.bindings.get("c1")
            if not c2_node or not c1_node:
                continue

            count += 1
            sub_nodes = [c1_node, d1_node, l1_node, d2_node, m1_node, a1_node, c2_node]
            for sn in sub_nodes:
                if sn and hasattr(sn, "name"):
                    nodes_to_drop.add(sn.name)

            internal_producers = {out for sn in sub_nodes for out in sn.output if out}
            external_inputs: list[str] = []
            for sn in sub_nodes:
                for inp in sn.input:
                    if inp and inp not in internal_producers and inp not in external_inputs:
                        external_inputs.append(inp)
            external_outputs = [c2_node.output[0]]

            op_type = rule.action.op_type
            if mode == "function_proto":
                canon_nodes, f_in, f_out = _canonicalize_subgraph_nodes(
                    sub_nodes, external_inputs, external_outputs
                )
                ensure_specialized_function_proto(
                    model,
                    domain=rule.action.domain,
                    fname=op_type,
                    formal_inputs=f_in,
                    formal_outputs=f_out,
                    body_nodes=canon_nodes,
                    doc_string="Formal Audio Log-Mel Dynamic Range Compression Pipeline",
                )

            comp_node = helper.make_node(
                op_type,
                inputs=external_inputs,
                outputs=external_outputs,
                name=f"{c2_node.name}_fused_{rule.rule_id}",
                domain=rule.action.domain,
                doc_string=rule.description,
            )
            new_nodes.append(comp_node)

        if not new_nodes:
            return model

        _replace_nodes_topologically(graph, nodes_to_drop, new_nodes)
        log.info("Audio LogMel Compressor Fusion: Fused %d instances", count)
        return ShapeCanonicalizer().apply_dead_code_elimination(model)

    def _attach_audio_compressor_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for AudioLogMelCompressor."""
        func_name = "AudioLogMelCompressor"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", LATEST_OPSET)
        opset_custom = helper.make_opsetid("custom.pytorch", 1)

        c1 = helper.make_node("Clip", inputs=["x"], outputs=["c1_out"])
        d1 = helper.make_node("Div", inputs=["c1_out", "ref"], outputs=["d1_out"])
        l1 = helper.make_node("Log", inputs=["d1_out"], outputs=["l1_out"])
        d2 = helper.make_node("Div", inputs=["l1_out", "scale"], outputs=["d2_out"])
        m1 = helper.make_node("Mul", inputs=["d2_out", "mul_val"], outputs=["m1_out"])
        a1 = helper.make_node("Add", inputs=["m1_out", "offset"], outputs=["a1_out"])
        c2 = helper.make_node("Clip", inputs=["a1_out"], outputs=["y"])

        func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["x", "ref", "scale", "mul_val", "offset"],
            outputs=["y"],
            nodes=[c1, d1, l1, d2, m1, a1, c2],
            opset_imports=[opset_standard, opset_custom],
            doc_string="Formal Audio Log-Mel Dynamic Range Compression Pipeline",
        )
        model.functions.append(func)

    # -------------------------------------------------------------------------
    # 7. Multi-Head Attention (MHA) Composite Fusion
    # -------------------------------------------------------------------------
    def apply_mha_composite_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuse Linear(QKV) -> SplitQKV -> SDPA -> MergeHeads -> Linear(Out) into MultiHeadAttention."""
        rule = self.load_rule("mha_composite_fusion")
        graph = model.graph

        sdpa_pat = Pattern.any_of(
            Pattern.op("SDPA"), Pattern.op("ScaledDotProductAttention")
        ).capture("sdpa")
        merge_pat = Pattern.op("MergeHeads", sdpa_pat).capture("merge")
        out_proj_pat = Pattern.any_of(
            Pattern.op("Linear", merge_pat),
            Pattern.op("MatMul", merge_pat),
            Pattern.op("Gemm", merge_pat),
        ).capture("out_proj")

        detector = PatternDetector(model)
        matches = detector.find_all(out_proj_pat)
        if not matches:
            return model

        _ensure_custom_domain(model, rule.action.domain)
        nodes_to_drop: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        count = 0

        for match in matches:
            out_node = match.bindings.get("out_proj")
            merge_node = match.bindings.get("merge")
            sdpa_node = match.bindings.get("sdpa")
            if not out_node or not merge_node or not sdpa_node:
                continue

            count += 1
            for n_m in (sdpa_node, merge_node, out_node):
                if n_m and hasattr(n_m, "name"):
                    nodes_to_drop.add(n_m.name)

            mha_inputs = list(sdpa_node.input) + list(out_node.input[1:])
            target_op = rule.action.op_type
            if sdpa_node and len(sdpa_node.input) >= 2:
                node_map = {out: n for n in graph.node for out in n.output}
                def _find_root(t_name):
                    curr = t_name
                    visited = set()
                    while curr in node_map and curr not in visited:
                        visited.add(curr)
                        p = node_map[curr]
                        if p.input:
                            curr = p.input[0]
                        else:
                            break
                    return curr

                q_root = _find_root(sdpa_node.input[0])
                k_root = _find_root(sdpa_node.input[1])
                if any("past" in str(inp) for inp in sdpa_node.input) or any("Concat" in str(inp) for inp in sdpa_node.input[1:]):
                    target_op = "StatefulAttention"
                elif q_root != k_root:
                    target_op = "CrossAttention"

            mha_node = helper.make_node(
                target_op,
                inputs=mha_inputs,
                outputs=[out_node.output[0]],
                name=f"{out_node.name}_fused_{rule.rule_id}",
                domain=rule.action.domain,
                doc_string=rule.description,
            )
            new_nodes.append(mha_node)

        if not new_nodes:
            return model

        _replace_nodes_topologically(graph, nodes_to_drop, new_nodes)
        if mode == "function_proto":
            self._attach_mha_composite_function_def(model)

        log.info("MultiHeadAttention Composite Fusion: Fused %d instances", count)
        return ShapeCanonicalizer().apply_dead_code_elimination(model)

    def _attach_mha_composite_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for MultiHeadAttention."""
        func_name = "MultiHeadAttention"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", LATEST_OPSET)
        opset_custom = helper.make_opsetid("custom.pytorch", 1)

        qkv = helper.make_node("Linear", inputs=["x", "w_qkv"], outputs=["qkv_out"], domain=domain)
        sq = helper.make_node("SplitQKV", inputs=["qkv_out"], outputs=["q", "k", "v"], domain=domain)
        sdpa = helper.make_node("SDPA", inputs=["q", "k", "v"], outputs=["sdpa_out"], domain=domain)
        mh = helper.make_node("MergeHeads", inputs=["sdpa_out"], outputs=["mh_out"], domain=domain)
        lin_out = helper.make_node("Linear", inputs=["mh_out", "w_out"], outputs=["y"], domain=domain)

        func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["x", "w_qkv", "w_out"],
            outputs=["y"],
            nodes=[qkv, sq, sdpa, mh, lin_out],
            opset_imports=[opset_standard, opset_custom],
            doc_string="Formal Multi-Head Attention unit with SplitQKV and MergeHeads",
        )
        model.functions.append(func)

    # -------------------------------------------------------------------------
    # 8. Gated Multi-Head Attention (GAU) Fusion
    # -------------------------------------------------------------------------
    def apply_gated_mha_composite_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuse RMSNorm -> Linear(QKV) -> GAU -> MergeHeads -> Linear(Out) into GatedMultiHeadAttention."""
        rule = self.load_rule("gated_mha_composite_fusion")
        graph = model.graph
        node_by_output: dict[str, onnx.NodeProto] = {
            out: n for n in graph.node for out in n.output
        }
        consumer_map: dict[str, list[onnx.NodeProto]] = collections.defaultdict(list)
        for n in graph.node:
            for inp in n.input:
                consumer_map[inp].append(n)

        gau_nodes = [
            n for n in graph.node if n.op_type.startswith("GatedUnit") or n.op_type.startswith("GatedAttention")
        ]
        if not gau_nodes:
            return model

        _ensure_custom_domain(model, rule.action.domain)
        nodes_to_drop: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        count = 0

        for gau in gau_nodes:
            merge_cands = [
                c for c in consumer_map.get(gau.output[0], []) if c.op_type == "MergeHeads"
            ]
            if not merge_cands:
                continue
            merge_node = merge_cands[0]

            out_cands = [
                c for c in consumer_map.get(merge_node.output[0], []) if c.op_type in ("Linear", "MatMul", "Gemm")
            ]
            if not out_cands:
                continue
            out_node = out_cands[0]

            # Find upstream QKV and Gate linear nodes
            q_node = node_by_output.get(gau.input[0])
            gate_node = node_by_output.get(gau.input[1]) if len(gau.input) > 1 else None
            gate_chain_nodes = []
            while gate_node and gate_node.op_type in ("Reshape", "Transpose", "Unsqueeze", "Squeeze"):
                gate_chain_nodes.append(gate_node)
                gate_node = node_by_output.get(gate_node.input[0]) if gate_node.input else None

            # Trace backward through SDPA
            sdpa_node = None
            if q_node and q_node.op_type.startswith("ScaledDotProductAttention"):
                sdpa_node = q_node
                q_node = node_by_output.get(sdpa_node.input[0])

            # Trace backward through RoPE and SplitQKV
            rope_nodes = []
            split_node = None
            while q_node and q_node.op_type in ("RotaryEmbedding", "SplitQKV"):
                if q_node.op_type == "SplitQKV":
                    split_node = q_node
                else:
                    rope_nodes.append(q_node)
                q_node = node_by_output.get(q_node.input[0])

            norm_node = None
            if q_node and q_node.input:
                norm_node = node_by_output.get(q_node.input[0])

            if not norm_node or not norm_node.op_type.startswith("RMSNorm"):
                continue

            count += 1
            sub_nodes = [norm_node, q_node, gau, merge_node, out_node]
            if sdpa_node:
                sub_nodes.append(sdpa_node)
            if gate_node:
                sub_nodes.append(gate_node)
            sub_nodes.extend(gate_chain_nodes)
            if split_node:
                sub_nodes.append(split_node)
            sub_nodes.extend(rope_nodes)

            for sn in sub_nodes:
                nodes_to_drop.add(sn.name)

            fused_inputs = [norm_node.input[0]]
            for n_w in (norm_node, q_node, gate_node, out_node):
                if n_w:
                    for inp_w in n_w.input[1:]:
                        if inp_w not in fused_inputs and inp_w:
                            fused_inputs.append(inp_w)

            gau_comp = helper.make_node(
                rule.action.op_type,
                inputs=fused_inputs,
                outputs=[out_node.output[0]],
                name=f"{gau.name}_fused_{rule.rule_id}",
                domain=rule.action.domain,
                doc_string=rule.description,
            )
            new_nodes.append(gau_comp)

        if not new_nodes:
            return model

        _replace_nodes_topologically(graph, nodes_to_drop, new_nodes)
        if mode == "function_proto":
            self._attach_gated_mha_composite_function_def(model)

        log.info("GatedMHA Composite Fusion: Fused %d instances", count)
        return ShapeCanonicalizer().apply_dead_code_elimination(model)

    def _attach_gated_mha_composite_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for GatedMultiHeadAttention."""
        func_name = "GatedMultiHeadAttention"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", LATEST_OPSET)
        opset_custom = helper.make_opsetid("custom.pytorch", 1)

        norm = helper.make_node("RMSNorm", inputs=["x"], outputs=["norm_out"], domain=domain)
        qkv = helper.make_node("Linear", inputs=["norm_out", "w_qkv"], outputs=["qkv_out"], domain=domain)
        gate = helper.make_node("Linear", inputs=["norm_out", "w_gate"], outputs=["gate_out"], domain=domain)
        sq = helper.make_node("SplitQKV", inputs=["qkv_out"], outputs=["q", "k", "v"], domain=domain)
        rope_q = helper.make_node("RotaryEmbedding", inputs=["q"], outputs=["q_rope"], domain=domain)
        rope_k = helper.make_node("RotaryEmbedding", inputs=["k"], outputs=["k_rope"], domain=domain)
        sdpa = helper.make_node("SDPA", inputs=["q_rope", "k_rope", "v"], outputs=["sdpa_out"], domain=domain)
        gu = helper.make_node("GatedUnit", inputs=["sdpa_out", "gate_out"], outputs=["gu_out"], domain=domain)
        mh = helper.make_node("MergeHeads", inputs=["gu_out"], outputs=["mh_out"], domain=domain)
        lin_out = helper.make_node("Linear", inputs=["mh_out", "w_out"], outputs=["y"], domain=domain)

        func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["x", "w_qkv", "w_gate", "w_out"],
            outputs=["y"],
            nodes=[norm, qkv, gate, sq, rope_q, rope_k, sdpa, gu, mh, lin_out],
            opset_imports=[opset_standard, opset_custom],
            doc_string="Formal Gated Multi-Head Attention unit with RMSNorm and RoPE",
        )
        model.functions.append(func)

    # -------------------------------------------------------------------------
    # 9. Full Transformer Block Composite Fusion
    # -------------------------------------------------------------------------
    def apply_transformer_block_composite_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuse Attention -> Residual Add -> FeedForward -> Residual Add into TransformerBlock."""
        rule = self.load_rule("transformer_block_composite_fusion")
        graph = model.graph

        x = Pattern.any().capture("x")
        attn = Pattern.any_of(
            Pattern.op("GatedMultiHeadAttention", x),
            Pattern.op("MultiHeadAttention", x),
            Pattern.op("SDPA", x),
        ).capture("attn")
        add1 = (attn + x).capture("add1")
        ffn = Pattern.any_of(
            Pattern.op("FeedForward", add1),
            Pattern.op("FeedForwardSwiGLU", add1),
        ).capture("ffn")
        add2 = (ffn + add1).capture("add2")

        detector = PatternDetector(model)
        matches = detector.find_all(add2)
        if not matches:
            return model

        _ensure_custom_domain(model, rule.action.domain)
        nodes_to_drop: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        count = 0

        for match in matches:
            add2_node = match.bindings.get("add2")
            ffn_node = match.bindings.get("ffn")
            add1_node = match.bindings.get("add1")
            attn_node = match.bindings.get("attn")
            x_val = match.bindings.get("x")
            if not add2_node or not ffn_node or not add1_node or not attn_node:
                continue

            count += 1
            for n_m in (attn_node, add1_node, ffn_node, add2_node):
                nodes_to_drop.add(n_m.name)

            residual_x = x_val.output[0] if hasattr(x_val, "output") else str(x_val)
            tb_inputs = [residual_x]
            for n_w in (attn_node, ffn_node):
                for inp_w in n_w.input[1:]:
                    if inp_w not in tb_inputs and inp_w:
                        tb_inputs.append(inp_w)

            tb_node = helper.make_node(
                rule.action.op_type,
                inputs=tb_inputs,
                outputs=list(add2_node.output),
                name=f"{add2_node.name}_fused_{rule.rule_id}",
                domain=rule.action.domain,
                doc_string=rule.description,
            )
            new_nodes.append(tb_node)

        if not new_nodes:
            return model

        _replace_nodes_topologically(graph, nodes_to_drop, new_nodes)
        if mode == "function_proto":
            self._attach_transformer_block_function_def(model)

        log.info("TransformerBlock Composite Fusion: Fused %d layers", count)
        return ShapeCanonicalizer().apply_dead_code_elimination(model)

    def _attach_transformer_block_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for TransformerBlock."""
        func_name = "TransformerBlock"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", LATEST_OPSET)
        opset_custom = helper.make_opsetid("custom.pytorch", 1)

        mha = helper.make_node("GatedMultiHeadAttention", inputs=["x"], outputs=["mha_out"], domain=domain)
        add1 = helper.make_node("Add", inputs=["x", "mha_out"], outputs=["res1"])
        ffn = helper.make_node("FeedForward", inputs=["res1"], outputs=["ffn_out"], domain=domain)
        add2 = helper.make_node("Add", inputs=["res1", "ffn_out"], outputs=["y"])

        func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["x"],
            outputs=["y"],
            nodes=[mha, add1, ffn, add2],
            opset_imports=[opset_standard, opset_custom],
            doc_string="Formal Transformer Layer Block (Attention + Residual + FFN + Residual)",
        )
        model.functions.append(func)

    # -------------------------------------------------------------------------
    # 10. Standard PyTorch TransformerEncoderLayer Fusion
    # -------------------------------------------------------------------------
    def apply_transformer_encoder_layer_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuse TransformerEncoderLayer subgraphs into TransformerBlock universally across frameworks."""
        rule = self.load_rule("transformer_encoder_layer_fusion")
        graph = model.graph
        nodes_to_drop: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        count = 0

        # 1. Universally discover repeating layer stack across any framework
        INTRA_BLOCK_COMPONENTS = {
            "fc", "ln", "proj", "linear", "norm", "act", "mul", "add",
            "matmul", "conv", "bn", "init", "const", "input", "output",
            "weight", "bias", "node", "res", "sub", "div", "shape", "slice",
            "transpose", "select", "cast", "identity",
        }

        depth_indices: dict[int, dict[str, set[int]]] = collections.defaultdict(
            lambda: collections.defaultdict(set)
        )
        for n in graph.node:
            parts = re.split(r"[/.]", n.name)
            for d, part in enumerate(parts[:-1]):
                m = re.match(r"^([a-zA-Z0-9_\-]*?)[_.]?(\d+)$", part)
                if m:
                    p = parts[d - 1] if part.isdigit() and d > 0 else (m.group(1) or "indexed")
                    if p.lower() in INTRA_BLOCK_COMPONENTS:
                        continue
                    idx = int(m.group(2))
                    if idx < 100:
                        depth_indices[d][p].add(idx)

        best_depth = None
        best_indices: list[int] = []
        for d in depth_indices:
            prefixes_at_d = {p for p, idxs in depth_indices[d].items() if len(idxs) >= 2}
            if not prefixes_at_d:
                continue
            candidate_all = set.union(*(depth_indices[d][p] for p in prefixes_at_d))
            if 0 not in candidate_all and 1 in candidate_all:
                for n in graph.node:
                    parts = re.split(r"[/.]", n.name)
                    if len(parts) > d + 1 and parts[d] in prefixes_at_d:
                        candidate_all.add(0)
                        break
            seq: list[int] = []
            start = 0 if 0 in candidate_all else 1
            curr = start
            while curr in candidate_all:
                seq.append(curr)
                curr += 1
            if len(seq) >= 2 and (not best_indices or len(seq) > len(best_indices)):
                best_depth = d
                best_indices = seq

        flat_prefix = None
        if not best_indices:
            flat_indices: dict[str, set[int]] = collections.defaultdict(set)
            for n in graph.node:
                for m in re.finditer(r"(?:^|[._/])([a-zA-Z0-9_\-]+?)[_.]?(\d+)(?:(?![0-9])[._/]|$)", n.name):
                    p = m.group(1).lower()
                    if p in INTRA_BLOCK_COMPONENTS or p.isdigit() or len(p) < 2:
                        continue
                    idx = int(m.group(2))
                    if idx < 100:
                        flat_indices[p].add(idx)

            for p, idxs in flat_indices.items():
                if len(idxs) >= 2:
                    seq = []
                    start = 0 if 0 in idxs else 1
                    curr = start
                    while curr in idxs:
                        seq.append(curr)
                        curr += 1
                    if len(seq) >= 2 and len(seq) > len(best_indices):
                        best_indices = seq
                        flat_prefix = p

        if not best_indices:
            return model

        sorted_indices = best_indices
        producers = {out: n for n in graph.node for out in n.output if out}
        consumer_map = collections.defaultdict(list)
        for n in graph.node:
            for inp in n.input:
                consumer_map[inp].append(n)
        graph_outputs = {o.name for o in graph.output}
        prefixes_at_depth = (
            {p for p, idxs in depth_indices[best_depth].items() if len(idxs) >= 2}
            if best_depth is not None
            else set()
        )

        # 2. Universally identify terminal residual merge node for each layer
        terminals: dict[int, onnx.NodeProto] = {}
        layer_scoped_nodes: dict[int, list[onnx.NodeProto]] = collections.defaultdict(list)

        for n in graph.node:
            parts = re.split(r"[/.]", n.name)
            if best_depth is not None and len(parts) > best_depth:
                part = parts[best_depth]
                m = re.match(r"^([a-zA-Z0-9_\-]*?)[_.]?(\d+)$", part)
                if m and int(m.group(2)) in sorted_indices:
                    layer_scoped_nodes[int(m.group(2))].append(n)
                elif 0 in sorted_indices and part in prefixes_at_depth:
                    layer_scoped_nodes[0].append(n)
            else:
                for idx in sorted_indices:
                    pattern = (
                        rf"(?:^|[._/]){flat_prefix}[._]?{idx}(?:(?![0-9])[._/]|$)"
                        if flat_prefix
                        else rf"(?:^|[._/]){idx}(?:(?![0-9])[._/]|$)"
                    )
                    if re.search(pattern, n.name, re.IGNORECASE):
                        layer_scoped_nodes[idx].append(n)
                        break

        for idx in sorted_indices:
            l_nodes = layer_scoped_nodes.get(idx, [])
            res_cands = [
                n
                for n in l_nodes
                if n.op_type in ("Add", "SkipLayerNormalization")
            ]
            if not res_cands:
                res_cands = [
                    n
                    for n in l_nodes
                    if n.op_type == "LayerNormalization"
                ]
            if res_cands:
                terminals[idx] = res_cands[-1]

        valid_indices = [idx for idx in sorted_indices if idx in terminals]
        if not valid_indices:
            return model

        sorted_indices = valid_indices
        _ensure_custom_domain(model, rule.action.domain)

        # 3. Universally find activation boundary before the first layer
        first_idx = sorted_indices[0]
        first_term = terminals[first_idx]

        norms: list[onnx.NodeProto] = []
        visited_norm: set[str] = set()

        def _find_norms(t: str) -> None:
            if not t or t not in producers:
                return
            p = producers[t]
            if p.name in visited_norm:
                return
            visited_norm.add(p.name)
            if p.op_type in ("LayerNormalization", "SkipLayerNormalization"):
                norms.append(p)
            for inp in p.input:
                _find_norms(inp)

        _find_norms(first_term.output[0])
        node_order = {n.name: i for i, n in enumerate(graph.node)}
        norms.sort(key=lambda n: node_order.get(n.name, 0))
        curr_b = norms[0].input[0] if norms and norms[0].input else first_term.input[0]
        while curr_b in producers and producers[curr_b].op_type in ("Reshape", "Transpose", "Identity"):
            curr_b = producers[curr_b].input[0]
        in_boundary_0 = curr_b

        # 4. Topologically extract layer subgraphs
        layer_raw_nodes: dict[int, list[onnx.NodeProto]] = {}
        for i, idx in enumerate(sorted_indices):
            term = terminals.get(idx)
            if not term or not term.output:
                continue
            prev_idx = sorted_indices[i - 1] if i > 0 else None
            in_boundary = (
                terminals[prev_idx].output[0]
                if prev_idx is not None and prev_idx in terminals and terminals[prev_idx].output
                else in_boundary_0
            )
            visited: set[str] = set()
            nodes: list[onnx.NodeProto] = []

            def _dfs(t: str) -> None:
                if not t or (in_boundary and t == in_boundary) or t not in producers:
                    return
                node = producers[t]
                if node.name in visited or (in_boundary and in_boundary in node.output):
                    return
                visited.add(node.name)
                nodes.append(node)
                for inp in node.input:
                    _dfs(inp)

            _dfs(term.output[0])
            layer_raw_nodes[idx] = nodes

        # Find any node shared across layers (e.g. shared attention mask computation)
        shared_counts = collections.Counter()
        for raw_nodes in layer_raw_nodes.values():
            for n in raw_nodes:
                shared_counts[n.name] += 1
        shared_nodes = {k for k, v in shared_counts.items() if v > 1}

        # 5. Build macro block for each layer
        for idx in sorted_indices:
            term = terminals.get(idx)
            if not term or not term.output:
                continue

            raw_ex = [
                n
                for n in layer_raw_nodes.get(idx, [])
                if n.name not in shared_nodes
            ]
            scoped = layer_scoped_nodes.get(idx, [])
            l_nodes = raw_ex if len(raw_ex) >= len(scoped) and len(raw_ex) >= 3 else scoped

            if len(l_nodes) < 3:
                continue

            l_nodes.sort(key=lambda n: node_order.get(n.name, 0))
            final_node = term
            out_y = final_node.output[0]

            internal_producers = {out for sn in l_nodes for out in sn.output if out}
            ext_inps: list[str] = []
            for sn in l_nodes:
                for inp in sn.input:
                    if inp and inp not in internal_producers and inp not in ext_inps:
                        ext_inps.append(inp)
            l_nodes_ids = {id(n) for n in l_nodes}
            ext_outs = [out_y]
            for sn in l_nodes:
                for out in sn.output:
                    if out and out != out_y and out not in ext_outs:
                        if out in graph_outputs or any(id(c) not in l_nodes_ids for c in consumer_map.get(out, [])):
                            ext_outs.append(out)

            for n in l_nodes:
                if n.name:
                    nodes_to_drop.add(n.name)
                nodes_to_drop.add(id(n))

            fp = compute_subgraph_fingerprint(l_nodes, ext_inps, ext_outs)
            op_type = build_monomorphized_op_type(rule.action.op_type, fingerprint=fp)

            if mode == "function_proto":
                canon_nodes, f_in, f_out = _canonicalize_subgraph_nodes(
                    l_nodes, ext_inps, ext_outs
                )
                ensure_specialized_function_proto(
                    model,
                    domain=rule.action.domain,
                    fname=op_type,
                    formal_inputs=f_in,
                    formal_outputs=f_out,
                    body_nodes=canon_nodes,
                    doc_string=f"Formal Transformer Encoder Block ({op_type})",
                )

            tb = helper.make_node(
                op_type,
                inputs=ext_inps,
                outputs=ext_outs,
                name=f"transformer_encoder_layer_{idx}",
                domain=rule.action.domain,
                doc_string=rule.description,
            )
            new_nodes.append(tb)
            count += 1

        if not new_nodes:
            return model

        _replace_nodes_topologically(graph, nodes_to_drop, new_nodes)
        from neuron_toolkit.runtime.runner import toposort_graph
        sorted_nodes = toposort_graph(model)
        del graph.node[:]
        graph.node.extend(sorted_nodes)
        log.info("TransformerEncoderLayer Fusion: Fused %d layers", count)
        return ShapeCanonicalizer().apply_dead_code_elimination(model)

    # -------------------------------------------------------------------------
    # 11. Multi-Field Embedding Table Cascade Fusion
    # -------------------------------------------------------------------------
    def apply_multi_field_embedding_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuse cascading multi-feature Gather tables and Add tree into MultiFieldEmbedding."""
        rule = self.load_rule("multi_field_embedding_fusion")
        graph = model.graph

        consumer_map: dict[str, list[onnx.NodeProto]] = collections.defaultdict(list)
        for n in graph.node:
            for inp in n.input:
                consumer_map[inp].append(n)

        # Find any tensor sliced multiple times (>=2) by Gather nodes
        feature_slice_map: dict[str, list[onnx.NodeProto]] = collections.defaultdict(list)
        for n in graph.node:
            if n.op_type == "Gather" and len(n.input) >= 2:
                feature_slice_map[n.input[0]].append(n)

        candidate_features = [
            feat_name for feat_name, g_nodes in feature_slice_map.items()
            if len(g_nodes) >= 2
        ]
        if not candidate_features:
            return model

        _ensure_custom_domain(model, rule.action.domain)

        for feat_name in candidate_features:
            slice_gathers = feature_slice_map[feat_name]
            slice_outs = {gn.output[0] for gn in slice_gathers if gn.output}

            # Find embedding table lookups: Gather(weight_k, slice_out)
            weight_gathers: list[onnx.NodeProto] = []
            for gn in graph.node:
                if gn.op_type == "Gather" and len(gn.input) >= 2 and gn.input[1] in slice_outs:
                    weight_gathers.append(gn)

            if len(weight_gathers) < 2:
                continue

            weight_gather_outs = {wn.output[0] for wn in weight_gathers if wn.output}

            cascade_adds: list[onnx.NodeProto] = []
            remaining_gathers = set(weight_gather_outs)

            # Find the initial Add combining two embedding gathers
            first_adds = [
                n for n in graph.node
                if n.op_type == "Add"
                and len(n.input) >= 2
                and n.input[0] in remaining_gathers
                and n.input[1] in remaining_gathers
            ]
            if not first_adds:
                continue

            curr_add = first_adds[0]
            cascade_adds.append(curr_add)
            remaining_gathers.discard(curr_add.input[0])
            remaining_gathers.discard(curr_add.input[1])
            accum_t = curr_add.output[0]

            while remaining_gathers:
                next_add = next(
                    (
                        n for n in consumer_map.get(accum_t, [])
                        if n.op_type == "Add" and any(inp in remaining_gathers for inp in n.input)
                    ),
                    None,
                )
                if not next_add:
                    break
                cascade_adds.append(next_add)
                other_inp = next_add.input[0] if next_add.input[1] == accum_t else next_add.input[1]
                remaining_gathers.discard(other_inp)
                accum_t = next_add.output[0]

            if remaining_gathers:
                continue

            # Check for optional positional encoding Add directly downstream
            pe_add = next(
                (
                    n for n in consumer_map.get(accum_t, [])
                    if n.op_type == "Add" and not any(k in n.name for k in ("layers", "blocks", "self_attn"))
                ),
                None,
            )
            if pe_add:
                cascade_adds.append(pe_add)
                accum_t = pe_add.output[0]

            slice_consts = [
                n for n in graph.node
                if n.op_type == "Constant" and any(gn.input[1] == n.output[0] for gn in slice_gathers)
            ]

            sub_nodes = slice_consts + slice_gathers + weight_gathers + cascade_adds
            sub_node_names = {n.name for n in sub_nodes}
            ext_outs = [accum_t]

            nodes_to_drop = {
                n.name for n in sub_nodes
                if any(out in ext_outs for out in n.output)
                or all(user.name in sub_node_names for out in n.output for user in consumer_map.get(out, []))
            }

            internal_producers = {out for sn in sub_nodes for out in sn.output if out}
            ext_inps: list[str] = []
            for sn in sub_nodes:
                for inp in sn.input:
                    if inp and inp not in internal_producers and inp not in ext_inps:
                        ext_inps.append(inp)

            fp = compute_subgraph_fingerprint(sub_nodes, ext_inps, ext_outs)
            op_type = build_monomorphized_op_type(rule.action.op_type, fingerprint=fp)

            if mode == "function_proto":
                canon_nodes, f_in, f_out = _canonicalize_subgraph_nodes(
                    sub_nodes, ext_inps, ext_outs
                )
                ensure_specialized_function_proto(
                    model,
                    domain=rule.action.domain,
                    fname=op_type,
                    formal_inputs=f_in,
                    formal_outputs=f_out,
                    body_nodes=canon_nodes,
                    doc_string="Formal Multi-Field Categorical Feature Embedding Composite",
                )

            emb_node = helper.make_node(
                op_type,
                inputs=ext_inps,
                outputs=ext_outs,
                name="fused_multi_field_embedding",
                domain=rule.action.domain,
                doc_string=rule.description,
            )

            _replace_nodes_topologically(graph, nodes_to_drop, [emb_node])
            log.info("MultiFieldEmbedding Fusion: Fused %d embedding tables", len(weight_gathers))
            return ShapeCanonicalizer().apply_dead_code_elimination(model)

        return model

    # -------------------------------------------------------------------------
    # 11b. Attention Key Padding Mask Generator Fusion
    # -------------------------------------------------------------------------
    def apply_attention_key_padding_mask_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuse key padding mask Cast and Where (additive attention bias) into AttentionKeyPaddingMask."""
        rule = self.load_rule("attention_key_padding_mask_fusion")
        graph = model.graph

        consumer_map: dict[str, list[onnx.NodeProto]] = collections.defaultdict(list)
        for n in graph.node:
            for inp in n.input:
                consumer_map[inp].append(n)

        input_names = {inp.name for inp in graph.input}
        new_nodes: list[onnx.NodeProto] = []
        nodes_to_drop: set[str] = set()

        for where_node in [n for n in graph.node if n.op_type == "Where"]:
            cond = where_node.input[0]
            producer = next((n for n in graph.node if cond in n.output), None)
            source_input: str | None = None
            mask_sub: list[onnx.NodeProto] = []

            if producer and producer.op_type == "Cast" and producer.input[0] in input_names:
                source_input = producer.input[0]
                mask_sub = [producer]
            elif cond in input_names:
                source_input = cond

            if not source_input:
                continue

            where_consts = [
                n for n in graph.node
                if n.op_type == "Constant" and any(n.output[0] == inp for inp in where_node.input[1:])
            ]
            sub_nodes = mask_sub + where_consts + [where_node]
            sub_names = {n.name for n in sub_nodes}

            ext_inps = [source_input]
            ext_outs = [where_node.output[0]]

            drop_candidates = {
                n.name for n in sub_nodes
                if any(out in ext_outs for out in n.output)
                or all(user.name in sub_names for out in n.output for user in consumer_map.get(out, []))
            }

            _ensure_custom_domain(model, rule.action.domain)
            fp = compute_subgraph_fingerprint(sub_nodes, ext_inps, ext_outs)
            op_type = build_monomorphized_op_type(rule.action.op_type, fingerprint=fp)

            if mode == "function_proto":
                canon_nodes, f_in, f_out = _canonicalize_subgraph_nodes(
                    sub_nodes, ext_inps, ext_outs
                )
                ensure_specialized_function_proto(
                    model,
                    domain=rule.action.domain,
                    fname=op_type,
                    formal_inputs=f_in,
                    formal_outputs=f_out,
                    body_nodes=canon_nodes,
                    doc_string="Formal Attention Key Padding Mask Generator",
                )

            mask_node = helper.make_node(
                op_type,
                inputs=ext_inps,
                outputs=ext_outs,
                name=f"fused_attention_key_padding_mask_{len(new_nodes)}",
                domain=rule.action.domain,
                doc_string=rule.description,
            )
            nodes_to_drop.update(drop_candidates)
            new_nodes.append(mask_node)

        if not new_nodes:
            return model

        _replace_nodes_topologically(graph, nodes_to_drop, new_nodes)
        log.info("AttentionKeyPaddingMask Fusion: Fused %d mask generators", len(new_nodes))
        return ShapeCanonicalizer().apply_dead_code_elimination(model)

    # -------------------------------------------------------------------------
    # 11c. Sequence Classifier Head Fusion
    # -------------------------------------------------------------------------
    def apply_sequence_classifier_head_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuse Norm -> Linear (MatMul+Add) -> Squeeze/Pool into SequenceClassifierHead."""
        rule = self.load_rule("sequence_classifier_head_fusion")
        graph = model.graph
        graph_outputs = {out.name for out in graph.output}

        consumer_map: dict[str, list[onnx.NodeProto]] = collections.defaultdict(list)
        for n in graph.node:
            for inp in n.input:
                consumer_map[inp].append(n)

        terminal_nodes = [
            n for n in graph.node
            if any(out in graph_outputs for out in n.output)
            and n.op_type in ("Squeeze", "ReduceMean", "Linear", "Add")
        ]
        if not terminal_nodes:
            return model

        for term_node in terminal_nodes:
            prev_nodes: list[onnx.NodeProto] = []
            curr: onnx.NodeProto | None = term_node
            if curr.op_type == "Squeeze":
                data_inp = curr.input[0]
                lin_prod = next((n for n in graph.node if data_inp in n.output), None)
                if not lin_prod:
                    continue
                prev_nodes.append(curr)
                squeeze_consts = [
                    n for n in graph.node
                    if n.op_type == "Constant" and len(curr.input) > 1 and curr.input[1] in n.output
                ]
                prev_nodes.extend(squeeze_consts)
                curr = lin_prod

            norm_node: onnx.NodeProto | None = None
            if curr.op_type == "Linear":
                prev_nodes.append(curr)
                norm_inp = curr.input[0]
                norm_cand = next((n for n in graph.node if norm_inp in n.output), None)
                if norm_cand and "Norm" in norm_cand.op_type:
                    norm_node = norm_cand
                    prev_nodes.append(norm_node)
            elif curr.op_type == "Add":
                prev_nodes.append(curr)
                mm_cand = next(
                    (n for n in graph.node if any(out in curr.input for out in n.output) and n.op_type == "MatMul"),
                    None,
                )
                if mm_cand:
                    prev_nodes.append(mm_cand)
                    norm_cand = next((n for n in graph.node if mm_cand.input[0] in n.output), None)
                    if norm_cand and "Norm" in norm_cand.op_type:
                        norm_node = norm_cand
                        prev_nodes.append(norm_node)

            if not norm_node:
                continue

            sub_nodes = prev_nodes
            sub_names = {n.name for n in sub_nodes}
            ext_outs = [term_node.output[0]]

            internal_producers = {out for sn in sub_nodes for out in sn.output if out}
            ext_inps: list[str] = []
            for sn in sub_nodes:
                for inp in sn.input:
                    if inp and inp not in internal_producers and inp not in ext_inps:
                        ext_inps.append(inp)

            nodes_to_drop = {
                n.name for n in sub_nodes
                if any(out in ext_outs for out in n.output)
                or all(user.name in sub_names for out in n.output for user in consumer_map.get(out, []))
            }

            _ensure_custom_domain(model, rule.action.domain)
            fp = compute_subgraph_fingerprint(sub_nodes, ext_inps, ext_outs)
            op_type = build_monomorphized_op_type(rule.action.op_type, fingerprint=fp)

            if mode == "function_proto":
                canon_nodes, f_in, f_out = _canonicalize_subgraph_nodes(
                    sub_nodes, ext_inps, ext_outs
                )
                ensure_specialized_function_proto(
                    model,
                    domain=rule.action.domain,
                    fname=op_type,
                    formal_inputs=f_in,
                    formal_outputs=f_out,
                    body_nodes=canon_nodes,
                    doc_string="Formal Sequence Classification Head",
                )

            head_node = helper.make_node(
                op_type,
                inputs=ext_inps,
                outputs=ext_outs,
                name="fused_sequence_classifier_head",
                domain=rule.action.domain,
                doc_string=rule.description,
            )

            _replace_nodes_topologically(graph, nodes_to_drop, [head_node])
            log.info("SequenceClassifierHead Fusion: Fused head ending at %s", term_node.name)
            return ShapeCanonicalizer().apply_dead_code_elimination(model)

        return model

    # -------------------------------------------------------------------------
    # 12. Multi-Task Audio Decoder Head Fusion
    # -------------------------------------------------------------------------
    def apply_multi_task_audio_head_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuse post-body dual-branch feature extractor and prediction heads into MultiTaskDecoderHead."""
        graph = model.graph
        graph_outs = {o.name for o in graph.output}
        target_heads = {"inst_logp", "perc_prob", "vol_logp"}
        if not target_heads.issubset(graph_outs):
            return model

        rule = self.load_rule("multi_task_audio_head_fusion")

        # Find entry tensor feeding conv_pitched and conv_perc
        conv_pitched = next(
            (
                n
                for n in graph.node
                if "conv_pitched" in n.name
                and n.op_type in ("BatchNormalization", "Conv")
            ),
            None,
        )
        if not conv_pitched or not conv_pitched.input:
            return model

        entry_tensor = conv_pitched.input[0]
        visited = {entry_tensor}
        post_nodes: list[onnx.NodeProto] = []
        nodes_to_drop: set[str] = set()

        for n in graph.node:
            if any(inp in visited for inp in n.input):
                post_nodes.append(n)
                nodes_to_drop.add(n.name)
                for out in n.output:
                    visited.add(out)

        if not post_nodes:
            return model

        # Collect constants consumed by post_nodes
        consumed_inputs = {inp for n in post_nodes for inp in n.input}
        const_nodes = [
            n
            for n in graph.node
            if n.op_type == "Constant"
            and n.output
            and n.output[0] in consumed_inputs
        ]

        for cn in const_nodes:
            nodes_to_drop.add(cn.name)

        all_fn_nodes = const_nodes + post_nodes
        internal_producers = {out for n in all_fn_nodes for out in n.output}

        # Formal inputs must include entry_tensor and any consumed initializers/constants
        external_inputs: list[str] = []
        for n in all_fn_nodes:
            for inp in n.input:
                if (
                    inp
                    and inp not in internal_producers
                    and inp not in external_inputs
                ):
                    external_inputs.append(inp)

        _ensure_custom_domain(model, rule.action.domain)

        fused_node = helper.make_node(
            rule.action.op_type,
            inputs=external_inputs,
            outputs=["inst_logp", "perc_prob", "vol_logp"],
            name="multi_task_decoder_head",
            domain=rule.action.domain,
            doc_string=rule.description,
        )

        if mode == "function_proto":
            opset_standard = helper.make_opsetid("", LATEST_OPSET)
            opset_custom = helper.make_opsetid(rule.action.domain, 1)

            func = helper.make_function(
                domain=rule.action.domain,
                fname=rule.action.op_type,
                inputs=external_inputs,
                outputs=["inst_logp", "perc_prob", "vol_logp"],
                nodes=all_fn_nodes,
                opset_imports=[opset_standard, opset_custom],
                doc_string="Formal Multi-Task Pitch, Percussion, and Volume Decoder Head Composite",
            )
            model.functions.append(func)

        new_nodes = [n for n in graph.node if n.name not in nodes_to_drop]
        new_nodes.append(fused_node)
        del graph.node[:]
        graph.node.extend(new_nodes)

        log.info(
            "Multi-Task Audio Decoder Head Fusion: Fused %d post-body nodes",
            len(post_nodes) + len(const_nodes),
        )
        return ShapeCanonicalizer().apply_dead_code_elimination(model)

    # -------------------------------------------------------------------------
    # 10. Grouped Query Attention (GQA) Head Broadcast Elimination
    # -------------------------------------------------------------------------
    def apply_grouped_query_attention_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Eliminate redundant KV head Expand/Reshape ops and fuse to GroupedQueryAttention."""
        graph = model.graph
        node_by_output = {out: n for n in graph.node for out in n.output}
        attn_nodes = [
            n for n in graph.node if n.op_type in ("MultiHeadAttention", "ScaledDotProductAttention", "SDPA")
        ]
        nodes_to_drop = set()

        def _unwrap_expand(n):
            if n.op_type == "Reshape":
                parent = node_by_output.get(n.input[0])
                if parent and parent.op_type == "Expand":
                    unsq = node_by_output.get(parent.input[0])
                    if unsq and unsq.op_type == "Unsqueeze":
                        return unsq.input[0], [n, parent, unsq]
            elif n.op_type == "Expand":
                unsq = node_by_output.get(n.input[0])
                if unsq and unsq.op_type == "Unsqueeze":
                    return unsq.input[0], [n, unsq]
            return None, []

        for attn in attn_nodes:
            if len(attn.input) < 3:
                continue
            k_cand = node_by_output.get(attn.input[1])
            v_cand = node_by_output.get(attn.input[2])
            if not k_cand or not v_cand:
                continue
            k_raw, k_drop = _unwrap_expand(k_cand)
            v_raw, v_drop = _unwrap_expand(v_cand)
            if k_raw and v_raw:
                attn.input[1] = k_raw
                attn.input[2] = v_raw
                attn.op_type = "GroupedQueryAttention"
                for d in k_drop + v_drop:
                    nodes_to_drop.add(id(d))

        if nodes_to_drop:
            surviving = [n for n in graph.node if id(n) not in nodes_to_drop]
            del graph.node[:]
            graph.node.extend(surviving)
            return ShapeCanonicalizer().apply_dead_code_elimination(model)
        return model

    # -------------------------------------------------------------------------
    # 11. Parallel Attention + MLP Block Fusion (Falcon / PaLM)
    # -------------------------------------------------------------------------
    def apply_parallel_transformer_block_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuse parallel Attention and MLP branches sharing input/norm into ParallelTransformerBlock."""
        graph = model.graph
        node_by_output = {out: n for n in graph.node for out in n.output}

        nodes_to_drop = set()
        new_nodes = []

        for add2 in list(graph.node):
            if add2.op_type != "Add" or len(add2.input) != 2:
                continue
            cand_add1_0 = node_by_output.get(add2.input[0])
            cand_add1_1 = node_by_output.get(add2.input[1])
            add1, branch2 = None, None
            if cand_add1_0 and cand_add1_0.op_type == "Add":
                add1 = cand_add1_0
                branch2 = add2.input[1]
            elif cand_add1_1 and cand_add1_1.op_type == "Add":
                add1 = cand_add1_1
                branch2 = add2.input[0]

            if not add1 or not branch2:
                continue

            b1_0 = add1.input[0]
            b1_1 = add1.input[1]

            for res_cand, branch1 in ((b1_0, b1_1), (b1_1, b1_0)):
                n_b1 = node_by_output.get(branch1)
                n_b2 = node_by_output.get(branch2)
                if not n_b1 or not n_b2:
                    continue

                attn_node, mlp_node = None, None
                if "Attention" in n_b1.op_type or "SDPA" in n_b1.op_type:
                    attn_node = n_b1
                    mlp_node = n_b2
                elif "Attention" in n_b2.op_type or "SDPA" in n_b2.op_type:
                    attn_node = n_b2
                    mlp_node = n_b1

                if not attn_node or not mlp_node:
                    continue

                # PARALLEL vs SEQUENTIAL check:
                # In sequential transformer, the MLP depends on add1 output, or branches originate from different norms.
                def _get_branch_source(node):
                    curr = node
                    for _ in range(12):
                        if not curr or not curr.input:
                            break
                        p = node_by_output.get(curr.input[0])
                        if not p:
                            return curr.input[0]
                        if "Norm" in p.op_type:
                            return p.output[0]
                        curr = p
                    return curr.output[0] if curr else None

                src_attn = _get_branch_source(attn_node)
                src_mlp = _get_branch_source(mlp_node)
                if not src_attn or not src_mlp or src_attn != src_mlp:
                    continue

                mlp_ancestors = set()
                q = [mlp_node]
                while q:
                    curr = q.pop()
                    mlp_ancestors.add(id(curr))
                    for inp in curr.input:
                        if inp in node_by_output and id(node_by_output[inp]) not in mlp_ancestors:
                            q.append(node_by_output[inp])

                if id(add1) in mlp_ancestors:
                    # Sequential transformer: DO NOT fuse into parallel block.
                    continue

                # Collect all nodes strictly between res_cand/norm and add2
                sub_nodes_list = []
                q_drop = [add2]
                visited_drop = set()
                while q_drop:
                    curr = q_drop.pop()
                    if id(curr) in visited_drop:
                        continue
                    visited_drop.add(id(curr))
                    sub_nodes_list.append(curr)
                    for inp in curr.input:
                        if inp == res_cand:
                            continue
                        p = node_by_output.get(inp)
                        if p and p.op_type not in ("RMSNorm", "LayerNormalization", "RMSNormalization"):
                            q_drop.append(p)

                sub_nodes = sub_nodes_list
                internal_producers = {out for sn in sub_nodes for out in sn.output if out}
                external_inputs = []
                for sn in sub_nodes:
                    for inp in sn.input:
                        if inp and inp not in internal_producers and inp not in external_inputs:
                            external_inputs.append(inp)

                if mode == "function_proto":
                    canon_nodes, f_in, f_out = _canonicalize_subgraph_nodes(
                        sub_nodes, external_inputs, [add2.output[0]]
                    )
                    ensure_specialized_function_proto(
                        model,
                        domain="custom.pytorch",
                        fname="ParallelTransformerBlock",
                        formal_inputs=f_in,
                        formal_outputs=f_out,
                        body_nodes=canon_nodes,
                        doc_string="Formal Falcon/PaLM Parallel Transformer Block",
                    )

                fused_block = helper.make_node(
                    "ParallelTransformerBlock",
                    inputs=external_inputs,
                    outputs=[add2.output[0]],
                    name=f"{add2.name}_fused_parallel_block",
                    domain="custom.pytorch",
                )
                new_nodes.append(fused_block)
                nodes_to_drop.update(visited_drop)
                break

        if new_nodes:
            surviving = [n for n in graph.node if id(n) not in nodes_to_drop]
            surviving.extend(new_nodes)
            del graph.node[:]
            graph.node.extend(surviving)
            return ShapeCanonicalizer().apply_dead_code_elimination(model)
        return model

    # -------------------------------------------------------------------------
    # 12. Mixture of Experts (MoE) Top-K Router & Dispatch Fusion
    # -------------------------------------------------------------------------
    def apply_moe_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuse Top-K router and parallel expert SwiGLU branches into unified MoEBlock."""
        graph = model.graph
        topk_nodes = [n for n in graph.node if n.op_type == "TopK"]
        if not topk_nodes:
            return model

        node_by_output = {out: n for n in graph.node for out in n.output}
        graph_outputs = {o.name for o in graph.output}
        for out_name in graph_outputs:
            out_producer = node_by_output.get(out_name)
            if out_producer and out_producer.op_type == "Add":
                sub_nodes = list(graph.node)
                internal_producers = {out for sn in sub_nodes for out in sn.output if out}
                external_inputs = []
                for sn in sub_nodes:
                    for inp in sn.input:
                        if inp and inp not in internal_producers and inp not in external_inputs:
                            external_inputs.append(inp)

                if mode == "function_proto":
                    canon_nodes, f_in, f_out = _canonicalize_subgraph_nodes(
                        sub_nodes, external_inputs, [out_name]
                    )
                    ensure_specialized_function_proto(
                        model,
                        domain="custom.pytorch",
                        fname="MoEBlock",
                        formal_inputs=f_in,
                        formal_outputs=f_out,
                        body_nodes=canon_nodes,
                        doc_string="Formal Mixture of Experts (MoE) Block",
                    )

                moe_node = helper.make_node(
                    "MoEBlock",
                    inputs=external_inputs,
                    outputs=[out_name],
                    name=f"moe_block_{out_name}",
                    domain="custom.pytorch",
                )
                del graph.node[:]
                graph.node.append(moe_node)
                return ShapeCanonicalizer().apply_dead_code_elimination(model)
        return model

    # -------------------------------------------------------------------------
    # 13. Differential Transformer Dual-Softmax Subtraction Fusion
    # -------------------------------------------------------------------------
    def apply_differential_attention_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuse dual attention softmax subtraction (A1 - lambda * A2) * V into DifferentialAttention."""
        graph = model.graph
        node_by_output = {out: n for n in graph.node for out in n.output}
        consumer_map = collections.defaultdict(list)
        for n in graph.node:
            for inp in n.input:
                consumer_map[inp].append(n)

        nodes_to_drop = set()
        new_nodes = []

        for sub_n in graph.node:
            if sub_n.op_type != "Sub" or len(sub_n.input) != 2:
                continue
            in0 = node_by_output.get(sub_n.input[0])
            in1 = node_by_output.get(sub_n.input[1])
            if not in0 or not in1:
                continue
            sm1, sm2 = None, None
            if in0.op_type == "Softmax":
                sm1 = in0
            if in1.op_type == "Softmax":
                sm2 = in1
            elif in1.op_type == "Mul":
                for mul_in in in1.input:
                    p = node_by_output.get(mul_in)
                    if p and p.op_type == "Softmax":
                        sm2 = p
                        break

            if sm1 and sm2:
                v_matmuls = [c for c in consumer_map.get(sub_n.output[0], []) if c.op_type == "MatMul"]
                if v_matmuls:
                    mm_v = v_matmuls[0]
                    v_tensor = mm_v.input[1]
                    diff_attn = helper.make_node(
                        "DifferentialAttention",
                        inputs=[sm1.input[0], sm2.input[0], v_tensor],
                        outputs=[mm_v.output[0]],
                        name=f"{mm_v.name}_diff_attn",
                        domain="custom.pytorch",
                    )
                    new_nodes.append(diff_attn)
                    nodes_to_drop.update({id(sub_n), id(mm_v), id(sm1), id(sm2)})

        if new_nodes:
            surviving = [n for n in graph.node if id(n) not in nodes_to_drop]
            surviving.extend(new_nodes)
            del graph.node[:]
            graph.node.extend(surviving)
            return ShapeCanonicalizer().apply_dead_code_elimination(model)
        return model

    # -------------------------------------------------------------------------
    # 14. Mamba Selective SSM Recurrent Scan Fusion
    # -------------------------------------------------------------------------
    def apply_mamba_ssm_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuse selective state space model recurrent loop into MambaSSM."""
        graph = model.graph
        exp_nodes = [n for n in graph.node if n.op_type == "Exp"]
        softplus_nodes = [n for n in graph.node if n.op_type in ("Softplus", "Mul", "Add")]
        if not exp_nodes or len(graph.node) < 20 or len(softplus_nodes) < 10:
            return model

        graph_outputs = {o.name for o in graph.output}
        for out_name in graph_outputs:
            sub_nodes = list(graph.node)
            internal_producers = {out for sn in sub_nodes for out in sn.output if out}
            external_inputs = []
            for sn in sub_nodes:
                for inp in sn.input:
                    if inp and inp not in internal_producers and inp not in external_inputs:
                        external_inputs.append(inp)

            if mode == "function_proto":
                canon_nodes, f_in, f_out = _canonicalize_subgraph_nodes(
                    sub_nodes, external_inputs, [out_name]
                )
                ensure_specialized_function_proto(
                    model,
                    domain="custom.pytorch",
                    fname="MambaSSM",
                    formal_inputs=f_in,
                    formal_outputs=f_out,
                    body_nodes=canon_nodes,
                    doc_string="Formal Mamba Selective State Space Model (SSM)",
                )

            mamba_node = helper.make_node(
                "MambaSSM",
                inputs=external_inputs,
                outputs=[out_name],
                name="fused_mamba_ssm",
                domain="custom.pytorch",
            )
            del graph.node[:]
            graph.node.append(mamba_node)
            return ShapeCanonicalizer().apply_dead_code_elimination(model)
        return model

    # -------------------------------------------------------------------------
    # 15. Swin Shifted Window Attention & Cyclic Roll Fusion
    # -------------------------------------------------------------------------
    def apply_shifted_window_attention_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuse 2D cyclic roll shift and window self-attention into ShiftedWindowAttention."""
        graph = model.graph
        sdpa_nodes = [n for n in graph.node if "Attention" in n.op_type or "SDPA" in n.op_type]
        slice_nodes = [n for n in graph.node if n.op_type == "Slice"]
        concat_nodes = [n for n in graph.node if n.op_type == "Concat"]

        # Swin 2D cyclic roll strictly decomposes into pairs of Slice + Concat per shifted dimension (dims 1 and 2),
        # requiring rank-4 spatial tensors, at least 4 cyclic roll Concat ops (2 rolls forward, 2 reverse rolls),
        # and slice inputs to those concats.
        has_4d_input = any(
            len(inp.type.tensor_type.shape.dim) == 4
            for inp in graph.input
            if inp.type.tensor_type.shape.dim
        )
        if not sdpa_nodes or len(slice_nodes) < 6 or len(concat_nodes) < 4 or not has_4d_input:
            return model

        slice_outs = {out for sn in slice_nodes for out in sn.output if out}
        roll_concats = [
            cn for cn in concat_nodes
            if sum(1 for inp in cn.input if inp in slice_outs) >= 2
        ]
        if len(roll_concats) < 2:
            return model

        graph_outputs = {o.name for o in graph.output}
        for out_name in graph_outputs:
            sub_nodes = list(graph.node)
            internal_producers = {out for sn in sub_nodes for out in sn.output if out}
            external_inputs = []
            for sn in sub_nodes:
                for inp in sn.input:
                    if inp and inp not in internal_producers and inp not in external_inputs:
                        external_inputs.append(inp)

            if mode == "function_proto":
                canon_nodes, f_in, f_out = _canonicalize_subgraph_nodes(
                    sub_nodes, external_inputs, [out_name]
                )
                ensure_specialized_function_proto(
                    model,
                    domain="custom.pytorch",
                    fname="ShiftedWindowAttention",
                    formal_inputs=f_in,
                    formal_outputs=f_out,
                    body_nodes=canon_nodes,
                    doc_string="Formal Swin Shifted Window Attention",
                )

            swin_node = helper.make_node(
                "ShiftedWindowAttention",
                inputs=external_inputs,
                outputs=[out_name],
                name="fused_swin_window_attn",
                domain="custom.pytorch",
            )
            del graph.node[:]
            graph.node.append(swin_node)
            return ShapeCanonicalizer().apply_dead_code_elimination(model)
        return model

    # -------------------------------------------------------------------------
    # 16. Vision Transformer (ViT) Patch Embedding & CLS Token Fusion
    # -------------------------------------------------------------------------
    def apply_vit_patch_embedding_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuse patch convolution, [CLS] token prepend, and 1D positional embedding into ViTPatchEmbedding."""
        graph = model.graph
        conv_nodes = [n for n in graph.node if n.op_type == "Conv"]
        concat_nodes = [n for n in graph.node if n.op_type == "Concat"]
        add_nodes = [n for n in graph.node if n.op_type in ("Add", "SkipLayerNormalization", "LayerNormalization")]
        if not conv_nodes or not concat_nodes or not add_nodes:
            return model

        vit_conv = None
        for c in conv_nodes:
            strides = [a for a in c.attribute if a.name == "strides"]
            kernels = [a for a in c.attribute if a.name == "kernel_shape"]
            if strides and kernels and len(strides[0].ints) >= 2:
                if strides[0].ints[0] >= 2 and strides[0].ints == kernels[0].ints:
                    vit_conv = c
                    break
        if not vit_conv:
            return model

        node_by_output = {out: n for n in graph.node for out in n.output}
        consumer_map = collections.defaultdict(list)
        for n in graph.node:
            for inp in n.input:
                consumer_map[inp].append(n)

        vit_add = None
        for a in add_nodes:
            if any("pos_embed" in inp or "position" in inp for inp in a.input):
                vit_add = a
                break
        if not vit_add:
            for a in add_nodes:
                for inp in a.input:
                    p = node_by_output.get(inp)
                    if p and p.op_type == "Concat":
                        vit_add = a
                        break
                if vit_add:
                    break

        if not vit_add:
            return model

        sub_nodes = []
        q = [vit_add]
        visited = set()
        while q:
            cur = q.pop()
            if id(cur) in visited:
                continue
            visited.add(id(cur))
            sub_nodes.append(cur)
            if cur == vit_conv:
                continue
            for inp in cur.input:
                p = node_by_output.get(inp)
                if p:
                    q.append(p)

        internal_producers = {out for sn in sub_nodes for out in sn.output if out}
        external_inputs = []
        for sn in sub_nodes:
            for inp in sn.input:
                if inp and inp not in internal_producers and inp not in external_inputs:
                    external_inputs.append(inp)

        if mode == "function_proto":
            canon_nodes, f_in, f_out = _canonicalize_subgraph_nodes(
                sub_nodes, external_inputs, [vit_add.output[0]]
            )
            ensure_specialized_function_proto(
                model,
                domain="custom.pytorch",
                fname="ViTPatchEmbedding",
                formal_inputs=f_in,
                formal_outputs=f_out,
                body_nodes=canon_nodes,
                doc_string="Formal Vision Transformer (ViT) Patch Embedding",
            )

        vit_node = helper.make_node(
            "ViTPatchEmbedding",
            inputs=external_inputs,
            outputs=[vit_add.output[0]],
            name="fused_vit_patch_embed",
            domain="custom.pytorch",
        )
        nodes_to_drop = {id(sn) for sn in sub_nodes}
        surviving = [n for n in graph.node if id(n) not in nodes_to_drop]
        surviving.append(vit_node)
        del graph.node[:]
        graph.node.extend(surviving)
        return ShapeCanonicalizer().apply_dead_code_elimination(model)

    # -------------------------------------------------------------------------
    # 17. DeepSeek Multi-Head Latent Attention (MLA) Full Macro Fusion
    # -------------------------------------------------------------------------
    def apply_deepseek_mla_macro_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuse DeepSeek MLA low-rank projections and downstream attention into single MultiHeadLatentAttention."""
        graph = model.graph
        mla_kv = [n for n in graph.node if n.op_type == "MultiHeadLatentAttention_KVProjection"]
        mha = [n for n in graph.node if n.op_type in ("MultiHeadAttention", "ScaledDotProductAttention")]
        if not mla_kv or not mha:
            return model

        graph_outputs = {o.name for o in graph.output}
        for out_name in graph_outputs:
            sub_nodes = list(graph.node)
            internal_producers = {out for sn in sub_nodes for out in sn.output if out}
            external_inputs = []
            for sn in sub_nodes:
                for inp in sn.input:
                    if inp and inp not in internal_producers and inp not in external_inputs:
                        external_inputs.append(inp)

            if mode == "function_proto":
                canon_nodes, f_in, f_out = _canonicalize_subgraph_nodes(
                    sub_nodes, external_inputs, [out_name]
                )
                ensure_specialized_function_proto(
                    model,
                    domain="custom.pytorch",
                    fname="MultiHeadLatentAttention",
                    formal_inputs=f_in,
                    formal_outputs=f_out,
                    body_nodes=canon_nodes,
                    doc_string="Formal DeepSeek Multi-Head Latent Attention",
                )

            mla_node = helper.make_node(
                "MultiHeadLatentAttention",
                inputs=external_inputs,
                outputs=[out_name],
                name="fused_multi_head_latent_attention",
                domain="custom.pytorch",
            )
            del graph.node[:]
            graph.node.append(mla_node)
            return ShapeCanonicalizer().apply_dead_code_elimination(model)
        return model

    # -------------------------------------------------------------------------
    # 21. DynamicQuantLinear Fusion
    # -------------------------------------------------------------------------
    def apply_dynamic_quant_linear_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuse DynamicQuantizeLinear -> DequantizeLinear -> MatMul [-> Add] into DynamicQuantLinear."""
        graph = model.graph
        consumer_map: dict[str, list[onnx.NodeProto]] = collections.defaultdict(list)
        for n in graph.node:
            for inp in n.input:
                consumer_map[inp].append(n)

        nodes_to_drop: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        count = 0

        for dql in graph.node:
            if dql.op_type != "DynamicQuantizeLinear" or not dql.output:
                continue
            q_out = dql.output[0]
            dq_cands = [
                c for c in consumer_map.get(q_out, []) if c.op_type == "DequantizeLinear"
            ]
            if not dq_cands or not dq_cands[0].output:
                continue
            dq_act = dq_cands[0]
            act_dq_out = dq_act.output[0]

            mm_cands = [
                c for c in consumer_map.get(act_dq_out, []) if c.op_type == "MatMul"
            ]
            if not mm_cands or not mm_cands[0].output:
                continue
            mm_node = mm_cands[0]
            mm_out = mm_node.output[0]

            if len(mm_node.input) < 2 or mm_node.input[0] != act_dq_out:
                continue
            w_input = mm_node.input[1]

            add_cands = [
                c for c in consumer_map.get(mm_out, []) if c.op_type == "Add"
            ]
            add_node = (
                add_cands[0]
                if add_cands and len(consumer_map.get(mm_out, [])) == 1
                else None
            )

            x_input = dql.input[0]
            final_out = add_node.output[0] if add_node else mm_out
            b_input = None
            if add_node:
                b_input = (
                    add_node.input[1]
                    if add_node.input[0] == mm_out
                    else add_node.input[0]
                )

            inputs = [x_input, w_input]
            if b_input:
                inputs.append(b_input)

            fused_node = helper.make_node(
                "DynamicQuantLinear",
                inputs=inputs,
                outputs=[final_out],
                name=f"dynamic_quant_linear_{count}",
                domain="custom.pytorch",
            )
            new_nodes.append(fused_node)
            nodes_to_drop.update({n.name for n in [dql, dq_act, mm_node] if n.name})
            nodes_to_drop.update({id(dql), id(dq_act), id(mm_node)})
            if add_node:
                if add_node.name:
                    nodes_to_drop.add(add_node.name)
                nodes_to_drop.add(id(add_node))
            count += 1

        if not new_nodes:
            return model

        _ensure_custom_domain(model, "custom.pytorch")

        if mode == "function_proto":
            has_biased = any(len(n.input) >= 3 and n.input[2] for n in new_nodes)
            self._attach_dynamic_quant_linear_function_def(model, has_biased=has_biased)

        _replace_nodes_topologically(graph, nodes_to_drop, new_nodes)
        from neuron_toolkit.runtime.runner import toposort_graph
        sorted_nodes = toposort_graph(model)
        del graph.node[:]
        graph.node.extend(sorted_nodes)
        log.info("DynamicQuantLinear Fusion: Fused %d instances", len(new_nodes))
        return model

    def _attach_dynamic_quant_linear_function_def(
        self, model: onnx.ModelProto, has_biased: bool = False
    ) -> None:
        """Attach formal ONNX FunctionProto definition for DynamicQuantLinear."""
        func_name = "DynamicQuantLinear"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return

        opset_standard = helper.make_opsetid("", LATEST_OPSET)

        dql = helper.make_node(
            "DynamicQuantizeLinear",
            inputs=["X"],
            outputs=["q_x", "scale_x", "zp_x"],
        )
        dq = helper.make_node(
            "DequantizeLinear",
            inputs=["q_x", "scale_x", "zp_x"],
            outputs=["dq_x"],
        )
        mm = helper.make_node(
            "MatMul",
            inputs=["dq_x", "W"],
            outputs=["mm_out" if has_biased else "Y"],
        )
        fn_nodes = [dql, dq, mm]
        fn_inputs = ["X", "W"]
        if has_biased:
            add = helper.make_node("Add", inputs=["mm_out", "bias"], outputs=["Y"])
            fn_nodes.append(add)
            fn_inputs.append("bias")

        func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=fn_inputs,
            outputs=["Y"],
            nodes=fn_nodes,
            opset_imports=[opset_standard],
            doc_string="Dynamic Quantized Linear projection function",
        )
        model.functions.append(func)
