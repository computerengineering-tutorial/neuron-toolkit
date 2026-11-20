"""FeedForward / MLP Block Synthesizer Pass.

Synthesizes macro-blocks:
- FeedForward GELU (RMSNorm -> Linear -> Gelu -> Linear) -> FeedForward
- FeedForward SwiGLU (RMSNorm -> Linear_gate * SiLU * Linear_up -> Linear_down) -> FeedForwardSwiGLU
Attaches formal FunctionProto definitions for Netron expandable hierarchical visualization.
"""

from __future__ import annotations

import collections
import logging

import onnx
from onnx import helper

from neuron_toolkit.compiler.architectures.fingerprint import (
    build_monomorphized_op_type,
    compute_subgraph_fingerprint,
    ensure_specialized_function_proto,
)
from neuron_toolkit.compiler.base import CompilerContext, Pass, LATEST_OPSET, ensure_model_latest_opset

log = logging.getLogger("neuron_toolkit.compiler.feedforward_synthesizer")


class FeedForwardSynthesizer(Pass):
    """Synthesizes high-level FeedForward / MLP transformer blocks."""

    @property
    def name(self) -> str:
        return "feedforward_synthesizer"

    @property
    def category(self) -> str:
        return "synthesis"

    def run(self, model: onnx.ModelProto, context: CompilerContext) -> onnx.ModelProto:
        nodes_before = len(model.graph.node)

        if context.should_run("feedforward"):
            model = self.apply_feedforward_fusion(model, mode=context.mode)
        if context.should_run("feedforward_swiglu") or context.should_run("swiglu"):
            model = self.apply_feedforward_swiglu_fusion(model, mode=context.mode)

        nodes_after = len(model.graph.node)
        context.record_pass(self.name, nodes_before, nodes_after)
        return model

    # -------------------------------------------------------------------------
    # FeedForward GELU Fusion
    # -------------------------------------------------------------------------
    def apply_feedforward_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuse 4-node FeedForward / MLP block (RMSNorm -> Linear -> Gelu -> Linear) into FeedForward."""
        graph = model.graph
        node_by_output: dict[str, onnx.NodeProto] = {
            out: n for n in graph.node for out in n.output
        }
        consumer_map: dict[str, list[onnx.NodeProto]] = collections.defaultdict(list)
        for n in graph.node:
            for inp in n.input:
                consumer_map[inp].append(n)

        nodes_to_drop: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        ff_count = 0

        for n in graph.node:
            if n.op_type not in ("Gelu", "FastGelu", "Relu", "SiLU", "Tanh", "Mish"):
                continue
            lin1 = node_by_output.get(n.input[0])
            if not lin1 or lin1.op_type != "Linear":
                continue
            norm = node_by_output.get(lin1.input[0])
            valid_norms = ("RMSNorm", "RMSNormalization", "LayerNormalization", "SimplifiedLayerNormalization")
            if not norm or norm.op_type not in valid_norms:
                continue

            cons = consumer_map.get(n.output[0], [])
            if len(cons) != 1 or cons[0].op_type != "Linear":
                continue
            lin2 = cons[0]

            lin1_cons = consumer_map.get(lin1.output[0], [])
            if len(lin1_cons) != 1:
                continue

            act_name = n.op_type
            norm_op = norm.op_type
            sub_nodes = [norm, lin1, n, lin2]
            fp = compute_subgraph_fingerprint(sub_nodes)
            specialized_op = build_monomorphized_op_type("FeedForward", act_name, fp)

            x = norm.input[0]
            norm_gamma = norm.input[1]
            fc1_w = lin1.input[1]
            fc1_b = lin1.input[2] if len(lin1.input) > 2 else ""
            fc2_w = lin2.input[1]
            fc2_b = lin2.input[2] if len(lin2.input) > 2 else ""
            final_out = lin2.output[0]

            ff_inputs = [x, norm_gamma, fc1_w, fc1_b, fc2_w, fc2_b]
            norm_beta = norm.input[2] if len(norm.input) > 2 else ""
            if norm_beta:
                ff_inputs.append(norm_beta)
            ff_node = helper.make_node(
                op_type=specialized_op,
                inputs=ff_inputs,
                outputs=[final_out],
                name=f"{norm.name}_fused_feedforward",
                domain="custom.pytorch",
                activation=act_name.lower(),
                norm_type=norm_op,
                doc_string=f"Fused polymorphic FeedForward ({act_name}) block",
            )
            new_nodes.append(ff_node)
            nodes_to_drop.update({lin1.name, n.name, lin2.name})
            # Only drop norm if it has no other consumers outside this FFN block
            graph_outputs = {o.name for o in graph.output}
            if (
                len(consumer_map.get(norm.output[0], [])) == 1
                and norm.output[0] not in graph_outputs
            ):
                nodes_to_drop.add(norm.name)
            ff_count += 1

            if mode in ("function_proto", "modern_onnx"):
                self._attach_specialized_feedforward_function(
                    model=model,
                    fname=specialized_op,
                    act_name=act_name,
                    norm_op=norm_op,
                    fingerprint=fp,
                    has_beta=bool(norm_beta),
                )

        if not new_nodes:
            return model

        kept = [n for n in graph.node if n.name not in nodes_to_drop]
        kept.extend(new_nodes)
        del graph.node[:]
        graph.node.extend(kept)

        log.info(
            "FeedForward Fusion: Fused %d FFN blocks with polymorphic fingerprints",
            ff_count,
        )
        return model

    def _attach_specialized_feedforward_function(
        self,
        model: onnx.ModelProto,
        fname: str,
        act_name: str,
        norm_op: str,
        fingerprint: str,
        has_beta: bool = False,
    ) -> None:
        """Attach formal, content-hashed ONNX FunctionProto definition for FeedForward."""
        if any(
            f.name == fname and f.domain == "custom.pytorch" for f in model.functions
        ):
            return
        ensure_model_latest_opset(model)
        opset_standard = helper.make_opsetid("", LATEST_OPSET)
        opset_custom = helper.make_opsetid("custom.pytorch", 1)

        norm_domain = "" if norm_op in ("LayerNormalization",) else "custom.pytorch"
        norm_inputs = ["X", "gamma", "beta"] if has_beta else ["X", "gamma"]
        norm = helper.make_node(
            norm_op,
            inputs=norm_inputs,
            outputs=["X_norm"],
            domain=norm_domain,
            epsilon=1e-12,
        )
        fc1 = helper.make_node(
            "Linear",
            inputs=["X_norm", "fc1_w", "fc1_b"],
            outputs=["H1"],
            domain="custom.pytorch",
        )
        act = helper.make_node(
            act_name,
            inputs=["H1"],
            outputs=["H_act"],
            domain="" if act_name in ("Relu", "Tanh") else "custom.pytorch",
        )
        fc2 = helper.make_node(
            "Linear",
            inputs=["H_act", "fc2_w", "fc2_b"],
            outputs=["Y"],
            domain="custom.pytorch",
        )

        formal_inputs = ["X", "gamma", "fc1_w", "fc1_b", "fc2_w", "fc2_b"]
        if has_beta:
            formal_inputs.append("beta")

        ensure_specialized_function_proto(
            model=model,
            domain="custom.pytorch",
            fname=fname,
            formal_inputs=formal_inputs,
            formal_outputs=["Y"],
            body_nodes=[norm, fc1, act, fc2],
            opset_imports=[opset_standard, opset_custom],
            doc_string=f"Content-hashed polymorphic FeedForward ({act_name}) block [{fingerprint}]",
        )

    # -------------------------------------------------------------------------
    # FeedForward SwiGLU Fusion
    # -------------------------------------------------------------------------
    def apply_feedforward_swiglu_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuse SwiGLU FFN block (RMSNorm -> Linear_gate * SiLU * Linear_up -> Linear_down) into FeedForwardSwiGLU."""
        graph = model.graph
        node_by_output = {out: n for n in graph.node for out in n.output}
        consumer_map: dict[str, list[onnx.NodeProto]] = collections.defaultdict(list)
        for n in graph.node:
            for inp in n.input:
                consumer_map[inp].append(n)

        nodes_to_drop: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        count = 0

        for cand_node in graph.node:
            gate_node, up_node, silu_node = None, None, None
            mul_node = None
            act_name = "SiLU"

            if cand_node.op_type.startswith("GatedUnit"):
                cand_act = None
                for attr in cand_node.attribute:
                    if attr.name == "activation":
                        cand_act = attr.s.decode("utf-8") if isinstance(attr.s, bytes) else str(attr.s)
                if cand_act is None:
                    parts = cand_node.op_type.split("_")
                    if len(parts) >= 2:
                        cand_act = parts[1].lower()
                if cand_act:
                    act_name = cand_act
                in0, in1 = cand_node.input[0], cand_node.input[1]
                p0 = node_by_output.get(in0)
                p1 = node_by_output.get(in1)
                if p0 and p1 and p0.op_type == "Linear" and p1.op_type == "Linear":
                    up_node = p0
                    gate_node = p1
                    mul_node = cand_node
            elif cand_node.op_type == "Mul":
                mul_node = cand_node
                in0, in1 = mul_node.input[0], mul_node.input[1]
                valid_acts = ("SiLU", "Gelu", "FastGelu", "Relu")
                act_node = None
                if in0 in node_by_output and node_by_output[in0].op_type in valid_acts:
                    act_node = node_by_output[in0]
                    up_node = node_by_output.get(in1)
                elif in1 in node_by_output and node_by_output[in1].op_type in valid_acts:
                    act_node = node_by_output[in1]
                    up_node = node_by_output.get(in0)

                if act_node and up_node and up_node.op_type == "Linear":
                    p_gate = node_by_output.get(act_node.input[0])
                    if p_gate and p_gate.op_type == "Linear":
                        gate_node = p_gate
                        silu_node = act_node
                        act_name = act_node.op_type

            if not gate_node or not up_node or not mul_node:
                continue

            # gate and up must share the same input
            if gate_node.input[0] != up_node.input[0]:
                continue
            x_common = gate_node.input[0]

            norm_node = node_by_output.get(x_common)
            norm_gamma = ""
            norm_beta = ""
            norm_op = ""
            valid_norms = ("RMSNorm", "RMSNormalization", "LayerNormalization", "SimplifiedLayerNormalization")
            if norm_node and norm_node.op_type in valid_norms:
                norm_op = norm_node.op_type
                x = norm_node.input[0]
                norm_gamma = norm_node.input[1]
                if len(norm_node.input) > 2 and norm_node.input[2]:
                    norm_beta = norm_node.input[2]
            else:
                norm_node = None
                x = x_common

            # Consumer of Mul / GatedUnit must be Linear down
            down_candidates = [
                c
                for c in consumer_map.get(mul_node.output[0], [])
                if c.op_type == "Linear"
            ]
            if not down_candidates:
                continue
            down_node = down_candidates[0]
            w_gate = gate_node.input[1]
            b_gate = gate_node.input[2] if len(gate_node.input) > 2 else ""
            w_up = up_node.input[1]
            b_up = up_node.input[2] if len(up_node.input) > 2 else ""
            w_down = down_node.input[1]
            b_down = down_node.input[2] if len(down_node.input) > 2 else ""
            final_out = down_node.output[0]

            nodes_to_drop.update(
                {
                    gate_node.name or gate_node.output[0],
                    up_node.name or up_node.output[0],
                    mul_node.name or mul_node.output[0],
                    down_node.name or down_node.output[0],
                }
            )
            if silu_node:
                nodes_to_drop.add(silu_node.name or silu_node.output[0])

            graph_outputs = {o.name for o in graph.output}
            if (
                norm_node
                and len(consumer_map.get(norm_node.output[0], [])) <= 2
                and norm_node.output[0] not in graph_outputs
            ):
                nodes_to_drop.add(norm_node.name or norm_node.output[0])

            has_norm = bool(norm_gamma)
            has_bias = bool(b_gate or b_up or b_down)

            if has_bias:
                if has_norm:
                    swiglu_inputs = [x, norm_gamma, w_gate, b_gate, w_up, b_up, w_down, b_down]
                else:
                    swiglu_inputs = [x, w_gate, b_gate, w_up, b_up, w_down, b_down]
            elif has_norm:
                swiglu_inputs = [x, norm_gamma, w_gate, w_up, w_down]
            else:
                swiglu_inputs = [x, w_gate, w_up, w_down]

            if act_name.lower() in ("gelu", "fastgelu"):
                base_op = "FeedForwardGeGLU"
            elif act_name.lower() in ("relu",):
                base_op = "FeedForwardReGLU"
            else:
                base_op = "FeedForwardSwiGLU"

            swiglu_node = helper.make_node(
                base_op,
                inputs=swiglu_inputs,
                outputs=[final_out],
                name=f"{base_op.lower()}_{count}",
                domain="custom.pytorch",
                activation=act_name.lower(),
                norm_type=norm_op,
            )
            new_nodes.append(swiglu_node)
            count += 1

        if not new_nodes:
            return model

        kept = [
            n
            for n in graph.node
            if (n.name or (n.output[0] if n.output else "")) not in nodes_to_drop
        ]
        kept.extend(new_nodes)
        del graph.node[:]
        graph.node.extend(kept)

        if mode == "function_proto" and new_nodes:
            seen_defs = set()
            for nn in new_nodes:
                act = "silu"
                for a in nn.attribute:
                    if a.name == "activation":
                        act = a.s.decode("utf-8") if isinstance(a.s, bytes) else str(a.s)
                norm_op = "RMSNormalization"
                for a in nn.attribute:
                    if a.name == "norm_type":
                        norm_op = a.s.decode("utf-8") if isinstance(a.s, bytes) else str(a.s)
                key = (nn.op_type, len(nn.input), act, norm_op)
                if key not in seen_defs:
                    seen_defs.add(key)
                    self._attach_feedforward_swiglu_function_def(
                        model,
                        func_name=nn.op_type,
                        act_name=act,
                        norm_op=norm_op,
                        input_count=len(nn.input),
                    )

        log.info("FeedForward SwiGLU Fusion: Fused %d SwiGLU blocks", count)
        return model

    def _attach_feedforward_swiglu_function_def(
        self,
        model: onnx.ModelProto,
        func_name: str = "FeedForwardSwiGLU",
        act_name: str = "silu",
        norm_op: str = "RMSNormalization",
        input_count: int = 5,
    ) -> None:
        """Attach formal ONNX FunctionProto definition for FeedForwardSwiGLU / GeGLU / ReGLU."""
        domain = "custom.pytorch"
        if any(
            f.name == func_name and f.domain == domain and len(f.input) == input_count
            for f in model.functions
        ):
            return
        surviving_funcs = [
            f
            for f in model.functions
            if not (f.name == func_name and f.domain == domain and len(f.input) == input_count)
        ]
        del model.functions[:]
        model.functions.extend(surviving_funcs)

        ensure_model_latest_opset(model)
        opset_standard = helper.make_opsetid("", LATEST_OPSET)
        opset_custom = helper.make_opsetid("custom.pytorch", 1)

        def make_act_nodes(in_t: str, out_t: str) -> list[onnx.NodeProto]:
            act_lower = act_name.lower()
            if act_lower in ("gelu", "fastgelu"):
                return [helper.make_node("Gelu", inputs=[in_t], outputs=[out_t], domain="custom.pytorch")]
            elif act_lower in ("relu",):
                return [helper.make_node("Relu", inputs=[in_t], outputs=[out_t])]
            else:
                return [
                    helper.make_node("Sigmoid", inputs=[in_t], outputs=["H_sig"]),
                    helper.make_node("Mul", inputs=[in_t, "H_sig"], outputs=[out_t]),
                ]

        def make_norm_node(x_in: str, gamma_in: str, norm_out: str) -> onnx.NodeProto:
            if norm_op in ("LayerNormalization",):
                return helper.make_node("LayerNormalization", inputs=[x_in, gamma_in], outputs=[norm_out])
            return helper.make_node("RMSNormalization", inputs=[x_in, gamma_in], outputs=[norm_out], domain="custom.pytorch")

        if input_count == 8:
            func_inputs = ["X", "gamma", "w_gate", "b_gate", "w_up", "b_up", "w_down", "b_down"]
            body_nodes = [
                make_norm_node("X", "gamma", "X_norm"),
                helper.make_node("MatMul", inputs=["X_norm", "w_gate"], outputs=["mm_gate"]),
                helper.make_node("Add", inputs=["mm_gate", "b_gate"], outputs=["H_gate_raw"]),
            ] + make_act_nodes("H_gate_raw", "H_act") + [
                helper.make_node("MatMul", inputs=["X_norm", "w_up"], outputs=["mm_up"]),
                helper.make_node("Add", inputs=["mm_up", "b_up"], outputs=["H_up"]),
                helper.make_node("Mul", inputs=["H_act", "H_up"], outputs=["H_mul"]),
                helper.make_node("MatMul", inputs=["H_mul", "w_down"], outputs=["mm_down"]),
                helper.make_node("Add", inputs=["mm_down", "b_down"], outputs=["Y"]),
            ]
        elif input_count == 4:
            func_inputs = ["X", "w_gate", "w_up", "w_down"]
            body_nodes = [
                helper.make_node("MatMul", inputs=["X", "w_gate"], outputs=["H_gate_raw"]),
            ] + make_act_nodes("H_gate_raw", "H_act") + [
                helper.make_node("MatMul", inputs=["X", "w_up"], outputs=["H_up"]),
                helper.make_node("Mul", inputs=["H_act", "H_up"], outputs=["H_mul"]),
                helper.make_node("MatMul", inputs=["H_mul", "w_down"], outputs=["Y"]),
            ]
        elif input_count == 7:
            func_inputs = ["X", "w_gate", "b_gate", "w_up", "b_up", "w_down", "b_down"]
            body_nodes = [
                helper.make_node("MatMul", inputs=["X", "w_gate"], outputs=["mm_gate"]),
                helper.make_node("Add", inputs=["mm_gate", "b_gate"], outputs=["H_gate_raw"]),
            ] + make_act_nodes("H_gate_raw", "H_act") + [
                helper.make_node("MatMul", inputs=["X", "w_up"], outputs=["mm_up"]),
                helper.make_node("Add", inputs=["mm_up", "b_up"], outputs=["H_up"]),
                helper.make_node("Mul", inputs=["H_act", "H_up"], outputs=["H_mul"]),
                helper.make_node("MatMul", inputs=["H_mul", "w_down"], outputs=["mm_down"]),
                helper.make_node("Add", inputs=["mm_down", "b_down"], outputs=["Y"]),
            ]
        else:
            func_inputs = ["X", "gamma", "w_gate", "w_up", "w_down"]
            body_nodes = [
                make_norm_node("X", "gamma", "X_norm"),
                helper.make_node("MatMul", inputs=["X_norm", "w_gate"], outputs=["H_gate_raw"]),
            ] + make_act_nodes("H_gate_raw", "H_act") + [
                helper.make_node("MatMul", inputs=["X_norm", "w_up"], outputs=["H_up"]),
                helper.make_node("Mul", inputs=["H_act", "H_up"], outputs=["H_mul"]),
                helper.make_node("MatMul", inputs=["H_mul", "w_down"], outputs=["Y"]),
            ]

        func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=func_inputs,
            outputs=["Y"],
            nodes=body_nodes,
            opset_imports=[opset_standard, opset_custom],
            doc_string=f"Formal {func_name} ({act_name}) block",
        )
        model.functions.append(func)
        ensure_model_latest_opset(model)
