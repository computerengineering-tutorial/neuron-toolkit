"""FeedForward / MLP Block Synthesizer Pass.

Synthesizes macro-blocks:
- FeedForward GELU (RMSNorm -> Linear -> Gelu -> Linear) -> FeedForward
- FeedForward SwiGLU (RMSNorm -> Linear_gate * SiLU * Linear_up -> Linear_down) -> FeedForwardSwiGLU
Attaches formal FunctionProto definitions for Netron expandable hierarchical visualization.
"""

from __future__ import annotations

import collections
import logging
from typing import Any

import onnx
from onnx import helper

from neuron_toolkit.compiler.architectures.fingerprint import (
    build_monomorphized_op_type,
    compute_subgraph_fingerprint,
    ensure_specialized_function_proto,
)
from neuron_toolkit.compiler.base import CompilerContext, Pass

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
    def apply_feedforward_fusion(self, model: onnx.ModelProto, mode: str = "function_proto") -> onnx.ModelProto:
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
            if not norm or norm.op_type not in ("RMSNorm", "RMSNormalization"):
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
            ff_node = helper.make_node(
                op_type=specialized_op,
                inputs=ff_inputs,
                outputs=[final_out],
                name=f"{norm.name}_fused_feedforward",
                domain="custom.pytorch",
                activation=act_name.lower(),
                doc_string=f"Fused polymorphic FeedForward ({act_name}) block",
            )
            new_nodes.append(ff_node)
            nodes_to_drop.update({lin1.name, n.name, lin2.name})
            # Only drop norm if it has no other consumers outside this FFN block
            graph_outputs = {o.name for o in graph.output}
            if len(consumer_map.get(norm.output[0], [])) == 1 and norm.output[0] not in graph_outputs:
                nodes_to_drop.add(norm.name)
            ff_count += 1

            if mode in ("function_proto", "modern_onnx"):
                self._attach_specialized_feedforward_function(
                    model=model,
                    fname=specialized_op,
                    act_name=act_name,
                    norm_op=norm_op,
                    fingerprint=fp,
                )

        if not new_nodes:
            return model

        kept = [n for n in graph.node if n.name not in nodes_to_drop]
        kept.extend(new_nodes)
        del graph.node[:]
        graph.node.extend(kept)

        log.info("FeedForward Fusion: Fused %d FFN blocks with polymorphic fingerprints", ff_count)
        return model

    def _attach_specialized_feedforward_function(
        self,
        model: onnx.ModelProto,
        fname: str,
        act_name: str,
        norm_op: str,
        fingerprint: str,
    ) -> None:
        """Attach formal, content-hashed ONNX FunctionProto definition for FeedForward."""
        if any(f.name == fname and f.domain == "custom.pytorch" for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", 14)
        opset_custom = helper.make_opsetid("custom.pytorch", 1)

        norm = helper.make_node(norm_op, inputs=["X", "gamma"], outputs=["X_norm"], domain="custom.pytorch", epsilon=1e-12)
        fc1 = helper.make_node("Linear", inputs=["X_norm", "fc1_w", "fc1_b"], outputs=["H1"], domain="custom.pytorch")
        act = helper.make_node(act_name, inputs=["H1"], outputs=["H_act"], domain="" if act_name in ("Relu", "Tanh") else "custom.pytorch")
        fc2 = helper.make_node("Linear", inputs=["H_act", "fc2_w", "fc2_b"], outputs=["Y"], domain="custom.pytorch")

        ensure_specialized_function_proto(
            model=model,
            domain="custom.pytorch",
            fname=fname,
            formal_inputs=["X", "gamma", "fc1_w", "fc1_b", "fc2_w", "fc2_b"],
            formal_outputs=["Y"],
            body_nodes=[norm, fc1, act, fc2],
            opset_imports=[opset_standard, opset_custom],
            doc_string=f"Content-hashed polymorphic FeedForward ({act_name}) block [{fingerprint}]",
        )

    # -------------------------------------------------------------------------
    # FeedForward SwiGLU Fusion
    # -------------------------------------------------------------------------
    def apply_feedforward_swiglu_fusion(self, model: onnx.ModelProto, mode: str = "function_proto") -> onnx.ModelProto:
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

        for mul_node in graph.node:
            if mul_node.op_type != "Mul":
                continue
            # Inputs to Mul: silu(gate) and up
            in0, in1 = mul_node.input[0], mul_node.input[1]
            silu_node, up_node = None, None
            if in0 in node_by_output and node_by_output[in0].op_type == "SiLU":
                silu_node = node_by_output[in0]
                up_node = node_by_output.get(in1)
            elif in1 in node_by_output and node_by_output[in1].op_type == "SiLU":
                silu_node = node_by_output[in1]
                up_node = node_by_output.get(in0)

            if not silu_node or not up_node:
                continue
            if up_node.op_type != "Linear":
                continue

            gate_node = node_by_output.get(silu_node.input[0])
            if not gate_node or gate_node.op_type != "Linear":
                continue

            # gate and up must share the same norm input
            norm_node = node_by_output.get(gate_node.input[0])
            if not norm_node or norm_node.op_type not in ("RMSNorm", "RMSNormalization"):
                continue
            if up_node.input[0] != norm_node.output[0]:
                continue

            # Consumer of Mul must be Linear down
            down_candidates = [c for c in consumer_map.get(mul_node.output[0], []) if c.op_type == "Linear"]
            if not down_candidates:
                continue
            down_node = down_candidates[0]

            x = norm_node.input[0]
            norm_gamma = norm_node.input[1]
            w_gate = gate_node.input[1]
            b_gate = gate_node.input[2] if len(gate_node.input) > 2 else ""
            w_up = up_node.input[1]
            b_up = up_node.input[2] if len(up_node.input) > 2 else ""
            w_down = down_node.input[1]
            b_down = down_node.input[2] if len(down_node.input) > 2 else ""
            final_out = down_node.output[0]

            nodes_to_drop.update({
                gate_node.name or gate_node.output[0],
                silu_node.name or silu_node.output[0],
                up_node.name or up_node.output[0],
                mul_node.name or mul_node.output[0],
                down_node.name or down_node.output[0],
            })
            graph_outputs = {o.name for o in graph.output}
            if len(consumer_map.get(norm_node.output[0], [])) <= 2 and norm_node.output[0] not in graph_outputs:
                nodes_to_drop.add(norm_node.name or norm_node.output[0])

            if b_gate or b_up or b_down:
                swiglu_inputs = [x, norm_gamma, w_gate, b_gate, w_up, b_up, w_down, b_down]
            else:
                swiglu_inputs = [x, norm_gamma, w_gate, w_up, w_down]

            swiglu_node = helper.make_node(
                "FeedForwardSwiGLU",
                inputs=swiglu_inputs,
                outputs=[final_out],
                name=f"feedforward_swiglu_{count}",
                domain="custom.pytorch",
            )
            new_nodes.append(swiglu_node)
            count += 1

        if not new_nodes:
            return model

        kept = [n for n in graph.node if (n.name or (n.output[0] if n.output else "")) not in nodes_to_drop]
        kept.extend(new_nodes)
        del graph.node[:]
        graph.node.extend(kept)

        if mode == "function_proto":
            self._attach_feedforward_swiglu_function_def(model)

        log.info("FeedForward SwiGLU Fusion: Fused %d SwiGLU blocks", count)
        return model

    def _attach_feedforward_swiglu_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for FeedForwardSwiGLU."""
        func_name = "FeedForwardSwiGLU"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", 14)
        opset_custom = helper.make_opsetid("custom.pytorch", 1)

        norm = helper.make_node("RMSNorm", inputs=["X", "gamma"], outputs=["X_norm"], domain="custom.pytorch", epsilon=1e-12)
        gate = helper.make_node("Linear", inputs=["X_norm", "w_gate"], outputs=["H_gate"], domain="custom.pytorch")
        up = helper.make_node("Linear", inputs=["X_norm", "w_up"], outputs=["H_up"], domain="custom.pytorch")
        silu = helper.make_node("SiLU", inputs=["H_gate"], outputs=["H_act"], domain="custom.pytorch")
        mul = helper.make_node("Mul", inputs=["H_act", "H_up"], outputs=["H_mul"])
        down = helper.make_node("Linear", inputs=["H_mul", "w_down"], outputs=["Y"], domain="custom.pytorch")

        func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["X", "gamma", "w_gate", "w_up", "w_down"],
            outputs=["Y"],
            nodes=[norm, gate, up, silu, mul, down],
            opset_imports=[opset_standard, opset_custom],
            doc_string="FeedForward SwiGLU Transformer block",
        )
        model.functions.append(func)
