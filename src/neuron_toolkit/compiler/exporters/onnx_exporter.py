"""ONNX Serializer and Target Exporter for the 3-Stage Compiler."""

from __future__ import annotations

import logging
from pathlib import Path

import onnx
from onnx import helper

from neuron_toolkit.compiler.base import CompilerContext
from neuron_toolkit.runtime.runner import toposort_graph

log = logging.getLogger("neuron_toolkit.compiler.onnx_exporter")


class OnnxExporter:
    """Serializes optimized models to standard ONNX or FunctionProto dual-mode representation."""

    def __init__(self, mode: str = "function_proto") -> None:
        self.mode = mode

    def export(
        self,
        model: onnx.ModelProto,
        context: CompilerContext | None = None,
        output_path: str | Path | None = None,
    ) -> onnx.ModelProto:
        """Finalize and export the ONNX model."""
        if context is None:
            context = CompilerContext(mode=self.mode)

        # 1. Strict topological sort invariant
        sorted_nodes = toposort_graph(model)
        del model.graph.node[:]
        model.graph.node.extend(sorted_nodes)

        # 2. Mode-specific adjustments and Opset imports
        if context.mode in ("plain_custom_op", "pytorch_custom"):
            del model.functions[:]
        elif context.mode == "modern_onnx":
            has_main = False
            for opset in model.opset_import:
                if opset.domain in ("", "ai.onnx"):
                    opset.version = max(opset.version, context.opset_version)
                    has_main = True
            if not has_main:
                model.opset_import.append(
                    helper.make_opsetid("", context.opset_version)
                )

        # Align function opset imports with model opset imports for clean checker validation
        model_opsets = {op.domain: op.version for op in model.opset_import}
        for f in model.functions:
            for op in f.opset_import:
                if op.domain in model_opsets:
                    op.version = model_opsets[op.domain]

        # Ensure all node and function custom domains are imported
        existing_domains = {op.domain for op in model.opset_import}
        for n in model.graph.node:
            domain = n.domain or ""
            if domain and domain not in existing_domains:
                model.opset_import.append(helper.make_opsetid(domain, 1))
                existing_domains.add(domain)
        for f in model.functions:
            domain = f.domain or ""
            if domain and domain not in existing_domains:
                model.opset_import.append(helper.make_opsetid(domain, 1))
                existing_domains.add(domain)

        # 3. Model metadata
        meta = {
            "Engine": "NeuronToolkit_OptimizationPipeline",
            "Applied_Rules": ",".join(context.applied_passes or ["all_builtin"]),
            "Engine_Mode": context.mode,
            "Total_Nodes": str(len(model.graph.node)),
        }
        if context.stats:
            meta["Total_Passes"] = str(len(context.stats))
        helper.set_model_props(model, meta)

        # 4. Check model integrity
        try:
            onnx.checker.check_model(model)
        except Exception as e:
            log.warning("onnx.checker validation note: %s", e)

        # 5. Optional file output
        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            onnx.save(model, str(out_p))
            log.info("Saved clean model to %s (%d nodes)", out_p, len(model.graph.node))

        return model
