"""PyTorch Execution Target Exporter for the 3-Stage Compiler."""

from __future__ import annotations

import logging

import onnx

from neuron_toolkit.compiler.base import CompilerContext
from neuron_toolkit.compiler.exporters.onnx_exporter import OnnxExporter
from neuron_toolkit.runtime.registry import OpRegistry
from neuron_toolkit.runtime.runner import PyTorchRunner

log = logging.getLogger("neuron_toolkit.compiler.pytorch_exporter")


class PyTorchExporter:
    """Exports an optimized ONNX model into an executable PyTorch graph runner."""

    def __init__(self, registry: OpRegistry | None = None) -> None:
        self.registry = registry or OpRegistry()

    def export(
        self,
        model: onnx.ModelProto,
        context: CompilerContext | None = None,
    ) -> PyTorchRunner:
        """Finalize model for PyTorch runtime and return a configured PyTorchRunner."""
        if context is None:
            context = CompilerContext(mode="pytorch_custom")

        # Prepare model in plain custom op mode for runtime runner
        onnx_exporter = OnnxExporter(mode="pytorch_custom")
        clean_model = onnx_exporter.export(model, context)

        runner = PyTorchRunner(clean_model, registry=self.registry)
        log.info(
            "Initialized PyTorchRunner for model with %d nodes",
            len(clean_model.graph.node),
        )
        return runner
