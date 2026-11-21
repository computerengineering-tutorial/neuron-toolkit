"""Dual-Mode ONNX Model Exporter for Netron and Custom Runtime Execution."""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any

import onnx
from onnx import helper

if TYPE_CHECKING:
    from neuron_toolkit.graph import NeuronGraph

log = logging.getLogger("neuron_toolkit.rules.exporter")


def export_simplified_onnx(
    model_or_graph: onnx.ModelProto | NeuronGraph,
    output_path: str,
    metadata: dict[str, Any] | None = None,
) -> onnx.ModelProto:
    """Export the simplified model with FunctionProto Netron expandability.

    Args:
        model_or_graph: ONNX ModelProto or NeuronGraph instance.
        output_path: Path to save the `.onnx` file.
        metadata: Optional dictionary of key-value pairs to store in model metadata.

    Returns:
        The validated onnx.ModelProto.
    """
    if hasattr(model_or_graph, "_backend") and hasattr(
        model_or_graph._backend, "model"
    ):
        model = copy.deepcopy(model_or_graph._backend.model)
    elif isinstance(model_or_graph, onnx.ModelProto):
        model = copy.deepcopy(model_or_graph)
    else:
        msg = (
            f"Unsupported input type for export_simplified_onnx: {type(model_or_graph)}"
        )
        raise TypeError(msg)

    # Ensure custom opset domain exists in model opset_imports if used
    custom_domain = "custom.pytorch"
    if not any(op.domain == custom_domain for op in model.opset_import):
        model.opset_import.append(helper.make_opsetid(custom_domain, 1))

    from neuron_toolkit.compiler.base import ensure_model_latest_opset
    ensure_model_latest_opset(model)

    # Attach metadata
    meta = {
        "Toolkit": "neuron-toolkit",
        "Export_Mode": "function_proto",
        "Total_Nodes": str(len(model.graph.node)),
    }
    if metadata:
        meta.update({k: str(v) for k, v in metadata.items()})
    helper.set_model_props(model, meta)

    # Validate model
    onnx.checker.check_model(model)

    # Save to disk
    onnx.save(model, output_path)
    log.info(
        "Successfully exported function_proto model with %d nodes to %s",
        len(model.graph.node),
        output_path,
    )
    return model
