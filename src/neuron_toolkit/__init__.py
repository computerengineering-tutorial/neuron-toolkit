"""neuron_toolkit.

Unified graph analysis, querying, pattern-matching, optimization, and rewriting library.
Supports ONNX and TFLite backends.
"""

from ._types import TensorMap
from .eval import GraphVerifier
from .exceptions import (
    BackendNotAvailableError,
    GraphParseError,
    NeuronToolkitError,
    PatternMatchError,
    RewriteError,
    VerificationMismatchError,
)
from .exporter import (
    build_onnx_model_from_json,
    build_tflite_model_from_json,
    compute_weight_stats,
    export_graph_dict,
    export_graph_json,
    export_safetensors,
    load_graph_from_json,
    load_safetensors,
)
from .graph import NeuronGraph
from .ir import IRGraph, IRNode, IRTensor
from .passes import (
    BasePass,
    CommonSubexpressionElimination,
    ConstantFolding,
    DeadCodeElimination,
    FuseConvActivation,
    FuseConvBN,
    FuseGELU,
    FuseLayerNorm,
    PassManager,
    PassStats,
)
from .pattern import MatchResult, Pattern, PatternDetector, RewriteRule
from .query import NeuronQuery
from .rewriter import NeuronRewriter
from .viz import to_html, to_mermaid

save_safetensors = export_safetensors
to_json = export_graph_json
from_json = load_graph_from_json

__all__ = [
    "BackendNotAvailableError",
    "BasePass",
    "CommonSubexpressionElimination",
    "ConstantFolding",
    "DeadCodeElimination",
    "FuseConvActivation",
    "FuseConvBN",
    "FuseGELU",
    "FuseLayerNorm",
    "GraphParseError",
    "GraphVerifier",
    "IRGraph",
    "IRNode",
    "IRTensor",
    "MatchResult",
    "NeuronGraph",
    "NeuronQuery",
    "NeuronRewriter",
    "NeuronToolkitError",
    "PassManager",
    "PassStats",
    "Pattern",
    "PatternDetector",
    "PatternMatchError",
    "RewriteError",
    "RewriteRule",
    "TensorMap",
    "VerificationMismatchError",
    "build_onnx_model_from_json",
    "build_tflite_model_from_json",
    "compute_weight_stats",
    "export_graph_dict",
    "export_graph_json",
    "export_safetensors",
    "from_json",
    "load_graph_from_json",
    "load_safetensors",
    "save_safetensors",
    "to_html",
    "to_json",
    "to_mermaid",
]
