"""neuron_toolkit.

Unified graph analysis, querying, pattern-matching, optimization, and rewriting library.
Supports ONNX and TFLite backends.
"""

from ._types import TensorMap
from .egraph import EClass, EGraph, ENode, Extractor, saturate
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
    FuseAttention,
    FuseConvActivation,
    FuseConvBN,
    FuseGELU,
    FuseLayerNorm,
    FuseRMSNorm,
    FuseRotaryEmbedding,
    FuseSwiGLU,
    PassManager,
    PassStats,
)
from .pattern import MatchResult, Pattern, PatternDetector, RewriteRule
from .profiler import ProfilerReport, profile_graph
from .query import NeuronQuery
from .rewriter import NeuronRewriter
from .rules import (
    JSONRule,
    JSONRuleEngine,
    NodePattern,
    RuleAction,
    RuleCondition,
    export_simplified_onnx,
)
from .runtime import (
    CrossVerificationReport,
    CrossVerifier,
    OpRegistry,
    OutputMetric,
    PyTorchRunner,
    global_registry,
    toposort_graph,
)
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
    "EClass",
    "EGraph",
    "ENode",
    "Extractor",
    "FuseAttention",
    "FuseConvActivation",
    "FuseConvBN",
    "FuseGELU",
    "FuseLayerNorm",
    "FuseRMSNorm",
    "FuseRotaryEmbedding",
    "FuseSwiGLU",
    "GraphParseError",
    "GraphVerifier",
    "IRGraph",
    "IRNode",
    "IRTensor",
    "JSONRule",
    "JSONRuleEngine",
    "MatchResult",
    "NeuronGraph",
    "NeuronQuery",
    "NeuronRewriter",
    "NeuronToolkitError",
    "NodePattern",
    "OpRegistry",
    "OutputMetric",
    "PassManager",
    "PassStats",
    "Pattern",
    "PatternDetector",
    "PatternMatchError",
    "ProfilerReport",
    "PyTorchRunner",
    "RewriteError",
    "RewriteRule",
    "RuleAction",
    "RuleCondition",
    "TensorMap",
    "VerificationMismatchError",
    "build_onnx_model_from_json",
    "build_tflite_model_from_json",
    "compute_weight_stats",
    "export_graph_dict",
    "export_graph_json",
    "export_safetensors",
    "export_simplified_onnx",
    "from_json",
    "global_registry",
    "load_graph_from_json",
    "load_safetensors",
    "profile_graph",
    "saturate",
    "save_safetensors",
    "to_html",
    "to_json",
    "to_mermaid",
    "toposort_graph",
    "CrossVerificationReport",
    "CrossVerifier",
]
