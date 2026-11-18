"""neuron_toolkit.graph.

Unified entry point for graph analysis, querying, and transformation.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from neuron_toolkit._utils import _GraphShim

if TYPE_CHECKING:
    from neuron_toolkit._utils import ShapeInfo
    from neuron_toolkit.pattern import MatchResult, Pattern
    from neuron_toolkit.query import NeuronQuery
    from neuron_toolkit.rewriter import NeuronRewriter

log = logging.getLogger("neuron_toolkit.graph")


class NeuronGraph:
    """Unified interface for a model graph (ONNX or TFLite)."""

    def __init__(self, source: object, **kwargs: Any) -> None:
        self._backend: Any
        if isinstance(source, str):
            # Auto-load if it's a path or JSON string
            if source.strip().startswith("{"):
                from neuron_toolkit.exporter import load_graph_from_json

                temp_g = load_graph_from_json(source, **kwargs)
                self._backend = temp_g._backend
            else:
                temp_g = self.load(source, **kwargs)
                self._backend = temp_g._backend
        elif isinstance(source, dict) and (
            source.get("format") == "neuron_toolkit_graph_v1" or "nodes" in source
        ):
            from neuron_toolkit.exporter import load_graph_from_json

            temp_g = load_graph_from_json(source, **kwargs)
            self._backend = temp_g._backend
        elif isinstance(source, bytes):
            # Check flatbuffer magic header for TFLite at offset 4
            if len(source) >= 8 and source[4:8] == b"TFL3":
                from neuron_toolkit.backends.tflite.parser import (
                    TFLiteParser,
                )

                self._backend = TFLiteParser(source, **kwargs)
            else:
                from neuron_toolkit.backends.onnx.parser import (
                    ONNXParser,
                )

                self._backend = ONNXParser(cast(Any, source), **kwargs)
        elif hasattr(source, "graph") and hasattr(
            source.graph, "node"
        ):  # ONNX ModelProto
            from neuron_toolkit.backends.onnx.parser import ONNXParser

            self._backend = ONNXParser(cast(Any, source), **kwargs)
        elif hasattr(source, "Subgraphs") and hasattr(
            source, "OperatorCodes"
        ):  # TFLite Model
            from neuron_toolkit.backends.tflite.parser import (
                TFLiteParser,
            )

            self._backend = TFLiteParser(source, **kwargs)
        else:
            # Assume it's already a backend parser
            self._backend = source

    @classmethod
    def load(cls, path: str, **kwargs: Any) -> NeuronGraph:
        """Load a model from path and return a NeuronGraph."""
        from pathlib import Path

        ext = Path(path).suffix.lower()
        if ext == ".onnx":
            from neuron_toolkit.backends.onnx.parser import ONNXParser

            return cls(ONNXParser(path, **kwargs))
        if ext in (".tflite", ".bin"):
            from neuron_toolkit.backends.tflite.parser import (
                TFLiteParser,
            )

            return cls(TFLiteParser(path, **kwargs))
        if ext == ".json":
            from neuron_toolkit.exporter import load_graph_from_json

            return load_graph_from_json(path, **kwargs)

        msg = f"Unsupported model format: {ext}"
        raise ValueError(msg)

    @classmethod
    def from_json(
        cls,
        json_source: str | dict[str, Any],
        weights: str | dict[str, Any] | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> NeuronGraph:
        """Rebuild a NeuronGraph from a JSON graph file/string/dict and optional safetensors weights."""
        from neuron_toolkit.exporter import load_graph_from_json

        return load_graph_from_json(
            json_source, weights_source=weights, seed=seed, **kwargs
        )

    @property
    def nodes(self) -> list[object]:
        """Return the list of nodes in the graph."""
        return getattr(self._backend, "nodes", [])

    @property
    def tensor_map(self) -> dict[str, object]:
        """Return the dictionary of initializers/parameter tensors."""
        return getattr(self._backend, "tensor_map", {})

    @property
    def shape_info(self) -> ShapeInfo:
        """Return shape and dtype information for each tensor."""
        return getattr(self._backend, "shape_info", {})

    @property
    def inputs(self) -> list[str]:
        """Return the list of graph input tensor names."""
        if hasattr(self._backend, "inputs"):
            return list(self._backend.inputs)
        if hasattr(self._backend, "graph_inputs"):
            return list(self._backend.graph_inputs)
        return []

    @property
    def outputs(self) -> list[str]:
        """Return the list of graph output tensor names."""
        if hasattr(self._backend, "outputs"):
            return list(self._backend.outputs)
        if hasattr(self._backend, "graph_outputs"):
            return list(self._backend.graph_outputs)
        return []

    def query(self) -> NeuronQuery:
        """Alias for find() — return a query over all nodes."""
        return self._backend.find()

    def match(self, pattern: Pattern) -> MatchResult | None:
        """Find the first occurrence of *pattern* in the graph."""
        results = self.findall(pattern)
        return results[0] if results else None

    def findall(self, pattern: Pattern) -> list[MatchResult]:
        """Find all occurrences of *pattern* in the graph."""
        from neuron_toolkit.pattern import PatternDetector

        det = PatternDetector(self._shim())
        return det.find_all(pattern)

    def replace(
        self,
        pattern: Pattern,
        new_op: str,
        name: str | None = None,
        **attrs: object,
    ) -> NeuronRewriter:
        """Replace all matches of *pattern* with a new operator.

        This is a fluent shortcut for finding all matches and applying a rewrite.
        """
        rewriter = self.rewriter()
        matches = self.findall(pattern)
        for m in matches:
            rewriter.replace_from_result(m, new_op, name=name, **attrs)
        return rewriter

    def to_ir(self) -> Any:
        """Convert this graph to a framework-agnostic IRGraph."""
        from neuron_toolkit.ir import (
            IRGraph,
            IRNode,
            IRTensor,
            ONNXAdapter,
            TFLiteAdapter,
        )

        if hasattr(self._backend, "model"):
            return ONNXAdapter().import_to_ir(self._backend.model)
        if hasattr(self._backend, "tflite_bytes"):
            return TFLiteAdapter().import_to_ir(self._backend.tflite_bytes)

        ir = IRGraph(name="graph")
        ir.inputs = list(self.inputs)
        ir.outputs = list(self.outputs)
        for name, data in self.tensor_map.items():
            shape = tuple(data.shape) if hasattr(data, "shape") else ()
            dtype = str(data.dtype) if hasattr(data, "dtype") else "float32"
            ir.add_tensor(
                IRTensor(
                    name=name,
                    shape=shape,
                    dtype=dtype,
                    data=data,
                    is_initializer=True,
                )
            )
        for node in self.nodes:
            attrs = (
                self._backend.get_node_attrs(node)
                if hasattr(self._backend, "get_node_attrs")
                else {}
            )
            ir.add_node(
                IRNode(
                    op_type=getattr(node, "op_type", ""),
                    name=getattr(node, "name", ""),
                    inputs=list(getattr(node, "input", [])),
                    outputs=list(getattr(node, "output", [])),
                    attributes=attrs,
                )
            )
        return ir

    def optimize(
        self, passes: list[Any] | Any | None = None
    ) -> tuple[NeuronGraph, Any]:
        """Apply a sequence of optimization passes or PassManager to the graph."""
        from neuron_toolkit.passes import (
            BasePass,
            DeadCodeElimination,
            FuseConvBN,
            PassManager,
        )

        if passes is None:
            pm = PassManager([FuseConvBN(), DeadCodeElimination()])
        elif isinstance(passes, PassManager):
            pm = passes
        elif isinstance(passes, list):
            pm = PassManager(passes)
        elif isinstance(passes, BasePass):
            pm = PassManager([passes])
        else:
            msg = f"Expected PassManager or list of BasePass, got {type(passes)}"
            raise TypeError(msg)

        return pm.run(self)

    def rewriter(self) -> NeuronRewriter:
        """Return a GraphRewriter bound to this model."""
        from neuron_toolkit.rewriter import NeuronRewriter

        return NeuronRewriter(self._backend.rewriter())

    def _shim(self) -> _GraphShim:
        return _GraphShim(
            self.nodes, self.tensor_map, self.shape_info, backend=self._backend
        )

    def apply_rules(self, rules: Any, max_iterations: int = 10) -> NeuronGraph:
        """Apply declarative RewriteRules iteratively until fixed point or max_iterations."""
        from neuron_toolkit.pattern.rules import RewriteRule

        rule_list = [rules] if isinstance(rules, RewriteRule) else list(rules)
        current_graph = self

        for _ in range(max_iterations):
            rewriter = current_graph.rewriter()
            matched_any = False

            for rule in rule_list:
                matches = current_graph.findall(rule.pattern)
                if matches:
                    matched_any = True
                    for m in matches:
                        rule.apply(m, rewriter)

            if not matched_any:
                break

            current_graph = rewriter.build_graph()

        return current_graph

    def equality_saturation(
        self,
        rules: Any = None,
        cost_model: Any = None,
        max_iterations: int = 10,
        node_limit: int = 5000,
    ) -> tuple[Any, dict[str, Any]]:
        """Apply equality saturation using E-Graphs and extract the optimal IRGraph.

        Returns:
            A tuple of (optimal_ir_graph, saturation_stats).
        """
        from neuron_toolkit.egraph import EGraph, Extractor, saturate

        ir = self.to_ir()
        egraph, tensor_map = EGraph.from_ir(ir)
        stats = saturate(
            egraph, rules=rules, max_iterations=max_iterations, node_limit=node_limit
        )

        root_ids = [tensor_map[out] for out in self.outputs if out in tensor_map]
        extractor = Extractor(egraph, cost_model=cost_model)
        graph_name = getattr(self, "name", getattr(ir, "name", "model"))
        optimal_ir = extractor.extract_to_ir(
            root_ids, graph_name=f"{graph_name}_optimized"
        )
        return optimal_ir, stats

    def slice(
        self,
        start_points: list[str | object] | str | object,
        end_points: list[str | object] | str | object,
        output_path: str | None = None,
    ) -> NeuronGraph:
        """Slice the model between start and end points and return a new NeuronGraph."""
        sliced_backend = self._backend.slice(start_points, end_points, output_path)
        return NeuronGraph(sliced_backend)

    def slice_node(
        self, node: str | object, output_path: str | None = None
    ) -> NeuronGraph:
        """Extract a single node as a new model and return a new NeuronGraph."""
        return self.slice(node, node, output_path)

    def save(self, path: str) -> None:
        """Save the current model to the specified path."""
        if hasattr(self._backend, "model"):  # ONNX
            import onnx

            onnx.save(self._backend.model, path)
        elif hasattr(self._backend, "source"):  # TFLite
            source = self._backend.source
            if isinstance(source, bytes):
                from pathlib import Path

                Path(path).write_bytes(source)
            elif isinstance(source, str):
                import shutil

                shutil.copy(source, path)
            else:
                msg = "Cannot save this model type directly. Source is a parsed object."
                raise ValueError(msg)
        else:
            msg = "Unsupported backend type for saving."
            raise ValueError(msg)

    def to_mermaid(self, max_nodes: int = 150) -> str:
        """Generate a Mermaid flowchart diagram representation."""
        from neuron_toolkit.viz import to_mermaid

        return to_mermaid(self, max_nodes=max_nodes)

    def to_html(self, output_path: str | None = None) -> str:
        """Export this graph as a standalone interactive HTML visualizer page."""
        from neuron_toolkit.viz import to_html

        return to_html(self, output_path=output_path)

    def get_weights(self) -> dict[str, Any]:
        """Return a dictionary of all parameter/initializer weight tensors in the graph."""
        return dict(self.tensor_map)

    def to_dict(self) -> dict[str, Any]:
        """Export the model graph structure as a Python dictionary without raw weights."""
        cached = getattr(self, "_cached_dict", None)
        if cached is not None:
            return cached.copy()
        from neuron_toolkit.exporter import export_graph_dict

        res = export_graph_dict(self)
        self._cached_dict = res
        return res.copy()

    def to_json(self, path: str | None = None, indent: int = 2) -> str | None:
        """Export the model graph structure as a JSON string or save to a file path."""
        from neuron_toolkit.exporter import export_graph_json

        return export_graph_json(self, path=path, indent=indent)

    def save_safetensors(
        self, path: str, metadata: dict[str, str] | None = None
    ) -> None:
        """Save model weight and parameter tensors into a .safetensors file."""
        from neuron_toolkit.exporter import export_safetensors

        export_safetensors(self, path, metadata=metadata)

    def profile(
        self,
        device: str | Any = "a100",
        dtype: str = "float16",
    ) -> Any:
        """Profile FLOPs, MACs, active working set memory, and Roofline latency bounds."""
        from neuron_toolkit.profiler import profile_graph

        return profile_graph(self, device=device, dtype=dtype)

    def load_safetensors(
        self,
        path: str,
        strict: bool = False,
    ) -> dict[str, list[str]]:
        """Load weights from a .safetensors file and replace model initializers in-place (like PyTorch load_state_dict)."""
        if hasattr(self._backend, "load_safetensors"):
            return self._backend.load_safetensors(path, strict=strict)
        from neuron_toolkit.exporter import load_safetensors

        weights = load_safetensors(path)
        return self.replace_weights(weights, strict=strict)

    def replace_weights(
        self,
        weights_dict: dict[str, Any] | Mapping[str, Any],
        strict: bool = False,
    ) -> dict[str, list[str]]:
        """Replace weights/initializers in-place from a dictionary (like PyTorch load_state_dict)."""
        self._cached_dict = None
        if hasattr(self._backend, "replace_weights"):
            return self._backend.replace_weights(weights_dict, strict=strict)
        if hasattr(self._backend, "tensor_map"):
            model_keys = set(self._backend.tensor_map.keys())
            provided_keys = set(weights_dict.keys())
            missing = sorted(model_keys - provided_keys)
            unexpected = sorted(provided_keys - model_keys)
            if strict:
                if missing:
                    msg = f"Missing key(s) in weights: {missing}"
                    raise ValueError(msg)
                if unexpected:
                    msg = f"Unexpected key(s) in weights: {unexpected}"
                    raise ValueError(msg)
            if isinstance(self._backend.tensor_map, dict):
                for k, v in weights_dict.items():
                    if k in model_keys:
                        self._backend.tensor_map[k] = v
            return {"missing_keys": missing, "unexpected_keys": unexpected}
        return {"missing_keys": [], "unexpected_keys": []}

    def compare_outputs(
        self,
        start_points: list[str | object] | str | object,
        end_points: list[str | object] | str | object,
        pytorch_module: Any,
        inputs_dict: dict[str, Any],
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> dict[str, Any]:
        """Execute the sliced model and PyTorch module on inputs and compare outputs."""
        from neuron_toolkit.eval import GraphVerifier

        return GraphVerifier(self).compare_outputs(
            start_points=start_points,
            end_points=end_points,
            pytorch_module=pytorch_module,
            inputs_dict=inputs_dict,
            rtol=rtol,
            atol=atol,
        )

    def verify_against_pytorch(
        self,
        start_points: list[str | object] | str | object,
        end_points: list[str | object] | str | object,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> Any:
        """Decorator to verify PyTorch layer/function outputs against the model's sliced subgraph."""
        from neuron_toolkit.eval import GraphVerifier

        return GraphVerifier(self).verify_against_pytorch(
            start_points=start_points,
            end_points=end_points,
            rtol=rtol,
            atol=atol,
        )

    def apply_json_rules(
        self,
        rules: list[str] | None = None,
        mode: str = "function_proto",
    ) -> NeuronGraph:
        """Apply declarative JSON pattern rules (Cast folding, GELU fusion, Math collapse)."""
        from neuron_toolkit.rules.engine import JSONRuleEngine

        engine = JSONRuleEngine(mode=mode)
        engine.load_rules_from_dir()
        opt_proto = engine.apply(self, rule_ids=rules)
        return NeuronGraph(opt_proto)

    def run_pytorch(
        self,
        inputs: dict[str, Any],
        device: str = "cpu",
    ) -> dict[str, Any]:
        """Execute graph natively using PyTorch runtime engine."""
        from neuron_toolkit.runtime.runner import PyTorchRunner

        runner = PyTorchRunner(self, device=device)
        return runner.run(inputs)

    def cross_verify(
        self,
        simplified_graph: NeuronGraph | object | str,
        inputs: dict[str, Any] | None = None,
        rtol: float = 1e-4,
        atol: float = 1e-4,
    ) -> Any:
        """Cross-verify numerical outputs between this model (on ORT) and a simplified model (on PyTorch)."""
        from neuron_toolkit.runtime.cross_verifier import CrossVerifier

        verifier = CrossVerifier(self, simplified_graph, rtol=rtol, atol=atol)
        return verifier.verify(inputs=inputs)

    @property
    def passes(self) -> GraphPasses:
        """Return a namespace for common graph optimization passes."""
        return GraphPasses(self)


class GraphPasses:
    """Namespace for common graph optimization passes."""

    def __init__(self, graph: NeuronGraph) -> None:
        self.graph = graph

    def fuse_conv_bn(self) -> NeuronRewriter:
        """Fuse a Conv followed by BatchNormalization into a single Conv.

        The BatchNormalization scale/shift are folded into the Conv weights
        (and bias), producing an equivalent graph containing only the Conv
        operator. Returns a rewriter; call ``.build()`` to apply it.
        """
        import numpy as np

        from neuron_toolkit.pattern import Pattern

        conv = Pattern.op("Conv").capture("conv")
        bn = Pattern.op("BatchNormalization", conv).capture("bn")
        matches = self.graph.findall(bn)
        rewriter = self.graph.rewriter()
        if not matches:
            return rewriter

        backend = self.graph._backend
        tensor_map = self.graph.tensor_map

        def _const(name: str) -> np.ndarray | None:
            val = tensor_map.get(name)
            return val if isinstance(val, np.ndarray) else None

        for m in matches:
            conv_node = m.bindings["conv"]
            bn_node = m.bindings["bn"]
            bn_inputs = list(getattr(bn_node, "input", []))
            conv_inputs = list(getattr(conv_node, "input", []))
            if len(bn_inputs) != 5 or len(conv_inputs) < 2:
                continue

            scale = _const(bn_inputs[1])
            beta = _const(bn_inputs[2])
            mean = _const(bn_inputs[3])
            var = _const(bn_inputs[4])
            weight = _const(conv_inputs[1])
            if (
                scale is None
                or beta is None
                or mean is None
                or var is None
                or weight is None
            ):
                continue
            scale = np.asarray(scale)
            beta = np.asarray(beta)
            mean = np.asarray(mean)
            var = np.asarray(var)
            weight = np.asarray(weight)
            if not (scale.shape == beta.shape == mean.shape == var.shape):
                continue
            if weight.shape[0] != scale.shape[0]:
                continue

            bn_attrs = backend.get_node_attrs(bn_node)
            if bn_attrs.get("training_mode", 0) not in (0, False):
                continue
            eps = float(bn_attrs.get("epsilon", 1e-5))

            dtype = weight.dtype
            multiplier = (scale / np.sqrt(var + eps)).astype(dtype)
            scale_b = multiplier.reshape((scale.shape[0],) + (1,) * (weight.ndim - 1))
            new_weight = (weight * scale_b).astype(dtype)

            gamma_scale = multiplier
            if len(conv_inputs) > 2 and conv_inputs[2]:
                conv_bias = _const(conv_inputs[2])
                if conv_bias is None:
                    continue
                new_bias = (
                    np.asarray(conv_bias) * gamma_scale + beta - mean * gamma_scale
                ).astype(dtype)
            else:
                new_bias = (beta - mean * gamma_scale).astype(dtype)

            bn_name = getattr(bn_node, "name", None)
            fused_w = f"{bn_name or 'fused_conv_bn'}_weight"
            fused_b = f"{bn_name or 'fused_conv_bn'}_bias"
            rewriter.register_initializer(fused_w, new_weight)
            rewriter.register_initializer(fused_b, new_bias)

            rewriter.replace(
                [conv_node, bn_node],
                "Conv",
                inputs=[conv_inputs[0], fused_w, fused_b],
                outputs=list(getattr(bn_node, "output", [])),
                name=bn_name or "conv_bn_fused",
                **backend.get_node_attrs(conv_node),
            )
        return rewriter

    def cse(self) -> NeuronGraph:
        """Apply CommonSubexpressionElimination pass."""
        from neuron_toolkit.passes import CommonSubexpressionElimination

        new_g, _ = CommonSubexpressionElimination().run(self.graph)
        return new_g

    def fuse_gelu(self) -> NeuronGraph:
        """Apply FuseGELU pass."""
        from neuron_toolkit.passes import FuseGELU

        new_g, _ = FuseGELU().run(self.graph)
        return new_g

    def fuse_layer_norm(self) -> NeuronGraph:
        """Apply FuseLayerNorm pass."""
        from neuron_toolkit.passes import FuseLayerNorm

        new_g, _ = FuseLayerNorm().run(self.graph)
        return new_g

    def fuse_conv_act(self) -> NeuronGraph:
        """Apply FuseConvActivation pass."""
        from neuron_toolkit.passes import FuseConvActivation

        new_g, _ = FuseConvActivation().run(self.graph)
        return new_g

    def dce(self) -> NeuronGraph:
        """Apply DeadCodeElimination pass."""
        from neuron_toolkit.passes import DeadCodeElimination

        new_g, _ = DeadCodeElimination().run(self.graph)
        return new_g

    def const_fold(self) -> NeuronGraph:
        """Apply ConstantFolding pass."""
        from neuron_toolkit.passes import ConstantFolding

        new_g, _ = ConstantFolding().run(self.graph)
        return new_g

    def fuse_rmsnorm(self) -> NeuronGraph:
        """Apply FuseRMSNorm pass."""
        from neuron_toolkit.passes import FuseRMSNorm

        new_g, _ = FuseRMSNorm().run(self.graph)
        return new_g

    def fuse_swiglu(self) -> NeuronGraph:
        """Apply FuseSwiGLU pass."""
        from neuron_toolkit.passes import FuseSwiGLU

        new_g, _ = FuseSwiGLU().run(self.graph)
        return new_g

    def fuse_attention(self) -> NeuronGraph:
        """Apply FuseAttention pass."""
        from neuron_toolkit.passes import FuseAttention

        new_g, _ = FuseAttention().run(self.graph)
        return new_g

    def fuse_rope(self) -> NeuronGraph:
        """Apply FuseRotaryEmbedding pass."""
        from neuron_toolkit.passes import FuseRotaryEmbedding

        new_g, _ = FuseRotaryEmbedding().run(self.graph)
        return new_g
