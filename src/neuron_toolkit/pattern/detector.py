from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

try:
    import networkx as nx
except ImportError:
    nx = None

from neuron_toolkit._utils import ShapeInfo, _GraphShim
from neuron_toolkit.pattern._matcher import MatchContext, MatchingMixin
from neuron_toolkit.pattern.dsl import _ANY_OF, _CONST_PAT, _WILDCARD
from neuron_toolkit.pattern.models import MatchResult

if TYPE_CHECKING:
    from neuron_toolkit.pattern.dsl import Pattern


class PatternDetector(MatchingMixin):
    """Match a Pattern against a subgraph of a model (ONNX, TFLite, etc.)."""

    _nodes: Sequence[object]
    _tensor_map: Mapping[str, object]
    _shape_info: ShapeInfo

    def __init__(
        self,
        model: object,
        start_node: object | str | None = None,
        end_node: object | str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize PatternDetector with model and optional start/end nodes."""
        if isinstance(model, _GraphShim):
            self._nodes = model.nodes
            self._tensor_map = model.tensor_map
            self._shape_info: ShapeInfo = model.shape_info
            self._backend = model.backend
        elif hasattr(model, "graph") and hasattr(
            model.graph, "node"
        ):  # ONNX ModelProto
            from neuron_toolkit.backends.onnx.parser import ONNXParser

            self._backend = ONNXParser(cast(Any, model), **kwargs)
            self._nodes = self._backend.nodes
            self._tensor_map = self._backend.tensor_map
            self._shape_info = self._backend.shape_info
        elif hasattr(model, "Subgraphs") and hasattr(
            model, "OperatorCodes"
        ):  # TFLite Model
            from neuron_toolkit.backends.tflite.parser import (
                TFLiteParser,
            )

            self._backend = TFLiteParser(cast(Any, model), **kwargs)
            self._nodes = self._backend.nodes
            self._tensor_map = self._backend.tensor_map
            self._shape_info = self._backend.shape_info
        else:
            # Assume it's already a backend parser (ONNXParser or TFLiteParser)
            self._backend = model
            self._nodes = getattr(model, "nodes", [])
            self._tensor_map = getattr(model, "tensor_map", {})
            self._shape_info = getattr(model, "shape_info", {})

        output_map = kwargs.pop("output_to_node", None)
        if output_map is not None:
            self.output_to_node = output_map
        else:
            self.output_to_node = {
                out: n for n in self._nodes for out in getattr(n, "output", [])
            }
        self.start = self._resolve(start_node)
        self.end = self._resolve(end_node)
        self._nx_graph: nx.DiGraph | None = None

    def match(self, pattern: Pattern) -> MatchResult | None:
        """Attempt to match *pattern* starting from self.start."""
        if self.start is None:
            return None

        ctx = MatchContext(detector=self)
        if not self._match_recursive(self.start, pattern, ctx):
            return None

        terminal = self.end if self.end is not None else self.start
        return MatchResult(
            start=self.start,
            end=terminal,
            nodes=ctx.trail,
            bindings=ctx.bindings,
        )

    @property
    def op_to_nodes(self) -> dict[str, list[object]]:
        """Index mapping operator types to lists of matching nodes in the graph."""
        if not hasattr(self, "_op_index"):
            idx: dict[str, list[object]] = {}
            for n in self._nodes:
                op = getattr(n, "op_type", None)
                if op:
                    idx.setdefault(op, []).append(n)
            self._op_index = idx
        return self._op_index

    def find_all(self, pattern: Pattern) -> list[MatchResult]:
        """Find all matches for *pattern* in the reachable subgraph using root-op indexing."""
        if self.start is not None:
            candidates = self._descendant_nodes()
            if pattern.op_type and pattern.op_type not in (
                _ANY_OF,
                _CONST_PAT,
                _WILDCARD,
            ):
                candidates = [
                    n
                    for n in candidates
                    if getattr(n, "op_type", None) == pattern.op_type
                ]
            elif pattern.op_type == _ANY_OF and pattern.alternatives:
                allowed = {
                    alt.op_type
                    for alt in pattern.alternatives
                    if alt.op_type
                    and alt.op_type not in (_ANY_OF, _CONST_PAT, _WILDCARD)
                }
                if len(allowed) == len(pattern.alternatives):
                    candidates = [
                        n for n in candidates if getattr(n, "op_type", None) in allowed
                    ]
        elif pattern.op_type and pattern.op_type not in (
            _ANY_OF,
            _CONST_PAT,
            _WILDCARD,
        ):
            candidates = self.op_to_nodes.get(pattern.op_type, [])
        elif pattern.op_type == _ANY_OF and pattern.alternatives:
            allowed = {
                alt.op_type
                for alt in pattern.alternatives
                if alt.op_type and alt.op_type not in (_ANY_OF, _CONST_PAT, _WILDCARD)
            }
            if len(allowed) == len(pattern.alternatives):
                cand_list: list[object] = []
                for op in allowed:
                    cand_list.extend(self.op_to_nodes.get(op, []))
                candidates = cand_list
            else:
                candidates = self._nodes
        else:
            candidates = self._nodes

        orig_start = self.start
        results: list[MatchResult] = []
        try:
            for node in candidates:
                if node is self.end:
                    continue
                self.start = node
                r = self.match(pattern)
                if r is not None:
                    results.append(r)
        finally:
            self.start = orig_start
        return results
