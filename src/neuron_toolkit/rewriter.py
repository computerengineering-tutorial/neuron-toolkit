from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neuron_toolkit.backends.base import BaseRewriter


class NeuronRewriter:
    """Base interface for model transformation engines.

    This class serves as a unified entry point for all backend-specific rewriters.
    """

    def __init__(self, backend_rewriter: BaseRewriter) -> None:
        self._impl = backend_rewriter

    def build(self, output_path: str | None = None) -> object:
        """Apply edits and return the modified model."""
        return self._impl.build(output_path=output_path)

    def build_graph(self, output_path: str | None = None) -> Any:
        """Apply edits and return a new NeuronGraph."""
        from neuron_toolkit.graph import NeuronGraph

        res = self.build(output_path=output_path)
        return NeuronGraph(res)

    def delete(self, nodes: Sequence[object] | object) -> NeuronRewriter:
        """Delete nodes from the graph."""
        node_seq = [nodes] if not isinstance(nodes, (list, tuple, set)) else list(nodes)
        if hasattr(self._impl, "delete"):
            self._impl.delete(node_seq)
        return self

    def remove(self, node: object) -> NeuronRewriter:
        """Alias for delete a single node."""
        return self.delete([node])

    def __getattr__(self, name: str) -> Any:
        """Delegate everything else to the implementation."""
        return getattr(self._impl, name)


__all__ = ["NeuronRewriter"]
