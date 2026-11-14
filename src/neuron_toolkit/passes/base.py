"""Base interface for graph transformation passes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neuron_toolkit.graph import NeuronGraph


class BasePass(ABC):
    """Abstract base class for all graph optimization and transformation passes."""

    name: str = "BasePass"

    @abstractmethod
    def run(self, graph: NeuronGraph) -> tuple[NeuronGraph, dict[str, Any]]:
        """Apply the pass to *graph*.

        Returns:
            A tuple of (transformed_graph, stats_dictionary).
        """
        raise NotImplementedError
