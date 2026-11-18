"""PassManager pipeline for sequencing and tracking graph transformation passes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from neuron_toolkit.passes.base import BasePass

if TYPE_CHECKING:
    from neuron_toolkit.graph import NeuronGraph


class PassResultsList(list[dict[str, Any]]):
    """List of pass results, supporting both integer indexing and lookup by pass name."""

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            for res in self:
                if res.get("pass_name") == item:
                    return res
            msg = f"Pass result for {item!r} not found in pass_results."
            raise KeyError(msg)
        return super().__getitem__(item)


@dataclass
class PassStats:
    """Statistics recorded during PassManager execution."""

    initial_nodes: int = 0
    final_nodes: int = 0
    pass_results: PassResultsList = field(default_factory=PassResultsList)

    def __getitem__(self, key: str | int) -> dict[str, Any]:
        """Support direct indexing stats['PassName'] or stats[0]."""
        return self.pass_results[key]

    @property
    def node_reduction(self) -> int:
        """Difference between initial and final node counts."""
        return self.initial_nodes - self.final_nodes

    @property
    def reduction_percentage(self) -> float:
        """Percentage of nodes removed."""
        if self.initial_nodes == 0:
            return 0.0
        return (self.node_reduction / self.initial_nodes) * 100.0

    def summary(self) -> str:
        """Return a formatted string summarizing execution results."""
        lines = [
            "PassManager Execution Summary:",
            f"  Initial node count: {self.initial_nodes}",
            f"  Final node count:   {self.final_nodes} ({self.reduction_percentage:.1f}% reduction)",
        ]
        for p in self.pass_results:
            p_name = p.get("pass_name", "UnknownPass")
            details = ", ".join(f"{k}={v}" for k, v in p.items() if k != "pass_name")
            lines.append(f"  - {p_name:24}: {details}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


class PassManager:
    """Manages the registration and execution of optimization passes on a NeuronGraph."""

    def __init__(self, passes: list[BasePass] | None = None) -> None:
        self.passes: list[BasePass] = list(passes) if passes else []

    def add_pass(self, opt_pass: BasePass) -> PassManager:
        """Add an optimization pass to the pipeline."""
        self.passes.append(opt_pass)
        return self

    def run(self, graph: NeuronGraph) -> tuple[NeuronGraph, PassStats]:
        """Execute all registered passes sequentially on *graph*."""
        stats = PassStats(initial_nodes=len(graph.nodes))
        curr_graph = graph

        for p in self.passes:
            curr_graph, p_stats = p.run(curr_graph)
            p_stats["pass_name"] = p.name
            stats.pass_results.append(p_stats)

        stats.final_nodes = len(curr_graph.nodes)
        return curr_graph, stats

    def run_until_fixed_point(
        self, graph: NeuronGraph, max_iterations: int = 10
    ) -> tuple[NeuronGraph, PassStats]:
        """Execute registered passes iteratively until graph node count stabilizes."""
        stats = PassStats(initial_nodes=len(graph.nodes))
        curr_graph = graph

        for iteration in range(max_iterations):
            prev_node_count = len(curr_graph.nodes)
            iter_changed = False
            for p in self.passes:
                before_nodes = len(curr_graph.nodes)
                curr_graph, p_stats = p.run(curr_graph)
                p_stats["pass_name"] = f"{p.name}_iter{iteration}"
                stats.pass_results.append(p_stats)
                if len(curr_graph.nodes) != before_nodes:
                    iter_changed = True

            # Early exit if passes reached a fixed point
            if not iter_changed and len(curr_graph.nodes) == prev_node_count:
                break

        stats.final_nodes = len(curr_graph.nodes)
        return curr_graph, stats
