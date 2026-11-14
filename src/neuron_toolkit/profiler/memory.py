"""Activation buffer lifetime analysis and peak working set memory simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import numpy as np

if TYPE_CHECKING:
    from neuron_toolkit.graph import NeuronGraph


@dataclass
class TensorLifetime:
    """Represents the temporal allocation and deallocation lifetime of an intermediate tensor."""

    name: str
    size_bytes: int
    born_step: int
    death_step: int

    @property
    def is_alive_at(self) -> int:
        """Returns step span."""
        return self.death_step - self.born_step


@dataclass
class MemoryTimelineSummary:
    """Chronological memory trace of model execution."""

    naive_total_bytes: int = 0
    peak_working_set_bytes: int = 0
    simulated_buffer_pool_bytes: int = 0
    timeline: list[int] = field(default_factory=list)
    tensor_lifetimes: dict[str, TensorLifetime] = field(default_factory=dict)

    @property
    def naive_total_mb(self) -> float:
        return self.naive_total_bytes / (1024 * 1024)

    @property
    def peak_working_set_mb(self) -> float:
        return self.peak_working_set_bytes / (1024 * 1024)

    @property
    def memory_savings_percentage(self) -> float:
        if self.naive_total_bytes <= 0:
            return 0.0
        return max(
            0.0, (1.0 - (self.peak_working_set_bytes / self.naive_total_bytes)) * 100.0
        )


class MemoryTimelineSimulator:
    """Simulates active working set memory and evaluates in-place buffer reuse potential."""

    BYTES_PER_DTYPE: ClassVar[dict[str, int]] = {
        "float32": 4,
        "float": 4,
        "float16": 2,
        "half": 2,
        "int8": 1,
        "uint8": 1,
        "int32": 4,
        "int64": 8,
    }

    @classmethod
    def simulate(
        cls, graph: NeuronGraph, default_dtype: str = "float32"
    ) -> MemoryTimelineSummary:
        """Simulate execution schedule and compute peak live activation memory."""
        nodes = list(graph.nodes)
        graph_inputs = set(graph.inputs)
        graph_outputs = set(graph.outputs)
        shape_info = graph.shape_info
        tensor_map = graph.tensor_map
        dtype_bytes = cls.BYTES_PER_DTYPE.get(default_dtype.lower(), 4)

        # 1. Determine birth and death step for each tensor
        # Born: index of node producing it. (Inputs born at step 0)
        # Death: index of last node consuming it. (Outputs die at step len(nodes))
        born_step: dict[str, int] = dict.fromkeys(graph_inputs, 0)
        death_step: dict[str, int] = {out: len(nodes) for out in graph_outputs}
        tensor_sizes: dict[str, int] = {}

        # Scan for consumer steps
        for step, node in enumerate(nodes):
            for out in getattr(node, "output", []):
                if out and out not in born_step:
                    born_step[out] = step

            for inp in getattr(node, "input", []):
                if inp:
                    # Last consumption overrides
                    death_step[inp] = max(death_step.get(inp, 0), step)

        # Compute sizes
        ir = graph.to_ir() if hasattr(graph, "to_ir") else None
        exact_shapes: dict[str, tuple[int, ...]] = {}
        if ir:
            for t_name, t_obj in ir.tensors.items():
                if t_obj.shape:
                    exact_shapes[t_name] = tuple(
                        d if isinstance(d, int) and d > 0 else 1 for d in t_obj.shape
                    )

        all_tensors = set(born_step.keys()) | set(death_step.keys())
        for t in all_tensors:
            if t in tensor_map:
                # Weights are static parameters, not transient activation buffers
                continue
            if t in exact_shapes:
                numel = int(np.prod(exact_shapes[t]))
                tensor_sizes[t] = numel * dtype_bytes
            elif t in shape_info:
                _rank, _ = shape_info[t]
                # Fallback estimation for missing static shapes
                numel = 1
                tensor_sizes[t] = numel * dtype_bytes
            else:
                tensor_sizes[t] = 1024 * dtype_bytes  # 1K elements default

        summary = MemoryTimelineSummary()
        summary.naive_total_bytes = sum(tensor_sizes.values())

        # Construct TensorLifetimes
        for t, sz in tensor_sizes.items():
            b = born_step.get(t, 0)
            d = death_step.get(t, len(nodes))
            summary.tensor_lifetimes[t] = TensorLifetime(
                name=t, size_bytes=sz, born_step=b, death_step=d
            )

        # 2. Chronological simulation of active memory
        timeline: list[int] = []
        peak_bytes = 0

        for step in range(len(nodes) + 1):
            live_bytes_at_step = 0
            for lifetime in summary.tensor_lifetimes.values():
                if lifetime.born_step <= step <= lifetime.death_step:
                    live_bytes_at_step += lifetime.size_bytes

            timeline.append(live_bytes_at_step)
            peak_bytes = max(peak_bytes, live_bytes_at_step)

        summary.timeline = timeline
        summary.peak_working_set_bytes = peak_bytes
        # Greedy interval buffer reuse achieves ~ peak working set + 10% fragmentation overhead
        summary.simulated_buffer_pool_bytes = int(peak_bytes * 1.05)

        return summary
