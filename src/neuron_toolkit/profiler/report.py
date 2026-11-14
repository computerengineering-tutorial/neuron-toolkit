"""Structured profiling reports, Markdown tables, and console summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neuron_toolkit.profiler.flops import FLOPsSummary
    from neuron_toolkit.profiler.memory import MemoryTimelineSummary
    from neuron_toolkit.profiler.roofline import RooflineAnalysis


@dataclass
class ProfilerReport:
    """Unified profiling report containing FLOPs, memory footprint, and Roofline metrics."""

    flops: FLOPsSummary
    memory: MemoryTimelineSummary
    roofline: RooflineAnalysis

    def to_dict(self) -> dict[str, Any]:
        """Convert profiling metrics to JSON-serializable dictionary."""
        return {
            "hardware_target": self.roofline.target.name,
            "precision": self.roofline.dtype,
            "total_parameters": self.flops.total_params,
            "total_gflops": round(self.flops.gflops, 4),
            "total_gmacs": round(self.flops.gmacs, 4),
            "naive_memory_mb": round(self.memory.naive_total_mb, 3),
            "peak_working_set_mb": round(self.memory.peak_working_set_mb, 3),
            "memory_savings_pct": round(self.memory.memory_savings_percentage, 1),
            "ridge_point": round(self.roofline.ridge_point, 2),
            "global_arithmetic_intensity": round(
                self.roofline.global_point.arithmetic_intensity, 2
            ),
            "bound_classification": "Compute-Bound"
            if self.roofline.global_point.is_compute_bound
            else "Memory-Bound",
            "estimated_latency_ms": round(
                self.roofline.global_point.estimated_latency_ms, 3
            ),
        }

    def summary(self) -> str:
        """Formatted plain-text summary suitable for terminal inspection."""
        d = self.to_dict()
        lines = [
            "=" * 60,
            f" NEURON TOOLKIT HARDWARE & MEMORY PROFILER: {d['hardware_target']}",
            "=" * 60,
            f"  Precision:              {d['precision'].upper()}",
            f"  Total Parameters:       {d['total_parameters']:,} params",
            f"  Compute:                {d['total_gflops']:.3f} GFLOPs ({d['total_gmacs']:.3f} GMACs)",
            f"  Naive Memory:           {d['naive_memory_mb']:.2f} MB",
            f"  Peak Working Set:       {d['peak_working_set_mb']:.2f} MB ({d['memory_savings_pct']:.1f}% buffer reuse)",
            "-" * 60,
            f"  Roofline Ridge Point:   {d['ridge_point']:.2f} FLOP/byte",
            f"  Arithmetic Intensity:   {d['global_arithmetic_intensity']:.2f} FLOP/byte",
            f"  Classification:         {d['bound_classification'].upper()}",
            f"  Estimated Latency:      {d['estimated_latency_ms']:.3f} ms",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Formatted GitHub Markdown table."""
        d = self.to_dict()
        return f"""### Hardware Profile: {d["hardware_target"]} ({d["precision"].upper()})

| Metric | Value |
| :--- | :--- |
| **Total Parameters** | {d["total_parameters"]:,} |
| **Compute Operations** | {d["total_gflops"]:.3f} GFLOPs ({d["total_gmacs"]:.3f} GMACs) |
| **Naive Memory Requirement** | {d["naive_memory_mb"]:.2f} MB |
| **Peak Working Set Memory** | {d["peak_working_set_mb"]:.2f} MB ({d["memory_savings_pct"]:.1f}% reduction) |
| **Arithmetic Intensity** | {d["global_arithmetic_intensity"]:.2f} FLOP/byte |
| **Hardware Ridge Point** | {d["ridge_point"]:.2f} FLOP/byte |
| **Bottleneck Regime** | **{d["bound_classification"]}** |
| **Predicted Minimum Latency** | **{d["estimated_latency_ms"]:.3f} ms** |
"""
