"""Hardware profiling, FLOPs/MACs calculation, Roofline modeling, and memory planning."""

from neuron_toolkit.profiler.flops import FLOPsProfiler, FLOPsSummary, OpProfile
from neuron_toolkit.profiler.memory import (
    MemoryTimelineSimulator,
    MemoryTimelineSummary,
    TensorLifetime,
)
from neuron_toolkit.profiler.report import ProfilerReport
from neuron_toolkit.profiler.roofline import (
    APPLE_M3_MAX,
    ARM_CORTEX_A78,
    GENERIC_CPU,
    HARDWARE_REGISTRY,
    NVIDIA_A100_SXM4_80GB,
    NVIDIA_H100_SXM5,
    NVIDIA_RTX_4090,
    HardwareTarget,
    RooflineAnalysis,
    RooflinePoint,
)


def profile_graph(
    graph: object,
    device: str | HardwareTarget = "a100",
    dtype: str = "float16",
) -> ProfilerReport:
    """Convenience helper to profile a NeuronGraph."""
    flops_summary = FLOPsProfiler.profile(graph, default_dtype=dtype)  # type: ignore[arg-type]
    mem_summary = MemoryTimelineSimulator.simulate(graph, default_dtype=dtype)  # type: ignore[arg-type]
    roofline_analysis = RooflineAnalysis.analyze(
        flops_summary, device=device, dtype=dtype
    )
    return ProfilerReport(
        flops=flops_summary, memory=mem_summary, roofline=roofline_analysis
    )


__all__ = [
    "APPLE_M3_MAX",
    "ARM_CORTEX_A78",
    "GENERIC_CPU",
    "HARDWARE_REGISTRY",
    "NVIDIA_A100_SXM4_80GB",
    "NVIDIA_H100_SXM5",
    "NVIDIA_RTX_4090",
    "FLOPsProfiler",
    "FLOPsSummary",
    "HardwareTarget",
    "MemoryTimelineSimulator",
    "MemoryTimelineSummary",
    "OpProfile",
    "ProfilerReport",
    "RooflineAnalysis",
    "RooflinePoint",
    "TensorLifetime",
    "profile_graph",
]
