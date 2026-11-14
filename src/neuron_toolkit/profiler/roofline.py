"""Roofline model and hardware latency estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuron_toolkit.profiler.flops import FLOPsSummary


@dataclass(frozen=True)
class HardwareTarget:
    """Hardware target profile specifying compute throughput and memory bandwidth.

    Attributes:
        name: Device identifier.
        peak_tflops_fp16: Peak half-precision tensor compute throughput (TFLOPS).
        peak_tflops_fp32: Peak single-precision compute throughput (TFLOPS).
        memory_bandwidth_gbs: Peak DRAM memory bandwidth (GB/s).
    """

    name: str
    peak_tflops_fp16: float
    peak_tflops_fp32: float
    memory_bandwidth_gbs: float

    def ridge_point(self, dtype: str = "float16") -> float:
        """Arithmetic Intensity threshold separating Memory-Bound from Compute-Bound workloads.

        Ridge Point = Peak FLOPS / Memory Bandwidth (FLOP/byte).
        """
        peak_tflops = self.peak_tflops_fp16 if "16" in dtype else self.peak_tflops_fp32
        peak_flops = peak_tflops * 1e12
        bandwidth_bytes_s = self.memory_bandwidth_gbs * 1e9
        return peak_flops / bandwidth_bytes_s


# Standard pre-defined hardware profiles
NVIDIA_A100_SXM4_80GB = HardwareTarget(
    name="NVIDIA A100 SXM4 80GB",
    peak_tflops_fp16=312.0,
    peak_tflops_fp32=19.5,
    memory_bandwidth_gbs=2039.0,
)

NVIDIA_H100_SXM5 = HardwareTarget(
    name="NVIDIA H100 SXM5",
    peak_tflops_fp16=989.0,
    peak_tflops_fp32=67.0,
    memory_bandwidth_gbs=3350.0,
)

NVIDIA_RTX_4090 = HardwareTarget(
    name="NVIDIA GeForce RTX 4090",
    peak_tflops_fp16=165.2,
    peak_tflops_fp32=82.6,
    memory_bandwidth_gbs=1008.0,
)

APPLE_M3_MAX = HardwareTarget(
    name="Apple M3 Max (40-core GPU)",
    peak_tflops_fp16=33.0,
    peak_tflops_fp32=16.5,
    memory_bandwidth_gbs=400.0,
)

ARM_CORTEX_A78 = HardwareTarget(
    name="ARM Cortex-A78 Mobile CPU",
    peak_tflops_fp16=0.2,
    peak_tflops_fp32=0.1,
    memory_bandwidth_gbs=34.0,
)

GENERIC_CPU = HardwareTarget(
    name="Generic x86-64 Server CPU",
    peak_tflops_fp16=1.0,
    peak_tflops_fp32=0.8,
    memory_bandwidth_gbs=64.0,
)

HARDWARE_REGISTRY: dict[str, HardwareTarget] = {
    "a100": NVIDIA_A100_SXM4_80GB,
    "h100": NVIDIA_H100_SXM5,
    "rtx4090": NVIDIA_RTX_4090,
    "4090": NVIDIA_RTX_4090,
    "m3": APPLE_M3_MAX,
    "apple_m3": APPLE_M3_MAX,
    "arm": ARM_CORTEX_A78,
    "cpu": GENERIC_CPU,
}


@dataclass
class RooflinePoint:
    """Roofline classification for an op or full model."""

    name: str
    flops: int
    bytes_transferred: int
    arithmetic_intensity: float
    is_compute_bound: bool
    estimated_latency_ms: float


@dataclass
class RooflineAnalysis:
    """Comprehensive roofline model analysis against a hardware target."""

    target: HardwareTarget
    dtype: str
    ridge_point: float
    global_point: RooflinePoint
    op_points: list[RooflinePoint]

    @classmethod
    def analyze(
        cls,
        summary: FLOPsSummary,
        device: str | HardwareTarget = "a100",
        dtype: str = "float16",
    ) -> RooflineAnalysis:
        target = (
            HARDWARE_REGISTRY.get(str(device).lower(), NVIDIA_A100_SXM4_80GB)
            if isinstance(device, str)
            else device
        )
        ridge = target.ridge_point(dtype=dtype)

        peak_tflops = (
            target.peak_tflops_fp16 if "16" in dtype else target.peak_tflops_fp32
        )
        peak_flops = peak_tflops * 1e12
        bandwidth_bytes_s = target.memory_bandwidth_gbs * 1e9

        def _calc_point(name: str, flops: int, nbytes: int) -> RooflinePoint:
            ai = (flops / nbytes) if nbytes > 0 else 0.0
            is_comp = ai >= ridge
            compute_time_s = flops / peak_flops
            memory_time_s = nbytes / bandwidth_bytes_s if bandwidth_bytes_s > 0 else 0.0
            latency_ms = max(compute_time_s, memory_time_s) * 1000.0
            return RooflinePoint(
                name=name,
                flops=flops,
                bytes_transferred=nbytes,
                arithmetic_intensity=ai,
                is_compute_bound=is_comp,
                estimated_latency_ms=latency_ms,
            )

        global_p = _calc_point("FullModel", summary.total_flops, summary.total_bytes)
        op_points = [
            _calc_point(op.name, op.flops, op.bytes_transferred)
            for op in summary.op_breakdown
        ]

        return cls(
            target=target,
            dtype=dtype,
            ridge_point=ridge,
            global_point=global_p,
            op_points=op_points,
        )
