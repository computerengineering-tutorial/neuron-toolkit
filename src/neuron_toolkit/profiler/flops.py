"""Analytical FLOPs and MACs profiler for neural computational graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import numpy as np

if TYPE_CHECKING:
    from neuron_toolkit.graph import NeuronGraph


@dataclass
class OpProfile:
    """Performance metrics for an individual operator node."""

    name: str
    op_type: str
    flops: int = 0
    macs: int = 0
    params: int = 0
    bytes_transferred: int = 0
    input_shapes: list[tuple[int, ...]] = field(default_factory=list)
    output_shapes: list[tuple[int, ...]] = field(default_factory=list)

    @property
    def arithmetic_intensity(self) -> float:
        """Arithmetic Intensity in FLOPs / byte."""
        if self.bytes_transferred <= 0:
            return 0.0
        return self.flops / self.bytes_transferred


@dataclass
class FLOPsSummary:
    """Summary of FLOPs, MACs, and parameter counts across the graph."""

    total_flops: int = 0
    total_macs: int = 0
    total_params: int = 0
    total_bytes: int = 0
    op_breakdown: list[OpProfile] = field(default_factory=list)

    @property
    def gflops(self) -> float:
        """GigaFLOPs (10^9)."""
        return self.total_flops / 1e9

    @property
    def gmacs(self) -> float:
        """GigaMACs (10^9)."""
        return self.total_macs / 1e9

    @property
    def arithmetic_intensity(self) -> float:
        """Global arithmetic intensity."""
        if self.total_bytes <= 0:
            return 0.0
        return self.total_flops / self.total_bytes


class FLOPsProfiler:
    """Profiles FLOPs, MACs, and I/O data volumes across graph nodes."""

    DTYPE_BYTES: ClassVar[dict[str, int]] = {
        "float32": 4,
        "float": 4,
        "float16": 2,
        "half": 2,
        "bfloat16": 2,
        "int8": 1,
        "uint8": 1,
        "int32": 4,
        "int64": 8,
    }

    @classmethod
    def profile(
        cls, graph: NeuronGraph, default_dtype: str = "float32"
    ) -> FLOPsSummary:
        """Compute analytical FLOPs, MACs, and data transfers for *graph*."""
        summary = FLOPsSummary()
        tensor_map = graph.tensor_map
        shape_info = graph.shape_info
        bytes_per_elem = cls.DTYPE_BYTES.get(default_dtype.lower(), 4)

        ir = graph.to_ir() if hasattr(graph, "to_ir") else None
        exact_shapes: dict[str, tuple[int, ...]] = {}
        if ir:
            for t_name, t_obj in ir.tensors.items():
                if t_obj.shape:
                    exact_shapes[t_name] = tuple(
                        d if isinstance(d, int) and d > 0 else 1 for d in t_obj.shape
                    )

        for node in graph.nodes:
            op_type = getattr(node, "op_type", "")
            node_name = (
                getattr(node, "name", "") or f"{op_type}_{len(summary.op_breakdown)}"
            )
            inputs = getattr(node, "input", [])
            outputs = getattr(node, "output", [])

            input_shapes: list[tuple[int, ...]] = []
            output_shapes: list[tuple[int, ...]] = []
            param_count = 0
            io_elements = 0

            # Gather input shapes & weights
            for inp in inputs:
                if inp in tensor_map:
                    arr = tensor_map[inp]
                    param_count += arr.size
                    input_shapes.append(tuple(arr.shape))
                    io_elements += arr.size
                elif inp in exact_shapes:
                    s = exact_shapes[inp]
                    input_shapes.append(s)
                    io_elements += int(np.prod(s))
                elif inp in shape_info:
                    rank, _ = shape_info[inp]
                    s = (1,) * max(rank, 1)
                    input_shapes.append(s)
                    io_elements += int(np.prod(s))

            for out in outputs:
                if out in exact_shapes:
                    s = exact_shapes[out]
                    output_shapes.append(s)
                    io_elements += int(np.prod(s))
                elif out in shape_info:
                    rank, _ = shape_info[out]
                    s = (1,) * max(rank, 1)
                    output_shapes.append(s)
                    io_elements += int(np.prod(s))

            flops, macs = cls._compute_op_flops(
                op_type, node, input_shapes, output_shapes, tensor_map
            )
            bytes_transferred = io_elements * bytes_per_elem

            profile = OpProfile(
                name=node_name,
                op_type=op_type,
                flops=flops,
                macs=macs,
                params=param_count,
                bytes_transferred=bytes_transferred,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
            )
            summary.op_breakdown.append(profile)
            summary.total_flops += flops
            summary.total_macs += macs
            summary.total_params += param_count
            summary.total_bytes += bytes_transferred

        return summary

    @classmethod
    def _compute_op_flops(
        cls,
        op_type: str,
        node: object,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
        tensor_map: dict[str, np.ndarray],
    ) -> tuple[int, int]:
        """Calculate FLOPs and MACs for a specific operator."""
        inputs = getattr(node, "input", [])

        # 1. Convolution (Conv, ConvTranspose)
        if (
            op_type in ("Conv", "ConvTranspose")
            and len(inputs) >= 2
            and inputs[1] in tensor_map
        ):
            w_shape = tensor_map[inputs[1]].shape
            out_shape = output_shapes[0] if output_shapes else (1, w_shape[0], 1, 1)
            b = out_shape[0] if len(out_shape) > 0 else 1
            c_out = w_shape[0]
            kernel_size = int(np.prod(w_shape[2:])) if len(w_shape) > 2 else 1
            spatial_out = int(np.prod(out_shape[2:])) if len(out_shape) > 2 else 1
            c_in_per_group = w_shape[1]

            macs = b * c_out * spatial_out * c_in_per_group * kernel_size
            has_bias = len(inputs) > 2 and inputs[2] in tensor_map
            flops = 2 * macs + (b * c_out * spatial_out if has_bias else 0)
            return flops, macs

        # 2. Matrix Multiplication (MatMul, Gemm)
        if op_type in ("MatMul", "Gemm") and len(inputs) >= 2:
            s0 = input_shapes[0] if input_shapes else (1, 1)
            s1 = input_shapes[1] if len(input_shapes) > 1 else (1, 1)
            m = s0[-2] if len(s0) >= 2 else s0[0]
            k = s0[-1]
            n = s1[-1]
            batch = int(np.prod(s0[:-2])) if len(s0) > 2 else 1

            macs = batch * m * n * k
            flops = 2 * macs
            return flops, macs

        # 3. Attention / SDPA
        if op_type in ("Attention", "ScaledDotProductAttention"):
            # Standard SDPA: 4 * B * H * S^2 * D + 3 * B * H * S^2
            s_q = input_shapes[0] if input_shapes else (1, 8, 32, 64)
            b = s_q[0] if len(s_q) > 0 else 1
            h = s_q[1] if len(s_q) > 1 else 8
            s = s_q[-2] if len(s_q) >= 2 else 32
            d = s_q[-1] if len(s_q) >= 1 else 64

            macs = 2 * b * h * s * s * d
            flops = 2 * macs + 3 * b * h * s * s
            return flops, macs

        # 4. Normalization (LayerNormalization, RMSNorm, BatchNormalization)
        if op_type in ("LayerNormalization", "RMSNorm", "BatchNormalization"):
            elements = int(np.prod(input_shapes[0])) if input_shapes else 1
            flops_per_elem = 5 if op_type == "LayerNormalization" else 3
            return elements * flops_per_elem, elements

        # 5. Activations and Elementwise
        if op_type in (
            "Relu",
            "Sigmoid",
            "Silu",
            "Gelu",
            "Tanh",
            "Add",
            "Sub",
            "Mul",
            "Div",
        ):
            elements = int(np.prod(input_shapes[0])) if input_shapes else 1
            flops_per_elem = 1
            if op_type in ("Silu", "Gelu"):
                flops_per_elem = 4
            elif op_type in ("Sigmoid", "Tanh"):
                flops_per_elem = 3
            return elements * flops_per_elem, 0

        return 0, 0
