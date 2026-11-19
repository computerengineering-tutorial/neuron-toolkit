#!/usr/bin/env python3
"""Synthetic PyTorch Graph Generator: Linear Variants.

Generates ONNX models for:
- MatMul + Add
- Gemm (transB=1)
- Einsum ('btd,df->btf')
- Conv1x1 (point-wise 2D convolution)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from torch import nn


class LinearMatMulAdd(nn.Module):
    """X @ W + B."""

    def __init__(self, in_features: int = 16, out_features: int = 32) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn(in_features, out_features))
        self.b = nn.Parameter(torch.randn(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.w) + self.b


class LinearGemm(nn.Module):
    """Linear projection exported as Gemm."""

    def __init__(self, in_features: int = 16, out_features: int = 32) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 2D input -> exports to Gemm in ONNX
        return self.linear(x)


class LinearEinsum(nn.Module):
    """torch.einsum('btd,df->btf')."""

    def __init__(self, in_features: int = 16, out_features: int = 32) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn(in_features, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("btd,df->btf", x, self.w)


class Conv1x1Linear(nn.Module):
    """1x1 Conv pointwise linear projection."""

    def __init__(self, in_channels: int = 16, out_channels: int = 32) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class CombinedLinearVariants(nn.Module):
    """Combines all linear variants in sequence."""

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = LinearMatMulAdd(16, 24)
        self.layer2 = LinearEinsum(24, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.layer1(x)
        return self.layer2(h1)


from generators.base import export_model as unified_export_model


def generate_all() -> None:
    models_dir = Path(".helper/models")
    dummy_3d = torch.randn(2, 8, 16)
    dummy_2d = torch.randn(4, 16)
    dummy_4d = torch.randn(2, 16, 8, 8)

    # 1. Combined model
    unified_export_model(
        CombinedLinearVariants(),
        (dummy_3d,),
        ["input"],
        ["output"],
        "linear_variants",
        models_dir,
        opset_versions=[14, 17, 18, 20],
        dynamic_axes={"input": {0: "batch", 1: "time"}, "output": {0: "batch", 1: "time"}},
        expected_target_node="Linear",
        min_expected_reduction=0.3,
    )

    # 2. Gemm variant
    unified_export_model(
        LinearGemm(16, 32),
        (dummy_2d,),
        ["input"],
        ["output"],
        "linear_gemm",
        models_dir,
        opset_versions=[14, 17, 18, 20],
        expected_target_node="Linear",
        min_expected_reduction=0.0,
    )

    # 3. Conv 1x1 variant
    unified_export_model(
        Conv1x1Linear(16, 32),
        (dummy_4d,),
        ["input"],
        ["output"],
        "linear_conv1x1",
        models_dir,
        opset_versions=[14, 17, 18, 20],
        expected_target_node="Conv",
        emit_json_descriptor=False,
    )


if __name__ == "__main__":
    generate_all()
