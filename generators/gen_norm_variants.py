#!/usr/bin/env python3
"""Synthetic PyTorch Graph Generator: Normalization Variants.

Generates ONNX models for:
- RMSNorm (ReduceL2 & MeanSquare formulations)
- LayerNorm (Mean/Var expansion)
- GroupNorm (InstanceNorm 1D adapter formulation)
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


class RMSNormL2Module(nn.Module):
    """Explicit ReduceL2-based RMSNorm: x * (sqrt(D) / clamp(norm(x), min=eps)) * gamma."""

    def __init__(self, d_model: int = 32, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # L2 norm along last dim
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        scale = (x.shape[-1] ** 0.5) / torch.clamp(norm, min=self.eps)
        return x * scale * self.gamma


class RMSNormMeanSqModule(nn.Module):
    """Explicit MeanSquare-based RMSNorm: x / sqrt(mean(x^2) + eps) * gamma."""

    def __init__(self, d_model: int = 32, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = torch.mean(torch.pow(x, 2), dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.gamma


class LayerNormExpandedModule(nn.Module):
    """Explicit LayerNorm mean/variance decomposition."""

    def __init__(self, d_model: int = 32, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.mean(torch.pow(x - mean, 2), dim=-1, keepdim=True)
        return ((x - mean) / torch.sqrt(var + self.eps)) * self.gamma + self.beta


class GroupNormModule(nn.Module):
    """Standard GroupNorm."""

    def __init__(self, num_groups: int = 2, num_channels: int = 16) -> None:
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gn(x)


class CombinedNormVariants(nn.Module):
    """Chains multiple normalizations in sequence."""

    def __init__(self, d_model: int = 32) -> None:
        super().__init__()
        self.rms = RMSNormL2Module(d_model)
        self.ln = LayerNormExpandedModule(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.rms(x)
        return self.ln(h)


from generators.base import export_model as unified_export_model


def generate_all() -> None:
    models_dir = Path(".helper/models")
    dummy_3d = torch.randn(2, 8, 32)
    dummy_4d = torch.randn(2, 16, 8, 8)

    # 1. Combined Norm model
    unified_export_model(
        CombinedNormVariants(32),
        (dummy_3d,),
        ["input"],
        ["output"],
        "norm_variants",
        models_dir,
        opset_versions=[14, 17, 18, 20],
        expected_target_node="LayerNormalization",
        emit_json_descriptor=False,
    )

    # 2. Individual RMSNorm L2
    unified_export_model(
        RMSNormL2Module(32),
        (dummy_3d,),
        ["input"],
        ["output"],
        "norm_rmsnorm_l2",
        models_dir,
        opset_versions=[14, 17, 18, 20],
        expected_target_node="RMSNorm",
        min_expected_reduction=0.5,
    )

    # 3. Individual LayerNorm
    unified_export_model(
        LayerNormExpandedModule(32),
        (dummy_3d,),
        ["input"],
        ["output"],
        "norm_layernorm",
        models_dir,
        opset_versions=[14, 17, 18, 20],
        expected_target_node="LayerNormalization",
        min_expected_reduction=0.5,
    )

    # 4. GroupNorm
    unified_export_model(
        GroupNormModule(2, 16),
        (dummy_4d,),
        ["input"],
        ["output"],
        "norm_groupnorm",
        models_dir,
        opset_versions=[14, 17, 18, 20],
        expected_target_node="GroupNorm",
        min_expected_reduction=0.0,
    )


if __name__ == "__main__":
    generate_all()
