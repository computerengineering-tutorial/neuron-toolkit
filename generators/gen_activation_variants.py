#!/usr/bin/env python3
"""Synthetic PyTorch Graph Generator: Activation Variants.

Generates ONNX models for:
- Standard ERF GELU expansion (Div -> Erf -> Add -> Mul -> Mul)
- Tanh GELU approximation
- SiLU (x * Sigmoid(x))
- HardSwish (x * Clip(x+3, 0, 6) / 6)
- Mish (x * Tanh(Softplus(x)))
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F
from torch import nn

from generators.base import export_model as unified_export_model


class ErfGeluModule(nn.Module):
    """Explicit ERF-based GELU formulation without native Gelu op."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 0.5 * x * (1 + erf(x / sqrt(2)))
        return 0.5 * x * (1.0 + torch.erf(x / 1.41421356))


class TanhGeluModule(nn.Module):
    """Explicit Tanh approximation GELU formulation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            0.5 * x * (1.0 + torch.tanh(0.79788456 * (x + 0.044715 * torch.pow(x, 3))))
        )


class SiluModule(nn.Module):
    """x * Sigmoid(x)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class HardSwishModule(nn.Module):
    """x * Clip(x + 3, 0, 6) / 6."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.clamp(x + 3.0, 0.0, 6.0) / 6.0


class MishModule(nn.Module):
    """x * Tanh(Softplus(x))."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class CombinedActivations(nn.Module):
    """Chains multiple activation variants for end-to-end multi-op canonicalization testing."""

    def __init__(self) -> None:
        super().__init__()
        self.erf_gelu = ErfGeluModule()
        self.silu = SiluModule()
        self.hswish = HardSwishModule()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.erf_gelu(x)
        h2 = self.silu(h1)
        return self.hswish(h2)



def generate_all() -> None:
    models_dir = Path(".helper/models")
    dummy = torch.randn(2, 16, 32)

    # 1. Combined Activations model
    unified_export_model(
        CombinedActivations(),
        (dummy,),
        ["input"],
        ["output"],
        "activation_variants",
        models_dir,
        opset_versions=[14, 17, 18, 20],
        expected_target_node="Gelu",
        emit_json_descriptor=False,
    )

    # 2. Individual ERF GELU
    unified_export_model(
        ErfGeluModule(),
        (dummy,),
        ["input"],
        ["output"],
        "activation_erf_gelu",
        models_dir,
        opset_versions=[14, 17, 18, 20],
        expected_target_node="Gelu",
        min_expected_reduction=0.5,
    )

    # 3. Individual SiLU
    unified_export_model(
        SiluModule(),
        (dummy,),
        ["input"],
        ["output"],
        "activation_silu",
        models_dir,
        opset_versions=[14, 17, 18, 20],
        expected_target_node="SiLU",
        min_expected_reduction=0.5,
    )

    # 4. Individual HardSwish
    unified_export_model(
        HardSwishModule(),
        (dummy,),
        ["input"],
        ["output"],
        "activation_hardswish",
        models_dir,
        opset_versions=[14, 17, 18, 20],
        expected_target_node="HardSwish",
        min_expected_reduction=0.5,
    )


if __name__ == "__main__":
    generate_all()
