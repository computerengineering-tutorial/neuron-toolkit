#!/usr/bin/env python3
"""Synthetic PyTorch Graph Generator: Activation Variants."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from generators.decorators import register_architecture
from generators.registry import ModelRegistry


@register_architecture(
    name="activation_erf_gelu",
    category="activation_variants",
    shapes=[(2, 16, 32)],
    input_names=["input"],
    output_names=["output"],
    expected_target_node="Gelu",
    min_expected_reduction=0.5,
)
class ErfGeluModule(nn.Module):
    """Explicit ERF-based GELU formulation without native Gelu op."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.erf(x / 1.41421356))


class TanhGeluModule(nn.Module):
    """Explicit Tanh approximation GELU formulation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            0.5 * x * (1.0 + torch.tanh(0.79788456 * (x + 0.044715 * torch.pow(x, 3))))
        )


@register_architecture(
    name="activation_silu",
    category="activation_variants",
    shapes=[(2, 16, 32)],
    input_names=["input"],
    output_names=["output"],
    expected_target_node="SiLU",
    min_expected_reduction=0.5,
)
class SiluModule(nn.Module):
    """x * Sigmoid(x)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


@register_architecture(
    name="activation_hardswish",
    category="activation_variants",
    shapes=[(2, 16, 32)],
    input_names=["input"],
    output_names=["output"],
    expected_target_node="HardSwish",
    min_expected_reduction=0.5,
)
class HardSwishModule(nn.Module):
    """x * Clip(x + 3, 0, 6) / 6."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.clamp(x + 3.0, 0.0, 6.0) / 6.0


class MishModule(nn.Module):
    """x * Tanh(Softplus(x))."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


@register_architecture(
    name="activation_variants",
    category="activation_variants",
    shapes=[(2, 16, 32)],
    input_names=["input"],
    output_names=["output"],
    expected_target_node="Gelu",
    emit_json_descriptor=False,
)
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
    ModelRegistry.export_all(category="activation_variants")


if __name__ == "__main__":
    generate_all()
