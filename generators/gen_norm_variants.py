#!/usr/bin/env python3
"""Synthetic PyTorch Graph Generator: Normalization Variants."""

from __future__ import annotations

import torch
from torch import nn

from generators.decorators import register_architecture
from generators.registry import ModelRegistry


@register_architecture(
    name="norm_rmsnorm_l2",
    category="norm_variants",
    shapes=[(2, 8, 32)],
    input_names=["input"],
    output_names=["output"],
    expected_target_node="RMSNorm",
    min_expected_reduction=0.5,
    init_args=(32,),
)
class RMSNormL2Module(nn.Module):
    """Explicit ReduceL2-based RMSNorm: x * (sqrt(D) / clamp(norm(x), min=eps)) * gamma."""

    def __init__(self, d_model: int = 32, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


@register_architecture(
    name="norm_layernorm",
    category="norm_variants",
    shapes=[(2, 8, 32)],
    input_names=["input"],
    output_names=["output"],
    expected_target_node="LayerNormalization",
    min_expected_reduction=0.5,
    init_args=(32,),
)
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


@register_architecture(
    name="norm_groupnorm",
    category="norm_variants",
    shapes=[(2, 16, 8, 8)],
    input_names=["input"],
    output_names=["output"],
    expected_target_node="GroupNorm",
    min_expected_reduction=0.0,
    init_args=(2, 16),
)
class GroupNormModule(nn.Module):
    """Standard GroupNorm."""

    def __init__(self, num_groups: int = 2, num_channels: int = 16) -> None:
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gn(x)


@register_architecture(
    name="norm_variants",
    category="norm_variants",
    shapes=[(2, 8, 32)],
    input_names=["input"],
    output_names=["output"],
    expected_target_node="LayerNormalization",
    emit_json_descriptor=False,
    init_args=(32,),
)
class CombinedNormVariants(nn.Module):
    """Chains multiple normalizations in sequence."""

    def __init__(self, d_model: int = 32) -> None:
        super().__init__()
        self.rms = RMSNormL2Module(d_model)
        self.ln = LayerNormExpandedModule(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.rms(x)
        return self.ln(h)


def generate_all() -> None:
    ModelRegistry.export_all(category="norm_variants")


if __name__ == "__main__":
    generate_all()
