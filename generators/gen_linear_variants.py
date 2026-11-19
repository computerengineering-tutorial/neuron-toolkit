#!/usr/bin/env python3
"""Synthetic PyTorch Graph Generator: Linear Variants."""

from __future__ import annotations

import torch
from torch import nn

from generators.decorators import register_architecture
from generators.registry import ModelRegistry


class LinearMatMulAdd(nn.Module):
    """X @ W + B."""

    def __init__(self, in_features: int = 16, out_features: int = 32) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn(in_features, out_features))
        self.b = nn.Parameter(torch.randn(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.w) + self.b


@register_architecture(
    name="linear_gemm",
    category="linear_variants",
    shapes=[(4, 16)],
    input_names=["input"],
    output_names=["output"],
    expected_target_node="Linear",
    init_args=(16, 32),
)
class LinearGemm(nn.Module):
    """Linear projection exported as Gemm."""

    def __init__(self, in_features: int = 16, out_features: int = 32) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class LinearEinsum(nn.Module):
    """torch.einsum('btd,df->btf')."""

    def __init__(self, in_features: int = 16, out_features: int = 32) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn(in_features, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("btd,df->btf", x, self.w)


@register_architecture(
    name="linear_conv1x1",
    category="linear_variants",
    shapes=[(2, 16, 8, 8)],
    input_names=["input"],
    output_names=["output"],
    expected_target_node="Conv",
    emit_json_descriptor=False,
    init_args=(16, 32),
)
class Conv1x1Linear(nn.Module):
    """1x1 Conv pointwise linear projection."""

    def __init__(self, in_channels: int = 16, out_channels: int = 32) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


@register_architecture(
    name="linear_variants",
    category="linear_variants",
    shapes=[(2, 8, 16)],
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch", 1: "time"}, "output": {0: "batch", 1: "time"}},
    expected_target_node="Linear",
    min_expected_reduction=0.3,
)
class CombinedLinearVariants(nn.Module):
    """Combines all linear variants in sequence."""

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = LinearMatMulAdd(16, 24)
        self.layer2 = LinearEinsum(24, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.layer1(x)
        return self.layer2(h1)


def generate_all() -> None:
    ModelRegistry.export_all(category="linear_variants")


if __name__ == "__main__":
    generate_all()
