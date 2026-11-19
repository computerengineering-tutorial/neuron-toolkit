#!/usr/bin/env python3
"""Synthetic PyTorch Graph Generator: Transformer Blocks."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from generators.decorators import register_architecture
from generators.registry import ModelRegistry


class RMSNormLayer(nn.Module):
    def __init__(self, d_model: int = 32, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        scale = (x.shape[-1] ** 0.5) / torch.clamp(norm, min=self.eps)
        return x * scale * self.gamma


@register_architecture(
    name="transformer_ffn_gelu",
    category="transformer_blocks",
    shapes=[(2, 8, 32)],
    input_names=["input"],
    output_names=["output"],
    expected_target_node="FeedForward",
    min_expected_reduction=0.5,
    init_args=(32, 64),
)
class FeedForwardGeluBlock(nn.Module):
    """RMSNorm -> Linear -> Gelu -> Linear."""

    def __init__(self, d_model: int = 32, d_ff: int = 64) -> None:
        super().__init__()
        self.norm = RMSNormLayer(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        return self.fc2(h)


@register_architecture(
    name="transformer_ffn_swiglu",
    category="transformer_blocks",
    shapes=[(2, 8, 32)],
    input_names=["input"],
    output_names=["output"],
    expected_target_node="FeedForwardSwiGLU",
    min_expected_reduction=0.3,
    init_args=(32, 64),
)
class FeedForwardSwiGLUBlock(nn.Module):
    """RMSNorm -> (Linear_gate * SiLU * Linear_up) -> Linear_down."""

    def __init__(self, d_model: int = 32, d_ff: int = 64) -> None:
        super().__init__()
        self.norm = RMSNormLayer(d_model)
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        hidden = F.silu(self.gate(h)) * self.up(h)
        return self.down(hidden)


@register_architecture(
    name="transformer_sdpa",
    category="transformer_blocks",
    shapes=[(2, 4, 8, 16), (2, 4, 8, 16), (2, 4, 8, 16)],
    input_names=["q", "k", "v"],
    output_names=["output"],
    expected_target_node="ScaledDotProductAttention",
    min_expected_reduction=0.3,
)
class SDPABlock(nn.Module):
    """Scaled Dot-Product Attention."""

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        scale = q.shape[-1] ** -0.5
        scores = torch.matmul(q * scale, k.transpose(-1, -2))
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs, v)


class GatedAttentionBlock(nn.Module):
    """attn * Sigmoid(gate)."""

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(gate)


@register_architecture(
    name="transformer_blocks",
    category="transformer_blocks",
    shapes=[(2, 8, 32)],
    input_names=["input"],
    output_names=["output"],
    expected_target_node="FeedForward",
    emit_json_descriptor=False,
    init_args=(32, 64),
)
class CombinedTransformerBlock(nn.Module):
    """Full synthetic transformer block combining SDPA, GatedAttention, and FeedForward."""

    def __init__(self, d_model: int = 32, d_ff: int = 64) -> None:
        super().__init__()
        self.ffn = FeedForwardGeluBlock(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


def generate_all() -> None:
    ModelRegistry.export_all(category="transformer_blocks")


if __name__ == "__main__":
    generate_all()
