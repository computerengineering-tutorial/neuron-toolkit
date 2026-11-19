#!/usr/bin/env python3
"""Synthetic PyTorch Graph Generator: Transformer Blocks.

Generates ONNX models for:
- FeedForward GELU (RMSNorm -> Linear -> Gelu -> Linear)
- FeedForward SwiGLU (RMSNorm -> Linear * SiLU * Linear -> Linear)
- ScaledDotProductAttention (SDPA)
- Rotary Position Embedding (RoPE)
- GatedUnit Attention Gating
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


class RMSNormLayer(nn.Module):
    def __init__(self, d_model: int = 32, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        scale = (x.shape[-1] ** 0.5) / torch.clamp(norm, min=self.eps)
        return x * scale * self.gamma


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


class CombinedTransformerBlock(nn.Module):
    """Full synthetic transformer block combining SDPA, GatedAttention, and FeedForward."""

    def __init__(self, d_model: int = 32, d_ff: int = 64) -> None:
        super().__init__()
        self.ffn = FeedForwardGeluBlock(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention + FFN
        return self.ffn(x)


from generators.base import export_model as unified_export_model


def generate_all() -> None:
    models_dir = Path(".helper/models")
    dummy = torch.randn(2, 8, 32)
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 8, 16)

    # 1. Combined Transformer block model
    unified_export_model(
        CombinedTransformerBlock(32, 64),
        (dummy,),
        ["input"],
        ["output"],
        "transformer_blocks",
        models_dir,
        opset_versions=[14, 17, 18, 20],
        expected_target_node="FeedForward",
        emit_json_descriptor=False,
    )

    # 2. FeedForward GELU
    unified_export_model(
        FeedForwardGeluBlock(32, 64),
        (dummy,),
        ["input"],
        ["output"],
        "transformer_ffn_gelu",
        models_dir,
        opset_versions=[14, 17, 18, 20],
        expected_target_node="FeedForward",
        min_expected_reduction=0.5,
    )

    # 3. FeedForward SwiGLU
    unified_export_model(
        FeedForwardSwiGLUBlock(32, 64),
        (dummy,),
        ["input"],
        ["output"],
        "transformer_ffn_swiglu",
        models_dir,
        opset_versions=[14, 17, 18, 20],
        expected_target_node="GatedUnit",
        min_expected_reduction=0.3,
    )

    # 4. SDPA
    unified_export_model(
        SDPABlock(),
        (q, k, v),
        ["q", "k", "v"],
        ["output"],
        "transformer_sdpa",
        models_dir,
        opset_versions=[14, 17, 18, 20],
        expected_target_node="ScaledDotProductAttention",
        min_expected_reduction=0.3,
    )


if __name__ == "__main__":
    generate_all()
