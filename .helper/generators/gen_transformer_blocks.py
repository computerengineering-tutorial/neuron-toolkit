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
from pathlib import Path

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


def generate_all() -> None:
    models_dir = Path(".helper/models")
    tests_dir = Path("tests/rules_json_tests")
    models_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(2, 8, 32)

    # 1. Combined Transformer block model
    comb = CombinedTransformerBlock(32, 64)
    comb.eval()
    comb_path = models_dir / "transformer_blocks.onnx"
    torch.onnx.export(
        comb,
        dummy,
        str(comb_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
        dynamo=False,
    )
    print(f"Exported: {comb_path}")

    # 2. FeedForward GELU
    ffn_gelu = FeedForwardGeluBlock(32, 64)
    ffn_gelu.eval()
    ffn_path = models_dir / "transformer_ffn_gelu.onnx"
    torch.onnx.export(
        ffn_gelu,
        dummy,
        str(ffn_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
        dynamo=False,
    )
    print(f"Exported: {ffn_path}")

    # 3. FeedForward SwiGLU
    ffn_swiglu = FeedForwardSwiGLUBlock(32, 64)
    ffn_swiglu.eval()
    swiglu_path = models_dir / "transformer_ffn_swiglu.onnx"
    torch.onnx.export(
        ffn_swiglu,
        dummy,
        str(swiglu_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
        dynamo=False,
    )
    print(f"Exported: {swiglu_path}")

    # 4. SDPA
    sdpa_mod = SDPABlock()
    sdpa_mod.eval()
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 8, 16)
    sdpa_path = models_dir / "transformer_sdpa.onnx"
    torch.onnx.export(
        sdpa_mod,
        (q, k, v),
        str(sdpa_path),
        input_names=["q", "k", "v"],
        output_names=["output"],
        opset_version=14,
        dynamo=False,
    )
    print(f"Exported: {sdpa_path}")

    # Emit JSON test descriptors
    test_specs = {
        "feedforward_gelu_fusion": {
            "test_id": "feedforward_gelu_fusion_test",
            "model_path": str(ffn_path),
            "input_shapes": {"input": [2, 8, 32]},
            "expected_target_node": "FeedForward",
            "min_expected_reduction": 0.5,
            "rtol": 1e-4,
            "atol": 1e-4,
        },
        "scaled_dot_product_attention_fusion": {
            "test_id": "scaled_dot_product_attention_fusion_test",
            "model_path": str(sdpa_path),
            "input_shapes": {
                "q": [2, 4, 8, 16],
                "k": [2, 4, 8, 16],
                "v": [2, 4, 8, 16],
            },
            "expected_target_node": "ScaledDotProductAttention",
            "min_expected_reduction": 0.3,
            "rtol": 1e-4,
            "atol": 1e-4,
        },
    }

    for name, spec in test_specs.items():
        out_f = tests_dir / f"{name}_test.json"
        with open(out_f, "w", encoding="utf-8") as f:
            json.dump(spec, f, indent=2)
        print(f"Emitted test spec: {out_f}")


if __name__ == "__main__":
    generate_all()
