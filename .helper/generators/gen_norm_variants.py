#!/usr/bin/env python3
"""Synthetic PyTorch Graph Generator: Normalization Variants.

Generates ONNX models for:
- RMSNorm (ReduceL2 & MeanSquare formulations)
- LayerNorm (Mean/Var expansion)
- GroupNorm (InstanceNorm 1D adapter formulation)
"""

from __future__ import annotations

import json
from pathlib import Path

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


def generate_all() -> None:
    models_dir = Path(".helper/models")
    tests_dir = Path("tests/rules_json_tests")
    models_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)

    dummy_3d = torch.randn(2, 8, 32)

    # 1. Combined Norm model
    comb = CombinedNormVariants(32)
    comb.eval()
    comb_path = models_dir / "norm_variants.onnx"
    torch.onnx.export(
        comb,
        dummy_3d,
        str(comb_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
        dynamo=False,
    )
    print(f"Exported: {comb_path}")

    # 2. Individual RMSNorm L2
    rms_mod = RMSNormL2Module(32)
    rms_mod.eval()
    rms_path = models_dir / "norm_rmsnorm_l2.onnx"
    torch.onnx.export(
        rms_mod,
        dummy_3d,
        str(rms_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
        dynamo=False,
    )
    print(f"Exported: {rms_path}")

    # 3. Individual LayerNorm
    ln_mod = LayerNormExpandedModule(32)
    ln_mod.eval()
    ln_path = models_dir / "norm_layernorm.onnx"
    torch.onnx.export(
        ln_mod,
        dummy_3d,
        str(ln_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
        dynamo=False,
    )
    print(f"Exported: {ln_path}")

    # 4. GroupNorm
    dummy_4d = torch.randn(2, 16, 8, 8)
    gn_mod = GroupNormModule(2, 16)
    gn_mod.eval()
    gn_path = models_dir / "norm_groupnorm.onnx"
    torch.onnx.export(
        gn_mod,
        dummy_4d,
        str(gn_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
        dynamo=False,
    )
    print(f"Exported: {gn_path}")

    # Emit JSON test descriptors
    test_specs = {
        "rmsnorm_fusion": {
            "test_id": "rmsnorm_fusion_test",
            "model_path": str(rms_path),
            "input_shapes": {"input": [2, 8, 32]},
            "expected_target_node": "RMSNorm",
            "min_expected_reduction": 0.5,
            "rtol": 1e-4,
            "atol": 1e-4,
        },
        "layernorm_fusion": {
            "test_id": "layernorm_fusion_test",
            "model_path": str(ln_path),
            "input_shapes": {"input": [2, 8, 32]},
            "expected_target_node": "LayerNormalization",
            "min_expected_reduction": 0.5,
            "rtol": 1e-4,
            "atol": 1e-4,
        },
        "groupnorm_fusion": {
            "test_id": "groupnorm_fusion_test",
            "model_path": str(gn_path),
            "input_shapes": {"input": [2, 16, 8, 8]},
            "expected_target_node": "GroupNorm",
            "min_expected_reduction": 0.0,
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
