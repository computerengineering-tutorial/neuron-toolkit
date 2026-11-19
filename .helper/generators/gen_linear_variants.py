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
from pathlib import Path

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


def generate_all() -> None:
    models_dir = Path(".helper/models")
    tests_dir = Path("tests/rules_json_tests")
    models_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)

    # 1. Combined model
    comb = CombinedLinearVariants()
    comb.eval()
    dummy_3d = torch.randn(2, 8, 16)
    comb_path = models_dir / "linear_variants.onnx"
    torch.onnx.export(
        comb,
        dummy_3d,
        str(comb_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 1: "time"},
            "output": {0: "batch", 1: "time"},
        },
        opset_version=14,
        dynamo=False,
    )
    print(f"Exported: {comb_path}")

    # 2. Gemm variant
    gemm_mod = LinearGemm(16, 32)
    gemm_mod.eval()
    dummy_2d = torch.randn(4, 16)
    gemm_path = models_dir / "linear_gemm.onnx"
    torch.onnx.export(
        gemm_mod,
        dummy_2d,
        str(gemm_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
        dynamo=False,
    )
    print(f"Exported: {gemm_path}")

    # 3. Conv 1x1 variant
    conv_mod = Conv1x1Linear(16, 32)
    conv_mod.eval()
    dummy_4d = torch.randn(2, 16, 8, 8)
    conv_path = models_dir / "linear_conv1x1.onnx"
    torch.onnx.export(
        conv_mod,
        dummy_4d,
        str(conv_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
        dynamo=False,
    )
    print(f"Exported: {conv_path}")

    # Emit JSON test descriptors
    test_specs = {
        "linear_matmul_add_fusion": {
            "test_id": "linear_matmul_add_fusion_test",
            "model_path": str(comb_path),
            "input_shapes": {"input": [2, 8, 16]},
            "expected_target_node": "Linear",
            "min_expected_reduction": 0.3,
            "rtol": 1e-4,
            "atol": 1e-4,
        },
        "linear_gemm_canonicalization": {
            "test_id": "linear_gemm_canonicalization_test",
            "model_path": str(gemm_path),
            "input_shapes": {"input": [4, 16]},
            "expected_target_node": "Linear",
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
