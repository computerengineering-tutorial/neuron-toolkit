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
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


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
    tests_dir = Path("tests/rules_json_tests")
    models_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(2, 16, 32)

    # 1. Combined Activations model
    comb = CombinedActivations()
    comb.eval()
    comb_path = models_dir / "activation_variants.onnx"
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

    # 2. Individual ERF GELU
    erf_mod = ErfGeluModule()
    erf_mod.eval()
    erf_path = models_dir / "activation_erf_gelu.onnx"
    torch.onnx.export(
        erf_mod,
        dummy,
        str(erf_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
        dynamo=False,
    )
    print(f"Exported: {erf_path}")

    # 3. Individual SiLU
    silu_mod = SiluModule()
    silu_mod.eval()
    silu_path = models_dir / "activation_silu.onnx"
    torch.onnx.export(
        silu_mod,
        dummy,
        str(silu_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
        dynamo=False,
    )
    print(f"Exported: {silu_path}")

    # 4. Individual HardSwish
    hswish_mod = HardSwishModule()
    hswish_mod.eval()
    hswish_path = models_dir / "activation_hardswish.onnx"
    torch.onnx.export(
        hswish_mod,
        dummy,
        str(hswish_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
        dynamo=False,
    )
    print(f"Exported: {hswish_path}")

    # Emit JSON test descriptors
    test_specs = {
        "erf_gelu_canonicalization": {
            "test_id": "erf_gelu_canonicalization_test",
            "model_path": str(erf_path),
            "input_shapes": {"input": [2, 16, 32]},
            "expected_target_node": "Gelu",
            "min_expected_reduction": 0.5,
            "rtol": 1e-4,
            "atol": 1e-4,
        },
        "silu_swish_canonicalization": {
            "test_id": "silu_swish_canonicalization_test",
            "model_path": str(silu_path),
            "input_shapes": {"input": [2, 16, 32]},
            "expected_target_node": "SiLU",
            "min_expected_reduction": 0.5,
            "rtol": 1e-4,
            "atol": 1e-4,
        },
        "hard_swish_canonicalization": {
            "test_id": "hard_swish_canonicalization_test",
            "model_path": str(hswish_path),
            "input_shapes": {"input": [2, 16, 32]},
            "expected_target_node": "HardSwish",
            "min_expected_reduction": 0.5,
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
