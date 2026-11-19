#!/usr/bin/env python3
"""Comprehensive Vanilla MLP, Feedforward & Gated MLP Architecture Family Generator.

Generates self-contained PyTorch models and multi-opset ONNX exports for:
1. DeepMLP_Norm_Act (Multi-layer deep perceptron with LayerNorm + GELU)
2. ResMLP_Block (Residual MLP with cross-channel linear + residual skip)
3. MLPMixer_Block (Token-mixing MLP + Channel-mixing MLP with Transpose)
4. gMLP_Block (Gated MLP with Spatial Gating Unit / SGU)
5. HighwayNetwork_Block (Information highway with Transform & Carry gates)
6. SwiGLU_MLP_Block (SwiGLU gated linear unit: (W1 x * silu(W3 x)) W2)
7. GeGLU_MLP_Block (GeGLU gated linear unit: (W1 x * gelu(W3 x)) W2)
8. ReGLU_MLP_Block (ReGLU gated linear unit: (W1 x * relu(W3 x)) W2)

Exports across ONNX Opsets 14, 17, 18, 20.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import onnx
import torch
import torch.nn.functional as F
from torch import nn

warnings.filterwarnings("ignore")

OUT_DIR = Path(".helper/models/mlp_vanilla")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. Deep MLP with LayerNorm & GELU
# =============================================================================
class DeepMLP_Norm_Act(nn.Module):
    def __init__(
        self, in_features: int = 64, hidden_features: int = 256, out_features: int = 64
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.norm1 = nn.LayerNorm(hidden_features)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.norm2 = nn.LayerNorm(hidden_features)
        self.act2 = nn.GELU()
        self.fc3 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act1(self.norm1(self.fc1(x)))
        out = self.act2(self.norm2(self.fc2(out)))
        return self.fc3(out)


# =============================================================================
# 2. ResMLP Block
# =============================================================================
class ResMLP_Block(nn.Module):
    def __init__(self, num_tokens: int = 16, dim: int = 64, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.cross_token = nn.Linear(num_tokens, num_tokens)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_channel1 = nn.Linear(dim, dim * mlp_ratio)
        self.act = nn.GELU()
        self.cross_channel2 = nn.Linear(dim * mlp_ratio, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cross-token mixing
        norm_x = self.norm1(x)
        token_out = self.cross_token(norm_x.transpose(1, 2)).transpose(1, 2)
        x = x + token_out

        # Cross-channel mixing
        norm_x2 = self.norm2(x)
        channel_out = self.cross_channel2(self.act(self.cross_channel1(norm_x2)))
        return x + channel_out


# =============================================================================
# 3. MLP-Mixer Block
# =============================================================================
class MLPMixer_Block(nn.Module):
    def __init__(
        self,
        num_patches: int = 16,
        num_channels: int = 64,
        tokens_mlp_dim: int = 32,
        channels_mlp_dim: int = 256,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(num_channels)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_patches, tokens_mlp_dim),
            nn.GELU(),
            nn.Linear(tokens_mlp_dim, num_patches),
        )
        self.norm2 = nn.LayerNorm(num_channels)
        self.channel_mlp = nn.Sequential(
            nn.Linear(num_channels, channels_mlp_dim),
            nn.GELU(),
            nn.Linear(channels_mlp_dim, num_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Token mixing: (B, S, C) -> transpose to (B, C, S) -> Linear -> transpose back
        norm_x = self.norm1(x).transpose(1, 2)
        token_mixed = self.token_mlp(norm_x).transpose(1, 2)
        x = x + token_mixed

        # Channel mixing: (B, S, C) -> Linear
        channel_mixed = self.channel_mlp(self.norm2(x))
        return x + channel_mixed


# =============================================================================
# 4. gMLP (Gated MLP with Spatial Gating Unit)
# =============================================================================
class SpatialGatingUnit(nn.Module):
    def __init__(self, dim: int = 64, seq_len: int = 16) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim // 2)
        self.proj = nn.Linear(seq_len, seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = torch.chunk(x, 2, dim=-1)
        v = self.norm(v)
        v = self.proj(v.transpose(1, 2)).transpose(1, 2)
        return u * v


class gMLP_Block(nn.Module):
    def __init__(self, dim: int = 64, d_ffn: int = 128, seq_len: int = 16) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.channel_in = nn.Linear(dim, d_ffn)
        self.act = nn.GELU()
        self.sgu = SpatialGatingUnit(dim=d_ffn, seq_len=seq_len)
        self.channel_out = nn.Linear(d_ffn // 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm(x)
        out = self.act(self.channel_in(out))
        out = self.sgu(out)
        out = self.channel_out(out)
        return residual + out


# =============================================================================
# 5. Highway Network Block
# =============================================================================
class HighwayNetwork_Block(nn.Module):
    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.transform_gate = nn.Linear(dim, dim)
        self.nonlinear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = torch.sigmoid(self.transform_gate(x))
        h = F.relu(self.nonlinear(x))
        return h * t + x * (1.0 - t)


# =============================================================================
# 6. GLU Activation Variants: SwiGLU, GeGLU, ReGLU
# =============================================================================
class SwiGLU_MLP_Block(nn.Module):
    def __init__(self, dim: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.w1(x) * F.silu(self.w3(x)))


class GeGLU_MLP_Block(nn.Module):
    def __init__(self, dim: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.w1(x) * F.gelu(self.w3(x)))


class ReGLU_MLP_Block(nn.Module):
    def __init__(self, dim: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.w1(x) * F.relu(self.w3(x)))


# =============================================================================
# Multi-Opset ONNX Export Helper
# =============================================================================
def export_model(
    model: nn.Module,
    dummy_inputs: tuple[torch.Tensor, ...],
    input_names: list[str],
    output_names: list[str],
    base_name: str,
    opset_versions: list[int] = (14, 17, 18, 20),
) -> list[str]:
    model.eval()
    exported = []
    for opset in opset_versions:
        out_path = OUT_DIR / f"{base_name}_opset{opset}.onnx"
        try:
            torch.onnx.export(
                model,
                dummy_inputs,
                str(out_path),
                input_names=input_names,
                output_names=output_names,
                opset_version=opset,
                do_constant_folding=True,
            )
            m_proto = onnx.load(str(out_path))
            onnx.checker.check_model(m_proto)
            size_kb = out_path.stat().st_size / 1024.0
            print(
                f"  [OK] Exported {out_path.name:<45} (opset={opset}, nodes={len(m_proto.graph.node):<3}, {size_kb:.1f} KB)"
            )
            exported.append(str(out_path))
        except Exception as e:
            print(f"  [FAIL] Failed to export {out_path.name} (opset={opset}): {e}")
    return exported


def main() -> None:
    print("==================================================================")
    print("  GENERATING VANILLA MLP & FEEDFORWARD ONNX SUITE")
    print("==================================================================")

    dummy_x = torch.randn(1, 16, 64)

    registry = [
        ("deep_mlp_norm_act", DeepMLP_Norm_Act(64, 256, 64)),
        ("resmlp_block", ResMLP_Block(num_tokens=16, dim=64)),
        ("mlpmixer_block", MLPMixer_Block(num_patches=16, num_channels=64)),
        ("gmlp_block", gMLP_Block(dim=64, d_ffn=128, seq_len=16)),
        ("highway_block", HighwayNetwork_Block(dim=64)),
        ("swiglu_mlp_block", SwiGLU_MLP_Block(dim=64, hidden_dim=128)),
        ("geglu_mlp_block", GeGLU_MLP_Block(dim=64, hidden_dim=128)),
        ("reglu_mlp_block", ReGLU_MLP_Block(dim=64, hidden_dim=128)),
    ]

    for name, module in registry:
        print(f"\nModel: {name}")
        export_model(
            module, (dummy_x,), ["x"], ["out"], name, opset_versions=[14, 17, 18, 20]
        )

    print("\n==================================================================")
    print(f"Vanilla MLP & Feedforward ONNX suite complete in: {OUT_DIR.resolve()}")
    print("==================================================================")


if __name__ == "__main__":
    main()
