"""Generator for Diffusion Transformer (DiT) & Generative U-Net architecture blocks.

Exports models to ONNX across opset versions 14, 17, 18, and 20.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import onnx
import torch
import torch.nn.functional as F
from torch import nn

OUT_DIR = Path(".helper/models/diffusion_dit")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Modulate token features via adaptive shift and scale."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# =============================================================================
# 1. Diffusion Transformer (DiT) Block with AdaLN-Zero
# =============================================================================
class DiT_AdaLNZero_Block(nn.Module):
    """Diffusion Transformer Block with Adaptive LayerNorm Zero Conditioning."""

    def __init__(self, d_model: int = 64, num_heads: int = 4, d_cond: int = 64) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)

        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

        # AdaLN modulation MLP: outputs 6 values (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_cond, 6 * d_model, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D), c: (B, d_cond) conditioning vector (timestep + label)
        B, S, D = x.shape

        # Compute adaptive modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )

        # 1. Self-Attention with AdaLN modulation
        normed = modulate(self.norm1(x), shift_msa, scale_msa)
        qkv = (
            self.attn_qkv(normed)
            .view(B, S, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(q / math.sqrt(self.head_dim), k.transpose(-1, -2))
        probs = torch.softmax(scores, dim=-1)
        attn = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, S, D)
        attn = self.attn_out(attn)

        # Residual update with zero-initialized adaptive gate
        x = x + gate_msa.unsqueeze(1) * attn

        # 2. MLP with AdaLN modulation
        normed2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(normed2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


# =============================================================================
# 2. U-Net ResBlock with Timestep Conditioning Injection
# =============================================================================
class UNet_ResBlock_WithTimestep(nn.Module):
    """Classic Diffusion U-Net Residual block with timestep embedding projection."""

    def __init__(
        self, in_channels: int = 32, out_channels: int = 32, d_time: int = 64
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=4, num_channels=out_channels)

        # Timestep linear projection to channels
        self.time_emb_proj = nn.Linear(d_time, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=4, num_channels=out_channels)

        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W), t_emb: (B, d_time)
        residual = self.skip(x)

        h = F.silu(self.norm1(self.conv1(x)))
        # Inject timestep embedding as channel broadcast
        time_proj = self.time_emb_proj(F.silu(t_emb))
        h = h + time_proj.unsqueeze(-1).unsqueeze(-1)

        h = F.silu(self.norm2(self.conv2(h)))
        return residual + h


# =============================================================================
# 3. Cross-Attention Downsample Block
# =============================================================================
class CrossAttention_DownBlock(nn.Module):
    """U-Net Spatial downsample with Text/Prompt Cross-Attention."""

    def __init__(
        self, channels: int = 32, d_context: int = 64, num_heads: int = 4
    ) -> None:
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(d_context, channels, bias=False)
        self.to_v = nn.Linear(d_context, channels, bias=False)
        self.to_out = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W), context: (B, S_ctx, d_context)
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        q = (
            self.to_q(x_flat)
            .view(B, H * W, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.to_k(context)
            .view(B, context.shape[1], self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.to_v(context)
            .view(B, context.shape[1], self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        scores = torch.matmul(q / math.sqrt(self.head_dim), k.transpose(-1, -2))
        probs = torch.softmax(scores, dim=-1)
        attn = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, H * W, C)
        out = self.to_out(attn)

        out_2d = out.view(B, H, W, C).permute(0, 3, 1, 2)
        return x + out_2d


# =============================================================================
# Exporter
# =============================================================================
def export_diffusion_zoo() -> list[dict[str, Any]]:
    models: dict[str, tuple[nn.Module, tuple[torch.Tensor, ...], list[str]]] = {
        "dit_adaln_zero": (
            DiT_AdaLNZero_Block(64, 4, 64),
            (torch.randn(2, 16, 64), torch.randn(2, 64)),
            ["x", "conditioning"],
        ),
        "unet_res_timestep": (
            UNet_ResBlock_WithTimestep(32, 32, 64),
            (torch.randn(2, 32, 16, 16), torch.randn(2, 64)),
            ["x", "t_emb"],
        ),
        "cross_attention_block": (
            CrossAttention_DownBlock(32, 64, 4),
            (torch.randn(2, 32, 16, 16), torch.randn(2, 8, 64)),
            ["x", "context"],
        ),
    }

    opsets = [14, 17, 18, 20]
    catalog = []

    for name, (mod, dummy_inputs, input_names) in models.items():
        mod.eval()
        for opset in opsets:
            out_path = OUT_DIR / f"{name}_opset{opset}.onnx"
            try:
                torch.onnx.export(
                    mod,
                    dummy_inputs,
                    str(out_path),
                    input_names=input_names,
                    output_names=["out"],
                    opset_version=opset,
                    do_constant_folding=True,
                )
                m = onnx.load(str(out_path))
                m.ir_version = min(m.ir_version, 10)
                onnx.save(m, str(out_path))
                catalog.append(
                    {
                        "name": name,
                        "opset": opset,
                        "path": str(out_path),
                        "nodes": len(m.graph.node),
                    }
                )
                print(
                    f"[DiT Zoo] Exported {out_path.name} (opset {opset}, {len(m.graph.node)} nodes)"
                )
            except Exception as e:
                print(f"[DiT Zoo] Failed to export {name} for opset {opset}: {e}")

    return catalog


if __name__ == "__main__":
    export_diffusion_zoo()
