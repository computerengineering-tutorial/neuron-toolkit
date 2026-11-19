"""Generator for Hierarchical & Advanced Vision architectures (Swin Window Attention, XCiT, PointNet).

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

OUT_DIR = Path(".helper/models/hierarchical_vit")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. Swin Transformer Window Multi-Head Self-Attention Block
# =============================================================================
class Swin_WindowAttentionBlock(nn.Module):
    """Shifted Window Attention Block (W-MSA)."""

    def __init__(self, dim: int = 64, window_size: int = 4, num_heads: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.relative_position_bias = nn.Parameter(
            torch.zeros(2 * window_size - 1, 2 * window_size - 1)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C) where H, W are divisible by window_size
        B, H, W, C = x.shape
        shortcut = x
        x_norm = self.norm(x)

        # Partition into windows of (window_size, window_size)
        ws = self.window_size
        x_win = (
            x_norm.view(B, H // ws, ws, W // ws, ws, C)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
        )
        num_windows = (H // ws) * (W // ws)
        x_win = x_win.view(B * num_windows, ws * ws, C)

        # Multi-head attention inside windows
        qkv = (
            self.qkv(x_win)
            .view(B * num_windows, ws * ws, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(q / math.sqrt(self.head_dim), k.transpose(-1, -2))
        probs = torch.softmax(scores, dim=-1)
        attn = (
            torch.matmul(probs, v)
            .transpose(1, 2)
            .contiguous()
            .view(B * num_windows, ws * ws, C)
        )
        out = self.proj(attn)

        # Merge windows back
        out = (
            out.view(B, H // ws, W // ws, ws, ws, C)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(B, H, W, C)
        )
        return shortcut + out


# =============================================================================
# 2. XCiT Cross-Covariance Attention (Transposed Attention over Channels)
# =============================================================================
class CrossCovarianceAttention_XCiT(nn.Module):
    """Attention computed over feature channels (C x C) rather than spatial tokens (N x N)."""

    def __init__(self, dim: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        B, N, C = x.shape
        q = (
            self.q(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        )  # (B, H, D, N)
        k = (
            self.k(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        )  # (B, H, D, N)
        v = (
            self.v(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        )  # (B, H, D, N)

        # Normalize along token dimension N for L2 stability
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Attention map is (D, D) per head rather than (N, N)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.temperature
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v).permute(0, 3, 1, 2).contiguous().view(B, N, C)
        return self.proj(out)


# =============================================================================
# 3. PointNet 3D Point Cloud Global Feature Block
# =============================================================================
class PointNet_GlobalFeatureBlock(nn.Module):
    """Shared 1D MLP + Global Max Pooling for 3D point cloud feature learning."""

    def __init__(self, in_channels: int = 3, out_channels: int = 64) -> None:
        super().__init__()
        self.mlp1 = nn.Conv1d(in_channels, 32, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.mlp2 = nn.Conv1d(32, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, num_points)
        x = F.relu(self.bn1(self.mlp1(x)))
        x = F.relu(self.bn2(self.mlp2(x)))
        # Symmetric global max pooling over points
        global_feat = torch.max(x, 2, keepdim=True)[0]  # (B, out_channels, 1)
        return global_feat.squeeze(-1)


# =============================================================================
# Exporter
# =============================================================================
def export_hierarchical_zoo() -> list[dict[str, Any]]:
    models: dict[str, tuple[nn.Module, tuple[torch.Tensor, ...], list[str]]] = {
        "swin_window_attention": (
            Swin_WindowAttentionBlock(64, 4, 4),
            (torch.randn(2, 8, 8, 64),),
            ["x"],
        ),
        "xcit_cross_covariance": (
            CrossCovarianceAttention_XCiT(64, 4),
            (torch.randn(2, 16, 64),),
            ["x"],
        ),
        "pointnet_global_block": (
            PointNet_GlobalFeatureBlock(3, 64),
            (torch.randn(2, 3, 32),),
            ["points"],
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
                    f"[Hierarchical Zoo] Exported {out_path.name} (opset {opset}, {len(m.graph.node)} nodes)"
                )
            except Exception as e:
                print(
                    f"[Hierarchical Zoo] Failed to export {name} for opset {opset}: {e}"
                )

    return catalog


if __name__ == "__main__":
    export_hierarchical_zoo()
