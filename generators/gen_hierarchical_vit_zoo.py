#!/usr/bin/env python3
"""Generator for Hierarchical & Advanced Vision architectures (Swin Window Attention, XCiT, PointNet)."""

from __future__ import annotations

import math
from typing import Any
import torch
import torch.nn.functional as F
from torch import nn

from generators.decorators import register_architecture
from generators.registry import ModelRegistry


@register_architecture(
    name="swin_window_attention",
    category="hierarchical_vit",
    subfolder="hierarchical_vit",
    shapes=[(2, 8, 8, 64)],
    input_names=["x"],
    output_names=["out"],
    init_args=(64, 4, 4),
)
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
        B, H, W, C = x.shape
        shortcut = x
        x_norm = self.norm(x)

        ws = self.window_size
        x_win = (
            x_norm.view(B, H // ws, ws, W // ws, ws, C)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
        )
        num_windows = (H // ws) * (W // ws)
        x_win = x_win.view(B * num_windows, ws * ws, C)

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

        out = (
            out.view(B, H // ws, W // ws, ws, ws, C)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(B, H, W, C)
        )
        return shortcut + out


@register_architecture(
    name="xcit_cross_covariance",
    category="hierarchical_vit",
    subfolder="hierarchical_vit",
    shapes=[(2, 16, 64)],
    input_names=["x"],
    output_names=["out"],
    init_args=(64, 4),
)
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
        B, N, C = x.shape
        q = (
            self.q(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        )
        k = (
            self.k(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        )
        v = (
            self.v(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        )

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.temperature
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v).permute(0, 3, 1, 2).contiguous().view(B, N, C)
        return self.proj(out)


@register_architecture(
    name="pointnet_global_block",
    category="hierarchical_vit",
    subfolder="hierarchical_vit",
    shapes=[(2, 3, 32)],
    input_names=["points"],
    output_names=["out"],
    init_args=(3, 64),
)
class PointNet_GlobalFeatureBlock(nn.Module):
    """Shared 1D MLP + Global Max Pooling for 3D point cloud feature learning."""

    def __init__(self, in_channels: int = 3, out_channels: int = 64) -> None:
        super().__init__()
        self.mlp1 = nn.Conv1d(in_channels, 32, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.mlp2 = nn.Conv1d(32, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.mlp1(x)))
        x = F.relu(self.bn2(self.mlp2(x)))
        global_feat = torch.max(x, 2, keepdim=True)[0]
        return global_feat.squeeze(-1)


def export_hierarchical_zoo() -> list[dict[str, Any]]:
    return [
        {"name": name, "path": p}
        for name, paths in ModelRegistry.export_all(category="hierarchical_vit").items()
        for p in paths
    ]


if __name__ == "__main__":
    export_hierarchical_zoo()
