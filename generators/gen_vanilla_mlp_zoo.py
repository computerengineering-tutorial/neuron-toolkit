#!/usr/bin/env python3
"""Comprehensive Vanilla MLP, FeedForward & Gated Architecture Family Generator."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from generators.decorators import register_architecture
from generators.registry import ModelRegistry


@register_architecture(
    name="deep_mlp_norm_act",
    category="mlp_vanilla",
    subfolder="mlp_vanilla",
    shapes=[(1, 16, 64)],
    input_names=["x"],
    output_names=["out"],
    init_args=(64, 256, 64),
)
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


@register_architecture(
    name="resmlp_block",
    category="mlp_vanilla",
    subfolder="mlp_vanilla",
    shapes=[(1, 16, 64)],
    input_names=["x"],
    output_names=["out"],
    init_kwargs={"num_tokens": 16, "dim": 64},
)
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
        norm_x = self.norm1(x)
        token_out = self.cross_token(norm_x.transpose(1, 2)).transpose(1, 2)
        x = x + token_out

        norm_x2 = self.norm2(x)
        channel_out = self.cross_channel2(self.act(self.cross_channel1(norm_x2)))
        return x + channel_out


@register_architecture(
    name="mlpmixer_block",
    category="mlp_vanilla",
    subfolder="mlp_vanilla",
    shapes=[(1, 16, 64)],
    input_names=["x"],
    output_names=["out"],
    init_kwargs={"num_patches": 16, "num_channels": 64},
)
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
        norm_x = self.norm1(x).transpose(1, 2)
        token_mixed = self.token_mlp(norm_x).transpose(1, 2)
        x = x + token_mixed

        channel_mixed = self.channel_mlp(self.norm2(x))
        return x + channel_mixed


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


@register_architecture(
    name="gmlp_block",
    category="mlp_vanilla",
    subfolder="mlp_vanilla",
    shapes=[(1, 16, 64)],
    input_names=["x"],
    output_names=["out"],
    init_kwargs={"dim": 64, "d_ffn": 128, "seq_len": 16},
)
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


@register_architecture(
    name="highway_block",
    category="mlp_vanilla",
    subfolder="mlp_vanilla",
    shapes=[(1, 16, 64)],
    input_names=["x"],
    output_names=["out"],
    init_args=(64,),
)
class HighwayNetwork_Block(nn.Module):
    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.transform_gate = nn.Linear(dim, dim)
        self.nonlinear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = torch.sigmoid(self.transform_gate(x))
        h = F.relu(self.nonlinear(x))
        return h * t + x * (1.0 - t)


@register_architecture(
    name="swiglu_mlp_block",
    category="mlp_vanilla",
    subfolder="mlp_vanilla",
    shapes=[(1, 16, 64)],
    input_names=["x"],
    output_names=["out"],
    init_kwargs={"dim": 64, "hidden_dim": 128},
)
class SwiGLU_MLP_Block(nn.Module):
    def __init__(self, dim: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.w1(x) * F.silu(self.w3(x)))


@register_architecture(
    name="geglu_mlp_block",
    category="mlp_vanilla",
    subfolder="mlp_vanilla",
    shapes=[(1, 16, 64)],
    input_names=["x"],
    output_names=["out"],
    init_kwargs={"dim": 64, "hidden_dim": 128},
)
class GeGLU_MLP_Block(nn.Module):
    def __init__(self, dim: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.w1(x) * F.gelu(self.w3(x)))


@register_architecture(
    name="reglu_mlp_block",
    category="mlp_vanilla",
    subfolder="mlp_vanilla",
    shapes=[(1, 16, 64)],
    input_names=["x"],
    output_names=["out"],
    init_kwargs={"dim": 64, "hidden_dim": 128},
)
class ReGLU_MLP_Block(nn.Module):
    def __init__(self, dim: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.w1(x) * F.relu(self.w3(x)))


def main() -> None:
    ModelRegistry.export_all(category="mlp_vanilla")


if __name__ == "__main__":
    main()
