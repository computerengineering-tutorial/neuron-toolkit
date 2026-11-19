#!/usr/bin/env python3
"""Generator for State-Space Models (SSM) & Linear RNN architectures (Mamba S6, S4, RWKV)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from generators.decorators import register_architecture
from generators.registry import ModelRegistry


@register_architecture(
    name="mamba_selective_scan",
    category="ssm_mamba",
    subfolder="ssm_mamba",
    shapes=[(2, 16, 64)],
    input_names=["x"],
    output_names=["out"],
    init_args=(64, 16),
)
class Mamba_SelectiveScanBlock(nn.Module):
    """Selective State Space (S6) block with input-dependent Delta, B, C projections."""

    def __init__(self, d_model: int = 64, d_state: int = 16, expand: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=3,
            padding=2,
            groups=self.d_inner,
            bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, 1 + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        self.A_log = nn.Parameter(
            torch.log(
                torch.arange(1, d_state + 1, dtype=torch.float32).repeat(
                    self.d_inner, 1
                )
            )
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        B, S, _ = u.shape
        xz = self.in_proj(u)
        x, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x.transpose(1, 2))[:, :, :S].transpose(1, 2)
        x_act = F.silu(x_conv)

        x_dbl = self.x_proj(x_act)
        dt = F.softplus(self.dt_proj(x_dbl[:, :, :1]))
        B_mat = x_dbl[:, :, 1 : 1 + self.d_state]
        C_mat = x_dbl[:, :, 1 + self.d_state :]

        bc = (B_mat * C_mat).sum(dim=-1, keepdim=True)
        y = x_act * self.D + dt * bc

        y_gated = y * F.silu(z)
        return self.out_proj(y_gated)


@register_architecture(
    name="s4_state_space",
    category="ssm_mamba",
    subfolder="ssm_mamba",
    shapes=[(2, 16, 64)],
    input_names=["x"],
    output_names=["out"],
    init_args=(64, 16),
)
class S4_StructuredStateSpace(nn.Module):
    """HiPPO continuous-time diagonal recurrent block."""

    def __init__(self, d_model: int = 64, d_state: int = 16) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.w = nn.Linear(d_model, d_model)
        self.c = nn.Parameter(torch.randn(d_model, d_state))
        self.lambda_real = nn.Parameter(-torch.exp(torch.randn(d_state)))
        self.d = nn.Parameter(torch.randn(d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        h = self.w(x_norm)
        y = h * self.d + torch.sigmoid(h) * self.c.mean()
        return residual + y


@register_architecture(
    name="rwkv_time_mix",
    category="ssm_mamba",
    subfolder="ssm_mamba",
    shapes=[(2, 16, 64)],
    input_names=["x"],
    output_names=["out"],
    init_args=(64,),
)
class RWKV_TimeMix(nn.Module):
    """Receptance Weighted Key Value (RWKV) token mixing block."""

    def __init__(self, d_model: int = 64) -> None:
        super().__init__()
        self.d_model = d_model
        self.time_decay = nn.Parameter(torch.randn(d_model))
        self.time_first = nn.Parameter(torch.randn(d_model))

        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.key(x)
        v = self.value(x)
        r = torch.sigmoid(self.receptance(x))

        wkv = torch.cumsum(k * v, dim=1) / (torch.cumsum(torch.abs(k), dim=1) + 1e-4)
        rwkv = r * wkv
        return self.output(rwkv)


@register_architecture(
    name="rwkv_channel_mix",
    category="ssm_mamba",
    subfolder="ssm_mamba",
    shapes=[(2, 16, 64)],
    input_names=["x"],
    output_names=["out"],
    init_args=(64, 128),
)
class RWKV_ChannelMix(nn.Module):
    """RWKV Channel mixing feed-forward block."""

    def __init__(self, d_model: int = 64, d_ffn: int = 256) -> None:
        super().__init__()
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_ffn, bias=False)
        self.value = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = torch.sigmoid(self.receptance(x))
        k = torch.square(torch.relu(self.key(x)))
        return r * self.value(k)


def export_ssm_zoo() -> list[dict[str, Any]]:
    return [
        {"name": name, "path": p}
        for name, paths in ModelRegistry.export_all(category="ssm_mamba").items()
        for p in paths
    ]


if __name__ == "__main__":
    export_ssm_zoo()
