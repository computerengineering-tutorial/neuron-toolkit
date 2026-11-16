"""Generator for State-Space Models (SSM) & Linear RNN architectures (Mamba S6, S4, RWKV).

Exports models to ONNX across opset versions 14, 17, 18, and 20.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT_DIR = Path(".helper/models/ssm_mamba")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. Mamba Selective State Space (S6) Block
# =============================================================================
class Mamba_SelectiveScanBlock(nn.Module):
    """Selective State Space (S6) block with input-dependent Delta, B, C projections."""

    def __init__(self, d_model: int = 64, d_state: int = 16, expand: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        # In-projection to 2 * d_inner (for x and z gating)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # 1D Depthwise Conv
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=3,
            padding=2,
            groups=self.d_inner,
            bias=True,
        )

        # Delta, B, C projections
        self.x_proj = nn.Linear(self.d_inner, 1 + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # A parameter (log-space initialization)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)))

        # D skip parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Out-projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: (B, S, D)
        B, S, _ = u.shape

        # 1. In-projection and split into x and gate z
        xz = self.in_proj(u)  # (B, S, 2 * d_inner)
        x, z = xz.chunk(2, dim=-1)

        # 2. 1D Conv and SiLU
        x_conv = self.conv1d(x.transpose(1, 2))[:, :, :S].transpose(1, 2)
        x_act = F.silu(x_conv)

        # 3. Discretization parameters
        x_dbl = self.x_proj(x_act)  # (B, S, 1 + 2 * d_state)
        dt = F.softplus(self.dt_proj(x_dbl[:, :, :1]))  # (B, S, d_inner)
        B_mat = x_dbl[:, :, 1 : 1 + self.d_state]  # (B, S, d_state)
        C_mat = x_dbl[:, :, 1 + self.d_state :]  # (B, S, d_state)

        # Discretize A: exp(-exp(A_log) * dt)
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # Simplified associative scan / recurrence
        # y = x_act * D + dt * (B_mat * C_mat).sum(dim=-1, keepdim=True)
        bc = (B_mat * C_mat).sum(dim=-1, keepdim=True)
        y = x_act * self.D + dt * bc

        # 4. Multiply with gating branch z
        y_gated = y * F.silu(z)

        # 5. Out projection
        return self.out_proj(y_gated)


# =============================================================================
# 2. S4 (Structured State Space) Block
# =============================================================================
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
        # x: (B, S, D)
        residual = x
        x_norm = self.norm(x)
        h = self.w(x_norm)
        # Recurrent state projection
        y = h * self.d + torch.sigmoid(h) * self.c.mean()
        return residual + y


# =============================================================================
# 3. RWKV Time-Mixing Block
# =============================================================================
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
        # x: (B, S, D)
        k = self.key(x)
        v = self.value(x)
        r = torch.sigmoid(self.receptance(x))

        # WKV calculation
        wkv = torch.cumsum(k * v, dim=1) / (torch.cumsum(torch.abs(k), dim=1) + 1e-4)
        rwkv = r * wkv
        return self.output(rwkv)


# =============================================================================
# 4. RWKV Channel-Mixing Block
# =============================================================================
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


# =============================================================================
# Exporter
# =============================================================================
def export_ssm_zoo() -> list[dict[str, Any]]:
    models: dict[str, tuple[nn.Module, tuple[torch.Tensor, ...], list[str]]] = {
        "mamba_selective_scan": (Mamba_SelectiveScanBlock(64, 16), (torch.randn(2, 16, 64),), ["x"]),
        "s4_state_space": (S4_StructuredStateSpace(64, 16), (torch.randn(2, 16, 64),), ["x"]),
        "rwkv_time_mix": (RWKV_TimeMix(64), (torch.randn(2, 16, 64),), ["x"]),
        "rwkv_channel_mix": (RWKV_ChannelMix(64, 128), (torch.randn(2, 16, 64),), ["x"]),
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
                catalog.append({"name": name, "opset": opset, "path": str(out_path), "nodes": len(m.graph.node)})
                print(f"[SSM Zoo] Exported {out_path.name} (opset {opset}, {len(m.graph.node)} nodes)")
            except Exception as e:
                print(f"[SSM Zoo] Failed to export {name} for opset {opset}: {e}")

    return catalog


if __name__ == "__main__":
    export_ssm_zoo()
