#!/usr/bin/env python3
"""Comprehensive Graph Neural Network (GNN) Architecture Family Generator."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from generators.decorators import register_architecture
from generators.registry import ModelRegistry


def _make_gnn_inputs(normalized: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    dummy_x = torch.randn(1, 16, 32)
    raw_adj = (torch.rand(1, 16, 16) > 0.7).float()
    eye = torch.eye(16).unsqueeze(0)
    raw_adj = torch.clamp(raw_adj + eye, 0.0, 1.0)
    if normalized:
        row_sum = raw_adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return (dummy_x, raw_adj / row_sum)
    return (dummy_x, raw_adj)


@register_architecture(
    name="gcn_layer",
    category="gnn",
    subfolder="gnn",
    inputs=lambda: _make_gnn_inputs(normalized=True),
    input_names=["x", "adj_norm"],
    output_names=["out"],
    init_args=(32, 32),
)
class GCN_Layer(nn.Module):
    """Spectral Graph Convolution: H' = A_norm @ H @ W + b."""

    def __init__(self, in_features: int = 32, out_features: int = 32) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        hw = torch.matmul(x, self.weight)
        ahw = torch.matmul(adj_norm, hw) + self.bias
        return self.relu(ahw)


@register_architecture(
    name="gat_layer",
    category="gnn",
    subfolder="gnn",
    inputs=lambda: _make_gnn_inputs(normalized=False),
    input_names=["x", "adj_mask"],
    output_names=["out"],
    init_args=(32, 32),
    init_kwargs={"num_heads": 4},
)
class GAT_Layer(nn.Module):
    """Graph Attention Network with multi-head attention over graph adjacency."""

    def __init__(
        self, in_features: int = 32, out_features: int = 32, num_heads: int = 4
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.attn_src = nn.Parameter(torch.randn(1, num_heads, 1, self.head_dim) * 0.02)
        self.attn_dst = nn.Parameter(torch.randn(1, num_heads, 1, self.head_dim) * 0.02)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, adj_mask: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        h = (
            self.linear(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        )
        score_src = (h * self.attn_src).sum(dim=-1, keepdim=True)
        score_dst = (h * self.attn_dst).sum(dim=-1, keepdim=True)
        raw_scores = score_src + score_dst.transpose(-1, -2)
        e = self.leaky_relu(raw_scores)

        mask = (1.0 - adj_mask.unsqueeze(1)) * -1e9
        alpha = torch.softmax(e + mask, dim=-1)

        out = torch.matmul(alpha, h).transpose(1, 2).contiguous().view(B, N, D)
        return F.elu(out)


@register_architecture(
    name="graphsage_layer",
    category="gnn",
    subfolder="gnn",
    inputs=lambda: _make_gnn_inputs(normalized=True),
    input_names=["x", "adj_mean"],
    output_names=["out"],
    init_args=(32, 32),
)
class GraphSAGE_MeanLayer(nn.Module):
    """GraphSAGE neighborhood mean aggregation followed by concatenation and linear."""

    def __init__(self, in_features: int = 32, out_features: int = 32) -> None:
        super().__init__()
        self.linear = nn.Linear(2 * in_features, out_features)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor, adj_mean: torch.Tensor) -> torch.Tensor:
        neighbor_agg = torch.matmul(adj_mean, x)
        concat = torch.cat([x, neighbor_agg], dim=-1)
        out = F.relu(self.linear(concat))
        return self.norm(out)


@register_architecture(
    name="gin_layer",
    category="gnn",
    subfolder="gnn",
    inputs=lambda: _make_gnn_inputs(normalized=False),
    input_names=["x", "adj"],
    output_names=["out"],
    init_args=(32, 32),
)
class GIN_Layer(nn.Module):
    """GIN: H' = MLP((1 + eps) * H + A @ H)."""

    def __init__(self, in_features: int = 32, out_features: int = 32) -> None:
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 2 * out_features),
            nn.ReLU(),
            nn.Linear(2 * out_features, out_features),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        agg = torch.matmul(adj, x)
        h = (1.0 + self.eps) * x + agg
        return self.mlp(h)


def main() -> None:
    ModelRegistry.export_all(category="gnn")


if __name__ == "__main__":
    main()
