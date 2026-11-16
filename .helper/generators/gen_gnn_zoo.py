#!/usr/bin/env python3
"""Comprehensive Graph Neural Network (GNN) Architecture Family Generator.

Generates self-contained PyTorch models and multi-opset ONNX exports for:
1. GCN_Layer (Graph Convolution Network: D^{-1/2} A D^{-1/2} X W)
2. GAT_Layer (Graph Attention Network with multi-head attention coefficients)
3. GraphSAGE_MeanLayer (Neighborhood mean aggregation + concat + linear projection)
4. GIN_Layer (Graph Isomorphism Network: MLP((1 + eps) X + A X))

Exports across ONNX Opsets 14, 17, 18, 20.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

OUT_DIR = Path(".helper/models/gnn")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. GCN (Graph Convolution Network)
# =============================================================================
class GCN_Layer(nn.Module):
    """Spectral Graph Convolution: H' = A_norm @ H @ W + b."""

    def __init__(self, in_features: int = 32, out_features: int = 32) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        # x: (B, N, F_in), adj_norm: (B, N, N)
        hw = torch.matmul(x, self.weight)
        ahw = torch.matmul(adj_norm, hw) + self.bias
        return self.relu(ahw)


# =============================================================================
# 2. GAT (Graph Attention Network)
# =============================================================================
class GAT_Layer(nn.Module):
    """Graph Attention Network with multi-head attention over graph adjacency."""

    def __init__(self, in_features: int = 32, out_features: int = 32, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.attn_src = nn.Parameter(torch.randn(1, num_heads, 1, self.head_dim) * 0.02)
        self.attn_dst = nn.Parameter(torch.randn(1, num_heads, 1, self.head_dim) * 0.02)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, adj_mask: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D), adj_mask: (B, N, N) (1 for connected, 0 for disconnected)
        B, N, D = x.shape
        h = self.linear(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D_h)

        # Compute source and target attention scores
        score_src = (h * self.attn_src).sum(dim=-1, keepdim=True)  # (B, H, N, 1)
        score_dst = (h * self.attn_dst).sum(dim=-1, keepdim=True)  # (B, H, N, 1)
        raw_scores = score_src + score_dst.transpose(-1, -2)  # (B, H, N, N)
        e = self.leaky_relu(raw_scores)

        # Mask disconnected edges with -1e9
        mask = (1.0 - adj_mask.unsqueeze(1)) * -1e9
        alpha = torch.softmax(e + mask, dim=-1)

        out = torch.matmul(alpha, h).transpose(1, 2).contiguous().view(B, N, D)
        return F.elu(out)


# =============================================================================
# 3. GraphSAGE (Mean Aggregation)
# =============================================================================
class GraphSAGE_MeanLayer(nn.Module):
    """GraphSAGE neighborhood mean aggregation followed by concatenation and linear."""

    def __init__(self, in_features: int = 32, out_features: int = 32) -> None:
        super().__init__()
        self.linear = nn.Linear(2 * in_features, out_features)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor, adj_mean: torch.Tensor) -> torch.Tensor:
        # adj_mean: row-normalized adjacency matrix (B, N, N)
        neighbor_agg = torch.matmul(adj_mean, x)  # (B, N, F)
        concat = torch.cat([x, neighbor_agg], dim=-1)  # (B, N, 2*F)
        out = F.relu(self.linear(concat))
        return self.norm(out)


# =============================================================================
# 4. GIN (Graph Isomorphism Network)
# =============================================================================
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
        # adj: binary adjacency (B, N, N)
        agg = torch.matmul(adj, x)
        h = (1.0 + self.eps) * x + agg
        return self.mlp(h)


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
            print(f"  [OK] Exported {out_path.name:<45} (opset={opset}, nodes={len(m_proto.graph.node):<3}, {size_kb:.1f} KB)")
            exported.append(str(out_path))
        except Exception as e:
            print(f"  [FAIL] Failed to export {out_path.name} (opset={opset}): {e}")
    return exported


def main() -> None:
    print("==================================================================")
    print("  GENERATING GRAPH NEURAL NETWORK (GNN) ONNX SUITE")
    print("==================================================================")

    dummy_x = torch.randn(1, 16, 32)
    # Adjacency matrix for 16 nodes
    raw_adj = (torch.rand(1, 16, 16) > 0.7).float()
    eye = torch.eye(16).unsqueeze(0)
    raw_adj = torch.clamp(raw_adj + eye, 0.0, 1.0)
    # Row normalized adj
    row_sum = raw_adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
    adj_norm = raw_adj / row_sum

    print("\n1. GCN_Layer")
    export_model(GCN_Layer(32, 32), (dummy_x, adj_norm), ["x", "adj_norm"], ["out"], "gcn_layer")

    print("\n2. GAT_Layer")
    export_model(GAT_Layer(32, 32, num_heads=4), (dummy_x, raw_adj), ["x", "adj_mask"], ["out"], "gat_layer")

    print("\n3. GraphSAGE_MeanLayer")
    export_model(GraphSAGE_MeanLayer(32, 32), (dummy_x, adj_norm), ["x", "adj_mean"], ["out"], "graphsage_layer")

    print("\n4. GIN_Layer")
    export_model(GIN_Layer(32, 32), (dummy_x, raw_adj), ["x", "adj"], ["out"], "gin_layer")

    print("\n==================================================================")
    print(f"GNN ONNX suite complete in: {OUT_DIR.resolve()}")
    print("==================================================================")


if __name__ == "__main__":
    main()
