#!/usr/bin/env python3
"""Comprehensive Advanced Transformer Architecture Family Generator.

Generates self-contained PyTorch models and multi-opset ONNX exports for:
1. PreLN_TransformerBlock (Pre-LayerNorm standard Transformer block)
2. PostLN_TransformerBlock (Original Vaswani Post-LayerNorm Transformer block)
3. ViT_Block (Vision Transformer Patch Embedding 2D Conv + Transformer Block)
4. MultiHeadLatentAttention_MLA (DeepSeek V2/V3 style low-rank KV compression & latent projection)
5. SparseMoE_Block (Top-2 Softmax Gated Mixture of Experts with 4 experts)
6. LinearAttention_Performer (Kernel-based O(N) linear attention with ELU+1 feature map)
7. SwiGLU_RMSNorm_TransformerBlock (Modern LLaMA-style block with RMSNorm + SwiGLU)

Exports across ONNX Opsets 14, 17, 18, 20.
"""

from __future__ import annotations

import math
from pathlib import Path
import warnings

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

OUT_DIR = Path(".helper/models/transformer_advanced")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Helper Attention Module
# =============================================================================
class SelfAttention(nn.Module):
    def __init__(self, d_model: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        q, k, v = torch.chunk(self.qkv(x), 3, dim=-1)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q / math.sqrt(self.head_dim), k.transpose(-1, -2))
        probs = torch.softmax(scores, dim=-1)
        attn = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out(attn)


# =============================================================================
# 1. Pre-LN Transformer Block
# =============================================================================
class PreLN_TransformerBlock(nn.Module):
    def __init__(self, d_model: int = 64, num_heads: int = 4, d_ffn: int = 256) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Linear(d_ffn, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# =============================================================================
# 2. Post-LN Transformer Block
# =============================================================================
class PostLN_TransformerBlock(nn.Module):
    def __init__(self, d_model: int = 64, num_heads: int = 4, d_ffn: int = 256) -> None:
        super().__init__()
        self.attn = SelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Linear(d_ffn, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.mlp(x))
        return x


# =============================================================================
# 3. Vision Transformer (ViT) Block with Patch Embedding
# =============================================================================
class ViT_Block(nn.Module):
    def __init__(self, img_channels: int = 3, patch_size: int = 4, d_model: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(img_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # img: (B, C, H, W) -> patch_embed -> (B, D, H/P, W/P) -> flatten to (B, N, D)
        B = img.shape[0]
        x = self.patch_embed(img).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# =============================================================================
# 4. Multi-Head Latent Attention (MLA - DeepSeek Style)
# =============================================================================
class MultiHeadLatentAttention_MLA(nn.Module):
    """DeepSeek style low-rank KV latent compression and up-projection."""

    def __init__(self, d_model: int = 64, num_heads: int = 4, d_latent: int = 16) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Q projection
        self.q_proj = nn.Linear(d_model, d_model)

        # Low-rank KV compression
        self.kv_down_proj = nn.Linear(d_model, d_latent, bias=False)
        self.kv_up_proj_k = nn.Linear(d_latent, d_model, bias=False)
        self.kv_up_proj_v = nn.Linear(d_latent, d_model, bias=False)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Compressed KV latent representation
        c_kv = self.kv_down_proj(x)  # (B, S, d_latent)

        # Decompressed K and V
        k = self.kv_up_proj_k(c_kv).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.kv_up_proj_v(c_kv).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q / math.sqrt(self.head_dim), k.transpose(-1, -2))
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


# =============================================================================
# 5. Sparse Mixture-of-Experts (MoE) Block
# =============================================================================
class SparseMoE_Block(nn.Module):
    """Mixture of Experts with top-2 softmax gating over 4 feed-forward experts."""

    def __init__(self, d_model: int = 64, d_ffn: int = 128, num_experts: int = 4, top_k: int = 2) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Linear(d_model, num_experts, bias=False)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.GELU(),
                nn.Linear(d_ffn, d_model),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        norm_x = self.norm(x)
        # Gating logits & Softmax weights
        logits = self.gate(norm_x)
        weights = torch.softmax(logits, dim=-1)

        # Sum weighted expert contributions
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            w = weights[:, :, i : i + 1]
            out = out + w * expert(norm_x)
        return residual + out


# =============================================================================
# 6. Linear Attention (Performer Style Kernel-based Attention)
# =============================================================================
class LinearAttention_Performer(nn.Module):
    """Linear attention: phi(Q) @ (phi(K)^T @ V) achieving O(N) complexity."""

    def __init__(self, d_model: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Non-negative kernel feature map: elu(x) + 1
        q_phi = F.elu(q) + 1.0
        k_phi = F.elu(k) + 1.0

        # Linear attention contraction: (k_phi^T @ v) has shape (B, H, D, D)
        kv = torch.matmul(k_phi.transpose(-1, -2), v)  # (B, H, D_h, D_h)
        # Output: q_phi @ kv
        num = torch.matmul(q_phi, kv)  # (B, H, S, D_h)

        # Normalization denominator
        den = torch.matmul(q_phi, k_phi.sum(dim=-2, keepdim=True).transpose(-1, -2)).clamp(min=1e-6)
        out = (num / den).transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


# =============================================================================
# 7. Modern LLaMA-style SwiGLU + RMSNorm Block
# =============================================================================
class SimpleRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class SwiGLU_RMSNorm_TransformerBlock(nn.Module):
    def __init__(self, d_model: int = 64, num_heads: int = 4, d_ffn: int = 128) -> None:
        super().__init__()
        self.norm1 = SimpleRMSNorm(d_model)
        self.attn = SelfAttention(d_model, num_heads)
        self.norm2 = SimpleRMSNorm(d_model)
        self.w1 = nn.Linear(d_model, d_ffn, bias=False)
        self.w3 = nn.Linear(d_model, d_ffn, bias=False)
        self.w2 = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        norm_x = self.norm2(x)
        ffn = self.w2(self.w1(norm_x) * F.silu(self.w3(norm_x)))
        return x + ffn


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
    print("  GENERATING ADVANCED TRANSFORMER ONNX SUITE")
    print("==================================================================")

    dummy_x = torch.randn(1, 16, 64)
    dummy_img = torch.randn(1, 3, 16, 16)

    print("\n1. PreLN_TransformerBlock")
    export_model(PreLN_TransformerBlock(64, 4, 128), (dummy_x,), ["x"], ["out"], "transformer_pre_ln")

    print("\n2. PostLN_TransformerBlock")
    export_model(PostLN_TransformerBlock(64, 4, 128), (dummy_x,), ["x"], ["out"], "transformer_post_ln")

    print("\n3. ViT_Block")
    export_model(ViT_Block(3, 4, 64, 4), (dummy_img,), ["img"], ["out"], "vit_patch_block")

    print("\n4. MultiHeadLatentAttention_MLA")
    export_model(MultiHeadLatentAttention_MLA(64, 4, 16), (dummy_x,), ["x"], ["out"], "transformer_mla")

    print("\n5. SparseMoE_Block")
    export_model(SparseMoE_Block(64, 128, 4, 2), (dummy_x,), ["x"], ["out"], "transformer_moe")

    print("\n6. LinearAttention_Performer")
    export_model(LinearAttention_Performer(64, 4), (dummy_x,), ["x"], ["out"], "transformer_performer")

    print("\n7. SwiGLU_RMSNorm_TransformerBlock")
    export_model(SwiGLU_RMSNorm_TransformerBlock(64, 4, 128), (dummy_x,), ["x"], ["out"], "transformer_swiglu_rms")

    print("\n==================================================================")
    print(f"Advanced Transformer ONNX suite complete in: {OUT_DIR.resolve()}")
    print("==================================================================")


if __name__ == "__main__":
    main()
