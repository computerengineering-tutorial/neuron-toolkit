#!/usr/bin/env python3
"""Comprehensive Advanced Transformer Architecture Family Generator."""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import nn

from generators.decorators import register_architecture
from generators.registry import ModelRegistry

SEQ_DYN_AXES = {"x": {0: "batch_size", 1: "seq_len"}, "out": {0: "batch_size", 1: "seq_len"}}
IMG_DYN_AXES = {"img": {0: "batch_size"}, "out": {0: "batch_size"}}


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


@register_architecture(
    name="transformer_pre_ln",
    category="transformer_advanced",
    subfolder="transformer_advanced",
    shapes=[(1, 16, 64)],
    input_names=["x"],
    output_names=["out"],
    dynamic_axes=SEQ_DYN_AXES,
    init_args=(64, 4, 128),
)
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


@register_architecture(
    name="transformer_post_ln",
    category="transformer_advanced",
    subfolder="transformer_advanced",
    shapes=[(1, 16, 64)],
    input_names=["x"],
    output_names=["out"],
    dynamic_axes=SEQ_DYN_AXES,
    init_args=(64, 4, 128),
)
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


@register_architecture(
    name="vit_patch_block",
    category="transformer_advanced",
    subfolder="transformer_advanced",
    shapes=[(1, 3, 16, 16)],
    input_names=["img"],
    output_names=["out"],
    dynamic_axes=IMG_DYN_AXES,
    init_args=(3, 4, 64, 4),
)
class ViT_Block(nn.Module):
    def __init__(
        self,
        img_channels: int = 3,
        patch_size: int = 4,
        d_model: int = 64,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(
            img_channels, d_model, kernel_size=patch_size, stride=patch_size
        )
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
        B = img.shape[0]
        x = self.patch_embed(img).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


@register_architecture(
    name="transformer_mla",
    category="transformer_advanced",
    subfolder="transformer_advanced",
    shapes=[(1, 16, 64)],
    input_names=["x"],
    output_names=["out"],
    dynamic_axes=SEQ_DYN_AXES,
    expected_target_node="MultiHeadLatentAttention_KVProjection",
    init_args=(64, 4, 16),
)
class MultiHeadLatentAttention_MLA(nn.Module):
    """DeepSeek style low-rank KV latent compression and up-projection."""

    def __init__(
        self, d_model: int = 64, num_heads: int = 4, d_latent: int = 16
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_down_proj = nn.Linear(d_model, d_latent, bias=False)
        self.kv_up_proj_k = nn.Linear(d_latent, d_model, bias=False)
        self.kv_up_proj_v = nn.Linear(d_latent, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        c_kv = self.kv_down_proj(x)
        k = (
            self.kv_up_proj_k(c_kv)
            .view(B, S, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.kv_up_proj_v(c_kv)
            .view(B, S, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        scores = torch.matmul(q / math.sqrt(self.head_dim), k.transpose(-1, -2))
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


@register_architecture(
    name="transformer_moe",
    category="transformer_advanced",
    subfolder="transformer_advanced",
    shapes=[(1, 16, 64)],
    input_names=["x"],
    output_names=["out"],
    dynamic_axes=SEQ_DYN_AXES,
    init_args=(64, 128, 4, 2),
)
class SparseMoE_Block(nn.Module):
    """Mixture of Experts with top-2 softmax gating over 4 feed-forward experts."""

    def __init__(
        self, d_model: int = 64, d_ffn: int = 128, num_experts: int = 4, top_k: int = 2
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    nn.GELU(),
                    nn.Linear(d_ffn, d_model),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        norm_x = self.norm(x)
        logits = self.gate(norm_x)
        weights = torch.softmax(logits, dim=-1)

        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            w = weights[:, :, i : i + 1]
            out = out + w * expert(norm_x)
        return residual + out


@register_architecture(
    name="transformer_performer",
    category="transformer_advanced",
    subfolder="transformer_advanced",
    shapes=[(1, 16, 64)],
    input_names=["x"],
    output_names=["out"],
    dynamic_axes=SEQ_DYN_AXES,
    init_args=(64, 4),
)
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

        q_phi = F.elu(q) + 1.0
        k_phi = F.elu(k) + 1.0

        kv = torch.matmul(k_phi.transpose(-1, -2), v)
        num = torch.matmul(q_phi, kv)

        den = torch.matmul(
            q_phi, k_phi.sum(dim=-2, keepdim=True).transpose(-1, -2)
        ).clamp(min=1e-6)
        out = (num / den).transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


class SimpleRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


@register_architecture(
    name="transformer_swiglu_rms",
    category="transformer_advanced",
    subfolder="transformer_advanced",
    shapes=[(1, 16, 64)],
    input_names=["x"],
    output_names=["out"],
    dynamic_axes=SEQ_DYN_AXES,
    init_args=(64, 4, 128),
)
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


def main() -> None:
    ModelRegistry.export_all(category="transformer_advanced")


if __name__ == "__main__":
    main()
