#!/usr/bin/env python3
"""Comprehensive Synthetic & HuggingFace Attention Family Generator.

Generates ONNX models for ALL major attention variants across deep learning:
1. StandardMHA_SeparateQKV (BERT, ViT, RoBERTa)
2. StandardMHA_FusedQKV (Standard Transformer Fused Projection)
3. CausalMHA (GPT-2, OPT Causal Masked Attention)
4. RotaryMHA (LLaMA, Mistral, Qwen RoPE Attention)
5. GroupedQueryAttention_GQA (LLaMA 3, Mistral 7B Grouped-Query Attention)
6. MultiQueryAttention_MQA (Falcon, PaLM Single-KV Attention)
7. CrossAttention (Whisper, BART, Transformer Decoder Cross-Attention)
8. GatedAttention_Sigmoid (Conformer Speech Attention)
9. GatedAttention_SiLU (Modern Speech Gated Variant)
10. GatedAttention_GELU (GELU Gated Variant)
11. T5_RelativePositionAttention (T5 Relative Position Bias)
12. ALiBi_Attention (BLOOM, MPT Linear Biases Slopes)
13. HuggingFace_GPT2_Attention (Real HuggingFace GPT-2 Self-Attention)
14. HuggingFace_Llama_Attention (Real HuggingFace LLaMA Attention)
15. HuggingFace_Whisper_Attention (Real HuggingFace Whisper Cross-Attention)

Exports across multiple ONNX Opset versions: 14, 17, 18, 20, 23.
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import onnx
import torch
import torch.nn.functional as F
from torch import nn

warnings.filterwarnings("ignore")

OUT_DIR = Path(".helper/models/attention_variants")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. Standard Multi-Head Attention (Separate Q, K, V Projections - BERT / ViT)
# =============================================================================
class StandardMHA_SeparateQKV(nn.Module):
    def __init__(self, d_model: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
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

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q * scale, k.transpose(-1, -2))
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


# =============================================================================
# 2. Standard Multi-Head Attention (Fused QKV Projection)
# =============================================================================
class StandardMHA_FusedQKV(nn.Module):
    def __init__(self, d_model: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q * scale, k.transpose(-1, -2))
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


# =============================================================================
# 3. Causal Masked Multi-Head Attention (GPT-2 Style)
# =============================================================================
class CausalMHA(nn.Module):
    def __init__(self, d_model: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q * scale, k.transpose(-1, -2))

        # Causal mask (triangular upper mask)
        mask = torch.triu(torch.full((S, S), -10000.0, device=x.device), diagonal=1)
        scores = scores + mask

        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


# =============================================================================
# 4. Rotary Multi-Head Attention (LLaMA Style with Pre-RMSNorm)
# =============================================================================
class RotaryMHA(nn.Module):
    def __init__(self, d_model: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.gamma = nn.Parameter(torch.ones(d_model))

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        # Precomputed RoPE frequencies
        theta = torch.randn(512, self.head_dim)
        self.register_buffer("theta", theta)

    def _apply_rope(self, t: torch.Tensor) -> torch.Tensor:
        S = t.shape[-2]
        th = self.theta[:S]
        cos = torch.cos(th)
        sin = torch.sin(th)
        t_r = t.reshape(*t.shape[:-1], -1, 2)
        rot = torch.stack((-t_r[..., 1], t_r[..., 0]), dim=-1).reshape(t.shape)
        return (t * cos) + (rot * sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        # RMSNorm
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        scale_rms = (D**0.5) / torch.clamp(norm, min=1e-12)
        x_norm = x * scale_rms * self.gamma

        qkv = self.qkv_proj(x_norm)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        q_rot = self._apply_rope(q)
        k_rot = self._apply_rope(k)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q_rot * scale, k_rot.transpose(-1, -2))
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


# =============================================================================
# 5. Grouped-Query Attention (GQA - LLaMA 3, Mistral 7B)
# =============================================================================
class GroupedQueryAttention_GQA(nn.Module):
    def __init__(
        self, d_model: int = 64, num_q_heads: int = 8, num_kv_heads: int = 2
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_q_heads // num_kv_heads
        self.head_dim = d_model // num_q_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Expand KV heads to match Q heads
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q * scale, k.transpose(-1, -2))
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


# =============================================================================
# 6. Multi-Query Attention (MQA - Falcon, PaLM)
# =============================================================================
class MultiQueryAttention_MQA(nn.Module):
    def __init__(self, d_model: int = 64, num_q_heads: int = 8) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.head_dim = d_model // num_q_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        # Single head for K and V
        self.k_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, 1, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, 1, self.head_dim).transpose(1, 2)

        # Broadcast KV across all query heads
        k = k.expand(B, self.num_q_heads, S, self.head_dim)
        v = v.expand(B, self.num_q_heads, S, self.head_dim)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q * scale, k.transpose(-1, -2))
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


# =============================================================================
# 7. Cross-Attention (Whisper, BART, Transformer Decoder)
# =============================================================================
class CrossAttention(nn.Module):
    def __init__(self, d_model: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        B, S_dec, D = hidden_states.shape
        _, S_enc, _ = encoder_hidden_states.shape

        q = (
            self.q_proj(hidden_states)
            .view(B, S_dec, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(encoder_hidden_states)
            .view(B, S_enc, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(encoder_hidden_states)
            .view(B, S_enc, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q * scale, k.transpose(-1, -2))
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, S_dec, D)
        return self.out_proj(out)


# =============================================================================
# 8. Gated Attention with Sigmoid (Conformer Style)
# =============================================================================
class GatedAttention_Sigmoid(nn.Module):
    def __init__(self, d_model: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.gate_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q * scale, k.transpose(-1, -2))
        probs = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(probs, v)

        # Gate projection
        gate = (
            self.gate_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        )
        gated_attn = attn_out * torch.sigmoid(gate)

        out = gated_attn.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


# =============================================================================
# 9. Gated Attention with SiLU (SwiGLU-style Attention Gating)
# =============================================================================
class GatedAttention_SiLU(nn.Module):
    def __init__(self, d_model: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.gate_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q * scale, k.transpose(-1, -2))
        probs = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(probs, v)

        gate = (
            self.gate_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        )
        gated_attn = attn_out * F.silu(gate)

        out = gated_attn.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


# =============================================================================
# 10. Gated Attention with GELU
# =============================================================================
class GatedAttention_GELU(nn.Module):
    def __init__(self, d_model: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.gate_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q * scale, k.transpose(-1, -2))
        probs = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(probs, v)

        gate = (
            self.gate_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        )
        gated_attn = attn_out * F.gelu(gate)

        out = gated_attn.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


# =============================================================================
# 11. T5 Relative Position Bias Attention
# =============================================================================
class T5_RelativePositionAttention(nn.Module):
    def __init__(
        self, d_model: int = 64, num_heads: int = 4, num_buckets: int = 32
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q * scale, k.transpose(-1, -2))

        # Relative position bias table lookup
        pos = torch.arange(S, device=x.device)
        rel_pos = torch.clamp((pos[:, None] - pos[None, :]) + 16, 0, 31)
        bias = self.relative_attention_bias(rel_pos).permute(2, 0, 1).unsqueeze(0)
        scores = scores + bias

        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


# =============================================================================
# 12. ALiBi Attention (Attention with Linear Biases - BLOOM / MPT)
# =============================================================================
class ALiBi_Attention(nn.Module):
    def __init__(self, d_model: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Geometric slope distribution
        slopes = torch.tensor(
            [2.0 ** (-8.0 * i / num_heads) for i in range(1, num_heads + 1)]
        )
        self.register_buffer("slopes", slopes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q * scale, k.transpose(-1, -2))

        # ALiBi slope bias: -slopes * |i - j|
        pos = torch.arange(S, device=x.device)
        distance = -torch.abs(pos[:, None] - pos[None, :]).float()
        alibi_bias = distance.unsqueeze(0) * self.slopes.view(self.num_heads, 1, 1)
        scores = scores + alibi_bias.unsqueeze(0)

        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


# =============================================================================
# 12b. Gemma-2 Multi-Head Attention with Logits Soft-Capping
# =============================================================================
class Gemma2_SoftCappedAttention(nn.Module):
    """Gemma-2 style Multi-Head Attention with Logits Soft-Capping."""

    def __init__(
        self, d_model: int = 64, num_heads: int = 4, softcap: float = 50.0
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.softcap = softcap

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q * scale, k.transpose(-1, -2))

        # Gemma-2 Logits Soft-Capping: cap * tanh(scores / cap)
        scores = self.softcap * torch.tanh(scores / self.softcap)

        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


# =============================================================================
# Hugging Face Real Pretrained Attention Modules
# =============================================================================
def get_hf_gpt2_attention():
    from transformers.models.gpt2.configuration_gpt2 import GPT2Config
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

    config = GPT2Config(n_embd=64, n_head=4, n_positions=128)
    layer = GPT2Attention(config)
    layer.eval()

    class HF_GPT2_Wrapper(nn.Module):
        def __init__(self, attn):
            super().__init__()
            self.attn = attn

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.attn(x)[0]

    return HF_GPT2_Wrapper(layer)


def get_hf_llama_attention():
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaRotaryEmbedding,
    )

    config = LlamaConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )
    config._attn_implementation = "eager"

    class HF_Llama_Wrapper(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.rotary = LlamaRotaryEmbedding(config)
            self.attn = LlamaAttention(config, layer_idx=0)
            self.eval()

        def forward(self, x: torch.Tensor, pos_ids: torch.Tensor) -> torch.Tensor:
            cos, sin = self.rotary(x, pos_ids)
            return self.attn(x, position_embeddings=(cos, sin))[0]

    return HF_Llama_Wrapper(config)


def get_hf_whisper_attention():
    from transformers.models.whisper.configuration_whisper import WhisperConfig
    from transformers.models.whisper.modeling_whisper import WhisperAttention

    config = WhisperConfig(
        d_model=64, decoder_attention_heads=4, max_target_positions=128
    )
    config._attn_implementation = "eager"

    class HF_Whisper_Wrapper(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.attn = WhisperAttention(
                embed_dim=64, num_heads=4, config=config, layer_idx=0, is_decoder=True
            )
            self.eval()

        def forward(
            self, x: torch.Tensor, key_val_states: torch.Tensor
        ) -> torch.Tensor:
            return self.attn(x, key_value_states=key_val_states)[0]

    return HF_Whisper_Wrapper(config)


from generators.base import export_model as unified_export_model


# =============================================================================
# Generation & Multi-Opset ONNX Export Driver
# =============================================================================
def export_model(
    model: nn.Module,
    dummy_inputs: tuple[torch.Tensor, ...],
    input_names: list[str],
    output_names: list[str],
    base_name: str,
    opset_versions: list[int] = (14, 18, 20),
    dynamic_axes: dict[str, dict[int, str]] | None = None,
) -> None:
    if dynamic_axes is None and input_names and output_names:
        dynamic_axes = {
            input_names[0]: {0: "batch_size", 1: "sequence_length"},
            output_names[0]: {0: "batch_size", 1: "sequence_length"},
        }
    unified_export_model(
        model=model,
        dummy_inputs=dummy_inputs,
        input_names=input_names,
        output_names=output_names,
        base_name=base_name,
        out_dir=OUT_DIR,
        opset_versions=opset_versions,
        dynamic_axes=dynamic_axes,
    )


def main() -> None:
    print("==================================================================")
    print("  GENERATING COMPREHENSIVE ATTENTION FAMILY ONNX SUITE")
    print("==================================================================")

    dummy_x = torch.randn(1, 16, 64)
    dummy_enc = torch.randn(1, 24, 64)
    dummy_pos = torch.arange(16).unsqueeze(0)

    # 1. StandardMHA_SeparateQKV
    print("\n1. StandardMHA_SeparateQKV (BERT / ViT)")
    export_model(
        StandardMHA_SeparateQKV(d_model=64, num_heads=4),
        (dummy_x,),
        ["x"],
        ["out"],
        "mha_separate_qkv",
        opset_versions=[14, 18, 20],
    )

    # 2. StandardMHA_FusedQKV
    print("\n2. StandardMHA_FusedQKV")
    export_model(
        StandardMHA_FusedQKV(d_model=64, num_heads=4),
        (dummy_x,),
        ["x"],
        ["out"],
        "mha_fused_qkv",
        opset_versions=[14, 18, 20],
    )

    # 3. CausalMHA
    print("\n3. CausalMHA (GPT-2)")
    export_model(
        CausalMHA(d_model=64, num_heads=4),
        (dummy_x,),
        ["x"],
        ["out"],
        "mha_causal",
        opset_versions=[14, 18, 20],
    )

    # 4. RotaryMHA
    print("\n4. RotaryMHA (LLaMA / Mistral)")
    export_model(
        RotaryMHA(d_model=64, num_heads=4),
        (dummy_x,),
        ["x"],
        ["out"],
        "mha_rope",
        opset_versions=[14, 18, 20],
    )

    # 5. GroupedQueryAttention_GQA
    print("\n5. GroupedQueryAttention_GQA (LLaMA 3)")
    export_model(
        GroupedQueryAttention_GQA(d_model=64, num_q_heads=8, num_kv_heads=2),
        (dummy_x,),
        ["x"],
        ["out"],
        "mha_gqa",
        opset_versions=[14, 18, 20],
    )

    # 6. MultiQueryAttention_MQA
    print("\n6. MultiQueryAttention_MQA (Falcon / PaLM)")
    export_model(
        MultiQueryAttention_MQA(d_model=64, num_q_heads=8),
        (dummy_x,),
        ["x"],
        ["out"],
        "mha_mqa",
        opset_versions=[14, 18, 20],
    )

    # 7. CrossAttention
    print("\n7. CrossAttention (Whisper / BART)")
    export_model(
        CrossAttention(d_model=64, num_heads=4),
        (dummy_x, dummy_enc),
        ["hidden_states", "encoder_hidden_states"],
        ["out"],
        "cross_attention",
        opset_versions=[14, 18, 20],
    )

    # 8. GatedAttention_Sigmoid
    print("\n8. GatedAttention_Sigmoid (Conformer)")
    export_model(
        GatedAttention_Sigmoid(d_model=64, num_heads=4),
        (dummy_x,),
        ["x"],
        ["out"],
        "gated_attn_sigmoid",
        opset_versions=[14, 18, 20],
    )

    # 9. GatedAttention_SiLU
    print("\n9. GatedAttention_SiLU")
    export_model(
        GatedAttention_SiLU(d_model=64, num_heads=4),
        (dummy_x,),
        ["x"],
        ["out"],
        "gated_attn_silu",
        opset_versions=[14, 18, 20],
    )

    # 10. GatedAttention_GELU
    print("\n10. GatedAttention_GELU")
    export_model(
        GatedAttention_GELU(d_model=64, num_heads=4),
        (dummy_x,),
        ["x"],
        ["out"],
        "gated_attn_gelu",
        opset_versions=[14, 18, 20],
    )

    # 11. T5_RelativePositionAttention
    print("\n11. T5_RelativePositionAttention")
    export_model(
        T5_RelativePositionAttention(d_model=64, num_heads=4),
        (dummy_x,),
        ["x"],
        ["out"],
        "mha_t5_relpos",
        opset_versions=[14, 18, 20],
    )

    # 12. ALiBi_Attention
    print("\n12. ALiBi_Attention (BLOOM / MPT)")
    export_model(
        ALiBi_Attention(d_model=64, num_heads=4),
        (dummy_x,),
        ["x"],
        ["out"],
        "mha_alibi",
        opset_versions=[14, 18, 20],
    )

    # 12b. Gemma2_SoftCappedAttention
    print("\n12b. Gemma2_SoftCappedAttention (Gemma-2 / Grok-1)")
    export_model(
        Gemma2_SoftCappedAttention(d_model=64, num_heads=4, softcap=50.0),
        (dummy_x,),
        ["x"],
        ["out"],
        "mha_gemma2_softcap",
        opset_versions=[14, 18, 20],
    )

    # 13. Real HuggingFace Models
    print("\n13. HuggingFace GPT-2 Self-Attention")
    try:
        hf_gpt2 = get_hf_gpt2_attention()
        export_model(
            hf_gpt2,
            (dummy_x,),
            ["x"],
            ["out"],
            "hf_gpt2_attention",
            opset_versions=[14, 18, 20],
        )
    except Exception as e:
        print("  Skipping HF GPT-2:", e)

    print("\n14. HuggingFace LLaMA Self-Attention")
    try:
        hf_llama = get_hf_llama_attention()
        export_model(
            hf_llama,
            (dummy_x, dummy_pos),
            ["x", "position_ids"],
            ["out"],
            "hf_llama_attention",
            opset_versions=[14, 18, 20],
        )
    except Exception as e:
        print("  Skipping HF LLaMA:", e)

    print("\n15. HuggingFace Whisper Cross-Attention")
    try:
        hf_whisper = get_hf_whisper_attention()
        export_model(
            hf_whisper,
            (dummy_x, dummy_enc),
            ["x", "key_value_states"],
            ["out"],
            "hf_whisper_attention",
            opset_versions=[14, 18, 20],
        )
    except Exception as e:
        print("  Skipping HF Whisper:", e)

    print("\n==================================================================")
    print(f"All attention variant ONNX models generated in: {OUT_DIR.resolve()}")
    print("==================================================================")


if __name__ == "__main__":
    main()
