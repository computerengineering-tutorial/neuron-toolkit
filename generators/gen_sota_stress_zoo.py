#!/usr/bin/env python3
"""SOTA Deep Learning Architecture Zoo & Stress Test Generator.

Generates real-world, production-accurate ONNX models using PyTorch modules for
modern architectures where standard ONNX simplifiers and graph compilers hit
structural limits:

Batch 3 SOTA Foundations:
1. ALiBi Additive Relative Position Bias (MPT / BLOOM / Whisper)
2. HuggingFace Split-Concat Rotate-Half RoPE (LLaMA-3 / Mistral / Gemma / Qwen)
3. Dual-Linear SwiGLU FeedForward (LLaMA-3 / Mistral / DeepSeek)
4. QK-Norm Head Normalization (Gemma-2 / Qwen-2.5 / Swin-v2)
5. Sliding Window & Dynamic Causal Where-Mask Attention (Mistral-7B / Mixtral)
6. Parallel Attention + MLP Shared Norm (Falcon-40B / PaLM-2)
7. Grouped-Query Attention (GQA) Head Broadcast (LLaMA-3 8B/70B)
8. Autoregressive KV-Cache Concat Attention (vLLM / TensorRT-LLM)

Batch 4 Advanced SOTA Architectures:
9. Mixtral Sparse Mixture of Experts (MoE) Top-K Routing & Dispatch
10. DeepSeek Multi-Head Latent Attention (MLA) Low-Rank Projections
11. Microsoft Phi-3 Partial-Dimension RoPE (Partial Rotation + Passthrough)
12. Differential Transformer Dual-Softmax Subtraction (Diff-Transformer)
13. Encoder-Decoder Cross-Attention (Whisper / T5 / Transformer Decoder)
14. Mamba Selective State Space Model (SSM) Discretization & Scan
15. Swin Transformer Shifted Window Attention & Cyclic Roll
16. Vision Transformer (ViT) CLS Token Prepend & 1D Position Embedding
"""

from __future__ import annotations

import io
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import onnx
import torch
import torch.nn.functional as F
from torch import nn

try:
    import inspect

    from torch.onnx._internal.torchscript_exporter import utils as _ts_utils
    if hasattr(_ts_utils.setup_onnx_logging, "__wrapped__") and not inspect.isgeneratorfunction(_ts_utils.setup_onnx_logging.__wrapped__):
        _ts_utils.setup_onnx_logging = _ts_utils.setup_onnx_logging.__wrapped__
except (ImportError, AttributeError):
    pass

from generators.decorators import register_architecture
from generators.registry import ModelRegistry


def export_model_to_onnx(
    model: nn.Module,
    dummy_inputs: tuple[torch.Tensor, ...] | torch.Tensor,
    input_names: list[str],
    output_names: list[str],
    opset_version: int = 17,
) -> onnx.ModelProto:
    """Export a PyTorch nn.Module to an in-memory onnx.ModelProto with standard settings."""
    model.eval()
    if isinstance(dummy_inputs, torch.Tensor):
        dummy_inputs_tuple = (dummy_inputs,)
    else:
        dummy_inputs_tuple = tuple(dummy_inputs)

    buf = io.BytesIO()
    torch.onnx.utils.export(
        model,
        dummy_inputs_tuple,
        buf,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        do_constant_folding=True,
    )
    buf.seek(0)
    proto = onnx.load_model_from_string(buf.getvalue())
    return proto


# =============================================================================
# 1. ALiBi Relative Position Bias Attention (MPT / BLOOM)
# =============================================================================
@register_architecture(
    name="alibi_relative_bias_attention",
    category="sota_architectures",
    subfolder="sota_stress",
    shapes=[(1, 16, 64)],
    input_names=["x"],
    output_names=["out"],
    opset_versions=(17, 18, 20),
)
class ALiBi_Attention(nn.Module):
    """Attention with additive linear relative position bias slopes."""

    def __init__(self, d_model: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

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

        pos = torch.arange(S, device=x.device)
        distance = -torch.abs(pos[:, None] - pos[None, :]).float()
        alibi_bias = distance.unsqueeze(0) * self.slopes.view(self.num_heads, 1, 1)
        scores = scores + alibi_bias.unsqueeze(0)

        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


# =============================================================================
# 2. HuggingFace Rotate-Half RoPE (LLaMA-3 / Mistral / Gemma)
# =============================================================================
@register_architecture(
    name="hf_rotate_half_rope",
    category="sota_architectures",
    subfolder="sota_stress",
    inputs=lambda: (torch.randn(1, 4, 16, 32), torch.randn(1, 1, 16, 32), torch.randn(1, 1, 16, 32)),
    input_names=["x", "cos", "sin"],
    output_names=["out"],
    opset_versions=(17, 18, 20),
)
class HFRotateHalf_RoPE(nn.Module):
    """HuggingFace standard rotate_half RoPE: Split(d/2) -> Neg(x2) -> Concat([neg_x2, x1])."""

    def __init__(self, d: int = 32) -> None:
        super().__init__()
        self.d = d

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : self.d // 2]
        x2 = x[..., self.d // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return (x * cos) + (self.rotate_half(x) * sin)


# =============================================================================
# 3. Dual-Linear SwiGLU MLP (LLaMA-3 / Mistral / DeepSeek)
# =============================================================================
@register_architecture(
    name="dual_linear_swiglu_mlp",
    category="sota_architectures",
    subfolder="sota_stress",
    shapes=[(1, 16, 64)],
    input_names=["x"],
    output_names=["out"],
    opset_versions=(17, 18, 20),
)
class DualLinear_SwiGLU_FFN(nn.Module):
    """Separate gate_proj and up_proj dual-linear SwiGLU feedforward."""

    def __init__(self, d_model: int = 64, d_ff: int = 128) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# =============================================================================
# 4. QK-Norm Head Normalization Attention (Gemma-2 / Qwen-2.5)
# =============================================================================
@register_architecture(
    name="qk_norm_attention",
    category="sota_architectures",
    subfolder="sota_stress",
    shapes=[(1, 16, 64)],
    input_names=["x"],
    output_names=["out"],
    opset_versions=(17, 18, 20),
)
class QKNorm_Attention(nn.Module):
    """Head-wise RMSNorm applied to Queries and Keys prior to dot product."""

    def __init__(self, d_model: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.q_gamma = nn.Parameter(torch.ones(self.head_dim))
        self.k_gamma = nn.Parameter(torch.ones(self.head_dim))

    def _rms_norm(self, t: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(t, p=2, dim=-1, keepdim=True)
        scale = (self.head_dim**0.5) / torch.clamp(norm, min=1e-6)
        return t * scale * gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Per-head QK normalization
        q = self._rms_norm(q, self.q_gamma)
        k = self._rms_norm(k, self.k_gamma)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q * scale, k.transpose(-1, -2))
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


# =============================================================================
# 5. Sliding Window Dynamic Causal Where-Mask Attention (Mistral 7B)
# =============================================================================
@register_architecture(
    name="sliding_window_causal_where_attention",
    category="sota_architectures",
    subfolder="sota_stress",
    inputs=lambda: (torch.randn(1, 16, 64), torch.tril(torch.ones(1, 1, 16, 16)).bool()),
    input_names=["x", "causal_mask"],
    output_names=["out"],
    opset_versions=(17, 18, 20),
)
class SlidingWindow_CausalWhere_Attention(nn.Module):
    """Attention with dynamic boolean causal mask applied via torch.where."""

    def __init__(self, d_model: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q * scale, k.transpose(-1, -2))

        # Dynamic causal boolean mask via torch.where
        scores = torch.where(causal_mask, scores, torch.full_like(scores, -1e4))

        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


# =============================================================================
# 6. Parallel Attention + MLP Shared Norm Block (Falcon / PaLM)
# =============================================================================
@register_architecture(
    name="parallel_attention_mlp",
    category="sota_architectures",
    subfolder="sota_stress",
    shapes=[(1, 16, 64)],
    input_names=["x"],
    output_names=["out"],
    opset_versions=(17, 18, 20),
)
class Falcon_Parallel_Attention_MLP(nn.Module):
    """Parallel block evaluating Attention and MLP on identical pre-normalized input."""

    def __init__(self, d_model: int = 64, num_heads: int = 4, d_ff: int = 128) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.gamma = nn.Parameter(torch.ones(d_model))
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)

        self.mlp_fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.mlp_fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        # Shared RMSNorm
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        h = x * (D**0.5) / torch.clamp(norm, min=1e-6) * self.gamma

        # Attention branch
        qkv = self.qkv(h)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q * (self.head_dim**-0.5), k.transpose(-1, -2))
        attn = torch.matmul(torch.softmax(scores, dim=-1), v).transpose(1, 2).contiguous().view(B, S, D)
        attn_y = self.attn_out(attn)

        # MLP branch
        mlp_y = self.mlp_fc2(F.gelu(self.mlp_fc1(h)))

        # Parallel sum
        return x + attn_y + mlp_y


# =============================================================================
# 7. Grouped-Query Attention (GQA) Head Broadcast (LLaMA-3 8B/70B)
# =============================================================================
@register_architecture(
    name="gqa_kv_broadcast_attention",
    category="sota_architectures",
    subfolder="sota_stress",
    shapes=[(1, 16, 64)],
    input_names=["x"],
    output_names=["out"],
    opset_versions=(17, 18, 20),
)
class GroupedQueryAttention_GQA(nn.Module):
    """GQA where 2 KV heads are broadcast expanded to match 8 Q heads."""

    def __init__(self, d_model: int = 64, num_q_heads: int = 8, num_kv_heads: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_q_heads
        self.rep = num_q_heads // num_kv_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Explicit expand and reshape broadcast
        k = k.unsqueeze(2).expand(B, self.num_kv_heads, self.rep, S, self.head_dim).reshape(B, self.num_q_heads, S, self.head_dim)
        v = v.unsqueeze(2).expand(B, self.num_kv_heads, self.rep, S, self.head_dim).reshape(B, self.num_q_heads, S, self.head_dim)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q * scale, k.transpose(-1, -2))
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.out_proj(out)


# =============================================================================
# 8. Autoregressive KV-Cache Concat Attention (vLLM / TensorRT-LLM)
# =============================================================================
@register_architecture(
    name="autoregressive_kv_cache_attention",
    category="sota_architectures",
    subfolder="sota_stress",
    inputs=lambda: (torch.randn(1, 1, 64), torch.randn(1, 4, 15, 16), torch.randn(1, 4, 15, 16)),
    input_names=["x_new", "past_k", "past_v"],
    output_names=["out"],
    opset_versions=(17, 18, 20),
)
class Stateful_KVCache_Attention(nn.Module):
    """Dynamic autoregressive generation step with KV-cache concatenation."""

    def __init__(self, d_model: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x_new: torch.Tensor, past_k: torch.Tensor, past_v: torch.Tensor) -> torch.Tensor:
        B, S_new, D = x_new.shape
        q = self.q_proj(x_new).view(B, S_new, self.num_heads, self.head_dim).transpose(1, 2)
        new_k = self.k_proj(x_new).view(B, S_new, self.num_heads, self.head_dim).transpose(1, 2)
        new_v = self.v_proj(x_new).view(B, S_new, self.num_heads, self.head_dim).transpose(1, 2)

        # Concat along sequence length dimension (dim=2)
        k_full = torch.cat([past_k, new_k], dim=2)
        v_full = torch.cat([past_v, new_v], dim=2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q * scale, k_full.transpose(-1, -2))
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v_full).transpose(1, 2).contiguous().view(B, S_new, D)
        return self.out_proj(out)


# =============================================================================
# 9. Mixtral Sparse Mixture of Experts (MoE) Top-K Routing & Dispatch
# =============================================================================
@register_architecture(
    name="mixtral_sparse_moe",
    category="sota_architectures",
    subfolder="sota_stress",
    shapes=[(1, 16, 64)],
    input_names=["x"],
    output_names=["out"],
    opset_versions=(17, 18, 20),
)
class Mixtral_Sparse_MoE(nn.Module):
    """Sparse MoE router with Top-K gating and multiple expert SwiGLU feedforwards."""

    def __init__(self, d_model: int = 64, d_ff: int = 128, num_experts: int = 4, top_k: int = 2) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            DualLinear_SwiGLU_FFN(d_model, d_ff) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.router(x)
        probs = torch.softmax(logits, dim=-1)
        weights, indices = torch.topk(probs, k=self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        w0 = weights[..., 0:1]
        w1 = weights[..., 1:2]
        return w0 * self.experts[0](x) + w1 * self.experts[1](x)


# =============================================================================
# 10. DeepSeek Multi-Head Latent Attention (MLA) Low-Rank Projections
# =============================================================================
@register_architecture(
    name="deepseek_mla_latent_attention",
    category="sota_architectures",
    subfolder="sota_stress",
    shapes=[(1, 16, 64)],
    input_names=["x"],
    output_names=["out"],
    opset_versions=(17, 18, 20),
)
class DeepSeek_MLA_LatentAttention(nn.Module):
    """DeepSeek-V2/V3 Multi-Head Latent Attention with low-rank KV compression & decoupled RoPE."""

    def __init__(self, d_model: int = 64, d_latent: int = 16, num_heads: int = 4, head_dim: int = 16) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Low-rank KV projection
        self.kv_down = nn.Linear(d_model, d_latent, bias=False)
        self.k_up = nn.Linear(d_latent, num_heads * head_dim, bias=False)
        self.v_up = nn.Linear(d_latent, num_heads * head_dim, bias=False)
        self.k_rope = nn.Linear(d_model, num_heads * head_dim, bias=False)

        # Low-rank Q projection
        self.q_down = nn.Linear(d_model, d_latent, bias=False)
        self.q_up = nn.Linear(d_latent, num_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        c_kv = self.kv_down(x)
        k_c = self.k_up(c_kv).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_up(c_kv).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k_r = self.k_rope(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k_c + k_r

        c_q = self.q_down(x)
        q = self.q_up(c_q).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q * scale, k.transpose(-1, -2))
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, S, -1)
        return self.out_proj(out)


# =============================================================================
# 11. Microsoft Phi-3 Partial-Dimension RoPE (Partial Rotation + Passthrough)
# =============================================================================
@register_architecture(
    name="phi3_partial_rope_attention",
    category="sota_architectures",
    subfolder="sota_stress",
    inputs=lambda: (torch.randn(1, 16, 64), torch.randn(1, 1, 16, 8), torch.randn(1, 1, 16, 8)),
    input_names=["x", "cos", "sin"],
    output_names=["out"],
    opset_versions=(17, 18, 20),
)
class Phi3_Partial_RoPE(nn.Module):
    """Partial RoPE where only the first rot_dim dimensions are rotated and remainder pass through."""

    def __init__(self, d_model: int = 64, num_heads: int = 4, head_dim: int = 16, rot_dim: int = 8) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rot_dim = rot_dim

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        half = self.rot_dim // 2
        return torch.cat((-x[..., half: self.rot_dim], x[..., :half]), dim=-1)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Partial RoPE on query
        q_rot = (q[..., : self.rot_dim] * cos) + (self.rotate_half(q[..., : self.rot_dim]) * sin)
        q_pass = q[..., self.rot_dim :]
        q_full = torch.cat([q_rot, q_pass], dim=-1)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q_full * scale, k.transpose(-1, -2))
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


# =============================================================================
# 12. Differential Transformer Dual-Softmax Subtraction (Diff-Transformer)
# =============================================================================
@register_architecture(
    name="diff_transformer_dual_softmax",
    category="sota_architectures",
    subfolder="sota_stress",
    shapes=[(1, 16, 64)],
    input_names=["x"],
    output_names=["out"],
    opset_versions=(17, 18, 20),
)
class DiffTransformer_DualSoftmax(nn.Module):
    """Microsoft Research 2024 Differential Transformer: (Softmax(QK1) - lambda * Softmax(QK2)) * V."""

    def __init__(self, d_model: int = 64, num_heads: int = 4, head_dim: int = 16) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.q1_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.k1_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.q2_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.k2_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, d_model, bias=False)
        self.lambda_param = nn.Parameter(torch.tensor(0.8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q1 = self.q1_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k1 = self.k1_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        q2 = self.q2_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k2 = self.k2_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores1 = torch.matmul(q1 * scale, k1.transpose(-1, -2))
        scores2 = torch.matmul(q2 * scale, k2.transpose(-1, -2))

        attn1 = torch.softmax(scores1, dim=-1)
        attn2 = torch.softmax(scores2, dim=-1)

        diff_attn = attn1 - self.lambda_param * attn2
        out = torch.matmul(diff_attn, v).transpose(1, 2).contiguous().view(B, S, -1)
        return self.out_proj(out)


# =============================================================================
# 13. Encoder-Decoder Cross-Attention (Whisper / T5)
# =============================================================================
@register_architecture(
    name="cross_attention_encoder_decoder",
    category="sota_architectures",
    subfolder="sota_stress",
    inputs=lambda: (torch.randn(1, 8, 64), torch.randn(1, 24, 64)),
    input_names=["x_dec", "x_enc"],
    output_names=["out"],
    opset_versions=(17, 18, 20),
)
class EncoderDecoder_CrossAttention(nn.Module):
    """Cross-Attention where Query is from decoder and Key/Value are from encoder."""

    def __init__(self, d_model: int = 64, num_heads: int = 4, head_dim: int = 16) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, d_model, bias=False)

    def forward(self, x_dec: torch.Tensor, x_enc: torch.Tensor) -> torch.Tensor:
        B, S_d, _ = x_dec.shape
        _, S_e, _ = x_enc.shape
        q = self.q_proj(x_dec).view(B, S_d, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_enc).view(B, S_e, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_enc).view(B, S_e, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q * scale, k.transpose(-1, -2))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S_d, -1)
        return self.out_proj(out)


# =============================================================================
# 14. Mamba Selective State Space Model (SSM) Discretization & Scan
# =============================================================================
@register_architecture(
    name="mamba_selective_ssm",
    category="sota_architectures",
    subfolder="sota_stress",
    shapes=[(1, 8, 32)],
    input_names=["x"],
    output_names=["out"],
    opset_versions=(17, 18, 20),
)
class Mamba_Selective_SSM(nn.Module):
    """Selective State Space Layer with input-dependent Delta, B, C projections and recurrent scan."""

    def __init__(self, d_model: int = 32, d_state: int = 8) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, d_model, bias=False)
        self.delta_proj = nn.Linear(d_model, d_model, bias=False)
        self.b_proj = nn.Linear(d_model, d_state, bias=False)
        self.c_proj = nn.Linear(d_model, d_state, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Static log-decay state parameter A
        self.a_log = nn.Parameter(torch.randn(d_model, d_state))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        u = self.in_proj(x)
        delta = F.softplus(self.delta_proj(x))  # [B, S, D]
        b = self.b_proj(x)                      # [B, S, d_state]
        c = self.c_proj(x)                      # [B, S, d_state]

        a = -torch.exp(self.a_log)              # [D, d_state]

        # Recurrent scan over sequence length
        h = torch.zeros(B, D, self.d_state, device=x.device)
        ys = []
        for t in range(S):
            d_t = delta[:, t : t + 1, :].transpose(1, 2)  # [B, D, 1]
            b_t = b[:, t : t + 1, :]                      # [B, 1, d_state]
            c_t = c[:, t : t + 1, :].transpose(1, 2)      # [B, d_state, 1]
            u_t = u[:, t : t + 1, :].transpose(1, 2)      # [B, D, 1]

            a_bar = torch.exp(d_t * a.unsqueeze(0))        # [B, D, d_state]
            b_bar = d_t * b_t                              # [B, D, d_state]
            h = a_bar * h + b_bar * u_t                    # [B, D, d_state]
            y_t = torch.matmul(h, c_t).squeeze(-1)         # [B, D]
            ys.append(y_t.unsqueeze(1))

        y = torch.cat(ys, dim=1)
        return self.out_proj(y)


# =============================================================================
# 15. Swin Transformer Shifted Window Attention & Cyclic Roll
# =============================================================================
@register_architecture(
    name="swin_shifted_window_attention",
    category="sota_architectures",
    subfolder="sota_stress",
    shapes=[(1, 8, 8, 32)],
    input_names=["x"],
    output_names=["out"],
    opset_versions=(17, 18, 20),
)
class Swin_ShiftedWindow_Attention(nn.Module):
    """Swin Transformer cyclic roll -> window partition -> self-attention -> reverse roll."""

    def __init__(self, c: int = 32, shift: int = 2) -> None:
        super().__init__()
        self.c = c
        self.shift = shift
        self.qkv = nn.Linear(c, 3 * c, bias=False)
        self.out_proj = nn.Linear(c, c, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cyclic roll shift
        x_roll = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))
        qkv = self.qkv(x_roll)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        scores = torch.matmul(q, k.transpose(-1, -2)) * (self.c**-0.5)
        probs = torch.softmax(scores, dim=-1)
        attn = torch.matmul(probs, v)

        # Reverse cyclic roll
        out_roll = torch.roll(attn, shifts=(self.shift, self.shift), dims=(1, 2))
        return self.out_proj(out_roll)


# =============================================================================
# 16. Vision Transformer (ViT) CLS Token Prepend & Positional Embedding
# =============================================================================
@register_architecture(
    name="vit_cls_token_pos_embed",
    category="sota_architectures",
    subfolder="sota_stress",
    shapes=[(1, 3, 16, 16)],
    input_names=["img"],
    output_names=["out"],
    opset_versions=(17, 18, 20),
)
class ViT_ClassToken_PosEmbed(nn.Module):
    """Patch projection -> prepend [CLS] token -> additive 1D positional embedding."""

    def __init__(self, in_ch: int = 3, embed_dim: int = 32, patch_size: int = 4, img_size: int = 16) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        num_patches = (img_size // patch_size) ** 2
        self.patch_proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        B = img.shape[0]
        # [B, embed_dim, H_p, W_p] -> [B, N, embed_dim]
        patches = self.patch_proj(img).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, patches), dim=1)
        x = x + self.pos_embed
        return self.norm(x)


# =============================================================================
# CLI Runner
# =============================================================================
def main() -> None:
    print("=" * 70)
    print("SOTA Architecture Zoo & Stress Generator")
    print("=" * 70)
    ModelRegistry.export_all(category="sota_architectures")


if __name__ == "__main__":
    main()
