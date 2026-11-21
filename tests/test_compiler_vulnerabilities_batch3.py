"""Exhaustive Adversarial Architecture Stress Tests & Compiler Vulnerabilities (Batch 3).

Generated using real PyTorch modules from the SOTA Generator Architecture Zoo
(`generators.gen_sota_stress_zoo`):
Tests complex real-world modern deep learning architectures where traditional
ONNX simplification and existing neuron-toolkit pattern matchers fail:
1. ALiBi Additive Relative Position Bias Attention (MPT / Bloom)
2. HuggingFace Split-Concat Rotate-Half RoPE (LLaMA / Mistral / Gemma / Qwen)
3. Dual-Linear SwiGLU FeedForward (LLaMA / Mistral / Gemma / Qwen)
4. QK-Norm Head Normalization Attention (Gemma-2 / Qwen-2.5 / Swin / ViT-22B)
5. Sliding Window & Dynamic Causal Boolean Masked Attention (Mistral 7B)
6. Parallel Attention + MLP with Shared Norm (Falcon / PaLM)
7. Grouped Query Attention (GQA) Head Broadcast Expand Memory Waste
8. Autoregressive Inference KV-Cache Concat Attention Fusion Failure
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from generators.gen_sota_stress_zoo import (
    ALiBi_Attention,
    DualLinear_SwiGLU_FFN,
    Falcon_Parallel_Attention_MLP,
    GroupedQueryAttention_GQA,
    HFRotateHalf_RoPE,
    QKNorm_Attention,
    SlidingWindow_CausalWhere_Attention,
    Stateful_KVCache_Attention,
    export_model_to_onnx,
)
from neuron_toolkit.compiler.pipeline import OptimizationPipeline


# ==============================================================================
# VULNERABILITY 39: [ALiBi BLOCKS SDPA] Additive Position Bias Prevents Attention Fusion
# ==============================================================================
def test_vuln_39_alibi_additive_relative_position_bias_blocks_sdpa():
    """Vulnerability 39: ALiBi relative position bias must fuse to SDPA.

    Model: ALiBi_Attention (MPT / BLOOM) generated via generators.gen_sota_stress_zoo
    FAILS WITH: AssertionError: ScaledDotProductAttention was not synthesized; ALiBi bias blocked fusion.
    """
    model = ALiBi_Attention(d_model=64, num_heads=4)
    dummy_x = torch.randn(1, 16, 64)
    proto = export_model_to_onnx(model, dummy_x, ["x"], ["out"], opset_version=17)

    pipeline = OptimizationPipeline()
    opt_m = pipeline.run(proto)

    sdpa_ops = [n for n in opt_m.graph.node if n.op_type in ("ScaledDotProductAttention", "SDPA", "MultiHeadAttention")]
    assert len(sdpa_ops) >= 1, (
        "SPEC DEFICIENCY: ALiBi relative position bias prevented ScaledDotProductAttention fusion! "
        f"Remaining ops: {[n.op_type for n in opt_m.graph.node]}"
    )


# ==============================================================================
# VULNERABILITY 40: [HUGGINGFACE ROPE IGNORED] Split-Concat Rotate-Half RoPE Not Recognized
# ==============================================================================
def test_vuln_40_huggingface_split_concat_rotate_half_rope_ignored():
    """Vulnerability 40: HuggingFace style rotate_half RoPE must fuse to RotaryEmbedding.

    Model: HFRotateHalf_RoPE (LLaMA-3 / Mistral) generated via generators.gen_sota_stress_zoo
    FAILS WITH: AssertionError: RotaryEmbedding was not synthesized for HuggingFace rotate_half RoPE.
    """
    d = 32
    model = HFRotateHalf_RoPE(d=d)
    x = torch.randn(1, 4, 16, d)
    cos = torch.randn(1, 1, 16, d)
    sin = torch.randn(1, 1, 16, d)
    proto = export_model_to_onnx(model, (x, cos, sin), ["x", "cos", "sin"], ["out"], opset_version=17)

    pipeline = OptimizationPipeline()
    opt_m = pipeline.run(proto)

    rope_ops = [n for n in opt_m.graph.node if n.op_type == "RotaryEmbedding"]
    assert len(rope_ops) >= 1, (
        "SPEC DEFICIENCY: HuggingFace rotate_half RoPE was completely ignored by RotaryEmbedding fusion! "
        f"0% reduction, remaining ops: {[n.op_type for n in opt_m.graph.node]}"
    )


# ==============================================================================
# VULNERABILITY 41: [DUAL-LINEAR SWIGLU] Separate Gate/Up Projections Fail Fusion
# ==============================================================================
def test_vuln_41_dual_linear_swiglu_mlp_unfused():
    """Vulnerability 41: Dual-Linear SwiGLU MLP must fuse to FeedForward_SwiGLU.

    Model: DualLinear_SwiGLU_FFN (LLaMA-3 / Mistral) generated via generators.gen_sota_stress_zoo
    FAILS WITH: AssertionError: FeedForward_SwiGLU was not synthesized for dual-projection MLP.
    """
    model = DualLinear_SwiGLU_FFN(d_model=64, d_ff=128)
    x = torch.randn(1, 16, 64)
    proto = export_model_to_onnx(model, x, ["x"], ["out"], opset_version=17)

    pipeline = OptimizationPipeline()
    opt_m = pipeline.run(proto)

    ffn_ops = [n for n in opt_m.graph.node if n.op_type.startswith("FeedForward")]
    assert len(ffn_ops) >= 1, (
        "SPEC DEFICIENCY: Dual-Linear SwiGLU MLP left separate linear and activation layers! "
        f"Remaining ops: {[n.op_type for n in opt_m.graph.node]}"
    )


# ==============================================================================
# VULNERABILITY 42: [QK-NORM ATTENTION] Head Normalization Blocks Macro-Fusion
# ==============================================================================
def test_vuln_42_qk_norm_attention_blocks_macro_attention_fusion():
    """Vulnerability 42: QK head normalization must not prevent MultiHeadAttention macro-fusion.

    Model: QKNorm_Attention (Gemma-2 / Qwen-2.5) generated via generators.gen_sota_stress_zoo
    FAILS WITH: AssertionError: MultiHeadAttention macro-block was not synthesized.
    """
    model = QKNorm_Attention(d_model=64, num_heads=4)
    x = torch.randn(1, 16, 64)
    proto = export_model_to_onnx(model, x, ["x"], ["out"], opset_version=17)

    pipeline = OptimizationPipeline()
    opt_m = pipeline.run(proto)

    # Contract: Whole attention block must fuse into a macro-block (MultiHeadAttention or GatedAttention)
    macro_ops = [n for n in opt_m.graph.node if n.op_type in ("MultiHeadAttention", "GatedAttention")]
    assert len(macro_ops) >= 1, (
        "SPEC DEFICIENCY: Per-head QK-normalization blocked macro MultiHeadAttention fusion! "
        f"Remaining disjoint ops: {[n.op_type for n in opt_m.graph.node]}"
    )


# ==============================================================================
# VULNERABILITY 43: [SLIDING WINDOW / CAUSAL WHERE] Dynamic Mask Blocks Attention
# ==============================================================================
def test_vuln_43_sliding_window_causal_mask_where_blocks_sdpa():
    """Vulnerability 43: Dynamic causal mask applied via Where operator must fuse into SDPA.

    Model: SlidingWindow_CausalWhere_Attention (Mistral 7B) generated via generators.gen_sota_stress_zoo
    FAILS WITH: AssertionError: ScaledDotProductAttention was not synthesized; Where operator blocked fusion.
    """
    model = SlidingWindow_CausalWhere_Attention(d_model=64, num_heads=4)
    x = torch.randn(1, 16, 64)
    causal_mask = torch.tril(torch.ones(1, 1, 16, 16)).bool()
    proto = export_model_to_onnx(model, (x, causal_mask), ["x", "causal_mask"], ["out"], opset_version=17)

    pipeline = OptimizationPipeline()
    opt_m = pipeline.run(proto)

    sdpa_ops = [n for n in opt_m.graph.node if n.op_type in ("ScaledDotProductAttention", "SDPA", "MultiHeadAttention")]
    assert len(sdpa_ops) >= 1, (
        "SPEC DEFICIENCY: Dynamic causal Where mask blocked ScaledDotProductAttention fusion! "
        f"Remaining ops: {[n.op_type for n in opt_m.graph.node]}"
    )


# ==============================================================================
# VULNERABILITY 44: [PARALLEL BLOCK] Attention + MLP Macro Fusion
# ==============================================================================
def test_vuln_44_parallel_attention_mlp_shared_norm_double_computation():
    """Vulnerability 44: Parallel Attention+MLP block must fuse into a parallel macro-block.

    Model: Falcon_Parallel_Attention_MLP (Falcon-40B / PaLM-2) generated via generators.gen_sota_stress_zoo
    FAILS WITH: AssertionError: Parallel Attention+MLP left 38 un-fused ops.
    """
    model = Falcon_Parallel_Attention_MLP(d_model=64, num_heads=4, d_ff=128)
    x = torch.randn(1, 16, 64)
    proto = export_model_to_onnx(model, x, ["x"], ["out"], opset_version=17)

    pipeline = OptimizationPipeline()
    opt_m = pipeline.run(proto)

    # Contract: Parallel attention + MLP block should collapse to a unified TransformerBlock or <= 10 ops
    assert len(opt_m.graph.node) < 15, (
        "SPEC DEFICIENCY: Parallel Attention+MLP block failed macro simplification! "
        f"Remaining ops ({len(opt_m.graph.node)}): {[n.op_type for n in opt_m.graph.node]}"
    )


# ==============================================================================
# VULNERABILITY 45: [GQA BROADCAST EXPAND] KV Head Expand Memory Waste
# ==============================================================================
def test_vuln_45_gqa_kv_head_broadcast_expand_unfolded_memory_waste():
    """Vulnerability 45: GQA KV head broadcast expand must fold into GroupedQueryAttention op.

    Model: GroupedQueryAttention_GQA (LLaMA-3 8B/70B) generated via generators.gen_sota_stress_zoo
    FAILS WITH: AssertionError: Redundant Expand/Reshape broadcast ops remain in graph.
    """
    model = GroupedQueryAttention_GQA(d_model=64, num_q_heads=8, num_kv_heads=2)
    x = torch.randn(1, 16, 64)
    proto = export_model_to_onnx(model, x, ["x"], ["out"], opset_version=17)

    pipeline = OptimizationPipeline()
    opt_m = pipeline.run(proto)

    # Contract: GQA broadcast Expand ops should be eliminated by native GQA stride handling
    expand_ops = [n for n in opt_m.graph.node if n.op_type == "Expand"]
    gqa_ops = [n for n in opt_m.graph.node if n.op_type == "GroupedQueryAttention"]
    assert len(expand_ops) == 0 and len(gqa_ops) >= 1, (
        f"SPEC DEFICIENCY: Grouped Query Attention did not eliminate KV head Expand ops! "
        f"Expand ops left: {len(expand_ops)}, GQA op synthesized: {len(gqa_ops)}, remaining ops: {[n.op_type for n in opt_m.graph.node]}"
    )


# ==============================================================================
# VULNERABILITY 46: [STATEFUL KV-CACHE] Concat Attention Fusion Failure
# ==============================================================================
def test_vuln_46_stateful_kv_cache_concat_blocks_attention_fusion():
    """Vulnerability 46: Autoregressive KV-cache concatenation must fuse into macro-attention.

    Model: Stateful_KVCache_Attention (vLLM / TensorRT-LLM) generated via generators.gen_sota_stress_zoo
    FAILS WITH: AssertionError: Stateful KV-cache concat left 15 un-fused ops.
    """
    model = Stateful_KVCache_Attention(d_model=64, num_heads=4)
    x_new = torch.randn(1, 1, 64)
    past_k = torch.randn(1, 4, 15, 16)
    past_v = torch.randn(1, 4, 15, 16)
    proto = export_model_to_onnx(model, (x_new, past_k, past_v), ["x_new", "past_k", "past_v"], ["out"], opset_version=17)

    pipeline = OptimizationPipeline()
    opt_m = pipeline.run(proto)

    # Contract: Stateful Attention layer with autoregressive KV-cache concat should fuse into a single
    # macro-block (StatefulAttention or MultiHeadAttention) rather than leaving 15 un-fused ops
    macro_ops = [n for n in opt_m.graph.node if n.op_type in ("StatefulAttention", "MultiHeadAttention")]
    assert len(macro_ops) >= 1, (
        "SPEC DEFICIENCY: Stateful KV-cache concatenation blocked macro-attention fusion! "
        f"Remaining ops ({len(opt_m.graph.node)}): {[n.op_type for n in opt_m.graph.node]}"
    )
