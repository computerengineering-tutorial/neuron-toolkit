"""Exhaustive Adversarial Architecture Stress Tests & Compiler Vulnerabilities (Batch 4).

Generated using real PyTorch modules from the SOTA Generator Architecture Zoo
(`generators.gen_sota_stress_zoo`):
Tests deep, complex architectural innovations across modern frontier models
where ONNX graph simplification and existing neuron-toolkit synthesis passes fail:
1. Mixtral Sparse Mixture of Experts (MoE) Top-K Routing & Expert Dispatch
2. DeepSeek-V2/V3 Multi-Head Latent Attention (MLA) Decoupled RoPE & Low-Rank Keys
3. Microsoft Phi-3 Partial-Dimension RoPE (Partial Head Rotation & Passthrough)
4. Microsoft Research Differential Transformer Dual-Softmax Subtraction
5. Encoder-Decoder Cross-Attention (Whisper / T5 / Transformer Decoders)
6. Mamba Selective State Space Model (SSM) Discretization & Recurrent Scan Bloat
7. Swin Transformer Shifted Window Attention & 2D Cyclic Roll Decompositions
8. Vision Transformer (ViT) CLS Token Prepend & Positional Embedding Folding
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from generators.gen_sota_stress_zoo import (
    DeepSeek_MLA_LatentAttention,
    DiffTransformer_DualSoftmax,
    EncoderDecoder_CrossAttention,
    Mamba_Selective_SSM,
    Mixtral_Sparse_MoE,
    Phi3_Partial_RoPE,
    Swin_ShiftedWindow_Attention,
    ViT_ClassToken_PosEmbed,
    export_model_to_onnx,
)
from neuron_toolkit.compiler.pipeline import OptimizationPipeline


# ==============================================================================
# VULNERABILITY 47: [SPARSE MOE ROUTING] Top-K Router & SwiGLU Expert Dispatch Unfused
# ==============================================================================
def test_vuln_47_mixtral_sparse_moe_unfused_dispatch():
    """Vulnerability 47: Sparse MoE router with Top-K gating must fuse into a unified MoEBlock.

    Model: Mixtral_Sparse_MoE (Mixtral 8x7B / DeepSeek-MoE) generated via generators.gen_sota_stress_zoo
    FAILS WITH: AssertionError: MoEBlock was not synthesized; 22 disjoint operators remain.
    """
    model = Mixtral_Sparse_MoE(d_model=64, d_ff=128, num_experts=4, top_k=2)
    x = torch.randn(1, 16, 64)
    proto = export_model_to_onnx(model, x, ["x"], ["out"], opset_version=17)

    pipeline = OptimizationPipeline()
    opt_m = pipeline.run(proto)

    moe_ops = [n for n in opt_m.graph.node if n.op_type in ("MoEBlock", "SparseMoeBlock")]
    assert len(moe_ops) >= 1, (
        "SPEC DEFICIENCY: Sparse Mixture of Experts router and expert dispatch were not synthesized into an MoEBlock! "
        f"22 disjoint ops remain: {[n.op_type for n in opt_m.graph.node]}"
    )


# ==============================================================================
# VULNERABILITY 48: [DEEPSEEK MLA] Latent Attention Low-Rank Projections Incomplete
# ==============================================================================
def test_vuln_48_deepseek_mla_latent_attention_unfused():
    """Vulnerability 48: Multi-Head Latent Attention must fuse into a unified MLA macro-block.

    Model: DeepSeek_MLA_LatentAttention (DeepSeek-V2 / V3) generated via generators.gen_sota_stress_zoo
    FAILS WITH: AssertionError: MultiHeadLatentAttention macro-block was not synthesized; 17 ops remain.
    """
    model = DeepSeek_MLA_LatentAttention(d_model=64, d_latent=16, num_heads=4, head_dim=16)
    x = torch.randn(1, 16, 64)
    proto = export_model_to_onnx(model, x, ["x"], ["out"], opset_version=17)

    pipeline = OptimizationPipeline()
    opt_m = pipeline.run(proto)

    # Contract: Whole MLA layer should collapse into a single MultiHeadLatentAttention macro-block
    assert len(opt_m.graph.node) < 10, (
        "SPEC DEFICIENCY: DeepSeek Multi-Head Latent Attention failed macro synthesis! "
        f"17 disjoint ops remain: {[n.op_type for n in opt_m.graph.node]}"
    )


# ==============================================================================
# VULNERABILITY 49: [PHI-3 PARTIAL ROPE] Sliced Partial-Dimension RoPE Ignored
# ==============================================================================
def test_vuln_49_phi3_partial_rope_ignored_by_compiler():
    """Vulnerability 49: Partial-dimension RoPE (first half rotated, second half passthrough) must fuse.

    Model: Phi3_Partial_RoPE (Phi-3 / ChatGLM) generated via generators.gen_sota_stress_zoo
    FAILS WITH: AssertionError: RotaryEmbedding was not synthesized; 38 ops remain un-fused.
    """
    d_model, num_heads, head_dim, rot_dim = 64, 4, 16, 8
    model = Phi3_Partial_RoPE(d_model, num_heads, head_dim, rot_dim)
    x = torch.randn(1, 16, d_model)
    cos = torch.randn(1, 1, 16, rot_dim)
    sin = torch.randn(1, 1, 16, rot_dim)
    proto = export_model_to_onnx(model, (x, cos, sin), ["x", "cos", "sin"], ["out"], opset_version=17)

    pipeline = OptimizationPipeline()
    opt_m = pipeline.run(proto)

    rope_ops = [n for n in opt_m.graph.node if "Rotary" in n.op_type or "RoPE" in n.op_type]
    assert len(rope_ops) >= 1, (
        "SPEC DEFICIENCY: Partial-dimension RoPE was completely ignored by RotaryEmbedding fusion! "
        f"38 ops remain un-fused: {[n.op_type for n in opt_m.graph.node]}"
    )


# ==============================================================================
# VULNERABILITY 50: [DIFF-TRANSFORMER] Dual Softmax Subtraction Blocks SDPA
# ==============================================================================
def test_vuln_50_diff_transformer_dual_softmax_blocks_sdpa():
    """Vulnerability 50: Differential Transformer dual-softmax subtraction must fuse to DifferentialAttention.

    Model: DiffTransformer_DualSoftmax (Microsoft Research 2024) generated via generators.gen_sota_stress_zoo
    FAILS WITH: AssertionError: ScaledDotProductAttention was not synthesized; 30 ops remain.
    """
    model = DiffTransformer_DualSoftmax(d_model=64, num_heads=4, head_dim=16)
    x = torch.randn(1, 16, 64)
    proto = export_model_to_onnx(model, x, ["x"], ["out"], opset_version=17)

    pipeline = OptimizationPipeline()
    opt_m = pipeline.run(proto)

    diff_ops = [n for n in opt_m.graph.node if "Diff" in n.op_type or "Attention" in n.op_type]
    assert len(diff_ops) >= 1, (
        "SPEC DEFICIENCY: Differential dual-softmax subtraction prevented attention fusion! "
        f"30 disjoint ops remain: {[n.op_type for n in opt_m.graph.node]}"
    )


# ==============================================================================
# VULNERABILITY 51: [CROSS-ATTENTION] Disjoint Query / Key-Value Input Projections
# ==============================================================================
def test_vuln_51_cross_attention_encoder_decoder_unfused_projections():
    """Vulnerability 51: Encoder-Decoder cross-attention must fuse into a unified CrossAttention block.

    Model: EncoderDecoder_CrossAttention (Whisper / T5) generated via generators.gen_sota_stress_zoo
    FAILS WITH: AssertionError: CrossAttention block was not synthesized; 14 disjoint ops remain.
    """
    model = EncoderDecoder_CrossAttention(d_model=64, num_heads=4, head_dim=16)
    x_dec = torch.randn(1, 8, 64)
    x_enc = torch.randn(1, 24, 64)
    proto = export_model_to_onnx(model, (x_dec, x_enc), ["x_dec", "x_enc"], ["out"], opset_version=17)

    pipeline = OptimizationPipeline()
    opt_m = pipeline.run(proto)

    cross_ops = [n for n in opt_m.graph.node if n.op_type in ("CrossAttention", "MultiHeadCrossAttention")]
    assert len(cross_ops) >= 1, (
        "SPEC DEFICIENCY: Cross-attention projections across decoder/encoder inputs failed macro fusion! "
        f"14 ops remain disjoint: {[n.op_type for n in opt_m.graph.node]}"
    )


# ==============================================================================
# VULNERABILITY 52: [MAMBA SSM BLOAT] Unfolded Recurrent Scan Creates Huge Graphs
# ==============================================================================
def test_vuln_52_mamba_selective_ssm_unfolded_scan_bloat():
    """Vulnerability 52: Mamba selective SSM recurrent loop must fuse into a native MambaSSM kernel.

    Model: Mamba_Selective_SSM (Mamba / S6) generated via generators.gen_sota_stress_zoo
    FAILS WITH: AssertionError: MambaSSM was not synthesized; 148 unrolled ops remain.
    """
    model = Mamba_Selective_SSM(d_model=32, d_state=8)
    x = torch.randn(1, 8, 32)
    proto = export_model_to_onnx(model, x, ["x"], ["out"], opset_version=17)

    pipeline = OptimizationPipeline()
    opt_m = pipeline.run(proto)

    mamba_ops = [n for n in opt_m.graph.node if "Mamba" in n.op_type or "SSM" in n.op_type]
    assert len(mamba_ops) >= 1, (
        "SPEC DEFICIENCY: Mamba selective state space scan was not synthesized into MambaSSM! "
        f"148 unrolled ops remain in graph: {[n.op_type for n in opt_m.graph.node][:10]}..."
    )


# ==============================================================================
# VULNERABILITY 53: [SWIN WINDOW ATTENTION] 2D Cyclic Roll Decompositions Un-folded
# ==============================================================================
def test_vuln_53_swin_shifted_window_roll_scattered_nodes():
    """Vulnerability 53: Swin cyclic roll and window attention must fuse into ShiftedWindowAttention.

    Model: Swin_ShiftedWindow_Attention (Swin Transformer) generated via generators.gen_sota_stress_zoo
    FAILS WITH: AssertionError: ShiftedWindowAttention was not synthesized; 31 ops remain.
    """
    model = Swin_ShiftedWindow_Attention(c=32, shift=2)
    x = torch.randn(1, 8, 8, 32)
    proto = export_model_to_onnx(model, x, ["x"], ["out"], opset_version=17)

    pipeline = OptimizationPipeline()
    opt_m = pipeline.run(proto)

    swin_ops = [n for n in opt_m.graph.node if "ShiftedWindow" in n.op_type or "Swin" in n.op_type]
    assert len(swin_ops) >= 1, (
        "SPEC DEFICIENCY: Swin cyclic roll and window partitioning failed macro synthesis! "
        f"31 ops remain: {[n.op_type for n in opt_m.graph.node]}"
    )


# ==============================================================================
# VULNERABILITY 54: [VIT CLS TOKEN PREPEND] CLS Token Concat & Pos Embed Unfused
# ==============================================================================
def test_vuln_54_vit_cls_token_concat_pos_embed_unfused():
    """Vulnerability 54: Vision Transformer patch projection + CLS token + pos_embed must fuse into ViTPatchEmbedding.

    Model: ViT_ClassToken_PosEmbed (Vision Transformer) generated via generators.gen_sota_stress_zoo
    FAILS WITH: AssertionError: ViTPatchEmbedding was not synthesized; 19 ops remain.
    """
    model = ViT_ClassToken_PosEmbed(in_ch=3, embed_dim=32, patch_size=4, img_size=16)
    img = torch.randn(1, 3, 16, 16)
    proto = export_model_to_onnx(model, img, ["img"], ["out"], opset_version=17)

    pipeline = OptimizationPipeline()
    opt_m = pipeline.run(proto)

    vit_ops = [n for n in opt_m.graph.node if "ViT" in n.op_type or "PatchEmbedding" in n.op_type]
    assert len(vit_ops) >= 1, (
        "SPEC DEFICIENCY: ViT patch convolution, CLS token prepend, and positional embedding failed synthesis! "
        f"19 ops remain: {[n.op_type for n in opt_m.graph.node]}"
    )
