"""Comprehensive Compiler Vulnerability & Adversarial Stress Suite (Batch 6).

Adversarial, topological, and quantization-aware edge-case tests designed to expose
subtle optimizer vulnerabilities, graph fragmentation, pointer aliasing, cross-block
tensor starvation, and nested block decomposition issues.
"""

from __future__ import annotations

import collections
import copy
import os
import tempfile
import numpy as np
import onnx
from onnx import helper, numpy_helper
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuron_toolkit.compiler.base import CompilerContext
from neuron_toolkit.compiler.pipeline import OptimizationPipeline
from neuron_toolkit.compiler.passes.canonicalization.shape_canonicalizer import ShapeCanonicalizer
from neuron_toolkit.compiler.passes.canonicalization.static_folder import StaticFolder
from neuron_toolkit.compiler.passes.canonicalization.norm_canonicalizer import NormCanonicalizer
from neuron_toolkit.compiler.passes.synthesis.macro_block_synthesizer import (
    MacroBlockSynthesizer,
    _replace_nodes_topologically,
)
from neuron_toolkit.compiler.architectures.fingerprint import (
    compute_subgraph_fingerprint,
    build_monomorphized_op_type,
    ensure_specialized_function_proto,
)


def _strip_all_node_names(model: onnx.ModelProto) -> None:
    """Anonymize all nodes in graph and functions to stress-test purely topological passes."""
    for n in model.graph.node:
        n.name = ""
    for f in model.functions:
        for fn in f.node:
            fn.name = ""


def _export_model_to_onnx(model: nn.Module, dummy_input: torch.Tensor, **kwargs) -> onnx.ModelProto:
    """Export a PyTorch module cleanly to ONNX using modern torch.export / dynamo exporter."""
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        kwargs.setdefault("dynamo", True)
        kwargs.setdefault("opset_version", 18)
        if "dynamic_axes" in kwargs and kwargs.get("dynamo", True):
            dyn_axes = kwargs.pop("dynamic_axes")
            dim_cache = {}
            inp_axis = dyn_axes.get("input", dyn_axes.get("x", {}))
            if inp_axis:
                shape_spec = {}
                for ax_idx, ax_name in inp_axis.items():
                    if ax_name not in dim_cache:
                        dim_cache[ax_name] = torch.export.Dim(ax_name)
                    shape_spec[ax_idx] = dim_cache[ax_name]
                kwargs["dynamic_shapes"] = (shape_spec,)
        torch.onnx.export(model, dummy_input, tmp_path, **kwargs)
        return onnx.load(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# =============================================================================
# 1. FAN-OUT RESHAPE REWIRING & RESIDUAL HIJACKING VULNERABILITY
# =============================================================================
def test_fanout_reshape_rewiring_preserves_residual_topology():
    """Verify consecutive Reshape collapsing does not split residual connections when

    the intermediate tensor is consumed by both a downstream Reshape and an Add residual.
    """
    # X -> Reshape1 (interm) -> Reshape2 -> MatMul -> Add (residual merge with interm)
    X = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 16, 32])
    Y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 16, 32])

    sh1_init = numpy_helper.from_array(np.array([1, 16, 32], dtype=np.int64), name="sh1")
    sh2_init = numpy_helper.from_array(np.array([16, 32], dtype=np.int64), name="sh2")
    sh3_init = numpy_helper.from_array(np.array([1, 16, 32], dtype=np.int64), name="sh3")
    w_init = numpy_helper.from_array(np.random.randn(32, 32).astype(np.float32), name="W")

    r1 = helper.make_node("Reshape", inputs=["X", "sh1"], outputs=["interm"], name="r1")
    r2 = helper.make_node("Reshape", inputs=["interm", "sh2"], outputs=["r2_out"], name="r2")
    mm = helper.make_node("MatMul", inputs=["r2_out", "W"], outputs=["mm_out"], name="mm")
    r3 = helper.make_node("Reshape", inputs=["mm_out", "sh3"], outputs=["attn_out"], name="r3")
    res_add = helper.make_node("Add", inputs=["interm", "attn_out"], outputs=["Y"], name="res_add")

    graph = helper.make_graph(
        [r1, r2, mm, r3, res_add],
        "fanout_test",
        [X],
        [Y],
        [sh1_init, sh2_init, sh3_init, w_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    # Apply reshape collapsing
    canonicalizer = ShapeCanonicalizer()
    optimized = canonicalizer.apply_reshape_collapsing(copy.deepcopy(model))

    # interm MUST remain as producer for res_add and r2 if fanout was preserved
    res_node = next(n for n in optimized.graph.node if n.name == "res_add")
    assert "interm" in res_node.input, (
        f"Residual add connection was severed! Inputs are {res_node.input}"
    )


# =============================================================================
# 2. CROSS-BLOCK SHARED QUANTIZATION SCALE/ZERO-POINT STARVATION
# =============================================================================
def test_shared_quantization_scale_not_swallowed_by_first_block():
    """Verify that when a single scale/zero_point initializer is shared across multiple

    quantized blocks, fusing the first block does not starve downstream blocks.
    """
    # Block 1 and Block 2 both consume shared scale "scale_shared"
    X = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 32])
    Y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 32])

    scale = numpy_helper.from_array(np.float32(0.05), name="scale_shared")
    zp = numpy_helper.from_array(np.int8(0), name="zp_shared")
    w1 = numpy_helper.from_array(np.random.randint(-127, 127, (32, 32), dtype=np.int8), name="W1")
    w2 = numpy_helper.from_array(np.random.randint(-127, 127, (32, 32), dtype=np.int8), name="W2")

    # Block 1: DQL -> DQ -> MatMul
    dql1 = helper.make_node("DynamicQuantizeLinear", inputs=["X"], outputs=["q1", "s1", "z1"], name="dql1")
    dq1 = helper.make_node("DequantizeLinear", inputs=["q1", "s1", "z1"], outputs=["dq1_out"], name="dq1")
    dq_w1 = helper.make_node("DequantizeLinear", inputs=["W1", "scale_shared", "zp_shared"], outputs=["w1_fp32"], name="dqw1")
    mm1 = helper.make_node("MatMul", inputs=["dq1_out", "w1_fp32"], outputs=["b1_out"], name="mm1")

    # Block 2: DQL -> DQ -> MatMul consuming shared scale
    dql2 = helper.make_node("DynamicQuantizeLinear", inputs=["b1_out"], outputs=["q2", "s2", "z2"], name="dql2")
    dq2 = helper.make_node("DequantizeLinear", inputs=["q2", "s2", "z2"], outputs=["dq2_out"], name="dq2")
    dq_w2 = helper.make_node("DequantizeLinear", inputs=["W2", "scale_shared", "zp_shared"], outputs=["w2_fp32"], name="dqw2")
    mm2 = helper.make_node("MatMul", inputs=["dq2_out", "w2_fp32"], outputs=["Y"], name="mm2")

    graph = helper.make_graph(
        [dql1, dq1, dq_w1, mm1, dql2, dq2, dq_w2, mm2],
        "shared_scale_graph",
        [X],
        [Y],
        [scale, zp, w1, w2],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    synth = MacroBlockSynthesizer()
    optimized = synth.apply_dynamic_quant_linear_fusion(copy.deepcopy(model), mode="function_proto")

    # Ensure scale_shared initializer is still present in graph for subsequent consumers
    init_names = {i.name for i in optimized.graph.initializer}
    assert "scale_shared" in init_names, "scale_shared initializer was mistakenly dropped!"
    assert "zp_shared" in init_names, "zp_shared initializer was mistakenly dropped!"


# =============================================================================
# 3. CROSS-ATTENTION EXTERNAL ENCODER DAG LEAKAGE VULNERABILITY
# =============================================================================
def test_cross_attention_does_not_swallow_external_encoder_kv():
    """Verify that in Cross-Attention (where Q comes from decoder, K/V come from an encoder),

    synthesizing the decoder TransformerBlock does NOT swallow the encoder's nodes into the block.
    """
    enc_in = helper.make_tensor_value_info("enc_in", onnx.TensorProto.FLOAT, [1, 16, 32])
    dec_in = helper.make_tensor_value_info("dec_in", onnx.TensorProto.FLOAT, [1, 16, 32])
    dec_out = helper.make_tensor_value_info("dec_out", onnx.TensorProto.FLOAT, [1, 16, 32])

    w_enc = numpy_helper.from_array(np.random.randn(32, 32).astype(np.float32), name="W_enc")
    w_q = numpy_helper.from_array(np.random.randn(32, 32).astype(np.float32), name="W_q")
    w_k = numpy_helper.from_array(np.random.randn(32, 32).astype(np.float32), name="W_k")
    w_v = numpy_helper.from_array(np.random.randn(32, 32).astype(np.float32), name="W_v")
    w_out = numpy_helper.from_array(np.random.randn(32, 32).astype(np.float32), name="W_out")
    w_ffn = numpy_helper.from_array(np.random.randn(32, 32).astype(np.float32), name="W_ffn")

    # Encoder subgraph: enc_in -> MatMul -> Relu -> enc_feat
    mm_enc = helper.make_node("MatMul", inputs=["enc_in", "W_enc"], outputs=["enc_mm"], name="mm_enc")
    relu_enc = helper.make_node("Relu", inputs=["enc_mm"], outputs=["enc_feat"], name="relu_enc")

    # Decoder layer:
    # Q from dec_in, K & V from enc_feat
    q_node = helper.make_node("MatMul", inputs=["dec_in", "W_q"], outputs=["q"], name="q_node")
    k_node = helper.make_node("MatMul", inputs=["enc_feat", "W_k"], outputs=["k"], name="k_node")
    v_node = helper.make_node("MatMul", inputs=["enc_feat", "W_v"], outputs=["v"], name="v_node")

    # Attention core: Q @ K.T -> Softmax -> @ V -> out_proj
    trans_k = helper.make_node("Transpose", inputs=["k"], outputs=["k_t"], perm=[0, 2, 1], name="trans_k")
    qk = helper.make_node("MatMul", inputs=["q", "k_t"], outputs=["scores"], name="qk")
    sm = helper.make_node("Softmax", inputs=["scores"], outputs=["probs"], axis=-1, name="sm")
    ctx = helper.make_node("MatMul", inputs=["probs", "v"], outputs=["ctx_vec"], name="ctx")
    proj = helper.make_node("MatMul", inputs=["ctx_vec", "W_out"], outputs=["attn_out"], name="proj")
    add1 = helper.make_node("Add", inputs=["dec_in", "attn_out"], outputs=["add1_out"], name="add1")

    # FFN:
    ffn = helper.make_node("MatMul", inputs=["add1_out", "W_ffn"], outputs=["ffn_out"], name="ffn")
    add2 = helper.make_node("Add", inputs=["add1_out", "ffn_out"], outputs=["dec_out"], name="add2")

    nodes = [mm_enc, relu_enc, q_node, k_node, v_node, trans_k, qk, sm, ctx, proj, add1, ffn, add2]
    graph = helper.make_graph(
        nodes,
        "cross_attn_test",
        [enc_in, dec_in],
        [dec_out],
        [w_enc, w_q, w_k, w_v, w_out, w_ffn],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    synth = MacroBlockSynthesizer()
    optimized = synth.apply_transformer_encoder_layer_fusion(copy.deepcopy(model))

    # Encoder nodes (mm_enc, relu_enc) MUST NOT be dropped from the outer graph!
    outer_node_names = {n.name for n in optimized.graph.node}
    assert "mm_enc" in outer_node_names, "mm_enc was mistakenly swallowed into decoder block!"
    assert "relu_enc" in outer_node_names, "relu_enc was mistakenly swallowed into decoder block!"


# =============================================================================
# 4. ANONYMOUS NODE MEMORY POINTER ALIASING & GC RE-USE STRESS TEST
# =============================================================================
def test_anonymous_node_pointer_address_aliasing_stress():
    """Verify that when all node names are empty (''), Python memory allocator address

    reuse during object recreation does not cause unrelated graph nodes to be dropped.
    """
    X = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 16])
    Y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [2, 16])

    w_init = numpy_helper.from_array(np.random.randn(16, 16).astype(np.float32), name="W")

    # Build a sequence of 20 anonymous Add/Mul nodes
    nodes = []
    curr = "X"
    for i in range(20):
        out_name = f"t_{i}" if i < 19 else "Y"
        op = "Add" if i % 2 == 0 else "Mul"
        n = helper.make_node(op, inputs=[curr, "W"], outputs=[out_name], name="")
        nodes.append(n)
        curr = out_name

    graph = helper.make_graph(nodes, "anonymous_gc_stress", [X], [Y], [w_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    # Selectively drop anonymous nodes 5 and 6 using their SSA output tensor names
    nodes_to_drop = {nodes[5].output[0], nodes[6].output[0]}
    replacement = [
        helper.make_node("Add", inputs=[nodes[4].output[0], "W"], outputs=[nodes[6].output[0]], name="")
    ]

    # Run _replace_nodes_topologically
    _replace_nodes_topologically(model.graph, nodes_to_drop, replacement)

    # 19 nodes should remain (20 - 2 dropped + 1 replacement)
    assert len(model.graph.node) == 19, f"Expected 19 nodes, got {len(model.graph.node)}"
    # Verify graph can be checked
    onnx.checker.check_model(model)


# =============================================================================
# 5. DYNAMIC SYMBOLIC BATCH SHAPE & CONCAT HEAD SPLITTING
# =============================================================================
def test_dynamic_symbolic_shape_concat_in_head_splitting():
    """Verify dynamic reshape with symbolic dimensions (-1, 'unk__0') in multi-head attention

    does not crash shape canonicalization or static folding.
    """
    class DynamicSDPAModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(64, 64)
            self.k = nn.Linear(64, 64)
            self.v = nn.Linear(64, 64)
            self.out = nn.Linear(64, 64)

        def forward(self, x):
            B, S, D = x.shape
            H = 4
            d_head = D // H
            q = self.q(x).view(B, S, H, d_head).transpose(1, 2)
            k = self.k(x).view(B, S, H, d_head).transpose(1, 2)
            v = self.v(x).view(B, S, H, d_head).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-2, -1)) / (d_head ** 0.5)
            attn = torch.softmax(scores, dim=-1)
            ctx = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, D)
            return self.out(ctx) + x

    m = DynamicSDPAModel()
    m.eval()
    x = torch.randn(2, 16, 64)
    onx = _export_model_to_onnx(
        m,
        x,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch", 1: "seq"}, "output": {0: "batch", 1: "seq"}},
    )

    pipeline = OptimizationPipeline(mode="function_proto")
    context = CompilerContext(mode="function_proto", options={"enable_macro_blocks": True})
    optimized = pipeline.run(onx, context=context)

    # Must pass ONNX checker with dynamic symbolic shapes
    onnx.checker.check_model(optimized)


# =============================================================================
# 6. MULTI-BRANCH CONFORMER DUAL RESIDUAL (ATTENTION + CONV)
# =============================================================================
def test_conformer_dual_branch_residuals_topological_detection():
    """Verify a Conformer block with two parallel branches (Attention branch + Depthwise Conv branch)

    both merging into the residual stream does not confuse residual matching.
    """
    class ConformerBlock(nn.Module):
        def __init__(self, d_model=32):
            super().__init__()
            self.norm1 = nn.LayerNorm(d_model)
            self.attn = nn.MultiheadAttention(d_model, num_heads=2, batch_first=True)
            self.norm2 = nn.LayerNorm(d_model)
            self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)

        def forward(self, x):
            # First residual: x + Attention(Norm1(x))
            x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
            # Second residual: x + Conv(Norm2(x))
            res = self.conv(self.norm2(x).transpose(1, 2)).transpose(1, 2)
            return x + res

    m = ConformerBlock()
    m.eval()
    x = torch.randn(2, 16, 32)
    onx = _export_model_to_onnx(m, x, input_names=["x"], output_names=["out"])

    _strip_all_node_names(onx)

    synth = MacroBlockSynthesizer()
    optimized = synth.apply_transformer_encoder_layer_fusion(onx, mode="function_proto")

    # Even with depthwise convolution second branch, first residual dual-branch should be recognized
    tb_nodes = [n for n in optimized.graph.node if "TransformerBlock" in n.op_type]
    assert len(tb_nodes) >= 1, "Expected at least 1 macro block detected in Conformer layer"
    onnx.checker.check_model(optimized)


# =============================================================================
# 8. NON-AFFINE LAYERNORM WITH EPSILON AS ONNX CONSTANT NODE
# =============================================================================
def test_norm_canonicalizer_non_affine_with_epsilon_constant_node():
    """Verify NormCanonicalizer handles LayerNorm where scale/bias are omitted

    and epsilon is supplied as an ONNX Constant node instead of an attribute.
    """
    # Raw subgraph: Sub(X, Mean) -> Square -> Mean -> Add(const_eps) -> Sqrt -> Reciprocal -> Mul
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 16, 32])
    y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [2, 16, 32])

    eps_const = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["eps_val"],
        value=numpy_helper.from_array(np.float32(1e-5), name="eps"),
    )
    mean1 = helper.make_node("ReduceMean", inputs=["X"], outputs=["mean1_out"], axes=[-1], keepdims=1)
    sub = helper.make_node("Sub", inputs=["X", "mean1_out"], outputs=["diff"])
    sq = helper.make_node("Mul", inputs=["diff", "diff"], outputs=["sq_out"])
    var = helper.make_node("ReduceMean", inputs=["sq_out"], outputs=["var_out"], axes=[-1], keepdims=1)
    add_eps = helper.make_node("Add", inputs=["var_out", "eps_val"], outputs=["var_eps"])
    sqrt = helper.make_node("Sqrt", inputs=["var_eps"], outputs=["std_out"])
    recip = helper.make_node("Reciprocal", inputs=["std_out"], outputs=["inv_std"])
    normed = helper.make_node("Mul", inputs=["diff", "inv_std"], outputs=["Y"])

    graph = helper.make_graph(
        [eps_const, mean1, sub, sq, var, add_eps, sqrt, recip, normed],
        "non_affine_ln_graph",
        [x],
        [y],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    norm_pass = NormCanonicalizer()
    optimized = norm_pass.apply_layernorm_fusion(copy.deepcopy(model), mode="standard")

    ln_nodes = [n for n in optimized.graph.node if n.op_type in ("LayerNormalization", "RMSNormalization")]
    assert len(ln_nodes) == 1, "Failed to fuse non-affine LayerNorm with Constant node epsilon!"


# =============================================================================
# 9. STATIC FOLDER 0-D SCALAR INITIALIZER DIMENSION PRESERVATION
# =============================================================================
def test_static_folder_0d_scalar_initializer_dimension_preservation():
    """Verify that constant folding operations on 0-d scalar initializers (shape=[])

    preserve empty shape [] rather than corrupting into 1-d [1] arrays.
    """
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 16])
    y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [2, 16])

    # Two 0-D scalar constants: a=2.0, b=3.0 -> Mul(a, b) = 6.0
    a_init = numpy_helper.from_array(np.float32(2.0), name="scalar_a")
    b_init = numpy_helper.from_array(np.float32(3.0), name="scalar_b")

    mul_scalars = helper.make_node("Mul", inputs=["scalar_a", "scalar_b"], outputs=["scalar_c"], name="mul_c")
    mul_x = helper.make_node("Mul", inputs=["X", "scalar_c"], outputs=["Y"], name="mul_y")

    graph = helper.make_graph([mul_scalars, mul_x], "scalar_0d_graph", [x], [y], [a_init, b_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    folder = StaticFolder()
    optimized = folder.apply_static_subgraph_folding(copy.deepcopy(model))

    # scalar_c must be folded into an initializer
    init_map = {i.name: numpy_helper.to_array(i) for i in optimized.graph.initializer}
    assert "scalar_c" in init_map, "scalar_c was not constant-folded!"
    assert init_map["scalar_c"].shape == (), f"Expected 0-D scalar shape (), got {init_map['scalar_c'].shape}"


# =============================================================================
# 10. SKIP-NORM RESIDUAL MULTI-CONSUMER LEAKAGE VULNERABILITY
# =============================================================================
def test_skip_norm_fusion_does_not_sever_multi_consumer_residuals():
    """Verify that when fusing Add + LayerNorm into SkipLayerNormalization, if the intermediate

    Add output is ALSO consumed by an auxiliary head or loss, it is preserved.
    """
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 16, 32])
    res = helper.make_tensor_value_info("Residual", onnx.TensorProto.FLOAT, [2, 16, 32])
    ln_out = helper.make_tensor_value_info("LN_Out", onnx.TensorProto.FLOAT, [2, 16, 32])
    aux_out = helper.make_tensor_value_info("Aux_Out", onnx.TensorProto.FLOAT, [2, 16, 32])

    gamma = numpy_helper.from_array(np.ones(32, dtype=np.float32), name="gamma")
    beta = numpy_helper.from_array(np.zeros(32, dtype=np.float32), name="beta")

    # Add(X, Residual) -> add_out
    add_node = helper.make_node("Add", inputs=["X", "Residual"], outputs=["add_out"], name="main_add")
    # LN consumes add_out
    ln_node = helper.make_node("LayerNormalization", inputs=["add_out", "gamma", "beta"], outputs=["LN_Out"], name="ln")
    # Auxiliary branch ALSO consumes add_out
    aux_node = helper.make_node("Relu", inputs=["add_out"], outputs=["Aux_Out"], name="aux_relu")

    graph = helper.make_graph([add_node, ln_node, aux_node], "skip_norm_graph", [x, res], [ln_out, aux_out], [gamma, beta])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    norm_pass = NormCanonicalizer()
    optimized = norm_pass.apply_skip_norm_fusion(copy.deepcopy(model), mode="standard")

    # Aux_Out must still receive valid producer
    onnx.checker.check_model(optimized)
    aux_consumer = next(n for n in optimized.graph.node if n.name == "aux_relu")
    producer_map = {out: n for n in optimized.graph.node for out in n.output}
    assert aux_consumer.input[0] in producer_map, f"Aux input {aux_consumer.input[0]} has no producer in graph!"


# =============================================================================
# 11. DEEP STACKED ANONYMOUS TRANSFORMERS CONVERGENCE & MONOMORPHIZATION
# =============================================================================
def test_deep_stacked_anonymous_transformers_monomorphization():
    """Verify a deep stack of 4 identical anonymous transformer layers monomorphizes into

    EXACTLY ONE specialized FunctionProto in model.functions (zero duplication).
    """
    class Stacked4Transformers(nn.Module):
        def __init__(self, d_model=32, num_heads=4, d_ffn=64):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dim_feedforward=d_ffn,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(4)
            ])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    m = Stacked4Transformers()
    m.eval()
    x = torch.randn(2, 8, 32)
    onx = _export_model_to_onnx(m, x, input_names=["input"], output_names=["output"])

    _strip_all_node_names(onx)

    synth = MacroBlockSynthesizer()
    optimized = synth.apply_transformer_encoder_layer_fusion(onx, mode="function_proto")

    tb_nodes = [n for n in optimized.graph.node if "TransformerBlock" in n.op_type]
    assert len(tb_nodes) == 4, f"Expected 4 TransformerBlocks, found {len(tb_nodes)}"

    # All 4 identical layers must share monomorphized FunctionProto
    unique_op_types = {n.op_type for n in tb_nodes}
    assert len(unique_op_types) == 1, (
        f"Expected all 4 layers to monomorphize to 1 unique op_type, but got {unique_op_types}"
    )
    assert len(optimized.functions) == 1, (
        f"Expected exactly 1 FunctionProto in model.functions, got {len(optimized.functions)}"
    )
    onnx.checker.check_model(optimized)


# =============================================================================
# 11. HIERARCHICAL NESTED BLOCKS WITHIN BLOCKS PRESERVATION
# =============================================================================
def test_nested_blocks_within_blocks_hierarchical_preservation():
    """Verify hierarchical block synthesis: where outer TransformerBlock contains

    sub-blocks (e.g. MHA or FFN), inner blocks are either cleanly folded into the outer
    function definition or maintained without graph breakage.
    """
    class HierarchicalTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(32)
            self.norm2 = nn.LayerNorm(32)
            # Inner sub-block 1: MHA
            self.mha = nn.MultiheadAttention(32, 4, batch_first=True)
            # Inner sub-block 2: MLP (Linear -> Gelu -> Linear)
            self.mlp_fc1 = nn.Linear(32, 64)
            self.mlp_fc2 = nn.Linear(64, 32)

        def forward(self, x):
            # Attention residual block
            res1 = x + self.mha(self.norm1(x), self.norm1(x), self.norm1(x))[0]
            # MLP residual block
            mlp_out = self.mlp_fc2(F.gelu(self.mlp_fc1(self.norm2(res1))))
            return res1 + mlp_out

    m = HierarchicalTransformer()
    m.eval()
    x = torch.randn(2, 10, 32)
    onx = _export_model_to_onnx(m, x, input_names=["x"], output_names=["out"])

    pipeline = OptimizationPipeline(mode="function_proto")
    context = CompilerContext(mode="function_proto", options={"enable_macro_blocks": True})
    optimized = pipeline.run(onx, context=context)

    # Clean verification of nested blocks
    onnx.checker.check_model(optimized)
    tb_nodes = [n for n in optimized.graph.node if "TransformerBlock" in n.op_type]
    assert len(tb_nodes) == 1, f"Expected 1 hierarchical TransformerBlock, found {len(tb_nodes)}"


# =============================================================================
# 14. MONOMORPHIZATION ATTRIBUTE SENSITIVITY (DIFFERENT EPSILONS / PARAMETERS)
# =============================================================================
def test_monomorphization_attribute_sensitivity_different_epsilons():
    """Verify compute_subgraph_fingerprint differentiates blocks with different attributes
    (e.g. LayerNorm epsilon=1e-5 vs epsilon=1e-12) so they are NOT incorrectly mapped to the
    same specialized FunctionProto.
    """
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 16])
    w = numpy_helper.from_array(np.ones(16, dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.zeros(16, dtype=np.float32), name="B")

    # Subgraph 1: LayerNorm with epsilon = 1e-5
    ln1 = helper.make_node("LayerNormalization", inputs=["X", "W", "B"], outputs=["Y1"], epsilon=1e-5)
    # Subgraph 2: LayerNorm with epsilon = 1e-12
    ln2 = helper.make_node("LayerNormalization", inputs=["X", "W", "B"], outputs=["Y2"], epsilon=1e-12)

    fp1 = compute_subgraph_fingerprint([ln1])
    fp2 = compute_subgraph_fingerprint([ln2])

    assert fp1 != fp2, (
        f"Fingerprint collision! Different epsilons (1e-5 vs 1e-12) produced identical fingerprint {fp1}"
    )


# =============================================================================
# 15. HIGHWAY LONG SKIP CONNECTION NOT SWALLOWED BY INNER BLOCK
# =============================================================================
def test_highway_long_skip_connection_not_swallowed_by_inner_block():
    """Verify that when input X bypasses Block 1 and merges directly into downstream Block 2
    (Highway / DenseNet topology), synthesizing Block 1 does NOT swallow the long skip connection.
    """
    X = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 16])
    Y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [2, 16])

    w1 = numpy_helper.from_array(np.random.randn(16, 16).astype(np.float32), name="W1")
    w2 = numpy_helper.from_array(np.random.randn(16, 16).astype(np.float32), name="W2")

    # Block 1: X -> Relu -> MatMul(W1) -> Add(X) -> B1_out
    r1 = helper.make_node("Relu", inputs=["X"], outputs=["r1_out"], name="r1")
    mm1 = helper.make_node("MatMul", inputs=["r1_out", "W1"], outputs=["mm1_out"], name="mm1")
    add1 = helper.make_node("Add", inputs=["X", "mm1_out"], outputs=["b1_out"], name="add1")

    # Block 2: b1_out -> Relu -> MatMul(W2) -> Add(X) -> Y (Notice Add merges with original X!)
    r2 = helper.make_node("Relu", inputs=["b1_out"], outputs=["r2_out"], name="r2")
    mm2 = helper.make_node("MatMul", inputs=["r2_out", "W2"], outputs=["mm2_out"], name="mm2")
    add2 = helper.make_node("Add", inputs=["X", "mm2_out"], outputs=["Y"], name="add2")

    graph = helper.make_graph([r1, mm1, add1, r2, mm2, add2], "highway_graph", [X], [Y], [w1, w2])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    synth = MacroBlockSynthesizer()
    optimized = synth.apply_transformer_encoder_layer_fusion(copy.deepcopy(model))

    # Node add2 MUST still have X in its inputs and MUST NOT be dropped
    outer_node_names = {n.name for n in optimized.graph.node}
    assert "add2" in outer_node_names, "Downstream Highway Add node was mistakenly dropped!"
    add2_node = next(n for n in optimized.graph.node if n.name == "add2")
    assert "X" in add2_node.input, f"Long skip connection from X was severed! Inputs: {add2_node.input}"


# =============================================================================
# 16. MULTI-CONSUMER QUANTIZED ACTIVATION BRANCHING
# =============================================================================
def test_multi_consumer_quantized_activation_branching():
    """Verify that when an activation tensor is quantized via DynamicQuantizeLinear and
    its quantized output (q, s, z) is consumed by two independent branches, fusing one
    branch does not drop the quantization node or starve the sibling branch.
    """
    X = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 32])
    Y1 = helper.make_tensor_value_info("Y1", onnx.TensorProto.FLOAT, [1, 32])
    Y2 = helper.make_tensor_value_info("Y2", onnx.TensorProto.FLOAT, [1, 32])

    w1 = numpy_helper.from_array(np.random.randn(32, 32).astype(np.float32), name="W1")
    w2 = numpy_helper.from_array(np.random.randn(32, 32).astype(np.float32), name="W2")

    # Quantize X
    dql = helper.make_node("DynamicQuantizeLinear", inputs=["X"], outputs=["q", "s", "z"], name="dql")

    # Branch 1 consumes (q, s, z)
    dq1 = helper.make_node("DequantizeLinear", inputs=["q", "s", "z"], outputs=["dq1_out"], name="dq1")
    mm1 = helper.make_node("MatMul", inputs=["dq1_out", "W1"], outputs=["Y1"], name="mm1")

    # Branch 2 ALSO consumes (q, s, z)
    dq2 = helper.make_node("DequantizeLinear", inputs=["q", "s", "z"], outputs=["dq2_out"], name="dq2")
    mm2 = helper.make_node("MatMul", inputs=["dq2_out", "W2"], outputs=["Y2"], name="mm2")

    graph = helper.make_graph([dql, dq1, mm1, dq2, mm2], "dql_branch_graph", [X], [Y1, Y2], [w1, w2])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    synth = MacroBlockSynthesizer()
    optimized = synth.apply_dynamic_quant_linear_fusion(copy.deepcopy(model), mode="standard")

    # Both outputs Y1 and Y2 must have valid producers in the graph
    onnx.checker.check_model(optimized)
    producer_map = {out: n for n in optimized.graph.node for out in n.output}
    assert "Y1" in producer_map, "Y1 producer missing!"
    assert "Y2" in producer_map, "Y2 producer missing!"


# =============================================================================
# 17. CAUSAL ATTENTION MASK CONSTANT FOLDING WITH NEGATIVE INFINITY
# =============================================================================
def test_causal_attention_mask_negative_inf_constant_folding():
    """Verify that constant folding on causal attention masks containing -inf floats
    does not turn -inf into NaNs or drop the mask.
    """
    scores = helper.make_tensor_value_info("scores", onnx.TensorProto.FLOAT, [1, 4, 8, 8])
    out = helper.make_tensor_value_info("probs", onnx.TensorProto.FLOAT, [1, 4, 8, 8])

    # Lower triangular mask: 0 where valid, -inf where masked
    mask_np = np.zeros((8, 8), dtype=np.float32)
    mask_np[np.triu_indices(8, k=1)] = -np.inf
    mask_init = numpy_helper.from_array(mask_np, name="causal_mask")

    # Add mask to scores, then Softmax
    add_mask = helper.make_node("Add", inputs=["scores", "causal_mask"], outputs=["masked_scores"], name="add_mask")
    sm = helper.make_node("Softmax", inputs=["masked_scores"], outputs=["probs"], axis=-1, name="sm")

    graph = helper.make_graph([add_mask, sm], "causal_mask_graph", [scores], [out], [mask_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    folder = StaticFolder()
    optimized = folder.apply_static_subgraph_folding(copy.deepcopy(model))

    onnx.checker.check_model(optimized)
    # Ensure causal_mask initializer still exists and has -inf preserved
    mask_val = next(numpy_helper.to_array(i) for i in optimized.graph.initializer if i.name == "causal_mask")
    assert np.isneginf(mask_val[0, 1]), "Causal mask -inf was corrupted during constant folding!"


# =============================================================================
# 18. FUNCTIONPROTO INTERNAL TOPOLOGICAL SORTING GUARANTEE
# =============================================================================
def test_function_proto_internal_nodes_strict_topological_ordering():
    """Verify that when nodes are packaged into a FunctionProto, their internal sequence
    in function.node strictly obeys topological ordering (producers precede consumers).
    """
    model = helper.make_model(
        helper.make_graph([], "empty_graph", [], []),
        opset_imports=[helper.make_opsetid("", 17)],
    )

    # Intentionally provide nodes in REVERSE topological order to ensure_specialized_function_proto
    node_consumer = helper.make_node("Mul", inputs=["interm", "scale"], outputs=["out0"])
    node_producer = helper.make_node("Add", inputs=["in0", "bias"], outputs=["interm"])

    # If passed out-of-order, the function synthesizer must topologically sort them internally!
    func = ensure_specialized_function_proto(
        model,
        domain="custom.pytorch",
        fname="OutOfOrderTest_func",
        formal_inputs=["in0", "bias", "scale"],
        formal_outputs=["out0"],
        body_nodes=[node_consumer, node_producer],  # Consumer passed BEFORE producer!
    )

    # In func.node, node_producer MUST come before node_consumer
    assert func.node[0].op_type == "Add", f"Expected Add producer first, but found {func.node[0].op_type}"
    assert func.node[1].op_type == "Mul", f"Expected Mul consumer second, but found {func.node[1].op_type}"
    onnx.checker.check_model(model)


# =============================================================================
# 19. GRAPH OUTPUT TENSOR NAME PRESERVATION IN MACRO SYNTHESIS
# =============================================================================
def test_graph_output_tensor_name_preservation_when_block_ends_at_graph_output():
    """Verify that when a macro block's final output tensor is an explicit graph output
    (e.g. named 'final_output'), synthesizing the block preserves the exact graph output name.
    """
    X = helper.make_tensor_value_info("input_tokens", onnx.TensorProto.FLOAT, [2, 16])
    FinalOut = helper.make_tensor_value_info("final_output", onnx.TensorProto.FLOAT, [2, 16])

    w_init = numpy_helper.from_array(np.random.randn(16, 16).astype(np.float32), name="W")

    # Simple 2-node sequence producing final_output
    relu = helper.make_node("Relu", inputs=["input_tokens"], outputs=["interm"], name="r")
    mm = helper.make_node("MatMul", inputs=["interm", "W"], outputs=["final_output"], name="mm")

    graph = helper.make_graph([relu, mm], "output_preservation_graph", [X], [FinalOut], [w_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    # Synthetic replacement of relu + mm with a single node
    replacement = [
        helper.make_node("CustomMacro", inputs=["input_tokens", "W"], outputs=["final_output"], domain="custom.pytorch")
    ]
    _replace_nodes_topologically(model, {"r", "mm"}, replacement)

    # Graph output 'final_output' must still be produced by the new node
    assert len(model.graph.node) == 1
    assert model.graph.node[0].output[0] == "final_output"
    assert model.graph.output[0].name == "final_output"
    onnx.checker.check_model(model)


# =============================================================================
# 20. MONOMORPHIZED OP_TYPE NAME VALID C IDENTIFIER SANITIZATION
# =============================================================================
def test_monomorphized_op_type_name_valid_c_identifier():
    """Verify that build_monomorphized_op_type generates strings that are strictly valid
    C/C++ identifiers (no slashes, hyphens, spaces, or dots), ensuring ONNX and engine compatibility.
    """
    import re
    ident_pattern = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

    test_cases = [
        ("Transformer/Encoder-Layer", "a1b2c3d4e5f6"),
        ("Model.Block-0", "1234567890ab"),
        ("Custom::Norm Layer", "feedfacecafe"),
    ]

    for base_name, hash_val in test_cases:
        op_type = build_monomorphized_op_type(base_name, hash_val)
        assert ident_pattern.match(op_type), (
            f"Generated op_type '{op_type}' is not a valid C identifier! (from base '{base_name}')"
        )

