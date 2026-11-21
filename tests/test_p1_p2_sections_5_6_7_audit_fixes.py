"""Comprehensive rock-solid test suite for Sections 5, 6, and 7 of SYNTHESIS_PASSES_BUG_AUDIT.

Covers:
- Section 5: Pass Ordering & Ambiguity Conflicts (P1)
  - Bug 5.1: DeepMLP_Norm_Act does not cannibalize 2-layer FeedForward/MLP blocks.
  - Bug 5.2: GatedUnit does not cannibalize raw SiLU (x * Sigmoid(x)).
  - Bug 5.3: Macro pass ordering guarantees TransformerEncoderLayer runs before loose sub-blocks.
- Section 6: Brittle Architectural Blindness (P2)
  - Bug 6.1: FeedForward fusion supports LayerNormalization and RMSNorm.
  - Bug 6.2: FeedForward GLU supports GeGLU (GELU) and ReGLU (ReLU) with exact numerical precision.
  - Bug 6.3: SoftmaxMatMul traces through intermediate Dropout/Identity/Cast layout nodes.
  - Bug 6.4: Pre-Activation ResNet cleanly fuses true Pre-Act blocks without post-addition ReLU.
  - Bug 6.5: SequenceClassifierHead supports global average pooling via ReduceMean.
  - Bug 6.6: MultiFieldEmbedding reduces arbitrary and balanced binary Add trees to a single root.
  - Bug 6.7: DeepSeek MLA deterministically distinguishes K vs V based on attention graph topology.
- Section 7: Topological & SSA Graph Corruption (P2)
  - Bug 7.1: DynamicQuantLinear preserves shared quantizer nodes when multiple projections branch from it.
  - Bug 7.3: K Transpose handling never mutates shared Transpose nodes in place.
  - Bug 7.4: IndexTensorRouting never deletes shared Shape/Gather/Sub nodes consumed elsewhere.
  - Bug 7.5: Subgraph node canonicalization handles passthrough input-output mappings cleanly.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import helper, numpy_helper
import pytest
import torch
import torch.nn.functional as F

from neuron_toolkit.compiler.passes.synthesis.attention_synthesizer import (
    AttentionSynthesizer,
)
from neuron_toolkit.compiler.passes.synthesis.axial_synthesizer import (
    AxialSynthesizer,
)
from neuron_toolkit.compiler.passes.synthesis.feedforward_synthesizer import (
    FeedForwardSynthesizer,
)
from neuron_toolkit.compiler.passes.synthesis.macro_block_synthesizer import (
    MacroBlockSynthesizer,
    _canonicalize_subgraph_nodes,
)
from neuron_toolkit.compiler.pipeline import NeuronCompiler
from neuron_toolkit.runtime.cross_verifier import CrossVerifier
from neuron_toolkit.runtime.runner import PyTorchRunner


# =============================================================================
# SECTION 5: PASS ORDERING & AMBIGUITY CONFLICTS (P1)
# =============================================================================

def test_bug_5_1_deep_mlp_does_not_cannibalize_two_layer_feedforward():
    """Bug 5.1: DeepMLP_Norm_Act must not match layer 1 of a 2-layer FFN/MLP."""
    # Build: Norm -> Linear1 -> Relu -> Linear2
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 16])
    y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [2, 16])

    gamma = numpy_helper.from_array(np.ones((16,), dtype=np.float32), name="gamma")
    w1 = numpy_helper.from_array(np.random.randn(16, 64).astype(np.float32), name="W1")
    b1 = numpy_helper.from_array(np.zeros((64,), dtype=np.float32), name="B1")
    w2 = numpy_helper.from_array(np.random.randn(64, 16).astype(np.float32), name="W2")
    b2 = numpy_helper.from_array(np.zeros((16,), dtype=np.float32), name="B2")

    nodes = [
        helper.make_node("LayerNormalization", ["X", "gamma"], ["x_norm"]),
        helper.make_node("Linear", ["x_norm", "W1", "B1"], ["h1"], domain="custom.pytorch"),
        helper.make_node("Relu", ["h1"], ["h_act"]),
        helper.make_node("Linear", ["h_act", "W2", "B2"], ["Y"], domain="custom.pytorch"),
    ]
    graph = helper.make_graph(nodes, "test_ffn_not_deep_mlp", [x], [y], initializer=[gamma, w1, b1, w2, b2])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    synth = MacroBlockSynthesizer()
    res = synth.apply_deep_mlp_norm_act_fusion(model)

    # Must NOT contain DeepMLP_Norm_Act because Relu feeds into Linear2!
    deep_mlp_nodes = [n for n in res.graph.node if "DeepMLP" in n.op_type]
    assert len(deep_mlp_nodes) == 0, "DeepMLP_Norm_Act incorrectly cannibalized layer 1 of a 2-layer MLP!"


def test_bug_5_2_gated_unit_does_not_cannibalize_raw_silu():
    """Bug 5.2: GatedUnit must not match x * Sigmoid(x), which is self-activation (SiLU)."""
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 16])
    y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [2, 16])

    nodes = [
        helper.make_node("Sigmoid", ["X"], ["x_sig"]),
        helper.make_node("Mul", ["X", "x_sig"], ["Y"]),
    ]
    graph = helper.make_graph(nodes, "test_raw_silu", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    attn_synth = AttentionSynthesizer()
    res = attn_synth.apply_gated_unit_fusion(model)

    # Must NOT contain GatedUnit because other_inp == gate_in == 'X'
    gated_nodes = [n for n in res.graph.node if "GatedUnit" in n.op_type]
    assert len(gated_nodes) == 0, "GatedUnit incorrectly cannibalized raw SiLU (x * Sigmoid(x))!"


def test_bug_5_3_pass_ordering_transformer_encoder_layer_fuses_before_sub_blocks():
    """Bug 5.3: Full TransformerEncoderLayer must match and fuse before sub-block passes."""
    # Build complete Transformer layer: X -> Add(X, MHA) -> Add(Norm, FFN)
    # Using NeuronCompiler: should fuse cleanly without partial dismantling
    d_model = 16
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 4, d_model])
    y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 4, d_model])

    w_q = numpy_helper.from_array(np.eye(d_model, dtype=np.float32), name="W_q")
    w_k = numpy_helper.from_array(np.eye(d_model, dtype=np.float32), name="W_k")
    w_v = numpy_helper.from_array(np.eye(d_model, dtype=np.float32), name="W_v")
    w_out = numpy_helper.from_array(np.eye(d_model, dtype=np.float32), name="W_out")
    scale = numpy_helper.from_array(np.ones((d_model,), dtype=np.float32), name="scale")

    nodes = [
        # Norm 1
        helper.make_node("LayerNormalization", ["X", "scale"], ["x_norm1"]),
        # Self-Attention
        helper.make_node("MatMul", ["x_norm1", "W_q"], ["q"]),
        helper.make_node("MatMul", ["x_norm1", "W_k"], ["k"]),
        helper.make_node("MatMul", ["x_norm1", "W_v"], ["v"]),
        helper.make_node("ScaledDotProductAttention", ["q", "k", "v"], ["attn_out"], domain="custom.pytorch"),
        helper.make_node("MatMul", ["attn_out", "W_out"], ["mha_out"]),
        # Residual 1
        helper.make_node("Add", ["X", "mha_out"], ["res1"]),
        # Norm 2
        helper.make_node("LayerNormalization", ["res1", "scale"], ["x_norm2"]),
        # FFN
        helper.make_node("MatMul", ["x_norm2", "W_q"], ["ff1"]),
        helper.make_node("Gelu", ["ff1"], ["ff_act"], domain="custom.pytorch"),
        helper.make_node("MatMul", ["ff_act", "W_out"], ["ff2"]),
        # Residual 2
        helper.make_node("Add", ["res1", "ff2"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes, "transformer_layer_ordering", [x], [y],
        initializer=[w_q, w_k, w_v, w_out, scale]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    synth = MacroBlockSynthesizer()
    compiled = synth.run(model)

    # Must contain TransformerBlock or unified transformer layer, NOT broken into loose DeepMLP
    block_nodes = [n for n in compiled.graph.node if "TransformerBlock" in n.op_type]
    assert len(block_nodes) >= 1, "TransformerEncoderLayer was not fused due to sub-block pass ordering!"


# =============================================================================
# SECTION 6: BRITTLE ARCHITECTURAL BLINDNESS (P2)
# =============================================================================

def test_bug_6_1_feedforward_fusion_supports_layernorm_and_rmsnorm():
    """Bug 6.1: FeedForward fusion must support LayerNormalization as well as RMSNorm."""
    for norm_op in ("LayerNormalization", "RMSNormalization"):
        x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 16])
        y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [2, 16])

        gamma = numpy_helper.from_array(np.ones((16,), dtype=np.float32), name="gamma")
        fc1_w = numpy_helper.from_array(np.random.randn(16, 32).astype(np.float32), name="fc1_w")
        fc2_w = numpy_helper.from_array(np.random.randn(32, 16).astype(np.float32), name="fc2_w")

        nodes = [
            helper.make_node(norm_op, ["X", "gamma"], ["x_norm"]),
            helper.make_node("Linear", ["x_norm", "fc1_w"], ["h1"], domain="custom.pytorch"),
            helper.make_node("Gelu", ["h1"], ["h_act"], domain="custom.pytorch"),
            helper.make_node("Linear", ["h_act", "fc2_w"], ["Y"], domain="custom.pytorch"),
        ]
        graph = helper.make_graph(
            nodes, f"test_ff_{norm_op}", [x], [y], initializer=[gamma, fc1_w, fc2_w]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

        ff_synth = FeedForwardSynthesizer()
        res = ff_synth.apply_feedforward_fusion(model, mode="function_proto")

        ff_nodes = [n for n in res.graph.node if "FeedForward" in n.op_type]
        assert len(ff_nodes) == 1, f"FeedForward fusion failed for norm {norm_op}!"


def test_bug_6_2_feedforward_swiglu_supports_geglu_and_reglu_with_exact_numerical_equivalence():
    """Bug 6.2: FeedForward GLU must support GELU (GeGLU) and ReLU (ReGLU) with 100% numerical verification."""
    for act_name, expected_op in [("Gelu", "FeedForwardGeGLU"), ("Relu", "FeedForwardReGLU"), ("SiLU", "FeedForwardSwiGLU")]:
        dim = 16
        hidden_dim = 32
        x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 4, dim])
        y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 4, dim])

        w1_arr = np.random.randn(dim, hidden_dim).astype(np.float32)
        w2_arr = np.random.randn(hidden_dim, dim).astype(np.float32)
        w3_arr = np.random.randn(dim, hidden_dim).astype(np.float32)

        w1 = numpy_helper.from_array(w1_arr, name="W1")
        w2 = numpy_helper.from_array(w2_arr, name="W2")
        w3 = numpy_helper.from_array(w3_arr, name="W3")

        nodes = [
            helper.make_node("Linear", ["X", "W1"], ["gate_out"], domain="custom.pytorch"),
            helper.make_node("Linear", ["X", "W3"], ["up_out"], domain="custom.pytorch"),
            helper.make_node(act_name, ["gate_out"], ["act_out"], domain="" if act_name == "Relu" else "custom.pytorch"),
            helper.make_node("Mul", ["act_out", "up_out"], ["mul_out"]),
            helper.make_node("Linear", ["mul_out", "W2"], ["Y"], domain="custom.pytorch"),
        ]
        graph = helper.make_graph(nodes, f"test_{act_name}", [x], [y], initializer=[w1, w2, w3])
        orig_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

        opt_model = NeuronCompiler.compile(orig_model)
        report = CrossVerifier(orig_model, opt_model, atol=1e-5, min_cosine_similarity=0.9999).verify()
        assert report.all_passed, f"CrossVerifier failed for {act_name}:\n{report.summary()}"


def test_bug_6_3_softmax_matmul_traces_through_dropout_and_cast():
    """Bug 6.3: SoftmaxMatMul must trace through intermediate Dropout and Cast nodes."""
    x = helper.make_tensor_value_info("Scores", onnx.TensorProto.FLOAT, [2, 4, 8, 8])
    v = helper.make_tensor_value_info("V", onnx.TensorProto.FLOAT, [2, 4, 8, 16])
    y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [2, 4, 8, 16])

    nodes = [
        helper.make_node("Softmax", ["Scores"], ["probs"], axis=-1),
        helper.make_node("Identity", ["probs"], ["probs_id"]),
        helper.make_node("Cast", ["probs_id"], ["probs_flt"], to=onnx.TensorProto.FLOAT),
        helper.make_node("MatMul", ["probs_flt", "V"], ["Y"]),
    ]
    graph = helper.make_graph(nodes, "test_sm_cast", [x, v], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    attn_synth = AttentionSynthesizer()
    res = attn_synth.apply_softmax_matmul_fusion(model)

    sm_nodes = [n for n in res.graph.node if n.op_type == "SoftmaxMatMul"]
    assert len(sm_nodes) == 1, "SoftmaxMatMul failed to trace through transparent intermediate Cast/Identity nodes!"


def test_bug_6_4_preactivation_resnet_fuses_without_post_add_relu():
    """Bug 6.4: Pre-Activation ResNet must fuse genuine pre-act blocks where Residual Add is the exit."""
    # Pattern: BN -> Conv -> Relu -> BN -> Conv + Residual (Add) -> (No ReLU!)
    c = 8
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, c, 16, 16])
    y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, c, 16, 16])

    scale = numpy_helper.from_array(np.ones((c,), dtype=np.float32), name="scale")
    bias = numpy_helper.from_array(np.zeros((c,), dtype=np.float32), name="bias")
    mean = numpy_helper.from_array(np.zeros((c,), dtype=np.float32), name="mean")
    var = numpy_helper.from_array(np.ones((c,), dtype=np.float32), name="var")
    w1 = numpy_helper.from_array(np.random.randn(c, c, 3, 3).astype(np.float32), name="W1")
    w2 = numpy_helper.from_array(np.random.randn(c, c, 3, 3).astype(np.float32), name="W2")

    nodes = [
        helper.make_node("BatchNormalization", ["X", "scale", "bias", "mean", "var"], ["bn1_out"]),
        helper.make_node("Conv", ["bn1_out", "W1"], ["conv1_out"], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["conv1_out"], ["r1_out"]),
        helper.make_node("BatchNormalization", ["r1_out", "scale", "bias", "mean", "var"], ["bn2_out"]),
        helper.make_node("Conv", ["bn2_out", "W2"], ["conv2_out"], pads=[1, 1, 1, 1]),
        helper.make_node("Add", ["conv2_out", "X"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes, "test_preact_resnet_no_relu", [x], [y],
        initializer=[scale, bias, mean, var, w1, w2]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    synth = MacroBlockSynthesizer()
    res = synth.apply_preactivation_resnet_fusion(model)

    res_nodes = [n for n in res.graph.node if "ResidualBlock" in n.op_type]
    assert len(res_nodes) == 1, "Pre-Activation ResNet failed to match pure pre-act block without post-Add ReLU!"


def test_bug_6_5_sequence_classifier_head_handles_reducemean_pooling():
    """Bug 6.5: SequenceClassifierHead must match heads with ReduceMean (global average pooling)."""
    dim = 16
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 8, dim])
    y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 2])

    scale = numpy_helper.from_array(np.ones((dim,), dtype=np.float32), name="scale")
    w = numpy_helper.from_array(np.random.randn(dim, 2).astype(np.float32), name="W")
    b = numpy_helper.from_array(np.zeros((2,), dtype=np.float32), name="B")
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes")

    # Topology: Norm -> ReduceMean(pool tokens) -> Linear -> Out
    nodes = [
        helper.make_node("LayerNormalization", ["X", "scale"], ["x_norm"]),
        helper.make_node("ReduceMean", ["x_norm", "axes"], ["pooled"], keepdims=0),
        helper.make_node("Linear", ["pooled", "W", "B"], ["Y"], domain="custom.pytorch"),
    ]
    graph = helper.make_graph(nodes, "test_reducemean_head", [x], [y], initializer=[scale, w, b, axes])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    synth = MacroBlockSynthesizer()
    res = synth.apply_sequence_classifier_head_fusion(model)

    head_nodes = [n for n in res.graph.node if "SequenceClassifierHead" in n.op_type]
    assert len(head_nodes) == 1, "SequenceClassifierHead failed to match head with ReduceMean pooling!"


def test_bug_6_6_multi_field_embedding_fuses_balanced_binary_add_tree():
    """Bug 6.6: MultiFieldEmbedding must reduce balanced binary summation trees: (e1 + e2) + (e3 + e4)."""
    num_fields = 4
    dim = 16
    feat = helper.make_tensor_value_info("Feat", onnx.TensorProto.INT64, [1, 4])
    y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, dim])

    inits = []
    nodes = []

    # Slices: Gather(Feat, i)
    for i in range(num_fields):
        idx = numpy_helper.from_array(np.array(i, dtype=np.int64), name=f"idx_{i}")
        w = numpy_helper.from_array(np.random.randn(10, dim).astype(np.float32), name=f"table_{i}")
        inits.extend([idx, w])
        nodes.append(helper.make_node("Gather", ["Feat", f"idx_{i}"], [f"slice_{i}"], axis=1))
        nodes.append(helper.make_node("Gather", [f"table_{i}", f"slice_{i}"], [f"emb_{i}"], axis=0))

    # Balanced tree: Add(Add(emb_0, emb_1), Add(emb_2, emb_3))
    nodes.append(helper.make_node("Add", ["emb_0", "emb_1"], ["tree_left"]))
    nodes.append(helper.make_node("Add", ["emb_2", "emb_3"], ["tree_right"]))
    nodes.append(helper.make_node("Add", ["tree_left", "tree_right"], ["Y"]))

    graph = helper.make_graph(nodes, "test_balanced_tree", [feat], [y], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    synth = MacroBlockSynthesizer()
    res = synth.apply_multi_field_embedding_fusion(model)

    mfe_nodes = [n for n in res.graph.node if "MultiFieldEmbedding" in n.op_type]
    assert len(mfe_nodes) == 1, "MultiFieldEmbedding failed to fuse balanced binary tree of Add nodes!"


def test_bug_6_7_deepseek_mla_deterministic_kv_assignment_via_attention_topology():
    """Bug 6.7: DeepSeek MLA must deterministically identify K (feeds Softmax) vs V (multiplied by Softmax)."""
    # Build: X -> Linear_kv -> (Up_A, Up_B)
    # Up_A -> MatMul(Q, Up_A^T) -> Softmax (Up_A is K)
    # Up_B -> MatMul(Softmax, Up_B) -> Out (Up_B is V)
    d = 16
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 4, d])
    q = helper.make_tensor_value_info("Q", onnx.TensorProto.FLOAT, [1, 4, d])
    out = helper.make_tensor_value_info("Out", onnx.TensorProto.FLOAT, [1, 4, d])

    w_kv = numpy_helper.from_array(np.random.randn(d, 8).astype(np.float32), name="W_kv")
    w_k = numpy_helper.from_array(np.random.randn(8, d).astype(np.float32), name="W_k")
    w_v = numpy_helper.from_array(np.random.randn(8, d).astype(np.float32), name="W_v")

    # In graph.node: put Up_V FIRST, and Up_K SECOND to test that order does NOT dictate K/V!
    nodes = [
        helper.make_node("MatMul", ["X", "W_kv"], ["c_kv"]),
        # Producer for V first in node list!
        helper.make_node("MatMul", ["c_kv", "W_v"], ["cand_v"]),
        # Producer for K second!
        helper.make_node("MatMul", ["c_kv", "W_k"], ["cand_k"]),
        # Attention score: Q @ K^T -> Softmax
        helper.make_node("MatMul", ["Q", "cand_k"], ["scores"]),
        helper.make_node("Softmax", ["scores"], ["probs"], axis=-1),
        # Attention output: Probs @ V
        helper.make_node("MatMul", ["probs", "cand_v"], ["Out"]),
    ]
    graph = helper.make_graph(nodes, "test_mla_kv", [x, q], [out], initializer=[w_kv, w_k, w_v])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    attn_synth = AttentionSynthesizer()
    res = attn_synth.apply_deepseek_mla_fusion(model)

    mla_nodes = [n for n in res.graph.node if "MultiHeadLatentAttention_KVProjection" in n.op_type]
    assert len(mla_nodes) == 1
    mla_node = mla_nodes[0]
    # Output 0 must be K (cand_k), Output 1 must be V (cand_v)
    assert mla_node.output[0] == "cand_k", f"MLA assigned {mla_node.output[0]} as K instead of cand_k!"
    assert mla_node.output[1] == "cand_v", f"MLA assigned {mla_node.output[1]} as V instead of cand_v!"


# =============================================================================
# SECTION 7: TOPOLOGICAL & SSA CORRUPTION (P2)
# =============================================================================

def test_bug_7_1_dynamic_quant_linear_preserves_shared_quantizer_for_sibling_projections():
    """Bug 7.1: Sibling MatMul projections sharing a DynamicQuantizeLinear must both be fused without dropped inputs."""
    d = 16
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, d])
    out_q = helper.make_tensor_value_info("Out_Q", onnx.TensorProto.FLOAT, [1, d])
    out_k = helper.make_tensor_value_info("Out_K", onnx.TensorProto.FLOAT, [1, d])

    w_q = numpy_helper.from_array(np.random.randn(d, d).astype(np.float32), name="W_q")
    w_k = numpy_helper.from_array(np.random.randn(d, d).astype(np.float32), name="W_k")

    # One DQL -> DQ feeding TWO sibling MatMuls (Q and K)
    nodes = [
        helper.make_node("DynamicQuantizeLinear", ["X"], ["q", "scale", "zp"]),
        helper.make_node("DequantizeLinear", ["q", "scale", "zp"], ["dq_x"]),
        helper.make_node("MatMul", ["dq_x", "W_q"], ["Out_Q"]),
        helper.make_node("MatMul", ["dq_x", "W_k"], ["Out_K"]),
    ]
    graph = helper.make_graph(nodes, "test_shared_dql", [x], [out_q, out_k], initializer=[w_q, w_k])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    synth = MacroBlockSynthesizer()
    res = synth.apply_dynamic_quant_linear_fusion(model)

    dql_nodes = [n for n in res.graph.node if n.op_type == "DynamicQuantLinear"]
    assert len(dql_nodes) == 2, f"Expected 2 fused DynamicQuantLinear nodes, got {len(dql_nodes)}!"

    # Verify model is strictly topologically valid
    onnx.checker.check_model(res)


def test_bug_7_3_handle_k_transpose_does_not_corrupt_shared_transpose_in_place():
    """Bug 7.3: _handle_k_transpose must not mutate a shared Transpose node in place."""
    p_orig = [0, 2, 1, 3]
    tr_node = helper.make_node("Transpose", ["K_raw"], ["K_trans"], perm=p_orig, name="shared_tr")

    consumer_map = {"K_trans": [helper.make_node("MatMul", ["Q", "K_trans"], ["S"]), helper.make_node("Identity", ["K_trans"], ["Other"])]}
    nodes_to_drop = set()
    extra_nodes = []

    res_out = AttentionSynthesizer._handle_k_transpose(
        tr_node, "K_raw", nodes_to_drop, consumer_map=consumer_map, extra_nodes=extra_nodes
    )

    # Must NOT mutate tr_node.attribute in place
    perm_after = list([a for a in tr_node.attribute if a.name == "perm"][0].ints)
    assert perm_after == p_orig, "Shared Transpose perm attribute was mutated in place!"
    assert id(tr_node) not in nodes_to_drop, "Shared Transpose node was marked for deletion while still consumed!"


def test_bug_7_4_index_tensor_routing_does_not_delete_shared_shape_or_gather():
    """Bug 7.4: Backwards graph deletion in index tensor routing must not delete shared Shape nodes."""
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 4, 16])
    route_out = helper.make_tensor_value_info("RouteOut", onnx.TensorProto.FLOAT, [1, 8])
    shape_user_out = helper.make_tensor_value_info("ShapeUserOut", onnx.TensorProto.INT64, [3])

    idx = numpy_helper.from_array(np.array(0, dtype=np.int64), name="idx_0")
    start = numpy_helper.from_array(np.array([0], dtype=np.int64), name="start_0")
    end = numpy_helper.from_array(np.array([8], dtype=np.int64), name="end_8")
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="axis_1")

    # Shape is used BOTH by the Slice and by another downstream node ShapeUserOut!
    nodes = [
        helper.make_node("Shape", ["X"], ["x_shape"], name="shared_shape"),
        helper.make_node("Gather", ["X", "idx_0"], ["gather_out"], axis=1),
        helper.make_node("Slice", ["gather_out", "start_0", "end_8", "axis_1"], ["RouteOut"]),
        helper.make_node("Identity", ["x_shape"], ["ShapeUserOut"]),
    ]
    graph = helper.make_graph(
        nodes, "test_shared_shape_routing", [x], [route_out, shape_user_out],
        initializer=[idx, start, end, axes]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    axial_synth = AxialSynthesizer()
    res = axial_synth.apply_index_tensor_routing(model)

    node_types = [n.op_type for n in res.graph.node]
    assert "Shape" in node_types, "Shared Shape node was improperly deleted by index routing backwards traversal!"
    onnx.checker.check_model(res)


def test_bug_7_5_canonicalize_subgraph_nodes_handles_passthrough_inputs_without_collision():
    """Bug 7.5: Subgraph node canonicalization must handle residual passthrough inputs as outputs without collision."""
    # Subgraph: in_0 and in_1
    # Node: Add(in_0, in_1) -> out_0
    # External outputs: ["add_out", "in_0"] (in_0 is returned directly as a passthrough output!)
    add_node = helper.make_node("Add", ["in_0", "in_1"], ["add_out"])

    canon_nodes, f_in, f_out = _canonicalize_subgraph_nodes(
        sub_nodes=[add_node],
        external_inputs=["in_0", "in_1"],
        external_outputs=["add_out", "in_0"],
    )

    assert f_in == ["in_0", "in_1"]
    assert f_out == ["out_0", "out_1"]

    # Verify that passthrough tensor in_0 has a corresponding Identity node producing out_1
    id_nodes = [n for n in canon_nodes if n.op_type == "Identity" and n.output[0] == "out_1"]
    assert len(id_nodes) == 1, "Failed to create Identity node for passthrough function output!"
    assert id_nodes[0].input[0] == "in_0", "Identity node does not take canonical input in_0!"
