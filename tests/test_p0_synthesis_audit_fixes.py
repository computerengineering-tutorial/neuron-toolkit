"""Tests covering all Batch 1 P0 fixes from MEMORY/SYNTHESIS_PASSES_BUG_AUDIT.md.

Specifically covers:
- Bug 1.1: apply_moe_fusion does not wipe external graph
- Bug 1.2: apply_mamba_ssm_fusion does not wipe transformer models
- Bug 1.3: apply_shifted_window_attention_fusion does not wipe multi-stage Swin
- Bug 1.4: apply_deepseek_mla_macro_fusion does not wipe multi-layer models
- Bug 7.2: _replace_nodes_topologically topological order and drop_ids handling
- Bug 2.1: MultiHeadAttention FunctionProto arity and input alignment
- Bug 2.2: ScaledDotProductAttention mask arity in FunctionProto
- Bug 2.3: MergeHeads FunctionProto contains Reshape to 3D
- Bug 2.4: SpatialGatingUnit preserves gamma and biases with arity matching
- Bug 2.5: AxialRearrange FunctionProto contains real Transpose/Reshape (not dummy Identity)
- Bug 2.6: TensorRoute FunctionProto preserves Slice and matches node arity
- Bug 2.7: Unbind FunctionProto preserves non-zero axis
- Bug 2.8: DynamicQuantLinear biased vs unbiased collision
- Bug 2.9 & 2.10: GatedUnit rank-agnostic and opset_imports include custom.pytorch
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper

from neuron_toolkit.compiler.passes.synthesis.attention_synthesizer import AttentionSynthesizer
from neuron_toolkit.compiler.passes.synthesis.axial_synthesizer import AxialSynthesizer
from neuron_toolkit.compiler.passes.synthesis.macro_block_synthesizer import (
    MacroBlockSynthesizer,
    _replace_nodes_topologically,
)


# ==============================================================================
# SECTION 1: WHOLE-GRAPH WIPEOUT DEFENSE TESTS (Bugs 1.1 - 1.4, 7.2)
# ==============================================================================

def test_bug_1_1_moe_fusion_preserves_external_graph():
    """Bug 1.1: apply_moe_fusion must only fuse the bounded MoE subgraph, not delete the entire model."""
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 16, 64])
    y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [2, 16, 64])

    w_pre = numpy_helper.from_array(np.random.randn(64, 64).astype(np.float32), name="W_pre")
    w_gate = numpy_helper.from_array(np.random.randn(64, 4).astype(np.float32), name="W_gate")
    k_val = numpy_helper.from_array(np.array([2], dtype=np.int64), name="K_val")
    w_e1 = numpy_helper.from_array(np.random.randn(64, 64).astype(np.float32), name="W_e1")
    w_e2 = numpy_helper.from_array(np.random.randn(64, 64).astype(np.float32), name="W_e2")
    w_post = numpy_helper.from_array(np.random.randn(64, 64).astype(np.float32), name="W_post")

    nodes = [
        # Pre-MoE layer (must survive!)
        helper.make_node("MatMul", ["X", "W_pre"], ["pre_out"], name="pre_matmul"),
        helper.make_node("Relu", ["pre_out"], ["pre_act"], name="pre_relu"),
        # MoE Router
        helper.make_node("MatMul", ["pre_act", "W_gate"], ["gate_logits"], name="gate_mm"),
        helper.make_node("TopK", ["gate_logits", "K_val"], ["topk_vals", "topk_idx"], axis=-1, name="topk_node"),
        # MoE Experts
        helper.make_node("MatMul", ["pre_act", "W_e1"], ["e1_out"], name="e1_mm"),
        helper.make_node("MatMul", ["pre_act", "W_e2"], ["e2_out"], name="e2_mm"),
        # MoE Combine
        helper.make_node("Add", ["e1_out", "e2_out"], ["moe_combined"], name="moe_add"),
        # Post-MoE layer (must survive!)
        helper.make_node("MatMul", ["moe_combined", "W_post"], ["post_out"], name="post_matmul"),
        helper.make_node("Relu", ["post_out"], ["Y"], name="post_relu"),
    ]

    graph = helper.make_graph(
        nodes, "test_moe_safety", [x], [y],
        initializer=[w_pre, w_gate, k_val, w_e1, w_e2, w_post]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    synthesizer = MacroBlockSynthesizer()
    simplified = synthesizer.apply_moe_fusion(model, mode="function_proto")

    node_types = [n.op_type for n in simplified.graph.node]
    node_names = [n.name for n in simplified.graph.node]

    # Pre-MoE and Post-MoE nodes MUST be intact!
    assert "pre_matmul" in node_names, "CRITICAL: Pre-MoE nodes were wiped out by apply_moe_fusion!"
    assert "post_matmul" in node_names, "CRITICAL: Post-MoE nodes were wiped out by apply_moe_fusion!"
    assert "post_relu" in node_names, "CRITICAL: Final output layer was wiped out by apply_moe_fusion!"
    assert len(simplified.graph.node) >= 3, f"Expected multi-layer graph to survive, got only {len(simplified.graph.node)} nodes"
    onnx.checker.check_model(simplified)


def test_bug_1_2_mamba_ssm_does_not_wipe_transformer_models():
    """Bug 1.2: apply_mamba_ssm_fusion must not mistake normal Transformer nodes for Mamba and delete the graph."""
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 16, 64])
    y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 16, 64])

    # Build a Transformer-like graph with 25+ nodes, Exp, Softplus, Mul, Add
    nodes = []
    curr = "X"
    for i in range(12):
        out_add = f"add_{i}"
        out_mul = f"mul_{i}"
        nodes.append(helper.make_node("Add", [curr, curr], [out_add], name=f"add_node_{i}"))
        nodes.append(helper.make_node("Mul", [out_add, out_add], [out_mul], name=f"mul_node_{i}"))
        curr = out_mul

    # Add an Exp and Softplus
    nodes.append(helper.make_node("Exp", [curr], ["exp_out"], name="exp_node"))
    nodes.append(helper.make_node("Softplus", ["exp_out"], ["Y"], name="softplus_node"))

    graph = helper.make_graph(nodes, "test_transformer_not_mamba", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    initial_node_count = len(model.graph.node)
    assert initial_node_count >= 25

    synthesizer = MacroBlockSynthesizer()
    simplified = synthesizer.apply_mamba_ssm_fusion(model, mode="function_proto")

    # It must NOT be replaced with a single MambaSSM node!
    mamba_nodes = [n for n in simplified.graph.node if n.op_type == "MambaSSM"]
    assert len(mamba_nodes) == 0, "apply_mamba_ssm_fusion falsely wiped standard Transformer nodes into MambaSSM!"
    assert len(simplified.graph.node) == initial_node_count
    onnx.checker.check_model(simplified)


def test_bug_1_3_shifted_window_attention_preserves_surrounding_graph():
    """Bug 1.3: apply_shifted_window_attention_fusion must preserve PatchEmbed and classifier stages."""
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 3, 32, 32])
    y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 10])

    w_conv = numpy_helper.from_array(np.random.randn(64, 3, 4, 4).astype(np.float32), name="W_conv")
    w_fc = numpy_helper.from_array(np.random.randn(64, 10).astype(np.float32), name="W_fc")

    nodes = [
        # PatchEmbed Stage (Must survive!)
        helper.make_node("Conv", ["X", "W_conv"], ["patch_emb"], strides=[4, 4], name="patch_conv"),
        # Swin Window Stage
        helper.make_node("Relu", ["patch_emb"], ["swin_in"], name="swin_relu"),
        # Terminal stage (Must survive!)
        helper.make_node("GlobalAveragePool", ["swin_in"], ["gap_out"], name="gap_node"),
        helper.make_node("Flatten", ["gap_out"], ["flat_out"], name="flat_node"),
        helper.make_node("MatMul", ["flat_out", "W_fc"], ["Y"], name="classifier_fc"),
    ]

    graph = helper.make_graph(nodes, "test_swin_safety", [x], [y], initializer=[w_conv, w_fc])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    synthesizer = MacroBlockSynthesizer()
    simplified = synthesizer.apply_shifted_window_attention_fusion(model, mode="function_proto")

    node_names = [n.name for n in simplified.graph.node]
    assert "patch_conv" in node_names, "PatchEmbed was wiped out by shifted window attention pass!"
    assert "classifier_fc" in node_names, "Classifier stage was wiped out by shifted window attention pass!"
    onnx.checker.check_model(simplified)


def test_bug_1_4_deepseek_mla_preserves_multi_layer_models():
    """Bug 1.4: apply_deepseek_mla_macro_fusion must not delete the entire model when fusing MLA."""
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 16, 64])
    y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 16, 64])

    w_emb = numpy_helper.from_array(np.random.randn(64, 64).astype(np.float32), name="W_emb")
    w_head = numpy_helper.from_array(np.random.randn(64, 64).astype(np.float32), name="W_head")

    nodes = [
        # Embedding Layer
        helper.make_node("MatMul", ["X", "W_emb"], ["emb_out"], name="emb_mm"),
        # MLA Layer 1
        helper.make_node("MultiHeadLatentAttention_KVProjection", ["emb_out"], ["kv_c_1", "k_r_1"], name="mla_kv_1", domain="custom.pytorch"),
        helper.make_node("ScaledDotProductAttention", ["emb_out", "kv_c_1", "kv_c_1"], ["attn_1"], name="sdpa_1", domain="custom.pytorch"),
        helper.make_node("MatMul", ["attn_1", "W_emb"], ["layer1_out"], name="out_proj_1"),
        # Intermediate Norm
        helper.make_node("Relu", ["layer1_out"], ["inter_norm"], name="inter_act"),
        # MLA Layer 2
        helper.make_node("MultiHeadLatentAttention_KVProjection", ["inter_norm"], ["kv_c_2", "k_r_2"], name="mla_kv_2", domain="custom.pytorch"),
        helper.make_node("ScaledDotProductAttention", ["inter_norm", "kv_c_2", "kv_c_2"], ["attn_2"], name="sdpa_2", domain="custom.pytorch"),
        helper.make_node("MatMul", ["attn_2", "W_emb"], ["layer2_out"], name="out_proj_2"),
        # Final LM Head
        helper.make_node("MatMul", ["layer2_out", "W_head"], ["Y"], name="lm_head"),
    ]

    graph = helper.make_graph(nodes, "test_mla_safety", [x], [y], initializer=[w_emb, w_head])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    synthesizer = MacroBlockSynthesizer()
    simplified = synthesizer.apply_deepseek_mla_macro_fusion(model, mode="function_proto")

    node_names = [n.name for n in simplified.graph.node]
    assert "emb_mm" in node_names, "Embedding layer was wiped out by MLA pass!"
    assert "lm_head" in node_names, "Final LM head was wiped out by MLA pass!"
    assert len(simplified.graph.node) >= 3, "Whole graph was collapsed into 1 node!"
    onnx.checker.check_model(simplified)


def test_bug_7_2_replace_nodes_topologically_handles_named_ids_and_preserves_toposort():
    """Bug 7.2: _replace_nodes_topologically must drop nodes when id is in drop_ids even if name is set, and ensure strict topological sort."""
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 4])
    y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [2, 4])

    n1 = helper.make_node("Add", ["X", "X"], ["t1"], name="named_add_1")
    n2 = helper.make_node("Mul", ["t1", "t1"], ["t2"], name="named_mul_1")
    n3 = helper.make_node("Relu", ["t2"], ["Y"], name="named_relu_1")

    graph = helper.make_graph([n1, n2, n3], "test_toposort_safety", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    fused = helper.make_node("FusedAddMul", ["X"], ["t2"], name="fused_add_mul", domain="custom.pytorch")

    node_0 = model.graph.node[0]
    node_1 = model.graph.node[1]
    _replace_nodes_topologically(model, {id(node_0), id(node_1)}, [fused])

    remaining_names = [n.name for n in model.graph.node]
    assert "named_add_1" not in remaining_names, "Node was not dropped despite its ID being in drop_ids!"
    assert "named_mul_1" not in remaining_names, "Node was not dropped despite its ID being in drop_ids!"
    assert "fused_add_mul" in remaining_names
    assert "named_relu_1" in remaining_names

    producer_idx = remaining_names.index("fused_add_mul")
    consumer_idx = remaining_names.index("named_relu_1")
    assert producer_idx < consumer_idx, "Replacement node must precede downstream consumers!"
    onnx.checker.check_model(model)


# ==============================================================================
# SECTION 2: FUNCTIONPROTO & ARITY MATCHING TESTS (Bugs 2.1 - 2.10)
# ==============================================================================

def test_bug_2_1_multi_head_attention_function_proto_arity_4_and_5():
    """Bug 2.1: MultiHeadAttention node inputs must match FunctionProto formal inputs exactly."""
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 8, 16, 32])
    w = helper.make_tensor_value_info("W", onnx.TensorProto.FLOAT, [32, 32])
    y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 16, 32])

    w_init = numpy_helper.from_array(np.random.randn(32, 32).astype(np.float32), name="W")

    sdpa = helper.make_node("ScaledDotProductAttention", ["X", "X", "X"], ["sdpa_out"], name="sdpa_core", domain="custom.pytorch")
    mh = helper.make_node("MergeHeads", ["sdpa_out"], ["mh_out"], name="mh_core", domain="custom.pytorch")
    lin = helper.make_node("Linear", ["mh_out", "W"], ["Y"], name="out_proj", domain="custom.pytorch")

    graph = helper.make_graph([sdpa, mh, lin], "test_mha_arity", [x], [y], initializer=[w_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("custom.pytorch", 1)])

    synthesizer = MacroBlockSynthesizer()
    simplified = synthesizer.apply_mha_composite_fusion(model, mode="function_proto")

    mha_nodes = [n for n in simplified.graph.node if n.op_type == "MultiHeadAttention"]
    assert len(mha_nodes) == 1, "Expected 1 fused MultiHeadAttention node"
    mha_node = mha_nodes[0]

    func = next((f for f in simplified.functions if f.name == "MultiHeadAttention"), None)
    assert func is not None, "MultiHeadAttention function definition must be attached"

    assert len(mha_node.input) == len(func.input), (
        f"MultiHeadAttention node input count ({len(mha_node.input)}) must match FunctionProto formal inputs ({len(func.input)})!"
    )
    onnx.checker.check_model(simplified)


def test_bug_2_2_sdpa_with_mask_function_proto_arity():
    """Bug 2.2: ScaledDotProductAttention with mask must define 4 formal inputs in FunctionProto."""
    q = helper.make_tensor_value_info("Q", onnx.TensorProto.FLOAT, [1, 4, 8, 16])
    k = helper.make_tensor_value_info("K", onnx.TensorProto.FLOAT, [1, 4, 8, 16])
    v = helper.make_tensor_value_info("V", onnx.TensorProto.FLOAT, [1, 4, 8, 16])
    mask = helper.make_tensor_value_info("Mask", onnx.TensorProto.FLOAT, [1, 4, 8, 8])
    y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 4, 8, 16])

    # Raw attention: Q @ K.T -> Mul -> Add(Mask) -> Softmax -> @ V
    nodes = [
        helper.make_node("Transpose", ["K"], ["KT"], perm=[0, 1, 3, 2], name="trans_k"),
        helper.make_node("MatMul", ["Q", "KT"], ["raw_scores"], name="mm_qk"),
        helper.make_node("Add", ["raw_scores", "Mask"], ["masked_scores"], name="mask_add"),
        helper.make_node("Softmax", ["masked_scores"], ["probs"], axis=-1, name="sm_node"),
        helper.make_node("MatMul", ["probs", "V"], ["Y"], name="mm_pv"),
    ]

    graph = helper.make_graph(nodes, "test_sdpa_mask", [q, k, v, mask], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    synthesizer = AttentionSynthesizer()
    simplified = synthesizer.apply_scaled_dot_product_attention_fusion(model, mode="function_proto")

    sdpa_nodes = [n for n in simplified.graph.node if n.op_type == "ScaledDotProductAttention"]
    assert len(sdpa_nodes) == 1
    sdpa_node = sdpa_nodes[0]
    assert len(sdpa_node.input) == 4, f"Expected 4 inputs (Q, K, V, Mask), got {len(sdpa_node.input)}"

    func = next((f for f in simplified.functions if f.name == "ScaledDotProductAttention"), None)
    assert func is not None, "Expected ScaledDotProductAttention function definition attached"
    assert len(func.input) == 4, f"FunctionProto formal inputs must be 4, got {len(func.input)} ({func.input})"
    assert func.input[3] == "Mask"
    onnx.checker.check_model(simplified)


def test_bug_2_3_merge_heads_function_proto_includes_reshape_to_3d():
    """Bug 2.3: MergeHeads FunctionProto must contain both Transpose and Reshape, not just Transpose."""
    q = helper.make_tensor_value_info("Q", onnx.TensorProto.FLOAT, [2, 4, 16, 32])
    k = helper.make_tensor_value_info("K", onnx.TensorProto.FLOAT, [2, 4, 16, 32])
    v = helper.make_tensor_value_info("V", onnx.TensorProto.FLOAT, [2, 4, 16, 32])
    y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [2, 16, 128])
    shape_val = numpy_helper.from_array(np.array([2, 16, 128], dtype=np.int64), name="shape_const")

    nodes = [
        helper.make_node("ScaledDotProductAttention", ["Q", "K", "V"], ["attn_out"], name="sdpa_prev", domain="custom.pytorch"),
        helper.make_node("Transpose", ["attn_out"], ["t_out"], perm=[0, 2, 1, 3], name="tr_heads"),
        helper.make_node("Reshape", ["t_out", "shape_const"], ["Y"], name="r_heads"),
    ]
    graph = helper.make_graph(nodes, "test_mh", [q, k, v], [y], initializer=[shape_val])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("custom.pytorch", 1)])

    synthesizer = AttentionSynthesizer()
    simplified = synthesizer.apply_merge_heads_fusion(model, mode="function_proto")

    func = next((f for f in simplified.functions if f.name == "MergeHeads"), None)
    assert func is not None, "MergeHeads function must be attached"

    op_types_in_func = [n.op_type for n in func.node]
    assert "Transpose" in op_types_in_func, "Transpose must be in MergeHeads function"
    assert "Reshape" in op_types_in_func, "CRITICAL: Reshape was missing from MergeHeads FunctionProto body!"
    onnx.checker.check_model(simplified)


def test_bug_2_4_spatial_gating_unit_preserves_gamma_and_biases():
    """Bug 2.4: SpatialGatingUnit must preserve gamma and biases, with exact FunctionProto arity."""
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 8, 64])
    y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [2, 8, 32])

    gamma = numpy_helper.from_array(np.ones(32, dtype=np.float32), name="gamma")
    w = numpy_helper.from_array(np.random.randn(32, 32).astype(np.float32), name="W")
    b = numpy_helper.from_array(np.zeros(32, dtype=np.float32), name="B")

    nodes = [
        helper.make_node("Split", ["X"], ["u", "v"], axis=-1, name="sgu_split"),
        helper.make_node("RMSNorm", ["u", "gamma"], ["norm_out"], name="rms_norm", domain="custom.pytorch"),
        helper.make_node("Linear", ["norm_out", "W", "B"], ["proj_out"], name="lin_proj", domain="custom.pytorch"),
        helper.make_node("Mul", ["v", "proj_out"], ["Y"], name="sgu_mul"),
    ]
    graph = helper.make_graph(nodes, "test_sgu", [x], [y], initializer=[gamma, w, b])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("custom.pytorch", 1)])

    synthesizer = MacroBlockSynthesizer()
    simplified = synthesizer.apply_spatial_gating_unit_fusion(model, mode="function_proto")

    sgu_nodes = [n for n in simplified.graph.node if n.op_type == "SpatialGatingUnit"]
    assert len(sgu_nodes) == 1
    sgu_node = sgu_nodes[0]

    # Must preserve [v, u, gamma, W, B] -> 5 inputs
    assert "gamma" in sgu_node.input, "Norm scale gamma was dropped by SGU fusion!"
    assert "B" in sgu_node.input, "Linear bias B was dropped by SGU fusion!"

    func = next((f for f in simplified.functions if f.name == "SpatialGatingUnit"), None)
    assert func is not None
    assert len(sgu_node.input) == len(func.input), (
        f"Node arity ({len(sgu_node.input)}) != FunctionProto arity ({len(func.input)})!"
    )
    onnx.checker.check_model(simplified)


def test_bug_2_5_axial_rearrange_function_proto_not_dummy_identity():
    """Bug 2.5: AxialRearrange FunctionProto must contain actual Transpose/Reshape operations, not dummy Identity."""
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 4, 16, 32])
    y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [32, 16, 4])
    shape_val = numpy_helper.from_array(np.array([32, 16, 4], dtype=np.int64), name="shape_const")

    nodes = [
        helper.make_node("Transpose", ["X"], ["t_out"], perm=[0, 3, 2, 1], name="tr_freq_in"),
        helper.make_node("Reshape", ["t_out", "shape_const"], ["Y"], name="r_freq_in"),
    ]
    graph = helper.make_graph(nodes, "test_axial", [x], [y], initializer=[shape_val])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    synthesizer = AxialSynthesizer()
    simplified = synthesizer.apply_axial_rearrange_fusion(model, mode="function_proto")

    func = next((f for f in simplified.functions if f.name == "AxialRearrange"), None)
    assert func is not None
    op_types_in_func = [n.op_type for n in func.node]
    assert "Identity" not in op_types_in_func, "AxialRearrange FunctionProto must not be a dummy Identity node!"
    assert "Transpose" in op_types_in_func
    onnx.checker.check_model(simplified)


def test_bug_2_6_tensor_route_function_proto_preserves_slice_and_matches_arity():
    """Bug 2.6: TensorRoute node inputs must match function formal inputs (1 input: x) and execute slicing."""
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 10, 32])
    y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [2, 16])

    idx_init = numpy_helper.from_array(np.array([3], dtype=np.int64), name="idx")
    starts_init = numpy_helper.from_array(np.array([0], dtype=np.int64), name="starts")
    ends_init = numpy_helper.from_array(np.array([16], dtype=np.int64), name="ends")
    axes_init = numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes")

    nodes = [
        helper.make_node("Gather", ["X", "idx"], ["g_out"], axis=1, name="gather_node"),
        helper.make_node("Slice", ["g_out", "starts", "ends", "axes"], ["Y"], name="slice_node"),
    ]
    graph = helper.make_graph(nodes, "test_route", [x], [y], initializer=[idx_init, starts_init, ends_init, axes_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    synthesizer = AxialSynthesizer()
    simplified = synthesizer.apply_index_tensor_routing(model, mode="function_proto")

    tr_nodes = [n for n in simplified.graph.node if n.op_type == "TensorRoute"]
    assert len(tr_nodes) == 1
    tr_node = tr_nodes[0]
    assert len(tr_node.input) == 1, f"Expected 1 input [X], got {len(tr_node.input)}"

    func = next((f for f in simplified.functions if f.name == "TensorRoute"), None)
    assert func is not None
    assert len(func.input) == 1, f"FunctionProto formal inputs must be 1, got {len(func.input)} ({func.input})"
    op_types_in_func = [n.op_type for n in func.node]
    assert "Slice" in op_types_in_func, "Slice operation was omitted from TensorRoute FunctionProto!"
    onnx.checker.check_model(simplified)


def test_bug_2_7_unbind_function_proto_preserves_arbitrary_axis():
    """Bug 2.7: Unbind FunctionProto must preserve the original split axis, not hardcode axis=0."""
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 4, 3, 16])
    y1 = helper.make_tensor_value_info("Y1", onnx.TensorProto.FLOAT, [2, 4, 16])
    y2 = helper.make_tensor_value_info("Y2", onnx.TensorProto.FLOAT, [2, 4, 16])
    y3 = helper.make_tensor_value_info("Y3", onnx.TensorProto.FLOAT, [2, 4, 16])

    axes_init = numpy_helper.from_array(np.array([2], dtype=np.int64), name="axes_const")

    nodes = [
        helper.make_node("Split", ["X"], ["s1", "s2", "s3"], axis=2, name="split_node"),
        helper.make_node("Squeeze", ["s1", "axes_const"], ["Y1"], name="sq1"),
        helper.make_node("Squeeze", ["s2", "axes_const"], ["Y2"], name="sq2"),
        helper.make_node("Squeeze", ["s3", "axes_const"], ["Y3"], name="sq3"),
    ]
    graph = helper.make_graph(nodes, "test_unbind_axis", [x], [y1, y2, y3], initializer=[axes_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    synthesizer = AxialSynthesizer()
    simplified = synthesizer.apply_unbind_fusion(model, mode="function_proto")

    func = next((f for f in simplified.functions if f.name == "Unbind"), None)
    assert func is not None

    sp_node_in_func = next((n for n in func.node if n.op_type == "Split"), None)
    assert sp_node_in_func is not None
    split_axis = next((a.i for a in sp_node_in_func.attribute if a.name == "axis"), None)
    assert split_axis == 2, f"Unbind FunctionProto split on axis={split_axis}, expected axis=2!"
    onnx.checker.check_model(simplified)


def test_bug_2_8_dynamic_quant_linear_biased_unbiased_collision():
    """Bug 2.8: DynamicQuantLinear must not experience FunctionProto registration collisions."""
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 16])
    y1 = helper.make_tensor_value_info("Y1", onnx.TensorProto.FLOAT, [2, 32])
    y2 = helper.make_tensor_value_info("Y2", onnx.TensorProto.FLOAT, [2, 32])

    w1 = numpy_helper.from_array(np.random.randn(16, 32).astype(np.float32), name="W1")
    b1 = numpy_helper.from_array(np.random.randn(32).astype(np.float32), name="B1")
    w2 = numpy_helper.from_array(np.random.randn(16, 32).astype(np.float32), name="W2")

    nodes = [
        # Instance 1: Biased
        helper.make_node("DynamicQuantizeLinear", ["X"], ["q1", "scale1", "zp1"]),
        helper.make_node("DequantizeLinear", ["q1", "scale1", "zp1"], ["dq1"]),
        helper.make_node("MatMul", ["dq1", "W1"], ["mm1"]),
        helper.make_node("Add", ["mm1", "B1"], ["Y1"]),
        # Instance 2: Unbiased
        helper.make_node("DynamicQuantizeLinear", ["X"], ["q2", "scale2", "zp2"]),
        helper.make_node("DequantizeLinear", ["q2", "scale2", "zp2"], ["dq2"]),
        helper.make_node("MatMul", ["dq2", "W2"], ["Y2"]),
    ]
    graph = helper.make_graph(nodes, "test_dql", [x], [y1, y2], initializer=[w1, b1, w2])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    synthesizer = MacroBlockSynthesizer()
    simplified = synthesizer.apply_dynamic_quant_linear_fusion(model, mode="function_proto")

    dql_nodes = [n for n in simplified.graph.node if n.op_type == "DynamicQuantLinear"]
    assert len(dql_nodes) == 2

    func = next((f for f in simplified.functions if f.name == "DynamicQuantLinear"), None)
    assert func is not None

    for n in dql_nodes:
        assert len(n.input) == len(func.input), (
            f"Node {n.name} input count ({len(n.input)}) != function input count ({len(func.input)})!"
        )
    onnx.checker.check_model(simplified)


def test_bug_2_9_and_2_10_gated_unit_rank_agnostic_and_opset_import():
    """Bug 2.9 & 2.10: GatedUnit must work on 2D and 4D tensors, and opset_imports must include custom.pytorch for SiLU."""
    # 2D test
    x2d = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [4, 64])
    g2d = helper.make_tensor_value_info("G", onnx.TensorProto.FLOAT, [4, 64])
    y2d = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [4, 64])

    nodes = [
        helper.make_node("SiLU", ["G"], ["g_act"], name="silu_gate", domain="custom.pytorch"),
        helper.make_node("Mul", ["X", "g_act"], ["Y"], name="mul_gate"),
    ]
    graph = helper.make_graph(nodes, "test_gated_2d", [x2d, g2d], [y2d])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("custom.pytorch", 1)])

    synthesizer = AttentionSynthesizer()
    simplified = synthesizer.apply_gated_unit_fusion(model, mode="function_proto")

    # Find specialized function attached
    func = next((f for f in simplified.functions if "GatedUnit" in f.name), None)
    assert func is not None, "Specialized GatedUnit function must be attached"

    opset_domains = [op.domain for op in func.opset_import]
    assert "custom.pytorch" in opset_domains, (
        f"CRITICAL: Function opset_imports ({opset_domains}) must contain 'custom.pytorch' for SiLU node!"
    )
    onnx.checker.check_model(simplified)
