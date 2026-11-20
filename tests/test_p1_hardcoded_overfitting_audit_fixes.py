"""Unit and regression tests for Section 4: Hardcoded Overfitting (P1) Audit Fixes.

Covers:
- Bug 4.1: apply_multi_task_audio_head_fusion structural multi-branch topology matching (zero name overfitting).
- Bug 4.2: apply_vit_patch_embedding_fusion structural forward traversal (zero pos_embed/position string checks).
- Bug 4.3: apply_mha_composite_fusion structural KV-cache loop detection (zero past/Concat string checks).
- Bug 4.4: apply_multi_field_embedding_fusion handles namespaced nodes (zero layers/blocks string exclusions).
- Bug 4.5: _find_base_frequency_tensor local DAG traversal (zero global ScatterND search).
- Bug 4.6: apply_axial_rearrange_fusion rank-4/rank-3 fidelity and dynamic permutation preservation.
"""

from __future__ import annotations

import collections
import numpy as np
import onnx
from onnx import helper, numpy_helper
import pytest
import torch

from neuron_toolkit.compiler.base import CompilerContext
from neuron_toolkit.compiler.passes.synthesis.macro_block_synthesizer import MacroBlockSynthesizer
from neuron_toolkit.compiler.passes.synthesis.attention_synthesizer import AttentionSynthesizer
from neuron_toolkit.compiler.passes.synthesis.axial_synthesizer import AxialSynthesizer
from neuron_toolkit.runtime.registry import OpRegistry


# ==============================================================================
# BUG 4.1: Multi-Task Audio Decoder Head Structural Matching
# ==============================================================================
def test_bug_4_1_multi_task_audio_head_arbitrary_output_and_node_names():
    """Bug 4.1: MultiTaskDecoderHead must be synthesized purely from structural fork topology,

    even when outputs and nodes have arbitrary names without 'inst_logp', 'perc_prob',
    'vol_logp', or 'conv_pitched'.
    """
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 16, 64])
    out_pitch = helper.make_tensor_value_info("pitch_semitone", onnx.TensorProto.FLOAT, [1, 16, 128])
    out_onset = helper.make_tensor_value_info("onset_probability", onnx.TensorProto.FLOAT, [1, 16, 1])

    w_fork_a = numpy_helper.from_array(np.random.randn(64, 64).astype(np.float32), name="w_fork_a")
    w_fork_b = numpy_helper.from_array(np.random.randn(64, 64).astype(np.float32), name="w_fork_b")
    w_pitch = numpy_helper.from_array(np.random.randn(64, 128).astype(np.float32), name="w_pitch")
    w_onset = numpy_helper.from_array(np.random.randn(64, 1).astype(np.float32), name="w_onset")

    # Backbone trunk node
    stem_node = helper.make_node("Relu", ["X"], ["trunk_feature"], name="backbone_exit")

    # Dual-branch feature extractors diverging from trunk_feature
    fork_a = helper.make_node("Linear", ["trunk_feature", "w_fork_a"], ["feat_a"], name="extractor_branch_a", domain="custom.pytorch")
    fork_b = helper.make_node("Linear", ["trunk_feature", "w_fork_b"], ["feat_b"], name="extractor_branch_b", domain="custom.pytorch")

    # Head prediction decoders leading to arbitrary graph outputs
    head_pitch = helper.make_node("Linear", ["feat_a", "w_pitch"], ["pitch_semitone"], name="head_pitch_layer", domain="custom.pytorch")
    head_onset = helper.make_node("Linear", ["feat_b", "w_onset"], ["onset_logits"], name="head_onset_layer", domain="custom.pytorch")
    act_onset = helper.make_node("Sigmoid", ["onset_logits"], ["onset_probability"], name="onset_sigmoid")

    nodes = [stem_node, fork_a, fork_b, head_pitch, head_onset, act_onset]
    graph = helper.make_graph(
        nodes,
        "test_multi_task_audio",
        [x],
        [out_pitch, out_onset],
        initializer=[w_fork_a, w_fork_b, w_pitch, w_onset],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("custom.pytorch", 1)],
    )

    synthesizer = MacroBlockSynthesizer()
    simplified = synthesizer.apply_multi_task_audio_head_fusion(model, mode="function_proto")

    head_nodes = [n for n in simplified.graph.node if n.op_type == "MultiTaskDecoderHead"]
    assert len(head_nodes) == 1, f"Expected 1 fused MultiTaskDecoderHead, got {len(head_nodes)}"
    fused_node = head_nodes[0]

    # Verify that the outputs match the actual model outputs, not hardcoded strings
    assert fused_node.output == ["pitch_semitone", "onset_probability"]
    assert fused_node.input[0] == "trunk_feature"

    # Verify FunctionProto definition
    func = next((f for f in simplified.functions if f.name == "MultiTaskDecoderHead"), None)
    assert func is not None, "Expected MultiTaskDecoderHead FunctionProto definition"
    assert func.output == ["pitch_semitone", "onset_probability"]
    onnx.checker.check_model(simplified)


def test_bug_4_1_backward_compatibility_with_model4():
    """Verify that model4.onnx (the original target of multi-task head) still fuses 100% cleanly."""
    from pathlib import Path
    model_path = Path(".temp/diffrent_models/model4.onnx")
    if not model_path.exists():
        pytest.skip("model4.onnx not present in .temp")

    m = onnx.load(str(model_path))
    synthesizer = MacroBlockSynthesizer()
    simplified = synthesizer.apply_multi_task_audio_head_fusion(m, mode="function_proto")

    head_nodes = [n for n in simplified.graph.node if n.op_type == "MultiTaskDecoderHead"]
    assert len(head_nodes) == 1
    assert head_nodes[0].output == ["inst_logp", "perc_prob", "vol_logp"]


# ==============================================================================
# BUG 4.2: ViT Patch Embedding Forward Structural Traversal
# ==============================================================================
def test_bug_4_2_vit_patch_embedding_arbitrary_position_tensor_name():
    """Bug 4.2: ViTPatchEmbedding must be synthesized even when positional embedding

    is named 'spatial_pe_weight' (zero 'pos_embed' or 'position' in name) and without Concat.
    """
    img = helper.make_tensor_value_info("img", onnx.TensorProto.FLOAT, [1, 3, 16, 16])
    out = helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [1, 16, 32])

    w_conv = numpy_helper.from_array(np.random.randn(32, 3, 4, 4).astype(np.float32), name="conv_w")
    b_conv = numpy_helper.from_array(np.zeros(32, dtype=np.float32), name="conv_b")
    # Non-standard tensor name without 'pos_embed' or 'position'
    pe_weight = numpy_helper.from_array(np.random.randn(1, 16, 32).astype(np.float32), name="spatial_pe_weight")
    shape_val = numpy_helper.from_array(np.array([1, 32, 16], dtype=np.int64), name="shape_const")

    conv = helper.make_node(
        "Conv",
        inputs=["img", "conv_w", "conv_b"],
        outputs=["conv_out"],
        kernel_shape=[4, 4],
        strides=[4, 4],
        name="patch_conv",
    )
    reshape = helper.make_node("Reshape", inputs=["conv_out", "shape_const"], outputs=["reshaped"], name="flatten_spatial")
    transpose = helper.make_node("Transpose", inputs=["reshaped"], outputs=["transposed"], perm=[0, 2, 1], name="permute_seq")
    add_pe = helper.make_node("Add", inputs=["transposed", "spatial_pe_weight"], outputs=["out"], name="add_pe_node")

    nodes = [conv, reshape, transpose, add_pe]
    graph = helper.make_graph(
        nodes,
        "test_vit_no_heuristics",
        [img],
        [out],
        initializer=[w_conv, b_conv, pe_weight, shape_val],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("custom.pytorch", 1)])

    synthesizer = MacroBlockSynthesizer()
    simplified = synthesizer.apply_vit_patch_embedding_fusion(model, mode="function_proto")

    vit_nodes = [n for n in simplified.graph.node if n.op_type == "ViTPatchEmbedding"]
    assert len(vit_nodes) == 1, f"Expected 1 fused ViTPatchEmbedding, got {len(vit_nodes)}"
    assert vit_nodes[0].output == ["out"]
    onnx.checker.check_model(simplified)


# ==============================================================================
# BUG 4.3: MHA Stateful Attention Structural KV-Cache Loop Detection
# ==============================================================================
def test_bug_4_3_mha_kv_cache_concat_structural_detection():
    """Bug 4.3: StatefulAttention must be chosen when K or V is formed via Concat,

    even if the tensor is named 'kv_history_tensor' (zero 'past' substring).
    """
    q_val = helper.make_tensor_value_info("Q", onnx.TensorProto.FLOAT, [1, 8, 1, 32])
    k_curr = helper.make_tensor_value_info("K_curr", onnx.TensorProto.FLOAT, [1, 8, 1, 32])
    v_curr = helper.make_tensor_value_info("V_curr", onnx.TensorProto.FLOAT, [1, 8, 1, 32])
    cache_k = helper.make_tensor_value_info("cache_k", onnx.TensorProto.FLOAT, [1, 8, 15, 32])
    cache_v = helper.make_tensor_value_info("cache_v", onnx.TensorProto.FLOAT, [1, 8, 15, 32])
    y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 1, 32])

    w_out = numpy_helper.from_array(np.random.randn(32, 32).astype(np.float32), name="W_out")

    # KV concatenation on sequence axis (axis=2)
    cat_k = helper.make_node("Concat", ["cache_k", "K_curr"], ["kv_history_k"], axis=2, name="cat_k_node")
    cat_v = helper.make_node("Concat", ["cache_v", "V_curr"], ["kv_history_v"], axis=2, name="cat_v_node")

    sdpa = helper.make_node("ScaledDotProductAttention", ["Q", "kv_history_k", "kv_history_v"], ["sdpa_out"], name="sdpa_core", domain="custom.pytorch")
    merge = helper.make_node("MergeHeads", ["sdpa_out"], ["mh_out"], name="mh_core", domain="custom.pytorch")
    out_proj = helper.make_node("Linear", ["mh_out", "W_out"], ["Y"], name="out_proj", domain="custom.pytorch")

    nodes = [cat_k, cat_v, sdpa, merge, out_proj]
    graph = helper.make_graph(
        nodes,
        "test_stateful_mha",
        [q_val, k_curr, v_curr, cache_k, cache_v],
        [y],
        initializer=[w_out],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("custom.pytorch", 1)])

    synthesizer = MacroBlockSynthesizer()
    simplified = synthesizer.apply_mha_composite_fusion(model, mode="function_proto")

    stateful_nodes = [n for n in simplified.graph.node if n.op_type == "StatefulAttention"]
    assert len(stateful_nodes) == 1, "Expected StatefulAttention to be chosen via Concat topology!"


def test_bug_4_3_no_false_positive_stateful_attention_from_past_in_tensor_name():
    """Bug 4.3: Having 'past' in tensor name (e.g. 'past_token_embedding') must NOT

    cause normal MultiHeadAttention to be misclassified as StatefulAttention.
    """
    x = helper.make_tensor_value_info("past_token_embedding", onnx.TensorProto.FLOAT, [1, 8, 16, 32])
    w = helper.make_tensor_value_info("W", onnx.TensorProto.FLOAT, [32, 32])
    y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 16, 32])
    w_init = numpy_helper.from_array(np.random.randn(32, 32).astype(np.float32), name="W")

    sdpa = helper.make_node("ScaledDotProductAttention", ["past_token_embedding", "past_token_embedding", "past_token_embedding"], ["sdpa_out"], name="sdpa_core", domain="custom.pytorch")
    mh = helper.make_node("MergeHeads", ["sdpa_out"], ["mh_out"], name="mh_core", domain="custom.pytorch")
    lin = helper.make_node("Linear", ["mh_out", "W"], ["Y"], name="out_proj", domain="custom.pytorch")

    graph = helper.make_graph([sdpa, mh, lin], "test_no_false_positive_past", [x], [y], initializer=[w_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("custom.pytorch", 1)])

    synthesizer = MacroBlockSynthesizer()
    simplified = synthesizer.apply_mha_composite_fusion(model, mode="function_proto")

    mha_nodes = [n for n in simplified.graph.node if n.op_type == "MultiHeadAttention"]
    assert len(mha_nodes) == 1, "Must synthesize standard MultiHeadAttention, NOT StatefulAttention!"


# ==============================================================================
# BUG 4.4: Multi-Field Embedding In Namespaced Nodes
# ==============================================================================
def test_bug_4_4_multi_field_embedding_with_layers_and_blocks_in_names():
    """Bug 4.4: MultiFieldEmbedding must NOT reject nodes whose names contain

    'layers' or 'blocks'.
    """
    feat_x = helper.make_tensor_value_info("feat_input", onnx.TensorProto.INT64, [1, 10])
    out = helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [1, 10, 32])

    w_emb1 = numpy_helper.from_array(np.random.randn(100, 32).astype(np.float32), name="w_emb1")
    w_emb2 = numpy_helper.from_array(np.random.randn(50, 32).astype(np.float32), name="w_emb2")
    pe = numpy_helper.from_array(np.random.randn(1, 10, 32).astype(np.float32), name="pe_weights")

    idx0 = numpy_helper.from_array(np.array(0, dtype=np.int64), name="idx0")
    idx1 = numpy_helper.from_array(np.array(1, dtype=np.int64), name="idx1")

    # Nodes with 'layers' and 'blocks' in names
    s1 = helper.make_node("Gather", ["feat_input", "idx0"], ["f1"], axis=1, name="model.layers.0.slice_f1")
    s2 = helper.make_node("Gather", ["feat_input", "idx1"], ["f2"], axis=1, name="model.layers.0.slice_f2")
    g1 = helper.make_node("Gather", ["w_emb1", "f1"], ["e1"], name="model.layers.0.embed_table_1")
    g2 = helper.make_node("Gather", ["w_emb2", "f2"], ["e2"], name="model.layers.0.embed_table_2")
    add_feats = helper.make_node("Add", ["e1", "e2"], ["sum_emb"], name="model.blocks.0.sum_embeds")
    add_pe = helper.make_node("Add", ["sum_emb", "pe_weights"], ["out"], name="model.blocks.0.add_positional")

    nodes = [s1, s2, g1, g2, add_feats, add_pe]
    graph = helper.make_graph(
        nodes,
        "test_multi_field_namespaced",
        [feat_x],
        [out],
        initializer=[w_emb1, w_emb2, pe, idx0, idx1],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("custom.pytorch", 1)])

    synthesizer = MacroBlockSynthesizer()
    simplified = synthesizer.apply_multi_field_embedding_fusion(model, mode="function_proto")

    emb_nodes = [n for n in simplified.graph.node if "MultiFieldEmbedding" in n.op_type]
    assert len(emb_nodes) == 1, "MultiFieldEmbedding must be synthesized despite 'layers'/'blocks' in node names!"
    onnx.checker.check_model(simplified)


# ==============================================================================
# BUG 4.5: RoPE Base Frequency Local DAG Traversal
# ==============================================================================
def test_bug_4_5_find_base_frequency_does_not_jump_to_unrelated_scatternd():
    """Bug 4.5: _find_base_frequency_tensor must not jump to an unrelated ScatterND in the graph."""
    synthesizer = AttentionSynthesizer()

    # Create a local DAG: theta -> Reshape -> Einsum -> freq_table
    # And an unrelated ScatterND node
    node_map = {
        "theta": helper.make_node("Reshape", ["raw_theta", "shape_c"], ["theta"], name="reshape_theta"),
        "raw_theta": helper.make_node("Einsum", ["pos_idx", "inv_freq"], ["raw_theta"], name="einsum_theta"),
        # Unrelated ScatterND
        "unrelated_out": helper.make_node("ScatterND", ["data_t", "idx_t", "up_t"], ["unrelated_out"], name="unrelated_scatter"),
    }

    base = synthesizer._find_base_frequency_tensor("theta", node_map)
    # Must stay on the local path of theta and raw_theta, NOT jumping to unrelated_scatter's input 'data_t'
    assert base != "data_t", "Must not grab input of unrelated ScatterND node!"
    assert base in ("raw_theta", "theta")


# ==============================================================================
# BUG 4.6: Axial Rearrange Permutation and Rank Fidelity
# ==============================================================================
def test_bug_4_6_axial_rearrange_arbitrary_permutation_and_function_proto_fidelity():
    """Bug 4.6: AxialRearrange must handle arbitrary permutations and produce

    a valid FunctionProto definition without 3D/4D rank mismatches.
    """
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [16, 8, 32])  # 3D: (B*T, F, C)
    y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [8, 16, 32])  # 3D: (B*F, T, C)

    shape_4d = numpy_helper.from_array(np.array([1, 16, 8, 32], dtype=np.int64), name="shape_4d")
    shape_3d = numpy_helper.from_array(np.array([8, 16, 32], dtype=np.int64), name="shape_3d")

    # Reshape 3D -> 4D, Transpose 4D -> 4D, Reshape 4D -> 3D
    r1 = helper.make_node("Reshape", ["X", "shape_4d"], ["x_4d"], name="r_unflatten")
    tr = helper.make_node("Transpose", ["x_4d"], ["t_out"], perm=[0, 2, 1, 3], name="tr_swap")
    r2 = helper.make_node("Reshape", ["t_out", "shape_3d"], ["Y"], name="r_flatten")

    nodes = [r1, tr, r2]
    graph = helper.make_graph(nodes, "test_axial_fidelity", [x], [y], initializer=[shape_4d, shape_3d])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("custom.pytorch", 1)])

    synthesizer = AxialSynthesizer()
    simplified = synthesizer.apply_axial_rearrange_fusion(model, mode="function_proto")

    axial_nodes = [n for n in simplified.graph.node if n.op_type == "AxialRearrange"]
    assert len(axial_nodes) == 1, "Expected 1 fused AxialRearrange node"

    # Verify FunctionProto definition exists and is valid ONNX
    func = next((f for f in simplified.functions if f.name == "AxialRearrange"), None)
    assert func is not None, "AxialRearrange FunctionProto must be registered"
    onnx.checker.check_model(simplified)

    # Verify execution in registry
    reg = OpRegistry()
    inp_t = torch.randn(16, 8, 32)
    out_t = reg.get("custom.pytorch", "AxialRearrange")([inp_t], {"mode": "freq_to_time", "batch_size": 1, "perm": [0, 2, 1, 3]})
    assert out_t.shape == torch.Size([8, 16, 32])
