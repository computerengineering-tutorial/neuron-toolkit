"""Unit tests for Advanced Model Simplification Passes.

Tests:
1. Chained Reshape Collapse: Reshape(s1) -> Reshape(s2) => Reshape(s2)
2. Pad into Conv Fusion: Pad + Conv => Conv(pads=combined)
3. Chained Cast Elimination: Cast(A -> B) -> Cast(B -> C) => Cast(A -> C)
4. Unsqueeze + Squeeze Cancellation
5. Gemma-2 Logits Soft-Capping Fusion: Div -> Tanh -> Mul => LogitsSoftCap
6. DeepSeek Multi-Head Latent Attention (MLA) low-rank KV fusion
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import helper, numpy_helper

from neuron_toolkit.compiler.passes.canonicalization.shape_canonicalizer import (
    ShapeCanonicalizer,
)
from neuron_toolkit.compiler.passes.synthesis.attention_synthesizer import (
    AttentionSynthesizer,
)


def test_consecutive_reshape_collapsing():
    """Verify Reshape(X, s1) -> Reshape(interm, s2) collapses to a single Reshape(X, s2)."""
    # Graph: X [1, 16] -> Reshape([4, 4]) -> interm -> Reshape([2, 8]) -> Y
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 16])
    y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [2, 8])

    s1 = numpy_helper.from_array(np.array([4, 4], dtype=np.int64), name="s1")
    s2 = numpy_helper.from_array(np.array([2, 8], dtype=np.int64), name="s2")

    r1 = helper.make_node("Reshape", inputs=["X", "s1"], outputs=["interm"], name="r1")
    r2 = helper.make_node("Reshape", inputs=["interm", "s2"], outputs=["Y"], name="r2")

    graph = helper.make_graph([r1, r2], "test_reshape_graph", [x], [y], [s1, s2])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

    shape_pass = ShapeCanonicalizer()
    simplified = shape_pass.apply_reshape_collapsing(model)

    reshape_nodes = [n for n in simplified.graph.node if n.op_type == "Reshape"]
    assert len(reshape_nodes) == 1, f"Expected 1 Reshape node, found {len(reshape_nodes)}"
    assert reshape_nodes[0].input[0] == "X"
    assert reshape_nodes[0].output[0] == "Y"


def test_pad_into_conv_fusion():
    """Verify Pad(data, pads) -> Conv(pads) fuses into Conv with combined pads."""
    # Input X [1, 1, 8, 8]
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 1, 8, 8])
    y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 1, 8, 8])

    # Pad with [0, 0, 1, 1, 0, 0, 1, 1] (top=1, left=1, bottom=1, right=1)
    pads_init = numpy_helper.from_array(
        np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64), name="pads_init"
    )
    weight_init = numpy_helper.from_array(
        np.random.randn(1, 1, 3, 3).astype(np.float32), name="conv_w"
    )

    pad_node = helper.make_node(
        "Pad", inputs=["X", "pads_init"], outputs=["padded"], name="pad_0"
    )
    conv_node = helper.make_node(
        "Conv",
        inputs=["padded", "conv_w"],
        outputs=["Y"],
        name="conv_0",
        pads=[0, 0, 0, 0],
    )

    graph = helper.make_graph(
        [pad_node, conv_node],
        "test_pad_conv",
        [x],
        [y],
        [pads_init, weight_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

    shape_pass = ShapeCanonicalizer()
    simplified = shape_pass.apply_pad_conv_fusion(model)

    assert not any(n.op_type == "Pad" for n in simplified.graph.node)
    convs = [n for n in simplified.graph.node if n.op_type == "Conv"]
    assert len(convs) == 1
    assert convs[0].input[0] == "X"

    # Check that pads attribute is [1, 1, 1, 1]
    pads_attr = next(a for a in convs[0].attribute if a.name == "pads")
    assert list(pads_attr.ints) == [1, 1, 1, 1]


def test_redundant_cast_elimination():
    """Verify chained Cast(FLOAT -> FLOAT16) -> Cast(FLOAT16 -> FLOAT) collapses."""
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [4, 4])
    y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [4, 4])

    cast1 = helper.make_node(
        "Cast", inputs=["X"], outputs=["interm"], to=onnx.TensorProto.FLOAT16, name="cast1"
    )
    cast2 = helper.make_node(
        "Cast", inputs=["interm"], outputs=["Y"], to=onnx.TensorProto.FLOAT, name="cast2"
    )

    graph = helper.make_graph([cast1, cast2], "test_cast", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

    shape_pass = ShapeCanonicalizer()
    simplified = shape_pass.apply_redundant_cast_elimination(model)

    casts = [n for n in simplified.graph.node if n.op_type == "Cast"]
    assert len(casts) == 1
    assert casts[0].input[0] == "X"
    assert casts[0].output[0] == "Y"


def test_squeeze_unsqueeze_cancellation():
    """Verify Unsqueeze followed by matching Squeeze along same axis cancels out."""
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [4, 4])
    y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [4, 4])

    axes_init = numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes")

    unsq = helper.make_node(
        "Unsqueeze", inputs=["X", "axes"], outputs=["unsq_out"], name="unsq"
    )
    sq = helper.make_node(
        "Squeeze", inputs=["unsq_out", "axes"], outputs=["Y"], name="sq"
    )

    graph = helper.make_graph([unsq, sq], "test_sq_unsq", [x], [y], [axes_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

    shape_pass = ShapeCanonicalizer()
    simplified = shape_pass.apply_squeeze_unsqueeze_cancellation(model)

    assert not any(n.op_type in ("Squeeze", "Unsqueeze") for n in simplified.graph.node)


def test_gemma2_softcapping_fusion():
    """Verify Div -> Tanh -> Mul logits soft-capping fuses into LogitsSoftCap."""
    # Graph: scores -> Div(scores, 50.0) -> Tanh -> Mul(..., 50.0) -> out
    scores = helper.make_tensor_value_info("scores", onnx.TensorProto.FLOAT, [1, 4, 16, 16])
    out = helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [1, 4, 16, 16])

    cap_init = numpy_helper.from_array(np.array([50.0], dtype=np.float32), name="cap_50")

    div_node = helper.make_node(
        "Div", inputs=["scores", "cap_50"], outputs=["div_out"], name="div_0"
    )
    tanh_node = helper.make_node(
        "Tanh", inputs=["div_out"], outputs=["tanh_out"], name="tanh_0"
    )
    mul_node = helper.make_node(
        "Mul", inputs=["tanh_out", "cap_50"], outputs=["out"], name="mul_0"
    )

    graph = helper.make_graph(
        [div_node, tanh_node, mul_node],
        "test_gemma2_softcap",
        [scores],
        [out],
        [cap_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

    attn_pass = AttentionSynthesizer()
    simplified = attn_pass.apply_gemma2_softcapping_fusion(model, mode="function_proto")

    softcap_nodes = [n for n in simplified.graph.node if n.op_type == "LogitsSoftCap"]
    assert len(softcap_nodes) == 1
    assert softcap_nodes[0].domain == "custom.pytorch"
    assert softcap_nodes[0].input[0] == "scores"
    assert softcap_nodes[0].output[0] == "out"

    # Verify formal FunctionProto is attached
    assert any(f.name == "LogitsSoftCap" for f in simplified.functions)


def test_deepseek_mla_fusion():
    """Verify DeepSeek MLA low-rank KV latent compression fuses into MLA KV block."""
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 16, 64])
    out_k = helper.make_tensor_value_info("out_k", onnx.TensorProto.FLOAT, [1, 16, 64])
    out_v = helper.make_tensor_value_info("out_v", onnx.TensorProto.FLOAT, [1, 16, 64])

    w_down = numpy_helper.from_array(np.random.randn(64, 16).astype(np.float32), name="w_down")
    w_up_k = numpy_helper.from_array(np.random.randn(16, 64).astype(np.float32), name="w_up_k")
    w_up_v = numpy_helper.from_array(np.random.randn(16, 64).astype(np.float32), name="w_up_v")

    down_proj = helper.make_node("MatMul", inputs=["X", "w_down"], outputs=["c_kv"], name="down")
    up_k = helper.make_node("MatMul", inputs=["c_kv", "w_up_k"], outputs=["out_k"], name="up_k")
    up_v = helper.make_node("MatMul", inputs=["c_kv", "w_up_v"], outputs=["out_v"], name="up_v")

    graph = helper.make_graph(
        [down_proj, up_k, up_v],
        "test_mla",
        [x],
        [out_k, out_v],
        [w_down, w_up_k, w_up_v],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

    attn_pass = AttentionSynthesizer()
    simplified = attn_pass.apply_deepseek_mla_fusion(model)

    mla_nodes = [
        n for n in simplified.graph.node if n.op_type == "MultiHeadLatentAttention_KVProjection"
    ]
    assert len(mla_nodes) == 1
    assert mla_nodes[0].domain == "custom.pytorch"
    assert mla_nodes[0].output == ["out_k", "out_v"]
