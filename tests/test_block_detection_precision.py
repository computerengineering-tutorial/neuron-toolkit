"""Unit tests for Architectural Block Detection (Topology-Verified)."""

from __future__ import annotations

import onnx
from onnx import helper

from neuron_toolkit.compiler.analyzer.block_predictor import ArchitectureBlockPredictor


def test_block_detection_gemma2_softcap():
    """Verify block detector identifies Gemma2 SoftCapped Attention Block."""
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 4, 16, 16])
    y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 4, 16, 16])

    softcap_node = helper.make_node(
        "LogitsSoftCap",
        inputs=["X"],
        outputs=["Y"],
        domain="custom.pytorch",
        name="softcap_0",
        softcap=50.0,
    )
    graph = helper.make_graph([softcap_node], "gemma2_graph", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

    predictor = ArchitectureBlockPredictor()
    manifest = predictor.analyze(model)

    block_types = [b.block_type for b in manifest.detected_blocks]
    assert "Gemma2 SoftCapped Attention Block" in block_types
    block = next(b for b in manifest.detected_blocks if b.block_type == "Gemma2 SoftCapped Attention Block")
    assert block.category == "transformer"
    assert block.confidence >= 0.95


def test_block_detection_deepseek_mla():
    """Verify block detector identifies MultiHeadLatentAttention (MLA) Block."""
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 16, 64])
    k = helper.make_tensor_value_info("K", onnx.TensorProto.FLOAT, [1, 16, 64])
    v = helper.make_tensor_value_info("V", onnx.TensorProto.FLOAT, [1, 16, 64])

    mla_node = helper.make_node(
        "MultiHeadLatentAttention_KVProjection",
        inputs=["X", "w1", "w2", "w3"],
        outputs=["K", "V"],
        domain="custom.pytorch",
        name="mla_0",
    )
    graph = helper.make_graph([mla_node], "mla_graph", [x], [k, v])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

    predictor = ArchitectureBlockPredictor()
    manifest = predictor.analyze(model)

    block_types = [b.block_type for b in manifest.detected_blocks]
    assert "MultiHeadLatentAttention (MLA) Block" in block_types
    block = next(b for b in manifest.detected_blocks if b.block_type == "MultiHeadLatentAttention (MLA) Block")
    assert block.confidence >= 0.95


def test_block_detection_swiglu():
    """Verify block detector identifies SwiGLU FeedForward Block."""
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 16, 64])
    y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 16, 64])

    silu_node = helper.make_node("SiLU", inputs=["X"], outputs=["act_out"], name="silu")
    mul_node = helper.make_node("Mul", inputs=["act_out", "X"], outputs=["Y"], name="mul")

    graph = helper.make_graph([silu_node, mul_node], "swiglu_graph", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

    predictor = ArchitectureBlockPredictor()
    manifest = predictor.analyze(model)

    block_types = [b.block_type for b in manifest.detected_blocks]
    assert "SwiGLU FeedForward Block" in block_types
