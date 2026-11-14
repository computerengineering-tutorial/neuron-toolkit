"""Tests for the FLOPs profiler, memory timeline simulator, and Roofline model."""

import numpy as np
import onnx
from onnx import TensorProto, helper

from neuron_toolkit import NeuronGraph
from neuron_toolkit.profiler import (
    NVIDIA_A100_SXM4_80GB,
    FLOPsProfiler,
    MemoryTimelineSimulator,
    RooflineAnalysis,
)


def _make_conv_model() -> onnx.ModelProto:
    """Create a Conv + Relu ONNX model."""
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 32, 32])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 32, 32])

    w_arr = np.random.randn(16, 3, 3, 3).astype(np.float32)
    b_arr = np.random.randn(16).astype(np.float32)

    w = helper.make_tensor("W", TensorProto.FLOAT, [16, 3, 3, 3], w_arr.flatten())
    b = helper.make_tensor("B", TensorProto.FLOAT, [16], b_arr.flatten())

    conv = helper.make_node("Conv", ["X", "W", "B"], ["conv_out"], pads=[1, 1, 1, 1])
    relu = helper.make_node("Relu", ["conv_out"], ["Y"])

    graph = helper.make_graph([conv, relu], "conv_net", [x], [y], initializer=[w, b])
    return helper.make_model(
        graph, producer_name="test", opset_imports=[helper.make_opsetid("", 17)]
    )


def _make_transformer_layer_model() -> onnx.ModelProto:
    """Create an Attention + RMSNorm ONNX model."""
    q = helper.make_tensor_value_info("Q", TensorProto.FLOAT, [1, 8, 32, 64])
    k = helper.make_tensor_value_info("K", TensorProto.FLOAT, [1, 8, 32, 64])
    v = helper.make_tensor_value_info("V", TensorProto.FLOAT, [1, 8, 32, 64])
    out = helper.make_tensor_value_info("Out", TensorProto.FLOAT, [1, 8, 32, 64])

    attn = helper.make_node("ScaledDotProductAttention", ["Q", "K", "V"], ["attn_out"])
    norm = helper.make_node("RMSNorm", ["attn_out"], ["Out"])

    graph = helper.make_graph([attn, norm], "transformer_layer", [q, k, v], [out])
    return helper.make_model(
        graph, producer_name="test", opset_imports=[helper.make_opsetid("", 17)]
    )


def test_flops_profiler_conv():
    model = _make_conv_model()
    graph = NeuronGraph(model)

    summary = FLOPsProfiler.profile(graph, default_dtype="float32")
    # Expected Conv MACs: 1 * 16 * (32*32) * 3 * (3*3) = 16 * 1024 * 27 = 442,368 MACs
    assert summary.total_macs == 442368
    # Conv FLOPs: 2 * 442368 + (16 * 1024 bias) = 884736 + 16384 = 901,120 FLOPs + 1024 Relu FLOPs
    assert summary.total_flops > 900000
    assert summary.total_params == (16 * 3 * 3 * 3) + 16


def test_memory_timeline_simulator():
    model = _make_conv_model()
    graph = NeuronGraph(model)

    mem_summary = MemoryTimelineSimulator.simulate(graph, default_dtype="float32")
    assert mem_summary.naive_total_bytes > 0
    assert mem_summary.peak_working_set_bytes > 0
    assert mem_summary.peak_working_set_bytes <= mem_summary.naive_total_bytes
    assert len(mem_summary.timeline) > 0


def test_roofline_model_analysis():
    model = _make_conv_model()
    graph = NeuronGraph(model)
    flops_summary = FLOPsProfiler.profile(graph, default_dtype="float16")

    # Analyze on NVIDIA A100
    analysis = RooflineAnalysis.analyze(flops_summary, device="a100", dtype="float16")
    assert analysis.target.name == NVIDIA_A100_SXM4_80GB.name
    # A100 FP16 Ridge point = 312e12 / 2039e9 = ~153 FLOP/byte
    assert 150 < analysis.ridge_point < 160
    assert analysis.global_point.estimated_latency_ms > 0

    # Analyze on Apple M3 Max
    m3_analysis = RooflineAnalysis.analyze(flops_summary, device="m3", dtype="float16")
    assert "Apple M3" in m3_analysis.target.name
    assert m3_analysis.ridge_point > 0


def test_full_graph_profile_and_reports():
    model = _make_transformer_layer_model()
    graph = NeuronGraph(model)

    report = graph.profile(device="a100", dtype="float16")

    # Dictionary serialization
    d = report.to_dict()
    assert "hardware_target" in d
    assert "total_gflops" in d
    assert "peak_working_set_mb" in d
    assert "bound_classification" in d

    # Text summary
    text_summary = report.summary()
    assert "NEURON TOOLKIT HARDWARE & MEMORY PROFILER" in text_summary

    # Markdown export
    md = report.to_markdown()
    assert "| **Hardware Profile:" in md or "### Hardware Profile:" in md
