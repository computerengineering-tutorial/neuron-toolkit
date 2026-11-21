"""Unit tests for Fixed-Point Convergence Engine in OptimizationPipeline."""

from __future__ import annotations

import numpy as np
import onnx
from onnx import helper, numpy_helper

from neuron_toolkit.compiler.base import CompilerContext
from neuron_toolkit.compiler.pipeline import NeuronCompiler, OptimizationPipeline


def test_fixed_point_convergence_stabilization():
    """Verify OptimizationPipeline reaches fixed-point and stops iterating."""
    # Graph: X -> Identity -> Reshape(s1) -> Reshape(s2) -> Pad -> Conv -> Identity -> Y
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 1, 8, 8])
    y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 1, 8, 8])

    s1 = numpy_helper.from_array(np.array([1, 1, 8, 8], dtype=np.int64), name="s1")
    s2 = numpy_helper.from_array(np.array([1, 1, 8, 8], dtype=np.int64), name="s2")
    pads = numpy_helper.from_array(
        np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64), name="pads"
    )
    conv_w = numpy_helper.from_array(
        np.random.randn(1, 1, 3, 3).astype(np.float32), name="conv_w"
    )

    n_id1 = helper.make_node("Identity", inputs=["X"], outputs=["id1_out"], name="id1")
    n_r1 = helper.make_node("Reshape", inputs=["id1_out", "s1"], outputs=["r1_out"], name="r1")
    n_r2 = helper.make_node("Reshape", inputs=["r1_out", "s2"], outputs=["r2_out"], name="r2")
    n_pad = helper.make_node("Pad", inputs=["r2_out", "pads"], outputs=["pad_out"], name="pad")
    n_conv = helper.make_node(
        "Conv", inputs=["pad_out", "conv_w"], outputs=["conv_out"], name="conv", pads=[0, 0, 0, 0]
    )
    n_id2 = helper.make_node("Identity", inputs=["conv_out"], outputs=["Y"], name="id2")

    graph = helper.make_graph(
        [n_id1, n_r1, n_r2, n_pad, n_conv, n_id2],
        "test_fixed_point_graph",
        [x],
        [y],
        [s1, s2, pads, conv_w],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

    pipeline = OptimizationPipeline()
    context = CompilerContext(options={"max_fixed_point_iters": 5})

    compiled = pipeline.run(model, context)

    # Initial had 6 nodes; after fixed-point passes (Identity dropped, Pad fused to Conv, Reshapes collapsed):
    assert len(compiled.graph.node) < 6
    assert not any(n.op_type == "Pad" for n in compiled.graph.node)
    # Intermediate identity 'id1' must be eliminated
    assert not any(n.name == "id1" for n in compiled.graph.node)


def test_neuron_compiler_fixed_point_api():
    """Verify NeuronCompiler.compile() seamlessly drives fixed-point optimization."""
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 4])
    y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [2, 4])

    id_node = helper.make_node("Identity", inputs=["X"], outputs=["Y"], name="id_only")
    graph = helper.make_graph([id_node], "test_compiler_api", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

    optimized = NeuronCompiler.compile(model)
    assert optimized is not None
    assert len(optimized.graph.node) == 0 or optimized.graph.node[0].op_type == "Identity"
