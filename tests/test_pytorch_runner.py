"""Tests for Vectorized PyTorch Runtime Graph Execution Engine."""

from __future__ import annotations

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import pytest
import torch

from neuron_toolkit.runtime.runner import PyTorchRunner, toposort_graph
from neuron_toolkit.runtime.registry import OpRegistry


def test_pytorch_runner_basic_math_and_activations():
    """Test PyTorch execution of basic arithmetic and activation functions."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])

    c2 = helper.make_node("Constant", inputs=[], outputs=["c2"], value=helper.make_tensor("c2_t", TensorProto.FLOAT, [1], [2.0]))
    mul = helper.make_node("Mul", inputs=["X", "c2"], outputs=["mul_out"])
    relu = helper.make_node("Relu", inputs=["mul_out"], outputs=["Y"])

    graph = helper.make_graph(nodes=[c2, mul, relu], name="MathAct", inputs=[X], outputs=[Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

    runner = PyTorchRunner(model)
    x_input = np.array([[-1.0, 2.0, -3.0, 4.0], [0.5, -0.5, 1.5, -2.5]], dtype=np.float32)
    outs = runner.run({"X": x_input})

    expected = np.maximum(0.0, x_input * 2.0)
    assert np.allclose(outs["Y"], expected)


def test_pytorch_runner_conv2d_and_pooling():
    """Test PyTorch execution of Conv2d and MaxPool2d."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2, 8, 8])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 4, 4])

    w_arr = np.ones((4, 2, 3, 3), dtype=np.float32) * 0.1
    b_arr = np.zeros(4, dtype=np.float32)
    w_init = numpy_helper.from_array(w_arr, name="W")
    b_init = numpy_helper.from_array(b_arr, name="B")

    conv = helper.make_node(
        "Conv",
        inputs=["X", "W", "B"],
        outputs=["conv_out"],
        pads=[1, 1, 1, 1],
        kernel_shape=[3, 3],
    )
    pool = helper.make_node(
        "MaxPool",
        inputs=["conv_out"],
        outputs=["Y"],
        kernel_shape=[2, 2],
        strides=[2, 2],
    )

    graph = helper.make_graph(
        nodes=[conv, pool],
        name="ConvPool",
        inputs=[X],
        outputs=[Y],
        initializer=[w_init, b_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

    runner = PyTorchRunner(model)
    x_input = np.ones((1, 2, 8, 8), dtype=np.float32)
    outs = runner.run({"X": x_input})

    assert outs["Y"].shape == (1, 4, 4, 4)
    assert not np.isnan(outs["Y"]).any()


def test_pytorch_runner_custom_gelu_and_math():
    """Test PyTorch execution of custom domain Gelu and Math ops."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])

    gelu = helper.make_node("Gelu", inputs=["X"], outputs=["gelu_out"], domain="custom.pytorch")
    math_node = helper.make_node(
        "Math",
        inputs=["gelu_out"],
        outputs=["Y"],
        domain="custom.pytorch",
        expr="x * 2.0 + 1.0",
    )

    graph = helper.make_graph(nodes=[gelu, math_node], name="CustomOps", inputs=[X], outputs=[Y])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 14), helper.make_opsetid("custom.pytorch", 1)],
    )

    runner = PyTorchRunner(model)
    x_input = np.array([[0.0, 1.0, -1.0, 2.0]], dtype=np.float32)
    outs = runner.run({"X": x_input})

    # Torch reference
    t_x = torch.from_numpy(x_input)
    expected = (torch.nn.functional.gelu(t_x) * 2.0 + 1.0).numpy()
    assert np.allclose(outs["Y"], expected, atol=1e-5)


def test_toposort_graph():
    """Verify graph topological sorting algorithm."""
    # Nodes in reverse dependency order
    c = helper.make_node("Add", inputs=["b_out", "const"], outputs=["c_out"])
    b = helper.make_node("Relu", inputs=["a_out"], outputs=["b_out"])
    a = helper.make_node("Mul", inputs=["X", "const"], outputs=["a_out"])
    const = helper.make_node("Constant", inputs=[], outputs=["const"], value=helper.make_tensor("c", TensorProto.FLOAT, [1], [1.0]))

    graph = helper.make_graph(
        nodes=[c, b, a, const],
        name="Unsorted",
        inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])],
        outputs=[helper.make_tensor_value_info("c_out", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

    sorted_nodes = toposort_graph(model)
    names = [n.op_type for n in sorted_nodes]
    # 'a' must come before 'b', 'b' must come before 'c'
    idx_a = names.index("Mul")
    idx_b = names.index("Relu")
    idx_c = names.index("Add")
    assert idx_a < idx_b < idx_c
