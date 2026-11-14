"""Unit tests for PassManager and standard optimization passes."""

import numpy as np
import onnx
from onnx import TensorProto, helper

from neuron_toolkit.graph import NeuronGraph
from neuron_toolkit.passes import (
    ConstantFolding,
    DeadCodeElimination,
    FuseConvBN,
    PassManager,
)


def create_conv_bn_model() -> onnx.ModelProto:
    """Create a Conv -> BatchNormalization -> Relu model."""
    in_val = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 10, 10])
    out_val = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 10, 10])

    w_conv = helper.make_tensor("W_conv", TensorProto.FLOAT, [3, 3, 3, 3], np.ones((3, 3, 3, 3), dtype=np.float32).flatten())
    b_conv = helper.make_tensor("B_conv", TensorProto.FLOAT, [3], np.zeros(3, dtype=np.float32))

    scale = helper.make_tensor("scale", TensorProto.FLOAT, [3], np.ones(3, dtype=np.float32))
    b_bn = helper.make_tensor("B_bn", TensorProto.FLOAT, [3], np.zeros(3, dtype=np.float32))
    mean = helper.make_tensor("mean", TensorProto.FLOAT, [3], np.zeros(3, dtype=np.float32))
    var = helper.make_tensor("var", TensorProto.FLOAT, [3], np.ones(3, dtype=np.float32))

    conv_node = helper.make_node("Conv", ["X", "W_conv", "B_conv"], ["conv_out"], name="conv0", pads=[1, 1, 1, 1])
    bn_node = helper.make_node("BatchNormalization", ["conv_out", "scale", "B_bn", "mean", "var"], ["bn_out"], name="bn0")
    relu_node = helper.make_node("Relu", ["bn_out"], ["Y"], name="relu0")

    graph = helper.make_graph(
        [conv_node, bn_node, relu_node],
        "conv_bn_graph",
        [in_val],
        [out_val],
        [w_conv, b_conv, scale, b_bn, mean, var],
    )
    return helper.make_model(graph, producer_name="test")


def create_model_with_dead_and_const_nodes() -> onnx.ModelProto:
    """Create a model with a dead branch and a constant arithmetic node."""
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2])

    c1 = helper.make_tensor("C1", TensorProto.FLOAT, [2, 2], np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32).flatten())
    c2 = helper.make_tensor("C2", TensorProto.FLOAT, [2, 2], np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32).flatten())

    # Constant folding candidate: C1 + C2 -> C_sum
    add_const = helper.make_node("Add", ["C1", "C2"], ["C_sum"], name="const_add")

    # Main active branch: X + C_sum -> Y
    add_main = helper.make_node("Add", ["X", "C_sum"], ["Y"], name="main_add")

    # Dead branch: X -> Relu -> dead_out (never consumed, not in graph outputs)
    dead_node = helper.make_node("Relu", ["X"], ["dead_out"], name="dead_relu")

    graph = helper.make_graph(
        [add_const, add_main, dead_node],
        "dead_const_graph",
        [x],
        [y],
        [c1, c2],
    )
    return helper.make_model(graph, producer_name="test")


def test_fuse_conv_bn_pass():
    model = create_conv_bn_model()
    graph = NeuronGraph(model)
    assert len(graph.nodes) == 3

    pm = PassManager([FuseConvBN()])
    opt_graph, stats = pm.run(graph)

    assert len(opt_graph.nodes) == 2  # Conv and Relu remain, BN fused away
    assert stats.pass_results[0]["fused_pairs"] == 1


def test_dead_code_elimination():
    model = create_model_with_dead_and_const_nodes()
    graph = NeuronGraph(model)
    assert len(graph.nodes) == 3

    pm = PassManager([DeadCodeElimination()])
    opt_graph, stats = pm.run(graph)

    assert len(opt_graph.nodes) == 2
    op_types = [n.op_type for n in opt_graph.nodes]
    assert "Relu" not in op_types
    assert stats.pass_results[0]["removed_nodes"] == 1


def test_constant_folding():
    model = create_model_with_dead_and_const_nodes()
    graph = NeuronGraph(model)

    pm = PassManager([ConstantFolding()])
    opt_graph, stats = pm.run(graph)

    assert stats.pass_results[0]["folded_nodes"] == 1
    assert "C_sum" in opt_graph.tensor_map
    np.testing.assert_allclose(
        opt_graph.tensor_map["C_sum"],
        np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
    )


def test_pass_manager_pipeline_and_summary():
    model = create_model_with_dead_and_const_nodes()
    graph = NeuronGraph(model)

    pm = PassManager([ConstantFolding(), DeadCodeElimination()])
    opt_graph, stats = pm.run(graph)

    assert len(opt_graph.nodes) == 1
    assert opt_graph.nodes[0].name == "main_add"
    assert stats.initial_nodes == 3
    assert stats.final_nodes == 1
    assert stats.node_reduction == 2
    assert "PassManager Execution Summary:" in stats.summary()


def test_graph_optimize_convenience_method():
    model = create_conv_bn_model()
    graph = NeuronGraph(model)

    opt_graph, stats = graph.optimize()
    assert len(opt_graph.nodes) == 2
    assert stats.node_reduction == 1
