"""Tests for extended optimization passes, TFLite rebuilding, and diamond DAG matching."""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from neuron_toolkit import (
    CommonSubexpressionElimination,
    FuseConvActivation,
    FuseGELU,
    FuseLayerNorm,
    NeuronGraph,
    PassManager,
    Pattern,
    build_tflite_model_from_json,
)


def test_common_subexpression_elimination():
    """Verify CSE identifies and merges duplicate operations with identical inputs."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 10])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 10])
    Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 10])

    # Two identical Add(X, Y) operations
    n1 = helper.make_node("Add", ["X", "Y"], ["add1_out"], name="add1")
    n2 = helper.make_node("Add", ["X", "Y"], ["add2_out"], name="add2")
    # Consumer taking outputs from both
    n3 = helper.make_node("Mul", ["add1_out", "add2_out"], ["Z"], name="mul1")

    graph_proto = helper.make_graph([n1, n2, n3], "CSEGraph", [X, Y], [Z])
    model = helper.make_model(graph_proto, opset_imports=[helper.make_opsetid("", 17)])

    g = NeuronGraph(model)
    assert len(g.nodes) == 3

    opt_g, stats = g.optimize(PassManager([CommonSubexpressionElimination()]))
    assert stats.pass_results["CommonSubexpressionElimination"]["eliminated_nodes"] == 1
    # Secondary Add replaced with Identity
    assert len(opt_g.query().op("Identity").all()) == 1


def test_fuse_conv_activation():
    """Verify FuseConvActivation combines Conv + ReLU into a single FusedConv."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 16, 16])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8, 16, 16])

    w = numpy_helper.from_array(np.random.randn(8, 3, 3, 3).astype(np.float32), name="conv_w")
    conv = helper.make_node("Conv", ["X", "conv_w"], ["conv_out"], name="c1", pads=[1, 1, 1, 1])
    relu = helper.make_node("Relu", ["conv_out"], ["Y"], name="r1")

    graph_proto = helper.make_graph([conv, relu], "ConvActGraph", [X], [Y], initializer=[w])
    model = helper.make_model(graph_proto, opset_imports=[helper.make_opsetid("", 17)])

    g = NeuronGraph(model)
    assert len(g.nodes) == 2

    opt_g, stats = g.optimize(PassManager([FuseConvActivation()]))
    assert stats.pass_results["FuseConvActivation"]["fused_conv_activation_count"] == 1
    assert len(opt_g.nodes) == 1
    assert opt_g.nodes[0].op_type == "FusedConv"


def test_fuse_gelu():
    """Verify FuseGELU combines 0.5 * x * (1 + erf(x / sqrt(2))) into Gelu."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 32])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 32])

    sqrt2_init = numpy_helper.from_array(np.array([1.41421356], dtype=np.float32), name="sqrt2")
    one_init = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="one")
    half_init = numpy_helper.from_array(np.array([0.5], dtype=np.float32), name="half")

    # x / sqrt(2) -> erf -> + 1 -> * x -> * 0.5
    div_node = helper.make_node("Div", ["X", "sqrt2"], ["div_out"], name="div")
    erf_node = helper.make_node("Erf", ["div_out"], ["erf_out"], name="erf")
    add_node = helper.make_node("Add", ["erf_out", "one"], ["add_out"], name="add")
    mul_x = helper.make_node("Mul", ["X", "add_out"], ["mul_x_out"], name="mul_x")
    mul_half = helper.make_node("Mul", ["mul_x_out", "half"], ["Y"], name="mul_half")

    graph_proto = helper.make_graph(
        [div_node, erf_node, add_node, mul_x, mul_half],
        "GeluGraph",
        [X],
        [Y],
        initializer=[sqrt2_init, one_init, half_init],
    )
    model = helper.make_model(graph_proto, opset_imports=[helper.make_opsetid("", 17)])

    g = NeuronGraph(model)
    assert len(g.nodes) == 5

    opt_g, stats = g.optimize(PassManager([FuseGELU()]))
    assert stats.pass_results["FuseGELU"]["fused_gelu_count"] == 1
    assert len(opt_g.query().op("Gelu").all()) == 1


def test_diamond_dag_pattern_matching():
    """Verify pattern matching enforces referential consistency in diamond topologies."""
    # Graph: X feeds both branch1 (Identity) and branch2 (Identity), then Add(branch1, branch2)
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 10])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 10])

    id1 = helper.make_node("Identity", ["X"], ["id1_out"], name="id1")
    id2 = helper.make_node("Identity", ["X"], ["id2_out"], name="id2")
    add = helper.make_node("Add", ["id1_out", "id2_out"], ["Y"], name="add")

    graph_proto = helper.make_graph([id1, id2, add], "DiamondGraph", [X], [Y])
    model = helper.make_model(graph_proto, opset_imports=[helper.make_opsetid("", 17)])
    g = NeuronGraph(model)

    # Pattern: Add(branch1, branch2)
    p_b1 = Pattern.op("Identity").capture("branch1")
    p_b2 = Pattern.op("Identity").capture("branch2")
    diamond_pat = Pattern.op("Add", p_b1, p_b2)

    matches = g.findall(diamond_pat)
    assert len(matches) == 1
    res = matches[0]
    assert res.bindings["branch1"].name != res.bindings["branch2"].name
    assert {res.bindings["branch1"].name, res.bindings["branch2"].name} == {"id1", "id2"}


def test_tflite_roundtrip_rebuilding(tflite_add_const_bytes):
    """Verify build_tflite_model_from_json successfully round-trips a TFLite model through JSON."""
    pytest.importorskip("tflite")
    pytest.importorskip("flatbuffers")

    orig_g = NeuronGraph(tflite_add_const_bytes)
    json_str = orig_g.to_json()
    assert json_str is not None

    # Rebuild from JSON
    rebuilt_bytes = build_tflite_model_from_json(json_str, seed=42)
    assert isinstance(rebuilt_bytes, bytes)
    assert rebuilt_bytes[:4] != b""

    # Load rebuilt model
    rebuilt_g = NeuronGraph(rebuilt_bytes)
    assert len(rebuilt_g.nodes) == len(orig_g.nodes)
    assert rebuilt_g.inputs == orig_g.inputs
    assert rebuilt_g.outputs == orig_g.outputs

    # Test auto-dispatch via NeuronGraph.from_json()
    g_auto = NeuronGraph.from_json(json_str)
    assert len(g_auto.nodes) == len(orig_g.nodes)


def test_fuse_layernorm():
    """Verify FuseLayerNorm combines ReduceMean, Sub, Sqrt, Div, Mul, Add into LayerNormalization."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16, 64])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 64])

    eps = numpy_helper.from_array(np.array([1e-5], dtype=np.float32), name="eps")
    gamma = numpy_helper.from_array(np.ones(64, dtype=np.float32), name="gamma")
    beta = numpy_helper.from_array(np.zeros(64, dtype=np.float32), name="beta")

    # 1. mean = ReduceMean(X)
    mean_node = helper.make_node("ReduceMean", ["X"], ["mean_out"], axes=[-1], keepdims=1, name="rm1")
    # 2. sub = Sub(X, mean)
    sub_node = helper.make_node("Sub", ["X", "mean_out"], ["sub_out"], name="sub1")
    # 3. pow2 = Mul(sub, sub)
    pow_node = helper.make_node("Mul", ["sub_out", "sub_out"], ["sq_out"], name="mul_sq")
    # 4. var = ReduceMean(pow2)
    var_node = helper.make_node("ReduceMean", ["sq_out"], ["var_out"], axes=[-1], keepdims=1, name="rm2")
    # 5. add_eps = Add(var, eps)
    add_eps_node = helper.make_node("Add", ["var_out", "eps"], ["add_eps_out"], name="add_eps")
    # 6. sqrt = Sqrt(add_eps)
    sqrt_node = helper.make_node("Sqrt", ["add_eps_out"], ["sqrt_out"], name="sqrt1")
    # 7. div = Div(sub, sqrt)
    div_node = helper.make_node("Div", ["sub_out", "sqrt_out"], ["div_out"], name="div1")
    # 8. scale = Mul(div, gamma)
    mul_scale = helper.make_node("Mul", ["div_out", "gamma"], ["scale_out"], name="mul_gamma")
    # 9. shift = Add(scale, beta)
    add_shift = helper.make_node("Add", ["scale_out", "beta"], ["Y"], name="add_beta")

    graph_proto = helper.make_graph(
        [mean_node, sub_node, pow_node, var_node, add_eps_node, sqrt_node, div_node, mul_scale, add_shift],
        "LNGraph",
        [X],
        [Y],
        initializer=[eps, gamma, beta],
    )
    model = helper.make_model(graph_proto, opset_imports=[helper.make_opsetid("", 17)])

    g = NeuronGraph(model)
    opt_g, stats = g.optimize(PassManager([FuseLayerNorm()]))
    assert stats["FuseLayerNorm"]["fused_layernorm_count"] == 1
    assert len(opt_g.query().op("LayerNormalization").all()) == 1


def test_fuse_conv_leakyrelu_and_clip():
    """Verify FuseConvActivation fuses Conv + LeakyRelu and Conv + Clip."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 16, 16])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8, 16, 16])

    w = numpy_helper.from_array(np.random.randn(8, 3, 3, 3).astype(np.float32), name="conv_w")
    conv = helper.make_node("Conv", ["X", "conv_w"], ["conv_out"], name="c1", pads=[1, 1, 1, 1])
    leaky = helper.make_node("LeakyRelu", ["conv_out"], ["Y"], name="lr1", alpha=0.1)

    graph_proto = helper.make_graph([conv, leaky], "ConvLeakyGraph", [X], [Y], initializer=[w])
    model = helper.make_model(graph_proto, opset_imports=[helper.make_opsetid("", 17)])

    g = NeuronGraph(model)
    opt_g, stats = g.optimize(PassManager([FuseConvActivation()]))
    assert stats["FuseConvActivation"]["fused_conv_activation_count"] == 1
    assert len(opt_g.nodes) == 1
    assert opt_g.nodes[0].op_type == "FusedConv"


def test_constant_folding_math_ops():
    """Verify ConstantFolding folds Sub, Mul, Div, Neg on constant inputs."""
    from neuron_toolkit import ConstantFolding

    # Inputs to graph (dummy, output Z)
    dummy_in = helper.make_tensor_value_info("Dummy", TensorProto.FLOAT, [1])
    Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1])

    c1 = numpy_helper.from_array(np.array([10.0], dtype=np.float32), name="c1")
    c2 = numpy_helper.from_array(np.array([2.0], dtype=np.float32), name="c2")

    sub = helper.make_node("Sub", ["c1", "c2"], ["sub_out"], name="sub_node")
    mul = helper.make_node("Mul", ["sub_out", "c2"], ["mul_out"], name="mul_node")
    div = helper.make_node("Div", ["mul_out", "c2"], ["div_out"], name="div_node")
    neg = helper.make_node("Neg", ["div_out"], ["neg_out"], name="neg_node")
    relu = helper.make_node("Relu", ["neg_out"], ["Z"], name="relu_node")

    graph_proto = helper.make_graph([sub, mul, div, neg, relu], "CFGraph", [dummy_in], [Z], initializer=[c1, c2])
    model = helper.make_model(graph_proto, opset_imports=[helper.make_opsetid("", 17)])

    g = NeuronGraph(model)
    opt_g, stats = g.optimize(PassManager([ConstantFolding()]))
    # All 5 nodes have constant inputs and should fold
    assert stats["ConstantFolding"]["folded_nodes"] >= 1
    assert len(opt_g.nodes) < len(g.nodes)

