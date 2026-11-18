"""Tests for Declarative JSON Pattern Rules Engine."""

from __future__ import annotations

import tempfile

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from neuron_toolkit.rules.engine import JSONRuleEngine
from neuron_toolkit.rules.exporter import export_simplified_onnx


def test_json_rule_schema_loading():
    """Verify loading and validating JSON rule schema."""
    engine = JSONRuleEngine()
    rules = engine.load_rules_from_dir()
    assert len(rules) >= 3
    rule_ids = {r.rule_id for r in rules}
    assert "cast_initializer_folding" in rule_ids
    assert "erf_gelu_canonicalization" in rule_ids
    assert "continuous_math_collapse" in rule_ids


def test_cast_conv_weight_folding_synthetic():
    """Verify that a Cast node converting FP16 weight initializer to FP32 before Conv is folded."""
    # Build synthetic ONNX graph: X -> Conv(X, Cast(W_fp16)) -> Y
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16, 8, 8])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 8, 8])

    w_arr = np.random.randn(16, 16, 3, 3).astype(np.float16)
    w_init = numpy_helper.from_array(w_arr, name="W_fp16")

    cast_node = helper.make_node(
        "Cast",
        inputs=["W_fp16"],
        outputs=["W_fp32"],
        name="cast_w",
        to=int(TensorProto.FLOAT),
    )
    conv_node = helper.make_node(
        "Conv",
        inputs=["X", "W_fp32"],
        outputs=["Y"],
        name="conv_layer",
        pads=[1, 1, 1, 1],
        kernel_shape=[3, 3],
    )

    graph = helper.make_graph(
        nodes=[cast_node, conv_node],
        name="CastConvGraph",
        inputs=[X],
        outputs=[Y],
        initializer=[w_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

    assert len(model.graph.node) == 2

    engine = JSONRuleEngine()
    m_opt = engine.apply_cast_conv_weight_folding(model)

    # Cast node must be removed, Conv remains
    assert len(m_opt.graph.node) == 1
    assert m_opt.graph.node[0].op_type == "Conv"
    # Weight initializer must now be W_fp32 in float32
    inits = {i.name: numpy_helper.to_array(i) for i in m_opt.graph.initializer}
    assert "W_fp32" in inits
    assert inits["W_fp32"].dtype == np.float32
    assert np.allclose(inits["W_fp32"], w_arr.astype(np.float32))


def test_erf_gelu_fusion_synthetic():
    """Verify that 7-node standard GELU expansion is fused into Gelu."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 32, 10, 10])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 32, 10, 10])

    c_sqrt2 = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["c_sqrt2"],
        value=helper.make_tensor("c_sqrt2_t", TensorProto.FLOAT, [1], [1.41421356]),
    )
    div = helper.make_node("Div", inputs=["X", "c_sqrt2"], outputs=["div_out"])
    erf = helper.make_node("Erf", inputs=["div_out"], outputs=["erf_out"])
    c_one = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["c_one"],
        value=helper.make_tensor("c_one_t", TensorProto.FLOAT, [1], [1.0]),
    )
    add = helper.make_node("Add", inputs=["erf_out", "c_one"], outputs=["add_out"])
    mul1 = helper.make_node("Mul", inputs=["X", "add_out"], outputs=["mul1_out"])
    c_half = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["c_half"],
        value=helper.make_tensor("c_half_t", TensorProto.FLOAT, [1], [0.5]),
    )
    mul2 = helper.make_node("Mul", inputs=["mul1_out", "c_half"], outputs=["Y"])

    graph = helper.make_graph(
        nodes=[c_sqrt2, div, erf, c_one, add, mul1, c_half, mul2],
        name="GeluGraph",
        inputs=[X],
        outputs=[Y],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

    assert len(model.graph.node) == 8

    engine = JSONRuleEngine(mode="function_proto")
    m_opt = engine.apply_erf_gelu_fusion(model)

    assert len(m_opt.graph.node) == 1
    assert m_opt.graph.node[0].op_type == "Gelu"
    assert m_opt.graph.node[0].domain == "custom.pytorch"
    assert len(m_opt.functions) == 1
    assert m_opt.functions[0].name == "Gelu"


def test_continuous_math_collapse_synthetic():
    """Verify that continuous constant arithmetic operations are folded into Constant with expr metadata."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])

    c1 = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["c1"],
        value=helper.make_tensor("c1_t", TensorProto.FLOAT, [1], [2.0]),
    )
    c2 = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["c2"],
        value=helper.make_tensor("c2_t", TensorProto.FLOAT, [1], [3.0]),
    )
    c3 = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["c3"],
        value=helper.make_tensor("c3_t", TensorProto.FLOAT, [1], [2.0]),
    )

    # (2.0 / 3.0) + 2.0
    div = helper.make_node("Div", inputs=["c1", "c2"], outputs=["div_out"])
    add = helper.make_node("Add", inputs=["div_out", "c3"], outputs=["folded_const"])
    # Dynamic add with X
    final_add = helper.make_node("Add", inputs=["X", "folded_const"], outputs=["Y"])

    graph = helper.make_graph(
        nodes=[c1, c2, c3, div, add, final_add],
        name="MathGraph",
        inputs=[X],
        outputs=[Y],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

    assert len(model.graph.node) == 6

    engine = JSONRuleEngine()
    m_opt = engine.apply_continuous_math_collapse(model)

    # Div and Add on constants should be folded
    assert len(m_opt.graph.node) < 6
    ops = [n.op_type for n in m_opt.graph.node]
    assert "Add" in ops


def test_dual_mode_export():
    """Verify exporting in both function_proto and plain_custom_op modes."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    gelu = helper.make_node(
        "Gelu", inputs=["X"], outputs=["Y"], domain="custom.pytorch"
    )

    func_def = helper.make_function(
        domain="custom.pytorch",
        fname="Gelu",
        inputs=["x"],
        outputs=["y"],
        nodes=[helper.make_node("Relu", inputs=["x"], outputs=["y"])],
        opset_imports=[helper.make_opsetid("", 14)],
    )

    graph = helper.make_graph(nodes=[gelu], name="GeluM", inputs=[X], outputs=[Y])
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 14),
            helper.make_opsetid("custom.pytorch", 1),
        ],
        functions=[func_def],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        fp_path = f"{tmpdir}/func_proto.onnx"
        plain_path = f"{tmpdir}/plain.onnx"

        m_fp = export_simplified_onnx(model, fp_path, mode="function_proto")
        assert len(m_fp.functions) == 1

        m_plain = export_simplified_onnx(model, plain_path, mode="plain_custom_op")
        assert len(m_plain.functions) == 0
