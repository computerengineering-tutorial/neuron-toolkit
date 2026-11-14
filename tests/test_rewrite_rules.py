"""Unit tests for declarative RewriteRule transformation engine."""

import onnx
from onnx import TensorProto, helper

from neuron_toolkit.graph import NeuronGraph
from neuron_toolkit.pattern import Pattern, RewriteRule


def create_silu_candidate_model() -> onnx.ModelProto:
    """Create a graph with: In -> Relu -> X; X -> Sigmoid -> sig_out; (X, sig_out) -> Mul -> Y."""
    in_val = helper.make_tensor_value_info("In", TensorProto.FLOAT, [1, 10])
    y_val = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 10])

    relu_node = helper.make_node("Relu", ["In"], ["X"], name="relu0")
    sigmoid_node = helper.make_node("Sigmoid", ["X"], ["sig_out"], name="sig0")
    mul_node = helper.make_node("Mul", ["X", "sig_out"], ["Y"], name="mul0")

    graph = helper.make_graph(
        [relu_node, sigmoid_node, mul_node], "silu_graph", [in_val], [y_val]
    )
    return helper.make_model(graph, producer_name="test")


def test_declarative_rewrite_rule():
    model = create_silu_candidate_model()
    graph = NeuronGraph(model)
    assert len(graph.nodes) == 3

    # Define declarative rule: X * Sigmoid(X) -> Silu(X)
    x = Pattern.any().capture("x")
    silu_pat = x * Pattern.op("Sigmoid", x)

    rule = RewriteRule(
        pattern=silu_pat,
        replacement=lambda match, rewriter: rewriter.replace_from_result(
            match, "Silu", name="fused_silu"
        ),
        name="FuseSiLU",
    )

    transformed = graph.apply_rules([rule])
    assert len(transformed.nodes) == 1
    op_types = [n.op_type for n in transformed.nodes]
    assert "Silu" in op_types
    assert "Sigmoid" not in op_types
    assert "Mul" not in op_types
    assert "Relu" not in op_types


def test_rewrite_rule_string_replacement_and_single_rule():
    """Verify RewriteRule using a string operator name and passing a single rule object."""
    model = create_silu_candidate_model()
    graph = NeuronGraph(model)

    x = Pattern.any().capture("x")
    silu_pat = x * Pattern.op("Sigmoid", x)

    # String replacement
    rule = RewriteRule(pattern=silu_pat, replacement="Silu", name="StringReplaceSiLU")

    # Pass rule as a single instance rather than list
    transformed = graph.apply_rules(rule)
    assert len(transformed.nodes) == 1
    assert transformed.nodes[0].op_type == "Silu"


def test_rewrite_rule_invalid_replacement():
    """Verify TypeError when replacement is neither a callable nor a string."""
    import pytest

    x = Pattern.any()
    rule = RewriteRule(pattern=x, replacement=12345)  # type: ignore[arg-type]

    model = create_silu_candidate_model()
    graph = NeuronGraph(model)

    with pytest.raises(TypeError, match="Unsupported replacement type"):
        graph.apply_rules(rule)
