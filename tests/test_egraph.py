"""Tests for the Equality Saturation (E-Graphs) engine."""

from onnx import TensorProto, helper

from neuron_toolkit import EGraph, ENode, Extractor, NeuronGraph
from neuron_toolkit.egraph.core import UnionFind
from neuron_toolkit.egraph.cost import NodeCountCostModel
from neuron_toolkit.egraph.rules import (
    ConstantFoldRule,
    IdentityRule,
    SiluFusionRule,
    saturate,
)


def test_union_find_basic():
    uf = UnionFind()
    for i in range(5):
        uf.make_set(i)

    assert uf.find(0) == 0
    assert uf.find(1) == 1

    uf.union(0, 1)
    assert uf.find(0) == uf.find(1)

    uf.union(2, 3)
    uf.union(1, 3)
    assert uf.find(0) == uf.find(2)
    assert uf.find(4) != uf.find(0)


def test_egraph_hash_consing():
    eg = EGraph()
    x_id = eg.add(ENode(op="Var", attrs=(("name", "x"),)))
    y_id = eg.add(ENode(op="Var", attrs=(("name", "y"),)))

    add1 = eg.add(ENode(op="Add", children=(x_id, y_id)))
    add2 = eg.add(ENode(op="Add", children=(x_id, y_id)))

    # Must hash-cons to the exact same EClassId
    assert add1 == add2
    assert eg.total_nodes() == 3


def test_congruence_closure():
    """If a == b, then f(a) == f(b)."""
    eg = EGraph()
    a = eg.add(ENode(op="Var", attrs=(("name", "a"),)))
    b = eg.add(ENode(op="Var", attrs=(("name", "b"),)))

    f_a = eg.add(ENode(op="Relu", children=(a,)))
    f_b = eg.add(ENode(op="Relu", children=(b,)))

    assert eg.find(f_a) != eg.find(f_b)

    # Union a and b
    eg.union(a, b)
    # Run congruence closure
    eg.rebuild()

    # Now f(a) and f(b) must be in the same equivalence class!
    assert eg.find(f_a) == eg.find(f_b)


def test_algebraic_identity_saturation():
    """x + 0 -> x."""
    eg = EGraph()
    x = eg.add(ENode(op="Var", attrs=(("name", "x"),)))
    zero = eg.add(ENode(op="Const", attrs=(("value", 0.0),)))
    add_node = eg.add(ENode(op="Add", children=(x, zero)))

    stats = saturate(eg, rules=[IdentityRule("Add", 0.0)])
    assert stats["total_unions"] >= 1
    assert eg.find(add_node) == eg.find(x)

    # Extract optimal expression: cost of x (0.1) < cost of x + 0 (1.2)
    extractor = Extractor(eg, NodeCountCostModel())
    _cost, best = extractor.find_best(add_node)
    assert best.op == "Var"
    assert dict(best.attrs)["name"] == "x"


def test_constant_folding_saturation():
    """Const(2.0) + Const(3.0) -> Const(5.0)."""
    eg = EGraph()
    c2 = eg.add(ENode(op="Const", attrs=(("value", 2.0),)))
    c3 = eg.add(ENode(op="Const", attrs=(("value", 3.0),)))
    add = eg.add(ENode(op="Add", children=(c2, c3)))

    stats = saturate(eg, rules=[ConstantFoldRule()])
    assert stats["total_unions"] >= 1

    extractor = Extractor(eg, NodeCountCostModel())
    _cost, best = extractor.find_best(add)
    assert best.op == "Const"
    assert dict(best.attrs)["value"] == 5.0


def test_silu_fusion_saturation():
    """x * Sigmoid(x) -> Silu(x)."""
    eg = EGraph()
    x = eg.add(ENode(op="Var", attrs=(("name", "x"),)))
    sig = eg.add(ENode(op="Sigmoid", children=(x,)))
    mul = eg.add(ENode(op="Mul", children=(x, sig)))

    stats = saturate(eg, rules=[SiluFusionRule()])
    assert stats["total_unions"] >= 1

    # Extract: Silu(x) (cost ~ 1.1) vs Mul(x, Sigmoid(x)) (cost ~ 2.2)
    extractor = Extractor(eg, NodeCountCostModel())
    _cost, best = extractor.find_best(mul)
    assert best.op == "Silu"
    assert eg.find(best.children[0]) == eg.find(x)


def test_end_to_end_graph_equality_saturation():
    # Construct ONNX model: y = x + 0.0
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 10])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 10])
    zero = helper.make_tensor("zero", TensorProto.FLOAT, [1], [0.0])
    node = helper.make_node("Add", ["X", "zero"], ["Y"])
    graph_proto = helper.make_graph([node], "id_test", [x], [y], initializer=[zero])
    model = helper.make_model(
        graph_proto, producer_name="test", opset_imports=[helper.make_opsetid("", 17)]
    )

    ng = NeuronGraph(model)
    opt_ir, stats = ng.equality_saturation()

    assert stats["total_unions"] >= 1
    # Check that extracted IR simplified
    assert len(opt_ir.nodes) == 0  # identity Add eliminated!
