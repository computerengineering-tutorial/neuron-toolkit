"""Property-based testing with Hypothesis verifying graph invariants."""

from hypothesis import given, settings, strategies as st
from onnx import TensorProto, helper

from neuron_toolkit.graph import NeuronGraph


@settings(max_examples=30, deadline=None)
@given(
    num_nodes=st.integers(min_value=2, max_value=20),
)
def test_hypothesis_linear_chain_invariants(num_nodes: int):
    """Test reachability and descendant invariants on linear chains of varying length."""
    inputs = [helper.make_tensor_value_info("X_0", TensorProto.FLOAT, [1, 2])]
    outputs = [
        helper.make_tensor_value_info(f"X_{num_nodes}", TensorProto.FLOAT, [1, 2])
    ]

    nodes = []
    for i in range(num_nodes):
        nodes.append(
            helper.make_node("Relu", [f"X_{i}"], [f"X_{i + 1}"], name=f"relu_{i}")
        )

    graph_proto = helper.make_graph(nodes, f"hypo_chain_{num_nodes}", inputs, outputs)
    model = helper.make_model(graph_proto, producer_name="hypothesis")

    graph = NeuronGraph(model)
    q = graph.query()

    # Invariant 1: Total node count equals num_nodes
    assert len(graph.nodes) == num_nodes
    assert q.count() == num_nodes

    # Invariant 2: Descendants of node 0 must include all subsequent nodes
    node_0 = q.name("relu_0", exact=True)
    descendants = node_0.descendants()
    assert descendants.count() == num_nodes - 1

    # Invariant 3: Ancestors of the last node must include all preceding nodes
    last_node = q.name(f"relu_{num_nodes - 1}", exact=True)
    ancestors = last_node.ancestors()
    assert ancestors.count() == num_nodes - 1

    # Invariant 4: IR representation preserves topological ordering
    ir = graph.to_ir()
    assert ir.is_topologically_sorted() is True

    # Invariant 5: Reachability duality (B in descendants(A) <=> A in ancestors(B))
    node_0_descendants = {n.name for n in node_0.descendants().all()}
    last_name = f"relu_{num_nodes - 1}"
    assert last_name in node_0_descendants
    assert "relu_0" in {n.name for n in last_node.ancestors().all()}

    # Invariant 6: JSON serialization round-trip isomorphism
    json_str = graph.to_json()
    rebuilt = NeuronGraph.from_json(json_str)
    assert len(rebuilt.nodes) == len(graph.nodes)
    assert rebuilt.inputs == graph.inputs
    assert rebuilt.outputs == graph.outputs
    assert [n.op_type for n in rebuilt.nodes] == [n.op_type for n in graph.nodes]
