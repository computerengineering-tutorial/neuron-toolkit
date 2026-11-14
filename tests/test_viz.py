"""Unit tests for visualization utilities."""

import onnx
from onnx import TensorProto, helper

from neuron_toolkit.backends.tflite.parser import TFLiteParser
from neuron_toolkit.graph import NeuronGraph


def create_simple_onnx() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])
    z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 2])
    n1 = helper.make_node("Relu", ["X"], ["Y"], name="relu_node")
    n2 = helper.make_node("Tanh", ["Y"], ["Z"], name="tanh_node")
    graph = helper.make_graph([n1, n2], "simple", [x], [z])
    return helper.make_model(graph, producer_name="test")


def test_to_mermaid_and_html(tmp_path):
    model = create_simple_onnx()
    graph = NeuronGraph(model)

    mermaid_str = graph.to_mermaid()
    assert "flowchart TD" in mermaid_str
    assert "Relu" in mermaid_str
    assert "Tanh" in mermaid_str
    assert "X" in mermaid_str
    assert "Z" in mermaid_str

    out_html = tmp_path / "graph.html"
    html_str = graph.to_html(str(out_html))
    assert "<!DOCTYPE html>" in html_str
    assert "mermaid.initialize" in html_str
    assert out_html.exists()
    assert out_html.read_text(encoding="utf-8") == html_str

    # Test html string generation without output_path
    html_str_no_path = graph.to_html()
    assert "<!DOCTYPE html>" in html_str_no_path


def test_to_mermaid_truncation():
    model = create_simple_onnx()
    graph = NeuronGraph(model)

    # Max nodes 1 forces truncation since model has 2 nodes
    mermaid_str = graph.to_mermaid(max_nodes=1)
    assert 'trunc["... (+1 more nodes)"]' in mermaid_str


def test_tflite_visualization(tflite_abs_bytes):
    graph = NeuronGraph(tflite_abs_bytes)
    assert isinstance(graph._backend, TFLiteParser)

    mermaid_str = graph.to_mermaid()
    assert "flowchart TD" in mermaid_str

    html_str = graph.to_html()
    assert "<!DOCTYPE html>" in html_str
    assert "NeuronGraph Topology" in html_str
