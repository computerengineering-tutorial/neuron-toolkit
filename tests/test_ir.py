"""Unit tests for Unified Intermediate Representation (IR)."""

import numpy as np
import onnx
from onnx import TensorProto, helper

from neuron_toolkit.graph import NeuronGraph
from neuron_toolkit.ir import IRGraph, IRNode, IRTensor, ONNXAdapter, TFLiteAdapter


def create_sample_onnx() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    w = helper.make_tensor(
        "W", TensorProto.FLOAT, [4, 4], np.eye(4, dtype=np.float32).flatten()
    )

    matmul = helper.make_node("MatMul", ["X", "W"], ["mm_out"], name="matmul0")
    relu = helper.make_node("Relu", ["mm_out"], ["Y"], name="relu0")

    graph = helper.make_graph([matmul, relu], "ir_test_graph", [x], [y], [w])
    return helper.make_model(graph, producer_name="test")


def test_ir_graph_construction_and_serialization():
    ir = IRGraph(name="test_ir")
    tensor = IRTensor(name="T1", shape=(2, 3), dtype="float32", is_initializer=True)
    ir.add_tensor(tensor)

    node = IRNode(
        op_type="Conv",
        name="conv1",
        inputs=["in1", "T1"],
        outputs=["out1"],
        attributes={"kernel_shape": [3, 3]},
    )
    ir.add_node(node)

    assert tensor.rank == 2
    assert tensor.numel == 6
    assert node.has_attr("kernel_shape")
    assert node.get_attr("kernel_shape") == [3, 3]

    # Serialization roundtrip
    d = ir.to_dict()
    rebuilt = IRGraph.from_dict(d)
    assert rebuilt.name == "test_ir"
    assert len(rebuilt.nodes) == 1
    assert rebuilt.nodes[0].op_type == "Conv"
    assert "T1" in rebuilt.tensors


def test_onnx_adapter_import_to_ir():
    model = create_sample_onnx()
    adapter = ONNXAdapter()
    assert adapter.can_handle(model) is True

    ir = adapter.import_to_ir(model)
    assert len(ir.nodes) == 2
    assert len(ir.inputs) == 1
    assert len(ir.outputs) == 1
    assert "W" in ir.tensors
    assert ir.tensors["W"].is_initializer is True
    assert ir.is_topologically_sorted() is True

    # Producer and consumer lookups
    producers = ir.producers
    consumers = ir.consumers
    assert "mm_out" in producers
    assert producers["mm_out"].name == "matmul0"
    assert "mm_out" in consumers
    assert consumers["mm_out"][0].name == "relu0"


def test_neuron_graph_to_ir():
    model = create_sample_onnx()
    graph = NeuronGraph(model)

    ir = graph.to_ir()
    assert isinstance(ir, IRGraph)
    assert len(ir.nodes) == 2
    assert ir.nodes[0].op_type == "MatMul"
    assert ir.nodes[1].op_type == "Relu"


def test_tflite_adapter_import_to_ir(tflite_add_const_bytes):
    """Verify TFLiteAdapter imports a TFLite flatbuffer into IRGraph."""
    adapter = TFLiteAdapter()
    assert adapter.can_handle(tflite_add_const_bytes) is True
    assert adapter.can_handle(b"NOT_A_TFLITE") is False

    ir = adapter.import_to_ir(tflite_add_const_bytes)
    assert isinstance(ir, IRGraph)
    assert ir.name == "tflite_model"
    assert len(ir.nodes) == 1
    assert ir.nodes[0].op_type == "ADD"
    assert "const_w" in ir.tensors
    assert ir.tensors["const_w"].is_initializer is True
    assert ir.is_topologically_sorted() is True

    # Test via NeuronGraph.to_ir()
    g = NeuronGraph(tflite_add_const_bytes)
    ir_from_g = g.to_ir()
    assert isinstance(ir_from_g, IRGraph)
    assert len(ir_from_g.nodes) == 1
    assert ir_from_g.nodes[0].op_type == "ADD"


def test_ir_topological_sort_failure_and_quant_params():
    """Verify is_topologically_sorted returns False when an input is never produced."""
    ir = IRGraph(name="cycle_or_missing")
    ir.inputs = ["in_val"]

    # node1 requires "unproduced_val" which is neither input nor initializer
    node1 = IRNode(
        op_type="Relu", name="n1", inputs=["unproduced_val"], outputs=["out1"]
    )
    ir.add_node(node1)

    assert ir.is_topologically_sorted() is False

    # Quant params serialization
    tensor = IRTensor(
        name="QWeight",
        shape=(10, 10),
        dtype="uint8",
        is_initializer=True,
        quant_params={"scale": [0.05], "zero_point": [128]},
    )
    ir.add_tensor(tensor)
    d = ir.to_dict()
    rebuilt = IRGraph.from_dict(d)
    assert rebuilt.tensors["QWeight"].quant_params == {
        "scale": [0.05],
        "zero_point": [128],
    }
