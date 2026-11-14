"""Tests for LLM and Transformer optimization passes."""

import numpy as np
import onnx
from onnx import TensorProto, helper

from neuron_toolkit import NeuronGraph
from neuron_toolkit.passes import (
    FuseRMSNorm,
    PassManager,
)


def _make_rmsnorm_model(
    use_pow: bool = True, has_gamma: bool = True
) -> onnx.ModelProto:
    """Create a synthetic RMSNorm decomposed ONNX model."""
    # x -> Pow(2) or Mul(x, x) -> ReduceMean(axis=-1) -> Add(eps) -> Sqrt -> Div(x, sqrt) -> Mul(gamma)
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16, 64])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 64])

    nodes = []
    initializers = []

    if use_pow:
        two_const = helper.make_tensor("two", TensorProto.FLOAT, [1], [2.0])
        initializers.append(two_const)
        nodes.append(helper.make_node("Pow", ["X", "two"], ["x_sq"]))
    else:
        nodes.append(helper.make_node("Mul", ["X", "X"], ["x_sq"]))

    # ReduceMean over last dimension
    mean_axes = helper.make_tensor("mean_axes", TensorProto.INT64, [1], [-1])
    initializers.append(mean_axes)
    nodes.append(
        helper.make_node("ReduceMean", ["x_sq", "mean_axes"], ["mean_sq"], keepdims=1)
    )

    # Add eps
    eps_const = helper.make_tensor("eps", TensorProto.FLOAT, [1], [1e-5])
    initializers.append(eps_const)
    nodes.append(helper.make_node("Add", ["mean_sq", "eps"], ["var_eps"]))

    # Sqrt
    nodes.append(helper.make_node("Sqrt", ["var_eps"], ["rms"]))

    # Div
    div_out = "norm_out" if has_gamma else "Y"
    nodes.append(helper.make_node("Div", ["X", "rms"], [div_out]))

    if has_gamma:
        gamma_const = helper.make_tensor(
            "gamma", TensorProto.FLOAT, [64], np.ones(64, dtype=np.float32).tolist()
        )
        initializers.append(gamma_const)
        nodes.append(helper.make_node("Mul", ["norm_out", "gamma"], ["Y"]))

    graph = helper.make_graph(
        nodes, "rmsnorm_graph", [x], [y], initializer=initializers
    )
    return helper.make_model(
        graph, producer_name="neuron_test", opset_imports=[helper.make_opsetid("", 17)]
    )


def _make_swiglu_model(with_projections: bool = True) -> onnx.ModelProto:
    """Create a synthetic SwiGLU ONNX model."""
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16, 32])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 64])

    nodes = []
    initializers = []

    if with_projections:
        w_gate = helper.make_tensor(
            "W_gate",
            TensorProto.FLOAT,
            [32, 64],
            np.random.randn(32, 64).astype(np.float32).flatten(),
        )
        w_up = helper.make_tensor(
            "W_up",
            TensorProto.FLOAT,
            [32, 64],
            np.random.randn(32, 64).astype(np.float32).flatten(),
        )
        initializers.extend([w_gate, w_up])
        nodes.append(helper.make_node("MatMul", ["X", "W_gate"], ["gate_proj"]))
        nodes.append(helper.make_node("MatMul", ["X", "W_up"], ["up_proj"]))
        nodes.append(helper.make_node("Silu", ["gate_proj"], ["gate_act"]))
        nodes.append(helper.make_node("Mul", ["gate_act", "up_proj"], ["Y"]))
    else:
        nodes.append(helper.make_node("Silu", ["X"], ["gate_act"]))
        up_val = helper.make_tensor_value_info("Up", TensorProto.FLOAT, [1, 16, 32])
        nodes.append(helper.make_node("Mul", ["gate_act", "Up"], ["Y"]))
        graph = helper.make_graph(nodes, "swiglu_graph", [x, up_val], [y])
        return helper.make_model(
            graph,
            producer_name="neuron_test",
            opset_imports=[helper.make_opsetid("", 17)],
        )

    graph = helper.make_graph(nodes, "swiglu_graph", [x], [y], initializer=initializers)
    return helper.make_model(
        graph, producer_name="neuron_test", opset_imports=[helper.make_opsetid("", 17)]
    )


def _make_attention_model() -> onnx.ModelProto:
    """Create a synthetic Scaled Dot-Product Attention ONNX model."""
    q = helper.make_tensor_value_info("Q", TensorProto.FLOAT, [1, 8, 16, 32])
    k = helper.make_tensor_value_info("K", TensorProto.FLOAT, [1, 8, 16, 32])
    v = helper.make_tensor_value_info("V", TensorProto.FLOAT, [1, 8, 16, 32])
    out = helper.make_tensor_value_info("Out", TensorProto.FLOAT, [1, 8, 16, 32])

    nodes = [
        helper.make_node("MatMul", ["Q", "K"], ["raw_scores"]),
        helper.make_node("Mul", ["raw_scores", "scale"], ["scaled_scores"]),
        helper.make_node("Softmax", ["scaled_scores"], ["attn_weights"], axis=-1),
        helper.make_node("MatMul", ["attn_weights", "V"], ["Out"]),
    ]
    scale_const = helper.make_tensor("scale", TensorProto.FLOAT, [1], [0.1767])
    graph = helper.make_graph(
        nodes, "attn_graph", [q, k, v], [out], initializer=[scale_const]
    )
    return helper.make_model(
        graph, producer_name="neuron_test", opset_imports=[helper.make_opsetid("", 17)]
    )


def _make_rope_model() -> onnx.ModelProto:
    """Create a synthetic Rotary Position Embedding ONNX model."""
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16, 64])
    cos = helper.make_tensor_value_info("Cos", TensorProto.FLOAT, [1, 16, 64])
    sin = helper.make_tensor_value_info("Sin", TensorProto.FLOAT, [1, 16, 64])
    out = helper.make_tensor_value_info("Out", TensorProto.FLOAT, [1, 16, 64])

    nodes = [
        helper.make_node("Mul", ["X", "Cos"], ["x_cos"]),
        helper.make_node("Neg", ["X"], ["neg_x"]),
        helper.make_node("Concat", ["neg_x", "X"], ["rot_x"], axis=-1),
        helper.make_node("Mul", ["rot_x", "Sin"], ["rot_sin"]),
        helper.make_node("Add", ["x_cos", "rot_sin"], ["Out"]),
    ]
    graph = helper.make_graph(nodes, "rope_graph", [x, cos, sin], [out])
    return helper.make_model(
        graph, producer_name="neuron_test", opset_imports=[helper.make_opsetid("", 17)]
    )


def test_fuse_rmsnorm_pow():
    model = _make_rmsnorm_model(use_pow=True, has_gamma=True)
    graph = NeuronGraph(model)
    initial_nodes = len(graph.nodes)
    assert initial_nodes == 6

    opt_graph = graph.passes.fuse_rmsnorm()
    assert len(opt_graph.nodes) == 1
    assert opt_graph.nodes[0].op_type == "RMSNorm"
    assert opt_graph.nodes[0].output == ["Y"]


def test_fuse_rmsnorm_mul():
    model = _make_rmsnorm_model(use_pow=False, has_gamma=False)
    graph = NeuronGraph(model)
    pass_obj = FuseRMSNorm()
    opt_graph, stats = pass_obj.run(graph)

    assert stats["fused_rmsnorm_count"] == 1
    assert len(opt_graph.nodes) == 1
    assert opt_graph.nodes[0].op_type == "RMSNorm"


def test_fuse_swiglu_with_projections():
    model = _make_swiglu_model(with_projections=True)
    graph = NeuronGraph(model)
    assert len(graph.nodes) == 4

    opt_graph = graph.passes.fuse_swiglu()
    assert len(opt_graph.nodes) == 1
    assert opt_graph.nodes[0].op_type == "FusedSwiGLU"
    assert opt_graph.nodes[0].output == ["Y"]


def test_fuse_attention_sdpa():
    model = _make_attention_model()
    graph = NeuronGraph(model)
    assert len(graph.nodes) == 4

    opt_graph = graph.passes.fuse_attention()
    assert len(opt_graph.nodes) == 1
    assert opt_graph.nodes[0].op_type == "ScaledDotProductAttention"
    assert opt_graph.nodes[0].output == ["Out"]


def test_fuse_rotary_embedding():
    model = _make_rope_model()
    graph = NeuronGraph(model)
    assert len(graph.nodes) == 5

    opt_graph = graph.passes.fuse_rope()
    assert len(opt_graph.nodes) == 1
    assert opt_graph.nodes[0].op_type == "RotaryEmbedding"
    assert opt_graph.nodes[0].output == ["Out"]


def test_pass_manager_with_transformer_passes():
    model = _make_rmsnorm_model(use_pow=True, has_gamma=True)
    graph = NeuronGraph(model)

    pm = PassManager([FuseRMSNorm()])
    _opt_graph, stats = pm.run(graph)
    assert stats.initial_nodes == 6
    assert stats.final_nodes == 1
    assert stats["FuseRMSNorm"]["fused_rmsnorm_count"] == 1
