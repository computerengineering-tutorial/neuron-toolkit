"""NLP and Transformer architecture blocks for tests and benchmarking."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def make_mha_block(d_model: int = 64, num_heads: int = 4) -> onnx.ModelProto:
    """Generate a realistic Multi-Head Attention (MHA) block in ONNX."""
    head_dim = d_model // num_heads

    nodes = []
    initializers = []

    def _add_init(name: str, arr: np.ndarray) -> str:
        initializers.append(numpy_helper.from_array(arr.astype(np.float32), name=name))
        return name

    # 1. Q, K, V Linear Projections
    w_q = _add_init("w_q", np.random.randn(d_model, d_model) * 0.02)
    w_k = _add_init("w_k", np.random.randn(d_model, d_model) * 0.02)
    w_v = _add_init("w_v", np.random.randn(d_model, d_model) * 0.02)

    nodes.append(helper.make_node("MatMul", ["X", w_q], ["q_out"], name="q_proj"))
    nodes.append(helper.make_node("MatMul", ["X", w_k], ["k_out"], name="k_proj"))
    nodes.append(helper.make_node("MatMul", ["X", w_v], ["v_out"], name="v_proj"))

    # 2. Reshape and Transpose for Multi-Head
    # Target shape: [batch, num_heads, seq_len, head_dim]
    # For synthetic block: [1, 16, num_heads, head_dim] -> transpose(0, 2, 1, 3)
    shape_q = _add_init("shape_q", np.array([1, 16, num_heads, head_dim], dtype=np.int64))
    nodes.append(helper.make_node("Reshape", ["q_out", shape_q], ["q_reshaped"], name="q_reshape"))
    nodes.append(helper.make_node("Transpose", ["q_reshaped"], ["q_trans"], perm=[0, 2, 1, 3], name="q_trans"))

    shape_k = _add_init("shape_k", np.array([1, 16, num_heads, head_dim], dtype=np.int64))
    nodes.append(helper.make_node("Reshape", ["k_out", shape_k], ["k_reshaped"], name="k_reshape"))
    nodes.append(helper.make_node("Transpose", ["k_reshaped"], ["k_trans"], perm=[0, 2, 3, 1], name="k_trans"))

    shape_v = _add_init("shape_v", np.array([1, 16, num_heads, head_dim], dtype=np.int64))
    nodes.append(helper.make_node("Reshape", ["v_out", shape_v], ["v_reshaped"], name="v_reshape"))
    nodes.append(helper.make_node("Transpose", ["v_reshaped"], ["v_trans"], perm=[0, 2, 1, 3], name="v_trans"))

    # 3. Scaled Dot-Product Attention: Q x K^T / sqrt(head_dim)
    nodes.append(helper.make_node("MatMul", ["q_trans", "k_trans"], ["scores_raw"], name="qk_matmul"))
    scale_val = _add_init("scale_const", np.array([1.0 / np.sqrt(head_dim)], dtype=np.float32))
    nodes.append(helper.make_node("Mul", ["scores_raw", scale_val], ["scores_scaled"], name="scale_mul"))
    nodes.append(helper.make_node("Softmax", ["scores_scaled"], ["attn_weights"], axis=-1, name="attn_softmax"))

    # 4. Context MatMul: Attn x V
    nodes.append(helper.make_node("MatMul", ["attn_weights", "v_trans"], ["context_trans"], name="attn_v_matmul"))
    nodes.append(helper.make_node("Transpose", ["context_trans"], ["context_reshaped"], perm=[0, 2, 1, 3], name="out_trans"))
    shape_out = _add_init("shape_out", np.array([1, 16, d_model], dtype=np.int64))
    nodes.append(helper.make_node("Reshape", ["context_reshaped", shape_out], ["context_flat"], name="out_reshape"))

    # 5. Output Projection Linear
    w_out = _add_init("w_out", np.random.randn(d_model, d_model) * 0.02)
    nodes.append(helper.make_node("MatMul", ["context_flat", w_out], ["Y"], name="out_proj"))

    inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16, d_model])]
    outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, d_model])]

    graph = helper.make_graph(nodes, "MultiHeadAttention", inputs, outputs, initializer=initializers)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def make_transformer_layer(d_model: int = 64, num_heads: int = 4) -> onnx.ModelProto:
    """Generate a complete Transformer encoder layer with LayerNorm, MHA, Residual, and MLP."""
    mha_model = make_mha_block(d_model=d_model, num_heads=num_heads)
    nodes = list(mha_model.graph.node)
    initializers = list(mha_model.graph.initializer)

    def _add_init(name: str, arr: np.ndarray) -> str:
        initializers.append(numpy_helper.from_array(arr.astype(np.float32), name=name))
        return name

    # Change mha output from Y to mha_out
    for n in nodes:
        if n.output[0] == "Y":
            n.output[0] = "mha_out"

    # Residual 1: Add(X, mha_out)
    nodes.append(helper.make_node("Add", ["X", "mha_out"], ["res1_out"], name="res1_add"))

    # LayerNorm 1
    ln1_scale = _add_init("ln1_scale", np.ones(d_model))
    ln1_bias = _add_init("ln1_bias", np.zeros(d_model))
    nodes.append(
        helper.make_node(
            "LayerNormalization",
            ["res1_out", ln1_scale, ln1_bias],
            ["ln1_out"],
            axis=-1,
            epsilon=1e-5,
            name="layernorm1",
        )
    )

    # MLP: FC1 -> Relu -> FC2
    d_ff = d_model * 4
    w_fc1 = _add_init("w_fc1", np.random.randn(d_model, d_ff) * 0.02)
    nodes.append(helper.make_node("MatMul", ["ln1_out", w_fc1], ["fc1_out"], name="mlp_fc1"))
    nodes.append(helper.make_node("Relu", ["fc1_out"], ["fc1_act"], name="mlp_relu"))

    w_fc2 = _add_init("w_fc2", np.random.randn(d_ff, d_model) * 0.02)
    nodes.append(helper.make_node("MatMul", ["fc1_act", w_fc2], ["fc2_out"], name="mlp_fc2"))

    # Residual 2: Add(ln1_out, fc2_out)
    nodes.append(helper.make_node("Add", ["ln1_out", "fc2_out"], ["res2_out"], name="res2_add"))

    # LayerNorm 2
    ln2_scale = _add_init("ln2_scale", np.ones(d_model))
    ln2_bias = _add_init("ln2_bias", np.zeros(d_model))
    nodes.append(
        helper.make_node(
            "LayerNormalization",
            ["res2_out", ln2_scale, ln2_bias],
            ["Y"],
            axis=-1,
            epsilon=1e-5,
            name="layernorm2",
        )
    )

    inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16, d_model])]
    outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, d_model])]

    graph = helper.make_graph(nodes, "TransformerLayer", inputs, outputs, initializer=initializers)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
