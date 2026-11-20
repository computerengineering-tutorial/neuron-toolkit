"""Tests for Batch 2 (Silent Numerical Bugs P1: Math & Bias Bugs).

Audited & Remediated:
- Bug 3.1: RoPE Function computes rotate_half rotation instead of x * (cos + sin).
- Bug 3.2: ScaledDotProductAttention FunctionProto dynamically computes 1/sqrt(d_k) rather than hardcoding 0.125.
- Bug 3.3: Gemma-2 / Grok logits softcapping accurately parameterizes cap_val (e.g. 30.0, 50.0).
- Bug 3.4: HighwayBlock preserves both transform and carry biases.
- Bug 3.5: SqueezeExcitation preserves FC1 and FC2 biases, supporting both 5-input and 3-input forms.
- Bug 3.6: LayerNorm (with beta) is not silently corrupted or substituted with RMSNorm in SwiGLU.
"""

from pathlib import Path
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import pytest
import torch
import torch.nn.functional as F

from neuron_toolkit.compiler.passes.synthesis.attention_synthesizer import AttentionSynthesizer
from neuron_toolkit.compiler.passes.synthesis.feedforward_synthesizer import FeedForwardSynthesizer
from neuron_toolkit.compiler.passes.synthesis.macro_block_synthesizer import MacroBlockSynthesizer
from neuron_toolkit.runtime.runner import PyTorchRunner


# ==============================================================================
# Bug 3.1: RoPE Function computes rotate_half rotation
# ==============================================================================
def test_bug_3_1_rope_rotate_half_mathematical_fidelity():
    """Verify RoPE FunctionProto defines rotate_half: [x1*cos - x2*sin, x2*cos + x1*sin]."""
    x_val = np.array([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=np.float32)  # d=4: x1=[1,2], x2=[3,4]
    cos_val = np.array([[[[0.5, 0.5, 0.5, 0.5]]]], dtype=np.float32)
    sin_val = np.array([[[[0.8660254, 0.8660254, 0.8660254, 0.8660254]]]], dtype=np.float32)

    # True RoPE math:
    # rot(x) = [-x2, x1] = [-3, -4, 1, 2]
    # out = x * cos + rot(x) * sin
    # out[0] = 1.0 * 0.5 + (-3.0) * 0.8660254 = 0.5 - 2.5980762 = -2.0980762
    # out[1] = 2.0 * 0.5 + (-4.0) * 0.8660254 = 1.0 - 3.4641016 = -2.4641016
    # out[2] = 3.0 * 0.5 + 1.0 * 0.8660254    = 1.5 + 0.8660254  = 2.3660254
    # out[3] = 4.0 * 0.5 + 2.0 * 0.8660254    = 2.0 + 1.7320508  = 3.7320508
    expected_out = (x_val * cos_val) + (np.array([[[[-3.0, -4.0, 1.0, 2.0]]]], dtype=np.float32) * sin_val)

    # Broken old math would have been: x * (cos + sin)
    broken_out = x_val * (cos_val + sin_val)

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 1, 4])
    cos = helper.make_tensor_value_info("cos", TensorProto.FLOAT, [1, 1, 1, 4])
    sin = helper.make_tensor_value_info("sin", TensorProto.FLOAT, [1, 1, 1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 1, 4])

    rope_node = helper.make_node(
        "RotaryEmbedding",
        inputs=["x", "cos", "sin"],
        outputs=["y"],
        domain="custom.pytorch",
    )
    graph = helper.make_graph([rope_node], "test_rope", [x, cos, sin], [y])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("custom.pytorch", 1)],
    )

    attn_pass = AttentionSynthesizer()
    attn_pass._attach_rope_function_def(model, arity=3)

    # Check function structure
    func = next(f for f in model.functions if f.name == "RotaryEmbedding")
    op_types = [n.op_type for n in func.node]
    assert "Split" in op_types, "RoPE Function must contain Split for rotate_half"
    assert "Neg" in op_types, "RoPE Function must contain Neg for rotate_half"
    assert "Concat" in op_types, "RoPE Function must contain Concat for rotate_half"

    # Execute with PyTorchRunner
    runner = PyTorchRunner(model)
    res = runner.run({"x": x_val, "cos": cos_val, "sin": sin_val})
    actual_out = res["y"]

    np.testing.assert_allclose(actual_out, expected_out, rtol=1e-5, atol=1e-5)
    # Ensure it's not the broken x*(cos+sin)
    assert not np.allclose(actual_out, broken_out, atol=1e-2), "RoPE output matches broken unrotated addition!"


# ==============================================================================
# Bug 3.2: ScaledDotProductAttention FunctionProto dynamically computes 1/sqrt(d_k)
# ==============================================================================
@pytest.mark.parametrize("head_dim", [32, 64, 128])
def test_bug_3_2_sdpa_dynamic_head_dimension_scaling(head_dim: int):
    """Verify SDPA FunctionProto computes scale dynamically via Shape->Gather->Cast->Pow(-0.5)."""
    batch, heads, seq = 1, 2, 8
    q_val = np.random.randn(batch, heads, seq, head_dim).astype(np.float32)
    k_val = np.random.randn(batch, heads, seq, head_dim).astype(np.float32)
    v_val = np.random.randn(batch, heads, seq, head_dim).astype(np.float32)

    # Golden PyTorch reference
    q_t = torch.from_numpy(q_val)
    k_t = torch.from_numpy(k_val)
    v_t = torch.from_numpy(v_val)
    expected_out = F.scaled_dot_product_attention(q_t, k_t, v_t).numpy()

    q_info = helper.make_tensor_value_info("Q", TensorProto.FLOAT, [batch, heads, seq, head_dim])
    k_info = helper.make_tensor_value_info("K", TensorProto.FLOAT, [batch, heads, seq, head_dim])
    v_info = helper.make_tensor_value_info("V", TensorProto.FLOAT, [batch, heads, seq, head_dim])
    y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [batch, heads, seq, head_dim])

    sdpa_node = helper.make_node(
        "ScaledDotProductAttention",
        inputs=["Q", "K", "V"],
        outputs=["Y"],
        domain="custom.pytorch",
    )
    graph = helper.make_graph([sdpa_node], "test_sdpa", [q_info, k_info, v_info], [y_info])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("custom.pytorch", 1)],
    )

    attn_pass = AttentionSynthesizer()
    attn_pass._attach_sdpa_function_def(model, has_mask=False)

    func = next(f for f in model.functions if f.name == "ScaledDotProductAttention")
    func_ops = [n.op_type for n in func.node]
    assert "Shape" in func_ops, "SDPA function must query Shape dynamically"
    assert "Pow" in func_ops, "SDPA function must compute Pow(-0.5) dynamically"

    # Ensure no hardcoded 0.125 float constant in the function
    for n in func.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    arr = numpy_helper.to_array(a.t)
                    assert not (arr.size == 1 and np.isclose(arr.item(), 0.125) and head_dim != 64), (
                        "SDPA function contains hardcoded 0.125 constant!"
                    )

    # Validate execution
    runner = PyTorchRunner(model)
    res = runner.run({"Q": q_val, "K": k_val, "V": v_val})
    np.testing.assert_allclose(res["Y"], expected_out, rtol=1e-4, atol=1e-4)


# ==============================================================================
# Bug 3.3: Gemma-2 / Grok logits softcapping accurately parameterizes cap_val
# ==============================================================================
@pytest.mark.parametrize("target_cap", [30.0, 50.0, 15.0])
def test_bug_3_3_logits_softcap_parameterization(target_cap: float):
    """Verify Gemma-2 logits softcapping parameterizes the FunctionProto with target_cap."""
    scores_val = np.random.randn(1, 4, 16, 16).astype(np.float32)

    scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 4, 16, 16])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4, 16, 16])

    cap_init = numpy_helper.from_array(np.array([target_cap], dtype=np.float32), name="cap_init")
    div_node = helper.make_node("Div", inputs=["scores", "cap_init"], outputs=["div_out"])
    tanh_node = helper.make_node("Tanh", inputs=["div_out"], outputs=["tanh_out"])
    mul_node = helper.make_node("Mul", inputs=["tanh_out", "cap_init"], outputs=["out"])

    graph = helper.make_graph(
        [div_node, tanh_node, mul_node],
        "test_softcap",
        [scores],
        [out],
        [cap_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    attn_pass = AttentionSynthesizer()
    fused_model = attn_pass.apply_gemma2_softcapping_fusion(model, mode="function_proto")

    func = next(f for f in fused_model.functions if f.name == "LogitsSoftCap")
    cap_const = next(
        n for n in func.node if n.op_type == "Constant" and n.output[0] == "cap_val"
    )
    arr = numpy_helper.to_array(cap_const.attribute[0].t)
    assert np.isclose(float(arr.item()), target_cap), (
        f"Expected softcap FunctionProto to have cap={target_cap}, but got {arr.item()}"
    )

    runner = PyTorchRunner(fused_model)
    res = runner.run({"scores": scores_val})
    expected = target_cap * np.tanh(scores_val / target_cap)
    np.testing.assert_allclose(res["out"], expected, rtol=1e-5, atol=1e-5)


# ==============================================================================
# Bug 3.4: HighwayBlock preserves both transform and carry biases
# ==============================================================================
def test_bug_3_4_highway_block_preserves_biases():
    """Verify HighwayBlock preserves transform bias and carry bias."""
    dim = 16
    x_val = np.random.randn(2, 4, dim).astype(np.float32)
    wh_val = np.random.randn(dim, dim).astype(np.float32)
    bh_val = np.random.randn(dim).astype(np.float32)
    wt_val = np.random.randn(dim, dim).astype(np.float32)
    bt_val = np.random.randn(dim).astype(np.float32)

    # True Highway with biases:
    # H = relu(X @ W_h + b_h)
    # T = sigmoid(X @ W_t + b_t)
    # Y = H * T + X * (1 - T)
    h_ref = np.maximum(x_val @ wh_val + bh_val, 0)
    t_ref = 1.0 / (1.0 + np.exp(-(x_val @ wt_val + bt_val)))
    expected_y = h_ref * t_ref + x_val * (1.0 - t_ref)

    # Create ONNX graph with biases
    in_t = helper.make_tensor_value_info("in", TensorProto.FLOAT, [2, 4, dim])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4, dim])

    wh = numpy_helper.from_array(wh_val, name="wh")
    bh = numpy_helper.from_array(bh_val, name="bh")
    wt = numpy_helper.from_array(wt_val, name="wt")
    bt = numpy_helper.from_array(bt_val, name="bt")
    one = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="one")

    nodes = [
        helper.make_node("Identity", inputs=["in"], outputs=["X"]),
        helper.make_node("Linear", inputs=["X", "wh", "bh"], outputs=["h_raw"], domain="custom.pytorch"),
        helper.make_node("Relu", inputs=["h_raw"], outputs=["h_out"]),
        helper.make_node("Linear", inputs=["X", "wt", "bt"], outputs=["t_raw"], domain="custom.pytorch"),
        helper.make_node("Sigmoid", inputs=["t_raw"], outputs=["t_out"]),
        helper.make_node("Mul", inputs=["h_out", "t_out"], outputs=["term1"]),
        helper.make_node("Sub", inputs=["one", "t_out"], outputs=["one_minus_t"]),
        helper.make_node("Mul", inputs=["X", "one_minus_t"], outputs=["term2"]),
        helper.make_node("Add", inputs=["term1", "term2"], outputs=["Y"]),
    ]
    graph = helper.make_graph(nodes, "test_hw", [in_t], [y], initializer=[wh, bh, wt, bt, one])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("custom.pytorch", 1)])

    macro_pass = MacroBlockSynthesizer()
    fused = macro_pass.apply_highway_block_fusion(model, mode="function_proto")

    hw_nodes = [n for n in fused.graph.node if n.op_type == "HighwayBlock"]
    assert len(hw_nodes) == 1
    # Must preserve all 5 inputs: x, wh, bh, wt, bt
    assert len(hw_nodes[0].input) == 5, f"HighwayBlock dropped biases! Got inputs: {hw_nodes[0].input}"
    assert "bh" in hw_nodes[0].input
    assert "bt" in hw_nodes[0].input

    # Verify function definition has 5 inputs
    func = next(f for f in fused.functions if f.name == "HighwayBlock")
    assert len(func.input) == 5, f"HighwayBlock function expected 5 inputs, got {len(func.input)}"

    # Execute with PyTorchRunner
    runner = PyTorchRunner(fused)
    res = runner.run({"in": x_val})
    np.testing.assert_allclose(res["Y"], expected_y, rtol=1e-4, atol=1e-4)


# ==============================================================================
# Bug 3.5: SqueezeExcitation preserves FC1 and FC2 biases
# ==============================================================================
def test_bug_3_5_squeeze_excitation_preserves_biases():
    """Verify SqueezeExcitation preserves FC1 and FC2 biases."""
    batch, c, h, w = 2, 16, 8, 8
    reduced = 4
    x_val = np.random.randn(batch, c, h, w).astype(np.float32)
    w1_val = np.random.randn(reduced, c, 1, 1).astype(np.float32)
    b1_val = np.random.randn(reduced).astype(np.float32)
    w2_val = np.random.randn(c, reduced, 1, 1).astype(np.float32)
    b2_val = np.random.randn(c).astype(np.float32)

    # True SqueezeExcitation reference:
    # pooled = mean(x, (-2, -1), keepdim=True)
    # fc1 = relu(conv2d(pooled, w1, b1))
    # fc2 = sigmoid(conv2d(fc1, w2, b2))
    # y = x * fc2
    pooled_ref = x_val.mean(axis=(-2, -1), keepdims=True)
    fc1_ref = np.maximum(
        torch.nn.functional.conv2d(
            torch.from_numpy(pooled_ref), torch.from_numpy(w1_val), torch.from_numpy(b1_val)
        ).numpy(),
        0,
    )
    scale_ref = 1.0 / (
        1.0
        + np.exp(
            -torch.nn.functional.conv2d(
                torch.from_numpy(fc1_ref), torch.from_numpy(w2_val), torch.from_numpy(b2_val)
            ).numpy()
        )
    )
    expected_y = x_val * scale_ref

    in_t = helper.make_tensor_value_info("in", TensorProto.FLOAT, [batch, c, h, w])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [batch, c, h, w])

    w1 = numpy_helper.from_array(w1_val, name="w1")
    b1 = numpy_helper.from_array(b1_val, name="b1")
    w2 = numpy_helper.from_array(w2_val, name="w2")
    b2 = numpy_helper.from_array(b2_val, name="b2")

    nodes = [
        helper.make_node("Identity", inputs=["in"], outputs=["X"]),
        helper.make_node("GlobalAveragePool", inputs=["X"], outputs=["pooled"]),
        helper.make_node("Conv", inputs=["pooled", "w1", "b1"], outputs=["fc1_out"]),
        helper.make_node("Relu", inputs=["fc1_out"], outputs=["act_out"]),
        helper.make_node("Conv", inputs=["act_out", "w2", "b2"], outputs=["fc2_out"]),
        helper.make_node("Sigmoid", inputs=["fc2_out"], outputs=["scale"]),
        helper.make_node("Mul", inputs=["X", "scale"], outputs=["Y"]),
    ]
    graph = helper.make_graph(nodes, "test_se", [in_t], [y], initializer=[w1, b1, w2, b2])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    macro_pass = MacroBlockSynthesizer()
    fused = macro_pass.apply_squeeze_excitation_fusion(model, mode="function_proto")

    se_nodes = [n for n in fused.graph.node if n.op_type == "SqueezeExcitation"]
    assert len(se_nodes) == 1
    assert len(se_nodes[0].input) == 5, f"SqueezeExcitation dropped biases! Got inputs: {se_nodes[0].input}"
    assert "b1" in se_nodes[0].input
    assert "b2" in se_nodes[0].input

    func = next(f for f in fused.functions if f.name == "SqueezeExcitation")
    assert len(func.input) == 5

    runner = PyTorchRunner(fused)
    res = runner.run({"in": x_val})
    np.testing.assert_allclose(res["Y"], expected_y, rtol=1e-4, atol=1e-4)


# ==============================================================================
# Bug 3.6: LayerNorm (with beta) is not silently corrupted or substituted in SwiGLU
# ==============================================================================
def test_bug_3_6_layernorm_preserved_not_replaced_by_rmsnorm():
    """Verify LayerNorm (with beta) is NOT absorbed and converted into RMSNorm."""
    dim = 64
    x_val = np.random.randn(1, 16, dim).astype(np.float32)
    gamma_val = np.ones(dim, dtype=np.float32)
    beta_val = np.full(dim, 2.5, dtype=np.float32)  # Non-zero beta
    w_gate_val = np.random.randn(dim, 128).astype(np.float32)
    w_up_val = np.random.randn(dim, 128).astype(np.float32)
    w_down_val = np.random.randn(128, dim).astype(np.float32)

    # Reference computation:
    # 1. True LayerNorm (mean subtracted, scaled by gamma, shifted by beta)
    mean = x_val.mean(axis=-1, keepdims=True)
    var = x_val.var(axis=-1, keepdims=True)
    ln_ref = (x_val - mean) / np.sqrt(var + 1e-5) * gamma_val + beta_val
    # 2. SwiGLU FFN
    gate = ln_ref @ w_gate_val
    silu_gate = gate / (1.0 + np.exp(-gate))
    up = ln_ref @ w_up_val
    expected_y = (silu_gate * up) @ w_down_val

    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16, dim])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, dim])

    gamma = numpy_helper.from_array(gamma_val, name="gamma")
    beta = numpy_helper.from_array(beta_val, name="beta")
    w_gate = numpy_helper.from_array(w_gate_val, name="w_gate")
    w_up = numpy_helper.from_array(w_up_val, name="w_up")
    w_down = numpy_helper.from_array(w_down_val, name="w_down")

    nodes = [
        helper.make_node("LayerNorm", inputs=["X", "gamma", "beta"], outputs=["x_norm"], domain="custom.pytorch"),
        helper.make_node("Linear", inputs=["x_norm", "w_gate"], outputs=["h_gate"], domain="custom.pytorch"),
        helper.make_node("Linear", inputs=["x_norm", "w_up"], outputs=["h_up"], domain="custom.pytorch"),
        helper.make_node("SiLU", inputs=["h_gate"], outputs=["h_act"], domain="custom.pytorch"),
        helper.make_node("Mul", inputs=["h_act", "h_up"], outputs=["h_mul"]),
        helper.make_node("Linear", inputs=["h_mul", "w_down"], outputs=["Y"], domain="custom.pytorch"),
    ]
    graph = helper.make_graph(nodes, "test_ln_swiglu", [x], [y], initializer=[gamma, beta, w_gate, w_up, w_down])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("custom.pytorch", 1)])

    ffn_pass = FeedForwardSynthesizer()
    fused = ffn_pass.apply_feedforward_swiglu_fusion(model, mode="function_proto")

    # LayerNorm MUST remain in the graph and NOT be absorbed into FeedForwardSwiGLU as RMSNorm!
    ln_nodes = [n for n in fused.graph.node if n.op_type == "LayerNorm"]
    assert len(ln_nodes) == 1, "LayerNorm was deleted from the graph!"
    assert "beta" in ln_nodes[0].input, "LayerNorm lost its beta!"

    # FeedForwardSwiGLU should be fused without norm (4 inputs: [x_norm, w_gate, w_up, w_down])
    swiglu_nodes = [n for n in fused.graph.node if n.op_type == "FeedForwardSwiGLU"]
    assert len(swiglu_nodes) == 1
    assert len(swiglu_nodes[0].input) == 4
    assert swiglu_nodes[0].input[0] == "x_norm"

    runner = PyTorchRunner(fused)
    res = runner.run({"X": x_val})
    np.testing.assert_allclose(res["Y"], expected_y, rtol=1e-3, atol=1e-3)
