import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from neuron_toolkit.compiler.passes.canonicalization.activation_canonicalizer import (
    ActivationCanonicalizer,
)
from neuron_toolkit.compiler.passes.canonicalization.math_canonicalizer import (
    MathCanonicalizer,
)
from neuron_toolkit.compiler.passes.canonicalization.norm_canonicalizer import (
    NormCanonicalizer,
)
from neuron_toolkit.compiler.passes.canonicalization.shape_canonicalizer import (
    ShapeCanonicalizer,
)
from neuron_toolkit.compiler.passes.canonicalization.static_folder import StaticFolder
from neuron_toolkit.compiler.passes.synthesis.attention_synthesizer import (
    AttentionSynthesizer,
)
from neuron_toolkit.compiler.passes.synthesis.macro_block_synthesizer import (
    MacroBlockSynthesizer,
)
from neuron_toolkit.runtime.registry import OpRegistry


# ==============================================================================
# BUG 19: Constant Node Illegal folded_expr Attribute
# ==============================================================================
def test_vuln_19_continuous_math_illegal_folded_expr_attribute():
    """Bug 19: MathCanonicalizer.apply_continuous_math_collapse adds non-standard

    folded_expr attribute to standard ONNX Constant node, violating ONNX schema.
    """
    c1 = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["c1"],
        value=numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="c1"),
    )
    c2 = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["c2"],
        value=numpy_helper.from_array(np.array([2.0], dtype=np.float32), name="c2"),
    )
    add = helper.make_node("Add", inputs=["c1", "c2"], outputs=["sum_out"])
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])
    mul = helper.make_node("Mul", inputs=["X", "sum_out"], outputs=["Y"])

    graph = helper.make_graph([c1, c2, add, mul], "test_graph", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

    pass_obj = MathCanonicalizer()
    optimized = pass_obj.apply_continuous_math_collapse(model)

    # Must pass onnx.checker without unrecognized attribute error
    onnx.checker.check_model(optimized)


# ==============================================================================
# BUG 20: Linear Fusion Duplicate Initializer Name on Shared Weights
# ==============================================================================
def test_vuln_20_linear_fusion_shared_weight_duplicate_initializer():
    """Bug 20: MathCanonicalizer.apply_linear_fusion appends transposed initializers

    without checking for existing names, creating duplicate initializers on shared weights.
    """
    w_shared = np.random.randn(16, 32).astype(np.float32)
    w_init = numpy_helper.from_array(w_shared, name="W_shared")

    x1 = helper.make_tensor_value_info("X1", TensorProto.FLOAT, [1, 32])
    x2 = helper.make_tensor_value_info("X2", TensorProto.FLOAT, [1, 32])
    y1 = helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [1, 16])
    y2 = helper.make_tensor_value_info("Y2", TensorProto.FLOAT, [1, 16])

    gemm1 = helper.make_node(
        "Gemm", inputs=["X1", "W_shared"], outputs=["Y1"], transB=1
    )
    gemm2 = helper.make_node(
        "Gemm", inputs=["X2", "W_shared"], outputs=["Y2"], transB=1
    )

    graph = helper.make_graph(
        [gemm1, gemm2], "test_graph", [x1, x2], [y1, y2], initializer=[w_init]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

    pass_obj = MathCanonicalizer()
    optimized = pass_obj.apply_linear_fusion(model)

    init_names = [init.name for init in optimized.graph.initializer]
    assert len(init_names) == len(set(init_names)), (
        f"Duplicate initializers found: {init_names}"
    )
    onnx.checker.check_model(optimized)


# ==============================================================================
# BUG 21: GroupNormalization Opset Version Guard Violation (Opset <= 17)
# ==============================================================================
def test_vuln_21_groupnorm_opset_version_guard_violation():
    """Bug 21: GroupNormalization was introduced in ONNX Opset 18. Emitting standard

    GroupNormalization on Opset <= 17 triggers onnx.checker.ValidationError.
    """
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 8])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 8])

    sh_exp = numpy_helper.from_array(
        np.array([1, 2, 2, 8], dtype=np.int64), name="sh_exp"
    )
    sh_rest = numpy_helper.from_array(
        np.array([1, 4, 8], dtype=np.int64), name="sh_rest"
    )
    scale = numpy_helper.from_array(np.ones((4, 1), dtype=np.float32), name="scale")
    bias = numpy_helper.from_array(np.zeros((4, 1), dtype=np.float32), name="bias")

    r1 = helper.make_node("Reshape", inputs=["X", "sh_exp"], outputs=["r1_out"])
    inn = helper.make_node(
        "InstanceNormalization", inputs=["r1_out", "scale", "bias"], outputs=["inn_out"]
    )
    r2 = helper.make_node("Reshape", inputs=["inn_out", "sh_rest"], outputs=["r2_out"])
    mul = helper.make_node("Mul", inputs=["r2_out", "scale"], outputs=["mul_out"])
    add = helper.make_node("Add", inputs=["mul_out", "bias"], outputs=["Y"])

    graph = helper.make_graph(
        [r1, inn, r2, mul, add],
        "test_graph",
        [x],
        [y],
        initializer=[sh_exp, sh_rest, scale, bias],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    pass_obj = NormCanonicalizer()
    optimized = pass_obj.apply_groupnorm_fusion(model, mode="modern_onnx")

    onnx.checker.check_model(optimized)


# ==============================================================================
# BUG 22: RMSNormalization Opset Version Guard Violation (Opset <= 19)
# ==============================================================================
def test_vuln_22_rmsnorm_opset_version_guard_violation():
    """Bug 22: RMSNormalization was introduced in ONNX Opset 20. Emitting standard

    RMSNormalization on Opset <= 19 triggers onnx.checker.ValidationError.
    """
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 8])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 8])

    two = numpy_helper.from_array(np.array(2.0, dtype=np.float32), name="two")
    eps = numpy_helper.from_array(np.array(1e-5, dtype=np.float32), name="eps")
    gamma = numpy_helper.from_array(np.ones((8,), dtype=np.float32), name="gamma")

    pow_n = helper.make_node("Pow", inputs=["X", "two"], outputs=["p_out"], name="pow1")
    rm = helper.make_node(
        "ReduceMean",
        inputs=["p_out"],
        outputs=["rm_out"],
        axes=[-1],
        keepdims=1,
        name="rm1",
    )
    add = helper.make_node(
        "Add", inputs=["rm_out", "eps"], outputs=["add_out"], name="add1"
    )
    sqrt = helper.make_node(
        "Sqrt", inputs=["add_out"], outputs=["sqrt_out"], name="sqrt1"
    )
    recip = helper.make_node(
        "Reciprocal", inputs=["sqrt_out"], outputs=["recip_out"], name="recip1"
    )
    mul1 = helper.make_node(
        "Mul", inputs=["X", "recip_out"], outputs=["mul1_out"], name="mul1"
    )
    mul2 = helper.make_node(
        "Mul", inputs=["mul1_out", "gamma"], outputs=["Y"], name="mul2"
    )

    graph = helper.make_graph(
        [pow_n, rm, add, sqrt, recip, mul1, mul2],
        "test_graph",
        [x],
        [y],
        initializer=[two, eps, gamma],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    pass_obj = NormCanonicalizer()
    optimized = pass_obj.apply_rmsnorm_fusion(model, mode="modern_onnx")

    onnx.checker.check_model(optimized)


# ==============================================================================
# BUG 23: LayerNormalization Opset Version Guard Violation (Opset <= 16)
# ==============================================================================
def test_vuln_23_layernorm_opset_version_guard_violation():
    """Bug 23: LayerNormalization was introduced in ONNX Opset 17. Emitting standard

    LayerNormalization on Opset <= 16 triggers onnx.checker.ValidationError.
    """
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 8])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 8])

    two = numpy_helper.from_array(np.array(2.0, dtype=np.float32), name="two")
    eps = numpy_helper.from_array(np.array(1e-5, dtype=np.float32), name="eps")
    scale = numpy_helper.from_array(np.ones((8,), dtype=np.float32), name="scale")
    bias = numpy_helper.from_array(np.zeros((8,), dtype=np.float32), name="bias")

    rm_m = helper.make_node(
        "ReduceMean",
        inputs=["X"],
        outputs=["mean_out"],
        axes=[-1],
        keepdims=1,
        name="rm_m",
    )
    sub = helper.make_node(
        "Sub", inputs=["X", "mean_out"], outputs=["sub_out"], name="sub"
    )
    pow_n = helper.make_node(
        "Pow", inputs=["sub_out", "two"], outputs=["pow_out"], name="pow"
    )
    rm_v = helper.make_node(
        "ReduceMean",
        inputs=["pow_out"],
        outputs=["var_out"],
        axes=[-1],
        keepdims=1,
        name="rm_v",
    )
    add1 = helper.make_node(
        "Add", inputs=["var_out", "eps"], outputs=["var_eps"], name="add1"
    )
    sqrt = helper.make_node(
        "Sqrt", inputs=["var_eps"], outputs=["std_out"], name="sqrt"
    )
    div = helper.make_node(
        "Div", inputs=["sub_out", "std_out"], outputs=["norm_out"], name="div"
    )
    mul = helper.make_node(
        "Mul", inputs=["norm_out", "scale"], outputs=["scaled_out"], name="mul"
    )
    add2 = helper.make_node(
        "Add", inputs=["scaled_out", "bias"], outputs=["Y"], name="add2"
    )

    graph = helper.make_graph(
        [rm_m, sub, pow_n, rm_v, add1, sqrt, div, mul, add2],
        "test_graph",
        [x],
        [y],
        initializer=[two, eps, scale, bias],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

    pass_obj = NormCanonicalizer()
    optimized = pass_obj.apply_layernorm_fusion(model, mode="modern_onnx")

    onnx.checker.check_model(optimized)


# ==============================================================================
# BUG 24: CSE Drops Constant Nodes That Are Graph Outputs
# ==============================================================================
def test_vuln_24_cse_drops_constant_graph_outputs():
    """Bug 24: CSE Step 1 deduplicates Constants and deletes nodes even when

    the constant's output is an exposed graph.output.
    """
    c1 = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["c1_out"],
        value=numpy_helper.from_array(np.array([42.0], dtype=np.float32), name="c1"),
    )
    c2 = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["c2_out"],
        value=numpy_helper.from_array(np.array([42.0], dtype=np.float32), name="c2"),
    )

    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])
    c2_out = helper.make_tensor_value_info("c2_out", TensorProto.FLOAT, [1])

    mul = helper.make_node("Mul", inputs=["X", "c1_out"], outputs=["Y"])

    graph = helper.make_graph([c1, c2, mul], "test_graph", [x], [y, c2_out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

    pass_obj = ShapeCanonicalizer()
    optimized = pass_obj.apply_common_subexpression_elimination(model)

    onnx.checker.check_model(optimized)


# ==============================================================================
# BUG 25: Cast Folding Drops Initializers Consumed Inside Subgraphs
# ==============================================================================
def test_vuln_25_cast_folding_drops_control_flow_subgraph_initializers():
    """Bug 25: Cast-initializer folding prunes initializers not consumed in top-level

    graph.node inputs, wiping out initializers needed inside nested If/Loop subgraphs.
    """
    init_top = numpy_helper.from_array(
        np.array([1.0], dtype=np.float32), name="init_top"
    )
    init_sub = numpy_helper.from_array(
        np.array([10.0], dtype=np.float32), name="init_sub"
    )

    cast = helper.make_node(
        "Cast", inputs=["init_top"], outputs=["cast_out"], to=TensorProto.FLOAT
    )
    sub_node = helper.make_node(
        "Add", inputs=["sub_in", "init_sub"], outputs=["sub_out"]
    )
    sub_graph = helper.make_graph(
        [sub_node],
        "then_branch",
        [helper.make_tensor_value_info("sub_in", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("sub_out", TensorProto.FLOAT, [1])],
    )

    cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])

    if_node = helper.make_node(
        "If",
        inputs=["cond"],
        outputs=["Y"],
        then_branch=sub_graph,
        else_branch=sub_graph,
    )

    graph = helper.make_graph(
        [cast, if_node], "test_graph", [cond, x], [y], initializer=[init_top, init_sub]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

    pass_obj = ShapeCanonicalizer()
    optimized = pass_obj.apply_cast_initializer_folding(model)

    init_names = [i.name for i in optimized.graph.initializer]
    assert "init_sub" in init_names, (
        f"'init_sub' was erroneously pruned from graph initializers: {init_names}"
    )


# ==============================================================================
# BUG 26: Static Subgraph Folding Is Unimplemented Hollow No-Op
# ==============================================================================
def test_vuln_26_static_subgraph_folding_is_unimplemented_noop():
    """Bug 26: apply_static_subgraph_folding identifies foldable nodes but never

    evaluates them or replaces them with initializers, leaving static arithmetic uncollapsed.
    """
    arr_a = numpy_helper.from_array(
        np.array([2.0, 3.0], dtype=np.float32), name="static_a"
    )
    arr_b = numpy_helper.from_array(
        np.array([4.0, 5.0], dtype=np.float32), name="static_b"
    )

    add = helper.make_node("Add", inputs=["static_a", "static_b"], outputs=["static_c"])
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
    mul = helper.make_node("Mul", inputs=["X", "static_c"], outputs=["Y"])

    graph = helper.make_graph(
        [add, mul], "test_graph", [x], [y], initializer=[arr_a, arr_b]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

    pass_obj = StaticFolder()
    optimized = pass_obj.apply_static_subgraph_folding(model)

    remaining_ops = [n.op_type for n in optimized.graph.node]
    assert "Add" not in remaining_ops, (
        f"Static 'Add' node was not folded into an initializer: {remaining_ops}"
    )
    init_names = [i.name for i in optimized.graph.initializer]
    assert "static_c" in init_names, (
        f"Folded tensor 'static_c' not present in initializers: {init_names}"
    )


# ==============================================================================
# BUG 27: GatedAttention FunctionProto Formal Input Mismatch on Unbiased Linear
# ==============================================================================
def test_vuln_27_gated_attention_input_count_mismatch():
    """Bug 27: When output projection has no bias, apply_gated_attention_fusion pops

    trailing empty string creating 8 node inputs, while FunctionProto hardcodes 9 formal inputs.
    """
    model = onnx.ModelProto()
    model.ir_version = 8
    model.opset_import.append(helper.make_opsetid("", 14))
    model.opset_import.append(helper.make_opsetid("custom.pytorch", 1))

    # Synthetic block representing constituent nodes of GatedAttention
    norm = helper.make_node(
        "RMSNorm",
        inputs=["X", "gamma"],
        outputs=["x_norm"],
        name="norm1",
        domain="custom.pytorch",
    )
    to_qkv = helper.make_node(
        "Linear",
        inputs=["x_norm", "qkv_w"],
        outputs=["qkv"],
        name="to_qkv",
        domain="custom.pytorch",
    )
    to_gates = helper.make_node(
        "Linear",
        inputs=["x_norm", "gate_w"],
        outputs=["gates"],
        name="to_gates",
        domain="custom.pytorch",
    )
    split = helper.make_node(
        "SplitQKV",
        inputs=["qkv"],
        outputs=["q", "k", "v"],
        name="split",
        domain="custom.pytorch",
        num_heads=4,
    )
    rope_q = helper.make_node(
        "RotaryEmbedding",
        inputs=["q", "theta"],
        outputs=["q_rot"],
        name="rope_q",
        domain="custom.pytorch",
    )
    rope_k = helper.make_node(
        "RotaryEmbedding",
        inputs=["k", "theta"],
        outputs=["k_rot"],
        name="rope_k",
        domain="custom.pytorch",
    )
    sdpa = helper.make_node(
        "ScaledDotProductAttention",
        inputs=["q_rot", "k_rot", "v"],
        outputs=["attn_out"],
        name="sdpa",
        domain="custom.pytorch",
    )
    gate_unit = helper.make_node(
        "GatedUnit",
        inputs=["attn_out", "gates"],
        outputs=["gated_out"],
        name="gate_unit",
        domain="custom.pytorch",
        activation="silu",
    )
    merge = helper.make_node(
        "MergeHeads",
        inputs=["gated_out"],
        outputs=["merged"],
        name="merge",
        domain="custom.pytorch",
    )
    to_out = helper.make_node(
        "Linear",
        inputs=["merged", "out_w"],
        outputs=["Y"],
        name="to_out",
        domain="custom.pytorch",
    )  # NO out_b

    x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 16])
    y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 16])

    graph = helper.make_graph(
        [norm, to_qkv, to_gates, split, rope_q, rope_k, sdpa, gate_unit, merge, to_out],
        "test_graph",
        [x_info],
        [y_info],
    )
    model.graph.CopyFrom(graph)

    pass_obj = AttentionSynthesizer()
    optimized = pass_obj.apply_gated_attention_fusion(model, mode="function_proto")

    gated_node = [n for n in optimized.graph.node if "GatedAttention" in n.op_type][0]
    func_proto = [f for f in optimized.functions if f.name == gated_node.op_type][0]

    assert len(gated_node.input) == len(func_proto.input), (
        f"GatedAttention node input count ({len(gated_node.input)}) does not match "
        f"FunctionProto formal inputs ({len(func_proto.input)})."
    )


# ==============================================================================
# BUG 28: ScaledDotProductAttention Ignores 4th Input (attn_mask)
# ==============================================================================
def test_vuln_28_sdpa_ignores_causal_attention_mask():
    """Bug 28: OpRegistry._attention completely ignores 4th argument attn_mask,

    allowing future tokens to corrupt past token representations in causal masks.
    """
    registry = OpRegistry()
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)

    # Causal mask: upper triangle is -inf
    mask = torch.zeros(1, 1, 4, 4)
    mask[0, 0, 0, 1:] = -1e9  # Position 0 can only attend to position 0

    fn = registry.get("custom.pytorch", "ScaledDotProductAttention")
    out = fn([q, k, v, mask], {})

    # Calculate expected causal output for position 0
    expected_pos0 = v[:, :, :1, :]
    torch.testing.assert_close(out[:, :, :1, :], expected_pos0, atol=1e-3, rtol=1e-3)


# ==============================================================================
# BUG 29: RotaryEmbedding Crashes on Partial Head Dimension
# ==============================================================================
def test_vuln_29_rotary_embedding_crashes_on_partial_head_dim():
    """Bug 29: OpRegistry._rotary_embedding assumes cos.shape[-1] == x.shape[-1],

    crashing on partial rotary embeddings (e.g. ChatGLM, GPT-J with 50% rotary dim).
    """
    registry = OpRegistry()
    x = torch.randn(1, 2, 4, 64)  # 64 head dimension
    cos = torch.randn(1, 1, 4, 32)  # 32 partial rotary dimension
    sin = torch.randn(1, 1, 4, 32)

    fn = registry.get("custom.pytorch", "RotaryEmbedding")
    out = fn([x, cos, sin], {})
    assert out.shape == x.shape, (
        f"Rotary output shape mismatch: {out.shape} vs {x.shape}"
    )
    # Verify untouched channels are preserved
    torch.testing.assert_close(out[..., 32:], x[..., 32:])


# ==============================================================================
# BUG 30: GatedUnit Blindly Transposes Already Head-Aligned 3D Gate
# ==============================================================================
def test_vuln_30_gated_unit_crashes_on_head_aligned_3d_gate():
    """Bug 30: OpRegistry._gated_unit blindly transposes any 3D gate tensor with

    gate.transpose(1, 2), crashing when gate is already head-aligned (B, H, S).
    """
    registry = OpRegistry()
    x = torch.randn(2, 4, 8, 16)  # (B, H, S, D)
    gate = torch.randn(2, 4, 8)  # (B, H, S) - Already head-aligned!

    fn = registry.get("custom.pytorch", "GatedUnit")
    out = fn([x, gate], {"activation": "silu"})
    assert out.shape == x.shape, f"GatedUnit shape mismatch: {out.shape} vs {x.shape}"


# ==============================================================================
# BUG 31: DeepMLP_Norm_Act Crashes on Unbiased Norm
# ==============================================================================
def test_vuln_31_deep_mlp_norm_act_crashes_on_unbiased_norm():
    """Bug 31: OpRegistry._deep_mlp_norm_act treats inputs[2] as beta bias,

    passing weight matrix w as beta when unbiased RMSNorm or LayerNorm is used.
    """
    registry = OpRegistry()
    x = torch.randn(2, 16, 64)
    gamma = torch.randn(64)
    w = torch.randn(64, 128)

    fn = registry.get("custom.pytorch", "DeepMLP_Norm_Act")
    # 3 inputs: [x, gamma, w]
    out = fn([x, gamma, w], {"has_norm": 1, "activation": "GELU"})
    assert out.shape == (2, 16, 128), f"DeepMLP output shape mismatch: {out.shape}"


# ==============================================================================
# BUG 32: SpatialGatingUnit Crashes on Norm Gamma
# ==============================================================================
def test_vuln_32_spatial_gating_unit_crashes_on_norm_gamma():
    """Bug 32: OpRegistry._spatial_gating_unit assumes inputs[1] is w, but SGU fusion

    passes norm weight gamma as inputs[1], causing matmul dimension mismatch crash.
    """
    registry = OpRegistry()
    x = torch.randn(2, 8, 64)
    gamma = torch.randn(32)  # Norm scale for chunk v (dim 32)
    w = torch.randn(32, 32)  # Linear weight

    fn = registry.get("custom.pytorch", "SpatialGatingUnit")
    out = fn([x, gamma, w], {})
    assert out.shape == (2, 8, 32), f"SGU output shape mismatch: {out.shape}"


# ==============================================================================
# BUG 33: SqueezeExcitation Crashes on 3D (B, S, C) Inputs
# ==============================================================================
def test_vuln_33_squeeze_excitation_crashes_on_3d_input():
    """Bug 33: OpRegistry._squeeze_excitation pools dim=(-2, -1), averaging across channels

    on 3D tensors (B, S, C) instead of sequence length dim=-2, crashing matmul.
    """
    registry = OpRegistry()
    x = torch.randn(2, 16, 64)
    w1 = torch.randn(64, 16)
    b1 = torch.randn(16)
    w2 = torch.randn(16, 64)
    b2 = torch.randn(64)

    fn = registry.get("custom.pytorch", "SqueezeExcitation")
    out = fn([x, w1, b1, w2, b2], {})
    assert out.shape == x.shape, f"SE output shape mismatch: {out.shape} vs {x.shape}"


# ==============================================================================
# BUG 34: MultiHeadAttention Composite Ignores Attention Computation
# ==============================================================================
def test_vuln_34_mha_composite_ignores_multihead_attention():
    """Bug 34: OpRegistry._mha_composite completely bypasses SplitQKV, SDPA, and

    MergeHeads, computing only x @ out_w linear projection.
    """
    registry = OpRegistry()
    # x is QKV projection tensor of shape (1, 4, 3 * 2 * 8) = (B, S, 3 * num_heads * D)
    num_heads = 2
    head_dim = 8
    qkv = torch.randn(1, 4, 3 * num_heads * head_dim)
    out_w = torch.randn(num_heads * head_dim, num_heads * head_dim)

    fn = registry.get("custom.pytorch", "MultiHeadAttention")
    out = fn([qkv, out_w], {"num_heads": num_heads})

    # Linear projection alone: qkv @ out_w would crash (wrong shape) or produce different values
    # Must run full QKV attention pipeline
    assert out.shape == (1, 4, num_heads * head_dim)


# ==============================================================================
# BUG 35: Activation Canonicalizer Emits custom.pytorch Ops Without FunctionProtos
# ==============================================================================
def test_vuln_35_activation_canonicalizer_missing_function_protos():
    """Bug 35: ActivationCanonicalizer creates SiLU, HardSwish, Relu6, and Mish in

    custom.pytorch without attaching corresponding FunctionProto definitions.
    """
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])

    sig = helper.make_node("Sigmoid", inputs=["X"], outputs=["s_out"])
    mul = helper.make_node("Mul", inputs=["X", "s_out"], outputs=["Y"])

    graph = helper.make_graph([sig, mul], "test_graph", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

    pass_obj = ActivationCanonicalizer()
    optimized = pass_obj.apply_silu_fusion(model, mode="function_proto")

    func_names = [f.name for f in optimized.functions]
    assert "SiLU" in func_names, (
        f"SiLU FunctionProto missing from model.functions: {func_names}"
    )


# ==============================================================================
# BUG 36: SkipNorm Fusion Emits custom.pytorch Ops Without FunctionProtos
# ==============================================================================
def test_vuln_36_skip_norm_fusion_missing_function_protos():
    """Bug 36: NormCanonicalizer.apply_skip_norm_fusion creates SkipSimplifiedLayerNormalization

    and SkipLayerNormalization without attaching matching FunctionProto definitions.
    """
    x1 = helper.make_tensor_value_info("X1", TensorProto.FLOAT, [1, 4, 8])
    x2 = helper.make_tensor_value_info("X2", TensorProto.FLOAT, [1, 4, 8])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 8])
    gamma = numpy_helper.from_array(np.ones((8,), dtype=np.float32), name="gamma")

    add = helper.make_node("Add", inputs=["X1", "X2"], outputs=["add_out"])
    norm = helper.make_node(
        "RMSNorm", inputs=["add_out", "gamma"], outputs=["Y"], domain="custom.pytorch"
    )

    graph = helper.make_graph(
        [add, norm], "test_graph", [x1, x2], [y], initializer=[gamma]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

    pass_obj = NormCanonicalizer()
    optimized = pass_obj.apply_skip_norm_fusion(model, mode="function_proto")

    func_names = [f.name for f in optimized.functions]
    assert "SkipSimplifiedLayerNormalization" in func_names, (
        f"SkipSimplifiedLayerNormalization missing from functions: {func_names}"
    )


# ==============================================================================
# BUG 37: GroupNorm Duplicate Initializer Name on Shared Scale/Bias
# ==============================================================================
def test_vuln_37_groupnorm_shared_weight_duplicate_initializer():
    """Bug 37: NormCanonicalizer.apply_groupnorm_fusion flattens scale/bias into

    scale_1d without checking if initializer name exists, causing duplicate initializers.
    """
    x1 = helper.make_tensor_value_info("X1", TensorProto.FLOAT, [1, 4, 8])
    x2 = helper.make_tensor_value_info("X2", TensorProto.FLOAT, [1, 4, 8])
    y1 = helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [1, 4, 8])
    y2 = helper.make_tensor_value_info("Y2", TensorProto.FLOAT, [1, 4, 8])

    sh_exp = numpy_helper.from_array(
        np.array([1, 2, 2, 8], dtype=np.int64), name="sh_exp"
    )
    sh_rest = numpy_helper.from_array(
        np.array([1, 4, 8], dtype=np.int64), name="sh_rest"
    )
    scale = numpy_helper.from_array(
        np.ones((4, 1), dtype=np.float32), name="scale_shared"
    )
    bias = numpy_helper.from_array(
        np.zeros((4, 1), dtype=np.float32), name="bias_shared"
    )

    # Instance 1
    r1_1 = helper.make_node("Reshape", inputs=["X1", "sh_exp"], outputs=["r1_1_out"])
    inn1 = helper.make_node(
        "InstanceNormalization",
        inputs=["r1_1_out", "scale_shared", "bias_shared"],
        outputs=["inn1_out"],
    )
    r2_1 = helper.make_node(
        "Reshape", inputs=["inn1_out", "sh_rest"], outputs=["r2_1_out"]
    )
    mul1 = helper.make_node(
        "Mul", inputs=["r2_1_out", "scale_shared"], outputs=["mul1_out"]
    )
    add1 = helper.make_node("Add", inputs=["mul1_out", "bias_shared"], outputs=["Y1"])

    # Instance 2 (Sharing scale_shared and bias_shared)
    r1_2 = helper.make_node("Reshape", inputs=["X2", "sh_exp"], outputs=["r1_2_out"])
    inn2 = helper.make_node(
        "InstanceNormalization",
        inputs=["r1_2_out", "scale_shared", "bias_shared"],
        outputs=["inn2_out"],
    )
    r2_2 = helper.make_node(
        "Reshape", inputs=["inn2_out", "sh_rest"], outputs=["r2_2_out"]
    )
    mul2 = helper.make_node(
        "Mul", inputs=["r2_2_out", "scale_shared"], outputs=["mul2_out"]
    )
    add2 = helper.make_node("Add", inputs=["mul2_out", "bias_shared"], outputs=["Y2"])

    graph = helper.make_graph(
        [r1_1, inn1, r2_1, mul1, add1, r1_2, inn2, r2_2, mul2, add2],
        "test_graph",
        [x1, x2],
        [y1, y2],
        initializer=[sh_exp, sh_rest, scale, bias],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

    pass_obj = NormCanonicalizer()
    optimized = pass_obj.apply_groupnorm_fusion(model)

    init_names = [i.name for i in optimized.graph.initializer]
    assert len(init_names) == len(set(init_names)), (
        f"Duplicate initializers: {init_names}"
    )
    onnx.checker.check_model(optimized)


# ==============================================================================
# BUG 38: SGU FunctionProto Signature Parameter Mismatch
# ==============================================================================
def test_vuln_38_sgu_function_proto_param_mismatch():
    """Bug 38: MacroBlockSynthesizer._attach_sgu_function_def hardcodes 2 inputs (x, w)

    while fused SpatialGatingUnit node has inputs [x, gamma, w] or [x, gamma, beta, w, b].
    """
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 8, 64])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 8, 32])
    gamma = numpy_helper.from_array(np.ones((32,), dtype=np.float32), name="sgu_gamma")
    w = numpy_helper.from_array(
        np.random.randn(32, 32).astype(np.float32), name="sgu_w"
    )

    split = helper.make_node(
        "Split", inputs=["X"], outputs=["u", "v"], axis=-1, name="split1"
    )
    norm = helper.make_node(
        "RMSNorm",
        inputs=["v", "sgu_gamma"],
        outputs=["v_norm"],
        name="norm1",
        domain="custom.pytorch",
    )
    lin = helper.make_node(
        "Linear",
        inputs=["v_norm", "sgu_w"],
        outputs=["gate"],
        name="lin1",
        domain="custom.pytorch",
    )
    mul = helper.make_node("Mul", inputs=["u", "gate"], outputs=["Y"], name="mul1")

    graph = helper.make_graph(
        [split, norm, lin, mul], "test_graph", [x], [y], initializer=[gamma, w]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

    pass_obj = MacroBlockSynthesizer()
    optimized = pass_obj.apply_spatial_gating_unit_fusion(model, mode="function_proto")

    sgu_node = [n for n in optimized.graph.node if n.op_type == "SpatialGatingUnit"][0]
    func_proto = [f for f in optimized.functions if f.name == "SpatialGatingUnit"][0]

    assert len(sgu_node.input) == len(func_proto.input), (
        f"SpatialGatingUnit node input count ({len(sgu_node.input)}) does not match "
        f"FunctionProto formal inputs ({len(func_proto.input)})."
    )
