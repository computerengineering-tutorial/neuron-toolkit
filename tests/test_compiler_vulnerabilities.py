"""Exhaustive Test Suite of Active Compiler Vulnerabilities & Bugs.

Every test in this file asserts the CORRECT compiler invariant / contract.
Because these bugs are ACTIVE and UNRESOLVED in neuron-toolkit, EVERY TEST
FAILS when executed, proving the presence of each vulnerability in the compiler.
"""

from __future__ import annotations

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from neuron_toolkit.compiler import NeuronCompiler
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
from neuron_toolkit.compiler.passes.canonicalization.static_folder import (
    StaticFolder,
)
from neuron_toolkit.compiler.passes.synthesis.attention_synthesizer import (
    AttentionSynthesizer,
)
from neuron_toolkit.compiler.passes.synthesis.axial_synthesizer import (
    AxialSynthesizer,
)
from neuron_toolkit.compiler.passes.synthesis.feedforward_synthesizer import (
    FeedForwardSynthesizer,
)
from neuron_toolkit.compiler.passes.synthesis.macro_block_synthesizer import (
    MacroBlockSynthesizer,
)
from neuron_toolkit.runtime.registry import OpRegistry


# ==============================================================================
# VULNERABILITY 1: [FATAL CRASH] IndexError in NormCanonicalizer SkipNorm Fusion
# ==============================================================================
def test_vuln_1_layernorm_skip_fusion_missing_scale_crash():
    """Vulnerability 1: apply_skip_norm_fusion must safely handle LayerNorm without explicit scale.

    FAILS WITH: IndexError: list index out of range at norm_canonicalizer.py:549
    """
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16, 64])
    res = helper.make_tensor_value_info("Res", TensorProto.FLOAT, [1, 16, 64])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 64])

    nodes = [
        helper.make_node("Add", ["X", "Res"], ["sum_out"], name="add0"),
        helper.make_node("LayerNormalization", ["sum_out"], ["Y"], axis=-1, name="ln0"),
    ]
    graph = helper.make_graph(nodes, "TestGraph", [x, res], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    pass_obj = NormCanonicalizer()
    # Contract: Compiler must not crash when scale input is omitted
    fused_model = pass_obj.apply_skip_norm_fusion(model)
    assert fused_model is not None


# ==============================================================================
# VULNERABILITY 2: [ONNX SPEC VIOLATION] Unbiased Linear FunctionProto Missing Bias
# ==============================================================================
def test_vuln_2_unbiased_linear_function_missing_bias_spec_violation():
    """Vulnerability 2: Fused Linear node input count must match FunctionProto formal inputs.

    FAILS WITH: AssertionError: Linear node has 2 inputs, but FunctionProto requires 3 inputs
    """
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 32])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 64])

    w_arr = np.random.randn(32, 64).astype(np.float32)
    w_init = numpy_helper.from_array(w_arr, name="W")

    mm = helper.make_node("MatMul", ["X", "W"], ["Y"], name="mm0")
    graph = helper.make_graph([mm], "TestGraph", [x], [y], initializer=[w_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    pass_obj = MathCanonicalizer()
    fused_model = pass_obj.apply_linear_fusion(model, mode="function_proto")

    lin_node = next(n for n in fused_model.graph.node if n.op_type == "Linear")
    lin_func = next(f for f in fused_model.functions if f.name == "Linear")

    # Contract: Node input count must equal function definition parameter count
    assert len(lin_node.input) == len(lin_func.input), (
        f"ONNX SPEC VIOLATION: Linear node has {len(lin_node.input)} inputs, "
        f"but function definition expects {len(lin_func.input)} inputs!"
    )


# ==============================================================================
# VULNERABILITY 3: [GRAPH CORRUPTION] Positional Bias Detached in SoftmaxMatMul Fusion
# ==============================================================================
def test_vuln_3_positional_bias_detached_in_softmax_matmul():
    """Vulnerability 3: Compiled graph with SoftmaxMatMul must maintain valid topological order.

    FAILS WITH: onnx.checker.ValidationError: input 'scaled' is not output of any previous nodes
    """
    q = helper.make_tensor_value_info("Q", TensorProto.FLOAT, [1, 4, 16, 32])
    k = helper.make_tensor_value_info("K", TensorProto.FLOAT, [1, 4, 16, 32])
    v = helper.make_tensor_value_info("V", TensorProto.FLOAT, [1, 4, 16, 32])
    bias = helper.make_tensor_value_info("Bias", TensorProto.FLOAT, [1, 4, 16, 16])
    out = helper.make_tensor_value_info("Out", TensorProto.FLOAT, [1, 4, 16, 32])

    scale = helper.make_tensor("scale", TensorProto.FLOAT, [1], [0.125])
    nodes = [
        helper.make_node("MatMul", ["Q", "K"], ["raw_scores"]),
        helper.make_node("Mul", ["raw_scores", "scale"], ["scaled_scores"]),
        helper.make_node("Add", ["scaled_scores", "Bias"], ["scaled"]),
        helper.make_node("Softmax", ["scaled"], ["probs"], axis=-1),
        helper.make_node("MatMul", ["probs", "V"], ["Out"]),
    ]
    graph = helper.make_graph(
        nodes, "TestGraph", [q, k, v, bias], [out], initializer=[scale]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    compiler = NeuronCompiler()
    compiled = compiler.compile(model)

    # Contract: Compiled ONNX model must pass topological validity check
    onnx.checker.check_model(compiled)


# ==============================================================================
# VULNERABILITY 4: [SILENT CORRUPTION] Softmax Canonicalizer Hardcodes axis=-1 on Opset 13+
# ==============================================================================
def test_vuln_4_softmax_fusion_opset13_axis_ignored_fallback():
    """Vulnerability 4: Decomposed Softmax along axis 1 must preserve axis=1 on Opset 13+.

    FAILS WITH: AssertionError: Expected axis=1, but compiler silently defaulted to -1
    """
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 16])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 16])

    axes_tensor = helper.make_tensor("axes_axis_1", TensorProto.INT64, [1], [1])
    nodes = [
        helper.make_node("Exp", ["X"], ["exp_out"], name="exp0"),
        helper.make_node(
            "ReduceSum",
            ["exp_out", "axes_axis_1"],
            ["sum_out"],
            keepdims=1,
            name="sum0",
        ),
        helper.make_node("Div", ["exp_out", "sum_out"], ["Y"], name="div0"),
    ]
    graph = helper.make_graph(nodes, "TestGraph", [x], [y], initializer=[axes_tensor])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    pass_obj = ActivationCanonicalizer()
    fused_model = pass_obj.apply_softmax_fusion(model)

    sm_node = next(n for n in fused_model.graph.node if n.op_type == "Softmax")
    axis_attr = next(a.i for a in sm_node.attribute if a.name == "axis")

    # Contract: Softmax reduction axis MUST match original ReduceSum axis (1)
    assert axis_attr == 1, (
        f"SILENT MATH CORRUPTION: Expected Softmax axis=1, but compiler silently defaulted to {axis_attr}!"
    )


# ==============================================================================
# VULNERABILITY 5: [DATA LOSS] StaticFolder Unconditional ScatterND Folding
# ==============================================================================
def test_vuln_5_scatter_nd_unconditional_folding_erases_updates():
    """Vulnerability 5: ScatterND with dynamic updates must not be deleted and rewired to empty buffer.

    FAILS WITH: AssertionError: Dynamic ScatterND updates were deleted and rewired to empty buffer
    """
    cache_arr = np.zeros((4, 8), dtype=np.float32)
    cache_init = numpy_helper.from_array(cache_arr, name="kv_cache_buffer")

    indices_arr = np.array([[2]], dtype=np.int64)
    indices_init = numpy_helper.from_array(indices_arr, name="target_indices")

    updates = helper.make_tensor_value_info("Updates", TensorProto.FLOAT, [1, 8])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 8])

    scatter = helper.make_node(
        "ScatterND",
        ["kv_cache_buffer", "target_indices", "Updates"],
        ["scatter_out"],
        name="scatter0",
    )
    relu = helper.make_node("Relu", ["scatter_out"], ["Y"], name="relu0")

    graph = helper.make_graph(
        [scatter, relu],
        "TestGraph",
        [updates],
        [y],
        initializer=[cache_init, indices_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    pass_obj = StaticFolder()
    folded_model = pass_obj.apply_identity_scatter_nd_folding(model)

    # Contract: Dynamic ScatterND node must be preserved!
    scatter_nodes = [n for n in folded_model.graph.node if n.op_type == "ScatterND"]
    assert len(scatter_nodes) == 1, (
        "DATA LOSS: Dynamic ScatterND updates were wiped out and consumer rewired to empty cache!"
    )


# ==============================================================================
# VULNERABILITY 6: [MASS NODE DROP] Unnamed Node Drop Bug in Set Lookups
# ==============================================================================
def test_vuln_6_unnamed_node_drop_bug_in_rope_and_attention():
    """Vulnerability 6: Unnamed nodes must not trigger mass deletion of unrelated unnamed nodes.

    FAILS WITH: AssertionError: Expected 2 surviving Relu nodes, but found 0
    """
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16])

    scores = helper.make_tensor_value_info("Scores", TensorProto.FLOAT, [1, 4, 16, 16])
    v = helper.make_tensor_value_info("V", TensorProto.FLOAT, [1, 4, 16, 16])
    attn_out = helper.make_tensor_value_info(
        "AttnOut", TensorProto.FLOAT, [1, 4, 16, 16]
    )

    sm = helper.make_node("Softmax", ["Scores"], ["probs"], axis=-1, name="")
    mm = helper.make_node("MatMul", ["probs", "V"], ["AttnOut"], name="")

    r1 = helper.make_node("Relu", ["X"], ["r1_out"], name="")
    r2 = helper.make_node("Relu", ["r1_out"], ["Y"], name="")

    graph = helper.make_graph(
        [sm, mm, r1, r2], "TestGraph", [x, scores, v], [y, attn_out]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    pass_obj = AttentionSynthesizer()
    fused_model = pass_obj.apply_softmax_matmul_fusion(model, mode="function_proto")

    # Contract: Unrelated unnamed Relu nodes r1, r2 MUST survive
    surviving_relus = [n for n in fused_model.graph.node if n.op_type == "Relu"]
    assert len(surviving_relus) == 2, (
        f"MASS NODE DROP: Expected 2 surviving Relu nodes, but found {len(surviving_relus)}! "
        "Unnamed nodes were wiped out by empty string lookup."
    )


# ==============================================================================
# VULNERABILITY 7: [SILENT DIVERGENCE] SDPA FunctionProto Omits Scale Division
# ==============================================================================
def test_vuln_7_sdpa_function_proto_omits_scale_division():
    """Vulnerability 7: ScaledDotProductAttention FunctionProto must include scale division node.

    FAILS WITH: AssertionError: ScaledDotProductAttention FunctionProto computes unscaled QK^T
    """
    model = helper.make_model(
        helper.make_graph([], "EmptyGraph", [], []),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    pass_obj = AttentionSynthesizer()
    pass_obj._attach_sdpa_function_def(model)

    sdpa_func = next(
        f for f in model.functions if f.name == "ScaledDotProductAttention"
    )
    func_op_types = [n.op_type for n in sdpa_func.node]

    # Contract: FunctionProto must divide or multiply scores by scale factor (1/sqrt(d))
    has_scaling = "Div" in func_op_types or "Mul" in func_op_types
    assert has_scaling, (
        "SILENT NUMERICAL DIVERGENCE: ScaledDotProductAttention FunctionProto computes unscaled QK^T!"
    )


# ==============================================================================
# VULNERABILITY 8: [FUNCTIONPROTO SHADOWING] Tanh-Approximated GELU Lost in FunctionProto
# ==============================================================================
def test_vuln_8_tanh_gelu_function_proto_shadowed_by_erf():
    """Vulnerability 8: Gelu(approximate='tanh') must register a dedicated tanh FunctionProto.

    FAILS WITH: AssertionError: Gelu(approximate='tanh') has no tanh FunctionProto
    """
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16])

    nodes = [
        helper.make_node("Tanh", ["X"], ["tanh_out"], name="t0"),
        helper.make_node("Add", ["tanh_out", "c_one"], ["add_out"], name="a0"),
        helper.make_node("Mul", ["X", "add_out"], ["mul_x_out"], name="m0"),
        helper.make_node("Mul", ["mul_x_out", "c_half"], ["Y"], name="m1"),
    ]
    c_one = helper.make_tensor("c_one", TensorProto.FLOAT, [1], [1.0])
    c_half = helper.make_tensor("c_half", TensorProto.FLOAT, [1], [0.5])

    graph = helper.make_graph(nodes, "TestGraph", [x], [y], initializer=[c_one, c_half])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    pass_obj = ActivationCanonicalizer()
    fused_model = pass_obj.apply_tanh_gelu_fusion(model, mode="function_proto")

    # Contract: Dedicated FunctionProto implementing tanh must exist
    tanh_func = [
        f
        for f in fused_model.functions
        if "Gelu" in f.name and any(n.op_type == "Tanh" for n in f.node)
    ]
    assert len(tanh_func) >= 1, (
        "FUNCTIONPROTO SHADOWING: Gelu(approximate='tanh') has no tanh FunctionProto; "
        "shadowed by exact ERF Gelu!"
    )


# ==============================================================================
# VULNERABILITY 9: [NESTED SCOPE LEAK] DCE Prunes Control Flow Subgraph Producers
# ==============================================================================
def test_vuln_9_dce_prunes_control_flow_subgraph_producers():
    """Vulnerability 9: Dead code elimination must not delete producers consumed in If branches.

    FAILS WITH: AssertionError: DCE deleted producer consumed inside nested If branch
    """
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16])
    cond = helper.make_tensor_value_info("Cond", TensorProto.BOOL, [1])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16])

    producer = helper.make_node("Relu", ["X"], ["relu_out"], name="producer_relu")

    then_y = helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [1, 16])
    then_node = helper.make_node("Neg", ["relu_out"], ["then_out"], name="then_neg")
    then_graph = helper.make_graph([then_node], "then_branch", [], [then_y])

    else_y = helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [1, 16])
    else_node = helper.make_node("Identity", ["X"], ["else_out"], name="else_id")
    else_graph = helper.make_graph([else_node], "else_branch", [], [else_y])

    if_node = helper.make_node(
        "If",
        ["Cond"],
        ["Y"],
        name="if0",
        then_branch=then_graph,
        else_branch=else_graph,
    )

    graph = helper.make_graph([producer, if_node], "TestGraph", [x, cond], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    pass_obj = ShapeCanonicalizer()
    dce_model = pass_obj.apply_dead_code_elimination(model)

    # Contract: producer_relu MUST NOT be deleted because it is needed inside then_branch
    producer_survived = any(n.name == "producer_relu" for n in dce_model.graph.node)
    assert producer_survived, (
        "NESTED SCOPE LEAK: DCE deleted producer tensor consumed inside nested If branch!"
    )


# ==============================================================================
# VULNERABILITY 10: [CRASH] SplitQKV Runtime Registry Operator Crashes on 2D Tensors
# ==============================================================================
def test_vuln_10_split_qkv_rank_mismatch_crash():
    """Vulnerability 10: _split_qkv must support 2D flattened inputs without crashing.

    FAILS WITH: IndexError: tuple index out of range at registry.py:920
    """
    reg = OpRegistry()
    x_2d = torch.randn(2, 64 * 3)

    fn = reg.get("custom.pytorch", "SplitQKV")
    assert fn is not None

    # Contract: Must support 2D projection inputs without crashing
    out = fn([x_2d], {"num_heads": 4})
    assert len(out) == 3


# ==============================================================================
# VULNERABILITY 11: [SPEC VIOLATION] FeedForwardSwiGLU FunctionProto Parameter Mismatch
# ==============================================================================
def test_vuln_11_swiglu_biased_function_proto_param_mismatch():
    """Vulnerability 11: Biased SwiGLU node input count must match FunctionProto input count.

    FAILS WITH: AssertionError: SwiGLU node has 8 inputs, but function expects 5
    """
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16, 64])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 64])

    gamma = helper.make_tensor(
        "gamma", TensorProto.FLOAT, [64], np.ones(64, dtype=np.float32).tolist()
    )
    w_gate = helper.make_tensor(
        "w_gate",
        TensorProto.FLOAT,
        [64, 128],
        np.random.randn(64, 128).astype(np.float32).flatten(),
    )
    b_gate = helper.make_tensor(
        "b_gate", TensorProto.FLOAT, [128], np.zeros(128, dtype=np.float32).tolist()
    )
    w_up = helper.make_tensor(
        "w_up",
        TensorProto.FLOAT,
        [64, 128],
        np.random.randn(64, 128).astype(np.float32).flatten(),
    )
    w_down = helper.make_tensor(
        "w_down",
        TensorProto.FLOAT,
        [128, 64],
        np.random.randn(128, 64).astype(np.float32).flatten(),
    )

    nodes = [
        helper.make_node(
            "RMSNorm", ["X", "gamma"], ["x_norm"], domain="custom.pytorch"
        ),
        helper.make_node(
            "Linear",
            ["x_norm", "w_gate", "b_gate"],
            ["h_gate"],
            domain="custom.pytorch",
        ),
        helper.make_node(
            "Linear", ["x_norm", "w_up"], ["h_up"], domain="custom.pytorch"
        ),
        helper.make_node("SiLU", ["h_gate"], ["h_act"], domain="custom.pytorch"),
        helper.make_node("Mul", ["h_act", "h_up"], ["h_mul"]),
        helper.make_node("Linear", ["h_mul", "w_down"], ["Y"], domain="custom.pytorch"),
    ]
    graph = helper.make_graph(
        nodes, "TestGraph", [x], [y], initializer=[gamma, w_gate, b_gate, w_up, w_down]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    pass_obj = FeedForwardSynthesizer()
    fused_model = pass_obj.apply_feedforward_swiglu_fusion(model, mode="function_proto")

    swiglu_node = next(
        n for n in fused_model.graph.node if n.op_type == "FeedForwardSwiGLU"
    )
    func = next(f for f in fused_model.functions if f.name == "FeedForwardSwiGLU")

    # Contract: Node input count must equal function parameter count
    assert len(swiglu_node.input) == len(func.input), (
        f"SPEC VIOLATION: SwiGLU node has {len(swiglu_node.input)} inputs, "
        f"but function definition expects {len(func.input)} inputs!"
    )


# ==============================================================================
# VULNERABILITY 12: [SPEC VIOLATION] Unbind FunctionProto Hardcodes Exactly 3 Outputs
# ==============================================================================
def test_vuln_12_unbind_function_proto_hardcodes_3_outputs():
    """Vulnerability 12: Unbind FunctionProto must dynamically match 2 output splits.

    FAILS WITH: AssertionError: Unbind node has 2 outputs, but function expects 3
    """
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 16, 64])
    y0 = helper.make_tensor_value_info("Y0", TensorProto.FLOAT, [16, 64])
    y1 = helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [16, 64])

    split_lens = helper.make_tensor("split_lens", TensorProto.INT64, [2], [1, 1])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [0])

    nodes = [
        helper.make_node(
            "Split", ["X", "split_lens"], ["s0", "s1"], axis=0, name="sp0"
        ),
        helper.make_node("Squeeze", ["s0", "axes"], ["Y0"], name="sq0"),
        helper.make_node("Squeeze", ["s1", "axes"], ["Y1"], name="sq1"),
    ]
    graph = helper.make_graph(
        nodes, "TestGraph", [x], [y0, y1], initializer=[split_lens, axes]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    pass_obj = AxialSynthesizer()
    fused_model = pass_obj.apply_unbind_fusion(model, mode="function_proto")

    unbind_node = next(n for n in fused_model.graph.node if n.op_type == "Unbind")
    func = next(f for f in fused_model.functions if f.name == "Unbind")

    # Contract: Node output count (2) must match function definition output count
    assert len(unbind_node.output) == len(func.output), (
        f"OUTPUT COUNT CRASH: Unbind node has {len(unbind_node.output)} outputs, "
        f"but FunctionProto expects {len(func.output)} outputs!"
    )


# ==============================================================================
# VULNERABILITY 13: [SPEC VIOLATION] DeepMLP_Norm_Act FunctionProto Hardcodes 5 Inputs
# ==============================================================================
def test_vuln_13_deep_mlp_function_proto_hardcodes_5_inputs():
    """Vulnerability 13: DeepMLP FunctionProto must match input count on unbiased Linear/RMSNorm.

    FAILS WITH: AssertionError: DeepMLP node has 3 inputs, function expects 5
    """
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16, 64])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 64])

    gamma = helper.make_tensor(
        "gamma", TensorProto.FLOAT, [64], np.ones(64, dtype=np.float32).tolist()
    )
    w = helper.make_tensor(
        "W", TensorProto.FLOAT, [64, 64], np.eye(64, dtype=np.float32).flatten()
    )

    nodes = [
        helper.make_node(
            "RMSNorm", ["X", "gamma"], ["x_norm"], domain="custom.pytorch", name="norm0"
        ),
        helper.make_node(
            "Linear", ["x_norm", "W"], ["lin_out"], domain="custom.pytorch", name="lin0"
        ),
        helper.make_node("Relu", ["lin_out"], ["Y"], name="act0"),
    ]
    graph = helper.make_graph(nodes, "TestGraph", [x], [y], initializer=[gamma, w])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    pass_obj = MacroBlockSynthesizer()
    fused_model = pass_obj.apply_deep_mlp_norm_act_fusion(model, mode="function_proto")

    deep_mlp_node = next(n for n in fused_model.graph.node if "DeepMLP" in n.op_type)
    func = next(f for f in fused_model.functions if "DeepMLP" in f.name)

    # Contract: Node input count (3) must match function formal input count
    assert len(deep_mlp_node.input) == len(func.input), (
        f"SPEC VIOLATION: DeepMLP node has {len(deep_mlp_node.input)} inputs, "
        f"but FunctionProto expects {len(func.input)} inputs!"
    )


# ==============================================================================
# VULNERABILITY 14: [DATA LOSS & PATTERN MISS] SpatialGatingUnit SGU Short-Circuit
# ==============================================================================
def test_vuln_14_sgu_drops_norm_gamma_and_bias():
    """Vulnerability 14: SGU must properly match on standard (u, gate) order and preserve norm gamma.

    FAILS WITH: AssertionError: Pattern Matcher Bug: SpatialGatingUnit failed to match on (u, gate)
    """
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16, 64])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 32])

    split_lens = helper.make_tensor("split_lens", TensorProto.INT64, [2], [32, 32])
    gamma = helper.make_tensor(
        "gamma", TensorProto.FLOAT, [32], np.ones(32, dtype=np.float32).tolist()
    )
    w = helper.make_tensor(
        "W", TensorProto.FLOAT, [32, 32], np.eye(32, dtype=np.float32).flatten()
    )
    b = helper.make_tensor(
        "B", TensorProto.FLOAT, [32], np.zeros(32, dtype=np.float32).tolist()
    )

    nodes_order_a = [
        helper.make_node("Split", ["X", "split_lens"], ["u", "v"], axis=-1, name="sp0"),
        helper.make_node(
            "RMSNorm", ["v", "gamma"], ["v_norm"], domain="custom.pytorch", name="norm0"
        ),
        helper.make_node(
            "Linear",
            ["v_norm", "W", "B"],
            ["gate"],
            domain="custom.pytorch",
            name="lin0",
        ),
        helper.make_node("Mul", ["u", "gate"], ["Y"], name="mul0"),
    ]
    graph_a = helper.make_graph(
        nodes_order_a, "TestGraphA", [x], [y], initializer=[split_lens, gamma, w, b]
    )
    model_a = helper.make_model(graph_a, opset_imports=[helper.make_opsetid("", 17)])

    pass_obj = MacroBlockSynthesizer()
    fused_model_a = pass_obj.apply_spatial_gating_unit_fusion(
        model_a, mode="function_proto"
    )
    sgu_nodes_a = [
        n for n in fused_model_a.graph.node if n.op_type == "SpatialGatingUnit"
    ]

    # Contract: SGU fusion must match on standard (u, gate) ordering
    assert len(sgu_nodes_a) == 1, (
        "PATTERN MISSED: SGU fusion completely failed on standard (u, gate) order due to boolean short-circuit!"
    )


# ==============================================================================
# VULNERABILITY 15: [SPEC VIOLATION] HighwayBlock Passes 2 Inputs but FunctionProto Expects 3
# ==============================================================================
def test_vuln_15_highway_block_missing_skip_input_x():
    """Vulnerability 15: HighwayBlock node must pass carry input 'x' to match FunctionProto.

    FAILS WITH: AssertionError: HighwayBlock node has 2 inputs, function expects 3 (missing 'x')
    """
    h = helper.make_tensor_value_info("H", TensorProto.FLOAT, [1, 16, 64])
    gate_in = helper.make_tensor_value_info("GateIn", TensorProto.FLOAT, [1, 16, 64])
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16, 64])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 64])

    c_one = helper.make_tensor("c_one", TensorProto.FLOAT, [1], [1.0])

    nodes = [
        helper.make_node(
            "GatedUnit",
            ["H", "GateIn"],
            ["trans_out"],
            domain="custom.pytorch",
            name="gu0",
        ),
        helper.make_node("Sigmoid", ["GateIn"], ["t"], name="sig0"),
        helper.make_node("Sub", ["c_one", "t"], ["carry"], name="sub0"),
        helper.make_node("Mul", ["X", "carry"], ["carry_out"], name="mul_c"),
        helper.make_node("Add", ["trans_out", "carry_out"], ["Y"], name="add0"),
    ]
    graph = helper.make_graph(
        nodes, "TestGraph", [h, gate_in, x], [y], initializer=[c_one]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    pass_obj = MacroBlockSynthesizer()
    fused_model = pass_obj.apply_highway_block_fusion(model, mode="function_proto")

    hw_node = next(n for n in fused_model.graph.node if n.op_type == "HighwayBlock")
    func = next(f for f in fused_model.functions if f.name == "HighwayBlock")

    # Contract: Node input count must equal FunctionProto input count (3: h, gate_in, x)
    assert len(hw_node.input) == len(func.input), (
        f"SPEC VIOLATION: HighwayBlock node has {len(hw_node.input)} inputs, "
        f"but FunctionProto expects {len(func.input)} inputs (missing 'x')!"
    )


# ==============================================================================
# VULNERABILITY 16: [CRASH ON 2D WEIGHTS] SqueezeExcitation FunctionProto Hardcodes Conv
# ==============================================================================
def test_vuln_16_squeeze_excitation_hardcodes_conv_on_linear_weights():
    """Vulnerability 16: SqueezeExcitation FunctionProto must not hardcode Conv on Linear blocks.

    FAILS WITH: AssertionError: SqueezeExcitation FunctionProto hardcoded Conv for 2D Linear weights
    """
    model = helper.make_model(
        helper.make_graph([], "EmptyGraph", [], []),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    pass_obj = MacroBlockSynthesizer()
    pass_obj._attach_se_function_def(model)

    func = next(f for f in model.functions if f.name == "SqueezeExcitation")
    op_types = [n.op_type for n in func.node]

    # Contract: SE block from Linear/MatMul must NOT use Conv nodes inside FunctionProto
    assert "Conv" not in op_types, (
        "RANK MISMATCH BUG: SqueezeExcitation FunctionProto hardcoded Conv for 2D Linear weight matrices!"
    )


# ==============================================================================
# VULNERABILITY 17: [MISSING FUNCTIONPROTO] MultiHeadAttention Composite Fusion Has No FunctionProto
# ==============================================================================
def test_vuln_17_mha_composite_fusion_missing_function_proto():
    """Vulnerability 17: MultiHeadAttention composite fusion must register a FunctionProto definition.

    FAILS WITH: AssertionError: MultiHeadAttention node was created without registering FunctionProto
    """
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16, 64])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 64])
    w_out = helper.make_tensor(
        "W_out", TensorProto.FLOAT, [64, 64], np.eye(64, dtype=np.float32).flatten()
    )

    nodes = [
        helper.make_node(
            "SplitQKV", ["X"], ["q", "k", "v"], domain="custom.pytorch", name="sp0"
        ),
        helper.make_node(
            "ScaledDotProductAttention",
            ["q", "k", "v"],
            ["attn_out"],
            domain="custom.pytorch",
            name="sdpa0",
        ),
        helper.make_node(
            "MergeHeads",
            ["attn_out"],
            ["merged"],
            domain="custom.pytorch",
            name="merge0",
        ),
        helper.make_node(
            "Linear",
            ["merged", "W_out"],
            ["Y"],
            domain="custom.pytorch",
            name="lin_out",
        ),
    ]
    graph = helper.make_graph(nodes, "TestGraph", [x], [y], initializer=[w_out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    pass_obj = MacroBlockSynthesizer()
    fused_model = pass_obj.apply_mha_composite_fusion(model, mode="function_proto")

    # Contract: MultiHeadAttention node must have a corresponding FunctionProto
    has_mha_func = any(f.name == "MultiHeadAttention" for f in fused_model.functions)
    assert has_mha_func, (
        "MISSING FUNCTIONPROTO: MultiHeadAttention node was created without registering a FunctionProto definition!"
    )


# ==============================================================================
# VULNERABILITY 18: [GRAPH CORRUPTION] TensorRoute Deletes Multi-Consumer Gather Producers
# ==============================================================================
def test_vuln_18_tensor_route_deletes_multiconsumer_gather():
    """Vulnerability 18: TensorRoute in AxialSynthesizer must not delete Gather if consumed elsewhere.

    FAILS WITH: AssertionError: Multi-consumer Gather producer was deleted, corrupting consumer input
    """
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 32, 64])
    res = helper.make_tensor_value_info("Res", TensorProto.FLOAT, [1, 64])
    y1 = helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [1, 16])
    y2 = helper.make_tensor_value_info("Y2", TensorProto.FLOAT, [1, 64])

    idx = helper.make_tensor("idx", TensorProto.INT64, [1], [0])
    starts = helper.make_tensor("starts", TensorProto.INT64, [1], [0])
    ends = helper.make_tensor("ends", TensorProto.INT64, [1], [16])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [1])

    nodes = [
        helper.make_node(
            "Gather", ["X", "idx"], ["gather_out"], axis=1, name="gather0"
        ),
        helper.make_node(
            "Slice", ["gather_out", "starts", "ends", "axes"], ["Y1"], name="slice0"
        ),
        helper.make_node("Add", ["gather_out", "Res"], ["Y2"], name="add_res"),
    ]
    graph = helper.make_graph(
        nodes, "TestGraph", [x, res], [y1, y2], initializer=[idx, starts, ends, axes]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    pass_obj = AxialSynthesizer()
    fused_model = pass_obj.apply_index_tensor_routing(model, mode="function_proto")

    # Contract: gather0 MUST NOT be deleted because add_res needs gather_out!
    gather_survived = any(n.name == "gather0" for n in fused_model.graph.node)
    assert gather_survived, (
        "DANGLING EDGE BUG: Multi-consumer Gather producer was deleted, corrupting consumer input!"
    )
