"""Exhaustive Adversarial Compiler Vulnerabilities & Edge-Case Stress Tests (Batch 5).

Focuses on edge cases, framework-agnostic universal patterns, and vulnerabilities
identified during real-world ONNX model simplifications across multiple architectures
(PyTorch, TensorFlow/TFLite, Keras, HuggingFace, etc.):
1. MathCanonicalizer hardcoded dimension fallback (out_features = 512).
2. MathCanonicalizer asymmetric MatMul blocking when biased Linear is present.
3. MacroBlockSynthesizer missing FunctionProto attachment for DynamicQuantLinear.
4. MacroBlockSynthesizer empty node name collision dropping all unnamed nodes.
5. NormCanonicalizer dropping intermediate nodes consumed externally / graph outputs.
6. NormCanonicalizer failing on bias-free / scale-free LayerNorm subgraphs.
7. MacroBlockSynthesizer missing transformer blocks with flat underscore naming.
8. MacroBlockSynthesizer dropping secondary outputs in multi-output transformer blocks.
9. Runtime OpRegistry DequantizeLinear dtype promotion violating ONNX specification.
10. StaticFolder empty node name collision deleting unrelated unnamed nodes.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest
import torch

from neuron_toolkit.compiler.passes.canonicalization.math_canonicalizer import MathCanonicalizer
from neuron_toolkit.compiler.passes.canonicalization.norm_canonicalizer import NormCanonicalizer
from neuron_toolkit.compiler.passes.canonicalization.static_folder import StaticFolder
from neuron_toolkit.compiler.passes.synthesis.macro_block_synthesizer import MacroBlockSynthesizer
from neuron_toolkit.runtime.registry import OpRegistry


# ==============================================================================
# VULNERABILITY 55: [MATH CANONICALIZER] Hardcoded 512 Dimension Fallback
# ==============================================================================
def test_vuln_55_math_canonicalizer_hardcoded_512_dim_fallback():
    """Vulnerability 55: Zero bias synthesis must not hardcode out_features = 512.

    When fusing unbiased linears in function_proto mode alongside biased linears,
    if weight is a graph input or dynamic tensor not present in initializers,
    the synthesizer must not blindly assume hidden dimension is 512.
    FAILS WITH: AssertionError: Expected zero-bias feature dimension 768, but got hardcoded 512!
    """
    hidden_dim = 768
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16, hidden_dim])
    W_dynamic = helper.make_tensor_value_info("W_dynamic", TensorProto.FLOAT, [hidden_dim, hidden_dim])
    Y1 = helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [1, 16, hidden_dim])
    Y2 = helper.make_tensor_value_info("Y2", TensorProto.FLOAT, [1, 16, hidden_dim])

    W_init = helper.make_tensor("W_init", TensorProto.FLOAT, [hidden_dim, hidden_dim], np.ones((hidden_dim, hidden_dim), dtype=np.float32).flatten())
    B_init = helper.make_tensor("B_init", TensorProto.FLOAT, [hidden_dim], np.zeros(hidden_dim, dtype=np.float32))

    gemm_biased = helper.make_node("Gemm", inputs=["X", "W_init", "B_init"], outputs=["Y1"], name="gemm_biased")
    gemm_unbiased = helper.make_node("Gemm", inputs=["X", "W_dynamic"], outputs=["Y2"], name="gemm_unbiased")

    graph = helper.make_graph(
        [gemm_biased, gemm_unbiased],
        "test_hardcoded_dim",
        [X, W_dynamic],
        [Y1, Y2],
        initializer=[W_init, B_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

    opt_m = MathCanonicalizer().apply_linear_fusion(model, mode="function_proto")

    zero_b_inits = [i for i in opt_m.graph.initializer if "zero_bias" in i.name]
    assert len(zero_b_inits) > 0, "Expected a synthesized zero-bias initializer for unbiased linear!"
    zero_b_arr = numpy_helper.to_array(zero_b_inits[0])
    assert zero_b_arr.shape[0] == hidden_dim, (
        f"VULNERABILITY DETECTED: Hardcoded 512 dimension fallback occurred! "
        f"Expected {hidden_dim}, but got {zero_b_arr.shape[0]}"
    )


# ==============================================================================
# VULNERABILITY 56: [MATH CANONICALIZER] Unbiased MatMul Blocked by Biased Linear
# ==============================================================================
def test_vuln_56_math_canonicalizer_blocks_unbiased_matmul_when_biased_linear_present():
    """Vulnerability 56: Unbiased MatMul should not be skipped when biased Linear is present.

    In function_proto mode, if any biased Linear exists, Step 3 skips converting all MatMuls.
    FAILS WITH: AssertionError: Unbiased MatMul was blocked from Linear fusion!
    """
    dim = 64
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16, dim])
    Y1 = helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [1, 16, dim])
    Y2 = helper.make_tensor_value_info("Y2", TensorProto.FLOAT, [1, 16, dim])

    W1 = helper.make_tensor("W1", TensorProto.FLOAT, [dim, dim], np.ones((dim, dim), dtype=np.float32).flatten())
    B1 = helper.make_tensor("B1", TensorProto.FLOAT, [dim], np.zeros(dim, dtype=np.float32))
    W2 = helper.make_tensor("W2", TensorProto.FLOAT, [dim, dim], np.ones((dim, dim), dtype=np.float32).flatten())

    n_biased = helper.make_node("Gemm", inputs=["X", "W1", "B1"], outputs=["Y1"], name="gemm_biased")
    n_unbiased = helper.make_node("MatMul", inputs=["X", "W2"], outputs=["Y2"], name="matmul_unbiased")

    graph = helper.make_graph([n_biased, n_unbiased], "test_matmul_blocking", [X], [Y1, Y2], initializer=[W1, B1, W2])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

    opt_m = MathCanonicalizer().apply_linear_fusion(model, mode="function_proto")

    remaining_matmuls = [n for n in opt_m.graph.node if n.op_type == "MatMul"]
    assert len(remaining_matmuls) == 0, (
        f"VULNERABILITY DETECTED: Unbiased MatMul was blocked from Linear fusion! "
        f"Remaining MatMuls: {[n.name for n in remaining_matmuls]}"
    )


# ==============================================================================
# VULNERABILITY 57: [MACRO SYNTHESIS] DynamicQuantLinear Missing FunctionProto
# ==============================================================================
def test_vuln_57_dynamic_quant_linear_missing_function_proto():
    """Vulnerability 57: DynamicQuantLinear must have FunctionProto in function_proto mode.

    FAILS WITH: AssertionError: DynamicQuantLinear FunctionProto was never registered in model.functions!
    """
    dim = 32
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8, dim])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8, dim])
    W = helper.make_tensor("W", TensorProto.FLOAT, [dim, dim], np.ones((dim, dim), dtype=np.float32).flatten())

    n_dql = helper.make_node("DynamicQuantizeLinear", inputs=["X"], outputs=["q_x", "scale_x", "zp_x"], name="dql")
    n_dq = helper.make_node("DequantizeLinear", inputs=["q_x", "scale_x", "zp_x"], outputs=["dq_x"], name="dq")
    n_mm = helper.make_node("MatMul", inputs=["dq_x", "W"], outputs=["Y"], name="mm")

    graph = helper.make_graph([n_dql, n_dq, n_mm], "test_dql", [X], [Y], initializer=[W])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

    opt_m = MacroBlockSynthesizer().apply_dynamic_quant_linear_fusion(model, mode="function_proto")

    custom_funcs = [f.name for f in opt_m.functions if f.domain == "custom.pytorch"]
    assert "DynamicQuantLinear" in custom_funcs, (
        "VULNERABILITY DETECTED: DynamicQuantLinear FunctionProto definition missing from model.functions! "
        f"Registered functions: {custom_funcs}"
    )


# ==============================================================================
# VULNERABILITY 58: [MACRO SYNTHESIS] Empty Node Name Drops All Unnamed Nodes
# ==============================================================================
def test_vuln_58_empty_node_name_drops_all_unnamed_nodes():
    """Vulnerability 58: Empty node name ('') in nodes_to_drop must not drop all unnamed nodes.

    Many converted models (TFLite/CoreML) have blank node names. When nodes_to_drop contains '',
    all unnamed nodes in the entire model match n.name in nodes_to_drop and get deleted.
    FAILS WITH: AssertionError: Unrelated unnamed Relu node was deleted!
    """
    dim = 16
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, dim])
    Y1 = helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [1, 4, dim])
    Y2 = helper.make_tensor_value_info("Y2", TensorProto.FLOAT, [1, 4, dim])
    W = helper.make_tensor("W", TensorProto.FLOAT, [dim, dim], np.ones((dim, dim), dtype=np.float32).flatten())

    # Nodes with name=""
    n_dql = helper.make_node("DynamicQuantizeLinear", inputs=["X"], outputs=["q_x", "scale_x", "zp_x"], name="")
    n_dq = helper.make_node("DequantizeLinear", inputs=["q_x", "scale_x", "zp_x"], outputs=["dq_x"], name="")
    n_mm = helper.make_node("MatMul", inputs=["dq_x", "W"], outputs=["Y1"], name="")
    n_unrelated_relu = helper.make_node("Relu", inputs=["X"], outputs=["Y2"], name="")

    graph = helper.make_graph([n_dql, n_dq, n_mm, n_unrelated_relu], "test_empty_name", [X], [Y1, Y2], initializer=[W])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

    opt_m = MacroBlockSynthesizer().apply_dynamic_quant_linear_fusion(model, mode="function_proto")

    relu_nodes = [n for n in opt_m.graph.node if n.op_type == "Relu"]
    assert len(relu_nodes) == 1, (
        "VULNERABILITY DETECTED: Unrelated unnamed Relu node was dropped due to empty name collision in nodes_to_drop!"
    )


# ==============================================================================
# VULNERABILITY 59: [NORM CANONICALIZER] Dropping Intermediate External Consumers
# ==============================================================================
def test_vuln_59_norm_canonicalizer_drops_intermediate_consumed_externally():
    """Vulnerability 59: Intermediate nodes must not be dropped if consumed outside normalization.

    When decomposing LayerNorm, rm_mean or sub may be used by auxiliary branches or graph outputs.
    Dropping them unconditionally leaves missing inputs in the graph.
    FAILS WITH: AssertionError: Dangling inputs detected after LayerNorm fusion!
    """
    dim = 16
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, dim])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, dim])
    Mean_out = helper.make_tensor_value_info("Mean_out", TensorProto.FLOAT, [1, 4, 1])

    scale = helper.make_tensor("scale", TensorProto.FLOAT, [dim], np.ones(dim, dtype=np.float32))
    bias = helper.make_tensor("bias", TensorProto.FLOAT, [dim], np.zeros(dim, dtype=np.float32))
    two = helper.make_tensor("two", TensorProto.FLOAT, [], np.array(2.0, dtype=np.float32))
    eps = helper.make_tensor("eps", TensorProto.FLOAT, [], np.array(1e-5, dtype=np.float32))

    axes = helper.make_tensor("axes", TensorProto.INT64, [1], np.array([-1], dtype=np.int64))

    # Standard LayerNorm decomposition
    n_mean = helper.make_node("ReduceMean", inputs=["X", "axes"], outputs=["mean_val"], name="rm_mean")
    n_sub = helper.make_node("Sub", inputs=["X", "mean_val"], outputs=["sub_val"], name="sub")
    n_pow = helper.make_node("Pow", inputs=["sub_val", "two"], outputs=["pow_val"], name="pow")
    n_var = helper.make_node("ReduceMean", inputs=["pow_val", "axes"], outputs=["var_val"], name="rm_var")
    n_add = helper.make_node("Add", inputs=["var_val", "eps"], outputs=["var_eps"], name="add_eps")
    n_sqrt = helper.make_node("Sqrt", inputs=["var_eps"], outputs=["std_val"], name="sqrt")
    n_div = helper.make_node("Div", inputs=["sub_val", "std_val"], outputs=["norm_val"], name="div")
    n_mul = helper.make_node("Mul", inputs=["norm_val", "scale"], outputs=["scaled_val"], name="mul")
    n_add_b = helper.make_node("Add", inputs=["scaled_val", "bias"], outputs=["Y"], name="add_bias")

    # Auxiliary branch consuming mean_val
    n_aux = helper.make_node("Abs", inputs=["mean_val"], outputs=["Mean_out"], name="aux_abs")

    graph = helper.make_graph(
        [n_mean, n_sub, n_pow, n_var, n_add, n_sqrt, n_div, n_mul, n_add_b, n_aux],
        "test_dangling_norm",
        [X],
        [Y, Mean_out],
        initializer=[scale, bias, two, eps, axes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    opt_m = NormCanonicalizer().apply_layernorm_fusion(model, mode="modern_onnx")

    producers = (
        {out for n in opt_m.graph.node for out in n.output}
        | {i.name for i in opt_m.graph.input}
        | {i.name for i in opt_m.graph.initializer}
    )
    missing_inputs = [inp for n in opt_m.graph.node for inp in n.input if inp and inp not in producers]
    assert len(missing_inputs) == 0, (
        f"VULNERABILITY DETECTED: Dangling inputs detected after LayerNorm fusion: {missing_inputs}"
    )


# ==============================================================================
# VULNERABILITY 60: [NORM CANONICALIZER] Bias-Free LayerNorm Skipped
# ==============================================================================
def test_vuln_60_norm_canonicalizer_fails_on_bias_free_layernorm():
    """Vulnerability 60: LayerNorm without additive bias must not be skipped.

    Modern LLMs frequently use bias-free LayerNorm or RMSNorm. Current code requires add_bias_candidates.
    FAILS WITH: AssertionError: Bias-free LayerNormalization was skipped from fusion!
    """
    dim = 16
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, dim])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, dim])

    scale = helper.make_tensor("scale", TensorProto.FLOAT, [dim], np.ones(dim, dtype=np.float32))
    two = helper.make_tensor("two", TensorProto.FLOAT, [], np.array(2.0, dtype=np.float32))
    eps = helper.make_tensor("eps", TensorProto.FLOAT, [], np.array(1e-5, dtype=np.float32))
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], np.array([-1], dtype=np.int64))

    n_mean = helper.make_node("ReduceMean", inputs=["X", "axes"], outputs=["mean_val"], name="rm_mean")
    n_sub = helper.make_node("Sub", inputs=["X", "mean_val"], outputs=["sub_val"], name="sub")
    n_pow = helper.make_node("Pow", inputs=["sub_val", "two"], outputs=["pow_val"], name="pow")
    n_var = helper.make_node("ReduceMean", inputs=["pow_val", "axes"], outputs=["var_val"], name="rm_var")
    n_add = helper.make_node("Add", inputs=["var_val", "eps"], outputs=["var_eps"], name="add_eps")
    n_sqrt = helper.make_node("Sqrt", inputs=["var_eps"], outputs=["std_val"], name="sqrt")
    n_div = helper.make_node("Div", inputs=["sub_val", "std_val"], outputs=["norm_val"], name="div")
    n_mul = helper.make_node("Mul", inputs=["norm_val", "scale"], outputs=["Y"], name="mul_no_bias")

    graph = helper.make_graph(
        [n_mean, n_sub, n_pow, n_var, n_add, n_sqrt, n_div, n_mul],
        "test_bias_free_ln",
        [X],
        [Y],
        initializer=[scale, two, eps, axes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    opt_m = NormCanonicalizer().apply_layernorm_fusion(model, mode="modern_onnx")

    ln_nodes = [n for n in opt_m.graph.node if "LayerNorm" in n.op_type]
    assert len(ln_nodes) >= 1, (
        "VULNERABILITY DETECTED: Bias-free LayerNormalization was skipped from fusion! "
        f"Nodes remain: {[n.op_type for n in opt_m.graph.node]}"
    )


# ==============================================================================
# VULNERABILITY 61: [MACRO SYNTHESIS] Flat-Named Transformer Layers Missed
# ==============================================================================
def test_vuln_61_transformer_encoder_layer_flat_naming_missed():
    """Vulnerability 61: Transformer layers with flat naming (layer_0_*) must be discovered.

    Current hierarchy depth search splits by [/.] and inspects parts[:-1], which is empty for flat names.
    FAILS WITH: AssertionError: Flat-named transformer blocks were not discovered!
    """
    dim = 16
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, dim])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, dim])

    nodes = []
    curr = "X"
    # Construct 2 consecutive repeating blocks with flat names
    for i in range(2):
        ln_out = f"l{i}_ln"
        add1_out = f"l{i}_add1"
        add2_out = f"l{i}_add2"
        # Layer nodes with flat underscore names (no '/' or '.')
        nodes.append(helper.make_node("LayerNormalization", inputs=[curr], outputs=[ln_out], name=f"layer_{i}_ln"))
        nodes.append(helper.make_node("Add", inputs=[curr, ln_out], outputs=[add1_out], name=f"layer_{i}_add_1"))
        nodes.append(helper.make_node("Add", inputs=[add1_out, ln_out], outputs=[add2_out], name=f"layer_{i}_add_2"))
        curr = add2_out

    # Final Identity to Y
    nodes.append(helper.make_node("Identity", inputs=[curr], outputs=["Y"], name="out_id"))

    graph = helper.make_graph(nodes, "test_flat_transformer", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    opt_m = MacroBlockSynthesizer().apply_transformer_encoder_layer_fusion(model, mode="function_proto")

    tb_nodes = [n for n in opt_m.graph.node if "Transformer" in n.op_type]
    assert len(tb_nodes) == 2, (
        f"VULNERABILITY DETECTED: Flat-named transformer blocks were not discovered! Found: {len(tb_nodes)}"
    )


# ==============================================================================
# VULNERABILITY 62: [MACRO SYNTHESIS] Multi-Output Transformer Blocks Lose Secondary Outputs
# ==============================================================================
def test_vuln_62_transformer_multi_output_dropped():
    """Vulnerability 62: Transformer blocks with secondary outputs must preserve them.

    Current code hardcodes ext_outs = [out_y], causing secondary outputs like attention probabilities
    or intermediate activations to be dropped.
    FAILS WITH: AssertionError: Secondary output was dropped during transformer block fusion!
    """
    dim = 16
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, dim])
    Y1 = helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [1, 4, dim])
    Y2 = helper.make_tensor_value_info("Y2", TensorProto.FLOAT, [1, 4, dim])

    nodes = []
    curr = "X"
    sec_out = None
    for i in range(2):
        ln_out = f"l{i}_ln"
        add1_out = f"l{i}_add1"
        add2_out = f"l{i}_add2"
        nodes.append(helper.make_node("LayerNormalization", inputs=[curr], outputs=[ln_out], name=f"encoder/layer_{i}/ln"))
        nodes.append(helper.make_node("Add", inputs=[curr, ln_out], outputs=[add1_out], name=f"encoder/layer_{i}/add_1"))
        nodes.append(helper.make_node("Add", inputs=[add1_out, ln_out], outputs=[add2_out], name=f"encoder/layer_{i}/add_2"))
        if i == 0:
            sec_out = add1_out  # Secondary output from layer 0 consumed externally
        curr = add2_out

    # Consumer of secondary output outside the block
    nodes.append(helper.make_node("Relu", inputs=[sec_out], outputs=["Y2"], name="aux_consumer"))
    nodes.append(helper.make_node("Identity", inputs=[curr], outputs=["Y1"], name="final_id"))

    graph = helper.make_graph(nodes, "test_multi_out_transformer", [X], [Y1, Y2])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    opt_m = MacroBlockSynthesizer().apply_transformer_encoder_layer_fusion(model, mode="function_proto")

    producers = (
        {out for n in opt_m.graph.node for out in n.output}
        | {i.name for i in opt_m.graph.input}
        | {i.name for i in opt_m.graph.initializer}
    )
    missing = [inp for n in opt_m.graph.node for inp in n.input if inp and inp not in producers]
    assert len(missing) == 0, (
        f"VULNERABILITY DETECTED: Secondary output was dropped during transformer block fusion! Missing: {missing}"
    )


# ==============================================================================
# VULNERABILITY 63: [RUNTIME REGISTRY] DequantizeLinear Dtype Promotion
# ==============================================================================
def test_vuln_63_dequantize_linear_dtype_promotion_violates_onnx_spec():
    """Vulnerability 63: DequantizeLinear must return output matching scale.dtype.

    PyTorch float32 - zp promotes float32 * float16 -> float32. ONNX spec requires scale.dtype.
    FAILS WITH: AssertionError: Expected output dtype torch.float16, got torch.float32!
    """
    reg = OpRegistry()
    x = torch.tensor([10, 20, 30], dtype=torch.uint8)
    scale = torch.tensor(0.5, dtype=torch.float16)
    zp = torch.tensor(0, dtype=torch.uint8)

    handler = reg.get("", "DequantizeLinear")
    assert handler is not None
    out = handler([x, scale, zp], {})
    assert out.dtype == torch.float16, (
        f"VULNERABILITY DETECTED: DequantizeLinear dtype promotion! Expected {torch.float16}, got {out.dtype}"
    )


# ==============================================================================
# VULNERABILITY 64: [STATIC FOLDER] Duplicate Node Name Collision Deletes Runtime Nodes
# ==============================================================================
def test_vuln_64_static_folder_duplicate_node_name_deletes_runtime_nodes():
    """Vulnerability 64: Duplicate node names in static folder must not drop unrelated runtime nodes.

    Many converted models (TFLite/CoreML) have duplicate default names (e.g. name='node').
    When folded_names uses string node names instead of unique identity id(n),
    unrelated runtime nodes with the same name get deleted.
    FAILS WITH: AssertionError: Unrelated runtime node was deleted by StaticFolder due to name collision!
    """
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    Y1 = helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [1, 4])
    Y2 = helper.make_tensor_value_info("Y2", TensorProto.FLOAT, [1, 4])

    C1 = helper.make_tensor("C1", TensorProto.FLOAT, [1, 4], np.ones((1, 4), dtype=np.float32).flatten())
    C2 = helper.make_tensor("C2", TensorProto.FLOAT, [1, 4], np.full((1, 4), 2.0, dtype=np.float32).flatten())

    # Static foldable node with name="op"
    n_static_add = helper.make_node("Add", inputs=["C1", "C2"], outputs=["folded_c"], name="op")
    # Runtime dynamic node with distinct name
    n_runtime_mul = helper.make_node("Mul", inputs=["X", "folded_c"], outputs=["Y1"], name="mul_runtime")
    # Unrelated runtime node sharing name="op"
    n_runtime_relu = helper.make_node("Relu", inputs=["X"], outputs=["Y2"], name="op")

    graph = helper.make_graph(
        [n_static_add, n_runtime_mul, n_runtime_relu],
        "test_static_name_collision",
        [X],
        [Y1, Y2],
        initializer=[C1, C2],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

    opt_m = StaticFolder().apply_static_subgraph_folding(model)

    relu_nodes = [n for n in opt_m.graph.node if n.op_type == "Relu"]
    assert len(relu_nodes) == 1, (
        "VULNERABILITY DETECTED: Unrelated runtime Relu node was deleted by StaticFolder due to name collision!"
    )
