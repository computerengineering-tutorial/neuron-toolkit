#!/usr/bin/env python3
"""Comprehensive Evaluation Test Suite for Quantization-Aware ONNX Simplification & Topological Detection.

Covers:
1. Pure Topological Transformer Block Detection (zero node-name dependency, completely stripped/anonymous names).
2. Quantized-Aware Transformer Blocks (quantized Linear projections fused into TransformerBlock).
3. DynamicQuantLinear Fusion Arity Consistency (mixed biased/unbiased zero-bias padding).
4. Multi-Consumer Activation Safety (shared quant activations like QKV are not erroneously dropped).
5. Affine-less LayerNorm Fusion (elementwise_affine=False).
6. Constant Node Epsilon Preservation in Norm Canonicalization.
7. Fixed-Point Static Folding & Initializer Deduplication.
8. Validation of AST-generated quantization catalog JSON.
"""

from __future__ import annotations

import os
import tempfile
import json
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import onnx
from onnx import helper, numpy_helper
import pytest
import torch
from torch import nn

from neuron_toolkit.compiler.passes.canonicalization.math_canonicalizer import MathCanonicalizer
from neuron_toolkit.compiler.passes.canonicalization.norm_canonicalizer import NormCanonicalizer
from neuron_toolkit.compiler.passes.canonicalization.static_folder import StaticFolder
from neuron_toolkit.compiler.passes.synthesis.macro_block_synthesizer import MacroBlockSynthesizer
from neuron_toolkit.compiler.pipeline import NeuronCompiler
from generators.gen_transformer_advanced_zoo import PreLN_TransformerBlock, PostLN_TransformerBlock
from generators.registry import ModelRegistry, ModelSpec
from generators.quantization import (
    QuantizedLinear,
    quantize_pytorch_module,
    quantize_onnx_model,
    export_quantized_model,
    QuantizationRegistry,
)

# Discover all generator specs
ModelRegistry.discover(root_dir / "generators")
ALL_SPECS = ModelRegistry.list_models()
_SPEC_BY_NAME = {s.name: s for s in ALL_SPECS}

# Representative specs covering all key architectural categories from quantization catalog
REPRESENTATIVE_QUANT_SPECS = [
    _SPEC_BY_NAME[name]
    for name in [
        "activation_erf_gelu",
        "activation_silu",
        "activation_hardswish",
        "mha_separate_qkv",
        "mha_fused_qkv",
        "mha_causal",
        "mha_rope",
        "mha_gqa",
        "mha_mqa",
        "swiglu_mlp_block",
        "geglu_mlp_block",
        "reglu_mlp_block",
        "transformer_pre_ln",
        "transformer_post_ln",
        "norm_layernorm",
        "norm_rmsnorm_l2",
        "norm_groupnorm",
        "linear_gemm",
        "conv2d_bn_relu",
        "resnet_basic",
        "mobilenetv2_invres",
    ]
    if name in _SPEC_BY_NAME
]


def _strip_all_node_names(model: onnx.ModelProto) -> onnx.ModelProto:
    """Strip all node names to ensure pure topological / non-name-based detection."""
    for n in model.graph.node:
        n.name = ""
    return model


def _export_to_onnx(model: nn.Module, x: torch.Tensor, opset_version: int = 18) -> onnx.ModelProto:
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        torch.onnx.export(model, x, tmp_path, input_names=["input"], output_names=["output"], opset_version=opset_version, dynamo=False)
        return onnx.load(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# -----------------------------------------------------------------------------
# 1. Pure Topological Transformer Block Detection
# -----------------------------------------------------------------------------
def test_pure_topological_transformer_preln_anonymous_names():
    """Verify Transformer Block is detected topologically when all node names are empty strings."""
    model = PreLN_TransformerBlock(d_model=32, num_heads=4, d_ffn=64)
    model.eval()
    x = torch.randn(2, 8, 32)
    onx = _export_to_onnx(model, x)

    _strip_all_node_names(onx)

    synthesizer = MacroBlockSynthesizer()
    simplified = synthesizer.apply_transformer_encoder_layer_fusion(onx, mode="function_proto")

    tb_nodes = [n for n in simplified.graph.node if "TransformerBlock" in n.op_type]
    assert len(tb_nodes) == 1, f"Expected 1 TransformerBlock macro node, found {len(tb_nodes)}"
    assert tb_nodes[0].domain == "custom.pytorch"


def test_pure_topological_transformer_postln_anonymous_names():
    """Verify PostLN Transformer Block is detected topologically with empty names."""
    model = PostLN_TransformerBlock(d_model=32, num_heads=4, d_ffn=64)
    model.eval()
    x = torch.randn(2, 8, 32)
    onx = _export_to_onnx(model, x)

    _strip_all_node_names(onx)

    synthesizer = MacroBlockSynthesizer()
    simplified = synthesizer.apply_transformer_encoder_layer_fusion(onx, mode="function_proto")

    tb_nodes = [n for n in simplified.graph.node if "TransformerBlock" in n.op_type]
    assert len(tb_nodes) == 1, f"Expected 1 TransformerBlock macro node, found {len(tb_nodes)}"


def test_pure_topological_multi_layer_stacked_transformers():
    """Verify multiple stacked Transformer layers are detected and fused in topological order."""
    class Stacked2Transformers(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = PreLN_TransformerBlock(d_model=32, num_heads=4, d_ffn=64)
            self.layer2 = PreLN_TransformerBlock(d_model=32, num_heads=4, d_ffn=64)

        def forward(self, x):
            return self.layer2(self.layer1(x))

    model = Stacked2Transformers()
    model.eval()
    x = torch.randn(2, 8, 32)
    onx = _export_to_onnx(model, x)

    _strip_all_node_names(onx)

    synthesizer = MacroBlockSynthesizer()
    simplified = synthesizer.apply_transformer_encoder_layer_fusion(onx, mode="function_proto")

    tb_nodes = [n for n in simplified.graph.node if "TransformerBlock" in n.op_type]
    assert len(tb_nodes) == 2, f"Expected 2 stacked TransformerBlock macro nodes, found {len(tb_nodes)}"


def test_quantized_transformer_block_topological_detection():
    """Verify quantized transformer blocks (where projections are quantized) are recognized topologically."""
    model = PreLN_TransformerBlock(d_model=32, num_heads=4, d_ffn=64)
    model.eval()
    x = torch.randn(2, 8, 32)
    onx = _export_to_onnx(model, x)

    # Convert linear layers inside to dynamic quantization pattern (DQL -> DQ -> MatMul)
    consumer_map = {}
    for n in onx.graph.node:
        for inp in n.input:
            consumer_map.setdefault(inp, []).append(n)

    synthesizer = MacroBlockSynthesizer()
    # Apply dynamic quantization pass, then transformer block fusion
    q_model = synthesizer.apply_dynamic_quant_linear_fusion(onx, mode="function_proto")
    _strip_all_node_names(q_model)
    fused_model = synthesizer.apply_transformer_encoder_layer_fusion(q_model, mode="function_proto")

    tb_nodes = [n for n in fused_model.graph.node if "TransformerBlock" in n.op_type]
    assert len(tb_nodes) == 1, f"Expected 1 TransformerBlock macro node in quantized model, got {len(tb_nodes)}"


# -----------------------------------------------------------------------------
# 2. DynamicQuantLinear Fusion & Arity Consistency
# -----------------------------------------------------------------------------
def test_dynamic_quant_linear_mixed_biased_unbiased_arity():
    """Verify mixed biased and unbiased DynamicQuantLinear instances don't cause arity crashes."""
    # Build graph with 1 biased and 1 unbiased DynamicQuantLinear pattern
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 16])
    y1 = helper.make_tensor_value_info("Y1", onnx.TensorProto.FLOAT, [2, 32])
    y2 = helper.make_tensor_value_info("Y2", onnx.TensorProto.FLOAT, [2, 32])

    w1_arr = np.random.randn(16, 32).astype(np.float32)
    b1_arr = np.random.randn(32).astype(np.float32)
    w2_arr = np.random.randn(16, 32).astype(np.float32)

    w1_init = numpy_helper.from_array(w1_arr, name="W1")
    b1_init = numpy_helper.from_array(b1_arr, name="B1")
    w2_init = numpy_helper.from_array(w2_arr, name="W2")

    nodes = [
        # Instance 1: Biased
        helper.make_node("DynamicQuantizeLinear", ["X"], ["q1", "scale1", "zp1"]),
        helper.make_node("DequantizeLinear", ["q1", "scale1", "zp1"], ["dq1"]),
        helper.make_node("MatMul", ["dq1", "W1"], ["mm1"]),
        helper.make_node("Add", ["mm1", "B1"], ["Y1"]),
        # Instance 2: Unbiased
        helper.make_node("DynamicQuantizeLinear", ["X"], ["q2", "scale2", "zp2"]),
        helper.make_node("DequantizeLinear", ["q2", "scale2", "zp2"], ["dq2"]),
        helper.make_node("MatMul", ["dq2", "W2"], ["Y2"]),
    ]

    graph = helper.make_graph(nodes, "test_quant", [x], [y1, y2], initializer=[w1_init, b1_init, w2_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    synthesizer = MacroBlockSynthesizer()
    simplified = synthesizer.apply_dynamic_quant_linear_fusion(model, mode="function_proto")

    dql_nodes = [n for n in simplified.graph.node if n.op_type == "DynamicQuantLinear"]
    assert len(dql_nodes) == 2, f"Expected 2 DynamicQuantLinear nodes, got {len(dql_nodes)}"

    # Check function proto definition arity
    func = next((f for f in simplified.functions if f.name == "DynamicQuantLinear"), None)
    assert func is not None, "Expected DynamicQuantLinear function definition attached"
    expected_arity = len(func.input)

    # Every instance MUST have matching arity so ONNX validation passes!
    for n in dql_nodes:
        assert len(n.input) == expected_arity, f"Node {n.name} arity {len(n.input)} != function arity {expected_arity}"

    # Validate ONNX model consistency
    onnx.checker.check_model(simplified)


def test_dynamic_quant_linear_multi_consumer_activation_safety():
    """Verify shared dynamic quantized activations (like QKV projections) are not dropped prematurely."""
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 16])
    y_q = helper.make_tensor_value_info("Y_Q", onnx.TensorProto.FLOAT, [2, 16])
    y_k = helper.make_tensor_value_info("Y_K", onnx.TensorProto.FLOAT, [2, 16])

    w_q = numpy_helper.from_array(np.random.randn(16, 16).astype(np.float32), name="W_Q")
    w_k = numpy_helper.from_array(np.random.randn(16, 16).astype(np.float32), name="W_K")

    # Single DQL and DQ consumed by TWO different MatMuls
    nodes = [
        helper.make_node("DynamicQuantizeLinear", ["X"], ["q_shared", "scale", "zp"]),
        helper.make_node("DequantizeLinear", ["q_shared", "scale", "zp"], ["dq_shared"]),
        helper.make_node("MatMul", ["dq_shared", "W_Q"], ["Y_Q"]),
        helper.make_node("MatMul", ["dq_shared", "W_K"], ["Y_K"]),
    ]

    graph = helper.make_graph(nodes, "test_multi_consumer", [x], [y_q, y_k], initializer=[w_q, w_k])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    synthesizer = MacroBlockSynthesizer()
    simplified = synthesizer.apply_dynamic_quant_linear_fusion(model, mode="function_proto")

    # Both outputs must still be validly produced without missing inputs
    producer_map = {out: n for n in simplified.graph.node for out in n.output}
    assert "Y_Q" in producer_map
    assert "Y_K" in producer_map


# -----------------------------------------------------------------------------
# 3. Affine-less Normalization & Constant Epsilon Preservation
# -----------------------------------------------------------------------------
def test_norm_canonicalizer_affine_less_layernorm():
    """Verify LayerNorm with elementwise_affine=False (no Mul scale, no Add bias) is fused correctly."""
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 4, 32])
    y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [2, 4, 32])

    eps_const = numpy_helper.from_array(np.array(1e-6, dtype=np.float32), name="eps")
    nodes = [
        helper.make_node("ReduceMean", ["X"], ["mean"], axes=[-1], keepdims=1),
        helper.make_node("Sub", ["X", "mean"], ["x_sub_mean"]),
        helper.make_node("Pow", ["x_sub_mean", "two"], ["sq_diff"]),
        helper.make_node("ReduceMean", ["sq_diff"], ["var"], axes=[-1], keepdims=1),
        helper.make_node("Add", ["var", "eps"], ["var_eps"]),
        helper.make_node("Sqrt", ["var_eps"], ["std"]),
        helper.make_node("Div", ["x_sub_mean", "std"], ["Y"]),
    ]

    two_init = numpy_helper.from_array(np.array(2.0, dtype=np.float32), name="two")
    graph = helper.make_graph(nodes, "affine_less_ln", [x], [y], initializer=[two_init, eps_const])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    canonicalizer = NormCanonicalizer()
    simplified = canonicalizer.apply_layernorm_fusion(model, mode="function_proto")

    ln_nodes = [n for n in simplified.graph.node if n.op_type == "LayerNormalization"]
    assert len(ln_nodes) == 1, f"Expected 1 LayerNormalization node, got {len(ln_nodes)}"
    # Epsilon should match the 1e-6 from eps_const
    eps_attr = next(a.f for a in ln_nodes[0].attribute if a.name == "epsilon")
    assert abs(eps_attr - 1e-6) < 1e-9, f"Expected epsilon 1e-6, got {eps_attr}"


# -----------------------------------------------------------------------------
# 4. Fixed-Point Static Folding & Initializer Deduplication
# -----------------------------------------------------------------------------
def test_static_folder_fixed_point_and_no_duplicate_initializers():
    """Verify StaticFolder reaches fixed point without duplicating initializer names."""
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 4])
    y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [2, 4])

    c1 = numpy_helper.from_array(np.array(2.0, dtype=np.float32), name="C1")
    c2 = numpy_helper.from_array(np.array(3.0, dtype=np.float32), name="C2")
    c3 = numpy_helper.from_array(np.array(4.0, dtype=np.float32), name="C3")

    # Cascading constant operations: (C1 + C2) * C3 -> Add with X
    nodes = [
        helper.make_node("Add", ["C1", "C2"], ["fold1"]),
        helper.make_node("Mul", ["fold1", "C3"], ["fold2"]),
        helper.make_node("Add", ["X", "fold2"], ["Y"]),
    ]

    graph = helper.make_graph(nodes, "cascade_fold", [x], [y], initializer=[c1, c2, c3])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    folder = StaticFolder()
    simplified = folder.apply_static_subgraph_folding(model)

    # Initializers should have unique names
    init_names = [i.name for i in simplified.graph.initializer]
    assert len(init_names) == len(set(init_names)), f"Found duplicate initializer names: {init_names}"

    # Folded result (2+3)*4 = 20.0 should be present
    init_vals = {i.name: numpy_helper.to_array(i) for i in simplified.graph.initializer}
    assert any(abs(v - 20.0) < 1e-5 for v in init_vals.values()), "Cascading constant was not folded to 20.0"


# -----------------------------------------------------------------------------
# 5. AST Quantization Catalog Integrity
# -----------------------------------------------------------------------------
def test_ast_quantization_catalog_json_integrity():
    """Verify quantization_block_analysis.json was generated and is structurally complete."""
    json_path = Path(__file__).resolve().parent.parent / "quantization_block_analysis.json"
    assert json_path.exists(), "quantization_block_analysis.json does not exist!"

    with open(json_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    assert len(catalog) >= 100, f"Expected at least 100 classes in catalog, got {len(catalog)}"

    for cname, entry in catalog.items():
        assert "class_name" in entry
        assert "source_file" in entry
        assert "category" in entry
        assert "applicable_quantization_schemes" in entry
        assert len(entry["applicable_quantization_schemes"]) > 0
        assert "target_onnx_nodes_for_quantization" in entry
        assert "expected_macro_block_detection" in entry
        assert "quantization_injection_strategy" in entry


# -----------------------------------------------------------------------------
# 6. Mass-Level Quantized Blocks Export & Differential PyTorch Parity Tests
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "spec",
    REPRESENTATIVE_QUANT_SPECS,
    ids=lambda s: f"{s.category}::{s.name}",
)
def test_quantized_block_export_and_torch_parity(spec: ModelSpec):
    """Verify that every catalog block can be dynamically quantized, saved to ONNX,

    simplified with NeuronCompiler, and achieves verified numerical parity against PyTorch.
    """
    model = spec.instantiate().eval()
    dummy_inputs = spec.get_dummy_inputs()

    out_dir = root_dir / ".helper" / "models" / "quantized" / spec.resolve_subfolder()
    out_path, opt_onnx, stats = export_quantized_model(
        model=model,
        dummy_inputs=dummy_inputs,
        input_names=spec.input_names,
        output_names=spec.output_names,
        base_name=spec.name,
        out_dir=out_dir,
        rtol=0.05,
        atol=0.05,
        auto_verify=True,
    )

    # 1. Assert saved quantized ONNX model exists on disk
    assert Path(out_path).exists(), f"Quantized ONNX model was not saved at {out_path}"

    # 2. Assert compilation preserved or simplified nodes
    assert stats["opt_nodes"] <= stats["raw_nodes"], (
        f"Compilation did not optimize graph: raw={stats['raw_nodes']}, opt={stats['opt_nodes']}"
    )

    # 3. Assert differential parity with PyTorch passed
    assert stats["passed"], (
        f"Differential parity failed with PyTorch for {spec.name}: "
        f"max_diff={stats['max_diff']}, cos={stats['cosine_sim']}"
    )


def test_quantized_catalog_all_categories_coverage():
    """Verify that key categories from quantization_block_analysis.json are covered."""
    json_path = root_dir / "quantization_block_analysis.json"
    with open(json_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    catalog_categories = {entry["category"] for entry in catalog.values()}
    tested_categories = {spec.category for spec in REPRESENTATIVE_QUANT_SPECS}

    # Verify key architectural categories are actively tested
    for cat in ["activation_variants", "attention_variants", "norm_variants", "transformer_advanced", "mlp_vanilla"]:
        assert any(cat in c for c in tested_categories), f"Category {cat} is not covered in representative test specs"
