"""Dynamic On-The-Fly Architectural & De-Obfuscation Invariance Test Suite.

Features:
- Zero static JSON descriptors on disk; all models and rules tested dynamically on-the-fly.
- Automatic program halt on any invariance or synthesis failure.
- Option 1 (De-Obfuscation Invariance): Proves that NeuronCompiler de-obfuscates models
  with identical numerical parity.
- Option 2 (tf2onnx Invariance): Tests resilience against tf2onnx-style NHWC ping-pongs,
  decomposed RMSNorm (Pow/Sqrt), and Tanh-GELU expansions.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import onnx
import pytest
import torch

try:
    import inspect

    from torch.onnx._internal.torchscript_exporter import utils as _ts_utils
    if hasattr(_ts_utils.setup_onnx_logging, "__wrapped__") and not inspect.isgeneratorfunction(_ts_utils.setup_onnx_logging.__wrapped__):
        _ts_utils.setup_onnx_logging = _ts_utils.setup_onnx_logging.__wrapped__
except (ImportError, AttributeError):
    pass

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from generators.obfuscation import ObfuscationRegistry
from generators.registry import ModelRegistry, ModelSpec
from neuron_toolkit.compiler.pipeline import NeuronCompiler
from neuron_toolkit.runtime.cross_verifier import CrossVerifier

# Discover all models dynamically from registry
ModelRegistry.discover(_REPO_ROOT / "generators")
RULE_SPECS = [s for s in ModelRegistry.list_models() if s.expected_target_node]


def _export_in_memory(spec: ModelSpec, opset: int = 18) -> onnx.ModelProto:
    """Export a ModelSpec to an in-memory ONNX ModelProto without touching disk."""
    model = spec.instantiate().eval()
    dummy = spec.get_dummy_inputs()
    buf = io.BytesIO()
    torch.onnx.utils.export(
        model,
        dummy,
        buf,
        input_names=spec.input_names,
        output_names=spec.output_names,
        opset_version=opset,
        dynamic_axes=spec.dynamic_axes,
    )
    buf.seek(0)
    return onnx.load_model_from_string(buf.getvalue())


# =============================================================================
# 1. On-The-Fly Canonical Op Synthesis & Parity Tests
# =============================================================================
@pytest.mark.parametrize("spec", RULE_SPECS, ids=lambda s: f"{s.category}::{s.name}")
def test_on_the_fly_canonical_rule(spec: ModelSpec) -> None:
    """Dynamically export, compile, and verify expected target node synthesis on-the-fly."""
    raw_model = _export_in_memory(spec, opset=18)
    opt_model = NeuronCompiler.compile(raw_model, mode="function_proto")

    # 1. Assert expected target op is synthesized
    expected_op = spec.expected_target_node
    assert expected_op is not None, f"Spec {spec.name} must define expected_target_node"

    opt_ops = {n.op_type for n in opt_model.graph.node}
    has_target = any(expected_op in op for op in opt_ops)
    assert has_target, (
        f"Expected target op '{expected_op}' was not synthesized in compiled graph for {spec.name}!\n"
        f"Available ops in compiled graph: {sorted(opt_ops)}"
    )

    # 2. Assert numerical parity with reference
    verifier = CrossVerifier(raw_model, opt_model, rtol=spec.rtol, atol=spec.atol)
    report = verifier.verify()
    assert report.all_passed, (
        f"Numerical parity verification FAILED for {spec.name} on-the-fly:\n{report.summary()}"
    )


# =============================================================================
# 2. Option 1: De-Obfuscation Architecture Block Invariance Tests
# =============================================================================
@pytest.mark.parametrize(
    "model_name",
    [
        "mha_separate_qkv",
        "convnext_block",
        "norm_layernorm",
        "norm_rmsnorm_l2",
        "transformer_ffn_gelu",
        "pytorch_obfuscated_complex",
    ],
)
def test_on_the_fly_deobfuscation_invariance(model_name: str) -> None:
    """Prove that NeuronCompiler de-obfuscates graphs so block detection matches clean models."""
    spec = ModelRegistry.get(model_name)

    # 1. Clean in-memory model
    raw_clean = _export_in_memory(spec, opset=18)

    # 2. Apply realistic multi-stage randomized obfuscation on-the-fly
    raw_obf, applied = ObfuscationRegistry.apply_onnx(raw_clean, seed=123)

    # 3. Compile both models with NeuronCompiler
    opt_clean = NeuronCompiler.compile(raw_clean, mode="function_proto")
    opt_obf = NeuronCompiler.compile(raw_obf, mode="function_proto")

    # 4. Parity Assertion: Numerical output between clean and obfuscated compiled models
    verifier = CrossVerifier(opt_clean, opt_obf, rtol=1e-4, atol=1e-4)
    report = verifier.verify()
    assert report.all_passed, (
        f"Numerical parity FAILED between clean and obfuscated compiled models for '{model_name}':\n"
        f"{report.summary()}"
    )


# =============================================================================
# 3. Option 2: tf2onnx Motifs & Special Decompositions Normalization Tests
# =============================================================================
def test_tf2onnx_nhwc_layout_normalization() -> None:
    """Verify that tf2onnx-style NHWC <-> NCHW layout ping-pongs are eliminated."""
    spec = ModelRegistry.get("convnext_block")
    raw = _export_in_memory(spec, opset=18)

    # Inject NHWC layout ping-pong
    mutated, applied = ObfuscationRegistry.apply_onnx(raw, strategy_names=["onnx_nhwc_layout_pingpong"], seed=42)
    assert "onnx_nhwc_layout_pingpong" in applied

    # Compile
    opt = NeuronCompiler.compile(mutated, mode="function_proto")
    verifier = CrossVerifier(raw, opt)
    report = verifier.verify()
    assert report.all_passed, f"NHWC ping-pong parity failed:\n{report.summary()}"


def test_tf2onnx_reciprocal_mul_normalization() -> None:
    """Verify that tf2onnx Reciprocal -> Mul is normalized and numerically invariant."""
    spec = ModelRegistry.get("mha_separate_qkv")
    raw = _export_in_memory(spec, opset=18)

    mutated, applied = ObfuscationRegistry.apply_onnx(raw, strategy_names=["onnx_reciprocal_mul_for_div"], seed=101)
    assert "onnx_reciprocal_mul_for_div" in applied

    opt = NeuronCompiler.compile(mutated, mode="function_proto")
    verifier = CrossVerifier(raw, opt)
    report = verifier.verify()
    assert report.all_passed, f"Reciprocal-Mul parity failed:\n{report.summary()}"


def test_tf2onnx_decomposed_rmsnorm_pow_normalization() -> None:
    """Verify that tf2onnx Pow(2) -> ReduceMean -> Sqrt -> Div RMSNorm is fused back to RMSNorm."""
    spec = ModelRegistry.get("norm_rmsnorm_l2")
    raw = _export_in_memory(spec, opset=18)

    mutated, applied = ObfuscationRegistry.apply_onnx(raw, strategy_names=["onnx_decompose_rmsnorm_pow"], seed=202)
    assert "onnx_decompose_rmsnorm_pow" in applied

    opt = NeuronCompiler.compile(mutated, mode="function_proto")
    opt_ops = {n.op_type for n in opt.graph.node}
    assert any("RMSNorm" in op for op in opt_ops), f"Decomposed RMSNorm was not fused back into RMSNorm! Ops: {opt_ops}"

    verifier = CrossVerifier(raw, opt)
    report = verifier.verify()
    assert report.all_passed, f"Decomposed RMSNorm parity failed:\n{report.summary()}"


def test_tf2onnx_decomposed_tanh_gelu_normalization() -> None:
    """Verify that tf2onnx Tanh GELU approximation formula is normalized into Gelu."""
    spec = ModelRegistry.get("activation_erf_gelu")
    raw = _export_in_memory(spec, opset=18)

    mutated, applied = ObfuscationRegistry.apply_onnx(raw, strategy_names=["onnx_decompose_tanh_gelu"], seed=303)
    assert "onnx_decompose_tanh_gelu" in applied

    opt = NeuronCompiler.compile(mutated, mode="function_proto")
    opt_ops = {n.op_type for n in opt.graph.node}
    assert any("Gelu" in op for op in opt_ops), f"Decomposed GELU was not fused back into Gelu! Ops: {opt_ops}"

    verifier = CrossVerifier(raw, opt, rtol=1e-3, atol=1e-3)
    report = verifier.verify()
    assert report.all_passed, f"Decomposed Tanh GELU parity failed:\n{report.summary()}"

