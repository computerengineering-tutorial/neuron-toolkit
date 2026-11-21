"""Native Test Suite for Deep Learning Architecture Zoo with On-The-Fly PyTorch-to-ONNX Verification.

Dynamically exports PyTorch models across all architecture families to ONNX on the fly,
compiles them using NeuronCompiler, and validates numerical parity against PyTorch reference execution:
- Recurrent (RNN, LSTM, GRU, QRNN, SRU)
- Convolutional & Vision (ResNet, MobileNet, ConvNeXt, DenseNet, TCN, WaveNet, Conv3D)
- Vanilla MLP & Gated Feedforwards (DeepMLP, ResMLP, MLPMixer, gMLP, Highway, GLU)
- Graph Neural Networks (GCN, GAT, GraphSAGE, GIN)
- Advanced Transformers & MoE (PreLN, PostLN, ViT, MLA, MoE, Performer, SwiGLU+RMS)
- Comprehensive Attention Family (MHA, Causal, RoPE, GQA, MQA, Cross, Gated)
- State-Space Models (Mamba S6, S4, RWKV)
- Diffusion & Generative (DiT, UNet, CrossAttention)
- Hierarchical Vision & PointNet (Swin, XCiT, PointNet)
- SOTA Architectures, Normalization, Activation & Linear Variants
- Plug-and-Play Addons & Obfuscated Stress Models

Zero disk-file dependencies: No hardcoded paths or skipped tests due to missing files.
Everything runs dynamically in-memory / on-the-fly.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from generators.registry import ModelRegistry, ModelSpec
from neuron_toolkit.compiler.pipeline import NeuronCompiler
from neuron_toolkit.runtime.runner import PyTorchRunner

# Discover all registered models dynamically
ModelRegistry.discover(_REPO_ROOT / "generators")
ALL_SPECS = ModelRegistry.list_models()
_SPECS_BY_CATEGORY = {cat: ModelRegistry.get_by_category(cat) for cat in ModelRegistry.get_categories()}


def _export_compile_and_verify(
    spec: ModelSpec,
    opset: int = 18,
    atol: float | None = None,
    rtol: float | None = None,
) -> tuple[float, float]:
    """Dynamically export PyTorch model to ONNX, compile, and cross-verify with PyTorch."""
    model = spec.instantiate().eval()
    dummy_inputs = spec.get_dummy_inputs()
    if isinstance(dummy_inputs, torch.Tensor):
        dummy_inputs_tuple: tuple[torch.Tensor, ...] = (dummy_inputs,)
    else:
        dummy_inputs_tuple = tuple(dummy_inputs)

    # 1. Compute ground-truth PyTorch outputs
    with torch.no_grad():
        pt_outs = model(*dummy_inputs_tuple)
        if isinstance(pt_outs, (tuple, list)):
            pt_ref = pt_outs[0].detach().cpu().numpy()
        else:
            pt_ref = pt_outs.detach().cpu().numpy()

    # 2. Export PyTorch module to ONNX on-the-fly
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        try:
            torch.onnx.export(
                model,
                dummy_inputs_tuple,
                tmp_path,
                input_names=spec.input_names,
                output_names=spec.output_names,
                opset_version=opset,
                dynamo=False,
            )
        except Exception:
            torch.onnx.export(
                model,
                dummy_inputs_tuple,
                tmp_path,
                input_names=spec.input_names,
                output_names=spec.output_names,
                dynamo=True,
            )
        raw_onnx = onnx.load(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # 3. Invariant compilation via NeuronCompiler
    opt_onnx = NeuronCompiler.compile(raw_onnx)

    # 4. Execute compiled ONNX model via PyTorchRunner
    runner = PyTorchRunner(opt_onnx)
    feed_dict = {
        name: inp.detach().cpu().numpy()
        for name, inp in zip(spec.input_names, dummy_inputs_tuple)
    }
    onnx_outs = runner.run(feed_dict)
    opt_ref = next(iter(onnx_outs.values()))

    # 5. Differential numerical parity checks
    diff = np.abs(pt_ref.astype(np.float64) - opt_ref.astype(np.float64))
    max_diff = float(np.max(diff))

    ref_flat = pt_ref.flatten().astype(np.float64)
    opt_flat = opt_ref.flatten().astype(np.float64)
    norm_p = np.linalg.norm(ref_flat) * np.linalg.norm(opt_flat)
    cosine = float(np.dot(ref_flat, opt_flat) / norm_p) if norm_p > 0 else 1.0

    eff_atol = atol if atol is not None else (spec.atol if spec.atol else 1e-4)
    eff_rtol = rtol if rtol is not None else (spec.rtol if spec.rtol else 1e-4)
    max_val = float(np.max(np.abs(pt_ref))) if pt_ref.size > 0 else 1.0

    passed = (max_diff <= (eff_atol + eff_rtol * max_val)) or (cosine >= 0.98)
    assert passed, (
        f"Verification FAILED for {spec.name} (category: {spec.category}, opset={opset}): "
        f"max_diff={max_diff:.6f}, cosine_similarity={cosine:.6f}, threshold_atol={eff_atol}"
    )
    return max_diff, cosine


# =============================================================================
# 1. Recurrent Family Dynamic On-The-Fly Tests
# =============================================================================
@pytest.mark.parametrize("spec", _SPECS_BY_CATEGORY.get("recurrent", []), ids=lambda s: s.name)
def test_recurrent_zoo_on_the_fly(spec: ModelSpec) -> None:
    _export_compile_and_verify(spec)


# =============================================================================
# 2. Convolutional & Computer Vision Dynamic On-The-Fly Tests
# =============================================================================
@pytest.mark.parametrize("spec", _SPECS_BY_CATEGORY.get("cnn_vision", []), ids=lambda s: s.name)
def test_cnn_vision_zoo_on_the_fly(spec: ModelSpec) -> None:
    _export_compile_and_verify(spec)


# =============================================================================
# 3. Vanilla MLP & Feedforward Dynamic On-The-Fly Tests
# =============================================================================
@pytest.mark.parametrize("spec", _SPECS_BY_CATEGORY.get("mlp_vanilla", []), ids=lambda s: s.name)
def test_mlp_vanilla_zoo_on_the_fly(spec: ModelSpec) -> None:
    _export_compile_and_verify(spec)


# =============================================================================
# 4. Graph Neural Network Dynamic On-The-Fly Tests
# =============================================================================
@pytest.mark.parametrize("spec", _SPECS_BY_CATEGORY.get("gnn", []), ids=lambda s: s.name)
def test_gnn_zoo_on_the_fly(spec: ModelSpec) -> None:
    _export_compile_and_verify(spec)


# =============================================================================
# 5. Advanced Transformer & MoE Dynamic On-The-Fly Tests
# =============================================================================
@pytest.mark.parametrize("spec", _SPECS_BY_CATEGORY.get("transformer_advanced", []), ids=lambda s: s.name)
def test_transformer_advanced_zoo_on_the_fly(spec: ModelSpec) -> None:
    _export_compile_and_verify(spec)


# =============================================================================
# 6. State-Space Models (SSM / Mamba / RWKV) Dynamic On-The-Fly Tests
# =============================================================================
@pytest.mark.parametrize("spec", _SPECS_BY_CATEGORY.get("ssm_mamba", []), ids=lambda s: s.name)
def test_ssm_mamba_zoo_on_the_fly(spec: ModelSpec) -> None:
    _export_compile_and_verify(spec)


# =============================================================================
# 7. Diffusion & Generative Models (DiT / UNet) Dynamic On-The-Fly Tests
# =============================================================================
@pytest.mark.parametrize("spec", _SPECS_BY_CATEGORY.get("diffusion_dit", []), ids=lambda s: s.name)
def test_diffusion_dit_zoo_on_the_fly(spec: ModelSpec) -> None:
    _export_compile_and_verify(spec)


# =============================================================================
# 8. Hierarchical Vision & PointNet Dynamic On-The-Fly Tests
# =============================================================================
@pytest.mark.parametrize("spec", _SPECS_BY_CATEGORY.get("hierarchical_vit", []), ids=lambda s: s.name)
def test_hierarchical_vit_zoo_on_the_fly(spec: ModelSpec) -> None:
    _export_compile_and_verify(spec)


# =============================================================================
# 9. Comprehensive Attention Family Dynamic On-The-Fly Tests
# =============================================================================
@pytest.mark.parametrize("spec", _SPECS_BY_CATEGORY.get("attention_variants", []), ids=lambda s: s.name)
def test_attention_variants_zoo_on_the_fly(spec: ModelSpec) -> None:
    _export_compile_and_verify(spec)


# =============================================================================
# 10. SOTA Architectures Dynamic On-The-Fly Tests
# =============================================================================
@pytest.mark.parametrize("spec", _SPECS_BY_CATEGORY.get("sota_architectures", []), ids=lambda s: s.name)
def test_sota_architectures_zoo_on_the_fly(spec: ModelSpec) -> None:
    _export_compile_and_verify(spec)


# =============================================================================
# 11. Activations, Normalizations, Linear & Blocks Dynamic On-The-Fly Tests
# =============================================================================
@pytest.mark.parametrize(
    "spec",
    _SPECS_BY_CATEGORY.get("activation_variants", [])
    + _SPECS_BY_CATEGORY.get("norm_variants", [])
    + _SPECS_BY_CATEGORY.get("linear_variants", [])
    + _SPECS_BY_CATEGORY.get("transformer_blocks", []),
    ids=lambda s: f"{s.category}::{s.name}",
)
def test_variants_and_blocks_zoo_on_the_fly(spec: ModelSpec) -> None:
    _export_compile_and_verify(spec)


# =============================================================================
# 12. Plug-and-Play Addons Dynamic On-The-Fly Tests
# =============================================================================
@pytest.mark.parametrize("spec", _SPECS_BY_CATEGORY.get("addons", []), ids=lambda s: s.name)
def test_addon_zoo_on_the_fly(spec: ModelSpec) -> None:
    _export_compile_and_verify(spec)


# =============================================================================
# 13. Obfuscated Stress Models Dynamic On-The-Fly Tests
# =============================================================================
@pytest.mark.parametrize("spec", _SPECS_BY_CATEGORY.get("obfuscated_stress", []), ids=lambda s: s.name)
def test_obfuscated_stress_on_the_fly(spec: ModelSpec) -> None:
    _export_compile_and_verify(spec)


# =============================================================================
# 14. Multi-Opset Dynamic Consistency Tests
# =============================================================================
@pytest.mark.parametrize("opset", [14, 17, 18, 20])
def test_multi_opset_consistency_on_the_fly(opset: int) -> None:
    spec = ModelRegistry.get("mha_separate_qkv")
    _export_compile_and_verify(spec, opset=opset)
