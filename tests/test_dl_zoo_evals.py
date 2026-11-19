"""Native Test Suite for Deep Learning Architecture Zoo across Multi-Opset ONNX models.

Tests optimization, custom op synthesis, and differential numerical parity
(ORT vs PyTorchRunner) on models generated across all architecture families:
- Recurrent (RNN, LSTM, GRU, QRNN, SRU)
- Convolutional & Vision (ResNet, MobileNet, ConvNeXt, DenseNet, TCN, WaveNet, Conv3D)
- Vanilla MLP & Gated Feedforwards (DeepMLP, ResMLP, MLPMixer, gMLP, Highway, GLU)
- Graph Neural Networks (GCN, GAT, GraphSAGE, GIN)
- Advanced Transformers & MoE (PreLN, PostLN, ViT, MLA, MoE, Performer, SwiGLU+RMS)
- Comprehensive Attention Family (MHA, Causal, RoPE, GQA, MQA, Cross, Gated)
"""

from __future__ import annotations

import sys
from pathlib import Path

import onnx
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from generators.registry import ModelRegistry
from neuron_toolkit.compiler.pipeline import NeuronCompiler
from neuron_toolkit.runtime.cross_verifier import CrossVerifier

ModelRegistry.discover(_REPO_ROOT / "generators")
_ADDON_SPECS = ModelRegistry.get_by_category("addons")

MODELS_DIR = Path(".helper/models")


def _run_verify_test(
    model_path: Path, min_cosine: float = 0.9999, atol: float = 1e-4
) -> None:
    assert model_path.exists(), f"Model {model_path} does not exist"
    orig_m = onnx.load(str(model_path), load_external_data=True)
    opt_m = NeuronCompiler.compile(orig_m, mode="function_proto")

    verifier = CrossVerifier(orig_m, opt_m, atol=atol, min_cosine_similarity=min_cosine)
    report = verifier.verify()
    assert report.all_passed, (
        f"Verification failed for {model_path.name}:\n{report.summary()}"
    )


# =============================================================================
# 1. Recurrent Family Tests
# =============================================================================
@pytest.mark.parametrize(
    "model_name",
    [
        "vanilla_rnn_single_opset18.onnx",
        "vanilla_rnn_multi_opset18.onnx",
        "lstm_uni_opset18.onnx",
        "lstm_bi_opset18.onnx",
        "gru_uni_opset18.onnx",
        "gru_bi_opset18.onnx",
        "custom_cell_lstm_opset18.onnx",
        "custom_cell_gru_opset18.onnx",
        "qrnn_opset18.onnx",
        "sru_cell_opset18.onnx",
    ],
)
def test_recurrent_zoo_opset18(model_name: str) -> None:
    p = MODELS_DIR / "recurrent" / model_name
    if not p.exists():
        pytest.skip(f"Model {model_name} not yet exported")
    _run_verify_test(p)


# =============================================================================
# 2. Convolutional & Computer Vision Tests
# =============================================================================
@pytest.mark.parametrize(
    "model_name",
    [
        "conv2d_bn_relu_opset18.onnx",
        "resnet_basic_opset18.onnx",
        "resnet_bottleneck_opset18.onnx",
        "mobilenetv2_invres_opset18.onnx",
        "mobilenetv3_block_opset18.onnx",
        "convnext_block_opset18.onnx",
        "densenet_block_opset18.onnx",
        "resnext_block_opset18.onnx",
        "tcn_dilated_opset18.onnx",
        "conv3d_spatiotemporal_opset18.onnx",
    ],
)
def test_cnn_vision_zoo_opset18(model_name: str) -> None:
    p = MODELS_DIR / "cnn_vision" / model_name
    if not p.exists():
        pytest.skip(f"Model {model_name} not yet exported")
    _run_verify_test(p)


# =============================================================================
# 3. Vanilla MLP & Feedforward Tests
# =============================================================================
@pytest.mark.parametrize(
    "model_name",
    [
        "deep_mlp_norm_act_opset18.onnx",
        "resmlp_block_opset18.onnx",
        "mlpmixer_block_opset18.onnx",
        "gmlp_block_opset18.onnx",
        "highway_block_opset18.onnx",
        "swiglu_mlp_block_opset18.onnx",
        "geglu_mlp_block_opset18.onnx",
        "reglu_mlp_block_opset18.onnx",
    ],
)
def test_mlp_vanilla_zoo_opset18(model_name: str) -> None:
    p = MODELS_DIR / "mlp_vanilla" / model_name
    if not p.exists():
        pytest.skip(f"Model {model_name} not yet exported")
    _run_verify_test(p)


# =============================================================================
# 4. Graph Neural Network Tests
# =============================================================================
@pytest.mark.parametrize(
    "model_name",
    [
        "gcn_layer_opset18.onnx",
        "gat_layer_opset18.onnx",
        "graphsage_layer_opset18.onnx",
        "gin_layer_opset18.onnx",
    ],
)
def test_gnn_zoo_opset18(model_name: str) -> None:
    p = MODELS_DIR / "gnn" / model_name
    if not p.exists():
        pytest.skip(f"Model {model_name} not yet exported")
    _run_verify_test(p)


# =============================================================================
# 5. Advanced Transformer & MoE Tests
# =============================================================================
@pytest.mark.parametrize(
    "model_name",
    [
        "transformer_pre_ln_opset18.onnx",
        "transformer_post_ln_opset18.onnx",
        "vit_patch_block_opset18.onnx",
        "transformer_mla_opset18.onnx",
        "transformer_moe_opset18.onnx",
        "transformer_performer_opset18.onnx",
        "transformer_swiglu_rms_opset18.onnx",
    ],
)
def test_transformer_advanced_zoo_opset18(model_name: str) -> None:
    p = MODELS_DIR / "transformer_advanced" / model_name
    if not p.exists():
        pytest.skip(f"Model {model_name} not yet exported")
    _run_verify_test(p)


# =============================================================================
# 6. Multi-Opset Cross-Consistency Test
# =============================================================================
@pytest.mark.parametrize("opset", [14, 17, 18, 20])
def test_multi_opset_consistency(opset: int) -> None:
    p = MODELS_DIR / "attention_variants" / f"mha_separate_qkv_opset{opset}.onnx"
    if p.exists():
        _run_verify_test(p)


# =============================================================================
# 7. State-Space Models (SSM / Mamba / RWKV) Tests
# =============================================================================
@pytest.mark.parametrize(
    "model_name",
    [
        "mamba_selective_scan_opset18.onnx",
        "s4_state_space_opset18.onnx",
        "rwkv_time_mix_opset18.onnx",
        "rwkv_channel_mix_opset18.onnx",
    ],
)
def test_ssm_mamba_zoo_opset18(model_name: str) -> None:
    p = MODELS_DIR / "ssm_mamba" / model_name
    if not p.exists():
        pytest.skip(f"Model {model_name} not yet exported")
    _run_verify_test(p)


# =============================================================================
# 8. Diffusion & Generative Models (DiT / UNet) Tests
# =============================================================================
@pytest.mark.parametrize(
    "model_name",
    [
        "dit_adaln_zero_opset18.onnx",
        "unet_res_timestep_opset18.onnx",
        "cross_attention_block_opset18.onnx",
    ],
)
def test_diffusion_dit_zoo_opset18(model_name: str) -> None:
    p = MODELS_DIR / "diffusion_dit" / model_name
    if not p.exists():
        pytest.skip(f"Model {model_name} not yet exported")
    _run_verify_test(p)


# =============================================================================
# 9. Hierarchical Vision & Obfuscated Stress Tests
# =============================================================================
@pytest.mark.parametrize(
    "model_name",
    [
        "swin_window_attention_opset18.onnx",
        "xcit_cross_covariance_opset18.onnx",
        "pointnet_global_block_opset18.onnx",
    ],
)
def test_hierarchical_vit_zoo_opset18(model_name: str) -> None:
    p = MODELS_DIR / "hierarchical_vit" / model_name
    if not p.exists():
        pytest.skip(f"Model {model_name} not yet exported")
    _run_verify_test(p)


@pytest.mark.parametrize("opset", [14, 18, 20])
def test_obfuscated_stress_models(opset: int) -> None:
    pure_p = (
        MODELS_DIR / "obfuscated_stress" / f"pure_onnx_decomposed_ln_opset{opset}.onnx"
    )
    pt_p = (
        MODELS_DIR
        / "obfuscated_stress"
        / f"pytorch_obfuscated_complex_opset{opset}.onnx"
    )
    if pure_p.exists():
        _run_verify_test(pure_p)
    if pt_p.exists():
        _run_verify_test(pt_p)


# =============================================================================
# 10. Dynamic Addon & Discovered Models Verification
# =============================================================================


@pytest.mark.parametrize("spec", _ADDON_SPECS, ids=lambda s: s.name)
def test_addon_zoo_models(spec) -> None:
    """Dynamic test exercising all registered addon models."""
    p = spec.get_model_path(opset=18)
    if not p.exists():
        pytest.skip(f"Addon model {spec.name} not yet exported")
    _run_verify_test(p, atol=spec.atol)

