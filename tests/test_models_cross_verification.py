"""Cross-by-cross differential verification tests on real models from .temp/diffrent_models/."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest

from neuron_toolkit.graph import NeuronGraph
from neuron_toolkit.rules.engine import JSONRuleEngine
from neuron_toolkit.runtime.cross_verifier import CrossVerifier

MODELS_DIR = Path(".temp/diffrent_models")


@pytest.mark.skipif(
    not (MODELS_DIR / "model2.onnx").exists(), reason="model2.onnx not found"
)
def test_cross_verify_model2_onnx():
    """Verify model2.onnx: Cast-Conv folding and ERF-GELU fusion with strict numerical parity."""
    model_path = str(MODELS_DIR / "model2.onnx")
    orig_model = onnx.load(model_path)
    orig_node_count = len(orig_model.graph.node)

    # 1. Apply JSONRuleEngine
    engine = JSONRuleEngine(mode="function_proto")
    opt_model = engine.apply(orig_model)

    opt_node_count = len(opt_model.graph.node)
    eliminated = orig_node_count - opt_node_count

    # Verify dramatic node reduction (from 490 down to 112 nodes: 378 nodes eliminated!)
    assert eliminated >= 350, f"Expected >= 350 nodes eliminated, got {eliminated}"

    # 2. Run CrossVerifier against ONNX Runtime
    verifier = CrossVerifier(orig_model, opt_model, rtol=1e-4, atol=1e-4)
    # Generate realistic inputs: x is [1, 4, 400, 32], beat is [1, 32, 2]
    np.random.seed(42)
    inputs = {
        "x": np.random.randn(1, 4, 400, 32).astype(np.float32),
        "beat": np.random.randn(1, 32, 2).astype(np.float32),
    }

    report = verifier.verify(inputs=inputs)
    print("\n" + report.summary())

    assert report.all_passed, f"Cross-verification failed: {report.summary()}"
    assert report.metrics["chord_logp"].cosine_similarity >= 0.99999
    assert report.metrics["bass_logp"].cosine_similarity >= 0.99999
    assert report.metrics["chord_logp"].mse < 1e-5
    assert report.metrics["bass_logp"].mse < 1e-5


@pytest.mark.skipif(
    not (MODELS_DIR / "model4.onnx").exists(), reason="model4.onnx not found"
)
def test_cross_verify_model4_onnx():
    """Verify model4.onnx: Index routing and slicing with numerical parity."""
    model_path = str(MODELS_DIR / "model4.onnx")
    orig_model = onnx.load(model_path)
    engine = JSONRuleEngine(mode="function_proto")
    opt_model = engine.apply(orig_model)

    verifier = CrossVerifier(orig_model, opt_model, rtol=1e-3, atol=1e-3)
    report = verifier.verify()
    assert report.all_passed
    assert report.metrics["perc_prob"].cosine_similarity >= 0.9999


@pytest.mark.skipif(
    not (MODELS_DIR / "model3.onnx").exists(), reason="model3.onnx not found"
)
def test_cross_verify_model3_onnx():
    """Verify model3.onnx on PyTorchRunner against ONNX Runtime."""
    model_path = str(MODELS_DIR / "model3.onnx")
    orig_model = onnx.load(model_path)

    # In model3, input shape is [0, 4, 352, 720] -> [1, 4, 352, 720]
    verifier = CrossVerifier(orig_model, orig_model, rtol=1e-4, atol=1e-4)
    inputs = {
        "input": np.random.randn(1, 4, 352, 720).astype(np.float32),
    }
    report = verifier.verify(inputs=inputs)
    assert report.all_passed
    assert report.metrics["onsets"].cosine_similarity >= 0.99999


def test_neurongraph_high_level_api_model2():
    """Verify the high-level NeuronGraph methods: apply_json_rules, run_pytorch, cross_verify."""
    model_path = str(MODELS_DIR / "model2.onnx")
    if not Path(model_path).exists():
        pytest.skip("model2.onnx not found")

    g = NeuronGraph.load(model_path)
    opt_g = g.apply_json_rules(mode="function_proto")

    assert len(opt_g.nodes) < len(g.nodes)

    np.random.seed(42)
    inputs = {
        "x": np.random.randn(1, 4, 400, 32).astype(np.float32),
        "beat": np.random.randn(1, 32, 2).astype(np.float32),
    }

    report = g.cross_verify(opt_g, inputs=inputs)
    assert report.all_passed


@pytest.mark.skipif(
    not (MODELS_DIR / "model1.onnx").exists(), reason="model1.onnx not found"
)
def test_cross_verify_model1_onnx():
    """Verify model1.onnx: RMSNorm, GELU, CSE, Identity elimination on large axial conformer."""
    model_path = str(MODELS_DIR / "model1.onnx")
    orig_model = onnx.load(model_path)
    orig_node_count = len(orig_model.graph.node)

    engine = JSONRuleEngine(mode="function_proto")
    opt_model = engine.apply(orig_model)

    opt_node_count = len(opt_model.graph.node)
    eliminated = orig_node_count - opt_node_count

    # Verify massive minification (from 4,740 down to <= 2,360: > 2,300 nodes eliminated!)
    assert eliminated >= 2000, f"Expected >= 2000 nodes eliminated, got {eliminated}"

    # Differential cross-verification against ONNX Runtime
    verifier = CrossVerifier(orig_model, opt_model, rtol=1e-4, atol=1e-4)
    np.random.seed(42)
    inputs = {
        "logmel": np.random.randn(1, 100, 128).astype(np.float32),
    }

    report = verifier.verify(inputs=inputs)
    print("\n" + report.summary())

    assert report.all_passed, f"Cross-verification failed: {report.summary()}"
    assert report.metrics["beat"].cosine_similarity >= 0.99999
    assert report.metrics["downbeat"].cosine_similarity >= 0.99999
    assert report.metrics["beat"].mse < 1e-5
    assert report.metrics["downbeat"].mse < 1e-5
