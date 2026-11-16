"""Automated Data-Driven Test Suite for Declarative JSON Rules.

Discovers and executes all JSON test descriptors in tests/rules_json_tests/
against their generated synthetic models in .helper/models/, verifying
numerical parity and pattern rewrite accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np
import onnx
import torch

from neuron_toolkit.compiler.pipeline import OptimizationPipeline
from neuron_toolkit.compiler.base import CompilerContext
from neuron_toolkit.runtime.cross_verifier import CrossVerifier


DESCRIPTOR_DIR = Path(__file__).parent / "rules_json_tests"
TEST_FILES = sorted(DESCRIPTOR_DIR.glob("*_test.json"))


@pytest.mark.parametrize("descriptor_path", TEST_FILES, ids=lambda p: p.stem)
def test_json_rule_descriptor(descriptor_path: Path):
    """Run automated verification for a single rule JSON test descriptor."""
    with open(descriptor_path, encoding="utf-8") as f:
        spec = json.load(f)

    model_path = Path(spec["model_path"])
    assert model_path.exists(), f"Model file {model_path} does not exist"

    raw_model = onnx.load(str(model_path))
    pipeline = OptimizationPipeline(mode="function_proto")
    context = CompilerContext(mode="function_proto")

    opt_model = pipeline.run(raw_model, context)

    # 1. Check expected target node presence
    expected_node = spec.get("expected_target_node")
    if expected_node:
        all_op_types = {n.op_type for n in opt_model.graph.node}
        assert expected_node in all_op_types, (
            f"Expected node '{expected_node}' not found in optimized graph ops: {all_op_types}"
        )

    # 2. Cross-verify numerical equivalence
    input_shapes = spec.get("input_shapes", {})
    inputs: dict[str, np.ndarray] = {}
    for name, shape in input_shapes.items():
        inputs[name] = np.random.randn(*shape).astype(np.float32)

    verifier = CrossVerifier(
        raw_model,
        opt_model,
        rtol=spec.get("rtol", 1e-4),
        atol=spec.get("atol", 1e-4),
    )
    report = verifier.verify(inputs if inputs else None)

    for out_name, metric in report.metrics.items():
        assert metric.cosine_similarity >= 0.99999, (
            f"Output '{out_name}' cosine similarity {metric.cosine_similarity} < 0.99999"
        )
        assert metric.mse < 1e-5, f"Output '{out_name}' MSE {metric.mse} >= 1e-5"
