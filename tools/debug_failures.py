#!/usr/bin/env python3
"""debug_failures.py - Reproduce the 6 failing zoo eval tests."""

import sys
from pathlib import Path
import numpy as np
import onnx

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neuron_toolkit.compiler.pipeline import NeuronCompiler
from neuron_toolkit.runtime.cross_verifier import CrossVerifier

MODELS_DIR = Path(".helper/models")

def test_model(cat, name):
    p = MODELS_DIR / cat / name
    print(f"\n=======================================================")
    print(f"Testing {cat}/{name} ({p})")
    print(f"=======================================================")
    if not p.exists():
        print("Model file does not exist!")
        return
    try:
        orig_m = onnx.load(str(p), load_external_data=True)
        opt_m = NeuronCompiler.compile(orig_m, mode="function_proto")
        verifier = CrossVerifier(orig_m, opt_m, atol=1e-4, min_cosine_similarity=0.9999)
        report = verifier.verify()
        print(report.summary())
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model("recurrent", "sru_cell_opset18.onnx")
    test_model("mlp_vanilla", "geglu_mlp_block_opset18.onnx")
    test_model("mlp_vanilla", "reglu_mlp_block_opset18.onnx")
    test_model("transformer_advanced", "vit_patch_block_opset18.onnx")
    test_model("ssm_mamba", "mamba_selective_scan_opset18.onnx")
    test_model("hierarchical_vit", "xcit_cross_covariance_opset18.onnx")
