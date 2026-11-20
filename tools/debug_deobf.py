#!/usr/bin/env python3
"""Debug test_on_the_fly_deobfuscation_invariance failure."""

import sys
from pathlib import Path
import numpy as np
import onnx

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generators.registry import ModelRegistry
from generators.obfuscation import ObfuscationRegistry
from neuron_toolkit.compiler.pipeline import NeuronCompiler
from neuron_toolkit.runtime.cross_verifier import CrossVerifier
from tests.test_on_the_fly_invariants import _export_in_memory

def debug(model_name):
    print(f"\n=== Debugging {model_name} ===")
    spec = ModelRegistry.get(model_name)
    raw_clean = _export_in_memory(spec, opset=18)
    raw_obf, applied = ObfuscationRegistry.apply_onnx(raw_clean, seed=123)
    print("Applied obfuscation strategies:", applied)
    
    opt_clean = NeuronCompiler.compile(raw_clean, mode="function_proto")
    opt_obf = NeuronCompiler.compile(raw_obf, mode="function_proto")
    
    print("opt_clean ops:", [n.op_type for n in opt_clean.graph.node])
    print("opt_obf ops:", [n.op_type for n in opt_obf.graph.node])
    
    verifier = CrossVerifier(opt_clean, opt_obf, rtol=1e-4, atol=1e-4)
    report = verifier.verify()
    print(report.summary())

if __name__ == "__main__":
    debug("norm_rmsnorm_l2")
    debug("transformer_ffn_gelu")
