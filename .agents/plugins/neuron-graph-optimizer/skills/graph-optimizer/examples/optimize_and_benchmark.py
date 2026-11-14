"""Example: Load, optimize, minify, export, and cross-verify an ONNX model."""

import time
import numpy as np
from neuron_toolkit.graph import NeuronGraph


def run_pipeline(model_path: str):
    print(f"Loading {model_path} via NeuronGraph...")
    g = NeuronGraph.load(model_path)
    initial_nodes = len(g.nodes)
    print(f"Initial node count: {initial_nodes}")

    t0 = time.perf_counter()
    # Apply all JSON rules, CSE, and identity pruning
    opt_g = g.apply_json_rules(mode="function_proto")
    opt_nodes = len(opt_g.nodes)
    elapsed = time.perf_counter() - t0

    eliminated = initial_nodes - opt_nodes
    pct = (eliminated / initial_nodes) * 100
    print(f"Optimized in {elapsed*1000:.1f}ms: {initial_nodes} -> {opt_nodes} nodes (-{eliminated} nodes, {pct:.1f}% reduction)")

    # Cross-verify outputs with automatic random inputs
    print("Running differential verification against ONNX Runtime...")
    report = g.cross_verify(opt_g)
    print("\n" + report.summary())

    if report.all_passed:
        out_path = f"optimized_{model_path.split('/')[-1]}"
        opt_g.export(out_path)
        print(f"Exported clean, Netron-ready model to: {out_path}")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else ".temp/diffrent_models/model2.onnx"
    run_pipeline(path)
