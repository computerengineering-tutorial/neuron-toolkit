#!/usr/bin/env python3
"""Analyze an ONNX model for node minification opportunities and repeating subgraphs."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def analyze_onnx_model(model_path: str | Path) -> dict[str, any]:
    """Inspect model structure and detect minification candidates."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = onnx.load(str(model_path))
    graph = model.graph
    total_nodes = len(graph.node)

    op_counts = Counter(n.op_type for n in graph.node)
    
    # 1. Constant redundancy analysis
    const_nodes = [n for n in graph.node if n.op_type == "Constant"]
    unique_const_hashes = set()
    duplicate_const_count = 0
    const_values = {}
    for n in const_nodes:
        if n.attribute and n.attribute[0].name == "value":
            arr = numpy_helper.to_array(n.attribute[0].t)
            h = (arr.dtype.str, tuple(arr.shape), arr.tobytes())
            const_values[n.output[0]] = arr
            if h in unique_const_hashes:
                duplicate_const_count += 1
            else:
                unique_const_hashes.add(h)

    # 2. Identity arithmetic (Mul 1.0, Add 0.0)
    for init in graph.initializer:
        const_values[init.name] = numpy_helper.to_array(init)

    mul_identity_count = 0
    add_identity_count = 0
    for n in graph.node:
        if n.op_type == "Mul" and len(n.input) == 2:
            for inp in n.input:
                if inp in const_values:
                    v = const_values[inp]
                    if v.size == 1 and float(v.item()) == 1.0:
                        mul_identity_count += 1
                        break
        elif n.op_type == "Add" and len(n.input) == 2:
            for inp in n.input:
                if inp in const_values:
                    v = const_values[inp]
                    if v.size == 1 and float(v.item()) == 0.0:
                        add_identity_count += 1
                        break

    # 3. RMSNorm patterns: ReduceL2 -> Clip -> Expand -> Div -> Mul -> Mul
    rmsnorm_candidates = 0
    for n in graph.node:
        if n.op_type == "ReduceL2":
            clips = [c for c in graph.node if n.output[0] in c.input and c.op_type == "Clip"]
            if clips:
                divs = [c for c in graph.node if c.op_type == "Div"]
                if divs:
                    rmsnorm_candidates += 1

    # 4. GELU patterns: Div -> Erf -> Add -> Mul -> Mul
    erf_gelu_candidates = sum(1 for n in graph.node if n.op_type == "Erf")

    # 5. RoPE patterns: Cos / Sin pairs
    cos_nodes = [n for n in graph.node if n.op_type == "Cos"]
    sin_nodes = [n for n in graph.node if n.op_type == "Sin"]
    rope_candidates = min(len(cos_nodes), len(sin_nodes))

    # 6. Deterministic shape calculation chains (Shape -> Gather -> Unsqueeze -> Concat)
    shape_count = op_counts.get("Shape", 0)
    gather_count = op_counts.get("Gather", 0)
    unsqueeze_count = op_counts.get("Unsqueeze", 0)
    concat_count = op_counts.get("Concat", 0)

    # Estimate potential elimination
    estimated_eliminated = (
        duplicate_const_count
        + mul_identity_count
        + add_identity_count
        + (rmsnorm_candidates * 8)
        + (erf_gelu_candidates * 6)
        + (shape_count // 2)
    )
    reduction_pct = (estimated_eliminated / max(total_nodes, 1)) * 100

    report = {
        "model_path": str(model_path),
        "total_nodes": total_nodes,
        "inputs": [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in graph.input],
        "outputs": [(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]) for o in graph.output],
        "op_breakdown": dict(op_counts.most_common(15)),
        "duplicate_constants": duplicate_const_count,
        "identity_ops": mul_identity_count + add_identity_count,
        "rmsnorm_candidates": rmsnorm_candidates,
        "gelu_candidates": erf_gelu_candidates,
        "rope_candidates": rope_candidates,
        "shape_ops": {
            "Shape": shape_count,
            "Gather": gather_count,
            "Unsqueeze": unsqueeze_count,
            "Concat": concat_count,
        },
        "estimated_nodes_eliminated": estimated_eliminated,
        "estimated_reduction_pct": reduction_pct,
    }
    return report


def print_report(r: dict[str, any]) -> None:
    print("=" * 65)
    print(f"ONNX MODEL MINIFICATION AUDIT: {Path(r['model_path']).name}")
    print("=" * 65)
    print(f"Total Nodes:               {r['total_nodes']:,}")
    print(f"Inputs:                    {r['inputs']}")
    print(f"Outputs:                   {r['outputs']}")
    print("-" * 65)
    print("Key Optimization Opportunities:")
    print(f"  * Duplicate Constants:   {r['duplicate_constants']:,} nodes")
    print(f"  * Identity Arithmetic:   {r['identity_ops']:,} nodes (Mul 1.0 / Add 0.0)")
    print(f"  * RMSNorm Subgraphs:     {r['rmsnorm_candidates']:,} instances (~{r['rmsnorm_candidates']*8:,} nodes)")
    print(f"  * GELU Subgraphs:        {r['gelu_candidates']:,} instances (~{r['gelu_candidates']*6:,} nodes)")
    print(f"  * RoPE Subgraphs:        {r['rope_candidates']:,} Cos/Sin pairs")
    print(f"  * Dynamic Shape Chaining: {sum(r['shape_ops'].values()):,} ops ({r['shape_ops']})")
    print("-" * 65)
    print(f"Estimated Node Reduction:  ~{r['estimated_nodes_eliminated']:,} nodes ({r['estimated_reduction_pct']:.1f}%)")
    print("=" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ONNX model for node minification opportunities.")
    parser.add_argument("model_path", help="Path to input .onnx file")
    args = parser.parse_args()

    r = analyze_onnx_model(args.model_path)
    print_report(r)


if __name__ == "__main__":
    main()
