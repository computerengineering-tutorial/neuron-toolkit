#!/usr/bin/env python3
"""Master Deep Learning Architecture Generator & Multi-Opset ONNX Zoo Builder.

Orchestrates all generators across:
1. Recurrent Architectures (RNN, LSTM, GRU, QRNN, SRU)
2. Convolutional & Vision (ResNet, MobileNet, ConvNeXt, DenseNet, TCN, WaveNet, 3D Conv)
3. Vanilla MLP & Gated Feedforwards (DeepMLP, ResMLP, MLPMixer, gMLP, Highway, GLU variants)
4. Graph Neural Networks (GCN, GAT, GraphSAGE, GIN)
5. Advanced Transformers (Pre-LN, Post-LN, ViT, MLA, MoE, Performer, SwiGLU+RMS)
6. Comprehensive Attention Family (MHA, Causal, RoPE, GQA, MQA, Cross, Gated Sigmoid/SiLU/GELU, ALiBi, T5)

Generates models across ONNX Opsets: 14, 17, 18, 20.
Maintains model catalog index in .helper/models/model_catalog.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import onnx

GENERATORS = [
    (".helper/generators/gen_recurrent_zoo.py", "Recurrent RNN/LSTM/GRU Family"),
    (
        ".helper/generators/gen_vision_cnn_zoo.py",
        "Convolutional & Computer Vision Family",
    ),
    (
        ".helper/generators/gen_vanilla_mlp_zoo.py",
        "Vanilla MLP & Gated Feedforward Family",
    ),
    (".helper/generators/gen_gnn_zoo.py", "Graph Neural Network (GNN) Family"),
    (
        ".helper/generators/gen_transformer_advanced_zoo.py",
        "Advanced Transformer & MoE Family",
    ),
    (
        ".helper/generators/gen_comprehensive_attentions.py",
        "Comprehensive Attention Family",
    ),
]


def build_catalog() -> dict:
    models_root = Path(".helper/models")
    all_onnx = sorted(models_root.glob("**/*.onnx"))
    catalog = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_models": len(all_onnx),
        "models": [],
    }

    by_category: dict[str, int] = {}
    by_opset: dict[int, int] = {}

    for p in all_onnx:
        rel = p.relative_to(models_root)
        cat = rel.parts[0] if len(rel.parts) > 1 else "root"
        by_category[cat] = by_category.get(cat, 0) + 1

        try:
            m = onnx.load(str(p), load_external_data=False)
            opset = m.opset_import[0].version if m.opset_import else -1
            by_opset[opset] = by_opset.get(opset, 0) + 1
            node_count = len(m.graph.node)
            inputs = [
                {
                    "name": inp.name,
                    "shape": [
                        d.dim_value if d.dim_value > 0 else -1
                        for d in inp.type.tensor_type.shape.dim
                    ],
                }
                for inp in m.graph.input
            ]
            outputs = [
                {
                    "name": out.name,
                    "shape": [
                        d.dim_value if d.dim_value > 0 else -1
                        for d in out.type.tensor_type.shape.dim
                    ],
                }
                for out in m.graph.output
            ]
        except Exception:
            opset = -1
            node_count = -1
            inputs, outputs = [], []

        catalog["models"].append(
            {
                "path": str(p),
                "name": p.name,
                "category": cat,
                "opset": opset,
                "nodes": node_count,
                "size_kb": round(p.stat().st_size / 1024.0, 2),
                "inputs": inputs,
                "outputs": outputs,
            }
        )

    catalog["breakdown_by_category"] = by_category
    catalog["breakdown_by_opset"] = by_opset

    catalog_path = models_root / "model_catalog.json"
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    return catalog


def main() -> None:
    print("==================================================================")
    print("      MASTER DEEP LEARNING ARCHITECTURE ZOO ORCHESTRATOR          ")
    print("==================================================================")

    for script_path, desc in GENERATORS:
        print(f"\n>>> Running Generator: {desc} ({script_path})")
        res = subprocess.run(
            [sys.executable, script_path], capture_output=True, text=True
        )
        if res.returncode != 0:
            print(f"Error running {script_path}:\n{res.stderr}")
        else:
            # Print last few lines of output
            lines = [
                l for l in res.stdout.strip().split("\n") if l.startswith("  [OK]")
            ]
            print(f"  Successfully exported {len(lines)} models.")

    print("\nIndexing and cataloging all models...")
    catalog = build_catalog()
    print("==================================================================")
    print("Catalog generated at .helper/models/model_catalog.json")
    print(f"Total Exported Models: {catalog['total_models']}")
    print("Breakdown by Category:")
    for cat, count in catalog["breakdown_by_category"].items():
        print(f"  - {cat:<25}: {count} models")
    print("Breakdown by Opset Version:")
    for opset, count in sorted(catalog["breakdown_by_opset"].items()):
        print(f"  - Opset {opset:<5}: {count} models")
    print("==================================================================")


if __name__ == "__main__":
    main()
