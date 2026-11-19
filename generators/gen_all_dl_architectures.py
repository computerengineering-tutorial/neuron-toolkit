#!/usr/bin/env python3
"""Master Deep Learning Architecture Generator & Multi-Opset ONNX Zoo Builder.

Orchestrates all generators across:
1. Recurrent Architectures (RNN, LSTM, GRU, QRNN, SRU)
2. Convolutional & Vision (ResNet, MobileNet, ConvNeXt, DenseNet, TCN, WaveNet, 3D Conv)
3. Vanilla MLP & Gated Feedforwards (DeepMLP, ResMLP, MLPMixer, gMLP, Highway, GLU variants)
4. Graph Neural Networks (GCN, GAT, GraphSAGE, GIN)
5. Advanced Transformers (Pre-LN, Post-LN, ViT, MLA, MoE, Performer, SwiGLU+RMS)
6. Comprehensive Attention Family (MHA, Causal, RoPE, GQA, MQA, Cross, Gated, ALiBi, T5)
7. State-Space Models (Mamba S6, S4, RWKV)
8. Diffusion & Generative (DiT, UNet, CrossAttention)
9. Hierarchical ViT & PointNet
10. Activation, Normalization & Linear Variants
11. Transformer Blocks & Plug-and-Play Addons

Generates models across ONNX Opsets: 14, 17, 18, 20.
Maintains model catalog index in .helper/models/model_catalog.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from generators.registry import ModelRegistry


def build_catalog(models_root: Path = Path(".helper/models")) -> dict:
    return ModelRegistry.build_catalog(models_root)


def main() -> None:
    print("==================================================================")
    print("      MASTER DEEP LEARNING ARCHITECTURE ZOO ORCHESTRATOR          ")
    print("==================================================================")

    # 1. Discover all architecture models and addons
    specs = ModelRegistry.discover()
    print(f"Discovered {len(specs)} architecture models across {len(ModelRegistry.get_categories())} categories.")

    # 2. Export all registered models across multi-opsets
    results = ModelRegistry.export_all()

    total_exported = sum(len(paths) for paths in results.values())
    print(f"\nCompleted export of {total_exported} ONNX artifacts across opsets.")

    # 3. Build & index catalog
    print("\nIndexing and cataloging all models...")
    catalog = ModelRegistry.build_catalog()

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
