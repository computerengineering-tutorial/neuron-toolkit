# Neuron Toolkit

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![ONNX](https://img.shields.io/badge/ONNX-Supported-green.svg)](https://onnx.ai/)
[![TFLite](https://img.shields.io/badge/TFLite-Supported-orange.svg)](https://www.tensorflow.org/lite)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)]()

> **A Unified ONNX and TFLite Parsing, Querying, Pattern-Matching, and Transformation Toolkit.**

`neuron-toolkit` provides a high-level, framework-agnostic Python interface for deep learning model surgery, pattern matching, graph querying, optimization passes, and numerical verification. It bridges the gap between disparate model formats (ONNX, TFLite, and Netron-compatible JSON), empowering compiler engineers, optimization researchers, and ML infrastructure developers to inspect, optimize, and manipulate neural computational graphs effortlessly.

---

## 🚀 Key Features

- 🔍 **Fluent Query Engine**: Chainable, SQL-like queries to traverse, filter, and inspect computational nodes (`graph.query().op("Conv").all()`).
- 🧩 **Pattern Matching DSL**: Declarative subgraph pattern matching with operator overloads (`x * Pattern.op("Sigmoid", x)`), captures, wildcards, and commutative matching.
- ⚡ **Optimization Passes (`PassManager`)**: Composable transformation pipelines including `FuseConvBN`, `ConstantFolding`, `DeadCodeElimination`, `CommonSubexpressionElimination`, `FuseGELU`, `FuseLayerNorm`, and `FuseConvActivation` with automated metric reports.
- 🚀 **3-Stage Compiler Pipeline**: Micro-canonicalization (Pad-Conv fusion, Reshape collapse, Cast folding), macro-block synthesis (SDPA, Gemma-2 softcapping, DeepSeek MLA), and Netron FunctionProto dual-mode export with fixed-point iterative convergence.
- 📜 **Declarative JSON Rules Engine**: 53 pre-built declarative JSON rewrite rules for automated graph rewriting and custom operator fusion.
- ⚡ **PyTorch Execution & Differential Cross-Verification**: Standalone vectorized runtime (`PyTorchRunner`) for standard & custom ONNX ops with automated bitwise and tolerance checks against ONNX Runtime (`CrossVerifier`).
- 🧠 **Architectural Block Detection & Fingerprinting**: Automatic detection of Transformer, DiT, Mamba/SSM, GNN, and ConvNeXt architectural motifs with polymorphic 8-character SHA-256 fingerprinting.
- 🌐 **Unified Intermediate Representation (IR)**: Decoupled `IRGraph`, `IRNode`, and `IRTensor` with $O(1)$ producer/consumer lookups and format SPI adapters.
- 🎨 **Visual Topology & Interactive HTML**: Instant GitHub-flavored Mermaid flowcharts (`graph.to_mermaid()`) and standalone interactive HTML visualizers (`graph.to_html()`).
- 💻 **Developer CLI (`neuron`)**: Zero-code terminal tool to inspect topologies, profile hardware bounds, diff models, run passes, export formats, and query AST codemaps (`neuron ast`).
- 📦 **Zero-Dependency Safetensors Serialization**: Built-in weight serializer and loader conforming to HuggingFace `.safetensors` standards without requiring Rust toolchains.
- 📊 **Netron-Compatible JSON Export & Rebuilding**: Export graph topologies with statistical weight summaries without saving raw weights; rebuild complete, runnable ONNX or TFLite models directly from JSON.
- 🔬 **Differential Verification**: Extract subgraphs between arbitrary points and verify numerical equivalence against PyTorch or ONNXRuntime with automated tolerance checks (MAE, MSE, cosine similarity).

---

## 📦 Installation

```bash
# Core installation with ONNX support:
pip install "neuron-toolkit[onnx]"

# Full installation with TFLite and evaluation support:
pip install "neuron-toolkit[all]"
```

---

## ⚡ Quickstart in 60 Seconds

### 1. Load and Query a Model

```python
from neuron_toolkit import NeuronGraph

# Load any ONNX or TFLite model:
graph = NeuronGraph.load("resnet18.onnx")

# Query all Convolution layers with 3x3 kernels:
conv_nodes = (
    graph.query()
    .op("Conv")
    .filter(lambda n: n.attrs.get("kernel_shape") == [3, 3])
    .all()
)
print(f"Found {len(conv_nodes)} 3x3 Conv layers.")
```

### 2. Match and Fuse Subgraphs (e.g. SiLU Activation)

```python
from neuron_toolkit import NeuronGraph, Pattern

graph = NeuronGraph.load("model.onnx")

# Define pattern: x * Sigmoid(x) -> SiLU
x = Pattern.any().capture("input_tensor")
silu_pattern = x * Pattern.op("Sigmoid", x)

# Find all occurrences and replace with a unified "Silu" op:
rewriter = graph.replace(silu_pattern, "Silu")
optimized_model = rewriter.build(output_path="model_fused.onnx")
```

### 3. Safetensors & Lightweight JSON Export

```python
# Save weights to standard .safetensors:
graph.save_safetensors("weights.safetensors")

# Export clean JSON structure with weight statistics for visualization:
graph.to_json("architecture.json", indent=2)

# Reconstruct a complete model from JSON + weights:
rebuilt = NeuronGraph.from_json("architecture.json", weights="weights.safetensors")
```

---

## 📚 Documentation & Audit Reports

- 📖 **Full Documentation**: Explore detailed user guides and architecture specifications in [`docs/`](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/docs/index.md):
  - [Architecture & Design](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/docs/architecture.md)
  - [Loading & Saving Models](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/docs/guides/loading_and_saving.md)
  - [Query Engine Guide](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/docs/guides/query_engine.md)
  - [Pattern Matching DSL](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/docs/guides/pattern_matching.md)
  - [Graph Rewriting & Passes](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/docs/guides/rewriter_and_passes.md)
  - [Transformer & LLM Passes](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/docs/guides/transformer_passes.md)
  - [3-Stage Compiler Pipeline](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/docs/guides/compiler_pipeline.md)
  - [Declarative JSON Rules Engine](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/docs/guides/json_rules_engine.md)
  - [PyTorch Runtime & Cross-Verification](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/docs/guides/cross_verification_runtime.md)
  - [Architectural Block Recognition](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/docs/guides/architecture_prediction.md)
  - [Equality Saturation & E-Graphs](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/docs/guides/equality_saturation.md)
  - [Hardware Profiler & Roofline Model](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/docs/guides/profiler_and_roofline.md)
  - [Model Verification & Slicing](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/docs/guides/verification.md)
  - [Model Visualization](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/docs/guides/visualization.md)
  - [Developer CLI & AST Tooling](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/docs/guides/cli.md)
  - [Unified Intermediate Representation](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/docs/guides/unified_ir.md)
  - [API Reference](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/docs/api_reference.md)

- 📊 **Architecture & Codebase Audit Reports**: Deep-dive strategy reports are available in [`.report/`](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/.report/README.md):
  - [01: Codebase Improvements & Critical Bug Fixes](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/.report/01_codebase_improvements.md)
  - [02: Deprecations, Removals & Redundancy Analysis](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/.report/02_deprecations_and_removals.md)
  - [03: Architecture Redesign & Unified IR](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/.report/03_architecture_redesign.md)
  - [04: Feature Roadmap & High-Value Additions](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/.report/04_feature_roadmap.md)
  - [05: Testing, Evaluation & Benchmarking Strategy](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/.report/05_testing_and_benchmarking_strategy.md)

---

## 📄 License

This project is licensed under the Apache 2.0 License.
