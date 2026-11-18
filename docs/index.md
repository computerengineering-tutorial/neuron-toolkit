# `neuron-toolkit` Documentation

**A Unified ONNX and TFLite Parsing, Querying, Pattern-Matching, and Transformation Toolkit.**

`neuron-toolkit` provides a high-level, framework-agnostic Python interface for deep learning model surgery, pattern matching, graph querying, and numerical verification. It bridges the gap between disparate model formats (ONNX, TFLite, and Netron-compatible JSON), empowering compiler engineers, optimization researchers, and ML infrastructure developers to manipulate neural computational graphs effortlessly.

---

## Key Highlights

- 🔍 **Fluent Query Engine**: Chainable, SQL-like queries to traverse, filter, and inspect computational nodes (`graph.query().op("Conv").has_params()`).
- 🧩 **Pattern Matching DSL**: Declarative subgraph pattern matching with operator overloads (`x * Pattern.op("Sigmoid", x)`), captures, wildcards, and constraints.
- ⚡ **Transformer & LLM Optimization Passes**: Pre-packaged fusion passes for `RMSNorm`, `SwiGLU`, `ScaledDotProductAttention`, and `RotaryEmbedding` (RoPE).
- 🔄 **Equality Saturation (E-Graphs)**: Solves the compiler phase ordering problem by applying rewrites non-destructively and extracting the optimal DAG using dynamic programming.
- 📊 **Hardware Profiler & Roofline Latency Modeling**: Analytical FLOPs/MACs calculator, chronological active working set memory simulation, and Roofline model bottleneck classification across NVIDIA A100/H100, RTX 4090, Apple Silicon, and ARM.
- ✂️ **Graph Surgery & Rewriting**: Staged, topological-preserving node replacement, insertion, and deletion (`rewriter.replace(...)`).
- 📦 **Zero-Dependency Safetensors Serialization**: Built-in weight serializer and loader conforming to HuggingFace `.safetensors` standards without requiring Rust toolchains.
- 📊 **Netron-Compatible JSON Export**: Export graph topologies with statistical weight summaries (min, max, mean, std, zero fraction) without saving raw weight buffers.
- 🔬 **Differential Verification**: Extract subgraphs between arbitrary points and verify numerical equivalence against PyTorch or ONNXRuntime with automated tolerance checks (MAE, MSE, cosine similarity).
- 🚀 **3-Stage Compiler Pipeline**: Iterative fixed-point convergence engine with micro-canonicalization (Pad-Conv fusion, Reshape collapse, Cast folding), macro-block synthesis (SDPA, Gemma-2 softcapping, DeepSeek MLA), and Netron FunctionProto dual-mode export (`NeuronCompiler.compile`).
- 🧠 **Architecture Recognition & Fingerprinting**: Automatic topology analysis predicting Transformer, DiT, Mamba/SSM, GNN, and ConvNeXt architectures with polymorphic subgraph fingerprinting.
- 📜 **Declarative JSON Rules Engine**: Extensible declarative graph transformation engine with 39 pre-built JSON rewrite rules (`JSONRuleEngine`).
- ⚡ **PyTorch Execution & Cross-Verification**: Standalone vectorized execution of custom ONNX graphs via `PyTorchRunner` (including `GlobalAveragePool`, `RNN`, `LogitsSoftCap`, `MLA`) with automated differential equivalence testing against ONNX Runtime.
- 🧭 **AST Code Intelligence & CLI**: Automated Python AST codemap generation, regex symbol querying, docstring indexing, and coverage reporting via `neuron ast`.

---

## Quick Start in 60 Seconds

### Installation

```bash
# Core installation with ONNX support:
pip install "neuron-toolkit[onnx]"

# Full installation with TFLite and evaluation support:
pip install "neuron-toolkit[all]"
```

### 1. Load and Query a Model

```python
from neuron_toolkit import NeuronGraph

# Load any ONNX or TFLite model:
graph = NeuronGraph.load("resnet18.onnx")

# Query all Convolution layers with kernel_size == 3:
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

### 3. Compile & Optimize with the 3-Stage Compiler

```python
import onnx
from neuron_toolkit.compiler import NeuronCompiler

# Compile model into dual-mode Netron FunctionProto representation:
model = onnx.load("llama_raw.onnx")
compiled_model = NeuronCompiler.compile(model, mode="function_proto", opset_version=23)
onnx.save(compiled_model, "llama_optimized.onnx")
```

### 4. Automatic Architecture Block Prediction

```python
from neuron_toolkit.compiler.analyzer.block_predictor import ArchitectureBlockPredictor

# Automatically inspect and predict model topology (Transformers, DiT, Mamba, CNNs, GNNs):
predictor = ArchitectureBlockPredictor()
manifest = predictor.analyze("model.onnx")
manifest.print_report()
```

### 5. High-Level Declarative Rules & Differential Verification

```python
from neuron_toolkit import NeuronGraph
import numpy as np

# Load model, apply declarative rewrite rules, and cross-verify in PyTorch:
graph = NeuronGraph.load("model.onnx")
opt_graph = graph.apply_json_rules(mode="function_proto")

inputs = {"input_ids": np.zeros((1, 128), dtype=np.int64)}
report = graph.cross_verify(opt_graph, inputs=inputs)
print(report.summary())
assert report.all_passed
```

### 6. Export to JSON (Netron Visualizer Compatible)

```python
# Export clean JSON structure with weight statistics:
json_string = graph.to_json(path="model_graph.json", indent=2)
```

---

## Documentation Navigation

```mermaid
graph LR
    Index["Docs Home\n(docs/index.md)"]
    Arch["Architecture\n(docs/architecture.md)"]
    API["API Reference\n(docs/api_reference.md)"]
    Guides["User Guides"]

    Index --> Arch
    Index --> Guides
    Index --> API

    Guides --> G1["Loading & Saving\n(guides/loading_and_saving.md)"]
    Guides --> G2["Query Engine\n(guides/query_engine.md)"]
    Guides --> G3["Pattern Matching\n(guides/pattern_matching.md)"]
    Guides --> G4["Rewriting & Passes\n(guides/rewriter_and_passes.md)"]
    Guides --> G5["Transformer Passes\n(guides/transformer_passes.md)"]
    Guides --> G6["Compiler Pipeline\n(guides/compiler_pipeline.md)"]
    Guides --> G7["Architecture Prediction\n(guides/architecture_prediction.md)"]
    Guides --> G8["JSON Rules Engine\n(guides/json_rules_engine.md)"]
    Guides --> G9["Runtime & Cross-Verification\n(guides/cross_verification_runtime.md)"]
    Guides --> G10["Equality Saturation\n(guides/equality_saturation.md)"]
    Guides --> G11["Hardware Profiler\n(guides/profiler_and_roofline.md)"]
    Guides --> G12["Developer CLI\n(guides/cli.md)"]
```

### Guides

1. [**Architecture & Design Philosophy**](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/docs/architecture.md): Decoupled multi-layer architecture, graph representations, and execution flow.
2. [**Loading & Saving Models**](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/docs/guides/loading_and_saving.md): Working with `.onnx`, `.tflite`, JSON representations, and `.safetensors`.
3. [**Query Engine Guide**](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/docs/guides/query_engine.md): Filtering by op/name/attribute, graph traversals, and boolean set operations.
4. [**Pattern Matching DSL**](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/docs/guides/pattern_matching.md): Building subgraph patterns, wildcards, arithmetic overloads, and commutative matching.
5. [**Graph Rewriting & Passes**](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/docs/guides/rewriter_and_passes.md): Performing in-place AST surgery, PassManager, and RewriteRule engine.
6. [**Transformer & LLM Passes**](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/docs/guides/transformer_passes.md): Fusing RMSNorm, SwiGLU, SDPA/Attention, and RotaryEmbeddings.
7. [**3-Stage Compiler Pipeline**](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/docs/guides/compiler_pipeline.md): Micro-canonicalization, macro-block synthesis, and dual-mode target exporters.
8. [**Architecture Recognition & Fingerprinting**](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/docs/guides/architecture_prediction.md): Semantic subnet recognition, architecture prediction, and polymorphic subgraph fingerprinting.
9. [**Declarative JSON Rules Engine**](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/docs/guides/json_rules_engine.md): Extensible pattern rules schema, built-in 29+ rules catalog, and custom rule authoring.
10. [**PyTorch Runtime & Cross-Verification**](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/docs/guides/cross_verification_runtime.md): Native vectorized PyTorch graph execution and automated numerical equivalence verification against ONNX Runtime.
11. [**Equality Saturation & E-Graphs**](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/docs/guides/equality_saturation.md): Non-destructive optimization and dynamic programming DAG extraction.
12. [**Hardware Profiler & Roofline Model**](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/docs/guides/profiler_and_roofline.md): FLOPs/MACs, peak activation working set simulation, and latency limits.
13. [**Model Verification & Slicing**](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/docs/guides/verification.md): Slicing subgraphs, LiteRT execution, and differential verification against PyTorch.
14. [**Model Visualization**](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/docs/guides/visualization.md): Mermaid diagrams and interactive HTML visualizer.
15. [**Developer CLI**](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/docs/guides/cli.md): Using the `neuron` command line tool (`inspect`, `profile`, `optimize`, `diff`, `export`).
16. [**Unified Intermediate Representation**](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/docs/guides/unified_ir.md): Working with framework-agnostic IRGraph, IRNode, and IRTensor.
17. [**Complete API Reference**](file:///home/jenil-sheth/Desktop/Backup/neuron-toolkit/docs/api_reference.md): Full signature and docstring documentation for all public modules.
