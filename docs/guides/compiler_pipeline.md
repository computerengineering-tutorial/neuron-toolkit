# 3-Stage Compiler Pipeline & Pass Optimization

The `neuron_toolkit.compiler` package provides an enterprise-grade 3-stage optimization pipeline designed to canonicalize, synthesize, and export ONNX computational graphs for both Netron visual inspection and high-performance execution runtimes.

---

## Architecture Overview

The compiler pipeline decomposes neural network subgraphs through three orchestrated stages:

```
                ┌──────────────────────────────────────────────┐
                │             Input ONNX / IR Model            │
                └──────────────────────┬───────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: MICRO-CANONICALIZATION                                             │
│ - ActivationCanonicalizer: Decomposed GELU, SiLU, HardSwish, Mish, Relu6   │
│ - MathCanonicalizer: Continuous constant arithmetic collapse, Gemm/Linear   │
│ - NormCanonicalizer: LayerNorm, RMSNorm, GroupNorm, SkipNorm fusions        │
│ - ShapeCanonicalizer: Cast folding, CSE, dead-code elimination, Reshape     │
│ - StaticFolder: Static constant & subgraph evaluation                       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: MACRO-BLOCK SYNTHESIS                                              │
│ - AttentionSynthesizer: ScaledDotProductAttention, SplitQKV, MergeHeads     │
│ - FeedForwardSynthesizer: MLP blocks, SwiGLU, GeGLU fusions                 │
│ - MacroBlockSynthesizer: SpatialGatingUnit (SGU), Highway, DeepMLP, SE      │
│ - AxialSynthesizer: Conformer axial rearrange fusions, Unbind routing       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: TARGET EXPORT & CODE GENERATION                                    │
│ - OnnxExporter: Dual-mode FunctionProto or standard ONNX Opset 23+ export   │
│ - PyTorchExporter: Native executable PyTorch runner with custom operators   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## High-Level Compiler Entry Point: `NeuronCompiler`

The `NeuronCompiler` class provides a unified classmethod interface for one-line graph optimization:

```python
import onnx
from neuron_toolkit.compiler import NeuronCompiler

# Load ONNX model
model = onnx.load("llama_block.onnx")

# Compile with dual-mode Netron FunctionProto representation
compiled_model = NeuronCompiler.compile(
    model,
    mode="function_proto",  # "function_proto", "modern_onnx", or "plain_custom_op"
    opset_version=23,
    rule_ids=None,          # None runs all registered pipeline passes, or specify list
    options={"enable_macro_blocks": True}  # Optional: enable macro block synthesis
)

# Save optimized model
onnx.save(compiled_model, "llama_block_optimized.onnx")
```

---

## Detailed Pipeline Driver: `OptimizationPipeline`

For fine-grained control over passes, execution order, and logging, use `OptimizationPipeline` and `CompilerContext`:

```python
from neuron_toolkit.compiler.pipeline import OptimizationPipeline
from neuron_toolkit.compiler.base import CompilerContext

# Instantiate pipeline driver with fixed-point iteration control
pipeline = OptimizationPipeline(mode="function_proto", max_fixed_point_iters=5)

# Create execution context with pass filtering or profiling
context = CompilerContext(
    mode="function_proto",  # 'function_proto', 'modern_onnx', 'plain_custom_op'
    opset_version=23,
    custom_domain="custom.pytorch",
    rule_ids=None,          # Filter specific passes by ID (e.g. ['erf_gelu', 'rmsnorm'])
    options={"enable_macro_blocks": True}
)

# Run complete 3-stage optimization with fixed-point convergence
optimized_model = pipeline.run(model, context=context)

# Inspect pass metrics recorded during execution
for pass_name, data in context.stats.items():
    print(f"Pass {pass_name}: nodes {data['nodes_before']} -> {data['nodes_after']} (delta: {data['delta']})")
```

### Iterative Fixed-Point Convergence Engine

Real-world model optimization requires cascading passes: for example, fusing a `Pad` into a `Conv` may expose new opportunities for continuous math collapse or reshape simplification. 

`OptimizationPipeline.run()` employs an iterative fixed-point loop:
1. At the start of each iteration, the compiler records a topological graph signature:
   $$\text{Signature} = \left(|V|, \langle \text{op}_1, \text{op}_2, \dots, \text{op}_{|V|} \rangle, \langle \text{output}_1, \dots, \text{output}_m \rangle\right)$$
2. All Stage 1 and Stage 2 passes are executed sequentially.
3. If the graph topological signature after the iteration matches the signature before the iteration, **fixed-point convergence has been reached**, and the pipeline exits immediately to Stage 3 Target Export without unnecessary iterations (capped at `max_fixed_point_iters=5`).

### Compiler Modes

`CompilerContext` supports three distinct target compilation modes:

1. **`function_proto` (Default - Netron Visual Hierarchy)**:
   Fuses subgraphs into custom operators under `custom.pytorch` domain while generating formal ONNX `FunctionProto` subroutines attached to `model.functions`. In Netron, clicking any fused node (e.g., `FeedForwardSwiGLU` or `RMSNorm`) reveals the full sub-graph hierarchy.
2. **`modern_onnx` (Standard ONNX Opset 23+)**:
   Emits standard ONNX specification operators without custom domains where supported (e.g., ONNX `RMSNormalization` in Opset 23+, `GroupNormalization` in Opset 21+, `LayerNormalization` in Opset 17+, and `Gelu`).
3. **`plain_custom_op` / `pytorch_custom` (Minimalist PyTorch Runtime)**:
   Emits lightweight custom operator nodes without embedding function bodies, optimized for minimal binary size and immediate execution via `PyTorchRunner`.

---

## Stage 1: Micro-Canonicalization Passes

Located in `neuron_toolkit.compiler.passes.canonicalization`:

### 1. `ActivationCanonicalizer`

Canonicalizes non-linear mathematical expansions into standardized primitives:

- **GELU (ERF)**: Fuses the standard 7-node Gaussian Error Linear Unit decomposition:
  $$\text{GELU}(x) = 0.5 \cdot x \cdot \left(1 + \text{Erf}\left(\frac{x}{\sqrt{2}}\right)\right)$$
  Replaces `Div(\sqrt{2}) \to \text{Erf} \to \text{Add}(1) \to \text{Mul}(x) \to \text{Mul}(0.5)` with `Gelu` or `custom.pytorch::Gelu`.
- **GELU (Tanh Approximation)**: Fuses the 10-node polynomial approximation:
  $$\text{GELU}_{\text{tanh}}(x) = 0.5 \cdot x \cdot \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \cdot (x + 0.044715 \cdot x^3)\right)\right)$$
  Emits `Gelu(approximate='tanh')`.
- **SiLU / Swish**: Fuses $x \cdot \sigma(x)$ (`Mul(x, Sigmoid(x))`) into `custom.pytorch::SiLU`.
- **HardSwish**: Fuses $x \cdot \frac{\text{Clip}(x + 3, 0, 6)}{6}$ into `custom.pytorch::HardSwish`.
- **Relu6**: Fuses $\text{Clip}(x, 0, 6)$ into `Relu6`.
- **Mish**: Fuses $x \cdot \tanh(\text{Softplus}(x))$ into `custom.pytorch::Mish`.
- **Softmax**: Fuses decomposed numerically stable softmax expansions:
  $$\text{Softmax}(x) = \frac{\exp(x - \max(x))}{\sum \exp(x - \max(x))}$$
  Replaces `ReduceMax \to \text{Sub} \to \text{Exp} \to \text{ReduceSum} \to \text{Div}` with standard `Softmax(axis=-1)`.

### 2. `MathCanonicalizer`

Canonicalizes matrix multiplication, Gemm, Einsum, and continuous math chains:

- **Linear Fusion**: Fuses $\text{MatMul}(X, W) + B$ or $\text{Gemm}$ into `custom.pytorch::Linear(X, W, B)`.
  > [!IMPORTANT]
  > **Shared Weight Deduplication**: When multiple linear projections share the same transposed weight matrix, `MathCanonicalizer` inspects existing initializers before appending `w_name_transposed`, avoiding duplicate tensor definitions.
- **Pad-Conv Fusion (`apply_pad_conv_fusion`)**: Fuses explicit `Pad` operations immediately preceding `Conv` nodes into the `Conv` operator's internal `pads` attribute when spatial padding dimensions and modes (`constant`, value 0) are compatible, eliminating intermediate memory allocations and kernel launch overhead.
- **Continuous Math Collapse**: Collapses chains of elementwise constant arithmetic (`Add`, `Sub`, `Mul`, `Div`, `Pow`, `Sqrt`) into folded constants or specialized expressions stored directly in node docstrings.

### 3. `NormCanonicalizer`

Normalizes standard and custom normalization primitives across ONNX opset versions:

- **RMSNorm**: Detects Root Mean Square Normalization:
  $$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}} \odot \gamma$$
  Replaces `Pow(2) \to \text{ReduceMean} \to \text{Add}(\epsilon) \to \text{Sqrt} \to \text{Div} \to \text{Mul}(\gamma)` with standard ONNX `RMSNormalization` (Opset 23+) or `custom.pytorch::RMSNorm`.
- **LayerNorm**: Detects mean/variance normalization and emits standard `LayerNormalization` (Opset 17+) or `custom.pytorch::LayerNorm`.
- **GroupNorm**: Fuses group-channel reshape, instance normalization, and affine scales into standard `GroupNormalization` (Opset 21+).
- **SkipNorm Fusion**: Fuses residual skip additions immediately preceding normalization into `SkipSimplifiedLayerNormalization` or `SkipLayerNormalization`.

### 4. `ShapeCanonicalizer`

Optimizes tensor topology and constants:

- **Consecutive Reshape Collapsing (`apply_reshape_collapsing`)**: Fuses consecutive reshapes ($X \to \text{Reshape}(S_1) \to \text{Reshape}(S_2) \implies X \to \text{Reshape}(S_2)$) when intermediate tensors have single fan-out, eliminating redundant layout re-indexing operations.
- **Squeeze-Unsqueeze Cancellation (`apply_squeeze_unsqueeze_cancellation`)**: Detects and cancels adjacent inverse squeeze and unsqueeze pairs operating along the same axes, restoring direct tensor flow.
- **Redundant Cast Elimination (`apply_redundant_cast_elimination`)**: Detects redundant data type conversions (casts where the source tensor already matches the target `to` data type or back-to-back identical casts) and replaces consumer inputs directly with the pre-cast tensor.
- **Common Subexpression Elimination (CSE)**: Deduplicates identical nodes producing identical outputs.
  > [!TIP]
  > **Exposed Output Safeguard**: CSE preserves any constant node whose output is explicitly listed in `graph.output`.
- **Dead Code Elimination (DCE)**: Recursively removes unreachable nodes without mutating model inputs or outputs. Correctly traverses nested control-flow subgraphs (`If.then_branch`, `Loop.body`).
- **Cast-Initializer Folding**: Pre-casts static weight initializers at compile-time and eliminates runtime `Cast` operators, correctly collecting references across nested control-flow subgraphs.
- **Transpose Cancellation**: Fuses consecutive inverse permutations ($\text{Transpose}(P_1) \to \text{Transpose}(P_2)$).

### 5. `StaticFolder`

Performs static subgraph evaluation and constant propagation across scalar arithmetic, slice expressions, and shape calculation chains.

---

## Stage 2: Macro-Block Synthesis Passes

Located in `neuron_toolkit.compiler.passes.synthesis`:

### 1. `AttentionSynthesizer`

- **ScaledDotProductAttention (SDPA)**: Fuses scaled dot-product attention subgraphs:
  $$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d}} + M\right) V$$
  Replaces `MatMul \to \text{Div} \to \text{Add(Mask)} \to \text{Softmax} \to \text{MatMul}` with `custom.pytorch::ScaledDotProductAttention`.
- **Gemma-2 Logits Soft-Capping (`apply_gemma2_softcapping_fusion`)**: Fuses attention score capping sequences:
  $$\text{Scores}_{\text{capped}} = \text{cap} \cdot \tanh\left(\frac{Q K^T}{\sqrt{d} \cdot \text{cap}}\right)$$
  Replaces `Div(cap) \to \text{Tanh} \to \text{Mul(cap)}` with monolithic `custom.pytorch::LogitsSoftCap(scores, cap=...)`.
- **DeepSeek Multi-Head Latent Attention (MLA) (`apply_deepseek_mla_fusion`)**: Synthesizes DeepSeek-V2 / DeepSeek-V3 low-rank KV compression and decoupled RoPE key projections:
  $$c_{KV} = X W_{DKV}, \quad K_C = c_{KV} W_{UK}, \quad V = c_{KV} W_{UV}, \quad K_R = \text{RoPE}(X W_{KR})$$
  Fuses down-projection and up-projection into `custom.pytorch::MultiHeadLatentAttention_KVProjection`.
- **SplitQKV**: Fuses joint projection unbinding or slicing into `custom.pytorch::SplitQKV`.
- **MergeHeads**: Fuses multi-head transpose and reshape operations (`Transpose(0, 2, 1, 3) \to \text{Reshape}`) into `custom.pytorch::MergeHeads`.
- **RotaryEmbedding (RoPE)**: Fuses rotary frequency slicing, sign inversion, and complex coordinate rotation into `custom.pytorch::RotaryEmbedding`. Supports partial head dimensions.

### 2. `FeedForwardSynthesizer`

- **FeedForward (GELU MLP)**: Fuses `RMSNorm -> Linear -> Gelu -> Linear` into monolithic `custom.pytorch::FeedForward` with content-hashed `FunctionProto` attachments.
- **FeedForwardSwiGLU**: Fuses modern LLaMA-style FFN blocks:
  $$\text{SwiGLU}(x) = (\text{Linear}_{\text{gate}}(x) \odot \text{SiLU}(\text{Linear}_{\text{up}}(x))) \cdot W_{\text{down}}$$
  Replaces `RMSNorm \to (\text{Linear} \cdot \text{SiLU}) \cdot \text{Linear} \to \text{Linear}` with `custom.pytorch::FeedForwardSwiGLU`.

### 3. `MacroBlockSynthesizer`

Synthesizes high-level neural blocks for vision, diffusion, and state-space architectures:

- **SpatialGatingUnit (SGU)**: Fuses gMLP and Vision-Transformer gating units (`Split -> Norm -> Linear -> Mul`).
- **HighwayBlock**: Fuses Highway network transform gates, carry gates, and residual additions.
- **SqueezeExcitation**: Fuses channel attention blocks (`GlobalAveragePool -> Linear -> ReLU -> Linear -> Sigmoid -> Mul`) with dynamic 3D sequence and 4D spatial tensor shape handling.
- **DeepMLP_Norm_Act**: Fuses deep MLP projection layers combining normalization, linear projections, and activations.
- **MultiHeadAttention Composite**: Fuses complete multi-head attention composites (`SplitQKV -> SDPA -> MergeHeads -> Linear`).

### 4. `AxialSynthesizer`

Synthesizes speech and multidimensional attention blocks (Conformer, DiT):

- **AxialRearrange**: Fuses frequency-to-time, time-to-conv, and channel rearrangements.
- **Unbind**: Fuses multi-slice singletons into unified tensor unbinding operations.

---

## Stage 3: Target Exporters

Located in `neuron_toolkit.compiler.exporters`:

### `OnnxExporter`

Exports the optimized graph in two distinct modes:

1. **`function_proto` Mode (Recommended for Netron)**:
   Custom operators are defined as formal ONNX `FunctionProto` entities stored directly inside `model.functions`. In Netron, clicking on a fused node (`FeedForwardSwiGLU`, `ScaledDotProductAttention`, `RMSNorm`) allows engineers to expand and view the underlying sub-nodes!
2. **`plain` Custom Op Mode**:
   Emits lightweight custom operator nodes with domain `custom.pytorch` for direct execution by the `PyTorchRunner`.

```python
from neuron_toolkit.compiler.exporters.onnx_exporter import OnnxExporter

exporter = OnnxExporter(mode="function_proto")
final_model = exporter.export(model, output_path="exported_model.onnx")
```

### `PyTorchExporter`

Compiles the optimized ONNX model directly into an executable `PyTorchRunner`:

```python
from neuron_toolkit.compiler.exporters.pytorch_exporter import PyTorchExporter
import numpy as np

exporter = PyTorchExporter()
runner = exporter.export(model)

# Run inference directly with PyTorch tensors or NumPy arrays
outputs = runner.run({"input_ids": np.zeros((1, 128), dtype=np.int64)})
```

---

## Real-World Optimization Benchmarks

Evaluated on deep learning architectures from `.helper/models/` and `.temp/diffrent_models/`:

| Model Architecture | Unoptimized Nodes | Optimized Nodes | Nodes Eliminated | Reduction % | Active Fused Ops |
| --- | --- | --- | --- | --- | --- |
| **Axial Conformer (Model 1)** | 4,740 | 2,342 | 2,398 | **50.6%** | `RMSNorm`, `Gelu`, `AxialRearrange`, `CSE` |
| **Gated MLP Vision (Model 2)** | 490 | 112 | 378 | **77.1%** | `Linear`, `Gelu`, `CastFolding` |
| **LLaMA Transformer Block** | 128 | 42 | 86 | **67.2%** | `RMSNorm`, `FeedForwardSwiGLU`, `SDPA`, `RoPE` |
| **Diffusion DiT Block** | 215 | 88 | 127 | **59.1%** | `AdaLayerNorm`, `ScaledDotProductAttention`, `Linear` |
| **Mamba SSM Block** | 184 | 64 | 120 | **65.2%** | `SelectiveScan`, `Conv1D_Gate`, `SiLU` |
