# Guide: Developer Command-Line Interface (`neuron`)

`neuron-toolkit` comes bundled with the `neuron` command-line tool, enabling zero-code inspection, pass optimization, differential comparisons, and format exports directly from the terminal.

---

## 1. Inspecting a Model (`neuron inspect`)

Inspect graph topology, input/output tensors, and operator distributions:

```bash
neuron inspect path/to/model.onnx
```

### Example Output

```text
=== NeuronGraph: path/to/model.onnx ===
Total Nodes: 68
Inputs (1): ['image']
Outputs (1): ['logits']

Operator Breakdown:
  Conv                    :   32 nodes
  BatchNormalization      :   32 nodes
  Relu                    :   32 nodes
  Gemm                    :    1 nodes
  Softmax                 :    1 nodes

Initializers/Weights: 65 tensors
```

---

## 2. Running Optimization Passes (`neuron optimize`)

Execute optimization passes from the terminal and save the transformed graph:

```bash
# Fuse Conv+BN and eliminate dead code
neuron optimize model.onnx --passes fuse_conv_bn,dce --output optimized.onnx

# Run multi-pass pipeline: constant folding, CSE, GELU, LayerNorm, Conv+Act, and DCE
neuron optimize model.onnx --passes const_fold,cse,gelu,layernorm,conv_act,dce -o opt.onnx
```

### Supported Pass Identifiers

| Pass Identifier | Class | Description |
| :--- | :--- | :--- |
| `fuse_conv_bn`, `conv_bn` | `FuseConvBN` | Folds BatchNormalization parameters into preceding Conv |
| `dce`, `dead_code` | `DeadCodeElimination` | Recursively removes unreachable/unconsumed nodes |
| `const_fold`, `constant_folding` | `ConstantFolding` | Folds static initializer arithmetic (Add, Sub, Mul, Div, Neg, Relu) |
| `cse` | `CommonSubexpressionElimination` | Merges duplicate operations with identical inputs |
| `gelu`, `fuse_gelu` | `FuseGELU` | Fuses multi-op Erf/Tanh GELU subgraphs into `Gelu` |
| `layernorm`, `fuse_layernorm` | `FuseLayerNorm` | Fuses 7-op normalization subgraphs into `LayerNormalization` |
| `conv_act`, `fuse_conv_act` | `FuseConvActivation` | Fuses Conv + Relu/LeakyRelu/Clip into `FusedConv` |
| `fuse_rmsnorm`, `rmsnorm` | `FuseRMSNorm` | Fuses multi-op RMSNorm subgraphs into unified `RMSNorm` |
| `fuse_swiglu`, `swiglu` | `FuseSwiGLU` | Fuses SiLU/Sigmoid gating branches and projections into `SwiGLU` |
| `fuse_attention`, `attention`, `sdpa` | `FuseAttention` | Fuses scaled dot-product attention subgraphs into `ScaledDotProductAttention` |
| `fuse_rope`, `rope` | `FuseRotaryEmbedding` | Fuses RoPE slicing, negation, and rotation into `RotaryEmbedding` |

---

## 3. Profiling Hardware Complexity & Roofline (`neuron profile`)

Analyze FLOPs, MACs, parameter counts, memory timeline, and Roofline latency limits directly from your CLI:

```bash
# Print formatted terminal summary for NVIDIA A100:
neuron profile model.onnx --device a100 --dtype float16

# Profile for Apple M3 Max and export results as JSON:
neuron profile model.onnx --device m3 --format json -o profile.json

# Export a GitHub-flavored Markdown report for documentation:
neuron profile model.onnx --device h100 --format markdown -o profile.md
```

Supported `--device` profiles: `a100`, `h100`, `4090`, `m3`, `arm`, `cpu`.

---

## 4. Diffing Two Models (`neuron diff`)

Compare two versions of a model to identify structural mutations, node reductions, and operator shifts:

```bash
neuron diff base_model.onnx optimized_model.onnx
```

### Example Output

```text
Diff: base_model.onnx -> optimized_model.onnx
  Node count: 68 -> 36 (-32)
  Inputs:     ['image'] -> ['image']
  Outputs:    ['logits'] -> ['logits']

Op differences:
  BatchNormalization  :   32 ->    0 (-32)
```

---

## 5. Exporting Formats (`neuron export`)

Convert graphs to Netron JSON, interactive HTML visualizers, Mermaid diagrams, or Safetensors weights:

```bash
# Export to Netron-compatible JSON
neuron export model.onnx --format json --output model.json

# Export to standalone interactive HTML
neuron export model.tflite --format html --output visualizer.html

# Export to Mermaid flowchart
neuron export model.onnx --format mermaid --output graph.mmd

# Export model parameters to zero-copy Safetensors
neuron export model.onnx --format safetensors --output weights.safetensors
```
