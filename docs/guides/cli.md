# Guide: Developer Command-Line Interface (`neuron`)

`neuron-toolkit` comes bundled with the `neuron` command-line tool, enabling zero-code inspection, pass optimization, differential comparisons, and format exports directly from the terminal.

---

## 1. Inspecting a Model (`neuron inspect`)

Inspect graph topology, input/output tensors, and operator distributions:

```bash
neuron inspect path/to/model.onnx
```

### Example Output:
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

### Supported Pass Identifiers:
| Pass Identifier | Class | Description |
| :--- | :--- | :--- |
| `fuse_conv_bn`, `conv_bn` | `FuseConvBN` | Folds BatchNormalization parameters into preceding Conv |
| `dce`, `dead_code` | `DeadCodeElimination` | Recursively removes unreachable/unconsumed nodes |
| `const_fold`, `constant_folding` | `ConstantFolding` | Folds static initializer arithmetic (Add, Sub, Mul, Div, Neg, Relu) |
| `cse` | `CommonSubexpressionElimination` | Merges duplicate operations with identical inputs |
| `gelu`, `fuse_gelu` | `FuseGELU` | Fuses multi-op Erf/Tanh GELU subgraphs into `Gelu` |
| `layernorm`, `fuse_layernorm` | `FuseLayerNorm` | Fuses 7-op normalization subgraphs into `LayerNormalization` |
| `conv_act`, `fuse_conv_act` | `FuseConvActivation` | Fuses Conv + Relu/LeakyRelu/Clip into `FusedConv` |

---

## 3. Diffing Two Models (`neuron diff`)

Compare two versions of a model to identify structural mutations, node reductions, and operator shifts:

```bash
neuron diff base_model.onnx optimized_model.onnx
```

### Example Output:
```text
Diff: base_model.onnx -> optimized_model.onnx
  Node count: 68 -> 36 (-32)
  Inputs:     ['image'] -> ['image']
  Outputs:    ['logits'] -> ['logits']

Op differences:
  BatchNormalization  :   32 ->    0 (-32)
```

---

## 4. Exporting Formats (`neuron export`)

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
