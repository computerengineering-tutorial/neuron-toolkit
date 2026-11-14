# Guide: Model Slicing & Numerical Verification

`neuron-toolkit` provides advanced model slicing and differential testing utilities to isolate subgraphs and verify that their numerical behavior matches reference implementations (such as PyTorch modules) within strict numerical tolerances.

---

## 1. Subgraph Slicing

Extract any section of a neural network into an independent, runnable model:

```python
from neuron_toolkit import NeuronGraph

graph = NeuronGraph.load("resnet50.onnx")

# Slice between two arbitrary boundary points:
sliced_graph = graph.slice(
    start_points=["conv1"],
    end_points=["layer1.2.relu_2"],
    output_path="subgraph_layer1.onnx"
)

print(f"Original graph nodes: {len(graph.nodes)}")
print(f"Sliced subgraph nodes: {len(sliced_graph.nodes)}")
```

### Slicing a Single Node

Extract an isolated node (e.g. for micro-benchmarking or debugging):

```python
single_node_model = graph.slice_node("layer2.0.conv1", output_path="isolated_conv.onnx")
```

---

## 2. Differential Verification Against PyTorch (`compare_outputs`)

Verify that a sliced subgraph in ONNX or TFLite produces identical numerical outputs to a PyTorch `nn.Module` when fed the same inputs:

```python
import numpy as np
import torch
import torch.nn as nn
from neuron_toolkit import NeuronGraph

# 1. Define reference PyTorch module
class ReferenceBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))

pytorch_block = ReferenceBlock().eval()

# 2. Load model
graph = NeuronGraph.load("model.onnx")

# 3. Prepare test inputs
inputs = {
    "input_0": np.random.randn(1, 3, 32, 32).astype(np.float32)
}

# 4. Compare outputs between PyTorch and the sliced subgraph
report = graph.compare_outputs(
    start_points=["conv0"],
    end_points=["relu0"],
    pytorch_module=pytorch_block,
    inputs_dict=inputs,
    rtol=1e-4,
    atol=1e-5
)

# 5. Inspect numerical verification metrics
for output_name, metrics in report.items():
    print(f"Output: {output_name}")
    print(f"  - All Close         : {metrics['all_close']}")
    print(f"  - Cosine Similarity : {metrics['cosine_similarity']:.6f}")
    print(f"  - Mean Absolute Err : {metrics['mae']:.6e}")
    print(f"  - Mean Squared Err  : {metrics['mse']:.6e}")
    print(f"  - Shape Matched     : {metrics['shape_match']}")
```

---

## 3. Decorator-Based PyTorch Verification

Wrap an existing PyTorch function or `forward()` call with `@verify_against_pytorch` to automatically assert equivalence during unit tests:

```python
from neuron_toolkit import NeuronGraph
import torch

graph = NeuronGraph.load("model.onnx")

@graph.verify_against_pytorch(
    start_points=["stem_conv"],
    end_points=["stem_relu"],
    rtol=1e-4,
    atol=1e-5
)
def forward_pass(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(my_pytorch_conv(x))

# Run the function:
sample_input = torch.randn(1, 3, 224, 224)
output = forward_pass(sample_input)

# Check attached similarity metrics:
report = forward_pass.similarity_report
assert report["stem_relu_out"]["all_close"] == True
```

---

## 4. Cross-Backend Execution (ONNXRuntime & LiteRT)

`compare_outputs` transparently delegates execution to the optimal engine based on model format:

- **ONNX Models**: Executed via `onnxruntime.InferenceSession`.
- **TFLite Models**: Dynamically discovers and loads the available execution interpreter across:
  1. `ai_edge_litert` (Google LiteRT 2.0+ runtime, recommended)
  2. `tflite_runtime` (Standalone TensorFlow Lite runtime)
  3. `tensorflow.lite` (Full TensorFlow installation)

This allows running differential tests on CPU environments without heavy TensorFlow installations.
