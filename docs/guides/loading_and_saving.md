# Guide: Loading & Saving Models

This guide covers how to load models across formats, extract and replace weight tensors, export Netron-compatible JSON representations, and save weights into `.safetensors`.

---

## 1. Loading Models

`NeuronGraph` provides a unified entry point that auto-detects model formats based on file extension, byte headers, or Python object types:

```python
from neuron_toolkit import NeuronGraph

# 1. Load from file path (auto-detects .onnx, .tflite, .bin, .json)
graph = NeuronGraph.load("resnet50.onnx")
tflite_graph = NeuronGraph.load("mobilenet_v2.tflite")

# 2. Load from raw bytes (checks TFL3 magic header automatically)
with open("model.onnx", "rb") as f:
    graph_from_bytes = NeuronGraph(f.read())

# 3. Load from existing in-memory protobuf
import onnx
model_proto = onnx.load("model.onnx")
graph_from_proto = NeuronGraph(model_proto)

# 4. Large Models with External Data
# NeuronGraph automatically handles models with external data buffers. If the external file
# is detached or missing, it gracefully falls back to topology parsing without crashing:
large_graph = NeuronGraph.load("huge_model.onnx", load_external_data=False)
```

---

## 2. Inspecting Nodes & Weights

Once loaded, inspect graph properties directly:

```python
# Access nodes in graph
print(f"Total nodes: {len(graph.nodes)}")

# Inspect weight / initializer tensors (returns LazyTensorMap with zero-copy eviction)
weights = graph.get_weights()
for name, tensor in weights.items():
    print(f"Tensor: {name:<30} Shape: {tensor.shape} Dtype: {tensor.dtype}")

# High-speed cached dictionary export (ideal for fast JSON serialization or UI tooling)
graph_dict = graph.to_dict()  # Memoized: subsequent calls return in O(1) time
```

# Access shape and dtype information

shapes = graph.shape_info
for name, (rank, dtype) in list(shapes.items())[:5]:
    print(f"{name}: Rank {rank}, Dtype {dtype}")

```

---

## 3. Working with Safetensors

`neuron-toolkit` includes a built-in, zero-dependency Safetensors serializer and loader conforming to the official HuggingFace `.safetensors` binary format.

### Saving Weights to Safetensors

```python
# Export all model weights into a .safetensors file with optional metadata:
graph.save_safetensors(
    "model_weights.safetensors",
    metadata={"author": "NeuronToolkit", "framework": "onnx"}
)
```

### Loading & In-Place Weight Replacement (like PyTorch `load_state_dict`)

```python
# Load weights from Safetensors and inject them directly into model initializers:
diff = graph.load_safetensors("model_weights.safetensors", strict=False)
print("Missing keys:", diff["missing_keys"])
print("Unexpected keys:", diff["unexpected_keys"])

# You can also replace weights directly from a Python dictionary:
import numpy as np
new_weights = {
    "conv1.weight": np.random.randn(64, 3, 7, 7).astype(np.float32)
}
graph.replace_weights(new_weights, strict=False)
```

---

## 4. Exporting to JSON & Rebuilding

You can export the entire graph structure into a clean, lightweight JSON format. This format is ideal for web visualizers (like Netron) and architecture inspection because it replaces heavy raw binary weights with statistical metrics (mean, std, min, max, zero fraction).

### Exporting Graph JSON

```python
# Export to a JSON file:
graph.to_json("architecture.json", indent=2)

# Or export directly to a Python dictionary:
graph_dict = graph.to_dict()
print("Model type:", graph_dict["model_type"])
print("Node count:", len(graph_dict["nodes"]))
```

### Rebuilding a Model from JSON

You can reconstruct a complete, fully-functional ONNX or TFLite model directly from the JSON structure. `NeuronGraph.from_json` auto-detects `"model_type": "TFLite"` or `"ONNX"`:

```python
# Reconstruct with reproducible random weights:
rebuilt_graph = NeuronGraph.from_json("architecture.json", seed=42)

# Or reconstruct with exact weights from a Safetensors file or dict:
rebuilt_graph = NeuronGraph.from_json(
    "architecture.json",
    weights="model_weights.safetensors"
)

# For TFLite explicitly:
from neuron_toolkit import build_tflite_model_from_json

tflite_bytes = build_tflite_model_from_json("tflite_architecture.json", seed=42)
tflite_graph = NeuronGraph(tflite_bytes)

# Save the rebuilt model to disk:
rebuilt_graph.save("rebuilt_model.onnx")
```

---

## 5. Saving & Exporting Models

Save modified models back to disk:

```python
# Standard ONNX / TFLite save
graph.save("optimized_model.onnx")

# Dual-Mode Netron Export:
# Serializes the graph with formal ONNX FunctionProto definitions attached.
# Clicking fused nodes (e.g., ScaledDotProductAttention, SwiGLU, RMSNorm) in Netron
# lets you expand and view the full underlying sub-nodes!
from neuron_toolkit.rules import export_simplified_onnx

export_simplified_onnx(
    model_or_graph=graph,
    output_path="model_netron_expandable.onnx",
    metadata={"optimizer": "NeuronCompiler", "version": "2.0"}
)
```
