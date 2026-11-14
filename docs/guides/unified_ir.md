# Guide: Unified Intermediate Representation (IR)

The Unified Intermediate Representation (`neuron_toolkit.ir`) provides a framework-agnostic DAG representation for computational graphs. By decoupling query, matching, and optimization algorithms from binary protobuf or flatbuffer implementations, it enables cross-framework interoperability.

---

## 1. Converting Models to IR

Any `NeuronGraph` can be converted to an `IRGraph` with a single call:

```python
from neuron_toolkit import NeuronGraph

graph = NeuronGraph.load("model.onnx")
ir = graph.to_ir()

print(f"IR Graph Name: {ir.name}")
print(f"Total IR Nodes: {len(ir.nodes)}")
print(f"Total IR Tensors: {len(ir.tensors)}")
```

---

## 2. Inspecting Nodes and Tensors

### `IRNode`

Represents an operation in the graph:

```python
node = ir.nodes[0]
print(node.op_type)     # e.g., "Conv"
print(node.inputs)      # ['X', 'W_conv', 'B_conv']
print(node.outputs)     # ['conv_out']
print(node.attributes)  # {'kernel_shape': [3, 3], 'pads': [1, 1, 1, 1]}
```

### `IRTensor`

Represents activations or weight parameters:

```python
tensor = ir.tensors["W_conv"]
print(tensor.shape)          # (64, 3, 3, 3)
print(tensor.dtype)          # "float32"
print(tensor.is_initializer) # True
print(tensor.rank)           # 4
print(tensor.numel)          # 1728
```

---

## 3. O(1) Producer & Consumer Lookups

`IRGraph` provides $O(1)$ dictionary lookups for upstream producers and downstream consumers:

```python
# Who produces 'conv_out'?
producer_node = ir.producers["conv_out"]

# Who consumes 'conv_out'?
consumer_nodes = ir.consumers["conv_out"]
for c in consumer_nodes:
    print(f"Consumed by: {c.op_type} ({c.name})")
```

---

## 4. Validating Graph Topological Order

Ensure valid execution ordering before deployment:

```python
is_valid = ir.is_topologically_sorted()
assert is_valid is True
```

---

## 5. Serialization Round-Trip

Serialize and deserialize the complete graph without losing metadata:

```python
# Export to dict
ir_dict = ir.to_dict()

# Reconstruct
from neuron_toolkit.ir import IRGraph
rebuilt_ir = IRGraph.from_dict(ir_dict)
```
