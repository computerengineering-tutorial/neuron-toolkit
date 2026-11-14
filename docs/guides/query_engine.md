# Guide: Graph Query Engine (`NeuronQuery`)

The `NeuronQuery` engine provides an expressive, chainable, and immutable interface for querying and traversing computational graphs. Inspired by modern ORMs and LINQ, it allows you to filter nodes, inspect parameters, trace dependencies, and perform boolean set operations.

---

## 1. Initializing Queries

Queries are spawned directly from any `NeuronGraph`:

```python
from neuron_toolkit import NeuronGraph

graph = NeuronGraph.load("model.onnx")

# Start an unbound query across all nodes in the model:
q = graph.query()
print(f"Querying across {q.count()} nodes.")
```

---

## 2. Filtering Nodes

Filtering methods return a new cloned `NeuronQuery` instance, allowing seamless method chaining:

### By Operator Type

```python
# Select all Conv operators
convs = graph.query().op("Conv")

# Select all BatchNormalization operators
bns = graph.query().op("BatchNormalization")
```

### By Node Name

```python
# Substring match (case-insensitive by default)
backbone_nodes = graph.query().name("layer1")

# Exact name match
stem_conv = graph.query().name("conv1_relu", exact=True)
```

### By Attributes

```python
# Filter by attribute existence or specific value
grouped_convs = graph.query().op("Conv").attr("group", 1)

# Pass a custom predicate callable to match attribute values
dilated_convs = graph.query().op("Conv").attr(
    "dilations", lambda d: d is not None and any(x > 1 for x in d)
)
```

### By Tensor Shape & Data Type

```python
# Filter nodes whose primary output has rank 4:
rank4_nodes = graph.query().rank(4)

# Filter nodes producing float16 outputs:
fp16_nodes = graph.query().dtype("float16")
```

### Custom Predicate Filtering

```python
# Arbitrary lambda or function predicate:
multi_input_nodes = graph.query().filter(lambda n: len(n.input) > 2)
```

### Parameter Presence

```python
# Filter nodes that own weight / bias initializer tensors:
nodes_with_weights = graph.query().has_params()
```

---

## 3. Graph Dependency Traversals

Navigate upstream and downstream dependencies through the computation graph:

```python
# Find a specific convolution layer:
conv = graph.query().name("conv1", exact=True)

# Immediate downstream consumers:
consumers = conv.children()

# Immediate upstream producers:
producers = conv.parents()

# All upstream ancestors (bounded or deep traversal):
all_ancestors = conv.ancestors(max_depth=50)

# All downstream descendants:
all_descendants = conv.descendants(max_depth=50)

# Graph boundary nodes:
entry_points = graph.query().entry_nodes  # Consumes model inputs
exit_points = graph.query().output_nodes   # Produces model outputs
```

---

## 4. Boolean Set Operations

`NeuronQuery` instances overload Python bitwise operators (`|`, `&`, `-`) to enable graph algebra:

```python
q = graph.query()

conv_nodes = q.op("Conv")
bn_nodes = q.op("BatchNormalization")
relu_nodes = q.op("Relu")

# Union (|): All Convs, BNs, and ReLUs
conv_bn_relu = conv_nodes | bn_nodes | relu_nodes

# Difference (-): All Convs except 1x1 convolutions
conv1x1 = conv_nodes.attr("kernel_shape", [1, 1])
non_1x1_convs = conv_nodes - conv1x1

# Intersection (&):
large_and_grouped = (
    conv_nodes.attr("group", 2) & conv_nodes.attr("kernel_shape", [7, 7])
)
```

---

## 5. Topological Sorting

Ensure nodes follow valid execution ordering:

```python
# Sort current selection in global topological order:
sorted_query = graph.query().op("Conv").topological_sort()

# Validate if current order satisfies topological constraints:
if sorted_query.is_topologically_sorted():
    print("Nodes are in valid topological order.")
```

---

## 6. Accessing Results

Extract matched nodes or execute callbacks:

```python
q = graph.query().op("Conv")

# Total count
total = q.count() # or len(q)

# List of unique operator types present in query
types = q.op_types() # e.g. ["Conv"]

# Fetch first or last node
first_conv = q.first()
last_conv = q.last()

# Extract single node (raises ValueError if count != 1)
single_head = graph.query().name("classifier", exact=True).single_node

# Fetch weight parameters for selected nodes:
params = q.tensor()

# Retrieve all nodes as a list:
node_list = q.all() # or list(q)
```
