# Guide: Graph Surgery, Rewriting & Optimization Passes

`neuron-toolkit` provides a transactional rewriter (`NeuronRewriter`), declarative transformation rules (`RewriteRule`), and a composable optimization pass pipeline (`PassManager`) to modify, insert, delete, and fuse nodes in computational graphs while preserving topological ordering and shape annotations.

---

## 1. Composable Optimization Pipeline (`PassManager`)

The modern way to run optimizations is using `PassManager`, which executes transformation passes sequentially and records node reduction statistics:

```python
from neuron_toolkit import NeuronGraph
from neuron_toolkit.passes import (
    CommonSubexpressionElimination,
    ConstantFolding,
    DeadCodeElimination,
    FuseConvActivation,
    FuseConvBN,
    FuseGELU,
    FuseLayerNorm,
    PassManager,
)

graph = NeuronGraph.load("model.onnx")

# Build a comprehensive optimization pipeline
pm = PassManager([
    ConstantFolding(),
    CommonSubexpressionElimination(),
    FuseConvBN(),
    FuseConvActivation(),
    FuseGELU(),
    FuseLayerNorm(),
    DeadCodeElimination(),
])

optimized_graph, stats = pm.run(graph)
print(stats.summary())
```

### Summary Output Example

```text
PassManager Execution Summary:
  Initial node count: 120
  Final node count:   76 (36.7% reduction)
  - ConstantFolding               : folded_nodes=6
  - CommonSubexpressionElimination: eliminated_nodes=4
  - FuseConvBN                    : fused_pairs=14
  - FuseConvActivation            : fused_conv_activation_count=12
  - FuseGELU                      : fused_gelu_count=4
  - FuseLayerNorm                 : fused_layernorm_count=2
  - DeadCodeElimination           : removed_nodes=6
```

### Direct Graph Pass Calls (`graph.passes`)

For quick, single-pass transformations without manually instantiating `PassManager`, use the `graph.passes` namespace:

```python
# Apply standard optimization passes:
g1 = graph.passes.const_fold()
g2 = g1.passes.cse()
g3 = g2.passes.fuse_gelu()
g4 = g3.passes.fuse_layer_norm()
g5 = g4.passes.fuse_conv_act()
g6 = g5.passes.dce()

# Apply transformer & LLM passes (see guides/transformer_passes.md):
g_llm = graph.passes.fuse_rmsnorm()
g_llm = g_llm.passes.fuse_swiglu()
g_llm = g_llm.passes.fuse_attention()
g_llm = g_llm.passes.fuse_rope()
```

### One-Line Pipeline (`graph.optimize`)

```python
# Automatically runs standard optimization passes
opt_graph, stats = graph.optimize()
opt_graph.save("optimized.onnx")
```

---

## 2. Declarative Transformation Engine (`RewriteRule`)

Instead of writing imperative loops over `findall()`, use declarative `RewriteRule` definitions applied iteratively until a fixed point is reached:

```python
from neuron_toolkit import NeuronGraph, Pattern, RewriteRule

graph = NeuronGraph.load("model.onnx")

# Define rule: x * Sigmoid(x) -> Silu(x)
x = Pattern.any().capture("x")
silu_pattern = x * Pattern.op("Sigmoid", x)

rule = RewriteRule(
    pattern=silu_pattern,
    replacement=lambda match, rewriter: rewriter.replace_from_result(
        match, "Silu", name="fused_silu"
    ),
    name="FuseSiLU",
)

# Apply rules iteratively across the graph
transformed_graph = graph.apply_rules([rule], max_iterations=10)
```

---

## 3. Low-Level Graph Surgery (`NeuronRewriter`)

Spawn a rewriter directly from any `NeuronGraph`:

```python
rw = graph.rewriter()
```

### Replacing Nodes

Replace one or more existing nodes with a new operator:

```python
old_add = graph.query().name("add1", exact=True).single_node

rw.replace(
    nodes=[old_add],
    new_op="Sub",
    inputs=list(old_add.input),
    outputs=list(old_add.output),
    name="sub_replacement",
)

new_graph = rw.build_graph(output_path="modified_model.onnx")
```

### Deleting Nodes

Completely remove obsolete or dead nodes from the graph:

```python
orphan_dropout = graph.query().op("Dropout").all()
rw.delete(orphan_dropout)
new_graph = rw.build_graph()
```

### Inserting Nodes

Insert a preprocessing or activation node before a target layer:

```python
target_conv = graph.query().name("conv2", exact=True).single_node

rw.insert_before(
    target_node=target_conv,
    new_op="Cast",
    inputs=["raw_input"],
    outputs=["cast_input"],
    to=1, # INT64 to FLOAT
)
new_graph = rw.build_graph()
```

---

## 4. Built-in Graph Passes Reference

| Pass Name | Description | Benefit |
| :--- | :--- | :--- |
| `FuseConvBN` | Folds BatchNormalization parameters (scale, bias, mean, var) into Conv weights | Eliminates runtime BN latency and intermediate activations |
| `ConstantFolding` | Statically computes ops where all inputs are initializers | Reduces inference-time arithmetic ops |
| `DeadCodeElimination` | Recursively removes orphan nodes not consumed by outputs | Prunes unreachable subgraphs and memory overhead |
| `CommonSubexpressionElimination` | Identifies duplicate operators with identical inputs and merges them into a single node | Eliminates redundant computation branches |
| `FuseGELU` | Folds multi-op GELU subgraphs (Erf-based or Tanh-approximation) into unified `Gelu` operator | Accelerates Transformer activation computation |
| `FuseLayerNorm` | Folds 7-op normalization subgraphs (ReduceMean, Sub, Pow, Div, Add, Mul) into `LayerNormalization` | Fuses memory-bandwidth-heavy reduction chains |
| `FuseConvActivation` | Folds `Conv` followed by `Relu`, `LeakyRelu`, or `Clip` into `FusedConv` | Minimizes memory bandwidth by fusing convolution and activation |
