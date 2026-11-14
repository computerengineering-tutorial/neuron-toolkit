# Guide: Pattern Matching DSL (`Pattern`)

The `Pattern` DSL allows you to define declarative subgraph templates to detect canonical architectural motifs (such as activation functions, residual blocks, normalization layers, and fusion candidates) across a computational graph.

---

## 1. Pattern Primitives

Import the `Pattern` class:

```python
from neuron_toolkit import Pattern
```

### Specific Operators (`Pattern.op`)
Specify an operator type and its upstream inputs:
```python
# Match a Relu node consuming any input:
relu_pat = Pattern.op("Relu", Pattern.any())

# Match a Conv followed by BatchNormalization:
conv = Pattern.op("Conv", Pattern.any())
bn = Pattern.op("BatchNormalization", conv)
```

### Wildcards (`Pattern.any`)
Matches any node regardless of op_type:
```python
wildcard = Pattern.any()
```

### Constant Matching (`Pattern.const`)
Matches weight initializers or constant tensors:
```python
# Match a constant float value 0.5:
half_const = Pattern.const(0.5)

# Match an Add where one input is constant 1.0:
add_one = Pattern.op("Add", Pattern.any(), Pattern.const(1.0))
```

### Alternative Branches (`Pattern.any_of`)
Matches any one of multiple alternative subgraphs:
```python
# Match either Relu or LeakyRelu:
act_pat = Pattern.any_of(Pattern.op("Relu"), Pattern.op("LeakyRelu"))
```

---

## 2. Captures & Constraints

### Named Captures (`.capture`)
Bind specific matched nodes to named variables for downstream inspection or rewriting:
```python
conv_node = Pattern.op("Conv", Pattern.any()).capture("conv_layer")
bn_node = Pattern.op("BatchNormalization", conv_node).capture("bn_layer")
```

### Attribute Constraints (`.where`)
Match nodes only if their attributes satisfy specific criteria:
```python
# Match Conv where group == 1 and dilations == [1, 1]
standard_conv = Pattern.op("Conv").where(group=1, dilations=[1, 1])
```

### Output Shape & Dtype Constraints
```python
# Match a node producing a 4D float32 tensor:
pat = (
    Pattern.op("Conv")
    .with_output_rank(4)
    .with_dtype("float32")
)
```

---

## 3. Arithmetic Operator Overloads

`Pattern` instances overload Python arithmetic operators (`+`, `-`, `*`, `/`, `**`, `neg`), allowing you to write mathematical graph formulas directly in Python:

```python
x = Pattern.any().capture("x")

# Swish / SiLU Activation: x * Sigmoid(x)
silu_pattern = x * Pattern.op("Sigmoid", x)

# Residual Connection: x + Conv(x)
conv_branch = Pattern.op("Conv", x)
residual_pattern = x + conv_branch

# Normalization step: (x - mean) / sqrt(var + eps)
mean = Pattern.any().capture("mean")
var = Pattern.any().capture("var")
normalized = (x - mean) / Pattern.op("Sqrt", var + Pattern.const(1e-5))
```

---

## 4. Built-in Activation Functions (`ActivationMixin`)

`Pattern` includes pre-built activation templates accessible directly as class methods:

| Method | Mathematical Definition / Pattern Structure |
| :--- | :--- |
| `Pattern.relu(x)` | `Relu(x)` |
| `Pattern.sigmoid(x)` | `Sigmoid(x)` |
| `Pattern.tanh(x)` | `Tanh(x)` |
| `Pattern.leaky_relu(x)` | `LeakyRelu(x)` |
| `Pattern.silu(x)` / `Pattern.swish(x)` | `x * Sigmoid(x)` |
| `Pattern.gelu(x)` | $0.5 \cdot x \cdot (1 + \text{Erf}(x / \sqrt{2}))$ |
| `Pattern.gelu_tanh(x)` | $0.5 \cdot x \cdot (1 + \tanh(\sqrt{2/\pi} \cdot (x + 0.044715 \cdot x^3)))$ |
| `Pattern.mish(x)` | `x * Tanh(Softplus(x))` |
| `Pattern.relu6(x)` | `Clip(x, 0.0, 6.0)` |
| `Pattern.softmax(x)` | `Softmax(x)` |

### Example: Detecting GeLU Activations

```python
x = Pattern.any().capture("gelu_input")
gelu_pattern = Pattern.gelu(x)

matches = graph.findall(gelu_pattern)
print(f"Detected {len(matches)} GELU activations.")
```

---

## 5. Executing Matches & Inspecting `MatchResult`

Run patterns against a `NeuronGraph` or `NeuronQuery`:

```python
# Find first occurrence
match = graph.match(silu_pattern)

# Find all occurrences
matches = graph.findall(silu_pattern)

for m in matches:
    print(f"Match root: {m.start.name} (Op: {m.start.op_type})")
    print(f"Matched nodes in subgraph: {[n.name for n in m.nodes]}")

    # Access captured nodes by name:
    captured_input = m.bindings["x"]
    print(f"Captured input node: {captured_input}")

    # Convert matched subgraph into a NeuronQuery for further exploration:
    subgraph_query = m.as_query()
    print("Subgraph parameters:", subgraph_query.has_params().count())
```

---

## 6. Commutative Matching & Diamond DAG Patterns

### Commutative Matching
Binary operations like `Add`, `Mul`, `Equal`, `And`, `Or`, `Xor`, `BitwiseAnd`, `BitwiseOr`, `BitwiseXor`, `Maximum`, and `Minimum` can have their inputs in either order in real-world graphs. `Pattern.commutative(...)` automatically matches regardless of operand ordering:

```python
# Matches Add(x, const) OR Add(const, x)
pat = Pattern.commutative("Add", Pattern.any().capture("x"), Pattern.const(1.0))
```

### Diamond DAG Multi-Branch Consistency
When pattern matching split-and-merge topologies (such as residual add blocks `Add(Conv(x), x)`), any named capture `.capture("x")` used across multiple branches is guaranteed to bind to the identical node instance across both branches. If the branches originate from different nodes, the pattern matcher automatically rejects the match, preventing false-positive cross-branch matches.
