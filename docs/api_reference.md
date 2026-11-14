# API Reference

Complete reference for all public classes, methods, and utility functions in `neuron-toolkit`.

---

## 1. `neuron_toolkit.NeuronGraph`

The primary class representing a loaded computational model (ONNX or TFLite).

### Constructors & Factories

::: python
#### `NeuronGraph(source: object, **kwargs: Any) -> NeuronGraph`
Initializes a `NeuronGraph` instance. Auto-detects input type:
- `str`: File path to a `.onnx`, `.tflite`, or JSON string.
- `bytes`: Model buffer (detects `TFL3` magic header for TFLite).
- `dict`: Pre-loaded JSON graph structure dictionary.
- `onnx.ModelProto`: In-memory ONNX protobuf.
- `tflite.Model`: In-memory parsed TFLite model.

#### `NeuronGraph.load(path: str, **kwargs: Any) -> NeuronGraph`
Loads a model from a file path based on its extension (`.onnx`, `.tflite`, `.bin`, `.json`).

#### `NeuronGraph.from_json(json_source: str | dict[str, Any], weights: str | dict[str, Any] | None = None, seed: int | None = None, **kwargs: Any) -> NeuronGraph`
Reconstructs a `NeuronGraph` from a JSON topology file or string, optionally initializing weights from a `.safetensors` file or reproducible random seed.
:::

### Core Properties

- **`nodes -> list[object]`**: List of all operator nodes in the graph.
- **`inputs -> list[str]`**: List of all graph input tensor names.
- **`outputs -> list[str]`**: List of all graph output tensor names.
- **`tensor_map -> dict[str, np.ndarray]`**: Dictionary mapping initializer/weight names to NumPy arrays.
- **`shape_info -> dict[str, tuple[int | None, str | None]]`**: Shape dimension ranks and dtype strings for all annotated graph tensors.
- **`passes -> GraphPasses`**: Namespace exposing pre-packaged graph optimization passes (`passes.const_fold()`, `passes.cse()`, `passes.dce()`, `passes.fuse_conv_act()`, `passes.fuse_conv_bn()`, `passes.fuse_gelu()`, `passes.fuse_layer_norm()`).

### Methods

- **`query() -> NeuronQuery`**: Creates an unbound query covering all nodes in the graph.
- **`match(pattern: Pattern) -> MatchResult | None`**: Finds the first match of *pattern* in the graph.
- **`findall(pattern: Pattern) -> list[MatchResult]`**: Finds all occurrences of *pattern* across the graph.
- **`replace(pattern: Pattern, new_op: str, name: str | None = None, **attrs: object) -> NeuronRewriter`**: Stages the replacement of all matches of *pattern* with a new operator.
- **`rewriter() -> NeuronRewriter`**: Spawns a transaction-staged graph rewriter bound to this model.
- **`to_ir() -> IRGraph`**: Converts this graph into a framework-agnostic `IRGraph`.
- **`optimize(passes: list[BasePass] | PassManager | None = None) -> tuple[NeuronGraph, PassStats]`**: Applies optimization passes.
- **`apply_rules(rules: list[RewriteRule] | RewriteRule, max_iterations: int = 10) -> NeuronGraph`**: Declaratively applies transformation rules.
- **`to_mermaid(max_nodes: int = 150) -> str`**: Emits GitHub-flavored Mermaid flowchart string.
- **`to_html(output_path: str | None = None) -> str`**: Exports an interactive standalone HTML visualization page.
- **`slice(start_points: list[str | object] | str, end_points: list[str | object] | str, output_path: str | None = None) -> NeuronGraph`**: Extracts a subgraph between boundary points as a new independent model.
- **`slice_node(node: str | object, output_path: str | None = None) -> NeuronGraph`**: Extracts a single isolated node into a new model.
- **`save(path: str) -> None`**: Serializes and writes the current model to disk.
- **`get_weights() -> dict[str, Any]`**: Returns all weight and parameter tensors in the graph.
- **`to_dict() -> dict[str, Any]`**: Exports graph structure into a Python dictionary with statistical weight summaries.
- **`to_json(path: str | None = None, indent: int = 2) -> str | None`**: Exports graph structure as a Netron-compatible JSON string or file.
- **`save_safetensors(path: str, metadata: dict[str, str] | None = None) -> None`**: Saves model parameters into a `.safetensors` file.
- **`load_safetensors(path: str, strict: bool = False) -> dict[str, list[str]]`**: Loads weights from a `.safetensors` file and replaces model initializers in-place.
- **`replace_weights(weights_dict: Mapping[str, Any], strict: bool = False) -> dict[str, list[str]]`**: Replaces weights in-place from a dictionary.
- **`compare_outputs(start_points, end_points, pytorch_module, inputs_dict, rtol=1e-5, atol=1e-8) -> dict[str, Any]`**: Executes sliced subgraph and reference PyTorch module on inputs, comparing numerical outputs (MAE, MSE, Cosine Similarity).
- **`verify_against_pytorch(start_points, end_points, rtol=1e-5, atol=1e-8) -> Callable`**: Function/method decorator asserting numerical equivalence with a PyTorch module.

---

## 2. `neuron_toolkit.query.NeuronQuery`

Fluent query builder for filtering and traversing model graphs.

### Filtering Methods
- **`filter(predicate: Callable[[object], bool]) -> NeuronQuery`**: Filters nodes using a custom function.
- **`op(op_type: str) -> NeuronQuery`**: Filters nodes by operator type.
- **`name(name: str, *, exact: bool = False) -> NeuronQuery`**: Filters nodes by name (substring or exact match).
- **`rank(rank: int) -> NeuronQuery`**: Filters nodes whose primary output tensor has the specified rank.
- **`dtype(dtype: str) -> NeuronQuery`**: Filters nodes whose primary output matches the given data type.
- **`attr(name: str, value: object = None) -> NeuronQuery`**: Filters nodes by attribute existence or value equality.
- **`has_params() -> NeuronQuery`**: Filters nodes that possess parameter/weight tensors.
- **`find_by_tensor(name: str) -> NeuronQuery`**: Filters nodes that produce or consume a given tensor.

### Traversal Methods
- **`children() -> NeuronQuery`**: Selects immediate downstream consumer nodes.
- **`parents() -> NeuronQuery`**: Selects immediate upstream producer nodes.
- **`ancestors(max_depth: int | None = None) -> NeuronQuery`**: Selects all upstream ancestor nodes (defaults to unbounded traversal).
- **`descendants(max_depth: int | None = None) -> NeuronQuery`**: Selects all downstream descendant nodes (defaults to unbounded traversal).
- **`entry_nodes -> NeuronQuery`**: Selects nodes consuming external graph inputs.
- **`output_nodes -> NeuronQuery`**: Selects nodes producing external graph outputs.

### Topological & Set Operations
- **`topological_sort() -> NeuronQuery`**: Sorts current nodes in global topological order.
- **`is_topologically_sorted() -> bool`**: Returns `True` if current nodes satisfy topological ordering.
- **`union(other: NeuronQuery) -> NeuronQuery`** (or `q1 | q2`): Union of node sets.
- **`intersection(other: NeuronQuery) -> NeuronQuery`** (or `q1 & q2`): Intersection of node sets.
- **`difference(other: NeuronQuery) -> NeuronQuery`** (or `q1 - q2`): Set difference of nodes.

### Result Accessors
- **`first() -> object | None`**: Returns the first node in the selection.
- **`last() -> object | None`**: Returns the last node in the selection.
- **`count() -> int`**: Returns the total number of selected nodes (supports `len(q)`).
- **`is_empty() -> bool`**: Returns `True` if no nodes are selected.
- **`op_types() -> list[str]`**: Returns unique operator type strings present in the selection.
- **`single_node -> object`**: Returns the single node (raises `ValueError` if `count() != 1`).
- **`single_tensor -> np.ndarray`**: Returns the single weight tensor for a single node.
- **`all() -> list[object]`**: Returns all selected nodes as a list (supports `iter(q)`).
- **`to_list() -> list[object]`**: Explicit alias for `all()`.

---

## 3. `neuron_toolkit.pattern.Pattern`

Declarative DSL for defining subgraph pattern motifs.

### Factory Constructors
- **`Pattern.op(op_type: str, *input_patterns: Pattern | object) -> Pattern`**: Matches an operator and its inputs.
- **`Pattern.commutative(op_type: str, *input_patterns: Pattern | object) -> Pattern`**: Matches a commutative binary operator regardless of input operand order.
- **`Pattern.any() -> Pattern`**: Wildcard pattern matching any node.
- **`Pattern.const(value: object) -> Pattern`**: Matches a constant tensor or initializer value.
- **`Pattern.any_of(*alternatives: Pattern) -> Pattern`**: Matches any of the given pattern alternatives.

### Chaining & Constraints
- **`capture(name: str) -> Pattern`**: Binds matched node to a named variable in `MatchResult.bindings`.
- **`where(**constraints: object) -> Pattern`**: Constrains node attributes to specified values.
- **`with_output_rank(rank: int) -> Pattern`**: Constrains output tensor rank.
- **`with_dtype(dtype: str) -> Pattern`**: Constrains output tensor dtype.
- **`then(op_type: str | Pattern) -> Pattern`**: Forward chains this pattern as input to another operator.

### Pre-Packaged Activation Mixins (`ActivationMixin`)
- `Pattern.relu(x)`: Rectified Linear Unit
- `Pattern.sigmoid(x)`: Sigmoid activation
- `Pattern.tanh(x)`: Hyperbolic tangent
- `Pattern.leaky_relu(x)`: Leaky ReLU with alpha parameter
- `Pattern.silu(x)` / `Pattern.swish(x)`: SiLU / Swish ($x \cdot \sigma(x)$)
- `Pattern.gelu(x)`: Standard Gaussian Error Linear Unit (Erf-based)
- `Pattern.gelu_tanh(x)`: Fast Tanh approximation of GELU
- `Pattern.mish(x)`: Mish ($x \cdot \tanh(\text{softplus}(x))$)
- `Pattern.relu6(x)`: Clip between 0.0 and 6.0
- `Pattern.softmax(x)`: Normalized exponential softmax
- `Pattern.elu(x)`: Exponential Linear Unit
- `Pattern.selu(x)`: Scaled Exponential Linear Unit
- `Pattern.softplus(x)`: Smooth approximation to ReLU ($\ln(1 + e^x)$)
- `Pattern.softsign(x)`: Softsign ($x / (1 + |x|)$)
- `Pattern.hardsigmoid(x)`: Piecewise linear approximation of sigmoid
- `Pattern.hardswish(x)`: Piecewise linear approximation of Swish
- `Pattern.prelu(x, slope)`: Parametric ReLU
- `Pattern.thresholded_relu(x)`: Thresholded ReLU
- `Pattern.log_softmax(x)`: Logarithm of softmax

---

## 4. `neuron_toolkit.pattern.MatchResult`

Contains the results of a successful pattern match.

### Fields
- **`start: object`**: The root/terminal node of the matched subgraph.
- **`end: object`**: The entry boundary node of the matched subgraph.
- **`nodes: list[object]`**: List of all nodes belonging to the matched pattern.
- **`bindings: dict[str, object]`**: Dictionary of named captures specified via `.capture()`.
- **`as_query() -> NeuronQuery`**: Converts the matched subgraph into a `NeuronQuery` instance.

---

## 5. `neuron_toolkit.rewriter.NeuronRewriter`

Transactional engine for graph surgery and AST rewriting.

### Methods
- **`replace(nodes: Sequence[object], new_op: str, inputs: list[str], outputs: list[str], name: str | None = None, **attrs: object) -> NeuronRewriter`**: Stages the replacement of `nodes` with a new operator.
- **`replace_from_result(result: MatchResult, new_op: str, inputs: list[str] | None = None, outputs: list[str] | None = None, name: str | None = None, **attrs: object) -> NeuronRewriter`**: Stages the replacement of a matched subgraph.
- **`delete(nodes: Sequence[object]) -> NeuronRewriter`**: Stages the deletion of nodes.
- **`insert_before(target_node: object, new_op: str, inputs: list[str], outputs: list[str], name: str | None = None, **attrs: object) -> NeuronRewriter`**: Stages insertion of a node before a target.
- **`register_initializer(name: str, value: Any) -> NeuronRewriter`**: Adds a new constant/weight tensor to the rewritten model.
- **`reset() -> NeuronRewriter`**: Clears all staged edits.
- **`build(output_path: str | None = None) -> object`**: Applies edits, sorts topologically, runs shape inference, and returns modified model.

---

## 6. `neuron_toolkit.exporter` Functions

- **`export_graph_dict(source: Any) -> dict[str, Any]`**: Exports graph topology to a dictionary without raw weights.
- **`export_graph_json(source: Any, path: str | None = None, indent: int = 2) -> str | None`**: Exports graph structure to JSON format.
- **`load_graph_from_json(json_source: str | dict[str, Any], weights_source: str | dict[str, Any] | None = None, seed: int | None = None, **kwargs: Any) -> NeuronGraph`**: Rebuilds graph from JSON (auto-detects ONNX or TFLite).
- **`build_onnx_model_from_json(json_dict_or_str: dict[str, Any] | str, weights: dict[str, np.ndarray] | str | None = None, seed: int | None = None) -> onnx.ModelProto`**: Reconstructs complete ONNX model.
- **`build_tflite_model_from_json(json_dict_or_str: dict[str, Any] | str, weights: dict[str, np.ndarray] | str | None = None, seed: int | None = None) -> bytes`**: Reconstructs complete runnable TFLite flatbuffer binary.
- **`compute_weight_stats(tensor: Any) -> dict[str, Any]`**: Calculates min, max, mean, std, zero fraction, NaN/Inf checks for a tensor.
- **`export_safetensors(source: Any, path: str, metadata: dict[str, str] | None = None) -> None`**: Writes weights into `.safetensors` format.
- **`load_safetensors(path: str) -> dict[str, np.ndarray]`**: Reads weights from `.safetensors` file into NumPy arrays.

---

## 7. `neuron_toolkit.passes`

### `PassManager(passes: list[BasePass] | None = None)`
Manages and executes an optimization pipeline. Supports indexing results by pass name or integer index (`stats["FuseConvBN"]` or `stats[0]`).
- **`add_pass(opt_pass: BasePass) -> PassManager`**: Registers a pass.
- **`run(graph: NeuronGraph) -> tuple[NeuronGraph, PassStats]`**: Applies all passes sequentially.

### Standard Passes
- **`FuseConvBN()`**: Folds BatchNormalization scale/bias/mean/var into Conv weight/bias tensors.
- **`DeadCodeElimination()`**: Prunes unconsumed nodes not contributing to graph outputs.
- **`ConstantFolding()`**: Statically computes operations whose inputs are constant initializers.
- **`CommonSubexpressionElimination()`**: Merges duplicate operator nodes with identical inputs into a single node.
- **`FuseGELU()`**: Folds multi-op GELU subgraphs (Erf-based or Tanh-approximation) into unified `Gelu` operator.
- **`FuseLayerNorm()`**: Folds 7-op normalization subgraphs (ReduceMean, Sub, Pow, Div, Add, Mul) into `LayerNormalization`.
- **`FuseConvActivation()`**: Combines `Conv` followed by `Relu`, `LeakyRelu`, or `Clip` into `FusedConv`.

---

## 8. `neuron_toolkit.pattern.RewriteRule`

Pairs a `Pattern` with a transformation callback or replacement operator.
```python
RewriteRule(
    pattern: Pattern,
    replacement: Callable[[MatchResult, NeuronRewriter], Any] | str,
    name: str = "RewriteRule",
)
```

---

## 9. `neuron_toolkit.ir` (Unified Intermediate Representation)

- **`IRGraph(name: str = "main_graph")`**: Framework-agnostic computational DAG.
  - Properties: `producers -> dict[str, IRNode]`, `consumers -> dict[str, list[IRNode]]`.
  - Methods: `add_node()`, `add_tensor()`, `is_topologically_sorted()`, `to_dict()`, `from_dict()`.
- **`IRNode(op_type: str, name: str, inputs: list[str], outputs: list[str], attributes: dict, domain: str)`**: Strongly-typed compute node.
- **`IRTensor(name: str, shape: tuple, dtype: str, data: np.ndarray | None, is_initializer: bool)`**: Strongly-typed tensor metadata.
- **`BackendAdapter`**: Protocol for format adapters (`ONNXAdapter`, `TFLiteAdapter`).

---

## 10. `neuron_toolkit.viz`

- **`to_mermaid(graph: NeuronGraph, max_nodes: int = 150) -> str`**: Emits GitHub-flavored Mermaid flowchart diagram.
- **`to_html(graph: NeuronGraph, output_path: str | None = None) -> str`**: Emits standalone, zero-dependency interactive HTML visualizer.

---

## 11. `neuron_toolkit.eval.GraphVerifier`

Differential numerical verification engine comparing sliced computational subgraphs against PyTorch reference layers or functions.

### Methods
- **`__init__(graph: NeuronGraph)`**: Initializes verifier bound to target graph.
- **`compare_outputs(start_points: list[str | object] | str | object, end_points: list[str | object] | str | object, pytorch_module: Any, inputs_dict: dict[str, Any], rtol: float = 1e-05, atol: float = 1e-08) -> dict[str, Any]`**:
  Executes the sliced subgraph and PyTorch module on provided input tensors and computes numerical divergence metrics:
  - `mae`: Mean Absolute Error
  - `mse`: Mean Squared Error
  - `cosine_similarity`: Cosine similarity between flattened output arrays
  - `all_close`: Boolean indicating whether outputs match within `rtol` and `atol`
  - `shape_match`: Boolean indicating whether output shapes match exactly
- **`verify_against_pytorch(start_points: list[str | object] | str | object, end_points: list[str | object] | str | object, rtol: float = 1e-05, atol: float = 1e-08) -> Callable`**:
  Function decorator that wraps a PyTorch callable, executes the matching subgraph on input tensors, and attaches `.similarity_report` containing the metrics report directly to the decorated function.
- **`_is_tflite_runnable() -> bool`**:
  Utility function checking whether a valid LiteRT / TFLite interpreter runtime is available on the host machine.

---

## 12. Exceptions (`neuron_toolkit.exceptions`)

- **`NeuronToolkitError`**: Base class for all library exceptions.
- **`GraphParseError`**: Raised on invalid or corrupt model files.
- **`QueryError`**: Raised on malformed query operations.
- **`PatternMatchError`**: Raised on invalid pattern definitions.
- **`RewriteError`**: Raised when graph surgery fails topological validation.
- **`VerificationMismatchError`**: Raised when differential outputs diverge.
- **`BackendNotAvailableError`**: Raised when an optional backend dependency is missing.
