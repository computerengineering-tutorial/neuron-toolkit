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
- **`passes -> GraphPasses`**: Namespace exposing pre-packaged graph optimization passes (`passes.const_fold()`, `passes.cse()`, `passes.dce()`, `passes.fuse_conv_act()`, `passes.fuse_conv_bn()`, `passes.fuse_gelu()`, `passes.fuse_layer_norm()`, `passes.fuse_rmsnorm()`, `passes.fuse_swiglu()`, `passes.fuse_attention()`, `passes.fuse_rope()`).

### Methods

- **`query() -> NeuronQuery`**: Creates an unbound query covering all nodes in the graph.
- **`match(pattern: Pattern) -> MatchResult | None`**: Finds the first match of *pattern* in the graph.
- **`findall(pattern: Pattern) -> list[MatchResult]`**: Finds all occurrences of *pattern* across the graph.
- **`replace(pattern: Pattern, new_op: str, name: str | None = None, **attrs: object) -> NeuronRewriter`**: Stages the replacement of all matches of *pattern* with a new operator.
- **`rewriter() -> NeuronRewriter`**: Spawns a transaction-staged graph rewriter bound to this model.
- **`to_ir() -> IRGraph`**: Converts this graph into a framework-agnostic `IRGraph`.
- **`optimize(passes: list[BasePass] | PassManager | None = None) -> tuple[NeuronGraph, PassStats]`**: Applies optimization passes.
- **`equality_saturation(rules: list[EGraphRule] | None = None, cost_model: CostModel | None = None, max_iterations: int = 10, node_limit: int = 5000) -> tuple[IRGraph, dict[str, Any]]`**: Runs equality saturation over an E-Graph and extracts the optimal program DAG.
- **`profile(device: str | HardwareTarget = "a100", dtype: str = "float16") -> ProfilerReport`**: Computes FLOPs, MACs, active working set memory, and Roofline model bottleneck latency predictions.
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
- **`apply_json_rules(rules: list[str] | None = None, mode: str = "function_proto") -> NeuronGraph`**: Applies declarative JSON rewrite rules (Cast folding, GELU, RMSNorm, SDPA, SwiGLU).
- **`run_pytorch(inputs: dict[str, Any], device: str = "cpu") -> dict[str, Any]`**: Executes the graph natively using the standalone PyTorch execution engine.
- **`cross_verify(simplified_graph: NeuronGraph | object | str, inputs: dict[str, Any] | None = None, rtol: float = 1e-4, atol: float = 1e-4) -> CrossVerificationReport`**: Cross-verifies numerical output parity between this model on ONNX Runtime and a simplified/rewritten model on PyTorchRunner.
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

### `neuron_toolkit.pattern.PatternDetector`

Graph matcher executing pattern searches with indexed candidate root lookup.

- **`match(pattern: Pattern) -> MatchResult | None`**: Finds the first match of *pattern*.
- **`find_all(pattern: Pattern) -> list[MatchResult]`**: Finds all occurrences of *pattern* in the graph using root-op indexing for $O(1)$ candidate root filtering.
- **`op_to_nodes -> dict[str, list[object]]`**: Cached index mapping operator type strings to lists of graph nodes.

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
- **`run_until_fixed_point(graph: NeuronGraph, max_iterations: int = 10) -> tuple[NeuronGraph, PassStats]`**: Executes passes iteratively in a loop until the graph node count stabilizes (reaches a fixed point) or `max_iterations` is reached.

### Standard & Transformer Passes

- **`FuseConvBN()`**: Folds BatchNormalization scale/bias/mean/var into Conv weight/bias tensors.
- **`DeadCodeElimination()`**: Prunes unconsumed nodes not contributing to graph outputs.
- **`ConstantFolding()`**: Statically computes operations whose inputs are constant initializers.
- **`CommonSubexpressionElimination()`**: Merges duplicate operator nodes with identical inputs into a single node.
- **`FuseGELU()`**: Folds multi-op GELU subgraphs (Erf-based or Tanh-approximation) into unified `Gelu` operator.
- **`FuseLayerNorm()`**: Folds 7-op normalization subgraphs (ReduceMean, Sub, Pow, Div, Add, Mul) into `LayerNormalization`.
- **`FuseConvActivation()`**: Combines `Conv` followed by `Relu`, `LeakyRelu`, or `Clip` into `FusedConv`.
- **`FuseRMSNorm()`**: Fuses decomposed RMSNorm subgraphs (`Pow/Mul` $\to$ `ReduceMean` $\to$ `Add` $\to$ `Sqrt` $\to$ `Div` $\to$ `Mul`) into unified `RMSNorm`.
- **`FuseSwiGLU()`**: Fuses SiLU/Sigmoid gating branches and linear projections into `SwiGLU` / `GeGLU`.
- **`FuseAttention()`**: Fuses Scaled Dot-Product Attention ($QK^T \to \text{scale} \to \text{mask} \to \text{softmax} \to V$) into `ScaledDotProductAttention`.
- **`FuseRotaryEmbedding()`**: Fuses RoPE slicing, negation, concatenation, and $\sin/\cos$ rotations into `RotaryEmbedding`.

---

## 8. `neuron_toolkit.egraph` (Equality Saturation Engine)

- **`EGraph()`**: Core Equivalence Graph structure.
  - Properties & Methods:
    - `add(enode: ENode) -> EClassId`: Hash-consed operator insertion.
    - `union(id1: EClassId, id2: EClassId) -> EClassId`: Unions two equivalence classes.
    - `rebuild() -> int`: Runs incremental congruence closure.
    - `find(id: EClassId) -> EClassId`: Finds canonical representative.
    - `total_nodes() -> int`: Returns total count of ENodes across all classes.
    - `classes_for_op(op: str) -> list[tuple[EClassId, EClass]]`: Fast lookup returning canonical (class_id, eclass) pairs containing at least one node with opcode *op*.
    - `from_ir(ir: IRGraph) -> tuple[EGraph, dict[str, EClassId]]`: Ingests an IRGraph.
- **`ENode(op: str, children: tuple[EClassId, ...], attrs: tuple[tuple[str, Any], ...])`**: Immutable operator node.
- **`EClass(id: EClassId, nodes: set[ENode], parents: list, data: Any)`**: Equivalence class grouping equivalent ENodes.
- **`UnionFind`**: Disjoint-set forest with path compression and rank-based union.
- **`saturate(egraph: EGraph, rules: list[EGraphRule] | None = None, max_iterations: int = 12, node_limit: int = 5000) -> dict[str, Any]`**: Executes equality saturation loop until fixed-point or resource limit.
- **`Extractor(egraph: EGraph, cost_model: CostModel | None = None)`**: Dynamic programming DAG extractor.
  - `find_best(class_id: EClassId) -> tuple[float, ENode]`: Computes minimum cost and optimal node.
  - `extract_to_ir(root_class_ids: list[EClassId], graph_name: str) -> IRGraph`: Reconstructs optimal DAG.
- **Cost Models**: `NodeCountCostModel(op_weights=None)`, `ComputeCostModel()`.
- **Rewrite Rules**: `CommutativeRule`, `IdentityRule`, `ConstantFoldRule`, `SiluFusionRule`, `default_rules()`.

---

## 9. `neuron_toolkit.profiler` (Hardware Cost Modeling & Memory Planning)

- **`FLOPsProfiler.profile(graph: NeuronGraph, default_dtype: str = "float32") -> FLOPsSummary`**: Computes analytical FLOPs, MACs, parameter counts, and bytes transferred.
- **`MemoryTimelineSimulator.simulate(graph: NeuronGraph, default_dtype: str = "float32") -> MemoryTimelineSummary`**: Traces tensor birth and death steps, evaluating naive memory vs active working set peak memory and simulated buffer reuse potential.
- **`RooflineAnalysis.analyze(summary: FLOPsSummary, device: str | HardwareTarget = "a100", dtype: str = "float16") -> RooflineAnalysis`**: Calculates Arithmetic Intensity ($AI$), hardware ridge points, compute vs memory-bound classification, and predicted execution latency.
- **`profile_graph(graph: NeuronGraph, device: str | HardwareTarget = "a100", dtype: str = "float16") -> ProfilerReport`**: Convenience entrypoint.
- **`ProfilerReport`**: Unified report exposing `.summary()`, `.to_dict()`, and `.to_markdown()`.
- **Hardware Targets**: `NVIDIA_A100_SXM4_80GB`, `NVIDIA_H100_SXM5`, `NVIDIA_RTX_4090`, `APPLE_M3_MAX`, `ARM_CORTEX_A78`, `GENERIC_CPU`, and `HARDWARE_REGISTRY`.

---

## 10. `neuron_toolkit.pattern.RewriteRule`

Pairs a `Pattern` with a transformation callback or replacement operator.

```python
RewriteRule(
    pattern: Pattern,
    replacement: Callable[[MatchResult, NeuronRewriter], Any] | str,
    name: str = "RewriteRule",
)
```

---

## 11. `neuron_toolkit.ir` (Unified Intermediate Representation)

- **`IRGraph(name: str = "main_graph")`**: Framework-agnostic computational DAG.
  - Properties: `producers -> dict[str, IRNode]`, `consumers -> dict[str, list[IRNode]]`.
  - Methods: `add_node()`, `add_tensor()`, `is_topologically_sorted()`, `to_dict()`, `from_dict()`.
- **`IRNode(op_type: str, name: str, inputs: list[str], outputs: list[str], attributes: dict, domain: str)`**: Strongly-typed compute node.
- **`IRTensor(name: str, shape: tuple, dtype: str, data: np.ndarray | None, is_initializer: bool)`**: Strongly-typed tensor metadata.
- **`BackendAdapter`**: Protocol for format adapters (`ONNXAdapter`, `TFLiteAdapter`).

---

## 12. `neuron_toolkit.viz`

- **`to_mermaid(graph: NeuronGraph, max_nodes: int = 150) -> str`**: Emits GitHub-flavored Mermaid flowchart diagram.
- **`to_html(graph: NeuronGraph, output_path: str | None = None) -> str`**: Emits standalone, zero-dependency interactive HTML visualizer.

---

## 13. `neuron_toolkit.eval.GraphVerifier`

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

## 14. Exceptions (`neuron_toolkit.exceptions`)

- **`NeuronToolkitError`**: Base class for all library exceptions.
- **`GraphParseError`**: Raised on invalid or corrupt model files.
- **`QueryError`**: Raised on malformed query operations.
- **`PatternMatchError`**: Raised on invalid pattern definitions.
- **`RewriteError`**: Raised when graph surgery fails topological validation.
- **`VerificationMismatchError`**: Raised when differential outputs diverge.
- **`BackendNotAvailableError`**: Raised when an optional backend dependency is missing.

---

## 15. `neuron_toolkit.compiler`

High-level 3-stage optimization pipeline and compilation layer for ONNX and unified graphs.

### `NeuronCompiler`

Class-based front-end compiler entry point.

- **`compile(model: onnx.ModelProto | NeuronGraph, mode: str = 'function_proto', rule_ids: list[str] | None = None, opset_version: int = 23, options: dict[str, Any] | None = None) -> onnx.ModelProto`**:
  Executes the full 3-stage optimization pipeline on the model and exports an optimized ONNX ModelProto.

### `OptimizationPipeline`

Enterprise-grade 3-stage optimization pipeline driver orchestrating canonicalization, synthesis, and export.

- **`__init__(mode: str = 'function_proto') -> None`**: Initializes pipeline with target representation mode (`'function_proto'` or `'plain'`).
- **`run(target: onnx.ModelProto | NeuronGraph, context: CompilerContext | None = None) -> onnx.ModelProto`**:
  Runs Stage 1 (Micro-Canonicalization), Stage 2 (Macro-Block Synthesis), and Stage 3 (Target Export).

### `CompilerContext`

Execution context and metrics tracker for compiler passes.

- **`record_pass(pass_name: str, nodes_before: int, nodes_after: int) -> None`**: Records node delta metrics for a compiler pass.
- **`should_run(rule_or_pass_id: str) -> bool`**: Returns boolean indicating whether a pass should be executed according to rule filtering options.

### Canonicalization Passes (`neuron_toolkit.compiler.passes.canonicalization`)

- **`ActivationCanonicalizer`**: Fuses mathematical expansions into standard activations (`apply_erf_gelu_fusion`, `apply_tanh_gelu_fusion`, `apply_silu_fusion`, `apply_hardswish_fusion`, `apply_relu6_fusion`, `apply_mish_fusion`, `apply_softmax_fusion`).
- **`MathCanonicalizer`**: Canonicalizes linear projections and matrix arithmetic (`apply_linear_fusion`, `apply_continuous_math_collapse`).
- **`NormCanonicalizer`**: Normalizes and fuses normalization expansions across opset versions (`apply_rmsnorm_fusion`, `apply_layernorm_fusion`, `apply_groupnorm_fusion`, `apply_skip_norm_fusion`).
- **`ShapeCanonicalizer`**: Optimizes tensor shapes, deduplicates subtrees, and removes unreachable nodes (`apply_common_subexpression_elimination`, `apply_dead_code_elimination`, `apply_cast_initializer_folding`, `apply_identity_elimination`).
- **`StaticFolder`**: Evaluates static constant subgraphs and propagates folded constants.

### Synthesis Passes (`neuron_toolkit.compiler.passes.synthesis`)

- **`AttentionSynthesizer`**: Synthesizes high-level attention primitives (`apply_scaled_dot_product_attention_fusion`, `apply_split_qkv_fusion`, `apply_merge_heads_fusion`, `apply_rope_fusion`).
- **`FeedForwardSynthesizer`**: Synthesizes transformer MLP and FFN blocks (`apply_feedforward_fusion`, `apply_feedforward_swiglu_fusion`).
- **`MacroBlockSynthesizer`**: Synthesizes neural blocks for vision, diffusion, and state-space architectures (`apply_deep_mlp_norm_act_fusion`, `apply_spatial_gating_unit_fusion`, `apply_highway_block_fusion`, `apply_squeeze_excitation_fusion`, `apply_mha_composite_fusion`).
- **`AxialSynthesizer`**: Synthesizes Conformer and multi-dimensional tensor arrangements (`apply_axial_rearrange_fusion`, `apply_unbind_fusion`).

### Exporters (`neuron_toolkit.compiler.exporters`)

- **`OnnxExporter(mode: str = 'function_proto')`**: Exports optimized model in dual-mode (FunctionProto or plain custom ops).
  - **`export(model: onnx.ModelProto, context: CompilerContext | None = None, output_path: str | Path | None = None) -> onnx.ModelProto`**
- **`PyTorchExporter(registry: OpRegistry | None = None)`**: Exports model directly into an executable `PyTorchRunner`.
  - **`export(model: onnx.ModelProto, context: CompilerContext | None = None) -> PyTorchRunner`**

---

## 16. `neuron_toolkit.compiler.architectures`

Subgraph discovery, declarative block signatures, and polymorphic fingerprinting.

### `ArchitectureCatalog`

Manages declarative architecture block signatures loaded from JSON.

- **`load() -> None`**: Loads block signatures from catalog file.
- **`get_signature(name: str) -> BlockSignature | None`**: Retrieves specific signature by identifier.
- **`list_signatures() -> list[str]`**: Lists all registered signature names.

### Fingerprinting Utilities (`neuron_toolkit.compiler.architectures.fingerprint`)

- **`compute_subgraph_fingerprint(nodes: list[onnx.NodeProto], external_inputs: list[str] | None = None, external_outputs: list[str] | None = None) -> str`**: Calculates deterministic 8-character SHA-256 fingerprint from subgraph topology and opcode sequence.
- **`build_monomorphized_op_type(family: str, variant_hint: str | None = None, fingerprint: str | None = None) -> str`**: Constructs monomorphized op name (e.g. `FeedForward_swiglu_a1b2c3d4`).
- **`ensure_specialized_function_proto(model: onnx.ModelProto, domain: str, fname: str, formal_inputs: list[str], formal_outputs: list[str], body_nodes: list[onnx.NodeProto], opset_imports: list[onnx.OperatorSetIdProto] | None = None, doc_string: str = '') -> onnx.FunctionProto`**: Registers specialized FunctionProto in `model.functions` if not present.

---

## 17. `neuron_toolkit.rules`

Declarative JSON pattern rules and graph rewriting engine.

### `JSONRuleEngine`

Facade engine that loads declarative JSON rewrite rules and transforms ONNX models.

- **`__init__(mode: str = 'function_proto') -> None`**
- **`load_rules_from_dir(directory: str | Path | None = None) -> list[JSONRule]`**: Loads all declarative JSON rules from directory (defaults to built-in `definitions/`).
- **`load_rule_from_file(path: str | Path) -> JSONRule`**: Loads and registers a single JSON rule file.
- **`apply(target: onnx.ModelProto | NeuronGraph, rule_ids: list[str] | None = None) -> onnx.ModelProto`**: Transforms the model applying all loaded or specified rules.

### Schema Entities (`neuron_toolkit.rules.schema`)

- **`JSONRule`**: Declarative rule specification containing metadata, pattern, conditions, and action.
- **`NodePattern`**: Subgraph node matching specification with opcode, input/output placeholders, and attribute filters.
- **`RuleCondition`**: Predicate condition on tensors, shapes, constants, or data types.
- **`RuleAction`**: Action definition specifying graph replacement or custom operator fusion.

### Exporter Function

- **`export_simplified_onnx(model_or_graph: onnx.ModelProto | NeuronGraph, output_path: str, mode: str = 'function_proto', metadata: dict[str, Any] | None = None) -> onnx.ModelProto`**: Exports transformed model in dual-mode representation with attached FunctionProtos.

---

## 18. `neuron_toolkit.runtime`

Native PyTorch execution runtime and differential cross-verification engine.

### `PyTorchRunner`

Vectorized, native PyTorch execution runtime for standard and rewritten ONNX graphs with custom operators.

- **`__init__(model_or_graph: onnx.ModelProto | NeuronGraph, registry: OpRegistry | None = None, device: str | torch.device = 'cpu') -> None`**
- **`run(inputs: dict[str, np.ndarray | torch.Tensor]) -> dict[str, np.ndarray]`**: Executes the computational graph using PyTorch and returns dictionary of numpy array outputs.

### `OpRegistry`

Registry of PyTorch implementations for ONNX standard and custom operations.

- **`register(domain: str, op_type: str, handler: OpHandler | None = None) -> Callable[[OpHandler], OpHandler] | None`**: Decorator / method to register a custom op handler.
- **`get(domain: str, op_type: str) -> OpHandler | None`**: Resolves handler for given domain and opcode.

### `CrossVerifier`

Differential verification suite comparing original models executed on ONNX Runtime against simplified/compiled models on `PyTorchRunner`.

- **`__init__(original_model: onnx.ModelProto | NeuronGraph | str, simplified_model: onnx.ModelProto | NeuronGraph | str, rtol: float = 0.0001, atol: float = 0.0001, min_cosine_similarity: float = 0.9999) -> None`**
- **`generate_dummy_inputs(batch_size: int = 1, default_dynamic_dim: int = 32) -> dict[str, np.ndarray]`**: Generates valid random inputs conforming to graph input tensor specifications.
- **`verify(inputs: dict[str, np.ndarray] | None = None) -> CrossVerificationReport`**: Executes both engines and returns comprehensive numerical divergence report.

### `CrossVerificationReport` & `OutputMetric`

- **`report.all_passed: bool`**: Overall pass/fail status across all output tensors.
- **`report.original_nodes: int`**: Node count in original unoptimized model.
- **`report.simplified_nodes: int`**: Node count in optimized model.
- **`report.eliminated_nodes: int`**: Net nodes removed.
- **`report.reduction_percentage: float`**: Percentage reduction in node count.
- **`report.standard_nodes: int`**: Count of remaining standard ONNX nodes.
- **`report.custom_nodes: int`**: Count of custom operator nodes.
- **`report.custom_ratio: float`**: Percentage of custom operator nodes in optimized model.
- **`report.custom_op_breakdown: dict[str, int]`**: Frequencies of active custom operators.
- **`report.metrics: dict[str, OutputMetric]`**: Per-output verification metrics:
  - `metric.name: str`: Output tensor name.
  - `metric.shape: tuple[int, ...]`: Shape of output tensor.
  - `metric.max_abs_diff: float`: Maximum absolute deviation between ORT and PyTorchRunner.
  - `metric.mse: float`: Mean squared error.
  - `metric.cosine_similarity: float`: Vector cosine similarity.
  - `metric.passed: bool`: Individual output validation status against `rtol`, `atol`, and `min_cosine_similarity`.
- **`report.summary() -> str`**: Formatted ASCII / Markdown verification table.

---

## 19. `neuron_toolkit.compiler.passes`

Advanced canonicalization, structural simplification, and modern attention synthesis passes.

### Canonicalization Passes (`neuron_toolkit.compiler.passes.canonicalization`)

- **`apply_reshape_collapsing(graph: onnx.GraphProto) -> int`**:
  Fuses consecutive linear/spatial reshapes ($X \to \text{Reshape}(S_1) \to \text{Reshape}(S_2) \implies X \to \text{Reshape}(S_2)$) when intermediate tensors have single fan-out.
- **`apply_pad_conv_fusion(graph: onnx.GraphProto) -> int`**:
  Fuses explicit spatial `Pad` operations preceding `Conv` nodes directly into the Conv operator's internal `pads` attribute for compatible constant zero-padding modes.
- **`apply_redundant_cast_elimination(graph: onnx.GraphProto) -> int`**:
  Eliminates redundant data type casts where the input tensor already matches the target `to` dtype or back-to-back identical cast operations.
- **`apply_squeeze_unsqueeze_cancellation(graph: onnx.GraphProto) -> int`**:
  Detects and cancels adjacent inverse squeeze and unsqueeze pairs operating along identical axes.

### Synthesis Passes (`neuron_toolkit.compiler.passes.synthesis`)

- **`apply_gemma2_softcapping_fusion(graph: onnx.GraphProto) -> int`**:
  Fuses Gemma-2 attention and output logits soft-capping sequences ($\text{cap} \cdot \tanh(x / \text{cap})$) into unified `custom.pytorch::LogitsSoftCap` nodes.
- **`apply_deepseek_mla_fusion(graph: onnx.GraphProto) -> int`**:
  Synthesizes DeepSeek-V2/V3 Multi-Head Latent Attention (MLA) low-rank KV compression projections into `custom.pytorch::MultiHeadLatentAttention_KVProjection`.

