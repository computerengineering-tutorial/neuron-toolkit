# Declarative JSON Rules Engine & Pattern Rewriting

The `neuron_toolkit.rules` package introduces a declarative, JSON-driven graph transformation engine. Instead of writing imperative Python code for every graph rewrite, compiler engineers can specify graph patterns, topological constraints, and replacements as declarative JSON descriptors.

---

## Architecture of `JSONRuleEngine`

The `JSONRuleEngine` loads declarative JSON rule files, matches target subgraphs against specified node patterns and topological conditions, and executes graph transformations:

```python
import onnx
from neuron_toolkit.rules.engine import JSONRuleEngine

# Instantiate engine
engine = JSONRuleEngine(mode="function_proto")

# Load all built-in JSON rewrite rules
rules = engine.load_rules_from_dir()
print(f"Loaded {len(rules)} declarative JSON rules.")

# Apply transformations to an ONNX model
model = onnx.load("raw_model.onnx")
transformed_model = engine.apply(model)

# Or apply specific rules selectively by rule ID
transformed_model = engine.apply(
    model,
    rule_ids=["erf_gelu_canonicalization", "feedforward_swiglu_fusion"]
)
```

---

## Declarative Rule Schema

Each JSON rule file defines four core sections:

1. **Metadata**: `rule_id`, `name`, `description`, `version`.
2. **Pattern**: Subgraph node definitions with opcode constraints, wildcards, and input-output connectivity.
3. **Conditions**: Predicates on tensor shapes, attributes, or constant values.
4. **Action**: Target replacement specification (`replace_subgraph`, `fuse_to_custom_op`, or `fold_constant`).

### Example Rule Descriptor: `erf_gelu_canonicalization.json`

```json
{
  "rule_id": "erf_gelu_canonicalization",
  "name": "ERF GELU Canonicalization",
  "description": "Fuses 7-node standard GELU expansion into standard Gelu or custom FunctionProto",
  "pattern": {
    "nodes": [
      {
        "id": "div_sqrt2",
        "op_type": "Div",
        "inputs": ["$input", "$const_sqrt2"],
        "outputs": ["$div_out"]
      },
      {
        "id": "erf_node",
        "op_type": "Erf",
        "inputs": ["$div_out"],
        "outputs": ["$erf_out"]
      },
      {
        "id": "add_one",
        "op_type": "Add",
        "inputs": ["$erf_out", "$const_one"],
        "outputs": ["$add_out"]
      },
      {
        "id": "mul_half",
        "op_type": "Mul",
        "inputs": ["$input", "$const_half"],
        "outputs": ["$mul_half_out"]
      },
      {
        "id": "mul_final",
        "op_type": "Mul",
        "inputs": ["$mul_half_out", "$add_out"],
        "outputs": ["$final_out"]
      }
    ]
  },
  "action": {
    "type": "fuse_to_custom_op",
    "domain": "custom.pytorch",
    "op_type": "Gelu",
    "inputs": ["$input"],
    "outputs": ["$final_out"]
  }
}
```

---

## Built-In Rules Catalog

The toolkit includes 29+ pre-built JSON rule definitions in `neuron_toolkit/rules/definitions/`:

| Rule Category | Rule ID | Description |
| --- | --- | --- |
| **Activations** | `erf_gelu_canonicalization` | Fuses Div -> Erf -> Add -> Mul -> Mul into `Gelu` |
| | `tanh_gelu_canonicalization` | Fuses Tanh polynomial approximation into `Gelu(approx='tanh')` |
| | `silu_swish_canonicalization` | Fuses Mul + Sigmoid into `SiLU` |
| | `hard_swish_canonicalization` | Fuses Mul + Clip + Div into `HardSwish` |
| | `clip_to_relu6_canonicalization` | Fuses Clip(0, 6) into `Relu6` |
| | `mish_activation_canonicalization` | Fuses Mul + Tanh + Softplus into `Mish` |
| **Normalizations** | `rmsnorm_meansquare_fusion` | Fuses Pow(2) -> ReduceMean -> Sqrt -> Div -> Mul into `RMSNorm` |
| | `rmsnorm_reducel2_fusion` | Fuses ReduceL2 -> Div -> Mul into `RMSNorm` |
| | `layernorm_mean_var_fusion` | Fuses Sub(Mean) -> Div(Var) -> Mul + Add into `LayerNorm` |
| | `groupnorm_instancenorm_fusion` | Fuses Reshape -> InstanceNorm -> Reshape into `GroupNorm` |
| | `skip_rmsnorm_fusion` | Fuses Add(Residual) directly before RMSNorm into `SkipRMSNorm` |
| | `skip_layernorm_fusion` | Fuses Add(Residual) directly before LayerNorm into `SkipLayerNorm` |
| **Attention & Linear** | `scaled_dot_product_attention_fusion` | Fuses MatMul + Scale + Mask + Softmax + MatMul into `SDPA` |
| | `linear_matmul_add_fusion` | Fuses MatMul + Add into `Linear` |
| | `linear_gemm_canonicalization` | Canonicalizes Gemm with transposed weights into `Linear` |
| | `split_qkv_fusion` | Fuses sliced projection outputs into `SplitQKV` |
| | `merge_heads_fusion` | Fuses Reshape + Transpose head merges into `MergeHeads` |
| | `rope_frequency_slice_fusion` | Fuses complex Rotary Embedding frequency slices into `RoPE` |
| **MLP Blocks** | `feedforward_gelu_fusion` | Fuses Norm -> Linear -> Gelu -> Linear into `FeedForward` |
| | `feedforward_swiglu_fusion` | Fuses Norm -> (Linear *SiLU* Linear) -> Linear into `FeedForwardSwiGLU` |
| | `gated_unit_fusion` | Fuses Split -> Norm -> Linear -> Mul into `GatedUnit` |
| **Graph Cleanups** | `cast_initializer_folding` | Folds static initializer `Cast` nodes at compile time |
| | `common_subexpression_elimination` | Prunes redundant identical computation subtrees |
| | `continuous_math_collapse` | Folds consecutive constant arithmetic into folded constants |
| | `dead_code_elimination` | Eliminates unreachable subgraphs |
| | `identity_elimination` | Removes non-functional `Identity` forwarding nodes |

---

## Dual-Mode Model Export: `export_simplified_onnx`

After applying transformations, models can be exported in Dual-Mode using `export_simplified_onnx`:

```python
from neuron_toolkit.rules.exporter import export_simplified_onnx

# Export with Netron FunctionProtos attached
export_simplified_onnx(
    model_or_graph=transformed_model,
    output_path="model_netron_viewable.onnx",
    mode="function_proto",
    metadata={"author": "NeuronCompiler", "version": "1.0"}
)
```

---

## Direct Integration on `NeuronGraph`

The declarative JSON rule engine is directly integrated into the high-level `NeuronGraph` interface:

```python
from neuron_toolkit import NeuronGraph

graph = NeuronGraph.load("model.onnx")

# Apply all JSON rules in FunctionProto mode:
optimized_graph = graph.apply_json_rules(mode="function_proto")

# Apply selective rules:
targeted_graph = graph.apply_json_rules(
    rules=["rmsnorm_meansquare_fusion", "scaled_dot_product_attention_fusion"],
    mode="function_proto"
)
```

---

## Authoring and Testing Custom JSON Rules

To add a new rewrite rule:

1. Create a `.json` descriptor in `src/neuron_toolkit/rules/definitions/<rule_name>.json`.
2. Define the subgraph node topology under `"pattern"`, specifying placeholder inputs like `"$x"` and constants like `"$const"`.
3. Add any type or value constraints in `"conditions"`.
4. Specify the target replacement under `"action"` (`"type": "fuse_to_custom_op"` or `"replace_subgraph"`).
5. Add a test descriptor in `tests/rules_json_tests/<rule_name>_test.json` verifying input and output node counts.
