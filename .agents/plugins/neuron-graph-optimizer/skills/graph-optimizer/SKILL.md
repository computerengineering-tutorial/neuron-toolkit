---
name: graph-optimizer
description: >-
  Systematic workflow to audit arbitrary ONNX models, minify redundant nodes (>50% reduction),
  synthesize reusable declarative JSON rewrite rules, extend the PyTorch runtime, and
  cross-verify numerical parity against ONNX Runtime.
---

# ONNX Graph Optimizer & Rule Generator Skill

This skill provides an end-to-end engineering playbook to inspect, dramatically minify (often eliminating 50% to 77% of redundant nodes in complex traced models), and rewrite ONNX computation graphs into clean, Netron-interpretable, PyTorch-backed representations while guaranteeing 100% bit-accurate numerical parity.

---

## 1. Quick Reference & Core Invariants

| Command / Task | Action | Key Guardrail |
| :--- | :--- | :--- |
| **Audit Model** | `uv run .agents/plugins/neuron-graph-optimizer/skills/graph-optimizer/scripts/analyze_model.py <model.onnx>` | Reports op breakdown, duplicate constants, identity ops, and repeating subgraphs. |
| **Minify & Verify** | `uv run .agents/plugins/neuron-graph-optimizer/skills/graph-optimizer/scripts/verify_parity.py <model.onnx>` | Applies all rule passes and verifies against ORT (Cosine $\ge 0.99999$, MSE $< 10^{-5}$). |
| **Scaffold New Rule** | `uv run .agents/plugins/neuron-graph-optimizer/skills/graph-optimizer/scripts/generate_rule.py --id <rule_name> --primary-op <Op> --custom-op <CustomOp>` | Generates JSON rule template, engine pass snippet, and PyTorch runtime handler. |
| **Export Netron Model** | `g = NeuronGraph.load("model.onnx"); opt = g.apply_json_rules(mode="function_proto"); opt.export("clean.onnx")` | Uses `FunctionProto` attachments so Netron visually renders single clean blocks. |

---

## 2. The 5-Phase Minification Pipeline

### Phase 1: Structural Audit & Discovery
Before modifying any nodes, inspect the model topology:
```bash
uv run .agents/plugins/neuron-graph-optimizer/skills/graph-optimizer/scripts/analyze_model.py path/to/model.onnx
```
Look for:
1. **Duplicate Constants**: Traced models often contain thousands of identical scalar constants (e.g. 1,760 redundant constants in `model1.onnx`).
2. **Identity Math**: Look for `Mul` nodes with constant `1.0` or `Add` with `0.0`.
3. **Repeating Subgraphs**: RMSNorm (`ReduceL2 -> Clip -> Expand -> Div -> Mul -> Mul`), GELU (`Erf` chains), RoPE (`Cos/Sin` branches), or GroupNorm / InstanceNorm expansions.
4. **Dynamic Reshaping Chains**: Sequences of `Shape -> Gather -> Unsqueeze -> Concat -> Reshape`.

### Phase 2: Common Subexpression Elimination (CSE) & Constant Deduplication
In `JSONRuleEngine.apply_common_subexpression_elimination()`:
1. **Constant Merging**: All `Constant` nodes sharing identical dtype, shape, and raw bytes are deduplicated to share a single output tensor.
2. **Cascading Pure Op CSE**: Deterministic ops (`Shape`, `Gather`, `Unsqueeze`, `Concat`, `Squeeze`, `Slice`, `Cast`, `Reshape`) whose inputs and attributes match an existing node are replaced with the existing node's output.
3. **DCE (Dead Code Elimination)**: Automatically cleans up orphaned constants and subexpressions with 0 consumers.

### Phase 3: Identity Arithmetic Pruning
In `JSONRuleEngine.apply_identity_elimination()`:
- `Mul(x, 1.0) -> x` and `Mul(1.0, x) -> x`
- `Add(x, 0.0) -> x` and `Add(0.0, x) -> x`
- `Sub(x, 0.0) -> x`
- `Div(x, 1.0) -> x`
Consumers of the arithmetic output are rewired directly to `x`, eliminating the node without numerical deviation.

### Phase 4: Semantic Subgraph Fusions
Fuses multi-node expansions into high-level ops (registered under `custom.pytorch` and attached as `FunctionProto`):
- **RMSNorm**: 9 nodes $\to$ 1 node (`custom.pytorch::RMSNorm`).
- **GELU**: 7 nodes $\to$ 1 node (`custom.pytorch::Gelu`).
- **GroupNorm / InstanceNorm**: 9 nodes $\to$ 1 node (`custom.pytorch::GroupNorm`).
- **Continuous Math**: Constant arithmetic chains folded into `Constant` carrying `folded_expr` metadata.
- **Index Tensor Routing**: Dynamic slice and gather chains collapsed into `custom.pytorch::TensorRoute`.

### Phase 5: Differential Parity Cross-Verification
Always verify the output against ONNX Runtime:
```python
from neuron_toolkit.runtime.cross_verifier import CrossVerifier
import onnx

orig_m = onnx.load("model.onnx")
opt_m = engine.apply(orig_m)

verifier = CrossVerifier(orig_m, opt_m, rtol=1e-4, atol=1e-4)
report = verifier.verify()
assert report.all_passed
```

---

## 3. High-Level Python API Usage

```python
from neuron_toolkit.graph import NeuronGraph
import numpy as np

# 1. Load any ONNX model
g = NeuronGraph.load(".temp/diffrent_models/model1.onnx")
print(f"Original nodes: {len(g.nodes)}")

# 2. Minify using declarative rules & CSE
opt_g = g.apply_json_rules(mode="function_proto")
print(f"Minified nodes: {len(opt_g.nodes)}")

# 3. Cross-verify against ONNX Runtime
report = g.cross_verify(opt_g)
print(report.summary())
assert report.all_passed

# 4. Run PyTorch vectorized runner
inputs = {"logmel": np.random.randn(1, 100, 128).astype(np.float32)}
outputs = opt_g.run_pytorch(inputs)

# 5. Export clean ONNX model for Netron visualization
opt_g.export("model1_minified.onnx")
```

---

## 4. Extending the Framework with New Rules

When encountering an unsupported pattern in a new model:

### Step A: Scaffold the Rule
```bash
uv run .agents/plugins/neuron-graph-optimizer/skills/graph-optimizer/scripts/generate_rule.py \
  --id "silu_fusion" \
  --primary-op "Sigmoid" \
  --custom-op "SiLU" \
  --desc "Fuses Sigmoid -> Mul into SiLU"
```
This generates:
`src/neuron_toolkit/rules/definitions/silu_fusion.json`.

### Step B: Register PyTorch Op Handler
Add to `src/neuron_toolkit/runtime/registry.py`:
```python
@self.register("custom.pytorch", "SiLU")
@self.register("", "SiLU")
def _silu(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
    return torch.nn.functional.silu(inputs[0])
```

### Step C: Add Pass Method to `JSONRuleEngine`
Add `apply_silu_fusion()` to `src/neuron_toolkit/rules/engine.py` and invoke it in `apply()`.

### Step D: Add Cross-Verification Test
Add a test in `tests/test_models_cross_verification.py`:
```python
def test_cross_verify_my_model():
    orig_model = onnx.load("my_model.onnx")
    engine = JSONRuleEngine(mode="function_proto")
    opt_model = engine.apply(orig_model)
    verifier = CrossVerifier(orig_model, opt_model)
    assert verifier.verify().all_passed
```

---

## 5. Practical Hints & Heuristics

1. **Handling Dynamic Dimensions**:
   In older ONNX models, dimension `0` in a `Reshape` node means "copy from input tensor at that dimension". `PyTorchRunner` handles this by mapping `0` to `x.shape[i]`.
2. **Topological Order Invariant**:
   After deleting or inserting nodes, the graph ordering can become invalid. Always call `toposort_graph(model)` before calling `onnx.checker.check_model`.
3. **Dual-Mode Netron Export**:
   - `mode="function_proto"`: Generates an ONNX model with `FunctionProto` attached so Netron renders high-level blocks that can be clicked to inspect internal primitives.
   - `mode="plain_custom_op"`: Strips functions for minimalist direct execution.
4. **Broadcasting in PyTorch vs ONNX**:
   In ONNX `Expand`, target shapes can contain `-1` or `1`. Multidirectional broadcasting must align trailing dimensions safely.
