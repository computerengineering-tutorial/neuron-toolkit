# PyTorch Runtime & Differential Cross-Verification

The `neuron_toolkit.runtime` package provides a standalone vectorized PyTorch execution engine for standard and rewritten ONNX computational graphs, coupled with an automated differential cross-verification suite.

---

## The Challenge of Custom Operator Execution

When compilers optimize ONNX models by fusing subgraphs into high-level custom operators (such as `ScaledDotProductAttention`, `RMSNorm`, `FeedForwardSwiGLU`, or `RotaryEmbedding`), standard ONNX Runtime (ORT) cannot execute them without building external C++ custom operator libraries.

`neuron-toolkit` solves this with a two-part runtime architecture:

1. **`PyTorchRunner`**: A pure-Python/PyTorch execution runtime capable of running standard ONNX nodes and custom operators natively.
2. **`CrossVerifier`**: A differential testing suite that compares output tensors between ONNX Runtime (running the original unfused model) and `PyTorchRunner` (running the compiled/rewritten model) across randomly generated or user-supplied inputs.

---

## Native Execution with `PyTorchRunner`

The `PyTorchRunner` sorts graph nodes topologically, resolves inputs and static weights, and executes operators using vectorized PyTorch operations registered in `OpRegistry`:

```python
import onnx
import numpy as np
from neuron_toolkit.runtime.runner import PyTorchRunner

# Load rewritten model containing custom ops (e.g. custom.pytorch::RMSNorm)
model = onnx.load("rewritten_llama.onnx")

# Initialize runner on CPU or CUDA
runner = PyTorchRunner(model, device="cuda")

# Execute with numpy arrays or torch.Tensor inputs
inputs = {
    "hidden_states": np.random.randn(2, 64, 4096).astype(np.float32)
}
outputs = runner.run(inputs)

print("Output shape:", outputs["output"].shape)
```

### Custom Operator Registry: `OpRegistry`

Custom operator handlers are managed dynamically by `OpRegistry`:

```python
from neuron_toolkit.runtime.registry import OpRegistry
import torch

registry = OpRegistry()

# Register a custom handler for custom.pytorch::MyCustomOp
@registry.register("custom.pytorch", "MyCustomOp")
def my_custom_handler(inputs: list[torch.Tensor], attrs: dict) -> list[torch.Tensor]:
    x, scale = inputs[0], inputs[1]
    bias = attrs.get("bias", 0.0)
    return [x * scale + bias]
```

Built-in operator support in `OpRegistry` covers:

- **Standard ONNX**: `MatMul`, `Gemm`, `Conv`, `GlobalAveragePool`, `RNN`, `Add`, `Sub`, `Mul`, `Div`, `Softmax`, `LayerNormalization`, `Reshape`, `Transpose`, `Split`, `Concat`, `Gather`, `Slice`, `Pow`, `Sqrt`, `Erf`, `Tanh`, `Sigmoid`, `Relu`, `Clip`, etc.
- **Custom PyTorch**: `RMSNorm`, `ScaledDotProductAttention`, `LogitsSoftCap`, `MultiHeadLatentAttention_KVProjection`, `RotaryEmbedding`, `SiLU`, `HardSwish`, `Mish`, `Linear`, `FeedForward`, `FeedForwardSwiGLU`, `SpatialGatingUnit`, `SqueezeExcitation`, `DeepMLP_Norm_Act`, `SplitQKV`, `MergeHeads`, `Unbind`.

---

## Differential Testing with `CrossVerifier`

The `CrossVerifier` guarantees compiler soundness by executing both the original model (via ONNX Runtime) and the transformed model (via `PyTorchRunner`), validating that numerical outputs match within strict tolerance thresholds:

```python
from neuron_toolkit.runtime.cross_verifier import CrossVerifier

# Initialize verifier with original and simplified models
verifier = CrossVerifier(
    original_model="llama_raw.onnx",
    simplified_model="llama_optimized.onnx",
    rtol=1e-4,
    atol=1e-4,
    min_cosine_similarity=0.9999
)

# Automatically generate dynamic shape inputs and verify
report = verifier.verify()

# Print formatted Markdown/ASCII verification summary table
print(report.summary())
```

### Verification Report Metrics

The generated `CrossVerificationReport` provides comprehensive numerical validation and graph reduction statistics:

```python
# Check overall status
if report.all_passed:
    print("✅ Compilation verified: numerical outputs are bitwise / tolerance accurate!")
else:
    print("❌ Verification failed! Regressions detected:")

print(f"Original Nodes:   {report.original_nodes}")
print(f"Simplified Nodes: {report.simplified_nodes}")
print(f"Nodes Eliminated: {report.eliminated_nodes} ({report.reduction_percentage:.1f}% reduction)")
print(f"Custom Ops Ratio: {report.custom_ratio:.1f}%")

# Inspect active custom ops breakdown:
for op, count in report.custom_op_breakdown.items():
    print(f"  - {op}: {count}")

# Inspect per-output tensor metrics
for output_name, metric in report.metrics.items():
    print(f"Output: {output_name} (shape: {metric.shape})")
    print(f"  - Max Absolute Diff: {metric.max_abs_diff:.6e}")
    print(f"  - MSE:               {metric.mse:.6e}")
    print(f"  - Cosine Similarity: {metric.cosine_similarity:.6f}")
    print(f"  - Passed:            {metric.passed}")
```

### Example Verification Summary Output

```
==================================================================
                   CROSS-VERIFICATION REPORT                     
==================================================================
Original Nodes:          150
Simplified Nodes:        42
  - Standard ONNX Nodes: 30 (71.4%)
  - Custom Ops Nodes:    12 (28.6%)
  - Custom / ONNX Ratio: 1 : 2.50 (28.6% custom)
Nodes Eliminated:        108 (72.0% reduction)
Active Custom Ops:       ScaledDotProductAttention (4), RMSNorm (4), FeedForwardSwiGLU (4)
Overall Status:          PASSED
------------------------------------------------------------------
Output               Shape            Max Diff     MSE          Cosine     Status
------------------------------------------------------------------
logits               (1, 128, 32000)  1.428e-05    3.109e-06    0.999998   PASS
==================================================================
```
