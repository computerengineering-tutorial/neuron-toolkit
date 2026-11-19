#!/usr/bin/env python3
"""Generator and diagnostic evaluation harness for heavily obfuscated ONNX graphs.

Features:
- Declarative registration of obfuscated architectures via @register_architecture.
- Pure ONNX programmatic graphs (decomposed LayerNorm, decomposed Softmax, dead branches, transpose ping-pongs).
- Complex PyTorch obfuscated architectures for stress-testing compiler passes.
- Verification harness that validates compiler pass invariants (Cosine Similarity >= 0.9999, differential parity).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper
from torch import nn

# Ensure workspace root and generators are in path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from generators.decorators import register_architecture

OUT_DIR = Path(".helper/models/obfuscated_stress")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. Complex PyTorch Obfuscated Architecture
# =============================================================================
@register_architecture(
    name="pytorch_obfuscated_complex",
    category="obfuscated_stress",
    subfolder="obfuscated_stress",
    shapes=[(2, 8, 32)],
    input_names=["x"],
    output_names=["out"],
    opset_versions=(14, 18, 20),
    description="Complex multi-layer PyTorch block with mathematical tautologies and ghost branches",
)
class PyTorchObfuscatedComplexBlock(nn.Module):
    """Stress-test architecture combining normalization, projections, and tautologies."""

    def __init__(self, d_model: int = 32) -> None:
        super().__init__()
        self.d_model = d_model
        self.ln1 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * 2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_model * 2, d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Tautology input mutation: (x + 0.0) * 1.0 == x
        x_in = (x + 0.0) * 1.0

        # Sub-layer 1: LayerNorm + Projection
        h1 = self.ln1(x_in)
        h2 = self.fc2(self.act(self.fc1(h1)))

        # Commutative addition & split residual: 0.5 * x + h2 + 0.5 * x == x + h2
        res1 = 0.5 * x + h2 + 0.5 * x

        # Sub-layer 2: Second norm with ping-pong scaling
        norm2 = self.ln2(res1)
        scaled = (norm2 * 2.0) * 0.5

        # Ghost branch: scaled + (scaled * 0.0)
        out = scaled + (scaled * 0.0)
        return (out * 1.0) + 0.0


# =============================================================================
# 2. Pure ONNX Programmatic Decomposed Block Builder
# =============================================================================
def build_pure_onnx_decomposed_ln(
    opset: int = 18,
    d_model: int = 32,
    seq_len: int = 8,
) -> onnx.ModelProto:
    """Construct an intentionally decomposed, multi-layer obfuscated block in raw ONNX."""
    nodes = []

    x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, seq_len, d_model])
    out_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, seq_len, d_model])

    zero_const = helper.make_tensor("zero_c", TensorProto.FLOAT, [], [0.0])
    eps_const = helper.make_tensor("eps_c", TensorProto.FLOAT, [], [1e-5])
    gamma_init = helper.make_tensor("gamma", TensorProto.FLOAT, [d_model], [1.0] * d_model)
    beta_init = helper.make_tensor("beta", TensorProto.FLOAT, [d_model], [0.0] * d_model)
    w_proj = helper.make_tensor(
        "w_proj",
        TensorProto.FLOAT,
        [d_model, d_model],
        np.random.randn(d_model, d_model).astype(np.float32).flatten().tolist(),
    )
    axes_neg1 = helper.make_tensor("axes_neg1", TensorProto.INT64, [1], [-1])

    inits = [zero_const, eps_const, gamma_init, beta_init, w_proj]
    if opset >= 13:
        inits.append(axes_neg1)

    # 1. Tautological arithmetic (X + 0 - 0) & Identity
    nodes.append(helper.make_node("Add", inputs=["X", "zero_c"], outputs=["x_add"], name="p_add0"))
    nodes.append(helper.make_node("Sub", inputs=["x_add", "zero_c"], outputs=["x_clean"], name="p_sub0"))
    nodes.append(helper.make_node("Identity", inputs=["x_clean"], outputs=["x_id"], name="p_id0"))

    # 2. Decomposed LayerNorm using Mul(diff, diff)
    if opset >= 18:
        nodes.append(helper.make_node("ReduceMean", inputs=["x_id", "axes_neg1"], outputs=["mean"], keepdims=1, name="p_mean1"))
        nodes.append(helper.make_node("Sub", inputs=["x_id", "mean"], outputs=["diff"], name="p_sub1"))
        nodes.append(helper.make_node("Mul", inputs=["diff", "diff"], outputs=["diff_sq"], name="p_sq1"))
        nodes.append(helper.make_node("ReduceMean", inputs=["diff_sq", "axes_neg1"], outputs=["var"], keepdims=1, name="p_var1"))
    elif opset >= 13:
        nodes.append(helper.make_node("ReduceMean", inputs=["x_id"], outputs=["mean"], axes=[-1], keepdims=1, name="p_mean1"))
        nodes.append(helper.make_node("Sub", inputs=["x_id", "mean"], outputs=["diff"], name="p_sub1"))
        nodes.append(helper.make_node("Mul", inputs=["diff", "diff"], outputs=["diff_sq"], name="p_sq1"))
        nodes.append(helper.make_node("ReduceMean", inputs=["diff_sq"], outputs=["var"], axes=[-1], keepdims=1, name="p_var1"))
    else:
        nodes.append(helper.make_node("ReduceMean", inputs=["x_id"], outputs=["mean"], axes=[-1], keepdims=1, name="p_mean1"))
        nodes.append(helper.make_node("Sub", inputs=["x_id", "mean"], outputs=["diff"], name="p_sub1"))
        nodes.append(helper.make_node("Mul", inputs=["diff", "diff"], outputs=["diff_sq"], name="p_sq1"))
        nodes.append(helper.make_node("ReduceMean", inputs=["diff_sq"], outputs=["var"], axes=[-1], keepdims=1, name="p_var1"))

    nodes.append(helper.make_node("Add", inputs=["var", "eps_c"], outputs=["var_eps"], name="p_add_eps"))
    nodes.append(helper.make_node("Sqrt", inputs=["var_eps"], outputs=["std"], name="p_sqrt"))
    nodes.append(helper.make_node("Div", inputs=["diff", "std"], outputs=["normed"], name="p_div"))
    nodes.append(helper.make_node("Mul", inputs=["normed", "gamma"], outputs=["scaled"], name="p_scale"))
    nodes.append(helper.make_node("Add", inputs=["scaled", "beta"], outputs=["ln_out"], name="p_ln_bias"))

    # 3. Decomposed Softmax: ReduceMax -> Sub -> Exp -> ReduceSum -> Div (strictly topologically sorted!)
    if opset >= 18:
        nodes.append(helper.make_node("ReduceMax", inputs=["ln_out", "axes_neg1"], outputs=["sm_max"], keepdims=1, name="p_max"))
    else:
        nodes.append(helper.make_node("ReduceMax", inputs=["ln_out"], outputs=["sm_max"], axes=[-1], keepdims=1, name="p_max"))

    nodes.append(helper.make_node("Sub", inputs=["ln_out", "sm_max"], outputs=["sm_sub"], name="p_sub_max"))
    nodes.append(helper.make_node("Exp", inputs=["sm_sub"], outputs=["sm_exp"], name="p_exp"))

    if opset >= 18 or opset >= 13:
        nodes.append(helper.make_node("ReduceSum", inputs=["sm_exp", "axes_neg1"], outputs=["sm_sum"], keepdims=1, name="p_sum"))
    else:
        nodes.append(helper.make_node("ReduceSum", inputs=["sm_exp"], outputs=["sm_sum"], axes=[-1], keepdims=1, name="p_sum"))

    nodes.append(helper.make_node("Div", inputs=["sm_exp", "sm_sum"], outputs=["sm_out"], name="p_div_sm"))

    # 4. Projection + Transpose Ping-Pong
    nodes.append(helper.make_node("MatMul", inputs=["sm_out", "w_proj"], outputs=["lin_out"], name="p_matmul"))
    nodes.append(helper.make_node("Transpose", inputs=["lin_out"], outputs=["tr1"], perm=[1, 0, 2], name="p_tr1"))
    nodes.append(helper.make_node("Transpose", inputs=["tr1"], outputs=["tr2"], perm=[1, 0, 2], name="p_tr2"))

    # 5. Dead branch (unreferenced)
    nodes.append(helper.make_node("Mul", inputs=["tr2", "zero_c"], outputs=["dead_mul"], name="p_dead_mul"))
    nodes.append(helper.make_node("Relu", inputs=["dead_mul"], outputs=["dead_relu"], name="p_dead_relu"))

    # 6. Residual addition: tr2 + X
    nodes.append(helper.make_node("Add", inputs=["tr2", "X"], outputs=["Y"], name="p_out_add"))

    graph = helper.make_graph(
        nodes=nodes,
        name=f"pure_onnx_decomposed_ln_opset{opset}",
        inputs=[x_info],
        outputs=[out_info],
        initializer=inits,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=8)


# Export wrapper to make pure ONNX model available as a registered architecture
class PureONNXDecomposedLNModule(nn.Module):
    """PyTorch wrapper that yields identical computation to pure_onnx_decomposed_ln."""

    def __init__(self, d_model: int = 32) -> None:
        super().__init__()
        self.d_model = d_model
        self.ln = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Exact math matching pure_onnx_decomposed_ln
        norm = self.ln(x)
        sm = torch.softmax(norm, dim=-1)
        proj = self.proj(sm)
        # Transpose ping-pong [1, 0, 2] -> [1, 0, 2]
        tr = proj.transpose(0, 1).transpose(0, 1)
        return tr + x


@register_architecture(
    name="pure_onnx_decomposed_ln",
    category="obfuscated_stress",
    subfolder="obfuscated_stress",
    shapes=[(1, 8, 32)],
    input_names=["X"],
    output_names=["Y"],
    opset_versions=(14, 18, 20),
    description="Pure decomposed LayerNorm, Softmax, Transpose ping-pong and dead branch stress model",
)
def create_pure_onnx_decomposed_ln() -> nn.Module:
    return PureONNXDecomposedLNModule(32)


# =============================================================================
# 3. Diagnostic Robustness Evaluation Harness
# =============================================================================
def verify_obfuscated_stress_zoo() -> dict[str, Any]:
    """Run full optimization pipeline and cross-verifier on all obfuscated models."""
    from neuron_toolkit.compiler.pipeline import NeuronCompiler
    from neuron_toolkit.runtime.cross_verifier import CrossVerifier

    print("\n" + "=" * 105)
    print("                     OBFUSCATED STRESS ZOO ROBUSTNESS EVALUATION REPORT")
    print("=" * 105)
    print(
        f"{'Model Name':<35} | {'Opset':<5} | {'Orig':<6} | {'Opt':<6} | {'Reduction':<10} | {'Cosine Sim':<11} | {'Status'}"
    )
    print("-" * 105)

    passed_count = 0
    total_count = 0
    results = []

    opsets = [14, 18, 20]
    # 1. Pure ONNX Decomposed LN models
    for opset in opsets:
        total_count += 1
        name = f"pure_onnx_decomposed_ln_opset{opset}"
        try:
            m = build_pure_onnx_decomposed_ln(opset=opset, d_model=32, seq_len=8)
            orig_nodes = len(m.graph.node)
            opt_m = NeuronCompiler.compile(m, mode="function_proto")
            opt_nodes = len(opt_m.graph.node)
            reduction = ((orig_nodes - opt_nodes) / orig_nodes) * 100.0 if orig_nodes > 0 else 0.0

            verifier = CrossVerifier(m, opt_m)
            report = verifier.verify()
            is_pass = report.all_passed
            min_cos = min((metric.cosine_similarity for metric in report.metrics.values()), default=1.0)
            status_str = "PASS" if is_pass else "FAIL"
            if is_pass:
                passed_count += 1

            results.append({"name": name, "opset": opset, "passed": is_pass, "reduction": reduction})
            print(
                f"{name:<35} | {opset:<5} | {orig_nodes:<6} | {opt_nodes:<6} | {reduction:>8.1f} % | {min_cos:>10.6f}  | {status_str}"
            )
        except Exception as e:
            print(f"{name:<35} | {opset:<5} | ERROR: {e}")
            results.append({"name": name, "opset": opset, "passed": False, "error": str(e)})

    # 2. PyTorch Obfuscated Complex models
    dummy = torch.randn(2, 8, 32)
    pt_mod = PyTorchObfuscatedComplexBlock(32).eval()
    for opset in opsets:
        total_count += 1
        name = f"pytorch_obfuscated_complex_opset{opset}"
        tmp_onnx = Path(f"/tmp/{name}.onnx")
        try:
            torch.onnx.export(
                pt_mod,
                (dummy,),
                str(tmp_onnx),
                input_names=["x"],
                output_names=["out"],
                opset_version=opset,
                dynamo=False,
            )
            m = onnx.load(str(tmp_onnx))
            tmp_onnx.unlink(missing_ok=True)
            orig_nodes = len(m.graph.node)
            opt_m = NeuronCompiler.compile(m, mode="function_proto")
            opt_nodes = len(opt_m.graph.node)
            reduction = ((orig_nodes - opt_nodes) / orig_nodes) * 100.0 if orig_nodes > 0 else 0.0

            verifier = CrossVerifier(m, opt_m)
            report = verifier.verify()
            is_pass = report.all_passed
            min_cos = min((metric.cosine_similarity for metric in report.metrics.values()), default=1.0)
            status_str = "PASS" if is_pass else "FAIL"
            if is_pass:
                passed_count += 1

            results.append({"name": name, "opset": opset, "passed": is_pass, "reduction": reduction})
            print(
                f"{name:<35} | {opset:<5} | {orig_nodes:<6} | {opt_nodes:<6} | {reduction:>8.1f} % | {min_cos:>10.6f}  | {status_str}"
            )
        except Exception as e:
            print(f"{name:<35} | {opset:<5} | ERROR: {e}")
            results.append({"name": name, "opset": opset, "passed": False, "error": str(e)})

    print("=" * 105)
    print(f"TOTAL: {passed_count} / {total_count} PASSED ({(passed_count / max(1, total_count)) * 100:.1f}% Pass Rate)")
    print("=" * 105)
    return {"total": total_count, "passed": passed_count, "results": results}


if __name__ == "__main__":
    verify_obfuscated_stress_zoo()
