"""Differential Cross-Verification Suite for ONNX Runtime vs PyTorch Runner."""

from __future__ import annotations

import collections
import copy
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import onnx
import onnxruntime as ort

from neuron_toolkit.runtime.runner import PyTorchRunner

if TYPE_CHECKING:
    from neuron_toolkit.graph import NeuronGraph

log = logging.getLogger("neuron_toolkit.runtime.cross_verifier")


@dataclass
class OutputMetric:
    """Metrics for a single model output tensor."""

    name: str
    shape: tuple[int, ...]
    max_abs_diff: float
    mse: float
    cosine_similarity: float
    passed: bool


@dataclass
class CrossVerificationReport:
    """Consolidated numerical verification report comparing ORT and PyTorch Runner."""

    original_nodes: int
    simplified_nodes: int
    eliminated_nodes: int
    reduction_percentage: float
    metrics: dict[str, OutputMetric]
    all_passed: bool
    standard_nodes: int = 0
    custom_nodes: int = 0
    custom_ratio: float = 0.0
    custom_op_breakdown: dict[str, int] = field(default_factory=dict)
    standard_op_breakdown: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        """Format an ASCII/Markdown summary table of the verification results."""
        std_pct = (self.standard_nodes / max(1, self.simplified_nodes)) * 100.0 if self.simplified_nodes > 0 else 0.0
        custom_ratio_str = (
            f"1 : {self.standard_nodes / self.custom_nodes:.2f}"
            if self.custom_nodes > 0
            else "N/A (0 custom ops)"
        )

        custom_ops_list = [f"{k} ({v})" for k, v in sorted(self.custom_op_breakdown.items(), key=lambda x: -x[1])]
        custom_ops_summary = ", ".join(custom_ops_list) if custom_ops_list else "None (all standard ONNX)"

        lines = [
            "==================================================================",
            "                   CROSS-VERIFICATION REPORT                     ",
            "==================================================================",
            f"Original Nodes:          {self.original_nodes}",
            f"Simplified Nodes:        {self.simplified_nodes}",
            f"  - Standard ONNX Nodes: {self.standard_nodes} ({std_pct:.1f}%)",
            f"  - Custom Ops Nodes:    {self.custom_nodes} ({self.custom_ratio:.1f}%)",
            f"  - Custom / ONNX Ratio: {custom_ratio_str} ({self.custom_ratio:.1f}% custom)",
            f"Nodes Eliminated:        {self.eliminated_nodes} ({self.reduction_percentage:.1f}% reduction)",
            f"Active Custom Ops:       {custom_ops_summary}",
            f"Overall Status:          {'PASSED' if self.all_passed else 'FAILED'}",
            "------------------------------------------------------------------",
            f"{'Output':<20} {'Shape':<16} {'Max Diff':<12} {'MSE':<12} {'Cosine':<10} {'Status'}",
            "------------------------------------------------------------------",
        ]
        for name, m in self.metrics.items():
            status_str = "PASS" if m.passed else "FAIL"
            shape_str = str(m.shape)
            lines.append(
                f"{name:<20} {shape_str:<16} {m.max_abs_diff:<12.3e} {m.mse:<12.3e} {m.cosine_similarity:<10.6f} {status_str}"
            )
        lines.append("==================================================================")
        return "\n".join(lines)


class CrossVerifier:
    """Runs differential verification comparing original model on ORT vs simplified on PyTorchRunner."""

    def __init__(
        self,
        original_model: onnx.ModelProto | NeuronGraph | str,
        simplified_model: onnx.ModelProto | NeuronGraph | str,
        rtol: float = 1e-4,
        atol: float = 1e-4,
        min_cosine_similarity: float = 0.9999,
    ) -> None:
        self.rtol = rtol
        self.atol = atol
        self.min_cosine_similarity = min_cosine_similarity

        self.orig_proto = self._to_proto(original_model)
        self.opt_proto = self._to_proto(simplified_model)

    def _to_proto(self, obj: Any) -> onnx.ModelProto:
        if isinstance(obj, str):
            return onnx.load(obj)
        elif hasattr(obj, "_backend") and hasattr(obj._backend, "model"):
            return copy.deepcopy(obj._backend.model)
        elif isinstance(obj, onnx.ModelProto):
            return copy.deepcopy(obj)
        msg = f"Cannot convert {type(obj)} to onnx.ModelProto"
        raise TypeError(msg)

    def generate_dummy_inputs(
        self,
        batch_size: int = 1,
        default_dynamic_dim: int = 32,
    ) -> dict[str, np.ndarray]:
        """Generate valid random float32/int64 inputs matching graph signatures."""
        inputs: dict[str, np.ndarray] = {}
        initializers = {init.name for init in self.orig_proto.graph.initializer}

        for inp in self.orig_proto.graph.input:
            if inp.name in initializers:
                continue
            shape: list[int] = []
            for d in inp.type.tensor_type.shape.dim:
                if d.dim_value > 0:
                    shape.append(d.dim_value)
                else:
                    # Dynamic dimension: use batch_size for first dim, default for others
                    shape.append(batch_size if not shape else default_dynamic_dim)

            elem_type = inp.type.tensor_type.elem_type
            if elem_type == onnx.TensorProto.INT64:
                inputs[inp.name] = np.random.randint(0, 2, size=shape, dtype=np.int64)
            elif elem_type == onnx.TensorProto.INT32:
                inputs[inp.name] = np.random.randint(0, 2, size=shape, dtype=np.int32)
            elif elem_type == onnx.TensorProto.BOOL:
                inputs[inp.name] = np.random.choice([True, False], size=shape)
            elif elem_type == onnx.TensorProto.FLOAT16:
                inputs[inp.name] = np.random.randn(*shape).astype(np.float16)
            elif elem_type == onnx.TensorProto.DOUBLE:
                inputs[inp.name] = np.random.randn(*shape).astype(np.float64)
            else:
                inputs[inp.name] = np.random.randn(*shape).astype(np.float32)

        return inputs

    def verify(
        self,
        inputs: dict[str, np.ndarray] | None = None,
    ) -> CrossVerificationReport:
        """Execute verification across both runtimes and compute differential metrics."""
        if inputs is None:
            inputs = self.generate_dummy_inputs()

        # 1. Run reference model on ONNX Runtime
        orig_bytes = self.orig_proto.SerializeToString()
        ort_session = ort.InferenceSession(orig_bytes)
        ort_out_names = [o.name for o in ort_session.get_outputs()]
        ort_raw_outs = ort_session.run(None, inputs)
        ort_outs = dict(zip(ort_out_names, ort_raw_outs, strict=False))

        # 2. Run simplified model on PyTorchRunner
        runner = PyTorchRunner(self.opt_proto)
        torch_outs = runner.run(inputs)

        # 3. Compute metrics
        orig_nodes = len(self.orig_proto.graph.node)
        opt_nodes = len(self.opt_proto.graph.node)
        eliminated = orig_nodes - opt_nodes
        reduction_pct = (eliminated / max(1, orig_nodes)) * 100.0

        all_passed = True
        metrics: dict[str, OutputMetric] = {}

        for name, ref_arr in ort_outs.items():
            if name not in torch_outs:
                log.error("Output tensor %r missing in PyTorchRunner outputs", name)
                all_passed = False
                continue

            test_arr = torch_outs[name]
            max_diff = float(np.max(np.abs(ref_arr - test_arr)))
            mse = float(np.mean((ref_arr - test_arr) ** 2))

            # Cosine similarity
            ref_flat = ref_arr.flatten().astype(np.float64)
            test_flat = test_arr.flatten().astype(np.float64)
            norm_prod = np.linalg.norm(ref_flat) * np.linalg.norm(test_flat)
            if norm_prod > 0:
                cosine_sim = float(np.dot(ref_flat, test_flat) / norm_prod)
            else:
                cosine_sim = 1.0 if np.allclose(ref_flat, test_flat) else 0.0

            passed = (
                max_diff <= (self.atol + self.rtol * float(np.max(np.abs(ref_arr))))
                or cosine_sim >= self.min_cosine_similarity
            )
            if not passed:
                all_passed = False

            metrics[name] = OutputMetric(
                name=name,
                shape=test_arr.shape,
                max_abs_diff=max_diff,
                mse=mse,
                cosine_similarity=cosine_sim,
                passed=passed,
            )

        # Classify nodes into Standard ONNX vs Custom Ops
        custom_op_names = {
            "Linear",
            "ScaledDotProductAttention",
            "Attention",
            "MergeHeads",
            "SplitQKV",
            "RotaryEmbedding",
            "GatedUnit",
            "SoftmaxMatMul",
            "Unbind",
            "RMSNorm",
            "Gelu",
            "GroupNorm",
            "InstanceNorm1d",
            "TensorRoute",
            "IndexSelect",
        }
        custom_counts: dict[str, int] = collections.defaultdict(int)
        standard_counts: dict[str, int] = collections.defaultdict(int)

        for n in self.opt_proto.graph.node:
            domain = n.domain or ""
            is_custom = (
                domain in ("custom.pytorch", "custom")
                or domain.startswith("custom.")
                or (domain not in ("", "ai.onnx", "ai.onnx.ml", "ai.onnx.preview.training"))
                or n.op_type in custom_op_names
            )
            if is_custom:
                custom_counts[n.op_type] += 1
            else:
                standard_counts[n.op_type] += 1

        custom_nodes = sum(custom_counts.values())
        standard_nodes = sum(standard_counts.values())
        custom_ratio = (custom_nodes / max(1, opt_nodes)) * 100.0

        return CrossVerificationReport(
            original_nodes=orig_nodes,
            simplified_nodes=opt_nodes,
            eliminated_nodes=eliminated,
            reduction_percentage=reduction_pct,
            metrics=metrics,
            all_passed=all_passed,
            standard_nodes=standard_nodes,
            custom_nodes=custom_nodes,
            custom_ratio=custom_ratio,
            custom_op_breakdown=dict(custom_counts),
            standard_op_breakdown=dict(standard_counts),
        )
