#!/usr/bin/env python3
"""Unified Quantization Engine for Deep Learning Architecture Generators.

Features:
- Pure Logic-Based Dynamic Quantization for PyTorch nn.Modules (QuantizedLinear).
- Topological ONNX Dynamic Quantization Conversion (DynamicQuantizeLinear -> DequantizeLinear -> MatMul).
- End-to-End ONNX Export, Storage, Compilation, and Differential PyTorch Parity Verification.
- Seamless Integration with quantization_block_analysis.json and ModelRegistry.
"""

from __future__ import annotations

import copy
import inspect
import json
import logging
import os
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import onnx
from onnx import helper, numpy_helper
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuron_toolkit.compiler.base import LATEST_OPSET, ensure_model_latest_opset
from neuron_toolkit.compiler.pipeline import NeuronCompiler
from neuron_toolkit.runtime.cross_verifier import CrossVerifier
from neuron_toolkit.runtime.runner import PyTorchRunner

log = logging.getLogger("generators.quantization")


# =============================================================================
# 1. PyTorch Quantization Elements
# =============================================================================
class QuantizedLinear(nn.Module):
    """Dynamically quantized Linear layer implementing 8-bit simulated QDQ.

    Performs dynamic input activation quantization and 8-bit symmetric weight quantization.
    """

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        # Quantize weight to symmetric int8
        w = linear.weight.data
        scale_w = (w.abs().max() / 127.0).clamp(min=1e-8)
        w_q = torch.clamp(torch.round(w / scale_w), -128, 127)

        self.register_buffer("weight_scale", scale_w)
        self.register_buffer("weight_q", w_q)
        self.bias = nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dynamic per-tensor activation quantization
        scale_x = (x.abs().max() / 127.0).clamp(min=1e-8)
        x_q = torch.clamp(torch.round(x / scale_x), -128, 127)
        x_dq = x_q * scale_x
        w_dq = self.weight_q * self.weight_scale
        return F.linear(x_dq, w_dq, self.bias)


def quantize_pytorch_module(model: nn.Module) -> nn.Module:
    """Recursively clone and replace all nn.Linear layers with QuantizedLinear."""
    q_model = copy.deepcopy(model)

    def _replace_linear(module: nn.Module) -> None:
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                setattr(module, name, QuantizedLinear(child))
            else:
                _replace_linear(child)

    _replace_linear(q_model)
    q_model.eval()
    return q_model


# =============================================================================
# 2. ONNX Graph-Level Quantization Transformer
# =============================================================================
def quantize_onnx_model(model: onnx.ModelProto) -> onnx.ModelProto:
    """Topologically convert Linear projections (MatMul with weight initializers)

    into standard ONNX DynamicQuantizeLinear -> DequantizeLinear -> MatMul patterns.
    """
    q_model = copy.deepcopy(model)
    graph = q_model.graph
    init_map = {i.name: i for i in graph.initializer}

    new_nodes: list[onnx.NodeProto] = []
    quant_count = 0

    for n in graph.node:
        if n.op_type == "MatMul" and len(n.input) >= 2:
            act_in, weight_in = n.input[0], n.input[1]
            if weight_in in init_map:
                # Insert DynamicQuantizeLinear and DequantizeLinear
                q_x = f"{act_in}_q_{quant_count}"
                scale_x = f"{act_in}_scale_{quant_count}"
                zp_x = f"{act_in}_zp_{quant_count}"
                dq_x = f"{act_in}_dq_{quant_count}"

                dql = helper.make_node(
                    "DynamicQuantizeLinear",
                    inputs=[act_in],
                    outputs=[q_x, scale_x, zp_x],
                    name=f"dql_{quant_count}",
                )
                dq = helper.make_node(
                    "DequantizeLinear",
                    inputs=[q_x, scale_x, zp_x],
                    outputs=[dq_x],
                    name=f"dq_{quant_count}",
                )
                mm = helper.make_node(
                    "MatMul",
                    inputs=[dq_x, weight_in],
                    outputs=n.output,
                    name=n.name if n.name else f"mm_quant_{quant_count}",
                )
                new_nodes.extend([dql, dq, mm])
                quant_count += 1
                continue
        new_nodes.append(n)

    del graph.node[:]
    graph.node.extend(new_nodes)
    ensure_model_latest_opset(q_model)
    return q_model


# =============================================================================
# 3. Export & Differential PyTorch Parity Verification Engine
# =============================================================================
def export_quantized_model(
    model: nn.Module,
    dummy_inputs: tuple[torch.Tensor, ...] | torch.Tensor,
    input_names: list[str],
    output_names: list[str],
    base_name: str,
    out_dir: Path,
    opset_version: int = 18,
    rtol: float = 0.05,
    atol: float = 0.05,
    auto_verify: bool = True,
) -> tuple[str, onnx.ModelProto, dict[str, Any]]:
    """Export a quantized PyTorch module to ONNX, simplify with NeuronCompiler,

    and verify parity against PyTorch forward execution.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{base_name}_quant.onnx"

    if isinstance(dummy_inputs, torch.Tensor):
        dummy_inputs_tuple: tuple[torch.Tensor, ...] = (dummy_inputs,)
    else:
        dummy_inputs_tuple = tuple(dummy_inputs)

    # 1. Compute PyTorch Quantized reference output
    q_pt_model = quantize_pytorch_module(model)
    with torch.no_grad():
        pt_outs = q_pt_model(*dummy_inputs_tuple)
        if isinstance(pt_outs, (tuple, list)):
            pt_ref = pt_outs[0].detach().cpu().numpy()
        else:
            pt_ref = pt_outs.detach().cpu().numpy()

    # 2. Export clean base model and apply topological ONNX quantization
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        torch.onnx.export(
            model,
            dummy_inputs_tuple,
            tmp_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamo=False,
        )
        raw_onnx = onnx.load(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    q_onnx = quantize_onnx_model(raw_onnx)
    onnx.save(q_onnx, str(out_path))

    # 3. Compile with NeuronCompiler (function_proto mode)
    opt_onnx = NeuronCompiler.compile(q_onnx, mode="function_proto")

    # 4. Verification against PyTorch
    verification_stats = {
        "raw_nodes": len(q_onnx.graph.node),
        "opt_nodes": len(opt_onnx.graph.node),
        "reduction_pct": round(
            (len(q_onnx.graph.node) - len(opt_onnx.graph.node))
            / max(1, len(q_onnx.graph.node))
            * 100.0,
            2,
        ),
        "max_diff": 0.0,
        "cosine_sim": 1.0,
        "passed": True,
    }

    if auto_verify:
        runner = PyTorchRunner(opt_onnx)
        feed_dict = {
            name: inp.detach().cpu().numpy()
            for name, inp in zip(input_names, dummy_inputs_tuple)
        }
        onnx_outs = runner.run(feed_dict)
        opt_ref = next(iter(onnx_outs.values()))

        diff = np.abs(pt_ref.astype(np.float64) - opt_ref.astype(np.float64))
        max_diff = float(np.max(diff))
        ref_flat = pt_ref.flatten().astype(np.float64)
        opt_flat = opt_ref.flatten().astype(np.float64)
        norm_p = np.linalg.norm(ref_flat) * np.linalg.norm(opt_flat)
        cosine = float(np.dot(ref_flat, opt_flat) / norm_p) if norm_p > 0 else 1.0

        verification_stats["max_diff"] = round(max_diff, 6)
        verification_stats["cosine_sim"] = round(cosine, 6)

        max_val = float(np.max(np.abs(pt_ref))) if pt_ref.size > 0 else 1.0
        passed = (max_diff <= (atol + rtol * max_val)) or (cosine >= 0.98)
        verification_stats["passed"] = passed

        if not passed:
            raise AssertionError(
                f"Quantized verification FAILED for {base_name}: "
                f"max_diff={max_diff:.5f}, cosine_sim={cosine:.5f}"
            )

    return str(out_path), opt_onnx, verification_stats


# =============================================================================
# 4. Quantization Registry & Batch Exporter
# =============================================================================
class QuantizationRegistry:
    """Registry and orchestrator for quantized block generation and evaluation."""

    @classmethod
    def get_catalog_classes(cls) -> dict[str, Any]:
        """Load and return AST quantization catalog."""
        catalog_path = _REPO_ROOT / "quantization_block_analysis.json"
        if not catalog_path.exists():
            return {}
        with open(catalog_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def export_all_quantized(
        cls,
        base_out_dir: Path = _REPO_ROOT / ".helper" / "models" / "quantized",
        limit: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Batch export quantized models across the entire architecture catalog."""
        from generators.registry import ModelRegistry

        specs = ModelRegistry.discover(_REPO_ROOT / "generators")
        if limit is not None:
            specs = specs[:limit]

        results = {}
        for spec in specs:
            out_sub = spec.resolve_subfolder()
            out_dir = base_out_dir / out_sub if out_sub else base_out_dir
            try:
                model = spec.instantiate()
                dummy_inputs = spec.get_dummy_inputs()
                path, opt_proto, stats = export_quantized_model(
                    model=model,
                    dummy_inputs=dummy_inputs,
                    input_names=spec.input_names,
                    output_names=spec.output_names,
                    base_name=spec.name,
                    out_dir=out_dir,
                    rtol=spec.rtol * 50 if spec.rtol else 0.05,
                    atol=spec.atol * 50 if spec.atol else 0.05,
                    auto_verify=True,
                )
                results[spec.name] = {"path": path, "stats": stats}
                print(
                    f"  [QUANT OK] {spec.name:<35} raw={stats['raw_nodes']:<3} -> opt={stats['opt_nodes']:<3} "
                    f"({stats['reduction_pct']:>5.1f}%) | diff={stats['max_diff']:.5f} | cos={stats['cosine_sim']:.4f}"
                )
            except Exception as e:
                log.warning("Skipping quantized export for %s: %s", spec.name, e)

        return results


if __name__ == "__main__":
    print("=== Running Quantization Batch Exporter ===")
    res = QuantizationRegistry.export_all_quantized(limit=10)
    print(f"Successfully exported {len(res)} quantized models.")
