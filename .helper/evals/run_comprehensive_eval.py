#!/usr/bin/env python3
"""Comprehensive Deep Learning Model Zoo Evaluation & Differential Verification Engine.

Discovers, optimizes, benchmarks, and cross-verifies ALL ONNX models in .helper/models/:
- Recurrent (RNN, LSTM, GRU, QRNN, SRU)
- Convolutional & Vision (ResNet, MobileNet, ConvNeXt, DenseNet, TCN, WaveNet, 3D Conv)
- Vanilla MLP & Gated Feedforwards (DeepMLP, ResMLP, MLPMixer, gMLP, Highway, GLU)
- Graph Neural Networks (GCN, GAT, GraphSAGE, GIN)
- Advanced Transformers & MoE (PreLN, PostLN, ViT, MLA, MoE, Performer, SwiGLU+RMS)
- Comprehensive Attention Family (15 distinct attention variants)

Evaluates:
1. Optimization / Minification: Original vs Simplified node counts and reduction %
2. Pattern & Custom Op Synthesis: Breakdown of recognized custom ops vs standard ops
3. Differential Numerical Parity: CrossVerifier (ORT vs PyTorchRunner) measuring Cosine Sim, Max Diff, MSE
4. Multi-Opset Robustness: Opset 14, 17, 18, 20

Saves results to .helper/evals/benchmark_results.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import onnx

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from neuron_toolkit.compiler.pipeline import NeuronCompiler
from neuron_toolkit.runtime.cross_verifier import CrossVerifier

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
log = logging.getLogger("eval_engine")


def evaluate_single_model(
    model_path: Path, mode: str = "function_proto"
) -> dict[str, Any]:
    """Run optimization and cross-verification on a single ONNX model."""
    t0 = time.perf_counter()
    res = {
        "path": str(model_path),
        "name": model_path.name,
        "category": model_path.parent.name
        if model_path.parent.name != "models"
        else "root",
        "passed": False,
        "error": None,
        "original_nodes": 0,
        "simplified_nodes": 0,
        "eliminated_nodes": 0,
        "reduction_pct": 0.0,
        "custom_ops_count": 0,
        "standard_ops_count": 0,
        "custom_ratio": 0.0,
        "custom_ops_breakdown": {},
        "min_cosine_sim": 0.0,
        "max_abs_diff": 0.0,
        "mse": 0.0,
        "opt_time_ms": 0.0,
        "verify_time_ms": 0.0,
        "opset": -1,
    }

    try:
        orig_model = onnx.load(str(model_path), load_external_data=True)
        res["original_nodes"] = len(orig_model.graph.node)
        res["opset"] = (
            orig_model.opset_import[0].version if orig_model.opset_import else -1
        )

        t_opt_start = time.perf_counter()
        opt_model = NeuronCompiler.compile(orig_model, mode=mode)
        t_opt_end = time.perf_counter()
        res["opt_time_ms"] = round((t_opt_end - t_opt_start) * 1000.0, 2)
        res["simplified_nodes"] = len(opt_model.graph.node)
        res["eliminated_nodes"] = res["original_nodes"] - res["simplified_nodes"]
        res["reduction_pct"] = round(
            (res["eliminated_nodes"] / max(1, res["original_nodes"])) * 100.0, 2
        )

        # Inspect custom vs standard ops
        custom_breakdown = collections.Counter()
        standard_breakdown = collections.Counter()
        for node in opt_model.graph.node:
            if getattr(node, "domain", "") == "custom.pytorch" or node.op_type in {
                "Linear",
                "Gelu",
                "RotaryEmbedding",
                "RMSNorm",
                "GroupNorm",
                "SplitQKV",
                "ScaledDotProductAttention",
                "GatedUnit",
                "MergeHeads",
                "FeedForward",
                "AxialRearrange",
                "SoftmaxMatMul",
                "TensorRoute",
                "Unbind",
                "SiLU",
            }:
                custom_breakdown[node.op_type] += 1
            else:
                standard_breakdown[node.op_type] += 1

        res["custom_ops_count"] = sum(custom_breakdown.values())
        res["standard_ops_count"] = sum(standard_breakdown.values())
        res["custom_ratio"] = round(
            (res["custom_ops_count"] / max(1, res["simplified_nodes"])) * 100.0, 2
        )
        res["custom_ops_breakdown"] = dict(custom_breakdown)

        # Cross Verification
        t_ver_start = time.perf_counter()
        verifier = CrossVerifier(orig_model, opt_model)
        report = verifier.verify()
        t_ver_end = time.perf_counter()
        res["verify_time_ms"] = round((t_ver_end - t_ver_start) * 1000.0, 2)

        res["passed"] = report.all_passed
        if report.metrics:
            cosines = [m.cosine_similarity for m in report.metrics.values()]
            diffs = [m.max_abs_diff for m in report.metrics.values()]
            mses = [m.mse for m in report.metrics.values()]
            res["min_cosine_sim"] = round(float(min(cosines)), 6)
            res["max_abs_diff"] = float(max(diffs))
            res["mse"] = float(max(mses))
        else:
            res["min_cosine_sim"] = 1.0 if report.all_passed else 0.0

    except Exception as e:
        res["passed"] = False
        res["error"] = str(e)

    return res


def run_evaluation(
    models_dir: Path = Path(".helper/models"),
    category: str | None = None,
    opset: int | None = None,
    output_json: Path = Path(".helper/evals/benchmark_results.json"),
) -> dict[str, Any]:
    """Execute evaluation over all matched models."""
    print("==================================================================")
    print("      NEURON TOOLKIT COMPREHENSIVE DL ZOO EVALUATION ENGINE       ")
    print("==================================================================")

    all_models = sorted(models_dir.glob("**/*.onnx"))
    filtered_models = []
    for p in all_models:
        if category and p.parent.name != category and category != "all":
            continue
        if opset and f"opset{opset}" not in p.name:
            continue
        filtered_models.append(p)

    print(f"Discovered {len(filtered_models)} model(s) matching criteria.")
    print(f"Targeting categories: {category or 'ALL'}")
    print(f"Targeting opsets:    {opset or 'ALL'}")
    print("-" * 80)

    results = []
    total_orig_nodes = 0
    total_opt_nodes = 0
    total_eliminated = 0
    passed_count = 0
    failed_count = 0

    cat_stats: dict[str, dict[str, Any]] = collections.defaultdict(
        lambda: {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "orig_nodes": 0,
            "opt_nodes": 0,
            "custom_ops": collections.Counter(),
        }
    )

    for idx, m_path in enumerate(filtered_models, 1):
        rel_path = m_path.relative_to(models_dir)
        print(
            f"[{idx:3d}/{len(filtered_models):3d}] Evaluating {rel_path!s:<50} ... ",
            end="",
            flush=True,
        )

        res = evaluate_single_model(m_path)
        results.append(res)

        cat = res["category"]
        cat_stats[cat]["total"] += 1
        cat_stats[cat]["orig_nodes"] += res["original_nodes"]
        cat_stats[cat]["opt_nodes"] += res["simplified_nodes"]
        for k, v in res.get("custom_ops_breakdown", {}).items():
            cat_stats[cat]["custom_ops"][k] += v

        if res["passed"]:
            passed_count += 1
            cat_stats[cat]["passed"] += 1
            total_orig_nodes += res["original_nodes"]
            total_opt_nodes += res["simplified_nodes"]
            total_eliminated += res["eliminated_nodes"]
            print(
                f"PASS (cos={res['min_cosine_sim']:.6f}, diff={res['max_abs_diff']:.1e}, {res['original_nodes']}->{res['simplified_nodes']} nodes, -{res['reduction_pct']}%)"
            )
        else:
            failed_count += 1
            cat_stats[cat]["failed"] += 1
            err_msg = res["error"] or "Numerical verification failed"
            print(f"FAIL ({err_msg[:45]})")

    # Aggregate Summary
    total_tested = len(filtered_models)
    pass_rate = (passed_count / max(1, total_tested)) * 100.0
    overall_red_pct = (
        (total_eliminated / max(1, total_orig_nodes)) * 100.0
        if total_orig_nodes > 0
        else 0.0
    )

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_models": total_tested,
        "passed": passed_count,
        "failed": failed_count,
        "pass_rate_pct": round(pass_rate, 2),
        "total_original_nodes": total_orig_nodes,
        "total_simplified_nodes": total_opt_nodes,
        "total_eliminated_nodes": total_eliminated,
        "overall_reduction_pct": round(overall_red_pct, 2),
        "category_summary": {},
        "results": results,
    }

    for cat, data in cat_stats.items():
        c_red = (
            ((data["orig_nodes"] - data["opt_nodes"]) / max(1, data["orig_nodes"]))
            * 100.0
            if data["orig_nodes"] > 0
            else 0.0
        )
        summary["category_summary"][cat] = {
            "total": data["total"],
            "passed": data["passed"],
            "failed": data["failed"],
            "orig_nodes": data["orig_nodes"],
            "opt_nodes": data["opt_nodes"],
            "reduction_pct": round(c_red, 2),
            "top_custom_ops": dict(data["custom_ops"].most_common(5)),
        }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print Final Summary Table
    print("\n" + "=" * 90)
    print(
        "                       CONSOLIDATED DL ZOO EVALUATION MATRIX                          "
    )
    print("=" * 90)
    print(
        f"{'Category':<24} | {'Total':<6} | {'Pass':<5} | {'Fail':<5} | {'Orig Nodes':<10} | {'Opt Nodes':<10} | {'Reduction %':<11} | {'Status'}"
    )
    print("-" * 90)
    for cat, data in summary["category_summary"].items():
        st = "ALL PASS" if data["failed"] == 0 else f"{data['failed']} FAIL"
        print(
            f"{cat:<24} | {data['total']:<6} | {data['passed']:<5} | {data['failed']:<5} | {data['orig_nodes']:<10} | {data['opt_nodes']:<10} | {data['reduction_pct']:<10.1f}% | {st}"
        )
    print("=" * 90)
    print(
        f"TOTAL: {passed_count} / {total_tested} PASSED ({pass_rate:.1f}% Pass Rate) | Overall Node Reduction: {overall_red_pct:.1f}%"
    )
    print(f"Full benchmark results written to: {output_json.resolve()}")
    print("=" * 90)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate DL Model Zoo against Neuron Toolkit"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(".helper/models"),
        help="Path to models root",
    )
    parser.add_argument(
        "--category", type=str, default=None, help="Filter by category folder name"
    )
    parser.add_argument(
        "--opset", type=int, default=None, help="Filter by opset version"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(".helper/evals/benchmark_results.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    run_evaluation(
        models_dir=args.models_dir,
        category=args.category,
        opset=args.opset,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
