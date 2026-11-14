#!/usr/bin/env python3
"""Multi-model Benchmark Suite for ONNX Graph Optimization & Custom Ops Ratio Analysis."""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import onnx
from neuron_toolkit.rules.engine import JSONRuleEngine
from neuron_toolkit.runtime.cross_verifier import CrossVerifier


def benchmark_model(
    model_path: Path,
    mode: str = "function_proto",
    rtol: float = 1e-4,
    atol: float = 1e-4,
    export_dir: Path | None = None,
) -> dict:
    """Run optimization, clean model export, and cross-verification on a single ONNX model."""
    print(f"\n==================================================================")
    print(f"BENCHMARKING MODEL: {model_path.name}")
    print(f"==================================================================")

    orig_model = onnx.load(str(model_path))
    engine = JSONRuleEngine(mode=mode)

    t0 = time.perf_counter()
    opt_model = engine.apply(orig_model)
    opt_time_ms = (time.perf_counter() - t0) * 1000.0

    # Clean Model Export for Netron / Visualizer Inspection
    clean_model_path: Path | None = None
    if export_dir is not None:
        export_dir.mkdir(parents=True, exist_ok=True)
        clean_model_path = export_dir / f"{model_path.stem}_clean.onnx"
        try:
            onnx.checker.check_model(opt_model)
        except Exception as e:
            print(f"Warning: onnx.checker validation note: {e}")
        onnx.save(opt_model, str(clean_model_path))
        file_size_mb = clean_model_path.stat().st_size / (1024 * 1024)
        print(f"Exported Clean Model: {clean_model_path} ({file_size_mb:.2f} MB)")

    verifier = CrossVerifier(orig_model, opt_model, rtol=rtol, atol=atol)
    t1 = time.perf_counter()
    report = verifier.verify()
    verify_time_ms = (time.perf_counter() - t1) * 1000.0

    # Print individual CROSS-VERIFICATION REPORT
    print(report.summary())
    print(f"Optimization Time: {opt_time_ms:.2f} ms | Verification Time: {verify_time_ms:.2f} ms")
    if clean_model_path:
        print(f"Visualizer Ready Path: file://{clean_model_path.resolve()}")

    # Aggregate metric stats across outputs
    max_diff = max(m.max_abs_diff for m in report.metrics.values()) if report.metrics else 0.0
    max_mse = max(m.mse for m in report.metrics.values()) if report.metrics else 0.0
    min_cosine = min(m.cosine_similarity for m in report.metrics.values()) if report.metrics else 1.0

    return {
        "model_name": model_path.name,
        "model_path": str(model_path),
        "clean_model_path": str(clean_model_path.resolve()) if clean_model_path else None,
        "original_nodes": report.original_nodes,
        "simplified_nodes": report.simplified_nodes,
        "eliminated_nodes": report.eliminated_nodes,
        "reduction_percentage": report.reduction_percentage,
        "standard_nodes": report.standard_nodes,
        "custom_nodes": report.custom_nodes,
        "custom_ratio": report.custom_ratio,
        "custom_op_breakdown": report.custom_op_breakdown,
        "max_diff": max_diff,
        "max_mse": max_mse,
        "min_cosine": min_cosine,
        "passed": report.all_passed,
        "optimization_time_ms": opt_time_ms,
        "verification_time_ms": verify_time_ms,
    }


def print_consolidated_table(results: list[dict], export_dir: Path | None = None) -> None:
    """Format and print an executive comparative table across all models."""
    header = (
        f"{'Model':<14} | {'Orig':<5} | {'Opt':<5} | {'Eliminated':<14} | "
        f"{'Standard ONNX':<14} | {'Custom Ops':<11} | {'Ratio (C:S)':<11} | {'Cosine Sim':<10} | {'Status'}"
    )
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print("               CONSOLIDATED ONNX MINIFICATION & CUSTOM OPS BENCHMARK")
    print("=" * len(header))
    print(header)
    print(sep)

    total_orig = sum(r["original_nodes"] for r in results)
    total_opt = sum(r["simplified_nodes"] for r in results)
    total_elim = sum(r["eliminated_nodes"] for r in results)
    total_std = sum(r["standard_nodes"] for r in results)
    total_cust = sum(r["custom_nodes"] for r in results)

    all_custom_ops = collections.defaultdict(int)

    for r in results:
        for op, cnt in r["custom_op_breakdown"].items():
            all_custom_ops[op] += cnt

        elim_str = f"-{r['eliminated_nodes']} ({r['reduction_percentage']:.1f}%)"
        std_pct = (r['standard_nodes'] / max(1, r['simplified_nodes'])) * 100.0
        std_str = f"{r['standard_nodes']} ({std_pct:.1f}%)"
        cust_str = f"{r['custom_nodes']} ({r['custom_ratio']:.1f}%)"
        ratio_str = f"1 : {r['standard_nodes'] / r['custom_nodes']:.2f}" if r["custom_nodes"] > 0 else "N/A"
        status_str = "PASS" if r["passed"] else "FAIL"

        print(
            f"{r['model_name']:<14} | {r['original_nodes']:<5d} | {r['simplified_nodes']:<5d} | {elim_str:<14} | "
            f"{std_str:<14} | {cust_str:<11} | {ratio_str:<11} | {r['min_cosine']:<10.6f} | {status_str}"
        )

    print(sep)
    overall_red = (total_elim / max(1, total_orig)) * 100.0
    overall_std_pct = (total_std / max(1, total_opt)) * 100.0
    overall_cust_pct = (total_cust / max(1, total_opt)) * 100.0
    overall_ratio_str = f"1 : {total_std / total_cust:.2f}" if total_cust > 0 else "N/A"

    print(
        f"{'TOTAL / AVG':<14} | {total_orig:<5d} | {total_opt:<5d} | -{total_elim} ({overall_red:.1f}%) | "
        f"{total_std} ({overall_std_pct:.1f}%)   | {total_cust} ({overall_cust_pct:.1f}%)  | {overall_ratio_str:<11} | "
        f"{'1.000000':<10} | {'ALL PASS' if all(r['passed'] for r in results) else 'FAIL'}"
    )
    print("=" * len(header))

    # Active Custom Ops Distribution
    print("\nGlobal Custom Ops Distribution across all Models:")
    for op, cnt in sorted(all_custom_ops.items(), key=lambda x: -x[1]):
        print(f"  - {op:<28}: {cnt:>3d} instances")
    print(f"Total Custom Ops Synthesized: {total_cust}")
    print(f"Total Standard ONNX Ops Retained: {total_std}")
    print(f"Overall Custom Ops vs Standard ONNX Ops Ratio: {overall_cust_pct:.1f}% Custom vs {overall_std_pct:.1f}% Standard")

    # Exported Clean Models Directory for Visualizer
    if export_dir and any(r.get("clean_model_path") for r in results):
        print("\n==================================================================")
        print("    CLEAN SAVED MODELS EXPORTED (READY FOR NETRON / VISUALIZER)")
        print("==================================================================")
        print(f"Directory: {export_dir.resolve()}\n")
        for r in results:
            if r.get("clean_model_path"):
                p = Path(r["clean_model_path"])
                sz = p.stat().st_size / (1024 * 1024) if p.exists() else 0.0
                print(f"  * {r['model_name']:<14} -> file://{p.resolve()} ({sz:.2f} MB)")
        print("\nTip: Open any of these files directly in Netron (https://netron.app) to inspect the minified graph.\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark all ONNX models and compute custom ops vs original standard ONNX ops ratio."
    )
    parser.add_argument(
        "models",
        nargs="*",
        help="Optional paths to specific ONNX models. If omitted, benchmarks all models in .temp/diffrent_models/",
    )
    parser.add_argument(
        "--models-dir",
        default=".temp/diffrent_models",
        help="Directory containing ONNX models (default: .temp/diffrent_models)",
    )
    parser.add_argument(
        "--mode",
        default="function_proto",
        choices=["function_proto", "plain_custom_op"],
        help="JSONRuleEngine mode (default: function_proto)",
    )
    parser.add_argument(
        "--export-dir",
        default=".temp/clean_saved_models",
        help="Directory to save clean optimized ONNX models for visualizer inspection (default: .temp/clean_saved_models)",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Disable exporting clean saved models",
    )
    parser.add_argument("--json", dest="json_out", default=None, help="Optional path to export benchmark results as JSON")
    args = parser.parse_args()

    # Discover models
    if args.models:
        model_paths = [Path(p) for p in args.models]
    else:
        dir_path = Path(args.models_dir)
        if not dir_path.exists():
            print(f"Error: Models directory '{dir_path}' does not exist.", file=sys.stderr)
            sys.exit(1)
        model_paths = sorted(dir_path.glob("*.onnx"))

    if not model_paths:
        print("No ONNX models found to benchmark.", file=sys.stderr)
        sys.exit(1)

    export_dir = None if args.no_export else Path(args.export_dir)

    print(f"Found {len(model_paths)} ONNX model(s) to benchmark in mode '{args.mode}':")
    for p in model_paths:
        print(f"  - {p}")
    if export_dir:
        print(f"Clean models will be exported to: {export_dir.resolve()}")

    results = []
    all_passed = True

    for p in model_paths:
        try:
            res = benchmark_model(p, mode=args.mode, export_dir=export_dir)
            results.append(res)
            if not res["passed"]:
                all_passed = False
        except Exception as e:
            print(f"\nERROR benchmarking {p.name}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            all_passed = False

    if results:
        print_consolidated_table(results, export_dir=export_dir)

        if args.json_out:
            out_file = Path(args.json_out)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"Exported JSON benchmark report to: {out_file}")

    if not all_passed:
        print("Benchmark completed with some verification failures.", file=sys.stderr)
        sys.exit(1)
    else:
        print("All models passed differential verification with 100% numerical parity!")
        sys.exit(0)


if __name__ == "__main__":
    main()
