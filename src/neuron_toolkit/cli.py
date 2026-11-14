"""Developer CLI for neuron-toolkit."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from neuron_toolkit.graph import NeuronGraph


def inspect_command(args: argparse.Namespace) -> int:
    """Inspect model topology, inputs, outputs, and parameters."""
    graph = NeuronGraph.load(args.model_path)
    print(f"=== NeuronGraph: {args.model_path} ===")
    print(f"Total Nodes: {len(graph.nodes)}")
    print(f"Inputs ({len(graph.inputs)}): {graph.inputs}")
    print(f"Outputs ({len(graph.outputs)}): {graph.outputs}")

    # Op summary
    op_counts: dict[str, int] = {}
    for n in graph.nodes:
        op = getattr(n, "op_type", "Unknown")
        op_counts[op] = op_counts.get(op, 0) + 1

    print("\nOperator Breakdown:")
    for op, cnt in sorted(op_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {op:24}: {cnt:4} nodes")

    print(f"\nInitializers/Weights: {len(graph.tensor_map)} tensors")
    return 0


def optimize_command(args: argparse.Namespace) -> int:
    """Run optimization passes on model and save output."""
    from neuron_toolkit.passes import (
        CommonSubexpressionElimination,
        ConstantFolding,
        DeadCodeElimination,
        FuseConvActivation,
        FuseConvBN,
        FuseGELU,
        FuseLayerNorm,
        PassManager,
    )

    graph = NeuronGraph.load(args.model_path)
    pm = PassManager()

    requested_passes = [p.strip().lower() for p in args.passes.split(",") if p.strip()]
    pass_map = {
        "fuse_conv_bn": FuseConvBN(),
        "conv_bn": FuseConvBN(),
        "dce": DeadCodeElimination(),
        "dead_code": DeadCodeElimination(),
        "const_fold": ConstantFolding(),
        "constant_folding": ConstantFolding(),
        "cse": CommonSubexpressionElimination(),
        "gelu": FuseGELU(),
        "fuse_gelu": FuseGELU(),
        "layernorm": FuseLayerNorm(),
        "fuse_layernorm": FuseLayerNorm(),
        "conv_act": FuseConvActivation(),
        "fuse_conv_act": FuseConvActivation(),
    }

    for p in requested_passes:
        if p in pass_map:
            pm.add_pass(pass_map[p])
        else:
            print(f"Warning: Unknown pass '{p}', skipping.")

    optimized, stats = pm.run(graph)
    print(stats.summary())

    if args.output:
        optimized.save(args.output)
        print(f"Saved optimized model to: {args.output}")

    return 0


def diff_command(args: argparse.Namespace) -> int:
    """Compare two models for structural and operator differences."""
    g1 = NeuronGraph.load(args.model1)
    g2 = NeuronGraph.load(args.model2)

    print(f"Diff: {args.model1} -> {args.model2}")
    print(f"  Node count: {len(g1.nodes)} -> {len(g2.nodes)} ({len(g2.nodes) - len(g1.nodes):+d})")
    print(f"  Inputs:     {g1.inputs} -> {g2.inputs}")
    print(f"  Outputs:    {g1.outputs} -> {g2.outputs}")

    ops1: dict[str, int] = {}
    for n in g1.nodes:
        ops1[getattr(n, "op_type", "")] = ops1.get(getattr(n, "op_type", ""), 0) + 1

    ops2: dict[str, int] = {}
    for n in g2.nodes:
        ops2[getattr(n, "op_type", "")] = ops2.get(getattr(n, "op_type", ""), 0) + 1

    all_ops = sorted(set(ops1.keys()) | set(ops2.keys()))
    print("\nOp differences:")
    for op in all_ops:
        c1 = ops1.get(op, 0)
        c2 = ops2.get(op, 0)
        if c1 != c2:
            print(f"  {op:20}: {c1:4} -> {c2:4} ({c2 - c1:+d})")

    return 0


def export_command(args: argparse.Namespace) -> int:
    """Export model to Netron-compatible JSON, standalone HTML, or Mermaid diagram."""
    graph = NeuronGraph.load(args.model_path)
    fmt = args.format.lower()

    if fmt == "json":

        data = graph.to_json() or ""
        if args.output:
            Path(args.output).write_text(data, encoding="utf-8")
            print(f"Exported JSON to {args.output}")
        else:
            print(data)
    elif fmt == "html":
        out = args.output or "model_viz.html"
        graph.to_html(out)
        print(f"Exported interactive HTML visualization to {out}")
    elif fmt == "mermaid":
        mermaid_text = graph.to_mermaid()
        if args.output:
            Path(args.output).write_text(mermaid_text, encoding="utf-8")
            print(f"Exported Mermaid diagram to {args.output}")
        else:
            print(mermaid_text)
    elif fmt == "safetensors":
        out = args.output or "weights.safetensors"
        graph.save_safetensors(out)
        print(f"Exported weights to Safetensors file: {out}")
    else:
        print(f"Error: Unknown format '{fmt}'. Choose from json, html, mermaid, safetensors.")
        return 1

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI main entry point."""
    parser = argparse.ArgumentParser(
        prog="neuron",
        description="Neuron Toolkit: Unified deep learning graph inspector, optimizer, and exporter.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Inspect
    p_inspect = subparsers.add_parser("inspect", help="Inspect model graph topology and statistics.")
    p_inspect.add_argument("model_path", help="Path to ONNX, TFLite, or JSON graph file.")

    # Optimize
    p_opt = subparsers.add_parser("optimize", help="Run optimization passes on model graph.")
    p_opt.add_argument("model_path", help="Path to input model.")
    p_opt.add_argument("--passes", default="fuse_conv_bn,dce", help="Comma-separated list of passes.")
    p_opt.add_argument("-o", "--output", help="Path to save optimized model.")

    # Diff
    p_diff = subparsers.add_parser("diff", help="Diff two models for structural changes.")
    p_diff.add_argument("model1", help="First model path.")
    p_diff.add_argument("model2", help="Second model path.")

    # Export
    p_export = subparsers.add_parser("export", help="Export graph to JSON, HTML, Mermaid, or Safetensors.")
    p_export.add_argument("model_path", help="Path to input model.")
    p_export.add_argument("--format", default="json", choices=["json", "html", "mermaid", "safetensors"])
    p_export.add_argument("-o", "--output", help="Path to save output file.")

    parsed = parser.parse_args(argv)

    if parsed.command == "inspect":
        return inspect_command(parsed)
    if parsed.command == "optimize":
        return optimize_command(parsed)
    if parsed.command == "diff":
        return diff_command(parsed)
    if parsed.command == "export":
        return export_command(parsed)

    return 0


if __name__ == "__main__":
    sys.exit(main())
