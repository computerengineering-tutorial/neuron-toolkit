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
        FuseAttention,
        FuseConvActivation,
        FuseConvBN,
        FuseGELU,
        FuseLayerNorm,
        FuseRMSNorm,
        FuseRotaryEmbedding,
        FuseSwiGLU,
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
        "fuse_rmsnorm": FuseRMSNorm(),
        "rmsnorm": FuseRMSNorm(),
        "fuse_swiglu": FuseSwiGLU(),
        "swiglu": FuseSwiGLU(),
        "fuse_attention": FuseAttention(),
        "attention": FuseAttention(),
        "sdpa": FuseAttention(),
        "fuse_rope": FuseRotaryEmbedding(),
        "rope": FuseRotaryEmbedding(),
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
    print(
        f"  Node count: {len(g1.nodes)} -> {len(g2.nodes)} ({len(g2.nodes) - len(g1.nodes):+d})"
    )
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
        print(
            f"Error: Unknown format '{fmt}'. Choose from json, html, mermaid, safetensors."
        )
        return 1

    return 0


def ast_command(args: argparse.Namespace) -> int:
    """AST-based codebase analysis, symbol querying, and codemap generation."""
    import json

    task_helper_path = (
        Path(__file__).resolve().parents[2] / ".task_helper" / "ast_graph.py"
    )
    if not task_helper_path.exists():
        task_helper_path = Path(".task_helper/ast_graph.py")
    if not task_helper_path.exists():
        print(
            f"Error: Could not locate ast_graph.py at {task_helper_path}",
            file=sys.stderr,
        )
        return 1

    try:
        task_helper_dir = str(task_helper_path.resolve().parent)
        if task_helper_dir not in sys.path:
            sys.path.insert(0, task_helper_dir)
        import ast_graph

        code_graph = ast_graph.CodeGraph()
    except Exception as e:
        print(f"Error loading ast_graph helper: {e}", file=sys.stderr)
        return 1

    if args.file:
        code_graph.parse_file(args.file)
    else:
        code_graph.parse_directory(args.dir)

    if args.query or args.doc_query or args.missing_docs:
        results = code_graph.query_symbols(
            name_pattern=args.query,
            doc_keyword=args.doc_query,
            missing_doc=args.missing_docs,
        )
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(f"Found {len(results)} matches:")
            for r in results:
                doc = (
                    r["docstring"].strip().splitlines()[0]
                    if r["docstring"]
                    else "NO DOCSTRING"
                )
                sig = f" {r.get('signature', '')}" if "signature" in r else ""
                print(f"[{r['type'].upper():8}] {r['qualname']}{sig}")
                print(
                    f"           File: {r['file']}:{r['lines'][0]}-{r['lines'][1]}"
                )
                print(f"           Doc:  {doc}")
        return 0

    if args.codemap:
        print(code_graph.generate_codemap(filter_file=args.file))
        return 0

    print(f"Parsed {len(code_graph.modules)} Python files in '{args.dir}'.")
    print("Use --codemap, --query <regex>, --doc-query <kw>, or --missing-docs.")
    return 0


def profile_command(args: argparse.Namespace) -> int:
    """Profile FLOPs, MACs, memory timeline, and Roofline latency bounds."""
    import json

    from neuron_toolkit.profiler import profile_graph

    graph = NeuronGraph.load(args.model_path)
    report = profile_graph(graph, device=args.device, dtype=args.dtype)

    if args.format == "json":
        data = json.dumps(report.to_dict(), indent=2)
        if args.output:
            Path(args.output).write_text(data, encoding="utf-8")
            print(f"Saved JSON profile to {args.output}")
        else:
            print(data)
    elif args.format == "markdown":
        md = report.to_markdown()
        if args.output:
            Path(args.output).write_text(md, encoding="utf-8")
            print(f"Saved Markdown profile to {args.output}")
        else:
            print(md)
    else:
        print(report.summary())

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI main entry point."""
    parser = argparse.ArgumentParser(
        prog="neuron",
        description="Neuron Toolkit: Unified deep learning graph inspector, optimizer, and exporter.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Inspect
    p_inspect = subparsers.add_parser(
        "inspect", help="Inspect model graph topology and statistics."
    )
    p_inspect.add_argument(
        "model_path", help="Path to ONNX, TFLite, or JSON graph file."
    )

    # Profile
    p_profile = subparsers.add_parser(
        "profile", help="Profile FLOPs, memory footprint, and Roofline bounds."
    )
    p_profile.add_argument("model_path", help="Path to input model.")
    p_profile.add_argument(
        "--device",
        default="a100",
        help="Hardware device (a100, h100, 4090, m3, arm, cpu).",
    )
    p_profile.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "float32", "int8"],
        help="Computation precision.",
    )
    p_profile.add_argument(
        "--format",
        default="table",
        choices=["table", "json", "markdown"],
        help="Output format.",
    )
    p_profile.add_argument(
        "-o", "--output", help="Optional output path to write profile."
    )

    # Optimize
    p_opt = subparsers.add_parser(
        "optimize", help="Run optimization passes on model graph."
    )
    p_opt.add_argument("model_path", help="Path to input model.")
    p_opt.add_argument(
        "--passes", default="fuse_conv_bn,dce", help="Comma-separated list of passes."
    )
    p_opt.add_argument("-o", "--output", help="Path to save optimized model.")

    # Diff
    p_diff = subparsers.add_parser(
        "diff", help="Diff two models for structural changes."
    )
    p_diff.add_argument("model1", help="First model path.")
    p_diff.add_argument("model2", help="Second model path.")

    # Export
    p_export = subparsers.add_parser(
        "export", help="Export graph to JSON, HTML, Mermaid, or Safetensors."
    )
    p_export.add_argument("model_path", help="Path to input model.")
    p_export.add_argument(
        "--format", default="json", choices=["json", "html", "mermaid", "safetensors"]
    )
    p_export.add_argument("-o", "--output", help="Path to save output file.")

    # AST Codebase Inspection & Codemap
    p_ast = subparsers.add_parser(
        "ast",
        help="AST-based codebase analysis, symbol querying, and codemap hierarchy.",
    )
    p_ast.add_argument(
        "--dir", default="src", help="Directory to parse (default: src)."
    )
    p_ast.add_argument("--file", help="Specific file to parse.")
    p_ast.add_argument(
        "--codemap", action="store_true", help="Print hierarchical ASCII codemap."
    )
    p_ast.add_argument("--query", help="Query symbol name (regex).")
    p_ast.add_argument("--doc-query", help="Search docstring keyword.")
    p_ast.add_argument(
        "--missing-docs", action="store_true", help="Find undocumented symbols."
    )
    p_ast.add_argument("--json", action="store_true", help="Output in JSON.")

    parsed = parser.parse_args(argv)

    if parsed.command == "inspect":
        return inspect_command(parsed)
    if parsed.command == "profile":
        return profile_command(parsed)
    if parsed.command == "optimize":
        return optimize_command(parsed)
    if parsed.command == "diff":
        return diff_command(parsed)
    if parsed.command == "export":
        return export_command(parsed)
    if parsed.command == "ast":
        return ast_command(parsed)

    return 0


if __name__ == "__main__":
    sys.exit(main())
