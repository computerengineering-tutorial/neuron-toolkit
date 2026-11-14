"""Visualization module for neuron-toolkit."""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuron_toolkit.graph import NeuronGraph


def to_mermaid(graph: NeuronGraph, max_nodes: int = 150) -> str:
    """Generate a Mermaid flowchart diagram from a NeuronGraph."""
    lines = ["flowchart TD"]

    # Graph inputs
    for idx, inp in enumerate(graph.inputs):
        safe_inp = html.escape(str(inp))
        lines.append(f'    in_{idx}(["Input: {safe_inp}"])')

    # Map output tensor name to producing node id
    producer_map: dict[str, str] = {}

    nodes_to_render = graph.nodes[:max_nodes]
    for i, node in enumerate(nodes_to_render):
        node_id = f"node_{i}"
        op_type = getattr(node, "op_type", "Unknown")
        name = getattr(node, "name", "") or f"node_{i}"
        label = f"{op_type}\\n({name})" if name else op_type
        lines.append(f'    {node_id}["{label}"]')

        for out in getattr(node, "output", []):
            if out:
                producer_map[out] = node_id

    # Connect inputs to first consuming nodes
    for idx, inp in enumerate(graph.inputs):
        for i, node in enumerate(nodes_to_render):
            if inp in getattr(node, "input", []):
                lines.append(f"    in_{idx} --> node_{i}")

    # Connect internal node edges
    for i, node in enumerate(nodes_to_render):
        node_id = f"node_{i}"
        lines.extend(
            f"    {producer_map[inp]} --> {node_id}"
            for inp in getattr(node, "input", [])
            if inp in producer_map and producer_map[inp] != node_id
        )

    # Graph outputs
    for idx, out in enumerate(graph.outputs):
        safe_out = html.escape(str(out))
        out_id = f"out_{idx}"
        lines.append(f'    {out_id}(["Output: {safe_out}"])')
        if out in producer_map:
            lines.append(f"    {producer_map[out]} --> {out_id}")

    if len(graph.nodes) > max_nodes:
        lines.append(f'    trunc["... (+{len(graph.nodes) - max_nodes} more nodes)"]')

    return "\n".join(lines)


def to_html(graph: NeuronGraph, output_path: str | None = None) -> str:
    """Export graph as a standalone interactive HTML visualizer page."""
    mermaid_code = to_mermaid(graph)
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NeuronGraph Visualizer - {html.escape(getattr(graph, "name", "Model"))}</title>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true, theme: 'neutral' }});
    </script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 24px;
            background: #f8fafc;
            color: #1e293b;
        }}
        .header {{
            margin-bottom: 24px;
            padding-bottom: 12px;
            border-bottom: 1px solid #e2e8f0;
        }}
        .badge {{
            display: inline-block;
            background: #0284c7;
            color: white;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 600;
        }}
        .canvas {{
            background: white;
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
            overflow: auto;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h2>NeuronGraph Topology</h2>
        <span class="badge">{len(graph.nodes)} Nodes</span>
        <span class="badge">{len(graph.inputs)} Inputs</span>
        <span class="badge">{len(graph.outputs)} Outputs</span>
    </div>
    <div class="canvas">
        <pre class="mermaid">
{mermaid_code}
        </pre>
    </div>
</body>
</html>
"""
    if output_path:
        from pathlib import Path

        Path(output_path).write_text(html_content, encoding="utf-8")

    return html_content
