#!/usr/bin/env python3
"""Prototype & verify Pure Topological Detection for Transformer Blocks."""

import io
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

import onnx
import torch
from torch import nn
import torch.nn.functional as F

from generators.gen_transformer_advanced_zoo import PreLN_TransformerBlock, PostLN_TransformerBlock


def export_to_onnx(model: nn.Module, x: torch.Tensor, strip_names: bool = False) -> onnx.ModelProto:
    buf = io.BytesIO()
    torch.onnx.export(
        model,
        x,
        buf,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
    )
    buf.seek(0)
    onx = onnx.load(buf)
    if strip_names:
        for i, n in enumerate(onx.graph.node):
            n.name = ""  # Strip all node names to test pure topological resilience
    return onx


def detect_transformer_layers_topologically(model: onnx.ModelProto):
    graph = model.graph
    producer_map: dict[str, onnx.NodeProto] = {
        out: n for n in graph.node for out in n.output if out
    }
    consumer_map: dict[str, list[onnx.NodeProto]] = {}
    for n in graph.node:
        for inp in n.input:
            consumer_map.setdefault(inp, []).append(n)

    def can_reach(start_tensor: str, target_tensor: str, max_depth: int = 25) -> bool:
        if not start_tensor or not target_tensor or target_tensor not in producer_map:
            return False
        visited = set()
        queue = [(target_tensor, 0)]
        while queue:
            curr, depth = queue.pop(0)
            if curr == start_tensor:
                return True
            if depth >= max_depth or curr in visited:
                continue
            visited.add(curr)
            p = producer_map.get(curr)
            if p:
                for inp in p.input:
                    if inp:
                        if inp == start_tensor:
                            return True
                        queue.append((inp, depth + 1))
        return False

    merge_nodes = [
        n for n in graph.node
        if n.op_type in ("Add", "SkipLayerNormalization") and len(n.input) >= 2
    ]

    detected_blocks = []
    used_merges = set()

    for m2 in merge_nodes:
        if id(m2) in used_merges:
            continue
        in_a, in_b = m2.input[0], m2.input[1]
        
        # Check if in_b is computed from in_a (FFN branch) or vice-versa
        r1 = None
        ffn_out = None
        if can_reach(in_a, in_b):
            r1 = in_a
            ffn_out = in_b
        elif can_reach(in_b, in_a):
            r1 = in_b
            ffn_out = in_a
        else:
            continue

        # Trace r1 back to find Attention Merge m1
        curr = r1
        # Handle optional intermediate norm (Post-LN or Pre-FFN norm)
        if curr in producer_map and producer_map[curr].op_type in ("LayerNormalization", "RMSNorm"):
            curr = producer_map[curr].input[0]

        if curr not in producer_map:
            continue
        m1 = producer_map[curr]
        if id(m1) in used_merges or m1.op_type not in ("Add", "SkipLayerNormalization") or len(m1.input) < 2:
            continue

        m1_a, m1_b = m1.input[0], m1.input[1]
        x_in = None
        attn_out = None
        if can_reach(m1_a, m1_b):
            x_in = m1_a
            attn_out = m1_b
        elif can_reach(m1_b, m1_a):
            x_in = m1_b
            attn_out = m1_a
        else:
            continue

        # Found candidate block: Input=x_in, Merge1=m1, Merge2=m2
        used_merges.add(id(m1))
        used_merges.add(id(m2))

        # Check for optional post-LN following m2
        final_out = m2.output[0]
        c_m2 = consumer_map.get(final_out, [])
        post_ln = None
        if len(c_m2) == 1 and c_m2[0].op_type in ("LayerNormalization", "RMSNorm"):
            post_ln = c_m2[0]
            final_out = post_ln.output[0]

        # Extract internal nodes between x_in and final_out
        visited_nodes = set()
        block_nodes = []
        def _collect(t: str):
            if not t or t == x_in or t not in producer_map:
                return
            node = producer_map[t]
            if id(node) in visited_nodes:
                return
            visited_nodes.add(id(node))
            block_nodes.append(node)
            for inp in node.input:
                _collect(inp)

        _collect(final_out)
        detected_blocks.append({
            "input": x_in,
            "output": final_out,
            "m1": m1,
            "m2": m2,
            "node_count": len(block_nodes),
            "block_nodes": block_nodes,
        })

    return detected_blocks


def main():
    print("Testing PreLN_TransformerBlock with completely stripped node names...")
    model_pre = PreLN_TransformerBlock(d_model=32, num_heads=4, d_ffn=64)
    x = torch.randn(2, 8, 32)
    onx_pre = export_to_onnx(model_pre, x, strip_names=True)
    blocks_pre = detect_transformer_layers_topologically(onx_pre)
    print(f"PreLN Detected: {len(blocks_pre)} block(s), nodes in block: {blocks_pre[0]['node_count'] if blocks_pre else 0}")
    assert len(blocks_pre) == 1, f"Expected 1 block, found {len(blocks_pre)}"

    print("Testing PostLN_TransformerBlock with completely stripped node names...")
    model_post = PostLN_TransformerBlock(d_model=32, num_heads=4, d_ffn=64)
    onx_post = export_to_onnx(model_post, x, strip_names=True)
    blocks_post = detect_transformer_layers_topologically(onx_post)
    print(f"PostLN Detected: {len(blocks_post)} block(s), nodes in block: {blocks_post[0]['node_count'] if blocks_post else 0}")
    assert len(blocks_post) == 1, f"Expected 1 block, found {len(blocks_post)}"

    print("Testing Multi-Layer Stack (2 stacked layers) with completely stripped names...")
    class TwoLayerTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = PreLN_TransformerBlock(d_model=32, num_heads=4, d_ffn=64)
            self.layer2 = PreLN_TransformerBlock(d_model=32, num_heads=4, d_ffn=64)
        def forward(self, x):
            return self.layer2(self.layer1(x))

    model_two = TwoLayerTransformer()
    onx_two = export_to_onnx(model_two, x, strip_names=True)
    blocks_two = detect_transformer_layers_topologically(onx_two)
    print(f"Stacked 2-Layer Detected: {len(blocks_two)} blocks!")
    for idx, b in enumerate(blocks_two):
        print(f"  Layer {idx}: input={b['input']}, output={b['output']}, nodes={b['node_count']}")
    assert len(blocks_two) == 2, f"Expected 2 blocks, found {len(blocks_two)}"

    print("All pure topological detection tests PASSED!")


if __name__ == "__main__":
    main()
