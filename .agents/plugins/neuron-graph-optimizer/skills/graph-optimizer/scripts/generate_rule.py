#!/usr/bin/env python3
"""Rule Generator: Scaffold declarative JSON pattern rules and PyTorch runtime op handlers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


JSON_RULE_TEMPLATE = {
    "rule_id": "{rule_id}",
    "description": "{description}",
    "pattern": {
        "nodes": [
            {
                "id": "root_node",
                "op_type": "{primary_op}",
                "inputs": ["$x"],
                "outputs": ["$intermediate"]
            }
        ],
        "conditions": [
            {
                "type": "is_initializer",
                "tensor": "$weight"
            }
        ]
    },
    "action": {
        "type": "fuse_to_function",
        "op_type": "{custom_op}",
        "domain": "custom.pytorch",
        "pytorch_module": "torch.nn.modules.{custom_op}",
        "inputs": ["$x"],
        "outputs": ["$out"],
        "attributes": {}
    }
}

PYTORCH_HANDLER_SNIPPET = '''
        @self.register("custom.pytorch", "{custom_op}")
        @self.register("", "{custom_op}")
        def _{fn_name}(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            x = inputs[0]
            # Custom PyTorch implementation for {custom_op}
            # e.g.:
            # return torch.ops.my_custom_op(x)
            return x
'''

ENGINE_METHOD_SNIPPET = '''
    # -------------------------------------------------------------------------
    # Rule: {custom_op} Fusion
    # -------------------------------------------------------------------------
    def apply_{fn_name}_fusion(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse {primary_op} subgraph into {custom_op}."""
        graph = model.graph
        nodes_to_remove: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        count = 0

        for n in graph.node:
            if n.op_type == "{primary_op}":
                # Match matching subgraph conditions here
                fused_node = helper.make_node(
                    op_type="{custom_op}",
                    inputs=n.input,
                    outputs=n.output,
                    name=f"{{n.name}}_fused_{fn_name}",
                    domain="custom.pytorch",
                )
                new_nodes.append(fused_node)
                nodes_to_remove.add(n.name or n.output[0])
                count += 1

        if not new_nodes:
            return model

        surviving = [n for n in graph.node if (n.name or n.output[0]) not in nodes_to_remove] + new_nodes
        del graph.node[:]
        graph.node.extend(surviving)

        log.info("{custom_op} Fusion: Fused %d instances", count)
        return model
'''


def scaffold_rule(rule_id: str, primary_op: str, custom_op: str, description: str, out_dir: str | Path | None = None) -> None:
    fn_name = rule_id.lower().replace("-", "_").replace(" ", "_")
    if out_dir is None:
        out_dir = Path("src/neuron_toolkit/rules/definitions")
    else:
        out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{fn_name}.json"

    data = json.loads(json.dumps(JSON_RULE_TEMPLATE))
    data["rule_id"] = fn_name
    data["description"] = description
    data["pattern"]["nodes"][0]["op_type"] = primary_op
    data["action"]["op_type"] = custom_op
    data["action"]["pytorch_module"] = f"torch.nn.modules.{custom_op}"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Generated declarative JSON rule at: {json_path}")
    print("\n" + "=" * 60)
    print("Code to paste into 'src/neuron_toolkit/runtime/registry.py':")
    print("=" * 60)
    print(PYTORCH_HANDLER_SNIPPET.format(custom_op=custom_op, fn_name=fn_name))

    print("=" * 60)
    print("Code to paste into 'src/neuron_toolkit/rules/engine.py':")
    print("=" * 60)
    print(ENGINE_METHOD_SNIPPET.format(primary_op=primary_op, custom_op=custom_op, fn_name=fn_name))


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaffold a new ONNX rewrite rule and PyTorch op handler.")
    parser.add_argument("--id", required=True, help="Rule ID (e.g. rotary_embedding_fusion)")
    parser.add_argument("--primary-op", required=True, help="Root/trigger ONNX op_type (e.g. Cos, ReduceL2)")
    parser.add_argument("--custom-op", required=True, help="Target custom op_type (e.g. RotaryEmbedding)")
    parser.add_argument("--desc", default="Fuses repeating subgraphs into a unified custom op", help="Rule description")
    parser.add_argument("--out-dir", default=None, help="Output directory for rule JSON")
    args = parser.parse_args()

    scaffold_rule(args.id, args.primary_op, args.custom_op, args.desc, args.out_dir)


if __name__ == "__main__":
    main()
