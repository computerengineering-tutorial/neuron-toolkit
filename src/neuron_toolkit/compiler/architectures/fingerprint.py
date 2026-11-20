"""Content-Hashed Subgraph Fingerprinting and Polymorphic FunctionProto Monomorphization.

Provides deterministic structural hashing of subgraphs to allow different architectural
dynamics (e.g. varying activations, gating ops, or structural permutations) to have their
own specialized, non-colliding FunctionProtos and op_types.
"""

from __future__ import annotations

import collections
import hashlib
import json
import re
from typing import Any

import onnx
from onnx import helper

from neuron_toolkit.compiler.base import LATEST_OPSET, ensure_model_latest_opset


def compute_subgraph_fingerprint(
    nodes: list[onnx.NodeProto],
    external_inputs: list[str] | None = None,
    external_outputs: list[str] | None = None,
) -> str:
    """Compute a deterministic 8-char SHA-256 fingerprint for a subgraph based on topology and ops.

    Tensors are normalized to canonical indices so node renaming does not alter the hash,
    while differences in op_type, wiring, or critical attributes produce distinct hashes.
    """
    if not nodes:
        return "empty"

    # Map external inputs to canonical _in0, _in1, ...
    internal_outputs = {out for node in nodes for out in node.output if out}
    name_map: dict[str, str] = {}
    in_counter = 0
    if external_inputs:
        for inp in external_inputs:
            if inp not in name_map:
                name_map[inp] = f"_in{in_counter}"
                in_counter += 1

    # Map remaining external inputs in topological order of first appearance
    for node in nodes:
        for inp in node.input:
            if inp and inp not in internal_outputs and inp not in name_map:
                name_map[inp] = f"_in{in_counter}"
                in_counter += 1

    # Map external outputs to canonical _out0, _out1, ...
    if external_outputs:
        for idx, out in enumerate(external_outputs):
            name_map[out] = f"_out{idx}"

    # Map internal intermediate tensors to _t0, _t1, ...
    t_counter = 0
    for node in nodes:
        for out in node.output:
            if out and out not in name_map:
                name_map[out] = f"_t{t_counter}"
                t_counter += 1

    canonical_repr: list[dict[str, Any]] = []
    for node in nodes:
        node_dict: dict[str, Any] = {
            "op": node.op_type,
            "domain": node.domain or "",
            "in": [name_map.get(inp, f"_ext_{inp}") for inp in node.input],
            "out": [name_map.get(out, out) for out in node.output],
        }

        # Include structural non-tensor attributes (axes, perm, epsilon, approximate, etc.)
        attrs: dict[str, Any] = {}
        for attr in sorted(node.attribute, key=lambda a: a.name):
            if attr.HasField("f"):
                attrs[attr.name] = round(attr.f, 6)
            elif attr.HasField("i"):
                attrs[attr.name] = attr.i
            elif attr.HasField("s"):
                attrs[attr.name] = attr.s.decode("utf-8", errors="ignore")
            elif attr.ints:
                attrs[attr.name] = list(attr.ints)
            elif attr.floats:
                attrs[attr.name] = [round(x, 6) for x in attr.floats]
            elif attr.strings:
                attrs[attr.name] = [
                    s.decode("utf-8", errors="ignore") for s in attr.strings
                ]
        if attrs:
            node_dict["attrs"] = attrs

        canonical_repr.append(node_dict)

    serialized = json.dumps(canonical_repr, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:8]


def _sanitize_identifier(name: str) -> str:
    """Sanitize any arbitrary string into a strictly valid C/C++ identifier."""
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if cleaned and cleaned[0].isdigit():
        cleaned = f"Op_{cleaned}"
    return cleaned or "Op"


def build_monomorphized_op_type(
    family: str,
    variant_hint: str | None = None,
    fingerprint: str | None = None,
) -> str:
    """Build a specialized, strictly valid C identifier: {family}_{variant_hint}_{fingerprint}."""
    clean_family = _sanitize_identifier(family)
    parts = [clean_family]
    if variant_hint:
        clean_hint = _sanitize_identifier(variant_hint)
        if clean_hint:
            parts.append(clean_hint)
    if fingerprint:
        clean_fp = _sanitize_identifier(fingerprint)
        if clean_fp:
            parts.append(clean_fp)
    return "_".join(parts)


def ensure_specialized_function_proto(
    model: onnx.ModelProto,
    domain: str,
    fname: str,
    formal_inputs: list[str],
    formal_outputs: list[str],
    body_nodes: list[onnx.NodeProto],
    opset_imports: list[onnx.OperatorSetIdProto] | None = None,
    doc_string: str = "",
) -> onnx.FunctionProto:
    """Register a specialized FunctionProto in model.functions if not already present."""
    for existing in model.functions:
        if existing.name == fname and existing.domain == domain:
            return existing

    # Always target modern latest ONNX opset for simplification
    ensure_model_latest_opset(model)

    if opset_imports is None:
        opset_imports = [
            helper.make_opsetid("", LATEST_OPSET),
            helper.make_opsetid("custom.pytorch", 1),
        ]

    # Ensure custom domain is present in model opset_imports
    if domain and not any(op.domain == domain for op in model.opset_import):
        model.opset_import.append(helper.make_opsetid(domain, 1))

    # Topologically sort body_nodes so producers strictly precede consumers
    if len(body_nodes) > 1:
        producer_idx: dict[str, int] = {}
        for idx, n in enumerate(body_nodes):
            for out in n.output:
                if out:
                    producer_idx[out] = idx

        in_degrees = [0] * len(body_nodes)
        graph_deps: dict[int, list[int]] = collections.defaultdict(list)
        for i, n in enumerate(body_nodes):
            for inp in n.input:
                if inp in producer_idx:
                    prod = producer_idx[inp]
                    if prod != i:
                        graph_deps[prod].append(i)
                        in_degrees[i] += 1

        queue = [i for i, deg in enumerate(in_degrees) if deg == 0]
        sorted_nodes = []
        while queue:
            curr = queue.pop(0)
            sorted_nodes.append(body_nodes[curr])
            for dep in graph_deps[curr]:
                in_degrees[dep] -= 1
                if in_degrees[dep] == 0:
                    queue.append(dep)

        if len(sorted_nodes) == len(body_nodes):
            body_nodes = sorted_nodes

    func = helper.make_function(
        domain=domain,
        fname=fname,
        inputs=formal_inputs,
        outputs=formal_outputs,
        nodes=body_nodes,
        opset_imports=opset_imports,
        doc_string=doc_string or f"Specialized polymorphic block {fname}",
    )
    model.functions.append(func)
    return func
