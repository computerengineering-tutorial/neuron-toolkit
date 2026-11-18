#!/usr/bin/env python3
"""Unified Generator Base Engine for Deep Learning Architecture Zoo.

Features:
- Multi-Opset Export (opsets 14, 17, 18, 20).
- Graceful Unsupported Opset Warning (warnings.warn instead of crash).
- Cross-Opset Graph Deduplication: If graph topology/weights are identical between
  opsets, redundant ONNX files are NOT saved (symlinked to canonical).
- Automatic JSON Test Descriptor Emission & Assignment.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path
from typing import Any, Sequence

import onnx
import torch
from torch import nn

TESTS_JSON_DIR = Path("tests/rules_json_tests")
TESTS_JSON_DIR.mkdir(parents=True, exist_ok=True)


def compute_graph_topology_hash(model: onnx.ModelProto) -> str:
    """Compute a deterministic hash of graph topology, nodes, inputs, outputs, and initializers."""
    g = model.graph
    hasher = hashlib.sha256()

    # 1. Inputs and Outputs
    for inp in g.input:
        shape = tuple(d.dim_value for d in inp.type.tensor_type.shape.dim) if inp.type.tensor_type.HasField("shape") else ()
        hasher.update(f"INP:{inp.name}:{shape}:".encode())
    for out in g.output:
        shape = tuple(d.dim_value for d in out.type.tensor_type.shape.dim) if out.type.tensor_type.HasField("shape") else ()
        hasher.update(f"OUT:{out.name}:{shape}:".encode())

    # 2. Nodes (order, op_type, inputs, outputs, attributes)
    for n in g.node:
        hasher.update(f"NODE:{n.op_type}:".encode())
        for i in n.input:
            hasher.update(f"i:{i},".encode())
        for o in n.output:
            hasher.update(f"o:{o},".encode())
        for a in sorted(n.attribute, key=lambda x: x.name):
            hasher.update(f"a:{a.name}:{a.type}:{a.i}:{a.f}:{a.s}:{tuple(a.ints)}:{tuple(a.floats)}:".encode())

    # 3. Initializers
    for init in sorted(g.initializer, key=lambda x: x.name):
        init_hash = hashlib.sha256(init.raw_data).hexdigest() if init.raw_data else ""
        hasher.update(f"INIT:{init.name}:{init.data_type}:{tuple(init.dims)}:{init_hash}:".encode())

    return hasher.hexdigest()


def export_model(
    model: nn.Module,
    dummy_inputs: tuple[torch.Tensor, ...] | torch.Tensor,
    input_names: list[str],
    output_names: list[str],
    base_name: str,
    out_dir: Path,
    opset_versions: Sequence[int] = (14, 17, 18, 20),
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    expected_target_node: str | None = None,
    emit_json_descriptor: bool = False,
    min_expected_reduction: float = 0.0,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> list[str]:
    """Export a PyTorch module across multiple ONNX opsets with deduplication and JSON test specs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    if isinstance(dummy_inputs, torch.Tensor):
        dummy_inputs_tuple: tuple[torch.Tensor, ...] = (dummy_inputs,)
    else:
        dummy_inputs_tuple = dummy_inputs

    exported_paths: list[str] = []
    # Map from topology_hash -> canonical Path
    seen_topologies: dict[str, Path] = {}

    for opset in opset_versions:
        out_path = out_dir / f"{base_name}_opset{opset}.onnx"
        tmp_path = out_dir / f".tmp_{base_name}_opset{opset}.onnx"

        # 1. Export with TorchScript ONNX exporter (dynamo=False for stability across 14-20)
        try:
            torch.onnx.export(
                model,
                dummy_inputs_tuple,
                str(tmp_path),
                input_names=input_names,
                output_names=output_names,
                opset_version=opset,
                do_constant_folding=True,
                dynamic_axes=dynamic_axes,
                dynamo=False,
            )
            m_proto = onnx.load(str(tmp_path))
            onnx.checker.check_model(m_proto)
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            warnings.warn(
                f"Opset {opset} unsupported for '{base_name}': {e}",
                UserWarning,
                stacklevel=2,
            )
            continue

        # 2. Compute topology hash for deduplication
        topo_hash = compute_graph_topology_hash(m_proto)

        if topo_hash in seen_topologies:
            # Identical graph topology already saved!
            canonical_path = seen_topologies[topo_hash]
            if tmp_path.exists():
                tmp_path.unlink()

            # Create symlink so disk space is 0 bytes duplicate, while preserving file accessibility
            if out_path.exists() or out_path.is_symlink():
                out_path.unlink()
            try:
                out_path.symlink_to(canonical_path.name)
                print(
                    f"  [DEDUP] {out_path.name:<45} (opset={opset}) -> identical to {canonical_path.name} (symlinked, 0 KB duplicate)"
                )
            except OSError:
                # Fallback if symlinks not permitted
                import shutil
                shutil.copyfile(str(canonical_path), str(out_path))
                print(
                    f"  [DEDUP] {out_path.name:<45} (opset={opset}) -> identical to {canonical_path.name}"
                )
            exported_paths.append(str(out_path))
        else:
            # First time seeing this topology: commit temporary file to out_path
            if out_path.exists() or out_path.is_symlink():
                out_path.unlink()
            tmp_path.rename(out_path)
            seen_topologies[topo_hash] = out_path

            size_kb = out_path.stat().st_size / 1024.0
            print(
                f"  [OK] Exported {out_path.name:<45} (opset={opset}, nodes={len(m_proto.graph.node):<3}, {size_kb:.1f} KB)"
            )
            exported_paths.append(str(out_path))

    # 3. Create unversioned alias (e.g. base_name.onnx) for backward compatibility & descriptors
    if exported_paths:
        base_alias = out_dir / f"{base_name}.onnx"
        canonical = Path(exported_paths[0])
        if not base_alias.exists() and not base_alias.is_symlink():
            try:
                base_alias.symlink_to(canonical.name)
            except OSError:
                import shutil
                shutil.copyfile(str(canonical), str(base_alias))

    # 4. Emit / assign JSON test descriptor for this block
    if emit_json_descriptor and exported_paths:
        primary_model_path = exported_paths[0]
        # Choose opset 18 if available, else first
        for p in exported_paths:
            if "opset18" in p:
                primary_model_path = p
                break
                primary_model_path = p
                break

        test_spec_file = TESTS_JSON_DIR / f"{base_name}_test.json"
        input_shapes = {
            name: list(t.shape)
            for name, t in zip(input_names, dummy_inputs_tuple)
        }

        spec = {
            "test_id": f"{base_name}_test",
            "model_path": str(primary_model_path),
            "input_shapes": input_shapes,
            "expected_target_node": expected_target_node,
            "min_expected_reduction": min_expected_reduction,
            "rtol": rtol,
            "atol": atol,
        }

        # Do not overwrite if already customized with specific reduction thresholds
        if not test_spec_file.exists():
            with open(test_spec_file, "w", encoding="utf-8") as f:
                json.dump(spec, f, indent=2)
            print(f"  [SPEC] Emitted JSON test descriptor: {test_spec_file.name}")

    return exported_paths
