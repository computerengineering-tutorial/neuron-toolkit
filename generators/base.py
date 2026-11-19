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
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import onnx
import torch
from torch import nn

try:
    import inspect

    from torch.onnx._internal.torchscript_exporter import utils as _ts_utils
    if hasattr(_ts_utils.setup_onnx_logging, "__wrapped__") and not inspect.isgeneratorfunction(_ts_utils.setup_onnx_logging.__wrapped__):
        _ts_utils.setup_onnx_logging = _ts_utils.setup_onnx_logging.__wrapped__
except (ImportError, AttributeError):
    pass

from generators.obfuscation import ObfuscationRegistry


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
    obfuscate_probability: float = 0.5,
    auto_verify: bool = True,
) -> list[str]:
    """Export a PyTorch module across multiple ONNX opsets with deduplication, optional 50% obfuscation, and automatic on-the-fly verification."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    if isinstance(dummy_inputs, torch.Tensor):
        dummy_inputs_tuple: tuple[torch.Tensor, ...] = (dummy_inputs,)
    else:
        dummy_inputs_tuple = dummy_inputs

    exported_paths: list[str] = []
    # Map from topology_hash -> canonical Path
    seen_topologies: dict[str, Path] = {}

    # Determine if this model export session will be obfuscated (50% probability by default)
    is_obfuscated_session = ObfuscationRegistry.should_obfuscate(obfuscate_probability)

    for opset in opset_versions:
        out_path = out_dir / f"{base_name}_opset{opset}.onnx"
        tmp_path = out_dir / f".tmp_{base_name}_opset{opset}.onnx"

        # Apply PyTorch mathematical obfuscation if selected
        current_model = model
        pt_applied: list[str] = []
        if is_obfuscated_session:
            current_model, pt_applied = ObfuscationRegistry.apply_pytorch(
                model, seed=opset * 100 + len(base_name)
            )

        # 1. Export with TorchScript ONNX exporter (utils.export for stability across 14-20)
        try:
            current_model.eval()
            torch.onnx.utils.export(
                current_model,
                dummy_inputs_tuple,
                str(tmp_path),
                input_names=input_names,
                output_names=output_names,
                opset_version=opset,
                do_constant_folding=True,
                dynamic_axes=dynamic_axes,
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

        # Apply ONNX graph mutation if selected
        onnx_applied: list[str] = []
        if is_obfuscated_session:
            try:
                mutated_proto, onnx_applied = ObfuscationRegistry.apply_onnx(
                    m_proto, seed=opset * 200 + len(base_name)
                )
                onnx.checker.check_model(mutated_proto)
                m_proto = mutated_proto
                onnx.save(m_proto, str(tmp_path))
            except Exception as e:
                warnings.warn(
                    f"ONNX obfuscation error for '{base_name}' (opset={opset}): {e}, retaining pre-mutation model",
                    UserWarning,
                    stacklevel=2,
                )

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
            obf_tag = ""
            if is_obfuscated_session:
                strats = pt_applied + onnx_applied
                obf_tag = f" [OBFUSCATED 50%: {', '.join(strats) if strats else 'active'}]"
            print(
                f"  [OK] Exported {out_path.name:<45} (opset={opset}, nodes={len(m_proto.graph.node):<3}, {size_kb:.1f} KB){obf_tag}"
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

    # 4. Automatic On-The-Fly Verification ("jab run kare tab hi automatic test ho jaaye aur jaise koi test naa ho vo rok de program ko")
    if auto_verify and exported_paths:
        canonical_path = Path(exported_paths[0])
        try:
            from neuron_toolkit.compiler.pipeline import NeuronCompiler
            from neuron_toolkit.runtime.cross_verifier import CrossVerifier

            m_loaded = onnx.load(str(canonical_path))
            opt_m = NeuronCompiler.compile(m_loaded, mode="function_proto")
            verifier = CrossVerifier(m_loaded, opt_m, rtol=rtol, atol=atol)
            report = verifier.verify()
            if not report.all_passed:
                raise AssertionError(
                    f"Automatic on-the-fly verification FAILED for {canonical_path.name}:\n{report.summary()}"
                )

            if expected_target_node:
                opt_ops = {n.op_type for n in opt_m.graph.node}
                if not any(expected_target_node in op for op in opt_ops):
                    raise AssertionError(
                        f"Expected target op '{expected_target_node}' not synthesized in {canonical_path.name}. Found ops: {opt_ops}"
                    )
            print(f"  [AUTO-VERIFY OK] On-the-fly verification & parity passed for {base_name}")
        except Exception as e:
            # Halt program immediately
            raise RuntimeError(
                f"[FATAL ON-THE-FLY TEST FAILURE] Verification halted for model '{base_name}': {e}"
            ) from e

    return exported_paths
