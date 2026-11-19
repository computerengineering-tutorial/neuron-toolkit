#!/usr/bin/env python3
"""Unified Registry, Metadata Specification & Discovery Engine for DL Architecture Generators."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import onnx
import torch
from torch import nn

from generators.base import export_model as unified_export_model

DEFAULT_CATEGORY_SUBFOLDERS: dict[str, str] = {
    "recurrent": "recurrent",
    "cnn_vision": "cnn_vision",
    "vision_cnn": "cnn_vision",
    "mlp_vanilla": "mlp_vanilla",
    "vanilla_mlp": "mlp_vanilla",
    "gnn": "gnn",
    "transformer_advanced": "transformer_advanced",
    "attention_variants": "attention_variants",
    "comprehensive_attentions": "attention_variants",
    "ssm_mamba": "ssm_mamba",
    "diffusion_dit": "diffusion_dit",
    "hierarchical_vit": "hierarchical_vit",
    "obfuscated_stress": "obfuscated_stress",
    "activation_variants": "",
    "linear_variants": "",
    "norm_variants": "",
    "transformer_blocks": "",
}


@dataclass
class ModelSpec:
    """Specification describing a deep learning model architecture and its export parameters."""

    name: str
    target: type[nn.Module] | Callable[[], nn.Module]
    category: str
    subfolder: str | None = None
    shapes: list[tuple[int, ...]] | None = None
    input_factory: Callable[[], tuple[torch.Tensor, ...] | torch.Tensor] | None = None
    input_names: list[str] = field(default_factory=lambda: ["x"])
    output_names: list[str] = field(default_factory=lambda: ["out"])
    opset_versions: Sequence[int] = (14, 17, 18, 20)
    dynamic_axes: dict[str, dict[int, str]] | None = None
    expected_target_node: str | None = None
    min_expected_reduction: float = 0.0
    rtol: float = 1e-4
    atol: float = 1e-4
    emit_json_descriptor: bool = False
    init_args: tuple = ()
    init_kwargs: dict = field(default_factory=dict)
    description: str = ""
    tags: list[str] = field(default_factory=list)

    def resolve_subfolder(self) -> str:
        if self.subfolder is not None:
            return self.subfolder
        return DEFAULT_CATEGORY_SUBFOLDERS.get(self.category, self.category)

    def instantiate(self) -> nn.Module:
        """Instantiate the PyTorch nn.Module instance."""
        if inspect.isclass(self.target) and issubclass(self.target, nn.Module):
            model = self.target(*self.init_args, **self.init_kwargs)
        elif callable(self.target):
            model = self.target(*self.init_args, **self.init_kwargs)
        else:
            raise TypeError(f"Target for {self.name} must be a class or callable, got {type(self.target)}")
        model.eval()
        return model

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Generate dummy input tensors for ONNX export."""
        if self.input_factory is not None:
            inputs = self.input_factory()
            if isinstance(inputs, torch.Tensor):
                return (inputs,)
            return tuple(inputs)
        if self.shapes is not None:
            return tuple(torch.randn(*s) for s in self.shapes)
        # Default fallback
        return (torch.randn(1, 16, 64),)

    def get_out_dir(self, base_dir: Path = Path(".helper/models")) -> Path:
        sub = self.resolve_subfolder()
        return base_dir / sub if sub else base_dir

    def get_model_path(self, opset: int | None = None, base_dir: Path = Path(".helper/models")) -> Path:
        out_dir = self.get_out_dir(base_dir)
        if opset is not None:
            return out_dir / f"{self.name}_opset{opset}.onnx"
        return out_dir / f"{self.name}.onnx"


class ModelRegistry:
    """Central registry and execution manager for DL Architecture Generators."""

    _registry: dict[str, ModelSpec] = {}

    @classmethod
    def register(cls, spec: ModelSpec) -> ModelSpec:
        """Register a model specification."""
        cls._registry[spec.name] = spec
        return spec

    @classmethod
    def get(cls, name: str) -> ModelSpec:
        if name not in cls._registry:
            raise KeyError(f"Model '{name}' not found in registry. Available: {sorted(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def list_models(cls) -> list[ModelSpec]:
        return list(cls._registry.values())

    @classmethod
    def get_by_category(cls, category: str) -> list[ModelSpec]:
        return [spec for spec in cls._registry.values() if spec.category == category]

    @classmethod
    def get_categories(cls) -> list[str]:
        return sorted({spec.category for spec in cls._registry.values()})

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()

    @classmethod
    def discover(cls, directory: str | Path | None = None) -> list[ModelSpec]:
        """Automatically discover and import all generator modules and addons."""
        if directory is None:
            directory = Path(__file__).resolve().parent
        else:
            directory = Path(directory).resolve()

        if str(directory.parent) not in sys.path:
            sys.path.insert(0, str(directory.parent))

        # 1. Search for gen_*.py files
        gen_files = sorted(directory.glob("gen_*.py"))
        # 2. Search for addons/*.py files
        addon_files = sorted((directory / "addons").glob("*.py")) if (directory / "addons").exists() else []

        all_files = [f for f in gen_files + addon_files if f.name not in ("gen_all_dl_architectures.py", "__init__.py")]

        for py_path in all_files:
            mod_name = py_path.stem
            pkg = directory.name
            full_mod_name = f"{pkg}.{mod_name}" if py_path.parent == directory else f"{pkg}.addons.{mod_name}"
            try:
                if full_mod_name not in sys.modules:
                    spec = importlib.util.spec_from_file_location(full_mod_name, str(py_path))
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        sys.modules[full_mod_name] = mod
                        spec.loader.exec_module(mod)
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to auto-discover module {py_path.name}: {e}", RuntimeWarning, stacklevel=2)

        return cls.list_models()

    @classmethod
    def export_spec(
        cls,
        spec: ModelSpec,
        base_out_dir: Path = Path(".helper/models"),
        opsets: Sequence[int] | None = None,
        obfuscate_probability: float = 0.5,
        auto_verify: bool = True,
    ) -> list[str]:
        """Export a single registered model specification across its opsets with optional 50% obfuscation and automatic verification."""
        try:
            model = spec.instantiate()
        except Exception as e:
            import warnings
            warnings.warn(f"Skipping model '{spec.name}' due to instantiation failure: {e}", UserWarning, stacklevel=2)
            return []

        dummy_inputs = spec.get_dummy_inputs()
        out_dir = spec.get_out_dir(base_out_dir)
        target_opsets = opsets or spec.opset_versions

        return unified_export_model(
            model=model,
            dummy_inputs=dummy_inputs,
            input_names=spec.input_names,
            output_names=spec.output_names,
            base_name=spec.name,
            out_dir=out_dir,
            opset_versions=target_opsets,
            dynamic_axes=spec.dynamic_axes,
            expected_target_node=spec.expected_target_node,
            min_expected_reduction=spec.min_expected_reduction,
            rtol=spec.rtol,
            atol=spec.atol,
            obfuscate_probability=obfuscate_probability,
            auto_verify=auto_verify,
        )

    @classmethod
    def export_all(
        cls,
        base_out_dir: Path = Path(".helper/models"),
        category: str | None = None,
        opsets: Sequence[int] | None = None,
        names: Sequence[str] | None = None,
        obfuscate_probability: float = 0.5,
        auto_verify: bool = True,
    ) -> dict[str, list[str]]:
        """Export all registered models matching category and names filter with optional 50% obfuscation and automatic verification."""
        specs = cls.list_models()
        if category:
            specs = [s for s in specs if s.category == category]
        if names:
            name_set = set(names)
            specs = [s for s in specs if s.name in name_set]

        results: dict[str, list[str]] = {}
        for spec in specs:
            print(f"\n>>> Exporting Architecture: [{spec.category}] {spec.name}")
            try:
                paths = cls.export_spec(
                    spec,
                    base_out_dir=base_out_dir,
                    opsets=opsets,
                    obfuscate_probability=obfuscate_probability,
                    auto_verify=auto_verify,
                )
                results[spec.name] = paths
            except Exception as e:
                print(f"  [ERROR] Failed to export {spec.name}: {e}")
                results[spec.name] = []

        return results

    @classmethod
    def build_catalog(cls, models_root: Path = Path(".helper/models")) -> dict:
        """Scan models directory and build model_catalog.json index."""
        all_onnx = sorted(models_root.glob("**/*.onnx"))
        catalog = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_models": len(all_onnx),
            "models": [],
        }

        by_category: dict[str, int] = {}
        by_opset: dict[int, int] = {}

        for p in all_onnx:
            rel = p.relative_to(models_root)
            cat = rel.parts[0] if len(rel.parts) > 1 else "root"
            by_category[cat] = by_category.get(cat, 0) + 1

            try:
                m = onnx.load(str(p), load_external_data=False)
                opset = m.opset_import[0].version if m.opset_import else -1
                by_opset[opset] = by_opset.get(opset, 0) + 1
                node_count = len(m.graph.node)
                inputs = [
                    {
                        "name": inp.name,
                        "shape": [
                            d.dim_value if d.dim_value > 0 else -1
                            for d in inp.type.tensor_type.shape.dim
                        ],
                    }
                    for inp in m.graph.input
                ]
                outputs = [
                    {
                        "name": out.name,
                        "shape": [
                            d.dim_value if d.dim_value > 0 else -1
                            for d in out.type.tensor_type.shape.dim
                        ],
                    }
                    for out in m.graph.output
                ]
            except Exception:
                opset = -1
                node_count = -1
                inputs, outputs = [], []

            catalog["models"].append(
                {
                    "path": str(p),
                    "name": p.name,
                    "category": cat,
                    "opset": opset,
                    "nodes": node_count,
                    "size_kb": round(p.stat().st_size / 1024.0, 2),
                    "inputs": inputs,
                    "outputs": outputs,
                }
            )

        catalog["breakdown_by_category"] = by_category
        catalog["breakdown_by_opset"] = by_opset

        catalog_path = models_root / "model_catalog.json"
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        with open(catalog_path, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2)

        return catalog

    @classmethod
    def cli(cls, argv: list[str] | None = None) -> None:
        """Unified command-line interface for generators."""
        parser = argparse.ArgumentParser(prog="neuron-generators", description="DL Architecture Zoo Generator CLI")
        subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

        # export command
        export_p = subparsers.add_parser("export", help="Export models to ONNX")
        export_p.add_argument("--category", "-c", type=str, default=None, help="Filter by category")
        export_p.add_argument("--model", "-m", type=str, default=None, help="Export specific model name")
        export_p.add_argument("--out-dir", "-o", type=Path, default=Path(".helper/models"), help="Output directory")
        export_p.add_argument("--opsets", nargs="+", type=int, default=None, help="Opsets (e.g. 14 17 18 20)")
        export_p.add_argument("--catalog", action="store_true", help="Rebuild catalog after export")
        export_p.add_argument("--obfuscate-prob", type=float, default=0.5, help="Probability of applying obfuscation (0.0 to 1.0, default: 0.5)")
        export_p.add_argument("--no-obfuscate", action="store_true", help="Disable randomized obfuscation")
        export_p.add_argument("--no-verify", action="store_true", help="Skip automatic on-the-fly verification")

        # list command
        list_p = subparsers.add_parser("list", help="List registered models and categories")
        list_p.add_argument("--category", "-c", type=str, default=None, help="Filter by category")

        # catalog command
        cat_p = subparsers.add_parser("catalog", help="Build or rebuild model_catalog.json")
        cat_p.add_argument("--out-dir", "-o", type=Path, default=Path(".helper/models"), help="Output directory")

        args = parser.parse_args(argv)

        # Auto-discover models
        cls.discover()

        if args.command == "export" or args.command is None:
            names = [args.model] if getattr(args, "model", None) else None
            prob = 0.0 if getattr(args, "no_obfuscate", False) else getattr(args, "obfuscate_prob", 0.5)
            verify = not getattr(args, "no_verify", False)
            cls.export_all(
                base_out_dir=getattr(args, "out_dir", Path(".helper/models")),
                category=getattr(args, "category", None),
                opsets=getattr(args, "opsets", None),
                names=names,
                obfuscate_probability=prob,
                auto_verify=verify,
            )
            if getattr(args, "catalog", True):
                cls.build_catalog(getattr(args, "out_dir", Path(".helper/models")))

        elif args.command == "list":
            specs = cls.list_models()
            if args.category:
                specs = [s for s in specs if s.category == args.category]
            print(f"\nRegistered Architecture Models ({len(specs)}):")
            print(f"{'Name':<35} | {'Category':<22} | {'Opsets':<15} | {'Expected Target'}")
            print("-" * 90)
            for s in specs:
                opsets_str = ",".join(str(o) for o in s.opset_versions)
                target = s.expected_target_node or "-"
                print(f"{s.name:<35} | {s.category:<22} | {opsets_str:<15} | {target}")
            print("-" * 90)

        elif args.command == "catalog":
            cat = cls.build_catalog(args.out_dir)
            print(f"Catalog rebuilt at {args.out_dir}/model_catalog.json. Total models: {cat['total_models']}")
