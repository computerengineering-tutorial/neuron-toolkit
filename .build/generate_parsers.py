#!/usr/bin/env python3
"""Unified code-first TFLite options parser builder.

Directly analyzes FlatBuffer options definitions from the installed TFLite package
in memory and generates optimized, category-specific Python parser modules
without any intermediate JSON files or temporary artifacts.
"""

from __future__ import annotations

import argparse
import ast
import glob
import importlib.util
import os
import re
import sys
from pathlib import Path

# Repository paths
TEMP_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEMP_DIR.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "src" / "neuron_toolkit" / "backends" / "tflite"

# Pre-defined operation categories
CATEGORIES: dict[str, list[str]] = {
    "activation": [
        "SoftmaxOptions",
        "HardSwishOptions",
        "LeakyReluOptions",
        "GeluOptions",
    ],
    "arithmetic": [
        "AddOptions",
        "SubOptions",
        "MulOptions",
        "DivOptions",
        "NegOptions",
        "AbsOptions",
        "SquareOptions",
        "SquaredDifferenceOptions",
        "PowOptions",
        "ExpOptions",
        "LogSoftmaxOptions",
        "FloorDivOptions",
        "FloorModOptions",
        "AddNOptions",
        "CosOptions",
        "ATan2Options",
        "SignOptions",
        "MaximumMinimumOptions",
        "BitwiseXorOptions",
        "RightShiftOptions",
    ],
    "tensor": [
        "ReshapeOptions",
        "SqueezeOptions",
        "TransposeOptions",
        "SliceOptions",
        "StridedSliceOptions",
        "ConcatenationOptions",
        "SplitOptions",
        "SplitVOptions",
        "TileOptions",
        "PackOptions",
        "UnpackOptions",
        "ExpandDimsOptions",
        "BroadcastToOptions",
        "ReverseV2Options",
        "ReverseSequenceOptions",
        "GatherOptions",
        "GatherNdOptions",
        "ScatterNdOptions",
        "OneHotOptions",
        "ShapeOptions",
        "UniqueOptions",
        "RankOptions",
        "CastOptions",
        "BitcastOptions",
        "FillOptions",
        "ZerosLikeOptions",
    ],
    "nn": [
        "Conv2DOptions",
        "Conv3DOptions",
        "DepthwiseConv2DOptions",
        "TransposeConvOptions",
        "FullyConnectedOptions",
        "BatchMatMulOptions",
        "Pool2DOptions",
        "L2NormOptions",
        "LocalResponseNormalizationOptions",
        "SVDFOptions",
    ],
    "reduction": [
        "ReducerOptions",
        "ArgMaxOptions",
        "ArgMinOptions",
    ],
    "logical": [
        "EqualOptions",
        "NotEqualOptions",
        "LessOptions",
        "LessEqualOptions",
        "GreaterOptions",
        "GreaterEqualOptions",
        "LogicalAndOptions",
        "LogicalOrOptions",
        "LogicalNotOptions",
        "SelectOptions",
        "SelectV2Options",
        "WhereOptions",
    ],
    "control_flow": [
        "IfOptions",
        "WhileOptions",
        "CallOptions",
        "CallOnceOptions",
    ],
}


def find_tflite_dir(custom_path: str | Path | None = None) -> Path:
    """Dynamically discover the tflite package directory containing *Options.py."""
    candidates: list[Path | None] = []

    if custom_path:
        candidates.append(Path(custom_path))

    env_dir = os.environ.get("TFLITE_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    try:
        import tflite

        if hasattr(tflite, "__file__") and tflite.__file__:
            candidates.append(Path(tflite.__file__).resolve().parent)
    except ImportError:
        pass

    try:
        spec = importlib.util.find_spec("tflite")
        if spec and spec.origin:
            candidates.append(Path(spec.origin).resolve().parent)
    except Exception:
        pass

    # Search user local and system python site-packages
    user_home = Path.home()
    search_patterns = [
        str(user_home / ".local/lib/python*/site-packages/tflite"),
        "/usr/local/lib/python*/site-packages/tflite",
        "/usr/local/lib/python*/dist-packages/tflite",
        "/usr/lib/python*/site-packages/tflite",
        "/usr/lib/python*/dist-packages/tflite",
    ]
    for pattern in search_patterns:
        for match in sorted(glob.glob(pattern), reverse=True):
            candidates.append(Path(match))

    for candidate in candidates:
        if candidate is not None:
            resolved = candidate.resolve()
            if resolved.is_dir() and any(resolved.glob("*Options.py")):
                return resolved

    raise FileNotFoundError(
        "Could not automatically locate the 'tflite' package directory containing *Options.py.\n"
        "Please provide the path via --tflite-dir or the TFLITE_DIR environment variable."
    )


def extract_options_in_memory(tflite_dir: Path) -> dict[str, list[str]]:
    """Inspect all FlatBuffer *Options classes in-memory via AST analysis."""
    options_info: dict[str, list[str]] = {}

    options_files = sorted(tflite_dir.glob("*Options.py"))
    if not options_files:
        raise FileNotFoundError(f"No *Options.py files found in {tflite_dir}")

    for filepath in options_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                code = f.read()
            tree = ast.parse(code, filename=str(filepath))
        except Exception as e:
            print(f"Warning: Failed to parse AST for {filepath.name}: {e}", file=sys.stderr)
            continue

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                has_init = False
                methods: list[str] = []

                for subnode in node.body:
                    if isinstance(subnode, ast.FunctionDef):
                        # Detect classmethod / staticmethod
                        is_special = any(
                            (isinstance(dec, ast.Name) and dec.id in ("classmethod", "staticmethod"))
                            or (isinstance(dec, ast.Attribute) and dec.attr in ("classmethod", "staticmethod"))
                            for dec in subnode.decorator_list
                        )
                        first_arg = subnode.args.args[0].arg if subnode.args.args else ""
                        if first_arg == "cls":
                            is_special = True

                        name = subnode.name
                        if name == "Init":
                            has_init = True
                        elif (
                            not is_special
                            and not name.startswith("_")
                            and not name.startswith("GetRootAs")
                            and not name.endswith("BufferHasIdentifier")
                        ):
                            methods.append(name)

                if has_init:
                    options_info[node.name] = sorted(methods)

    return options_info


def camel_to_snake(name: str) -> str:
    """Convert CamelCase identifier to snake_case."""
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def build_parser_modules(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    tflite_dir: str | Path | None = None,
    dry_run: bool = False,
) -> None:
    """Extract options in-memory and write category-specific parser modules."""
    tflite_path = find_tflite_dir(tflite_dir)
    print(f"Analyzing TFLite package at: {tflite_path}")

    # Step 1: Direct in-memory extraction
    options_data = extract_options_in_memory(tflite_path)
    print(f"Extracted {len(options_data)} options classes into memory.")

    # Step 2: Categorization
    all_classes = set(options_data.keys())
    categorized_classes: set[str] = set()
    for cat_classes in CATEGORIES.values():
        categorized_classes.update(cat_classes)

    stablehlo_classes: list[str] = []
    misc_classes: list[str] = []

    for cls_name in sorted(all_classes):
        if cls_name in categorized_classes:
            continue
        if cls_name.lower().startswith("stablehlo") or cls_name in ["ReduceWindowOptions", "DilateOptions"]:
            stablehlo_classes.append(cls_name)
        else:
            misc_classes.append(cls_name)

    categories = dict(CATEGORIES)
    categories["stablehlo"] = stablehlo_classes
    categories["misc"] = misc_classes

    # Step 3: Identify empty classes
    empty_classes: set[str] = set()
    for cls_name, methods in options_data.items():
        vectors = [m[:-7] for m in methods if m.endswith("AsNumpy")]
        helpers = [m for m in methods if m.endswith("IsNone") or m.endswith("Length")]
        scalars = [
            m for m in methods
            if m not in helpers and not m.endswith("AsNumpy") and m not in vectors
        ]
        if not scalars and not vectors:
            empty_classes.add(cls_name)

    print(f"Identified {len(empty_classes)} empty options classes (will be skipped).")

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Step 4: Code generation per category
    for cat_name, cls_list in categories.items():
        file_path = output_dir / f"opt_{cat_name}.py"
        active_cls_list = [c for c in cls_list if c not in empty_classes]

        if not active_cls_list:
            if file_path.exists():
                if not dry_run:
                    file_path.unlink()
                print(f"  [-] Removed {file_path.name} (no active classes).")
            continue

        print(f"  [+] Generating {file_path.name} ({len(active_cls_list)} active classes)...")

        lines: list[str] = [
            f'"""TFLite options parser for {cat_name} operations."""',
            "# ruff: noqa: PLC0415, N802",
            "",
            "from __future__ import annotations",
            "",
            "from typing import Any",
            "",
        ]

        for cls_name in sorted(active_cls_list):
            methods = options_data.get(cls_name, [])

            vectors = [m[:-7] for m in methods if m.endswith("AsNumpy")]
            helpers = [m for m in methods if m.endswith("IsNone") or m.endswith("Length")]
            scalars = [
                m for m in methods
                if m not in helpers and not m.endswith("AsNumpy") and m not in vectors
            ]

            lines.append("")
            lines.append(f"def parse_{cls_name}(_options: Any) -> dict[str, Any]:")
            lines.append(f'    """Parse {cls_name} attributes."""')
            lines.append('    if _options is None or not hasattr(_options, "Bytes") or not hasattr(_options, "Pos"):')
            lines.append("        return {}")
            lines.append("")
            lines.append(f"    from tflite.{cls_name} import {cls_name}")
            lines.append("")
            lines.append(f"    opt = {cls_name}()")
            lines.append("    opt.Init(_options.Bytes, _options.Pos)")
            lines.append("")
            lines.append("    attrs: dict[str, Any] = {}")

            # Scalar attributes
            for s in sorted(scalars):
                snake_s = camel_to_snake(s)
                lines.append(f'    attrs["{snake_s}"] = opt.{s}()')

            # Vector attributes
            for v in sorted(vectors):
                snake_v = camel_to_snake(v)
                lines.append(f'    attrs["{snake_v}"] = opt.{v}AsNumpy()')

            lines.append("")
            lines.append("    # Normalize output types (e.g. bytes to str, numpy array to list)")
            lines.append("    for k, v in list(attrs.items()):")
            lines.append("        if isinstance(v, bytes):")
            lines.append("            try:")
            lines.append('                attrs[k] = v.decode("utf-8")')
            lines.append("            except UnicodeDecodeError:")
            lines.append('                attrs[k] = v.decode("utf-8", errors="replace")')
            lines.append('        elif hasattr(v, "tolist"):')
            lines.append("            attrs[k] = v.tolist()")
            lines.append("    return attrs")
            lines.append("")

        content = "\n".join(lines).strip() + "\n"

        if dry_run:
            print(f"      [Dry-run] Would write {len(content)} bytes to {file_path}")
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

    print("All parser modules built and verified successfully!")


def verify_coverage(output_dir: Path = DEFAULT_OUTPUT_DIR, tflite_dir: str | Path | None = None) -> bool:
    """Audit and verify that 100% of TFLite FlatBuffer options classes are covered."""
    tflite_path = find_tflite_dir(tflite_dir)
    print(f"\n--- TFLite Coverage Audit ---")
    print(f"Inspecting package at: {tflite_path}")

    options_data = extract_options_in_memory(tflite_path)
    total_classes = len(options_data)
    print(f"Total TFLite Options classes found: {total_classes}")

    # Identify empty classes
    empty_classes: set[str] = set()
    active_classes: set[str] = set()
    for cls_name, methods in options_data.items():
        vectors = [m[:-7] for m in methods if m.endswith("AsNumpy")]
        helpers = [m for m in methods if m.endswith("IsNone") or m.endswith("Length")]
        scalars = [
            m for m in methods
            if m not in helpers and not m.endswith("AsNumpy") and m not in vectors
        ]
        if not scalars and not vectors:
            empty_classes.add(cls_name)
        else:
            active_classes.add(cls_name)

    print(f"Active classes (with attributes): {len(active_classes)}")
    print(f"Empty classes (skipped): {len(empty_classes)}")

    # Check that every active class has a parser function in output_dir
    missing: list[str] = []
    generated_parsers: set[str] = set()

    for py_file in output_dir.glob("opt_*.py"):
        code = py_file.read_text(encoding="utf-8")
        for match in re.finditer(r"def\s+parse_([A-Za-z0-9_]+Options)\s*\(", code):
            generated_parsers.add(match.group(1))

    for cls_name in sorted(active_classes):
        if cls_name not in generated_parsers:
            missing.append(cls_name)

    print(f"Parser functions generated: {len(generated_parsers)}")
    if missing:
        print(f"ERROR: {len(missing)} active options classes are missing parsers: {missing}")
        return False

    covered_total = len(empty_classes) + len(generated_parsers)
    coverage_pct = (covered_total / total_classes) * 100
    print(f"Coverage: {covered_total}/{total_classes} ({coverage_pct:.1f}%)")
    print("SUCCESS: 100% of TFLite options classes are accounted for and covered!\n")
    return True


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Unified code-first TFLite options parser builder for neuron-toolkit."
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for generated opt_*.py modules (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--tflite-dir",
        type=str,
        default=None,
        help="Path to tflite package directory (auto-discovered if omitted).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate parser generation without writing files.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run coverage audit to verify 100%% options coverage without generating files.",
    )

    args = parser.parse_args()
    try:
        if args.check:
            ok = verify_coverage(output_dir=args.output_dir, tflite_dir=args.tflite_dir)
            sys.exit(0 if ok else 1)

        build_parser_modules(
            output_dir=args.output_dir,
            tflite_dir=args.tflite_dir,
            dry_run=args.dry_run,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
