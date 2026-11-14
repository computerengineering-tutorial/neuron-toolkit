"""Unified table-driven and reflection-based TFLite BuiltinOptions parser.

Replaces ~2,500 lines of repetitive opt_*.py boilerplate across 8 files
with a single consolidated, robust attribute extractor.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import re
from typing import Any

log = logging.getLogger("neuron_toolkit.backends.tflite.options")

_PASCAL_TO_SNAKE_1 = re.compile(r"(.)([A-Z][a-z]+)")
_PASCAL_TO_SNAKE_2 = re.compile(r"([a-z0-9])([A-Z])")


def _pascal_to_snake(name: str) -> str:
    """Convert PascalCase or CamelCase string to snake_case."""
    s1 = _PASCAL_TO_SNAKE_1.sub(r"\1_\2", name)
    return _PASCAL_TO_SNAKE_2.sub(r"\1_\2", s1).lower()


_IGNORED_METHOD_NAMES = frozenset({"Init", "GetRootAs"})


def parse_tflite_builtin_options(opt_class_name: str, options: Any) -> dict[str, Any]:
    """Extract all attributes from a TFLite BuiltinOptions FlatBuffer table."""
    if options is None or not hasattr(options, "Bytes") or not hasattr(options, "Pos"):
        return {}

    try:
        mod = importlib.import_module(f"tflite.{opt_class_name}")
        cls = getattr(mod, opt_class_name)
    except (ImportError, AttributeError):
        return {}

    table = cls()
    table.Init(options.Bytes, options.Pos)

    attrs: dict[str, Any] = {}
    class_attrs = list(dict.fromkeys(dir(table) + dir(cls)))

    # 1. Identify vector attributes ending with AsNumpy
    as_numpy_fields = {
        m[:-7] for m in class_attrs if m.endswith("AsNumpy") and not m.startswith("_")
    }

    for attr_name in class_attrs:
        if (
            attr_name.startswith("_")
            or attr_name in _IGNORED_METHOD_NAMES
            or attr_name.endswith("BufferHasIdentifier")
            or attr_name.startswith("GetRootAs")
        ):
            continue

        # Skip length and is_none helper methods
        if attr_name.endswith("Length") or attr_name.endswith("IsNone"):
            continue

        method = getattr(table, attr_name, None)
        if not callable(method):
            continue

        # If this is an AsNumpy variant
        if attr_name.endswith("AsNumpy"):
            base_name = attr_name[:-7]
            snake_name = _pascal_to_snake(base_name)
            try:
                val = method()
                if hasattr(val, "tolist"):
                    attrs[snake_name] = val.tolist()
                elif isinstance(val, (int, float)) and val == 0:
                    attrs[snake_name] = []
                else:
                    attrs[snake_name] = val
            except Exception:
                attrs[snake_name] = []
            continue

        # If this field has an AsNumpy equivalent, skip the indexed/scalar getter
        if attr_name in as_numpy_fields:
            continue

        snake_name = _pascal_to_snake(attr_name)
        val: Any = None
        has_val = False
        with contextlib.suppress(Exception):
            try:
                val = method()
                has_val = True
            except TypeError:
                len_method = getattr(table, f"{attr_name}Length", None)
                if callable(len_method):
                    with contextlib.suppress(Exception):
                        count = len_method()
                        val = [method(i) for i in range(count)]
                        has_val = True

        if not has_val:
            continue

        # Normalize value
        if isinstance(val, bytes):
            try:
                attrs[snake_name] = val.decode("utf-8")
            except UnicodeDecodeError:
                attrs[snake_name] = val.decode("utf-8", errors="replace")
        elif hasattr(val, "tolist"):
            attrs[snake_name] = val.tolist()
        else:
            attrs[snake_name] = val

    return attrs


def __getattr__(name: str) -> Any:
    """Dynamically provide parse_<OptionClass> functions."""
    if name.startswith("parse_"):
        opt_class = name[6:]

        def _parser(options: Any) -> dict[str, Any]:
            return parse_tflite_builtin_options(opt_class, options)

        _parser.__name__ = name
        return _parser
    msg = f"module '{__name__}' has no attribute '{name}'"
    raise AttributeError(msg)
