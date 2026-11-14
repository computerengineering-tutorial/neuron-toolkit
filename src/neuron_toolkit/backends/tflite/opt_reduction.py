"""TFLite options parser for reduction operations."""
# ruff: noqa: PLC0415, N802

from __future__ import annotations

from typing import Any


def parse_ArgMaxOptions(_options: Any) -> dict[str, Any]:
    """Parse ArgMaxOptions attributes."""
    if _options is None or not hasattr(_options, "Bytes") or not hasattr(_options, "Pos"):
        return {}

    from tflite.ArgMaxOptions import ArgMaxOptions

    opt = ArgMaxOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["output_type"] = opt.OutputType()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            try:
                attrs[k] = v.decode("utf-8")
            except UnicodeDecodeError:
                attrs[k] = v.decode("utf-8", errors="replace")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_ArgMinOptions(_options: Any) -> dict[str, Any]:
    """Parse ArgMinOptions attributes."""
    if _options is None or not hasattr(_options, "Bytes") or not hasattr(_options, "Pos"):
        return {}

    from tflite.ArgMinOptions import ArgMinOptions

    opt = ArgMinOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["output_type"] = opt.OutputType()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            try:
                attrs[k] = v.decode("utf-8")
            except UnicodeDecodeError:
                attrs[k] = v.decode("utf-8", errors="replace")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs


def parse_ReducerOptions(_options: Any) -> dict[str, Any]:
    """Parse ReducerOptions attributes."""
    if _options is None or not hasattr(_options, "Bytes") or not hasattr(_options, "Pos"):
        return {}

    from tflite.ReducerOptions import ReducerOptions

    opt = ReducerOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["keep_dims"] = opt.KeepDims()

    # Normalize output types (e.g. bytes to str, numpy array to list)
    for k, v in list(attrs.items()):
        if isinstance(v, bytes):
            try:
                attrs[k] = v.decode("utf-8")
            except UnicodeDecodeError:
                attrs[k] = v.decode("utf-8", errors="replace")
        elif hasattr(v, "tolist"):
            attrs[k] = v.tolist()
    return attrs
