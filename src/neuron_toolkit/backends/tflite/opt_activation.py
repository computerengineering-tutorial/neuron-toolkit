"""TFLite options parser for activation operations."""
# ruff: noqa: PLC0415, N802

from __future__ import annotations

from typing import Any


def parse_GeluOptions(_options: Any) -> dict[str, Any]:
    """Parse GeluOptions attributes."""
    if _options is None or not hasattr(_options, "Bytes") or not hasattr(_options, "Pos"):
        return {}

    from tflite.GeluOptions import GeluOptions

    opt = GeluOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["approximate"] = opt.Approximate()

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


def parse_LeakyReluOptions(_options: Any) -> dict[str, Any]:
    """Parse LeakyReluOptions attributes."""
    if _options is None or not hasattr(_options, "Bytes") or not hasattr(_options, "Pos"):
        return {}

    from tflite.LeakyReluOptions import LeakyReluOptions

    opt = LeakyReluOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["alpha"] = opt.Alpha()

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


def parse_SoftmaxOptions(_options: Any) -> dict[str, Any]:
    """Parse SoftmaxOptions attributes."""
    if _options is None or not hasattr(_options, "Bytes") or not hasattr(_options, "Pos"):
        return {}

    from tflite.SoftmaxOptions import SoftmaxOptions

    opt = SoftmaxOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["beta"] = opt.Beta()

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
