"""TFLite options parser for tensor operations."""
# ruff: noqa: PLC0415, N802

from __future__ import annotations

from typing import Any


def parse_CastOptions(_options: Any) -> dict[str, Any]:
    """Parse CastOptions attributes."""
    if _options is None or not hasattr(_options, "Bytes") or not hasattr(_options, "Pos"):
        return {}

    from tflite.CastOptions import CastOptions

    opt = CastOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["in_data_type"] = opt.InDataType()
    attrs["out_data_type"] = opt.OutDataType()

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


def parse_ConcatenationOptions(_options: Any) -> dict[str, Any]:
    """Parse ConcatenationOptions attributes."""
    if _options is None or not hasattr(_options, "Bytes") or not hasattr(_options, "Pos"):
        return {}

    from tflite.ConcatenationOptions import ConcatenationOptions

    opt = ConcatenationOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["axis"] = opt.Axis()
    attrs["fused_activation_function"] = opt.FusedActivationFunction()

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


def parse_GatherOptions(_options: Any) -> dict[str, Any]:
    """Parse GatherOptions attributes."""
    if _options is None or not hasattr(_options, "Bytes") or not hasattr(_options, "Pos"):
        return {}

    from tflite.GatherOptions import GatherOptions

    opt = GatherOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["axis"] = opt.Axis()
    attrs["batch_dims"] = opt.BatchDims()

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


def parse_OneHotOptions(_options: Any) -> dict[str, Any]:
    """Parse OneHotOptions attributes."""
    if _options is None or not hasattr(_options, "Bytes") or not hasattr(_options, "Pos"):
        return {}

    from tflite.OneHotOptions import OneHotOptions

    opt = OneHotOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["axis"] = opt.Axis()

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


def parse_PackOptions(_options: Any) -> dict[str, Any]:
    """Parse PackOptions attributes."""
    if _options is None or not hasattr(_options, "Bytes") or not hasattr(_options, "Pos"):
        return {}

    from tflite.PackOptions import PackOptions

    opt = PackOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["axis"] = opt.Axis()
    attrs["values_count"] = opt.ValuesCount()

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


def parse_ReshapeOptions(_options: Any) -> dict[str, Any]:
    """Parse ReshapeOptions attributes."""
    if _options is None or not hasattr(_options, "Bytes") or not hasattr(_options, "Pos"):
        return {}

    from tflite.ReshapeOptions import ReshapeOptions

    opt = ReshapeOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["new_shape"] = opt.NewShapeAsNumpy()

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


def parse_ReverseSequenceOptions(_options: Any) -> dict[str, Any]:
    """Parse ReverseSequenceOptions attributes."""
    if _options is None or not hasattr(_options, "Bytes") or not hasattr(_options, "Pos"):
        return {}

    from tflite.ReverseSequenceOptions import ReverseSequenceOptions

    opt = ReverseSequenceOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["batch_dim"] = opt.BatchDim()
    attrs["seq_dim"] = opt.SeqDim()

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


def parse_ShapeOptions(_options: Any) -> dict[str, Any]:
    """Parse ShapeOptions attributes."""
    if _options is None or not hasattr(_options, "Bytes") or not hasattr(_options, "Pos"):
        return {}

    from tflite.ShapeOptions import ShapeOptions

    opt = ShapeOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["out_type"] = opt.OutType()

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


def parse_SplitOptions(_options: Any) -> dict[str, Any]:
    """Parse SplitOptions attributes."""
    if _options is None or not hasattr(_options, "Bytes") or not hasattr(_options, "Pos"):
        return {}

    from tflite.SplitOptions import SplitOptions

    opt = SplitOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["num_splits"] = opt.NumSplits()

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


def parse_SplitVOptions(_options: Any) -> dict[str, Any]:
    """Parse SplitVOptions attributes."""
    if _options is None or not hasattr(_options, "Bytes") or not hasattr(_options, "Pos"):
        return {}

    from tflite.SplitVOptions import SplitVOptions

    opt = SplitVOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["num_splits"] = opt.NumSplits()

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


def parse_SqueezeOptions(_options: Any) -> dict[str, Any]:
    """Parse SqueezeOptions attributes."""
    if _options is None or not hasattr(_options, "Bytes") or not hasattr(_options, "Pos"):
        return {}

    from tflite.SqueezeOptions import SqueezeOptions

    opt = SqueezeOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["squeeze_dims"] = opt.SqueezeDimsAsNumpy()

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


def parse_StridedSliceOptions(_options: Any) -> dict[str, Any]:
    """Parse StridedSliceOptions attributes."""
    if _options is None or not hasattr(_options, "Bytes") or not hasattr(_options, "Pos"):
        return {}

    from tflite.StridedSliceOptions import StridedSliceOptions

    opt = StridedSliceOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["begin_mask"] = opt.BeginMask()
    attrs["ellipsis_mask"] = opt.EllipsisMask()
    attrs["end_mask"] = opt.EndMask()
    attrs["new_axis_mask"] = opt.NewAxisMask()
    attrs["offset"] = opt.Offset()
    attrs["shrink_axis_mask"] = opt.ShrinkAxisMask()

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


def parse_UniqueOptions(_options: Any) -> dict[str, Any]:
    """Parse UniqueOptions attributes."""
    if _options is None or not hasattr(_options, "Bytes") or not hasattr(_options, "Pos"):
        return {}

    from tflite.UniqueOptions import UniqueOptions

    opt = UniqueOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["idx_out_type"] = opt.IdxOutType()

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


def parse_UnpackOptions(_options: Any) -> dict[str, Any]:
    """Parse UnpackOptions attributes."""
    if _options is None or not hasattr(_options, "Bytes") or not hasattr(_options, "Pos"):
        return {}

    from tflite.UnpackOptions import UnpackOptions

    opt = UnpackOptions()
    opt.Init(_options.Bytes, _options.Pos)

    attrs: dict[str, Any] = {}
    attrs["axis"] = opt.Axis()
    attrs["num"] = opt.Num()

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
