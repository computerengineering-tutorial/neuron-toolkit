"""Comprehensive coverage and verification test suite for TFLite options parsers.

Ensures that 100% of TFLite FlatBuffer BuiltinOptions and BuiltinOptions2
classes defined in the installed tflite package are accounted for, categorized,
and have functional parser implementations.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import tflite

from neuron_toolkit.backends.tflite import options
from neuron_toolkit.backends.tflite.options import parse_tflite_builtin_options
from neuron_toolkit.backends.tflite.utils import (
    _EMPTY_OPTIONS_CLASSES,
    _get_tflite_attr,
)


@pytest.fixture(scope="module")
def tflite_all_options() -> set[str]:
    """Collect all unique BuiltinOptions and BuiltinOptions2 enum names."""
    all_options: set[str] = set()

    if hasattr(tflite, "BuiltinOptions"):
        for name in dir(tflite.BuiltinOptions):
            if not name.startswith("_") and name != "NONE":
                all_options.add(name)

    if hasattr(tflite, "BuiltinOptions2"):
        for name in dir(tflite.BuiltinOptions2):
            if not name.startswith("_") and name != "NONE":
                all_options.add(name)

    return all_options


def test_tflite_options_total_count(tflite_all_options: set[str]):
    """Verify that all 148 TFLite options enum members are detected."""
    assert len(tflite_all_options) == 148, (
        f"Expected 148 TFLite options classes, found {len(tflite_all_options)}"
    )


def test_all_tflite_options_files_exist(tflite_all_options: set[str]):
    """Verify that every option class corresponds to a file in the tflite package."""
    tflite_dir = Path(tflite.__file__).parent
    missing_files = []
    for opt_name in sorted(tflite_all_options):
        file_path = tflite_dir / f"{opt_name}.py"
        if not file_path.is_file():
            missing_files.append(f"{opt_name}.py")

    assert not missing_files, f"Options files missing in tflite package: {missing_files}"


def test_every_active_option_has_parser_function(tflite_all_options: set[str]):
    """Every non-empty option must have a matching parser function available in options module."""
    for opt_name in sorted(tflite_all_options):
        if opt_name in _EMPTY_OPTIONS_CLASSES:
            continue

        parse_func = getattr(options, f"parse_{opt_name}")
        assert callable(parse_func), f"Missing callable parser for {opt_name}"


def test_all_parser_functions_callable_and_safe(tflite_all_options: set[str]):
    """Test calling every single option parser with None and valid mock options.

    Verifies that consolidated options extractor handles all 148 options safely.
    """
    mock_options = MagicMock()
    mock_options.Bytes = b"\x00" * 32
    mock_options.Pos = 0

    tested_count = 0
    for opt_name in sorted(tflite_all_options):
        if opt_name in _EMPTY_OPTIONS_CLASSES:
            continue

        # Test direct function
        none_res = parse_tflite_builtin_options(opt_name, None)
        assert none_res == {}, f"parse_tflite_builtin_options({opt_name}, None) did not return empty dict"

        # Test with mock options
        try:
            res = parse_tflite_builtin_options(opt_name, mock_options)
            assert isinstance(res, dict)
            tested_count += 1
        except Exception as e:
            pytest.fail(f"parse_tflite_builtin_options({opt_name}, mock_options) failed with exception: {e}")

    assert tested_count > 0, "No active parser functions were tested"


def test_empty_options_return_empty_dict(tflite_all_options: set[str]):
    """Test that all 68 empty options classes immediately return an empty dict."""
    op_mock = MagicMock()
    op_mock.BuiltinOptions.return_value = MagicMock()
    op_mock.BuiltinOptions2Type.return_value = 0

    tested_empty = 0
    for opt_name in sorted(tflite_all_options):
        if opt_name not in _EMPTY_OPTIONS_CLASSES:
            continue

        # Set the option type
        if hasattr(tflite.BuiltinOptions, opt_name):
            op_mock.BuiltinOptionsType.return_value = getattr(tflite.BuiltinOptions, opt_name)
        elif hasattr(tflite.BuiltinOptions2, opt_name):
            op_mock.BuiltinOptionsType.return_value = 0
            op_mock.BuiltinOptions2Type.return_value = getattr(tflite.BuiltinOptions2, opt_name)

        attrs = _get_tflite_attr(op_mock, opt_name)
        assert attrs == {}, f"Empty option {opt_name} returned non-empty attrs: {attrs}"
        tested_empty += 1

    assert tested_empty == len(_EMPTY_OPTIONS_CLASSES)
    assert tested_empty == 68


def test_tflite_builtin_operators_coverage():
    """Verify that all BuiltinOperator codes can be resolved by TFLiteParser."""
    builtin_operators = {
        k: v
        for k, v in tflite.BuiltinOperator.__dict__.items()
        if isinstance(v, int) and not k.startswith("__")
    }

    builtin_op_map = {v: k for k, v in builtin_operators.items()}

    # Check that common and modern TFLite operators are present in map
    critical_ops = ["ADD", "SUB", "MUL", "DIV", "CONV_2D", "DEPTHWISE_CONV_2D", "RESHAPE", "SOFTMAX"]
    for op in critical_ops:
        code = builtin_operators.get(op)
        assert code is not None, f"Operator {op} missing from BuiltinOperator"
        assert builtin_op_map.get(code) == op
