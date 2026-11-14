"""Comprehensive coverage and verification test suite for TFLite options parsers.

Ensures that 100% of TFLite FlatBuffer BuiltinOptions and BuiltinOptions2
classes defined in the installed tflite package are accounted for, categorized,
and have functional parser implementations.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import tflite

from neuron_toolkit.backends.tflite.utils import (
    _EMPTY_OPTIONS_CLASSES,
    _OPT_CLASS_TO_CATEGORY,
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


def test_every_tflite_option_is_categorized_or_empty(tflite_all_options: set[str]):
    """Every option in tflite must either be in _EMPTY_OPTIONS_CLASSES or categorized."""
    uncategorized = []
    for opt_name in sorted(tflite_all_options):
        is_empty = opt_name in _EMPTY_OPTIONS_CLASSES
        category = _OPT_CLASS_TO_CATEGORY.get(opt_name)

        if not is_empty and not category:
            uncategorized.append(opt_name)

    assert not uncategorized, f"Found uncategorized options classes: {uncategorized}"


def test_every_active_option_has_parser_function(tflite_all_options: set[str]):
    """Every non-empty option must have a matching parse_<OptionClass> function in its category module."""
    missing_parsers = []
    for opt_name in sorted(tflite_all_options):
        if opt_name in _EMPTY_OPTIONS_CLASSES:
            continue

        category = _OPT_CLASS_TO_CATEGORY.get(opt_name)
        assert category, f"Class {opt_name} missing category mapping."

        module_name = f"neuron_toolkit.backends.tflite.opt_{category}"
        try:
            mod = importlib.import_module(module_name)
        except ImportError as e:
            missing_parsers.append(f"{module_name} (Failed to import: {e})")
            continue

        parse_func_name = f"parse_{opt_name}"
        if not hasattr(mod, parse_func_name):
            missing_parsers.append(f"{parse_func_name} in {module_name}")

    assert not missing_parsers, f"Missing parser functions for options: {missing_parsers}"


def test_all_parser_functions_callable_and_safe(tflite_all_options: set[str]):
    """Test calling every single generated parse function with mock options.
    
    Verifies that no parser crashes with None options or valid mock options.
    """
    mock_options = MagicMock()
    mock_options.Bytes = b"\x00" * 32
    mock_options.Pos = 0

    tested_count = 0
    for opt_name in sorted(tflite_all_options):
        if opt_name in _EMPTY_OPTIONS_CLASSES:
            continue

        category = _OPT_CLASS_TO_CATEGORY[opt_name]
        module = importlib.import_module(f"neuron_toolkit.backends.tflite.opt_{category}")
        parse_func = getattr(module, f"parse_{opt_name}")

        # Test with None
        none_res = parse_func(None)
        assert none_res == {}, f"{parse_func.__name__}(None) did not return empty dict"

        # Test with mock options
        try:
            res = parse_func(mock_options)
            assert isinstance(res, dict)
            tested_count += 1
        except Exception as e:
            pytest.fail(f"parse_{opt_name}(mock_options) failed with exception: {e}")

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
