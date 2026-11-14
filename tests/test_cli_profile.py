"""Tests for CLI neuron profile and transformer pass optimization."""

import json
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from neuron_toolkit.cli import main


def _save_sample_model(path: Path) -> str:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 10])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 10])
    w = helper.make_tensor("W", TensorProto.FLOAT, [10, 10], [1.0] * 100)
    node = helper.make_node("MatMul", ["X", "W"], ["Y"])
    graph = helper.make_graph([node], "matmul_net", [x], [y], initializer=[w])
    model = helper.make_model(
        graph, producer_name="cli_test", opset_imports=[helper.make_opsetid("", 17)]
    )
    onnx.save(model, str(path))
    return str(path)


def test_cli_profile_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    model_file = tmp_path / "model.onnx"
    _save_sample_model(model_file)

    code = main(["profile", str(model_file), "--device", "a100", "--dtype", "float16"])
    assert code == 0
    captured = capsys.readouterr().out
    assert "NEURON TOOLKIT HARDWARE & MEMORY PROFILER" in captured
    assert "NVIDIA A100" in captured


def test_cli_profile_json(tmp_path: Path):
    model_file = tmp_path / "model.onnx"
    _save_sample_model(model_file)
    json_out = tmp_path / "profile.json"

    code = main(
        [
            "profile",
            str(model_file),
            "--device",
            "m3",
            "--format",
            "json",
            "-o",
            str(json_out),
        ]
    )
    assert code == 0
    assert json_out.exists()
    data = json.loads(json_out.read_text())
    assert "total_parameters" in data
    assert "ridge_point" in data


def test_cli_optimize_with_transformer_passes(tmp_path: Path):
    # Create RMSNorm model
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16])
    two = helper.make_tensor("two", TensorProto.FLOAT, [1], [2.0])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [-1])
    eps = helper.make_tensor("eps", TensorProto.FLOAT, [1], [1e-5])

    n1 = helper.make_node("Pow", ["X", "two"], ["sq"])
    n2 = helper.make_node("ReduceMean", ["sq", "axes"], ["mean"], keepdims=1)
    n3 = helper.make_node("Add", ["mean", "eps"], ["var_eps"])
    n4 = helper.make_node("Sqrt", ["var_eps"], ["rms"])
    n5 = helper.make_node("Div", ["X", "rms"], ["Y"])

    graph = helper.make_graph(
        [n1, n2, n3, n4, n5], "rmsnorm", [x], [y], initializer=[two, axes, eps]
    )
    model = helper.make_model(
        graph, producer_name="cli_test", opset_imports=[helper.make_opsetid("", 17)]
    )

    in_file = tmp_path / "rmsnorm.onnx"
    out_file = tmp_path / "opt.onnx"
    onnx.save(model, str(in_file))

    code = main(
        ["optimize", str(in_file), "--passes", "fuse_rmsnorm,dce", "-o", str(out_file)]
    )
    assert code == 0
    assert out_file.exists()

    opt_model = onnx.load(str(out_file))
    assert len(opt_model.graph.node) == 1
    assert opt_model.graph.node[0].op_type == "RMSNorm"
