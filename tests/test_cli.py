"""Unit tests for the neuron developer CLI."""

import os
from io import StringIO
from unittest.mock import patch

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from neuron_toolkit.cli import main


def create_cli_test_model(path: str) -> None:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])
    w = numpy_helper.from_array(np.array([[1.0, 2.0]], dtype=np.float32), name="W")
    node = helper.make_node("Relu", ["X"], ["Y"], name="test_relu")
    graph = helper.make_graph([node], "cli_model", [x], [y], [w])
    model = helper.make_model(graph, producer_name="cli_test")
    onnx.save(model, path)


def create_cli_second_model(path: str) -> None:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])
    z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 2])
    n1 = helper.make_node("Sigmoid", ["X"], ["H"], name="test_sig")
    n2 = helper.make_node("Tanh", ["H"], ["Z"], name="test_tanh")
    graph = helper.make_graph([n1, n2], "cli_model_2", [x], [z])
    model = helper.make_model(graph, producer_name="cli_test_2")
    onnx.save(model, path)


def test_cli_inspect(tmp_path):
    p = str(tmp_path / "model.onnx")
    create_cli_test_model(p)

    with patch("sys.stdout", new_callable=StringIO) as mock_out:
        ret = main(["inspect", p])
        assert ret == 0
        out = mock_out.getvalue()
        assert "Total Nodes: 1" in out
        assert "Relu" in out
        assert "Initializers/Weights: 1 tensors" in out


def test_cli_export_json_and_html(tmp_path):
    p = str(tmp_path / "model.onnx")
    create_cli_test_model(p)

    json_out = str(tmp_path / "model.json")
    ret_json = main(["export", p, "--format", "json", "-o", json_out])
    assert ret_json == 0
    assert os.path.exists(json_out)

    # Export json to stdout
    with patch("sys.stdout", new_callable=StringIO) as mock_out:
        ret_json_stdout = main(["export", p, "--format", "json"])
        assert ret_json_stdout == 0
        assert "cli_model" in mock_out.getvalue()

    html_out = str(tmp_path / "model.html")
    ret_html = main(["export", p, "--format", "html", "-o", html_out])
    assert ret_html == 0
    assert os.path.exists(html_out)


def test_cli_export_mermaid(tmp_path):
    p = str(tmp_path / "model.onnx")
    create_cli_test_model(p)

    mermaid_out = str(tmp_path / "model.mmd")
    ret = main(["export", p, "--format", "mermaid", "-o", mermaid_out])
    assert ret == 0
    assert os.path.exists(mermaid_out)
    content = (tmp_path / "model.mmd").read_text(encoding="utf-8")
    assert "flowchart TD" in content

    # Export to stdout
    with patch("sys.stdout", new_callable=StringIO) as mock_out:
        ret_stdout = main(["export", p, "--format", "mermaid"])
        assert ret_stdout == 0
        assert "flowchart TD" in mock_out.getvalue()


def test_cli_export_safetensors(tmp_path):
    p = str(tmp_path / "model.onnx")
    create_cli_test_model(p)

    sf_out = str(tmp_path / "weights.safetensors")
    ret = main(["export", p, "--format", "safetensors", "-o", sf_out])
    assert ret == 0
    assert os.path.exists(sf_out)
    assert os.path.getsize(sf_out) > 0


def test_cli_export_invalid_format(tmp_path):
    p = str(tmp_path / "model.onnx")
    create_cli_test_model(p)

    with pytest.raises(SystemExit):
        main(["export", p, "--format", "invalid_fmt"])


def test_cli_diff(tmp_path):
    p1 = str(tmp_path / "model1.onnx")
    p2 = str(tmp_path / "model2.onnx")
    create_cli_test_model(p1)
    create_cli_second_model(p2)

    with patch("sys.stdout", new_callable=StringIO) as mock_out:
        ret = main(["diff", p1, p2])
        assert ret == 0
        out = mock_out.getvalue()
        assert "Node count: 1 -> 2 (+1)" in out
        assert "Relu" in out
        assert "Sigmoid" in out


def test_cli_optimize(tmp_path):
    p = str(tmp_path / "model.onnx")
    create_cli_test_model(p)
    out_p = str(tmp_path / "model_opt.onnx")

    with patch("sys.stdout", new_callable=StringIO) as mock_out:
        ret = main(["optimize", p, "--passes", "dce,const_fold,cse", "-o", out_p])
        assert ret == 0
        out = mock_out.getvalue()
        assert "PassManager Execution Summary" in out
        assert os.path.exists(out_p)


def test_cli_optimize_unknown_pass(tmp_path):
    p = str(tmp_path / "model.onnx")
    create_cli_test_model(p)

    with patch("sys.stdout", new_callable=StringIO) as mock_out:
        ret = main(["optimize", p, "--passes", "nonexistent_pass_xyz"])
        assert ret == 0
        assert "Warning: Unknown pass 'nonexistent_pass_xyz', skipping." in mock_out.getvalue()
