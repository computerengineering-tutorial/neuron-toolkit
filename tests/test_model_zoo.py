"""Tests exercising NeuronGraph query, pattern matching, passes, and slicing on Zoo architectures."""

from __future__ import annotations

import sys
from pathlib import Path

tests_dir = Path(__file__).parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

from zoo.fixtures_cv import make_mobilenet_block, make_resnet_bottleneck_block
from zoo.fixtures_nlp import make_mha_block, make_transformer_layer

from neuron_toolkit.graph import NeuronGraph
from neuron_toolkit.passes import FuseConvBN, PassManager


def test_resnet_bottleneck_query_and_fusion():
    """Verify ResNet bottleneck block querying and FuseConvBN optimization."""
    # Bottleneck with downsample projection has 4 Convs and 4 BNs
    model = make_resnet_bottleneck_block(in_channels=64, out_channels=256, stride=2)
    g = NeuronGraph(model)

    assert len(g.query().op("Conv").all()) == 4
    assert len(g.query().op("BatchNormalization").all()) == 4
    assert len(g.query().op("Relu").all()) == 3
    assert len(g.query().op("Add").all()) == 1

    # Fuse Conv + BN
    opt_g, stats = g.optimize(PassManager([FuseConvBN()]))
    assert stats["FuseConvBN"]["fused_pairs"] == 4
    assert len(opt_g.query().op("BatchNormalization").all()) == 0
    assert len(opt_g.query().op("Conv").all()) == 4


def test_mobilenet_depthwise_query_and_slice():
    """Verify MobileNet block inverted residual, depthwise conv querying and slicing."""
    model = make_mobilenet_block(in_channels=32, out_channels=64, stride=1, expansion=6)
    g = NeuronGraph(model)

    # 3 Convs: expand (1x1), depthwise (3x3), project (1x1)
    convs = g.query().op("Conv").all()
    assert len(convs) == 3

    # Check depthwise conv has group == 192 (expansion * 32)
    dw_conv = g.query().op("Conv").attr("group", 192).all()
    assert len(dw_conv) == 1
    assert dw_conv[0].name == "conv_dw"

    # Slice the depthwise conv stage
    sliced = g.slice(start_points=["exp_relu6_out"], end_points=["dw_relu6_out"])
    assert len(sliced.nodes) >= 1
    assert any(n.op_type == "Conv" for n in sliced.nodes)


def test_transformer_mha_query_and_patterns():
    """Verify Multi-Head Attention block querying and pattern matching."""
    model = make_mha_block(d_model=64, num_heads=4)
    g = NeuronGraph(model)

    matmuls = g.query().op("MatMul").all()
    assert len(matmuls) == 6  # Q, K, V projections + QK + Attn*V + Out projection

    softmax_nodes = g.query().op("Softmax").all()
    assert len(softmax_nodes) == 1

    transposes = g.query().op("Transpose").all()
    assert len(transposes) == 4


def test_transformer_layer_roundtrip_and_residuals():
    """Verify complete Transformer encoder layer JSON round-trip and residual connections."""
    model = make_transformer_layer(d_model=32, num_heads=2)
    g = NeuronGraph(model)

    # Has 2 residual Adds and 2 LayerNorms
    adds = g.query().op("Add").all()
    assert len(adds) >= 2
    lns = g.query().op("LayerNormalization").all()
    assert len(lns) == 2

    # Round trip to JSON and back
    json_str = g.to_json()
    rebuilt_g = NeuronGraph.from_json(json_str, seed=42)
    assert len(rebuilt_g.nodes) == len(g.nodes)
    assert len(rebuilt_g.query().op("LayerNormalization").all()) == 2
