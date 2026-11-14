"""Computer vision architecture blocks for tests and benchmarking."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def make_resnet_bottleneck_block(
    in_channels: int = 64,
    out_channels: int = 256,
    stride: int = 1,
) -> onnx.ModelProto:
    """Generate a realistic ResNet bottleneck residual block with skip connection."""
    mid_channels = out_channels // 4

    nodes = []
    initializers = []

    def _add_init(name: str, shape: list[int], positive: bool = False) -> str:
        if positive:
            arr = np.abs(np.random.randn(*shape).astype(np.float32)) + 0.5
        else:
            arr = np.random.randn(*shape).astype(np.float32) * 0.01
        initializers.append(numpy_helper.from_array(arr, name=name))
        return name

    # Input: X
    # 1. 1x1 Conv + BN + Relu
    w1 = _add_init("w1", [mid_channels, in_channels, 1, 1])
    nodes.append(helper.make_node("Conv", ["X", w1], ["conv1_out"], name="conv1"))
    s1 = _add_init("s1", [mid_channels])
    b1 = _add_init("b1", [mid_channels])
    m1 = _add_init("m1", [mid_channels])
    v1 = _add_init("v1", [mid_channels], positive=True)
    nodes.append(
        helper.make_node(
            "BatchNormalization",
            ["conv1_out", s1, b1, m1, v1],
            ["bn1_out"],
            name="bn1",
        )
    )
    nodes.append(helper.make_node("Relu", ["bn1_out"], ["relu1_out"], name="relu1"))

    # 2. 3x3 Conv + BN + Relu
    w2 = _add_init("w2", [mid_channels, mid_channels, 3, 3])
    nodes.append(
        helper.make_node(
            "Conv",
            ["relu1_out", w2],
            ["conv2_out"],
            name="conv2",
            pads=[1, 1, 1, 1],
            strides=[stride, stride],
        )
    )
    s2 = _add_init("s2", [mid_channels])
    b2 = _add_init("b2", [mid_channels])
    m2 = _add_init("m2", [mid_channels])
    v2 = _add_init("v2", [mid_channels], positive=True)
    nodes.append(
        helper.make_node(
            "BatchNormalization",
            ["conv2_out", s2, b2, m2, v2],
            ["bn2_out"],
            name="bn2",
        )
    )
    nodes.append(helper.make_node("Relu", ["bn2_out"], ["relu2_out"], name="relu2"))

    # 3. 1x1 Conv + BN
    w3 = _add_init("w3", [out_channels, mid_channels, 1, 1])
    nodes.append(helper.make_node("Conv", ["relu2_out", w3], ["conv3_out"], name="conv3"))
    s3 = _add_init("s3", [out_channels])
    b3 = _add_init("b3", [out_channels])
    m3 = _add_init("m3", [out_channels])
    v3 = _add_init("v3", [out_channels], positive=True)
    nodes.append(
        helper.make_node(
            "BatchNormalization",
            ["conv3_out", s3, b3, m3, v3],
            ["bn3_out"],
            name="bn3",
        )
    )

    # 4. Shortcut connection
    if in_channels == out_channels and stride == 1:
        shortcut_out = "X"
    else:
        ws = _add_init("ws", [out_channels, in_channels, 1, 1])
        nodes.append(
            helper.make_node(
                "Conv",
                ["X", ws],
                ["conv_s_out"],
                name="conv_shortcut",
                strides=[stride, stride],
            )
        )
        ss = _add_init("ss", [out_channels])
        bs = _add_init("bs", [out_channels])
        ms = _add_init("ms", [out_channels])
        vs = _add_init("vs", [out_channels], positive=True)
        nodes.append(
            helper.make_node(
                "BatchNormalization",
                ["conv_s_out", ss, bs, ms, vs],
                ["shortcut_out"],
                name="bn_shortcut",
            )
        )
        shortcut_out = "shortcut_out"

    # 5. Add + final ReLU
    nodes.append(helper.make_node("Add", ["bn3_out", shortcut_out], ["add_out"], name="res_add"))
    nodes.append(helper.make_node("Relu", ["add_out"], ["Y"], name="relu3"))

    inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, in_channels, 56, 56])]
    outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, out_channels, 56 // stride, 56 // stride])]

    graph = helper.make_graph(nodes, "ResNetBottleneck", inputs, outputs, initializer=initializers)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def make_mobilenet_block(
    in_channels: int = 32,
    out_channels: int = 64,
    expansion: int = 6,
    stride: int = 1,
) -> onnx.ModelProto:
    """Generate a realistic MobileNetV2 inverted residual block with depthwise conv and ReLU6."""
    expanded = in_channels * expansion

    nodes = []
    initializers = []

    def _add_init(name: str, shape: list[int], positive: bool = False) -> str:
        if positive:
            arr = np.abs(np.random.randn(*shape).astype(np.float32)) + 0.5
        else:
            arr = np.random.randn(*shape).astype(np.float32) * 0.01
        initializers.append(numpy_helper.from_array(arr, name=name))
        return name

    # Clip limits
    clip_min_arr = np.array([0.0], dtype=np.float32)
    clip_max_arr = np.array([6.0], dtype=np.float32)
    initializers.append(numpy_helper.from_array(clip_min_arr, name="clip_min"))
    initializers.append(numpy_helper.from_array(clip_max_arr, name="clip_max"))

    # 1. 1x1 Expansion Conv + BN + Clip(0, 6)
    w1 = _add_init("w_exp", [expanded, in_channels, 1, 1])
    nodes.append(helper.make_node("Conv", ["X", w1], ["exp_conv_out"], name="conv_exp"))
    nodes.append(
        helper.make_node(
            "BatchNormalization",
            ["exp_conv_out", _add_init("s_exp", [expanded]), _add_init("b_exp", [expanded]), _add_init("m_exp", [expanded]), _add_init("v_exp", [expanded], positive=True)],
            ["exp_bn_out"],
            name="bn_exp",
        )
    )
    nodes.append(helper.make_node("Clip", ["exp_bn_out", "clip_min", "clip_max"], ["exp_relu6_out"], name="relu6_exp"))

    # 2. 3x3 Depthwise Conv + BN + Clip(0, 6)
    w_dw = _add_init("w_dw", [expanded, 1, 3, 3])
    nodes.append(
        helper.make_node(
            "Conv",
            ["exp_relu6_out", w_dw],
            ["dw_conv_out"],
            name="conv_dw",
            group=expanded,
            pads=[1, 1, 1, 1],
            strides=[stride, stride],
        )
    )
    nodes.append(
        helper.make_node(
            "BatchNormalization",
            ["dw_conv_out", _add_init("s_dw", [expanded]), _add_init("b_dw", [expanded]), _add_init("m_dw", [expanded]), _add_init("v_dw", [expanded], positive=True)],
            ["dw_bn_out"],
            name="bn_dw",
        )
    )
    nodes.append(helper.make_node("Clip", ["dw_bn_out", "clip_min", "clip_max"], ["dw_relu6_out"], name="relu6_dw"))

    # 3. 1x1 Linear Projection + BN (no non-linearity)
    w_proj = _add_init("w_proj", [out_channels, expanded, 1, 1])
    nodes.append(helper.make_node("Conv", ["dw_relu6_out", w_proj], ["proj_conv_out"], name="conv_proj"))
    nodes.append(
        helper.make_node(
            "BatchNormalization",
            ["proj_conv_out", _add_init("s_proj", [out_channels]), _add_init("b_proj", [out_channels]), _add_init("m_proj", [out_channels]), _add_init("v_proj", [out_channels], positive=True)],
            ["proj_bn_out"],
            name="bn_proj",
        )
    )

    # 4. Residual Add if in_channels == out_channels
    if in_channels == out_channels:
        nodes.append(helper.make_node("Add", ["X", "proj_bn_out"], ["Y"], name="res_add"))
    else:
        nodes.append(helper.make_node("Identity", ["proj_bn_out"], ["Y"], name="identity_out"))

    inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, in_channels, 28, 28])]
    outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, out_channels, 28, 28])]

    graph = helper.make_graph(nodes, "MobileNetV2Block", inputs, outputs, initializer=initializers)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
