#!/usr/bin/env python3
"""Comprehensive Convolutional & Computer Vision Architecture Family Generator."""

from __future__ import annotations

import torch
from torch import nn

from generators.decorators import register_architecture
from generators.registry import ModelRegistry


@register_architecture(
    name="conv2d_bn_relu",
    category="cnn_vision",
    subfolder="cnn_vision",
    shapes=[(1, 16, 16, 16)],
    input_names=["x"],
    output_names=["out"],
    init_args=(16, 32),
)
class StandardConv2D_BN_ReLU(nn.Module):
    def __init__(self, in_c: int = 16, out_c: int = 32) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


@register_architecture(
    name="resnet_basic",
    category="cnn_vision",
    subfolder="cnn_vision",
    shapes=[(1, 32, 16, 16)],
    input_names=["x"],
    output_names=["out"],
    init_args=(32,),
)
class ResNetBasicBlock(nn.Module):
    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.relu(out)


@register_architecture(
    name="resnet_bottleneck",
    category="cnn_vision",
    subfolder="cnn_vision",
    shapes=[(1, 64, 16, 16)],
    input_names=["x"],
    output_names=["out"],
    init_args=(64, 16, 64),
)
class ResNetBottleneckBlock(nn.Module):
    def __init__(self, in_c: int = 64, mid_c: int = 16, out_c: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, mid_c, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.conv2 = nn.Conv2d(mid_c, mid_c, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_c)
        self.conv3 = nn.Conv2d(mid_c, out_c, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + residual
        return self.relu(out)


@register_architecture(
    name="mobilenetv2_invres",
    category="cnn_vision",
    subfolder="cnn_vision",
    shapes=[(1, 32, 16, 16)],
    input_names=["x"],
    output_names=["out"],
    init_args=(32, 64, 32),
)
class MobileNetV2_InvertedResidual(nn.Module):
    def __init__(self, in_c: int = 32, expand_c: int = 64, out_c: int = 32) -> None:
        super().__init__()
        self.expand = nn.Conv2d(in_c, expand_c, kernel_size=1, bias=False)
        self.bn_exp = nn.BatchNorm2d(expand_c)
        self.depthwise = nn.Conv2d(
            expand_c, expand_c, kernel_size=3, padding=1, groups=expand_c, bias=False
        )
        self.bn_dw = nn.BatchNorm2d(expand_c)
        self.project = nn.Conv2d(expand_c, out_c, kernel_size=1, bias=False)
        self.bn_proj = nn.BatchNorm2d(out_c)
        self.relu6 = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu6(self.bn_exp(self.expand(x)))
        out = self.relu6(self.bn_dw(self.depthwise(out)))
        out = self.bn_proj(self.project(out))
        return out + residual


@register_architecture(
    name="mobilenetv3_block",
    category="cnn_vision",
    subfolder="cnn_vision",
    shapes=[(1, 32, 16, 16)],
    input_names=["x"],
    output_names=["out"],
    init_args=(32, 64, 32),
)
class MobileNetV3_Block(nn.Module):
    def __init__(
        self, in_c: int = 32, expand_c: int = 64, out_c: int = 32, se_reduce: int = 8
    ) -> None:
        super().__init__()
        self.expand = nn.Conv2d(in_c, expand_c, kernel_size=1, bias=False)
        self.bn_exp = nn.BatchNorm2d(expand_c)
        self.depthwise = nn.Conv2d(
            expand_c, expand_c, kernel_size=3, padding=1, groups=expand_c, bias=False
        )
        self.bn_dw = nn.BatchNorm2d(expand_c)

        self.se_pool = nn.AdaptiveAvgPool2d(1)
        self.se_fc1 = nn.Conv2d(expand_c, se_reduce, kernel_size=1)
        self.se_relu = nn.ReLU(inplace=True)
        self.se_fc2 = nn.Conv2d(se_reduce, expand_c, kernel_size=1)
        self.se_hsigmoid = nn.Hardsigmoid(inplace=True)

        self.project = nn.Conv2d(expand_c, out_c, kernel_size=1, bias=False)
        self.bn_proj = nn.BatchNorm2d(out_c)
        self.hardswish = nn.Hardswish(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.hardswish(self.bn_exp(self.expand(x)))
        out = self.hardswish(self.bn_dw(self.depthwise(out)))

        se = self.se_pool(out)
        se = self.se_relu(self.se_fc1(se))
        se = self.se_hsigmoid(self.se_fc2(se))
        out = out * se

        out = self.bn_proj(self.project(out))
        return out + residual


@register_architecture(
    name="convnext_block",
    category="cnn_vision",
    subfolder="cnn_vision",
    shapes=[(1, 32, 16, 16)],
    input_names=["x"],
    output_names=["out"],
    init_args=(32,),
)
class ConvNeXt_Block(nn.Module):
    def __init__(self, dim: int = 32, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, mlp_ratio * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(mlp_ratio * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.dwconv(x)
        out = out.permute(0, 2, 3, 1)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)
        out = out.permute(0, 3, 1, 2)
        return residual + out


@register_architecture(
    name="densenet_block",
    category="cnn_vision",
    subfolder="cnn_vision",
    shapes=[(1, 16, 16, 16)],
    input_names=["x"],
    output_names=["out"],
    init_args=(16, 16),
)
class DenseNet_Block(nn.Module):
    def __init__(self, in_c: int = 16, growth_rate: int = 16) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv1 = nn.Conv2d(in_c, growth_rate, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.relu(self.bn1(x)))
        return torch.cat([x, out], dim=1)


@register_architecture(
    name="resnext_block",
    category="cnn_vision",
    subfolder="cnn_vision",
    shapes=[(1, 64, 16, 16)],
    input_names=["x"],
    output_names=["out"],
    init_args=(64, 8, 8),
)
class ResNeXt_GroupedConvBlock(nn.Module):
    def __init__(self, in_c: int = 64, cardinality: int = 8, group_w: int = 8) -> None:
        super().__init__()
        hidden_dim = cardinality * group_w
        self.conv1 = nn.Conv2d(in_c, hidden_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=cardinality,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, in_c, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + residual)


@register_architecture(
    name="tcn_dilated",
    category="cnn_vision",
    subfolder="cnn_vision",
    shapes=[(1, 32, 32)],
    input_names=["x"],
    output_names=["out"],
    init_args=(32, 32),
    init_kwargs={"dilation": 2},
)
class TCN_DilatedConvBlock(nn.Module):
    def __init__(self, in_c: int = 32, out_c: int = 32, dilation: int = 2) -> None:
        super().__init__()
        self.dilation = dilation
        self.pad = (3 - 1) * dilation
        self.conv = nn.Conv1d(
            in_c, out_c, kernel_size=3, padding=self.pad, dilation=dilation
        )
        self.norm = nn.BatchNorm1d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S = x.shape[-1]
        out = self.conv(x)[:, :, :S]
        out = self.norm(out)
        return self.relu(out + x)


@register_architecture(
    name="wavenet_gated",
    category="cnn_vision",
    subfolder="cnn_vision",
    shapes=[(1, 32, 32)],
    input_names=["x"],
    output_names=["res", "skip"],
    init_args=(32,),
    init_kwargs={"dilation": 2},
)
class WaveNet_GatedDilatedConv(nn.Module):
    def __init__(self, in_c: int = 32, dilation: int = 2) -> None:
        super().__init__()
        self.pad = (2 - 1) * dilation
        self.conv = nn.Conv1d(
            in_c, 2 * in_c, kernel_size=2, padding=self.pad, dilation=dilation
        )
        self.res_proj = nn.Conv1d(in_c, in_c, kernel_size=1)
        self.skip_proj = nn.Conv1d(in_c, in_c, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        S = x.shape[-1]
        conv_out = self.conv(x)[:, :, :S]
        tanh_val, sig_val = torch.chunk(conv_out, 2, dim=1)
        gated = torch.tanh(tanh_val) * torch.sigmoid(sig_val)
        res_out = self.res_proj(gated) + x
        skip_out = self.skip_proj(gated)
        return res_out, skip_out


@register_architecture(
    name="conv3d_spatiotemporal",
    category="cnn_vision",
    subfolder="cnn_vision",
    shapes=[(1, 8, 4, 16, 16)],
    input_names=["x"],
    output_names=["out"],
    init_args=(8, 16),
)
class Conv3D_SpatioTemporalBlock(nn.Module):
    def __init__(self, in_c: int = 8, out_c: int = 16) -> None:
        super().__init__()
        self.conv3d = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn3d = nn.BatchNorm3d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn3d(self.conv3d(x)))


def main() -> None:
    ModelRegistry.export_all(category="cnn_vision")


if __name__ == "__main__":
    main()
