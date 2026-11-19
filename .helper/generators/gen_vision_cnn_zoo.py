#!/usr/bin/env python3
"""Comprehensive Convolutional & Computer Vision (CNN) Architecture Family Generator.

Generates self-contained PyTorch models and multi-opset ONNX exports for:
1. StandardConv2D_BN_ReLU (Classic Conv-BN-ReLU pipeline)
2. ResNetBasicBlock (ResNet-18/34 style 3x3 + 3x3 with identity shortcut)
3. ResNetBottleneckBlock (ResNet-50/101 style 1x1 -> 3x3 -> 1x1 expansion)
4. MobileNetV2_InvertedResidual (1x1 expand -> 3x3 depthwise separable -> 1x1 linear projection)
5. MobileNetV3_Block (Inverted residual + Squeeze-and-Excitation + Hardswish)
6. ConvNeXt_Block (7x7 depthwise -> LayerNorm (channels-last) -> 1x1 -> GELU -> 1x1 projection)
7. DenseNet_Block (Dense concatenation block [x, conv(x)])
8. ResNeXt_GroupedConvBlock (Grouped 3x3 convolution with cardinality groups)
9. TCN_DilatedConvBlock (1D Temporal Convolution with dilation + causal padding)
10. WaveNet_GatedDilatedConv (1D Dilated Conv with split Tanh * Sigmoid gating)
11. Conv3D_SpatioTemporalBlock (3D Spatio-Temporal video conv with BatchNorm3d)

Exports across ONNX Opsets 14, 17, 18, 20.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import onnx
import torch
from torch import nn

warnings.filterwarnings("ignore")

OUT_DIR = Path(".helper/models/cnn_vision")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. Standard Conv2D + BN + ReLU
# =============================================================================
class StandardConv2D_BN_ReLU(nn.Module):
    def __init__(self, in_c: int = 16, out_c: int = 32) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


# =============================================================================
# 2. ResNet BasicBlock
# =============================================================================
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


# =============================================================================
# 3. ResNet Bottleneck Block
# =============================================================================
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


# =============================================================================
# 4. MobileNetV2 Inverted Residual (Depthwise Separable)
# =============================================================================
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


# =============================================================================
# 5. MobileNetV3 Block with Squeeze-and-Excitation & Hardswish
# =============================================================================
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

        # Squeeze-and-Excitation
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

        # SE branch
        se = self.se_pool(out)
        se = self.se_relu(self.se_fc1(se))
        se = self.se_hsigmoid(self.se_fc2(se))
        out = out * se

        out = self.bn_proj(self.project(out))
        return out + residual


# =============================================================================
# 6. ConvNeXt Block
# =============================================================================
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
        # Channels last for LayerNorm & MLP
        out = out.permute(0, 2, 3, 1)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)
        out = out.permute(0, 3, 1, 2)
        return residual + out


# =============================================================================
# 7. DenseNet Dense Block
# =============================================================================
class DenseNet_Block(nn.Module):
    def __init__(self, in_c: int = 16, growth_rate: int = 16) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv1 = nn.Conv2d(in_c, growth_rate, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.relu(self.bn1(x)))
        return torch.cat([x, out], dim=1)


# =============================================================================
# 8. ResNeXt Grouped Conv Block
# =============================================================================
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


# =============================================================================
# 9. TCN Dilated Conv Block (1D Temporal ConvNet)
# =============================================================================
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
        # Causal slicing: remove future padding
        S = x.shape[-1]
        out = self.conv(x)[:, :, :S]
        out = self.norm(out)
        return self.relu(out + x)


# =============================================================================
# 10. WaveNet Gated Dilated Conv
# =============================================================================
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


# =============================================================================
# 11. Conv3D Spatio-Temporal Block (Video / Volumetric)
# =============================================================================
class Conv3D_SpatioTemporalBlock(nn.Module):
    def __init__(self, in_c: int = 8, out_c: int = 16) -> None:
        super().__init__()
        self.conv3d = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn3d = nn.BatchNorm3d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn3d(self.conv3d(x)))


# =============================================================================
# Multi-Opset ONNX Export Helper
# =============================================================================
def export_model(
    model: nn.Module,
    dummy_inputs: tuple[torch.Tensor, ...],
    input_names: list[str],
    output_names: list[str],
    base_name: str,
    opset_versions: list[int] = (14, 17, 18, 20),
) -> list[str]:
    model.eval()
    exported = []
    for opset in opset_versions:
        out_path = OUT_DIR / f"{base_name}_opset{opset}.onnx"
        try:
            torch.onnx.export(
                model,
                dummy_inputs,
                str(out_path),
                input_names=input_names,
                output_names=output_names,
                opset_version=opset,
                do_constant_folding=True,
            )
            m_proto = onnx.load(str(out_path))
            onnx.checker.check_model(m_proto)
            size_kb = out_path.stat().st_size / 1024.0
            print(
                f"  [OK] Exported {out_path.name:<45} (opset={opset}, nodes={len(m_proto.graph.node):<3}, {size_kb:.1f} KB)"
            )
            exported.append(str(out_path))
        except Exception as e:
            print(f"  [FAIL] Failed to export {out_path.name} (opset={opset}): {e}")
    return exported


def main() -> None:
    print("==================================================================")
    print("  GENERATING CONVOLUTIONAL & COMPUTER VISION ONNX SUITE")
    print("==================================================================")

    dummy_2d_16 = torch.randn(1, 16, 16, 16)
    dummy_2d_32 = torch.randn(1, 32, 16, 16)
    dummy_2d_64 = torch.randn(1, 64, 16, 16)
    dummy_1d = torch.randn(1, 32, 32)
    dummy_3d = torch.randn(1, 8, 4, 16, 16)

    # 1. StandardConv2D_BN_ReLU
    print("\n1. StandardConv2D_BN_ReLU")
    export_model(
        StandardConv2D_BN_ReLU(16, 32), (dummy_2d_16,), ["x"], ["out"], "conv2d_bn_relu"
    )

    # 2. ResNetBasicBlock
    print("\n2. ResNetBasicBlock")
    export_model(ResNetBasicBlock(32), (dummy_2d_32,), ["x"], ["out"], "resnet_basic")

    # 3. ResNetBottleneckBlock
    print("\n3. ResNetBottleneckBlock")
    export_model(
        ResNetBottleneckBlock(64, 16, 64),
        (dummy_2d_64,),
        ["x"],
        ["out"],
        "resnet_bottleneck",
    )

    # 4. MobileNetV2_InvertedResidual
    print("\n4. MobileNetV2_InvertedResidual")
    export_model(
        MobileNetV2_InvertedResidual(32, 64, 32),
        (dummy_2d_32,),
        ["x"],
        ["out"],
        "mobilenetv2_invres",
    )

    # 5. MobileNetV3_Block
    print("\n5. MobileNetV3_Block")
    export_model(
        MobileNetV3_Block(32, 64, 32),
        (dummy_2d_32,),
        ["x"],
        ["out"],
        "mobilenetv3_block",
    )

    # 6. ConvNeXt_Block
    print("\n6. ConvNeXt_Block")
    export_model(ConvNeXt_Block(32), (dummy_2d_32,), ["x"], ["out"], "convnext_block")

    # 7. DenseNet_Block
    print("\n7. DenseNet_Block")
    export_model(
        DenseNet_Block(16, 16), (dummy_2d_16,), ["x"], ["out"], "densenet_block"
    )

    # 8. ResNeXt_GroupedConvBlock
    print("\n8. ResNeXt_GroupedConvBlock")
    export_model(
        ResNeXt_GroupedConvBlock(64, 8, 8),
        (dummy_2d_64,),
        ["x"],
        ["out"],
        "resnext_block",
    )

    # 9. TCN_DilatedConvBlock
    print("\n9. TCN_DilatedConvBlock")
    export_model(
        TCN_DilatedConvBlock(32, 32, dilation=2),
        (dummy_1d,),
        ["x"],
        ["out"],
        "tcn_dilated",
    )

    # 10. WaveNet_GatedDilatedConv
    print("\n10. WaveNet_GatedDilatedConv")
    export_model(
        WaveNet_GatedDilatedConv(32, dilation=2),
        (dummy_1d,),
        ["x"],
        ["res", "skip"],
        "wavenet_gated",
    )

    # 11. Conv3D_SpatioTemporalBlock
    print("\n11. Conv3D_SpatioTemporalBlock")
    export_model(
        Conv3D_SpatioTemporalBlock(8, 16),
        (dummy_3d,),
        ["x"],
        ["out"],
        "conv3d_spatiotemporal",
    )

    print("\n==================================================================")
    print(f"CNN & Computer Vision ONNX suite complete in: {OUT_DIR.resolve()}")
    print("==================================================================")


if __name__ == "__main__":
    main()
