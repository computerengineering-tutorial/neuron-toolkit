"""Optimization passes and pipeline management."""

from neuron_toolkit.passes.base import BasePass
from neuron_toolkit.passes.manager import PassManager, PassStats
from neuron_toolkit.passes.standard import (
    CommonSubexpressionElimination,
    ConstantFolding,
    DeadCodeElimination,
    FuseConvActivation,
    FuseConvBN,
    FuseGELU,
    FuseLayerNorm,
)
from neuron_toolkit.passes.transformer import (
    FuseAttention,
    FuseRMSNorm,
    FuseRotaryEmbedding,
    FuseSwiGLU,
)

__all__ = [
    "BasePass",
    "CommonSubexpressionElimination",
    "ConstantFolding",
    "DeadCodeElimination",
    "FuseAttention",
    "FuseConvActivation",
    "FuseConvBN",
    "FuseGELU",
    "FuseLayerNorm",
    "FuseRMSNorm",
    "FuseRotaryEmbedding",
    "FuseSwiGLU",
    "PassManager",
    "PassStats",
]
