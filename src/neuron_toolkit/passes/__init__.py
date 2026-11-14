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

__all__ = [
    "BasePass",
    "CommonSubexpressionElimination",
    "ConstantFolding",
    "DeadCodeElimination",
    "FuseConvActivation",
    "FuseConvBN",
    "FuseGELU",
    "FuseLayerNorm",
    "PassManager",
    "PassStats",
]
