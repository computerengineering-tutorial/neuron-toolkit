#!/usr/bin/env python3
"""Example Plug-and-Play Addon Model Generator."""

import torch
from torch import nn

from generators.decorators import register_architecture


@register_architecture(
    name="example_addon_block",
    category="addons",
    shapes=[(1, 16, 32)],
    input_names=["x"],
    output_names=["out"],
    opset_versions=(14, 18, 20),
    expected_target_node="Linear",
    emit_json_descriptor=False,
)
class ExampleAddonBlock(nn.Module):
    """Simple linear projection with residual connection demonstration addon."""

    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.proj(x))
