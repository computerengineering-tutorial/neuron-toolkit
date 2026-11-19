#!/usr/bin/env python3
"""Decorator APIs for declarative DL Architecture registration."""

from __future__ import annotations

import re
from typing import Any, Callable, Sequence
import torch
from torch import nn

from generators.registry import ModelRegistry, ModelSpec
from generators.obfuscation import ObfuscationRegistry, register_obfuscation_strategy


def _to_snake_case(name: str) -> str:
    """Convert PascalCase or mixed name to snake_case."""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower().replace("__", "_")


def register_architecture(
    name: str | None = None,
    category: str | None = None,
    subfolder: str | None = None,
    shapes: list[tuple[int, ...]] | None = None,
    inputs: Callable[[], tuple[torch.Tensor, ...] | torch.Tensor] | None = None,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    opset_versions: Sequence[int] = (14, 17, 18, 20),
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    expected_target_node: str | None = None,
    min_expected_reduction: float = 0.0,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    init_args: tuple = (),
    init_kwargs: dict[str, Any] | None = None,
    emit_json_descriptor: bool = False,
    description: str = "",
    tags: list[str] | None = None,
) -> Callable[[Any], Any]:
    """Decorator to register a PyTorch architecture class or factory function.

    Example:
        @register_architecture(
            category="gnn",
            shapes=[(1, 16, 32), (1, 16, 16)],
            input_names=["x", "adj_norm"],
            expected_target_node="GCN",
        )
        class GCN_Layer(nn.Module):
            ...
    """

    def decorator(target: Any) -> Any:
        model_name = name or _to_snake_case(target.__name__)

        # Inferred category from module name if not provided
        model_category = category
        if model_category is None:
            mod = getattr(target, "__module__", "")
            base_mod = mod.split(".")[-1].replace("gen_", "").replace("_zoo", "")
            model_category = base_mod or "general"

        spec = ModelSpec(
            name=model_name,
            target=target,
            category=model_category,
            subfolder=subfolder,
            shapes=shapes,
            input_factory=inputs,
            input_names=input_names or ["x"],
            output_names=output_names or ["out"],
            opset_versions=opset_versions,
            dynamic_axes=dynamic_axes,
            expected_target_node=expected_target_node,
            min_expected_reduction=min_expected_reduction,
            rtol=rtol,
            atol=atol,
            init_args=init_args,
            init_kwargs=init_kwargs or {},
            emit_json_descriptor=emit_json_descriptor,
            description=description or (target.__doc__ or ""),
            tags=tags or [],
        )

        ModelRegistry.register(spec)
        return target

    return decorator
