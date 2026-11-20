"""Base classes and context definitions for the 3-Stage ONNX Compiler."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import onnx

LATEST_OPSET: int = onnx.defs.onnx_opset_version()


@dataclass
class CompilerContext:
    """Execution context and configuration for compiler optimization passes."""

    mode: str = "function_proto"
    # Mode can be:
    # - 'function_proto': Attaches formal ONNX FunctionProto definitions for Netron visualization.
    # - 'modern_onnx': Emits standard modern ONNX ops (e.g. RMSNormalization, Gelu, GroupNormalization).
    # - 'plain_custom_op' or 'pytorch_custom': Minimalist custom ops for PyTorch runtime.
    custom_domain: str = "custom.pytorch"
    max_fixed_point_iters: int = 5
    rule_ids: list[str] | None = None
    applied_passes: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)

    def record_pass(self, pass_name: str, nodes_before: int, nodes_after: int) -> None:
        """Record pass execution history and node reduction statistics."""
        self.applied_passes.append(pass_name)
        self.stats[pass_name] = {
            "nodes_before": nodes_before,
            "nodes_after": nodes_after,
            "delta": nodes_before - nodes_after,
        }

    def should_run(self, rule_or_pass_id: str) -> bool:
        """Determine whether a specific pass or rule should be executed based on rule_ids filter."""
        if not self.rule_ids:
            return True

        def _base_id(s: str) -> str:
            s = s.lower().strip()
            for suffix in (
                "_canonicalization",
                "_fusion",
                "_folding",
                "_elimination",
                "_projection",
            ):
                if s.endswith(suffix):
                    return s[: -len(suffix)]
            return s

        target = _base_id(rule_or_pass_id)
        for rid in self.rule_ids:
            r = _base_id(rid)
            if target == r:
                return True
            # Disallow non-skip target matching skip rules
            if r.startswith("skip_") and not target.startswith("skip"):
                continue
            if target in ("skip_norm", "skip") and r.startswith("skip_"):
                return True
            if r.startswith(f"{target}_") or target.startswith(f"{r}_"):
                return True
            if target in r.split("_") or r in target.split("_"):
                return True
        return False


class Pass(ABC):
    """Abstract base class for all compiler canonicalization and synthesis passes."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier name for this pass."""
        ...

    @property
    def category(self) -> str:
        """Category: 'canonicalization', 'synthesis', or 'optimization'."""
        return "general"

    @abstractmethod
    def run(self, model: onnx.ModelProto, context: CompilerContext) -> onnx.ModelProto:
        """Execute transformation pass on the given ONNX ModelProto."""
        ...

    def __call__(
        self, model: onnx.ModelProto, context: CompilerContext | None = None
    ) -> onnx.ModelProto:
        if context is None:
            context = CompilerContext()
        return self.run(model, context)


CompilerPass = Pass
