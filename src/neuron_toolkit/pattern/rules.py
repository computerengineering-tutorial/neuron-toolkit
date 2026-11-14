"""Declarative RewriteRule engine for graph transformations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neuron_toolkit.pattern.dsl import Pattern
    from neuron_toolkit.pattern.models import MatchResult
    from neuron_toolkit.rewriter import NeuronRewriter


@dataclass
class RewriteRule:
    """Pairs a pattern with a transformation action.

    Args:
        pattern: Pattern to match in the graph.
        replacement: Callback taking (MatchResult, NeuronRewriter) to apply the rewrite,
                     or string new_op name.
        name: Optional descriptive rule name.
    """

    pattern: Pattern
    replacement: Callable[[MatchResult, NeuronRewriter], Any] | str
    name: str = "RewriteRule"

    def apply(self, match: MatchResult, rewriter: NeuronRewriter) -> None:
        """Apply this rule to a specific match result."""
        if callable(self.replacement):
            self.replacement(match, rewriter)
        elif isinstance(self.replacement, str):
            rewriter.replace_from_result(match, self.replacement)
        else:
            msg = f"Unsupported replacement type: {type(self.replacement)}"
            raise TypeError(msg)
