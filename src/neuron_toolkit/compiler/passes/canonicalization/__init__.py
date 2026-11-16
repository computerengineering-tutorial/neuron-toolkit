"""Canonicalization micro-passes package."""

from neuron_toolkit.compiler.passes.canonicalization.activation_canonicalizer import (
    ActivationCanonicalizer,
)
from neuron_toolkit.compiler.passes.canonicalization.math_canonicalizer import (
    MathCanonicalizer,
)
from neuron_toolkit.compiler.passes.canonicalization.norm_canonicalizer import (
    NormCanonicalizer,
)
from neuron_toolkit.compiler.passes.canonicalization.shape_canonicalizer import (
    ShapeCanonicalizer,
)
from neuron_toolkit.compiler.passes.canonicalization.static_folder import (
    StaticFolder,
)

__all__ = [
    "ActivationCanonicalizer",
    "MathCanonicalizer",
    "NormCanonicalizer",
    "ShapeCanonicalizer",
    "StaticFolder",
]
