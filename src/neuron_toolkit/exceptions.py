"""Custom exceptions hierarchy for neuron-toolkit."""

from __future__ import annotations


class NeuronToolkitError(Exception):
    """Base exception for all neuron-toolkit errors."""


class GraphParseError(NeuronToolkitError):
    """Raised when parsing a model file or buffer fails."""


class QueryError(NeuronToolkitError):
    """Raised when an invalid query operation is executed."""


class PatternMatchError(NeuronToolkitError):
    """Raised when an invalid pattern syntax or constraint is evaluated."""


class RewriteError(NeuronToolkitError):
    """Raised when graph surgery or topological sorting fails."""


class VerificationMismatchError(NeuronToolkitError):
    """Raised when model outputs diverge beyond tolerance thresholds."""


class BackendNotAvailableError(NeuronToolkitError):
    """Raised when an optional backend dependency (e.g., tflite) is missing."""
