"""Unified Generator Framework for Deep Learning Architecture Zoo."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from generators.base import compute_graph_topology_hash, export_model
from generators.decorators import register_architecture
from generators.obfuscation import ObfuscationRegistry, register_obfuscation_strategy
from generators.registry import ModelRegistry, ModelSpec

__all__ = [
    "ModelRegistry",
    "ModelSpec",
    "ObfuscationRegistry",
    "compute_graph_topology_hash",
    "export_model",
    "register_architecture",
    "register_obfuscation_strategy",
]
