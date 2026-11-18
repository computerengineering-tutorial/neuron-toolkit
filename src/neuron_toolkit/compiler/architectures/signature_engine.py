"""Declarative signature engine for discovering and synthesizing architecture blocks from JSON."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import onnx

log = logging.getLogger(__name__)


@dataclass
class BlockSignature:
    """Specification of an architectural macro block."""

    name: str
    category: str
    domain: str
    description: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    sub_ops: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BlockSignature:
        return cls(
            name=data["name"],
            category=data.get("category", "general"),
            domain=data.get("domain", "custom.pytorch"),
            description=data.get("description", ""),
            inputs=list(data.get("inputs", [])),
            outputs=list(data.get("outputs", [])),
            sub_ops=list(data.get("sub_ops", [])),
            attributes=dict(data.get("attributes", {})),
        )


class ArchitectureCatalog:
    """Manages loaded architecture block signatures."""

    def __init__(self, catalog_path: str | Path | None = None) -> None:
        if catalog_path is None:
            catalog_path = Path(__file__).parent / "block_signatures.json"
        self.catalog_path = Path(catalog_path)
        self.signatures: dict[str, BlockSignature] = {}
        self.load()

    def load(self) -> None:
        if not self.catalog_path.exists():
            log.warning("Architecture catalog file not found: %s", self.catalog_path)
            return
        with open(self.catalog_path, encoding="utf-8") as f:
            data = json.load(f)
        for sig_dict in data.get("signatures", []):
            sig = BlockSignature.from_dict(sig_dict)
            self.signatures[sig.name] = sig
        log.info(
            "Loaded %d architectural block signatures from %s",
            len(self.signatures),
            self.catalog_path.name,
        )

    def register_signature(self, signature: BlockSignature) -> None:
        self.signatures[signature.name] = signature

    def get_signature(self, name: str) -> BlockSignature | None:
        return self.signatures.get(name)

    def list_signatures(self) -> list[str]:
        return sorted(self.signatures.keys())


class SignatureEngine:
    """Discovers motifs and synthesizes high-level architecture blocks into custom ops or FunctionProtos."""

    def __init__(self, catalog: ArchitectureCatalog | None = None) -> None:
        self.catalog = catalog or ArchitectureCatalog()

    def discover_subgraphs(
        self, model: onnx.ModelProto
    ) -> dict[str, list[dict[str, Any]]]:
        """Detect occurrence of architectural patterns across the model graph."""
        graph = model.graph
        discovered: dict[str, list[dict[str, Any]]] = {}
        op_counts = {}
        for n in graph.node:
            op_counts[n.op_type] = op_counts.get(n.op_type, 0) + 1

        for name, sig in self.catalog.signatures.items():
            # Check if all constituent ops exist in sufficient count
            sub_op_counts = {}
            for op in sig.sub_ops:
                sub_op_counts[op] = sub_op_counts.get(op, 0) + 1

            possible = True
            for op, req_count in sub_op_counts.items():
                if op_counts.get(op, 0) < req_count:
                    possible = False
                    break

            if possible:
                discovered[name] = [
                    {
                        "signature": sig.name,
                        "category": sig.category,
                        "matched_ops": sig.sub_ops,
                        "confidence": min(
                            1.0,
                            sum(op_counts.get(op, 0) for op in sig.sub_ops)
                            / max(1, len(graph.node)),
                        ),
                    }
                ]

        return discovered
