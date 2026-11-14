"""Unified Intermediate Representation (IR) for neuron-toolkit.

Framework-agnostic DAG representation of neural network computation graphs,
supporting ONNX, TFLite, PyTorch FX, and custom graph structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class IRTensor:
    """Represents an activation tensor, model parameter, or constant initializer."""

    name: str
    shape: tuple[int | str | None, ...] = ()
    dtype: str = "float32"
    data: np.ndarray | None = None
    is_initializer: bool = False
    quant_params: dict[str, Any] | None = None

    @property
    def rank(self) -> int:
        """Return the rank (number of dimensions) of the tensor."""
        return len(self.shape)

    @property
    def numel(self) -> int | None:
        """Return the total number of elements if shape is static, else None."""
        if any(d is None or isinstance(d, str) for d in self.shape):
            return None
        return int(np.prod(self.shape)) if self.shape else 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize tensor metadata to dictionary."""
        d: dict[str, Any] = {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "is_initializer": self.is_initializer,
        }
        if self.quant_params:
            d["quant_params"] = self.quant_params
        return d


@dataclass(slots=True)
class IRNode:
    """Represents a computation operation in the compute graph."""

    op_type: str
    name: str = ""
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    domain: str = ""

    def has_attr(self, key: str) -> bool:
        """Check if attribute exists."""
        return key in self.attributes

    def get_attr(self, key: str, default: Any = None) -> Any:
        """Retrieve attribute value with default."""
        return self.attributes.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Serialize node to dictionary."""
        return {
            "op_type": self.op_type,
            "name": self.name,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "attributes": dict(self.attributes),
            "domain": self.domain,
        }


class IRGraph:
    """Framework-agnostic Directed Acyclic Graph (DAG) representation."""

    def __init__(self, name: str = "main_graph") -> None:
        self.name: str = name
        self.nodes: list[IRNode] = []
        self.tensors: dict[str, IRTensor] = {}
        self.inputs: list[str] = []
        self.outputs: list[str] = []
        self.metadata: dict[str, Any] = {}

    def add_node(self, node: IRNode) -> None:
        """Append an IRNode to the graph."""
        self.nodes.append(node)

    def add_tensor(self, tensor: IRTensor) -> None:
        """Register an IRTensor in the graph."""
        self.tensors[tensor.name] = tensor

    @property
    def producers(self) -> dict[str, IRNode]:
        """Map from tensor name to the IRNode that produces it."""
        res: dict[str, IRNode] = {}
        for node in self.nodes:
            for out in node.outputs:
                if out:
                    res[out] = node
        return res

    @property
    def consumers(self) -> dict[str, list[IRNode]]:
        """Map from tensor name to list of IRNodes that consume it."""
        res: dict[str, list[IRNode]] = {}
        for node in self.nodes:
            for inp in node.inputs:
                if inp:
                    res.setdefault(inp, []).append(node)
        return res

    def is_topologically_sorted(self) -> bool:
        """Verify whether graph nodes are in valid topological execution order."""
        produced: set[str] = set(self.inputs)
        for tensor in self.tensors.values():
            if tensor.is_initializer:
                produced.add(tensor.name)

        for node in self.nodes:
            for inp in node.inputs:
                if inp and inp not in produced:
                    return False
            for out in node.outputs:
                if out:
                    produced.add(out)
        return True

    def to_dict(self) -> dict[str, Any]:
        """Export IRGraph to dictionary representation."""
        return {
            "name": self.name,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "nodes": [n.to_dict() for n in self.nodes],
            "tensors": {k: t.to_dict() for k, t in self.tensors.items()},
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IRGraph:
        """Construct IRGraph from dictionary."""
        g = cls(name=data.get("name", "main_graph"))
        g.inputs = list(data.get("inputs", []))
        g.outputs = list(data.get("outputs", []))
        g.metadata = dict(data.get("metadata", {}))

        for n_data in data.get("nodes", []):
            node = IRNode(
                op_type=n_data["op_type"],
                name=n_data.get("name", ""),
                inputs=list(n_data.get("inputs", [])),
                outputs=list(n_data.get("outputs", [])),
                attributes=dict(n_data.get("attributes", {})),
                domain=n_data.get("domain", ""),
            )
            g.add_node(node)

        for t_name, t_data in data.get("tensors", {}).items():
            tensor = IRTensor(
                name=t_name,
                shape=tuple(t_data.get("shape", ())),
                dtype=t_data.get("dtype", "float32"),
                is_initializer=t_data.get("is_initializer", False),
                quant_params=t_data.get("quant_params"),
            )
            g.add_tensor(tensor)

        return g
