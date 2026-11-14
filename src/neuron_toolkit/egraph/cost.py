"""Cost models for evaluating expressions and nodes within an E-Graph."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from neuron_toolkit.egraph.core import ENode


class CostModel(ABC):
    """Abstract interface for scoring candidate ENodes during extraction."""

    @abstractmethod
    def cost(self, enode: ENode, child_costs: list[float]) -> float:
        """Compute the cost of enode given the pre-calculated costs of its children."""
        raise NotImplementedError


class NodeCountCostModel(CostModel):
    """Minimizes total number of nodes in the extracted DAG."""

    def __init__(self, op_weights: dict[str, float] | None = None) -> None:
        self.weights = op_weights or {}

    def cost(self, enode: ENode, child_costs: list[float]) -> float:
        base_cost = self.weights.get(enode.op, 1.0)
        # Constants / Variables have lower base cost
        if enode.op in ("Const", "Constant", "Var", "Input"):
            base_cost = 0.1
        return base_cost + sum(child_costs)


class ComputeCostModel(CostModel):
    """Heuristic cost model favoring fused and hardware-efficient operators."""

    OP_COMPUTE_WEIGHTS: ClassVar[dict[str, float]] = {
        "Conv": 10.0,
        "MatMul": 10.0,
        "Gemm": 10.0,
        "RMSNorm": 2.0,
        "LayerNormalization": 3.0,
        "ScaledDotProductAttention": 15.0,
        "SwiGLU": 4.0,
        "Gelu": 2.0,
        "Silu": 2.0,
        "Relu": 1.0,
        "Add": 1.0,
        "Mul": 1.0,
        "Sub": 1.0,
        "Div": 2.0,
        "Sqrt": 2.0,
        "ReduceMean": 3.0,
        "Pow": 3.0,
        "Const": 0.05,
        "Var": 0.05,
    }

    def cost(self, enode: ENode, child_costs: list[float]) -> float:
        base = self.OP_COMPUTE_WEIGHTS.get(enode.op, 2.0)
        return base + sum(child_costs)
