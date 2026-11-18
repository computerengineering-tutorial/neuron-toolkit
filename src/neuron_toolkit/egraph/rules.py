"""Declarative equality saturation rewrite rules and saturation loop."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from neuron_toolkit.egraph.core import EClassId, ENode

if TYPE_CHECKING:
    from neuron_toolkit.egraph.core import EGraph


class EGraphRule(ABC):
    """Abstract base class for equality rewrite rules on an E-Graph."""

    name: str = "EGraphRule"

    @abstractmethod
    def apply(self, egraph: EGraph) -> int:
        """Apply rule to egraph.

        Returns:
            The number of unions or new nodes introduced.
        """
        raise NotImplementedError


class CommutativeRule(EGraphRule):
    """Symmetric rewrite: op(a, b) = op(b, a)."""

    def __init__(self, op: str) -> None:
        self.op = op
        self.name = f"Commutative_{op}"

    def apply(self, egraph: EGraph) -> int:
        applied = 0
        to_union: list[tuple[EClassId, ENode]] = []

        for can_id, eclass in egraph.classes_for_op(self.op):
            for node in eclass.nodes:
                if node.op == self.op and len(node.children) == 2:
                    a, b = node.children
                    can_a = egraph.find(a)
                    can_b = egraph.find(b)
                    if can_a != can_b:
                        swapped = ENode(
                            op=self.op, children=(can_b, can_a), attrs=node.attrs
                        )
                        to_union.append((can_id, swapped))

        for target_class, new_node in to_union:
            new_id = egraph.add(new_node)
            if egraph.find(target_class) != egraph.find(new_id):
                egraph.union(target_class, new_id)
                applied += 1

        return applied


class IdentityRule(EGraphRule):
    """Eliminates arithmetic identity: Add(x, 0) = x, Mul(x, 1) = x, Sub(x, 0) = x."""

    def __init__(self, op: str, identity_value: float) -> None:
        self.op = op
        self.val = identity_value
        self.name = f"Identity_{op}_{identity_value}"

    def apply(self, egraph: EGraph) -> int:
        applied = 0
        to_union: list[tuple[EClassId, EClassId]] = []

        for can_id, eclass in egraph.classes_for_op(self.op):
            for node in eclass.nodes:
                if node.op == self.op and len(node.children) == 2:
                    c0, c1 = node.children
                    # Check if c0 is identity
                    is_c0_id = False
                    is_c1_id = False

                    class_0 = egraph.classes.get(egraph.find(c0))
                    class_1 = egraph.classes.get(egraph.find(c1))

                    if class_0:
                        for n0 in class_0.nodes:
                            if (
                                n0.op in ("Const", "Constant")
                                and dict(n0.attrs).get("value") == self.val
                            ):
                                is_c0_id = True
                                break

                    if class_1:
                        for n1 in class_1.nodes:
                            if (
                                n1.op in ("Const", "Constant")
                                and dict(n1.attrs).get("value") == self.val
                            ):
                                is_c1_id = True
                                break

                    if is_c1_id:
                        to_union.append((can_id, c0))
                    elif is_c0_id and self.op in ("Add", "Mul"):
                        to_union.append((can_id, c1))

        for c_orig, c_target in to_union:
            if egraph.find(c_orig) != egraph.find(c_target):
                egraph.union(c_orig, c_target)
                applied += 1

        return applied


class ConstantFoldRule(EGraphRule):
    """Folds arithmetic on known constant nodes into a unified constant."""

    name: str = "ConstantFoldRule"

    def apply(self, egraph: EGraph) -> int:
        applied = 0
        to_union: list[tuple[EClassId, float]] = []

        for class_id, eclass in list(egraph.classes.items()):
            can_id = egraph.find(class_id)
            for node in eclass.nodes:
                if node.op in ("Add", "Mul", "Sub") and len(node.children) == 2:
                    c0_class = egraph.classes.get(egraph.find(node.children[0]))
                    c1_class = egraph.classes.get(egraph.find(node.children[1]))
                    if not c0_class or not c1_class:
                        continue

                    val0 = None
                    val1 = None
                    for n in c0_class.nodes:
                        if n.op in ("Const", "Constant") and "value" in dict(n.attrs):
                            val0 = dict(n.attrs)["value"]
                            break
                    for n in c1_class.nodes:
                        if n.op in ("Const", "Constant") and "value" in dict(n.attrs):
                            val1 = dict(n.attrs)["value"]
                            break

                    if val0 is not None and val1 is not None:
                        res = 0.0
                        if node.op == "Add":
                            res = val0 + val1
                        elif node.op == "Mul":
                            res = val0 * val1
                        elif node.op == "Sub":
                            res = val0 - val1
                        to_union.append((can_id, res))

        for c_id, val in to_union:
            c_node = ENode(op="Const", children=(), attrs=(("value", val),))
            new_id = egraph.add(c_node)
            if egraph.find(c_id) != egraph.find(new_id):
                egraph.union(c_id, new_id)
                applied += 1

        return applied


class SiluFusionRule(EGraphRule):
    """Fuses Mul(x, Sigmoid(x)) = Silu(x)."""

    name: str = "SiluFusionRule"

    def apply(self, egraph: EGraph) -> int:
        applied = 0
        to_add: list[tuple[EClassId, EClassId]] = []

        for can_id, eclass in egraph.classes_for_op("Mul"):
            for node in eclass.nodes:
                if node.op == "Mul" and len(node.children) == 2:
                    c0, c1 = node.children
                    can_0 = egraph.find(c0)
                    can_1 = egraph.find(c1)

                    # Check if c0 is Sigmoid(c1)
                    sig_input = None
                    class_0 = egraph.classes.get(can_0)
                    if class_0:
                        for n in class_0.nodes:
                            if (
                                n.op == "Sigmoid"
                                and len(n.children) == 1
                                and egraph.find(n.children[0]) == can_1
                            ):
                                sig_input = can_1
                                break

                    if sig_input is None:
                        class_1 = egraph.classes.get(can_1)
                        if class_1:
                            for n in class_1.nodes:
                                if (
                                    n.op == "Sigmoid"
                                    and len(n.children) == 1
                                    and egraph.find(n.children[0]) == can_0
                                ):
                                    sig_input = can_0
                                    break

                    if sig_input is not None:
                        to_add.append((can_id, sig_input))

        for target_class, inp_class in to_add:
            silu_node = ENode(op="Silu", children=(inp_class,))
            silu_id = egraph.add(silu_node)
            if egraph.find(target_class) != egraph.find(silu_id):
                egraph.union(target_class, silu_id)
                applied += 1

        return applied


def default_rules() -> list[EGraphRule]:
    """Return standard compiler and algebraic rewrite rules."""
    return [
        CommutativeRule("Add"),
        CommutativeRule("Mul"),
        IdentityRule("Add", 0.0),
        IdentityRule("Mul", 1.0),
        IdentityRule("Sub", 0.0),
        ConstantFoldRule(),
        SiluFusionRule(),
    ]


def saturate(
    egraph: EGraph,
    rules: list[EGraphRule] | None = None,
    max_iterations: int = 12,
    node_limit: int = 15000,
) -> dict[str, Any]:
    """Run equality saturation until fixed point, iteration limit, or node limit is reached."""
    active_rules = rules or default_rules()
    total_unions = 0
    iteration = 0

    for _iteration in range(1, max_iterations + 1):
        iteration = _iteration
        if egraph.total_nodes() >= node_limit:
            break

        unions_this_iter = 0
        for rule in active_rules:
            unions_this_iter += rule.apply(egraph)

        egraph.rebuild()
        total_unions += unions_this_iter

        if unions_this_iter == 0:
            break

    return {
        "iterations": iteration,
        "total_unions": total_unions,
        "final_classes": len(egraph.classes),
        "total_nodes": egraph.total_nodes(),
    }
