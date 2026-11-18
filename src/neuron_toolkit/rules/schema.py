"""Schema definitions for declarative JSON pattern rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NodePattern:
    """Pattern specification for a single node inside a subgraph."""

    id: str
    op_type: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodePattern:
        return cls(
            id=data["id"],
            op_type=data.get("op_type", "*"),
            inputs=list(data.get("inputs", [])),
            outputs=list(data.get("outputs", [])),
            attributes=dict(data.get("attributes", {})),
        )


@dataclass
class RuleCondition:
    """Predicate condition on tensors, shapes, constants, or dtypes."""

    condition_type: str  # 'is_initializer', 'dtype_eq', 'scalar_approx', 'is_constant'
    tensor: str  # Binding variable e.g. '$weight_in'
    value: Any = None
    tolerance: float = 1e-4

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuleCondition:
        return cls(
            condition_type=data.get("type", "is_initializer"),
            tensor=data.get("tensor", ""),
            value=data.get("value"),
            tolerance=float(data.get("tolerance", 1e-4)),
        )


@dataclass
class RuleAction:
    """Action definition for graph replacement and rewriting."""

    action_type: str  # 'fuse_to_function', 'cast_fold', 'math_collapse', 'replace_op'
    op_type: str = ""
    domain: str = "custom.pytorch"
    pytorch_module: str = ""
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    targets: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuleAction:
        action_type = data.get("type")
        op_type = data.get("op_type", "")
        domain = data.get("domain", "custom.pytorch")
        inputs = list(data.get("inputs", []))
        outputs = list(data.get("outputs", []))
        attributes = dict(data.get("attributes", {}))

        # Support target_pattern format with nested nodes list
        if not op_type and "nodes" in data and len(data["nodes"]) > 0:
            first_node = data["nodes"][0]
            op_type = first_node.get("op_type", "")
            domain = first_node.get("domain", domain)
            if not inputs:
                inputs = list(first_node.get("inputs", []))
            if not outputs:
                outputs = list(first_node.get("outputs", []))
            if not attributes:
                attributes = dict(first_node.get("attributes", {}))
            if not action_type:
                action_type = "replace_op"

        if not action_type:
            action_type = "replace_op"
        elif action_type == "replace":
            action_type = "replace_op"

        return cls(
            action_type=action_type,
            op_type=op_type,
            domain=domain,
            pytorch_module=data.get("pytorch_module", ""),
            inputs=inputs,
            outputs=outputs,
            attributes=attributes,
            metadata=dict(data.get("metadata", {})),
            targets=dict(data.get("targets", {})),
        )


@dataclass
class JSONRule:
    """Declarative rewrite rule loaded from JSON."""

    rule_id: str
    description: str
    nodes: list[NodePattern]
    conditions: list[RuleCondition] = field(default_factory=list)
    action: RuleAction = field(default_factory=lambda: RuleAction("replace_op"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JSONRule:
        pattern_dict = data.get("pattern") or data.get("source_pattern") or {}
        action_dict = data.get("action") or data.get("target_pattern") or {}
        return cls(
            rule_id=data.get("rule_id", "unnamed_rule"),
            description=data.get("description", ""),
            nodes=[NodePattern.from_dict(n) for n in pattern_dict.get("nodes", [])],
            conditions=[
                RuleCondition.from_dict(c) for c in pattern_dict.get("conditions", [])
            ],
            action=RuleAction.from_dict(action_dict),
        )
