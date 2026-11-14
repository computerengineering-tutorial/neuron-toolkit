"""Optimal expression and DAG extraction from an E-Graph."""

from __future__ import annotations

from typing import TYPE_CHECKING

from neuron_toolkit.egraph.core import EClassId, ENode
from neuron_toolkit.egraph.cost import CostModel, NodeCountCostModel
from neuron_toolkit.ir.core import IRGraph, IRNode

if TYPE_CHECKING:
    from neuron_toolkit.egraph.core import EGraph


class Extractor:
    """Extracts the lowest-cost program DAG from an EGraph according to a CostModel."""

    def __init__(self, egraph: EGraph, cost_model: CostModel | None = None) -> None:
        self.egraph = egraph
        self.cost_model = cost_model or NodeCountCostModel()
        self.costs: dict[EClassId, tuple[float, ENode]] = {}
        self._compute_costs()

    def _compute_costs(self) -> None:
        """Find the optimal ENode and cost for every EClass via fixed-point iteration."""
        # Initialize
        for class_id in self.egraph.classes:
            canonical_id = self.egraph.find(class_id)
            if canonical_id not in self.costs:
                self.costs[canonical_id] = (float("inf"), ENode(op="None"))

        changed = True
        iterations = 0
        max_iters = max(len(self.egraph.classes) * 2, 50)

        while changed and iterations < max_iters:
            changed = False
            iterations += 1

            for class_id, eclass in list(self.egraph.classes.items()):
                can_id = self.egraph.find(class_id)
                current_best_cost, _current_best_node = self.costs.get(
                    can_id, (float("inf"), ENode(op="None"))
                )

                for node in eclass.nodes:
                    # Check if all children have finite costs
                    child_costs: list[float] = []
                    all_children_known = True

                    for child in node.children:
                        can_child = self.egraph.find(child)
                        c_cost, _ = self.costs.get(
                            can_child, (float("inf"), ENode(op="None"))
                        )
                        if c_cost == float("inf"):
                            all_children_known = False
                            break
                        child_costs.append(c_cost)

                    if not all_children_known:
                        continue

                    node_cost = self.cost_model.cost(node, child_costs)
                    if node_cost < current_best_cost:
                        current_best_cost = node_cost
                        self.costs[can_id] = (node_cost, node)
                        changed = True

    def find_best(self, class_id: EClassId) -> tuple[float, ENode]:
        """Return the lowest cost and chosen ENode for a given class."""
        can_id = self.egraph.find(class_id)
        return self.costs.get(can_id, (float("inf"), ENode(op="None")))

    def extract_to_ir(
        self,
        root_class_ids: list[EClassId] | EClassId,
        graph_name: str = "extracted_graph",
    ) -> IRGraph:
        """Reconstruct an IRGraph containing the optimal DAG for the specified root classes."""
        ir = IRGraph(name=graph_name)
        roots = (
            [root_class_ids]
            if isinstance(root_class_ids, int)
            else list(root_class_ids)
        )

        memo_extracted: dict[EClassId, str] = {}
        node_counter = 0

        def _extract_rec(cid: EClassId) -> str:
            nonlocal node_counter
            can_cid = self.egraph.find(cid)
            if can_cid in memo_extracted:
                return memo_extracted[can_cid]

            cost, best_node = self.find_best(can_cid)
            if cost == float("inf") or best_node.op == "None":
                fallback_name = f"t_{can_cid}"
                memo_extracted[can_cid] = fallback_name
                return fallback_name

            # Leaf variable or constant
            if best_node.op in ("Var", "Input"):
                var_name = dict(best_node.attrs).get("name", f"in_{can_cid}")
                memo_extracted[can_cid] = var_name
                if var_name not in ir.inputs:
                    ir.inputs.append(var_name)
                return var_name

            if best_node.op in ("Const", "Constant"):
                c_name = dict(best_node.attrs).get("name", f"const_{can_cid}")
                memo_extracted[can_cid] = c_name
                return c_name

            # Recursively extract child inputs
            input_names = [_extract_rec(ch) for ch in best_node.children]
            out_name = f"{best_node.op.lower()}_{node_counter}_out"
            node_name = f"{best_node.op.lower()}_{node_counter}"
            node_counter += 1

            ir_node = IRNode(
                op_type=best_node.op,
                name=node_name,
                inputs=input_names,
                outputs=[out_name],
                attributes=dict(best_node.attrs),
            )
            ir.add_node(ir_node)
            memo_extracted[can_cid] = out_name
            return out_name

        for r in roots:
            out_tensor = _extract_rec(r)
            if out_tensor not in ir.outputs:
                ir.outputs.append(out_tensor)

        return ir
