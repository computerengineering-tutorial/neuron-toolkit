"""Core E-Graph (Equivalence Graph) data structures and Congruence Closure algorithm."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

EClassId = int


@dataclass(frozen=True, slots=True)
class ENode:
    """An operator node in an E-Graph with EClassId children.

    Attributes:
        op: Operation name or symbol (e.g. 'Add', 'Mul', 'Conv', 'Const').
        children: Tuple of canonical EClassIds that are inputs to this node.
        attrs: Sorted tuple of key-value attributes.
    """

    op: str
    children: tuple[EClassId, ...] = ()
    attrs: tuple[tuple[str, Any], ...] = ()

    def canonicalize(self, find_fn: Callable[[EClassId], EClassId]) -> ENode:
        """Return a new ENode where all child class IDs are resolved to canonical representatives."""
        canonical_children = tuple(find_fn(c) for c in self.children)
        if canonical_children == self.children:
            return self
        return ENode(op=self.op, children=canonical_children, attrs=self.attrs)


class UnionFind:
    """Disjoint-set forest with path compression and union by rank."""

    def __init__(self) -> None:
        self.parent: dict[EClassId, EClassId] = {}
        self.rank: dict[EClassId, int] = {}

    def make_set(self, x: EClassId) -> None:
        """Register a new element."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: EClassId) -> EClassId:
        """Find the canonical representative of x with path compression."""
        path: list[EClassId] = []
        curr = x
        while self.parent[curr] != curr:
            path.append(curr)
            curr = self.parent[curr]
        for node in path:
            self.parent[node] = curr
        return curr

    def union(self, x: EClassId, y: EClassId) -> EClassId:
        """Merge the sets containing x and y. Returns the new canonical representative."""
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return root_x

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        self.parent[root_y] = root_x
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        return root_x


@dataclass
class EClass:
    """An equivalence class containing multiple equivalent ENodes."""

    id: EClassId
    nodes: set[ENode] = field(default_factory=set)
    parents: list[tuple[ENode, EClassId]] = field(default_factory=list)
    data: Any = None


class EGraph:
    """Equivalence Graph (E-Graph) supporting hash-consing and incremental congruence closure."""

    def __init__(self) -> None:
        self.classes: dict[EClassId, EClass] = {}
        self.memo: dict[ENode, EClassId] = {}
        self.unionfind: UnionFind = UnionFind()
        self.worklist: set[EClassId] = set()
        self._next_id: int = 0

    def find(self, class_id: EClassId) -> EClassId:
        """Find canonical representative EClassId."""
        return self.unionfind.find(class_id)

    def total_nodes(self) -> int:
        """Total number of ENodes across all equivalence classes."""
        return sum(len(c.nodes) for c in self.classes.values())

    def add(self, enode: ENode) -> EClassId:
        """Insert an ENode using hash-consing.

        If an equivalent node already exists, returns its existing EClassId.
        Otherwise, creates a new EClass.
        """
        can_node = enode.canonicalize(self.find)
        if can_node in self.memo:
            return self.find(self.memo[can_node])

        # Allocate new EClass
        new_id = self._next_id
        self._next_id += 1
        self.unionfind.make_set(new_id)

        eclass = EClass(id=new_id, nodes={can_node})
        self.classes[new_id] = eclass
        self.memo[can_node] = new_id

        # Register parent pointers in child classes
        for child_id in can_node.children:
            can_child = self.find(child_id)
            if can_child in self.classes:
                self.classes[can_child].parents.append((can_node, new_id))

        return new_id

    def union(self, id1: EClassId, id2: EClassId) -> EClassId:
        """Union two equivalence classes and schedule for congruence repair."""
        root1 = self.find(id1)
        root2 = self.find(id2)
        if root1 == root2:
            return root1

        new_root = self.unionfind.union(root1, root2)
        old_root = root2 if new_root == root1 else root1

        # Move nodes, parents, and data to new_root
        c_new = self.classes[new_root]
        c_old = self.classes[old_root]

        c_new.nodes.update(c_old.nodes)
        c_new.parents.extend(c_old.parents)
        if c_new.data is None and c_old.data is not None:
            c_new.data = c_old.data

        del self.classes[old_root]
        self.worklist.add(new_root)
        return new_root

    def rebuild(self) -> int:
        """Congruence closure algorithm.

        Restores canonical invariants across memo table and unions parents
        whose children have become equivalent.
        """
        repaired = 0
        while self.worklist:
            curr_id = self.worklist.pop()
            curr_root = self.find(curr_id)
            if curr_root not in self.classes:
                continue

            eclass = self.classes[curr_root]
            # Deduplicate and canonicalize parents
            new_parents: list[tuple[ENode, EClassId]] = []
            seen: dict[ENode, EClassId] = {}

            for p_node, p_class in eclass.parents:
                can_p = p_node.canonicalize(self.find)
                can_p_class = self.find(p_class)

                if can_p in self.memo:
                    existing_class = self.find(self.memo[can_p])
                    if existing_class != can_p_class:
                        self.union(existing_class, can_p_class)
                        repaired += 1

                self.memo[can_p] = can_p_class

                if can_p in seen:
                    other_class = seen[can_p]
                    if other_class != can_p_class:
                        self.union(other_class, can_p_class)
                        repaired += 1
                else:
                    seen[can_p] = can_p_class
                    new_parents.append((can_p, can_p_class))

            eclass.parents = new_parents

        return repaired

    @classmethod
    def from_ir(cls, ir: Any) -> tuple[EGraph, dict[str, EClassId]]:
        """Construct an EGraph from an IRGraph.

        Returns:
            A tuple of (egraph, tensor_to_class_map).
        """
        egraph = cls()
        tensor_map: dict[str, EClassId] = {}

        # Register inputs
        for inp_name in getattr(ir, "inputs", []):
            var_node = ENode(op="Input", children=(), attrs=(("name", inp_name),))
            tensor_map[inp_name] = egraph.add(var_node)

        # Register initializers/constants
        for t_name, t_obj in getattr(ir, "tensors", {}).items():
            if getattr(t_obj, "is_initializer", False) and t_name not in tensor_map:
                val = getattr(t_obj, "data", None)
                scalar_val = (
                    float(val.item()) if val is not None and val.size == 1 else None
                )
                attrs = [("name", t_name)]
                if scalar_val is not None:
                    attrs.append(("value", scalar_val))
                c_node = ENode(op="Const", children=(), attrs=tuple(attrs))
                tensor_map[t_name] = egraph.add(c_node)

        # Ingest nodes in topological order
        for node in getattr(ir, "nodes", []):
            child_ids: list[EClassId] = []
            for inp in getattr(node, "inputs", []):
                if inp in tensor_map:
                    child_ids.append(tensor_map[inp])
                else:
                    # Allocate fallback input
                    fallback_node = ENode(
                        op="Input", children=(), attrs=(("name", inp),)
                    )
                    cid = egraph.add(fallback_node)
                    tensor_map[inp] = cid
                    child_ids.append(cid)

            attrs = tuple(
                sorted(getattr(node, "attributes", {}).items(), key=lambda x: str(x[0]))
            )
            enode = ENode(
                op=getattr(node, "op_type", ""), children=tuple(child_ids), attrs=attrs
            )
            out_cid = egraph.add(enode)

            for out in getattr(node, "outputs", []):
                if out:
                    tensor_map[out] = out_cid

        return egraph, tensor_map
