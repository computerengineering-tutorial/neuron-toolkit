"""Shape & Tensor Cleansing Canonicalizer Pass.

Handles:
- Cast-initializer folding (absorb static casting into weights)
- Identity operation pruning (Mul 1.0, Add 0.0, Sub 0.0, Div 1.0)
- Common Subexpression Elimination (CSE)
- Dead Code Elimination (DCE)
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper

from neuron_toolkit.compiler.base import CompilerContext, Pass

log = logging.getLogger("neuron_toolkit.compiler.shape_canonicalizer")


class ShapeCanonicalizer(Pass):
    """Canonicalizes shapes, eliminates duplicate deterministic ops, folds casts and prunes dead code."""

    @property
    def name(self) -> str:
        return "shape_canonicalizer"

    @property
    def category(self) -> str:
        return "canonicalization"

    def run(self, model: onnx.ModelProto, context: CompilerContext) -> onnx.ModelProto:
        nodes_before = len(model.graph.node)

        if context.should_run("cast"):
            model = self.apply_cast_initializer_folding(model)
            model = self.apply_redundant_cast_elimination(model)
        if context.should_run("identity"):
            model = self.apply_identity_elimination(model)
        if context.should_run("reshape") or context.should_run("shape"):
            model = self.apply_reshape_collapsing(model)
            model = self.apply_squeeze_unsqueeze_cancellation(model)
        if context.should_run("pad_conv") or context.should_run("conv"):
            model = self.apply_pad_conv_fusion(model)
        if context.should_run("transpose") or context.should_run("shape"):
            model = self.apply_transpose_cancellation(model)
        if context.should_run("cse") or context.should_run("common_subexpression"):
            model = self.apply_common_subexpression_elimination(model)
        if context.should_run("dce") or context.should_run("dead_code"):
            model = self.apply_dead_code_elimination(model)

        nodes_after = len(model.graph.node)
        context.record_pass(self.name, nodes_before, nodes_after)
        return model

    # -------------------------------------------------------------------------
    # Rule: Cast-Initializer Folding
    # -------------------------------------------------------------------------
    def apply_cast_initializer_folding(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fold all Cast nodes converting raw initializers into casted initializers for any consumer op."""
        graph = model.graph
        init_arrays = {
            init.name: numpy_helper.to_array(init) for init in graph.initializer
        }

        consumer_map: dict[str, list[onnx.NodeProto]] = {}
        for n in graph.node:
            for inp in n.input:
                consumer_map.setdefault(inp, []).append(n)

        nodes_to_remove: set[str] = set()
        new_initializers: dict[str, onnx.TensorProto] = {}
        modified = False

        for node in graph.node:
            if (
                node.op_type == "Cast"
                and len(node.input) == 1
                and len(node.output) == 1
            ):
                raw_in = node.input[0]
                if raw_in in init_arrays:
                    to_type = 1
                    for attr in node.attribute:
                        if attr.name == "to":
                            to_type = attr.i
                            break
                    raw_arr = init_arrays[raw_in]
                    if to_type == 1:
                        target_arr = raw_arr.astype(np.float32)
                    elif to_type == 10:
                        target_arr = raw_arr.astype(np.float16)
                    elif to_type == 7:
                        target_arr = raw_arr.astype(np.int64)
                    else:
                        target_arr = raw_arr

                    out_name = node.output[0]
                    folded_init = numpy_helper.from_array(target_arr, name=out_name)
                    new_initializers[out_name] = folded_init
                    nodes_to_remove.add(node.name or out_name)
                    modified = True

        if not modified:
            return model

        # Update initializers: remove old inits if no remaining consumers outside nodes_to_remove
        consumed_raw: set[str] = set()

        def collect_consumed(g_proto: onnx.GraphProto) -> None:
            for n in g_proto.node:
                if (n.name or (n.output[0] if n.output else "")) not in nodes_to_remove:
                    for inp in n.input:
                        if inp:
                            consumed_raw.add(inp)
                for attr in n.attribute:
                    if attr.type == onnx.AttributeProto.GRAPH and attr.g:
                        collect_consumed(attr.g)
                    elif attr.type == onnx.AttributeProto.GRAPHS:
                        for sub_g in attr.graphs:
                            collect_consumed(sub_g)

        collect_consumed(graph)
        filtered_inits = [
            init
            for init in graph.initializer
            if init.name in consumed_raw and init.name not in new_initializers
        ]
        for folded in new_initializers.values():
            filtered_inits.append(folded)

        del graph.initializer[:]
        graph.initializer.extend(filtered_inits)

        if hasattr(graph, "input"):
            retained_inputs = [
                inp
                for inp in graph.input
                if inp.name in consumed_raw or inp.name not in init_arrays
            ]
            del graph.input[:]
            graph.input.extend(retained_inputs)

        filtered_nodes = [
            n
            for n in graph.node
            if (n.name or n.output[0] if n.output else "") not in nodes_to_remove
            and not (
                n.op_type == "Cast" and n.output and n.output[0] in new_initializers
            )
        ]
        del graph.node[:]
        graph.node.extend(filtered_nodes)

        log.info(
            "Cast-Initializer Folding: Removed %d Cast nodes", len(nodes_to_remove)
        )
        return model

    def apply_cast_conv_weight_folding(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Alias for backward compatibility."""
        return self.apply_cast_initializer_folding(model)

    # -------------------------------------------------------------------------
    # Rule: Identity Arithmetic Elimination
    # -------------------------------------------------------------------------
    def apply_identity_elimination(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Eliminate redundant identity operations (Mul 1.0, Add 0.0, Sub 0.0, Div 1.0, Identity)."""
        graph = model.graph
        const_map: dict[str, np.ndarray] = {}
        for n in graph.node:
            if n.op_type == "Constant":
                for attr in n.attribute:
                    if attr.name == "value":
                        try:
                            const_map[n.output[0]] = numpy_helper.to_array(attr.t)
                        except Exception:
                            pass
        for init in graph.initializer:
            try:
                const_map[init.name] = numpy_helper.to_array(init)
            except Exception:
                pass

        rewrites: dict[str, str] = {}
        identity_drops: set[str] = set()
        graph_outputs = {o.name for o in graph.output}

        for n in graph.node:
            if (
                n.op_type == "Identity"
                and len(n.input) == 1
                and n.output[0] not in graph_outputs
            ):
                rewrites[n.output[0]] = n.input[0]
                identity_drops.add(n.name or n.output[0])
            elif (
                n.op_type == "Mul"
                and len(n.input) == 2
                and n.output[0] not in graph_outputs
            ):
                for i in (0, 1):
                    inp, other = n.input[i], n.input[1 - i]
                    if inp in const_map:
                        v = const_map[inp]
                        if v.size == 1 and float(v.item()) == 1.0:
                            rewrites[n.output[0]] = other
                            identity_drops.add(n.name or n.output[0])
                            break
            elif (
                n.op_type == "Add"
                and len(n.input) == 2
                and n.output[0] not in graph_outputs
            ):
                for i in (0, 1):
                    inp, other = n.input[i], n.input[1 - i]
                    if inp in const_map:
                        v = const_map[inp]
                        if v.size == 1 and float(v.item()) == 0.0:
                            rewrites[n.output[0]] = other
                            identity_drops.add(n.name or n.output[0])
                            break
            elif (
                n.op_type == "Sub"
                and len(n.input) == 2
                and n.output[0] not in graph_outputs
            ):
                inp, other = n.input[1], n.input[0]
                if inp in const_map:
                    v = const_map[inp]
                    if v.size == 1 and float(v.item()) == 0.0:
                        rewrites[n.output[0]] = other
                        identity_drops.add(n.name or n.output[0])
            elif (
                n.op_type == "Div"
                and len(n.input) == 2
                and n.output[0] not in graph_outputs
            ):
                inp, other = n.input[1], n.input[0]
                if inp in const_map:
                    v = const_map[inp]
                    if v.size == 1 and float(v.item()) == 1.0:
                        rewrites[n.output[0]] = other
                        identity_drops.add(n.name or n.output[0])

        if not identity_drops:
            return model

        def canonical(name: str) -> str:
            while name in rewrites:
                name = rewrites[name]
            return name

        new_nodes: list[onnx.NodeProto] = []
        for n in graph.node:
            if (n.name or n.output[0]) in identity_drops:
                continue
            new_n = copy.deepcopy(n)
            for idx, inp in enumerate(new_n.input):
                new_n.input[idx] = canonical(inp)
            new_nodes.append(new_n)

        del graph.node[:]
        graph.node.extend(new_nodes)
        log.info("Identity Elimination: Removed %d identity ops", len(identity_drops))
        return model

    # -------------------------------------------------------------------------
    # Rule: Transpose Ping-Pong & Identity Transpose Cancellation
    # -------------------------------------------------------------------------
    def apply_transpose_cancellation(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Cancel redundant identity transposes and collapse consecutive Transpose pairs."""
        graph = model.graph
        graph_outputs = {o.name for o in graph.output}

        def get_perm(n: onnx.NodeProto) -> list[int] | None:
            for a in n.attribute:
                if a.name == "perm":
                    return list(a.ints)
            return None

        for _ in range(25):
            node_by_output = {out: n for n in graph.node for out in n.output}
            consumer_map: dict[str, list[onnx.NodeProto]] = {}
            for n in graph.node:
                for inp in n.input:
                    consumer_map.setdefault(inp, []).append(n)

            rewrites: dict[str, str] = {}
            nodes_to_drop: set[str] = set()
            modified = False

            for n in graph.node:
                if n.op_type != "Transpose" or len(n.input) != 1 or len(n.output) != 1:
                    continue

                perm = get_perm(n)
                if perm is None:
                    continue

                # Case 1: Standalone identity transpose (e.g. perm = [0, 1, 2])
                if perm == list(range(len(perm))):
                    if n.output[0] not in graph_outputs:
                        rewrites[n.output[0]] = n.input[0]
                        nodes_to_drop.add(n.name or n.output[0])
                        modified = True
                        continue
                    n.op_type = "Identity"
                    del n.attribute[:]
                    modified = True
                    continue

                # Case 2: Consecutive Transpose nodes
                prev_node = node_by_output.get(n.input[0])
                if (
                    prev_node
                    and prev_node.op_type == "Transpose"
                    and len(prev_node.input) == 1
                    and (prev_node.name or prev_node.output[0]) not in nodes_to_drop
                ):
                    p1 = get_perm(prev_node)
                    p2 = perm
                    if p1 is not None and len(p1) == len(p2):
                        # Composed permutation: dim i comes from dim p1[p2[i]] of raw input
                        p_comp = [p1[idx] for idx in p2]
                        if p_comp == list(range(len(p_comp))):
                            if n.output[0] not in graph_outputs:
                                rewrites[n.output[0]] = prev_node.input[0]
                                nodes_to_drop.add(n.name or n.output[0])
                                if (
                                    len(consumer_map.get(prev_node.output[0], [])) <= 1
                                    and prev_node.output[0] not in graph_outputs
                                ):
                                    nodes_to_drop.add(
                                        prev_node.name or prev_node.output[0]
                                    )
                                modified = True
                            else:
                                n.op_type = "Identity"
                                n.input[0] = prev_node.input[0]
                                del n.attribute[:]
                                if (
                                    len(consumer_map.get(prev_node.output[0], [])) <= 1
                                    and prev_node.output[0] not in graph_outputs
                                ):
                                    nodes_to_drop.add(
                                        prev_node.name or prev_node.output[0]
                                    )
                                modified = True
                        elif (
                            len(consumer_map.get(prev_node.output[0], [])) <= 1
                            and prev_node.output[0] not in graph_outputs
                        ):
                            n.input[0] = prev_node.input[0]
                            for a in n.attribute:
                                if a.name == "perm":
                                    del a.ints[:]
                                    a.ints.extend(p_comp)
                            nodes_to_drop.add(prev_node.name or prev_node.output[0])
                            modified = True

            if not modified:
                break

            def canonical(name: str, rewrites_map: dict[str, str] = rewrites) -> str:
                while name in rewrites_map:
                    name = rewrites_map[name]
                return name

            new_nodes: list[onnx.NodeProto] = []
            for n in graph.node:
                if (n.name or n.output[0]) in nodes_to_drop:
                    continue
                new_n = copy.deepcopy(n)
                for idx, inp in enumerate(new_n.input):
                    new_n.input[idx] = canonical(inp)
                new_nodes.append(new_n)

            del graph.node[:]
            graph.node.extend(new_nodes)

        return self.apply_dead_code_elimination(model)

    # -------------------------------------------------------------------------
    # Rule: Consecutive Reshape Collapse
    # -------------------------------------------------------------------------
    def apply_reshape_collapsing(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Collapse consecutive Reshape operations: Reshape(X, s1) -> Reshape(..., s2) => Reshape(X, s2)."""
        graph = model.graph
        graph_outputs = {o.name for o in graph.output}

        for _ in range(25):
            node_by_output = {out: n for n in graph.node for out in n.output}
            consumer_map: dict[str, list[onnx.NodeProto]] = {}
            for n in graph.node:
                for inp in n.input:
                    consumer_map.setdefault(inp, []).append(n)

            nodes_to_drop: set[str] = set()
            modified = False

            for n in graph.node:
                if n.op_type != "Reshape" or len(n.input) < 2 or len(n.output) != 1:
                    continue

                prev_node = node_by_output.get(n.input[0])
                if (
                    prev_node
                    and prev_node.op_type == "Reshape"
                    and len(prev_node.input) >= 2
                    and (prev_node.name or prev_node.output[0]) not in nodes_to_drop
                ):
                    interm = prev_node.output[0]
                    if interm not in graph_outputs:
                        n.input[0] = prev_node.input[0]
                        if len(consumer_map.get(interm, [])) <= 1:
                            nodes_to_drop.add(prev_node.name or interm)
                        modified = True

            if not modified:
                break

            new_nodes = [
                n for n in graph.node if (n.name or n.output[0]) not in nodes_to_drop
            ]
            del graph.node[:]
            graph.node.extend(new_nodes)

        return self.apply_dead_code_elimination(model)

    # -------------------------------------------------------------------------
    # Rule: Pad into Conv Fusion
    # -------------------------------------------------------------------------
    def apply_pad_conv_fusion(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse Pad -> Conv into Conv with combined pads when Pad mode is constant 0."""
        from onnx import helper

        graph = model.graph
        graph_outputs = {o.name for o in graph.output}
        node_by_output = {out: n for n in graph.node for out in n.output}
        consumer_map: dict[str, list[onnx.NodeProto]] = {}
        for n in graph.node:
            for inp in n.input:
                consumer_map.setdefault(inp, []).append(n)

        init_map = {
            init.name: numpy_helper.to_array(init) for init in graph.initializer
        }
        nodes_to_drop: set[str] = set()
        modified = False

        for n in graph.node:
            if n.op_type != "Conv" or len(n.input) < 2:
                continue

            pad_node = node_by_output.get(n.input[0])
            if not pad_node or pad_node.op_type != "Pad":
                continue

            pad_out = pad_node.output[0]
            if pad_out in graph_outputs or len(consumer_map.get(pad_out, [])) > 1:
                continue

            # Verify pad mode is constant (or default)
            pad_mode = "constant"
            for a in pad_node.attribute:
                if a.name == "mode":
                    pad_mode = (
                        a.s.decode("utf-8") if isinstance(a.s, bytes) else str(a.s)
                    )
                    break
            if pad_mode.lower() != "constant":
                continue

            # Check constant value is 0 (if provided)
            if len(pad_node.input) >= 3 and pad_node.input[2]:
                cval_name = pad_node.input[2]
                if cval_name in init_map:
                    arr = init_map[cval_name]
                    if arr.size == 1 and float(arr.item()) != 0.0:
                        continue

            # Get pads array
            pads = None
            if len(pad_node.input) >= 2 and pad_node.input[1] in init_map:
                pads = list(init_map[pad_node.input[1]])
            else:
                for a in pad_node.attribute:
                    if a.name == "pads":
                        pads = list(a.ints)
                        break

            if pads is None:
                continue

            if (
                len(pads) == 8
                and pads[0] == 0
                and pads[1] == 0
                and pads[4] == 0
                and pads[5] == 0
            ):
                pad_top, pad_left = pads[2], pads[3]
                pad_bottom, pad_right = pads[6], pads[7]
            elif len(pads) == 4:
                pad_top, pad_left, pad_bottom, pad_right = (
                    pads[0],
                    pads[1],
                    pads[2],
                    pads[3],
                )
            else:
                continue

            # Extract existing Conv pads
            conv_pads = [0, 0, 0, 0]
            conv_pad_attr = None
            for a in n.attribute:
                if a.name == "pads":
                    conv_pads = list(a.ints)
                    conv_pad_attr = a
                    break

            if len(conv_pads) == 4:
                new_pads = [
                    int(conv_pads[0] + pad_top),
                    int(conv_pads[1] + pad_left),
                    int(conv_pads[2] + pad_bottom),
                    int(conv_pads[3] + pad_right),
                ]
            else:
                continue

            # Rewire Conv input[0] to Pad input[0]
            n.input[0] = pad_node.input[0]
            if conv_pad_attr is not None:
                del conv_pad_attr.ints[:]
                conv_pad_attr.ints.extend(new_pads)
            else:
                n.attribute.append(helper.make_attribute("pads", new_pads))

            nodes_to_drop.add(pad_node.name or pad_out)
            modified = True

        if modified:
            new_nodes = [
                n for n in graph.node if (n.name or n.output[0]) not in nodes_to_drop
            ]
            del graph.node[:]
            graph.node.extend(new_nodes)

        return self.apply_dead_code_elimination(model)

    # -------------------------------------------------------------------------
    # Rule: Redundant Cast Elimination
    # -------------------------------------------------------------------------
    def apply_redundant_cast_elimination(
        self, model: onnx.ModelProto
    ) -> onnx.ModelProto:
        """Eliminate redundant chained Cast ops: Cast(A -> B) -> Cast(B -> C) => Cast(A -> C)."""
        graph = model.graph
        graph_outputs = {o.name for o in graph.output}

        for _ in range(3):
            node_by_output = {out: n for n in graph.node for out in n.output}
            consumer_map: dict[str, list[onnx.NodeProto]] = {}
            for n in graph.node:
                for inp in n.input:
                    consumer_map.setdefault(inp, []).append(n)

            nodes_to_drop: set[str] = set()
            modified = False

            for n in graph.node:
                if n.op_type != "Cast" or len(n.input) != 1 or len(n.output) != 1:
                    continue

                prev_node = node_by_output.get(n.input[0])
                if (
                    prev_node
                    and prev_node.op_type == "Cast"
                    and len(prev_node.input) == 1
                    and len(prev_node.output) == 1
                    and (prev_node.name or prev_node.output[0]) not in nodes_to_drop
                ):
                    interm = prev_node.output[0]
                    if (
                        interm not in graph_outputs
                        and len(consumer_map.get(interm, [])) <= 1
                    ):
                        n.input[0] = prev_node.input[0]
                        nodes_to_drop.add(prev_node.name or interm)
                        modified = True

            if not modified:
                break

            new_nodes = [
                n for n in graph.node if (n.name or n.output[0]) not in nodes_to_drop
            ]
            del graph.node[:]
            graph.node.extend(new_nodes)

        return self.apply_dead_code_elimination(model)

    # -------------------------------------------------------------------------
    # Rule: Squeeze / Unsqueeze Cancellation
    # -------------------------------------------------------------------------
    def apply_squeeze_unsqueeze_cancellation(
        self, model: onnx.ModelProto
    ) -> onnx.ModelProto:
        """Cancel matching Unsqueeze immediately followed by Squeeze along the same axes."""
        graph = model.graph
        graph_outputs = {o.name for o in graph.output}
        init_map = {
            init.name: numpy_helper.to_array(init) for init in graph.initializer
        }

        for _ in range(3):
            node_by_output = {out: n for n in graph.node for out in n.output}
            consumer_map: dict[str, list[onnx.NodeProto]] = {}
            for n in graph.node:
                for inp in n.input:
                    consumer_map.setdefault(inp, []).append(n)

            rewrites: dict[str, str] = {}
            nodes_to_drop: set[str] = set()
            modified = False

            def get_axes(node: onnx.NodeProto) -> list[int] | None:
                if len(node.input) >= 2 and node.input[1] in init_map:
                    return [int(x) for x in init_map[node.input[1]].flatten()]
                for a in node.attribute:
                    if a.name == "axes":
                        return list(a.ints)
                return None

            for n in graph.node:
                if n.op_type != "Squeeze" or len(n.input) < 1:
                    continue

                prev_node = node_by_output.get(n.input[0])
                if (
                    prev_node
                    and prev_node.op_type == "Unsqueeze"
                    and (prev_node.name or prev_node.output[0]) not in nodes_to_drop
                ):
                    interm = prev_node.output[0]
                    if interm not in graph_outputs:
                        axes_sq = get_axes(n)
                        axes_unsq = get_axes(prev_node)
                        if (
                            axes_sq is not None
                            and axes_unsq is not None
                            and sorted(axes_sq) == sorted(axes_unsq)
                        ):
                            rewrites[n.output[0]] = prev_node.input[0]
                            nodes_to_drop.add(n.name or n.output[0])
                            if len(consumer_map.get(interm, [])) <= 1:
                                nodes_to_drop.add(prev_node.name or interm)
                            modified = True

            if not modified:
                break

            def canonical(name: str) -> str:
                while name in rewrites:
                    name = rewrites[name]
                return name

            new_nodes = []
            for n in graph.node:
                if (n.name or n.output[0]) in nodes_to_drop:
                    continue
                new_n = copy.deepcopy(n)
                for idx, inp in enumerate(new_n.input):
                    new_n.input[idx] = canonical(inp)
                new_nodes.append(new_n)
            del graph.node[:]
            graph.node.extend(new_nodes)

        return self.apply_dead_code_elimination(model)

    # -------------------------------------------------------------------------
    # Rule: Common Subexpression Elimination (CSE)
    # -------------------------------------------------------------------------
    def apply_common_subexpression_elimination(
        self, model: onnx.ModelProto
    ) -> onnx.ModelProto:
        """Deduplicate identical deterministic subexpressions across the graph."""
        graph = model.graph
        rewrites: dict[str, str] = {}
        nodes_to_drop: set[str] = set()
        graph_outputs = {o.name for o in graph.output}

        # Step 1: Constant Deduplication
        const_val_map: dict[tuple[str, tuple[int, ...], bytes], str] = {}
        for n in graph.node:
            if n.op_type == "Constant":
                for attr in n.attribute:
                    if attr.name == "value":
                        try:
                            arr = numpy_helper.to_array(attr.t)
                            key = (arr.dtype.str, tuple(arr.shape), arr.tobytes())
                            if key in const_val_map:
                                if n.output and n.output[0] not in graph_outputs:
                                    rewrites[n.output[0]] = const_val_map[key]
                                    nodes_to_drop.add(n.name or n.output[0])
                            else:
                                const_val_map[key] = n.output[0]
                        except Exception:
                            pass

        def canonical(name: str) -> str:
            while name in rewrites:
                name = rewrites[name]
            return name

        # Step 2: Cascading CSE on deterministic pure operations
        pure_ops = {
            "Shape",
            "Gather",
            "Unsqueeze",
            "Concat",
            "Squeeze",
            "Slice",
            "Cast",
            "Reshape",
        }
        for _ in range(25):
            sig_map: dict[
                tuple[str, tuple[str, ...], tuple[tuple[str, Any], ...]], str
            ] = {}
            new_drops = 0
            for n in graph.node:
                if (n.name or n.output[0]) in nodes_to_drop:
                    continue
                c_inputs = tuple(canonical(inp) for inp in n.input)
                if n.op_type in pure_ops and len(n.output) == 1:
                    attrs = tuple(
                        sorted(
                            (
                                a.name,
                                (
                                    a.i
                                    if a.type == onnx.AttributeProto.INT
                                    else tuple(a.ints)
                                    if a.type == onnx.AttributeProto.INTS
                                    else a.f
                                    if a.type == onnx.AttributeProto.FLOAT
                                    else tuple(a.floats)
                                    if a.type == onnx.AttributeProto.FLOATS
                                    else a.s
                                    if a.type == onnx.AttributeProto.STRING
                                    else str(a)
                                ),
                            )
                            for a in n.attribute
                        )
                    )
                    sig = (n.op_type, c_inputs, attrs)
                    if sig in sig_map and n.output[0] not in graph_outputs:
                        rewrites[n.output[0]] = sig_map[sig]
                        nodes_to_drop.add(n.name or n.output[0])
                        new_drops += 1
                    else:
                        sig_map[sig] = n.output[0]
            if new_drops == 0:
                break

        if not nodes_to_drop:
            return model

        rewritten_nodes: list[onnx.NodeProto] = []
        for n in graph.node:
            if (n.name or n.output[0]) in nodes_to_drop:
                continue
            new_n = copy.deepcopy(n)
            for idx, inp in enumerate(new_n.input):
                new_n.input[idx] = canonical(inp)
            rewritten_nodes.append(new_n)

        del graph.node[:]
        graph.node.extend(rewritten_nodes)
        log.info(
            "Common Subexpression Elimination: Removed %d duplicate ops",
            len(nodes_to_drop),
        )
        return self.apply_dead_code_elimination(model)

    # -------------------------------------------------------------------------
    # Rule: Dead Code Elimination (DCE)
    # -------------------------------------------------------------------------
    def apply_dead_code_elimination(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Remove unused unreachable nodes whose outputs have no consumers."""
        graph = model.graph
        graph_outputs = {o.name for o in graph.output}

        def collect_used_inputs(nodes) -> set[str]:
            used = set()
            for n in nodes:
                for inp in n.input:
                    used.add(inp)
                for attr in n.attribute:
                    if attr.HasField("g"):
                        used.update(collect_used_inputs(attr.g.node))
                    for sub_g in attr.graphs:
                        used.update(collect_used_inputs(sub_g.node))
            return used

        for _ in range(25):
            used_outputs = set(graph_outputs)
            used_outputs.update(collect_used_inputs(graph.node))

            surviving = [
                n
                for n in graph.node
                if any(out in used_outputs for out in n.output)
                or any(out in graph_outputs for out in n.output)
            ]
            if len(surviving) == len(graph.node):
                break
            del graph.node[:]
            graph.node.extend(surviving)

        return model
