"""Compile-Time Static Subgraph Evaluator & Constant Folding Pass.

Automatically identifies and evaluates compile-time static generator subgraphs:
- Static mathematical sub-DAGs (e.g. RoPE theta tables, anchor grids, position frequency tensors)
- Redundant cache-update ScatterND operations where data is already precomputed
- Pure constant-dependent expression trees evaluated into static Initializers

Zero hardcoded node-name heuristics: relies strictly on DAG topological reachability and algebraic identity.
"""

from __future__ import annotations

import collections
import copy
import logging
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper

try:
    from onnx.reference import ReferenceEvaluator
except ImportError:
    ReferenceEvaluator = None

from neuron_toolkit.compiler.base import CompilerContext, Pass
from neuron_toolkit.compiler.passes.canonicalization.shape_canonicalizer import (
    ShapeCanonicalizer,
)

log = logging.getLogger("neuron_toolkit.compiler.static_folder")


class StaticFolder(Pass):
    """Evaluates static generator subgraphs compile-time and prunes redundant scatter-updates."""

    @property
    def name(self) -> str:
        return "static_folder"

    @property
    def category(self) -> str:
        return "canonicalization"

    def run(self, model: onnx.ModelProto, context: CompilerContext) -> onnx.ModelProto:
        nodes_before = len(model.graph.node)

        if context.should_run("static_folder") or context.should_run("constant_folding"):
            model = self.apply_identity_scatter_nd_folding(model)
            model = self.apply_static_subgraph_folding(model)

        nodes_after = len(model.graph.node)
        context.record_pass(self.name, nodes_before, nodes_after)
        return model

    # -------------------------------------------------------------------------
    # Identity ScatterND Folding
    # -------------------------------------------------------------------------
    def apply_identity_scatter_nd_folding(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fold ScatterND(data, indices, updates) where data is static and already contains updates.
        
        In models with cached positional buffers (e.g. RoPE frequency tables),
        forward passes often scatter updates back into a static precomputed cache.
        When updates match data[indices], the operation is an exact identity write.
        Rewiring consumers directly to data breaks cycles and eliminates the generator DAG.
        """
        graph = model.graph
        init_arrays: dict[str, np.ndarray] = {
            i.name: numpy_helper.to_array(i) for i in graph.initializer
        }
        for n in graph.node:
            if n.op_type == "Constant" and len(n.output) == 1:
                for a in n.attribute:
                    if a.name == "value" and a.HasField("t"):
                        try:
                            init_arrays[n.output[0]] = numpy_helper.to_array(a.t)
                        except Exception:
                            pass

        scatter_nodes = [n for n in graph.node if n.op_type == "ScatterND" and len(n.input) >= 3]
        if not scatter_nodes:
            return model

        rewired_outputs: dict[str, str] = {}
        for sn in scatter_nodes:
            data_name = sn.input[0]
            indices_name = sn.input[1]
            updates_name = sn.input[2]
            # Resolve through previous rewires if chained
            while data_name in rewired_outputs:
                data_name = rewired_outputs[data_name]
            while indices_name in rewired_outputs:
                indices_name = rewired_outputs[indices_name]
            while updates_name in rewired_outputs:
                updates_name = rewired_outputs[updates_name]

            if data_name not in init_arrays or indices_name not in init_arrays or updates_name not in init_arrays:
                continue

            data_arr = init_arrays[data_name]
            indices_arr = init_arrays[indices_name]
            updates_arr = init_arrays[updates_name]

            try:
                idx_tuple = tuple(indices_arr[..., i] for i in range(indices_arr.shape[-1]))
                target_slice = data_arr[idx_tuple]
                if not np.array_equal(target_slice, updates_arr):
                    continue
            except Exception:
                continue

            out_name = sn.output[0]
            # When ScatterND updates data with matching frequencies or self-identity,
            # consumers (RotaryEmbedding, Slice) slice up to seq_len.
            rewired_outputs[out_name] = data_name
            log.info(
                "Identity ScatterND Fold: Rewired %r -> %r (shape=%s)",
                out_name,
                data_name,
                data_arr.shape,
            )

        if not rewired_outputs:
            return model

        # Rewire all consumers
        for n in graph.node:
            for i in range(len(n.input)):
                inp = n.input[i]
                if inp in rewired_outputs:
                    n.input[i] = rewired_outputs[inp]

        # Run Dead Code Elimination to prune now-dead generator DAGs
        shape_pass = ShapeCanonicalizer()
        prev_count = len(graph.node)
        while True:
            model = shape_pass.apply_dead_code_elimination(model)
            if len(model.graph.node) == prev_count:
                break
            prev_count = len(model.graph.node)

        return model

    # -------------------------------------------------------------------------
    # Pure Static Subgraph Folding
    # -------------------------------------------------------------------------
    def apply_static_subgraph_folding(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Evaluate nodes whose inputs are all transitively static (constants/initializers)."""
        graph = model.graph
        init_names = {i.name for i in graph.initializer}
        graph_input_names = {i.name for i in graph.input if i.name not in init_names}
        graph_output_names = {o.name for o in graph.output}

        # Track static tensors
        static_tensors: set[str] = set(init_names)
        node_by_output: dict[str, onnx.NodeProto] = {}
        for n in graph.node:
            for out in n.output:
                if out:
                    node_by_output[out] = n
            if n.op_type == "Constant":
                for out in n.output:
                    if out:
                        static_tensors.add(out)

        # Propagate static property
        changed = True
        while changed:
            changed = False
            for n in graph.node:
                if all(out in static_tensors for out in n.output if out):
                    continue
                # If all inputs are static and non-empty
                if n.input and all(inp in static_tensors for inp in n.input if inp):
                    for out in n.output:
                        if out and out not in static_tensors:
                            static_tensors.add(out)
                            changed = True

        # Now identify static nodes that can be folded into initializers
        # (Exclude initializers themselves and nodes whose outputs are graph outputs)
        foldable_nodes: list[onnx.NodeProto] = []
        for n in graph.node:
            if n.op_type == "Constant":
                continue
            if all(out in static_tensors for out in n.output if out):
                # Only fold if none of outputs are graph outputs
                if not any(out in graph_output_names for out in n.output if out):
                    foldable_nodes.append(n)

        if not foldable_nodes:
            return model

        # Collect static array values for initializers and constants
        static_values: dict[str, np.ndarray] = {
            i.name: numpy_helper.to_array(i) for i in graph.initializer
        }
        for n in graph.node:
            if n.op_type == "Constant":
                for attr in n.attribute:
                    if attr.name == "value" and n.output:
                        try:
                            static_values[n.output[0]] = numpy_helper.to_array(attr.t)
                        except Exception:
                            pass

        folded_nodes_set: set[int] = set()
        for n in foldable_nodes:
            if not n.input or not all(inp in static_values for inp in n.input if inp):
                continue
            try:
                # Try evaluating via ReferenceEvaluator
                sub_inputs = [
                    helper.make_tensor_value_info(
                        inp,
                        onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[static_values[inp].dtype],
                        static_values[inp].shape,
                    )
                    for inp in n.input
                    if inp
                ]
                sub_outputs = [
                    helper.make_tensor_value_info(out, onnx.TensorProto.FLOAT, None)
                    for out in n.output
                    if out
                ]
                sub_graph = helper.make_graph([n], "sub", sub_inputs, sub_outputs)
                sub_model = helper.make_model(sub_graph, opset_imports=model.opset_import)
                sess = ReferenceEvaluator(sub_model)
                in_tensors = {inp: static_values[inp] for inp in n.input if inp}
                res = sess.run(None, in_tensors)
                for out_name, out_arr in zip(n.output, res):
                    if out_name:
                        static_values[out_name] = out_arr
                        graph.initializer.append(numpy_helper.from_array(out_arr, name=out_name))
                folded_nodes_set.add(id(n))
            except Exception:
                # Fallback to direct numpy arithmetic for basic ops
                try:
                    if n.op_type == "Add" and len(n.input) == 2:
                        val = static_values[n.input[0]] + static_values[n.input[1]]
                    elif n.op_type == "Sub" and len(n.input) == 2:
                        val = static_values[n.input[0]] - static_values[n.input[1]]
                    elif n.op_type == "Mul" and len(n.input) == 2:
                        val = static_values[n.input[0]] * static_values[n.input[1]]
                    elif n.op_type == "Div" and len(n.input) == 2:
                        val = static_values[n.input[0]] / static_values[n.input[1]]
                    else:
                        continue
                    out_name = n.output[0]
                    static_values[out_name] = val
                    graph.initializer.append(numpy_helper.from_array(val, name=out_name))
                    folded_nodes_set.add(id(n))
                except Exception:
                    pass

        if folded_nodes_set:
            remaining_nodes = [n for n in graph.node if id(n) not in folded_nodes_set]
            del graph.node[:]
            graph.node.extend(remaining_nodes)

        # Prune via DCE after folding constants
        shape_pass = ShapeCanonicalizer()
        model = shape_pass.apply_dead_code_elimination(model)
        return model
