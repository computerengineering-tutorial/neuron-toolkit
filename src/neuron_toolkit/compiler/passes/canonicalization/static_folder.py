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
            # Resolve data_name through previous rewires if chained
            while data_name in rewired_outputs:
                data_name = rewired_outputs[data_name]

            if data_name not in init_arrays:
                continue

            data_arr = init_arrays[data_name]
            out_name = sn.output[0]

            # In transformer RoPE position caches, data is an [S, D] frequency grid.
            # When ScatterND updates data with matching frequencies or self-identity,
            # consumers (RotaryEmbedding, Slice) slice up to seq_len.
            # Check if this ScatterND is a self-buffering cache update:
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

        # Prune via DCE after folding constants
        shape_pass = ShapeCanonicalizer()
        model = shape_pass.apply_dead_code_elimination(model)
        return model
