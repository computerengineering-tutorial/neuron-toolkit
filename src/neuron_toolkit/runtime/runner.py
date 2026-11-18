"""PyTorch Graph Execution Engine for ONNX Models with Custom Ops."""

from __future__ import annotations

import collections
import copy
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import onnx
import torch
from onnx import numpy_helper

from neuron_toolkit.runtime.registry import OpRegistry, global_registry

if TYPE_CHECKING:
    from neuron_toolkit.graph import NeuronGraph

log = logging.getLogger("neuron_toolkit.runtime.runner")


def toposort_graph(model: onnx.ModelProto) -> list[onnx.NodeProto]:
    """Sort graph nodes topologically according to tensor dependencies."""
    nodes = list(model.graph.node)
    node_map = {id(n): n for n in nodes}
    out_to_node_id = {out: id(n) for n in nodes for out in n.output if out}

    dep_graph = collections.defaultdict(list)
    in_degree = collections.defaultdict(int)

    for n in nodes:
        nid = id(n)
        dep_count = 0
        for inp in n.input:
            if inp and inp in out_to_node_id:
                dep_graph[out_to_node_id[inp]].append(nid)
                dep_count += 1
        in_degree[nid] = dep_count

    queue = collections.deque([nid for nid in node_map if in_degree[nid] == 0])
    sorted_nodes: list[onnx.NodeProto] = []
    while queue:
        nid = queue.popleft()
        sorted_nodes.append(node_map[nid])
        for nxt in dep_graph[nid]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)

    # In case of disconnected or cyclic components, append remaining nodes
    if len(sorted_nodes) < len(nodes):
        seen = {id(n) for n in sorted_nodes}
        for n in nodes:
            if id(n) not in seen:
                sorted_nodes.append(n)

    return sorted_nodes


class PyTorchRunner:
    """Vectorized, native PyTorch execution runtime for standard and rewritten ONNX graphs."""

    def __init__(
        self,
        model_or_graph: onnx.ModelProto | NeuronGraph,
        registry: OpRegistry | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize runner with an ONNX model or NeuronGraph."""
        if hasattr(model_or_graph, "_backend") and hasattr(
            model_or_graph._backend, "model"
        ):
            self.model: onnx.ModelProto = copy.deepcopy(model_or_graph._backend.model)
        elif isinstance(model_or_graph, onnx.ModelProto):
            self.model = copy.deepcopy(model_or_graph)
        else:
            msg = f"Unsupported input type for PyTorchRunner: {type(model_or_graph)}"
            raise TypeError(msg)

        self.registry = registry or global_registry
        self.device = torch.device(device)

        # Topological ordering
        self.sorted_nodes = toposort_graph(self.model)

        # Convert initializers to PyTorch tensors
        self.initializers: dict[str, torch.Tensor] = {}
        for init in self.model.graph.initializer:
            try:
                arr = numpy_helper.to_array(init)
                self.initializers[init.name] = torch.from_numpy(arr.copy()).to(
                    self.device
                )
            except Exception as e:
                log.warning("Could not convert initializer %r: %s", init.name, e)

        # Catalog input and output names
        init_names = set(self.initializers.keys())
        self.input_names = [
            inp.name for inp in self.model.graph.input if inp.name not in init_names
        ]
        self.output_names = [out.name for out in self.model.graph.output]

        # Catalog Model Functions
        self.functions: dict[tuple[str, str], onnx.FunctionProto] = {
            (f.domain, f.name): f for f in self.model.functions
        }

        # Precompile execution plan: pre-resolve handlers and attributes once
        self._execution_plan: list[
            tuple[Any, list[str], list[str], dict[str, Any], Any, str]
        ] = []
        for node in self.sorted_nodes:
            op_type = node.op_type
            domain = node.domain or ""
            attrs = self._extract_attributes(node)
            handler = self.registry.get(domain, op_type)
            fn_proto = (
                self.functions.get((domain, op_type)) if handler is None else None
            )
            op_desc = f"{domain}::{op_type}" if domain else op_type
            self._execution_plan.append(
                (handler, list(node.input), list(node.output), attrs, fn_proto, op_desc)
            )

    def _extract_attributes(self, node: onnx.NodeProto) -> dict[str, Any]:
        """Convert ONNX NodeProto attributes to a Python dict."""
        attrs: dict[str, Any] = {}
        for attr in node.attribute:
            if attr.HasField("f"):
                attrs[attr.name] = attr.f
            elif attr.HasField("i"):
                attrs[attr.name] = attr.i
            elif attr.HasField("s"):
                attrs[attr.name] = attr.s.decode("utf-8", errors="ignore")
            elif attr.HasField("t"):
                attrs[attr.name] = numpy_helper.to_array(attr.t)
            elif attr.floats:
                attrs[attr.name] = list(attr.floats)
            elif attr.ints:
                attrs[attr.name] = list(attr.ints)
            elif attr.strings:
                attrs[attr.name] = [
                    s.decode("utf-8", errors="ignore") for s in attr.strings
                ]
        return attrs

    def run(
        self,
        inputs: dict[str, np.ndarray | torch.Tensor],
    ) -> dict[str, np.ndarray]:
        """Execute the graph using PyTorch and return a dictionary of numpy array outputs."""
        tensor_env: dict[str, torch.Tensor] = dict(self.initializers)

        # Populate user inputs
        for name, val in inputs.items():
            if isinstance(val, torch.Tensor):
                tensor_env[name] = val.to(self.device)
            else:
                tensor_env[name] = torch.from_numpy(np.asarray(val)).to(self.device)

        with torch.no_grad():
            for (
                handler,
                node_inputs_names,
                node_outputs_names,
                attrs,
                fn_proto,
                op_desc,
            ) in self._execution_plan:
                # Fast input extraction
                node_inputs: list[torch.Tensor | None] = []
                for inp_name in node_inputs_names:
                    if not inp_name:
                        node_inputs.append(None)
                    elif inp_name in tensor_env:
                        node_inputs.append(tensor_env[inp_name])
                    else:
                        msg = f"Operation requires missing input {inp_name!r}"
                        raise RuntimeError(msg)

                # 1. Check registry handler
                if handler is not None:
                    out = handler(node_inputs, attrs)
                    if isinstance(out, (list, tuple)):
                        for idx, out_tensor in enumerate(out):
                            if (
                                idx < len(node_outputs_names)
                                and node_outputs_names[idx]
                            ):
                                tensor_env[node_outputs_names[idx]] = out_tensor
                    elif len(node_outputs_names) >= 1:
                        tensor_env[node_outputs_names[0]] = out
                    continue

                # 2. Check if it's a FunctionProto in model.functions
                if fn_proto is not None:
                    fn_outputs = self._run_function(
                        fn_proto, node_inputs, attrs, parent_env=tensor_env
                    )
                    for idx, out_tensor in enumerate(fn_outputs):
                        if idx < len(node_outputs_names) and node_outputs_names[idx]:
                            tensor_env[node_outputs_names[idx]] = out_tensor
                    continue

                # 3. Op not found
                msg = f"No PyTorch runtime handler registered for operation {op_desc!r}"
                raise NotImplementedError(msg)

        # Collect outputs
        results: dict[str, np.ndarray] = {}
        for out_name in self.output_names:
            if out_name in tensor_env:
                t = tensor_env[out_name]
                results[out_name] = t.detach().cpu().numpy()
            else:
                msg = f"Graph output {out_name!r} was not computed during forward pass."
                raise RuntimeError(msg)

        return results

    def _run_function(
        self,
        fn: onnx.FunctionProto,
        inputs: list[torch.Tensor],
        attrs: dict[str, Any],
        parent_env: dict[str, torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        """Execute an ONNX FunctionProto using the PyTorch registry."""
        fn_env: dict[str, torch.Tensor] = dict(parent_env or self.initializers)
        for formal_inp, actual_tensor in zip(fn.input, inputs, strict=False):
            if actual_tensor is not None:
                fn_env[formal_inp] = actual_tensor

        for node in fn.node:
            op_type = node.op_type
            domain = node.domain or ""
            node_inputs = [fn_env.get(inp) for inp in node.input]
            node_attrs = self._extract_attributes(node)

            handler = self.registry.get(domain, op_type)
            if handler is None:
                msg = f"Inside Function {fn.name}: no handler for op {op_type!r}"
                raise NotImplementedError(msg)

            out = handler(node_inputs, node_attrs)
            if isinstance(out, (list, tuple)):
                for idx, out_tensor in enumerate(out):
                    if idx < len(node.output):
                        fn_env[node.output[idx]] = out_tensor
            elif len(node.output) >= 1:
                fn_env[node.output[0]] = out

        return [fn_env[formal_out] for formal_out in fn.output if formal_out in fn_env]
