"""Axial, Routing & Dynamic Tensor Arrangement Synthesizer Pass.

Synthesizes tensor re-orderings and layout transformations:
- AxialRearrange (freq_in, freq_to_time, time_to_conv, add_channel for Conformer blocks)
- TensorRoute (multi-index slicing, constant gather and dynamic padding routing)
- Unbind (Split + Squeeze decomposition into multi-output Unbind)
Attaches formal FunctionProto definitions for Netron expandable hierarchical visualization.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper

from neuron_toolkit.compiler.base import CompilerContext, Pass
from neuron_toolkit.compiler.passes.canonicalization.shape_canonicalizer import (
    ShapeCanonicalizer,
)

log = logging.getLogger("neuron_toolkit.compiler.axial_synthesizer")


class AxialSynthesizer(Pass):
    """Synthesizes tensor routes, axial rearranges, and unbind decompositions."""

    @property
    def name(self) -> str:
        return "axial_synthesizer"

    @property
    def category(self) -> str:
        return "synthesis"

    def run(self, model: onnx.ModelProto, context: CompilerContext) -> onnx.ModelProto:
        nodes_before = len(model.graph.node)

        if context.should_run("index") or context.should_run("routing"):
            model = self.apply_index_tensor_routing(model, mode=context.mode)
        if context.should_run("unbind"):
            model = self.apply_unbind_fusion(model, mode=context.mode)
        if context.should_run("axial") or context.should_run("rearrange"):
            model = self.apply_axial_rearrange_fusion(model, mode=context.mode)

        nodes_after = len(model.graph.node)
        context.record_pass(self.name, nodes_before, nodes_after)
        return model

    # -------------------------------------------------------------------------
    # Index Tensor Routing
    # -------------------------------------------------------------------------
    def apply_index_tensor_routing(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuse Gather on constant index and dynamic slice chains into TensorRoute."""
        graph = model.graph
        node_by_output: dict[str, onnx.NodeProto] = {
            out: n for n in graph.node for out in n.output
        }
        consumer_map: dict[str, list[onnx.NodeProto]] = {}
        for n in graph.node:
            for inp in n.input:
                consumer_map.setdefault(inp, []).append(n)
        graph_outputs = {o.name for o in graph.output}

        constant_values: dict[str, np.ndarray] = {
            init.name: numpy_helper.to_array(init) for init in graph.initializer
        }
        for n in graph.node:
            if n.op_type == "Constant" and len(n.output) == 1:
                for attr in n.attribute:
                    if attr.name == "value":
                        try:
                            constant_values[n.output[0]] = numpy_helper.to_array(attr.t)
                        except Exception:
                            pass

        nodes_to_remove: set[str] = set()
        new_route_nodes: list[onnx.NodeProto] = []
        route_count = 0

        for gather_node in graph.node:
            if gather_node.op_type != "Gather" or len(gather_node.input) < 2:
                continue

            idx_name = gather_node.input[1]
            if idx_name not in constant_values:
                continue
            idx_arr = constant_values[idx_name]
            if idx_arr.size != 1:
                continue
            index_val = int(idx_arr.item())

            gather_axis = 1
            for attr in gather_node.attribute:
                if attr.name == "axis":
                    gather_axis = attr.i

            gather_out = gather_node.output[0]
            consumers = consumer_map.get(gather_out, [])

            slice_candidates = [c for c in consumers if c.op_type == "Slice"]
            if slice_candidates:
                slice_node = slice_candidates[0]
                starts_name = slice_node.input[1] if len(slice_node.input) > 1 else None
                ends_name = slice_node.input[2] if len(slice_node.input) > 2 else None
                axes_name = slice_node.input[3] if len(slice_node.input) > 3 else None

                slice_axis = (
                    int(constant_values[axes_name][0])
                    if axes_name and axes_name in constant_values
                    else 1
                )
                slice_start = (
                    int(constant_values[starts_name][0])
                    if starts_name and starts_name in constant_values
                    else 0
                )
                slice_end = None

                if ends_name:
                    if ends_name in constant_values:
                        end_val = int(constant_values[ends_name][0])
                        if end_val < 9223372036854775000:
                            slice_end = end_val
                    elif ends_name in node_by_output:
                        end_node = node_by_output[ends_name]
                        sub_node = (
                            end_node
                            if end_node.op_type == "Sub"
                            else node_by_output.get(end_node.input[0])
                        )
                        if (
                            sub_node
                            and sub_node.op_type == "Sub"
                            and len(sub_node.input) >= 2
                        ):
                            pad_inp = sub_node.input[1]
                            if pad_inp in constant_values:
                                pad_val = int(constant_values[pad_inp].item())
                                slice_end = -pad_val

                route_expr = f"x[:, {index_val}, {slice_start}:{slice_end if slice_end is not None else ''}]"
                final_out = slice_node.output[0]

                route_attrs: dict[str, Any] = {
                    "axis": gather_axis,
                    "index": index_val,
                    "slice_axis": slice_axis,
                    "slice_start": slice_start,
                    "route_expr": route_expr,
                }
                if slice_end is not None:
                    route_attrs["slice_end"] = slice_end

                new_route = helper.make_node(
                    "TensorRoute",
                    inputs=[gather_node.input[0]],
                    outputs=[final_out],
                    name=f"tensor_route_{route_count}",
                    domain="custom.pytorch",
                    doc_string=f"Fused index selection and routing: {route_expr}",
                    **route_attrs,
                )
                new_route_nodes.append(new_route)
                route_count += 1

                if (
                    len(consumer_map.get(gather_out, [])) <= 1
                    and gather_out not in graph_outputs
                ):
                    nodes_to_remove.add(gather_node.name or gather_out)
                nodes_to_remove.add(slice_node.name or final_out)

                if ends_name and ends_name in node_by_output:
                    curr_n = node_by_output[ends_name]
                    while curr_n:
                        nodes_to_remove.add(curr_n.name or curr_n.output[0])
                        if curr_n.input and curr_n.input[0] in node_by_output:
                            prev = node_by_output[curr_n.input[0]]
                            if prev.op_type in (
                                "Unsqueeze",
                                "Sub",
                                "Gather",
                                "Shape",
                            ):
                                curr_n = prev
                            else:
                                break
                        else:
                            break

        if not new_route_nodes:
            return model

        kept_nodes = [
            n
            for n in graph.node
            if (n.name or (n.output[0] if n.output else "")) not in nodes_to_remove
        ]
        kept_nodes.extend(new_route_nodes)
        del graph.node[:]
        graph.node.extend(kept_nodes)

        custom_domain = "custom.pytorch"
        if not any(op.domain == custom_domain for op in model.opset_import):
            model.opset_import.append(helper.make_opsetid(custom_domain, 1))

        if mode == "function_proto":
            self._attach_tensor_route_function_def(model)

        log.info("Index Tensor Routing: Fused %d routing patterns", route_count)
        return model

    def _attach_tensor_route_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for TensorRoute."""
        func_name = "TensorRoute"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", 14)
        gather_node = helper.make_node(
            "Gather", inputs=["x", "idx"], outputs=["y"], axis=1
        )
        route_func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["x", "idx"],
            outputs=["y"],
            nodes=[gather_node],
            opset_imports=[opset_standard],
            doc_string="Tensor Index Routing layer",
        )
        model.functions.append(route_func)

    # -------------------------------------------------------------------------
    # Unbind Fusion
    # -------------------------------------------------------------------------
    def apply_unbind_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuse Split into multiple singletons followed by Squeeze nodes into a single Unbind."""
        graph = model.graph
        nodes_to_remove: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        unbind_count = 0

        for s in graph.node:
            if s.op_type == "Split" and len(s.output) > 1:
                squeezes = []
                for out in s.output:
                    c = [
                        cn
                        for cn in graph.node
                        if out in cn.input and cn.op_type == "Squeeze"
                    ]
                    if len(c) == 1:
                        squeezes.append(c[0])
                if len(squeezes) == len(s.output):
                    unbind_outputs = [sq.output[0] for sq in squeezes]
                    axis = 0
                    for a in s.attribute:
                        if a.name == "axis":
                            axis = a.i
                    unbind_node = helper.make_node(
                        "Unbind",
                        inputs=[s.input[0]],
                        outputs=unbind_outputs,
                        name=f"{s.name}_unbind",
                        domain="custom.pytorch",
                        axis=axis,
                    )
                    new_nodes.append(unbind_node)
                    nodes_to_remove.add(s.name)
                    for sq in squeezes:
                        nodes_to_remove.add(sq.name)
                    unbind_count += 1

        if not new_nodes:
            return model

        surviving = [n for n in graph.node if n.name not in nodes_to_remove] + new_nodes
        del graph.node[:]
        graph.node.extend(surviving)

        if mode == "function_proto":
            num_outputs = len(new_nodes[0].output) if new_nodes else 3
            self._attach_unbind_function_def(model, num_outputs=num_outputs)

        log.info("Unbind Fusion: Fused %d Unbind instances", unbind_count)
        return ShapeCanonicalizer().apply_dead_code_elimination(model)

    def _attach_unbind_function_def(
        self, model: onnx.ModelProto, num_outputs: int = 3
    ) -> None:
        """Attach formal ONNX FunctionProto definition for Unbind."""
        func_name = "Unbind"
        domain = "custom.pytorch"
        if any(
            f.name == func_name and f.domain == domain and len(f.output) == num_outputs
            for f in model.functions
        ):
            return
        surviving = [
            f
            for f in model.functions
            if not (f.name == func_name and f.domain == domain)
        ]
        del model.functions[:]
        model.functions.extend(surviving)

        opset_standard = helper.make_opsetid("", 14)
        s_outputs = [f"s{i}" for i in range(num_outputs)]
        y_outputs = [f"y{i}" for i in range(num_outputs)]
        sp_node = helper.make_node("Split", inputs=["x"], outputs=s_outputs, axis=0)
        sq_nodes = [
            helper.make_node("Squeeze", inputs=[s_outputs[i]], outputs=[y_outputs[i]])
            for i in range(num_outputs)
        ]
        unbind_func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["x"],
            outputs=y_outputs,
            nodes=[sp_node] + sq_nodes,
            opset_imports=[opset_standard],
            doc_string="Unbind multi-head projections along axis",
        )
        model.functions.append(unbind_func)

    # -------------------------------------------------------------------------
    # Axial Conformer Rearrange Fusion
    # -------------------------------------------------------------------------
    def apply_axial_rearrange_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuse Conformer axial rearrangements (freq_in, freq_to_time, time_to_conv, add_channel)."""
        graph = model.graph
        node_by_output: dict[str, onnx.NodeProto] = {
            out: n for n in graph.node for out in n.output
        }
        nodes_to_drop: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        rearrange_count = 0

        constant_values: dict[str, np.ndarray] = {
            init.name: numpy_helper.to_array(init) for init in graph.initializer
        }
        for n in graph.node:
            if n.op_type == "Constant" and len(n.output) == 1:
                for attr in n.attribute:
                    if attr.name == "value":
                        try:
                            constant_values[n.output[0]] = numpy_helper.to_array(attr.t)
                        except Exception:
                            pass

        for n in graph.node:
            # 1. freq_in: Transpose(perm=[0, 3, 2, 1]) -> Reshape
            if n.op_type == "Transpose":
                perm = [a.ints for a in n.attribute if a.name == "perm"]
                if perm and list(perm[0]) == [0, 3, 2, 1]:
                    consumers = [cn for cn in graph.node if n.output[0] in cn.input]
                    if len(consumers) == 1 and consumers[0].op_type == "Reshape":
                        r = consumers[0]
                        nodes_to_drop.update({n.name, r.name})
                        new_nodes.append(
                            helper.make_node(
                                "AxialRearrange",
                                inputs=[n.input[0]],
                                outputs=[r.output[0]],
                                name=f"{n.name}_fused_freq_in",
                                domain="custom.pytorch",
                                mode="freq_in",
                                batch_size=1,
                            )
                        )
                        rearrange_count += 1

            # 2. freq_to_time: Reshape -> Transpose(perm=[0, 2, 1, 3]) -> Reshape
            if n.op_type == "Transpose" and n.name not in nodes_to_drop:
                perm = [a.ints for a in n.attribute if a.name == "perm"]
                if perm and list(perm[0]) == [0, 2, 1, 3]:
                    prod = node_by_output.get(n.input[0])
                    consumers = [cn for cn in graph.node if n.output[0] in cn.input]
                    if (
                        prod
                        and prod.op_type == "Reshape"
                        and len(consumers) == 1
                        and consumers[0].op_type == "Reshape"
                    ):
                        r1 = prod
                        r2 = consumers[0]
                        nodes_to_drop.update({r1.name, n.name, r2.name})
                        new_nodes.append(
                            helper.make_node(
                                "AxialRearrange",
                                inputs=[r1.input[0]],
                                outputs=[r2.output[0]],
                                name=f"{n.name}_fused_freq_to_time",
                                domain="custom.pytorch",
                                mode="freq_to_time",
                                batch_size=1,
                            )
                        )
                        rearrange_count += 1

            # 3. time_to_conv: Reshape -> Transpose(perm=[0, 3, 1, 2])
            if n.op_type == "Transpose" and n.name not in nodes_to_drop:
                perm = [a.ints for a in n.attribute if a.name == "perm"]
                if perm and list(perm[0]) == [0, 3, 1, 2]:
                    prod = node_by_output.get(n.input[0])
                    if prod and prod.op_type == "Reshape":
                        r3 = prod
                        target_shape = []
                        if len(r3.input) > 1 and r3.input[1] in constant_values:
                            target_shape = [
                                int(v) for v in constant_values[r3.input[1]].flatten()
                            ]
                        nodes_to_drop.update({r3.name, n.name})
                        kwargs = {"mode": "time_to_conv", "batch_size": 1}
                        if target_shape:
                            kwargs["target_shape"] = target_shape
                        new_nodes.append(
                            helper.make_node(
                                "AxialRearrange",
                                inputs=[r3.input[0]],
                                outputs=[n.output[0]],
                                name=f"{n.name}_fused_time_to_conv",
                                domain="custom.pytorch",
                                **kwargs,
                            )
                        )
                        rearrange_count += 1

            # 4. flatten_fc: Transpose(perm=[0, 3, 1, 2]) -> Reshape
            if n.op_type == "Transpose" and n.name not in nodes_to_drop:
                perm = [a.ints for a in n.attribute if a.name == "perm"]
                if perm and list(perm[0]) == [0, 3, 1, 2]:
                    consumers = [cn for cn in graph.node if n.output[0] in cn.input]
                    if len(consumers) == 1 and consumers[0].op_type == "Reshape":
                        r = consumers[0]
                        nodes_to_drop.update({n.name, r.name})
                        new_nodes.append(
                            helper.make_node(
                                "AxialRearrange",
                                inputs=[n.input[0]],
                                outputs=[r.output[0]],
                                name=f"{n.name}_fused_flatten_fc",
                                domain="custom.pytorch",
                                mode="flatten_fc",
                                batch_size=1,
                            )
                        )
                        rearrange_count += 1

            # 5. add_channel: Reshape inserting channel dimension before Conv
            if n.op_type == "Reshape" and n.name not in nodes_to_drop:
                consumers = [cn for cn in graph.node if n.output[0] in cn.input]
                prod = node_by_output.get(n.input[0])
                if prod and any(
                    c.op_type in ("Conv", "ConvTranspose") for c in consumers
                ):
                    shape_node = node_by_output.get(n.input[1])
                    if (
                        shape_node
                        and shape_node.op_type == "Concat"
                        and len(shape_node.input) == 4
                    ):
                        dim1_const = constant_values.get(shape_node.input[1])
                        if (
                            dim1_const is not None
                            and dim1_const.size == 1
                            and int(dim1_const.item()) == 1
                        ):
                            nodes_to_drop.add(n.name)
                            new_nodes.append(
                                helper.make_node(
                                    "AxialRearrange",
                                    inputs=[n.input[0]],
                                    outputs=[n.output[0]],
                                    name=f"{n.name}_fused_add_channel",
                                    domain="custom.pytorch",
                                    mode="add_channel",
                                    batch_size=1,
                                )
                            )
                            rearrange_count += 1

        if not new_nodes:
            return model

        surviving = [
            node for node in graph.node if node.name not in nodes_to_drop
        ] + new_nodes
        del graph.node[:]
        graph.node.extend(surviving)

        custom_domain = "custom.pytorch"
        if not any(op.domain == custom_domain for op in model.opset_import):
            model.opset_import.append(helper.make_opsetid(custom_domain, 1))

        if mode == "function_proto":
            self._attach_axial_rearrange_function_def(model)

        log.info(
            "AxialRearrange Fusion: Fused %d AxialRearrange instances", rearrange_count
        )
        return ShapeCanonicalizer().apply_dead_code_elimination(model)

    def _attach_axial_rearrange_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for AxialRearrange."""
        func_name = "AxialRearrange"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        model_opset = 14
        for op in model.opset_import:
            if op.domain in ("", "ai.onnx"):
                model_opset = op.version
        opset_standard = helper.make_opsetid("", model_opset)
        id_node = helper.make_node("Identity", inputs=["x"], outputs=["y"])
        rearrange_func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["x"],
            outputs=["y"],
            nodes=[id_node],
            opset_imports=[opset_standard],
            doc_string="Axial conformer rearrangement function (freq_in, freq_to_time, time_to_conv, add_channel)",
        )
        model.functions.append(rearrange_func)
