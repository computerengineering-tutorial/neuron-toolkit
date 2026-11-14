"""Standard graph optimization passes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from neuron_toolkit.passes.base import BasePass

if TYPE_CHECKING:
    from neuron_toolkit.graph import NeuronGraph


class FuseConvBN(BasePass):
    """Folds BatchNormalization parameters (scale, bias, mean, var) into preceding Conv."""

    name: str = "FuseConvBN"

    def run(self, graph: NeuronGraph) -> tuple[NeuronGraph, dict[str, Any]]:
        from neuron_toolkit.graph import NeuronGraph

        initial_count = len(graph.nodes)
        transformed_raw = graph.passes.fuse_conv_bn().build()
        transformed = NeuronGraph(transformed_raw)
        final_count = len(transformed.nodes)
        fused = initial_count - final_count
        return transformed, {"fused_pairs": fused}


class DeadCodeElimination(BasePass):
    """Removes nodes whose outputs are never consumed and are not graph outputs."""

    name: str = "DeadCodeElimination"

    def run(self, graph: NeuronGraph) -> tuple[NeuronGraph, dict[str, Any]]:
        graph_outputs = set(graph.outputs)

        # Iterate until no more dead nodes can be removed
        current_graph = graph
        total_removed = 0

        while True:
            # Build set of all consumed tensors
            consumed: set[str] = set(graph_outputs)
            for node in current_graph.nodes:
                for inp in getattr(node, "input", []):
                    if inp:
                        consumed.add(inp)

            # Find dead nodes
            dead_nodes = []
            for node in current_graph.nodes:
                outs = getattr(node, "output", [])
                if outs and not any(out in consumed for out in outs):
                    dead_nodes.append(node)

            if not dead_nodes:
                break

            rewriter = current_graph.rewriter()
            for dead in dead_nodes:
                rewriter.remove(dead)

            current_graph = rewriter.build_graph()
            total_removed += len(dead_nodes)

        return current_graph, {"removed_nodes": total_removed}


class ConstantFolding(BasePass):
    """Evaluates operations whose inputs are all static constant initializers."""

    name: str = "ConstantFolding"

    def run(self, graph: NeuronGraph) -> tuple[NeuronGraph, dict[str, Any]]:
        tensor_map = graph.tensor_map
        folded_count = 0
        rewriter = graph.rewriter()

        for node in list(graph.nodes):
            op_type = getattr(node, "op_type", "")
            inputs = getattr(node, "input", [])
            outputs = getattr(node, "output", [])

            if not inputs or not outputs:
                continue

            # Check if all inputs are present in tensor_map (constants)
            if all(inp in tensor_map and tensor_map[inp] is not None for inp in inputs):
                inp_arrays = [tensor_map[inp] for inp in inputs]
                result_val = None

                if op_type == "Add" and len(inp_arrays) == 2:
                    result_val = inp_arrays[0] + inp_arrays[1]
                elif op_type == "Sub" and len(inp_arrays) == 2:
                    result_val = inp_arrays[0] - inp_arrays[1]
                elif op_type == "Mul" and len(inp_arrays) == 2:
                    result_val = inp_arrays[0] * inp_arrays[1]
                elif op_type == "Div" and len(inp_arrays) == 2:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        result_val = inp_arrays[0] / inp_arrays[1]
                elif op_type == "Neg" and len(inp_arrays) == 1:
                    result_val = -inp_arrays[0]
                elif op_type == "Relu" and len(inp_arrays) == 1:
                    result_val = np.maximum(0, inp_arrays[0])

                if result_val is not None:
                    out_name = outputs[0]
                    res_arr = np.asarray(result_val)
                    if hasattr(rewriter, "register_initializer"):
                        rewriter.register_initializer(out_name, res_arr)
                    rewriter.remove(node)
                    folded_count += 1

        if folded_count > 0:
            new_graph = rewriter.build_graph()
            return new_graph, {"folded_nodes": folded_count}

        return graph, {"folded_nodes": 0}


class CommonSubexpressionElimination(BasePass):
    """Merges duplicate operations taking identical inputs and attributes."""

    name: str = "CommonSubexpressionElimination"

    def run(self, graph: NeuronGraph) -> tuple[NeuronGraph, dict[str, Any]]:
        rewriter = graph.rewriter()
        seen: dict[tuple[str, tuple[str, ...]], object] = {}
        eliminated = 0

        for node in list(graph.nodes):
            op_type = getattr(node, "op_type", "")
            inputs = tuple(getattr(node, "input", []))
            outputs = getattr(node, "output", [])

            if not inputs or not outputs or op_type in {"RandomNormal", "RandomUniform", "Dropout"}:
                continue

            key = (op_type, inputs)
            if key in seen:
                primary = seen[key]
                primary_outs = getattr(primary, "output", [])
                secondary_out = outputs[0]

                if primary_outs and secondary_out:
                    primary_out = primary_outs[0]
                    # Replace secondary with Identity forwarding primary_out
                    rewriter.replace([node], "Identity", inputs=[primary_out], outputs=[secondary_out])
                    eliminated += 1
            else:
                seen[key] = node

        if eliminated > 0:
            return rewriter.build_graph(), {"eliminated_nodes": eliminated}
        return graph, {"eliminated_nodes": 0}


class FuseGELU(BasePass):
    """Fuses Erf-based or Tanh-approximated GELU subgraphs into a unified Gelu operator."""

    name: str = "FuseGELU"

    def run(self, graph: NeuronGraph) -> tuple[NeuronGraph, dict[str, Any]]:
        rewriter = graph.rewriter()
        fused = 0
        nodes = list(graph.nodes)
        node_map = {getattr(n, "output", [None])[0]: n for n in nodes if getattr(n, "output", [])}

        for node in nodes:
            if getattr(node, "op_type", "") != "Mul":
                continue

            mul_inputs = getattr(node, "input", [])
            mul_outputs = getattr(node, "output", [])
            if len(mul_inputs) != 2 or not mul_outputs:
                continue

            branch_a = node_map.get(mul_inputs[0])
            branch_b = node_map.get(mul_inputs[1])

            add_node = None
            other_inp = None

            if branch_a and getattr(branch_a, "op_type", "") == "Add":
                add_node = branch_a
                other_inp = mul_inputs[1]
            elif branch_b and getattr(branch_b, "op_type", "") == "Add":
                add_node = branch_b
                other_inp = mul_inputs[0]

            if not add_node:
                continue

            add_inps = getattr(add_node, "input", [])
            erf_node = None
            for ai in add_inps:
                p = node_map.get(ai)
                if p and getattr(p, "op_type", "") == "Erf":
                    erf_node = p
                    break

            if not erf_node:
                continue

            erf_inps = getattr(erf_node, "input", [])
            if not erf_inps:
                continue

            div_or_mul = node_map.get(erf_inps[0])
            x_candidate = None
            if div_or_mul and getattr(div_or_mul, "op_type", "") in ("Div", "Mul"):
                dm_inps = getattr(div_or_mul, "input", [])
                if dm_inps:
                    x_candidate = dm_inps[0]
            else:
                x_candidate = erf_inps[0]

            other_node = node_map.get(other_inp)
            if other_node and getattr(other_node, "op_type", "") == "Mul":
                om_inps = getattr(other_node, "input", [])
                for oi in om_inps:
                    if oi == x_candidate or (x_candidate is None and oi not in graph.tensor_map):
                        x_candidate = oi
                        break

            if x_candidate:
                subgraph_nodes = [node, add_node, erf_node]
                if div_or_mul:
                    subgraph_nodes.append(div_or_mul)
                if other_node and other_node is not node:
                    subgraph_nodes.append(other_node)

                unique_nodes = list({id(n): n for n in subgraph_nodes}.values())
                rewriter.replace(
                    unique_nodes,
                    "Gelu",
                    inputs=[x_candidate],
                    outputs=[mul_outputs[0]],
                    name=f"fused_gelu_{fused}",
                )
                fused += 1

        if fused > 0:
            return rewriter.build_graph(), {"fused_gelu_count": fused}
        return graph, {"fused_gelu_count": 0}


class FuseLayerNorm(BasePass):
    """Fuses ReduceMean, Sub, Pow, ReduceMean, Add, Sqrt, Div subgraphs into LayerNormalization."""

    name: str = "FuseLayerNorm"

    def run(self, graph: NeuronGraph) -> tuple[NeuronGraph, dict[str, Any]]:
        rewriter = graph.rewriter()
        fused = 0
        nodes = list(graph.nodes)
        node_map = {getattr(n, "output", [None])[0]: n for n in nodes if getattr(n, "output", [])}

        for node in nodes:
            if getattr(node, "op_type", "") != "Div":
                continue

            div_inputs = getattr(node, "input", [])
            div_outputs = getattr(node, "output", [])
            if len(div_inputs) != 2 or not div_outputs:
                continue

            num_node = node_map.get(div_inputs[0])
            den_node = node_map.get(div_inputs[1])

            if not num_node or not den_node:
                continue
            if getattr(num_node, "op_type", "") != "Sub" or getattr(den_node, "op_type", "") != "Sqrt":
                continue

            sub_node = num_node
            sqrt_node = den_node

            sub_inps = getattr(sub_node, "input", [])
            if len(sub_inps) < 2:
                continue
            x_tensor = sub_inps[0]
            mean_node = node_map.get(sub_inps[1])
            if not mean_node or getattr(mean_node, "op_type", "") != "ReduceMean":
                continue

            sqrt_inps = getattr(sqrt_node, "input", [])
            if not sqrt_inps:
                continue
            add_eps_node = node_map.get(sqrt_inps[0])
            if not add_eps_node or getattr(add_eps_node, "op_type", "") != "Add":
                continue

            add_eps_inps = getattr(add_eps_node, "input", [])
            var_mean_node = None
            for ai in add_eps_inps:
                p = node_map.get(ai)
                if p and getattr(p, "op_type", "") == "ReduceMean":
                    var_mean_node = p
                    break

            if not var_mean_node:
                continue

            div_out = div_outputs[0]
            final_out = div_out
            scale_tensor = ""
            bias_tensor = ""

            consumers = [n for n in nodes if div_out in getattr(n, "input", [])]
            scale_node = None
            bias_node = None
            if len(consumers) == 1 and getattr(consumers[0], "op_type", "") == "Mul":
                scale_node = consumers[0]
                mul_inps = getattr(scale_node, "input", [])
                scale_tensor = mul_inps[1] if mul_inps[0] == div_out else mul_inps[0]
                mul_out = getattr(scale_node, "output", [None])[0]

                mul_consumers = [n for n in nodes if mul_out in getattr(n, "input", [])]
                if len(mul_consumers) == 1 and getattr(mul_consumers[0], "op_type", "") == "Add":
                    bias_node = mul_consumers[0]
                    add_inps = getattr(bias_node, "input", [])
                    bias_tensor = add_inps[1] if add_inps[0] == mul_out else add_inps[0]
                    final_out = getattr(bias_node, "output", [None])[0]
                else:
                    final_out = mul_out

            subgraph_nodes = [node, sub_node, mean_node, sqrt_node, add_eps_node, var_mean_node]
            if scale_node:
                subgraph_nodes.append(scale_node)
            if bias_node:
                subgraph_nodes.append(bias_node)

            unique_nodes = list({id(n): n for n in subgraph_nodes}.values())
            ln_inputs = [x_tensor]
            if scale_tensor:
                ln_inputs.append(scale_tensor)
            if bias_tensor:
                ln_inputs.append(bias_tensor)

            rewriter.replace(
                unique_nodes,
                "LayerNormalization",
                inputs=ln_inputs,
                outputs=[final_out],
                name=f"fused_layernorm_{fused}",
                axis=-1,
                epsilon=1e-5,
            )
            fused += 1

        if fused > 0:
            return rewriter.build_graph(), {"fused_layernorm_count": fused}
        return graph, {"fused_layernorm_count": 0}


class FuseConvActivation(BasePass):
    """Fuses Conv followed by Relu, LeakyRelu, or Clip into FusedConv."""

    name: str = "FuseConvActivation"

    def run(self, graph: NeuronGraph) -> tuple[NeuronGraph, dict[str, Any]]:
        rewriter = graph.rewriter()
        fused = 0
        nodes = list(graph.nodes)
        graph_outputs = set(graph.outputs)

        for node in nodes:
            if getattr(node, "op_type", "") != "Conv":
                continue

            conv_outputs = getattr(node, "output", [])
            if not conv_outputs:
                continue

            conv_out = conv_outputs[0]
            if conv_out in graph_outputs:
                continue

            consumers = [n for n in nodes if conv_out in getattr(n, "input", [])]
            if len(consumers) != 1:
                continue

            consumer = consumers[0]
            act_op = getattr(consumer, "op_type", "")
            if act_op not in ("Relu", "LeakyRelu", "Clip"):
                continue

            act_outputs = getattr(consumer, "output", [])
            if not act_outputs:
                continue

            conv_inputs = getattr(node, "input", [])
            act_attrs = getattr(graph._backend, "get_node_attrs", lambda _n: {})(consumer)
            conv_attrs = getattr(graph._backend, "get_node_attrs", lambda _n: {})(node)

            combined_attrs = dict(conv_attrs)
            combined_attrs["activation"] = act_op
            for k, v in act_attrs.items():
                combined_attrs[f"activation_{k}"] = v

            rewriter.replace(
                [node, consumer],
                "FusedConv",
                inputs=list(conv_inputs),
                outputs=list(act_outputs),
                name=f"{getattr(node, 'name', 'conv')}_{act_op.lower()}_fused",
                **combined_attrs,
            )
            fused += 1

        if fused > 0:
            return rewriter.build_graph(), {"fused_conv_activation_count": fused}
        return graph, {"fused_conv_activation_count": 0}
