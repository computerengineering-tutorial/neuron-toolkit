"""Declarative JSON Pattern Rules Engine for ONNX Graph Transformation."""

from __future__ import annotations

import collections
import copy
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import onnx
from onnx import helper, numpy_helper

from neuron_toolkit.rules.schema import JSONRule

if TYPE_CHECKING:
    from neuron_toolkit.graph import NeuronGraph

log = logging.getLogger("neuron_toolkit.rules.engine")


class JSONRuleEngine:
    """Engine that loads declarative JSON rewrite rules and transforms ONNX models."""

    def __init__(self, mode: str = "function_proto") -> None:
        """Initialize engine.

        Args:
            mode: 'function_proto' (dual-mode with FunctionProto attached) or
                  'plain_custom_op' (minimalist custom op node without function body).
        """
        self.mode = mode
        self.rules: list[JSONRule] = []

    def load_rule_from_file(self, path: str | Path) -> JSONRule:
        """Load and register a single JSON rule file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        rule = JSONRule.from_dict(data)
        self.rules.append(rule)
        return rule

    def load_rules_from_dir(self, directory: str | Path | None = None) -> list[JSONRule]:
        """Load all JSON rules from a directory (defaults to built-in definitions)."""
        if directory is None:
            directory = Path(__file__).parent / "definitions"
        else:
            directory = Path(directory)

        loaded: list[JSONRule] = []
        for json_path in sorted(directory.glob("*.json")):
            loaded.append(self.load_rule_from_file(json_path))
        return loaded

    def apply(
        self,
        target: onnx.ModelProto | NeuronGraph,
        rule_ids: list[str] | None = None,
    ) -> onnx.ModelProto:
        """Apply loaded rules to the model.

        Returns an optimized/simplified ONNX ModelProto.
        """
        if hasattr(target, "_backend") and hasattr(target._backend, "model"):
            model = copy.deepcopy(target._backend.model)
        elif isinstance(target, onnx.ModelProto):
            model = copy.deepcopy(target)
        else:
            msg = f"Unsupported target type for JSONRuleEngine: {type(target)}"
            raise TypeError(msg)

        # Check which rules to run
        active_rules = self.rules
        if rule_ids is not None:
            active_rules = [r for r in self.rules if r.rule_id in rule_ids]

        # Apply specific rule implementations based on registered rules
        applied_names = [r.rule_id for r in active_rules]

        if any("cast" in r.rule_id for r in active_rules) or not active_rules:
            model = self.apply_cast_initializer_folding(model)

        if any("erf_gelu" in r.rule_id for r in active_rules) or not active_rules:
            model = self.apply_erf_gelu_fusion(model)

        if any("rmsnorm" in r.rule_id for r in active_rules) or not active_rules:
            model = self.apply_rmsnorm_fusion(model)

        if any("groupnorm" in r.rule_id for r in active_rules) or not active_rules:
            model = self.apply_groupnorm_fusion(model)

        if any("index" in r.rule_id or "routing" in r.rule_id for r in active_rules) or not active_rules:
            model = self.apply_index_tensor_routing(model)

        if any("identity" in r.rule_id for r in active_rules) or not active_rules:
            model = self.apply_identity_elimination(model)

        if any("rope" in r.rule_id for r in active_rules) or not active_rules:
            model = self.apply_rope_fusion(model)

        if any("gated_unit" in r.rule_id for r in active_rules) or not active_rules:
            model = self.apply_gated_unit_fusion(model)

        if any("softmax_matmul" in r.rule_id for r in active_rules) or not active_rules:
            model = self.apply_softmax_matmul_fusion(model)

        if any("sdpa" in r.rule_id or "attention" in r.rule_id for r in active_rules) or not active_rules:
            model = self.apply_scaled_dot_product_attention_fusion(model)

        if any("linear" in r.rule_id for r in active_rules) or not active_rules:
            model = self.apply_linear_fusion(model)

        if any("merge_heads" in r.rule_id for r in active_rules) or not active_rules:
            model = self.apply_merge_heads_fusion(model)

        if any("axial" in r.rule_id or "rearrange" in r.rule_id for r in active_rules) or not active_rules:
            model = self.apply_axial_rearrange_fusion(model)

        if any("unbind" in r.rule_id for r in active_rules) or not active_rules:
            model = self.apply_unbind_fusion(model)

        if any("split_qkv" in r.rule_id for r in active_rules) or not active_rules:
            model = self.apply_split_qkv_fusion(model)

        if any("cse" in r.rule_id or "common_subexpression" in r.rule_id for r in active_rules) or not active_rules:
            model = self.apply_common_subexpression_elimination(model)

        if any("continuous_math" in r.rule_id for r in active_rules) or not active_rules:
            model = self.apply_continuous_math_collapse(model)

        if any("dce" in r.rule_id or "dead_code" in r.rule_id for r in active_rules) or not active_rules:
            model = self.apply_dead_code_elimination(model)

        # Ensure graph is strictly topologically sorted
        from neuron_toolkit.runtime.runner import toposort_graph

        sorted_nodes = toposort_graph(model)
        del model.graph.node[:]
        model.graph.node.extend(sorted_nodes)

        if self.mode == "plain_custom_op":
            del model.functions[:]

        # Record rule metadata
        helper.set_model_props(
            model,
            {
                "Engine": "NeuronToolkit_JSONRuleEngine",
                "Applied_Rules": ",".join(applied_names or ["all_builtin"]),
                "Engine_Mode": self.mode,
            },
        )
        return model

    # -------------------------------------------------------------------------
    # Core Rule 1: Cast-Initializer Folding
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
            if node.op_type == "Cast" and len(node.input) == 1 and len(node.output) == 1:
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
        consumed_raw = {
            inp
            for n in graph.node
            if (n.name or n.output[0]) not in nodes_to_remove
            for inp in n.input
        }
        filtered_inits = [
            init
            for init in graph.initializer
            if init.name in consumed_raw and init.name not in new_initializers
        ]
        for folded in new_initializers.values():
            filtered_inits.append(folded)

        del graph.initializer[:]
        graph.initializer.extend(filtered_inits)

        filtered_nodes = [
            n
            for n in graph.node
            if (n.name or n.output[0] if n.output else "") not in nodes_to_remove
            and not (
                n.op_type == "Cast"
                and n.output
                and n.output[0] in new_initializers
            )
        ]
        del graph.node[:]
        graph.node.extend(filtered_nodes)

        log.info("Cast-Initializer Folding: Removed %d Cast nodes", len(nodes_to_remove))
        return model

    def apply_cast_conv_weight_folding(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Alias for backward compatibility."""
        return self.apply_cast_initializer_folding(model)

    # -------------------------------------------------------------------------
    # Rule: GroupNorm Fusion (9-node Reshape -> IN -> Reshape -> Mul -> Add)
    # -------------------------------------------------------------------------
    def apply_groupnorm_fusion(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuses 9-node GroupNorm pattern into unified GroupNorm op."""
        graph = model.graph
        node_by_output: dict[str, onnx.NodeProto] = {
            o: n for n in graph.node for o in n.output
        }
        consumer_map: dict[str, list[onnx.NodeProto]] = {}
        for n in graph.node:
            for inp in n.input:
                consumer_map.setdefault(inp, []).append(n)

        inits_dict = {
            i.name: numpy_helper.to_array(i) for i in graph.initializer
        }
        nodes_to_remove: set[str] = set()
        new_gn_nodes: list[onnx.NodeProto] = []
        gn_count = 0

        for inn in graph.node:
            if inn.op_type != "InstanceNormalization":
                continue
            reshape_expand = node_by_output.get(inn.input[0])
            if not (reshape_expand and reshape_expand.op_type == "Reshape"):
                continue
            in_cons = consumer_map.get(inn.output[0], [])
            if len(in_cons) != 1 or in_cons[0].op_type != "Reshape":
                continue
            reshape_restore = in_cons[0]
            r_cons = consumer_map.get(reshape_restore.output[0], [])
            if len(r_cons) != 1 or r_cons[0].op_type != "Mul":
                continue
            mul_node = r_cons[0]
            m_cons = consumer_map.get(mul_node.output[0], [])
            if len(m_cons) != 1 or m_cons[0].op_type != "Add":
                continue
            add_node = m_cons[0]

            x_3d = reshape_expand.input[0]
            final_out = add_node.output[0]
            scale_name = [
                i for i in mul_node.input if i != reshape_restore.output[0]
            ][0]
            bias_name = [
                i for i in add_node.input if i != mul_node.output[0]
            ][0]

            # Convert scale and bias inits to 1D
            if scale_name in inits_dict:
                scale_1d = inits_dict[scale_name].flatten()
                scale_1d_name = f"{scale_name}_1d"
                graph.initializer.append(
                    numpy_helper.from_array(scale_1d, name=scale_1d_name)
                )
            else:
                scale_1d_name = scale_name

            if bias_name in inits_dict:
                bias_1d = inits_dict[bias_name].flatten()
                bias_1d_name = f"{bias_name}_1d"
                graph.initializer.append(
                    numpy_helper.from_array(bias_1d, name=bias_1d_name)
                )
            else:
                bias_1d_name = bias_name

            gn_node = helper.make_node(
                "GroupNorm",
                inputs=[x_3d, scale_1d_name, bias_1d_name],
                outputs=[final_out],
                name=f"groupnorm_{gn_count}",
                domain="custom.pytorch",
                num_groups=1,
                epsilon=1e-5,
                doc_string="Fused 1D GroupNorm / InstanceNorm adapter",
            )
            new_gn_nodes.append(gn_node)
            gn_count += 1

            for n in [reshape_expand, inn, reshape_restore, mul_node, add_node]:
                nodes_to_remove.add(n.name or n.output[0])

            shape_in = reshape_restore.input[1]
            if shape_in in node_by_output and node_by_output[shape_in].op_type == "Shape":
                nodes_to_remove.add(node_by_output[shape_in].name or shape_in)
            for c_inp in inn.input[1:]:
                if c_inp in node_by_output and node_by_output[c_inp].op_type == "Constant":
                    nodes_to_remove.add(node_by_output[c_inp].name or c_inp)
            exp_c = reshape_expand.input[1]
            if exp_c in node_by_output and node_by_output[exp_c].op_type == "Constant":
                nodes_to_remove.add(node_by_output[exp_c].name or exp_c)

        if not new_gn_nodes:
            return model

        kept = [
            n
            for n in graph.node
            if (n.name or (n.output[0] if n.output else "")) not in nodes_to_remove
        ]
        kept.extend(new_gn_nodes)
        del graph.node[:]
        graph.node.extend(kept)

        custom_domain = "custom.pytorch"
        if not any(op.domain == custom_domain for op in model.opset_import):
            model.opset_import.append(helper.make_opsetid(custom_domain, 1))

        if self.mode == "function_proto":
            self._attach_groupnorm_function_def(model)

        log.info("GroupNorm Fusion: Fused %d GroupNorm instances", gn_count)
        return model

    def _attach_groupnorm_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for GroupNorm."""
        func_name = "GroupNorm"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", 14)
        in_node = helper.make_node(
            "InstanceNormalization",
            inputs=["x", "scale", "bias"],
            outputs=["y"],
            epsilon=1e-5,
        )
        gn_func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["x", "scale", "bias"],
            outputs=["y"],
            nodes=[in_node],
            opset_imports=[opset_standard],
            doc_string="GroupNorm normalization function",
        )
        model.functions.append(gn_func)

    # -------------------------------------------------------------------------
    # Rule: Index Tensor Routing & Slicing Fusion
    # -------------------------------------------------------------------------
    def apply_index_tensor_routing(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse Gather on constant index and dynamic slice chains into TensorRoute."""
        graph = model.graph
        node_by_output: dict[str, onnx.NodeProto] = {
            out: n for n in graph.node for out in n.output
        }
        consumer_map: dict[str, list[onnx.NodeProto]] = {}
        for n in graph.node:
            for inp in n.input:
                consumer_map.setdefault(inp, []).append(n)

        # Track constant values
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
                starts_name = (
                    slice_node.input[1] if len(slice_node.input) > 1 else None
                )
                ends_name = (
                    slice_node.input[2] if len(slice_node.input) > 2 else None
                )
                axes_name = (
                    slice_node.input[3] if len(slice_node.input) > 3 else None
                )

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

                # Check dynamic or static crop
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

        if self.mode == "function_proto":
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
    # Core Rule 2: ERF-GELU Expansion Fusion
    # -------------------------------------------------------------------------
    def apply_erf_gelu_fusion(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuses 7-node standard GELU expansion (Div -> Erf -> Add -> Mul -> Mul) into Gelu."""
        graph = model.graph
        node_by_output: dict[str, onnx.NodeProto] = {}
        consumer_map: dict[str, list[onnx.NodeProto]] = {}
        for n in graph.node:
            for out in n.output:
                node_by_output[out] = n
            for inp in n.input:
                consumer_map.setdefault(inp, []).append(n)

        # Track initializers and constant nodes
        constants: dict[str, float] = {}
        constant_node_names: dict[str, str] = {}
        for init in graph.initializer:
            try:
                arr = numpy_helper.to_array(init)
                if arr.size == 1:
                    constants[init.name] = float(arr.item())
            except Exception:
                pass

        for n in graph.node:
            if n.op_type == "Constant" and len(n.output) == 1:
                for attr in n.attribute:
                    if attr.name == "value":
                        try:
                            arr = numpy_helper.to_array(attr.t)
                            if arr.size == 1:
                                constants[n.output[0]] = float(arr.item())
                                constant_node_names[n.output[0]] = n.name or n.output[0]
                        except Exception:
                            pass

        def is_approx(name: str, target: float, tol: float = 0.05) -> bool:
            if name in constants:
                return bool(abs(constants[name] - target) < tol)
            return False

        nodes_to_remove: set[str] = set()
        gelu_nodes_to_insert: list[onnx.NodeProto] = []
        gelu_count = 0

        for erf_node in graph.node:
            if erf_node.op_type != "Erf":
                continue

            # 1. Div input to Erf
            div_out = erf_node.input[0]
            if div_out not in node_by_output:
                continue
            div_node = node_by_output[div_out]
            if div_node.op_type != "Div" or len(div_node.input) < 2:
                continue

            x_tensor = div_node.input[0]
            sqrt2_tensor = div_node.input[1]
            if not is_approx(sqrt2_tensor, 1.41421356, tol=0.05):
                continue

            # 2. Add after Erf
            erf_out = erf_node.output[0]
            add_candidates = [
                c
                for c in consumer_map.get(erf_out, [])
                if c.op_type == "Add" and len(c.input) == 2
            ]
            if not add_candidates:
                continue
            add_node = add_candidates[0]
            c_one_tensor = (
                add_node.input[1] if add_node.input[0] == erf_out else add_node.input[0]
            )
            if not is_approx(c_one_tensor, 1.0, tol=0.01):
                continue

            # 3. Mul1: x * (1 + Erf(...))
            add_out = add_node.output[0]
            mul1_candidates = [
                c
                for c in consumer_map.get(add_out, [])
                if c.op_type == "Mul" and len(c.input) == 2
            ]
            if not mul1_candidates:
                continue
            mul1_node = mul1_candidates[0]
            other_input = (
                mul1_node.input[0] if mul1_node.input[1] == add_out else mul1_node.input[1]
            )
            if other_input != x_tensor:
                continue

            # 4. Mul2: mul1_out * 0.5
            mul1_out = mul1_node.output[0]
            mul2_candidates = [
                c
                for c in consumer_map.get(mul1_out, [])
                if c.op_type == "Mul" and len(c.input) == 2
            ]
            if not mul2_candidates:
                continue
            mul2_node = mul2_candidates[0]
            c_half_tensor = (
                mul2_node.input[1]
                if mul2_node.input[0] == mul1_out
                else mul2_node.input[0]
            )
            if not is_approx(c_half_tensor, 0.5, tol=0.01):
                continue

            final_output = mul2_node.output[0]

            # Mark all intermediate nodes for deletion
            for n in (div_node, erf_node, add_node, mul1_node, mul2_node):
                nodes_to_remove.add(n.name or n.output[0])

            # Check if constants can be removed (if only used in this GELU)
            for c_tensor in (sqrt2_tensor, c_one_tensor, c_half_tensor):
                if c_tensor in constant_node_names:
                    consumers = consumer_map.get(c_tensor, [])
                    if all((c.name or (c.output[0] if c.output else "")) in nodes_to_remove for c in consumers):
                        nodes_to_remove.add(constant_node_names[c_tensor])
                        nodes_to_remove.add(c_tensor)

            # Create replacement Gelu node
            gelu_name = f"gelu_{gelu_count}"
            gelu_count += 1
            domain = "custom.pytorch"

            new_gelu = helper.make_node(
                op_type="Gelu",
                inputs=[x_tensor],
                outputs=[final_output],
                name=gelu_name,
                domain=domain,
                doc_string="Fused standard ERF GELU activation",
            )
            gelu_nodes_to_insert.append(new_gelu)

        if not gelu_nodes_to_insert:
            return model

        # Filter nodes and insert Gelu nodes
        kept_nodes = [
            n
            for n in graph.node
            if (n.name or (n.output[0] if n.output else "")) not in nodes_to_remove
        ]
        kept_nodes.extend(gelu_nodes_to_insert)
        del graph.node[:]
        graph.node.extend(kept_nodes)

        # Register custom opset domain
        custom_domain = "custom.pytorch"
        has_domain = any(op.domain == custom_domain for op in model.opset_import)
        if not has_domain:
            model.opset_import.append(helper.make_opsetid(custom_domain, 1))

        # Attach FunctionProto if mode == 'function_proto'
        if self.mode == "function_proto":
            self._attach_gelu_function_def(model)

        log.info("ERF-GELU Fusion: Fused %d GELU instances", gelu_count)
        return model

    def _attach_gelu_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for Gelu."""
        func_name = "Gelu"
        domain = "custom.pytorch"

        # Check if already present
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return

        opset_standard = helper.make_opsetid("", 14)
        c_sqrt2 = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["c_sqrt2"],
            value=helper.make_tensor(
                "c_sqrt2_t", onnx.TensorProto.FLOAT, [1], [1.41421356]
            ),
        )
        div = helper.make_node("Div", inputs=["x", "c_sqrt2"], outputs=["div_out"])
        erf = helper.make_node("Erf", inputs=["div_out"], outputs=["erf_out"])
        c_one = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["c_one"],
            value=helper.make_tensor("c_one_t", onnx.TensorProto.FLOAT, [1], [1.0]),
        )
        add = helper.make_node("Add", inputs=["erf_out", "c_one"], outputs=["add_out"])
        mul1 = helper.make_node("Mul", inputs=["x", "add_out"], outputs=["mul1_out"])
        c_half = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["c_half"],
            value=helper.make_tensor("c_half_t", onnx.TensorProto.FLOAT, [1], [0.5]),
        )
        mul2 = helper.make_node("Mul", inputs=["mul1_out", "c_half"], outputs=["y"])

        gelu_func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["x"],
            outputs=["y"],
            nodes=[c_sqrt2, div, erf, c_one, add, mul1, c_half, mul2],
            opset_imports=[opset_standard],
            doc_string="Formal FunctionProto definition of standard ERF GELU",
        )
        model.functions.append(gelu_func)

    # -------------------------------------------------------------------------
    # Core Rule 3: Continuous Math Collapse
    # -------------------------------------------------------------------------
    def apply_continuous_math_collapse(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Collapse consecutive constant and tensor arithmetic chains into Math node or folded constant."""
        graph = model.graph
        initializers = {
            init.name: numpy_helper.to_array(init) for init in graph.initializer
        }

        # Build producer mapping
        node_by_output: dict[str, onnx.NodeProto] = {
            out: n for n in graph.node for out in n.output
        }
        consumer_map: dict[str, list[onnx.NodeProto]] = {}
        for n in graph.node:
            for inp in n.input:
                consumer_map.setdefault(inp, []).append(n)

        # Constant value mapping
        constant_values: dict[str, np.ndarray] = dict(initializers)
        constant_node_names: dict[str, str] = {}
        for n in graph.node:
            if n.op_type == "Constant" and len(n.output) == 1:
                for attr in n.attribute:
                    if attr.name == "value":
                        try:
                            constant_values[n.output[0]] = numpy_helper.to_array(attr.t)
                            constant_node_names[n.output[0]] = n.name or n.output[0]
                        except Exception:
                            pass

        nodes_to_remove: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        math_count = 0

        # Perform pure constant folding for chained math operations
        for node in list(graph.node):
            if node.op_type in ("Add", "Sub", "Mul", "Div"):
                if len(node.input) == 2:
                    in1, in2 = node.input[0], node.input[1]
                    if in1 in constant_values and in2 in constant_values:
                        val1 = constant_values[in1]
                        val2 = constant_values[in2]
                        try:
                            if node.op_type == "Add":
                                res = val1 + val2
                                expr_str = f"({in1} + {in2})"
                            elif node.op_type == "Sub":
                                res = val1 - val2
                                expr_str = f"({in1} - {in2})"
                            elif node.op_type == "Mul":
                                res = val1 * val2
                                expr_str = f"({in1} * {in2})"
                            elif node.op_type == "Div":
                                res = val1 / val2
                                expr_str = f"({in1} / {in2})"
                            else:
                                continue

                            out_name = node.output[0]
                            constant_values[out_name] = res

                            # Replace with folded Constant node carrying expr attribute
                            new_const = helper.make_node(
                                "Constant",
                                inputs=[],
                                outputs=[out_name],
                                name=f"math_fold_{math_count}",
                                value=numpy_helper.from_array(res, name=out_name),
                            )
                            # Add string attribute for formula readability in Netron
                            new_const.attribute.append(
                                helper.make_attribute("folded_expr", expr_str)
                            )
                            nodes_to_remove.add(node.name or out_name)
                            nodes_to_remove.add(out_name)

                            # Check if inputs were standalone constants that can also be cleaned up
                            for in_tensor in (in1, in2):
                                if in_tensor in constant_node_names:
                                    consumers = consumer_map.get(in_tensor, [])
                                    if all(
                                        (c.name or (c.output[0] if c.output else "")) in nodes_to_remove
                                        for c in consumers
                                    ):
                                        nodes_to_remove.add(constant_node_names[in_tensor])
                                        nodes_to_remove.add(in_tensor)

                            new_nodes.append(new_const)
                            math_count += 1
                        except Exception:
                            pass

        if not nodes_to_remove:
            return model

        kept_nodes = [
            n
            for n in graph.node
            if (n.name or (n.output[0] if n.output else "")) not in nodes_to_remove
        ]
        kept_nodes.extend(new_nodes)
        del graph.node[:]
        graph.node.extend(kept_nodes)

        log.info("Continuous Math Collapse: Folded %d constant math operations", math_count)
        return model

    # -------------------------------------------------------------------------
    # Rule: RMSNorm Fusion
    # -------------------------------------------------------------------------
    def apply_rmsnorm_fusion(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse 9-node RMSNorm expansion (ReduceL2 -> Clip -> Expand -> Div -> Mul -> Mul) into RMSNorm."""
        graph = model.graph
        node_by_output: dict[str, onnx.NodeProto] = {
            out: n for n in graph.node for out in n.output
        }
        nodes_to_remove: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        rms_count = 0

        for n in graph.node:
            if n.op_type != "ReduceL2":
                continue
            x = n.input[0]
            clips = [c for c in graph.node if n.output[0] in c.input and c.op_type == "Clip"]
            if not clips:
                continue
            clip = clips[0]
            expands = [c for c in graph.node if clip.output[0] in c.input and c.op_type == "Expand"]
            if not expands:
                continue
            expand = expands[0]
            divs = [c for c in graph.node if expand.output[0] in c.input and c.op_type == "Div"]
            if not divs:
                continue
            div = divs[0]
            muls1 = [c for c in graph.node if div.output[0] in c.input and c.op_type == "Mul"]
            if not muls1:
                continue
            mul1 = muls1[0]
            muls2 = [c for c in graph.node if mul1.output[0] in c.input and c.op_type == "Mul"]
            if not muls2:
                continue
            mul2 = muls2[0]

            gamma = mul2.input[1] if mul2.input[0] == mul1.output[0] else mul2.input[0]
            nodes_to_remove.update({n.name, clip.name, expand.name, div.name, mul1.name, mul2.name})

            # Check if expand's shape input is an isolated Shape node
            if len(expand.input) > 1 and expand.input[1] in node_by_output:
                sh_node = node_by_output[expand.input[1]]
                if sh_node.op_type == "Shape":
                    sh_consumers = [c for c in graph.node if sh_node.output[0] in c.input]
                    if len(sh_consumers) == 1:
                        nodes_to_remove.add(sh_node.name)

            rmsnorm_node = helper.make_node(
                op_type="RMSNorm",
                inputs=[x, gamma],
                outputs=[mul2.output[0]],
                name=f"{n.name}_fused_rmsnorm",
                domain="custom.pytorch",
                epsilon=1e-12,
            )
            new_nodes.append(rmsnorm_node)
            rms_count += 1

        if not new_nodes:
            return model

        surviving = [n for n in graph.node if n.name not in nodes_to_remove] + new_nodes
        del graph.node[:]
        graph.node.extend(surviving)

        if self.mode == "function_proto":
            self._attach_rmsnorm_function_def(model)

        log.info("RMSNorm Fusion: Fused %d RMSNorm instances", rms_count)
        return model

    def _attach_rmsnorm_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for RMSNorm."""
        func_name = "RMSNorm"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", 14)
        r_node = helper.make_node("ReduceL2", inputs=["x"], outputs=["norm"], axes=[-1], keepdims=1)
        const_eps = helper.make_node("Constant", inputs=[], outputs=["eps"], value=helper.make_tensor("eps", onnx.TensorProto.FLOAT, [], [1e-12]))
        c_node = helper.make_node("Clip", inputs=["norm", "eps"], outputs=["clipped_norm"])
        d_node = helper.make_node("Div", inputs=["x", "clipped_norm"], outputs=["normed_x"])
        m_node = helper.make_node("Mul", inputs=["normed_x", "gamma"], outputs=["y"])
        rms_func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["x", "gamma"],
            outputs=["y"],
            nodes=[r_node, const_eps, c_node, d_node, m_node],
            opset_imports=[opset_standard],
            doc_string="RMSNorm root mean square normalization function",
        )
        model.functions.append(rms_func)

    # -------------------------------------------------------------------------
    # Rule: Rotary Position Embedding (RoPE) Fusion
    # -------------------------------------------------------------------------
    def apply_rope_fusion(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse 12-node Rotary Position Embedding expansions into RotaryEmbedding."""
        graph = model.graph
        node_by_output: dict[str, onnx.NodeProto] = {
            out: n for n in graph.node for out in n.output
        }
        nodes_to_remove: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        rope_count = 0

        for n in graph.node:
            if n.op_type == "Add" and len(n.input) == 2:
                in0 = node_by_output.get(n.input[0])
                in1 = node_by_output.get(n.input[1])
                if in0 and in1 and in0.op_type == "Mul" and in1.op_type == "Mul":
                    m_cos, m_sin = None, None
                    for mul_node in (in0, in1):
                        for inp in mul_node.input:
                            p = node_by_output.get(inp)
                            if p and p.op_type == "Cos":
                                m_cos = (mul_node, inp)
                            elif p and p.op_type == "Sin":
                                m_sin = (mul_node, inp)

                    if m_cos and m_sin:
                        mul_c, cos_tensor = m_cos
                        mul_s, sin_tensor = m_sin
                        x_slice = mul_c.input[0] if mul_c.input[1] == cos_tensor else mul_c.input[1]
                        rot_tensor = mul_s.input[0] if mul_s.input[1] == sin_tensor else mul_s.input[1]

                        subgraph_nodes = {n.name, mul_c.name, mul_s.name}
                        curr = node_by_output.get(rot_tensor)
                        if curr and curr.op_type == "Reshape":
                            subgraph_nodes.add(curr.name)
                            concat_n = node_by_output.get(curr.input[0])
                            if concat_n and concat_n.op_type == "Concat":
                                subgraph_nodes.add(concat_n.name)
                                for u_inp in concat_n.input:
                                    u_n = node_by_output.get(u_inp)
                                    if u_n and u_n.op_type == "Unsqueeze":
                                        subgraph_nodes.add(u_n.name)
                                        p_n = node_by_output.get(u_n.input[0])
                                        if p_n:
                                            subgraph_nodes.add(p_n.name)
                                            if p_n.op_type == "Neg":
                                                sq = node_by_output.get(p_n.input[0])
                                                if sq and sq.op_type == "Squeeze":
                                                    subgraph_nodes.add(sq.name)
                                                    sp = node_by_output.get(sq.input[0])
                                                    if sp and sp.op_type == "Split":
                                                        subgraph_nodes.add(sp.name)
                                            elif p_n.op_type == "Squeeze":
                                                sp = node_by_output.get(p_n.input[0])
                                                if sp and sp.op_type == "Split":
                                                    subgraph_nodes.add(sp.name)

                        nodes_to_remove.update(subgraph_nodes)
                        
                        # Check if cos and sin share the same theta input
                        cos_n = node_by_output.get(cos_tensor)
                        sin_n = node_by_output.get(sin_tensor)
                        if cos_n and sin_n and cos_n.op_type == "Cos" and sin_n.op_type == "Sin" and cos_n.input == sin_n.input:
                            theta_tensor = cos_n.input[0]
                            base_theta = self._find_base_frequency_tensor(theta_tensor, node_by_output)
                            rope_inputs = [x_slice, base_theta]
                            nodes_to_remove.update({cos_n.name, sin_n.name})
                        else:
                            rope_inputs = [x_slice, cos_tensor, sin_tensor]

                        rope_node = helper.make_node(
                            op_type="RotaryEmbedding",
                            inputs=rope_inputs,
                            outputs=[n.output[0]],
                            name=f"{n.name}_fused_rope",
                            domain="custom.pytorch",
                        )
                        new_nodes.append(rope_node)
                        rope_count += 1

        if not new_nodes:
            return model

        # Check for dummy Concat and dummy Slice around RotaryEmbedding
        rewrites: dict[str, str] = {}
        for r_node in new_nodes:
            r_out = r_node.output[0]
            # 1. Concat cleanup
            concats = [c for c in graph.node if c.op_type == "Concat" and r_out in c.input]
            if concats:
                concat = concats[0]
                rewrites[concat.output[0]] = r_out
                nodes_to_remove.add(concat.name)

            # 2. Slice middle cleanup
            x_in = r_node.input[0]
            p = node_by_output.get(x_in)
            if p and p.op_type == "Slice":
                r_node.input[0] = p.input[0]
                nodes_to_remove.add(p.name)

        def canonical(name: str) -> str:
            while name in rewrites:
                name = rewrites[name]
            return name

        surviving = []
        for n in graph.node:
            if n.name in nodes_to_remove:
                continue
            for idx, inp in enumerate(n.input):
                n.input[idx] = canonical(inp)
            surviving.append(n)
        surviving.extend(new_nodes)

        del graph.node[:]
        graph.node.extend(surviving)

        if self.mode == "function_proto":
            self._attach_rope_function_def(model)

        log.info("RoPE Fusion: Fused %d RotaryEmbedding instances", rope_count)
        return self.apply_dead_code_elimination(model)

    def _attach_rope_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for RotaryEmbedding."""
        func_name = "RotaryEmbedding"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", 14)
        cos_node = helper.make_node("Cos", inputs=["theta"], outputs=["cos"])
        sin_node = helper.make_node("Sin", inputs=["theta"], outputs=["sin"])
        m_cos = helper.make_node("Mul", inputs=["x", "cos"], outputs=["q_cos"])
        m_sin = helper.make_node("Mul", inputs=["x", "sin"], outputs=["q_sin"])
        add_node = helper.make_node("Add", inputs=["q_cos", "q_sin"], outputs=["y"])
        rope_func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["x", "theta"],
            outputs=["y"],
            nodes=[cos_node, sin_node, m_cos, m_sin, add_node],
            opset_imports=[opset_standard],
            doc_string="Rotary position embedding function",
        )
        model.functions.append(rope_func)

    def _find_base_frequency_tensor(
        self,
        tensor_name: str,
        node_by_output: dict[str, onnx.NodeProto],
    ) -> str:
        """Trace back through low-level slice/reshape nodes to find the root frequency tensor."""
        curr = tensor_name
        visited = set()
        while curr in node_by_output and curr not in visited:
            visited.add(curr)
            n = node_by_output[curr]
            if n.op_type == "ScatterND":
                return curr
            if n.op_type == "Slice":
                curr = n.input[0]
            elif n.op_type in ("Reshape", "Expand", "Unsqueeze", "Cast"):
                curr = n.input[0]
            elif n.op_type == "Einsum":
                prefix = n.name.rsplit("/", 1)[0]
                scatter_candidates = [
                    out
                    for out, node in node_by_output.items()
                    if node.op_type == "ScatterND" and node.name.startswith(prefix)
                ]
                if scatter_candidates:
                    return scatter_candidates[0]
                return curr
            else:
                break
        return curr

    # -------------------------------------------------------------------------
    # Rule: Unbind Fusion (Split + Squeeze -> Unbind)
    # -------------------------------------------------------------------------
    def apply_unbind_fusion(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse Split into multiple singletons followed by Squeeze nodes into a single Unbind."""
        graph = model.graph
        nodes_to_remove: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        unbind_count = 0

        for s in graph.node:
            if s.op_type == "Split" and len(s.output) > 1:
                squeezes = []
                for out in s.output:
                    c = [cn for cn in graph.node if out in cn.input and cn.op_type == "Squeeze"]
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

        if self.mode == "function_proto":
            self._attach_unbind_function_def(model)

        log.info("Unbind Fusion: Fused %d Unbind instances", unbind_count)
        return self.apply_dead_code_elimination(model)

    def _attach_unbind_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for Unbind."""
        func_name = "Unbind"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", 14)
        sp_node = helper.make_node("Split", inputs=["x"], outputs=["s0", "s1", "s2"], axis=0)
        sq0 = helper.make_node("Squeeze", inputs=["s0"], outputs=["y0"])
        sq1 = helper.make_node("Squeeze", inputs=["s1"], outputs=["y1"])
        sq2 = helper.make_node("Squeeze", inputs=["s2"], outputs=["y2"])
        unbind_func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["x"],
            outputs=["y0", "y1", "y2"],
            nodes=[sp_node, sq0, sq1, sq2],
            opset_imports=[opset_standard],
            doc_string="Unbind multi-head projections along axis",
        )
        model.functions.append(unbind_func)

    # -------------------------------------------------------------------------
    # Rule: Gated Unit (Sigmoid Attention Gating) Fusion
    # -------------------------------------------------------------------------
    def apply_gated_unit_fusion(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse Sigmoid -> Mul gating into GatedUnit."""
        graph = model.graph
        node_by_output: dict[str, onnx.NodeProto] = {
            out: n for n in graph.node for out in n.output
        }
        nodes_to_remove: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        gate_count = 0

        for n in graph.node:
            if n.op_type == "Mul" and len(n.input) == 2:
                sig_in = [inp for inp in n.input if inp in node_by_output and node_by_output[inp].op_type == "Sigmoid"]
                if sig_in:
                    sig_node = node_by_output[sig_in[0]]
                    other_inp = [inp for inp in n.input if inp != sig_node.output[0]][0]
                    gate_in = sig_node.input[0]

                    nodes_to_remove.update({n.name, sig_node.name})
                    base_gate = self._find_base_gate_tensor(gate_in, node_by_output)
                    gated_node = helper.make_node(
                        op_type="GatedUnit",
                        inputs=[other_inp, base_gate],
                        outputs=[n.output[0]],
                        name=f"{n.name}_fused_gated",
                        domain="custom.pytorch",
                    )
                    new_nodes.append(gated_node)
                    gate_count += 1

        if not new_nodes:
            return model

        surviving = [n for n in graph.node if n.name not in nodes_to_remove] + new_nodes
        del graph.node[:]
        graph.node.extend(surviving)

        if self.mode == "function_proto":
            self._attach_gated_unit_function_def(model)

        log.info("GatedUnit Fusion: Fused %d GatedUnit instances", gate_count)
        return self.apply_dead_code_elimination(model)

    def _find_base_gate_tensor(
        self,
        gate_in: str,
        node_by_output: dict[str, onnx.NodeProto],
    ) -> str:
        """Trace back through gate reshape/transpose pipeline to the root linear projection output."""
        curr = gate_in
        visited = set()
        while curr in node_by_output and curr not in visited:
            visited.add(curr)
            n = node_by_output[curr]
            if n.op_type in ("Reshape", "Transpose"):
                curr = n.input[0]
            else:
                break
        return curr

    def _attach_gated_unit_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for GatedUnit."""
        func_name = "GatedUnit"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", 14)
        t_node = helper.make_node("Transpose", inputs=["gate"], outputs=["gate_t"], perm=[0, 2, 1])
        axes_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["axes_val"],
            value=helper.make_tensor("axes_val", onnx.TensorProto.INT64, [1], [-1]),
        )
        u_node = helper.make_node("Unsqueeze", inputs=["gate_t", "axes_val"], outputs=["gate_4d"])
        sig_node = helper.make_node("Sigmoid", inputs=["gate_4d"], outputs=["sig_out"])
        mul_node = helper.make_node("Mul", inputs=["x", "sig_out"], outputs=["y"])
        gate_func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["x", "gate"],
            outputs=["y"],
            nodes=[t_node, axes_node, u_node, sig_node, mul_node],
            opset_imports=[opset_standard],
            doc_string="Gated unit attention projection function",
        )
        model.functions.append(gate_func)

    # -------------------------------------------------------------------------
    # Rule: Softmax MatMul Attention Core Fusion
    # -------------------------------------------------------------------------
    def apply_softmax_matmul_fusion(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse Softmax -> MatMul into SoftmaxMatMul."""
        graph = model.graph
        node_by_output: dict[str, onnx.NodeProto] = {
            out: n for n in graph.node for out in n.output
        }
        nodes_to_remove: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        sm_count = 0

        for n in graph.node:
            if n.op_type == "MatMul" and len(n.input) == 2:
                in0 = node_by_output.get(n.input[0])
                if in0 and in0.op_type == "Softmax":
                    scores = in0.input[0]
                    v = n.input[1]
                    axis = -1
                    for a in in0.attribute:
                        if a.name == "axis":
                            axis = a.i
                    nodes_to_remove.update({n.name, in0.name})
                    sm_node = helper.make_node(
                        op_type="SoftmaxMatMul",
                        inputs=[scores, v],
                        outputs=[n.output[0]],
                        name=f"{n.name}_fused_sm_matmul",
                        domain="custom.pytorch",
                        axis=axis,
                    )
                    new_nodes.append(sm_node)
                    sm_count += 1

        if not new_nodes:
            return model

        surviving = [n for n in graph.node if n.name not in nodes_to_remove] + new_nodes
        del graph.node[:]
        graph.node.extend(surviving)

        if self.mode == "function_proto":
            self._attach_softmax_matmul_function_def(model)

        log.info("SoftmaxMatMul Fusion: Fused %d SoftmaxMatMul instances", sm_count)
        return model

    def _attach_softmax_matmul_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for SoftmaxMatMul."""
        func_name = "SoftmaxMatMul"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", 14)
        sm_node = helper.make_node("Softmax", inputs=["scores"], outputs=["probs"], axis=-1)
        mm_node = helper.make_node("MatMul", inputs=["probs", "v"], outputs=["y"])
        func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["scores", "v"],
            outputs=["y"],
            nodes=[sm_node, mm_node],
            opset_imports=[opset_standard],
            doc_string="Softmax-MatMul attention function",
        )
        model.functions.append(func)


    # -------------------------------------------------------------------------
    # Rule: Axial Conformer Rearrange Fusion
    # -------------------------------------------------------------------------
    def apply_axial_rearrange_fusion(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse Conformer axial rearrangements (freq_in, freq_to_time, time_to_conv, add_channel)."""
        graph = model.graph
        node_by_output: dict[str, onnx.NodeProto] = {
            out: n for n in graph.node for out in n.output
        }
        nodes_to_drop: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        rearrange_count = 0

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
            if n.op_type == "Transpose":
                perm = [a.ints for a in n.attribute if a.name == "perm"]
                if perm and list(perm[0]) == [0, 2, 1, 3]:
                    prod = node_by_output.get(n.input[0])
                    consumers = [cn for cn in graph.node if n.output[0] in cn.input]
                    if prod and prod.op_type == "Reshape" and len(consumers) == 1 and consumers[0].op_type == "Reshape":
                        r1 = prod
                        r2 = consumers[0]
                        if "partial" in n.name and "attn" not in n.name:
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
            if n.op_type == "Transpose":
                perm = [a.ints for a in n.attribute if a.name == "perm"]
                if perm and list(perm[0]) == [0, 3, 1, 2]:
                    prod = node_by_output.get(n.input[0])
                    if prod and prod.op_type == "Reshape" and "partial" in n.name:
                        r3 = prod
                        nodes_to_drop.update({r3.name, n.name})
                        new_nodes.append(
                            helper.make_node(
                                "AxialRearrange",
                                inputs=[r3.input[0]],
                                outputs=[n.output[0]],
                                name=f"{n.name}_fused_time_to_conv",
                                domain="custom.pytorch",
                                mode="time_to_conv",
                                batch_size=1,
                            )
                        )
                        rearrange_count += 1

            # 4. add_channel: Stem Reshape inserting channel dimension
            if n.op_type == "Reshape" and "add_channel" in n.name.lower():
                prod = node_by_output.get(n.input[0])
                if prod and "bn" in prod.name.lower():
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

        surviving = [node for node in graph.node if node.name not in nodes_to_drop] + new_nodes
        del graph.node[:]
        graph.node.extend(surviving)

        custom_domain = "custom.pytorch"
        if not any(op.domain == custom_domain for op in model.opset_import):
            model.opset_import.append(helper.make_opsetid(custom_domain, 1))

        if self.mode == "function_proto":
            self._attach_axial_rearrange_function_def(model)

        log.info("AxialRearrange Fusion: Fused %d AxialRearrange instances", rearrange_count)
        return self.apply_dead_code_elimination(model)

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

    # -------------------------------------------------------------------------
    # Rule: Scaled Dot-Product Attention Fusion
    # -------------------------------------------------------------------------
    def apply_scaled_dot_product_attention_fusion(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse attention scaling, transposition, dot-product, and softmax into ScaledDotProductAttention."""
        graph = model.graph
        node_by_output: dict[str, onnx.NodeProto] = {
            out: n for n in graph.node for out in n.output
        }
        sm_nodes = [n for n in graph.node if n.op_type == "SoftmaxMatMul"]
        new_nodes: list[onnx.NodeProto] = []
        nodes_to_drop: set[str] = set()

        for sm in sm_nodes:
            scores_inp, v_inp = sm.input[0], sm.input[1]
            mm = node_by_output.get(scores_inp)
            if not mm or mm.op_type != "MatMul":
                continue
            q_mul = node_by_output.get(mm.input[0])
            k_mul = node_by_output.get(mm.input[1])
            if not q_mul or not k_mul or q_mul.op_type != "Mul" or k_mul.op_type != "Mul":
                continue
            q_inp = q_mul.input[0]
            k_trans = node_by_output.get(k_mul.input[0])
            if not k_trans or k_trans.op_type != "Transpose":
                continue
            k_inp = k_trans.input[0]

            # Bypass redundant Cast nodes on Q, K, V
            cast_q = node_by_output.get(q_inp)
            if cast_q and cast_q.op_type == "Cast":
                q_inp = cast_q.input[0]
            cast_k = node_by_output.get(k_inp)
            if cast_k and cast_k.op_type == "Cast":
                k_inp = cast_k.input[0]
            cast_v = node_by_output.get(v_inp)
            if cast_v and cast_v.op_type == "Cast":
                v_inp = cast_v.input[0]

            sdpa = helper.make_node(
                "ScaledDotProductAttention",
                inputs=[q_inp, k_inp, v_inp],
                outputs=list(sm.output),
                name=f"{sm.name}_sdpa",
                domain="custom.pytorch",
            )
            new_nodes.append(sdpa)
            nodes_to_drop.add(sm.name)
            nodes_to_drop.add(mm.name)
            nodes_to_drop.add(q_mul.name)
            nodes_to_drop.add(k_mul.name)
            nodes_to_drop.add(k_trans.name)

        if not new_nodes:
            return model

        kept = [n for n in graph.node if n.name not in nodes_to_drop]
        kept.extend(new_nodes)
        del graph.node[:]
        graph.node.extend(kept)

        if self.mode == "function_proto":
            self._attach_sdpa_function_def(model)

        log.info("ScaledDotProductAttention Fusion: Fused %d attention blocks", len(new_nodes))
        return self.apply_dead_code_elimination(model)

    def _attach_sdpa_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for ScaledDotProductAttention."""
        func_name = "ScaledDotProductAttention"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", 14)
        k_trans = helper.make_node("Transpose", inputs=["K"], outputs=["KT"], perm=[0, 1, 3, 2])
        mm1 = helper.make_node("MatMul", inputs=["Q", "KT"], outputs=["Scores"])
        sm = helper.make_node("Softmax", inputs=["Scores"], outputs=["Probs"], axis=-1)
        mm2 = helper.make_node("MatMul", inputs=["Probs", "V"], outputs=["Y"])
        func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            nodes=[k_trans, mm1, sm, mm2],
            opset_imports=[opset_standard],
            doc_string="Scaled Dot-Product Attention function",
        )
        model.functions.append(func)


    # -------------------------------------------------------------------------
    # Rule: Axial Conformer Rearrange Fusion
    # -------------------------------------------------------------------------
    def apply_axial_rearrange_fusion(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse Conformer axial rearrangements (freq_in, freq_to_time, time_to_conv, add_channel)."""
        graph = model.graph
        node_by_output: dict[str, onnx.NodeProto] = {
            out: n for n in graph.node for out in n.output
        }
        nodes_to_drop: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        rearrange_count = 0

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
            if n.op_type == "Transpose":
                perm = [a.ints for a in n.attribute if a.name == "perm"]
                if perm and list(perm[0]) == [0, 2, 1, 3]:
                    prod = node_by_output.get(n.input[0])
                    consumers = [cn for cn in graph.node if n.output[0] in cn.input]
                    if prod and prod.op_type == "Reshape" and len(consumers) == 1 and consumers[0].op_type == "Reshape":
                        r1 = prod
                        r2 = consumers[0]
                        if "partial" in n.name and "attn" not in n.name:
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
            if n.op_type == "Transpose":
                perm = [a.ints for a in n.attribute if a.name == "perm"]
                if perm and list(perm[0]) == [0, 3, 1, 2]:
                    prod = node_by_output.get(n.input[0])
                    if prod and prod.op_type == "Reshape" and "partial" in n.name:
                        r3 = prod
                        nodes_to_drop.update({r3.name, n.name})
                        new_nodes.append(
                            helper.make_node(
                                "AxialRearrange",
                                inputs=[r3.input[0]],
                                outputs=[n.output[0]],
                                name=f"{n.name}_fused_time_to_conv",
                                domain="custom.pytorch",
                                mode="time_to_conv",
                                batch_size=1,
                            )
                        )
                        rearrange_count += 1

            # 4. add_channel: Stem Reshape inserting channel dimension
            if n.op_type == "Reshape" and "add_channel" in n.name.lower():
                prod = node_by_output.get(n.input[0])
                if prod and "bn" in prod.name.lower():
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

        surviving = [node for node in graph.node if node.name not in nodes_to_drop] + new_nodes
        del graph.node[:]
        graph.node.extend(surviving)

        custom_domain = "custom.pytorch"
        if not any(op.domain == custom_domain for op in model.opset_import):
            model.opset_import.append(helper.make_opsetid(custom_domain, 1))

        if self.mode == "function_proto":
            self._attach_axial_rearrange_function_def(model)

        log.info("AxialRearrange Fusion: Fused %d AxialRearrange instances", rearrange_count)
        return self.apply_dead_code_elimination(model)

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

    # -------------------------------------------------------------------------
    # Rule: Linear Layer Fusion
    # -------------------------------------------------------------------------
    def apply_linear_fusion(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse MatMul(X, W) + bias into custom.pytorch::Linear(X, W, bias)."""
        graph = model.graph
        inits = {t.name for t in graph.initializer}
        node_by_output: dict[str, onnx.NodeProto] = {
            out: n for n in graph.node for out in n.output
        }
        consumer_map: dict[str, list[onnx.NodeProto]] = collections.defaultdict(list)
        for n in graph.node:
            for inp in n.input:
                consumer_map[inp].append(n)

        graph_outputs = {o.name for o in graph.output}
        new_nodes: list[onnx.NodeProto] = []
        nodes_to_drop: set[str] = set()

        for n in graph.node:
            if n.op_type == "Add" and len(n.input) == 2:
                inp0, inp1 = n.input[0], n.input[1]
                mm_node, bias_name = None, None
                if inp0 in inits and inp1 in node_by_output and node_by_output[inp1].op_type == "MatMul":
                    bias_name, mm_node = inp0, node_by_output[inp1]
                elif inp1 in inits and inp0 in node_by_output and node_by_output[inp0].op_type == "MatMul":
                    bias_name, mm_node = inp1, node_by_output[inp0]

                if mm_node and len(consumer_map.get(mm_node.output[0], [])) == 1 and mm_node.output[0] not in graph_outputs:
                    lin = helper.make_node(
                        "Linear",
                        inputs=[mm_node.input[0], mm_node.input[1], bias_name],
                        outputs=list(n.output),
                        name=f"{n.name}_linear",
                        domain="custom.pytorch",
                    )
                    new_nodes.append(lin)
                    nodes_to_drop.add(n.name)
                    nodes_to_drop.add(mm_node.name)

        if not new_nodes:
            return model

        kept = [n for n in graph.node if n.name not in nodes_to_drop]
        kept.extend(new_nodes)
        del graph.node[:]
        graph.node.extend(kept)

        if self.mode == "function_proto":
            self._attach_linear_function_def(model)

        log.info("Linear Fusion: Fused %d MatMul+Add pairs into Linear", len(new_nodes))
        return model

    def _attach_linear_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for Linear."""
        func_name = "Linear"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", 14)
        mm = helper.make_node("MatMul", inputs=["X", "W"], outputs=["MM"])
        add = helper.make_node("Add", inputs=["MM", "bias"], outputs=["Y"])
        func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["X", "W", "bias"],
            outputs=["Y"],
            nodes=[mm, add],
            opset_imports=[opset_standard],
            doc_string="Linear layer projection function",
        )
        model.functions.append(func)


    # -------------------------------------------------------------------------
    # Rule: Axial Conformer Rearrange Fusion
    # -------------------------------------------------------------------------
    def apply_axial_rearrange_fusion(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse Conformer axial rearrangements (freq_in, freq_to_time, time_to_conv, add_channel)."""
        graph = model.graph
        node_by_output: dict[str, onnx.NodeProto] = {
            out: n for n in graph.node for out in n.output
        }
        nodes_to_drop: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        rearrange_count = 0

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
            if n.op_type == "Transpose":
                perm = [a.ints for a in n.attribute if a.name == "perm"]
                if perm and list(perm[0]) == [0, 2, 1, 3]:
                    prod = node_by_output.get(n.input[0])
                    consumers = [cn for cn in graph.node if n.output[0] in cn.input]
                    if prod and prod.op_type == "Reshape" and len(consumers) == 1 and consumers[0].op_type == "Reshape":
                        r1 = prod
                        r2 = consumers[0]
                        if "partial" in n.name and "attn" not in n.name:
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
            if n.op_type == "Transpose":
                perm = [a.ints for a in n.attribute if a.name == "perm"]
                if perm and list(perm[0]) == [0, 3, 1, 2]:
                    prod = node_by_output.get(n.input[0])
                    if prod and prod.op_type == "Reshape" and "partial" in n.name:
                        r3 = prod
                        nodes_to_drop.update({r3.name, n.name})
                        new_nodes.append(
                            helper.make_node(
                                "AxialRearrange",
                                inputs=[r3.input[0]],
                                outputs=[n.output[0]],
                                name=f"{n.name}_fused_time_to_conv",
                                domain="custom.pytorch",
                                mode="time_to_conv",
                                batch_size=1,
                            )
                        )
                        rearrange_count += 1

            # 4. add_channel: Stem Reshape inserting channel dimension
            if n.op_type == "Reshape" and "add_channel" in n.name.lower():
                prod = node_by_output.get(n.input[0])
                if prod and "bn" in prod.name.lower():
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

        surviving = [node for node in graph.node if node.name not in nodes_to_drop] + new_nodes
        del graph.node[:]
        graph.node.extend(surviving)

        custom_domain = "custom.pytorch"
        if not any(op.domain == custom_domain for op in model.opset_import):
            model.opset_import.append(helper.make_opsetid(custom_domain, 1))

        if self.mode == "function_proto":
            self._attach_axial_rearrange_function_def(model)

        log.info("AxialRearrange Fusion: Fused %d AxialRearrange instances", rearrange_count)
        return self.apply_dead_code_elimination(model)

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

    # -------------------------------------------------------------------------
    # Rule: Merge Heads Fusion
    # -------------------------------------------------------------------------
    def apply_merge_heads_fusion(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse Transpose(perm=[0, 2, 1, 3]) and Reshape after GatedUnit into MergeHeads."""
        graph = model.graph
        consumer_map: dict[str, list[onnx.NodeProto]] = collections.defaultdict(list)
        for n in graph.node:
            for inp in n.input:
                consumer_map[inp].append(n)

        new_nodes: list[onnx.NodeProto] = []
        nodes_to_drop: set[str] = set()

        gated_units = [n for n in graph.node if n.op_type == "GatedUnit"]
        for gu in gated_units:
            users = consumer_map.get(gu.output[0], [])
            trans = [u for u in users if u.op_type == "Transpose"]
            if not trans:
                continue
            t_node = trans[0]
            perm = next((list(a.ints) for a in t_node.attribute if a.name == "perm"), [])
            if perm != [0, 2, 1, 3]:
                continue
            reshapes = [u for u in consumer_map.get(t_node.output[0], []) if u.op_type == "Reshape"]
            if not reshapes:
                continue
            r_node = reshapes[0]
            mh = helper.make_node(
                "MergeHeads",
                inputs=[gu.output[0]],
                outputs=list(r_node.output),
                name=f"{r_node.name}_merge_heads",
                domain="custom.pytorch",
            )
            new_nodes.append(mh)
            nodes_to_drop.add(t_node.name)
            nodes_to_drop.add(r_node.name)

        if not new_nodes:
            return model

        kept = [n for n in graph.node if n.name not in nodes_to_drop]
        kept.extend(new_nodes)
        del graph.node[:]
        graph.node.extend(kept)

        if self.mode == "function_proto":
            self._attach_merge_heads_function_def(model)

        log.info("MergeHeads Fusion: Fused %d MergeHeads instances", len(new_nodes))
        return self.apply_dead_code_elimination(model)

    def _attach_merge_heads_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for MergeHeads."""
        func_name = "MergeHeads"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", 14)
        t = helper.make_node("Transpose", inputs=["X"], outputs=["Y"], perm=[0, 2, 1, 3])
        func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["X"],
            outputs=["Y"],
            nodes=[t],
            opset_imports=[opset_standard],
            doc_string="Merge attention heads function",
        )
        model.functions.append(func)

    # -------------------------------------------------------------------------
    # Rule: Axial Conformer Rearrange Fusion
    # -------------------------------------------------------------------------
    def apply_axial_rearrange_fusion(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse Conformer axial rearrangements (freq_in, freq_to_time, time_to_conv, add_channel)."""
        graph = model.graph
        node_by_output: dict[str, onnx.NodeProto] = {
            out: n for n in graph.node for out in n.output
        }
        nodes_to_drop: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        rearrange_count = 0

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
            if n.op_type == "Transpose":
                perm = [a.ints for a in n.attribute if a.name == "perm"]
                if perm and list(perm[0]) == [0, 2, 1, 3]:
                    prod = node_by_output.get(n.input[0])
                    consumers = [cn for cn in graph.node if n.output[0] in cn.input]
                    if prod and prod.op_type == "Reshape" and len(consumers) == 1 and consumers[0].op_type == "Reshape":
                        r1 = prod
                        r2 = consumers[0]
                        if "partial" in n.name and "attn" not in n.name:
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
            if n.op_type == "Transpose":
                perm = [a.ints for a in n.attribute if a.name == "perm"]
                if perm and list(perm[0]) == [0, 3, 1, 2]:
                    prod = node_by_output.get(n.input[0])
                    if prod and prod.op_type == "Reshape" and "partial" in n.name:
                        r3 = prod
                        nodes_to_drop.update({r3.name, n.name})
                        new_nodes.append(
                            helper.make_node(
                                "AxialRearrange",
                                inputs=[r3.input[0]],
                                outputs=[n.output[0]],
                                name=f"{n.name}_fused_time_to_conv",
                                domain="custom.pytorch",
                                mode="time_to_conv",
                                batch_size=1,
                            )
                        )
                        rearrange_count += 1

            # 4. add_channel: Stem Reshape inserting channel dimension
            if n.op_type == "Reshape" and "add_channel" in n.name.lower():
                prod = node_by_output.get(n.input[0])
                if prod and "bn" in prod.name.lower():
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

        surviving = [node for node in graph.node if node.name not in nodes_to_drop] + new_nodes
        del graph.node[:]
        graph.node.extend(surviving)

        custom_domain = "custom.pytorch"
        if not any(op.domain == custom_domain for op in model.opset_import):
            model.opset_import.append(helper.make_opsetid(custom_domain, 1))

        if self.mode == "function_proto":
            self._attach_axial_rearrange_function_def(model)

        log.info("AxialRearrange Fusion: Fused %d AxialRearrange instances", rearrange_count)
        return self.apply_dead_code_elimination(model)

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

    # -------------------------------------------------------------------------
    # Rule: Split QKV Fusion
    # -------------------------------------------------------------------------
    def apply_split_qkv_fusion(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse Reshape -> Transpose(perm=[2, 0, 3, 1, 4]) -> Unbind into SplitQKV."""
        graph = model.graph
        node_by_output: dict[str, onnx.NodeProto] = {
            out: n for n in graph.node for out in n.output
        }
        new_nodes: list[onnx.NodeProto] = []
        nodes_to_drop: set[str] = set()

        unbind_nodes = [n for n in graph.node if n.op_type == "Unbind"]
        for unbind in unbind_nodes:
            trans = node_by_output.get(unbind.input[0])
            if not trans or trans.op_type != "Transpose":
                continue
            perm = next((list(a.ints) for a in trans.attribute if a.name == "perm"), [])
            if perm != [2, 0, 3, 1, 4]:
                continue
            reshape = node_by_output.get(trans.input[0])
            if not reshape or reshape.op_type != "Reshape":
                continue

            concat = node_by_output.get(reshape.input[1])
            num_heads = 4
            if concat and concat.op_type == "Concat" and len(concat.input) >= 4:
                h_node = node_by_output.get(concat.input[3])
                if h_node and h_node.op_type == "Constant":
                    try:
                        num_heads = int(numpy_helper.to_array(h_node.attribute[0].t).item())
                    except Exception:
                        pass

            qkv_inp = reshape.input[0]
            sqkv = helper.make_node(
                "SplitQKV",
                inputs=[qkv_inp],
                outputs=list(unbind.output),
                name=f"{unbind.name}_split_qkv",
                domain="custom.pytorch",
                num_heads=num_heads,
            )
            new_nodes.append(sqkv)
            nodes_to_drop.add(unbind.name)
            nodes_to_drop.add(trans.name)
            nodes_to_drop.add(reshape.name)

        if not new_nodes:
            return model

        kept = [n for n in graph.node if n.name not in nodes_to_drop]
        kept.extend(new_nodes)
        del graph.node[:]
        graph.node.extend(kept)

        if self.mode == "function_proto":
            self._attach_split_qkv_function_def(model)

        log.info("SplitQKV Fusion: Fused %d SplitQKV instances", len(new_nodes))
        return self.apply_dead_code_elimination(model)

    def _attach_split_qkv_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for SplitQKV."""
        func_name = "SplitQKV"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", 14)
        s = helper.make_node("Split", inputs=["QKV"], outputs=["Q", "K", "V"], axis=0)
        func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["QKV"],
            outputs=["Q", "K", "V"],
            nodes=[s],
            opset_imports=[opset_standard],
            doc_string="Split QKV projections into Q, K, V",
        )
        model.functions.append(func)


    # -------------------------------------------------------------------------
    # Rule: Axial Conformer Rearrange Fusion
    # -------------------------------------------------------------------------
    def apply_axial_rearrange_fusion(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse Conformer axial rearrangements (freq_in, freq_to_time, time_to_conv, add_channel)."""
        graph = model.graph
        node_by_output: dict[str, onnx.NodeProto] = {
            out: n for n in graph.node for out in n.output
        }
        nodes_to_drop: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        rearrange_count = 0

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
            if n.op_type == "Transpose":
                perm = [a.ints for a in n.attribute if a.name == "perm"]
                if perm and list(perm[0]) == [0, 2, 1, 3]:
                    prod = node_by_output.get(n.input[0])
                    consumers = [cn for cn in graph.node if n.output[0] in cn.input]
                    if prod and prod.op_type == "Reshape" and len(consumers) == 1 and consumers[0].op_type == "Reshape":
                        r1 = prod
                        r2 = consumers[0]
                        if "partial" in n.name and "attn" not in n.name:
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
            if n.op_type == "Transpose":
                perm = [a.ints for a in n.attribute if a.name == "perm"]
                if perm and list(perm[0]) == [0, 3, 1, 2]:
                    prod = node_by_output.get(n.input[0])
                    if prod and prod.op_type == "Reshape" and "partial" in n.name:
                        r3 = prod
                        nodes_to_drop.update({r3.name, n.name})
                        new_nodes.append(
                            helper.make_node(
                                "AxialRearrange",
                                inputs=[r3.input[0]],
                                outputs=[n.output[0]],
                                name=f"{n.name}_fused_time_to_conv",
                                domain="custom.pytorch",
                                mode="time_to_conv",
                                batch_size=1,
                            )
                        )
                        rearrange_count += 1

            # 4. add_channel: Stem Reshape inserting channel dimension
            if n.op_type == "Reshape" and "add_channel" in n.name.lower():
                prod = node_by_output.get(n.input[0])
                if prod and "bn" in prod.name.lower():
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

        surviving = [node for node in graph.node if node.name not in nodes_to_drop] + new_nodes
        del graph.node[:]
        graph.node.extend(surviving)

        custom_domain = "custom.pytorch"
        if not any(op.domain == custom_domain for op in model.opset_import):
            model.opset_import.append(helper.make_opsetid(custom_domain, 1))

        if self.mode == "function_proto":
            self._attach_axial_rearrange_function_def(model)

        log.info("AxialRearrange Fusion: Fused %d AxialRearrange instances", rearrange_count)
        return self.apply_dead_code_elimination(model)

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

    # -------------------------------------------------------------------------
    # Rule: Identity Arithmetic Elimination
    # -------------------------------------------------------------------------
    def apply_identity_elimination(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Eliminate redundant identity operations (Mul 1.0, Add 0.0, Sub 0.0, Div 1.0)."""
        graph = model.graph
        const_map: dict[str, np.ndarray] = {}
        for n in graph.node:
            if n.op_type == "Constant" and n.attribute and n.attribute[0].name == "value":
                try:
                    const_map[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
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
            if n.op_type == "Mul" and len(n.input) == 2 and n.output[0] not in graph_outputs:
                for i in (0, 1):
                    inp, other = n.input[i], n.input[1 - i]
                    if inp in const_map:
                        v = const_map[inp]
                        if v.size == 1 and float(v.item()) == 1.0:
                            rewrites[n.output[0]] = other
                            identity_drops.add(n.name or n.output[0])
                            break
            elif n.op_type == "Add" and len(n.input) == 2 and n.output[0] not in graph_outputs:
                for i in (0, 1):
                    inp, other = n.input[i], n.input[1 - i]
                    if inp in const_map:
                        v = const_map[inp]
                        if v.size == 1 and float(v.item()) == 0.0:
                            rewrites[n.output[0]] = other
                            identity_drops.add(n.name or n.output[0])
                            break
            elif n.op_type == "Sub" and len(n.input) == 2 and n.output[0] not in graph_outputs:
                inp, other = n.input[1], n.input[0]
                if inp in const_map:
                    v = const_map[inp]
                    if v.size == 1 and float(v.item()) == 0.0:
                        rewrites[n.output[0]] = other
                        identity_drops.add(n.name or n.output[0])
            elif n.op_type == "Div" and len(n.input) == 2 and n.output[0] not in graph_outputs:
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
    # Rule: Common Subexpression Elimination (CSE)
    # -------------------------------------------------------------------------
    def apply_common_subexpression_elimination(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Deduplicate identical deterministic subexpressions across the graph."""
        graph = model.graph
        rewrites: dict[str, str] = {}
        nodes_to_drop: set[str] = set()
        graph_outputs = {o.name for o in graph.output}

        # Step 1: Constant Deduplication
        const_val_map: dict[tuple[str, tuple[int, ...], bytes], str] = {}
        for n in graph.node:
            if n.op_type == "Constant" and n.attribute and n.attribute[0].name == "value":
                try:
                    arr = numpy_helper.to_array(n.attribute[0].t)
                    key = (arr.dtype.str, tuple(arr.shape), arr.tobytes())
                    if key in const_val_map:
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
        for _ in range(5):
            sig_map: dict[tuple[str, tuple[str, ...], tuple[tuple[str, Any], ...]], str] = {}
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
        log.info("Common Subexpression Elimination: Removed %d duplicate ops", len(nodes_to_drop))
        return self.apply_dead_code_elimination(model)

    # -------------------------------------------------------------------------
    # Rule: Dead Code Elimination (DCE)
    # -------------------------------------------------------------------------
    def apply_dead_code_elimination(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Remove unused unreachable nodes whose outputs have no consumers."""
        graph = model.graph
        graph_outputs = {o.name for o in graph.output}

        for _ in range(5):
            used_outputs = set(graph_outputs)
            for n in graph.node:
                for inp in n.input:
                    used_outputs.add(inp)

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
