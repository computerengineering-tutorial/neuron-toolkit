"""Activation Canonicalizer Pass.

Canonicalizes non-linear activation patterns into standard or custom ops:
- ERF GELU -> standard Gelu(approximate="none")
- Tanh / Fast GELU -> standard Gelu(approximate="tanh")
- SiLU / Swish -> SiLU (x * Sigmoid(x))
- HardSwish -> HardSwish (x * Clip(x + 3, 0, 6) / 6)
- Relu6 -> Relu6 (Clip 0 to 6)
- Mish -> Mish (x * Tanh(Softplus(x)))
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper

from neuron_toolkit.compiler.base import CompilerContext, Pass

log = logging.getLogger("neuron_toolkit.compiler.activation_canonicalizer")


class ActivationCanonicalizer(Pass):
    """Canonicalizes non-linear activations."""

    @property
    def name(self) -> str:
        return "activation_canonicalizer"

    @property
    def category(self) -> str:
        return "canonicalization"

    def run(self, model: onnx.ModelProto, context: CompilerContext) -> onnx.ModelProto:
        nodes_before = len(model.graph.node)

        if context.should_run("softmax"):
            model = self.apply_softmax_fusion(model)
        if context.should_run("erf_gelu") or context.should_run("gelu"):
            model = self.apply_erf_gelu_fusion(model, mode=context.mode)
        if context.should_run("tanh_gelu") or context.should_run("fast_gelu"):
            model = self.apply_tanh_gelu_fusion(model, mode=context.mode)
        if context.should_run("silu") or context.should_run("swish"):
            model = self.apply_silu_fusion(model, mode=context.mode)
        if context.should_run("hard_swish") or context.should_run("hardswish"):
            model = self.apply_hardswish_fusion(model, mode=context.mode)
        if context.should_run("mish"):
            model = self.apply_mish_fusion(model, mode=context.mode)
        if context.should_run("relu6"):
            model = self.apply_relu6_fusion(model, mode=context.mode)

        nodes_after = len(model.graph.node)
        context.record_pass(self.name, nodes_before, nodes_after)
        return model

    # -------------------------------------------------------------------------
    # ERF-GELU Fusion
    # -------------------------------------------------------------------------
    def apply_erf_gelu_fusion(self, model: onnx.ModelProto, mode: str = "function_proto") -> onnx.ModelProto:
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

            # 1. Div or Mul input to Erf
            div_out = erf_node.input[0]
            if div_out not in node_by_output:
                continue
            div_node = node_by_output[div_out]
            if len(div_node.input) < 2:
                continue

            x_tensor = None
            sqrt2_tensor = None
            if div_node.op_type == "Div":
                x_tensor = div_node.input[0]
                sqrt2_tensor = div_node.input[1]
                if not is_approx(sqrt2_tensor, 1.41421356, tol=0.05):
                    continue
            elif div_node.op_type == "Mul":
                if is_approx(div_node.input[1], 0.70710678, tol=0.05):
                    x_tensor = div_node.input[0]
                    sqrt2_tensor = div_node.input[1]
                elif is_approx(div_node.input[0], 0.70710678, tol=0.05):
                    x_tensor = div_node.input[1]
                    sqrt2_tensor = div_node.input[0]
                else:
                    continue
            else:
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

            # 3. Mul with add_out
            add_out = add_node.output[0]
            mul_candidates = [
                c
                for c in consumer_map.get(add_out, [])
                if c.op_type == "Mul" and len(c.input) == 2
            ]
            if not mul_candidates:
                continue
            mul_cand = mul_candidates[0]
            other_input = (
                mul_cand.input[0] if mul_cand.input[1] == add_out else mul_cand.input[1]
            )

            c_half_tensor = None
            if other_input == x_tensor:
                # Form 1: Mul(x, add_out) -> Mul(mul1_out, 0.5)
                mul1_node = mul_cand
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
                for n in (div_node, erf_node, add_node, mul1_node, mul2_node):
                    nodes_to_remove.add(n.name or n.output[0])
            elif is_approx(other_input, 0.5, tol=0.01):
                # Form 3: Mul(add_out, 0.5) -> Mul(mul1_out, x)
                mul1_node = mul_cand
                mul1_out = mul1_node.output[0]
                mul2_candidates = [
                    c
                    for c in consumer_map.get(mul1_out, [])
                    if c.op_type == "Mul" and len(c.input) == 2
                ]
                if not mul2_candidates:
                    continue
                mul2_node = mul2_candidates[0]
                if x_tensor not in mul2_node.input:
                    continue
                c_half_tensor = other_input
                final_output = mul2_node.output[0]
                for n in (div_node, erf_node, add_node, mul1_node, mul2_node):
                    nodes_to_remove.add(n.name or n.output[0])
            elif other_input in node_by_output and node_by_output[other_input].op_type == "Mul":
                # Form 2: Mul(x, 0.5) -> Mul(half_x, add_out)
                half_mul = node_by_output[other_input]
                h0, h1 = half_mul.input[0], half_mul.input[1]
                half_c = h1 if h0 == x_tensor else (h0 if h1 == x_tensor else None)
                if half_c is None or not is_approx(half_c, 0.5, tol=0.01):
                    continue
                c_half_tensor = half_c
                final_output = mul_cand.output[0]
                for n in (div_node, erf_node, add_node, half_mul, mul_cand):
                    nodes_to_remove.add(n.name or n.output[0])
            else:
                continue

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

            if mode == "modern_onnx":
                new_gelu = helper.make_node(
                    op_type="Gelu",
                    inputs=[x_tensor],
                    outputs=[final_output],
                    name=gelu_name,
                    approximate="none",
                    doc_string="Standard ONNX Gelu activation",
                )
            else:
                domain = "custom.pytorch"
                new_gelu = helper.make_node(
                    op_type="Gelu",
                    inputs=[x_tensor],
                    outputs=[final_output],
                    name=gelu_name,
                    domain=domain,
                    approximate="none",
                    doc_string="Fused standard ERF GELU activation",
                )
            gelu_nodes_to_insert.append(new_gelu)

        if not gelu_nodes_to_insert:
            return model

        kept_nodes = [
            n
            for n in graph.node
            if (n.name or (n.output[0] if n.output else "")) not in nodes_to_remove
        ]
        kept_nodes.extend(gelu_nodes_to_insert)
        del graph.node[:]
        graph.node.extend(kept_nodes)

        custom_domain = "custom.pytorch"
        has_domain = any(op.domain == custom_domain for op in model.opset_import)
        if not has_domain:
            model.opset_import.append(helper.make_opsetid(custom_domain, 1))

        if mode == "function_proto":
            self._attach_gelu_function_def(model)

        log.info("ERF-GELU Fusion: Fused %d GELU instances", gelu_count)
        return model

    def _attach_gelu_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for Gelu."""
        func_name = "Gelu"
        domain = "custom.pytorch"

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
    # Tanh GELU Fusion
    # -------------------------------------------------------------------------
    def apply_tanh_gelu_fusion(self, model: onnx.ModelProto, mode: str = "function_proto") -> onnx.ModelProto:
        """Fuses Tanh-based GELU polynomial approximation into Gelu(approximate='tanh')."""
        graph = model.graph
        node_by_output = {out: n for n in graph.node for out in n.output}
        consumer_map: dict[str, list[onnx.NodeProto]] = {}
        for n in graph.node:
            for inp in n.input:
                consumer_map.setdefault(inp, []).append(n)

        # Look for Tanh node
        nodes_to_remove: set[str] = set()
        fused_nodes: list[onnx.NodeProto] = []
        count = 0

        for tanh_node in graph.node:
            if tanh_node.op_type != "Tanh" or len(tanh_node.input) != 1:
                continue

            # Check consumer: Add 1.0 -> Mul x -> Mul 0.5
            tanh_out = tanh_node.output[0]
            add_candidates = [c for c in consumer_map.get(tanh_out, []) if c.op_type == "Add"]
            if not add_candidates:
                continue
            add_node = add_candidates[0]
            add_out = add_node.output[0]

            mul_x_candidates = [c for c in consumer_map.get(add_out, []) if c.op_type == "Mul"]
            if not mul_x_candidates:
                continue
            mul_x_node = mul_x_candidates[0]
            mul_x_out = mul_x_node.output[0]

            mul_half_candidates = [c for c in consumer_map.get(mul_x_out, []) if c.op_type == "Mul"]
            if not mul_half_candidates:
                continue
            mul_half_node = mul_half_candidates[0]
            final_out = mul_half_node.output[0]

            # Determine x_in
            x_in = mul_x_node.input[0] if mul_x_node.input[1] == add_out else mul_x_node.input[1]

            # Collect intermediate nodes connected to tanh input
            tanh_in = tanh_node.input[0]
            sub_nodes = [tanh_node, add_node, mul_x_node, mul_half_node]
            curr = tanh_in
            while curr in node_by_output and curr != x_in:
                p_node = node_by_output[curr]
                if p_node in sub_nodes:
                    break
                sub_nodes.append(p_node)
                curr = p_node.input[0] if p_node.input else None

            for sn in sub_nodes:
                nodes_to_remove.add(sn.name or sn.output[0])

            node_name = f"gelu_tanh_{count}"
            count += 1
            if mode == "modern_onnx":
                new_node = helper.make_node(
                    "Gelu",
                    inputs=[x_in],
                    outputs=[final_out],
                    name=node_name,
                    approximate="tanh",
                )
            else:
                new_node = helper.make_node(
                    "Gelu",
                    inputs=[x_in],
                    outputs=[final_out],
                    name=node_name,
                    domain="custom.pytorch",
                    approximate="tanh",
                )
            fused_nodes.append(new_node)

        if not fused_nodes:
            return model

        kept = [n for n in graph.node if (n.name or (n.output[0] if n.output else "")) not in nodes_to_remove]
        kept.extend(fused_nodes)
        del graph.node[:]
        graph.node.extend(kept)

        if mode == "function_proto":
            self._attach_tanh_gelu_function_def(model)

        return model

    def _attach_tanh_gelu_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for Gelu with Tanh approximation."""
        domain = "custom.pytorch"
        opset_standard = helper.make_opsetid("", 14)
        custom_domain = "custom.pytorch"
        if not any(op.domain == custom_domain for op in model.opset_import):
            model.opset_import.append(helper.make_opsetid(custom_domain, 1))

        # Check if a tanh-specific function or Gelu function already exists with Tanh
        if any("Gelu" in f.name and f.domain == domain and any(n.op_type == "Tanh" for n in f.node) for f in model.functions):
            return

        c_sqrt = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["c_sqrt"],
            value=helper.make_tensor("c_sqrt_t", onnx.TensorProto.FLOAT, [1], [0.79788456]),
        )
        c_coeff = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["c_coeff"],
            value=helper.make_tensor("c_coeff_t", onnx.TensorProto.FLOAT, [1], [0.044715]),
        )
        c_three = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["c_three"],
            value=helper.make_tensor("c_three_t", onnx.TensorProto.FLOAT, [1], [3.0]),
        )
        c_one = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["c_one"],
            value=helper.make_tensor("c_one_t", onnx.TensorProto.FLOAT, [1], [1.0]),
        )
        c_half = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["c_half"],
            value=helper.make_tensor("c_half_t", onnx.TensorProto.FLOAT, [1], [0.5]),
        )
        pow3 = helper.make_node("Pow", inputs=["x", "c_three"], outputs=["x_cubed"])
        mul_c = helper.make_node("Mul", inputs=["x_cubed", "c_coeff"], outputs=["mul_c_out"])
        add_x = helper.make_node("Add", inputs=["x", "mul_c_out"], outputs=["add_x_out"])
        mul_sqrt = helper.make_node("Mul", inputs=["add_x_out", "c_sqrt"], outputs=["mul_sqrt_out"])
        tanh_n = helper.make_node("Tanh", inputs=["mul_sqrt_out"], outputs=["tanh_out"])
        add1 = helper.make_node("Add", inputs=["tanh_out", "c_one"], outputs=["add1_out"])
        mul_x = helper.make_node("Mul", inputs=["x", "add1_out"], outputs=["mul_x_out"])
        mul_h = helper.make_node("Mul", inputs=["mul_x_out", "c_half"], outputs=["y"])

        fname = "Gelu" if not any(f.name == "Gelu" and f.domain == domain for f in model.functions) else "Gelu_tanh"
        func = helper.make_function(
            domain=domain,
            fname=fname,
            inputs=["x"],
            outputs=["y"],
            nodes=[c_sqrt, c_coeff, c_three, c_one, c_half, pow3, mul_c, add_x, mul_sqrt, tanh_n, add1, mul_x, mul_h],
            opset_imports=[opset_standard],
            doc_string="Formal FunctionProto definition of Tanh-approximated GELU",
        )
        model.functions.append(func)

    # -------------------------------------------------------------------------
    # SiLU / Swish Fusion
    # -------------------------------------------------------------------------
    def apply_silu_fusion(self, model: onnx.ModelProto, mode: str = "function_proto") -> onnx.ModelProto:
        """Fuses x * Sigmoid(x) into SiLU."""
        graph = model.graph
        node_by_output = {out: n for n in graph.node for out in n.output}
        nodes_to_remove: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        count = 0

        for n in graph.node:
            if n.op_type == "Mul" and len(n.input) == 2:
                in0, in1 = n.input[0], n.input[1]
                sig_node = None
                x_tensor = None
                if in0 in node_by_output and node_by_output[in0].op_type == "Sigmoid":
                    sig_node = node_by_output[in0]
                    x_tensor = in1
                elif in1 in node_by_output and node_by_output[in1].op_type == "Sigmoid":
                    sig_node = node_by_output[in1]
                    x_tensor = in0

                if sig_node is not None and sig_node.input[0] == x_tensor:
                    nodes_to_remove.add(n.name or n.output[0])
                    nodes_to_remove.add(sig_node.name or sig_node.output[0])
                    out_tensor = n.output[0]
                    node_name = f"silu_{count}"
                    count += 1
                    if mode == "modern_onnx":
                        silu_node = helper.make_node(
                            "SiLU",
                            inputs=[x_tensor],
                            outputs=[out_tensor],
                            name=node_name,
                        )
                    else:
                        silu_node = helper.make_node(
                            "SiLU",
                            inputs=[x_tensor],
                            outputs=[out_tensor],
                            name=node_name,
                            domain="custom.pytorch",
                        )
                    new_nodes.append(silu_node)

        if not new_nodes:
            return model

        kept = [n for n in graph.node if (n.name or (n.output[0] if n.output else "")) not in nodes_to_remove]
        kept.extend(new_nodes)
        del graph.node[:]
        graph.node.extend(kept)

        custom_domain = "custom.pytorch"
        if mode != "modern_onnx":
            if not any(op.domain == custom_domain for op in model.opset_import):
                model.opset_import.append(helper.make_opsetid(custom_domain, 1))
            if mode == "function_proto":
                self._attach_silu_function_def(model)

        return model

    # -------------------------------------------------------------------------
    # HardSwish Fusion
    # -------------------------------------------------------------------------
    def apply_hardswish_fusion(self, model: onnx.ModelProto, mode: str = "function_proto") -> onnx.ModelProto:
        """Fuses x * Clip(x + 3, 0, 6) / 6 into HardSwish."""
        graph = model.graph
        node_by_output = {out: n for n in graph.node for out in n.output}
        consumer_map: dict[str, list[onnx.NodeProto]] = {}
        for n in graph.node:
            for inp in n.input:
                consumer_map.setdefault(inp, []).append(n)

        nodes_to_remove: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        count = 0

        for clip_node in graph.node:
            if clip_node.op_type != "Clip":
                continue
            # Clip input: Add(x, 3)
            if not clip_node.input:
                continue
            add_in = clip_node.input[0]
            if add_in not in node_by_output or node_by_output[add_in].op_type != "Add":
                continue
            add_node = node_by_output[add_in]
            clip_out = clip_node.output[0]

            matched = False
            for x_in in add_node.input:
                if matched:
                    break
                # Clip consumers: Mul(x, clip_out) or Div(clip_out, 6)
                for c in consumer_map.get(clip_out, []):
                    if c.op_type == "Mul" and x_in in c.input:
                        mul_out = c.output[0]
                        # Consumer of mul_out: Div by 6 or Mul by 1/6
                        div_candidates = consumer_map.get(mul_out, [])
                        if div_candidates:
                            div_node = div_candidates[0]
                            final_out = div_node.output[0]
                            nodes_to_remove.update({
                                add_node.name or add_node.output[0],
                                clip_node.name or clip_node.output[0],
                                c.name or c.output[0],
                                div_node.name or div_node.output[0],
                            })
                            hswish = helper.make_node(
                                "HardSwish",
                                inputs=[x_in],
                                outputs=[final_out],
                                name=f"hardswish_{count}",
                                domain="" if mode == "modern_onnx" else "custom.pytorch",
                            )
                            new_nodes.append(hswish)
                            count += 1
                            matched = True
                            break

        if not new_nodes:
            return model

        kept = [n for n in graph.node if (n.name or (n.output[0] if n.output else "")) not in nodes_to_remove]
        kept.extend(new_nodes)
        del graph.node[:]
        graph.node.extend(kept)

        custom_domain = "custom.pytorch"
        if mode != "modern_onnx":
            if not any(op.domain == custom_domain for op in model.opset_import):
                model.opset_import.append(helper.make_opsetid(custom_domain, 1))
            if mode == "function_proto":
                self._attach_hardswish_function_def(model)

        return model

    # -------------------------------------------------------------------------
    # Relu6 Fusion
    # -------------------------------------------------------------------------
    def apply_relu6_fusion(self, model: onnx.ModelProto, mode: str = "function_proto") -> onnx.ModelProto:
        """Fuses Clip(min=0, max=6) into Relu6."""
        graph = model.graph
        inits = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
        for n in graph.node:
            if n.op_type == "Constant" and len(n.output) == 1:
                for attr in n.attribute:
                    if attr.name == "value":
                        try:
                            inits[n.output[0]] = numpy_helper.to_array(attr.t)
                        except Exception:
                            pass

        for n in graph.node:
            if n.op_type == "Clip":
                # Check min/max attrs (Opset < 11) or inputs (Opset >= 11)
                min_val, max_val = None, None
                for attr in n.attribute:
                    if attr.name == "min":
                        min_val = attr.f
                    elif attr.name == "max":
                        max_val = attr.f
                if min_val is None and len(n.input) > 1 and n.input[1] in inits:
                    arr = inits[n.input[1]]
                    if arr.size == 1:
                        min_val = float(arr.item())
                if max_val is None and len(n.input) > 2 and n.input[2] in inits:
                    arr = inits[n.input[2]]
                    if arr.size == 1:
                        max_val = float(arr.item())

                if min_val == 0.0 and max_val == 6.0:
                    n.op_type = "Relu6"
                    del n.attribute[:]
                    if len(n.input) > 1:
                        del n.input[1:]
                    if mode != "modern_onnx":
                        n.domain = "custom.pytorch"

        if mode != "modern_onnx":
            custom_domain = "custom.pytorch"
            if any(n.op_type == "Relu6" and n.domain == custom_domain for n in graph.node):
                if not any(op.domain == custom_domain for op in model.opset_import):
                    model.opset_import.append(helper.make_opsetid(custom_domain, 1))
                if mode == "function_proto":
                    self._attach_relu6_function_def(model)

        return model

    # -------------------------------------------------------------------------
    # Mish Fusion
    # -------------------------------------------------------------------------
    def apply_mish_fusion(self, model: onnx.ModelProto, mode: str = "function_proto") -> onnx.ModelProto:
        """Fuses x * Tanh(Softplus(x)) into Mish."""
        graph = model.graph
        node_by_output = {out: n for n in graph.node for out in n.output}
        consumer_map: dict[str, list[onnx.NodeProto]] = {}
        for n in graph.node:
            for inp in n.input:
                consumer_map.setdefault(inp, []).append(n)

        nodes_to_remove: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        count = 0

        for sp_node in graph.node:
            if sp_node.op_type != "Softplus":
                continue
            x_in = sp_node.input[0]
            sp_out = sp_node.output[0]
            tanh_candidates = [c for c in consumer_map.get(sp_out, []) if c.op_type == "Tanh"]
            if not tanh_candidates:
                continue
            tanh_node = tanh_candidates[0]
            tanh_out = tanh_node.output[0]

            mul_candidates = [c for c in consumer_map.get(tanh_out, []) if c.op_type == "Mul" and x_in in c.input]
            if not mul_candidates:
                continue
            mul_node = mul_candidates[0]
            final_out = mul_node.output[0]

            nodes_to_remove.update({
                sp_node.name or sp_node.output[0],
                tanh_node.name or tanh_node.output[0],
                mul_node.name or mul_node.output[0],
            })
            mish = helper.make_node(
                "Mish",
                inputs=[x_in],
                outputs=[final_out],
                name=f"mish_{count}",
                domain="" if mode == "modern_onnx" else "custom.pytorch",
            )
            new_nodes.append(mish)
            count += 1

        if not new_nodes:
            return model

        kept = [n for n in graph.node if (n.name or (n.output[0] if n.output else "")) not in nodes_to_remove]
        kept.extend(new_nodes)
        del graph.node[:]
        graph.node.extend(kept)

        custom_domain = "custom.pytorch"
        if mode != "modern_onnx":
            if not any(op.domain == custom_domain for op in model.opset_import):
                model.opset_import.append(helper.make_opsetid(custom_domain, 1))
            if mode == "function_proto":
                self._attach_mish_function_def(model)

        return model

    # -------------------------------------------------------------------------
    # Decomposed Softmax Fusion
    # -------------------------------------------------------------------------
    def apply_softmax_fusion(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse decomposed Softmax (ReduceMax -> Sub -> Exp -> ReduceSum -> Div) into Softmax."""
        graph = model.graph
        node_by_output = {out: n for n in graph.node for out in n.output}
        consumer_map: dict[str, list[onnx.NodeProto]] = {}
        for n in graph.node:
            for inp in n.input:
                consumer_map.setdefault(inp, []).append(n)
        graph_outputs = {o.name for o in graph.output}

        nodes_to_remove: set[str] = set()
        new_nodes: list[onnx.NodeProto] = []
        count = 0

        for div_node in graph.node:
            if div_node.op_type != "Div" or len(div_node.input) < 2:
                continue

            num_in = div_node.input[0]
            denom_in = div_node.input[1]

            exp_node = node_by_output.get(num_in)
            sum_node = node_by_output.get(denom_in)

            if not exp_node or exp_node.op_type != "Exp":
                continue
            if not sum_node or sum_node.op_type != "ReduceSum":
                continue
            if not sum_node.input or sum_node.input[0] != exp_node.output[0]:
                continue

            # Extract axis from sum_node
            axis = -1
            for attr in sum_node.attribute:
                if attr.name == "axes" and attr.ints:
                    axis = attr.ints[-1]
            if len(sum_node.input) > 1:
                axes_name = sum_node.input[1]
                init_map = {i.name: i for i in graph.initializer}
                if axes_name in init_map:
                    try:
                        arr = numpy_helper.to_array(init_map[axes_name])
                        if arr.size > 0:
                            axis = int(arr.flat[-1])
                    except Exception:
                        pass
                else:
                    cand_const = node_by_output.get(axes_name)
                    if cand_const and cand_const.op_type == "Constant":
                        for a in cand_const.attribute:
                            if a.name == "value":
                                try:
                                    arr = numpy_helper.to_array(a.t)
                                    if arr.size > 0:
                                        axis = int(arr.flat[-1])
                                except Exception:
                                    pass

            # Case A: Shifted softmax with ReduceMax -> Sub
            exp_in = exp_node.input[0]
            sub_node = node_by_output.get(exp_in)
            x_tensor = None
            max_node = None
            if sub_node and sub_node.op_type == "Sub" and len(sub_node.input) >= 2:
                cand_x = sub_node.input[0]
                cand_max = node_by_output.get(sub_node.input[1])
                if cand_max and cand_max.op_type == "ReduceMax" and cand_max.input and cand_max.input[0] == cand_x:
                    x_tensor = cand_x
                    max_node = cand_max

            # Case B: Unshifted softmax: Exp(x) / ReduceSum(Exp(x))
            if x_tensor is None:
                x_tensor = exp_in

            sm_node = helper.make_node(
                "Softmax",
                inputs=[x_tensor],
                outputs=list(div_node.output),
                name=f"{div_node.name}_fused_softmax",
                axis=axis,
            )
            new_nodes.append(sm_node)
            count += 1

            nodes_to_remove.add(div_node.name or div_node.output[0])
            if len(consumer_map.get(sum_node.output[0], [])) <= 1 and sum_node.output[0] not in graph_outputs:
                nodes_to_remove.add(sum_node.name or sum_node.output[0])
            if len(consumer_map.get(exp_node.output[0], [])) <= 2 and exp_node.output[0] not in graph_outputs:
                nodes_to_remove.add(exp_node.name or exp_node.output[0])
            if sub_node and max_node:
                if len(consumer_map.get(sub_node.output[0], [])) <= 1 and sub_node.output[0] not in graph_outputs:
                    nodes_to_remove.add(sub_node.name or sub_node.output[0])
                if len(consumer_map.get(max_node.output[0], [])) <= 1 and max_node.output[0] not in graph_outputs:
                    nodes_to_remove.add(max_node.name or max_node.output[0])

        if not new_nodes:
            return model

        kept = [n for n in graph.node if (n.name or (n.output[0] if n.output else "")) not in nodes_to_remove]
        kept.extend(new_nodes)
        del graph.node[:]
        graph.node.extend(kept)
        log.info("Softmax Canonicalization: Fused %d decomposed Softmax instances", count)
        return model

    def _attach_silu_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for SiLU."""
        func_name = "SiLU"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", 14)
        sig = helper.make_node("Sigmoid", inputs=["x"], outputs=["sig_out"])
        mul = helper.make_node("Mul", inputs=["x", "sig_out"], outputs=["y"])
        func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["x"],
            outputs=["y"],
            nodes=[sig, mul],
            opset_imports=[opset_standard],
            doc_string="SiLU activation function (x * sigmoid(x))",
        )
        model.functions.append(func)

    def _attach_hardswish_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for HardSwish."""
        func_name = "HardSwish"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", 14)
        hsig = helper.make_node("HardSigmoid", inputs=["x"], outputs=["hsig_out"], alpha=1.0 / 6.0, beta=0.5)
        mul = helper.make_node("Mul", inputs=["x", "hsig_out"], outputs=["y"])
        func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["x"],
            outputs=["y"],
            nodes=[hsig, mul],
            opset_imports=[opset_standard],
            doc_string="HardSwish activation function",
        )
        model.functions.append(func)

    def _attach_relu6_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for Relu6."""
        func_name = "Relu6"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", 14)
        zero = helper.make_node("Constant", inputs=[], outputs=["zero"], value=helper.make_tensor("zero", onnx.TensorProto.FLOAT, [], [0.0]))
        six = helper.make_node("Constant", inputs=[], outputs=["six"], value=helper.make_tensor("six", onnx.TensorProto.FLOAT, [], [6.0]))
        clip = helper.make_node("Clip", inputs=["x", "zero", "six"], outputs=["y"])
        func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["x"],
            outputs=["y"],
            nodes=[zero, six, clip],
            opset_imports=[opset_standard],
            doc_string="Relu6 activation function",
        )
        model.functions.append(func)

    def _attach_mish_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for Mish."""
        func_name = "Mish"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", 14)
        sp = helper.make_node("Softplus", inputs=["x"], outputs=["sp_out"])
        tanh = helper.make_node("Tanh", inputs=["sp_out"], outputs=["t_out"])
        mul = helper.make_node("Mul", inputs=["x", "t_out"], outputs=["y"])
        func = helper.make_function(
            domain=domain,
            fname=func_name,
            inputs=["x"],
            outputs=["y"],
            nodes=[sp, tanh, mul],
            opset_imports=[opset_standard],
            doc_string="Mish activation function (x * tanh(softplus(x)))",
        )
        model.functions.append(func)
