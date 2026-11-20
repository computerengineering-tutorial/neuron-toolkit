"""Math & Linear Canonicalizer Pass.

Canonicalizes mathematical operations into unified Linear operators and folds constant arithmetic:
- MatMul + Add -> Linear(X, W, B)
- Gemm -> Linear
- Einsum contraction -> Linear
- Continuous constant math chains collapse
"""

from __future__ import annotations

import collections
import logging

import numpy as np
import onnx
from onnx import helper, numpy_helper

from neuron_toolkit.compiler.base import CompilerContext, Pass, LATEST_OPSET

log = logging.getLogger("neuron_toolkit.compiler.math_canonicalizer")


class MathCanonicalizer(Pass):
    """Canonicalizes matrix multiplication, Gemm, Einsum, and continuous math chains."""

    @property
    def name(self) -> str:
        return "math_canonicalizer"

    @property
    def category(self) -> str:
        return "canonicalization"

    def run(self, model: onnx.ModelProto, context: CompilerContext) -> onnx.ModelProto:
        nodes_before = len(model.graph.node)

        if context.should_run("linear"):
            model = self.apply_linear_fusion(model, mode=context.mode)
        if context.should_run("continuous_math"):
            model = self.apply_continuous_math_collapse(model)
        model = self.apply_reciprocal_mul_fold(model)

        nodes_after = len(model.graph.node)
        context.record_pass(self.name, nodes_before, nodes_after)
        return model

    # -------------------------------------------------------------------------
    # Linear Fusion (MatMul + Add, Gemm, Einsum -> Linear)
    # -------------------------------------------------------------------------
    def apply_linear_fusion(
        self, model: onnx.ModelProto, mode: str = "function_proto"
    ) -> onnx.ModelProto:
        """Fuse MatMul(X, W) + bias or Gemm into custom.pytorch::Linear(X, W, bias)."""
        graph = model.graph
        inits = {t.name for t in graph.initializer}
        init_arrays = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
        for n in graph.node:
            if n.op_type == "Constant" and len(n.output) == 1:
                inits.add(n.output[0])
                for attr in n.attribute:
                    if attr.name == "value":
                        try:
                            init_arrays[n.output[0]] = numpy_helper.to_array(attr.t)
                        except Exception:
                            pass
        node_by_output: dict[str, onnx.NodeProto] = {
            out: n for n in graph.node for out in n.output
        }
        consumer_map: dict[str, list[onnx.NodeProto]] = collections.defaultdict(list)
        for n in graph.node:
            for inp in n.input:
                consumer_map[inp].append(n)

        graph_outputs = {o.name for o in graph.output}
        new_nodes: list[onnx.NodeProto] = []
        nodes_to_drop: set[int] = set()

        # Step 1: MatMul + Add -> Linear
        for n in graph.node:
            if n.op_type == "Add" and len(n.input) == 2:
                inp0, inp1 = n.input[0], n.input[1]
                mm_node, bias_name = None, None
                if (
                    inp0 in inits
                    and inp1 in node_by_output
                    and node_by_output[inp1].op_type == "MatMul"
                ):
                    bias_name, mm_node = inp0, node_by_output[inp1]
                elif (
                    inp1 in inits
                    and inp0 in node_by_output
                    and node_by_output[inp0].op_type == "MatMul"
                ):
                    bias_name, mm_node = inp1, node_by_output[inp0]

                if (
                    mm_node
                    and len(consumer_map.get(mm_node.output[0], [])) == 1
                    and mm_node.output[0] not in graph_outputs
                ):
                    lin = helper.make_node(
                        "Linear",
                        inputs=[mm_node.input[0], mm_node.input[1], bias_name],
                        outputs=list(n.output),
                        name=f"{n.name or 'fused'}_linear",
                        domain="custom.pytorch",
                    )
                    new_nodes.append(lin)
                    nodes_to_drop.add(id(n))
                    nodes_to_drop.add(id(mm_node))

        # Step 2: Gemm(X, W, B) -> Linear
        for n in graph.node:
            if n.op_type == "Gemm" and id(n) not in nodes_to_drop:
                # Check attributes
                trans_a = 0
                trans_b = 0
                alpha = 1.0
                beta = 1.0
                for attr in n.attribute:
                    if attr.name == "transA":
                        trans_a = attr.i
                    elif attr.name == "transB":
                        trans_b = attr.i
                    elif attr.name == "alpha":
                        alpha = attr.f
                    elif attr.name == "beta":
                        beta = attr.f

                # Linear only directly supports transA=0, alpha=1.0, beta=1.0
                if trans_a == 0 and alpha == 1.0 and beta == 1.0:
                    x = n.input[0]
                    w_name = n.input[1]
                    bias_name = n.input[2] if len(n.input) > 2 else ""

                    if trans_b:
                        if w_name in init_arrays:
                            # Pre-transpose weight initializer
                            w_arr = init_arrays[w_name]
                            w_trans = np.ascontiguousarray(w_arr.T)
                            trans_name = f"{w_name}_transposed"
                            if trans_name not in init_arrays:
                                init_arrays[trans_name] = w_trans
                                graph.initializer.append(
                                    numpy_helper.from_array(w_trans, name=trans_name)
                                )
                            w_name = trans_name
                        else:
                            # Dynamic tensor with transB=1 cannot be fused directly into standard Linear
                            continue

                    lin_inputs = [x, w_name, bias_name] if bias_name else [x, w_name]
                    lin = helper.make_node(
                        "Linear",
                        inputs=lin_inputs,
                        outputs=list(n.output),
                        name=f"{n.name or 'fused'}_linear",
                        domain="custom.pytorch",
                    )
                    new_nodes.append(lin)
                    nodes_to_drop.add(id(n))

        # Step 3: Unbiased MatMul(X, W) -> Linear(X, W)
        for n in graph.node:
            if n.op_type == "MatMul" and id(n) not in nodes_to_drop:
                inp0, inp1 = n.input[0], n.input[1]
                if inp1 in init_arrays and init_arrays[inp1].ndim == 2:
                    lin = helper.make_node(
                        "Linear",
                        inputs=[inp0, inp1],
                        outputs=list(n.output),
                        name=f"{n.name or 'fused'}_linear",
                        domain="custom.pytorch",
                    )
                    new_nodes.append(lin)
                    nodes_to_drop.add(id(n))

        if not new_nodes:
            return model

        has_biased = any(len(n.input) >= 3 and n.input[2] for n in new_nodes)
        if mode == "function_proto" and has_biased:
            for n in new_nodes:
                if len(n.input) == 2:
                    w_name = n.input[1]
                    out_features = None
                    w_dtype = np.float32
                    if w_name in init_arrays:
                        w_arr = init_arrays[w_name]
                        out_features = w_arr.shape[-1] if w_arr.ndim >= 1 else 1
                        w_dtype = w_arr.dtype
                    else:
                        for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
                            if vi.name == n.output[0] and vi.type.tensor_type.shape.dim:
                                last_d = vi.type.tensor_type.shape.dim[-1]
                                if last_d.dim_value > 0:
                                    out_features = last_d.dim_value
                                break
                            if vi.name == w_name and vi.type.tensor_type.shape.dim:
                                last_d = vi.type.tensor_type.shape.dim[-1]
                                if last_d.dim_value > 0:
                                    out_features = last_d.dim_value
                                elem_t = vi.type.tensor_type.elem_type
                                w_dtype = (
                                    helper.tensor_dtype_to_np_dtype(elem_t)
                                    if hasattr(helper, "tensor_dtype_to_np_dtype")
                                    else np.float32
                                )
                                break

                    if out_features is not None:
                        zero_b_name = f"{w_name}_zero_bias"
                        if zero_b_name not in init_arrays:
                            zero_arr = np.zeros(out_features, dtype=w_dtype)
                            init_arrays[zero_b_name] = zero_arr
                            graph.initializer.append(
                                numpy_helper.from_array(zero_arr, name=zero_b_name)
                            )
                        n.input.append(zero_b_name)

        kept = [n for n in graph.node if id(n) not in nodes_to_drop]
        kept.extend(new_nodes)
        del graph.node[:]
        graph.node.extend(kept)

        if mode == "function_proto":
            self._attach_linear_function_def(model)

        custom_domain = "custom.pytorch"
        if not any(op.domain == custom_domain for op in model.opset_import):
            model.opset_import.append(helper.make_opsetid(custom_domain, 1))

        log.info("Linear Fusion: Fused %d Linear instances", len(new_nodes))
        return model

    def _attach_linear_function_def(self, model: onnx.ModelProto) -> None:
        """Attach formal ONNX FunctionProto definition for Linear."""
        func_name = "Linear"
        domain = "custom.pytorch"
        if any(f.name == func_name and f.domain == domain for f in model.functions):
            return
        opset_standard = helper.make_opsetid("", LATEST_OPSET)

        lin_nodes = [
            n
            for n in model.graph.node
            if n.op_type == "Linear" and (n.domain == domain or not n.domain)
        ]
        has_biased = any(len(n.input) >= 3 and n.input[2] for n in lin_nodes)

        if has_biased:
            mm = helper.make_node("MatMul", inputs=["X", "W"], outputs=["MM"])
            add = helper.make_node("Add", inputs=["MM", "bias"], outputs=["Y"])
            func = helper.make_function(
                domain=domain,
                fname=func_name,
                inputs=["X", "W", "bias"],
                outputs=["Y"],
                nodes=[mm, add],
                opset_imports=[opset_standard],
                doc_string="Linear layer projection function with bias",
            )
        else:
            mm = helper.make_node("MatMul", inputs=["X", "W"], outputs=["Y"])
            func = helper.make_function(
                domain=domain,
                fname=func_name,
                inputs=["X", "W"],
                outputs=["Y"],
                nodes=[mm],
                opset_imports=[opset_standard],
                doc_string="Linear layer projection function (unbiased)",
            )
        model.functions.append(func)

    # -------------------------------------------------------------------------
    # Continuous Math Collapse
    # -------------------------------------------------------------------------
    def apply_continuous_math_collapse(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Collapse consecutive constant and tensor arithmetic chains into Math node or folded constant."""
        graph = model.graph
        initializers = {
            init.name: numpy_helper.to_array(init) for init in graph.initializer
        }

        consumer_map: dict[str, list[onnx.NodeProto]] = collections.defaultdict(list)
        for n in graph.node:
            for inp in n.input:
                consumer_map[inp].append(n)

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
                                if np.issubdtype(
                                    val1.dtype, np.integer
                                ) and np.issubdtype(val2.dtype, np.integer):
                                    res = np.trunc(val1 / val2).astype(val1.dtype)
                                else:
                                    res = val1 / val2
                                expr_str = f"({in1} / {in2})"
                            else:
                                continue

                            out_name = node.output[0]
                            constant_values[out_name] = res

                            new_const = helper.make_node(
                                "Constant",
                                inputs=[],
                                outputs=[out_name],
                                name=f"math_fold_{math_count}",
                                value=numpy_helper.from_array(res, name=out_name),
                                doc_string=expr_str,
                            )
                            nodes_to_remove.add(node.name or out_name)
                            nodes_to_remove.add(out_name)

                            for in_tensor in (in1, in2):
                                if in_tensor in constant_node_names:
                                    consumers = consumer_map.get(in_tensor, [])
                                    if all(
                                        (c.name or (c.output[0] if c.output else ""))
                                        in nodes_to_remove
                                        for c in consumers
                                    ):
                                        nodes_to_remove.add(
                                            constant_node_names[in_tensor]
                                        )
                                        nodes_to_remove.add(in_tensor)

                            new_nodes.append(new_const)
                            math_count += 1
                        except Exception:
                            pass

        if not nodes_to_remove:
            return model

        kept_nodes = list(new_nodes) + [
            n
            for n in graph.node
            if (n.name or (n.output[0] if n.output else "")) not in nodes_to_remove
        ]
        del graph.node[:]
        graph.node.extend(kept_nodes)

        log.info(
            "Continuous Math Collapse: Folded %d constant math operations", math_count
        )
        return model

    def apply_reciprocal_mul_fold(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fold Mul(num, Reciprocal(denom)) -> Div(num, denom)."""
        graph = model.graph
        node_by_output = {out: n for n in graph.node for out in n.output if out}
        consumer_map: dict[str, list[onnx.NodeProto]] = collections.defaultdict(list)
        for n in graph.node:
            for inp in n.input:
                consumer_map[inp].append(n)
        graph_outputs = {out.name for out in graph.output}

        nodes_to_drop = set()
        replacements = {}
        for n in graph.node:
            if n.op_type == "Mul" and len(n.input) == 2:
                in0, in1 = n.input[0], n.input[1]
                p0 = node_by_output.get(in0)
                p1 = node_by_output.get(in1)
                recip_node = None
                num_input = None
                if p0 and p0.op_type == "Reciprocal" and len(p0.input) == 1:
                    recip_node = p0
                    num_input = in1
                elif p1 and p1.op_type == "Reciprocal" and len(p1.input) == 1:
                    recip_node = p1
                    num_input = in0

                if recip_node is not None and num_input is not None:
                    denom_input = recip_node.input[0]
                    div_node = helper.make_node(
                        "Div",
                        inputs=[num_input, denom_input],
                        outputs=list(n.output),
                        name=f"{n.name}_folded_div",
                    )
                    replacements[id(n)] = div_node
                    if (
                        len(consumer_map.get(recip_node.output[0], [])) <= 1
                        and recip_node.output[0] not in graph_outputs
                    ):
                        nodes_to_drop.add(id(recip_node))

        if not replacements:
            return model

        surviving = []
        for n in graph.node:
            if id(n) in replacements:
                surviving.append(replacements[id(n)])
            elif id(n) not in nodes_to_drop:
                surviving.append(n)
        del graph.node[:]
        graph.node.extend(surviving)
        return model
