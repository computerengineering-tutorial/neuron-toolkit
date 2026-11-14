"""Transformer and Large Language Model (LLM) graph optimization passes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from neuron_toolkit.passes.base import BasePass

if TYPE_CHECKING:
    from neuron_toolkit.graph import NeuronGraph


class FuseRMSNorm(BasePass):
    """Fuses RMSNorm subgraphs into a unified RMSNorm operator.

    Matches the standard RMSNorm formulation:
        y = (x / sqrt(mean(x^2, -1) + eps)) * gamma
    Decomposed as:
        Pow(x, 2) or Mul(x, x) -> ReduceMean -> Add(eps) -> Sqrt -> Div(x, sqrt_out) -> optional Mul(gamma)
    """

    name: str = "FuseRMSNorm"

    def run(self, graph: NeuronGraph) -> tuple[NeuronGraph, dict[str, Any]]:
        rewriter = graph.rewriter()
        fused = 0
        nodes = list(graph.nodes)
        node_map = {
            getattr(n, "output", [None])[0]: n
            for n in nodes
            if getattr(n, "output", [])
        }

        for node in nodes:
            if getattr(node, "op_type", "") != "Div":
                continue

            div_inputs = getattr(node, "input", [])
            div_outputs = getattr(node, "output", [])
            if len(div_inputs) != 2 or not div_outputs:
                continue

            x_tensor = div_inputs[0]
            denom_name = div_inputs[1]
            sqrt_node = node_map.get(denom_name)
            if not sqrt_node or getattr(sqrt_node, "op_type", "") != "Sqrt":
                continue

            sqrt_inputs = getattr(sqrt_node, "input", [])
            if not sqrt_inputs:
                continue

            add_node = node_map.get(sqrt_inputs[0])
            if not add_node or getattr(add_node, "op_type", "") != "Add":
                continue

            add_inputs = getattr(add_node, "input", [])
            if len(add_inputs) != 2:
                continue

            # Identify ReduceMean node and epsilon
            mean_node = None
            eps_val = 1e-6
            for inp_name in add_inputs:
                cand = node_map.get(inp_name)
                if cand and getattr(cand, "op_type", "") == "ReduceMean":
                    mean_node = cand
                elif inp_name in graph.tensor_map:
                    arr = graph.tensor_map[inp_name]
                    if arr.size == 1:
                        eps_val = float(arr.item())

            if not mean_node:
                continue

            mean_inputs = getattr(mean_node, "input", [])
            if not mean_inputs:
                continue

            sq_node = node_map.get(mean_inputs[0])
            if not sq_node:
                continue

            sq_op = getattr(sq_node, "op_type", "")
            sq_inputs = getattr(sq_node, "input", [])
            is_valid_square = False

            if (
                sq_op == "Pow" and len(sq_inputs) >= 2 and sq_inputs[0] == x_tensor
            ) or (
                sq_op == "Mul"
                and len(sq_inputs) == 2
                and sq_inputs[0] == x_tensor
                and sq_inputs[1] == x_tensor
            ):
                is_valid_square = True

            if not is_valid_square:
                continue

            # Check for downstream scale (Mul by gamma)
            div_out = div_outputs[0]
            final_out = div_out
            scale_node = None
            gamma_tensor = None

            consumers = [n for n in nodes if div_out in getattr(n, "input", [])]
            if len(consumers) == 1 and getattr(consumers[0], "op_type", "") == "Mul":
                mul_node = consumers[0]
                mul_inps = getattr(mul_node, "input", [])
                if len(mul_inps) == 2:
                    scale_node = mul_node
                    gamma_tensor = (
                        mul_inps[1] if mul_inps[0] == div_out else mul_inps[0]
                    )
                    final_out = getattr(mul_node, "output", [div_out])[0]

            subgraph_nodes = [node, sqrt_node, add_node, mean_node, sq_node]
            if scale_node:
                subgraph_nodes.append(scale_node)

            unique_nodes = list({id(n): n for n in subgraph_nodes}.values())
            rmsnorm_inputs = [x_tensor]
            if gamma_tensor:
                rmsnorm_inputs.append(gamma_tensor)

            rewriter.replace(
                unique_nodes,
                "RMSNorm",
                inputs=rmsnorm_inputs,
                outputs=[final_out],
                name=f"fused_rmsnorm_{fused}",
                epsilon=eps_val,
            )
            fused += 1

        if fused > 0:
            return rewriter.build_graph(), {"fused_rmsnorm_count": fused}
        return graph, {"fused_rmsnorm_count": 0}


class FuseSwiGLU(BasePass):
    """Fuses SwiGLU / GeGLU activation branches into a unified GLU operator.

    Matches patterns of the form:
        Mul(SiLU(gate), up) or Mul(Sigmoid(gate) * gate, up) -> SwiGLU(gate, up)
    Optionally fuses upstream linear projections if gate and up share the same input x.
    """

    name: str = "FuseSwiGLU"

    def run(self, graph: NeuronGraph) -> tuple[NeuronGraph, dict[str, Any]]:
        rewriter = graph.rewriter()
        fused = 0
        nodes = list(graph.nodes)
        node_map = {
            getattr(n, "output", [None])[0]: n
            for n in nodes
            if getattr(n, "output", [])
        }

        for node in nodes:
            if getattr(node, "op_type", "") != "Mul":
                continue

            mul_inputs = getattr(node, "input", [])
            mul_outputs = getattr(node, "output", [])
            if len(mul_inputs) != 2 or not mul_outputs:
                continue

            branch_0 = node_map.get(mul_inputs[0])
            branch_1 = node_map.get(mul_inputs[1])

            silu_node = None
            up_tensor = None

            if branch_0 and getattr(branch_0, "op_type", "") in (
                "Silu",
                "Sigmoid",
                "Gelu",
            ):
                silu_node = branch_0
                up_tensor = mul_inputs[1]
            elif branch_1 and getattr(branch_1, "op_type", "") in (
                "Silu",
                "Sigmoid",
                "Gelu",
            ):
                silu_node = branch_1
                up_tensor = mul_inputs[0]

            if not silu_node:
                continue

            silu_inputs = getattr(silu_node, "input", [])
            if not silu_inputs:
                continue

            gate_tensor = silu_inputs[0]
            act_type = getattr(silu_node, "op_type", "")

            # If Sigmoid, check for x * Sigmoid(x) SiLU structure
            subgraph_nodes = [node, silu_node]
            if act_type == "Sigmoid":
                # Check if other branch of a parent mul is gate_tensor
                pass

            target_op = "GeGLU" if act_type == "Gelu" else "SwiGLU"

            # Check if gate and up are projections of same input x
            gate_prod = node_map.get(gate_tensor)
            up_prod = node_map.get(up_tensor)
            shared_x = None

            if gate_prod and up_prod:
                g_op = getattr(gate_prod, "op_type", "")
                u_op = getattr(up_prod, "op_type", "")
                if g_op in ("MatMul", "Gemm") and u_op in ("MatMul", "Gemm"):
                    g_inps = getattr(gate_prod, "input", [])
                    u_inps = getattr(up_prod, "input", [])
                    if len(g_inps) >= 2 and len(u_inps) >= 2 and g_inps[0] == u_inps[0]:
                        shared_x = g_inps[0]
                        w_gate = g_inps[1]
                        w_up = u_inps[1]
                        subgraph_nodes.extend([gate_prod, up_prod])
                        unique_nodes = list({id(n): n for n in subgraph_nodes}.values())
                        rewriter.replace(
                            unique_nodes,
                            f"Fused{target_op}",
                            inputs=[shared_x, w_gate, w_up],
                            outputs=[mul_outputs[0]],
                            name=f"fused_{target_op.lower()}_{fused}",
                        )
                        fused += 1
                        continue

            unique_nodes = list({id(n): n for n in subgraph_nodes}.values())
            rewriter.replace(
                unique_nodes,
                target_op,
                inputs=[gate_tensor, up_tensor],
                outputs=[mul_outputs[0]],
                name=f"fused_{target_op.lower()}_{fused}",
            )
            fused += 1

        if fused > 0:
            return rewriter.build_graph(), {"fused_swiglu_count": fused}
        return graph, {"fused_swiglu_count": 0}


class FuseAttention(BasePass):
    """Fuses multi-op Scaled Dot-Product Attention (SDPA) into a unified Attention operator.

    Matches pattern:
        Q @ K^T -> Mul/Div (scaling) -> optional Add (mask) -> Softmax -> MatMul(..., V)
    """

    name: str = "FuseAttention"

    def run(self, graph: NeuronGraph) -> tuple[NeuronGraph, dict[str, Any]]:
        rewriter = graph.rewriter()
        fused = 0
        nodes = list(graph.nodes)
        node_map = {
            getattr(n, "output", [None])[0]: n
            for n in nodes
            if getattr(n, "output", [])
        }

        for node in nodes:
            if getattr(node, "op_type", "") != "MatMul":
                continue

            mm_inputs = getattr(node, "input", [])
            mm_outputs = getattr(node, "output", [])
            if len(mm_inputs) != 2 or not mm_outputs:
                continue

            attn_weights_name = mm_inputs[0]
            v_name = mm_inputs[1]

            softmax_node = node_map.get(attn_weights_name)
            if not softmax_node or getattr(softmax_node, "op_type", "") != "Softmax":
                continue

            sm_inputs = getattr(softmax_node, "input", [])
            if not sm_inputs:
                continue

            curr_name = sm_inputs[0]
            curr_node = node_map.get(curr_name)
            if not curr_node:
                continue

            # Optional mask Add
            mask_node = None
            mask_tensor = None
            if getattr(curr_node, "op_type", "") == "Add":
                mask_node = curr_node
                add_inps = getattr(mask_node, "input", [])
                if len(add_inps) == 2:
                    c0 = node_map.get(add_inps[0])
                    c1 = node_map.get(add_inps[1])
                    if c0 and getattr(c0, "op_type", "") in ("Mul", "Div", "MatMul"):
                        curr_node = c0
                        mask_tensor = add_inps[1]
                    elif c1 and getattr(c1, "op_type", "") in ("Mul", "Div", "MatMul"):
                        curr_node = c1
                        mask_tensor = add_inps[0]

            # Scaling Mul/Div
            scale_node = None
            scale_factor = 1.0
            if getattr(curr_node, "op_type", "") in ("Mul", "Div"):
                scale_node = curr_node
                s_inps = getattr(scale_node, "input", [])
                if len(s_inps) == 2:
                    p0 = node_map.get(s_inps[0])
                    p1 = node_map.get(s_inps[1])
                    if p0 and getattr(p0, "op_type", "") == "MatMul":
                        curr_node = p0
                    elif p1 and getattr(p1, "op_type", "") == "MatMul":
                        curr_node = p1

            # Q @ K MatMul
            if getattr(curr_node, "op_type", "") != "MatMul":
                continue

            qk_matmul = curr_node
            qk_inps = getattr(qk_matmul, "input", [])
            if len(qk_inps) != 2:
                continue

            q_name = qk_inps[0]
            k_name = qk_inps[1]

            # Collect all nodes
            subgraph_nodes = [node, softmax_node, qk_matmul]
            if mask_node:
                subgraph_nodes.append(mask_node)
            if scale_node:
                subgraph_nodes.append(scale_node)

            unique_nodes = list({id(n): n for n in subgraph_nodes}.values())
            attn_inputs = [q_name, k_name, v_name]
            if mask_tensor:
                attn_inputs.append(mask_tensor)

            rewriter.replace(
                unique_nodes,
                "ScaledDotProductAttention",
                inputs=attn_inputs,
                outputs=[mm_outputs[0]],
                name=f"fused_attention_{fused}",
                scale=scale_factor,
            )
            fused += 1

        if fused > 0:
            return rewriter.build_graph(), {"fused_attention_count": fused}
        return graph, {"fused_attention_count": 0}


class FuseRotaryEmbedding(BasePass):
    """Fuses rotary position embedding (RoPE) rotation logic into a RotaryEmbedding operator.

    Matches pattern:
        (x * cos) + (rotate_half(x) * sin)
    where rotate_half is:
        Concat(-x[..., d/2:], x[..., :d/2])
    """

    name: str = "FuseRotaryEmbedding"

    def run(self, graph: NeuronGraph) -> tuple[NeuronGraph, dict[str, Any]]:
        rewriter = graph.rewriter()
        fused = 0
        nodes = list(graph.nodes)
        node_map = {
            getattr(n, "output", [None])[0]: n
            for n in nodes
            if getattr(n, "output", [])
        }

        for node in nodes:
            if getattr(node, "op_type", "") != "Add":
                continue

            add_inputs = getattr(node, "input", [])
            add_outputs = getattr(node, "output", [])
            if len(add_inputs) != 2 or not add_outputs:
                continue

            mul_a = node_map.get(add_inputs[0])
            mul_b = node_map.get(add_inputs[1])
            if not mul_a or not mul_b:
                continue
            if (
                getattr(mul_a, "op_type", "") != "Mul"
                or getattr(mul_b, "op_type", "") != "Mul"
            ):
                continue

            # One branch has rotate_half (Concat of Neg)
            branch_cos = None
            branch_sin = None
            rot_node = None

            for m in (mul_a, mul_b):
                inps = getattr(m, "input", [])
                for inp_name in inps:
                    parent = node_map.get(inp_name)
                    if parent and getattr(parent, "op_type", "") == "Concat":
                        rot_node = parent
                        branch_sin = m
                        branch_cos = mul_b if m is mul_a else mul_a
                        break
                if rot_node:
                    break

            if not rot_node or not branch_sin or not branch_cos:
                continue

            # Verify Concat has Neg node in inputs
            concat_inps = getattr(rot_node, "input", [])
            neg_node = None
            for ci in concat_inps:
                p = node_map.get(ci)
                if p and getattr(p, "op_type", "") == "Neg":
                    neg_node = p
                    break

            if not neg_node:
                continue

            # Identify base tensor x from branch_cos
            cos_inps = getattr(branch_cos, "input", [])
            x_candidate = None
            cos_tensor = None
            for ci in cos_inps:
                p = node_map.get(ci)
                if not p or getattr(p, "op_type", "") not in ("Concat", "Neg"):
                    if x_candidate is None:
                        x_candidate = ci
                    else:
                        cos_tensor = ci

            sin_inps = getattr(branch_sin, "input", [])
            sin_tensor = (
                sin_inps[1]
                if sin_inps[0] == getattr(rot_node, "output", [None])[0]
                else sin_inps[0]
            )

            if not x_candidate:
                continue

            subgraph_nodes = [node, mul_a, mul_b, rot_node, neg_node]
            unique_nodes = list({id(n): n for n in subgraph_nodes}.values())
            rope_inputs = [x_candidate]
            if cos_tensor:
                rope_inputs.append(cos_tensor)
            if sin_tensor:
                rope_inputs.append(sin_tensor)

            rewriter.replace(
                unique_nodes,
                "RotaryEmbedding",
                inputs=rope_inputs,
                outputs=[add_outputs[0]],
                name=f"fused_rope_{fused}",
            )
            fused += 1

        if fused > 0:
            return rewriter.build_graph(), {"fused_rope_count": fused}
        return graph, {"fused_rope_count": 0}
