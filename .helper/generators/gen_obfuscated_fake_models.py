"""Generator and evaluation harness for heavily obfuscated, randomized, stress-testing ONNX graphs.

Covers major DL architecture blocks from across the zoo:
- Multi-Head Attention (Standard, Rotary, GQA)
- Vision (ConvNeXt, MobileNetV2 Inverted Residual)
- Modern & Vanilla MLP (DeepMLP_Norm_Act, Highway Network, SwiGLU FFN)
- State-Space Models (Mamba S6 Selective Scan)
- Diffusion & Generative (DiT with AdaLN conditioning)
- Hierarchical ViT (Swin Window Attention)
- Pure Programmatic ONNX (Decomposed LayerNorm, decomposed Softmax, dead branches, transpose ping-pongs)

Applies multi-stage randomized obfuscation:
1. PyTorch-level graph mutation (tautologies, ghost branches, commutative additions, reshape ping-pongs).
2. ONNX-level graph mutation (commutative input swapping, identity insertion, cancelling transpose ping-pongs,
   decomposed softmax/layernorm, dead-code branches).

Verifies that compiler passes and rules apply smartly and invariantly, not in a brittle or hardcoded way.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add helper generators to path
HELPERS_DIR = Path(__file__).resolve().parent
if str(HELPERS_DIR) not in sys.path:
    sys.path.insert(0, str(HELPERS_DIR))

from gen_comprehensive_attentions import (
    GroupedQueryAttention_GQA,
    RotaryMHA,
    StandardMHA_SeparateQKV,
)
from gen_diffusion_dit_zoo import DiT_AdaLNZero_Block
from gen_hierarchical_vit_zoo import Swin_WindowAttentionBlock
from gen_ssm_mamba_zoo import Mamba_SelectiveScanBlock
from gen_vanilla_mlp_zoo import (
    DeepMLP_Norm_Act,
    HighwayNetwork_Block,
    SwiGLU_MLP_Block,
)
from gen_vision_cnn_zoo import ConvNeXt_Block, MobileNetV2_InvertedResidual

OUT_DIR = Path(".helper/models/obfuscated_stress")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. PyTorch-Level Randomized Block Mutator
# =============================================================================
class PyTorchBlockMutator(nn.Module):
    """Wraps any deep learning block and injects randomized, realistic real-world graph obfuscations."""

    def __init__(
        self,
        block: nn.Module,
        seed: int = 42,
        variant_mode: int = 0,
        d_model: int = 32,
    ) -> None:
        super().__init__()
        self.block = block
        self.seed = seed
        self.variant_mode = variant_mode
        self.d_model = d_model

        # Ghost parameter for dead side-chain computation
        self.ghost_weight = nn.Parameter(torch.zeros(d_model, d_model))

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # Step 1: Input Tautological Math Mutations
        mode = self.variant_mode % 5
        if mode == 0:
            # Identity arithmetic: (x + 0.0) * 1.0
            x_in = (x + 0.0) * 1.0
        elif mode == 1:
            # Multiplicative ping-pong: (x * 2.0) * 0.5
            x_in = (x * 2.0) * 0.5
        elif mode == 2:
            # Additive ping-pong: (x + 3.1415) - 3.1415
            x_in = (x + 3.1415) - 3.1415
        elif mode == 3:
            # Redundant identity clone
            x_in = x.clone()
        else:
            x_in = x

        # Step 2: Under-the-hood Block Execution
        out_block = self.block(x_in, *args, **kwargs)

        # Step 3: Residual Highway Obfuscation & Commutative Additions
        if out_block.shape == x.shape:
            res_mode = (self.variant_mode // 2) % 5
            if res_mode == 0:
                # Flipped addition: Block(x) + x (tests commutativity of Add)
                out = out_block + x
            elif res_mode == 1:
                # Ghost side-chain: x + Block(x) + 0.0 * (x * 1.0)
                ghost = (x * 0.0)
                out = x + out_block + ghost
            elif res_mode == 2:
                # Split residual: 0.5 * x + Block(x) + 0.5 * x
                out = 0.5 * x + out_block + 0.5 * x
            elif res_mode == 3:
                # Commutative with zero subtraction: (out_block + x) - 0.0
                out = (out_block + x) - 0.0
            else:
                out = x + out_block
        else:
            out = out_block

        # Step 4: Output Tautology
        if self.variant_mode % 2 == 1:
            out = (out * 1.0) + 0.0

        return out


# =============================================================================
# 2. ONNX-Level Graph Mutator (Programmatic Graph Obfuscation)
# =============================================================================
def mutate_onnx_graph(
    model: onnx.ModelProto,
    seed: int = 42,
    swap_commutative: bool = True,
    insert_identities: bool = True,
    insert_transpose_ping_pong: bool = True,
    decompose_softmax: bool = True,
    decompose_layernorm: bool = True,
    inject_dead_nodes: bool = True,
) -> onnx.ModelProto:
    """Apply randomized structural and mathematical graph obfuscations to an ONNX model."""
    random.seed(seed)
    graph = model.graph
    graph_outputs = {o.name for o in graph.output}
    inits = {init.name for init in graph.initializer}

    opset_ver = 14
    for op in model.opset_import:
        if op.domain in ("", "ai.onnx"):
            opset_ver = op.version

    # Perform shape inference to accurately identify tensor ranks and dtypes
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
        value_info_map = {vi.name: vi for vi in inferred.graph.value_info}
        for inp in inferred.graph.input:
            value_info_map[inp.name] = inp
        for out in inferred.graph.output:
            value_info_map[out.name] = out
    except Exception:
        value_info_map = {}

    float_tensors = [
        name for name, vi in value_info_map.items()
        if vi.type.tensor_type.elem_type == TensorProto.FLOAT and name not in graph_outputs and name not in inits
    ]
    tensors_3d = [
        name for name in float_tensors
        if len(value_info_map[name].type.tensor_type.shape.dim) == 3
    ]
    tensors_4d = [
        name for name in float_tensors
        if len(value_info_map[name].type.tensor_type.shape.dim) == 4
    ]

    new_nodes: list[onnx.NodeProto] = []
    new_inits: list[onnx.TensorProto] = []

    # Constant initializers for tautologies
    zero_const = helper.make_tensor(f"mut_zero_{seed}", TensorProto.FLOAT, [], [0.0])
    one_const = helper.make_tensor(f"mut_one_{seed}", TensorProto.FLOAT, [], [1.0])
    new_inits.extend([zero_const, one_const])

    # Pick candidate targets upfront for topologically valid inline insertion
    chosen_tr_t = None
    tr_perm = None
    if insert_transpose_ping_pong:
        if tensors_3d:
            chosen_tr_t = random.choice(tensors_3d)
            tr_perm = [1, 0, 2]
        elif tensors_4d:
            chosen_tr_t = random.choice(tensors_4d)
            tr_perm = [0, 2, 1, 3]

    chosen_id_t = random.choice(float_tensors) if (insert_identities and float_tensors) else None
    chosen_dead_t = random.choice(float_tensors) if (inject_dead_nodes and float_tensors) else None

    tr_ping = f"{chosen_tr_t}_t_ping" if chosen_tr_t else None
    tr_pong = f"{chosen_tr_t}_t_pong" if chosen_tr_t else None
    id_out = f"{chosen_id_t}_mut_id" if chosen_id_t else None

    tr_produced = False
    id_produced = False

    for node in graph.node:
        current_node = copy.deepcopy(node)

        # Rewire inputs if producer has already run
        if tr_produced:
            for idx, inp in enumerate(current_node.input):
                if inp == chosen_tr_t:
                    current_node.input[idx] = tr_pong
        if id_produced:
            for idx, inp in enumerate(current_node.input):
                if inp == chosen_id_t:
                    current_node.input[idx] = id_out

        # 1. Commutative Input Swapping (Add, Mul, Equal, Min, Max)
        if swap_commutative and current_node.op_type in ("Add", "Mul", "Equal", "Min", "Max") and len(current_node.input) == 2:
            if random.random() < 0.5:
                current_node.input[0], current_node.input[1] = current_node.input[1], current_node.input[0]

        # 2. Decompose Softmax into ReduceMax -> Sub -> Exp -> ReduceSum -> Div
        if decompose_softmax and current_node.op_type == "Softmax" and current_node.output[0] not in graph_outputs:
            axis = -1
            for a in current_node.attribute:
                if a.name == "axis":
                    axis = a.i
            x_in = current_node.input[0]
            out_name = current_node.output[0]
            prefix = f"{current_node.name or out_name}_decomp"

            if opset_ver >= 18:
                axes_t = helper.make_tensor(f"{prefix}_axes", TensorProto.INT64, [1], [axis])
                new_inits.append(axes_t)
                rmax = helper.make_node("ReduceMax", inputs=[x_in, axes_t.name], outputs=[f"{prefix}_max"], keepdims=1, name=f"{prefix}_rmax")
                rsum = helper.make_node("ReduceSum", inputs=[f"{prefix}_exp", axes_t.name], outputs=[f"{prefix}_sum"], keepdims=1, name=f"{prefix}_rsum")
            elif opset_ver >= 13:
                axes_t = helper.make_tensor(f"{prefix}_axes", TensorProto.INT64, [1], [axis])
                new_inits.append(axes_t)
                rmax = helper.make_node("ReduceMax", inputs=[x_in], outputs=[f"{prefix}_max"], axes=[axis], keepdims=1, name=f"{prefix}_rmax")
                rsum = helper.make_node("ReduceSum", inputs=[f"{prefix}_exp", axes_t.name], outputs=[f"{prefix}_sum"], keepdims=1, name=f"{prefix}_rsum")
            else:
                rmax = helper.make_node("ReduceMax", inputs=[x_in], outputs=[f"{prefix}_max"], axes=[axis], keepdims=1, name=f"{prefix}_rmax")
                rsum = helper.make_node("ReduceSum", inputs=[f"{prefix}_exp"], outputs=[f"{prefix}_sum"], axes=[axis], keepdims=1, name=f"{prefix}_rsum")

            sub = helper.make_node("Sub", inputs=[x_in, f"{prefix}_max"], outputs=[f"{prefix}_sub"], name=f"{prefix}_sub")
            exp = helper.make_node("Exp", inputs=[f"{prefix}_sub"], outputs=[f"{prefix}_exp"], name=f"{prefix}_exp")
            div = helper.make_node("Div", inputs=[f"{prefix}_exp", f"{prefix}_sum"], outputs=[out_name], name=f"{prefix}_div")

            new_nodes.extend([rmax, sub, exp, rsum, div])
            continue

        # 3. Decompose LayerNormalization into ReduceMean -> Sub -> Mul -> ReduceMean -> Sqrt -> Div -> Mul -> Add
        if decompose_layernorm and current_node.op_type == "LayerNormalization" and len(current_node.input) >= 2:
            x_in = current_node.input[0]
            scale_in = current_node.input[1]
            bias_in = current_node.input[2] if len(current_node.input) > 2 else None
            out_name = current_node.output[0]
            prefix = f"{current_node.name or out_name}_decomp_ln"

            eps = 1e-5
            for a in current_node.attribute:
                if a.name == "epsilon":
                    eps = a.f
            eps_tensor = helper.make_tensor(f"{prefix}_eps", TensorProto.FLOAT, [], [eps])
            new_inits.append(eps_tensor)

            if opset_ver >= 18:
                axes_neg1 = helper.make_tensor(f"{prefix}_axes_neg1", TensorProto.INT64, [1], [-1])
                new_inits.append(axes_neg1)
                mean = helper.make_node("ReduceMean", inputs=[x_in, axes_neg1.name], outputs=[f"{prefix}_mean"], keepdims=1, name=f"{prefix}_rmean")
                var = helper.make_node("ReduceMean", inputs=[f"{prefix}_sq", axes_neg1.name], outputs=[f"{prefix}_var"], keepdims=1, name=f"{prefix}_rvar")
            else:
                mean = helper.make_node("ReduceMean", inputs=[x_in], outputs=[f"{prefix}_mean"], axes=[-1], keepdims=1, name=f"{prefix}_rmean")
                var = helper.make_node("ReduceMean", inputs=[f"{prefix}_sq"], outputs=[f"{prefix}_var"], axes=[-1], keepdims=1, name=f"{prefix}_rvar")

            sub = helper.make_node("Sub", inputs=[x_in, f"{prefix}_mean"], outputs=[f"{prefix}_diff"], name=f"{prefix}_sub")
            sq = helper.make_node("Mul", inputs=[f"{prefix}_diff", f"{prefix}_diff"], outputs=[f"{prefix}_sq"], name=f"{prefix}_sq")
            var_eps = helper.make_node("Add", inputs=[f"{prefix}_var", eps_tensor.name], outputs=[f"{prefix}_vareps"], name=f"{prefix}_add_eps")
            std = helper.make_node("Sqrt", inputs=[f"{prefix}_vareps"], outputs=[f"{prefix}_std"], name=f"{prefix}_sqrt")
            normed = helper.make_node("Div", inputs=[f"{prefix}_diff", f"{prefix}_std"], outputs=[f"{prefix}_normed"], name=f"{prefix}_div")
            scaled = helper.make_node("Mul", inputs=[f"{prefix}_normed", scale_in], outputs=[f"{prefix}_scaled"], name=f"{prefix}_mul_scale")

            if bias_in:
                add_bias = helper.make_node("Add", inputs=[f"{prefix}_scaled", bias_in], outputs=[out_name], name=f"{prefix}_add_bias")
                new_nodes.extend([mean, sub, sq, var, var_eps, std, normed, scaled, add_bias])
            else:
                scaled.output[0] = out_name
                new_nodes.extend([mean, sub, sq, var, var_eps, std, normed, scaled])
            continue

        new_nodes.append(current_node)

        # Inline injection immediately after producer (topologically sound!)
        if chosen_tr_t and chosen_tr_t in current_node.output and not tr_produced:
            ping_node = helper.make_node("Transpose", inputs=[chosen_tr_t], outputs=[tr_ping], perm=tr_perm, name=f"obf_ping_{seed}")
            pong_node = helper.make_node("Transpose", inputs=[tr_ping], outputs=[tr_pong], perm=tr_perm, name=f"obf_pong_{seed}")
            new_nodes.extend([ping_node, pong_node])
            tr_produced = True

        if chosen_id_t and chosen_id_t in current_node.output and not id_produced:
            id_node = helper.make_node("Identity", inputs=[chosen_id_t], outputs=[id_out], name=f"obf_id_{seed}")
            new_nodes.append(id_node)
            id_produced = True

        if chosen_dead_t and chosen_dead_t in current_node.output:
            dead1 = helper.make_node("Mul", inputs=[chosen_dead_t, zero_const.name], outputs=[f"dead1_{seed}"], name=f"obf_dead_mul_{seed}")
            dead2 = helper.make_node("Relu", inputs=[f"dead1_{seed}"], outputs=[f"dead2_{seed}"], name=f"obf_dead_relu_{seed}")
            new_nodes.extend([dead1, dead2])
            chosen_dead_t = None

    # Reconstruct graph
    mutated_graph = helper.make_graph(
        nodes=new_nodes,
        name=f"{graph.name}_mutated_s{seed}",
        inputs=list(graph.input),
        outputs=list(graph.output),
        initializer=list(graph.initializer) + new_inits,
    )

    mutated_model = helper.make_model(
        mutated_graph,
        opset_imports=list(model.opset_import),
        ir_version=min(model.ir_version, 10),
    )
    return mutated_model


# =============================================================================
# 3. Pure ONNX Programmatic Obfuscated Graph Builders
# =============================================================================
def build_pure_onnx_decomposed_block(
    opset: int = 14,
    d_model: int = 32,
    seq_len: int = 8,
    variant: str = "ln_softmax",
) -> onnx.ModelProto:
    """Construct an intentionally decomposed, multi-layer obfuscated block in raw ONNX."""
    nodes = []

    x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, seq_len, d_model])
    out_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, seq_len, d_model])

    zero_const = helper.make_tensor("zero_c", TensorProto.FLOAT, [], [0.0])
    one_const = helper.make_tensor("one_c", TensorProto.FLOAT, [], [1.0])
    two_const = helper.make_tensor("two_c", TensorProto.FLOAT, [], [2.0])
    eps_const = helper.make_tensor("eps_c", TensorProto.FLOAT, [], [1e-5])
    gamma_init = helper.make_tensor("gamma", TensorProto.FLOAT, [d_model], [1.0] * d_model)
    beta_init = helper.make_tensor("beta", TensorProto.FLOAT, [d_model], [0.0] * d_model)
    w_proj = helper.make_tensor("w_proj", TensorProto.FLOAT, [d_model, d_model], np.random.randn(d_model, d_model).astype(np.float32).flatten().tolist())
    axes_neg1 = helper.make_tensor("axes_neg1", TensorProto.INT64, [1], [-1])

    inits = [zero_const, one_const, two_const, eps_const, gamma_init, beta_init, w_proj]
    if opset >= 13:
        inits.append(axes_neg1)

    # Obfuscation 1: Tautological arithmetic (X + 0 - 0)
    nodes.append(helper.make_node("Add", inputs=["X", "zero_c"], outputs=["x_add"], name="p_add0"))
    nodes.append(helper.make_node("Sub", inputs=["x_add", "zero_c"], outputs=["x_clean"], name="p_sub0"))

    # Obfuscation 2: Identity chain
    nodes.append(helper.make_node("Identity", inputs=["x_clean"], outputs=["x_id"], name="p_id0"))

    # Obfuscation 3: Decomposed LayerNorm using Mul(diff, diff)
    if opset >= 18:
        nodes.append(helper.make_node("ReduceMean", inputs=["x_id", "axes_neg1"], outputs=["mean"], keepdims=1, name="p_mean1"))
    else:
        nodes.append(helper.make_node("ReduceMean", inputs=["x_id"], outputs=["mean"], axes=[-1], keepdims=1, name="p_mean1"))
    nodes.append(helper.make_node("Sub", inputs=["x_id", "mean"], outputs=["diff"], name="p_sub1"))
    nodes.append(helper.make_node("Mul", inputs=["diff", "diff"], outputs=["diff_sq"], name="p_sq1"))
    if opset >= 18:
        nodes.append(helper.make_node("ReduceMean", inputs=["diff_sq", "axes_neg1"], outputs=["var"], keepdims=1, name="p_var1"))
    else:
        nodes.append(helper.make_node("ReduceMean", inputs=["diff_sq"], outputs=["var"], axes=[-1], keepdims=1, name="p_var1"))
    nodes.append(helper.make_node("Add", inputs=["var", "eps_c"], outputs=["var_eps"], name="p_add_eps"))
    nodes.append(helper.make_node("Sqrt", inputs=["var_eps"], outputs=["std"], name="p_sqrt"))
    nodes.append(helper.make_node("Div", inputs=["diff", "std"], outputs=["normed"], name="p_div"))
    nodes.append(helper.make_node("Mul", inputs=["normed", "gamma"], outputs=["scaled"], name="p_scale"))
    nodes.append(helper.make_node("Add", inputs=["scaled", "beta"], outputs=["ln_out"], name="p_ln_bias"))

    # Obfuscation 4: Decomposed Softmax
    if opset >= 18:
        nodes.append(helper.make_node("ReduceMax", inputs=["ln_out", "axes_neg1"], outputs=["sm_max"], keepdims=1, name="p_max"))
        nodes.append(helper.make_node("ReduceSum", inputs=["sm_exp", "axes_neg1"], outputs=["sm_sum"], keepdims=1, name="p_sum"))
    elif opset >= 13:
        nodes.append(helper.make_node("ReduceMax", inputs=["ln_out"], outputs=["sm_max"], axes=[-1], keepdims=1, name="p_max"))
        nodes.append(helper.make_node("ReduceSum", inputs=["sm_exp", "axes_neg1"], outputs=["sm_sum"], keepdims=1, name="p_sum"))
    else:
        nodes.append(helper.make_node("ReduceMax", inputs=["ln_out"], outputs=["sm_max"], axes=[-1], keepdims=1, name="p_max"))
        nodes.append(helper.make_node("ReduceSum", inputs=["sm_exp"], outputs=["sm_sum"], axes=[-1], keepdims=1, name="p_sum"))

    nodes.append(helper.make_node("Sub", inputs=["ln_out", "sm_max"], outputs=["sm_sub"], name="p_sub_max"))
    nodes.append(helper.make_node("Exp", inputs=["sm_sub"], outputs=["sm_exp"], name="p_exp"))
    # Reorder sum after exp
    idx_sum = len(nodes) - 2
    nodes.append(helper.make_node("Div", inputs=["sm_exp", "sm_sum"], outputs=["sm_out"], name="p_div_sm"))

    # Linear projection
    nodes.append(helper.make_node("MatMul", inputs=["sm_out", "w_proj"], outputs=["lin_out"], name="p_matmul"))

    # Obfuscation 5: Transpose ping-pong
    nodes.append(helper.make_node("Transpose", inputs=["lin_out"], outputs=["tr1"], perm=[1, 0, 2], name="p_tr1"))
    nodes.append(helper.make_node("Transpose", inputs=["tr1"], outputs=["tr2"], perm=[1, 0, 2], name="p_tr2"))

    # Obfuscation 6: Dead branch
    nodes.append(helper.make_node("Mul", inputs=["tr2", "zero_c"], outputs=["dead_mul"], name="p_dead_mul"))
    nodes.append(helper.make_node("Relu", inputs=["dead_mul"], outputs=["dead_relu"], name="p_dead_relu"))

    # Commutative residual addition: tr2 + X
    nodes.append(helper.make_node("Add", inputs=["tr2", "X"], outputs=["Y"], name="p_out_add"))

    graph = helper.make_graph(
        nodes=nodes,
        name=f"pure_onnx_decomposed_{variant}_opset{opset}",
        inputs=[x_info],
        outputs=[out_info],
        initializer=inits,
    )
    opset_import = helper.make_opsetid("", opset)
    return helper.make_model(graph, opset_imports=[opset_import], ir_version=8)


# =============================================================================
# 4. Multi-Variant Architecture Stress Zoo Exporter
# =============================================================================
def export_obfuscated_stress_zoo() -> list[dict[str, Any]]:
    """Export a comprehensive suite of obfuscated models across deep learning families and opsets."""
    catalog: list[dict[str, Any]] = []
    opsets = [14, 17, 18, 20]

    # --- Category A: Pure ONNX Programmatic Decomposed Blocks ---
    for opset in opsets:
        out_path = OUT_DIR / f"pure_onnx_decomposed_block_opset{opset}.onnx"
        m = build_pure_onnx_decomposed_block(opset=opset, d_model=32, seq_len=8)
        onnx.save(m, str(out_path))
        catalog.append({"name": "pure_onnx_decomposed_block", "family": "pure_onnx", "opset": opset, "path": str(out_path), "nodes": len(m.graph.node)})

    # --- Category B: Multi-Head Attention (Standard, Rotary, GQA) ---
    attn_configs = [
        ("mha_standard", StandardMHA_SeparateQKV(32, 4), (2, 8, 32)),
        ("mha_rotary", RotaryMHA(32, 4), (2, 8, 32)),
        ("mha_gqa", GroupedQueryAttention_GQA(32, 4, 2), (2, 8, 32)),
    ]
    for name, block, dummy_shape in attn_configs:
        dummy = torch.randn(*dummy_shape)
        for idx, opset in enumerate(opsets):
            mutated_mod = PyTorchBlockMutator(block, seed=42 + idx, variant_mode=idx, d_model=32)
            mutated_mod.eval()
            out_path = OUT_DIR / f"obfuscated_{name}_opset{opset}.onnx"
            try:
                torch.onnx.export(
                    mutated_mod,
                    (dummy,),
                    str(out_path),
                    input_names=["x"],
                    output_names=["out"],
                    opset_version=opset,
                    do_constant_folding=True,
                    dynamo=False,
                )
                m = onnx.load(str(out_path))
                # Apply post-export graph mutations (commutative swapping, identity injection, transpose ping-pongs)
                m_mut = mutate_onnx_graph(m, seed=100 + idx, decompose_softmax=True, insert_transpose_ping_pong=True)
                onnx.save(m_mut, str(out_path))
                catalog.append({"name": f"obfuscated_{name}", "family": "attention", "opset": opset, "path": str(out_path), "nodes": len(m_mut.graph.node)})
            except Exception as e:
                print(f"[Obfuscated Zoo] Failed to export {out_path.name}: {e}")

    # --- Category C: Vision & Convolutional (ConvNeXt, MobileNetV2) ---
    vision_configs = [
        ("vision_convnext", ConvNeXt_Block(32), (2, 32, 8, 8)),
        ("vision_mobilenetv2", MobileNetV2_InvertedResidual(32, 64, 32), (2, 32, 8, 8)),
    ]
    for name, block, dummy_shape in vision_configs:
        dummy = torch.randn(*dummy_shape)
        for idx, opset in enumerate(opsets):
            mutated_mod = PyTorchBlockMutator(block, seed=142 + idx, variant_mode=idx, d_model=32)
            mutated_mod.eval()
            out_path = OUT_DIR / f"obfuscated_{name}_opset{opset}.onnx"
            try:
                torch.onnx.export(
                    mutated_mod,
                    (dummy,),
                    str(out_path),
                    input_names=["x"],
                    output_names=["out"],
                    opset_version=opset,
                    do_constant_folding=True,
                    dynamo=False,
                )
                m = onnx.load(str(out_path))
                m_mut = mutate_onnx_graph(m, seed=200 + idx, decompose_layernorm=True)
                onnx.save(m_mut, str(out_path))
                catalog.append({"name": f"obfuscated_{name}", "family": "vision", "opset": opset, "path": str(out_path), "nodes": len(m_mut.graph.node)})
            except Exception as e:
                print(f"[Obfuscated Zoo] Failed to export {out_path.name}: {e}")

    # --- Category D: Modern & Vanilla MLP (DeepMLP, Highway, SwiGLU) ---
    mlp_configs = [
        ("mlp_deepnormact", DeepMLP_Norm_Act(32), (2, 8, 32)),
        ("mlp_highway", HighwayNetwork_Block(32), (2, 8, 32)),
        ("mlp_swiglu", SwiGLU_MLP_Block(32, 64), (2, 8, 32)),
    ]
    for name, block, dummy_shape in mlp_configs:
        dummy = torch.randn(*dummy_shape)
        for idx, opset in enumerate(opsets):
            mutated_mod = PyTorchBlockMutator(block, seed=242 + idx, variant_mode=idx, d_model=32)
            mutated_mod.eval()
            out_path = OUT_DIR / f"obfuscated_{name}_opset{opset}.onnx"
            try:
                torch.onnx.export(
                    mutated_mod,
                    (dummy,),
                    str(out_path),
                    input_names=["x"],
                    output_names=["out"],
                    opset_version=opset,
                    do_constant_folding=True,
                    dynamo=False,
                )
                m = onnx.load(str(out_path))
                m_mut = mutate_onnx_graph(m, seed=300 + idx, decompose_layernorm=True, insert_identities=True)
                onnx.save(m_mut, str(out_path))
                catalog.append({"name": f"obfuscated_{name}", "family": "mlp", "opset": opset, "path": str(out_path), "nodes": len(m_mut.graph.node)})
            except Exception as e:
                print(f"[Obfuscated Zoo] Failed to export {out_path.name}: {e}")

    # --- Category E: State-Space Models (Mamba S6) ---
    mamba_block = Mamba_SelectiveScanBlock(32, 8)
    dummy_mamba = torch.randn(2, 8, 32)
    for idx, opset in enumerate(opsets):
        mutated_mod = PyTorchBlockMutator(mamba_block, seed=342 + idx, variant_mode=idx, d_model=32)
        mutated_mod.eval()
        out_path = OUT_DIR / f"obfuscated_ssm_mamba_opset{opset}.onnx"
        try:
            torch.onnx.export(
                mutated_mod,
                (dummy_mamba,),
                str(out_path),
                input_names=["x"],
                output_names=["out"],
                opset_version=opset,
                do_constant_folding=True,
                dynamo=False,
            )
            m = onnx.load(str(out_path))
            m_mut = mutate_onnx_graph(m, seed=400 + idx, swap_commutative=True)
            onnx.save(m_mut, str(out_path))
            catalog.append({"name": "obfuscated_ssm_mamba", "family": "ssm", "opset": opset, "path": str(out_path), "nodes": len(m_mut.graph.node)})
        except Exception as e:
            print(f"[Obfuscated Zoo] Failed to export {out_path.name}: {e}")

    # --- Category F: Hierarchical ViT (Swin Window Attention) ---
    swin_block = Swin_WindowAttentionBlock(32, 2, 4)
    dummy_swin = torch.randn(2, 4, 4, 32)
    for idx, opset in enumerate(opsets):
        mutated_mod = PyTorchBlockMutator(swin_block, seed=442 + idx, variant_mode=idx, d_model=32)
        mutated_mod.eval()
        out_path = OUT_DIR / f"obfuscated_vit_swin_opset{opset}.onnx"
        try:
            torch.onnx.export(
                mutated_mod,
                (dummy_swin,),
                str(out_path),
                input_names=["x"],
                output_names=["out"],
                opset_version=opset,
                do_constant_folding=True,
                dynamo=False,
            )
            m = onnx.load(str(out_path))
            m_mut = mutate_onnx_graph(m, seed=500 + idx, decompose_softmax=True)
            onnx.save(m_mut, str(out_path))
            catalog.append({"name": "obfuscated_vit_swin", "family": "vit", "opset": opset, "path": str(out_path), "nodes": len(m_mut.graph.node)})
        except Exception as e:
            print(f"[Obfuscated Zoo] Failed to export {out_path.name}: {e}")

    print(f"\n[Obfuscated Stress Zoo] Successfully exported {len(catalog)} obfuscated ONNX models.")
    return catalog


# =============================================================================
# 5. Diagnostic Evaluation & Verification Harness
# =============================================================================
def verify_obfuscated_stress_zoo() -> dict[str, Any]:
    """Run full optimization pipeline and cross-verifier on all obfuscated models."""
    from neuron_toolkit.compiler.pipeline import OptimizationPipeline
    from neuron_toolkit.runtime.cross_verifier import CrossVerifier

    catalog = export_obfuscated_stress_zoo()
    pipeline = OptimizationPipeline(mode="function_proto")

    results = []
    print("\n" + "=" * 105)
    print("                     OBFUSCATED STRESS ZOO ROBUSTNESS EVALUATION REPORT")
    print("=" * 105)
    print(f"{'Model Name':<32} | {'Opset':<5} | {'Orig':<6} | {'Opt':<6} | {'Reduction':<10} | {'Cosine Sim':<11} | {'Status'}")
    print("-" * 105)

    passed_count = 0
    total_count = len(catalog)

    for item in catalog:
        model_path = item["path"]
        name = item["name"]
        opset = item["opset"]

        try:
            raw_model = onnx.load(model_path)
            orig_nodes = len(raw_model.graph.node)

            # Optimize
            opt_model = pipeline.run(raw_model)
            opt_nodes = len(opt_model.graph.node)
            reduction = ((orig_nodes - opt_nodes) / orig_nodes) * 100.0 if orig_nodes > 0 else 0.0

            # Verify differential parity
            verifier = CrossVerifier(raw_model, opt_model)
            report = verifier.verify()
            is_pass = report.all_passed
            min_cos = min((m.cosine_similarity for m in report.metrics.values()), default=1.0)
            status_str = "PASS" if is_pass else "FAIL"

            if is_pass:
                passed_count += 1

            results.append({
                "name": name,
                "opset": opset,
                "orig_nodes": orig_nodes,
                "opt_nodes": opt_nodes,
                "reduction_pct": reduction,
                "cosine_sim": min_cos,
                "passed": is_pass,
            })

            print(f"{name:<32} | {opset:<5} | {orig_nodes:<6} | {opt_nodes:<6} | {reduction:>8.1f} % | {min_cos:>10.6f}  | {status_str}")
        except Exception as e:
            print(f"{name:<32} | {opset:<5} | ERROR: {e}")
            results.append({"name": name, "opset": opset, "passed": False, "error": str(e)})

    print("=" * 105)
    print(f"TOTAL: {passed_count} / {total_count} PASSED ({(passed_count / total_count) * 100:.1f}% Pass Rate)")
    print("=" * 105)
    return {"total": total_count, "passed": passed_count, "results": results}


if __name__ == "__main__":
    verify_obfuscated_stress_zoo()
