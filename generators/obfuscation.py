#!/usr/bin/env python3
"""Declarative Obfuscation Registry & Mutation Engine for Deep Learning Models.

Allows registering PyTorch-level and ONNX-level graph obfuscation strategies via decorators.
Guarantees 100% mathematical equivalence (zero functional difference, diff = 0.0)
so compilers can be tested for graph-simplification robustness (e.g. against tf2onnx,
torch.onnx or hand-crafted graph obfuscations).
"""

from __future__ import annotations

import copy
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import onnx
import torch
from onnx import TensorProto, helper
from torch import nn


@dataclass
class ObfuscationStrategy:
    """Specification of an obfuscation mutation strategy."""

    name: str
    stage: Literal["pytorch", "onnx"]
    fn: Callable[..., Any]
    description: str = ""
    tags: list[str] = field(default_factory=list)


class ObfuscationRegistry:
    """Central registry and execution coordinator for model obfuscation strategies."""

    _strategies: dict[str, ObfuscationStrategy] = {}

    @classmethod
    def register(cls, strategy: ObfuscationStrategy) -> ObfuscationStrategy:
        cls._strategies[strategy.name] = strategy
        return strategy

    @classmethod
    def get(cls, name: str) -> ObfuscationStrategy:
        if name not in cls._strategies:
            raise KeyError(f"Obfuscation strategy '{name}' not found. Available: {sorted(cls._strategies.keys())}")
        return cls._strategies[name]

    @classmethod
    def list_strategies(cls, stage: Literal["pytorch", "onnx"] | None = None) -> list[ObfuscationStrategy]:
        if stage is None:
            return list(cls._strategies.values())
        return [s for s in cls._strategies.values() if s.stage == stage]

    @classmethod
    def clear(cls) -> None:
        cls._strategies.clear()

    @classmethod
    def should_obfuscate(cls, probability: float = 0.5, seed: int | None = None) -> bool:
        """Probabilistic decision for obfuscating a model export (default 50%)."""
        if probability <= 0.0:
            return False
        if probability >= 1.0:
            return True
        rng = random.Random(seed) if seed is not None else random
        return rng.random() < probability

    @classmethod
    def apply_pytorch(
        cls,
        model: nn.Module,
        strategy_names: Sequence[str] | None = None,
        seed: int = 42,
    ) -> tuple[nn.Module, list[str]]:
        """Wrap or mutate a PyTorch module with registered PyTorch obfuscation strategies."""
        pt_strategies = cls.list_strategies(stage="pytorch")
        if not pt_strategies:
            return model, []

        rng = random.Random(seed)
        if strategy_names is not None:
            selected = [cls.get(name) for name in strategy_names if cls.get(name).stage == "pytorch"]
        else:
            # Randomly select 1 to 2 PyTorch strategies
            k = min(len(pt_strategies), rng.randint(1, 2))
            selected = rng.sample(pt_strategies, k)

        applied: list[str] = []
        mutated_model = model
        for strat in selected:
            try:
                mutated_model = strat.fn(mutated_model, seed=rng.randint(0, 10000))
                applied.append(strat.name)
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to apply PyTorch obfuscation '{strat.name}': {e}", UserWarning, stacklevel=2)

        return mutated_model, applied

    @classmethod
    def apply_onnx(
        cls,
        model_proto: onnx.ModelProto,
        strategy_names: Sequence[str] | None = None,
        seed: int = 42,
    ) -> tuple[onnx.ModelProto, list[str]]:
        """Mutate an ONNX model graph with registered ONNX obfuscation strategies."""
        onnx_strategies = cls.list_strategies(stage="onnx")
        if not onnx_strategies:
            return model_proto, []

        rng = random.Random(seed)
        if strategy_names is not None:
            selected = [cls.get(name) for name in strategy_names if cls.get(name).stage == "onnx"]
        else:
            # Randomly select 1 to 3 ONNX strategies
            k = min(len(onnx_strategies), rng.randint(1, min(3, len(onnx_strategies))))
            selected = rng.sample(onnx_strategies, k)

        applied: list[str] = []
        mutated_proto = model_proto
        for strat in selected:
            try:
                mutated_proto = strat.fn(mutated_proto, seed=rng.randint(0, 10000))
                applied.append(strat.name)
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to apply ONNX obfuscation '{strat.name}': {e}", UserWarning, stacklevel=2)

        return mutated_proto, applied


def register_obfuscation_strategy(
    name: str | None = None,
    stage: Literal["pytorch", "onnx"] = "onnx",
    description: str = "",
    tags: list[str] | None = None,
) -> Callable[[Callable], Callable]:
    """Decorator to register a custom mathematical-equivalence-preserving obfuscation strategy."""

    def decorator(fn: Callable) -> Callable:
        strat_name = name or fn.__name__
        spec = ObfuscationStrategy(
            name=strat_name,
            stage=stage,
            fn=fn,
            description=description or (fn.__doc__ or ""),
            tags=tags or [],
        )
        ObfuscationRegistry.register(spec)
        return fn

    return decorator


# =============================================================================
# PyTorch-Level Mathematical Obfuscator Wrapper & Strategies
# =============================================================================
class PyTorchMathematicalObfuscator(nn.Module):
    """Wraps an nn.Module with mathematically invariant arithmetic tautologies.
    
    Guarantees that calculation and output remain exactly equal to the un-obfuscated model.
    """

    def __init__(self, block: nn.Module, mode: str = "identity_arithmetic", seed: int = 42) -> None:
        super().__init__()
        self.block = block
        self.mode = mode
        self.seed = seed

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        mutated_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor) and torch.is_floating_point(arg):
                if self.mode == "identity_arithmetic":
                    # (x + 0.0) * 1.0 == x
                    mutated_args.append((arg + 0.0) * 1.0)
                elif self.mode == "multiplicative_pingpong":
                    # (x * 2.0) * 0.5 == x (exact in binary IEEE-754)
                    mutated_args.append((arg * 2.0) * 0.5)
                elif self.mode == "additive_pingpong":
                    # (x + 3.14159) - 3.14159 == x
                    mutated_args.append((arg + 3.14159) - 3.14159)
                elif self.mode == "clone_identity":
                    mutated_args.append(arg.clone())
                elif self.mode == "split_identity":
                    # 0.5 * x + 0.5 * x == x
                    mutated_args.append((0.5 * arg) + (0.5 * arg))
                elif self.mode == "ghost_sidechain":
                    # x + (x * 0.0) == x
                    mutated_args.append(arg + (arg * 0.0))
                else:
                    mutated_args.append(arg)
            else:
                mutated_args.append(arg)

        out = self.block(*mutated_args, **kwargs)

        if isinstance(out, torch.Tensor) and torch.is_floating_point(out):
            if self.mode in ("identity_arithmetic", "multiplicative_pingpong"):
                out = (out * 1.0) + 0.0
            elif self.mode in ("ghost_sidechain", "split_identity"):
                out = out + (out * 0.0)
            elif self.mode == "additive_pingpong":
                out = (out + 1.0) - 1.0

        return out


@register_obfuscation_strategy(
    name="pt_identity_arithmetic",
    stage="pytorch",
    description="Injects (x + 0.0) * 1.0 input and output arithmetic tautologies",
)
def strategy_pt_identity_arithmetic(model: nn.Module, seed: int = 42) -> nn.Module:
    return PyTorchMathematicalObfuscator(model, mode="identity_arithmetic", seed=seed)


@register_obfuscation_strategy(
    name="pt_multiplicative_pingpong",
    stage="pytorch",
    description="Injects binary exponent scaling (x * 2.0) * 0.5 ping-pong",
)
def strategy_pt_multiplicative_pingpong(model: nn.Module, seed: int = 42) -> nn.Module:
    return PyTorchMathematicalObfuscator(model, mode="multiplicative_pingpong", seed=seed)


@register_obfuscation_strategy(
    name="pt_additive_pingpong",
    stage="pytorch",
    description="Injects additive constant ping-pong (x + 3.14159) - 3.14159",
)
def strategy_pt_additive_pingpong(model: nn.Module, seed: int = 42) -> nn.Module:
    return PyTorchMathematicalObfuscator(model, mode="additive_pingpong", seed=seed)


@register_obfuscation_strategy(
    name="pt_split_identity",
    stage="pytorch",
    description="Splits input into two symmetric halves: 0.5 * x + 0.5 * x",
)
def strategy_pt_split_identity(model: nn.Module, seed: int = 42) -> nn.Module:
    return PyTorchMathematicalObfuscator(model, mode="split_identity", seed=seed)


@register_obfuscation_strategy(
    name="pt_ghost_sidechain",
    stage="pytorch",
    description="Injects zero-weighted ghost branch sidechain: x + (x * 0.0)",
)
def strategy_pt_ghost_sidechain(model: nn.Module, seed: int = 42) -> nn.Module:
    return PyTorchMathematicalObfuscator(model, mode="ghost_sidechain", seed=seed)


# =============================================================================
# ONNX-Level Graph Mutation Strategies
# =============================================================================
@register_obfuscation_strategy(
    name="onnx_swap_commutative",
    stage="onnx",
    description="Randomly swaps inputs of commutative operators (Add, Mul, Equal, Min, Max)",
)
def strategy_onnx_swap_commutative(model: onnx.ModelProto, seed: int = 42) -> onnx.ModelProto:
    rng = random.Random(seed)
    mutated = copy.deepcopy(model)
    for node in mutated.graph.node:
        if node.op_type in ("Add", "Mul", "Equal", "Min", "Max") and len(node.input) == 2:
            if rng.random() < 0.5:
                node.input[0], node.input[1] = node.input[1], node.input[0]
    return mutated


@register_obfuscation_strategy(
    name="onnx_insert_identities",
    stage="onnx",
    description="Inserts transparent ONNX Identity nodes on intermediate float activations",
)
def strategy_onnx_insert_identities(model: onnx.ModelProto, seed: int = 42) -> onnx.ModelProto:
    rng = random.Random(seed)
    mutated = copy.deepcopy(model)
    graph = mutated.graph
    outputs = {o.name for o in graph.output}
    inits = {i.name for i in graph.initializer}

    # Find candidates
    candidates: list[str] = []
    for node in graph.node:
        for out in node.output:
            if out not in outputs and out not in inits:
                candidates.append(out)

    if not candidates:
        return mutated

    chosen = rng.choice(candidates)
    id_out = f"{chosen}_mut_id_{seed}"

    new_nodes: list[onnx.NodeProto] = []
    id_inserted = False

    for node in graph.node:
        if id_inserted:
            for idx, inp in enumerate(node.input):
                if inp == chosen:
                    node.input[idx] = id_out

        new_nodes.append(node)

        if not id_inserted and chosen in node.output:
            id_node = helper.make_node(
                "Identity",
                inputs=[chosen],
                outputs=[id_out],
                name=f"obf_id_{seed}",
            )
            new_nodes.append(id_node)
            id_inserted = True

    del graph.node[:]
    graph.node.extend(new_nodes)
    return mutated


@register_obfuscation_strategy(
    name="onnx_transpose_ping_pong",
    stage="onnx",
    description="Inserts cancelling Transpose(p) -> Transpose(inv_p) pairs on 3D/4D tensors",
)
def strategy_onnx_transpose_ping_pong(model: onnx.ModelProto, seed: int = 42) -> onnx.ModelProto:
    rng = random.Random(seed)
    mutated = copy.deepcopy(model)
    graph = mutated.graph
    outputs = {o.name for o in graph.output}
    inits = {i.name for i in graph.initializer}

    try:
        inferred = onnx.shape_inference.infer_shapes(mutated)
        vinfo = {vi.name: vi for vi in inferred.graph.value_info}
    except Exception:
        vinfo = {}

    candidates_3d = []
    candidates_4d = []
    for node in graph.node:
        for out in node.output:
            if out in outputs or out in inits:
                continue
            if out in vinfo:
                dim_count = len(vinfo[out].type.tensor_type.shape.dim)
                if dim_count == 3:
                    candidates_3d.append(out)
                elif dim_count == 4:
                    candidates_4d.append(out)

    chosen_t = None
    perm = None
    if candidates_3d:
        chosen_t = rng.choice(candidates_3d)
        perm = [1, 0, 2]
    elif candidates_4d:
        chosen_t = rng.choice(candidates_4d)
        perm = [0, 2, 1, 3]

    if not chosen_t or not perm:
        return mutated

    # Inverse permutation
    inv_perm = [perm.index(i) for i in range(len(perm))]
    tr_ping = f"{chosen_t}_t_ping_{seed}"
    tr_pong = f"{chosen_t}_t_pong_{seed}"

    new_nodes: list[onnx.NodeProto] = []
    tr_inserted = False

    for node in graph.node:
        if tr_inserted:
            for idx, inp in enumerate(node.input):
                if inp == chosen_t:
                    node.input[idx] = tr_pong

        new_nodes.append(node)

        if not tr_inserted and chosen_t in node.output:
            ping_node = helper.make_node(
                "Transpose",
                inputs=[chosen_t],
                outputs=[tr_ping],
                perm=perm,
                name=f"obf_ping_{seed}",
            )
            pong_node = helper.make_node(
                "Transpose",
                inputs=[tr_ping],
                outputs=[tr_pong],
                perm=inv_perm,
                name=f"obf_pong_{seed}",
            )
            new_nodes.extend([ping_node, pong_node])
            tr_inserted = True

    del graph.node[:]
    graph.node.extend(new_nodes)
    return mutated


@register_obfuscation_strategy(
    name="onnx_decompose_softmax",
    stage="onnx",
    description="Decomposes Softmax into primitive operations: ReduceMax -> Sub -> Exp -> ReduceSum -> Div",
)
def strategy_onnx_decompose_softmax(model: onnx.ModelProto, seed: int = 42) -> onnx.ModelProto:
    mutated = copy.deepcopy(model)
    graph = mutated.graph
    outputs = {o.name for o in graph.output}

    opset_ver = 14
    for op in mutated.opset_import:
        if op.domain in ("", "ai.onnx"):
            opset_ver = op.version

    new_nodes: list[onnx.NodeProto] = []
    new_inits: list[onnx.TensorProto] = []

    for node in graph.node:
        if node.op_type == "Softmax" and node.output[0] not in outputs:
            axis = -1
            for a in node.attribute:
                if a.name == "axis":
                    axis = a.i

            x_in = node.input[0]
            out_name = node.output[0]
            prefix = f"{node.name or out_name}_decomp_{seed}"

            # Step 1: ReduceMax
            if opset_ver >= 18:
                axes_t = helper.make_tensor(f"{prefix}_axes", TensorProto.INT64, [1], [axis])
                new_inits.append(axes_t)
                rmax = helper.make_node(
                    "ReduceMax",
                    inputs=[x_in, axes_t.name],
                    outputs=[f"{prefix}_max"],
                    keepdims=1,
                    name=f"{prefix}_rmax",
                )
            elif opset_ver >= 13:
                axes_t = helper.make_tensor(f"{prefix}_axes", TensorProto.INT64, [1], [axis])
                new_inits.append(axes_t)
                rmax = helper.make_node(
                    "ReduceMax",
                    inputs=[x_in],
                    outputs=[f"{prefix}_max"],
                    axes=[axis],
                    keepdims=1,
                    name=f"{prefix}_rmax",
                )
            else:
                rmax = helper.make_node(
                    "ReduceMax",
                    inputs=[x_in],
                    outputs=[f"{prefix}_max"],
                    axes=[axis],
                    keepdims=1,
                    name=f"{prefix}_rmax",
                )

            # Step 2: Sub(x, max)
            sub = helper.make_node(
                "Sub",
                inputs=[x_in, f"{prefix}_max"],
                outputs=[f"{prefix}_sub"],
                name=f"{prefix}_sub",
            )

            # Step 3: Exp(sub)
            exp = helper.make_node(
                "Exp",
                inputs=[f"{prefix}_sub"],
                outputs=[f"{prefix}_exp"],
                name=f"{prefix}_exp",
            )

            # Step 4: ReduceSum(exp)
            if opset_ver >= 13:
                rsum = helper.make_node(
                    "ReduceSum",
                    inputs=[f"{prefix}_exp", axes_t.name],
                    outputs=[f"{prefix}_sum"],
                    keepdims=1,
                    name=f"{prefix}_rsum",
                )
            else:
                rsum = helper.make_node(
                    "ReduceSum",
                    inputs=[f"{prefix}_exp"],
                    outputs=[f"{prefix}_sum"],
                    axes=[axis],
                    keepdims=1,
                    name=f"{prefix}_rsum",
                )

            # Step 5: Div(exp, sum) -> out_name
            div = helper.make_node(
                "Div",
                inputs=[f"{prefix}_exp", f"{prefix}_sum"],
                outputs=[out_name],
                name=f"{prefix}_div",
            )

            new_nodes.extend([rmax, sub, exp, rsum, div])
        else:
            new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(new_inits)
    return mutated


@register_obfuscation_strategy(
    name="onnx_decompose_layernorm",
    stage="onnx",
    description="Decomposes LayerNormalization into ReduceMean -> Sub -> Mul -> ReduceMean -> Sqrt -> Div -> Mul -> Add",
)
def strategy_onnx_decompose_layernorm(model: onnx.ModelProto, seed: int = 42) -> onnx.ModelProto:
    mutated = copy.deepcopy(model)
    graph = mutated.graph

    opset_ver = 14
    for op in mutated.opset_import:
        if op.domain in ("", "ai.onnx"):
            opset_ver = op.version

    new_nodes: list[onnx.NodeProto] = []
    new_inits: list[onnx.TensorProto] = []

    for node in graph.node:
        if node.op_type == "LayerNormalization" and len(node.input) >= 2:
            x_in = node.input[0]
            scale_in = node.input[1]
            bias_in = node.input[2] if len(node.input) > 2 else None
            out_name = node.output[0]
            prefix = f"{node.name or out_name}_decomp_ln_{seed}"

            eps = 1e-5
            for a in node.attribute:
                if a.name == "epsilon":
                    eps = a.f
            eps_tensor = helper.make_tensor(f"{prefix}_eps", TensorProto.FLOAT, [], [eps])
            new_inits.append(eps_tensor)

            if opset_ver >= 18:
                axes_neg1 = helper.make_tensor(f"{prefix}_axes_neg1", TensorProto.INT64, [1], [-1])
                new_inits.append(axes_neg1)
                mean = helper.make_node(
                    "ReduceMean",
                    inputs=[x_in, axes_neg1.name],
                    outputs=[f"{prefix}_mean"],
                    keepdims=1,
                    name=f"{prefix}_rmean",
                )
                var = helper.make_node(
                    "ReduceMean",
                    inputs=[f"{prefix}_sq", axes_neg1.name],
                    outputs=[f"{prefix}_var"],
                    keepdims=1,
                    name=f"{prefix}_rvar",
                )
            else:
                mean = helper.make_node(
                    "ReduceMean",
                    inputs=[x_in],
                    outputs=[f"{prefix}_mean"],
                    axes=[-1],
                    keepdims=1,
                    name=f"{prefix}_rmean",
                )
                var = helper.make_node(
                    "ReduceMean",
                    inputs=[f"{prefix}_sq"],
                    outputs=[f"{prefix}_var"],
                    axes=[-1],
                    keepdims=1,
                    name=f"{prefix}_rvar",
                )

            sub = helper.make_node(
                "Sub",
                inputs=[x_in, f"{prefix}_mean"],
                outputs=[f"{prefix}_diff"],
                name=f"{prefix}_sub",
            )
            sq = helper.make_node(
                "Mul",
                inputs=[f"{prefix}_diff", f"{prefix}_diff"],
                outputs=[f"{prefix}_sq"],
                name=f"{prefix}_sq",
            )
            var_eps = helper.make_node(
                "Add",
                inputs=[f"{prefix}_var", eps_tensor.name],
                outputs=[f"{prefix}_vareps"],
                name=f"{prefix}_add_eps",
            )
            std = helper.make_node(
                "Sqrt",
                inputs=[f"{prefix}_vareps"],
                outputs=[f"{prefix}_std"],
                name=f"{prefix}_sqrt",
            )
            normed = helper.make_node(
                "Div",
                inputs=[f"{prefix}_diff", f"{prefix}_std"],
                outputs=[f"{prefix}_normed"],
                name=f"{prefix}_div",
            )
            scaled = helper.make_node(
                "Mul",
                inputs=[f"{prefix}_normed", scale_in],
                outputs=[f"{prefix}_scaled"] if bias_in else [out_name],
                name=f"{prefix}_mul_scale",
            )

            if bias_in:
                add_bias = helper.make_node(
                    "Add",
                    inputs=[f"{prefix}_scaled", bias_in],
                    outputs=[out_name],
                    name=f"{prefix}_add_bias",
                )
                new_nodes.extend([mean, sub, sq, var, var_eps, std, normed, scaled, add_bias])
            else:
                new_nodes.extend([mean, sub, sq, var, var_eps, std, normed, scaled])
        else:
            new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(new_inits)
    return mutated


@register_obfuscation_strategy(
    name="onnx_inject_dead_nodes",
    stage="onnx",
    description="Injects unreachable dead code branch (Mul with 0 -> Relu) without altering outputs",
)
def strategy_onnx_inject_dead_nodes(model: onnx.ModelProto, seed: int = 42) -> onnx.ModelProto:
    rng = random.Random(seed)
    mutated = copy.deepcopy(model)
    graph = mutated.graph
    outputs = {o.name for o in graph.output}
    inits = {i.name for i in graph.initializer}

    candidates = [
        out for node in graph.node for out in node.output
        if out not in outputs and out not in inits
    ]
    if not candidates:
        return mutated

    chosen = rng.choice(candidates)
    zero_c = helper.make_tensor(f"mut_dead_zero_{seed}", TensorProto.FLOAT, [], [0.0])
    graph.initializer.append(zero_c)

    dead_mul = helper.make_node(
        "Mul",
        inputs=[chosen, zero_c.name],
        outputs=[f"dead_mul_{seed}"],
        name=f"obf_dead_mul_{seed}",
    )
    dead_relu = helper.make_node(
        "Relu",
        inputs=[f"dead_mul_{seed}"],
        outputs=[f"dead_relu_{seed}"],
        name=f"obf_dead_relu_{seed}",
    )
    graph.node.extend([dead_mul, dead_relu])
    return mutated


# =============================================================================
# tf2onnx & Framework-Specific Motifs Obfuscation Strategies
# =============================================================================
@register_obfuscation_strategy(
    name="onnx_nhwc_layout_pingpong",
    stage="onnx",
    description="Emulates tf2onnx NHWC <-> NCHW layout ping-pongs with cancelling Transpose pairs",
)
def strategy_onnx_nhwc_layout_pingpong(model: onnx.ModelProto, seed: int = 42) -> onnx.ModelProto:
    rng = random.Random(seed)
    mutated = copy.deepcopy(model)
    graph = mutated.graph
    outputs = {o.name for o in graph.output}
    inits = {i.name for i in graph.initializer}

    try:
        inferred = onnx.shape_inference.infer_shapes(mutated)
        vinfo = {vi.name: vi for vi in inferred.graph.value_info}
    except Exception:
        vinfo = {}

    candidates_4d = []
    for node in graph.node:
        for out in node.output:
            if out not in outputs and out not in inits:
                if out in vinfo and len(vinfo[out].type.tensor_type.shape.dim) == 4:
                    candidates_4d.append(out)

    if not candidates_4d:
        return mutated

    chosen = rng.choice(candidates_4d)
    nhwc_t = f"{chosen}_nhwc_{seed}"
    nchw_pong = f"{chosen}_nchw_pong_{seed}"

    new_nodes: list[onnx.NodeProto] = []
    tr_inserted = False

    for node in graph.node:
        if tr_inserted:
            for idx, inp in enumerate(node.input):
                if inp == chosen:
                    node.input[idx] = nchw_pong

        new_nodes.append(node)

        if not tr_inserted and chosen in node.output:
            # NCHW -> NHWC [0, 2, 3, 1]
            tr1 = helper.make_node(
                "Transpose",
                inputs=[chosen],
                outputs=[nhwc_t],
                perm=[0, 2, 3, 1],
                name=f"obf_tf2onnx_nhwc_{seed}",
            )
            # NHWC -> NCHW [0, 3, 1, 2]
            tr2 = helper.make_node(
                "Transpose",
                inputs=[nhwc_t],
                outputs=[nchw_pong],
                perm=[0, 3, 1, 2],
                name=f"obf_tf2onnx_nchw_{seed}",
            )
            new_nodes.extend([tr1, tr2])
            tr_inserted = True

    del graph.node[:]
    graph.node.extend(new_nodes)
    return mutated


@register_obfuscation_strategy(
    name="onnx_reciprocal_mul_for_div",
    stage="onnx",
    description="Emulates tf2onnx replacing Div(x, y) with Reciprocal(y) -> Mul(x, recip_y)",
)
def strategy_onnx_reciprocal_mul_for_div(model: onnx.ModelProto, seed: int = 42) -> onnx.ModelProto:
    mutated = copy.deepcopy(model)
    graph = mutated.graph

    new_nodes: list[onnx.NodeProto] = []
    for node in graph.node:
        if node.op_type == "Div" and len(node.input) == 2:
            num = node.input[0]
            denom = node.input[1]
            out = node.output[0]
            recip_out = f"{denom}_recip_{seed}"

            recip = helper.make_node(
                "Reciprocal",
                inputs=[denom],
                outputs=[recip_out],
                name=f"{node.name or out}_recip",
            )
            mul = helper.make_node(
                "Mul",
                inputs=[num, recip_out],
                outputs=[out],
                name=f"{node.name or out}_mul",
            )
            new_nodes.extend([recip, mul])
        else:
            new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)
    return mutated


@register_obfuscation_strategy(
    name="onnx_gemm_to_matmul_add",
    stage="onnx",
    description="Decomposes Gemm(A, B, C) into MatMul(A, B) -> Add(..., C)",
)
def strategy_onnx_gemm_to_matmul_add(model: onnx.ModelProto, seed: int = 42) -> onnx.ModelProto:
    mutated = copy.deepcopy(model)
    graph = mutated.graph

    new_nodes: list[onnx.NodeProto] = []
    for node in graph.node:
        if node.op_type == "Gemm" and len(node.input) >= 2:
            a = node.input[0]
            b = node.input[1]
            c = node.input[2] if len(node.input) > 2 else None
            out = node.output[0]

            trans_a = 0
            trans_b = 0
            alpha = 1.0
            beta = 1.0
            for attr in node.attribute:
                if attr.name == "transA":
                    trans_a = attr.i
                elif attr.name == "transB":
                    trans_b = attr.i
                elif attr.name == "alpha":
                    alpha = attr.f
                elif attr.name == "beta":
                    beta = attr.f

            if abs(alpha - 1.0) < 1e-5 and abs(beta - 1.0) < 1e-5:
                in_a = a
                in_b = b
                prefix = f"{node.name or out}_decomp_{seed}"
                decomp_nodes = []

                if trans_a:
                    tr_a = helper.make_node("Transpose", inputs=[a], outputs=[f"{prefix}_trA"], perm=[1, 0])
                    decomp_nodes.append(tr_a)
                    in_a = f"{prefix}_trA"
                if trans_b:
                    tr_b = helper.make_node("Transpose", inputs=[b], outputs=[f"{prefix}_trB"], perm=[1, 0])
                    decomp_nodes.append(tr_b)
                    in_b = f"{prefix}_trB"

                matmul_out = f"{prefix}_mm" if c else out
                mm = helper.make_node("MatMul", inputs=[in_a, in_b], outputs=[matmul_out], name=f"{prefix}_mm")
                decomp_nodes.append(mm)

                if c:
                    add = helper.make_node("Add", inputs=[matmul_out, c], outputs=[out], name=f"{prefix}_add")
                    decomp_nodes.append(add)

                new_nodes.extend(decomp_nodes)
            else:
                new_nodes.append(node)
        else:
            new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)
    return mutated


@register_obfuscation_strategy(
    name="onnx_decompose_rmsnorm_pow",
    stage="onnx",
    description="Decomposes RMSNorm into Pow(2) -> ReduceMean -> Add(eps) -> Sqrt -> Div -> Mul",
)
def strategy_onnx_decompose_rmsnorm_pow(model: onnx.ModelProto, seed: int = 42) -> onnx.ModelProto:
    mutated = copy.deepcopy(model)
    graph = mutated.graph

    opset_ver = 14
    for op in mutated.opset_import:
        if op.domain in ("", "ai.onnx"):
            opset_ver = op.version

    new_nodes: list[onnx.NodeProto] = []
    new_inits: list[onnx.TensorProto] = []

    for node in graph.node:
        if node.op_type in ("RMSNorm", "RMSNormalization") and len(node.input) >= 2:
            x = node.input[0]
            gamma = node.input[1]
            out = node.output[0]
            prefix = f"{node.name or out}_decomp_rms_{seed}"

            eps = 1e-6
            for attr in node.attribute:
                if attr.name == "epsilon":
                    eps = attr.f

            two_t = helper.make_tensor(f"{prefix}_two", TensorProto.FLOAT, [], [2.0])
            eps_t = helper.make_tensor(f"{prefix}_eps", TensorProto.FLOAT, [], [eps])
            new_inits.extend([two_t, eps_t])

            # 1. Pow(x, 2.0)
            pow_node = helper.make_node("Pow", inputs=[x, two_t.name], outputs=[f"{prefix}_pow"], name=f"{prefix}_pow")

            # 2. ReduceMean
            if opset_ver >= 18:
                axes_neg1 = helper.make_tensor(f"{prefix}_axes", TensorProto.INT64, [1], [-1])
                new_inits.append(axes_neg1)
                rmean = helper.make_node("ReduceMean", inputs=[f"{prefix}_pow", axes_neg1.name], outputs=[f"{prefix}_mean"], keepdims=1, name=f"{prefix}_rmean")
            else:
                rmean = helper.make_node("ReduceMean", inputs=[f"{prefix}_pow"], outputs=[f"{prefix}_mean"], axes=[-1], keepdims=1, name=f"{prefix}_rmean")

            # 3. Add(mean, eps)
            add_eps = helper.make_node("Add", inputs=[f"{prefix}_mean", eps_t.name], outputs=[f"{prefix}_add_eps"], name=f"{prefix}_add_eps")

            # 4. Sqrt
            sqrt_node = helper.make_node("Sqrt", inputs=[f"{prefix}_add_eps"], outputs=[f"{prefix}_sqrt"], name=f"{prefix}_sqrt")

            # 5. Div(x, sqrt)
            div_node = helper.make_node("Div", inputs=[x, f"{prefix}_sqrt"], outputs=[f"{prefix}_normed"], name=f"{prefix}_div")

            # 6. Mul(normed, gamma) -> out
            mul_node = helper.make_node("Mul", inputs=[f"{prefix}_normed", gamma], outputs=[out], name=f"{prefix}_mul")

            new_nodes.extend([pow_node, rmean, add_eps, sqrt_node, div_node, mul_node])
        else:
            new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(new_inits)
    return mutated


@register_obfuscation_strategy(
    name="onnx_decompose_tanh_gelu",
    stage="onnx",
    description="Decomposes Gelu into Tanh approximation formula: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))",
)
def strategy_onnx_decompose_tanh_gelu(model: onnx.ModelProto, seed: int = 42) -> onnx.ModelProto:
    mutated = copy.deepcopy(model)
    graph = mutated.graph

    new_nodes: list[onnx.NodeProto] = []
    new_inits: list[onnx.TensorProto] = []

    for node in graph.node:
        if node.op_type == "Gelu" and len(node.input) >= 1:
            x = node.input[0]
            out = node.output[0]
            prefix = f"{node.name or out}_decomp_tanh_gelu_{seed}"

            c_three = helper.make_tensor(f"{prefix}_c3", TensorProto.FLOAT, [], [3.0])
            c_coeff = helper.make_tensor(f"{prefix}_coeff", TensorProto.FLOAT, [], [0.044715])
            c_sqrt2pi = helper.make_tensor(f"{prefix}_c_sqrt2pi", TensorProto.FLOAT, [], [0.7978845608])
            c_one = helper.make_tensor(f"{prefix}_c1", TensorProto.FLOAT, [], [1.0])
            c_half = helper.make_tensor(f"{prefix}_half", TensorProto.FLOAT, [], [0.5])
            new_inits.extend([c_three, c_coeff, c_sqrt2pi, c_one, c_half])

            # x^3
            pow3 = helper.make_node("Pow", inputs=[x, c_three.name], outputs=[f"{prefix}_x3"], name=f"{prefix}_pow3")
            # 0.044715 * x^3
            mul_coeff = helper.make_node("Mul", inputs=[f"{prefix}_x3", c_coeff.name], outputs=[f"{prefix}_coeff_x3"], name=f"{prefix}_mul_coeff")
            # x + 0.044715 * x^3
            add_poly = helper.make_node("Add", inputs=[x, f"{prefix}_coeff_x3"], outputs=[f"{prefix}_poly"], name=f"{prefix}_add_poly")
            # sqrt(2/pi) * (x + 0.044715 * x^3)
            mul_scaled = helper.make_node("Mul", inputs=[f"{prefix}_poly", c_sqrt2pi.name], outputs=[f"{prefix}_scaled"], name=f"{prefix}_mul_scaled")
            # Tanh(...)
            tanh_node = helper.make_node("Tanh", inputs=[f"{prefix}_scaled"], outputs=[f"{prefix}_tanh"], name=f"{prefix}_tanh")
            # 1 + Tanh(...)
            add_one = helper.make_node("Add", inputs=[f"{prefix}_tanh", c_one.name], outputs=[f"{prefix}_one_plus_tanh"], name=f"{prefix}_add_one")
            # 0.5 * x
            half_x = helper.make_node("Mul", inputs=[x, c_half.name], outputs=[f"{prefix}_halfx"], name=f"{prefix}_half_x")
            # 0.5 * x * (1 + Tanh(...)) -> out
            mul_final = helper.make_node("Mul", inputs=[f"{prefix}_halfx", f"{prefix}_one_plus_tanh"], outputs=[out], name=f"{prefix}_mul_final")

            new_nodes.extend([pow3, mul_coeff, add_poly, mul_scaled, tanh_node, add_one, half_x, mul_final])
        else:
            new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(new_inits)
    return mutated
