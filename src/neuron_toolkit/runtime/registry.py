"""OpRegistry for PyTorch Runtime Graph Execution Engine."""

from __future__ import annotations

import ast
import logging
import operator
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger("neuron_toolkit.runtime.registry")

_SAFE_MATH_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval_math(expr_str: str, env: dict[str, Any]) -> Any:
    """Safely evaluate mathematical expressions using AST parsing without arbitrary code execution."""
    tree = ast.parse(expr_str, mode="eval")

    def _eval_node(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            raise ValueError(f"Unknown variable in Math expr: {node.id}")
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type in _SAFE_MATH_OPS:
                left = _eval_node(node.left)
                right = _eval_node(node.right)
                return _SAFE_MATH_OPS[op_type](left, right)
            raise ValueError(f"Unsupported binary op: {op_type}")
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type in _SAFE_MATH_OPS:
                operand = _eval_node(node.operand)
                return _SAFE_MATH_OPS[op_type](operand)
            raise ValueError(f"Unsupported unary op: {op_type}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in env:
                fn = env[node.func.id]
                args = [_eval_node(a) for a in node.args]
                return fn(*args)
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in env
            ):
                mod = env[node.func.value.id]
                fn = getattr(mod, node.func.attr)
                args = [_eval_node(a) for a in node.args]
                return fn(*args)
            raise ValueError(f"Unsupported call: {node.func}")
        raise ValueError(f"Unsupported AST node: {type(node)}")

    return _eval_node(tree)


# Op handler signature: (inputs: list[torch.Tensor], attrs: dict[str, Any], initializers: dict[str, torch.Tensor]) -> torch.Tensor | list[torch.Tensor]
OpHandler = Callable[..., Any]


class OpRegistry:
    """Registry of PyTorch implementations for ONNX standard and custom operations."""

    def __init__(self) -> None:
        self._registry: dict[tuple[str, str], OpHandler] = {}
        self._register_builtins()

    def register(
        self,
        domain: str,
        op_type: str,
        handler: OpHandler | None = None,
    ) -> Callable[[OpHandler], OpHandler] | None:
        """Register a PyTorch handler for (domain, op_type)."""

        def decorator(fn: OpHandler) -> OpHandler:
            self._registry[(domain, op_type)] = fn
            return fn

        if handler is not None:
            self._registry[(domain, op_type)] = handler
            return None
        return decorator

    def get(self, domain: str, op_type: str) -> OpHandler | None:
        """Retrieve handler for (domain, op_type) with fallback to default domain and prefix matches."""
        if (domain, op_type) in self._registry:
            return self._registry[(domain, op_type)]
        if ("", op_type) in self._registry:
            return self._registry[("", op_type)]
        # Prefix fallback for monomorphized specialized custom ops
        for prefix, base_op in [
            ("DeepMLP_", "DeepMLP_Norm_Act"),
            ("FeedForward_", "FeedForward"),
            ("FeedForwardSwiGLU_", "FeedForwardSwiGLU"),
            ("GatedUnit_", "GatedUnit"),
            ("GatedAttention_", "GatedAttention"),
        ]:
            if op_type.startswith(prefix):
                if (domain, base_op) in self._registry:
                    return self._registry[(domain, base_op)]
                if ("", base_op) in self._registry:
                    return self._registry[("", base_op)]
        return None

    def _register_builtins(self) -> None:
        """Register built-in ONNX standard operators and custom PyTorch ops."""

        # -------------------------------------------------------------
        # Basic Math
        # -------------------------------------------------------------
        @self.register("", "Add")
        def _add(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            return inputs[0] + inputs[1]

        @self.register("", "Sub")
        def _sub(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            return inputs[0] - inputs[1]

        @self.register("", "Mul")
        def _mul(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            return inputs[0] * inputs[1]

        @self.register("", "Div")
        def _div(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            if inputs[0].dtype in (
                torch.int64,
                torch.int32,
                torch.int16,
                torch.int8,
                torch.uint8,
            ):
                return torch.div(inputs[0], inputs[1], rounding_mode="trunc")
            return inputs[0] / inputs[1]

        @self.register("", "Neg")
        def _neg(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            return -inputs[0]

        @self.register("", "Abs")
        def _abs(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            return torch.abs(inputs[0])

        @self.register("", "CumSum")
        def _cumsum(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            x = inputs[0]
            axis = (
                int(inputs[1].item())
                if len(inputs) > 1 and inputs[1] is not None
                else int(attrs.get("axis", 0))
            )
            exclusive = int(attrs.get("exclusive", 0))
            reverse = int(attrs.get("reverse", 0))
            if reverse:
                x = torch.flip(x, [axis])
            res = torch.cumsum(x, dim=axis)
            if exclusive:
                res = res - x
            if reverse:
                res = torch.flip(res, [axis])
            return res

        @self.register("", "Sqrt")
        def _sqrt(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            return torch.sqrt(inputs[0])

        @self.register("", "Reciprocal")
        def _reciprocal(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            return torch.reciprocal(inputs[0])

        @self.register("", "Pow")
        def _pow(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            return torch.pow(inputs[0], inputs[1])

        @self.register("", "Mod")
        def _mod(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            a, b = inputs[0], inputs[1]
            fmod = attrs.get("fmod", 0)
            if fmod == 1:
                return torch.fmod(a, b)
            return torch.remainder(a, b)

        @self.register("", "Erf")
        def _erf(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            return torch.erf(inputs[0])

        @self.register("", "Exp")
        def _exp(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            return torch.exp(inputs[0])

        @self.register("", "Log")
        def _log(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            return torch.log(inputs[0])

        @self.register("", "Cos")
        def _cos(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            return torch.cos(inputs[0])

        @self.register("", "Sin")
        def _sin(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            return torch.sin(inputs[0])

        @self.register("", "Equal")
        def _equal(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            return torch.eq(inputs[0], inputs[1])

        @self.register("", "Greater")
        def _greater(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            return torch.gt(inputs[0], inputs[1])

        @self.register("", "Less")
        def _less(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            return torch.lt(inputs[0], inputs[1])

        @self.register("", "GreaterOrEqual")
        def _greater_equal(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            return torch.ge(inputs[0], inputs[1])

        @self.register("", "LessOrEqual")
        def _less_equal(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            return torch.le(inputs[0], inputs[1])

        @self.register("", "Where")
        def _where(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            cond = inputs[0] if inputs[0].dtype == torch.bool else inputs[0].bool()
            return torch.where(cond, inputs[1], inputs[2])

        @self.register("", "Not")
        def _not(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            cond = inputs[0] if inputs[0].dtype == torch.bool else inputs[0].bool()
            return torch.logical_not(cond)

        @self.register("", "Pad")
        def _pad(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            data = inputs[0]
            if len(inputs) > 1 and inputs[1] is not None:
                pads_list = [int(p) for p in inputs[1].view(-1).tolist()]
            else:
                pads_list = list(attrs.get("pads", []))

            val = 0.0
            if len(inputs) > 2 and inputs[2] is not None and inputs[2].numel() > 0:
                val = float(inputs[2].item())
            elif "value" in attrs:
                val = float(attrs["value"])

            axes = None
            if len(inputs) > 3 and inputs[3] is not None and inputs[3].numel() > 0:
                axes = [int(a) for a in inputs[3].view(-1).tolist()]

            r = data.dim()
            mode = attrs.get("mode", "constant")
            if isinstance(mode, bytes):
                mode = mode.decode("utf-8")

            if axes is not None:
                full_begin = [0] * r
                full_end = [0] * r
                num_axes = len(axes)
                for i, ax in enumerate(axes):
                    ax_norm = ax if ax >= 0 else ax + r
                    full_begin[ax_norm] = pads_list[i]
                    full_end[ax_norm] = pads_list[num_axes + i]
                pads_list = full_begin + full_end

            pt_pad: list[int] = []
            for d in reversed(range(r)):
                pt_pad.append(pads_list[d])
                pt_pad.append(pads_list[r + d])

            if all(p == 0 for p in pt_pad):
                return data

            pt_mode = "replicate" if mode == "edge" else mode
            if pt_mode == "constant":
                pad_val = (
                    bool(val)
                    if data.dtype == torch.bool
                    else (
                        int(val)
                        if data.dtype
                        in (
                            torch.int32,
                            torch.int64,
                            torch.int16,
                            torch.int8,
                            torch.uint8,
                        )
                        else val
                    )
                )
                return F.pad(data, tuple(pt_pad), mode="constant", value=pad_val)
            return F.pad(data, tuple(pt_pad), mode=pt_mode)

        @self.register("", "DynamicQuantizeLinear")
        def _dynamic_quantize_linear(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> list[torch.Tensor]:
            x = inputs[0]
            x_min = torch.minimum(
                torch.tensor(0.0, device=x.device, dtype=x.dtype), torch.min(x)
            )
            x_max = torch.maximum(
                torch.tensor(0.0, device=x.device, dtype=x.dtype), torch.max(x)
            )
            denom = 255.0
            scale = (x_max - x_min) / denom
            if scale == 0:
                scale = torch.tensor(1.0, device=x.device, dtype=torch.float32)
            else:
                scale = scale.to(torch.float32)
            zero_point = torch.clamp(torch.round(-x_min / scale), 0, 255).to(
                torch.uint8
            )
            y = torch.clamp(
                torch.round(x / scale) + zero_point.to(x.dtype), 0, 255
            ).to(torch.uint8)
            return [y, scale, zero_point]

        @self.register("", "DequantizeLinear")
        def _dequantize_linear(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            x = inputs[0]
            scale = inputs[1]
            zp = inputs[2] if len(inputs) > 2 and inputs[2] is not None else None
            axis = attrs.get("axis", 1)

            if scale.dim() > 0 and scale.numel() > 1 and x.dim() > 1:
                axis_norm = axis if axis >= 0 else axis + x.dim()
                new_shape = [1] * x.dim()
                new_shape[axis_norm] = scale.numel()
                scale = scale.view(*new_shape)
                if zp is not None and zp.dim() > 0 and zp.numel() > 1:
                    zp = zp.view(*new_shape)

            if zp is not None:
                diff = x.to(torch.float32) - zp.to(torch.float32)
            else:
                diff = x.to(torch.float32)

            return (diff * scale).to(scale.dtype)

        @self.register("", "Expand")
        def _expand(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            x = inputs[0]
            raw_shape = [int(v) for v in inputs[1].cpu().tolist()]
            x_shape = list(x.shape)
            rank_diff = len(raw_shape) - len(x_shape)
            if rank_diff > 0:
                x_shape = [1] * rank_diff + x_shape
            elif rank_diff < 0:
                raw_shape = [1] * (-rank_diff) + raw_shape

            out_shape = []
            for d_x, d_target in zip(x_shape, raw_shape, strict=False):
                if d_target == -1:
                    out_shape.append(d_x)
                elif d_x == 1:
                    out_shape.append(d_target)
                elif d_target == 1:
                    out_shape.append(d_x)
                else:
                    out_shape.append(max(d_x, d_target))
            return x.expand(out_shape)

        @self.register("", "ReduceL2")
        def _reduce_l2(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            axes = attrs.get("axes")
            keepdims = bool(attrs.get("keepdims", 1))
            if axes is not None:
                dim = (
                    tuple(axes)
                    if isinstance(axes, list) and len(axes) > 1
                    else (axes[0] if isinstance(axes, list) else axes)
                )
            elif len(inputs) > 1 and inputs[1] is not None:
                axes_list = [int(v) for v in inputs[1].cpu().tolist()]
                dim = tuple(axes_list) if len(axes_list) > 1 else axes_list[0]
            else:
                dim = -1
            return torch.linalg.norm(inputs[0], ord=2, dim=dim, keepdim=keepdims)

        @self.register("", "ReduceMean")
        def _reduce_mean(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            axes = attrs.get("axes")
            keepdims = bool(attrs.get("keepdims", 1))
            if axes is not None:
                dim = (
                    tuple(axes)
                    if isinstance(axes, list) and len(axes) > 1
                    else (axes[0] if isinstance(axes, list) else axes)
                )
            elif len(inputs) > 1 and inputs[1] is not None:
                axes_list = [int(v) for v in inputs[1].cpu().tolist()]
                dim = (
                    tuple(axes_list)
                    if len(axes_list) > 1
                    else (axes_list[0] if axes_list else None)
                )
            else:
                dim = None
            if dim is None:
                return (
                    torch.mean(
                        inputs[0], dim=tuple(range(inputs[0].dim())), keepdim=True
                    )
                    if keepdims
                    else torch.mean(inputs[0])
                )
            return torch.mean(inputs[0], dim=dim, keepdim=keepdims)

        @self.register("", "ReduceSum")
        def _reduce_sum(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            axes = attrs.get("axes")
            keepdims = bool(attrs.get("keepdims", 1))
            if axes is not None:
                dim = (
                    tuple(axes)
                    if isinstance(axes, list) and len(axes) > 1
                    else (axes[0] if isinstance(axes, list) else axes)
                )
            elif len(inputs) > 1 and inputs[1] is not None:
                axes_list = [int(v) for v in inputs[1].cpu().tolist()]
                dim = (
                    tuple(axes_list)
                    if len(axes_list) > 1
                    else (axes_list[0] if axes_list else None)
                )
            else:
                dim = None
            if dim is None:
                return (
                    torch.sum(
                        inputs[0], dim=tuple(range(inputs[0].dim())), keepdim=True
                    )
                    if keepdims
                    else torch.sum(inputs[0])
                )
            return torch.sum(inputs[0], dim=dim, keepdim=keepdims)

        @self.register("", "ReduceMax")
        def _reduce_max(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            axes = attrs.get("axes")
            keepdims = bool(attrs.get("keepdims", 1))
            if axes is not None:
                dim = (
                    tuple(axes)
                    if isinstance(axes, list) and len(axes) > 1
                    else (axes[0] if isinstance(axes, list) else axes)
                )
            elif len(inputs) > 1 and inputs[1] is not None:
                axes_list = [int(v) for v in inputs[1].cpu().tolist()]
                dim = (
                    tuple(axes_list)
                    if len(axes_list) > 1
                    else (axes_list[0] if axes_list else None)
                )
            else:
                dim = None
            if dim is None:
                return (
                    torch.max(inputs[0])
                    if not keepdims
                    else torch.max(inputs[0]).view([1] * inputs[0].dim())
                )
            if isinstance(dim, tuple):
                res = inputs[0]
                for d in sorted(dim, reverse=True):
                    res = torch.max(res, dim=d, keepdim=keepdims).values
                return res
            return torch.max(inputs[0], dim=dim, keepdim=keepdims).values

        @self.register("", "ReduceMin")
        def _reduce_min(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            axes = attrs.get("axes")
            keepdims = bool(attrs.get("keepdims", 1))
            if axes is not None:
                dim = (
                    tuple(axes)
                    if isinstance(axes, list) and len(axes) > 1
                    else (axes[0] if isinstance(axes, list) else axes)
                )
            elif len(inputs) > 1 and inputs[1] is not None:
                axes_list = [int(v) for v in inputs[1].cpu().tolist()]
                dim = (
                    tuple(axes_list)
                    if len(axes_list) > 1
                    else (axes_list[0] if axes_list else None)
                )
            else:
                dim = None
            if dim is None:
                return (
                    torch.min(inputs[0])
                    if not keepdims
                    else torch.min(inputs[0]).view([1] * inputs[0].dim())
                )
            if isinstance(dim, tuple):
                res = inputs[0]
                for d in sorted(dim, reverse=True):
                    res = torch.min(res, dim=d, keepdim=keepdims).values
                return res
            return torch.min(inputs[0], dim=dim, keepdim=keepdims).values

        @self.register("", "Elu")
        def _elu(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            alpha = float(attrs.get("alpha", 1.0))
            return F.elu(inputs[0], alpha=alpha)

        @self.register("", "LSTM")
        def _lstm(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> Any:
            X = inputs[0]
            W = inputs[1]
            R = inputs[2]
            B = (
                inputs[3]
                if len(inputs) > 3 and inputs[3] is not None and inputs[3].numel() > 0
                else None
            )
            initial_h = (
                inputs[5]
                if len(inputs) > 5 and inputs[5] is not None and inputs[5].numel() > 0
                else None
            )
            initial_c = (
                inputs[6]
                if len(inputs) > 6 and inputs[6] is not None and inputs[6].numel() > 0
                else None
            )

            layout = int(attrs.get("layout", 0))
            hidden_size = int(attrs.get("hidden_size", W.shape[1] // 4))
            direction = attrs.get("direction", "forward")
            if isinstance(direction, bytes):
                direction = direction.decode("utf-8")

            if layout == 1:
                X_t = X.transpose(0, 1)
            else:
                X_t = X

            T, B_size, _ = X_t.shape
            num_directions = W.shape[0]

            all_dir_outputs = []
            final_h = []
            final_c = []

            for d_idx in range(num_directions):
                w_d = W[d_idx]
                r_d = R[d_idx]

                b_w = (
                    B[d_idx, : 4 * hidden_size]
                    if B is not None
                    else torch.zeros(
                        4 * hidden_size, device=X_t.device, dtype=X_t.dtype
                    )
                )
                b_r = (
                    B[d_idx, 4 * hidden_size :]
                    if B is not None
                    else torch.zeros(
                        4 * hidden_size, device=X_t.device, dtype=X_t.dtype
                    )
                )
                bias = b_w + b_r

                h_t = (
                    initial_h[d_idx]
                    if initial_h is not None
                    else torch.zeros(
                        B_size, hidden_size, device=X_t.device, dtype=X_t.dtype
                    )
                )
                c_t = (
                    initial_c[d_idx]
                    if initial_c is not None
                    else torch.zeros(
                        B_size, hidden_size, device=X_t.device, dtype=X_t.dtype
                    )
                )

                is_reverse = direction == "reverse" or (
                    direction == "bidirectional" and d_idx == 1
                )
                time_steps = range(T - 1, -1, -1) if is_reverse else range(T)

                dir_outs = [None] * T
                for t in time_steps:
                    x_step = X_t[t]
                    gates = (
                        torch.matmul(x_step, w_d.T)
                        + torch.matmul(h_t, r_d.T)
                        + bias.unsqueeze(0)
                    )
                    i_gate, o_gate, f_gate, c_gate = torch.chunk(gates, 4, dim=-1)
                    i = torch.sigmoid(i_gate)
                    o = torch.sigmoid(o_gate)
                    f = torch.sigmoid(f_gate)
                    c = torch.tanh(c_gate)
                    c_t = f * c_t + i * c
                    h_t = o * torch.tanh(c_t)
                    dir_outs[t] = h_t

                all_dir_outputs.append(torch.stack(dir_outs, dim=0))
                final_h.append(h_t)
                final_c.append(c_t)

            if num_directions == 1:
                Y = all_dir_outputs[0].unsqueeze(1)
                Y_h = final_h[0].unsqueeze(0)
                Y_c = final_c[0].unsqueeze(0)
            else:
                Y = torch.stack(all_dir_outputs, dim=1)
                Y_h = torch.stack(final_h, dim=0)
                Y_c = torch.stack(final_c, dim=0)

            if layout == 1:
                Y = Y.permute(2, 0, 1, 3)

            return [Y, Y_h, Y_c]

        @self.register("", "GRU")
        def _gru(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> Any:
            X = inputs[0]
            W = inputs[1]
            R = inputs[2]
            B = (
                inputs[3]
                if len(inputs) > 3 and inputs[3] is not None and inputs[3].numel() > 0
                else None
            )
            initial_h = (
                inputs[5]
                if len(inputs) > 5 and inputs[5] is not None and inputs[5].numel() > 0
                else None
            )

            layout = int(attrs.get("layout", 0))
            hidden_size = int(attrs.get("hidden_size", W.shape[1] // 3))
            direction = attrs.get("direction", "forward")
            linear_before_reset = int(attrs.get("linear_before_reset", 0))
            if isinstance(direction, bytes):
                direction = direction.decode("utf-8")

            if layout == 1:
                X_t = X.transpose(0, 1)
            else:
                X_t = X

            T, B_size, _ = X_t.shape
            num_directions = W.shape[0]

            all_dir_outputs = []
            final_h = []

            for d_idx in range(num_directions):
                w_d = W[d_idx]
                r_d = R[d_idx]

                b_w = (
                    B[d_idx, : 3 * hidden_size]
                    if B is not None
                    else torch.zeros(
                        3 * hidden_size, device=X_t.device, dtype=X_t.dtype
                    )
                )
                b_r = (
                    B[d_idx, 3 * hidden_size :]
                    if B is not None
                    else torch.zeros(
                        3 * hidden_size, device=X_t.device, dtype=X_t.dtype
                    )
                )

                h_t = (
                    initial_h[d_idx]
                    if initial_h is not None
                    else torch.zeros(
                        B_size, hidden_size, device=X_t.device, dtype=X_t.dtype
                    )
                )

                is_reverse = direction == "reverse" or (
                    direction == "bidirectional" and d_idx == 1
                )
                time_steps = range(T - 1, -1, -1) if is_reverse else range(T)

                dir_outs = [None] * T
                for t in time_steps:
                    x_step = X_t[t]
                    w_x = torch.matmul(x_step, w_d.T) + b_w.unsqueeze(0)
                    w_z, w_r, w_h = torch.chunk(w_x, 3, dim=-1)

                    if linear_before_reset == 1:
                        r_h_all = torch.matmul(h_t, r_d.T) + b_r.unsqueeze(0)
                        r_z, r_r, r_h = torch.chunk(r_h_all, 3, dim=-1)
                        z = torch.sigmoid(w_z + r_z)
                        r = torch.sigmoid(w_r + r_r)
                        h = torch.tanh(w_h + r * r_h)
                    else:
                        r_zr = torch.matmul(h_t, r_d[: 2 * hidden_size].T) + b_r[
                            : 2 * hidden_size
                        ].unsqueeze(0)
                        r_z, r_r = torch.chunk(r_zr, 2, dim=-1)
                        z = torch.sigmoid(w_z + r_z)
                        r = torch.sigmoid(w_r + r_r)
                        r_h = torch.matmul(r * h_t, r_d[2 * hidden_size :].T) + b_r[
                            2 * hidden_size :
                        ].unsqueeze(0)
                        h = torch.tanh(w_h + r_h)

                    h_t = (1.0 - z) * h + z * h_t
                    dir_outs[t] = h_t

                all_dir_outputs.append(torch.stack(dir_outs, dim=0))
                final_h.append(h_t)

            if num_directions == 1:
                Y = all_dir_outputs[0].unsqueeze(1)
                Y_h = final_h[0].unsqueeze(0)
            else:
                Y = torch.stack(all_dir_outputs, dim=1)
                Y_h = torch.stack(final_h, dim=0)

            if layout == 1:
                Y = Y.permute(2, 0, 1, 3)

            return [Y, Y_h]

        @self.register("", "RNN")
        def _rnn(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> Any:
            X = inputs[0]
            W = inputs[1]
            R = inputs[2]
            B = (
                inputs[3]
                if len(inputs) > 3 and inputs[3] is not None and inputs[3].numel() > 0
                else None
            )
            initial_h = (
                inputs[5]
                if len(inputs) > 5 and inputs[5] is not None and inputs[5].numel() > 0
                else None
            )

            layout = int(attrs.get("layout", 0))
            hidden_size = int(attrs.get("hidden_size", W.shape[1]))
            direction = attrs.get("direction", "forward")
            if isinstance(direction, bytes):
                direction = direction.decode("utf-8")
            activations = attrs.get("activations", ["Tanh", "Tanh"])
            if isinstance(activations, (list, tuple)) and activations:
                act_fn_name = (
                    activations[0].decode("utf-8")
                    if isinstance(activations[0], bytes)
                    else str(activations[0])
                )
            else:
                act_fn_name = "Tanh"

            act_fn = torch.relu if act_fn_name.lower() == "relu" else torch.tanh

            if layout == 1:
                X_t = X.transpose(0, 1)
            else:
                X_t = X

            T, B_size, _ = X_t.shape
            num_directions = W.shape[0]

            all_dir_outputs = []
            final_h = []

            for d_idx in range(num_directions):
                w_d = W[d_idx]
                r_d = R[d_idx]

                b_w = (
                    B[d_idx, :hidden_size]
                    if B is not None
                    else torch.zeros(hidden_size, device=X_t.device, dtype=X_t.dtype)
                )
                b_r = (
                    B[d_idx, hidden_size:]
                    if B is not None
                    else torch.zeros(hidden_size, device=X_t.device, dtype=X_t.dtype)
                )
                bias = b_w + b_r

                h_t = (
                    initial_h[d_idx]
                    if initial_h is not None
                    else torch.zeros(
                        B_size, hidden_size, device=X_t.device, dtype=X_t.dtype
                    )
                )

                dir_outs = [None] * T
                time_steps = (
                    range(T)
                    if (direction == "forward" or d_idx == 0)
                    else reversed(range(T))
                )

                for t in time_steps:
                    x_t = X_t[t]
                    h_t = act_fn(
                        torch.matmul(x_t, w_d.T) + torch.matmul(h_t, r_d.T) + bias
                    )
                    dir_outs[t] = h_t

                all_dir_outputs.append(torch.stack(dir_outs, dim=0))
                final_h.append(h_t)

            if num_directions == 1:
                Y = all_dir_outputs[0].unsqueeze(1)
                Y_h = final_h[0].unsqueeze(0)
            else:
                Y = torch.stack(all_dir_outputs, dim=1)
                Y_h = torch.stack(final_h, dim=0)

            if layout == 1:
                Y = Y.permute(2, 0, 1, 3)

            return [Y, Y_h]

        @self.register("", "GlobalAveragePool")
        def _global_average_pool(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            x = inputs[0]
            dims = list(range(2, x.ndim))
            if dims:
                return torch.mean(x, dim=dims, keepdim=True)
            return x

        @self.register("", "Range")
        def _range(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            device = inputs[0].device
            start = inputs[0].item()
            limit = inputs[1].item()
            delta = inputs[2].item() if len(inputs) > 2 and inputs[2] is not None else 1
            return torch.arange(
                start, limit, delta, dtype=inputs[0].dtype, device=device
            )

        @self.register("", "Einsum")
        def _einsum(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            eq = attrs.get("equation", "")
            if isinstance(eq, bytes):
                eq = eq.decode("utf-8")
            return torch.einsum(eq, *inputs)

        @self.register("", "ConstantOfShape")
        def _constant_of_shape(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            shape = [int(v) for v in inputs[0].cpu().tolist()]
            device = inputs[0].device
            val_tensor = attrs.get("value")
            if val_tensor is not None:
                if isinstance(val_tensor, np.ndarray):
                    fill_val = val_tensor.item()
                    dtype = torch.from_numpy(val_tensor.copy()).dtype
                elif isinstance(val_tensor, torch.Tensor):
                    fill_val = val_tensor.item()
                    dtype = val_tensor.dtype
                else:
                    fill_val = 0.0
                    dtype = torch.float32
            else:
                fill_val = 0.0
                dtype = torch.float32
            return torch.full(shape, fill_val, dtype=dtype, device=device)

        @self.register("", "ScatterND")
        def _scatter_nd(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            data, indices, updates = inputs[0], inputs[1], inputs[2]
            out = data.clone()
            idx_tuple = tuple(
                indices.select(-1, i).long() for i in range(indices.shape[-1])
            )
            out[idx_tuple] = updates
            return out

        # -------------------------------------------------------------
        # Activations
        # -------------------------------------------------------------
        @self.register("", "Relu")
        def _relu(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            return F.relu(inputs[0])

        @self.register("", "LeakyRelu")
        @self.register("custom.pytorch", "LeakyRelu")
        def _leaky_relu(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            alpha = attrs.get("alpha", 0.01)
            return F.leaky_relu(inputs[0], negative_slope=alpha)

        @self.register("", "Sigmoid")
        def _sigmoid(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            return torch.sigmoid(inputs[0])

        @self.register("", "Tanh")
        def _tanh(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            return torch.tanh(inputs[0])

        @self.register("", "LogSoftmax")
        def _log_softmax(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            axis = attrs.get("axis", -1)
            return F.log_softmax(inputs[0], dim=axis)

        @self.register("", "Softmax")
        def _softmax(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            axis = attrs.get("axis", -1)
            return F.softmax(inputs[0], dim=axis)

        @self.register("", "Clip")
        def _clip(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            val = inputs[0]
            min_val = (
                inputs[1].item()
                if len(inputs) > 1 and inputs[1] is not None and inputs[1].numel() > 0
                else attrs.get("min")
            )
            max_val = (
                inputs[2].item()
                if len(inputs) > 2 and inputs[2] is not None and inputs[2].numel() > 0
                else attrs.get("max")
            )
            return torch.clamp(val, min=min_val, max=max_val)

        # -------------------------------------------------------------
        # Custom Ops: Gelu & Math
        # -------------------------------------------------------------
        @self.register("custom.pytorch", "Gelu")
        @self.register("", "Gelu")
        def _gelu(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            approximate = "tanh" if attrs.get("approximate") == "tanh" else "none"
            return F.gelu(inputs[0], approximate=approximate)

        @self.register("custom.pytorch", "SiLU")
        @self.register("", "SiLU")
        @self.register("custom.pytorch", "Swish")
        @self.register("", "Swish")
        def _silu(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            return F.silu(inputs[0])

        @self.register("custom.pytorch", "HardSwish")
        @self.register("", "HardSwish")
        def _hardswish(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            return F.hardswish(inputs[0])

        @self.register("custom.pytorch", "HardSigmoid")
        @self.register("", "HardSigmoid")
        def _hardsigmoid(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            alpha = attrs.get("alpha", 0.2)
            beta = attrs.get("beta", 0.5)
            return torch.clamp(inputs[0] * alpha + beta, min=0.0, max=1.0)

        @self.register("custom.pytorch", "Mish")
        @self.register("", "Mish")
        def _mish(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            return F.mish(inputs[0])

        @self.register("custom.pytorch", "Relu6")
        @self.register("", "Relu6")
        def _relu6(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            return F.relu6(inputs[0])

        @self.register("", "Softplus")
        def _softplus(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            return F.softplus(inputs[0])

        @self.register("custom.pytorch", "Math")
        def _math_op(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            expr = attrs.get("expr") or attrs.get("folded_expr")
            device = inputs[0].device if inputs else None
            if expr and not inputs:
                # Constant expression
                val = _safe_eval_math(expr, {"torch": torch, "np": np})
                return torch.tensor(val, dtype=torch.float32, device=device)
            if expr and len(inputs) == 1:
                x = inputs[0]
                val = _safe_eval_math(expr, {"x": x, "torch": torch, "np": np})
                return (
                    val
                    if isinstance(val, torch.Tensor)
                    else torch.tensor(val, dtype=x.dtype, device=x.device)
                )
            if inputs:
                return inputs[0]
            msg = f"Unable to evaluate Math op with attrs={attrs}"
            raise ValueError(msg)

        # -------------------------------------------------------------
        # Neural Network Layers: Conv, Normalization, Pooling
        # -------------------------------------------------------------
        @self.register("", "Conv")
        def _conv(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            x = inputs[0]
            w = inputs[1]
            b = (
                inputs[2]
                if len(inputs) > 2 and inputs[2] is not None and inputs[2].numel() > 0
                else None
            )

            pads = attrs.get("pads", [0, 0, 0, 0])
            strides = attrs.get("strides", [1, 1])
            dilations = attrs.get("dilations", [1, 1])
            group = attrs.get("group", 1)

            # Check 1D vs 2D
            if x.dim() == 4:
                # ONNX pads: [top, left, bottom, right] or [x1_begin, x2_begin, x1_end, x2_end]
                if len(pads) == 4:
                    if pads[0] == pads[2] and pads[1] == pads[3]:
                        padding = (pads[0], pads[1])
                    else:
                        # PyTorch F.pad takes (left, right, top, bottom)
                        x = F.pad(x, (pads[1], pads[3], pads[0], pads[2]))
                        padding = (0, 0)
                else:
                    padding = (0, 0)

                return F.conv2d(
                    x,
                    w,
                    b,
                    stride=tuple(strides) if isinstance(strides, list) else strides,
                    padding=padding,
                    dilation=tuple(dilations)
                    if isinstance(dilations, list)
                    else dilations,
                    groups=group,
                )
            if x.dim() == 3:
                padding = pads[0] if len(pads) >= 2 and pads[0] == pads[1] else 0
                return F.conv1d(
                    x,
                    w,
                    b,
                    stride=strides[0] if isinstance(strides, list) else strides,
                    padding=padding,
                    dilation=dilations[0] if isinstance(dilations, list) else dilations,
                    groups=group,
                )
            if x.dim() == 5:
                if pads:
                    if (
                        len(pads) == 6
                        and pads[0] == pads[3]
                        and pads[1] == pads[4]
                        and pads[2] == pads[5]
                    ):
                        padding = (pads[0], pads[1], pads[2])
                    else:
                        x = F.pad(
                            x, (pads[2], pads[5], pads[1], pads[4], pads[0], pads[3])
                        )
                        padding = (0, 0, 0)
                else:
                    padding = (0, 0, 0)
                return F.conv3d(
                    x,
                    w,
                    b,
                    stride=tuple(strides) if isinstance(strides, list) else strides,
                    padding=padding,
                    dilation=tuple(dilations)
                    if isinstance(dilations, list)
                    else dilations,
                    groups=group,
                )
            msg = f"Unsupported Conv input dimension: {x.dim()}"
            raise NotImplementedError(msg)

        @self.register("", "BatchNormalization")
        def _batch_norm(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            x, scale, b, mean, var = (
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
            )
            eps = attrs.get("epsilon", 1e-5)
            return F.batch_norm(
                x, mean, var, weight=scale, bias=b, training=False, eps=eps
            )

        @self.register("custom.pytorch", "GroupNorm")
        @self.register("", "GroupNorm")
        @self.register("", "GroupNormalization")
        def _group_norm(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            x, scale, b = inputs[0], inputs[1], inputs[2]
            num_groups = int(attrs.get("num_groups", 1))
            eps = float(attrs.get("epsilon", 1e-5))
            w = scale.view(-1) if scale is not None and scale.numel() > 0 else None
            bias = b.view(-1) if b is not None and b.numel() > 0 else None
            return F.group_norm(x, num_groups=num_groups, weight=w, bias=bias, eps=eps)

        @self.register("custom.pytorch", "InstanceNorm1d")
        @self.register("", "InstanceNormalization")
        def _instance_norm(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            x, scale, b = inputs[0], inputs[1], inputs[2]
            eps = attrs.get("epsilon", 1e-5)
            w = scale.view(-1) if scale is not None and scale.numel() > 0 else None
            bias = b.view(-1) if b is not None and b.numel() > 0 else None
            return F.instance_norm(x, weight=w, bias=bias, eps=eps)

        @self.register("custom.pytorch", "RMSNorm")
        @self.register("", "RMSNorm")
        @self.register("", "RMSNormalization")
        @self.register("com.microsoft", "SimplifiedLayerNormalization")
        def _rms_norm(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            x = inputs[0]
            gamma = (
                inputs[1]
                if len(inputs) > 1 and inputs[1] is not None and inputs[1].numel() > 0
                else None
            )
            eps = float(attrs.get("epsilon", 1e-12))
            norm = torch.norm(x, p=2, dim=-1, keepdim=True)
            scale = (x.shape[-1] ** 0.5) / torch.clamp(norm, min=eps)
            res = x * scale
            if gamma is not None:
                res = res * gamma
            return res

        @self.register("com.microsoft", "SkipSimplifiedLayerNormalization")
        @self.register("custom.pytorch", "SkipRMSNorm")
        @self.register("", "SkipRMSNorm")
        @self.register("custom.pytorch", "SkipSimplifiedLayerNormalization")
        def _skip_rms_norm(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            x = inputs[0] + inputs[1]
            gamma = (
                inputs[2]
                if len(inputs) > 2 and inputs[2] is not None and inputs[2].numel() > 0
                else None
            )
            bias = (
                inputs[3]
                if len(inputs) > 3 and inputs[3] is not None and inputs[3].numel() > 0
                else None
            )
            eps = float(attrs.get("epsilon", 1e-12))
            norm = torch.norm(x, p=2, dim=-1, keepdim=True)
            scale = (x.shape[-1] ** 0.5) / torch.clamp(norm, min=eps)
            res = x * scale
            if gamma is not None:
                res = res * gamma
            if bias is not None:
                res = res + bias
            return res

        @self.register("custom.pytorch", "FeedForward")
        @self.register("", "FeedForward")
        def _feed_forward(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            x = inputs[0]
            gamma = (
                inputs[1]
                if len(inputs) > 1 and inputs[1] is not None and inputs[1].numel() > 0
                else None
            )
            fc1_w = inputs[2]
            fc1_b = (
                inputs[3]
                if len(inputs) > 3 and inputs[3] is not None and inputs[3].numel() > 0
                else None
            )
            fc2_w = inputs[4] if len(inputs) > 4 and inputs[4] is not None else None
            fc2_b = (
                inputs[5]
                if len(inputs) > 5 and inputs[5] is not None and inputs[5].numel() > 0
                else None
            )
            eps = float(attrs.get("epsilon", 1e-12))
            # 1. RMSNorm
            norm = torch.norm(x, p=2, dim=-1, keepdim=True)
            scale = (x.shape[-1] ** 0.5) / torch.clamp(norm, min=eps)
            x_norm = x * scale
            if gamma is not None:
                x_norm = x_norm * gamma
            # 2. Linear 1
            h = torch.matmul(x_norm, fc1_w)
            if fc1_b is not None:
                h = h + fc1_b
            # 3. Activation
            act = attrs.get("activation", "gelu")
            if act == "gelu":
                h = F.gelu(h)
            elif act == "relu":
                h = F.relu(h)
            elif act == "silu":
                h = F.silu(h)
            # 4. Linear 2
            if fc2_w is not None:
                out = torch.matmul(h, fc2_w)
                if fc2_b is not None:
                    out = out + fc2_b
            else:
                out = h
            return out

        @self.register("custom.pytorch", "FeedForwardSwiGLU")
        @self.register("", "FeedForwardSwiGLU")
        def _feed_forward_swiglu(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            x = inputs[0]
            if len(inputs) in (5, 8):
                gamma = (
                    inputs[1]
                    if inputs[1] is not None and inputs[1].numel() > 0
                    else None
                )
                eps = float(attrs.get("epsilon", 1e-12))
                norm = torch.norm(x, p=2, dim=-1, keepdim=True)
                scale = (x.shape[-1] ** 0.5) / torch.clamp(norm, min=eps)
                x_norm = x * scale
                if gamma is not None:
                    x_norm = x_norm * gamma
                rest = inputs[2:]
            else:
                x_norm = x
                rest = inputs[1:]

            if len(rest) == 3:
                w_gate, w_up, w_down = rest
                b_gate, b_up, b_down = None, None, None
            elif len(rest) >= 6:
                w_gate, b_gate, w_up, b_up, w_down, b_down = rest[:6]
            else:
                w_gate, w_up, w_down = rest[0], rest[1], rest[2]
                b_gate, b_up, b_down = None, None, None

            gate = torch.matmul(x_norm, w_gate)
            if b_gate is not None and b_gate.numel() > 0:
                gate = gate + b_gate
            up = torch.matmul(x_norm, w_up)
            if b_up is not None and b_up.numel() > 0:
                up = up + b_up
            hidden = F.silu(gate) * up
            out = torch.matmul(hidden, w_down)
            if b_down is not None and b_down.numel() > 0:
                out = out + b_down
            return out

        @self.register("custom.pytorch", "RotaryEmbedding")
        @self.register("custom.pytorch", "PartialRotaryEmbedding")
        @self.register("", "RotaryEmbedding")
        def _rotary_embedding(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            x = inputs[0]
            seq_len = x.shape[-2]
            if len(inputs) >= 3:
                cos, sin = inputs[1], inputs[2]
                if cos.shape[-2] > seq_len:
                    cos = cos[..., :seq_len, :]
                    sin = sin[..., :seq_len, :]
            else:
                theta = inputs[1]
                if theta.shape[-2] > seq_len:
                    theta = theta[..., :seq_len, :]
                cos = torch.cos(theta)
                sin = torch.sin(theta)

            rotary_dim = cos.shape[-1]
            head_dim = x.shape[-1]

            if rotary_dim < head_dim:
                x_rot = x[..., :rotary_dim]
                x_pass = x[..., rotary_dim:]
            else:
                x_rot = x
                x_pass = None

            interleaved = int(attrs.get("interleaved", 0 if "rotate_half" in str(attrs.get("mode", "")) or attrs.get("interleaved") == 0 else 1))
            if interleaved == 0:
                d = x_rot.shape[-1]
                x1 = x_rot[..., : d // 2]
                x2 = x_rot[..., d // 2 :]
                rot = torch.cat((-x2, x1), dim=-1)
            else:
                orig_rot_shape = x_rot.shape
                x_reshaped = x_rot.reshape(*orig_rot_shape[:-1], -1, 2)
                rot = torch.stack(
                    (-x_reshaped[..., 1], x_reshaped[..., 0]), dim=-1
                ).reshape(orig_rot_shape)

            out_rot = (x_rot * cos) + (rot * sin)

            if x_pass is not None:
                return torch.cat((out_rot, x_pass), dim=-1)
            return out_rot

        @self.register("custom.pytorch", "Unbind")
        @self.register("", "Unbind")
        def _unbind(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> list[torch.Tensor]:
            dim = attrs.get("axis", attrs.get("dim", 0))
            return list(torch.unbind(inputs[0], dim=dim))

        @self.register("custom.pytorch", "GatedUnit")
        @self.register("", "GatedUnit")
        def _gated_unit(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            x, gate = inputs[0], inputs[1]
            act_name = attrs.get("activation", "sigmoid").lower()
            if gate.dim() == 3 and x.dim() == 4:
                # x: (B, H, S, D)
                if gate.shape[1] == x.shape[1]:
                    gate = gate.unsqueeze(-1)
                elif gate.shape[2] == x.shape[1]:
                    gate = gate.transpose(1, 2).unsqueeze(-1)
                else:
                    gate = gate.unsqueeze(-1)
            if act_name in ("silu", "swish"):
                g_act = F.silu(gate)
            elif act_name == "gelu":
                g_act = F.gelu(gate)
            elif act_name == "relu":
                g_act = F.relu(gate)
            elif act_name == "tanh":
                g_act = torch.tanh(gate)
            else:
                g_act = torch.sigmoid(gate)
            return x * g_act

        @self.register("custom.pytorch", "SoftmaxMatMul")
        @self.register("", "SoftmaxMatMul")
        def _softmax_matmul(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            scores, v = inputs[0], inputs[1]
            axis = attrs.get("axis", -1)
            return torch.matmul(torch.softmax(scores, dim=axis), v)

        @self.register("custom.pytorch", "Linear")
        @self.register("", "Linear")
        def _linear(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            x, w = inputs[0], inputs[1]
            bias = (
                inputs[2]
                if len(inputs) > 2 and inputs[2] is not None and inputs[2].numel() > 0
                else None
            )
            out = torch.matmul(x, w)
            if bias is not None:
                out = out + bias
            return out

        @self.register("custom.pytorch", "LogitsSoftCap")
        @self.register("", "LogitsSoftCap")
        def _logits_softcap(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            x = inputs[0]
            softcap = float(attrs.get("softcap", 50.0))
            return softcap * torch.tanh(x / softcap)

        @self.register("custom.pytorch", "MultiHeadLatentAttention_KVProjection")
        @self.register("", "MultiHeadLatentAttention_KVProjection")
        def _mla_kv_proj(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> list[torch.Tensor]:
            x, w_down, w_up_k, w_up_v = inputs[0], inputs[1], inputs[2], inputs[3]
            c_kv = torch.matmul(x, w_down)
            k = torch.matmul(c_kv, w_up_k)
            v = torch.matmul(c_kv, w_up_v)
            return [k, v]

        @self.register("custom.pytorch", "ScaledDotProductAttention")
        @self.register("custom.pytorch", "Attention")
        @self.register("", "ScaledDotProductAttention")
        def _attention(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            q, k, v = inputs[0], inputs[1], inputs[2]
            scale = attrs.get("scale")
            if scale is not None:
                s = float(scale) ** 0.5
            else:
                s = q.shape[-1] ** -0.25
            scores = torch.matmul(q * s, (k * s).transpose(-1, -2))
            if "softcap" in attrs:
                sc = float(attrs["softcap"])
                scores = sc * torch.tanh(scores / sc)
            if len(inputs) > 3 and inputs[3] is not None and inputs[3].numel() > 0:
                scores = scores + inputs[3]
            probs = torch.softmax(scores, dim=-1)
            return torch.matmul(probs, v)

        @self.register("custom.pytorch", "SplitQKV")
        @self.register("", "SplitQKV")
        def _split_qkv(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> list[torch.Tensor]:
            x = inputs[0]
            num_heads = int(attrs.get("num_heads", 4))
            orig_shape = x.shape
            if len(orig_shape) == 2:
                B, total_d = orig_shape[0], orig_shape[1]
                D = total_d // (3 * num_heads)
                reshaped = x.view(B, 3, num_heads, D).permute(1, 0, 2, 3)
                return [reshaped[0], reshaped[1], reshaped[2]]
            B, S, total_d = orig_shape[0], orig_shape[1], orig_shape[2]
            D = total_d // (3 * num_heads)
            reshaped = x.view(B, S, 3, num_heads, D).permute(2, 0, 3, 1, 4)
            return [reshaped[0], reshaped[1], reshaped[2]]

        @self.register("custom.pytorch", "MergeHeads")
        @self.register("", "MergeHeads")
        def _merge_heads(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            x = inputs[0]
            # x shape: (B, H, S, D)
            B, H, S, D = x.shape
            return x.transpose(1, 2).reshape(B, S, H * D)

        @self.register("custom.pytorch", "GatedAttention")
        @self.register("", "GatedAttention")
        def _gated_attention(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            x = inputs[0]
            norm_gamma = (
                inputs[1]
                if len(inputs) > 1 and inputs[1] is not None and inputs[1].numel() > 0
                else None
            )
            qkv_w = inputs[2]
            qkv_b = (
                inputs[3]
                if len(inputs) > 3 and inputs[3] is not None and inputs[3].numel() > 0
                else None
            )
            gates_w = inputs[4]
            gates_b = (
                inputs[5]
                if len(inputs) > 5 and inputs[5] is not None and inputs[5].numel() > 0
                else None
            )
            theta = inputs[6]
            out_w = inputs[7]
            out_b = (
                inputs[8]
                if len(inputs) > 8 and inputs[8] is not None and inputs[8].numel() > 0
                else None
            )

            # 1. RMSNorm
            eps = float(attrs.get("epsilon", 1e-12))
            norm = torch.norm(x, p=2, dim=-1, keepdim=True)
            scale = (x.shape[-1] ** 0.5) / torch.clamp(norm, min=eps)
            x_norm = x * scale
            if norm_gamma is not None:
                x_norm = x_norm * norm_gamma

            # 2. to_qkv
            qkv = torch.matmul(x_norm, qkv_w)
            if qkv_b is not None:
                qkv = qkv + qkv_b

            # 3. to_gates
            gates = torch.matmul(x_norm, gates_w)
            if gates_b is not None:
                gates = gates + gates_b

            # 4. SplitQKV
            num_heads = int(attrs.get("num_heads", 4))
            orig_shape = qkv.shape
            B, S, total_d = orig_shape[0], orig_shape[1], orig_shape[2]
            D = total_d // (3 * num_heads)
            reshaped_qkv = qkv.view(B, S, 3, num_heads, D).permute(2, 0, 3, 1, 4)
            q, k, v = reshaped_qkv[0], reshaped_qkv[1], reshaped_qkv[2]

            # 5. RoPE
            def _apply_rope(t: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
                seq_len = t.shape[-2]
                if th.shape[-2] > seq_len:
                    th = th[..., :seq_len, :]
                cos = torch.cos(th)
                sin = torch.sin(th)
                rotary_dim = cos.shape[-1]
                head_dim = t.shape[-1]
                if rotary_dim < head_dim:
                    t_rot = t[..., :rotary_dim]
                    t_pass = t[..., rotary_dim:]
                else:
                    t_rot = t
                    t_pass = None
                o_shp = t_rot.shape
                t_r = t_rot.reshape(*o_shp[:-1], -1, 2)
                rot = torch.stack((-t_r[..., 1], t_r[..., 0]), dim=-1).reshape(o_shp)
                res = (t_rot * cos) + (rot * sin)
                if t_pass is not None:
                    res = torch.cat([res, t_pass], dim=-1)
                return res

            q_rot = _apply_rope(q, theta)
            k_rot = _apply_rope(k, theta)

            # 6. SDPA
            head_dim = q_rot.shape[-1]
            scale_factor = 1.0 / (head_dim**0.5)
            scores = torch.matmul(q_rot, k_rot.transpose(-1, -2)) * scale_factor
            attn_weights = torch.softmax(scores, dim=-1)
            attn_out = torch.matmul(attn_weights, v)

            # 7. GatedUnit
            if gates.dim() == 3:
                gates = gates.transpose(1, 2).unsqueeze(-1)
            gated_scores = attn_out * torch.sigmoid(gates)

            # 8. MergeHeads
            merged = (
                gated_scores.transpose(1, 2).contiguous().reshape(B, S, num_heads * D)
            )

            # 9. to_out
            out = torch.matmul(merged, out_w)
            if out_b is not None:
                out = out + out_b
            return out

        @self.register("custom.pytorch", "AxialRearrange")
        @self.register("", "AxialRearrange")
        def _axial_rearrange(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            x = inputs[0]
            mode = attrs.get("mode", "freq_to_time")
            b = int(attrs.get("batch_size", 1))
            if mode == "add_channel":
                return x.unsqueeze(1)
            if mode == "freq_in":
                # (B, C, F, T) -> (B*T, F, C)
                B, C, F, T = x.shape
                return x.permute(0, 3, 2, 1).reshape(B * T, F, C)
            if mode == "freq_to_time":
                # (B*T, F, C) -> (B*F, T, C)
                BT, F, C = x.shape
                T = BT // b
                return x.view(b, T, F, C).permute(0, 2, 1, 3).reshape(b * F, T, C)
            if mode == "time_to_conv":
                target_shape = attrs.get("target_shape")
                if target_shape:
                    target = list(target_shape)
                    if (
                        x.dim() >= 1
                        and x.shape[0] != target[0]
                        and target[0] in (1, -1)
                    ):
                        target[0] = x.shape[0]
                    return x.reshape(*target).permute(0, 3, 1, 2)
                # (B*F, T, C) -> (B, C, F, T)
                BF, T, C = x.shape
                F = BF // b
                return x.view(b, F, T, C).permute(0, 3, 1, 2)
            if mode == "flatten_fc":
                # (B, C, F, T) -> (B, T, C * F)
                B, C, F, T = x.shape
                return x.permute(0, 3, 1, 2).reshape(B, T, C * F)
            raise ValueError(f"Unknown AxialRearrange mode: {mode}")

        @self.register("custom.pytorch", "TensorRoute")
        @self.register("custom.pytorch", "IndexSelect")
        def _tensor_route(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            x = inputs[0]
            axis = attrs.get("axis", 1)
            index = attrs.get("index")
            if index is not None:
                x = torch.select(x, dim=axis, index=int(index))
            slice_axis = attrs.get("slice_axis")
            slice_start = attrs.get("slice_start")
            slice_end = attrs.get("slice_end")
            if slice_axis is not None and (
                slice_start is not None or slice_end is not None
            ):
                slices = [slice(None)] * x.dim()
                start = slice_start if slice_start is not None else 0
                end = slice_end
                if end is not None and end < 0:
                    end = x.shape[slice_axis] + end
                slices[slice_axis] = slice(start, end)
                x = x[tuple(slices)]
            return x

        @self.register("", "LayerNormalization")
        @self.register("", "LayerNorm")
        @self.register("custom.pytorch", "LayerNormalization")
        @self.register("custom.pytorch", "LayerNorm")
        def _layer_norm(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            x = inputs[0]
            scale = (
                inputs[1]
                if len(inputs) > 1 and inputs[1] is not None and inputs[1].numel() > 0
                else None
            )
            bias = (
                inputs[2]
                if len(inputs) > 2 and inputs[2] is not None and inputs[2].numel() > 0
                else None
            )
            axis = attrs.get("axis", -1)
            eps = attrs.get("epsilon", 1e-5)
            normalized_shape = (
                list(x.shape[axis:])
                if axis >= 0
                else list(x.shape[len(x.shape) + axis :])
            )
            if scale is not None and scale.dtype != x.dtype:
                scale = scale.to(dtype=x.dtype)
            if bias is not None and bias.dtype != x.dtype:
                bias = bias.to(dtype=x.dtype)
            return F.layer_norm(x, normalized_shape, weight=scale, bias=bias, eps=eps)

        # -------------------------------------------------------------
        # Macro Block Handlers (ResMLP, SGU, Highway, SE, MHA, DeepMLP)
        # -------------------------------------------------------------
        @self.register("custom.pytorch", "DeepMLP_Norm_Act")
        @self.register("", "DeepMLP_Norm_Act")
        def _deep_mlp_norm_act(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            has_norm = attrs.get("has_norm", 1)
            act_type = attrs.get("activation", "GELU")
            if has_norm:
                x = inputs[0]
                gamma = inputs[1]
                rem = inputs[2:]
                beta = None
                w = None
                b = None
                if rem:
                    if rem[0].dim() == 1 and rem[0].shape == gamma.shape:
                        beta = rem[0]
                        if len(rem) > 1:
                            w = rem[1]
                        if len(rem) > 2:
                            b = rem[2]
                    else:
                        w = rem[0]
                        if len(rem) > 1:
                            b = rem[1]
                if beta is not None and beta.numel() > 0:
                    x_norm = F.layer_norm(x, [gamma.shape[0]], weight=gamma, bias=beta)
                else:
                    norm = torch.norm(x, p=2, dim=-1, keepdim=True)
                    scale = (x.shape[-1] ** 0.5) / torch.clamp(norm, min=1e-12)
                    x_norm = x * scale * gamma

                if w is not None:
                    h = torch.matmul(x_norm, w)
                    if b is not None and b.numel() > 0:
                        h = h + b
                else:
                    h = x_norm
            else:
                x = inputs[0]
                w = inputs[1]
                b = (
                    inputs[2]
                    if len(inputs) > 2
                    and inputs[2] is not None
                    and inputs[2].numel() > 0
                    else None
                )
                h = torch.matmul(x, w)
                if b is not None:
                    h = h + b
            if act_type in ("Gelu", "GELU", "FastGelu"):
                return F.gelu(h)
            if act_type in ("SiLU", "Swish"):
                return F.silu(h)
            if act_type == "Relu":
                return F.relu(h)
            return h

        @self.register("custom.pytorch", "SpatialGatingUnit")
        @self.register("", "SpatialGatingUnit")
        def _spatial_gating_unit(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            x = inputs[0]
            u, v = torch.chunk(x, 2, dim=-1)
            rem = inputs[1:]
            gamma = None
            beta = None
            w = None
            b = None
            w_idx = None
            for idx, t in enumerate(rem):
                if t is not None and t.dim() == 2:
                    w_idx = idx
                    break
            if w_idx is not None:
                norm_params = rem[:w_idx]
                if len(norm_params) >= 1:
                    gamma = norm_params[0]
                if len(norm_params) >= 2:
                    beta = norm_params[1]
                w = rem[w_idx]
                if len(rem) > w_idx + 1:
                    b = rem[w_idx + 1]
            else:
                w = rem[0]
                if len(rem) > 1:
                    b = rem[1]

            if gamma is not None:
                v_norm = F.layer_norm(v, [v.shape[-1]], weight=gamma, bias=beta)
            else:
                v_norm = F.layer_norm(v, [v.shape[-1]])

            if w.dim() == 2 and w.shape[-1] == v.shape[1] and w.shape[0] == v.shape[1]:
                gate = torch.matmul(v_norm.transpose(1, 2), w).transpose(1, 2)
            else:
                gate = torch.matmul(v_norm, w)
            if b is not None and b.numel() > 0:
                gate = gate + b
            return u * gate

        @self.register("custom.pytorch", "HighwayBlock")
        @self.register("", "HighwayBlock")
        def _highway_block(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            h, gate = inputs[0], inputs[1]
            t = torch.sigmoid(gate)
            x = inputs[2] if len(inputs) > 2 else h
            return h * t + x * (1.0 - t)

        @self.register("custom.pytorch", "SqueezeExcitation")
        @self.register("", "SqueezeExcitation")
        def _squeeze_excitation(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            x = inputs[0]
            w1, b1 = inputs[1], inputs[2]
            w2, b2 = inputs[3], inputs[4]
            if x.dim() == 4:
                pooled = x.mean(dim=(-2, -1), keepdim=True)
                fc1 = (
                    F.conv2d(pooled, w1, b1)
                    if w1.dim() == 4
                    else torch.matmul(pooled.squeeze(-1).squeeze(-1), w1) + b1
                )
                act = F.relu(fc1)
                fc2 = (
                    F.conv2d(act, w2, b2)
                    if w2.dim() == 4
                    else torch.matmul(act, w2) + b2
                )
                scale = torch.sigmoid(
                    fc2 if w1.dim() == 4 else fc2.unsqueeze(-1).unsqueeze(-1)
                )
            else:
                pooled = x.mean(dim=-2)
                fc1 = torch.matmul(pooled, w1) + b1
                act = F.relu(fc1)
                fc2 = torch.matmul(act, w2) + b2
                scale = torch.sigmoid(fc2).unsqueeze(-2)
            return x * scale

        @self.register("custom.pytorch", "MultiHeadAttention")
        @self.register("custom.pytorch", "CrossAttention")
        @self.register("custom.pytorch", "StatefulAttention")
        @self.register("custom.pytorch", "GroupedQueryAttention")
        @self.register("", "MultiHeadAttention")
        def _mha_composite(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            num_heads = int(attrs.get("num_heads", 4))
            scale = attrs.get("scale")

            # Case 1: Separate Q, K, V passed (e.g. [q, k, v, out_w, out_b] or [q, k, v, out_w])
            if len(inputs) >= 3 and inputs[0] is not None and inputs[0].dim() == 4 and inputs[1] is not None and inputs[1].dim() == 4:
                q, k, v = inputs[0], inputs[1], inputs[2]
                out_w = inputs[3] if len(inputs) > 3 and inputs[3] is not None and inputs[3].dim() == 2 else None
                out_b = inputs[4] if len(inputs) > 4 and inputs[4] is not None and inputs[4].numel() > 0 else None

                # Handle GQA KV head repetition if k heads < q heads
                if k.shape[1] < q.shape[1]:
                    repeat_factor = q.shape[1] // k.shape[1]
                    k = k.repeat_interleave(repeat_factor, dim=1)
                    v = v.repeat_interleave(repeat_factor, dim=1)

                d_head = q.shape[-1]
                s = float(scale) if scale is not None else (1.0 / (d_head ** 0.5))
                scores = torch.matmul(q * s, k.transpose(-1, -2))
                probs = torch.softmax(scores, dim=-1)
                attn_out = torch.matmul(probs, v)

                B, H, S, D = attn_out.shape
                merged = attn_out.transpose(1, 2).contiguous().view(B, S, H * D)
                if out_w is not None:
                    out = torch.matmul(merged, out_w)
                    if out_b is not None:
                        out = out + out_b
                    return out
                return merged

            # Case 2: Unified QKV tensor
            qkv = inputs[0]
            out_w = inputs[1] if len(inputs) > 1 and inputs[1] is not None and inputs[1].dim() == 2 else None
            out_b = (
                inputs[2]
                if len(inputs) > 2 and inputs[2] is not None and inputs[2].numel() > 0
                else None
            )

            if qkv.dim() == 2:
                qkv = qkv.unsqueeze(0)
            if qkv.dim() == 4:
                B, H, S, D = qkv.shape
                qkv = qkv.transpose(1, 2).contiguous().view(B, S, H * D)

            B, S, total_d = qkv.shape
            d_head = total_d // (3 * num_heads) if total_d % (3 * num_heads) == 0 else total_d // num_heads
            if total_d % (3 * num_heads) == 0:
                qkv_reshaped = qkv.view(B, S, 3, num_heads, d_head)
                q = qkv_reshaped[:, :, 0].transpose(1, 2)
                k = qkv_reshaped[:, :, 1].transpose(1, 2)
                v = qkv_reshaped[:, :, 2].transpose(1, 2)
            else:
                q = k = v = qkv.view(B, S, num_heads, d_head).transpose(1, 2)

            s = float(scale) ** 0.5 if scale is not None else d_head**-0.25
            scores = torch.matmul(q * s, (k * s).transpose(-1, -2))
            probs = torch.softmax(scores, dim=-1)
            attn_out = torch.matmul(probs, v)

            merged = (
                attn_out.transpose(1, 2).contiguous().view(B, S, num_heads * d_head)
            )
            if out_w is not None:
                out = torch.matmul(merged, out_w)
                if out_b is not None:
                    out = out + out_b
                return out
            return merged

        @self.register("custom.pytorch", "GatedMultiHeadAttention")
        @self.register("", "GatedMultiHeadAttention")
        def _gated_mha_composite(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            x = inputs[0]
            if len(inputs) > 1 and inputs[1] is not None and inputs[1].numel() > 0:
                w = inputs[1]
                if w.dim() == 2 and x.dim() >= 2 and x.shape[-1] == w.shape[0]:
                    return torch.matmul(x, w)
            return x

        @self.register("custom.pytorch", "DynamicQuantLinear")
        @self.register("", "DynamicQuantLinear")
        def _dynamic_quant_linear(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            x = inputs[0]
            w = inputs[1]
            b = inputs[2] if len(inputs) > 2 and inputs[2] is not None else None

            # Dynamic uint8 quantization on x
            x_min = torch.minimum(
                torch.tensor(0.0, device=x.device, dtype=x.dtype), torch.min(x)
            )
            x_max = torch.maximum(
                torch.tensor(0.0, device=x.device, dtype=x.dtype), torch.max(x)
            )
            denom = 255.0
            scale = (x_max - x_min) / denom
            if scale == 0:
                scale = torch.tensor(1.0, device=x.device, dtype=torch.float32)
            else:
                scale = scale.to(torch.float32)
            zp = torch.clamp(torch.round(-x_min / scale), 0, 255).to(torch.uint8)
            q_x = torch.clamp(
                torch.round(x / scale) + zp.to(x.dtype), 0, 255
            ).to(torch.uint8)
            dq_x = (q_x.to(torch.float32) - zp.to(torch.float32)) * scale

            if w.dtype != dq_x.dtype:
                w = w.to(dq_x.dtype)
            out = torch.matmul(dq_x, w)
            if b is not None:
                if b.dtype != out.dtype:
                    b = b.to(out.dtype)
                out = out + b
            return out

        @self.register("custom.pytorch", "MultiFieldEmbedding")
        @self.register("", "MultiFieldEmbedding")
        def _multi_field_embedding(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            features = inputs[0]
            weights = inputs[1:]
            pe = None
            if len(weights) > 1 and weights[-1].dim() == features.dim():
                pe = weights[-1]
                weights = weights[:-1]
            out = None
            for idx, w in enumerate(weights):
                idx_slice = features[..., idx]
                emb = F.embedding(idx_slice, w)
                out = emb if out is None else out + emb
            if pe is not None:
                out = out + pe
            return out

        @self.register("custom.pytorch", "AttentionKeyPaddingMask")
        @self.register("", "AttentionKeyPaddingMask")
        def _attention_key_padding_mask(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            pad_mask = inputs[0]
            val = float(attrs.get("mask_value", -10000.0))
            mask = torch.zeros_like(pad_mask, dtype=torch.float32)
            return mask.masked_fill(pad_mask.bool(), val)

        @self.register("custom.pytorch", "SequenceClassifierHead")
        @self.register("", "SequenceClassifierHead")
        def _sequence_classifier_head(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            x = inputs[0]
            norm_w = inputs[1] if len(inputs) > 1 else None
            norm_b = inputs[2] if len(inputs) > 2 else None
            lin_w = inputs[3] if len(inputs) > 3 else None
            lin_b = inputs[4] if len(inputs) > 4 else None

            if norm_w is not None:
                x = F.layer_norm(x, [x.shape[-1]], weight=norm_w, bias=norm_b)
            if lin_w is not None:
                if lin_w.dim() == 2 and lin_w.shape[0] == x.shape[-1]:
                    x = torch.matmul(x, lin_w)
                    if lin_b is not None:
                        x = x + lin_b
                else:
                    x = F.linear(x, lin_w, lin_b)
            return x.squeeze(-1)



        @self.register("com.microsoft", "SkipLayerNormalization")
        @self.register("custom.pytorch", "SkipLayerNormalization")
        @self.register("", "SkipLayerNormalization")
        def _skip_layer_norm(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            x = inputs[0] + inputs[1]
            scale = (
                inputs[2]
                if len(inputs) > 2 and inputs[2] is not None and inputs[2].numel() > 0
                else None
            )
            bias = (
                inputs[3]
                if len(inputs) > 3 and inputs[3] is not None and inputs[3].numel() > 0
                else None
            )
            axis = attrs.get("axis", -1)
            eps = attrs.get("epsilon", 1e-5)
            normalized_shape = (
                list(x.shape[axis:])
                if axis >= 0
                else list(x.shape[len(x.shape) + axis :])
            )
            if scale is not None and scale.dtype != x.dtype:
                scale = scale.to(dtype=x.dtype)
            if bias is not None and bias.dtype != x.dtype:
                bias = bias.to(dtype=x.dtype)
            return F.layer_norm(x, normalized_shape, weight=scale, bias=bias, eps=eps)

        @self.register("", "MaxPool")
        def _max_pool(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            x = inputs[0]
            kernel_shape = attrs.get("kernel_shape", [2, 2])
            strides = attrs.get("strides", kernel_shape)
            pads = attrs.get("pads", [0, 0, 0, 0])
            if x.dim() == 4:
                if len(pads) == 4 and (pads[0] != pads[2] or pads[1] != pads[3]):
                    x = F.pad(x, (pads[1], pads[3], pads[0], pads[2]))
                    pads = [0, 0, 0, 0]
                padding = (pads[0], pads[1]) if len(pads) >= 2 else 0
                return F.max_pool2d(
                    x,
                    kernel_size=tuple(kernel_shape),
                    stride=tuple(strides),
                    padding=padding,
                )
            if x.dim() == 3:
                return F.max_pool1d(
                    x,
                    kernel_size=kernel_shape[0],
                    stride=strides[0],
                    padding=pads[0] if pads else 0,
                )
            msg = f"Unsupported MaxPool input dimension: {x.dim()}"
            raise NotImplementedError(msg)

        # -------------------------------------------------------------
        # Linear & Matrix Ops
        # -------------------------------------------------------------
        @self.register("", "MatMul")
        def _matmul(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            return torch.matmul(inputs[0], inputs[1])

        @self.register("", "Gemm")
        def _gemm(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            a, b = inputs[0], inputs[1]
            c = (
                inputs[2]
                if len(inputs) > 2 and inputs[2] is not None and inputs[2].numel() > 0
                else None
            )
            alpha = attrs.get("alpha", 1.0)
            beta = attrs.get("beta", 1.0)
            trans_a = attrs.get("transA", 0)
            trans_b = attrs.get("transB", 0)

            if trans_a:
                a = a.transpose(-1, -2)
            if trans_b:
                b = b.transpose(-1, -2)

            res = torch.matmul(a, b) * alpha
            if c is not None:
                res = res + (c * beta)
            return res

        # -------------------------------------------------------------
        # Tensor Layout & Shape Transformations
        # -------------------------------------------------------------
        @self.register("", "Transpose")
        def _transpose(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            perm = attrs.get("perm")
            if perm is not None:
                return inputs[0].permute(list(perm))
            return inputs[0].transpose(-1, -2)

        @self.register("", "Reshape")
        def _reshape(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            x = inputs[0]
            raw_shape = [int(v) for v in inputs[1].cpu().tolist()]
            resolved_shape = []
            for idx, dim in enumerate(raw_shape):
                if dim == 0:
                    resolved_shape.append(x.shape[idx])
                else:
                    resolved_shape.append(dim)
            return x.reshape(resolved_shape)

        @self.register("", "Concat")
        def _concat(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            axis = attrs.get("axis", 0)
            return torch.cat(inputs, dim=axis)

        @self.register("", "Split")
        def _split(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> list[torch.Tensor]:
            x = inputs[0]
            axis = attrs.get("axis", 0)
            if len(inputs) > 1 and inputs[1] is not None and inputs[1].numel() > 0:
                split_sizes = [int(v) for v in inputs[1].cpu().tolist()]
                return list(torch.split(x, split_sizes, dim=axis))
            if "split" in attrs:
                split_sizes = list(attrs["split"])
                return list(torch.split(x, split_sizes, dim=axis))
            if "num_outputs" in attrs:
                num = attrs["num_outputs"]
                split_size = x.shape[axis] // num
                return list(torch.split(x, split_size, dim=axis))
            return list(torch.split(x, 1, dim=axis))

        @self.register("", "Slice")
        def _slice(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            data = inputs[0]
            starts = [int(v) for v in inputs[1].cpu().tolist()]
            ends = [int(v) for v in inputs[2].cpu().tolist()]
            axes = (
                [int(v) for v in inputs[3].cpu().tolist()]
                if len(inputs) > 3
                else list(range(len(starts)))
            )
            steps = (
                [int(v) for v in inputs[4].cpu().tolist()]
                if len(inputs) > 4
                else [1] * len(starts)
            )

            # Build slice objects per dimension
            slices = [slice(None)] * data.dim()
            for start, end, axis, step in zip(starts, ends, axes, steps, strict=False):
                # Handle INT_MAX in ONNX
                end_val = (
                    None if (end >= 2147483647 or end >= 9223372036854775807) else end
                )
                slices[axis] = slice(start, end_val, step)
            return data[tuple(slices)]

        @self.register("", "Squeeze")
        def _squeeze(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            x = inputs[0]
            axes = inputs[1].tolist() if len(inputs) > 1 else attrs.get("axes")
            if axes:
                for ax in sorted(axes, reverse=True):
                    x = x.squeeze(ax)
                return x
            return x.squeeze()

        @self.register("", "Unsqueeze")
        def _unsqueeze(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            x = inputs[0]
            axes = inputs[1].tolist() if len(inputs) > 1 else attrs.get("axes", [])
            for ax in sorted(axes):
                x = x.unsqueeze(ax)
            return x

        @self.register("", "Tile")
        def _tile(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            x, repeats = inputs[0], inputs[1]
            rep_list = [int(v) for v in repeats.cpu().tolist()]
            return x.repeat(*rep_list)

        @self.register("", "Flatten")
        def _flatten(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            axis = attrs.get("axis", 1)
            x = inputs[0]
            pre = int(np.prod(x.shape[:axis]))
            post = int(np.prod(x.shape[axis:]))
            return x.reshape(pre, post)

        @self.register("", "Shape")
        def _shape(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            return torch.tensor(
                inputs[0].shape, dtype=torch.int64, device=inputs[0].device
            )

        @self.register("", "Gather")
        def _gather(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            data, indices = inputs[0], inputs[1]
            axis = attrs.get("axis", 0)
            if axis < 0:
                axis += data.dim()
            dim_size = data.shape[axis]
            if indices.dim() == 0:
                idx = int(indices.item())
                if idx < 0:
                    idx += dim_size
                return torch.select(data, axis, idx)
            indices_wrapped = torch.where(indices < 0, indices + dim_size, indices)
            return torch.index_select(data, axis, indices_wrapped.view(-1)).reshape(
                *data.shape[:axis], *indices.shape, *data.shape[axis + 1 :]
            )

        @self.register("", "Cast")
        def _cast(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            to_type = attrs.get("to", 1)
            # ONNX TensorProto enum: 1=FLOAT, 2=UINT8, 3=INT8, 4=UINT16, 5=INT16, 6=INT32, 7=INT64, 9=BOOL, 10=FLOAT16, 11=DOUBLE, 16=BFLOAT16
            type_map = {
                1: torch.float32,
                2: torch.uint8,
                3: torch.int8,
                4: getattr(torch, "uint16", torch.int32),
                5: torch.int16,
                6: torch.int32,
                7: torch.int64,
                9: torch.bool,
                10: torch.float16,
                11: torch.float64,
                16: torch.bfloat16,
            }
            target_torch = type_map.get(to_type, torch.float32)
            return inputs[0].to(target_torch)

        @self.register("", "Identity")
        def _identity(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            return inputs[0]

        @self.register("", "Constant")
        def _constant(
            inputs: list[torch.Tensor], attrs: dict[str, Any]
        ) -> torch.Tensor:
            if "value" in attrs:
                val = attrs["value"]
                if isinstance(val, np.ndarray):
                    return torch.from_numpy(val.copy())
                if isinstance(val, torch.Tensor):
                    return val
            if "value_float" in attrs:
                return torch.tensor(attrs["value_float"], dtype=torch.float32)
            if "value_int" in attrs:
                return torch.tensor(attrs["value_int"], dtype=torch.int64)
            if "value_floats" in attrs:
                return torch.tensor(attrs["value_floats"], dtype=torch.float32)
            if "value_ints" in attrs:
                return torch.tensor(attrs["value_ints"], dtype=torch.int64)
            msg = f"Constant op missing value attribute: {list(attrs.keys())}"
            raise ValueError(msg)


# Global default registry instance
global_registry = OpRegistry()
