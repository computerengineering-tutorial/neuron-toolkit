"""OpRegistry for PyTorch Runtime Graph Execution Engine."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger("neuron_toolkit.runtime.registry")

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
        """Retrieve handler for (domain, op_type) with fallback to default domain."""
        if (domain, op_type) in self._registry:
            return self._registry[(domain, op_type)]
        if ("", op_type) in self._registry:
            return self._registry[("", op_type)]
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
            return inputs[0] / inputs[1]

        @self.register("", "Neg")
        def _neg(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            return -inputs[0]

        @self.register("", "Abs")
        def _abs(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            return torch.abs(inputs[0])

        @self.register("", "Sqrt")
        def _sqrt(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            return torch.sqrt(inputs[0])

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

        @self.register("", "Where")
        def _where(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            cond = inputs[0] if inputs[0].dtype == torch.bool else inputs[0].bool()
            return torch.where(cond, inputs[1], inputs[2])

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
        def _reduce_l2(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            axes = attrs.get("axes")
            keepdims = bool(attrs.get("keepdims", 1))
            if axes is not None:
                dim = tuple(axes) if isinstance(axes, list) and len(axes) > 1 else (axes[0] if isinstance(axes, list) else axes)
            elif len(inputs) > 1 and inputs[1] is not None:
                axes_list = [int(v) for v in inputs[1].cpu().tolist()]
                dim = tuple(axes_list) if len(axes_list) > 1 else axes_list[0]
            else:
                dim = -1
            return torch.linalg.norm(inputs[0], ord=2, dim=dim, keepdim=keepdims)

        @self.register("", "Range")
        def _range(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            start = inputs[0].item()
            limit = inputs[1].item()
            delta = inputs[2].item() if len(inputs) > 2 and inputs[2] is not None else 1
            return torch.arange(start, limit, delta, dtype=inputs[0].dtype)

        @self.register("", "Einsum")
        def _einsum(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            eq = attrs.get("equation", "")
            if isinstance(eq, bytes):
                eq = eq.decode("utf-8")
            return torch.einsum(eq, *inputs)

        @self.register("", "ConstantOfShape")
        def _constant_of_shape(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            shape = [int(v) for v in inputs[0].cpu().tolist()]
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
            return torch.full(shape, fill_val, dtype=dtype)

        @self.register("", "ScatterND")
        def _scatter_nd(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            data, indices, updates = inputs[0], inputs[1], inputs[2]
            out = data.clone()
            idx_tuple = tuple(indices.select(-1, i).long() for i in range(indices.shape[-1]))
            out[idx_tuple] = updates
            return out

        # -------------------------------------------------------------
        # Activations
        # -------------------------------------------------------------
        @self.register("", "Relu")
        def _relu(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            return F.relu(inputs[0])

        @self.register("", "Sigmoid")
        def _sigmoid(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            return torch.sigmoid(inputs[0])

        @self.register("", "Tanh")
        def _tanh(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            return torch.tanh(inputs[0])

        @self.register("", "LogSoftmax")
        def _log_softmax(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
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

        @self.register("custom.pytorch", "Math")
        def _math_op(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            expr = attrs.get("expr") or attrs.get("folded_expr")
            if expr and not inputs:
                # Constant expression
                return torch.tensor(eval(expr, {"__builtins__": {}}), dtype=torch.float32)
            if expr and len(inputs) == 1:
                x = inputs[0]
                return eval(expr, {"__builtins__": {}, "x": x, "torch": torch})
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
            b = inputs[2] if len(inputs) > 2 and inputs[2] is not None and inputs[2].numel() > 0 else None

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
                    dilation=tuple(dilations) if isinstance(dilations, list) else dilations,
                    groups=group,
                )
            elif x.dim() == 3:
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
            msg = f"Unsupported Conv input dimension: {x.dim()}"
            raise NotImplementedError(msg)

        @self.register("", "BatchNormalization")
        def _batch_norm(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            x, scale, b, mean, var = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
            eps = attrs.get("epsilon", 1e-5)
            return F.batch_norm(x, mean, var, weight=scale, bias=b, training=False, eps=eps)

        @self.register("custom.pytorch", "GroupNorm")
        @self.register("", "GroupNorm")
        def _group_norm(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            x, scale, b = inputs[0], inputs[1], inputs[2]
            num_groups = int(attrs.get("num_groups", 1))
            eps = float(attrs.get("epsilon", 1e-5))
            w = scale.view(-1) if scale is not None and scale.numel() > 0 else None
            bias = b.view(-1) if b is not None and b.numel() > 0 else None
            return F.group_norm(x, num_groups=num_groups, weight=w, bias=bias, eps=eps)

        @self.register("custom.pytorch", "InstanceNorm1d")
        @self.register("", "InstanceNormalization")
        def _instance_norm(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            x, scale, b = inputs[0], inputs[1], inputs[2]
            eps = attrs.get("epsilon", 1e-5)
            w = scale.view(-1) if scale is not None and scale.numel() > 0 else None
            bias = b.view(-1) if b is not None and b.numel() > 0 else None
            return F.instance_norm(x, weight=w, bias=bias, eps=eps)

        @self.register("custom.pytorch", "RMSNorm")
        @self.register("", "RMSNorm")
        def _rms_norm(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            x = inputs[0]
            gamma = inputs[1] if len(inputs) > 1 and inputs[1] is not None and inputs[1].numel() > 0 else None
            eps = float(attrs.get("epsilon", 1e-12))
            norm = torch.norm(x, p=2, dim=-1, keepdim=True)
            scale = (x.shape[-1] ** 0.5) / torch.clamp(norm, min=eps)
            res = x * scale
            if gamma is not None:
                res = res * gamma
            return res

        @self.register("custom.pytorch", "RotaryEmbedding")
        @self.register("", "RotaryEmbedding")
        def _rotary_embedding(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            x = inputs[0]
            seq_len = x.shape[-2]
            if len(inputs) == 3:
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
            orig_shape = x.shape
            x_reshaped = x.reshape(*orig_shape[:-1], -1, 2)
            rot = torch.stack((-x_reshaped[..., 1], x_reshaped[..., 0]), dim=-1).reshape(orig_shape)
            return (x * cos) + (rot * sin)

        @self.register("custom.pytorch", "Unbind")
        @self.register("", "Unbind")
        def _unbind(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> list[torch.Tensor]:
            dim = attrs.get("axis", attrs.get("dim", 0))
            return list(torch.unbind(inputs[0], dim=dim))

        @self.register("custom.pytorch", "GatedUnit")
        @self.register("", "GatedUnit")
        def _gated_unit(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            x, gate = inputs[0], inputs[1]
            if gate.dim() == 3 and x.dim() == 4:
                # gate: (B, S, H) -> (B, H, S, 1) to match x: (B, H, S, D)
                gate = gate.transpose(1, 2).unsqueeze(-1)
            return x * torch.sigmoid(gate)

        @self.register("custom.pytorch", "SoftmaxMatMul")
        @self.register("", "SoftmaxMatMul")
        def _softmax_matmul(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            scores, v = inputs[0], inputs[1]
            axis = attrs.get("axis", -1)
            return torch.matmul(torch.softmax(scores, dim=axis), v)

        @self.register("custom.pytorch", "Linear")
        @self.register("", "Linear")
        def _linear(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            x, w = inputs[0], inputs[1]
            bias = inputs[2] if len(inputs) > 2 and inputs[2] is not None and inputs[2].numel() > 0 else None
            out = torch.matmul(x, w)
            if bias is not None:
                out = out + bias
            return out

        @self.register("custom.pytorch", "ScaledDotProductAttention")
        @self.register("custom.pytorch", "Attention")
        @self.register("", "ScaledDotProductAttention")
        def _attention(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            q, k, v = inputs[0], inputs[1], inputs[2]
            scale = attrs.get("scale", None)
            if scale is not None:
                s = float(scale) ** 0.5
            else:
                s = q.shape[-1] ** -0.25
            scores = torch.matmul(q * s, (k * s).transpose(-1, -2))
            probs = torch.softmax(scores, dim=-1)
            return torch.matmul(probs, v)

        @self.register("custom.pytorch", "SplitQKV")
        @self.register("", "SplitQKV")
        def _split_qkv(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> list[torch.Tensor]:
            x = inputs[0]
            num_heads = int(attrs.get("num_heads", 4))
            orig_shape = x.shape
            B, S, total_d = orig_shape[0], orig_shape[1], orig_shape[2]
            D = total_d // (3 * num_heads)
            reshaped = x.view(B, S, 3, num_heads, D).permute(2, 0, 3, 1, 4)
            return [reshaped[0], reshaped[1], reshaped[2]]

        @self.register("custom.pytorch", "MergeHeads")
        @self.register("", "MergeHeads")
        def _merge_heads(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            x = inputs[0]
            # x shape: (B, H, S, D)
            B, H, S, D = x.shape
            return x.transpose(1, 2).reshape(B, S, H * D)

        @self.register("custom.pytorch", "AxialRearrange")
        @self.register("", "AxialRearrange")
        def _axial_rearrange(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            x = inputs[0]
            mode = attrs.get("mode", "freq_to_time")
            b = int(attrs.get("batch_size", 1))
            if mode == "add_channel":
                return x.unsqueeze(1)
            elif mode == "freq_in":
                # (B, C, F, T) -> (B*T, F, C)
                B, C, F, T = x.shape
                return x.permute(0, 3, 2, 1).reshape(B * T, F, C)
            elif mode == "freq_to_time":
                # (B*T, F, C) -> (B*F, T, C)
                BT, F, C = x.shape
                T = BT // b
                return x.view(b, T, F, C).permute(0, 2, 1, 3).reshape(b * F, T, C)
            elif mode == "time_to_conv":
                # (B*F, T, C) -> (B, C, F, T)
                BF, T, C = x.shape
                F = BF // b
                return x.view(b, F, T, C).permute(0, 3, 1, 2)
            elif mode == "flatten_fc":
                # (B, C, F, T) -> (B, T, F*C)
                B, C, F, T = x.shape
                return x.permute(0, 3, 2, 1).reshape(B, T, F * C)
            raise ValueError(f"Unknown AxialRearrange mode: {mode}")

        @self.register("custom.pytorch", "TensorRoute")
        @self.register("custom.pytorch", "IndexSelect")
        def _tensor_route(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            x = inputs[0]
            axis = attrs.get("axis", 1)
            index = attrs.get("index")
            if index is not None:
                x = torch.select(x, dim=axis, index=int(index))
            slice_axis = attrs.get("slice_axis")
            slice_start = attrs.get("slice_start")
            slice_end = attrs.get("slice_end")
            if slice_axis is not None and (slice_start is not None or slice_end is not None):
                slices = [slice(None)] * x.dim()
                start = slice_start if slice_start is not None else 0
                end = slice_end
                if end is not None and end < 0:
                    end = x.shape[slice_axis] + end
                slices[slice_axis] = slice(start, end)
                x = x[tuple(slices)]
            return x

        @self.register("", "LayerNormalization")
        def _layer_norm(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            x = inputs[0]
            scale = inputs[1]
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
            return F.layer_norm(x, normalized_shape, weight=scale, bias=bias, eps=eps)

        @self.register("", "MaxPool")
        def _max_pool(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
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
            elif x.dim() == 3:
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
            c = inputs[2] if len(inputs) > 2 and inputs[2] is not None and inputs[2].numel() > 0 else None
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
        def _transpose(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
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
        def _split(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> list[torch.Tensor]:
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
            axes = [int(v) for v in inputs[3].cpu().tolist()] if len(inputs) > 3 else list(range(len(starts)))
            steps = [int(v) for v in inputs[4].cpu().tolist()] if len(inputs) > 4 else [1] * len(starts)

            # Build slice objects per dimension
            slices = [slice(None)] * data.dim()
            for start, end, axis, step in zip(starts, ends, axes, steps, strict=False):
                # Handle INT_MAX in ONNX
                if end >= 2147483647 or end >= 9223372036854775807:
                    end = None
                slices[axis] = slice(start, end, step)
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
        def _unsqueeze(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            x = inputs[0]
            axes = inputs[1].tolist() if len(inputs) > 1 else attrs.get("axes", [])
            for ax in sorted(axes):
                x = x.unsqueeze(ax)
            return x

        @self.register("", "Flatten")
        def _flatten(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            axis = attrs.get("axis", 1)
            x = inputs[0]
            pre = int(np.prod(x.shape[:axis]))
            post = int(np.prod(x.shape[axis:]))
            return x.reshape(pre, post)

        @self.register("", "Shape")
        def _shape(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            return torch.tensor(inputs[0].shape, dtype=torch.int64)

        @self.register("", "Gather")
        def _gather(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            data, indices = inputs[0], inputs[1]
            axis = attrs.get("axis", 0)
            return torch.index_select(data, axis, indices.view(-1)).reshape(
                *data.shape[:axis], *indices.shape, *data.shape[axis + 1 :]
            ) if indices.dim() > 0 else torch.select(data, axis, int(indices.item()))

        @self.register("", "Cast")
        def _cast(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            to_type = attrs.get("to", 1)
            # ONNX TensorProto enum: 1=FLOAT, 7=INT64, 6=INT32, 10=FLOAT16, 9=BOOL, 11=DOUBLE
            type_map = {
                1: torch.float32,
                6: torch.int32,
                7: torch.int64,
                9: torch.bool,
                10: torch.float16,
                11: torch.float64,
            }
            target_torch = type_map.get(to_type, torch.float32)
            return inputs[0].to(target_torch)

        @self.register("", "Identity")
        def _identity(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
            return inputs[0]

        @self.register("", "Constant")
        def _constant(inputs: list[torch.Tensor], attrs: dict[str, Any]) -> torch.Tensor:
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
