"""Evaluation and differential numerical verification suite."""

from __future__ import annotations

import contextlib
import functools
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from neuron_toolkit.graph import NeuronGraph

log = logging.getLogger("neuron_toolkit.eval")


def _get_tflite_interpreter_cls() -> Any:
    """Resolve available TFLite/LiteRT Interpreter class across backends."""
    for mod_path, cls_name in [
        ("ai_edge_litert.interpreter", "Interpreter"),
        ("tflite_runtime.interpreter", "Interpreter"),
        ("tensorflow.lite", "Interpreter"),
    ]:
        with contextlib.suppress(Exception):
            mod = __import__(mod_path, fromlist=[cls_name])
            return getattr(mod, cls_name)
    return None


@functools.lru_cache(maxsize=1)
def _is_tflite_runnable() -> bool:
    """Check if LiteRT/TFLite interpreter can be loaded and run."""
    return _get_tflite_interpreter_cls() is not None


class GraphVerifier:
    """Verifies numerical equivalence between a NeuronGraph (or subgraph) and an external model/module."""

    def __init__(self, graph: NeuronGraph) -> None:
        self.graph = graph

    def compare_outputs(
        self,
        start_points: list[str | object] | str | object,
        end_points: list[str | object] | str | object,
        pytorch_module: Any,
        inputs_dict: dict[str, Any],
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> dict[str, Any]:
        """Execute the sliced model and PyTorch module on inputs and compare outputs."""
        import torch

        # 1. Run the PyTorch module
        with torch.no_grad():
            torch_inputs = []
            if len(inputs_dict) == 1:
                val = next(iter(inputs_dict.values()))
                torch_inputs.append(torch.from_numpy(np.asarray(val)))
            else:
                import inspect

                callable_module = getattr(pytorch_module, "forward", pytorch_module)
                params = list(inspect.signature(callable_module).parameters.keys())
                unused = dict(inputs_dict)
                torch_inputs.extend(
                    torch.from_numpy(np.asarray(unused.pop(param)))
                    for param in params
                    if param in unused
                )
                while len(torch_inputs) < len(params) and unused:
                    key = next(iter(unused))
                    val = unused.pop(key)
                    torch_inputs.append(torch.from_numpy(np.asarray(val)))

            pytorch_outputs = pytorch_module(*torch_inputs)

        # Normalize PyTorch outputs to numpy arrays
        if isinstance(pytorch_outputs, torch.Tensor):
            pytorch_outputs_np = [pytorch_outputs.cpu().numpy()]
        elif isinstance(pytorch_outputs, tuple):
            pytorch_outputs_np = [
                t.cpu().numpy() for t in pytorch_outputs if isinstance(t, torch.Tensor)
            ]
        elif isinstance(pytorch_outputs, dict):
            pytorch_outputs_np = {
                k: v.cpu().numpy()
                for k, v in pytorch_outputs.items()
                if isinstance(v, torch.Tensor)
            }
        else:
            pytorch_outputs_np = [pytorch_outputs]

        # 2. Run the sliced model
        sliced_graph = self.graph.slice(start_points, end_points)
        if hasattr(sliced_graph._backend, "model"):  # ONNX
            import onnxruntime as ort

            model_bytes = sliced_graph._backend.model.SerializeToString()
            session = ort.InferenceSession(model_bytes)

            run_inputs = {}
            session_inputs = [i.name for i in session.get_inputs()]
            used_keys: set[str] = set()
            for name in session_inputs:
                if name in inputs_dict:
                    run_inputs[name] = inputs_dict[name]
                    used_keys.add(name)
                    continue
                matched = False
                for k, v in inputs_dict.items():
                    if k in used_keys:
                        continue
                    if k in name or name in k:
                        run_inputs[name] = v
                        matched = True
                        used_keys.add(k)
                        break
                if not matched and len(inputs_dict) == 1:
                    run_inputs[name] = next(iter(inputs_dict.values()))
                elif not matched and len(inputs_dict) == len(session_inputs):
                    for k, v in inputs_dict.items():
                        if k not in used_keys:
                            run_inputs[name] = v
                            used_keys.add(k)
                            matched = True
                            break

            output_names = [o.name for o in session.get_outputs()]
            outputs_list = session.run(output_names, run_inputs)
            model_outputs = dict(zip(output_names, outputs_list, strict=False))
        else:  # TFLite
            InterpreterCls = _get_tflite_interpreter_cls()
            if InterpreterCls is None:
                msg = "LiteRT / TFLite interpreter is not available or not usable on this CPU."
                raise RuntimeError(msg)

            model_bytes = sliced_graph._backend.source
            if not isinstance(model_bytes, bytes):
                msg = "Sliced TFLite model source must be bytes."
                raise ValueError(msg)

            interpreter = InterpreterCls(model_content=model_bytes)
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            tflite_used_keys: set[str] = set()
            for detail in input_details:
                name = detail["name"]
                if name in inputs_dict:
                    interpreter.set_tensor(detail["index"], inputs_dict[name])
                    tflite_used_keys.add(name)
                    continue
                matched = False
                for k, v in inputs_dict.items():
                    if k in tflite_used_keys:
                        continue
                    if k in name or name in k:
                        interpreter.set_tensor(detail["index"], v)
                        matched = True
                        tflite_used_keys.add(k)
                        break
                if not matched and len(inputs_dict) == 1:
                    interpreter.set_tensor(
                        detail["index"], next(iter(inputs_dict.values()))
                    )
                elif not matched and len(inputs_dict) == len(input_details):
                    for k, v in inputs_dict.items():
                        if k not in tflite_used_keys:
                            interpreter.set_tensor(detail["index"], v)
                            tflite_used_keys.add(k)
                            matched = True
                            break

            interpreter.invoke()

            model_outputs = {}
            for detail in output_details:
                model_outputs[detail["name"]] = interpreter.get_tensor(detail["index"])

        # 3. Match and compare
        declared_endpoints = (
            [end_points]
            if isinstance(end_points, str)
            else (
                list(end_points)
                if isinstance(end_points, (list, tuple))
                else [end_points]
            )
        )
        if declared_endpoints and all(isinstance(e, str) for e in declared_endpoints):
            model_outs_list = [
                model_outputs[e] for e in declared_endpoints if e in model_outputs
            ]
        else:
            model_outs_list = list(model_outputs.values())
        matches = []
        if isinstance(pytorch_outputs_np, dict):
            for k, py_val in pytorch_outputs_np.items():
                model_val = None
                for mk, mv in model_outputs.items():
                    if k in mk or mk in k:
                        model_val = mv
                        break
                if model_val is None and len(model_outputs) == 1:
                    model_val = next(iter(model_outputs.values()))

                if model_val is not None:
                    matches.append((k, py_val, model_val))
        else:
            matched_endpoint = False
            if len(pytorch_outputs_np) == 1:
                end_pts_list = (
                    [end_points] if not isinstance(end_points, list) else end_points
                )
                for ep in end_pts_list:
                    if isinstance(ep, str) and ep in model_outputs:
                        matches.append((ep, pytorch_outputs_np[0], model_outputs[ep]))
                        matched_endpoint = True
                        break

            if not matched_endpoint:
                model_names = list(model_outputs.keys())
                for i, py_val in enumerate(pytorch_outputs_np):
                    if i < len(model_outs_list):
                        name = model_names[i] if i < len(model_names) else f"output_{i}"
                        matches.append((name, py_val, model_outs_list[i]))

        report = {}
        for name, py_val, raw_model_val in matches:
            curr_model_val = raw_model_val
            if py_val.shape != curr_model_val.shape:
                with contextlib.suppress(ValueError):
                    curr_model_val = curr_model_val.reshape(py_val.shape)

            mae = np.mean(np.abs(py_val - curr_model_val))
            mse = np.mean((py_val - curr_model_val) ** 2)

            py_flat = py_val.flatten()
            model_flat = curr_model_val.flatten()
            norm_py = np.linalg.norm(py_flat)
            norm_model = np.linalg.norm(model_flat)
            if norm_py > 0 and norm_model > 0:
                cos_sim = np.dot(py_flat, model_flat) / (norm_py * norm_model)
            else:
                cos_sim = 1.0 if norm_py == norm_model else 0.0

            all_close = np.allclose(py_val, curr_model_val, rtol=rtol, atol=atol)

            report[name] = {
                "mae": float(mae),
                "mse": float(mse),
                "cosine_similarity": float(cos_sim),
                "all_close": bool(all_close),
                "shape_match": py_val.shape == curr_model_val.shape,
            }

        return report

    def verify_against_pytorch(
        self,
        start_points: list[str | object] | str | object,
        end_points: list[str | object] | str | object,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> Any:
        """Decorator to verify PyTorch layer/function outputs against the model's sliced subgraph."""

        def decorator(func: Any) -> Any:
            from functools import wraps

            import torch

            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                py_out = func(*args, **kwargs)
                inputs_dict = {}
                func_args = args
                if (
                    len(args) > 0
                    and hasattr(args[0], "__class__")
                    and isinstance(args[0], torch.nn.Module)
                ):
                    func_args = args[1:]

                for idx, arg in enumerate(func_args):
                    if isinstance(arg, torch.nn.Module):
                        continue
                    if isinstance(arg, torch.Tensor):
                        inputs_dict[f"input_{idx}"] = arg.cpu().detach().numpy()
                    elif isinstance(arg, np.ndarray):
                        inputs_dict[f"input_{idx}"] = arg

                for k, v in kwargs.items():
                    if isinstance(v, torch.Tensor):
                        inputs_dict[k] = v.cpu().detach().numpy()
                    elif isinstance(v, np.ndarray):
                        inputs_dict[k] = v

                try:
                    report = self.compare_outputs(
                        start_points=start_points,
                        end_points=end_points,
                        pytorch_module=func,
                        inputs_dict=inputs_dict,
                        rtol=rtol,
                        atol=atol,
                    )
                    wrapper.similarity_report = report
                except Exception as e:
                    log.warning("PyTorch verification comparison failed: %s", e)
                    wrapper.similarity_report = {"error": str(e)}

                return py_out

            return wrapper

        return decorator
