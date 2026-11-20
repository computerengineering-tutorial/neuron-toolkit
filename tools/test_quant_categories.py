#!/usr/bin/env python3
"""Test quantization and compilation across representative blocks from each category."""

import copy
import sys
import tempfile
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, numpy_helper
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neuron_toolkit.compiler.pipeline import NeuronCompiler
from neuron_toolkit.runtime.cross_verifier import CrossVerifier
from neuron_toolkit.runtime.runner import PyTorchRunner
from generators.registry import ModelRegistry

class QuantizedLinear(nn.Module):
    """Dynamically quantized Linear layer with 8-bit simulated QDQ."""
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        w = linear.weight.data
        scale_w = (w.abs().max() / 127.0).clamp(min=1e-8)
        w_q = torch.clamp(torch.round(w / scale_w), -128, 127)
        self.register_buffer("weight_scale", scale_w)
        self.register_buffer("weight_q", w_q)
        self.bias = nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale_x = (x.abs().max() / 127.0).clamp(min=1e-8)
        x_q = torch.clamp(torch.round(x / scale_x), -128, 127)
        x_dq = x_q * scale_x
        w_dq = self.weight_q * self.weight_scale
        return F.linear(x_dq, w_dq, self.bias)

def quantize_pytorch_module(model: nn.Module) -> nn.Module:
    """Recursively replace nn.Linear with QuantizedLinear."""
    q_model = copy.deepcopy(model)
    def _replace_layers(module: nn.Module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                setattr(module, name, QuantizedLinear(child))
            else:
                _replace_layers(child)
    _replace_layers(q_model)
    q_model.eval()
    return q_model

def quantize_onnx_model(model: onnx.ModelProto) -> onnx.ModelProto:
    """Convert MatMul / Linear projections in ONNX to DynamicQuantizeLinear pattern."""
    q_model = copy.deepcopy(model)
    graph = q_model.graph
    init_map = {i.name: i for i in graph.initializer}
    
    new_nodes = []
    quant_count = 0
    
    for n in graph.node:
        if n.op_type == "MatMul" and len(n.input) == 2:
            act_in, weight_in = n.input[0], n.input[1]
            if weight_in in init_map:
                q_x = f"{act_in}_q_{quant_count}"
                scale_x = f"{act_in}_scale_{quant_count}"
                zp_x = f"{act_in}_zp_{quant_count}"
                dq_x = f"{act_in}_dq_{quant_count}"
                
                dql = helper.make_node(
                    "DynamicQuantizeLinear",
                    inputs=[act_in],
                    outputs=[q_x, scale_x, zp_x],
                    name=f"dql_{quant_count}",
                )
                dq = helper.make_node(
                    "DequantizeLinear",
                    inputs=[q_x, scale_x, zp_x],
                    outputs=[dq_x],
                    name=f"dq_{quant_count}",
                )
                mm = helper.make_node(
                    "MatMul",
                    inputs=[dq_x, weight_in],
                    outputs=n.output,
                    name=n.name if n.name else f"mm_quant_{quant_count}",
                )
                new_nodes.extend([dql, dq, mm])
                quant_count += 1
                continue
        new_nodes.append(n)
        
    del graph.node[:]
    graph.node.extend(new_nodes)
    onnx.checker.check_model(q_model)
    return q_model

def main():
    specs = ModelRegistry.discover(REPO_ROOT / "generators")
    spec_dict = {s.name: s for s in specs}
    
    test_models = [
        "activation_erf_gelu",
        "mha_separate_qkv",
        "mlp_standard",
        "swiglu_mlp_block",
        "norm_rms",
        "transformer_pre_ln",
        "transformer_post_ln",
        "linear_standard",
    ]
    
    for mname in test_models:
        if mname not in spec_dict:
            print(f"Skipping {mname} (not found)")
            continue
        spec = spec_dict[mname]
        try:
            pt_model = spec.instantiate().eval()
            dummy_inputs = spec.get_dummy_inputs()
            
            # 1. PyTorch quantization
            q_pt_model = quantize_pytorch_module(pt_model)
            with torch.no_grad():
                pt_out = q_pt_model(*dummy_inputs)
                if isinstance(pt_out, (tuple, list)):
                    pt_out = pt_out[0]
                pt_out_np = pt_out.detach().numpy()
                
            # 2. ONNX export & quantize
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
                tmp_path = tmp.name
            torch.onnx.export(pt_model, dummy_inputs, tmp_path, input_names=spec.input_names, output_names=spec.output_names, opset_version=18, dynamo=False)
            raw_onx = onnx.load(tmp_path)
            q_onx = quantize_onnx_model(raw_onx)
            
            # 3. Compile with NeuronCompiler
            opt_onx = NeuronCompiler.compile(q_onx, mode="function_proto")
            
            # 4. Run PyTorchRunner on compiled ONNX
            runner = PyTorchRunner(opt_onx)
            feed_dict = {inp_name: inp_t.numpy() for inp_name, inp_t in zip(spec.input_names, dummy_inputs)}
            onx_outs = runner.run(feed_dict)
            main_out = next(iter(onx_outs.values()))
            
            # 5. Differential check
            diff = np.abs(pt_out_np - main_out)
            max_diff = np.max(diff)
            print(f"[{mname}] raw_nodes={len(q_onx.graph.node)} -> opt_nodes={len(opt_onx.graph.node)} | max_diff={max_diff:.5f} | PASS")
        except Exception as e:
            print(f"[{mname}] FAILED: {e}")

if __name__ == "__main__":
    main()
