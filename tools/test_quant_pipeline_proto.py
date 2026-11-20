#!/usr/bin/env python3
"""Prototype script to verify PyTorch and ONNX quantization transformations."""

import copy
import sys
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
                # Insert DQL -> DQ -> MatMul
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
    from generators.gen_transformer_advanced_zoo import PreLN_TransformerBlock
    block = PreLN_TransformerBlock(d_model=32, num_heads=4, d_ffn=64).eval()
    x = torch.randn(2, 8, 32)
    
    # 1. Quantize in PyTorch
    q_block = quantize_pytorch_module(block)
    pt_out = q_block(x).detach().numpy()
    
    # 2. Export base block to ONNX
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_path = tmp.name
    torch.onnx.export(block, x, tmp_path, input_names=["x"], output_names=["out"], opset_version=18, dynamo=False)
    raw_onx = onnx.load(tmp_path)
    
    # 3. Quantize ONNX
    q_onx = quantize_onnx_model(raw_onx)
    print("Raw Quantized ONNX node count:", len(q_onx.graph.node))
    
    # 4. Compile with NeuronCompiler
    opt_onx = NeuronCompiler.compile(q_onx, mode="function_proto")
    print("Optimized ONNX node count:", len(opt_onx.graph.node))
    tb_nodes = [n for n in opt_onx.graph.node if "TransformerBlock" in n.op_type]
    print(f"Detected TransformerBlock nodes: {len(tb_nodes)}")
    
    # 5. Cross verification with PyTorchRunner
    runner = PyTorchRunner(opt_onx)
    onx_out = runner.run({"x": x.numpy()})["out"]
    
    diff = np.abs(pt_out - onx_out)
    max_diff = np.max(diff)
    print(f"Max diff between PyTorch quantized and ONNX simplified: {max_diff:.6f}")
    assert max_diff < 0.1, f"Max diff too large: {max_diff}"
    print("SUCCESS: Prototype verification passed!")

if __name__ == "__main__":
    main()
