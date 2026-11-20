#!/usr/bin/env python3
"""AST Analysis & Quantization Architecture Catalog Generator.

Scans all generator classes across generators/ using Python AST and produces
quantization_block_analysis.json mapping each class to its quantization
schemes, target ONNX nodes, and expected macro block fusions.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path


class ArchitectureASTVisitor(ast.NodeVisitor):
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.classes: dict[str, dict] = {}
        self.current_class: str | None = None
        self.current_class_info: dict | None = None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # We only care about nn.Module subclasses or generator architecture classes
        base_names = [ast.unparse(b) for b in node.bases]
        doc = ast.get_docstring(node) or ""

        self.current_class = node.name
        self.current_class_info = {
            "class_name": node.name,
            "source_file": self.filename,
            "docstring": doc.strip(),
            "bases": base_names,
            "submodules": [],
            "operations": set(),
        }

        # Traverse class body
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                self._analyze_function(item)

        self.current_class_info["operations"] = sorted(
            list(self.current_class_info["operations"])
        )
        self.classes[node.name] = self._enrich_quantization_profile(
            self.current_class_info
        )
        self.current_class = None
        self.current_class_info = None

    def _analyze_function(self, func: ast.FunctionDef) -> None:
        for n in ast.walk(func):
            # Detect submodule instantiations in __init__ (self.foo = nn.Bar(...))
            if isinstance(n, ast.Assign):
                for target in n.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        val_str = ast.unparse(n.value)
                        self.current_class_info["submodules"].append(
                            {"attribute": target.attr, "expression": val_str}
                        )

            # Detect function calls in forward
            if isinstance(n, ast.Call):
                call_name = ast.unparse(n.func)
                self.current_class_info["operations"].add(call_name)

            # Detect matrix multiplications and binary ops
            if isinstance(n, ast.BinOp):
                if isinstance(n.op, ast.MatMult):
                    self.current_class_info["operations"].add("@ (matmul)")
                elif isinstance(n.op, ast.Add):
                    self.current_class_info["operations"].add("+ (add)")
                elif isinstance(n.op, ast.Mult):
                    self.current_class_info["operations"].add("* (mul)")

    def _enrich_quantization_profile(self, info: dict) -> dict:
        cname = info["class_name"]
        file = info["source_file"]
        submods = [s["expression"] for s in info["submodules"]]
        submods_str = " ".join(submods).lower()
        ops_str = " ".join(info["operations"]).lower()
        doc_str = info["docstring"].lower()

        is_linear = any(
            k in submods_str or k in cname.lower() or k in doc_str
            for k in ["linear", "gemm", "dense", "matmul", "fc"]
        )
        is_conv = any(
            k in submods_str or k in cname.lower()
            for k in ["conv", "resnet", "mobilenet"]
        )
        is_attention = any(
            k in submods_str or k in cname.lower() or k in doc_str
            for k in [
                "attention",
                "mha",
                "sdpa",
                "multihead",
                "qkv",
                "rotary",
                "gqa",
                "alibi",
            ]
        )
        is_transformer = any(
            k in cname.lower() or "transformer" in file
            for k in [
                "transformer",
                "block",
                "encoder",
                "vit",
                "dit",
                "swin",
                "preln",
                "postln",
            ]
        )
        is_norm = any(
            k in submods_str or k in cname.lower()
            for k in ["norm", "layernorm", "rmsnorm", "groupnorm"]
        )
        is_act = (
            any(
                k in submods_str or k in cname.lower()
                for k in ["gelu", "silu", "swish", "mish", "relu"]
            )
            or "activation" in file
        )
        is_recurrent = any(
            k in submods_str or k in cname.lower()
            for k in ["rnn", "lstm", "gru"]
        )
        is_ssm = any(
            k in submods_str or k in cname.lower()
            for k in ["mamba", "s4", "rwkv", "scan"]
        )

        schemes = []
        target_nodes = []
        macro_block = "Standard"
        strategy = ""

        if is_transformer or (is_attention and is_linear):
            category = "transformer_and_attention_blocks"
            macro_block = "TransformerBlock"
            schemes = [
                "Dynamic QDQ (W8A8 Int8)",
                "Static QDQ (Int8/UInt8)",
                "Weight-Only Quantization (W8A16 / W4A16 AWQ/GPTQ)",
                "QOperator Dialect (QLinearMatMul)",
            ]
            target_nodes = [
                "MatMul",
                "Gemm",
                "DynamicQuantizeLinear",
                "DequantizeLinear",
                "LayerNormalization",
                "Add",
            ]
            strategy = (
                "Quantize Q, K, V linear projections and FFN intermediate/down projections via DynamicQuantLinear "
                "or QLinearMatMul; retain LayerNorm and Softmax in FP32/FP16 for numerical stability. "
                "The entire dual-residual DAG is topologically fused into a TransformerBlock macro-node."
            )
        elif is_attention:
            category = "attention_mechanisms"
            macro_block = "MultiHeadAttention"
            schemes = [
                "Dynamic QDQ (W8A8 Int8)",
                "Weight-Only (W8A16 / W4A16)",
                "QOperator Dialect (QLinearMatMul)",
            ]
            target_nodes = [
                "MatMul",
                "DynamicQuantizeLinear",
                "DequantizeLinear",
                "Softmax",
            ]
            strategy = (
                "Quantize input projections (Q, K, V) and out projection into DynamicQuantLinear nodes. "
                "Attention score matmul (Q @ K.T) and context matmul (Score @ V) can utilize dynamic 8-bit integer dot products."
            )
        elif is_conv:
            category = "convolutional_vision"
            macro_block = "QuantizedConv2D"
            schemes = [
                "Static QDQ (Int8/UInt8 per-channel)",
                "QOperator Dialect (QLinearConv)",
                "Channel-wise Symmetric Int8 Weights",
            ]
            target_nodes = [
                "Conv",
                "QuantizeLinear",
                "DequantizeLinear",
                "Add",
                "Relu",
            ]
            strategy = (
                "Calibrate activation scale/zero-point across dataset; quantize weights per-channel (axis=0). "
                "Fuse Conv + BN + ReLU into QLinearConv or QuantizedConv2D macro-block."
            )
        elif is_linear:
            category = "linear_projections_and_mlp"
            macro_block = "DynamicQuantLinear"
            schemes = [
                "Dynamic QDQ (DynamicQuantizeLinear -> DequantizeLinear -> MatMul)",
                "Weight-Only Quantization (W8A16, W4A16)",
                "QOperator Dialect (QLinearMatMul, IntegerGemm)",
            ]
            target_nodes = [
                "MatMul",
                "Gemm",
                "DynamicQuantizeLinear",
                "DequantizeLinear",
                "Add",
            ]
            strategy = (
                "Fuse activation dynamic quantization and dequantization with MatMul/Gemm and bias Add "
                "into unified DynamicQuantLinear macro-node."
            )
        elif is_norm:
            category = "normalization_variants"
            macro_block = "LayerNormalization" if "layer" in cname.lower() else "RMSNorm"
            schemes = [
                "Preserved High-Precision (FP32/FP16 Accumulation)",
                "Quantized Affine Scales (Int8 Weight Scale)",
            ]
            target_nodes = [
                "ReduceMean",
                "Sub",
                "Pow",
                "Sqrt",
                "Div",
                "Mul",
                "Add",
            ]
            strategy = (
                "Retain reduction and statistics in float precision to avoid division by zero; "
                "canonicalize decomposed variance/mean arithmetic into unified LayerNormalization or RMSNorm."
            )
        elif is_act:
            category = "activation_variants"
            macro_block = "FusedActivation"
            schemes = [
                "LUT-based Integer Quantization",
                "Preserved Math Approximation",
            ]
            target_nodes = ["Erf", "Mul", "Add", "Tanh", "Sigmoid"]
            strategy = (
                "Canonicalize decomposed activation formulas (e.g. 0.5 * x * (1 + erf(x / sqrt(2)))) "
                "into standard Gelu/SiLU nodes prior to integer QDQ pipeline."
            )
        elif is_recurrent or is_ssm:
            category = "recurrent_and_state_space"
            macro_block = "QuantizedRecurrentBlock"
            schemes = [
                "Weight-Only Quantization (W8A16 / W4A16)",
                "Static State Quantization",
            ]
            target_nodes = ["MatMul", "Add", "Sigmoid", "Tanh", "Split"]
            strategy = (
                "Quantize recurrent projection weights to Int8/Int4 to reduce memory bandwidth bottleneck; "
                "accumulate hidden states in FP16/FP32."
            )
        else:
            category = "general_deep_learning"
            macro_block = "QuantizedBlock"
            schemes = ["Dynamic QDQ (Int8)", "Static QDQ (Int8)"]
            target_nodes = ["MatMul", "Conv", "Add"]
            strategy = "Apply universal QDQ insertion around heavy compute nodes."

        return {
            "class_name": cname,
            "source_file": file,
            "category": category,
            "docstring": info["docstring"],
            "base_classes": info["bases"],
            "submodule_count": len(info["submodules"]),
            "operations_detected": info["operations"],
            "applicable_quantization_schemes": schemes,
            "target_onnx_nodes_for_quantization": target_nodes,
            "expected_macro_block_detection": macro_block,
            "quantization_injection_strategy": strategy,
        }


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    gen_dir = root / "generators"
    gen_files = sorted(gen_dir.glob("gen_*.py"))

    catalog: dict[str, dict] = {}
    for gfile in gen_files:
        with open(gfile, "r", encoding="utf-8") as f:
            src = f.read()
        tree = ast.parse(src, filename=str(gfile))
        visitor = ArchitectureASTVisitor(gfile.name)
        visitor.visit(tree)
        catalog.update(visitor.classes)

    out_json = root / "quantization_block_analysis.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    print(f"Successfully analyzed {len(catalog)} generator classes.")
    print(f"Catalog saved to {out_json}")


if __name__ == "__main__":
    main()
