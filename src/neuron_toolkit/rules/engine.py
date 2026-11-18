"""Declarative JSON Pattern Rules Engine for ONNX Graph Transformation.

Refactored facade delegating to the modular 3-Stage Optimization Pipeline:
Stage 1: Micro-Canonicalization (Shape, Act, Norm, Math)
Stage 2: Macro-Block Synthesis (Attention, FFN, RoPE, Axial)
Stage 3: Target Exporters (ONNX / Netron FunctionProto / PyTorch)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import onnx

from neuron_toolkit.compiler.base import CompilerContext
from neuron_toolkit.compiler.pipeline import OptimizationPipeline
from neuron_toolkit.rules.schema import JSONRule

if TYPE_CHECKING:
    from neuron_toolkit.graph import NeuronGraph

log = logging.getLogger("neuron_toolkit.rules.engine")


class JSONRuleEngine:
    """Facade engine that loads declarative JSON rewrite rules and transforms ONNX models.

    Delegates execution directly to the modular 3-Stage Optimization Pipeline.
    """

    def __init__(self, mode: str = "function_proto") -> None:
        """Initialize engine.

        Args:
            mode: 'modern_onnx' (converts to modern standard ONNX Opset 23+ ops like RMSNormalization, Gelu, GroupNormalization),
                  'function_proto' (dual-mode with formal FunctionProto attached for Netron visual hierarchy),
                  'plain_custom_op' or 'pytorch_custom' (minimalist custom op node without function body for PyTorch runtime).
        """
        self.mode = mode
        self.rules: list[JSONRule] = []
        self.pipeline = OptimizationPipeline(mode=mode)

    def load_rule_from_file(self, path: str | Path) -> JSONRule:
        """Load and register a single JSON rule file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        rule = JSONRule.from_dict(data)
        self.rules.append(rule)
        return rule

    def load_rules_from_dir(
        self, directory: str | Path | None = None
    ) -> list[JSONRule]:
        """Load all JSON rules from a directory (defaults to built-in definitions)."""
        if directory is None:
            directory = Path(__file__).parent / "definitions"
        else:
            directory = Path(directory)

        loaded: list[JSONRule] = []
        for json_path in sorted(directory.glob("*.json")):
            loaded.append(self.load_rule_from_file(json_path))
        return loaded

    def apply(
        self,
        target: onnx.ModelProto | NeuronGraph,
        rule_ids: list[str] | None = None,
    ) -> onnx.ModelProto:
        """Apply loaded rules to the model.

        Returns an optimized/simplified ONNX ModelProto.
        """
        context = CompilerContext(
            mode=self.mode,
            rule_ids=rule_ids
            if rule_ids is not None
            else ([r.rule_id for r in self.rules] if self.rules else None),
        )
        return self.pipeline.run(target, context)

    # -------------------------------------------------------------------------
    # Backward-Compatible Pass Delegates
    # -------------------------------------------------------------------------
    def apply_cast_initializer_folding(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fold Cast nodes converting static initializers."""
        return self.pipeline.shape_pass.apply_cast_initializer_folding(model)

    def apply_cast_conv_weight_folding(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Alias for backward compatibility."""
        return self.pipeline.shape_pass.apply_cast_conv_weight_folding(model)

    def apply_erf_gelu_fusion(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse ERF GELU expansion."""
        return self.pipeline.act_pass.apply_erf_gelu_fusion(model, mode=self.mode)

    def _attach_gelu_function_def(self, model: onnx.ModelProto) -> None:
        self.pipeline.act_pass._attach_gelu_function_def(model)

    def apply_tanh_gelu_fusion(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse Tanh GELU approximation."""
        return self.pipeline.act_pass.apply_tanh_gelu_fusion(model, mode=self.mode)

    def apply_silu_fusion(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse SiLU / Swish."""
        return self.pipeline.act_pass.apply_silu_fusion(model, mode=self.mode)

    def apply_hardswish_fusion(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse HardSwish."""
        return self.pipeline.act_pass.apply_hardswish_fusion(model, mode=self.mode)

    def apply_relu6_fusion(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse Relu6."""
        return self.pipeline.act_pass.apply_relu6_fusion(model, mode=self.mode)

    def apply_mish_fusion(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse Mish."""
        return self.pipeline.act_pass.apply_mish_fusion(model, mode=self.mode)

    def apply_rmsnorm_fusion(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse RMSNorm expansions."""
        return self.pipeline.norm_pass.apply_rmsnorm_fusion(model, mode=self.mode)

    def _attach_rmsnorm_function_def(self, model: onnx.ModelProto) -> None:
        self.pipeline.norm_pass._attach_rmsnorm_function_def(model)

    def apply_groupnorm_fusion(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse GroupNorm expansions."""
        return self.pipeline.norm_pass.apply_groupnorm_fusion(model, mode=self.mode)

    def _attach_groupnorm_function_def(self, model: onnx.ModelProto) -> None:
        self.pipeline.norm_pass._attach_groupnorm_function_def(model)

    def apply_layernorm_fusion(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse LayerNorm expansions."""
        return self.pipeline.norm_pass.apply_layernorm_fusion(model, mode=self.mode)

    def apply_skip_norm_fusion(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse SkipNorms."""
        return self.pipeline.norm_pass.apply_skip_norm_fusion(model, mode=self.mode)

    def apply_index_tensor_routing(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse index routing."""
        return self.pipeline.axial_pass.apply_index_tensor_routing(
            model, mode=self.mode
        )

    def _attach_tensor_route_function_def(self, model: onnx.ModelProto) -> None:
        self.pipeline.axial_pass._attach_tensor_route_function_def(model)

    def apply_identity_elimination(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Prune identity nodes."""
        return self.pipeline.shape_pass.apply_identity_elimination(model)

    def apply_rope_fusion(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse RoPE."""
        return self.pipeline.attn_pass.apply_rope_fusion(model, mode=self.mode)

    def _attach_rope_function_def(self, model: onnx.ModelProto) -> None:
        self.pipeline.attn_pass._attach_rope_function_def(model)

    def apply_gated_unit_fusion(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse GatedUnit."""
        return self.pipeline.attn_pass.apply_gated_unit_fusion(model, mode=self.mode)

    def _attach_gated_unit_function_def(self, model: onnx.ModelProto) -> None:
        self.pipeline.attn_pass._attach_gated_unit_function_def(model)

    def apply_softmax_matmul_fusion(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse SoftmaxMatMul."""
        return self.pipeline.attn_pass.apply_softmax_matmul_fusion(
            model, mode=self.mode
        )

    def _attach_softmax_matmul_function_def(self, model: onnx.ModelProto) -> None:
        self.pipeline.attn_pass._attach_softmax_matmul_function_def(model)

    def apply_scaled_dot_product_attention_fusion(
        self, model: onnx.ModelProto
    ) -> onnx.ModelProto:
        """Fuse SDPA."""
        return self.pipeline.attn_pass.apply_scaled_dot_product_attention_fusion(
            model, mode=self.mode
        )

    def _attach_sdpa_function_def(self, model: onnx.ModelProto) -> None:
        self.pipeline.attn_pass._attach_sdpa_function_def(model)

    def apply_linear_fusion(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse Linear."""
        return self.pipeline.math_pass.apply_linear_fusion(model, mode=self.mode)

    def _attach_linear_function_def(self, model: onnx.ModelProto) -> None:
        self.pipeline.math_pass._attach_linear_function_def(model)

    def apply_feedforward_fusion(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse FeedForward GELU."""
        return self.pipeline.ffn_pass.apply_feedforward_fusion(model, mode=self.mode)

    def _attach_feedforward_function_def(self, model: onnx.ModelProto) -> None:
        self.pipeline.ffn_pass._attach_feedforward_function_def(model)

    def apply_feedforward_swiglu_fusion(
        self, model: onnx.ModelProto
    ) -> onnx.ModelProto:
        """Fuse FeedForward SwiGLU."""
        return self.pipeline.ffn_pass.apply_feedforward_swiglu_fusion(
            model, mode=self.mode
        )

    def apply_merge_heads_fusion(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse MergeHeads."""
        return self.pipeline.attn_pass.apply_merge_heads_fusion(model, mode=self.mode)

    def _attach_merge_heads_function_def(self, model: onnx.ModelProto) -> None:
        self.pipeline.attn_pass._attach_merge_heads_function_def(model)

    def apply_axial_rearrange_fusion(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse AxialRearrange."""
        return self.pipeline.axial_pass.apply_axial_rearrange_fusion(
            model, mode=self.mode
        )

    def _attach_axial_rearrange_function_def(self, model: onnx.ModelProto) -> None:
        self.pipeline.axial_pass._attach_axial_rearrange_function_def(model)

    def apply_unbind_fusion(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse Unbind."""
        return self.pipeline.axial_pass.apply_unbind_fusion(model, mode=self.mode)

    def _attach_unbind_function_def(self, model: onnx.ModelProto) -> None:
        self.pipeline.axial_pass._attach_unbind_function_def(model)

    def apply_split_qkv_fusion(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse SplitQKV."""
        return self.pipeline.attn_pass.apply_split_qkv_fusion(model, mode=self.mode)

    def _attach_split_qkv_function_def(self, model: onnx.ModelProto) -> None:
        self.pipeline.attn_pass._attach_split_qkv_function_def(model)

    def apply_common_subexpression_elimination(
        self, model: onnx.ModelProto
    ) -> onnx.ModelProto:
        """Eliminate duplicate subexpressions."""
        return self.pipeline.shape_pass.apply_common_subexpression_elimination(model)

    def apply_continuous_math_collapse(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Collapse constant math."""
        return self.pipeline.math_pass.apply_continuous_math_collapse(model)

    def apply_dead_code_elimination(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Prune dead code."""
        return self.pipeline.shape_pass.apply_dead_code_elimination(model)
