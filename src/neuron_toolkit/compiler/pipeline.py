"""3-Stage Optimization Pipeline Driver for the ONNX Compiler.

Orchestrates:
- Stage 1: Micro-Canonicalization (Shapes, Activations, Norms, Math)
- Stage 2: Macro-Block Synthesis (Attention, FeedForward, RoPE, Axial)
- Stage 3: Target Export & Code Generation (ONNX Opset 23+ / Netron FunctionProto / PyTorch)
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any

import onnx

from neuron_toolkit.compiler.base import CompilerContext
from neuron_toolkit.compiler.exporters.onnx_exporter import OnnxExporter
from neuron_toolkit.compiler.passes.canonicalization.activation_canonicalizer import (
    ActivationCanonicalizer,
)
from neuron_toolkit.compiler.passes.canonicalization.math_canonicalizer import (
    MathCanonicalizer,
)
from neuron_toolkit.compiler.passes.canonicalization.norm_canonicalizer import (
    NormCanonicalizer,
)
from neuron_toolkit.compiler.passes.canonicalization.shape_canonicalizer import (
    ShapeCanonicalizer,
)
from neuron_toolkit.compiler.passes.canonicalization.static_folder import (
    StaticFolder,
)
from neuron_toolkit.compiler.passes.synthesis.attention_synthesizer import (
    AttentionSynthesizer,
)
from neuron_toolkit.compiler.passes.synthesis.axial_synthesizer import (
    AxialSynthesizer,
)
from neuron_toolkit.compiler.passes.synthesis.feedforward_synthesizer import (
    FeedForwardSynthesizer,
)
from neuron_toolkit.compiler.passes.synthesis.macro_block_synthesizer import (
    MacroBlockSynthesizer,
)

if TYPE_CHECKING:
    from neuron_toolkit.graph import NeuronGraph

log = logging.getLogger("neuron_toolkit.compiler.pipeline")


class OptimizationPipeline:
    """Enterprise-grade 3-stage optimization pipeline driver."""

    def __init__(self, mode: str = "function_proto") -> None:
        self.mode = mode
        self.static_pass = StaticFolder()
        self.shape_pass = ShapeCanonicalizer()
        self.act_pass = ActivationCanonicalizer()
        self.norm_pass = NormCanonicalizer()
        self.math_pass = MathCanonicalizer()
        self.attn_pass = AttentionSynthesizer()
        self.ffn_pass = FeedForwardSynthesizer()
        self.axial_pass = AxialSynthesizer()
        self.macro_pass = MacroBlockSynthesizer()
        self.exporter = OnnxExporter(mode=mode)

    def run(
        self,
        target: onnx.ModelProto | NeuronGraph,
        context: CompilerContext | None = None,
    ) -> onnx.ModelProto:
        """Run the full 3-stage optimization pipeline on target model."""
        if hasattr(target, "_backend") and hasattr(target._backend, "model"):
            model = copy.deepcopy(target._backend.model)
        elif isinstance(target, onnx.ModelProto):
            model = copy.deepcopy(target)
        else:
            msg = f"Unsupported target type for OptimizationPipeline: {type(target)}"
            raise TypeError(msg)

        if context is None:
            context = CompilerContext(mode=self.mode)

        log.info(
            "Starting 3-stage optimization pipeline (mode=%s, initial_nodes=%d)",
            context.mode,
            len(model.graph.node),
        )

        max_iters = int(context.options.get("max_fixed_point_iters", getattr(context, "max_fixed_point_iters", 5)))
        if not context.options.get("enable_fixed_point", True):
            max_iters = 1

        for iteration in range(max_iters):
            node_count_before = len(model.graph.node)
            op_sig_before = tuple(n.op_type for n in model.graph.node)
            out_sig_before = tuple(n.output[0] for n in model.graph.node if n.output)

            # -------------------------------------------------------------
            # Stage 1: Micro-Canonicalization Pipeline
            # -------------------------------------------------------------
            if context.should_run("static_folder") or context.should_run(
                "constant_folding"
            ):
                model = self.static_pass.apply_static_subgraph_folding(model)

            if context.should_run("cast"):
                model = self.shape_pass.apply_cast_initializer_folding(model)
                model = self.shape_pass.apply_redundant_cast_elimination(model)

            if context.should_run("softmax"):
                model = self.act_pass.apply_softmax_fusion(model)

            if context.should_run("erf_gelu"):
                model = self.act_pass.apply_erf_gelu_fusion(model, mode=context.mode)

            if context.should_run("tanh_gelu") or context.should_run("fast_gelu"):
                model = self.act_pass.apply_tanh_gelu_fusion(model, mode=context.mode)

            if context.should_run("silu") or context.should_run("swish"):
                model = self.act_pass.apply_silu_fusion(model, mode=context.mode)

            if context.should_run("hard_swish"):
                model = self.act_pass.apply_hardswish_fusion(model, mode=context.mode)

            if context.should_run("relu6"):
                model = self.act_pass.apply_relu6_fusion(model, mode=context.mode)

            if context.should_run("mish"):
                model = self.act_pass.apply_mish_fusion(model, mode=context.mode)

            if context.should_run("groupnorm"):
                model = self.norm_pass.apply_groupnorm_fusion(model, mode=context.mode)

            if context.should_run("layernorm"):
                model = self.norm_pass.apply_layernorm_fusion(model, mode=context.mode)

            if context.should_run("rmsnorm"):
                model = self.norm_pass.apply_rmsnorm_fusion(model, mode=context.mode)

            if context.should_run("index") or context.should_run("routing"):
                model = self.axial_pass.apply_index_tensor_routing(
                    model, mode=context.mode
                )

            if context.should_run("identity"):
                model = self.shape_pass.apply_identity_elimination(model)

            if context.should_run("reshape") or context.should_run("shape"):
                model = self.shape_pass.apply_reshape_collapsing(model)
                model = self.shape_pass.apply_squeeze_unsqueeze_cancellation(model)

            if context.should_run("pad_conv") or context.should_run("conv"):
                model = self.shape_pass.apply_pad_conv_fusion(model)

            if (
                context.should_run("transpose")
                or context.should_run("identity")
                or context.should_run("shape")
            ):
                model = self.shape_pass.apply_transpose_cancellation(model)

            # -------------------------------------------------------------
            # Stage 2: Macro-Block Synthesis Pipeline
            # -------------------------------------------------------------
            if context.should_run("rope"):
                model = self.attn_pass.apply_rope_fusion(model, mode=context.mode)

            if (
                context.should_run("static_folder")
                or context.should_run("constant_folding")
                or context.should_run("scatter")
            ):
                model = self.static_pass.apply_identity_scatter_nd_folding(model)

            if context.should_run("gemma2_softcap") or context.should_run("softcap"):
                model = self.attn_pass.apply_gemma2_softcapping_fusion(
                    model, mode=context.mode
                )

            if context.should_run("deepseek_mla") or context.should_run("mla"):
                model = self.attn_pass.apply_deepseek_mla_fusion(
                    model, mode=context.mode
                )

            if context.should_run("gated_unit"):
                model = self.attn_pass.apply_gated_unit_fusion(model, mode=context.mode)

            if context.should_run("softmax_matmul"):
                model = self.attn_pass.apply_softmax_matmul_fusion(
                    model, mode=context.mode
                )

            if context.should_run("sdpa") or context.should_run("attention"):
                model = self.attn_pass.apply_scaled_dot_product_attention_fusion(
                    model, mode=context.mode
                )

            if context.should_run("linear"):
                model = self.math_pass.apply_linear_fusion(model, mode=context.mode)

            if context.should_run("feedforward"):
                model = self.ffn_pass.apply_feedforward_fusion(model, mode=context.mode)

            if context.should_run("feedforward_swiglu") or context.should_run("swiglu"):
                model = self.ffn_pass.apply_feedforward_swiglu_fusion(
                    model, mode=context.mode
                )

            if context.should_run("merge_heads"):
                model = self.attn_pass.apply_merge_heads_fusion(
                    model, mode=context.mode
                )

            if context.should_run("axial") or context.should_run("rearrange"):
                model = self.axial_pass.apply_axial_rearrange_fusion(
                    model, mode=context.mode
                )

            if context.should_run("unbind"):
                model = self.axial_pass.apply_unbind_fusion(model, mode=context.mode)

            if context.should_run("split_qkv"):
                model = self.attn_pass.apply_split_qkv_fusion(model, mode=context.mode)

            if context.should_run("gated_attention") or context.should_run("attention"):
                model = self.attn_pass.apply_gated_attention_fusion(
                    model, mode=context.mode
                )

            if context.options.get("enable_macro_blocks", False) or (
                context.rule_ids and "macro_blocks" in context.rule_ids
            ):
                model = self.macro_pass.run(model, mode=context.mode)

            if context.should_run("skip_norm"):
                model = self.norm_pass.apply_skip_norm_fusion(model, mode=context.mode)

            # -------------------------------------------------------------
            # Graph Cleansing at each iteration
            # -------------------------------------------------------------
            if context.should_run("cse") or context.should_run("common_subexpression"):
                model = self.shape_pass.apply_common_subexpression_elimination(model)

            if context.should_run("continuous_math"):
                model = self.math_pass.apply_continuous_math_collapse(model)

            if context.should_run("dce") or context.should_run("dead_code"):
                model = self.shape_pass.apply_dead_code_elimination(model)

            node_count_after = len(model.graph.node)
            op_sig_after = tuple(n.op_type for n in model.graph.node)
            out_sig_after = tuple(n.output[0] for n in model.graph.node if n.output)

            if (
                node_count_before == node_count_after
                and op_sig_before == op_sig_after
                and out_sig_before == out_sig_after
            ):
                log.info(
                    "Fixed-point convergence achieved at iteration %d (%d nodes)",
                    iteration + 1,
                    node_count_after,
                )
                break

        # -------------------------------------------------------------
        # Stage 3: Target Exporter & Serialization
        # -------------------------------------------------------------
        model = self.exporter.export(model, context)

        log.info(
            "Optimization complete. Final nodes: %d",
            len(model.graph.node),
        )
        return model


class NeuronCompiler:
    """High-level entry point for ONNX graph compilation and optimization."""

    @classmethod
    def compile(
        cls,
        model: onnx.ModelProto | NeuronGraph,
        mode: str = "function_proto",
        rule_ids: list[str] | None = None,
        opset_version: int = 23,
        options: dict[str, Any] | None = None,
    ) -> onnx.ModelProto:
        """Compile and optimize model with specified mode and rules."""
        pipeline = OptimizationPipeline(mode=mode)
        context = CompilerContext(
            mode=mode,
            opset_version=opset_version,
            rule_ids=rule_ids,
            options=options or {},
        )
        return pipeline.run(model, context)
