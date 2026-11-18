"""Semantic Subnet Recognizer & Architecture Block Predictor for ONNX Models.

Analyzes graph topology, operator motifs, and algebraic patterns to automatically
predict and interpret deep learning architectural components (Transformer, DiT,
ConvNeXt, MobileNet, Mamba/SSM, ResMLP, GNN, RNN) with confidence scores.
"""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import onnx

from neuron_toolkit.compiler.architectures.signature_engine import ArchitectureCatalog

log = logging.getLogger(__name__)


@dataclass
class DetectedBlock:
    """An identified architectural block or component within the model graph."""

    block_type: str
    category: str
    confidence: float
    nodes_involved: list[str]
    properties: dict[str, Any] = field(default_factory=dict)
    summary: str = ""


@dataclass
class ArchitectureManifest:
    """Consolidated semantic architecture manifest and interpretability analysis."""

    model_name: str
    total_nodes: int
    primary_family: str
    detected_blocks: list[DetectedBlock] = field(default_factory=list)
    operator_distribution: dict[str, int] = field(default_factory=dict)
    interpretability_summary: str = ""
    optimization_recommendations: list[str] = field(default_factory=list)

    def print_report(self) -> None:
        """Print an interpretable, human-readable summary of the detected architecture."""
        print("=" * 80)
        print(f"      NEURON ARCHITECTURE INTERPRETABILITY MANIFEST: {self.model_name}")
        print("=" * 80)
        print(f"Total Nodes:            {self.total_nodes}")
        print(f"Predicted Family:       {self.primary_family}")
        print(f"Identified Blocks:      {len(self.detected_blocks)}")
        print("-" * 80)
        print(
            f"{'Block Type':<25} | {'Category':<12} | {'Confidence':<10} | {'Nodes Involved'}"
        )
        print("-" * 80)
        for b in self.detected_blocks:
            print(
                f"{b.block_type:<25} | {b.category:<12} | {b.confidence * 100:>8.1f}% | {len(b.nodes_involved)} nodes"
            )
            if b.summary:
                print(f"  └─> {b.summary}")
        print("-" * 80)
        print(f"Summary: {self.interpretability_summary}")
        if self.optimization_recommendations:
            print("\nRecommended Synthesis Passes:")
            for rec in self.optimization_recommendations:
                print(f"  [+] {rec}")
        print("=" * 80)


class ArchitectureBlockPredictor:
    """Automatic pattern matcher and semantic architecture predictor."""

    def __init__(self, catalog: ArchitectureCatalog | None = None) -> None:
        self.catalog = catalog or ArchitectureCatalog()

    def analyze(self, model: onnx.ModelProto | str | Path) -> ArchitectureManifest:
        """Analyze an ONNX model and return a comprehensive ArchitectureManifest."""
        if isinstance(model, (str, Path)):
            model_path = Path(model)
            model_proto = onnx.load(str(model_path), load_external_data=False)
            model_name = model_path.name
        else:
            model_proto = model
            model_name = model_proto.graph.name or "onnx_model"

        graph = model_proto.graph
        nodes = list(graph.node)
        op_counts = collections.Counter(n.op_type for n in nodes)

        detected_blocks: list[DetectedBlock] = []

        # 1. Detect Attention Motifs (Standard MHA, RoPE, MLA, Causal, Gated)
        self._detect_attention_motifs(graph, detected_blocks)

        # 2. Detect Feed-Forward / MLP Motifs (SwiGLU, GeGLU, ResMLP, SGU, Highway)
        self._detect_ffn_motifs(graph, detected_blocks)

        # 3. Detect Normalization Motifs (RMSNorm, LayerNorm, GroupNorm)
        self._detect_norm_motifs(graph, detected_blocks)

        # 4. Detect Vision / Convolutional Motifs (ConvNeXt, InvertedResidual, SE)
        self._detect_vision_motifs(graph, detected_blocks)

        # 5. Detect Recurrent & State-Space Motifs (LSTM, GRU, Mamba SSM)
        self._detect_recurrent_ssm_motifs(graph, detected_blocks)

        # 6. Detect Graph Neural Network Motifs (GCN, GAT, GIN)
        self._detect_gnn_motifs(graph, detected_blocks)

        # Deduce Primary Architecture Family
        family = self._deduce_primary_family(detected_blocks, op_counts)

        # Build Interpretability Summary
        summary = self._generate_summary(family, detected_blocks, op_counts)

        # Build Optimization Recommendations
        recs = self._generate_recommendations(detected_blocks)

        return ArchitectureManifest(
            model_name=model_name,
            total_nodes=len(nodes),
            primary_family=family,
            detected_blocks=detected_blocks,
            operator_distribution=dict(op_counts),
            interpretability_summary=summary,
            optimization_recommendations=recs,
        )

    def _detect_attention_motifs(
        self, graph: onnx.GraphProto, blocks: list[DetectedBlock]
    ) -> None:
        nodes = graph.node
        node_by_output = {out: n for n in nodes for out in n.output}

        # Check for ScaledDotProductAttention, SoftmaxMatMul, or custom Attention
        sdpa_nodes = [
            n for n in nodes if n.op_type in ("ScaledDotProductAttention", "Attention")
        ]
        softmax_nodes = [n for n in nodes if n.op_type == "Softmax"]
        matmul_nodes = [n for n in nodes if n.op_type == "MatMul"]

        if sdpa_nodes:
            blocks.append(
                DetectedBlock(
                    block_type="MultiHeadAttention (Synthesized)",
                    category="transformer",
                    confidence=1.0,
                    nodes_involved=[n.name for n in sdpa_nodes],
                    properties={"instances": len(sdpa_nodes)},
                    summary=f"Detected {len(sdpa_nodes)} fused ScaledDotProductAttention instance(s).",
                )
            )
        elif len(softmax_nodes) >= 1 and len(matmul_nodes) >= 2:
            # Check for Softmax -> MatMul pattern
            sm_matmul_count = 0
            involved = set()
            for mm in matmul_nodes:
                for inp in mm.input:
                    parent = node_by_output.get(inp)
                    if parent and parent.op_type == "Softmax":
                        sm_matmul_count += 1
                        involved.update({mm.name, parent.name})
            if sm_matmul_count > 0:
                blocks.append(
                    DetectedBlock(
                        block_type="Unrolled MultiHeadAttention",
                        category="transformer",
                        confidence=0.95,
                        nodes_involved=list(involved),
                        properties={"sm_matmul_pairs": sm_matmul_count},
                        summary=f"Detected {sm_matmul_count} unrolled Softmax-MatMul attention score projection(s).",
                    )
                )

        # Check for Rotary Embedding (RoPE)
        sin_cos_nodes = [n for n in nodes if n.op_type in ("Sin", "Cos")]
        if len(sin_cos_nodes) >= 2:
            blocks.append(
                DetectedBlock(
                    block_type="RotaryEmbedding (RoPE)",
                    category="transformer",
                    confidence=0.92,
                    nodes_involved=[n.name for n in sin_cos_nodes],
                    summary="Detected Sin/Cos frequency projections characteristic of Rotary Position Embeddings.",
                )
            )

    def _detect_ffn_motifs(
        self, graph: onnx.GraphProto, blocks: list[DetectedBlock]
    ) -> None:
        nodes = graph.node
        node_by_output = {out: n for n in nodes for out in n.output}

        # Check for SwiGLU: Mul(Linear(x), Silu(Linear(x))) or Mul(x, Sigmoid(gate))
        silu_nodes = [n for n in nodes if n.op_type in ("SiLU", "Swish")]
        mul_nodes = [n for n in nodes if n.op_type == "Mul"]
        gelu_nodes = [n for n in nodes if n.op_type in ("Gelu", "FastGelu")]

        swiglu_detected = False
        swiglu_nodes = set()
        for m in mul_nodes:
            in_nodes = [
                node_by_output.get(inp) for inp in m.input if inp in node_by_output
            ]
            has_silu = any(n.op_type in ("SiLU", "Swish") for n in in_nodes if n)
            if has_silu:
                swiglu_detected = True
                swiglu_nodes.add(m.name)

        if swiglu_detected:
            blocks.append(
                DetectedBlock(
                    block_type="SwiGLU FeedForward Block",
                    category="mlp",
                    confidence=0.96,
                    nodes_involved=list(swiglu_nodes),
                    summary="Detected gated SwiGLU activation (SiLU gating combined with linear projection).",
                )
            )

        # Check for GeGLU
        geglu_detected = False
        geglu_nodes = set()
        for m in mul_nodes:
            in_nodes = [
                node_by_output.get(inp) for inp in m.input if inp in node_by_output
            ]
            has_gelu = any(n.op_type in ("Gelu", "FastGelu") for n in in_nodes if n)
            if has_gelu:
                geglu_detected = True
                geglu_nodes.add(m.name)

        if geglu_detected:
            blocks.append(
                DetectedBlock(
                    block_type="GeGLU FeedForward Block",
                    category="mlp",
                    confidence=0.95,
                    nodes_involved=list(geglu_nodes),
                    summary="Detected GeGLU gated linear activation motif.",
                )
            )

        # Check for Highway Network: Mul + Sub(1, sig) + Add
        sub_nodes = [n for n in nodes if n.op_type == "Sub"]
        for sub in sub_nodes:
            for inp in sub.input:
                p = node_by_output.get(inp)
                if p and p.op_type == "Sigmoid":
                    blocks.append(
                        DetectedBlock(
                            block_type="Highway Network Gate",
                            category="mlp",
                            confidence=0.90,
                            nodes_involved=[sub.name, p.name],
                            summary="Detected Highway Carry & Transform gating motif (1 - T(x)).",
                        )
                    )
                    break

    def _detect_norm_motifs(
        self, graph: onnx.GraphProto, blocks: list[DetectedBlock]
    ) -> None:
        nodes = graph.node
        rms_nodes = [
            n
            for n in nodes
            if n.op_type in ("RMSNorm", "SimplifiedLayerNormalization", "RMSNorm_Fused")
        ]
        ln_nodes = [n for n in nodes if n.op_type == "LayerNormalization"]
        gn_nodes = [
            n for n in nodes if n.op_type in ("GroupNormalization", "GroupNorm")
        ]

        if rms_nodes:
            blocks.append(
                DetectedBlock(
                    block_type="RMSNorm Layers",
                    category="normalization",
                    confidence=1.0,
                    nodes_involved=[n.name for n in rms_nodes],
                    properties={"count": len(rms_nodes)},
                    summary=f"Found {len(rms_nodes)} Root Mean Square Normalization layer(s).",
                )
            )
        if ln_nodes:
            blocks.append(
                DetectedBlock(
                    block_type="LayerNormalization",
                    category="normalization",
                    confidence=1.0,
                    nodes_involved=[n.name for n in ln_nodes],
                    properties={"count": len(ln_nodes)},
                    summary=f"Found {len(ln_nodes)} Standard LayerNormalization layer(s).",
                )
            )
        if gn_nodes:
            blocks.append(
                DetectedBlock(
                    block_type="GroupNorm / InstanceNorm",
                    category="normalization",
                    confidence=1.0,
                    nodes_involved=[n.name for n in gn_nodes],
                    properties={"count": len(gn_nodes)},
                    summary=f"Found {len(gn_nodes)} GroupNormalization / InstanceNorm layer(s).",
                )
            )

    def _detect_vision_motifs(
        self, graph: onnx.GraphProto, blocks: list[DetectedBlock]
    ) -> None:
        nodes = graph.node
        conv_nodes = [n for n in nodes if n.op_type == "Conv"]
        gap_nodes = [
            n for n in nodes if n.op_type in ("GlobalAveragePool", "ReduceMean")
        ]

        # Squeeze-and-Excitation
        if len(conv_nodes) >= 2 and len(gap_nodes) >= 1:
            sig_nodes = [n for n in nodes if n.op_type in ("Sigmoid", "HardSigmoid")]
            if sig_nodes:
                blocks.append(
                    DetectedBlock(
                        block_type="Squeeze-and-Excitation (SE)",
                        category="vision",
                        confidence=0.88,
                        nodes_involved=[gap_nodes[0].name, sig_nodes[0].name],
                        summary="Detected channel attention Squeeze-and-Excitation block.",
                    )
                )

        # Inverted Residual / Depthwise Separable
        group_convs = []
        for c in conv_nodes:
            for attr in c.attribute:
                if attr.name == "group" and attr.i > 1:
                    group_convs.append(c)
        if group_convs:
            blocks.append(
                DetectedBlock(
                    block_type="Depthwise / Grouped Convolution Block",
                    category="vision",
                    confidence=0.92,
                    nodes_involved=[c.name for c in group_convs],
                    properties={"grouped_conv_count": len(group_convs)},
                    summary=f"Found {len(group_convs)} grouped/depthwise convolution layer(s) (MobileNet/ConvNeXt style).",
                )
            )

    def _detect_recurrent_ssm_motifs(
        self, graph: onnx.GraphProto, blocks: list[DetectedBlock]
    ) -> None:
        nodes = graph.node
        lstm_nodes = [n for n in nodes if n.op_type == "LSTM"]
        gru_nodes = [n for n in nodes if n.op_type == "GRU"]
        rnn_nodes = [n for n in nodes if n.op_type == "RNN"]

        if lstm_nodes:
            blocks.append(
                DetectedBlock(
                    block_type="Standard ONNX LSTM",
                    category="recurrent",
                    confidence=1.0,
                    nodes_involved=[n.name for n in lstm_nodes],
                    summary=f"Detected {len(lstm_nodes)} Long Short-Term Memory (LSTM) recurrent cell(s).",
                )
            )
        if gru_nodes:
            blocks.append(
                DetectedBlock(
                    block_type="Standard ONNX GRU",
                    category="recurrent",
                    confidence=1.0,
                    nodes_involved=[n.name for n in gru_nodes],
                    summary=f"Detected {len(gru_nodes)} Gated Recurrent Unit (GRU) cell(s).",
                )
            )
        if rnn_nodes:
            blocks.append(
                DetectedBlock(
                    block_type="Vanilla ONNX RNN",
                    category="recurrent",
                    confidence=1.0,
                    nodes_involved=[n.name for n in rnn_nodes],
                    summary=f"Detected {len(rnn_nodes)} Vanilla Recurrent Neural Network layer(s).",
                )
            )

    def _detect_gnn_motifs(
        self, graph: onnx.GraphProto, blocks: list[DetectedBlock]
    ) -> None:
        nodes = graph.node
        node_by_output = {out: n for n in nodes for out in n.output}
        # In GNNs: MatMul(Adjacency, Feature) or ScatterAdd/ReduceSum neighborhood aggregation
        reduce_sums = [n for n in nodes if n.op_type == "ReduceSum"]
        leaky_relus = [n for n in nodes if n.op_type == "LeakyRelu"]

        if leaky_relus and (reduce_sums or any(n.op_type == "MatMul" for n in nodes)):
            blocks.append(
                DetectedBlock(
                    block_type="Graph Attention Network (GAT) Aggregator",
                    category="gnn",
                    confidence=0.89,
                    nodes_involved=[n.name for n in leaky_relus],
                    summary="Detected LeakyReLU attention coefficients with neighbor aggregation.",
                )
            )

    def _deduce_primary_family(
        self, blocks: list[DetectedBlock], op_counts: dict[str, int]
    ) -> str:
        cat_counts = collections.Counter(b.category for b in blocks)
        if (
            cat_counts["transformer"] >= 1
            or op_counts.get("ScaledDotProductAttention", 0) > 0
        ):
            if any("RoPE" in b.block_type for b in blocks):
                return (
                    "Autoregressive / Modern Transformer (LLaMA/Mistral/DeepSeek Style)"
                )
            return "Transformer Architecture (Multi-Head Attention Backbone)"
        if cat_counts["vision"] >= 1 or op_counts.get("Conv", 0) >= 3:
            return "Convolutional Neural Network (CNN / Vision Backbone)"
        if cat_counts["recurrent"] >= 1:
            return "Recurrent Neural Network (LSTM / GRU / RNN)"
        if cat_counts["gnn"] >= 1:
            return "Graph Neural Network (GNN / Message Passing)"
        if (
            cat_counts["mlp"] >= 1
            or op_counts.get("Linear", 0) >= 2
            or op_counts.get("Gemm", 0) >= 2
        ):
            return "Multilayer Perceptron / Modern MLP (Mixer/ResMLP/Highway)"
        return "Generic Deep Learning Graph"

    def _generate_summary(
        self, family: str, blocks: list[DetectedBlock], op_counts: dict[str, int]
    ) -> str:
        block_names = [b.block_type for b in blocks]
        top_ops = ", ".join(f"{k} ({v})" for k, v in op_counts.most_common(5))
        return (
            f"Model classified as '{family}'. Identified key macro components: "
            f"{', '.join(block_names) if block_names else 'Standard Dense Ops'}. "
            f"Top operator distribution: {top_ops}."
        )

    def _generate_recommendations(self, blocks: list[DetectedBlock]) -> list[str]:
        recs = []
        block_types = {b.block_type for b in blocks}
        if any("Unrolled MultiHeadAttention" in bt for bt in block_types):
            recs.append(
                "Apply ScaledDotProductAttention fusion to eliminate unrolled Softmax-MatMul overhead."
            )
        if any("SwiGLU" in bt for bt in block_types):
            recs.append(
                "Synthesize FeedForwardSwiGLU into custom domain to fuse gating multiplication."
            )
        if any("Highway" in bt for bt in block_types):
            recs.append(
                "Preserve Sigmoid multi-consumer graph connections while fusing Transform & Carry units."
            )
        if any("SE" in bt for bt in block_types):
            recs.append(
                "Fuse GlobalAveragePool + Conv/Linear + Sigmoid channel gating into SqueezeExcitation."
            )
        if not recs:
            recs.append(
                "Run Micro-Canonicalization (Static Subgraph Folding, Dead Code Elimination, Norm Fusion)."
            )
        return recs
