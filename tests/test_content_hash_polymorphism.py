"""Targeted tests for Content-Hashed Subgraph Fingerprinting and Polymorphic FunctionProto Monomorphization.

Verifies that when multiple layers have identical structure conceptually but different
inner dynamics (e.g. Layer 0-1 with GELU, Layer 2 with ReLU, Layer 3 with SiLU),
the compiler generates distinct, content-hashed FunctionProtos without collision or math corruption.
"""

from __future__ import annotations

import os
import tempfile

import onnx
import torch
import torch.nn.functional as F
from torch import nn

from neuron_toolkit.compiler import NeuronCompiler
from neuron_toolkit.runtime.cross_verifier import CrossVerifier


class AlternatingFeedForwardModel(nn.Module):
    """Real-world style model where intermediate layers use different activations (e.g. GELU vs ReLU vs SiLU)."""

    def __init__(self, d_model: int = 64, d_ff: int = 128) -> None:
        super().__init__()
        # Layer 0 & 1: GELU
        self.norm0 = nn.RMSNorm(d_model)
        self.fc1_0 = nn.Linear(d_model, d_ff)
        self.fc2_0 = nn.Linear(d_ff, d_model)

        self.norm1 = nn.RMSNorm(d_model)
        self.fc1_1 = nn.Linear(d_model, d_ff)
        self.fc2_1 = nn.Linear(d_ff, d_model)

        # Layer 2: ReLU (e.g. special dynamic / alternating layer)
        self.norm2 = nn.RMSNorm(d_model)
        self.fc1_2 = nn.Linear(d_model, d_ff)
        self.fc2_2 = nn.Linear(d_ff, d_model)

        # Layer 3: SiLU (Swish)
        self.norm3 = nn.RMSNorm(d_model)
        self.fc1_3 = nn.Linear(d_model, d_ff)
        self.fc2_3 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 0: GELU
        x = x + self.fc2_0(F.gelu(self.fc1_0(self.norm0(x))))
        # Block 1: GELU (identical dynamics to Block 0)
        x = x + self.fc2_1(F.gelu(self.fc1_1(self.norm1(x))))
        # Block 2: ReLU (different dynamics)
        x = x + self.fc2_2(F.relu(self.fc1_2(self.norm2(x))))
        # Block 3: SiLU (different dynamics)
        x = x + self.fc2_3(F.silu(self.fc1_3(self.norm3(x))))
        return x


class AlternatingGatedAttentionModel(nn.Module):
    """Model with multiple gating mechanisms (Sigmoid gating vs SiLU gating)."""

    def __init__(self, d_model: int = 64) -> None:
        super().__init__()
        self.proj_val1 = nn.Linear(d_model, d_model)
        self.proj_gate1 = nn.Linear(d_model, d_model)

        self.proj_val2 = nn.Linear(d_model, d_model)
        self.proj_gate2 = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gate 1: Sigmoid gating
        v1 = self.proj_val1(x)
        g1 = torch.sigmoid(self.proj_gate1(x))
        x = x + (v1 * g1)

        # Gate 2: SiLU gating
        v2 = self.proj_val2(x)
        g2 = F.silu(self.proj_gate2(x))
        x = x + (v2 * g2)
        return x


class AlternatingDeepMLPModel(nn.Module):
    """Model with DeepMLP blocks using different activations (LayerNorm -> Linear -> Gelu vs Relu)."""

    def __init__(self, d_model: int = 64) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model)

        self.ln2 = nn.LayerNorm(d_model)
        self.fc2 = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1: GELU
        h1 = F.gelu(self.fc1(self.ln1(x)))
        # Block 2: ReLU
        h2 = F.relu(self.fc2(self.ln2(h1)))
        return h2


def _export_to_onnx(
    torch_model: nn.Module, dummy_input: torch.Tensor
) -> onnx.ModelProto:
    torch_model.eval()
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        torch.onnx.export(
            torch_model,
            dummy_input,
            tmp_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=18,
            do_constant_folding=True,
        )
        return onnx.load(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_content_hashed_feedforward_polymorphism():
    """Verify that GELU, ReLU, and SiLU FFN blocks get distinct fingerprints and correct FunctionProtos."""
    torch.manual_seed(42)
    py_model = AlternatingFeedForwardModel()
    dummy_input = torch.randn(2, 8, 64)
    raw_onnx = _export_to_onnx(py_model, dummy_input)

    opt_onnx = NeuronCompiler.compile(raw_onnx, mode="function_proto")

    # 1. Inspect synthesized nodes
    ff_nodes = [n for n in opt_onnx.graph.node if n.op_type.startswith("FeedForward_")]
    assert len(ff_nodes) == 4, f"Expected 4 fused FFN nodes, got {len(ff_nodes)}"

    # Layer 0 & 1 should share the exact same op_type and fingerprint
    assert ff_nodes[0].op_type == ff_nodes[1].op_type
    assert "Gelu" in ff_nodes[0].op_type

    # Layer 2 should have Relu and a different op_type
    assert ff_nodes[2].op_type != ff_nodes[0].op_type
    assert "Relu" in ff_nodes[2].op_type

    # Layer 3 should have SiLU and a different op_type
    assert ff_nodes[3].op_type != ff_nodes[0].op_type
    assert ff_nodes[3].op_type != ff_nodes[2].op_type
    assert "SiLU" in ff_nodes[3].op_type

    # 2. Inspect FunctionProtos in opt_onnx.functions
    fn_map = {f.name: f for f in opt_onnx.functions}
    assert ff_nodes[0].op_type in fn_map
    assert ff_nodes[2].op_type in fn_map
    assert ff_nodes[3].op_type in fn_map

    # Exactly 3 distinct FFN FunctionProtos should be registered
    ff_funcs = [name for name in fn_map if name.startswith("FeedForward_")]
    assert len(ff_funcs) == 3, (
        f"Expected 3 distinct FFN FunctionProtos, got {len(ff_funcs)}: {ff_funcs}"
    )

    # Verify inner nodes inside the FunctionProtos
    gelu_fn_nodes = [n.op_type for n in fn_map[ff_nodes[0].op_type].node]
    assert "Gelu" in gelu_fn_nodes

    relu_fn_nodes = [n.op_type for n in fn_map[ff_nodes[2].op_type].node]
    assert "Relu" in relu_fn_nodes

    silu_fn_nodes = [n.op_type for n in fn_map[ff_nodes[3].op_type].node]
    assert "SiLU" in silu_fn_nodes

    # 3. Cross-Verification: Mathematical Parity
    verifier = CrossVerifier(raw_onnx, opt_onnx)
    report = verifier.verify({"input": dummy_input.numpy()})
    assert report.all_passed, f"Cross-verification failed: {report.summary()}"
    for metric in report.metrics.values():
        assert metric.cosine_similarity >= 0.999999, (
            f"Cosine similarity too low: {metric.cosine_similarity}"
        )


def test_content_hashed_gated_unit_polymorphism():
    """Verify that Sigmoid and SiLU gating blocks receive distinct fingerprints and separate FunctionProtos."""
    torch.manual_seed(42)
    py_model = AlternatingGatedAttentionModel()
    dummy_input = torch.randn(2, 16, 64)
    raw_onnx = _export_to_onnx(py_model, dummy_input)

    opt_onnx = NeuronCompiler.compile(raw_onnx, mode="function_proto")

    gated_nodes = [n for n in opt_onnx.graph.node if n.op_type.startswith("GatedUnit_")]
    assert len(gated_nodes) == 2, f"Expected 2 GatedUnit nodes, got {len(gated_nodes)}"

    # Gate 1 should be Sigmoid, Gate 2 should be SiLU
    assert "Sigmoid" in gated_nodes[0].op_type
    assert "SiLU" in gated_nodes[1].op_type
    assert gated_nodes[0].op_type != gated_nodes[1].op_type

    # Verify FunctionProtos
    fn_map = {f.name: f for f in opt_onnx.functions}
    assert gated_nodes[0].op_type in fn_map
    assert gated_nodes[1].op_type in fn_map

    # Cross-Verification
    verifier = CrossVerifier(raw_onnx, opt_onnx)
    report = verifier.verify({"input": dummy_input.numpy()})
    assert report.all_passed, f"Cross-verification failed: {report.summary()}"
    for metric in report.metrics.values():
        assert metric.cosine_similarity >= 0.999999, (
            f"Cosine similarity: {metric.cosine_similarity}"
        )


def test_content_hashed_deep_mlp_polymorphism():
    """Verify that DeepMLP blocks with GELU vs ReLU are correctly monomorphized."""
    torch.manual_seed(42)
    py_model = AlternatingDeepMLPModel()
    dummy_input = torch.randn(2, 8, 64)
    raw_onnx = _export_to_onnx(py_model, dummy_input)

    opt_onnx = NeuronCompiler.compile(
        raw_onnx, mode="function_proto", options={"enable_macro_blocks": True}
    )

    mlp_nodes = [n for n in opt_onnx.graph.node if n.op_type.startswith("DeepMLP_")]
    assert len(mlp_nodes) == 2, f"Expected 2 DeepMLP nodes, got {len(mlp_nodes)}"

    assert "Gelu" in mlp_nodes[0].op_type
    assert "Relu" in mlp_nodes[1].op_type
    assert mlp_nodes[0].op_type != mlp_nodes[1].op_type

    # Cross-Verification
    verifier = CrossVerifier(raw_onnx, opt_onnx)
    report = verifier.verify({"input": dummy_input.numpy()})
    assert report.all_passed, f"Cross-verification failed: {report.summary()}"
    for metric in report.metrics.values():
        assert metric.cosine_similarity >= 0.999999, (
            f"Cosine similarity: {metric.cosine_similarity}"
        )
