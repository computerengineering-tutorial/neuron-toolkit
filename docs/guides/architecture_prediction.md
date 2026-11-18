# Architecture Recognition & Subgraph Fingerprinting

The `neuron_toolkit.compiler.analyzer` and `neuron_toolkit.compiler.architectures` packages provide automated semantic subnet recognition, architecture prediction, and content-hashed subgraph fingerprinting for ONNX models.

---

## Semantic Architecture Block Predictor

When optimizing large computational graphs, understanding high-level architectural motifs (e.g. knowing whether a model is a LLaMA Transformer, a Mamba SSM, a ConvNeXt vision model, or a Diffusion DiT) allows compilers to apply targeted macro fusions.

The `ArchitectureBlockPredictor` analyzes graph topology, operator motifs, and algebraic connectivity to predict architectural blocks with confidence metrics:

```python
from neuron_toolkit.compiler.analyzer.block_predictor import ArchitectureBlockPredictor

# Initialize predictor with the default architecture catalog
predictor = ArchitectureBlockPredictor()

# Analyze an ONNX model file or ModelProto
manifest = predictor.analyze("model.onnx")

# Print formatted human-readable architecture report
manifest.print_report()
```

### Supported Architectural Blocks

The predictor detects blocks across modern deep learning paradigms:

| Family | Detected Block Types | Characteristic Motifs |
| --- | --- | --- |
| **Transformer** | `MultiHeadAttention`, `FeedForward_GELU`, `FeedForward_SwiGLU`, `RoPE`, `TransformerBlock` | SDPA patterns, RMSNorm / LayerNorm, QKV projections |
| **Diffusion (DiT)** | `DiT_Block`, `AdaLayerNorm`, `TimestepEmbedding` | Scale/shift modulation, Cross-Attention, Timestep MLPs |
| **State Space (SSM)** | `Mamba_SSM_Block`, `SelectiveScan`, `Conv1D_Gate` | 1D Depthwise Conv + Discretized State-Space equations |
| **Vision (CNN/ViT)** | `ConvNeXt_Block`, `SqueezeExcitation`, `InvertedResidual`, `PatchEmbedding` | 7x7 Depthwise Conv, SE squeeze-excite ratio, Linear projection |
| **MLP Architectures** | `SpatialGatingUnit` (gMLP), `ResMLP_Block` | Channel splitting, spatial normalization, affine gates |
| **Graph Neural Nets** | `MessagePassing_GNN`, `GraphAttention_GAT` | Scatter/gather reductions, node/edge feature aggregations |
| **Recurrent** | `LSTM_Cell`, `GRU_Cell` | Gate bifurcations (sigmoid/tanh), hidden state updates |

---

## Inspecting the Architecture Manifest

The resulting `ArchitectureManifest` object provides structured access to all detected blocks and metrics:

```python
manifest = predictor.analyze("transformer.onnx")

print(f"Model Name:                  {manifest.model_name}")
print(f"Total Nodes:                 {manifest.total_nodes}")
print(f"Primary Architecture Family: {manifest.primary_family}")
print(f"Identified Blocks:           {len(manifest.detected_blocks)}")
print(f"Interpretability Summary:    {manifest.interpretability_summary}")

# Inspect operator distribution:
for op, count in sorted(manifest.operator_distribution.items(), key=lambda x: -x[1])[:5]:
    print(f"  {op}: {count}")

# Inspect each detected block:
for block in manifest.detected_blocks:
    print(f"- [{block.block_type}] ({block.category}): confidence={block.confidence * 100:.1f}%")
    print(f"  Nodes involved: {len(block.nodes_involved)} nodes")
    if block.summary:
        print(f"  Summary:        {block.summary}")
    if block.properties:
        print(f"  Properties:     {block.properties}")

# Inspect compiler pass recommendations:
if manifest.optimization_recommendations:
    print("Recommended Synthesis Passes:")
    for rec in manifest.optimization_recommendations:
        print(f"  [+] {rec}")
```

---

## Polymorphic Subgraph Fingerprinting

In real-world models, identical architectural blocks often appear with minor parameter variations (e.g., varying hidden dimensions, activation functions, or epsilon values).

The `neuron_toolkit.compiler.architectures.fingerprint` module calculates deterministic 8-character SHA-256 topology fingerprints to enable **polymorphic function monomorphization**:

```python
from neuron_toolkit.compiler.architectures.fingerprint import (
    compute_subgraph_fingerprint,
    build_monomorphized_op_type,
    ensure_specialized_function_proto
)

# Compute deterministic fingerprint from subgraph nodes
fingerprint = compute_subgraph_fingerprint(
    nodes=subgraph_nodes,
    external_inputs=["x", "weight"],
    external_outputs=["y"]
)

# Monomorphize operator name with variant hint and fingerprint
op_type = build_monomorphized_op_type(
    family="FeedForward",
    variant_hint="swiglu",
    fingerprint=fingerprint
)
# Result: 'FeedForward_swiglu_a1b2c3d4'

# Ensure specialized FunctionProto is registered in model.functions
ensure_specialized_function_proto(
    model=model,
    domain="custom.pytorch",
    fname=op_type,
    formal_inputs=["x", "w_gate", "w_up", "w_down"],
    formal_outputs=["y"],
    body_nodes=subgraph_nodes,
    doc_string="Specialized SwiGLU FeedForward block"
)
```

---

## Declarative Signature Catalog

The `ArchitectureCatalog` loads JSON-defined block signatures from `block_signatures.json`:

```python
from neuron_toolkit.compiler.architectures.signature_engine import ArchitectureCatalog

catalog = ArchitectureCatalog()
catalog.load()

# List available signatures
signatures = catalog.list_signatures()
print(f"Loaded signatures: {signatures}")

# Retrieve specific block signature
mha_sig = catalog.get_signature("multi_head_attention")
if mha_sig:
    print(f"Name: {mha_sig.name}, Family: {mha_sig.family}")
    print(f"Required Ops: {mha_sig.required_ops}")
```

---

## Discovering Architectural Subgraphs with `SignatureEngine`

To programmatically find all occurrences of registered architectural block motifs across an ONNX graph:

```python
from neuron_toolkit.compiler.architectures.signature_engine import ArchitectureCatalog, SignatureEngine
import onnx

model = onnx.load("model.onnx")

catalog = ArchitectureCatalog()
catalog.load()
engine = SignatureEngine(catalog=catalog)

# Discover occurrences across the model graph
matches = engine.discover_subgraphs(model)

for sig_name, occurrences in matches.items():
    print(f"Signature '{sig_name}': found {len(occurrences)} instances")
    for inst in occurrences:
        print(f"  - Nodes: {inst.get('nodes', [])}")
        print(f"    Inputs: {inst.get('inputs', [])}")
        print(f"    Outputs: {inst.get('outputs', [])}")
```

---

## Automated Multi-Zoo Evaluation

The test suite in `tests/test_dl_zoo_evals.py` and evaluation engine in `.helper/evals/run_comprehensive_eval.py` validate the architecture predictor across hundreds of synthetic and real models:

- `test_recurrent_zoo_opset18`: RNN, LSTM, GRU, QRNN, SRU
- `test_cnn_vision_zoo_opset18`: ResNet, ConvNeXt, MobileNet, DenseNet
- `test_mlp_vanilla_zoo_opset18`: DeepMLP, ResMLP, MLPMixer, gMLP, Highway
- `test_gnn_zoo_opset18`: GCN, GAT, GraphSAGE, GIN
- `test_transformer_advanced_zoo_opset18`: PreLN, PostLN, ViT, MLA, MoE, Performer
- `test_ssm_mamba_zoo_opset18`: Mamba, SelectiveScan, S4
- `test_diffusion_dit_zoo_opset18`: Diffusion Transformer blocks, AdaLayerNorm modulation
