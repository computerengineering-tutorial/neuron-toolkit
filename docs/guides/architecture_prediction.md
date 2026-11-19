# Architecture Signatures & Subgraph Fingerprinting

The `neuron_toolkit.compiler.architectures` package provides declarative block signatures, motif discovery, and content-hashed subgraph fingerprinting for ONNX models.

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

The test suite in `tests/test_dl_zoo_evals.py` and evaluation engine in `.helper/evals/run_comprehensive_eval.py` validate architecture block representations across hundreds of synthetic and real models:

- `test_recurrent_zoo_opset18`: RNN, LSTM, GRU, QRNN, SRU
- `test_cnn_vision_zoo_opset18`: ResNet, ConvNeXt, MobileNet, DenseNet
- `test_mlp_vanilla_zoo_opset18`: DeepMLP, ResMLP, MLPMixer, gMLP, Highway
- `test_gnn_zoo_opset18`: GCN, GAT, GraphSAGE, GIN
- `test_transformer_advanced_zoo_opset18`: PreLN, PostLN, ViT, MLA, MoE, Performer
- `test_ssm_mamba_zoo_opset18`: Mamba, SelectiveScan, S4
- `test_diffusion_dit_zoo_opset18`: Diffusion Transformer blocks, AdaLayerNorm modulation
