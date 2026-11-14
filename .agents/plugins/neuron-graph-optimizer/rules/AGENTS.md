# Graph Optimization & Rule Generation Guardrails

When working with ONNX models, pattern rewrites, and graph simplifications in this workspace:

1. **Parity Before Claiming Minification**:
   - Never claim an ONNX model is simplified or minified without performing differential cross-verification using `CrossVerifier` against ONNX Runtime (ORT).
   - Invariant: Cosine Similarity must be $\ge 0.99999$ and Mean Squared Error (MSE) must be $< 10^{-5}$ (typically $< 10^{-10}$ for exact float parity).

2. **Topological Ordering Invariant**:
   - Every graph transformation must ensure that `model.graph.node` remains strictly topologically sorted.
   - Always run `toposort_graph(model)` before returning or exporting a modified model.
   - Run `onnx.checker.check_model(model)` to confirm graph integrity.

3. **Dual-Mode Preservation**:
   - Always support both `mode="function_proto"` (attaches formal ONNX `FunctionProto` bodies for Netron visualization and standard runtimes) and `mode="plain_custom_op"` (clean minimalist op nodes for lightweight PyTorch execution).

4. **Preserve Graph I/O Interfaces**:
   - Never modify or eliminate tensors declared in `graph.input` or `graph.output`.
   - Output names, dtypes, and shapes must remain identical to the original model.

5. **Declarative Rule Separation**:
   - Whenever a new repeating subgraph pattern is identified, create a corresponding declarative JSON rule in `src/neuron_toolkit/rules/definitions/<rule_name>.json`.
   - Implement the vectorized PyTorch execution handler in `src/neuron_toolkit/runtime/registry.py` under the `custom.pytorch` domain.
   - Register the engine pass method in `JSONRuleEngine` (`src/neuron_toolkit/rules/engine.py`).

6. **Environment & Dependency Safety**:
   - Do not reinstall packages or modify virtual environment packages directly with `pip install`. Use the existing `.venv` via `uv run` or `.venv/bin/python`.
