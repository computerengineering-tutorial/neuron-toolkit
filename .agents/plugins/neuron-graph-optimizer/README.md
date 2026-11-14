# Neuron Graph Optimizer & Rule Generator Plugin

A production-grade workspace plugin for Antigravity agents that enables deep ONNX model inspection, dramatic node minification (eliminating 50%–77% of redundant nodes in complex traced models), automated declarative JSON rule generation, and differential parity verification against ONNX Runtime.

## Contents
- **Skill**: `graph-optimizer` (`skills/graph-optimizer/SKILL.md`)
  - Automated analysis of model structure, op breakdown, and constant duplication.
  - Multi-pass Common Subexpression Elimination (CSE) and Identity operation pruning.
  - Subgraph fusions: RMSNorm, GELU, RoPE, GroupNorm, Continuous Math collapsing.
  - Automated rule generator scaffolding for new patterns.
- **Rules**: `rules/AGENTS.md`
  - Invariants for safe graph rewriting, topological sorting, and numerical parity constraints.
- **Scripts**:
  - `analyze_model.py`: Profile any ONNX model and uncover minification opportunities.
  - `generate_rule.py`: Scaffold declarative JSON rules and PyTorch handlers.
  - `verify_parity.py`: Differential verification of optimized models against ONNX Runtime.
- **Examples**:
  - `examples/optimize_and_benchmark.py`: End-to-end minification and benchmarking pipeline.
