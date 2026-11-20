# Comprehensive Bug Audit & Vulnerability Analysis: Synthesis Passes

**Files Audited:**
1. `src/neuron_toolkit/compiler/passes/synthesis/macro_block_synthesizer.py` (2,826 lines)
2. `src/neuron_toolkit/compiler/passes/synthesis/attention_synthesizer.py` (1,605 lines)
3. `src/neuron_toolkit/compiler/passes/synthesis/feedforward_synthesizer.py` (432 lines)
4. `src/neuron_toolkit/compiler/passes/synthesis/axial_synthesizer.py` (531 lines)

Total Scope: **5,394 lines of compiler synthesis codebase**.

---

## Executive Summary of Findings

Across the 4 synthesis pass files, this deep forensic audit identified **39 distinct structural bugs, architectural vulnerabilities, and failure modes**, classified into 7 severity tiers:

| Severity | Count | Primary Impact | Batch Status |
|---|---|---|---|
| **CRITICAL (P0): Whole-Graph Wipeouts** | 4 | Subgraph capture grabs `list(graph.node)` and deletes the entire model into 1 custom node | **BATCH 1: 100% FIXED & VERIFIED** |
| **FATAL (P0): FunctionProto Mismatches** | 10 | Function definition inputs/outputs or bodies mismatch the instantiated node, causing ONNX checker or runtime aborts | **BATCH 1: 100% FIXED & VERIFIED** |
| **SILENT NUMERICAL (P1): Math & Bias Bugs** | 6 | Biases dropped into thin air; RoPE with no rotation; hardcoded scale (0.125) and softcap (50.0) | **BATCH 2: 100% FIXED & VERIFIED** |
| **HARDCODED OVERFITTING (P1): String Matching** | 6 | Name-based checks (`"conv_pitched"`, `"pos_embed"`, `inst_logp`, `"past"`, global `ScatterND`, rank-3/4 perm) | **BATCH 3: 100% FIXED & VERIFIED** |
| **PASS ORDERING & AMBIGUITY (P1)** | 3 | Pass ordering collisions cannibalizing sub-blocks (e.g. DeepMLP eating FFN; GatedUnit eating SiLU) | **BATCH 4: 100% FIXED & VERIFIED** |
| **BRITTLE BLINDNESS (P2): Architectural Gaps** | 7 | Strict RMSNorm-only requirement; adjacent peephole assumptions; rigid permutations | **BATCH 5: 100% FIXED & VERIFIED** |
| **TOPOLOGICAL & SSA CORRUPTION (P2)** | 5 | Shared producers dropped; unplaced replacements appended to end of graph; in-place node mutations | **BATCH 6: 100% FIXED & VERIFIED** |

---

## Batch 1 Remediation Log (P0 Whole-Graph Wipeouts & FunctionProto Mismatches)

* **Bug 1.1 (`apply_moe_fusion`)**: Fixed whole-graph deletion (`del graph.node[:]`). Now uses bounded router & expert cluster extraction with `_replace_nodes_topologically`. Verified by `test_bug_1_1_moe_fusion_preserves_external_graph`.
* **Bug 1.2 (`apply_mamba_ssm_fusion`)**: Removed loose `len > 20 and 10 mul/add` trigger and whole-graph deletion. Now precisely identifies selective scan loops (`Softplus` + multi-`Exp` scan accumulation + `Concat`) and fuses only the SSM block. Verified by `test_bug_1_2_mamba_ssm_does_not_wipe_transformer_models` and `test_vuln_52_mamba_selective_ssm_unfolded_scan_bloat`.
* **Bug 1.3 (`apply_shifted_window_attention_fusion`)**: Replaced whole-graph wipeout with bidirectional pre/post cyclic roll tracing (`Slice` -> `Concat`) around attention, stopping at block boundaries. Verified by `test_bug_1_3_shifted_window_attention_preserves_surrounding_graph` and `test_vuln_53_swin_shifted_window_roll_scattered_nodes`.
* **Bug 1.4 (`apply_deepseek_mla_macro_fusion`)**: Removed whole-graph wipeout and removed improper `len(sub_nodes) >= len(graph.node)` skip. Uses forward BFS to locate downstream attention and backward BFS stopping at `root_input` boundary. Verified by `test_bug_1_4_deepseek_mla_preserves_multi_layer_models` and `test_vuln_48_deepseek_mla_latent_attention_unfused`.
* **Bug 7.2 (`_replace_nodes_topologically`)**: Fixed `is_dropped` logic so nodes matched by integer `id(node)` in `drop_ids` are always dropped even if they have a `name`. Added post-replacement topological sort (`_toposort_nodes`). Verified by `test_bug_7_2_replace_nodes_topologically_handles_named_ids_and_preserves_toposort`.
* **Bug 2.1 (`MultiHeadAttention` arity)**: Dynamic FunctionProto formal inputs matching instantiated node arity (3 to 6 inputs), preserving mask and out bias. Verified by `test_bug_2_1_multi_head_attention_function_proto_arity_4_and_5`.
* **Bug 2.2 (`ScaledDotProductAttention` mask formal input)**: Formal input `Mask` and `Add(ScaledScores, Mask)` added to FunctionProto when mask is present (4 formal inputs). Verified by `test_bug_2_2_sdpa_with_mask_function_proto_arity`.
* **Bug 2.3 (`MergeHeads` FunctionProto Reshape to 3D)**: Function body now includes both `Transpose` and `Reshape(T, [0, 0, -1])`, guaranteeing 3D output `[B, S, H*D]`. Verified by `test_bug_2_3_merge_heads_function_proto_includes_reshape_to_3d`.
* **Bug 2.4 (`SpatialGatingUnit` norm gamma & bias preservation)**: SGU now appends `gamma` and `beta` from norm, and bias from linear projection. Formal inputs dynamically match node arity (3, 4, or 5). Verified by `test_bug_2_4_spatial_gating_unit_preserves_gamma_and_biases`.
* **Bug 2.5 (`AxialRearrange` dummy Identity)**: Replaced dummy `Identity` with real `Transpose` and `Reshape` ops reflecting the exact permutation and target shape. Verified by `test_bug_2_5_axial_rearrange_function_proto_not_dummy_identity`.
* **Bug 2.6 (`TensorRoute` FunctionProto matching arity)**: Node arity matched to 1 input (`x`), and FunctionProto generates `Gather` + `Slice`. Verified by `test_bug_2_6_tensor_route_function_proto_preserves_slice_and_matches_arity`.
* **Bug 2.7 (`Unbind` FunctionProto axis)**: Dynamic axis preservation (`axis=axis`) rather than hardcoded 0. Verified by `test_bug_2_7_unbind_function_proto_preserves_arbitrary_axis`.
* **Bug 2.8 (`DynamicQuantLinear` collision)**: Upgrades FunctionProto when biased and unbiased instances coexist. Verified by `test_bug_2_8_dynamic_quant_linear_biased_unbiased_collision`.
* **Bugs 2.9 & 2.10 (`GatedUnit` rank-agnostic body & opset import)**: Replaced hardcoded 3D transpose with rank-agnostic `Sigmoid(gate) -> Mul(x, sig)`, and added `custom.pytorch` to `opset_imports` when non-standard activations like `SiLU` are used. Verified by `test_bug_2_9_and_2_10_gated_unit_rank_agnostic_and_opset_import`.

---

## SECTION 1: CRITICAL BUGS (P0) — WHOLE-GRAPH WIPEOUTS

### Bug 1.1: `apply_moe_fusion` Replaces Entire Model Graph
* **File:** `macro_block_synthesizer.py` (Lines 2248-2294)
* **Code:**
  ```python
  for out_name in graph_outputs:
      out_producer = node_by_output.get(out_name)
      if out_producer and out_producer.op_type == "Add":
          sub_nodes = list(graph.node)
          ...
          moe_node = helper.make_node("MoEBlock", inputs=external_inputs, outputs=[out_name], ...)
          del graph.node[:]
          graph.node.append(moe_node)
  ```
* **Failure Scenario:** If any model has an `Add` producing an output and at least one `TopK` node anywhere, `sub_nodes = list(graph.node)` grabs **every single node in the entire model** (token embeddings, all 32 transformer layers, final LM head) and replaces the entire ONNX model with a single custom node `MoEBlock`. Furthermore, in multi-output models, it terminates on the first output and drops all remaining outputs.

### Bug 1.2: `apply_mamba_ssm_fusion` Replaces Entire Model Graph
* **File:** `macro_block_synthesizer.py` (Lines 2358-2402)
* **Code:**
  ```python
  if not exp_nodes or len(graph.node) < 20 or len(softplus_nodes) < 10:
      return model
  for out_name in graph_outputs:
      sub_nodes = list(graph.node)
      ...
      del graph.node[:]
      graph.node.append(mamba_node)
  ```
* **Failure Scenario:** In any model with at least 1 `Exp` and 10 `Mul`/`Add`/`Softplus` nodes (which virtually every Transformer or hybrid Mamba-Transformer model has), it sweeps the **entire ONNX model** into `MambaSSM` and deletes all other nodes.

### Bug 1.3: `apply_shifted_window_attention_fusion` Replaces Entire Model Graph
* **File:** `macro_block_synthesizer.py` (Lines 2407-2469)
* **Code:**
  ```python
  for out_name in graph_outputs:
      sub_nodes = list(graph.node)
      ...
      del graph.node[:]
      graph.node.append(swin_node)
  ```
* **Failure Scenario:** In a multi-stage Swin Transformer (e.g., Swin-B with PatchEmbed, Stage 1-4, Norm, and Linear Classifier), it deletes all 24 Swin stages and the classifier head, bundling everything into one single `ShiftedWindowAttention` node.

### Bug 1.4: `apply_deepseek_mla_macro_fusion` Replaces Entire Model Graph
* **File:** `macro_block_synthesizer.py` (Lines 2590-2634)
* **Code:**
  ```python
  for out_name in graph_outputs:
      sub_nodes = list(graph.node)
      ...
      del graph.node[:]
      graph.node.append(mla_node)
  ```
* **Failure Scenario:** Whenever `MultiHeadLatentAttention_KVProjection` and `ScaledDotProductAttention` are present in a model, it deletes all nodes across all layers and wraps the entire graph into a single node.

---

## SECTION 2: FATAL BUGS (P0) — FUNCTIONPROTO & GRAPH SERIALIZATION FAILURES

### Bug 2.1: `MultiHeadAttention` Input Arity Mismatch
* **File:** `macro_block_synthesizer.py` (Lines 992-1056)
* **Code:**
  Node creation: `mha_inputs = list(sdpa_node.input) + list(out_node.input[1:])` (4 to 6 inputs: `[q, k, v, mask, w_out, b_out]`).
  FunctionProto definition: `inputs=["x", "w_qkv", "w_out"]` (Exactly 3 inputs!).
* **Failure Scenario:** ONNX Runtime / `onnx.checker` throws `FAIL : Node (name) has 5 inputs, but function (MultiHeadAttention) defines 3 inputs`. The model cannot be serialized or loaded in Netron / ORT.

### Bug 2.2: `ScaledDotProductAttention` Mask Input Discarded in Function Definition
* **File:** `attention_synthesizer.py` (Lines 785, 840)
* **Code:**
  Node creation: `inputs=[q_inp, k_inp, v_inp] + extra_attn_inputs` (4 inputs if mask is present).
  FunctionProto definition: `inputs=["Q", "K", "V"]` (Only 3 inputs! Mask is omitted).
* **Failure Scenario:** Any attention block with an attention mask (BERT, GPT, TFLite encoder, causal decoders) fails ONNX checker with input arity mismatch.

### Bug 2.3: `MergeHeads` Function Definition Drops `Reshape` (Outputs 4D instead of 3D)
* **File:** `attention_synthesizer.py` (Lines 986, 1014-1025)
* **Code:**
  Pass drops both `Transpose(perm=[0, 2, 1, 3])` AND `Reshape` (`[B, S, H, D] -> [B, S, H*D]`).
  Function definition: `nodes=[t]` (Only Transpose, NO RESHAPE!).
* **Failure Scenario:** When inlining or executing `MergeHeads`, it outputs a 4D tensor `[B, S, H, D]`. The downstream `Linear` projection expects 2D/3D `[B, S, H*D]`, causing an immediate matrix dimension mismatch crash (`MatMul: Incompatible dimensions`).

### Bug 2.4: `SpatialGatingUnit` Function Definition Drops `gamma` & Biases
* **File:** `macro_block_synthesizer.py` (Lines 390-433)
* **Code:**
  Function definition: `norm = helper.make_node("RMSNorm", inputs=["u"], outputs=["norm_out"])` (Missing `gamma`!).
  `func_inputs=["v", "u", "w"]` (Only 3 inputs!).
* **Failure Scenario:** `RMSNorm` requires `gamma` (scale parameter). Running or expanding this function fails because `gamma` is missing and bias `b` is ignored.

### Bug 2.5: `AxialRearrange` Function Definition Body Contains Dummy `Identity`
* **File:** `axial_synthesizer.py` (Lines 515-530)
* **Code:**
  `id_node = helper.make_node("Identity", inputs=["x"], outputs=["y"])`
* **Failure Scenario:** `AxialRearrange` fuses complex frequency/time layout transformations (`Transpose(perm=[0, 3, 2, 1])`, `Reshape`). But its FunctionProto body is literally an `Identity` node! If any framework inlines or executes this model via ONNX Function expansion, all axial rearrangements become no-ops, destroying audio spectrogram layout and producing numerical garbage.

### Bug 2.6: `TensorRoute` Function Definition Completely Omits Slicing
* **File:** `axial_synthesizer.py` (Lines 225-235)
* **Code:**
  `gather_node = helper.make_node("Gather", inputs=["x", "idx"], outputs=["y"], axis=1)`
* **Failure Scenario:** `apply_index_tensor_routing` fuses `Gather` + `Slice` (e.g. `x[:, idx, start:end]`). But the function definition only performs `Gather`, discarding start, end, and slice_axis. Inlined models will output the un-sliced tensor.

### Bug 2.7: `Unbind` Function Definition Hardcodes `axis=0`
* **File:** `axial_synthesizer.py` (Lines 318-328)
* **Code:**
  `sp_node = helper.make_node("Split", inputs=["x"], outputs=s_outputs, axis=0)`
* **Failure Scenario:** Even if the original Split was on `axis=1` (heads) or `axis=2` (hidden chunks), the function body splits on `axis=0` (batch dimension). Inlined model splits along batch dimension instead of heads or channels!

### Bug 2.8: `DynamicQuantLinear` Function Definition Collision (Biased vs Unbiased)
* **File:** `macro_block_synthesizer.py` (Lines 2787-2790)
* **Code:**
  ```python
  if any(f.name == func_name and f.domain == domain for f in model.functions):
      return
  ```
* **Failure Scenario:** If a model has both biased and unbiased `DynamicQuantLinear` layers, the first layer registers `DynamicQuantLinear` with 2 inputs (`["X", "W"]`). When the second (biased) layer calls it with 3 inputs (`["X", "W", "bias"]`), registration is skipped. The biased node is linked to the 2-input function, causing an ONNX validator arity crash.

### Bug 2.9: `GatedUnit` Function Definition Forces 3D Transpose & Unsqueeze
* **File:** `attention_synthesizer.py` (Lines 468-478)
* **Code:**
  ```python
  t_node = helper.make_node("Transpose", inputs=["gate"], outputs=["gate_t"], perm=[0, 2, 1])
  u_node = helper.make_node("Unsqueeze", inputs=["gate_t", "axes_val"], outputs=["gate_4d"])
  ```
* **Failure Scenario:** If `gate` is a 4D tensor `[B, H, S, D]` or 2D tensor `[B, D]`, `Transpose(perm=[0, 2, 1])` crashes because permutation length 3 does not match tensor rank!

### Bug 2.10: Missing `custom.pytorch` in `_attach_specialized_gated_unit_function` Opset Imports
* **File:** `attention_synthesizer.py` (Lines 423-439)
* **Code:**
  `domain="" if act_name in ("Relu", "Tanh", "Sigmoid") else "custom.pytorch"`
  `opset_imports=[opset_standard]` (Only imports domain `""`!).
* **Failure Scenario:** When `act_name == "SiLU"`, the body node uses domain `"custom.pytorch"`, but `custom.pytorch` is NOT listed in `opset_imports`. ONNX validation throws: `Domain custom.pytorch used by node sig_out is not listed in opset_imports of function`.

---

## SECTION 3: SILENT NUMERICAL BUGS (P1) [BATCH 2 — 100% FIXED & VERIFIED]

### Bug 3.1: RoPE Function Computes $x \cdot (\cos + \sin)$ Without Rotation
* **File:** `attention_synthesizer.py`
* **Status:** **FIXED & VERIFIED**
* **Fix Summary:** In `_attach_rope_function_def`, implemented true `rotate_half` via `Split(x, num_outputs=2, axis=-1) -> Neg(x2) -> Concat([neg_x2, x1], axis=-1) -> rot`, followed by `x * cos + rot * sin`. Updated `registry.py` `_rotary_embedding` default to `interleaved=0` (rotate_half) to match modern LLM RoPE standards.
* **Verification:** Verified by `test_bug_3_1_rope_rotate_half_mathematical_fidelity` in `tests/test_p1_numerical_audit_fixes.py`.

### Bug 3.2: SDPA Function Definition Hardcodes Scaling Factor to `0.125`
* **File:** `attention_synthesizer.py`
* **Status:** **FIXED & VERIFIED**
* **Fix Summary:** Completely replaced hardcoded `0.125` with dynamic shape-based scaling inside FunctionProto: `Shape(Q) -> Gather(-1) -> Cast(FLOAT) -> Pow(-0.5) -> scale_const`. Works universally for any head dimension ($d_k=32, 64, 128, 256$, etc.).
* **Verification:** Verified across head dimensions 32, 64, 128 by `test_bug_3_2_sdpa_dynamic_head_dimension_scaling` in `tests/test_p1_numerical_audit_fixes.py`.

### Bug 3.3: Gemma-2 Softcapping Hardcodes `cap_val = 50.0`
* **File:** `attention_synthesizer.py`
* **Status:** **FIXED & VERIFIED**
* **Fix Summary:** `_attach_logits_softcap_function_def` now dynamically accepts and attaches `cap_val` (e.g. 30.0 for LM logits, 50.0 for attention) extracted from the graph during fusion instead of hardcoding 50.0.
* **Verification:** Verified for caps 15.0, 30.0, 50.0 by `test_bug_3_3_logits_softcap_parameterization` in `tests/test_p1_numerical_audit_fixes.py`.

### Bug 3.4: Highway Block Discards Biases
* **File:** `macro_block_synthesizer.py`, `runtime/registry.py`, `pattern/_matcher.py`
* **Status:** **FIXED & VERIFIED**
* **Fix Summary:** `apply_highway_block_fusion` preserves both transform bias (`b_h`) and carry bias (`b_t`), constructing 5-input Highway nodes `[x, wh, bh, wt, bt]`. `_attach_highway_function_def` defines 5-input FunctionProto with biased Linear operations. `registry.py` `_highway_block` seamlessly executes 5-input, 3-input (weights), and 3-input (activations) variants. Fixed `_matcher.py` parent tracking to properly support DAGs/diamonds where outputs have multiple consumers.
* **Verification:** Verified by `test_bug_3_4_highway_block_preserves_biases` in `tests/test_p1_numerical_audit_fixes.py`.

### Bug 3.5: SqueezeExcitation Discards FC1 and FC2 Biases
* **File:** `macro_block_synthesizer.py`, `runtime/registry.py`
* **Status:** **FIXED & VERIFIED**
* **Fix Summary:** `apply_squeeze_excitation_fusion` preserves `fc1` and `fc2` biases (`se_inputs = [x, w1, b1, w2, b2]`). `_attach_se_function_def` defines 5-input FunctionProto with biased Linear ops. `registry.py` `_squeeze_excitation` supports both 5-input (biased) and 3-input (unbiased) execution without crashing.
* **Verification:** Verified by `test_bug_3_5_squeeze_excitation_preserves_biases` in `tests/test_p1_numerical_audit_fixes.py`.

### Bug 3.6: LayerNorm -> RMSNorm Substitution in `FeedForwardSwiGLU`
* **File:** `feedforward_synthesizer.py`
* **Status:** **FIXED & VERIFIED**
* **Fix Summary:** Removed `"LayerNorm"` from `apply_feedforward_swiglu_fusion` norm check so that LayerNorm (with mean-subtraction and beta) is never silently converted into RMSNorm. LayerNorm is preserved intact in the graph, and the FFN is cleanly fused as 4-input or 7-input FeedForwardSwiGLU.
* **Verification:** Verified by `test_bug_3_6_layernorm_preserved_not_replaced_by_rmsnorm` in `tests/test_p1_numerical_audit_fixes.py`.

---

## SECTION 4: HARDCODED OVERFITTING (P1) [BATCH 3 — 100% FIXED & VERIFIED]

### Bug 4.1: `apply_multi_task_audio_head_fusion` Overfitted to Output & Node Names
* **File:** `macro_block_synthesizer.py`, `multi_task_audio_head_fusion.json`
* **Status:** **FIXED & VERIFIED**
* **Fix Summary:** Refactored from checking hardcoded names (`"conv_pitched"` and `{"inst_logp", "perc_prob", "vol_logp"}`) to universal structural fork topology matching. Now locates the trunk branch point feeding multi-branch feature extractors whose forward transitive closure reaches $\ge 2$ graph outputs. Fused outputs and FunctionProto definition dynamically bind to the actual model graph outputs (`target_outputs`).
* **Verification:** Verified by `test_bug_4_1_multi_task_audio_head_arbitrary_output_and_node_names` and `test_bug_4_1_backward_compatibility_with_model4` in `tests/test_p1_hardcoded_overfitting_audit_fixes.py`.

### Bug 4.2: `apply_vit_patch_embedding_fusion` String Matched `"pos_embed"`
* **File:** `macro_block_synthesizer.py`
* **Status:** **FIXED & VERIFIED**
* **Fix Summary:** Replaced string matching (`any("pos_embed" in inp or "position" in inp)`) and mandatory Concat check with forward BFS topological traversal from `vit_conv` through layout/token manipulation ops (`Reshape`, `Transpose`, `Flatten`, `Unsqueeze`, `Concat`) to locate the positional addition (`Add` or `SkipLayerNormalization`). Works regardless of whether position weights are named `w_pe`, `spatial_pe_weight`, or anonymous IDs, and supports ViT architectures with or without `[CLS]` token Concat.
* **Verification:** Verified by `test_bug_4_2_vit_patch_embedding_arbitrary_position_tensor_name` in `tests/test_p1_hardcoded_overfitting_audit_fixes.py` and `test_vuln_54_vit_cls_token_concat_pos_embed_unfused` in `tests/test_compiler_vulnerabilities_batch4.py`.

### Bug 4.3: `apply_mha_composite_fusion` String Matched `"past"` and `"Concat"`
* **File:** `macro_block_synthesizer.py`
* **Status:** **FIXED & VERIFIED**
* **Fix Summary:** Eliminated substring checks (`"past" in str(inp)` and `"Concat" in str(inp)`). Now structurally inspects the DAG producers of Key and Value tensors feeding SDPA: if K or V is produced by an ONNX `Concat` node (concatenating past KV cache with new projections), it is classified as `StatefulAttention`. A tensor simply named `"past_token_embedding"` is no longer misclassified as stateful.
* **Verification:** Verified by `test_bug_4_3_mha_kv_cache_concat_structural_detection` and `test_bug_4_3_no_false_positive_stateful_attention_from_past_in_tensor_name` in `tests/test_p1_hardcoded_overfitting_audit_fixes.py`.

### Bug 4.4: Exclusion String Check in Multi-Field Embedding
* **File:** `macro_block_synthesizer.py`
* **Status:** **FIXED & VERIFIED**
* **Fix Summary:** Replaced string exclusions (`not any(k in n.name for k in ("layers", "blocks", "self_attn"))`) with structural tensor classification. Differentiates positional encoding additions from transformer residual connections by checking whether the other input to `Add` is static (an initializer, Constant, or Gather/Unsqueeze table) rather than a dynamic activation produced by an Attention/FFN block. Multi-field embeddings in models with hierarchical namespaces (e.g. `model.layers.0`) are now cleanly fused.
* **Verification:** Verified by `test_bug_4_4_multi_field_embedding_with_layers_and_blocks_in_names` in `tests/test_p1_hardcoded_overfitting_audit_fixes.py`.

### Bug 4.5: RoPE `_find_base_frequency_tensor` Global `ScatterND` Scan
* **File:** `attention_synthesizer.py`
* **Status:** **FIXED & VERIFIED**
* **Fix Summary:** Removed arbitrary whole-graph search for `ScatterND` when encountering `Einsum`. Now traverses only the local ancestor sub-DAG of the frequency calculation inputs (`Einsum`, `Mul`, `MatMul`), preventing erroneous jumps to unrelated `ScatterND` nodes elsewhere in the graph.
* **Verification:** Verified by `test_bug_4_5_find_base_frequency_does_not_jump_to_unrelated_scatternd` in `tests/test_p1_hardcoded_overfitting_audit_fixes.py`.

### Bug 4.6: `apply_axial_rearrange_fusion` Rank/Permutation Hardcoding
* **File:** `axial_synthesizer.py`, `runtime/registry.py`
* **Status:** **FIXED & VERIFIED**
* **Fix Summary:** Relaxed hardcoded `perm == [0, 2, 1, 3]` to accept dynamic permutations and attach `perm` as an attribute. Corrected `_attach_axial_rearrange_function_def` for `freq_to_time` mode to properly include 3D $\to$ 4D unflattening before Transpose and 4D $\to$ 3D flattening after Transpose, eliminating ONNX rank mismatch errors during function inlining and verification. Updated `registry.py` to support dynamic `perm` across all rearrangement modes.
* **Verification:** Verified by `test_bug_4_6_axial_rearrange_arbitrary_permutation_and_function_proto_fidelity` in `tests/test_p1_hardcoded_overfitting_audit_fixes.py`.

---

## SECTION 5: PASS ORDERING & AMBIGUITY CONFLICTS (P1)

### Bug 5.1: `apply_deep_mlp_norm_act_fusion` Cannibalizes FeedForward Blocks
* **File:** `macro_block_synthesizer.py`
* **Status:** **FIXED & VERIFIED**
* **Fix Summary:** Guarded `apply_deep_mlp_norm_act_fusion` by inspecting downstream consumers of `act_node.output[0]`. If the activation output directly feeds a downstream projection (`Linear`, `MatMul`, `Gemm`), it belongs to a multi-layer FFN/MLP and is skipped, preserving the block intact for high-level macro synthesis.
* **Verification:** Verified by `test_bug_5_1_deep_mlp_does_not_cannibalize_two_layer_feedforward` in `tests/test_p1_p2_sections_5_6_7_audit_fixes.py`.

### Bug 5.2: `apply_gated_unit_fusion` Cannibalizes `SiLU` (`x * Sigmoid(x)`)
* **File:** `attention_synthesizer.py`
* **Status:** **FIXED & VERIFIED**
* **Fix Summary:** In `apply_gated_unit_fusion`, added an identity check `if other_inp == gate_in: continue`. When both branches of `Mul` originate from the exact same tensor, it represents canonical raw SiLU ($x \cdot \sigma(x)$), not a dual-branch gated unit ($x \cdot \sigma(y)$). Raw SiLU passes through untampered to downstream activation canonicalization passes.
* **Verification:** Verified by `test_bug_5_2_gated_unit_does_not_cannibalize_raw_silu` in `tests/test_p1_p2_sections_5_6_7_audit_fixes.py`.

### Bug 5.3: Global Wipeout Passes Run Before `TransformerEncoderLayer`
* **File:** `macro_block_synthesizer.py`
* **Status:** **FIXED & VERIFIED**
* **Fix Summary:** Reordered `MacroBlockSynthesizer.run` hierarchically: full macro blocks (`TransformerEncoderLayer`, `ParallelTransformerBlock`) execute FIRST before constituent sub-blocks (`HighwayBlock`, `SqueezeExcitation`, `ResidualBlock`), and generic low-level fallbacks (`DeepMLP_Norm_Act`) execute LAST.
* **Verification:** Verified by `test_bug_5_3_pass_ordering_transformer_encoder_layer_fuses_before_sub_blocks` in `tests/test_p1_p2_sections_5_6_7_audit_fixes.py`.

---

## SECTION 6: BRITTLE ARCHITECTURAL BLINDNESS (P2)

### Bug 6.1: `apply_feedforward_fusion` Strictly Requires `RMSNorm`
* **File:** `feedforward_synthesizer.py`
* **Status:** **FIXED & VERIFIED**
* **Fix Summary:** Expanded normalization acceptance in `apply_feedforward_fusion` to include standard `LayerNormalization`, `SimplifiedLayerNormalization`, and `RMSNorm` with optional bias (`beta`). Updated runtime registry handler `_feed_forward` to support `F.layer_norm` with optional bias.
* **Verification:** Verified by `test_bug_6_1_feedforward_fusion_supports_layernorm_and_rmsnorm` in `tests/test_p1_p2_sections_5_6_7_audit_fixes.py`.

### Bug 6.2: `apply_feedforward_swiglu_fusion` Ignores ReGLU & GeGLU
* **File:** `feedforward_synthesizer.py`, `runtime/registry.py`
* **Status:** **FIXED & VERIFIED**
* **Fix Summary:** Generalized GLU fusion into a unified parametric synthesizer supporting `FeedForwardSwiGLU` (SiLU/Swish), `FeedForwardGeGLU` (GELU), and `FeedForwardReGLU` (ReLU). Generates exact specialized FunctionProtos and registered corresponding runtime handlers in `registry.py` with full numerical verification.
* **Verification:** Verified by `test_bug_6_2_feedforward_swiglu_supports_geglu_and_reglu_with_exact_numerical_equivalence` in `tests/test_p1_p2_sections_5_6_7_audit_fixes.py` and model zoo tests `geglu_mlp_block_opset18.onnx` and `reglu_mlp_block_opset18.onnx`.

### Bug 6.3: `apply_softmax_matmul_fusion` Fails When Layout Ops Exist
* **File:** `attention_synthesizer.py`
* **Status:** **FIXED & VERIFIED**
* **Fix Summary:** Extended `apply_softmax_matmul_fusion` to trace backwards through transparent layout and casting nodes (`Dropout`, `Identity`, `Cast`) between `MatMul` and `Softmax`. Subgraph nodes are safely absorbed into the fused SDPA block.
* **Verification:** Verified by `test_bug_6_3_softmax_matmul_traces_through_dropout_and_cast` in `tests/test_p1_p2_sections_5_6_7_audit_fixes.py`.

### Bug 6.4: Pre-Activation ResNet Expects Non-Existent ReLU After Add
* **File:** `macro_block_synthesizer.py`
* **Status:** **FIXED & VERIFIED**
* **Fix Summary:** Refactored `apply_preactivation_resnet_fusion` to use robust topological matching for the canonical Pre-Activation ResNet invariant ($y = x + F(x)$) where residual `Add` is the primary exit, while gracefully absorbing an optional post-addition `Relu` if present. Independent of whether `x` is a graph input or intermediate node.
* **Verification:** Verified by `test_bug_6_4_preactivation_resnet_fuses_without_post_add_relu` in `tests/test_p1_p2_sections_5_6_7_audit_fixes.py`.

### Bug 6.5: Sequence Classifier Head Blind to `ReduceMean`
* **File:** `macro_block_synthesizer.py`
* **Status:** **FIXED & VERIFIED**
* **Fix Summary:** Added support for `ReduceMean` (global average pooling) in `apply_sequence_classifier_head_fusion`, both when `ReduceMean` acts as the terminal pooling stage or as an intermediate stage feeding Linear/Gemm layers.
* **Verification:** Verified by `test_bug_6_5_sequence_classifier_head_handles_reducemean_pooling` in `tests/test_p1_p2_sections_5_6_7_audit_fixes.py`.

### Bug 6.6: `apply_multi_field_embedding_fusion` Fails on Balanced Binary Trees
* **File:** `macro_block_synthesizer.py`
* **Status:** **FIXED & VERIFIED**
* **Fix Summary:** Replaced linear chain iteration with a bottom-up iterative summation reduction. Iteratively reduces pairs of terms until a single root accumulator remains, seamlessly supporting linear cascades, balanced binary trees, and arbitrary binary DAG trees of embeddings.
* **Verification:** Verified by `test_bug_6_6_multi_field_embedding_fuses_balanced_binary_add_tree` in `tests/test_p1_p2_sections_5_6_7_audit_fixes.py`.

### Bug 6.7: Arbitrary K/V Assignment in `apply_deepseek_mla_fusion`
* **File:** `attention_synthesizer.py`
* **Status:** **FIXED & VERIFIED**
* **Fix Summary:** Refactored `apply_deepseek_mla_fusion` to disambiguate Key and Value projections topologically: Key reaches `Softmax` (in $Q \cdot K^T$ score calculation), while Value is consumed by `SoftmaxMatMul` or `MatMul(Softmax, V)`. Deterministic regardless of dict ordering or node positions.
* **Verification:** Verified by `test_bug_6_7_deepseek_mla_deterministic_kv_assignment_via_attention_topology` in `tests/test_p1_p2_sections_5_6_7_audit_fixes.py`.

---

## SECTION 7: TOPOLOGICAL & SSA CORRUPTION (P2)

### Bug 7.1: `apply_dynamic_quant_linear_fusion` Drops Shared Quantizers
* **File:** `macro_block_synthesizer.py`
* **Status:** **FIXED & VERIFIED**
* **Fix Summary:** Refactored `apply_dynamic_quant_linear_fusion` to iterate over and fuse ALL sibling `MatMul` consumers sharing the same activation quantizer into distinct `DynamicQuantLinear` nodes. The shared `DynamicQuantizeLinear` and `DequantizeLinear` nodes are only dropped once all of their consumers have been fused.
* **Verification:** Verified by `test_bug_7_1_dynamic_quant_linear_preserves_shared_quantizer_for_sibling_projections` in `tests/test_p1_p2_sections_5_6_7_audit_fixes.py`.

### Bug 7.2: `_replace_nodes_topologically` Appends Replacements to End of Graph
* **File:** `macro_block_synthesizer.py`
* **Status:** **FIXED & VERIFIED**
* **Fix Summary:** Inserted an automatic post-replacement topological sort (`_toposort_nodes`) that enforces strict topological ordering across the entire graph, and fixed `is_dropped` logic to check `id(n) in drop_ids` regardless of node naming.
* **Verification:** Verified by `test_bug_7_2_replace_nodes_topologically_handles_named_ids_and_preserves_toposort` in `tests/test_p0_synthesis_audit_fixes.py`.

### Bug 7.3: In-Place Node Mutation in `_handle_k_transpose`
* **File:** `attention_synthesizer.py`
* **Status:** **FIXED & VERIFIED**
* **Fix Summary:** Updated `_handle_k_transpose` to inspect consumer counts before modifying Transpose nodes. If a Transpose node has multiple consumers, a new dedicated Transpose node is created for attention fusion, preventing in-place corruption of shared graph nodes.
* **Verification:** Verified by `test_bug_7_3_handle_k_transpose_does_not_corrupt_shared_transpose_in_place` in `tests/test_p1_p2_sections_5_6_7_audit_fixes.py`.

### Bug 7.4: Backwards Graph Deletion in `apply_index_tensor_routing`
* **File:** `axial_synthesizer.py`
* **Status:** **FIXED & VERIFIED**
* **Fix Summary:** Guarded ancestor deletion loop in `apply_index_tensor_routing`. Shared ancestor nodes (`Shape`, `Gather`, `Sub`, `Unsqueeze`) are only marked for deletion if ALL of their downstream consumers are within the fused subgraph.
* **Verification:** Verified by `test_bug_7_4_index_tensor_routing_does_not_delete_shared_shape_or_gather` in `tests/test_p1_p2_sections_5_6_7_audit_fixes.py`.

### Bug 7.5: Lexical Scoping Collision in `_canonicalize_subgraph_nodes`
* **File:** `macro_block_synthesizer.py`
* **Status:** **FIXED & VERIFIED**
* **Fix Summary:** Overhauled `_canonicalize_subgraph_nodes` to partition namespaces into distinct `input_map`, `output_map`, and `internal_map`. Added automatic generation of `Identity` nodes for passthrough inputs directly returned as outputs, and added recursive canonicalization for nested subgraphs (`If`, `Loop`).
* **Verification:** Verified by `test_bug_7_5_canonicalize_subgraph_nodes_handles_passthrough_inputs_without_collision` in `tests/test_p1_p2_sections_5_6_7_audit_fixes.py`.
