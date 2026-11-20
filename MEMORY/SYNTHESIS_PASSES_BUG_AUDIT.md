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

| Severity | Count | Primary Impact |
|---|---|---|
| **CRITICAL (P0): Whole-Graph Wipeouts** | 4 | Subgraph capture grabs `list(graph.node)` and deletes the entire model into 1 custom node |
| **FATAL (P0): FunctionProto Mismatches** | 10 | Function definition inputs/outputs or bodies mismatch the instantiated node, causing ONNX checker or runtime aborts |
| **SILENT NUMERICAL (P1): Math & Bias Bugs** | 6 | Biases dropped into thin air; RoPE with no rotation; hardcoded scale (0.125) and softcap (50.0) |
| **HARDCODED OVERFITTING (P1): String Matching** | 4 | Name-based checks (`"conv_pitched"`, `"pos_embed"`, `inst_logp`, `"past"`) |
| **PASS ORDERING & AMBIGUITY (P1)** | 3 | Pass ordering collisions cannibalizing sub-blocks (e.g. DeepMLP eating FFN; GatedUnit eating SiLU) |
| **BRITTLE BLINDNESS (P2): Architectural Gaps** | 7 | Strict RMSNorm-only requirement; adjacent peephole assumptions; rigid permutations |
| **TOPOLOGICAL & SSA CORRUPTION (P2)** | 5 | Shared producers dropped; unplaced replacements appended to end of graph; in-place node mutations |

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

## SECTION 3: SILENT NUMERICAL BUGS (P1)

### Bug 3.1: RoPE Function Computes $x \cdot (\cos + \sin)$ Without Rotation
* **File:** `attention_synthesizer.py` (Lines 274-278)
* **Code:**
  ```python
  m_cos = helper.make_node("Mul", inputs=["x", "cos"], outputs=["q_cos"])
  m_sin = helper.make_node("Mul", inputs=["x", "sin"], outputs=["q_sin"])
  add_node = helper.make_node("Add", inputs=["q_cos", "q_sin"], outputs=["y"])
  ```
* **Failure Scenario:** Mathematical RoPE requires rotating coordinates: $[-x_2 \sin, x_1 \sin]$. The function body multiplies the unrotated $x$ by both $\cos$ and $\sin$, producing $x \cdot (\cos \theta + \sin \theta)$.

### Bug 3.2: SDPA Function Definition Hardcodes Scaling Factor to `0.125`
* **File:** `attention_synthesizer.py` (Lines 826-834)
* **Code:**
  `scale_val = helper.make_tensor("scale_val", onnx.TensorProto.FLOAT, [1], [0.125])`
* **Failure Scenario:** `0.125` is only valid for $d_k = 64$ ($1/\sqrt{64} = 0.125$). For $d_k = 32$ ($0.1768$), $d_k = 128$ ($0.0884$), or $d_k = 256$ ($0.0625$), the function definition evaluates with the wrong scale, corrupting attention logits.

### Bug 3.3: Gemma-2 Softcapping Hardcodes `cap_val = 50.0`
* **File:** `attention_synthesizer.py` (Lines 1518-1525)
* **Code:**
  `value=helper.make_tensor("cap_val", onnx.TensorProto.FLOAT, [1], [50.0])`
* **Failure Scenario:** In Gemma-2, attention logits use `cap = 50.0`, but final LM output logits use `cap = 30.0`. Grok models use `cap = 30.0`. All models expanding this function silently evaluate with `50.0`.

### Bug 3.4: Highway Block Discards Biases
* **File:** `macro_block_synthesizer.py` (Lines 527-532)
* **Code:**
  ```python
  hw_inputs = [x_node.output[0] if hasattr(x_node, "output") else str(x_node)]
  if lin_h_node and len(lin_h_node.input) > 1:
      hw_inputs.append(lin_h_node.input[1])  # Only weight!
  if lin_t_node and len(lin_t_node.input) > 1:
      hw_inputs.append(lin_t_node.input[1])  # Only weight!
  ```
* **Failure Scenario:** `lin_h_node.input[2]` (transform bias) and `lin_t_node.input[2]` (carry gate bias) are dropped. The resulting Highway block computes without bias.

### Bug 3.5: SqueezeExcitation Discards FC1 and FC2 Biases
* **File:** `macro_block_synthesizer.py` (Lines 665-668)
* **Code:**
  `for fc in (fc1_node, fc2_node): if len(fc.input) > 1: se_inputs.append(fc.input[1])`
* **Failure Scenario:** PyTorch SE blocks (SENet, MobileNetV3, EfficientNet) always include bias in FC1/FC2. Dropping bias shifts channel excitation weights, causing model degradation.

### Bug 3.6: LayerNorm -> RMSNorm Substitution in `FeedForwardSwiGLU`
* **File:** `feedforward_synthesizer.py` (Lines 268, 376, 413)
* **Code:**
  Pass matches if `norm_node.op_type in ("RMSNorm", "RMSNormalization", "LayerNorm")`.
  Function definition: Always generates `RMSNormalization(inputs=["X", "gamma"])`!
* **Failure Scenario:** Models with `LayerNorm` (and bias `beta`) have their normalization silently replaced by `RMSNorm`, discarding `beta` and changing the mathematical formula of the norm!

---

## SECTION 4: HARDCODED OVERFITTING (P1) — VIOLATIONS OF ZERO-HARDCODING

### Bug 4.1: `apply_multi_task_audio_head_fusion` Matches Exact Output & Node Names
* **File:** `macro_block_synthesizer.py` (Lines 1952-1965)
* **Code:**
  `target_heads = {"inst_logp", "perc_prob", "vol_logp"}`
  `if "conv_pitched" in n.name and n.op_type in ("BatchNormalization", "Conv"):`
* **Failure Scenario:** Only matches one specific handwritten test model (`model2.onnx`). Any production audio model with different output names or node names is completely ignored.

### Bug 4.2: `apply_vit_patch_embedding_fusion` String Matches `"pos_embed"`
* **File:** `macro_block_synthesizer.py` (Line 2504)
* **Code:**
  `if any("pos_embed" in inp or "position" in inp for inp in a.input):`
* **Failure Scenario:** Fails on any anonymized ONNX model, obfuscated graph, or framework where positional embeddings are named `w_pe`, `embed_positions`, or numeric IDs.

### Bug 4.3: `apply_mha_composite_fusion` String Matches `"past"`
* **File:** `macro_block_synthesizer.py` (Line 1010)
* **Code:**
  `if any("past" in str(inp) for inp in sdpa_node.input) or any("Concat" in str(inp) for inp in sdpa_node.input[1:]): target_op = "StatefulAttention"`
* **Failure Scenario:** Unreliable heuristic: an input named `"past_dropout"` or `"past_norm"` gets incorrectly classified as `StatefulAttention`. Conversely, standard KV caches named `kv_cache` or `cache_k` are missed.

### Bug 4.4: Exclusion String Check in Multi-Field Embedding
* **File:** `macro_block_synthesizer.py` (Line 1681)
* **Code:**
  `if n.op_type == "Add" and not any(k in n.name for k in ("layers", "blocks", "self_attn")):`
* **Failure Scenario:** Any embedding table whose nodes happen to contain the word "layers" or "blocks" in their namespace is rejected.

---

## SECTION 5: PASS ORDERING & AMBIGUITY CONFLICTS (P1)

### Bug 5.1: `apply_deep_mlp_norm_act_fusion` Cannibalizes FeedForward Blocks
* **File:** `macro_block_synthesizer.py` (Lines 189, 215-300)
* **Code:**
  Runs at the beginning of `MacroBlockSynthesizer.run`. It matches `Norm -> Linear -> Gelu/Relu/SiLU`.
* **Failure Scenario:** The first half of a standard 2-layer FFN is `Norm -> Linear -> Gelu`. `DeepMLP_Norm_Act` eats this half, turning the FFN into `DeepMLP_Norm_Act -> Linear`. Downstream `FeedForward` and `TransformerBlock` passes fail to recognize the FFN because it has been dismantled.

### Bug 5.2: `apply_gated_unit_fusion` Cannibalizes `SiLU` (`x * Sigmoid(x)`)
* **File:** `attention_synthesizer.py` (Lines 345-385)
* **Code:**
  `Mul(x, Sigmoid(x))` matches `sig_in = [Sigmoid]`.
* **Failure Scenario:** If `SiLU` is in raw form (`x * Sigmoid(x)`), `apply_gated_unit_fusion` matches it and turns it into `GatedUnit_Sigmoid(x, x)`! Downstream `apply_silu_fusion` never gets to run.

### Bug 5.3: Global Wipeout Passes Run Before `TransformerEncoderLayer`
* **File:** `macro_block_synthesizer.py` (Lines 198-207)
* **Code:**
  `apply_moe_fusion` (L198), `apply_mamba_ssm_fusion` (L200), and `apply_shifted_window_attention_fusion` (L201) run BEFORE `apply_transformer_encoder_layer_fusion` (L207).
* **Failure Scenario:** If any loose trigger condition matches in these passes, the whole graph is wiped out, completely bypassing `TransformerEncoderLayer`.

---

## SECTION 6: BRITTLE ARCHITECTURAL BLINDNESS (P2)

### Bug 6.1: `apply_feedforward_fusion` Strictly Requires `RMSNorm`
* **File:** `feedforward_synthesizer.py` (Lines 76-78)
* **Code:**
  `if not norm or norm.op_type not in ("RMSNorm", "RMSNormalization"): continue`
* **Failure Scenario:** Rejects `LayerNormalization` and Post-LN. Standard BERT, RoBERTa, ViT, GPT-2, Whisper, DeBERTa, and TFLite encoder models fail to fuse FFN.

### Bug 6.2: `apply_feedforward_swiglu_fusion` Ignores ReGLU & GeGLU
* **File:** `feedforward_synthesizer.py` (Line 232)
* **Code:**
  `if cand_act and cand_act.lower() not in ("silu", "swish"): continue`
* **Failure Scenario:** ReGLU (PaLM, TFLite encoder_0) and GeGLU (Gemma, T5-v1.1) are rejected. There is no `FeedForwardReGLU` or `FeedForwardGeGLU` pass.

### Bug 6.3: `apply_softmax_matmul_fusion` Fails When Layout Ops Exist
* **File:** `attention_synthesizer.py` (Lines 515-516)
* **Code:**
  `in0 = node_by_output.get(n.input[0]); if in0 and in0.op_type == "Softmax":`
* **Failure Scenario:** When multi-head attention contains `Softmax -> Reshape -> Transpose -> MatMul`, `in0` is `Transpose` or `Reshape`. `SoftmaxMatMul` fails to match, which cascades into SDPA and MHA fusion failing.

### Bug 6.4: Pre-Activation ResNet Expects Non-Existent ReLU After Add
* **File:** `macro_block_synthesizer.py` (Line 734)
* **Code:**
  `relu2 = Pattern.op("Relu", add).capture("relu2")`
* **Failure Scenario:** Genuine Pre-Act ResNet (He et al., 2016) has NO activation after the residual addition ($y = x + F(x)$). The pass expects an activation that does not exist, so it never matches true Pre-Act ResNets.

### Bug 6.5: Sequence Classifier Head Blind to `ReduceMean`
* **File:** `macro_block_synthesizer.py` (Lines 1848-1870)
* **Code:**
  Line 1848 filters for `("Squeeze", "ReduceMean", "Linear", "Add")`. Line 1856 only handles `curr.op_type == "Squeeze"`.
* **Failure Scenario:** If a classifier head pools with `ReduceMean` (global average pooling in ViT / ConvNeXt), `curr.op_type == "ReduceMean"` is unhandled, `norm_node` remains `None`, and the pass terminates with 0 matches.

### Bug 6.6: `apply_multi_field_embedding_fusion` Fails on Balanced Binary Trees
* **File:** `macro_block_synthesizer.py` (Lines 1643-1670)
* **Code:**
  Expects a sequential linear chain `Add(Add(Add(e1, e2), e3), e4)`.
* **Failure Scenario:** In PyTorch models exported with balanced summation `Add(Add(e1, e2), Add(e3, e4))`, `next_add` is `None` and the pass fails.

### Bug 6.7: Arbitrary K/V Assignment in `apply_deepseek_mla_fusion`
* **File:** `attention_synthesizer.py` (Lines 1570-1574)
* **Code:**
  `up_k, up_v = up_proj_consumers[0], up_proj_consumers[1]`
* **Failure Scenario:** Depends on random dictionary iteration order or graph node position. If consumer 0 is V and consumer 1 is K, Keys and Values are swapped, producing garbage attention.

---

## SECTION 7: TOPOLOGICAL & SSA CORRUPTION (P2)

### Bug 7.1: `apply_dynamic_quant_linear_fusion` Drops Shared Quantizers
* **File:** `macro_block_synthesizer.py` (Lines 2657-2712)
* **Code:**
  When `DynamicQuantizeLinear` -> `DequantizeLinear` output feeds multiple branches (e.g. Q, K, and V projections in Self-Attention): `mm_cands[0]` is picked, and `nodes_to_drop.update({dql.name, dq_act.name})` unconditionally deletes the shared quantizer and dequantizer nodes!
* **Failure Scenario:** Sibling projections (K and V) are left with missing input tensors, triggering `RuntimeError: Operation requires missing input`.

### Bug 7.2: `_replace_nodes_topologically` Appends Replacements to End of Graph
* **File:** `macro_block_synthesizer.py` (Lines 84-94)
* **Code:**
  `new_graph_nodes.extend(new_node_map.values())`
* **Failure Scenario:** If a replacement node is not matched to a dropped node's output, it is appended to the bottom of the graph. If intermediate nodes consume its output, ONNX checker validation fails with: `Nodes in a graph must be topologically sorted, however input 'X' of node 'Y' is not output of any previous nodes`.

### Bug 7.3: In-Place Node Mutation in `_handle_k_transpose`
* **File:** `attention_synthesizer.py` (Lines 595-597)
* **Code:**
  `perm_attrs[0].ints[:] = new_perm`
  `tr_node.input[0] = k_raw_input`
* **Failure Scenario:** Mutates the shared node in place. If `tr_node` is consumed elsewhere, other paths receive corrupted permutations and inputs.

### Bug 7.4: Backwards Graph Deletion in `apply_index_tensor_routing`
* **File:** `axial_synthesizer.py` (Lines 178-195)
* **Code:**
  `while curr_n: nodes_to_remove.add(curr_n.name); ... if prev.op_type in ("Unsqueeze", "Sub", "Gather", "Shape"): curr_n = prev`
* **Failure Scenario:** Recursively deletes all ancestor `Shape`, `Gather`, `Sub` nodes without checking consumer counts, deleting the model's global `Shape` node.

### Bug 7.5: Lexical Scoping Collision in `_canonicalize_subgraph_nodes`
* **File:** `macro_block_synthesizer.py` (Lines 146-152)
* **Code:**
  `tensor_map = {**input_map, **output_map, **internal_map}`
* **Failure Scenario:** If an external input is passed directly as an external output (residual pass-through), `output_map` overwrites `input_map`, renaming the input tensor to `out_0`. Subgraphs (`If`, `Loop`) are not recursively canonicalized.
