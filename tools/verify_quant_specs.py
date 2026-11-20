#!/usr/bin/env python3
"""Verify representative quantized specs."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generators.registry import ModelRegistry
from generators.quantization import export_quantized_model

def main():
    specs = ModelRegistry.discover(REPO_ROOT / "generators")
    spec_dict = {s.name: s for s in specs}
    
    test_names = [
        "activation_erf_gelu",
        "activation_silu",
        "activation_hardswish",
        "mha_separate_qkv",
        "mha_fused_qkv",
        "mha_causal",
        "mha_rope",
        "mha_gqa",
        "mha_mqa",
        "swiglu_mlp_block",
        "geglu_mlp_block",
        "reglu_mlp_block",
        "transformer_pre_ln",
        "transformer_post_ln",
        "norm_layernorm",
        "norm_rmsnorm_l2",
        "norm_groupnorm",
        "linear_gemm",
        "conv2d_bn_relu",
        "resnet_basic",
        "mobilenetv2_invres",
    ]
    
    passed = 0
    failed = []
    
    for name in test_names:
        if name not in spec_dict:
            print(f"[MISSING] {name}")
            continue
        spec = spec_dict[name]
        try:
            m = spec.instantiate().eval()
            inputs = spec.get_dummy_inputs()
            out_dir = REPO_ROOT / ".helper" / "models" / "quantized" / spec.resolve_subfolder()
            path, opt_proto, stats = export_quantized_model(
                model=m,
                dummy_inputs=inputs,
                input_names=spec.input_names,
                output_names=spec.output_names,
                base_name=spec.name,
                out_dir=out_dir,
                rtol=0.05,
                atol=0.05,
                auto_verify=True,
            )
            print(f"[PASS] {name:<25} raw={stats['raw_nodes']:<3} -> opt={stats['opt_nodes']:<3} ({stats['reduction_pct']:>5.1f}%) | diff={stats['max_diff']:.5f} | cos={stats['cosine_sim']:.4f}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name:<25} Error: {e}")
            failed.append((name, str(e)))

    print(f"\nSummary: {passed} passed, {len(failed)} failed out of {len(test_names)}")

if __name__ == "__main__":
    main()
