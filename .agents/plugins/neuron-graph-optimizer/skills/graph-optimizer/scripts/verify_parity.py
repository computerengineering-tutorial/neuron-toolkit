#!/usr/bin/env python3
"""Run differential cross-verification between original and optimized ONNX models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx
from neuron_toolkit.rules.engine import JSONRuleEngine
from neuron_toolkit.runtime.cross_verifier import CrossVerifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-verify an ONNX model against ONNX Runtime.")
    parser.add_argument("original_model", help="Path to original ONNX model")
    parser.add_argument("--optimized-model", default=None, help="Path to optimized ONNX model (if omitted, JSONRuleEngine runs on original)")
    parser.add_argument("--mode", default="function_proto", choices=["function_proto", "plain_custom_op"], help="Engine mode")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance")
    args = parser.parse_args()

    orig_path = Path(args.original_model)
    if not orig_path.exists():
        print(f"Error: Original model '{orig_path}' not found.", file=sys.stderr)
        sys.exit(1)

    orig_model = onnx.load(str(orig_path))

    if args.optimized_model:
        opt_path = Path(args.optimized_model)
        if not opt_path.exists():
            print(f"Error: Optimized model '{opt_path}' not found.", file=sys.stderr)
            sys.exit(1)
        opt_model = onnx.load(str(opt_path))
    else:
        print(f"Applying JSONRuleEngine (mode={args.mode}) on {orig_path.name}...")
        engine = JSONRuleEngine(mode=args.mode)
        opt_model = engine.apply(orig_model)

    verifier = CrossVerifier(orig_model, opt_model, rtol=args.rtol, atol=args.atol)
    report = verifier.verify()
    print("\n" + report.summary())

    if not report.all_passed:
        print("Cross-verification FAILED! Parity checks did not meet tolerance.", file=sys.stderr)
        sys.exit(1)
    else:
        print("Cross-verification SUCCEEDED! 100% numerical parity confirmed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
