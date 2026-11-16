"""Demo script for ArchitectureBlockPredictor on .temp/diffrent_models."""

from pathlib import Path
from neuron_toolkit.compiler.analyzer.block_predictor import ArchitectureBlockPredictor

def main() -> None:
    predictor = ArchitectureBlockPredictor()
    models_dir = Path(".temp/diffrent_models")
    print("\n" + "#" * 80)
    print("   AUTOMATIC ARCHITECTURE BLOCK PREDICTION & MOTIF ANALYSIS DEMO   ")
    print("#" * 80 + "\n")

    for p in sorted(models_dir.glob("model*.onnx")):
        manifest = predictor.analyze(p)
        manifest.print_report()
        print("\n")

if __name__ == "__main__":
    main()
