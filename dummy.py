import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn


# 1. Ek simple PyTorch model define karein
class SimpleModel(nn.Module):

    def __init__(self, input_dim=10, hidden_dim=32, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def main():
    # Model ko eval mode mein laana bahut zaroori hai (Dropout/BatchNorm ke liye)
    model = SimpleModel(input_dim=10, hidden_dim=32, output_dim=2)
    model.eval()

    # 2. Dummy input tensor banayein (batch_size=1, features=10)
    dummy_input = torch.randn(1, 10, dtype=torch.float32)

    onnx_path = "simple_model.onnx"

    # 3. Model ko ONNX mein export karein
    batch_dim = torch.export.Dim("batch_size", min=1)

    torch.onnx.export(
        model,
        (dummy_input,),
        onnx_path,
        export_params=True,
        opset_version=28,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_shapes={"x": {0: batch_dim}},
    )
    print(f"Model successfully exported to {onnx_path}")

    # 4. Exported ONNX model ka structure verify karein
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model check passed!")

    # 5. PyTorch vs ONNXRuntime output compare karein
    with torch.no_grad():
        torch_out = model(dummy_input).numpy()

    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_out = ort_session.run(None, ort_inputs)[0]

    max_diff = np.max(np.abs(torch_out - ort_out))
    print(f"Max difference between PyTorch and ONNX Runtime: {max_diff:.6e}")


if __name__ == "__main__":
    main()