#!/usr/bin/env python3
"""Comprehensive Recurrent Neural Network (RNN) Architecture Family Generator.

Generates self-contained PyTorch models and multi-opset ONNX exports for:
1. VanillaRNN_Single (1-layer Tanh RNN)
2. VanillaRNN_Multi (2-layer ReLU RNN)
3. StandardLSTM_Uni (1-layer Unidirectional LSTM)
4. StandardLSTM_Bi (1-layer Bidirectional LSTM)
5. StandardGRU_Uni (1-layer Unidirectional GRU)
6. StandardGRU_Bi (1-layer Bidirectional GRU)
7. CustomCellLSTM (Explicit cell unrolling: Input, Forget, Cell, Output gates)
8. CustomCellGRU (Explicit cell unrolling: Reset, Update, Candidate gates)
9. QuasiRNN_QRNN (Conv1D + parallel pooling elementwise recurrence)
10. SRU_Cell (Simple Recurrent Unit with highway connection)

Exports across ONNX Opsets 14, 17, 18, 20.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import onnx
import torch
from torch import nn

warnings.filterwarnings("ignore")

OUT_DIR = Path(".helper/models/recurrent")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. Vanilla RNN Variants
# =============================================================================
class VanillaRNN_Single(nn.Module):
    def __init__(self, input_size: int = 32, hidden_size: int = 32) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            nonlinearity="tanh",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return out


class VanillaRNN_Multi(nn.Module):
    def __init__(
        self, input_size: int = 32, hidden_size: int = 32, num_layers: int = 2
    ) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="relu",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return out


# =============================================================================
# 2. Standard LSTM Variants
# =============================================================================
class StandardLSTM_Uni(nn.Module):
    def __init__(self, input_size: int = 32, hidden_size: int = 32) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h_n, c_n) = self.lstm(x)
        return out


class StandardLSTM_Bi(nn.Module):
    def __init__(self, input_size: int = 32, hidden_size: int = 32) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out


# =============================================================================
# 3. Standard GRU Variants
# =============================================================================
class StandardGRU_Uni(nn.Module):
    def __init__(self, input_size: int = 32, hidden_size: int = 32) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return out


class StandardGRU_Bi(nn.Module):
    def __init__(self, input_size: int = 32, hidden_size: int = 32) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return out


# =============================================================================
# 4. Custom Unrolled LSTM Cell (Decomposed Graph Pattern)
# =============================================================================
class CustomCellLSTM(nn.Module):
    """Explicit unrolled LSTM step-by-step exposing gate math to ONNX."""

    def __init__(self, input_size: int = 32, hidden_size: int = 32) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.x2h = nn.Linear(input_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        c = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(S):
            xt = x[:, t, :]
            gates = self.x2h(xt) + self.h2h(h)
            i_gate, f_gate, g_gate, o_gate = torch.chunk(gates, 4, dim=-1)
            i = torch.sigmoid(i_gate)
            f = torch.sigmoid(f_gate)
            g = torch.tanh(g_gate)
            o = torch.sigmoid(o_gate)
            c = f * c + i * g
            h = o * torch.tanh(c)
            outputs.append(h.unsqueeze(1))
        return torch.cat(outputs, dim=1)


# =============================================================================
# 5. Custom Unrolled GRU Cell (Decomposed Graph Pattern)
# =============================================================================
class CustomCellGRU(nn.Module):
    """Explicit unrolled GRU step-by-step exposing reset/update gates to ONNX."""

    def __init__(self, input_size: int = 32, hidden_size: int = 32) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.x2rz = nn.Linear(input_size, 2 * hidden_size)
        self.h2rz = nn.Linear(hidden_size, 2 * hidden_size)
        self.x2n = nn.Linear(input_size, hidden_size)
        self.h2n = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(S):
            xt = x[:, t, :]
            rz = torch.sigmoid(self.x2rz(xt) + self.h2rz(h))
            r, z = torch.chunk(rz, 2, dim=-1)
            n = torch.tanh(self.x2n(xt) + r * self.h2n(h))
            h = (1.0 - z) * n + z * h
            outputs.append(h.unsqueeze(1))
        return torch.cat(outputs, dim=1)


# =============================================================================
# 6. Quasi-RNN (QRNN - Conv1D + Recurrent Pooling)
# =============================================================================
class QuasiRNN_QRNN(nn.Module):
    """Quasi-RNN: Convolutional feature extraction followed by f-pooling."""

    def __init__(
        self, input_size: int = 32, hidden_size: int = 32, kernel_size: int = 3
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.conv = nn.Conv1d(
            input_size,
            3 * hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D) -> (B, D, S)
        B, S, _ = x.shape
        x_t = x.transpose(1, 2)
        conv_out = self.conv(x_t)[:, :, :S]  # Causal slicing
        z, f, o = torch.chunk(conv_out, 3, dim=1)
        z = torch.tanh(z).transpose(1, 2)
        f = torch.sigmoid(f).transpose(1, 2)
        o = torch.sigmoid(o).transpose(1, 2)

        # F-pooling recurrence
        h = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(S):
            h = f[:, t, :] * h + (1.0 - f[:, t, :]) * z[:, t, :]
            c = o[:, t, :] * h
            outputs.append(c.unsqueeze(1))
        return torch.cat(outputs, dim=1)


# =============================================================================
# 7. Simple Recurrent Unit (SRU)
# =============================================================================
class SRU_Cell(nn.Module):
    """Simple Recurrent Unit with highway skip connections."""

    def __init__(self, input_size: int = 32, hidden_size: int = 32) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_size, 3 * hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        u, f, r = torch.chunk(self.linear(x), 3, dim=-1)
        f = torch.sigmoid(f)
        r = torch.sigmoid(r)

        c = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(S):
            c = f[:, t, :] * c + (1.0 - f[:, t, :]) * u[:, t, :]
            h = r[:, t, :] * torch.tanh(c) + (1.0 - r[:, t, :]) * x[:, t, :]
            outputs.append(h.unsqueeze(1))
        return torch.cat(outputs, dim=1)


# =============================================================================
# Multi-Opset ONNX Export Helper
# =============================================================================
def export_model(
    model: nn.Module,
    dummy_inputs: tuple[torch.Tensor, ...],
    input_names: list[str],
    output_names: list[str],
    base_name: str,
    opset_versions: list[int] = (14, 17, 18, 20),
) -> list[str]:
    model.eval()
    exported = []
    for opset in opset_versions:
        out_path = OUT_DIR / f"{base_name}_opset{opset}.onnx"
        try:
            torch.onnx.export(
                model,
                dummy_inputs,
                str(out_path),
                input_names=input_names,
                output_names=output_names,
                opset_version=opset,
                do_constant_folding=True,
            )
            m_proto = onnx.load(str(out_path))
            onnx.checker.check_model(m_proto)
            size_kb = out_path.stat().st_size / 1024.0
            print(
                f"  [OK] Exported {out_path.name:<45} (opset={opset}, nodes={len(m_proto.graph.node):<3}, {size_kb:.1f} KB)"
            )
            exported.append(str(out_path))
        except Exception as e:
            print(f"  [FAIL] Failed to export {out_path.name} (opset={opset}): {e}")
    return exported


def main() -> None:
    print("==================================================================")
    print("  GENERATING RECURRENT ARCHITECTURE FAMILY ONNX SUITE")
    print("==================================================================")

    dummy_x = torch.randn(1, 8, 32)

    registry = [
        ("vanilla_rnn_single", VanillaRNN_Single(32, 32)),
        ("vanilla_rnn_multi", VanillaRNN_Multi(32, 32, num_layers=2)),
        ("lstm_uni", StandardLSTM_Uni(32, 32)),
        ("lstm_bi", StandardLSTM_Bi(32, 32)),
        ("gru_uni", StandardGRU_Uni(32, 32)),
        ("gru_bi", StandardGRU_Bi(32, 32)),
        ("custom_cell_lstm", CustomCellLSTM(32, 32)),
        ("custom_cell_gru", CustomCellGRU(32, 32)),
        ("qrnn", QuasiRNN_QRNN(32, 32)),
        ("sru_cell", SRU_Cell(32, 32)),
    ]

    for name, module in registry:
        print(f"\nModel: {name}")
        export_model(
            module,
            (dummy_x,),
            ["x"],
            ["out"],
            name,
            opset_versions=[14, 17, 18, 20],
        )

    print("\n==================================================================")
    print(f"Recurrent ONNX suite complete in: {OUT_DIR.resolve()}")
    print("==================================================================")


if __name__ == "__main__":
    main()
