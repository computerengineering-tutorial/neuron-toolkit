#!/usr/bin/env python3
"""Comprehensive Recurrent Neural Network (RNN / LSTM / GRU / QRNN / SRU) Family Generator."""

from __future__ import annotations

import torch
from torch import nn

from generators.decorators import register_architecture
from generators.registry import ModelRegistry


@register_architecture(
    name="vanilla_rnn_single",
    category="recurrent",
    subfolder="recurrent",
    shapes=[(1, 8, 32)],
    input_names=["x"],
    output_names=["out"],
    init_args=(32, 32),
)
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


@register_architecture(
    name="vanilla_rnn_multi",
    category="recurrent",
    subfolder="recurrent",
    shapes=[(1, 8, 32)],
    input_names=["x"],
    output_names=["out"],
    init_args=(32, 32),
    init_kwargs={"num_layers": 2},
)
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


@register_architecture(
    name="lstm_uni",
    category="recurrent",
    subfolder="recurrent",
    shapes=[(1, 8, 32)],
    input_names=["x"],
    output_names=["out"],
    init_args=(32, 32),
)
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


@register_architecture(
    name="lstm_bi",
    category="recurrent",
    subfolder="recurrent",
    shapes=[(1, 8, 32)],
    input_names=["x"],
    output_names=["out"],
    init_args=(32, 32),
)
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


@register_architecture(
    name="gru_uni",
    category="recurrent",
    subfolder="recurrent",
    shapes=[(1, 8, 32)],
    input_names=["x"],
    output_names=["out"],
    init_args=(32, 32),
)
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


@register_architecture(
    name="gru_bi",
    category="recurrent",
    subfolder="recurrent",
    shapes=[(1, 8, 32)],
    input_names=["x"],
    output_names=["out"],
    init_args=(32, 32),
)
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


@register_architecture(
    name="custom_cell_lstm",
    category="recurrent",
    subfolder="recurrent",
    shapes=[(1, 8, 32)],
    input_names=["x"],
    output_names=["out"],
    init_args=(32, 32),
)
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


@register_architecture(
    name="custom_cell_gru",
    category="recurrent",
    subfolder="recurrent",
    shapes=[(1, 8, 32)],
    input_names=["x"],
    output_names=["out"],
    init_args=(32, 32),
)
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


@register_architecture(
    name="qrnn",
    category="recurrent",
    subfolder="recurrent",
    shapes=[(1, 8, 32)],
    input_names=["x"],
    output_names=["out"],
    init_args=(32, 32),
)
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
        B, S, _ = x.shape
        x_t = x.transpose(1, 2)
        conv_out = self.conv(x_t)[:, :, :S]
        z, f, o = torch.chunk(conv_out, 3, dim=1)
        z = torch.tanh(z).transpose(1, 2)
        f = torch.sigmoid(f).transpose(1, 2)
        o = torch.sigmoid(o).transpose(1, 2)

        h = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(S):
            h = f[:, t, :] * h + (1.0 - f[:, t, :]) * z[:, t, :]
            c = o[:, t, :] * h
            outputs.append(c.unsqueeze(1))
        return torch.cat(outputs, dim=1)


@register_architecture(
    name="sru_cell",
    category="recurrent",
    subfolder="recurrent",
    shapes=[(1, 8, 32)],
    input_names=["x"],
    output_names=["out"],
    init_args=(32, 32),
)
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


def main() -> None:
    ModelRegistry.export_all(category="recurrent")


if __name__ == "__main__":
    main()
