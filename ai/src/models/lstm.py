from __future__ import annotations

import torch
from torch import nn


class TemporalLSTMClassifier(nn.Module):
    """LSTM classifier for future binary network-state prediction."""

    def __init__(
        self,
        input_size: int = 21,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_output, _ = self.lstm(x)

        # Use the final timestep representation.
        final_state = sequence_output[:, -1, :]

        return self.classifier(final_state)
