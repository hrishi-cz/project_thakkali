"""1D CNN encoder for sequential modalities (ECG, EEG, sensor data).

Input:  (batch, seq_len, channels)
Output: (batch, output_dim)

Stacked 1D Conv → BatchNorm → ReLU → MaxPool → GlobalAvgPool → Linear projection.
Works on ECG (1 channel), EEG (N channels), or generic multi-channel sensor arrays.
"""
from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

# Default output dimension (matches tabular encoder default for compatibility)
TIMESERIES_OUTPUT_DIM: int = 64


class TimeSeriesEncoder(nn.Module):
    """
    Stacked 1-D convolutional encoder for time-series / sequential data.

    Parameters
    ----------
    input_channels : int
        Number of input channels per time step (e.g. 1 for ECG lead-II,
        12 for 12-lead ECG, N for EEG channels).
    output_dim : int
        Embedding dimension of the output vector.  Default 64.
    kernel_sizes : tuple of int
        Kernel widths for each conv layer.  Length determines depth.
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_dim: int = TIMESERIES_OUTPUT_DIM,
        kernel_sizes: Tuple[int, ...] = (7, 5, 3),
    ) -> None:
        super().__init__()
        self._output_dim = int(output_dim)

        channel_plan = [input_channels, 32, 64, 128]
        conv_layers: list = []
        for i, ks in enumerate(kernel_sizes):
            in_ch = channel_plan[i]
            out_ch = channel_plan[i + 1]
            conv_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=ks, padding=ks // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
        self.conv_stack = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(channel_plan[-1], self._output_dim)

        logger.info(
            "TimeSeriesEncoder: input_channels=%d  kernel_sizes=%s  output_dim=%d",
            input_channels, kernel_sizes, self._output_dim,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor  shape (batch, seq_len, channels)

        Returns
        -------
        Tensor  shape (batch, output_dim)
        """
        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.conv_stack(x)
        x = self.pool(x).squeeze(-1)        # (batch, 128)
        return self.projection(x)           # (batch, output_dim)

    def get_output_dim(self) -> int:
        return self._output_dim
