"""
modelss/encoders/tabular.py

Strict 3-layer MLP tabular encoder: input_dim → 64 → 32 → 16.

Architecture (NO BatchNorm, NO Dropout per specification)
---------------------------------------------------------
  nn.Linear(input_dim, 64) → nn.ReLU()
  nn.Linear(64, 32)        → nn.ReLU()
  nn.Linear(32, 16)
      ↓
  [N, 16]

The fixed 16-dim output is a hard architectural constant.  The fusion layer
depends on receiving exactly 16 dimensions from the tabular branch.

``input_dim`` is the number of columns produced by the upstream
``ColumnTransformer`` (StandardScaler + median imputation).  It must be
set at construction time from the Phase 3 preprocessor's output shape.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Default architectural constant (may be overridden via output_dim parameter)
TABULAR_OUTPUT_DIM: int = 16


class TabularEncoder(nn.Module):
    """
    Strict 3-layer MLP encoder for preprocessed tabular feature vectors.

    Parameters
    ----------
    input_dim : int
        Number of features output by the upstream ``ColumnTransformer``.
    output_dim : int, optional
        Output embedding dimension.  Defaults to 16 (legacy constant).
        Pass ``ctx.encoder_output_dims["tabular"]`` to use schema-driven sizing.
    """

    def __init__(self, input_dim: int, output_dim: int = TABULAR_OUTPUT_DIM) -> None:
        super().__init__()

        self.input_dim: int = input_dim
        self._output_dim: int = int(output_dim)
        self.input_dropout: nn.Module = nn.Identity()

        # ── 3-layer MLP (strict spec – NO BatchNorm, NO Dropout) ─────────
        self.network: nn.Sequential = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self._output_dim),
        )

        logger.info(
            "TabularEncoder: input_dim=%d  topology=%d→64→32→%d  "
            "output_dim=%d",
            input_dim, input_dim, self._output_dim, self._output_dim,
        )

    def configure(self, plan: Optional[Dict[str, Any]]) -> None:
        """Apply optional runtime overrides from preprocessing planner."""
        if not isinstance(plan, dict):
            return
        dropout = plan.get("input_dropout")
        if dropout is None:
            return
        try:
            p = float(dropout)
        except Exception:
            return
        p = max(0.0, min(0.8, p))
        self.input_dropout = nn.Dropout(p) if p > 0 else nn.Identity()

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of tabular feature vectors.

        Parameters
        ----------
        x : torch.Tensor
            Float tensor of shape ``(N, input_dim)`` — scaled tabular
            features produced by the Phase 3 ``ColumnTransformer``.

        Returns
        -------
        torch.Tensor
            Shape ``(N, 16)`` — encoded tabular embeddings.
        """
        return self.network(self.input_dropout(x))

    def get_output_dim(self) -> int:
        """Return the output dimensionality."""
        return self._output_dim


class GRNTabularEncoder(nn.Module):
    """
    Gated Residual Network encoder for preprocessed tabular feature vectors.

    Architecture (adapted from Temporal Fusion Transformers)::

        FC1:  Linear(input_dim, hidden_dim) -> ELU
        FC2:  Linear(hidden_dim, hidden_dim)
        Gate: Sigmoid(Linear(input_dim, hidden_dim))
        Skip: Linear(input_dim, hidden_dim)
        Out:  LayerNorm(gate * FC2 + skip) -> Linear(hidden_dim, 16)

    Parameters
    ----------
    input_dim : int
        Number of features from upstream ``ColumnTransformer``.
    hidden_dim : int
        Width of GRN hidden layers.  Default 64.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = TABULAR_OUTPUT_DIM,
    ) -> None:
        super().__init__()
        self.input_dim: int = input_dim
        self._output_dim: int = int(output_dim)
        self.input_dropout: nn.Module = nn.Identity()

        # Core transformation
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Gating mechanism
        self.gate_linear = nn.Linear(input_dim, hidden_dim)
        self.gate_activation = nn.Sigmoid()

        # Skip (residual) connection — projects input to hidden_dim
        self.skip = nn.Linear(input_dim, hidden_dim)

        # Normalization + final projection
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, self._output_dim)

        logger.info(
            "GRNTabularEncoder: input_dim=%d  hidden_dim=%d  output_dim=%d",
            input_dim, hidden_dim, self._output_dim,
        )

    def configure(self, plan: Optional[Dict[str, Any]]) -> None:
        """Apply optional runtime overrides from preprocessing planner."""
        if not isinstance(plan, dict):
            return
        dropout = plan.get("input_dropout")
        if dropout is None:
            return
        try:
            p = float(dropout)
        except Exception:
            return
        p = max(0.0, min(0.8, p))
        self.input_dropout = nn.Dropout(p) if p > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of tabular feature vectors.

        Parameters
        ----------
        x : torch.Tensor
            Float tensor of shape ``(N, input_dim)``.

        Returns
        -------
        torch.Tensor
            Shape ``(N, 16)`` — encoded tabular embeddings.
        """
        x = self.input_dropout(x)
        h = self.elu(self.fc1(x))
        h = self.fc2(h)

        gate = self.gate_activation(self.gate_linear(x))
        h = gate * h

        skip = self.skip(x)
        h = self.layer_norm(h + skip)

        return self.output_projection(h)

    def get_output_dim(self) -> int:
        """Return the output dimensionality."""
        return self._output_dim


# ---------------------------------------------------------------------------
# FTTransformerEncoder — Gorishniy et al., NeurIPS 2021
# "Revisiting Deep Learning Models for Tabular Data"
# ---------------------------------------------------------------------------

class FTTransformerEncoder(nn.Module):
    """
    Feature Tokenizer + Transformer tabular encoder.

    Algorithm (Gorishniy et al., NeurIPS 2021):
      1. Feature Tokenization: each of the n_features scalar values is
         projected to a d-dim token via a per-feature affine transform:
           token_j = x_j * W_j + b_j   (W_j, b_j ∈ R^d)
         A learnable [CLS] token is prepended, giving shape (N, n+1, d).
      2. L Transformer encoder layers with pre-LN (LayerNorm → Attention
         → residual, LayerNorm → FFN → residual).
      3. The [CLS] token at the last layer is extracted and projected to
         output_dim via a two-layer MLP with GELU activation.

    Outperforms MLP and GBM on many tabular benchmarks.  Especially strong
    on datasets with many features and complex non-linear interactions.

    Parameters
    ----------
    input_dim : int
        Number of preprocessed tabular features.
    output_dim : int
        Output embedding dimension (default 64 — richer than MLP's 16).
    d_token : int
        Token embedding dimension inside the Transformer.  Default 96.
    n_layers : int
        Number of Transformer encoder layers.  Default 3.
    n_heads : int
        Number of attention heads.  Default 8.
    ffn_factor : float
        FFN hidden dim = round(d_token * ffn_factor).  Default 1.33.
    dropout : float
        Dropout rate inside Transformer and final MLP.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 64,
        d_token: int = 96,
        n_layers: int = 3,
        n_heads: int = 8,
        ffn_factor: float = 1.33,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self._output_dim = int(output_dim)
        self.d_token = d_token

        # Per-feature affine tokenizer: W ∈ R^{n, d}, b ∈ R^{n, d}
        self.feature_weights = nn.Parameter(torch.empty(input_dim, d_token))
        self.feature_biases  = nn.Parameter(torch.zeros(input_dim, d_token))
        nn.init.xavier_uniform_(self.feature_weights)

        # Learnable [CLS] token prepended before each sample's feature tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)

        # Transformer encoder (pre-LN norm_first=True matches the paper)
        ffn_dim = max(d_token, int(round(d_token * ffn_factor)))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Final MLP: [CLS] token → output_dim
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, output_dim),
        )
        self.input_dropout: nn.Module = nn.Identity()

        logger.info(
            "FTTransformerEncoder [NeurIPS 2021]: input_dim=%d  d_token=%d  "
            "n_layers=%d  n_heads=%d  output_dim=%d",
            input_dim, d_token, n_layers, n_heads, output_dim,
        )

    def configure(self, plan: Optional[Dict[str, Any]]) -> None:
        if not isinstance(plan, dict):
            return
        dropout = plan.get("input_dropout")
        if dropout is not None:
            try:
                p = float(max(0.0, min(0.8, float(dropout))))
                self.input_dropout = nn.Dropout(p) if p > 0 else nn.Identity()
            except Exception:
                pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, input_dim)  — preprocessed tabular features

        Returns
        -------
        (N, output_dim)  — rich tabular embeddings
        """
        x = self.input_dropout(x)
        # Tokenize: x_j * W_j + b_j  → (N, F, d_token)
        tokens = x.unsqueeze(-1) * self.feature_weights.unsqueeze(0) + self.feature_biases.unsqueeze(0)
        # Prepend CLS token → (N, F+1, d_token)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        out = self.transformer(tokens)       # (N, F+1, d_token)
        cls_out = out[:, 0]                  # (N, d_token)
        return self.head(cls_out)            # (N, output_dim)

    def get_output_dim(self) -> int:
        return self._output_dim
