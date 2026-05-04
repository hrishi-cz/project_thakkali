"""
modelss/encoders/text.py

BERT / GPT-2 text encoder producing a fixed 768-dim sequence representation.

Architecture
------------
  transformers.AutoModel.from_pretrained(model_name)
      ↓
  [CLS] token extraction  (position 0 for BERT-style encoder models)
  Last non-pad token      (for GPT-2-style causal decoder models)
      ↓
  [N, 768]  sentence-level embedding

The output dimensionality is always 768.  If the loaded model's
``hidden_size`` differs from 768 (e.g. a distilled / smaller variant),
a ``nn.Linear(hidden_size, 768)`` projection is inserted automatically and
a warning is logged.

Supported models (non-exhaustive)
----------------------------------
  bert-base-uncased        – standard encoder; CLS pooling
  bert-large-uncased       – 1024-dim; projected to 768
  gpt2                     – causal decoder; last-token pooling
  distilbert-base-uncased  – 768-dim; CLS pooling
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

# Architectural constant – fusion layer expects exactly this dim
TEXT_OUTPUT_DIM: int = 768


class TextEncoder(nn.Module):
    """
    HuggingFace transformer encoder with CLS (or last-token) pooling.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.  Default ``"bert-base-uncased"``.
    max_length : int
        Tokeniser truncation length.  Default ``128``.
    freeze_backbone : bool
        Freeze all transformer weights so only the optional projection
        layer is trained.  Default ``False``.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 128,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        self.model_name: str = model_name
        self.max_length: int = max_length

        # ── Transformer backbone ──────────────────────────────────────────
        self.transformer: nn.Module = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # GPT-2 has no pad token by default; reuse eos so that padding works
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if freeze_backbone:
            for param in self.transformer.parameters():
                param.requires_grad = False

        # ── Pooling strategy ──────────────────────────────────────────────
        # Causal models (GPT-2) use the last non-padding token as the
        # sequence representation; encoder models use the [CLS] token.
        self._is_causal: bool = "gpt2" in model_name.lower()
        self.pooling_strategy: str = "last_token" if self._is_causal else "cls"

        # ── Optional projection to enforce TEXT_OUTPUT_DIM = 768 ─────────
        hidden_size: int = self.transformer.config.hidden_size
        if hidden_size != TEXT_OUTPUT_DIM:
            logger.warning(
                "TextEncoder: %s hidden_size=%d != %d — "
                "inserting Linear(%d, %d) projection.",
                model_name, hidden_size, TEXT_OUTPUT_DIM,
                hidden_size, TEXT_OUTPUT_DIM,
            )
            self._projection: Optional[nn.Linear] = nn.Linear(
                hidden_size, TEXT_OUTPUT_DIM
            )
        else:
            self._projection = None

        logger.info(
            "TextEncoder: model=%s  max_length=%d  causal=%s  "
            "freeze=%s  output_dim=%d",
            model_name, max_length, self._is_causal,
            freeze_backbone, TEXT_OUTPUT_DIM,
        )

    def configure(self, plan: Optional[Dict[str, Any]]) -> None:
        """Apply runtime encoder overrides produced by the preprocessing planner."""
        if not isinstance(plan, dict):
            return

        pooling = str(plan.get("pooling", "")).strip().lower()
        if pooling == "auto":
            pooling = "last_token" if self._is_causal else "cls"
        if pooling in {"cls", "last_token", "mean"}:
            self.pooling_strategy = pooling

        max_length = plan.get("max_length")
        if max_length is not None:
            try:
                self.max_length = max(8, int(max_length))
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        texts: List[str],
        return_all_tokens: bool = False,
    ) -> torch.Tensor:
        """
        Tokenise and encode a batch of raw text strings.

        Parameters
        ----------
        texts : List[str]
            Batch of N raw text strings.
        return_all_tokens : bool
            False (default): return pooled sentence embedding ``(N, 768)``.
            True: return full last-hidden-state ``(N, T, hidden_size)`` for
            use by ``UnifiedLatentFusion`` token-sequence mode.

        Returns
        -------
        torch.Tensor
            ``(N, 768)`` when ``return_all_tokens=False``;
            ``(N, T, hidden_size)`` when ``return_all_tokens=True``.
        """
        # Resolve device from transformer parameters (handles CPU/GPU/multi-GPU)
        device: torch.device = next(self.transformer.parameters()).device

        # ── Tokenise ─────────────────────────────────────────────────────
        encoded: Dict[str, torch.Tensor] = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # Move tokeniser outputs to the model device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # ── Transformer forward ──────────────────────────────────────────
        outputs = self.transformer(**encoded)
        last_hidden: torch.Tensor = outputs.last_hidden_state  # (N, seq, hidden)

        # ── Token-sequence mode (for ULA cross-modal fusion) ─────────────
        if return_all_tokens:
            # Return full sequence; projection applied per-token if needed
            if self._projection is not None:
                last_hidden = self._projection(last_hidden)  # (N, T, 768)
            return last_hidden

        # ── Pooling ──────────────────────────────────────────────────────
        if self.pooling_strategy == "last_token":
            # Last non-padding token pooling for causal/decoder-style models.
            attn_mask: torch.Tensor = encoded["attention_mask"]
            seq_lens: torch.Tensor = (attn_mask.sum(dim=1) - 1).clamp(min=0)
            batch_idx = torch.arange(last_hidden.size(0), device=device)
            pooled = last_hidden[batch_idx, seq_lens]
        elif self.pooling_strategy == "mean":
            # Masked mean pooling can improve robustness on long documents.
            attn_mask = encoded["attention_mask"].unsqueeze(-1).float()
            denom = attn_mask.sum(dim=1).clamp_min(1.0)
            pooled = (last_hidden * attn_mask).sum(dim=1) / denom
        else:
            # BERT / encoder-only: [CLS] token always lives at position 0
            pooled = last_hidden[:, 0, :]   # (N, hidden)

        # ── Optional projection ──────────────────────────────────────────
        if self._projection is not None:
            pooled = self._projection(pooled)

        return pooled   # (N, 768)

    def get_output_dim(self) -> int:
        """Return the fixed output dimensionality (768)."""
        return TEXT_OUTPUT_DIM
