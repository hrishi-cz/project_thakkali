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
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


def _is_hf_cached(model_name: str) -> bool:
    """Return True if model weights exist in the local HuggingFace cache."""
    cache = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    safe = model_name.replace("/", "--")
    return (cache / "hub" / f"models--{safe}").exists()

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
        _cached = _is_hf_cached(model_name)
        # Suppress HF LOAD REPORT and update-check noise during model load.
        # transformers.utils.logging is the canonical path across all versions.
        _restore_verbosity = lambda: None  # noqa: E731
        try:
            from transformers.utils import logging as _hf_log
            _prev_verbosity = _hf_log.get_verbosity()
            _hf_log.set_verbosity_error()
            _restore_verbosity = lambda: _hf_log.set_verbosity(_prev_verbosity)  # noqa: E731
        except Exception:
            pass
        def _load_tokenizer(name: str, **kwargs: object) -> object:
            """Load tokenizer; fallback to slow (use_fast=False) on any fast-tokenizer failure.

            Handles tiktoken/sentencepiece/protobuf incompatibilities generically.
            Re-raises the original exception only when slow tokenizer also fails.
            """
            try:
                return AutoTokenizer.from_pretrained(name, **kwargs)
            except Exception as _e:
                logger.info(
                    "TextEncoder: fast tokenizer failed for '%s' (%s: %s) — "
                    "falling back to slow tokenizer (use_fast=False)",
                    name, type(_e).__name__, _e,
                )
                try:
                    return AutoTokenizer.from_pretrained(
                        name, use_fast=False,
                        **{k: v for k, v in kwargs.items() if k != "use_fast"},
                    )
                except Exception:
                    raise _e  # slow also failed → surface original error

        try:
            try:
                self.transformer: nn.Module = AutoModel.from_pretrained(
                    model_name, local_files_only=_cached
                )
                self.tokenizer = _load_tokenizer(model_name, local_files_only=_cached)
            except OSError:
                self.transformer = AutoModel.from_pretrained(model_name)
                self.tokenizer = _load_tokenizer(model_name)
        finally:
            _restore_verbosity()

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
        texts: Any,
        return_all_tokens: bool = False,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Tokenise and encode a batch of raw text strings.

        Parameters
        ----------
        texts : List[str] | dict | torch.Tensor
            Batch of N raw text strings, a HuggingFace tokenizer dict, or
            pre-tokenised input ids.
        return_all_tokens : bool
            False (default): return pooled sentence embedding ``(N, 768)``.
            True: return full last-hidden-state ``(N, T, hidden_size)`` for
            use by ``UnifiedLatentFusion`` token-sequence mode.
        output_attentions : bool
            True: return the full HuggingFace output object with attention
            tensors for XAI callers.

        Returns
        -------
        torch.Tensor
            ``(N, 768)`` when ``return_all_tokens=False``;
            ``(N, T, hidden_size)`` when ``return_all_tokens=True``.
        """
        # Resolve device from transformer parameters (handles CPU/GPU/multi-GPU)
        device: torch.device = next(self.transformer.parameters()).device

        # ── Tokenise ─────────────────────────────────────────────────────
        if isinstance(texts, dict):
            encoded: Dict[str, torch.Tensor] = {
                k: v
                for k, v in texts.items()
                if isinstance(v, torch.Tensor)
            }
        elif torch.is_tensor(texts):
            input_ids = texts.long()
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = 0
            encoded = {
                "input_ids": input_ids,
                "attention_mask": (input_ids != int(pad_id)).long(),
            }
        else:
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        # Move tokeniser outputs to the model device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # ── Transformer forward ──────────────────────────────────────────
        outputs = self.transformer(
            **encoded,
            output_attentions=bool(output_attentions),
            return_dict=True,
        )
        if output_attentions:
            return outputs
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


# ---------------------------------------------------------------------------
# CLIP Text Encoder (77-token BPE, 512-dim CLIP joint space)
# ---------------------------------------------------------------------------

class CLIPTextEncoder(nn.Module):
    """
    CLIP text encoder producing 512-dim embeddings in the CLIP joint latent space.

    Max context: 77 BPE tokens (hard CLIP limit — do NOT use for long sequences).
    Use only for short meme captions (typically ≤ 40 tokens).

    Encoder family: "clip" — MUST be paired with a CLIP vision encoder.
    Pairing with SigLIP vision encoder will cause latent space misalignment.
    """

    ENCODER_FAMILY: str = "clip"

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.max_length = 77
        self._output_dim = 512

        try:
            from transformers import CLIPTextModel, CLIPTokenizer
            _cached = _is_hf_cached(model_name)
            try:
                self._model = CLIPTextModel.from_pretrained(model_name, local_files_only=_cached)
                self.tokenizer = CLIPTokenizer.from_pretrained(model_name, local_files_only=_cached)
            except OSError:
                self._model = CLIPTextModel.from_pretrained(model_name)
                self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        except Exception as exc:
            logger.warning("CLIPTextEncoder: failed to load '%s': %s", model_name, exc)
            self._model = None
            self.tokenizer = None

        if self._model is not None and freeze_backbone:
            for p in self._model.parameters():
                p.requires_grad_(False)

        logger.info("CLIPTextEncoder: model=%s  freeze=%s  output_dim=%d  max_length=%d",
                    model_name, freeze_backbone, self._output_dim, self.max_length)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> torch.Tensor:
        if self._model is None:
            return torch.zeros(input_ids.shape[0], self._output_dim, device=input_ids.device)
        out = self._model(input_ids=input_ids, attention_mask=attention_mask)
        return out.pooler_output  # (N, 512) — CLS-equivalent in CLIP space

    def get_output_dim(self) -> int:
        return self._output_dim

    def configure(self, plan: Optional[Dict[str, Any]]) -> None:
        pass


# ---------------------------------------------------------------------------
# SigLIP Text Encoder (64-token, 768-dim SigLIP joint space)
# ---------------------------------------------------------------------------

class SigLIPTextEncoder(nn.Module):
    """
    SigLIP text encoder producing 768-dim embeddings in the SigLIP joint latent space.

    Max context: 64 tokens. Trained with per-pair sigmoid loss (not softmax NT-Xent),
    making it more stable at small batch sizes.

    Encoder family: "siglip" — MUST be paired with a SigLIP vision encoder.
    Pairing with CLIP vision encoder will cause latent space misalignment.
    """

    ENCODER_FAMILY: str = "siglip"

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.max_length = 64
        self._output_dim = 768

        try:
            from transformers import SiglipTextModel, AutoTokenizer
            _cached = _is_hf_cached(model_name)
            try:
                self._model = SiglipTextModel.from_pretrained(model_name, local_files_only=_cached)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=_cached)
            except OSError:
                self._model = SiglipTextModel.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Detect actual hidden size
            hidden = getattr(self._model.config, "hidden_size", 768)
            if isinstance(hidden, int) and hidden > 0:
                self._output_dim = hidden
        except Exception as exc:
            logger.warning("SigLIPTextEncoder: failed to load '%s': %s", model_name, exc)
            self._model = None
            self.tokenizer = None

        if self._model is not None and freeze_backbone:
            for p in self._model.parameters():
                p.requires_grad_(False)

        logger.info("SigLIPTextEncoder: model=%s  freeze=%s  output_dim=%d  max_length=%d",
                    model_name, freeze_backbone, self._output_dim, self.max_length)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> torch.Tensor:
        if self._model is None:
            return torch.zeros(input_ids.shape[0], self._output_dim, device=input_ids.device)
        out = self._model(input_ids=input_ids, attention_mask=attention_mask)
        return out.pooler_output  # (N, 768)

    def get_output_dim(self) -> int:
        return self._output_dim

    def configure(self, plan: Optional[Dict[str, Any]]) -> None:
        pass
