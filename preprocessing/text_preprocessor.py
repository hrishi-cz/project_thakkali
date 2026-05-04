"""Text preprocessing – HuggingFace BERT tokeniser (PyTorch DataLoader-safe)."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    BERT-compatible text preprocessor.

    Tokenises raw strings with ``bert-base-uncased`` via HuggingFace
    ``AutoTokenizer``.  Output tensors are squeezed from ``[1, 128]`` to
    ``[128]`` so that ``DataLoader`` batching produces ``[B, 128]`` rather
    than ``[B, 1, 128]`` (which would crash Phase 5 model forward passes).

    Usage
    -----
    >>> tp = TextPreprocessor()
    >>> out = tp("Diagnosis shows severe inflammation.")
    >>> out["input_ids"].shape
    torch.Size([128])
    """

    _PRETRAINED: str = "bert-base-uncased"
    _MAX_LENGTH: int = 128

    def __init__(self) -> None:
        self._tokenizer: Optional[Any] = None  # lazy-loaded on first call
        self._pretrained_model: str = str(self._PRETRAINED)
        self.max_length: int = int(self._MAX_LENGTH)
        self.pooling: str = "cls"

    # ------------------------------------------------------------------
    # Lazy tokeniser property
    # ------------------------------------------------------------------

    @property
    def tokenizer(self) -> Any:
        """Load the BERT tokeniser once; re-use thereafter."""
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self._pretrained_model)
                logger.info("TextPreprocessor: loaded tokeniser '%s'", self._pretrained_model)
            except Exception as exc:
                raise RuntimeError(
                    f"TextPreprocessor: could not load '{self._pretrained_model}'. "
                    "Install transformers: pip install transformers"
                ) from exc
        return self._tokenizer

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def preprocess(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenise *text* and return a dict of fixed-length tensors.

        Returns
        -------
        dict with keys:
          ``input_ids``      : ``torch.LongTensor`` of shape ``[128]``
          ``attention_mask`` : ``torch.LongTensor`` of shape ``[128]``

        The ``.squeeze(0)`` call converts the tokeniser's ``[1, 128]``
        output to ``[128]`` so DataLoader can stack a batch to ``[B, 128]``.
        """
        # Sanitize NaN/None inputs  --  tokenizing "nan"/"None" as words produces
        # misleading embeddings.  Replace with empty string instead.
        if text is None:
            text = ""
        else:
            text = str(text)
            if text.lower() in ("nan", "none", "null", "<na>"):
                text = ""

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),           # [128]
            "attention_mask": encoding["attention_mask"].squeeze(0), # [128]
        }

    def __call__(self, text: str) -> Dict[str, torch.Tensor]:
        """Delegate to :meth:`preprocess` – makes instances callable as transforms."""
        return self.preprocess(text)

    # ------------------------------------------------------------------
    # Config helper (used by /preprocess API endpoint)
    # ------------------------------------------------------------------

    def configure(self, plan: Optional[Dict[str, Any]]) -> None:
        """Apply optional runtime overrides from the preprocessing planner (G16 enhanced)."""
        if not isinstance(plan, dict):
            return

        tokenizer = plan.get("tokenizer")
        if isinstance(tokenizer, str):
            model_name = tokenizer.strip()
            if model_name and model_name != self._pretrained_model:
                self._pretrained_model = model_name
                self._tokenizer = None

        max_length = plan.get("max_length")
        if max_length is not None:
            try:
                self.max_length = max(8, int(max_length))
            except Exception:
                pass

        pooling = str(plan.get("pooling", self.pooling)).strip().lower()
        if pooling in {"cls", "mean", "max", "none"}:
            self.pooling = pooling

        # G16: context-aware adaptation from feature_intelligence signals
        sig = plan.get("feature_intelligence") or {}
        text_sig = sig.get("text") or {}
        avg_tokens = text_sig.get("avg_tokens_per_sample")
        if avg_tokens and avg_tokens > 0:
            self.max_length = max(16, min(512, int(avg_tokens * 1.3)))

        linguistic_complexity = text_sig.get("linguistic_complexity", 0.0)
        if linguistic_complexity > 0.7 and not tokenizer:
            # Switch to multilingual model for complex / multilingual text
            self._pretrained_model = "bert-base-multilingual-cased"
            self._tokenizer = None
            logger.info("TextPreprocessor: switching to bert-base-multilingual-cased (linguistic_complexity=%.3f)", linguistic_complexity)

        # vocab_size: large vocabularies (> 50k unique tokens) indicate rich
        # domain-specific language that benefits from a larger context window.
        # Also guards against over-truncation when the text is lexically diverse.
        vocab_size = text_sig.get("vocab_size") or sig.get("vocab_size")
        if vocab_size and isinstance(vocab_size, (int, float)) and vocab_size > 0:
            if vocab_size > 50_000 and self.max_length < 256:
                self.max_length = min(512, self.max_length * 2)
                logger.info(
                    "TextPreprocessor: vocab_size=%d > 50k — extended max_length to %d",
                    int(vocab_size), self.max_length,
                )
            elif vocab_size < 5_000 and self.max_length > 128:
                # Small vocabulary (e.g. templated text, short labels) — truncate aggressively
                self.max_length = 64
                logger.info(
                    "TextPreprocessor: vocab_size=%d < 5k — truncated max_length to 64 (simple text)",
                    int(vocab_size),
                )

        text_task_type = plan.get("text_task_type") or text_sig.get("text_task_type")
        if text_task_type == "ner_sequence":
            self.pooling = "none"
            self._task_type = "ner"
            logger.info("TextPreprocessor: NER task  --  pooling=none")
        elif text_task_type == "seq2seq":
            self._task_type = "seq2seq"
        else:
            self._task_type = "classification"

        # long_doc_indicator: documents with avg >200 tokens need max-length
        # capped at 512 (BERT limit) and mean pooling (not CLS) so the full
        # document contributes to the representation rather than just the
        # first 512 tokens.
        long_doc = bool(text_sig.get("long_doc_indicator", False))
        if long_doc:
            self.max_length = 512  # saturate at BERT maximum
            if self.pooling not in ("none",):  # don't override NER token-level
                self.pooling = "mean"           # mean over all non-pad tokens
            logger.info(
                "TextPreprocessor: long_doc_indicator=True  --  "
                "max_length=512, pooling=mean for full-document coverage"
            )

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "model": self._pretrained_model,
            "max_length": self.max_length,
            "pooling": self.pooling,
            "padding": "max_length",
            "truncation": True,
            "output_keys": ["input_ids", "attention_mask"],
            "output_shape": f"[{self.max_length}] per key",
        }
