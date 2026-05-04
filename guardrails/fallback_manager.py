"""Fallback policy manager for encoder selection failures."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


ENCODER_FALLBACK_CHAIN: Dict[str, List[str]] = {
    "image": ["ConvNeXt-Tiny", "ResNet50", "MobileNetV3"],
    "text": ["DeBERTa-v3-base", "BERT-base-uncased", "MiniLM-L6-v2"],
    "tabular": ["GRN", "MLP"],
}


class FallbackManager:
    """Track failed encoders and choose the next candidate in chain."""

    def __init__(self) -> None:
        self._failed: Dict[str, Set[str]] = defaultdict(set)

    def get_encoder(self, modality: str, preferred: str) -> str:
        chain = ENCODER_FALLBACK_CHAIN.get(modality, [preferred])
        if preferred in chain:
            start_idx = chain.index(preferred)
            ordered = chain[start_idx:] + chain[:start_idx]
        else:
            ordered = chain

        for candidate in ordered:
            if candidate not in self._failed[modality]:
                return candidate
        return ordered[-1]

    def mark_failed(self, modality: str, encoder: str) -> None:
        self._failed[modality].add(encoder)
        logger.warning(
            "FallbackManager: marked encoder failure modality=%s encoder=%s",
            modality,
            encoder,
        )

    def get_failed(self) -> Dict[str, List[str]]:
        return {modality: sorted(list(names)) for modality, names in self._failed.items()}
