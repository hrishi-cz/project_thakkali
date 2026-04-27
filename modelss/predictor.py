"""Tombstone for the retired multimodal predictor implementation.

`MultimodalPredictor` was never part of the active training/inference path.
The canonical runtime path is `automl.trainer._MultimodalHead`.
"""

from __future__ import annotations

from typing import Any


class MultimodalPredictor:
    """Deprecated import-compatibility stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "MultimodalPredictor has been retired. "
            "Use automl.trainer._MultimodalHead instead."
        )
