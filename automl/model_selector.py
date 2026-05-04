"""
ModelSelector – thin deprecation wrapper around AdvancedModelSelector.

The old ``ModelSelector`` class contained its own encoder catalogue and
selection logic that has been superseded by ``AdvancedModelSelector``.
This module keeps the public surface alive to prevent import errors in
any legacy callers while routing all work to the canonical implementation.

New code should import ``AdvancedModelSelector`` directly::

    from automl.advanced_selector import AdvancedModelSelector
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from automl.advanced_selector import AdvancedModelSelector, SelectionResult

logger = logging.getLogger(__name__)


class ModelSelector:
    """
    Deprecated: delegates entirely to :class:`AdvancedModelSelector`.

    The only method retained for backward compatibility is
    ``recommend_models(problem_type, modalities)``, which is called by
    the ``/select-model`` API endpoint.
    """

    def __init__(self) -> None:
        self._delegate = AdvancedModelSelector()
        logger.debug(
            "ModelSelector: delegating to AdvancedModelSelector "
            "(this class is deprecated)"
        )

    def recommend_models(
        self,
        problem_type: str,
        modalities: List[str],
        dataset_size: int = 10_000,
        avg_tokens: int = 128,
    ) -> List[Dict[str, Any]]:
        """
        Proxy for :meth:`AdvancedModelSelector.recommend_models`.

        Returns a ranked list of model recommendation dicts that conform to
        the Streamlit frontend JSON contract consumed by ``/select-model``.
        """
        return self._delegate.recommend_models(
            problem_type=problem_type,
            modalities=modalities,
            dataset_size=dataset_size,
            avg_tokens=avg_tokens,
        )
