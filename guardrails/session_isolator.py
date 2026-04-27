"""Session isolation checks to prevent cross-session dataset leakage."""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class SessionIsolator:
    """Filter dataset hash maps to the set allowed by session context."""

    def validate(
        self,
        session_id: str,
        dataset_hashes: Dict[str, Any],
        ctx: Any,
    ) -> Dict[str, Any]:
        active_ids = set(getattr(ctx, "active_dataset_ids", []) or []) if ctx is not None else set()
        if not active_ids:
            return dict(dataset_hashes or {})

        valid = {
            str(key): value
            for key, value in (dataset_hashes or {}).items()
            if str(key) in active_ids
        }
        leaked = set(str(k) for k in (dataset_hashes or {}).keys()) - active_ids
        if leaked:
            logger.warning(
                "SessionIsolator: rejected %d dataset(s) outside session=%s: %s",
                len(leaked),
                session_id,
                sorted(leaked),
            )
        return valid
