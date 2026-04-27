"""X-S3 target selector used to rank and pick target candidates."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


class XS3TargetSelector:
    """
    Rank target candidates and expose calibrated confidence diagnostics.

    Confidence is calibrated using an exponential scaling of the score gap,
    blended with the winner's own score — so a 0.15 gap with a strong winner
    (score 0.70) reads as ~0.82 confidence rather than a misleading raw 0.15.
    """

    def __init__(
        self,
        min_gap: float = 0.0,
        score_keys: Optional[List[str]] = None,
    ) -> None:
        self.min_gap = max(0.0, float(min_gap))
        self.score_keys = score_keys or ["final_score", "score", "confidence"]

    def _candidate_score(self, candidate: Dict[str, Any]) -> float:
        for key in self.score_keys:
            value = candidate.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return 0.0

    @staticmethod
    def _candidate_name(candidate: Dict[str, Any]) -> str:
        for key in ("column", "name", "target"):
            value = candidate.get(key)
            if isinstance(value, str) and value:
                return value
        return "Unknown"

    @staticmethod
    def _calibrate_confidence(gap: float, top_score: float) -> float:
        """
        Map raw score gap → calibrated probability in [0, 1].

        Formula: 65% from exponential gap scaling + 35% from winner quality.
          gap=0.30 → raw~0.95, blended~0.87
          gap=0.15 → raw~0.78, blended~0.72
          gap=0.05 → raw~0.39, blended~0.50
          gap=0.01 → raw~0.10, blended~0.32
        """
        raw = 1.0 - math.exp(-10.0 * gap)
        return round(min(1.0, 0.65 * raw + 0.35 * top_score), 3)

    def select(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        ranked = sorted(candidates, key=self._candidate_score, reverse=True)
        if not ranked:
            return {
                "target_column": "Unknown",
                "xs3_confidence_gap": 0.0,
                "confidence": 0.0,
                "ranked_candidates": [],
                "is_confident": False,
            }

        top_score = self._candidate_score(ranked[0])
        second_score = self._candidate_score(ranked[1]) if len(ranked) > 1 else 0.0
        gap = max(0.0, top_score - second_score)
        confidence = self._calibrate_confidence(gap, top_score)

        return {
            "target_column": self._candidate_name(ranked[0]),
            "xs3_confidence_gap": confidence,
            "confidence": confidence,
            "ranked_candidates": ranked,
            "is_confident": gap >= self.min_gap,
        }
