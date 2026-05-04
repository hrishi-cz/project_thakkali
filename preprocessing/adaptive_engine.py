"""Adaptive preprocessing policy builder driven by ExecutionContext intelligence."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


class AdaptivePreprocessingEngine:
    """Compute preprocessing policies from context predictability and data stats."""

    def __init__(self, ctx: Any) -> None:
        self._ctx = ctx

    def _get_effective_predictability(self) -> Dict[str, float]:
        if hasattr(self._ctx, "get_effective_predictability_scores"):
            try:
                scores = self._ctx.get_effective_predictability_scores()
                if isinstance(scores, dict):
                    return {
                        str(k): float(v)
                        for k, v in scores.items()
                        if isinstance(v, (int, float))
                    }
            except Exception:
                pass

        pred_scores = dict(getattr(self._ctx, "predictability_scores", {}) or {})
        drift_scores = dict(getattr(self._ctx, "drift_adjusted_predictability", {}) or {})
        if getattr(self._ctx, "drift_feedback_applied", False) and drift_scores:
            pred_scores = drift_scores
        return {
            str(k): float(v)
            for k, v in pred_scores.items()
            if isinstance(v, (int, float))
        }

    def build_tabular_config(self, X: pd.DataFrame) -> Dict[str, Any]:
        pred_scores = self._get_effective_predictability()
        drifted_features = [str(col) for col in (getattr(self._ctx, "drifted_features", []) or [])]

        tab_values = [
            float(v)
            for k, v in pred_scores.items()
            if "tabular" in str(k).lower() and isinstance(v, (int, float))
        ]
        tab_pred = float(np.mean(tab_values)) if tab_values else 0.5

        stats = self._compute_column_stats(X)

        if stats["missing_rate"] > 0.10:
            imputer = "most_frequent"
        else:
            imputer = "median"

        scaler = "robust" if (stats["outlier_rate"] > 0.05 or drifted_features) else "standard"

        return {
            "imputer_strategy": imputer,
            "near_unique_ratio": 0.75 if tab_pred < 0.30 else 0.50,
            "max_cardinality": 32 if tab_pred < 0.30 else 50,
            "scaler": scaler,
            "drifted_features": list(drifted_features),
            "add_polynomial": bool(tab_pred < 0.30),
            "poly_degree": 2,
            "interaction_only": True,
            "weak_feature_threshold": float(max(0.1, 1.0 - tab_pred)),
            "drop_correlation_threshold": 0.95 if tab_pred > 0.70 else None,
        }

    def get_weak_modalities(self) -> List[str]:
        pred_scores = self._get_effective_predictability()
        return [str(modality) for modality, score in pred_scores.items() if isinstance(score, (int, float)) and float(score) < 0.25]

    def get_modality_predictability(self) -> Dict[str, float]:
        """Return the effective modality predictability scores."""
        return self._get_effective_predictability()

    def build_context_contract(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Build a rich preprocessing snapshot for context persistence."""
        predictability = self.get_modality_predictability()
        weak_modalities = [
            str(modality)
            for modality, score in predictability.items()
            if isinstance(score, (int, float)) and float(score) < 0.25
        ]
        strong_modalities = [
            str(modality)
            for modality, score in predictability.items()
            if isinstance(score, (int, float)) and float(score) > 0.75
        ]
        return {
            "adaptive_tabular_config": self.build_tabular_config(X),
            "fusion_recommendation": self.get_fusion_recommendation(),
            "modality_predictability": predictability,
            "weak_modalities": weak_modalities,
            "strong_modalities": strong_modalities,
            "drifted_features": [
                str(col) for col in (getattr(self._ctx, "drifted_features", []) or [])
            ],
        }

    def get_fusion_recommendation(self) -> str:
        pred_scores = self._get_effective_predictability()
        values = [float(v) for v in pred_scores.values() if isinstance(v, (int, float))]
        if len(values) < 2:
            return "concatenation"
        variance = float(np.var(values))
        if variance > 0.10:
            return "uncertainty"
        if len(values) >= 3:
            return "uncertainty_graph"
        return "attention"

    @staticmethod
    def _compute_column_stats(X: pd.DataFrame) -> Dict[str, float]:
        if X.empty:
            return {"missing_rate": 0.0, "outlier_rate": 0.0}

        missing_rate = float(X.isnull().mean().mean())
        numeric_cols = X.select_dtypes(include=[np.number])
        if numeric_cols.empty:
            return {"missing_rate": missing_rate, "outlier_rate": 0.0}

        q1 = numeric_cols.quantile(0.25)
        q3 = numeric_cols.quantile(0.75)
        iqr = q3 - q1
        outliers = ((numeric_cols < (q1 - 1.5 * iqr)) | (numeric_cols > (q3 + 1.5 * iqr))).mean().mean()
        return {
            "missing_rate": missing_rate,
            "outlier_rate": float(outliers),
        }
