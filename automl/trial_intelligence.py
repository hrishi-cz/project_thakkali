from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


class TrialIntelligence:
    """Simple trial-behavior analyzer for adaptive training policies."""

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    @staticmethod
    def _slope(values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values), dtype=float)
        y = np.asarray(values, dtype=float)
        slope, _ = np.polyfit(x, y, 1)
        return float(slope)

    def analyze(self, train_losses: List[float], val_losses: List[float]) -> Dict[str, Any]:
        train_slope = self._slope(train_losses)
        val_slope = self._slope(val_losses)

        if train_slope < -1e-3 and val_slope > 1e-3:
            fit_type = "overfitting"
        elif abs(train_slope) < 1e-3 and abs(val_slope) < 1e-3:
            fit_type = "underfitting"
        else:
            fit_type = "good"

        return {
            "fit_type": fit_type,
            "train_slope": train_slope,
            "val_slope": val_slope,
            "generalization_gap": float((val_losses[-1] if val_losses else 0.0) - (train_losses[-1] if train_losses else 0.0)),
        }

    def consistency_gate(
        self,
        analysis: Dict[str, Any],
        min_margin: float = 1e-3,
    ) -> Dict[str, Any]:
        """
        Validate whether fit diagnosis is stable enough to drive adaptation.

        Returns a small decision dict used by training code to avoid
        oscillatory updates when train/val trends are nearly flat.
        """
        train_slope = float(analysis.get("train_slope", 0.0) or 0.0)
        val_slope = float(analysis.get("val_slope", 0.0) or 0.0)
        fit_type = str(analysis.get("fit_type", "good"))

        stable = (abs(train_slope) >= min_margin) or (abs(val_slope) >= min_margin)
        if fit_type == "good":
            stable = True

        return {
            "stable": bool(stable),
            "fit_type": fit_type,
            "reason": "stable_trend" if stable else "low_signal",
        }

    def update_memory(self, record: Dict[str, Any]) -> None:
        self.records.append(dict(record))

    def adjust_hyperparams(self, base_params: Dict[str, Any]) -> Dict[str, Any]:
        params = dict(base_params)
        fit_type = self.records[-1].get("fit_type") if self.records else "good"

        if fit_type == "overfitting":
            params["lr"] = float(params.get("lr", 1e-3)) * 0.7
            params["dropout"] = min(0.6, float(params.get("dropout", 0.1)) + 0.05)
            params["weight_decay"] = float(params.get("weight_decay", 1e-5)) * 2.0
            params["epochs"] = max(3, int(params.get("epochs", 10) * 0.9))
        elif fit_type == "underfitting":
            params["lr"] = float(params.get("lr", 1e-3)) * 1.2
            params["dropout"] = max(0.0, float(params.get("dropout", 0.1)) - 0.03)
            params["hidden_dim"] = int(params.get("hidden_dim", 256) * 1.25)
            params["epochs"] = int(params.get("epochs", 10) * 1.2)
        else:
            params["lr"] = float(params.get("lr", 1e-3)) * 0.98

        return params

    def estimate_epochs(self, base: int = 10, fit_type: str = "good", flat_epoch: int = None) -> int:
        """Extended signature for G20: prune-aware epoch capping.

        If flat_epoch is given and fit_type suggests convergence, cap at
        flat_epoch * 1.2 to avoid wasted compute beyond the prune step.
        """
        if flat_epoch is not None and fit_type in {"good", "overfitting"}:
            return max(3, min(base, int(flat_epoch * 1.2)))
        if not self.records:
            return int(base)
        epochs = [int(r.get("epochs", base)) for r in self.records]
        return max(3, int(round(sum(epochs) / max(1, len(epochs)))))

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    def summarize_trials(self, trial_diagnostics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build a stable cross-trial summary for downstream planning/selection.

        The output is intentionally compact and numeric so callers can
        persist it in ``ExecutionContext.training_fit_analysis`` and derive
        deterministic next-run adjustments.
        """
        rows = [dict(row) for row in list(trial_diagnostics or []) if isinstance(row, dict)]
        if not rows:
            return {
                "fit_type": "unknown",
                "num_trials": 0,
                "fit_counts": {"overfitting": 0, "underfitting": 0, "good": 0, "unknown": 0},
                "fit_type_distribution": {
                    "overfitting": 0.0,
                    "underfitting": 0.0,
                    "good": 0.0,
                    "unknown": 1.0,
                },
                "train_slope": 0.0,
                "val_slope": 0.0,
                "generalization_gap": 0.0,
                "calibration_proxy": 0.0,
                "adaptive_penalty": 0.0,
            }

        fit_counts: Dict[str, int] = {
            "overfitting": 0,
            "underfitting": 0,
            "good": 0,
            "unknown": 0,
        }
        train_slopes: List[float] = []
        val_slopes: List[float] = []
        gaps: List[float] = []
        penalties: List[float] = []

        for row in rows:
            fit_type = str(row.get("fit_type", "unknown") or "unknown").lower()
            if fit_type not in fit_counts:
                fit_type = "unknown"
            fit_counts[fit_type] += 1

            if row.get("train_slope") is not None:
                train_slopes.append(self._safe_float(row.get("train_slope")))
            if row.get("val_slope") is not None:
                val_slopes.append(self._safe_float(row.get("val_slope")))
            if row.get("generalization_gap") is not None:
                gaps.append(self._safe_float(row.get("generalization_gap")))
            if row.get("adaptive_penalty") is not None:
                penalties.append(self._safe_float(row.get("adaptive_penalty")))

        n = float(max(1, len(rows)))
        fit_distribution = {
            key: round(float(count) / n, 4)
            for key, count in fit_counts.items()
        }

        dominant_fit = max(
            ("overfitting", "underfitting", "good"),
            key=lambda key: fit_counts[key],
        )
        if fit_counts[dominant_fit] == 0:
            dominant_fit = "unknown"

        avg_train_slope = float(np.mean(train_slopes)) if train_slopes else 0.0
        avg_val_slope = float(np.mean(val_slopes)) if val_slopes else 0.0
        avg_gap = float(np.mean(gaps)) if gaps else 0.0

        # Gap-normalized proxy in [0, 1]: higher means better calibration.
        calibration_proxy = 1.0 - (abs(avg_gap) / (1.0 + abs(avg_gap)))

        if penalties:
            adaptive_penalty = float(np.clip(np.mean(penalties), 0.0, 0.8))
        else:
            slope_divergence = abs(avg_val_slope - avg_train_slope)
            overfit_ratio = fit_distribution.get("overfitting", 0.0)
            adaptive_penalty = float(
                np.clip(
                    0.40 * overfit_ratio + 0.30 * max(0.0, avg_gap) + 0.20 * slope_divergence,
                    0.0,
                    0.8,
                )
            )

        return {
            "fit_type": dominant_fit,
            "num_trials": int(len(rows)),
            "fit_counts": fit_counts,
            "fit_type_distribution": fit_distribution,
            "train_slope": round(avg_train_slope, 6),
            "val_slope": round(avg_val_slope, 6),
            "generalization_gap": round(avg_gap, 6),
            "calibration_proxy": round(float(np.clip(calibration_proxy, 0.0, 1.0)), 6),
            "adaptive_penalty": round(adaptive_penalty, 6),
        }
