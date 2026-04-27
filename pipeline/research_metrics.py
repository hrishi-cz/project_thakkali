"""Research-oriented summary metrics for trained models."""

from __future__ import annotations

from typing import Any, Dict

from core.types import Phase
from pipeline.evaluation import EvaluationAdapter


class ResearchMetrics:
    """Compute publishable summary metrics from phase outputs."""

    def __init__(self) -> None:
        self._evaluator = EvaluationAdapter()

    @staticmethod
    def _f(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    def compute(self, phase_results: Dict[Any, Dict[str, Any]]) -> Dict[str, Any]:
        training = phase_results.get(Phase.TRAINING, {}) or {}
        drift = phase_results.get(Phase.DRIFT_DETECTION, {}) or {}

        problem_type = str(training.get("problem_type", "classification_binary"))
        train_eval = self._evaluator.evaluate_training(training, problem_type=problem_type)
        monitor_eval = self._evaluator.evaluate_monitoring(drift)

        train_duration = self._f(training.get("duration_seconds"), 0.0)
        efficiency = 1.0 / (1.0 + train_duration / 900.0)

        multimodal_bonus = 0.0
        input_dims = training.get("input_dims", {})
        if isinstance(input_dims, dict) and len(input_dims) >= 2:
            multimodal_bonus = 0.05

        reproducibility = 1.0 if training.get("n_complete", 0) else 0.5
        publishability = (
            0.45 * train_eval.get("overall_score", 0.0)
            + 0.25 * monitor_eval.get("health_score", 0.0)
            + 0.20 * efficiency
            + 0.10 * reproducibility
            + multimodal_bonus
        )

        return {
            "training_quality": train_eval,
            "monitoring_quality": monitor_eval,
            "efficiency_score": round(float(efficiency), 6),
            "reproducibility_score": round(float(reproducibility), 6),
            "publishability_index": round(float(min(1.0, publishability)), 6),
        }
