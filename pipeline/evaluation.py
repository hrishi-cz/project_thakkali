"""Evaluation helpers for training and monitoring summaries."""

from __future__ import annotations

from typing import Any, Dict


class EvaluationAdapter:
    """Compute compact quality metrics for model selection and monitoring."""

    @staticmethod
    def _f(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    def evaluate_training(self, training_result: Dict[str, Any], problem_type: str) -> Dict[str, Any]:
        val_loss = max(0.0, self._f(training_result.get("best_val_loss"), 0.0))
        val_acc = max(0.0, min(1.0, self._f(training_result.get("best_val_acc"), 0.0)))
        val_f1 = max(0.0, min(1.0, self._f(training_result.get("best_val_f1"), 0.0)))
        train_acc = max(0.0, min(1.0, self._f(training_result.get("best_train_acc"), 0.0)))

        loss_score = 1.0 / (1.0 + val_loss)
        generalization_gap = max(0.0, train_acc - val_acc)
        stability = max(0.0, 1.0 - generalization_gap)

        if "regression" in str(problem_type):
            performance = loss_score
        else:
            performance = 0.6 * val_acc + 0.4 * val_f1

        overall = 0.5 * performance + 0.3 * loss_score + 0.2 * stability

        return {
            "performance": round(float(performance), 6),
            "loss_score": round(float(loss_score), 6),
            "generalization_gap": round(float(generalization_gap), 6),
            "stability": round(float(stability), 6),
            "overall_score": round(float(overall), 6),
        }

    def evaluate_monitoring(self, drift_result: Dict[str, Any]) -> Dict[str, Any]:
        composite = self._f(drift_result.get("composite_score"), 0.0)
        retrain = bool(drift_result.get("retrain_triggered", False))
        drift_detected = bool(drift_result.get("drift_detected", False))

        risk = min(2.0, max(0.0, composite))
        health = max(0.0, 1.0 - min(1.0, risk))
        return {
            "drift_detected": drift_detected,
            "risk_score": round(float(risk), 6),
            "health_score": round(float(health), 6),
            "retrain_triggered": retrain,
        }

    def evaluate_from_summary(
        self,
        training_summary: Dict[str, Any],
        drift_summary: Dict[str, Any],
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        training_eval = self.evaluate_training(
            {
                "best_val_loss": training_summary.get("best_val_loss"),
                "best_val_acc": training_summary.get("best_val_acc"),
                "best_val_f1": training_summary.get("best_val_f1"),
                "best_train_acc": training_summary.get("best_train_acc"),
            },
            problem_type=(metadata or {}).get("config", {}).get("problem_type", "classification_binary"),
        )
        drift_eval = self.evaluate_monitoring(drift_summary)
        score = 0.7 * training_eval["overall_score"] + 0.3 * drift_eval["health_score"]
        return {
            "training": training_eval,
            "monitoring": drift_eval,
            "combined_score": round(float(score), 6),
        }
