"""
Post-hoc probability calibration utilities for classification models.

Calibration is learned on held-out validation logits in Phase 5 and persisted
as an artifact for inference-time probability correction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    denom = np.sum(exp_x, axis=1, keepdims=True)
    denom = np.where(denom <= 0.0, 1.0, denom)
    return exp_x / denom


def _clip_probs(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)


def _expected_calibration_error(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute ECE from confidence and correctness vectors."""
    conf = np.asarray(confidences, dtype=np.float64).ravel()
    corr = np.asarray(correctness, dtype=np.float64).ravel()
    if conf.size == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = float(conf.size)
    for idx in range(n_bins):
        left, right = bins[idx], bins[idx + 1]
        if idx == n_bins - 1:
            mask = (conf >= left) & (conf <= right)
        else:
            mask = (conf >= left) & (conf < right)
        if not np.any(mask):
            continue
        conf_bin = conf[mask]
        corr_bin = corr[mask]
        ece += (conf_bin.size / n) * abs(float(conf_bin.mean()) - float(corr_bin.mean()))
    return float(ece)


def _binary_metrics(probs: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    probs = _clip_probs(np.asarray(probs).reshape(-1))
    y = np.asarray(y_true).reshape(-1).astype(int)
    correctness = (probs >= 0.5).astype(int) == y
    try:
        nll = float(log_loss(y, np.vstack([1.0 - probs, probs]).T, labels=[0, 1]))
    except Exception:
        nll = float("nan")
    try:
        brier = float(brier_score_loss(y, probs))
    except Exception:
        brier = float("nan")
    ece = _expected_calibration_error(probs, correctness.astype(float))
    return {"nll": nll, "brier": brier, "ece": ece}


def _multiclass_metrics(probs: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    probs = _clip_probs(np.asarray(probs))
    y = np.asarray(y_true).reshape(-1).astype(int)
    if probs.ndim != 2 or probs.shape[0] == 0:
        return {"nll": float("nan"), "brier": float("nan"), "ece": float("nan")}

    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    correctness = (preds == y).astype(float)

    try:
        nll = float(log_loss(y, probs, labels=list(range(probs.shape[1]))))
    except Exception:
        nll = float("nan")

    try:
        one_hot = np.eye(probs.shape[1], dtype=np.float64)[y]
        brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    except Exception:
        brier = float("nan")

    ece = _expected_calibration_error(confs, correctness)
    return {"nll": nll, "brier": brier, "ece": ece}


def _multilabel_metrics(probs: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    probs = _clip_probs(np.asarray(probs))
    y = np.asarray(y_true)
    if probs.ndim != 2 or y.ndim != 2 or probs.shape != y.shape:
        return {"ece": float("nan")}

    preds = (probs >= 0.5).astype(int)
    correctness = (preds == y.astype(int)).astype(float)
    ece = _expected_calibration_error(probs.ravel(), correctness.ravel())
    return {"ece": ece}


@dataclass
class ProbabilityCalibrator:
    """Serializable post-hoc calibrator for classification probabilities."""

    min_samples: int = 50
    max_temperature: float = 5.0
    clip_eps: float = 1e-6

    fitted: bool = False
    mode: str = "identity"
    problem_type: str = ""
    n_classes: int = 0

    temperature: float = 1.0
    binary_isotonic: Optional[IsotonicRegression] = None
    multilabel_isotonic: List[Optional[IsotonicRegression]] = field(default_factory=list)

    fit_report: Dict[str, Any] = field(default_factory=dict)

    def fit(
        self,
        logits: np.ndarray,
        targets: np.ndarray,
        *,
        problem_type: str,
        execution_context: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Fit calibration mapping from held-out logits and targets."""
        self.problem_type = str(problem_type)

        logits_np = np.asarray(logits)
        targets_np = np.asarray(targets)

        report: Dict[str, Any]
        if self.problem_type == "classification_binary":
            report = self._fit_binary(logits_np, targets_np)
        elif self.problem_type.startswith("classification"):
            report = self._fit_multiclass(logits_np, targets_np)
        elif self.problem_type == "multilabel_classification":
            report = self._fit_multilabel(logits_np, targets_np)
        else:
            self.fitted = False
            self.mode = "identity"
            self.fit_report = {
                "enabled": False,
                "mode": "identity",
                "reason": f"Unsupported problem type: {self.problem_type}",
            }
            report = dict(self.fit_report)

        self._log_to_execution_context(execution_context, report)
        return report

    def _log_to_execution_context(
        self,
        execution_context: Optional[Any],
        report: Dict[str, Any],
    ) -> None:
        if execution_context is None or not hasattr(execution_context, "log_decision"):
            return

        try:
            mode = str(report.get("mode", self.mode) or "identity")
            enabled = bool(report.get("enabled", False))
            metrics_after = report.get("metrics_after", {})
            nll = None
            brier = None
            ece = None
            if isinstance(metrics_after, dict):
                nll = metrics_after.get("nll")
                brier = metrics_after.get("brier")
                ece = metrics_after.get("ece")

            execution_context.log_decision(
                "calibration",
                (
                    "Calibration "
                    f"{'enabled' if enabled else 'skipped'}: mode={mode}"
                ),
                evidence=(
                    f"nll={nll}; brier={brier}; ece={ece}; "
                    f"problem_type={self.problem_type}"
                ),
            )
        except Exception:
            return

    def calibrate(self, logits: np.ndarray, probs: np.ndarray) -> np.ndarray:
        """Apply learned calibration and return calibrated probabilities."""
        probs_np = np.asarray(probs, dtype=np.float64)
        logits_np = np.asarray(logits, dtype=np.float64)

        if not self.fitted:
            return _clip_probs(probs_np, self.clip_eps)

        if self.mode == "isotonic_binary" and self.binary_isotonic is not None:
            flat = probs_np.reshape(-1)
            calibrated = self.binary_isotonic.transform(flat)
            return _clip_probs(calibrated.reshape(probs_np.shape), self.clip_eps)

        if self.mode == "temperature_multiclass":
            if logits_np.ndim != 2:
                return _clip_probs(probs_np, self.clip_eps)
            calibrated = _softmax(logits_np / max(self.temperature, self.clip_eps))
            return _clip_probs(calibrated, self.clip_eps)

        if self.mode == "isotonic_multilabel" and self.multilabel_isotonic:
            calibrated = np.asarray(probs_np, dtype=np.float64).copy()
            if calibrated.ndim != 2:
                return _clip_probs(calibrated, self.clip_eps)
            max_c = min(calibrated.shape[1], len(self.multilabel_isotonic))
            for col in range(max_c):
                iso = self.multilabel_isotonic[col]
                if iso is None:
                    continue
                calibrated[:, col] = iso.transform(calibrated[:, col])
            return _clip_probs(calibrated, self.clip_eps)

        return _clip_probs(probs_np, self.clip_eps)

    def _fit_binary(self, logits: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits[:, 0]
        y = targets.reshape(-1)
        try:
            y = y.astype(int)
        except Exception:
            y = (y > 0).astype(int)

        raw_probs = _sigmoid(logits.reshape(-1))
        before = _binary_metrics(raw_probs, y)

        unique = np.unique(y)
        if y.size < self.min_samples or unique.size < 2:
            self.fitted = False
            self.mode = "identity"
            self.fit_report = {
                "enabled": False,
                "mode": "identity",
                "problem_type": self.problem_type,
                "n_samples": int(y.size),
                "reason": "Not enough samples or class diversity for binary calibration",
                "metrics_before": before,
                "metrics_after": before,
            }
            return dict(self.fit_report)

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_probs, y)
        calibrated = _clip_probs(iso.transform(raw_probs), self.clip_eps)
        after = _binary_metrics(calibrated, y)

        self.binary_isotonic = iso
        self.fitted = True
        self.mode = "isotonic_binary"
        self.n_classes = 2
        self.fit_report = {
            "enabled": True,
            "mode": self.mode,
            "problem_type": self.problem_type,
            "n_samples": int(y.size),
            "metrics_before": before,
            "metrics_after": after,
        }
        return dict(self.fit_report)

    def _fit_multiclass(self, logits: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
        if logits.ndim != 2:
            self.fitted = False
            self.mode = "identity"
            self.fit_report = {
                "enabled": False,
                "mode": "identity",
                "problem_type": self.problem_type,
                "reason": "Multiclass calibration expects 2D logits",
            }
            return dict(self.fit_report)

        y = targets.reshape(-1).astype(int)
        raw_probs = _softmax(logits)
        before = _multiclass_metrics(raw_probs, y)

        n_samples, n_classes = logits.shape
        self.n_classes = int(n_classes)

        if n_samples < self.min_samples or n_classes < 2:
            self.fitted = False
            self.mode = "identity"
            self.fit_report = {
                "enabled": False,
                "mode": "identity",
                "problem_type": self.problem_type,
                "n_samples": int(n_samples),
                "n_classes": int(n_classes),
                "reason": "Not enough samples for temperature scaling",
                "metrics_before": before,
                "metrics_after": before,
            }
            return dict(self.fit_report)

        temperatures = np.linspace(0.5, float(self.max_temperature), 40)
        best_t = 1.0
        best_loss = float("inf")

        for t in temperatures:
            probs_t = _softmax(logits / t)
            try:
                loss_t = float(log_loss(y, probs_t, labels=list(range(n_classes))))
            except Exception:
                continue
            if loss_t < best_loss:
                best_loss = loss_t
                best_t = float(t)

        self.temperature = best_t
        calibrated = _softmax(logits / max(best_t, self.clip_eps))
        after = _multiclass_metrics(calibrated, y)

        self.fitted = True
        self.mode = "temperature_multiclass"
        self.fit_report = {
            "enabled": True,
            "mode": self.mode,
            "problem_type": self.problem_type,
            "n_samples": int(n_samples),
            "n_classes": int(n_classes),
            "temperature": float(best_t),
            "metrics_before": before,
            "metrics_after": after,
        }
        return dict(self.fit_report)

    def _fit_multilabel(self, logits: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
        probs = _sigmoid(logits)
        y = np.asarray(targets, dtype=np.float64)
        if probs.ndim != 2 or y.ndim != 2 or probs.shape != y.shape:
            self.fitted = False
            self.mode = "identity"
            self.fit_report = {
                "enabled": False,
                "mode": "identity",
                "problem_type": self.problem_type,
                "reason": "Multilabel calibration expects matching 2D logits/targets",
            }
            return dict(self.fit_report)

        before = _multilabel_metrics(probs, y)
        n_samples, n_classes = probs.shape
        self.n_classes = int(n_classes)

        fitted_count = 0
        calibrators: List[Optional[IsotonicRegression]] = []
        calibrated = np.asarray(probs, dtype=np.float64).copy()

        for col in range(n_classes):
            y_col = y[:, col].astype(int)
            p_col = probs[:, col]
            if n_samples < self.min_samples or np.unique(y_col).size < 2:
                calibrators.append(None)
                continue

            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p_col, y_col)
            calibrated[:, col] = iso.transform(p_col)
            calibrators.append(iso)
            fitted_count += 1

        self.multilabel_isotonic = calibrators
        self.fitted = fitted_count > 0
        self.mode = "isotonic_multilabel" if self.fitted else "identity"

        after = _multilabel_metrics(calibrated, y)
        self.fit_report = {
            "enabled": bool(self.fitted),
            "mode": self.mode,
            "problem_type": self.problem_type,
            "n_samples": int(n_samples),
            "n_classes": int(n_classes),
            "fitted_classes": int(fitted_count),
            "metrics_before": before,
            "metrics_after": after,
        }
        return dict(self.fit_report)


# ---------------------------------------------------------------------------
# ConformalCalibrator — split conformal prediction sets / intervals
# ---------------------------------------------------------------------------

class ConformalCalibrator:
    """
    Split conformal prediction (Angelopoulos & Bates, 2022).

    Provides coverage guarantees:
        P(y ∈ C(x)) ≥ 1 - alpha    for classification (prediction sets)
        P(y ∈ [ŷ - q, ŷ + q]) ≥ 1 - alpha  for regression (intervals)

    Usage
    -----
    >>> cal = ConformalCalibrator(alpha=0.1, problem_type="classification_multiclass")
    >>> cal.calibrate(cal_logits, cal_labels)          # fit on calibration set
    >>> sets = cal.predict_set(test_logits)             # prediction sets
    >>> assert cal.coverage >= 0.90                     # guaranteed coverage

    References
    ----------
    Angelopoulos & Bates. "A Gentle Introduction to Conformal Prediction."
    arXiv 2022.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        problem_type: str = "classification_multiclass",
    ) -> None:
        self.alpha = float(np.clip(alpha, 0.01, 0.49))
        self.problem_type = problem_type
        self.q_hat: Optional[float] = None
        self.coverage: float = 0.0
        self.fitted: bool = False

    def calibrate(
        self,
        cal_logits: np.ndarray,
        cal_labels: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compute conformal quantile from a calibration set.

        Parameters
        ----------
        cal_logits : np.ndarray  ``(N_cal, C)`` or ``(N_cal,)``
        cal_labels : np.ndarray  ``(N_cal,)``
        """
        cal_logits = np.asarray(cal_logits, dtype=np.float64)
        cal_labels = np.asarray(cal_labels)
        n = len(cal_labels)

        if self.problem_type == "regression":
            # Nonconformity score: |y - ŷ|
            preds = cal_logits.ravel()
            scores = np.abs(cal_labels.astype(float) - preds)
        elif self.problem_type == "classification_binary":
            probs = _sigmoid(cal_logits.ravel())
            scores = 1.0 - np.where(cal_labels == 1, probs, 1.0 - probs)
        else:
            # Multiclass: 1 - softmax probability of the true class
            probs = _softmax(cal_logits if cal_logits.ndim == 2
                             else cal_logits.reshape(-1, 1))
            true_probs = probs[np.arange(n), cal_labels.astype(int)]
            scores = 1.0 - true_probs

        # Finite-sample corrected quantile (Vovk 2005)
        level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.q_hat = float(np.quantile(scores, level))
        self.fitted = True

        # Estimate empirical coverage
        if self.problem_type == "regression":
            preds = cal_logits.ravel()
            self.coverage = float(np.mean(np.abs(cal_labels.astype(float) - preds) <= self.q_hat))
        else:
            self.coverage = float(np.mean(scores <= self.q_hat))

        return {
            "q_hat": self.q_hat,
            "alpha": self.alpha,
            "empirical_coverage": self.coverage,
            "n_calibration": n,
        }

    def predict_set(
        self,
        logits: np.ndarray,
    ) -> List[Any]:
        """
        Return prediction sets (classification) or intervals (regression).

        For classification: returns list of label arrays where every class
        whose nonconformity score ≤ q_hat is included.
        For regression: returns list of (lower, upper) tuples.
        """
        if not self.fitted or self.q_hat is None:
            raise RuntimeError("Call calibrate() before predict_set().")

        logits = np.asarray(logits, dtype=np.float64)

        if self.problem_type == "regression":
            preds = logits.ravel()
            return [(float(p - self.q_hat), float(p + self.q_hat)) for p in preds]

        if self.problem_type == "classification_binary":
            probs = _sigmoid(logits.ravel())
            return [
                [1] if (1 - p) <= self.q_hat else (
                    [0] if p <= self.q_hat else [0, 1]
                )
                for p in probs
            ]

        # Multiclass
        probs = _softmax(logits if logits.ndim == 2 else logits.reshape(-1, 1))
        return [
            [c for c in range(probs.shape[1]) if (1.0 - probs[i, c]) <= self.q_hat]
            for i in range(len(probs))
        ]
