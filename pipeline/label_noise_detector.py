"""
Label Noise Detector — pre-training cross-validation based noise detection.

Runs a lightweight 5-fold cross-validation with a fast sklearn RandomForest
on tabular features (and, optionally, hashed text/image path features) to
identify samples where the model consistently disagrees with the given label.
These samples are flagged as *potentially mislabelled* and their loss weight
is reduced so they have less influence on model training.

Usage in training_orchestrator.py (before DataLoader creation):
    from pipeline.label_noise_detector import LabelNoiseDetector
    detector = LabelNoiseDetector(n_folds=5, noise_threshold=0.8)
    result = detector.detect(X_tabular, y, feature_names)
    sample_weights = result["sample_weights"]   # np.ndarray, shape (N,)
    ctx.suspicious_label_indices = result["suspicious_indices"]
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class LabelNoiseDetector:
    """
    Cross-validation label noise detector.

    For each sample, the detector records how often the cross-validation
    ensemble disagrees with its given label.  Samples with disagreement rate
    ≥ ``noise_threshold`` are flagged as suspicious.

    Parameters
    ----------
    n_folds : int
        Number of cross-validation folds (default 5).
    noise_threshold : float
        Fraction of CV folds that must disagree for a sample to be flagged
        (default 0.8 — flagged when ≥ 4/5 folds predict the wrong label).
    min_samples_to_run : int
        Minimum dataset size to run detection.  Skipped on tiny datasets.
    weight_floor : float
        Minimum sample weight assigned to suspicious samples (default 0.2).
        Keeps them in training but heavily down-weighted.
    max_features : int
        Cap on tabular feature count fed to the RF (for speed on wide datasets).
    """

    def __init__(
        self,
        n_folds: int = 5,
        noise_threshold: float = 0.8,
        min_samples_to_run: int = 200,
        weight_floor: float = 0.2,
        max_features: int = 100,
    ) -> None:
        self.n_folds = int(n_folds)
        self.noise_threshold = float(noise_threshold)
        self.min_samples_to_run = int(min_samples_to_run)
        self.weight_floor = float(weight_floor)
        self.max_features = int(max_features)

    def detect(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        problem_type: str = "classification",
    ) -> Dict[str, Any]:
        """
        Run cross-validation noise detection.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix ``(N, F)``.  Must be numeric (pre-processed).
        y : np.ndarray
            Target labels ``(N,)``.
        feature_names : list of str, optional
            Column names for the result report.
        problem_type : str
            ``"classification_binary"`` / ``"classification_multiclass"`` /
            ``"regression"`` — regression returns uniform weights.

        Returns
        -------
        dict with keys:
            ``suspicious_indices`` : list[int]
            ``sample_weights``     : np.ndarray shape (N,), values in [weight_floor, 1.0]
            ``disagreement_rates`` : np.ndarray shape (N,) — fraction of folds that disagreed
            ``n_suspicious``       : int
            ``skipped``            : bool — True when dataset was too small to run
        """
        N = int(X.shape[0])
        _empty = {
            "suspicious_indices": [],
            "sample_weights": np.ones(N, dtype=np.float32),
            "disagreement_rates": np.zeros(N, dtype=np.float32),
            "n_suspicious": 0,
            "skipped": True,
        }

        # Skip for regression or tiny datasets
        if "regression" in str(problem_type).lower():
            logger.debug("LabelNoiseDetector: skipped (regression)")
            return _empty
        if N < self.min_samples_to_run:
            logger.debug(
                "LabelNoiseDetector: skipped (N=%d < min=%d)", N, self.min_samples_to_run
            )
            return _empty

        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import StratifiedKFold
            from sklearn.preprocessing import LabelEncoder
        except ImportError:
            logger.warning("LabelNoiseDetector: sklearn not available — skipping")
            return _empty

        try:
            # Cap feature count for speed
            X_use = X[:, : self.max_features] if X.shape[1] > self.max_features else X
            X_use = np.nan_to_num(X_use, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            # Encode labels to integers
            le = LabelEncoder()
            y_enc = le.fit_transform(y.ravel())

            # Track per-sample disagreement count
            disagree_counts = np.zeros(N, dtype=np.int32)

            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            rf_kwargs: Dict[str, Any] = {
                "n_estimators": 50,
                "max_depth": 8,
                "n_jobs": -1,
                "random_state": 42,
                "class_weight": "balanced",
            }

            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_use, y_enc)):
                try:
                    clf = RandomForestClassifier(**rf_kwargs)
                    clf.fit(X_use[train_idx], y_enc[train_idx])
                    preds = clf.predict(X_use[val_idx])
                    wrong_mask = preds != y_enc[val_idx]
                    disagree_counts[val_idx] += wrong_mask.astype(np.int32)
                    logger.debug(
                        "  Fold %d/%d — val_error=%.3f",
                        fold_idx + 1, self.n_folds,
                        float(wrong_mask.mean()),
                    )
                except Exception as fold_exc:
                    logger.debug("  Fold %d failed: %s", fold_idx + 1, fold_exc)

            disagree_rates = disagree_counts.astype(np.float32) / self.n_folds
            suspicious_mask = disagree_rates >= self.noise_threshold
            suspicious_indices = np.where(suspicious_mask)[0].tolist()

            # Build sample weights: suspicious → weight_floor, clean → 1.0
            sample_weights = np.where(suspicious_mask, self.weight_floor, 1.0).astype(np.float32)

            n_susp = int(suspicious_mask.sum())
            logger.info(
                "LabelNoiseDetector: %d/%d samples flagged as potentially noisy "
                "(threshold=%.0f%%)",
                n_susp, N, self.noise_threshold * 100,
            )

            return {
                "suspicious_indices": suspicious_indices,
                "sample_weights": sample_weights,
                "disagreement_rates": disagree_rates,
                "n_suspicious": n_susp,
                "skipped": False,
            }

        except Exception as exc:
            logger.warning("LabelNoiseDetector.detect() failed: %s", exc)
            return _empty
