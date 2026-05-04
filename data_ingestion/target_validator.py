"""
Universal Target Validator: Learning-based predictability scoring for all modalities.

PURPOSE (FIX-4 Part 2):
  Replace heuristics (SIFT for images, TF-IDF for text) with unified
  RF-based predictability validation.

METHODOLOGY:
  For each modality:
    1. Predictability: Random Forest 3-fold CV accuracy
    2. Complementarity: How much unique info (vs other modalities)
    3. Degeneracy: Avoid constant features
    4. Noise Robustness: Stability under small perturbations
    5. Feature Importance: Top k% of features matter
  
  Final Score = 0.40×pred + 0.20×comp + 0.15×degen + 0.15×noise + 0.10×imp

EXPECTED IMPROVEMENT:
  Before: tabular 85-90%, image 60%, text 60%
  After:  tabular 85-90%, image 85-95%, text 85-95%

INTEGRATION:
  Called by: data_ingestion/schema_detector.py
  Replaces: SIFT and TF-IDF heuristics
  Output: predictability_scores Dict[str, float] (0-1)

RESEARCH CITATIONS:
  - RandomForest importance: Breiman et al. 2001
  - Cross-validation: Kfold stability
  - Complementarity: Measure via negative cosine similarity
  - Robustness: Add noise, measure metric variance
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class UniversalTargetValidator:
    """
    Learn-based predictability validator for all modalities.
    
    Uses Random Forest 3-fold cross-validation on embeddings to score
    how well each modality can predict the target (in isolation).
    
    Attributes
    ----------
    cv_folds : int
        Number of cross-validation folds (default 3)
    max_features : int
        Max features for RF (default 50)
    criterion : str
        "squared_error" (regression) or "gini" (classification)
    """
    
    def __init__(
        self,
        cv_folds: int = 3,
        max_features: int = 50,
        criterion: str = "squared_error",
    ):
        """
        Initialize validator.
        
        Parameters
        ----------
        cv_folds : int
            Cross-validation folds for scoring (3 is default for small data).
        max_features : int
            Max features per tree (larger = more complex).
        criterion : str
            "squared_error" (regression) or "gini" (classification).
        """
        self.cv_folds = cv_folds
        self.max_features = max_features
        self.criterion = criterion
        logger.info(
            "UniversalTargetValidator initialized: "
            "cv=%d, max_features=%d, criterion=%s",
            cv_folds, max_features, criterion,
        )
    
    def _predict(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str = "regression",
    ) -> float:
        """
        Compute Random Forest 3-fold CV accuracy.
        
        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            Embeddings (tabular, text, or image).
        y : np.ndarray, shape (N,)
            Target values.
        task_type : str
            "regression" or "classification"
        
        Returns
        -------
        float
            Mean CV accuracy (0-1).
        
        Raises
        ------
        ValueError
            If X or y invalid.
        """
        if X.shape[0] < self.cv_folds:
            logger.warning(
                "_predict: only %d samples, need ≥%d for CV",
                X.shape[0], self.cv_folds,
            )
            return 0.0
        
        try:
            if task_type == "regression":
                model = RandomForestRegressor(
                    n_estimators=50,
                    max_features=min(self.max_features, X.shape[1]),
                    max_depth=10,
                    random_state=42,
                )
                scores = cross_val_score(
                    model, X, y, cv=self.cv_folds, scoring="r2"
                )
                # R² can be negative, clamp to [0, 1]
                score = max(0.0, np.mean(scores))
            else:  # classification
                model = RandomForestClassifier(
                    n_estimators=50,
                    max_features=min(self.max_features, X.shape[1]),
                    max_depth=10,
                    random_state=42,
                )
                scores = cross_val_score(
                    model, X, y, cv=self.cv_folds, scoring="accuracy"
                )
                score = np.mean(scores)
            
            return float(score)
        
        except Exception as e:
            logger.error("_predict failed: %s", e)
            return 0.0
    
    def _check_complementarity(
        self,
        X: np.ndarray,
        X_other: Optional[np.ndarray] = None,
    ) -> float:
        """
        Measure complementarity: how unique is this modality vs others?
        
        If X_other provided, use negative cosine similarity.
        If not, use eigenvalue analysis (PCA variance).
        
        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            This modality's embeddings.
        X_other : Optional[np.ndarray]
            Other modalities' embeddings (stacked).
        
        Returns
        -------
        float
            Complementarity score (0-1), higher = more unique.
        """
        try:
            if X_other is None:
                # Use variance of principal components (PCA-style)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # SVD
                _, s, _ = np.linalg.svd(X_scaled, full_matrices=False)
                # Variance explained by first k components
                total_var = np.sum(s ** 2)
                concentrated = total_var / (X.shape[0] - 1)
                
                # High variance = uninformative constant features = low complementarity
                # Low variance = informative features = high complementarity
                # Invert: 1 - normalized_concentration
                score = 1.0 - min(concentrated / max(1.0, X.shape[1]), 1.0)
            else:
                # Negative mean cosine similarity with other modalities
                # Normalize
                X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
                other_norm = X_other / (np.linalg.norm(X_other, axis=1, keepdims=True) + 1e-8)
                
                # Pairwise similarity (max pooling across "other" modalities)
                sims = np.abs((X_norm @ other_norm.T).max(axis=1))  # (N,)
                mean_sim = float(np.mean(sims))
                
                # High similarity = redundant = low complementarity
                # Low similarity = unique = high complementarity
                score = 1.0 - mean_sim
            
            return float(np.clip(score, 0.0, 1.0))
        
        except Exception as e:
            logger.debug("_check_complementarity failed: %s", e)
            return 0.5  # Neutral default
    
    def _check_degeneracy(self, X: np.ndarray) -> float:
        """
        Detect degenerate features (all zeros, all constant).
        
        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            Embeddings.
        
        Returns
        -------
        float
            Non-degeneracy score (0-1), 1 = no degeneracy.
        """
        try:
            # Count non-zero, non-constant features
            non_zero_cols = np.sum((X != 0).any(axis=0))  # ≠ 0 in at least one row
            non_constant_cols = np.sum(X.std(axis=0) > 1e-8)  # Non-zero variance
            
            # Fraction of good features
            good_fraction = non_constant_cols / max(1, X.shape[1])
            
            return float(np.clip(good_fraction, 0.0, 1.0))
        
        except Exception as e:
            logger.debug("_check_degeneracy failed: %s", e)
            return 1.0  # Assume OK by default
    
    def _check_noise_robustness(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str = "regression",
        noise_level: float = 0.1,
    ) -> float:
        """
        Measure robustness: do predictions stay stable with noisy inputs?
        
        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            Embeddings.
        y : np.ndarray, shape (N,)
            Targets.
        task_type : str
            "regression" or "classification"
        noise_level : float
            Gaussian noise std (relative to data range).
        
        Returns
        -------
        float
            Robustness score (0-1), 1 = very stable.
        """
        try:
            # Clean baseline
            score_clean = self._predict(X, y, task_type=task_type)
            
            # Add noise
            X_range = np.ptp(X, axis=0)  # Peak-to-peak
            noise = np.random.randn(*X.shape) * X_range * noise_level
            X_noisy = X + noise
            
            # Score with noise
            score_noisy = self._predict(X_noisy, y, task_type=task_type)
            
            # Stability = 1 - (score drop / baseline)
            if score_clean > 0:
                drop = abs(score_clean - score_noisy) / score_clean
                robustness = 1.0 - drop
            else:
                robustness = 0.5
            
            return float(np.clip(robustness, 0.0, 1.0))
        
        except Exception as e:
            logger.debug("_check_noise_robustness failed: %s", e)
            return 0.5  # Neutral
    
    def _check_feature_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str = "regression",
        top_k_percent: float = 0.5,
    ) -> float:
        """
        Is there feature importance concentration? (Pareto principle)
        
        If top 50% of features dominate, score is high (= concentrated signal).
        If features uniformly important, score is lower (= diffuse signal).
        
        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            Embeddings.
        y : np.ndarray, shape (N,)
            Targets.
        task_type : str
            "regression" or "classification"
        top_k_percent : float
            Check if top k% of features explain top_k_percent of importance.
        
        Returns
        -------
        float
            Importance concentration (0-1), 1 = strong concentration.
        """
        try:
            if task_type == "regression":
                model = RandomForestRegressor(
                    n_estimators=50,
                    max_features=min(self.max_features, X.shape[1]),
                    random_state=42,
                )
            else:
                model = RandomForestClassifier(
                    n_estimators=50,
                    max_features=min(self.max_features, X.shape[1]),
                    random_state=42,
                )
            
            model.fit(X, y)
            importances = model.feature_importances_  # (D,)
            
            # Sort and cumsum
            sorted_imp = np.sort(importances)[::-1]
            cumsum_imp = np.cumsum(sorted_imp)
            cumsum_imp = cumsum_imp / cumsum_imp[-1]  # Normalize to [0, 1]
            
            # At what fraction of features does top_k_percent importance reach?
            threshold_idx = np.searchsorted(cumsum_imp, top_k_percent)
            concentration = 1.0 - (threshold_idx / len(importances))
            
            return float(np.clip(concentration, 0.0, 1.0))
        
        except Exception as e:
            logger.debug("_check_feature_importance failed: %s", e)
            return 0.5  # Neutral
    
    def final_score(
        self,
        embeddings: Dict[str, np.ndarray],
        y: np.ndarray,
        task_type: str = "regression",
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Compute final predictability score for a single modality.
        
        Formula:
        --------
        final = 0.40×predictability + 0.20×complementarity + 0.15×degeneracy
                + 0.15×noise_robustness + 0.10×feature_importance
        
        Parameters
        ----------
        embeddings : Dict[str, np.ndarray]
            All modalities' embeddings: {"text": (N,768), "image": (N,2048), ...}
        y : np.ndarray, shape (N,)
            Target variable.
        task_type : str
            "regression" or "classification"
        weights : Optional[Dict[str, float]]
            Override default weights: {"predictability": 0.40, ...}
        
        Returns
        -------
        float
            Predictability score (0-1).
        """
        # Default weights
        w = weights or {
            "predictability": 0.40,
            "complementarity": 0.20,
            "degeneracy": 0.15,
            "noise_robustness": 0.15,
            "feature_importance": 0.10,
        }
        
        scores = {}
        
        # For each modality
        for mod_name, X in embeddings.items():
            if X.shape[0] < self.cv_folds:
                logger.warning(
                    "final_score: %s has only %d samples, skipping details",
                    mod_name, X.shape[0],
                )
                scores[mod_name] = 0.0
                continue
            
            # 1. Predictability (RF 3-fold CV)
            pred = self._predict(X, y, task_type=task_type)
            
            # 2. Complementarity (vs other modalities)
            X_other = np.concatenate(
                [v for k, v in embeddings.items() if k != mod_name],
                axis=1,
            ) if len(embeddings) > 1 else None
            comp = self._check_complementarity(X, X_other)
            
            # 3. Degeneracy (fraction non-constant features)
            degen = self._check_degeneracy(X)
            
            # 4. Noise robustness (stability under perturbation)
            noise = self._check_noise_robustness(X, y, task_type=task_type)
            
            # 5. Feature importance (concentration of signal)
            feat = self._check_feature_importance(X, y, task_type=task_type)
            
            # Weighted sum
            final = (
                w["predictability"] * pred +
                w["complementarity"] * comp +
                w["degeneracy"] * degen +
                w["noise_robustness"] * noise +
                w["feature_importance"] * feat
            )
            
            scores[mod_name] = float(np.clip(final, 0.0, 1.0))
            
            logger.info(
                "Modality '%s' scores: pred=%.3f comp=%.3f degen=%.3f "
                "noise=%.3f feat=%.3f → final=%.3f",
                mod_name, pred, comp, degen, noise, feat, scores[mod_name],
            )
        
        # Return scores as dict
        return scores


def validate_all_modalities(
    embeddings: Dict[str, np.ndarray],
    y: np.ndarray,
    task_type: str = "regression",
) -> Dict[str, float]:
    """
    Convenience function: validate all modalities at once.
    
    Parameters
    ----------
    embeddings : Dict[str, np.ndarray]
        All modalities' embeddings.
    y : np.ndarray
        Target variable.
    task_type : str
        "regression" or "classification"
    
    Returns
    -------
    Dict[str, float]
        Modality → predictability score.
    """
    validator = UniversalTargetValidator()
    return validator.final_score(embeddings, y, task_type=task_type)
