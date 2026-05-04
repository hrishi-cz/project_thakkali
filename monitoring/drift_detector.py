"""
Production-grade drift detector using rigorous statistical tests.

Metrics implemented
-------------------
KS  (Kolmogorov-Smirnov)       – ``scipy.stats.ks_2samp`` per numerical
                                  feature; report the *maximum* statistic
                                  across all features.
PSI (Population Stability Index)– Binning-based stability measure computed
                                  per feature and averaged across features.
                                  Formula: Σ (prod_% − ref_%) × ln(prod_%/ref_%)
FDD (Feature Drift Distance)   – Maximum Mean Discrepancy (MMD) with an
                                  RBF kernel evaluated on a random subsample
                                  of up to 500 rows.  Captures multivariate
                                  distributional shift in the full feature
                                  space.
DDM (Drift Detection Method)   – Online concept-drift detector tracking
                                  prediction error rates (Gama et al., 2004).
                                  Raises "warning" / "drift" signals from
                                  a sequential stream of binary error events
                                  using a Gaussian approximation of binomial
                                  variance.  Reference:
                                    Gama, J. et al., "Learning with drift
                                    detection", SBIA 2004.
DriftLens cosine drift         – Mean pairwise cosine distance in the
                                  PCA-reduced embedding subspace (IEEE 2024).
                                  Complements MMD by catching rotational
                                  shifts invisible to Euclidean kernels.

PDF Thresholds (hardcoded)
--------------------------
PSI  > 0.25  →  drift
KS   > 0.30  →  drift
FDD  > 0.50  →  drift

If **any** threshold is breached, ``DriftReport.drift_detected`` is ``True``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PDF thresholds (fixed per spec)
# ---------------------------------------------------------------------------

PSI_THRESHOLD: float = 0.25
KS_THRESHOLD:  float = 0.30
FDD_THRESHOLD: float = 0.50


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DriftReport:
    """
    Complete output of a single drift detection run.

    Attributes
    ----------
    psi            : Mean PSI across all features (scalar).
    ks_statistic   : Maximum KS statistic across all features (scalar).
    fdd            : Feature Drift Distance (MMD, scalar).
    drift_detected : ``True`` when **any** metric breaches its threshold.
    status         : Per-metric breach flags:
                     ``{"psi": bool, "ks_statistic": bool, "fdd": bool}``.
    per_feature_ks  : KS statistic for each individual feature.
    per_feature_psi : PSI value for each individual feature.
    n_features      : Number of features analysed.
    n_reference     : Sample count in the reference (older) split.
    n_production    : Sample count in the production (recent) split.
    """

    psi: float
    ks_statistic: float
    fdd: float
    drift_detected: bool
    status: Dict[str, bool] = field(default_factory=dict)
    per_feature_ks: Dict[str, float] = field(default_factory=dict)
    per_feature_psi: Dict[str, float] = field(default_factory=dict)
    n_features: int = 0
    n_reference: int = 0
    n_production: int = 0
    composite_score: float = 0.0
    retrain_triggered: bool = False
    retrain_info: Dict[str, Any] = field(default_factory=dict)
    reference_sample: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DriftDetector:
    """
    Stateless drift detector.  All state lives in the returned
    :class:`DriftReport`; the instance can be reused across multiple calls.

    Usage
    -----
    >>> dd = DriftDetector()
    >>> report = dd.detect(reference_array, production_array, feature_names)
    >>> report.drift_detected
    False
    """

    PSI_BINS: int = 10
    MMD_SUBSAMPLE: int = 500  # subsample cap for O(n²) MMD kernel computation

    def __init__(
        self,
        retraining_orchestrator: Optional[Any] = None,
        cooldown_seconds: int = 3600,
    ) -> None:
        self.retraining_orchestrator = retraining_orchestrator
        self.cooldown_seconds = max(0, int(cooldown_seconds))
        self._last_retrain_by_dataset: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def detect(
        self,
        reference: np.ndarray,
        production: np.ndarray,
        feature_names: Optional[List[str]] = None,
        dataset_id: str = "default",
    ) -> DriftReport:
        """
        Run KS, PSI, and FDD tests between *reference* and *production*.

        Parameters
        ----------
        reference    : 2-D float array, shape ``(n_ref, n_features)``.
        production   : 2-D float array, shape ``(n_prod, n_features)``.
        feature_names: Optional list of feature names for labelling per-feature
                       results.  Auto-generated as ``feature_0 …`` when absent.

        Returns
        -------
        :class:`DriftReport`
        """
        reference  = np.asarray(reference,  dtype=np.float64)
        production = np.asarray(production, dtype=np.float64)

        # Force 2-D
        if reference.ndim == 1:
            reference  = reference.reshape(-1, 1)
        if production.ndim == 1:
            production = production.reshape(-1, 1)

        n_ref,  n_feat = reference.shape
        n_prod, _      = production.shape

        if feature_names is None or len(feature_names) != n_feat:
            feature_names = [f"feature_{i}" for i in range(n_feat)]

        # ── Per-feature KS and PSI ────────────────────────────────────────
        per_ks:  Dict[str, float] = {}
        per_psi: Dict[str, float] = {}

        for i, fname in enumerate(feature_names):
            ref_col  = reference[:, i]
            prod_col = production[:, i]
            per_ks[fname]  = self._compute_ks(ref_col, prod_col)
            per_psi[fname] = self._compute_psi(ref_col, prod_col)

        ks_statistic: float = (
            float(max(per_ks.values())) if per_ks else 0.0
        )
        psi: float = (
            float(np.mean(list(per_psi.values()))) if per_psi else 0.0
        )

        # ── Multivariate FDD (MMD with RBF kernel) ────────────────────────
        fdd: float = self._compute_mmd(reference, production)

        # ── Threshold checks (PDF spec) ───────────────────────────────────
        status: Dict[str, bool] = {
            "psi":          psi          > PSI_THRESHOLD,
            "ks_statistic": ks_statistic > KS_THRESHOLD,
            "fdd":          fdd          > FDD_THRESHOLD,
        }
        drift_detected: bool = any(status.values())
        composite_score: float = self._compute_composite_score(
            psi=psi,
            ks_statistic=ks_statistic,
            fdd=fdd,
        )

        retrain_triggered = False
        retrain_info: Dict[str, Any] = {}
        if self.retraining_orchestrator is not None:
            drift_payload = {
                "drift_detected": drift_detected,
                "composite_score": composite_score,
                "metrics": {
                    "psi": psi,
                    "ks_statistic": ks_statistic,
                    "fdd": fdd,
                },
                "status": status,
            }
            try:
                if self.retraining_orchestrator.should_retrain(drift_payload):
                    now = time.time()
                    last = self._last_retrain_by_dataset.get(dataset_id)
                    if last is None or (now - last) >= self.cooldown_seconds:
                        retrain_info = self.retraining_orchestrator.trigger_retraining(
                            dataset_id=dataset_id,
                            drift_report=drift_payload,
                        )
                        retrain_triggered = bool(retrain_info.get("triggered", False))
                        if retrain_triggered:
                            self._last_retrain_by_dataset[dataset_id] = now
                    else:
                        retrain_info = {
                            "triggered": False,
                            "status": "cooldown_blocked",
                            "cooldown_remaining_seconds": round(
                                max(0.0, self.cooldown_seconds - (now - last)), 2
                            ),
                        }
            except Exception as retrain_exc:
                logger.warning(
                    "Retraining trigger failed; drift report will still be returned: %s",
                    retrain_exc,
                    exc_info=True,
                )
                retrain_triggered = False
                retrain_info = {
                    "triggered": False,
                    "status": "error",
                    "error": str(retrain_exc),
                }

        logger.info(
            "DriftDetector: psi=%.4f (>%.2f? %s)  ks=%.4f (>%.2f? %s)  "
            "fdd=%.4f (>%.2f? %s)  drift=%s  composite=%.3f  retrain=%s",
            psi,          PSI_THRESHOLD, status["psi"],
            ks_statistic, KS_THRESHOLD,  status["ks_statistic"],
            fdd,          FDD_THRESHOLD, status["fdd"],
            drift_detected,
            composite_score,
            retrain_triggered,
        )

        reference_sample = None
        if n_ref > 0:
            sample_n = min(int(self.MMD_SUBSAMPLE), int(n_ref))
            reference_sample = np.asarray(reference[:sample_n]).copy()

        return DriftReport(
            psi=psi,
            ks_statistic=ks_statistic,
            fdd=fdd,
            drift_detected=drift_detected,
            status=status,
            per_feature_ks=per_ks,
            per_feature_psi=per_psi,
            n_features=n_feat,
            n_reference=n_ref,
            n_production=n_prod,
            composite_score=composite_score,
            retrain_triggered=retrain_triggered,
            retrain_info=retrain_info,
            reference_sample=reference_sample,
        )

    def detect_modality_drift(
        self,
        reference_df: Any,
        production_df: Any,
        modality_columns: Dict[str, List[str]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute drift metrics separately for each modality column group.

        Parameters
        ----------
        reference_df : pandas.DataFrame-like
            Reference split dataframe.
        production_df : pandas.DataFrame-like
            Production split dataframe.
        modality_columns : dict
            Mapping modality name -> source column names.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Per-modality metrics and threshold breach status.
        """
        try:
            import pandas as pd
        except Exception:
            return {}

        if not isinstance(reference_df, pd.DataFrame) or not isinstance(production_df, pd.DataFrame):
            return {}

        modality_report: Dict[str, Dict[str, Any]] = {}
        for modality, cols in modality_columns.items():
            if not isinstance(cols, list) or not cols:
                continue

            valid_cols = [c for c in cols if c in reference_df.columns and c in production_df.columns]
            if not valid_cols:
                continue

            ref_block = reference_df[valid_cols].select_dtypes(include=[np.number]).fillna(0.0)
            prod_block = production_df[valid_cols].select_dtypes(include=[np.number]).fillna(0.0)

            if ref_block.empty or prod_block.empty:
                modality_report[modality] = {
                    "drift_detected": False,
                    "reason": "no_numeric_features",
                    "n_features": 0,
                }
                continue

            ref_arr = ref_block.to_numpy(dtype=np.float64)
            prod_arr = prod_block.to_numpy(dtype=np.float64)
            feature_names = list(ref_block.columns)

            per_ks: Dict[str, float] = {}
            per_psi: Dict[str, float] = {}
            for i, fname in enumerate(feature_names):
                per_ks[fname] = self._compute_ks(ref_arr[:, i], prod_arr[:, i])
                per_psi[fname] = self._compute_psi(ref_arr[:, i], prod_arr[:, i])

            ks_statistic = float(max(per_ks.values())) if per_ks else 0.0
            psi = float(np.mean(list(per_psi.values()))) if per_psi else 0.0
            fdd = self._compute_mmd(ref_arr, prod_arr)
            status = {
                "psi": psi > PSI_THRESHOLD,
                "ks_statistic": ks_statistic > KS_THRESHOLD,
                "fdd": fdd > FDD_THRESHOLD,
            }
            drift_detected = any(status.values())

            modality_report[str(modality)] = {
                "drift_detected": drift_detected,
                "metrics": {
                    "psi": psi,
                    "ks_statistic": ks_statistic,
                    "fdd": fdd,
                },
                "status": status,
                "n_features": len(feature_names),
                "feature_names": feature_names,
                "composite_score": self._compute_composite_score(
                    psi=psi,
                    ks_statistic=ks_statistic,
                    fdd=fdd,
                ),
            }

        return modality_report

    @staticmethod
    def _compute_composite_score(
        psi: float,
        ks_statistic: float,
        fdd: float,
    ) -> float:
        """Aggregate normalized drift metrics into a single risk score."""
        psi_ratio = psi / PSI_THRESHOLD if PSI_THRESHOLD > 0 else 0.0
        ks_ratio = ks_statistic / KS_THRESHOLD if KS_THRESHOLD > 0 else 0.0
        fdd_ratio = fdd / FDD_THRESHOLD if FDD_THRESHOLD > 0 else 0.0
        return float((psi_ratio + ks_ratio + fdd_ratio) / 3.0)

    # ------------------------------------------------------------------ #
    # KS test (per feature)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_ks(ref: np.ndarray, prod: np.ndarray) -> float:
        """
        Two-sample Kolmogorov-Smirnov statistic for one feature column.

        Uses ``scipy.stats.ks_2samp`` (exact two-sample variant).
        Returns the test *statistic* (0–1), not the p-value.
        """
        if len(ref) < 2 or len(prod) < 2:
            return 0.0
        result = stats.ks_2samp(ref, prod)
        return float(result.statistic)

    # ------------------------------------------------------------------ #
    # PSI (per feature)
    # ------------------------------------------------------------------ #

    def _compute_psi(
        self,
        ref: np.ndarray,
        prod: np.ndarray,
    ) -> float:
        """
        Population Stability Index for a single feature.

        Algorithm
        ---------
        1. Bin edges are derived from the combined min/max of both arrays so
           every observed value falls within a bin.
        2. Reference  → expected proportions (``ref_%``).
           Production → actual   proportions (``prod_%``).
        3. PSI formula::

               PSI = Σ (prod_% − ref_%) × ln(prod_% / ref_%)

        Epsilon 1e-8 is added to every bin count before normalising to avoid
        ``log(0)`` and division-by-zero on empty bins.

        PSI interpretation (conventional)
        -----------------------------------
        < 0.10  : negligible change
        0.10–0.25 : moderate change
        > 0.25  : significant drift (PDF threshold)
        """
        if len(ref) < 2 or len(prod) < 2:
            return 0.0

        combined_min = float(min(ref.min(), prod.min()))
        combined_max = float(max(ref.max(), prod.max()))

        if combined_min == combined_max:
            # Constant feature – no distributional information
            return 0.0

        bin_edges = np.linspace(combined_min, combined_max, self.PSI_BINS + 1)

        ref_counts,  _ = np.histogram(ref,  bins=bin_edges)
        prod_counts, _ = np.histogram(prod, bins=bin_edges)

        eps: float = 1e-8
        ref_pct  = (ref_counts  + eps) / (len(ref)  + eps * self.PSI_BINS)
        prod_pct = (prod_counts + eps) / (len(prod)  + eps * self.PSI_BINS)

        # PSI is mathematically non-negative; clamp any floating-point noise
        psi: float = float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))
        return max(psi, 0.0)

    # ------------------------------------------------------------------ #
    # FDD – Maximum Mean Discrepancy with RBF kernel
    # ------------------------------------------------------------------ #

    def _compute_mmd(
        self,
        ref: np.ndarray,
        prod: np.ndarray,
    ) -> float:
        """
        Multivariate Maximum Mean Discrepancy with a Gaussian (RBF) kernel.

        A random subsample of up to ``MMD_SUBSAMPLE`` rows is drawn from each
        split before computing the O(n²) kernel matrices.

        Bandwidth
        ---------
        Median heuristic: ``σ² = median(positive pairwise squared distances) / 2``.
        This is scale-invariant and avoids manual bandwidth selection.

        MMD² (unbiased estimate)
        ------------------------
        ::

            MMD²(X, Y) =  [Σᵢ≠ⱼ k(xᵢ,xⱼ)] / [n(n-1)]
                        + [Σᵢ≠ⱼ k(yᵢ,yⱼ)] / [m(m-1)]
                        − 2 × mean_ij k(xᵢ,yⱼ)

        Returns
        -------
        ``sqrt(max(MMD², 0))`` — always non-negative.
        """
        rng = np.random.default_rng(seed=42)

        n_ref_sub  = min(self.MMD_SUBSAMPLE, len(ref))
        n_prod_sub = min(self.MMD_SUBSAMPLE, len(prod))
        idx_r = rng.choice(len(ref),  n_ref_sub,  replace=False)
        idx_p = rng.choice(len(prod), n_prod_sub, replace=False)
        X: np.ndarray = ref[idx_r].astype(np.float64)
        Y: np.ndarray = prod[idx_p].astype(np.float64)

        # Median-heuristic bandwidth
        all_pts  = np.vstack([X, Y])
        sq_dists = np.sum(
            (all_pts[:, None, :] - all_pts[None, :, :]) ** 2,
            axis=-1,
        )
        positive = sq_dists[sq_dists > 0]
        sigma_sq: float = float(np.median(positive)) / 2.0 if positive.size > 0 else 1.0

        def _rbf(A: np.ndarray, B: np.ndarray) -> np.ndarray:
            d = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
            return np.exp(-d / (2.0 * sigma_sq))

        Kxx = _rbf(X, X)
        Kyy = _rbf(Y, Y)
        Kxy = _rbf(X, Y)

        n, m = len(X), len(Y)

        term_xx: float = (
            (np.sum(Kxx) - np.trace(Kxx)) / (n * (n - 1))
            if n > 1 else 0.0
        )
        term_yy: float = (
            (np.sum(Kyy) - np.trace(Kyy)) / (m * (m - 1))
            if m > 1 else 0.0
        )
        term_xy: float = 2.0 * float(np.mean(Kxy))

        mmd_sq: float = term_xx + term_yy - term_xy
        return float(np.sqrt(max(mmd_sq, 0.0)))

    # ------------------------------------------------------------------ #
    # Embedding-space drift (Phase 6)
    # ------------------------------------------------------------------ #

    def detect_embedding_drift(
        self,
        reference_embeddings: np.ndarray,
        current_embeddings: np.ndarray,
        n_components: int = 32,
        modality_name: str = "embedding",
    ) -> Dict[str, Any]:
        """
        Dual-metric embedding-space drift detection.

        Combines two complementary distances (Gama/MMD-RBF + DriftLens
        cosine, IEEE 2024) so that both Euclidean magnitude shifts and
        directional/rotational shifts are caught.

        MMD-RBF (``mmd_score``)
            Maximum Mean Discrepancy in PCA-reduced embedding space.
            Sensitive to global distributional shift.

        DriftLens cosine (``cosine_drift_score``, IEEE 2024)
            Mean pairwise cosine distance between a subsample of reference
            and current embeddings in the PCA subspace.  Captures semantic
            drift (topic/concept shifts) invisible to Euclidean kernels.

        Both scores are evaluated independently; the final ``drift_detected``
        flag is True when **either** threshold is breached.

        Parameters
        ----------
        reference_embeddings : (N_ref, D) float array
            Embeddings collected at training time.
        current_embeddings : (N_cur, D) float array
            Embeddings from recent production traffic.
        n_components : int
            PCA target dimensionality; skipped when D ≤ n_components.
        modality_name : str
            Label used in the returned dict ("text", "image", etc.).

        Returns
        -------
        dict with keys: mmd_score, cosine_drift_score,
                        drift_detected, severity,
                        n_ref, n_cur, n_components_used, modality.
        """
        import os as _os

        ref = np.asarray(reference_embeddings, dtype=np.float64)
        cur = np.asarray(current_embeddings, dtype=np.float64)

        if ref.ndim == 1:
            ref = ref.reshape(-1, 1)
        if cur.ndim == 1:
            cur = cur.reshape(-1, 1)

        n_ref, D = ref.shape
        n_cur = cur.shape[0]

        n_components_used = D
        if D > n_components and n_ref >= n_components and n_cur >= n_components:
            try:
                from sklearn.decomposition import PCA

                pca = PCA(n_components=n_components, random_state=42)
                ref = pca.fit_transform(ref)
                cur = pca.transform(cur)
                n_components_used = n_components
            except Exception as exc:
                logger.debug(
                    "detect_embedding_drift: PCA failed (%s); using raw embeddings", exc
                )

        embed_drift_threshold = float(_os.getenv("APEX_EMBED_DRIFT_THRESHOLD", "0.25"))
        cosine_drift_threshold = float(_os.getenv("APEX_COSINE_DRIFT_THRESHOLD", "0.15"))

        # ── MMD-RBF (Gama-family, Euclidean kernel) ───────────────────────
        mmd = self._compute_mmd(ref, cur)

        # ── DriftLens cosine distance (IEEE 2024) ─────────────────────────
        cosine_drift = self._compute_cosine_drift(ref, cur)

        # Severity from the *worse* of the two signals
        def _severity(score: float, threshold: float) -> str:
            if score < threshold * 0.5:
                return "low"
            if score < threshold:
                return "medium"
            return "high"

        sev_rank = {"low": 1, "medium": 2, "high": 3}
        mmd_sev = _severity(mmd, embed_drift_threshold)
        cos_sev = _severity(cosine_drift, cosine_drift_threshold)
        severity = mmd_sev if sev_rank[mmd_sev] >= sev_rank[cos_sev] else cos_sev

        drift_detected = (mmd > embed_drift_threshold) or (cosine_drift > cosine_drift_threshold)

        return {
            "modality": modality_name,
            "mmd_score": float(mmd),
            "cosine_drift_score": float(cosine_drift),
            "drift_detected": drift_detected,
            "severity": severity,
            "mmd_severity": mmd_sev,
            "cosine_severity": cos_sev,
            "n_ref": int(n_ref),
            "n_cur": int(n_cur),
            "n_components_used": int(n_components_used),
            "mmd_threshold": float(embed_drift_threshold),
            "cosine_threshold": float(cosine_drift_threshold),
        }

    @staticmethod
    def _compute_cosine_drift(ref: np.ndarray, cur: np.ndarray, subsample: int = 300) -> float:
        """
        Mean pairwise cosine distance between reference and current embeddings.

        Uses random subsampling (cap=300) for tractability.  Cosine distance
        is 1 − cosine_similarity, so 0 = identical directions, 2 = opposite.
        Values near 0 indicate no directional shift.

        Paper: DriftLens (IEEE 2024) — embedding-space drift via angular
        separability in the principal-component subspace.
        """
        rng = np.random.default_rng(seed=0)
        n_r = min(subsample, len(ref))
        n_c = min(subsample, len(cur))
        R = ref[rng.choice(len(ref), n_r, replace=False)]
        C = cur[rng.choice(len(cur), n_c, replace=False)]

        # L2-normalise each row
        r_norm = np.linalg.norm(R, axis=1, keepdims=True)
        c_norm = np.linalg.norm(C, axis=1, keepdims=True)
        R = R / np.where(r_norm > 0, r_norm, 1.0)
        C = C / np.where(c_norm > 0, c_norm, 1.0)

        # cosine similarity matrix (n_r, n_c); cosine distance = 1 - sim
        sim_matrix = R @ C.T  # (n_r, n_c)
        cos_distance = 1.0 - float(np.mean(sim_matrix))
        return max(0.0, cos_distance)

    # ------------------------------------------------------------------ #
    # Severity-based retraining depth selection (Phase 6)
    # ------------------------------------------------------------------ #

    def select_retraining_depth(
        self,
        covariate_report: Optional["DriftReport"] = None,
        concept_drift: bool = False,
        embedding_reports: Optional[Dict[str, Dict[str, Any]]] = None,
        execution_context: Optional[Any] = None,
    ) -> str:
        """
        Choose the minimum sufficient retraining depth given all drift signals.

        Policy
        ------
        concept_drift=True            → "full"
        any embedding severity "high" → "full"
        covariate drift high          → "full"
        covariate drift medium
          OR embedding severity "medium" → "head_only"
        covariate drift low
          OR embedding severity "low"   → "calibration_only"
        no drift at all               → "none"

        Stores the decision in *execution_context* when supplied.

        Returns
        -------
        One of "none", "calibration_only", "head_only", "full".
        """
        embedding_reports = embedding_reports or {}

        # Covariate severity from composite score
        covariate_severity = "none"
        if covariate_report is not None and covariate_report.drift_detected:
            cs = covariate_report.composite_score
            if cs >= 2.0:
                covariate_severity = "high"
            elif cs >= 1.0:
                covariate_severity = "medium"
            else:
                covariate_severity = "low"

        # Embedding severity (worst case across modalities)
        embed_severities = [
            r.get("severity", "none")
            for r in embedding_reports.values()
            if r.get("drift_detected", False)
        ]
        sev_rank = {"none": 0, "low": 1, "medium": 2, "high": 3}
        embedding_severity = max(
            embed_severities, key=lambda s: sev_rank.get(s, 0), default="none"
        )

        # Policy decision
        if concept_drift:
            depth = "full"
        elif covariate_severity == "high" or sev_rank.get(embedding_severity, 0) >= 3:
            depth = "full"
        elif covariate_severity == "medium" or sev_rank.get(embedding_severity, 0) >= 2:
            depth = "head_only"
        elif covariate_severity == "low" or sev_rank.get(embedding_severity, 0) >= 1:
            depth = "calibration_only"
        else:
            depth = "none"

        if execution_context is not None:
            try:
                execution_context.retraining_depth_required = depth
                if hasattr(execution_context, "log_decision"):
                    execution_context.log_decision(
                        "retraining_policy",
                        f"Retraining depth selected: {depth}",
                        evidence=(
                            f"covariate={covariate_severity}, "
                            f"concept={concept_drift}, "
                            f"embedding={embedding_severity}"
                        ),
                    )
            except Exception as exc:
                logger.debug("select_retraining_depth: ctx update failed: %s", exc)

        return depth


# ---------------------------------------------------------------------------
# DDM — Drift Detection Method (Gama et al., SBIA 2004)
# [10] Online concept-drift detector for sequential error streams
# ---------------------------------------------------------------------------

class DDMConceptDriftDetector:
    """
    Online concept-drift detector based on the Drift Detection Method
    (DDM, Gama et al., 2004).

    Monitors a stream of binary prediction-error events (1 = error,
    0 = correct) and raises warnings/alarms when the running error rate
    deviates significantly from the historical minimum.

    Theory
    ------
    For a binomial process with observed error rate p̄ over n examples,
    the standard deviation is approximately::

        s̄ ≈ sqrt( p̄(1 − p̄) / n )

    DDM tracks the minimum (p_min, s_min) ever seen.  Two thresholds:

    Warning level  : p̄ + s̄  ≥  p_min + 2 · s_min
    Drift level    : p̄ + s̄  ≥  p_min + 3 · s_min

    On confirmed drift the internal statistics are reset (a new concept
    is assumed to have started).

    Paper reference
    ---------------
    Gama, J., Medas, P., Castillo, G., & Rodrigues, P. (2004).
    "Learning with drift detection." SBIA 2004, LNAI 3171, pp. 286-295.

    Usage
    -----
    >>> ddm = DDMConceptDriftDetector()
    >>> for prediction, label in stream:
    ...     error = int(prediction != label)
    ...     status = ddm.update(bool(error))
    ...     if status == "drift":
    ...         retrain()
    ...         ddm.reset()
    """

    # Gama et al. (2004) use 2σ for warning, 3σ for drift
    WARNING_LEVEL: float = 2.0
    DRIFT_LEVEL:   float = 3.0
    MIN_INSTANCES: int   = 30   # minimum samples before issuing any alarm

    def __init__(self) -> None:
        self._n:     int   = 0
        self._error: float = 0.0   # running error rate p̄
        self._p_min: float = float("inf")
        self._s_min: float = float("inf")
        self._in_warning: bool = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def update(self, error_occurred: bool) -> str:
        """
        Ingest one binary error event and return the detector status.

        Parameters
        ----------
        error_occurred : bool
            True if the model made an error on this sample.

        Returns
        -------
        str
            ``"stable"`` | ``"warning"`` | ``"drift"``
        """
        self._n += 1
        self._error += (float(bool(error_occurred)) - self._error) / self._n

        # Too few samples for a reliable estimate
        if self._n < self.MIN_INSTANCES:
            return "stable"

        s = float(np.sqrt(self._error * (1.0 - self._error) / self._n))

        # Update minimum statistics
        if self._error + s < self._p_min + self._s_min:
            self._p_min = self._error
            self._s_min = s

        # Guard against degenerate (zero-error) base
        if self._s_min == 0.0:
            return "stable"

        level = self._error + s

        if level >= self._p_min + self.DRIFT_LEVEL * self._s_min:
            self._in_warning = False
            logger.warning(
                "DDM: DRIFT detected at n=%d, error_rate=%.4f (p_min=%.4f, s_min=%.4f)",
                self._n, self._error, self._p_min, self._s_min,
            )
            return "drift"

        if level >= self._p_min + self.WARNING_LEVEL * self._s_min:
            self._in_warning = True
            logger.info(
                "DDM: WARNING zone at n=%d, error_rate=%.4f", self._n, self._error
            )
            return "warning"

        self._in_warning = False
        return "stable"

    def reset(self) -> None:
        """
        Reset all internal statistics.

        Call this after confirmed drift once a new model has been deployed
        so the detector starts fresh on the new concept.
        """
        self._n     = 0
        self._error = 0.0
        self._p_min = float("inf")
        self._s_min = float("inf")
        self._in_warning = False
        logger.debug("DDM: statistics reset (new concept epoch started)")

    @property
    def error_rate(self) -> float:
        """Current running error rate p̄."""
        return float(self._error)

    @property
    def n_samples(self) -> int:
        """Total samples processed since last reset."""
        return int(self._n)

    @property
    def in_warning(self) -> bool:
        """True when the detector is in the warning zone."""
        return bool(self._in_warning)

    def state_dict(self) -> Dict[str, Any]:
        """Serialise detector state for checkpointing."""
        return {
            "n": self._n,
            "error": self._error,
            "p_min": self._p_min,
            "s_min": self._s_min,
            "in_warning": self._in_warning,
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "DDMConceptDriftDetector":
        """Restore from a serialised state dict."""
        obj = cls()
        obj._n         = int(state.get("n", 0))
        obj._error     = float(state.get("error", 0.0))
        obj._p_min     = float(state.get("p_min", float("inf")))
        obj._s_min     = float(state.get("s_min", float("inf")))
        obj._in_warning = bool(state.get("in_warning", False))
        return obj
