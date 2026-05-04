"""Tabular preprocessing – ColumnTransformer pipeline (PyTorch-compatible)."""

from __future__ import annotations

import logging
import re
import threading
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

logger = logging.getLogger(__name__)
_CONFIG_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Heuristic thresholds for automatic column filtering
# ---------------------------------------------------------------------------
_MAX_OHE_CARDINALITY: int = 50          # one-hot encode only if ≤ this many unique values
_NEAR_UNIQUE_RATIO: float = 0.5         # drop if unique_ratio > this (likely IDs)
_PATH_PATTERN = re.compile(              # detect file paths / URLs
    r"[/\\]|\.(?:csv|json|dat|hea|png|jpg|wav|mp3|parquet|zip)", re.IGNORECASE
)
_ID_NAME_PATTERN = re.compile(           # detect ID-like column names
    r"(?:^|_)(?:id|idx|index|key|serial|pk|fk)(?:$|_)",
    re.IGNORECASE,
)
_NEAR_UNIQUE_RATIO_STRICT: float = 0.9  # drop ANY integer col above this
_ID_UNIQUE_RATIO: float = 0.1           # drop ID-named cols above this


class TabularPreprocessor:
    """
    Scikit-learn ColumnTransformer pipeline that produces ``np.float32``
    arrays ready for ``torch.tensor()``.

    Pipeline
    --------
    Numeric columns  : SimpleImputer(median) → StandardScaler
    Categorical cols : SimpleImputer(most_frequent) → OneHotEncoder(sparse=False)

    Automatic filtering (universal pipeline safety):
    - Columns with near-unique values (>50% unique) are dropped (IDs, paths, timestamps).
    - Categorical columns with >50 unique values are dropped (prevents OHE explosion).
    - File-path and URL columns are detected and dropped.

    ``sparse_output=False`` is mandatory: ``torch.tensor()`` cannot consume
    scipy sparse matrices and will raise a ``TypeError``.

    Usage
    -----
    >>> tp = TabularPreprocessor()
    >>> arr = tp.fit_transform(train_df)      # np.float32, shape (N, D)
    >>> arr_test = tp.transform(test_df)      # same D
    >>> dim = tp.get_output_dim()
    """

    def __init__(
        self,
        adaptive_config: Optional[Dict[str, Any]] = None,
        drifted_features: Optional[List[str]] = None,
    ) -> None:
        self._transformer: Optional[ColumnTransformer] = None
        self._feature_names_in: List[str] = []
        self._dropped_cols: List[str] = []
        self._imputer_strategy: str = "median"
        self._scaler_mode: str = "standard"
        self._adaptive_config: Dict[str, Any] = dict(adaptive_config or {})
        self._drifted_features: List[str] = [str(col) for col in (drifted_features or [])]

        if self._adaptive_config:
            self.configure(self._adaptive_config)

    # ------------------------------------------------------------------
    # Column filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _is_path_column(series: pd.Series) -> bool:
        """Return True if most non-null values look like file paths or URLs."""
        sample = series.dropna().astype(str).head(50)
        if len(sample) == 0:
            return False
        return _PATH_PATTERN.search(sample.iloc[0]) is not None and (
            sample.str.contains(r"[/\\]", na=False).mean() > 0.5
        )

    @staticmethod
    def _is_datetime_like(series: pd.Series) -> bool:
        """Return True if a string/object column looks like dates or timestamps."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        sample = series.dropna().astype(str).head(30)
        if len(sample) == 0:
            return False
        try:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            return parsed.notna().mean() > 0.7
        except Exception:
            return False

    def _filter_categorical(
        self,
        df: pd.DataFrame,
        categorical_cols: List[str],
    ) -> List[str]:
        """
        Remove categorical columns that would cause OHE explosion.

        Drops:
        - Near-unique columns (unique ratio > 50%) – IDs, hashes, timestamps
        - High-cardinality columns (> _MAX_OHE_CARDINALITY unique values)
        - File-path / URL columns
        """
        kept: List[str] = []
        n_rows = len(df)

        for col in categorical_cols:
            n_unique = df[col].nunique(dropna=True)
            unique_ratio = n_unique / max(n_rows, 1)

            # Check 1: near-unique → likely an ID or hash
            if unique_ratio > _NEAR_UNIQUE_RATIO:
                logger.info(
                    "  DROP '%s': near-unique (%.0f%% unique, %d values) – likely ID/hash",
                    col, unique_ratio * 100, n_unique,
                )
                self._dropped_cols.append(col)
                continue

            # Check 2: file-path column
            if self._is_path_column(df[col]):
                logger.info("  DROP '%s': detected as file-path column", col)
                self._dropped_cols.append(col)
                continue

            # Check 3: datetime-like string column
            if self._is_datetime_like(df[col]):
                logger.info("  DROP '%s': detected as datetime string column", col)
                self._dropped_cols.append(col)
                continue

            # Check 4: too many unique values for OHE
            if n_unique > _MAX_OHE_CARDINALITY:
                logger.info(
                    "  DROP '%s': cardinality %d exceeds OHE limit (%d)",
                    col, n_unique, _MAX_OHE_CARDINALITY,
                )
                self._dropped_cols.append(col)
                continue

            kept.append(col)

        return kept

    def _filter_numeric(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
    ) -> List[str]:
        """
        Remove numeric columns that carry no signal.

        Drops:
        - Constant columns (zero variance)
        - Near-unique integer columns with ID-like names (e.g. patient_id, ecg_id)
        - Very high uniqueness integers (>90% unique) regardless of name
        - Datetime-typed numeric columns (e.g. Unix timestamps stored as int64)
        """
        kept: List[str] = []
        n_rows = len(df)

        for col in numeric_cols:
            n_unique = df[col].nunique(dropna=True)

            # Constant column
            if n_unique <= 1:
                logger.info("  DROP '%s': constant (single value)", col)
                self._dropped_cols.append(col)
                continue

            # Integer column checks
            if pd.api.types.is_integer_dtype(df[col]):
                unique_ratio = n_unique / max(n_rows, 1)

                # Check 1: Column name matches ID pattern + any meaningful uniqueness
                if _ID_NAME_PATTERN.search(col) and unique_ratio > _ID_UNIQUE_RATIO:
                    logger.info(
                        "  DROP '%s': ID-like name + %.0f%% unique values",
                        col, unique_ratio * 100,
                    )
                    self._dropped_cols.append(col)
                    continue

                # Check 2: Very high uniqueness (>90%) — almost certainly auto-increment
                if unique_ratio > _NEAR_UNIQUE_RATIO_STRICT:
                    logger.info(
                        "  DROP '%s': near-unique integer (%.0f%% unique) "
                        "– likely auto-increment ID",
                        col, unique_ratio * 100,
                    )
                    self._dropped_cols.append(col)
                    continue

            # Datetime-typed numeric column (e.g. datetime64 parsed as nanoseconds)
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                logger.info("  DROP '%s': datetime numeric column", col)
                self._dropped_cols.append(col)
                continue

            kept.append(col)

        return kept

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        """
        Fit the ColumnTransformer on *df*.

        Numeric and categorical columns are detected automatically.
        Useless columns (IDs, paths, datetimes, high-cardinality) are dropped.
        """
        self._dropped_cols = []

        # Auto-detect and drop datetime64 columns before numeric/categorical split
        datetime_cols: List[str] = df.select_dtypes(
            include=["datetime64", "datetimetz"]
        ).columns.tolist()
        if datetime_cols:
            logger.info(
                "  DROP %d datetime columns: %s",
                len(datetime_cols), datetime_cols,
            )
            self._dropped_cols.extend(datetime_cols)
            df = df.drop(columns=datetime_cols)

        numeric_cols: List[str] = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols: List[str] = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Smart filtering
        numeric_cols = self._filter_numeric(df, numeric_cols)
        categorical_cols = self._filter_categorical(df, categorical_cols)

        numeric_robust_cols: List[str] = []
        numeric_standard_cols: List[str] = list(numeric_cols)
        if self._drifted_features:
            numeric_robust_cols = [col for col in numeric_cols if col in self._drifted_features]
            numeric_standard_cols = [col for col in numeric_cols if col not in numeric_robust_cols]

        if self._scaler_mode == "robust" and not numeric_robust_cols:
            numeric_robust_cols = list(numeric_standard_cols)
            numeric_standard_cols = []

        self._feature_names_in = numeric_standard_cols + numeric_robust_cols + categorical_cols

        if self._dropped_cols:
            logger.info(
                "TabularPreprocessor: dropped %d useless columns: %s",
                len(self._dropped_cols),
                self._dropped_cols,
            )

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy=self._imputer_strategy)),
            ("scaler", StandardScaler()),
        ])

        robust_numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy=self._imputer_strategy)),
            ("scaler", RobustScaler()),
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        transformers = []
        if numeric_standard_cols:
            transformers.append(("numeric", numeric_pipeline, numeric_standard_cols))
        if numeric_robust_cols:
            transformers.append(("numeric_robust", robust_numeric_pipeline, numeric_robust_cols))
        if categorical_cols:
            transformers.append(("categorical", categorical_pipeline, categorical_cols))

        if not transformers:
            raise ValueError(
                "TabularPreprocessor.fit: DataFrame has no usable numeric or categorical columns "
                f"(dropped {len(self._dropped_cols)} column(s) as IDs/paths/high-cardinality)."
            )

        self._transformer = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )
        self._transformer.fit(df)
        logger.info(
            "TabularPreprocessor fitted: %d numeric (%d robust), %d categorical columns → output dim %d",
            len(numeric_standard_cols) + len(numeric_robust_cols),
            len(numeric_robust_cols),
            len(categorical_cols),
            self.get_output_dim(),
        )
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Apply the fitted transformer.

        Returns
        -------
        np.ndarray of dtype ``float32``, shape ``(N, output_dim)``.
        """
        if self._transformer is None:
            raise RuntimeError("TabularPreprocessor must be fitted before transform().")
        # Align columns to match the fitted feature set
        if self._feature_names_in:
            missing = [c for c in self._feature_names_in if c not in df.columns]
            if missing:
                df = df.copy()
                for col in missing:
                    df[col] = 0.0
                logger.warning(
                    "TabularPreprocessor.transform: zero-filled %d missing columns: %s",
                    len(missing), missing,
                )
            df = df[self._feature_names_in]
        result = self._transformer.transform(df)
        return result.astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one call."""
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def configure(self, plan: Optional[Dict[str, Any]]) -> None:
        """
        Apply PreprocessingPlanner overrides before fit().

        Supported keys: max_cardinality (int), near_unique_ratio (float),
        imputer_strategy ("median"|"mean"|"most_frequent")
        """
        if not isinstance(plan, dict):
            return

        merged_plan: Dict[str, Any] = {}
        merged_plan.update(self._adaptive_config)
        merged_plan.update(plan)
        plan = merged_plan

        with _CONFIG_LOCK:
            global _MAX_OHE_CARDINALITY, _NEAR_UNIQUE_RATIO

            max_card = plan.get("max_cardinality")
            if max_card is not None:
                try:
                    _MAX_OHE_CARDINALITY = max(2, int(max_card))
                    logger.info(
                        "TabularPreprocessor.configure: max_cardinality -> %d",
                        _MAX_OHE_CARDINALITY,
                    )
                except (TypeError, ValueError):
                    pass

            near_unique = plan.get("near_unique_ratio")
            if near_unique is not None:
                try:
                    _NEAR_UNIQUE_RATIO = float(max(0.05, min(0.99, float(near_unique))))
                    logger.info(
                        "TabularPreprocessor.configure: near_unique_ratio -> %.2f",
                        _NEAR_UNIQUE_RATIO,
                    )
                except (TypeError, ValueError):
                    pass

        imputer_strat = plan.get("imputer_strategy")
        if imputer_strat in {"median", "mean", "most_frequent"}:
            self._imputer_strategy = str(imputer_strat)
            logger.info(
                "TabularPreprocessor.configure: imputer_strategy -> %s",
                self._imputer_strategy,
            )

        scaler = str(plan.get("scaler", self._scaler_mode)).lower()
        if scaler in {"standard", "robust"}:
            self._scaler_mode = scaler
            logger.info(
                "TabularPreprocessor.configure: scaler -> %s",
                self._scaler_mode,
            )

        drifted_features = plan.get("drifted_features")
        if isinstance(drifted_features, (list, tuple, set)):
            self._drifted_features = [str(col) for col in drifted_features]
            logger.info(
                "TabularPreprocessor.configure: drifted_features -> %d column(s)",
                len(self._drifted_features),
            )

        # ── Intelligence-driven tabular configuration ──────────────────────
        # Pull interaction_summary and uncertainty_summary from feature_intelligence
        # to select the best imputer and scaler for this dataset's signal profile.
        _fi = plan.get("feature_intelligence") or {}
        # feature_intelligence may be per-dataset or direct dict
        if isinstance(_fi, dict) and _fi:
            _tab_fi = _fi.get("tabular") or next(
                (v for v in _fi.values() if isinstance(v, dict)), {}
            )
            _interaction = dict(_tab_fi.get("interaction_summary") or {})
            _uncertainty  = dict(_tab_fi.get("uncertainty_summary") or {})
            _biz_patterns = dict(_tab_fi.get("business_patterns") or {})

            # High interaction score (mean > 0.5) → features are correlated →
            # RobustScaler handles outliers better than StandardScaler
            if _interaction:
                _mean_interaction = sum(float(v) for v in _interaction.values()) / max(1, len(_interaction))
                if _mean_interaction > 0.5 and self._scaler_mode == "standard":
                    self._scaler_mode = "robust"
                    logger.info(
                        "TabularPreprocessor: high interaction score (%.2f) -- "
                        "switching scaler to robust", _mean_interaction,
                    )

            # High-uncertainty columns (aleatoric uncertainty > 0.7) often have
            # high missingness → prefer 'most_frequent' imputation for categoricals
            if _uncertainty:
                _high_uncert = [
                    col for col, v in _uncertainty.items() if float(v or 0) > 0.7
                ]
                if _high_uncert and self._imputer_strategy == "median":
                    self._imputer_strategy = "most_frequent"
                    logger.info(
                        "TabularPreprocessor: high-uncertainty columns %s -- "
                        "switching imputer to most_frequent", _high_uncert[:3],
                    )

            # Business patterns: columns flagged as id_columns should be dropped
            # (already filtered by _NEAR_UNIQUE_RATIO, but semantic role confirms)
            _id_cols = list(_biz_patterns.get("id_like_columns", []) or [])
            if _id_cols:
                existing = list(self._drifted_features or [])
                # Mark id_cols as "drifted" so they get zero-coefficient treatment
                self._drifted_features = list({*existing, *_id_cols})
                logger.info(
                    "TabularPreprocessor: semantic id_columns detected %s -- "
                    "added to drifted_features for soft-drop treatment", _id_cols[:5],
                )

    def get_output_dim(self) -> int:
        """Return the number of output features after transformation."""
        if self._transformer is None:
            return 0
        try:
            return sum(
                len(t.get_feature_names_out())
                for _, t, _ in self._transformer.transformers_
                if hasattr(t, "get_feature_names_out")
                   and not isinstance(t, str)
            )
        except Exception:
            # Fallback: derive output dim from a single-row transform.
            # n_features_in_ is the INPUT count (pre-OHE), not output.
            try:
                dummy = self._transformer.transform(
                    pd.DataFrame(
                        np.zeros((1, self._transformer.n_features_in_)),
                        columns=self._transformer.feature_names_in_,
                    )
                )
                return dummy.shape[1]
            except Exception:
                return getattr(self._transformer, "n_features_in_", 0)

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "numeric_pipeline": ["SimpleImputer(median)", "StandardScaler"],
            "categorical_pipeline": ["SimpleImputer(most_frequent)", "OneHotEncoder(sparse=False)"],
            "output_dtype": "float32",
            "output_shape": "(N, output_dim)",
        }


# ---------------------------------------------------------------------------
# TabularFeatureTokenizer — per-feature tokens for ULA cross-modal attention
# ---------------------------------------------------------------------------

try:
    import torch as _torch
    import torch.nn as _nn

    class TabularFeatureTokenizer(_nn.Module):
        """
        Projects flattened tabular features ``(N, D)`` into a per-feature
        token sequence ``(N, D, token_dim)`` for ``UnifiedLatentFusion``.

        Each feature gets its own ``Linear(1 → token_dim)`` projection plus
        a learnable feature-type embedding, so the ULA transformer can attend
        to individual features rather than a single opaque tabular vector.
        """

        def __init__(self, n_features: int, token_dim: int = 256) -> None:
            super().__init__()
            self.n_features = n_features
            self.token_dim = token_dim
            self.projections = _nn.ModuleList([
                _nn.Linear(1, token_dim) for _ in range(n_features)
            ])
            self.feature_embeddings = _nn.Embedding(n_features, token_dim)

        def forward(self, x: "_torch.Tensor") -> "_torch.Tensor":
            """(N, D) → (N, D, token_dim)"""
            D = min(x.shape[1], self.n_features)
            device = x.device
            tokens = [self.projections[i](x[:, i : i + 1]) for i in range(D)]
            feat_ids = _torch.arange(D, device=device)
            type_embs = self.feature_embeddings(feat_ids)   # (D, token_dim)
            stacked = _torch.stack(tokens, dim=1)           # (N, D, token_dim)
            return stacked + type_embs.unsqueeze(0)

        def get_output_dim(self) -> int:
            return self.token_dim

except ImportError:
    class TabularFeatureTokenizer:  # type: ignore[no-redef]
        """Stub when PyTorch is unavailable."""
        def __init__(self, n_features: int, token_dim: int = 256) -> None:
            self.n_features = n_features
            self.token_dim = token_dim
        def get_output_dim(self) -> int:
            return self.token_dim
