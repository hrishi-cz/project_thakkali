"""
AutoVision+ Tier-2 Multi-Dataset Schema Engine — UNIVERSAL VERSION

Two-tier detection architecture:

  Tier 1  _detect_single(dataset_id, lazy_data)
          Accepts any of: Polars LazyFrame, Dask DataFrame, pandas DataFrame,
          or a PyTorch Dataset.  Materialises at most 500 rows into a pandas
          sample; never loads the whole dataset.  Computes per-column
          heuristics and returns an IndividualSchema.

  Tier 2  detect_global_schema(datasets)
          Loops over every lazy reference, calls Tier 1, then aggregates the
          results into a single GlobalSchema (global problem type, union of
          modalities, primary target).

Supported modalities:   tabular | text | image | timeseries | mask
Supported problem types: classification_binary | classification_multiclass
                         | regression | multilabel_classification
                         | segmentation | unsupervised
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data_ingestion.schema import GlobalSchema, IndividualSchema

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backward-compat alias so any code still importing SingleDatasetSchema works
# ---------------------------------------------------------------------------
SingleDatasetSchema = IndividualSchema


# ===========================================================================
# Main Engine
# ===========================================================================

class MultiDatasetSchemaDetector:
    """
    Universal schema detector.

    Public API
    ----------
    detect_global_schema(datasets)  – primary entry-point (lazy-aware)
    detect_schema(datasets)         – legacy API (accepts pandas DataFrames,
                                      returns dict for backward compatibility)
    """

    TARGET_KEYWORDS: List[str] = [
        "target", "label", "class", "y",
        "diagnosis", "condition", "output",
        "result", "severity",
    ]

    IMAGE_EXTENSIONS: Tuple[str, ...] = (
        ".png", ".jpg", ".jpeg", ".bmp", ".tiff",
    )

    # -----------------------------------------------------------------------
    # Tier-2: collective inference (PRIMARY PUBLIC METHOD)
    # -----------------------------------------------------------------------

    def detect_global_schema(
        self,
        datasets: Dict[str, Any],
        dataset_group: Optional[List[int]] = None,
    ) -> GlobalSchema:
        """
        Tier-2 collective inference across all datasets in the current session.

        Args:
            datasets: Mapping of dataset_id -> lazy reference.
                      Accepted types: polars.LazyFrame, dask.dataframe.DataFrame,
                      pandas.DataFrame (legacy), or torch.utils.data.Dataset.
            dataset_group: Optional list of dataset indices to use for Tier-2
                      aggregation.  When provided, only the specified datasets
                      participate in the global schema; others are still
                      detected individually but excluded from aggregation.

        Returns:
            GlobalSchema with resolved problem type, union of modalities,
            primary target, and serialised per-dataset breakdowns.
        """
        per_dataset_results: List[Dict[str, Any]] = []

        for dataset_id, lazy_data in datasets.items():
            try:
                schema: IndividualSchema = self._detect_single(
                    dataset_id, lazy_data
                )
                per_dataset_results.append(asdict(schema))
                logger.info(
                    "Schema Tier-1 [%s]: modalities=%s  target=%s  problem=%s  conf=%.2f",
                    dataset_id,
                    schema.modalities,
                    schema.target_column,
                    schema.problem_type,
                    schema.confidence,
                )
            except Exception as exc:
                logger.warning(
                    "Schema Tier-1 failed for [%s]: %s", dataset_id, exc
                )
                # Produce a safe fallback so the pipeline never hard-crashes
                per_dataset_results.append(
                    asdict(
                        IndividualSchema(
                            dataset_id=dataset_id,
                            detected_columns={
                                "image": [], "text": [], "tabular": [],
                                "timeseries": [], "mask": [],
                            },
                            target_column="Unknown",
                            problem_type="unsupervised",
                            modalities=["tabular"],
                            confidence=0.0,
                        )
                    )
                )

        global_modalities: List[str] = self._aggregate_modalities(per_dataset_results)
        global_problem: str = self._aggregate_problem_type(per_dataset_results)
        primary_target: str = self._select_primary_target(per_dataset_results)
        fusion_ready: bool = len(global_modalities) > 1

        # Cross-dataset relatedness checking
        groups, relatedness_report = self._check_relatedness(per_dataset_results)

        # Determine which datasets to use for aggregation
        if dataset_group is not None:
            # User-selected group: override auto-selection
            valid_indices = [i for i in dataset_group if i < len(per_dataset_results)]
            if valid_indices:
                filtered = [per_dataset_results[i] for i in valid_indices]
                global_modalities = self._aggregate_modalities(filtered)
                global_problem = self._aggregate_problem_type(filtered)
                primary_target = self._select_primary_target(filtered)
                fusion_ready = len(global_modalities) > 1
                relatedness_report["selected_group"] = valid_indices
                relatedness_report["selection_method"] = "user"
                logger.info(
                    "Schema Tier-2: using user-selected group (indices %s) "
                    "for global schema",
                    valid_indices,
                )
        elif len(groups) > 1:
            logger.warning(
                "Schema Tier-2: detected %d unrelated dataset groups: %s",
                len(groups), groups,
            )
            # Re-aggregate using only the largest related group
            largest_group = max(groups, key=len)
            filtered = [per_dataset_results[i] for i in largest_group]
            global_modalities = self._aggregate_modalities(filtered)
            global_problem = self._aggregate_problem_type(filtered)
            primary_target = self._select_primary_target(filtered)
            fusion_ready = len(global_modalities) > 1
            relatedness_report["selected_group"] = list(largest_group)
            relatedness_report["selection_method"] = "auto"
            logger.info(
                "Schema Tier-2: using largest related group (indices %s) "
                "for global schema",
                largest_group,
            )

        confidence: float = (
            float(np.mean([d["confidence"] for d in per_dataset_results]))
            if per_dataset_results
            else 0.0
        )

        logger.info(
            "Schema Tier-2 global: modalities=%s  problem=%s  target=%s  fusion=%s",
            global_modalities,
            global_problem,
            primary_target,
            fusion_ready,
        )

        return GlobalSchema(
            global_problem_type=global_problem,
            global_modalities=global_modalities,
            primary_target=primary_target,
            fusion_ready=fusion_ready,
            detection_confidence=round(confidence, 3),
            per_dataset=per_dataset_results,
            relatedness_report=relatedness_report,
        )

    # -----------------------------------------------------------------------
    # Legacy wrapper – accepts Dict[str, pd.DataFrame], returns plain dict
    # -----------------------------------------------------------------------

    def detect_schema(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Backward-compatible entry-point.  Accepts pandas DataFrames and
        returns the asdict() serialisation of the GlobalSchema.

        Delegates to detect_global_schema, which handles pandas DataFrames
        through _materialise_sample's pandas branch.
        """
        global_schema: GlobalSchema = self.detect_global_schema(datasets)
        return asdict(global_schema)

    # -----------------------------------------------------------------------
    # Tier-1: single-dataset inspector
    # -----------------------------------------------------------------------

    def _detect_single(
        self,
        dataset_id: str,
        lazy_data: Any,
    ) -> IndividualSchema:
        """
        Tier-1: inspect one lazy dataset.

        Materialises at most 500 rows for heuristic computation.  For PyTorch
        Datasets (image-only) it returns an image-modality schema directly
        without any column analysis.
        """
        sample_df: Optional[pd.DataFrame] = self._materialise_sample(
            lazy_data, n=500
        )

        if sample_df is None:
            # PyTorch Dataset or unrecognised type → treat as image dataset
            return IndividualSchema(
                dataset_id=dataset_id,
                detected_columns={
                    "image": ["__image_path__"],
                    "text": [],
                    "tabular": [],
                    "timeseries": [],
                    "mask": [],
                },
                target_column="Unknown",
                problem_type="unsupervised",
                modalities=["image"],
                confidence=0.5,
            )

        return self._inspect_dataframe(dataset_id, sample_df)

    # -----------------------------------------------------------------------
    # Sample materialisation – lazy → pandas (at most n rows)
    # -----------------------------------------------------------------------

    @staticmethod
    def _materialise_sample(
        lazy_data: Any,
        n: int = 500,
    ) -> Optional[pd.DataFrame]:
        """
        Materialise at most *n* rows from a lazy reference into a pandas
        DataFrame.  Returns None for PyTorch Datasets (image-only sources).

        Supported input types:
          - polars.LazyFrame   → .head(n).collect().to_pandas()
          - dask.dataframe.DataFrame → .head(n, compute=True)
          - pandas.DataFrame   → .head(n)
          - torch Dataset      → None  (image-only; no column structure)
        """
        # --- Polars LazyFrame ---
        try:
            import polars as pl  # noqa: F401 – guarded import
            if isinstance(lazy_data, pl.LazyFrame):
                return lazy_data.head(n).collect().to_pandas()
        except ImportError:
            pass

        # --- Dask DataFrame ---
        try:
            import dask.dataframe as dd  # noqa: F401
            if isinstance(lazy_data, dd.DataFrame):
                return lazy_data.head(n, compute=True)
        except ImportError:
            pass

        # --- Plain pandas DataFrame (legacy / already materialised) ---
        if isinstance(lazy_data, pd.DataFrame):
            return lazy_data.head(n)

        # --- PyTorch Dataset (image collections) ---
        try:
            from torch.utils.data import Dataset  # noqa: F401
            if isinstance(lazy_data, Dataset):
                return None  # handled upstream as image-only schema
        except ImportError:
            pass

        # Unknown type – attempt pandas coercion as last resort
        try:
            return pd.DataFrame(lazy_data).head(n)
        except Exception:
            return None

    # -----------------------------------------------------------------------
    # Core column-level analysis (runs on the pandas sample)
    # -----------------------------------------------------------------------

    def _inspect_dataframe(
        self,
        dataset_id: str,
        df: pd.DataFrame,
    ) -> IndividualSchema:
        """
        Run all four signals on the pandas sample and return an IndividualSchema.
        """
        detected: Dict[str, List[str]] = {
            "image": [],
            "text": [],
            "tabular": [],
            "timeseries": [],
            "mask": [],
        }

        for col in df.columns:
            series = df[col]
            if self._is_image(series):
                # Distinguish mask columns from regular image columns by name
                if self._is_mask_column(col):
                    detected["mask"].append(col)
                else:
                    detected["image"].append(col)
            elif self._is_timeseries(series):
                detected["timeseries"].append(col)
            elif self._is_text(series):
                detected["text"].append(col)
            else:
                detected["tabular"].append(col)

        target_col, confidence = self._detect_target(df)
        problem_type = self._infer_problem(df, target_col, detected_columns=detected)

        # For segmentation, the target is the mask column, not a tabular label
        if problem_type == "segmentation" and detected.get("mask"):
            target_col = detected["mask"][0]
            confidence = max(confidence, 0.8)

        modalities: List[str] = sorted(k for k, v in detected.items() if v)

        return IndividualSchema(
            dataset_id=dataset_id,
            detected_columns=detected,
            target_column=target_col,
            problem_type=problem_type,
            modalities=modalities,
            confidence=confidence,
        )

    # -----------------------------------------------------------------------
    # Column-type checks
    # -----------------------------------------------------------------------

    def _is_image(self, series: pd.Series) -> bool:
        if series.dtype != "object":
            return False
        sample = series.dropna().astype(str).head(50)
        if len(sample) == 0:
            return False
        hits = sum(
            any(ext in v.lower() for ext in self.IMAGE_EXTENSIONS)
            for v in sample
        )
        return hits > max(3, len(sample) * 0.3)

    def _is_text(self, series: pd.Series) -> bool:
        """
        Return True when the column looks like free-form text.

        Threshold: mean string length > 20 characters.
        (Lowered from 50 to detect short clinical text such as
        ECG reports, which average ~25 characters per entry.)
        """
        if series.dtype != "object":
            return False
        sample = series.dropna().astype(str).head(50)
        if len(sample) == 0:
            return False
        return sample.str.len().mean() > 20  # lowered: 50 missed short clinical text

    def _is_timeseries(self, series: pd.Series) -> bool:
        if series.dtype != "object":
            return False
        sample = series.dropna().astype(str).head(30)
        if len(sample) == 0:
            return False
        return sample.str.contains(r"\[.*\]", na=False).mean() > 0.5

    # -- mask column name detection ----------------------------------------

    _MASK_COLUMN_RE = re.compile(
        r"(?:^|_)(?:mask|seg_mask|segmentation|annotation|label_map)(?:$|_)",
        re.IGNORECASE,
    )

    def _is_mask_column(self, col_name: str) -> bool:
        """Return True when the column name suggests segmentation masks."""
        return bool(self._MASK_COLUMN_RE.search(col_name))

    # -----------------------------------------------------------------------
    # Target detection – 4-signal universal reasoning
    # -----------------------------------------------------------------------

    def _detect_target(
        self,
        df: pd.DataFrame,
    ) -> Tuple[str, float]:
        """
        Universal multi-signal target detection.

        Returns (best_column_name, confidence_score).
        """
        best_col: str = "Unknown"
        best_score: float = 0.0
        n_rows: int = max(len(df), 1)

        for col in df.columns:
            name = col.lower()
            series = df[col]
            score = 0.0

            # Signal 1 – semantic keyword match
            if any(k in name for k in self.TARGET_KEYWORDS):
                score += 0.35

            # Signal 2 – statistical behaviour (unique ratio)
            try:
                unique_ratio = series.nunique(dropna=True) / n_rows
            except Exception:
                unique_ratio = 1.0

            if 0.001 < unique_ratio < 0.2:
                score += 0.30
            if unique_ratio < 0.01:
                score += 0.20

            if series.dtype == "object":
                score += 0.10  # categorical hint

            # Signal 3 – structural pattern (JSON / list values → multi-label)
            if series.dtype == "object":
                sample = series.dropna().astype(str).head(50)
                if len(sample) > 0:
                    json_like = sample.str.contains(r"\{.*\}", na=False).mean()
                    list_like = sample.str.contains(r"\[.*\]", na=False).mean()
                    if json_like > 0.3:
                        score += 0.60
                    if list_like > 0.3:
                        score += 0.40

            # Signal 4 – regression signal (numeric, many unique, spread out)
            if pd.api.types.is_numeric_dtype(series):
                nunique = series.nunique(dropna=True)
                if nunique > 20 and unique_ratio > 0.05:
                    score += 0.25

            # Penalty for ID columns — use word-boundary check to avoid
            # false positives on words containing "id" (humidity, valid, rapid)
            import re as _re_sd
            if _re_sd.search(r'(?:^|_)id(?:$|_)', name):
                score -= 0.45

            if score > best_score:
                best_score = score
                best_col = col

        if best_score < 0.35:
            return "Unknown", 0.0

        return best_col, round(min(best_score, 1.0), 3)

    # -----------------------------------------------------------------------
    # Problem-type inference
    # -----------------------------------------------------------------------

    def _infer_problem(self, df: pd.DataFrame, target: str,
                        detected_columns: Optional[Dict[str, List[str]]] = None) -> str:
        """
        Infer the ML problem type from the target column.

        Rules (in priority order):
          0. Segmentation        – mask columns detected
          1. Multilabel – target values look like JSON dicts
          2. Binary classification      – 2 unique values
          3. Multiclass classification  – 3-20 unique integer values
          4. Regression                 – numeric float or >20 unique values
          5. Unsupervised               – no valid target detected
        """
        # Rule 0 – segmentation (mask column present)
        if detected_columns and detected_columns.get("mask"):
            return "segmentation"

        if target == "Unknown":
            return "unsupervised"

        s = df[target]

        # Rule 1 – multilabel (JSON dict values)
        if s.dtype == "object":
            sample = s.dropna().astype(str).head(50)
            if len(sample) > 0 and sample.str.contains(r"\{.*\}", na=False).mean() > 0.3:
                return "multilabel_classification"

        n_unique: int = int(s.nunique(dropna=True))

        # Rule 2 – binary
        if n_unique == 2:
            return "classification_binary"

        # Rule 3 – multiclass (small integer cardinality)
        if 3 <= n_unique <= 20:
            return "classification_multiclass"

        # Rule 4 – regression (float dtype OR many unique numeric values)
        if pd.api.types.is_numeric_dtype(s):
            if pd.api.types.is_float_dtype(s) or n_unique > 20:
                return "regression"
            # Integer with > 20 unique values but not clearly float → multiclass
            return "classification_multiclass"

        # Categorical string with > 20 unique values → multiclass
        return "classification_multiclass"

    # -----------------------------------------------------------------------
    # Tier-2 aggregation helpers
    # -----------------------------------------------------------------------

    def _aggregate_modalities(
        self,
        results: List[Dict[str, Any]],
    ) -> List[str]:
        """Union all per-dataset modalities into a sorted list."""
        mods: set = set()
        for r in results:
            mods.update(r.get("modalities", []))
        return sorted(mods)

    def _aggregate_problem_type(
        self,
        results: List[Dict[str, Any]],
    ) -> str:
        """
        Resolve a single global problem type.

        Priority rule: regression beats classification (a regression dataset
        mixed with a classification dataset should produce a regression run).
        Otherwise use majority vote.
        """
        types: List[str] = [
            r["problem_type"]
            for r in results
            if r.get("problem_type", "unsupervised") != "unsupervised"
        ]

        if not types:
            return "unsupervised"

        # Segmentation takes priority when detected
        if "segmentation" in types:
            return "segmentation"

        # Regression takes priority when mixed
        if "regression" in types:
            return "regression"

        return max(set(types), key=types.count)

    def _select_primary_target(
        self,
        results: List[Dict[str, Any]],
    ) -> str:
        """
        Pick the primary target column.

        Prefer the target with the highest confidence; fall back to first
        non-Unknown value; final fallback is "Unknown".
        """
        best_target = "Unknown"
        best_conf = -1.0
        for r in results:
            col = r.get("target_column", "Unknown")
            conf = r.get("confidence", 0.0)
            if col != "Unknown" and conf > best_conf:
                best_conf = conf
                best_target = col
        return best_target

    # -------------------------------------------------------------------
    # Cross-dataset relatedness
    # -------------------------------------------------------------------

    def _check_relatedness(
        self,
        per_dataset_results: List[Dict[str, Any]],
    ) -> Tuple[List[List[int]], Dict[str, Any]]:
        """
        Check pairwise relatedness of datasets and return groups.

        Signals (each contributes to a [0, 1] score):
          1. Column name overlap  (Jaccard) — weight 0.40
          2. Target column match             — weight 0.30
          3. Modality set overlap (Jaccard)  — weight 0.20
          4. Problem type match              — weight 0.10

        Datasets with pairwise score >= 0.5 are grouped via union-find.

        Returns
        -------
        (groups, report)
            groups : list of lists of dataset indices
            report : dict with pairwise scores for logging / frontend
        """
        n = len(per_dataset_results)
        if n <= 1:
            return [list(range(n))], {"single_dataset": True, "n_groups": 1}

        scores: Dict[Tuple[int, int], float] = {}
        for i in range(n):
            for j in range(i + 1, n):
                a = per_dataset_results[i]
                b = per_dataset_results[j]

                # Signal 1: column overlap (Jaccard)
                cols_a: set = set()
                cols_b: set = set()
                for mod_cols in a.get("detected_columns", {}).values():
                    cols_a.update(mod_cols)
                for mod_cols in b.get("detected_columns", {}).values():
                    cols_b.update(mod_cols)
                union = cols_a | cols_b
                col_jaccard = len(cols_a & cols_b) / len(union) if union else 0.0

                # Signal 2: target compatibility
                target_match = 1.0 if (
                    a.get("target_column", "X") == b.get("target_column", "Y")
                    and a.get("target_column") != "Unknown"
                ) else 0.0

                # Signal 3: modality match
                mods_a = set(a.get("modalities", []))
                mods_b = set(b.get("modalities", []))
                mod_union = mods_a | mods_b
                mod_jaccard = (
                    len(mods_a & mods_b) / len(mod_union) if mod_union else 0.0
                )

                # Signal 4: problem type match
                prob_match = (
                    1.0 if a.get("problem_type") == b.get("problem_type") else 0.0
                )

                score = (
                    0.40 * col_jaccard
                    + 0.30 * target_match
                    + 0.20 * mod_jaccard
                    + 0.10 * prob_match
                )
                scores[(i, j)] = score

        # Union-Find grouping at threshold 0.5
        parent = list(range(n))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(x: int, y: int) -> None:
            px, py = _find(x), _find(y)
            if px != py:
                parent[px] = py

        for (i, j), score in scores.items():
            if score >= 0.5:
                _union(i, j)

        from collections import defaultdict
        group_map: Dict[int, List[int]] = defaultdict(list)
        for i in range(n):
            group_map[_find(i)].append(i)

        groups = list(group_map.values())
        report = {
            "n_datasets": n,
            "n_groups": len(groups),
            "pairwise_scores": {
                f"{i}-{j}": round(s, 3) for (i, j), s in scores.items()
            },
            "groups": groups,
        }
        return groups, report
