from __future__ import annotations

import warnings
from typing import Any, Dict, List

import pandas as pd


class SemanticAnalyzer:
    """Deterministic semantic profiling for tabular dataframes."""

    def infer_column_roles(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        roles: Dict[str, List[str]] = {
            "id_columns": [],
            "categorical": [],
            "numeric": [],
            "text": [],
            "time_series": [],
        }

        if df is None or df.empty:
            return roles

        n_rows = max(1, len(df))

        for col in df.columns:
            series = df[col]
            col_l = str(col).lower()
            non_na = series.dropna()
            unique_ratio = float(non_na.nunique()) / max(1, len(non_na)) if len(non_na) else 0.0

            if self._is_time_series(col_l, series):
                roles["time_series"].append(col)
                continue

            if self._is_identifier(col_l, unique_ratio, non_na, series):
                roles["id_columns"].append(col)
                continue

            if pd.api.types.is_numeric_dtype(series):
                roles["numeric"].append(col)
                continue

            avg_len = float(non_na.astype(str).str.len().mean()) if len(non_na) else 0.0
            whitespace_ratio = (
                float(non_na.astype(str).str.contains(r"\s", regex=True).mean())
                if len(non_na)
                else 0.0
            )
            if avg_len > 20 or (avg_len > 12 and whitespace_ratio > 0.5) or unique_ratio > 0.5:
                roles["text"].append(col)
                continue

            if (not pd.api.types.is_numeric_dtype(series)) and (non_na.nunique() <= max(20, int(0.2 * n_rows))):
                roles["categorical"].append(col)
            else:
                roles["text"].append(col)

        return roles

    def detect_business_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df is None or df.empty:
            return {
                "has_time_axis": False,
                "duplicate_row_ratio": 0.0,
                "high_missing_columns": [],
                "long_tail_categoricals": [],
                "potential_target_candidates": [],
            }

        roles = self.infer_column_roles(df)
        missing_ratio = df.isna().mean()

        long_tail: List[str] = []
        for col in roles["categorical"]:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            top1 = series.value_counts(normalize=True).iloc[0]
            if float(top1) < 0.4:
                long_tail.append(col)

        target_candidates: List[str] = []
        for col in roles["categorical"]:
            n_unique = df[col].nunique(dropna=True)
            if 2 <= n_unique <= 20:
                target_candidates.append(col)

        return {
            "has_time_axis": bool(roles["time_series"]),
            "duplicate_row_ratio": float(df.duplicated().mean()),
            "high_missing_columns": [
                col for col, ratio in missing_ratio.items() if float(ratio) > 0.3
            ],
            "long_tail_categoricals": long_tail,
            "potential_target_candidates": target_candidates,
        }

    @staticmethod
    def _is_identifier(
        col_name: str,
        unique_ratio: float,
        non_na: pd.Series,
        original_series: pd.Series,
    ) -> bool:
        if any(token in col_name for token in ["id", "uuid", "guid", "pk", "key"]):
            return True
        if pd.api.types.is_numeric_dtype(original_series):
            return False
        if unique_ratio > 0.98 and len(non_na) > 20:
            return True
        return False

    @staticmethod
    def _is_time_series(col_name: str, series: pd.Series) -> bool:
        if any(token in col_name for token in ["date", "time", "timestamp", "datetime"]):
            return True
        if pd.api.types.is_datetime64_any_dtype(series):
            return True

        non_na = series.dropna()
        if len(non_na) < 5:
            return False

        if non_na.dtype == "object":
            sample = non_na.astype(str).head(200)
            looks_temporal = sample.str.contains(
                r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}:\d{2}|T\d{2}:\d{2}",
                regex=True,
                na=False,
            ).mean() > 0.6
            if not looks_temporal:
                return False
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(sample, errors="coerce", utc=False)
            parse_ratio = float(parsed.notna().mean())
            return parse_ratio > 0.85

        return False
