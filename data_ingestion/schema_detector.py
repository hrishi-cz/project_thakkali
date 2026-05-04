from __future__ import annotations

import logging
import re as _re_import
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Verb-prefix patterns: "is_fraud", "will_churn", "has_disease", "did_convert"
_VERB_PREFIX_RE = _re_import.compile(
    r'^(is|has|will|did|was|had|can|should|would|got|get|flag)_',
    _re_import.IGNORECASE,
)
# Suffix patterns strongly associated with prediction targets
_TARGET_SUFFIX_RE = _re_import.compile(
    r'_(flag|indicator|label|binary|outcome|result|target|class|category|'
    r'status|decision|prediction|verdict|signal|event|response|output|'
    r'survived|converted|purchased|clicked|churned|defaulted|approved)$',
    _re_import.IGNORECASE,
)


from data_ingestion.schema import GlobalSchema, IndividualSchema
from data_ingestion.integrator import Integrator
from data_ingestion.data_bridge import materialize_sample
from data_ingestion.semantic_analyzer import SemanticAnalyzer
from data_ingestion.xs3_target_selector import XS3TargetSelector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TargetScore dataclass (used throughout for candidate scoring)
# ---------------------------------------------------------------------------
@dataclass
class TargetScore:
    column: str
    final_score: float
    nan_ratio: float = 0.0
    valid: bool = True
    reason: str = "Valid"
    quality: float = 0.0
    semantic_score: float = 0.0
    semantics: Dict[str, Any] = field(default_factory=dict)
    explanation: List[str] = field(default_factory=list)
    semantic_role: str = ""
    interaction_score: float = 0.0
    uncertainty: float = 0.0
    # --- Previously missing fields ---
    keyword_score: float = 0.0
    uniqueness_score: float = 0.0
    regression_score: float = 0.0
    json_score: float = 0.0
    predictability_score: float = 0.0
    complementarity_score: float = 0.0
    cross_dataset_score: float = 0.0
    degeneracy_penalty: float = 0.0


# ===========================================================================
# Main Engine
# ===========================================================================

class COGMASchemaDetector:
    """
    COGMA-ready schema detector: 6-stage intelligence pipeline.
    All detection/validation flows through Integrator. No heuristics.
    """

    TARGET_KEYWORDS = [
        # ML standard
        "target", "label", "class", "output", "code",
        # Medical / clinical
        "diagnosis", "outcome", "prognosis", "disease", "condition", "grade",
        "severity", "risk", "mortality", "survived", "survival",
        # Business / e-commerce
        "churn", "fraud", "default", "conversion", "purchased", "clicked",
        "revenue", "approved", "cancelled",
        # Scientific / research
        "result", "response", "category", "type", "status", "group", "arm",
        # NLP / annotation
        "sentiment", "intent", "entity", "tag", "annotation", "toxicity",
    ]
    TARGET_SUFFIX_KEYWORDS = ["id", "val", "score"]
    IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    _BINARY_ATTRIBUTE_VALUES = {"yes", "no", "true", "false", "0", "1", "m", "f"}

    def __init__(self, fix4_engine=None):
        self.fix4_engine = fix4_engine
        self.last_target_candidates: List[TargetScore] = []
        self.semantic_analyzer = SemanticAnalyzer()

    def _build_global_schema(self, per_dataset_results: List[Dict[str, Any]]) -> GlobalSchema:
        """Aggregate per-dataset results into a GlobalSchema."""
        global_modalities = sorted({m for s in per_dataset_results for m in s.get("modalities", [])})
        
        if per_dataset_results:
            primary_target = max(per_dataset_results, key=lambda s: s.get("confidence", 0.0)).get("target_column", "Unknown")
            detection_confidence = float(np.mean([s.get("confidence", 0.0) for s in per_dataset_results]))
        else:
            primary_target = "Unknown"
            detection_confidence = 0.0
            
        fusion_ready = len(global_modalities) > 1

        global_schema = GlobalSchema(
            global_problem_type="classification_multiclass",  # Could be refined
            global_modalities=global_modalities,
            primary_target=primary_target,
            fusion_ready=fusion_ready,
            detection_confidence=round(detection_confidence, 3),
            per_dataset=per_dataset_results,
            multimodal_signals=self._compute_cross_modality_signals(per_dataset_results),
        )
        return global_schema

    def _select_primary_target(self, per_dataset_results: List[Dict[str, Any]]) -> str:
        """
        Select the primary target column across datasets, boosting targets that appear in multiple datasets.
        """
        target_scores = {}  # Initialize target scores
        target_appearances = {}

        for s in per_dataset_results:
            t = s.get("target_column")
            score = s.get("confidence", 0)
            if t and t != "Unknown":
                target_scores[t] = target_scores.get(t, 0) + score
                target_appearances[t] = target_appearances.get(t, 0) + 1

        # Boost score for targets appearing in multiple datasets (cross-dataset bonus)
        for target in target_scores:
            count = target_appearances[target]
            # Linear boost: appearing in N datasets → +0.1 * (N-1)
            cross_dataset_bonus = 0.1 * max(0, count - 1)
            target_scores[target] += cross_dataset_bonus

        if target_scores:
            return max(target_scores, key=target_scores.get)
        return "Unknown"

    def _compute_modality_importance(self, per_dataset_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute the importance of each modality based on confidence scores across datasets.
        """
        scores = {"tabular": 0.0, "text": 0.0, "image": 0.0, "timeseries": 0.0}
        for ds in per_dataset_results:
            conf = float(ds.get("confidence", 0.0))
            for m in ds.get("modalities", []):
                if m in scores:
                    scores[m] += conf
        total = sum(scores.values()) or 1.0
        return {k: round(v / total, 3) for k, v in scores.items()}

    # -----------------------------------------------------------------------
    # PATCH C1: Image target scoring
    # -----------------------------------------------------------------------

    def _score_image_target_candidates(
        self, sample_df: pd.DataFrame, image_cols: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Score candidate target columns for image datasets. Returns ranked list of candidate dicts with final_score.
        """
        candidates = []
        non_image_cols = [c for c in sample_df.columns if c not in image_cols]

        for col in non_image_cols:
            series = sample_df[col].dropna()
            if len(series) == 0:
                continue

            n_unique   = series.nunique()
            nan_ratio  = 1.0 - len(series) / max(len(sample_df), 1)
            col_lower  = col.lower()

            # 1. Cardinality score → prefer [2, 100]
            if n_unique < 2:
                cardinality_score = 0.0
            elif n_unique <= 10:
                cardinality_score = 1.0
            elif n_unique <= 50:
                cardinality_score = 0.7
            elif n_unique <= 100:
                cardinality_score = 0.4
            else:
                cardinality_score = 0.1  # likely a path/id column

            # 2. Class balance score (Gini-based evenness)
            try:
                val_counts = series.value_counts(normalize=True)
                balance = 1.0 - float(val_counts.std())  # high std → imbalanced
                balance_score = max(0.0, min(1.0, balance))
            except Exception:
                balance_score = 0.5

            # 3. Semantic keyword match
            sem_hits = [kw for kw in self.TARGET_KEYWORDS if kw in col_lower]
            semantic_score = min(1.0, len(sem_hits) * 0.4)

            # 4. Bbox / JSON detection penalty
            try:
                sample_str = str(series.iloc[0])
                is_bbox = sample_str.startswith("{") or "xmin" in col_lower or "bbox" in col_lower
                bbox_penalty = 0.3 if is_bbox else 0.0
            except Exception:
                bbox_penalty = 0.0

            final = (
                0.35 * cardinality_score
                + 0.25 * balance_score
                + 0.25 * semantic_score
                - 0.15 * nan_ratio
                - bbox_penalty
            )

            candidates.append({
                "column":           col,
                "cardinality_score": round(cardinality_score, 3),
                "balance_score":    round(balance_score, 3),
                "semantic_score":   round(semantic_score, 3),
                "nan_ratio":        round(nan_ratio, 3),
                "final_score":      round(max(0.0, final), 3),
                "n_unique":         int(n_unique),
            })

        return sorted(candidates, key=lambda x: x["final_score"], reverse=True)

    # -----------------------------------------------------------------------
    # PATCH C2: Text target scoring
    # -----------------------------------------------------------------------

    def _score_text_target_candidates(
        self, sample_df: pd.DataFrame, text_cols: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Score candidate target columns for text/NLP datasets. Returns ranked list of candidate dicts with type + final_score.
        """
        import re

        _IOB_PATTERN  = re.compile(r"^(B|I|O)-[A-Z]{1,10}$")
        _SEQ2SEQ_TOKENS = {"</s>", "[SEP]", "<eos>", "[CLS]", "<pad>"}

        candidates = []
        non_text_content_cols = [c for c in sample_df.columns if c not in text_cols]

        for col in non_text_content_cols:
            series   = sample_df[col].dropna().astype(str)
            if len(series) == 0:
                continue

            col_lower = col.lower()
            n_unique  = series.nunique()
            nan_ratio = 1.0 - series.count() / max(len(sample_df), 1)
            sample_vals = series.head(50).tolist()

            target_type    = "classification"
            type_conf      = 0.0
            semantic_score = min(1.0, sum(kw in col_lower for kw in self.TARGET_KEYWORDS) * 0.35)

            # --- Classification detection ---
            if 2 <= n_unique <= 50:
                target_type = "text_classification"
                if n_unique <= 5:
                    type_conf = 0.9
                elif n_unique <= 20:
                    type_conf = 0.7
                else:
                    type_conf = 0.5

            # --- NER / IOB detection (overrides classification) ---
            iob_matches = sum(
                1 for v in sample_vals
                if isinstance(v, str) and any(_IOB_PATTERN.match(tok) for tok in v.split())
            )
            if iob_matches / max(len(sample_vals), 1) > 0.3:
                target_type = "ner_sequence"
                type_conf   = min(1.0, iob_matches / len(sample_vals) * 1.5)

            # --- Seq2seq detection ---
            avg_len = float(series.str.len().mean())
            special_tok_hits = sum(1 for v in sample_vals if any(t in v for t in _SEQ2SEQ_TOKENS))
            if avg_len > 30 and (special_tok_hits > 0 or "output" in col_lower or "response" in col_lower):
                target_type = "seq2seq"
                type_conf   = 0.65 + 0.1 * min(1.0, special_tok_hits / max(len(sample_vals), 1))

            final = (
                0.40 * type_conf
                + 0.30 * semantic_score
                - 0.15 * nan_ratio
                - (0.20 if target_type == "seq2seq" and n_unique > 100 else 0.0)  # penalise unique seq2seq
            )

            candidates.append({
                "column":      col,
                "target_type": target_type,
                "type_conf":   round(type_conf, 3),
                "semantic_score": round(semantic_score, 3),
                "n_unique":    int(n_unique),
                "avg_len":     round(avg_len, 1),
                "nan_ratio":   round(nan_ratio, 3),
                "final_score": round(max(0.0, final), 3),
            })

        return sorted(candidates, key=lambda x: x["final_score"], reverse=True)

    def _infer_problem_from_override(
        self, col: str, datasets: Dict[str, Any]
    ) -> str:
        """
        Re-infer problem type from an override column's unique values. Materialises only the single target column (memory-safe).
        """
        try:
            import polars as pl
            for lazy_ref in datasets.values():
                if isinstance(lazy_ref, pl.LazyFrame):
                    schema_names = lazy_ref.collect_schema().names()
                    if col not in schema_names:
                        continue
                    series = lazy_ref.select(col).head(5000).collect()[col]
                    n_unique = series.n_unique()
                    if series.dtype in (pl.Utf8, pl.String, pl.Categorical):
                        return "classification_multiclass" if n_unique > 2 else "classification_binary"
                    if n_unique <= 2:
                        return "classification_binary"
                    if n_unique <= 20:
                        return "classification_multiclass"
                    return "regression"
        except Exception:
            pass
        return "classification_multiclass"  # safe default for unknown

    # -----------------------------------------------------------------------
    # Tier-1: single-dataset inspector
    # -----------------------------------------------------------------------

    def _detect_single(
        self,
        dataset_id: str,
        lazy_data: Any,
        target_override: Optional[str] = None,
    ) -> IndividualSchema:
        """
        Tier-1: inspect one lazy dataset. Materialises at most 500 rows for
        heuristic computation. For PyTorch Datasets (image-only) it returns an
        image-modality schema directly without any column analysis.

        Args:
            target_override: When set, bypass auto target detection and mark
                             this column as the target directly.
        """
        sample_df: Optional[pd.DataFrame] = self._materialise_sample(
            lazy_data, n=500
        )

        if sample_df is None:
            # PyTorch Dataset or unrecognised type → treat as image dataset
            from data_ingestion.loader import detect_image_structure
            from pathlib import Path

            cache_path = Path("./data/dataset_cache") / dataset_id
            structure = detect_image_structure(cache_path)

            if structure["type"] == "classification":
                target_col = "__image_label__"
                prob_type = "classification_multiclass"
                conf = 0.9
                reasoning = {"reason": "Detected class folders", "selected": {"column": "__image_label__", "final_score": 0.9}}
                candidates = [{"column": "__image_label__", "final_score": 0.9, "reason": "Detected class folders"}]
            else:
                target_col = "Unknown"
                prob_type = "unsupervised"
                conf = 0.5
                reasoning = {"reason": "No label structures detected"}
                candidates = [
                    {"column": "__image_label__", "final_score": 0.0, "reason": "No folder patterns"},
                    {"column": "__filename_pattern__", "final_score": 0.0, "reason": "No matched pattern"},
                    {"column": "__unsupervised__", "final_score": 0.5, "reason": "Default fallback"}
                ]

            return IndividualSchema(
                dataset_id=dataset_id,
                detected_columns={
                    "image": ["__image_path__"],
                    "text": [],
                    "tabular": [],
                    "timeseries": [],
                },
                target_column=target_col,
                problem_type=prob_type,
                modalities=["image"],
                confidence=conf,
                reasoning=reasoning,
                candidates=candidates,
            )

        return self._inspect_dataframe(dataset_id, sample_df, target_override=target_override)

    # -----------------------------------------------------------------------
    # Advanced Semantic Target Heuristics (Blueprint)
    # -----------------------------------------------------------------------

    def _safe_div(self, a: float, b: float) -> float:
        return float(a) / float(b) if b else 0.0

    def _nan_ratio(self, s: pd.Series) -> float:
        return float(s.isna().mean()) if len(s) else 1.0

    def _unique_ratio(self, s: pd.Series) -> float:
        return self._safe_div(s.nunique(dropna=True), max(len(s), 1))

    def _avg_len(self, s: pd.Series) -> float:
        try:
            sample = s.dropna().astype(str).head(50)
            return float(sample.str.len().mean()) if len(sample) else 0.0
        except Exception:
            return 0.0

    def _json_ratio(self, s: pd.Series) -> float:
        sample = s.dropna().astype(str).head(50)
        if len(sample) == 0:
            return 0.0
        return float(sample.str.contains(r"^\s*\{.*\}\s*$", na=False).mean())

    def _list_ratio(self, s: pd.Series) -> float:
        sample = s.dropna().astype(str).head(50)
        if len(sample) == 0:
            return 0.0
        return float(sample.str.contains(r"^\s*\[.*\]\s*$", na=False).mean())

    def _looks_like_path(self, s: pd.Series) -> float:
        sample = s.dropna().astype(str).head(50)
        if len(sample) == 0:
            return 0.0
        return float(sample.str.contains(r"[/\\]|\.(?:png|jpg|jpeg|bmp|tif|tiff|csv|json|parquet)$", case=False, regex=True).mean())

    def _is_image_path_series(self, s: pd.Series) -> bool:
        return self._looks_like_path(s) > 0.5 and self._avg_len(s) < 250

    def _is_text_series(self, s: pd.Series) -> bool:
        if s.dtype != "object":
            return False
        return self._avg_len(s) > 30 and self._unique_ratio(s) > 0.1

    def _is_structured_label(self, s: pd.Series) -> bool:
        if s.dtype != "object":
            return False
        return (self._json_ratio(s) > 0.2) or (self._list_ratio(s) > 0.2)

    def _target_quality_score(self, s: pd.Series, name: str) -> float:
        unique_ratio = self._unique_ratio(s)
        nan_ratio = self._nan_ratio(s)
        name_l = name.lower()
        score = 0.0
        if nan_ratio < 0.5:
            score += 0.20
        if 2 <= s.nunique(dropna=True) <= 50:
            score += 0.25
        if self._is_structured_label(s):
            score += 0.35
        if s.dtype == "object" and self._avg_len(s) < 50:
            score += 0.10
        if unique_ratio > 0.98 and s.dtype != "object":
            score -= 0.35
        if any(k in name_l for k in ["target", "label", "class", "output", "code", "diagnosis", "result"]):
            score += 0.10
        return float(max(0.0, min(1.0, score)))

    def _infer_image_schema(self, dataset_id: str, lazy_data: Any) -> IndividualSchema:
        targets = getattr(lazy_data, "targets", None) or getattr(lazy_data, "labels", None)
        semantic = self._image_label_quality(lazy_data)

        detected = {"image": ["__image__"], "text": [], "tabular": [], "timeseries": []}
        reasoning = {"mode": "image", "reason": [], "notes": []}

        if semantic == 1.0 and targets is not None:
            n_classes = len(getattr(lazy_data, "classes", []))
            problem_type = "classification_binary" if n_classes == 2 else "classification_multiclass"
            reasoning["reason"].append("Detected class metadata in image dataset")
            reasoning["notes"].append(f"{n_classes} classes found")

            return IndividualSchema(
                dataset_id=dataset_id,
                detected_columns=detected,
                target_column="__image_label__",
                problem_type=problem_type,
                modalities=["image"],
                confidence=0.90,
                target_profile={"semantic_score": 1.0, "predictability_score": 0.8, "quality_score": 1.0},
                reasoning=reasoning,
                candidates=[{"column": "__image_label__", "final_score": 1.0, "reason": "Image class metadata detected", "valid": True}],
                rejected_candidates=[],
                preprocessing_hints={"image": {"mode": "supervised", "resize": [224, 224], "augment": True, "label_source": "metadata"}}
            )

        reasoning["reason"].append("No explicit image labels found")
        reasoning["notes"].append("Fallback to self-supervised / representation learning")

        return IndividualSchema(
            dataset_id=dataset_id,
            detected_columns=detected,
            target_column="Unknown",
            problem_type="unsupervised",
            modalities=["image"],
            confidence=0.50,
            target_profile={"semantic_score": 0.0, "predictability_score": 0.0, "quality_score": 0.0},
            reasoning=reasoning,
            candidates=[],
            rejected_candidates=[],
            preprocessing_hints={"image": {"mode": "self_supervised", "resize": [224, 224], "augment": True, "label_source": None}}
        )

    def _build_preprocessing_hints(self, modalities: List[str], target_col: str, problem_type: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        best = candidates[0] if candidates else {}
        target_type = "semantic" if best.get("semantic_score", 0) > 0.4 else "predictive"
        hints = {
            "target_type": target_type,
            "feature_selection": "minimal" if target_type == "semantic" else "strict",
            "tabular": {
                "use_mi_shap": True,
                "top_k_ratio": 0.8 if target_type == "semantic" else 0.4,
                "keep_interactions": True
            },
            "text": {
                "mode": "structured_label" if best.get("list_ratio", 0) > 0.2 or best.get("json_ratio", 0) > 0.2 else "free_text",
                "max_length": 256 if target_type == "semantic" else 128
            },
            "image": {
                "mode": "supervised" if problem_type.startswith("classification") else "self_supervised",
                "resize": [256, 256] if target_type == "semantic" else [224, 224],
                "augment": True
            },
            "multimodal": {
                "fusion_ready": len([m for m in modalities if m in ["tabular", "text", "image"]]) > 1,
                "weights": {
                    "tabular": 0.5 if "tabular" in modalities else 0.0,
                    "text": 0.4 if "text" in modalities else 0.0,
                    "image": 0.1 if "image" in modalities else 0.0,
                }
            }
        }
        return hints

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
        target_override: Optional[str] = None,
    ) -> IndividualSchema:
        detected: Dict[str, List[str]] = {"image": [], "text": [], "tabular": [], "timeseries": []}
        candidates: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []

        semantic_summary = {}
        interaction_summary = {}
        uncertainty_summary = {}

        for col in df.columns:
            s = df[col]
            if self._is_image(s, col_name=str(col)):
                detected["image"].append(col)
            elif self._is_timeseries(s):
                detected["timeseries"].append(col)
            elif self._is_text(s):
                detected["text"].append(col)
            else:
                detected["tabular"].append(col)

        # Determine which columns are confirmed content features (never targets)
        # Image path columns: file paths, never labels
        _image_cols = set(detected.get("image", []))
        # Long-text content columns: documents / captions / reviews
        # (avg_len > 80 AND many unique values → document, not label)
        _content_text_cols: set = set()
        for _col in detected.get("text", []):
            _s = df[_col]
            _avg = self._avg_len(_s)
            _ur  = self._unique_ratio(_s)
            if _avg > 80 and _ur > 0.05:
                _content_text_cols.add(_col)

        # score candidates across all modalities
        for col in df.columns:
            s = df[col]
            # Image path columns are never targets
            if col in _image_cols:
                rejected.append({
                    "column": col, "final_score": 0.0, "valid": False,
                    "reason": "Image path feature — not a target column",
                })
                continue
            # Long-text content features are suppressed as target candidates
            _is_content_feature = col in _content_text_cols

            profile: Dict[str, Any] = {
                "column": col,
                "dtype": "text" if self._is_text(s) else ("image" if self._is_image(s, col_name=str(col)) else "tabular"),
                "nan_ratio": self._nan_ratio(s),
                "unique_ratio": self._unique_ratio(s),
                "avg_len": self._avg_len(s),
                "json_ratio": self._json_ratio(s),
                "list_ratio": self._list_ratio(s),
            }

            quality = self._target_quality_score(s, col)

            # --- New Paper-Aligned Telemetry ---
            interaction_score = self._compute_interaction_score(df, col)
            uncertainty = self._compute_uncertainty(s)
            semantic_role = self._infer_semantic_role(col, s)

            semantic_summary[col] = semantic_role
            interaction_summary[col] = interaction_score
            uncertainty_summary[col] = uncertainty

            fix4_scores: Dict[str, Any] = {}

            if self.fix4_engine is not None:
                if pd.api.types.is_numeric_dtype(s):
                    problem_type = "regression"
                else:
                    problem_type = (
                        "classification_binary"
                        if s.nunique(dropna=True) == 2
                        else "classification_multiclass"
                    )

                modality_map = {
                    "text": detected.get("text", []),
                    "image": detected.get("image", []),
                    "tabular": detected.get("tabular", []),
                }

                fix4_scores = self.fix4_engine.score_target_candidates_fix4(
                    df[[col] + [c for c in df.columns if c != col]],
                    [col],
                    problem_type,
                    modality_map
                )

                if col in fix4_scores:
                    fix4_col_scores = fix4_scores[col]
                    predictability = fix4_col_scores.get("predictability_score", 0.0)
                    complementarity = fix4_col_scores.get("complementarity_score", 0.0)
                    semantic = fix4_col_scores.get("semantic_score", 0.0)
                    reason = "FIX-4: Learning-based target validation"
                    logger.debug(
                        "FIX-4 scoring [%s]: pred=%.3f, comp=%.3f, sem=%.3f",
                        col, predictability, complementarity, semantic
                    )
                else:
                    predictability = self._predictability_score(df, col)
                    complementarity = self._complementarity_score(df, col)
                    semantic = 0.0
                    reason = "Heuristic fallback"
            else:
                predictability = self._predictability_score(df, col)
                complementarity = self._complementarity_score(df, col)

            if self.fix4_engine is None or col not in fix4_scores:
                if profile["dtype"] == "text":
                    semantic = 0.0
                    if self._is_structured_label(s):
                        semantic += 0.60
                    if profile["avg_len"] < 50 and 2 <= s.nunique(dropna=True) <= 50:
                        semantic += 0.30
                    if profile["avg_len"] > 100:
                        semantic -= 0.15
                    reason = "Text semantic score"
                    # Boost predictability with text-specific TF-IDF probe (numeric probe returns 0 for text)
                    text_pred = self._text_predictability_score(s, df)
                    if text_pred > predictability:
                        predictability = text_pred

                elif profile["dtype"] == "image":
                    semantic = 0.0
                    if self._looks_like_path(s) > 0.5:
                        semantic += 0.20
                    reason = "Image-path feature (not usually target)"

                else:
                    semantic = 0.0
                    if self._is_structured_label(s):
                        semantic += 0.35
                    if 2 <= s.nunique(dropna=True) <= 50:
                        semantic += 0.20
                    # Exact match to universally-recognised target column names
                    _col_lower = col.strip().lower()
                    _EXACT_TARGET_NAMES = {
                        "label", "target", "class", "category", "output",
                        "diagnosis", "sentiment", "outcome", "result", "type",
                        "status", "group", "tag", "annotation", "scp_codes",
                        "survived", "defaulted", "churned", "converted",
                        # Common real-world dataset target columns
                        "income", "salary", "price", "sold", "fraud",
                        "hate", "toxic", "spam", "clicked", "purchased",
                        "approved", "accepted", "hired", "admitted",
                        "default", "churn", "attrition", "readmitted",
                        "genre", "rating", "score", "grade", "pass",
                    }
                    if _col_lower in _EXACT_TARGET_NAMES:
                        semantic = 1.0
                        predictability = min(1.0, predictability + 0.15)
                    reason = "Tabular target quality"

            # ── Dataset-level understanding signals ─────────────────────
            # Verb-phrase / suffix name pattern (language-level target signal)
            verb_score = self._verb_phrase_score(col)
            semantic = min(1.0, semantic + verb_score * 0.5)

            # MI-based predictability (handles mixed types; blended with RF probe)
            mi_pred = self._mi_predictability_score(df, col)
            predictability = max(predictability, mi_pred * 0.85)

            # Causal direction: targets are predicted BY features, not predictors OF them
            directionality = self._directionality_score(df, col, forward_predictability=predictability)

            # Value-set fingerprint: read actual cell values for semantic understanding
            fingerprint = self._value_set_fingerprint(s)

            # Leakage penalty: suspiciously perfect predictability = derived column
            leakage_pen = self._leakage_penalty(predictability)

            # Content-feature penalty: long text document columns are features, not targets
            content_pen = 0.35 if _is_content_feature else 0.0

            # Cross-modal convergence: for multimodal datasets, how many modalities
            # independently predict this column? Only computed when 2+ modalities present
            _active_modalities = [m for m, cols in detected.items() if cols]
            if len(_active_modalities) >= 2 and not _is_content_feature:
                convergence = self._cross_modal_convergence(df, col, detected)
            else:
                convergence = 0.0

            # X-S³ Scoring — multimodal-aware, dataset-understanding enhanced
            final = (
                0.18 * predictability +
                0.15 * complementarity +
                0.11 * semantic +
                0.11 * interaction_score +
                0.07 * 0.0 +        # cross_dataset defaults 0 here
                0.11 * (1.0 - uncertainty) +
                0.11 * directionality +   # causal direction
                0.06 * fingerprint +      # value-set understanding
                0.10 * convergence        # cross-modal convergent evidence
            ) - leakage_pen - content_pen

            profile.update({
                "semantic_score": semantic,
                "predictability_score": predictability,
                "quality_score": quality,
                "interaction_score": interaction_score,
                "uncertainty": uncertainty,
                "semantic_role": semantic_role,
                "final_score": float(final),
                "reason": reason,
                "valid": final >= 0.20,
            })

            if profile["valid"]:
                candidates.append(profile)
            else:
                rejected.append(profile)

        def validate_override(df_local, target):
            """Enhanced validation with support for edge-case targets (multilabel, hierarchical)."""
            if target not in df_local.columns:
                return False, "Target not found"
            if df_local[target].isna().mean() > 0.8:
                return False, "Too many NaNs"
            if df_local[target].nunique(dropna=True) <= 1:
                return False, "No variance"

            try:
                series = df_local[target].dropna()
                if len(series) > 0:
                    sample = series.astype(str).head(50)
                    is_json = sample.str.contains(r'\{.*\}|\[.*\]', na=False).mean() > 0.3
                    if is_json:
                        logger.info(f"Target '{target}' detected as structured/multilabel")
                        return True, "Valid (structured target)"
            except Exception:
                pass

            return True, "Valid"

        if target_override:
            valid, msg = validate_override(df, target_override)
            if valid:
                target_col = target_override
                best_cand = next((c for c in candidates if c.get("column") == target_override), None)
                orig_conf = best_cand.get("final_score", 0.0) if best_cand else 0.0
                confidence = max(orig_conf, 0.5)
                if best_cand:
                    candidates.remove(best_cand)
                    candidates.insert(0, best_cand)
                else:
                    candidates.insert(0, {"column": target_override, "final_score": confidence, "reason": "User Override applied", "valid": True})
            else:
                logger.warning("Override rejected: %s", msg)
                target_override = None

        if not target_override:
            if not candidates:
                target_col = "Unknown"
                confidence = 0.0
            else:
                selection = XS3TargetSelector().select(candidates)
                candidates = selection["ranked_candidates"]
                target_col = selection["target_column"]
                confidence = float(selection["xs3_confidence_gap"])

        problem_type = self._infer_problem(df, target_col)
        modalities = sorted(k for k, v in detected.items() if v)

        avg_text_len = 0.0
        if detected.get("text"):
            text_lengths: List[float] = []
            for text_col in detected["text"]:
                if text_col not in df.columns:
                    continue
                try:
                    text_sample = df[text_col].dropna().astype(str)
                    if len(text_sample) == 0:
                        continue
                    text_lengths.append(float(text_sample.str.len().mean()))
                except Exception:
                    continue
            if text_lengths:
                avg_text_len = float(np.mean(text_lengths))

        text_task_type: Optional[str] = None
        if detected.get("text") and target_col in df.columns:
            text_task_type = self._infer_text_task_type(target_col, df[target_col])
            if text_task_type is None and problem_type.startswith("classification"):
                text_task_type = "text_classification"

        image_dataset_size = 0
        image_label_separability = 0.0
        image_class_balance = 0.0
        if detected.get("image"):
            try:
                image_mask = pd.Series(False, index=df.index)
                for image_col in detected["image"]:
                    if image_col not in df.columns:
                        continue
                    series = df[image_col].fillna("").astype(str).str.strip()
                    valid = series.ne("") & ~series.str.lower().isin({"nan", "none", "null", "<na>"})
                    image_mask = image_mask | valid
                image_dataset_size = int(image_mask.sum())
            except Exception:
                image_dataset_size = 0

            if target_col in df.columns:
                image_label_separability = float(
                    self._image_label_separability_score(df, target_col)
                )
                image_class_balance = float(
                    self._gini_class_balance(df[target_col])
                )

        reasoning = {
            "selected": candidates[0] if candidates else {},
            "why_not_others": [
                {"column": c["column"], "reason": c.get("reason", ""), "score": c.get("final_score", 0)}
                for c in candidates[1:5]
            ],
            "confidence_gap": confidence,
            "xs3_confidence_gap": confidence,
            "avg_text_len": round(avg_text_len, 2),
        }

        preprocessing_hints = self._build_preprocessing_hints(modalities, target_col, problem_type, candidates)
        if avg_text_len > 0:
            text_hints = dict(preprocessing_hints.get("text", {}) or {})
            text_hints.setdefault("avg_text_len", round(avg_text_len, 2))
            if text_task_type:
                text_hints["task_type"] = text_task_type
            preprocessing_hints["text"] = text_hints
        if detected.get("image"):
            image_hints = dict(preprocessing_hints.get("image", {}) or {})
            image_hints.setdefault("dataset_size", int(image_dataset_size))
            image_hints.setdefault("label_separability", round(image_label_separability, 4))
            image_hints.setdefault("class_balance", round(image_class_balance, 4))
            preprocessing_hints["image"] = image_hints
        semantic_roles = self.semantic_analyzer.infer_column_roles(df)
        business_patterns = self.semantic_analyzer.detect_business_patterns(df)

        return IndividualSchema(
            dataset_id=dataset_id,
            detected_columns=detected,
            target_column=target_col,
            problem_type=problem_type,
            modalities=modalities,
            confidence=round(confidence, 3),
            target_profile=(candidates[0] if candidates else {}),
            reasoning=reasoning,
            candidates=candidates,
            rejected_candidates=rejected,
            preprocessing_hints=preprocessing_hints,
            selection_mode="manual_override" if target_override else "auto",
            semantic_summary=semantic_summary,
            interaction_summary=interaction_summary,
            uncertainty_summary=uncertainty_summary,
            semantic_roles=semantic_roles,
            business_patterns=business_patterns,
            text_task_type=text_task_type,
            num_features=int(df.shape[1]),
            has_relational_columns=bool(semantic_roles.get("id_columns")),
            image_label_separability=float(image_label_separability),
            image_class_balance=float(image_class_balance),
            image_dataset_size=int(image_dataset_size),
            feature_signals=self._build_feature_signals(df, detected),
        )


    # -----------------------------------------------------------------------
    # G7: Text feature signals helper
    # -----------------------------------------------------------------------

    @staticmethod
    def _extract_text_feature_signals(df, text_cols):
        """Compute text-modality feature signals for preprocessing context (G7)."""
        if not text_cols:
            return {}
        try:
            all_tokens = []
            all_vocab = set()
            all_type_token = []
            long_doc_count = 0
            for col in text_cols:
                if col not in df.columns:
                    continue
                sample = df[col].dropna().astype(str).head(500)
                if len(sample) == 0:
                    continue
                token_counts = sample.str.split().str.len().dropna()
                if len(token_counts) > 0:
                    all_tokens.append(float(token_counts.mean()))
                words = " ".join(sample.tolist()).split()
                all_vocab.update(words)
                if words:
                    ttr = len(set(words)) / max(1, len(words))
                    all_type_token.append(ttr)
                long_doc_count += int((token_counts > 200).sum())

            signals = {}
            if all_tokens:
                signals["avg_tokens_per_sample"] = round(float(sum(all_tokens) / len(all_tokens)), 2)
            signals["vocab_size"] = len(all_vocab)
            if all_type_token:
                signals["linguistic_complexity"] = round(
                    float(sum(all_type_token) / len(all_type_token)), 4
                )
            signals["long_doc_indicator"] = long_doc_count > 0
            try:
                from langdetect import detect as _ld_detect
                lang_sample = df[text_cols[0]].dropna().astype(str).head(20).tolist()
                langs = []
                for txt in lang_sample[:10]:
                    try:
                        langs.append(_ld_detect(txt[:200]))
                    except Exception:
                        pass
                if langs:
                    from collections import Counter
                    signals["language_id"] = Counter(langs).most_common(1)[0][0]
            except ImportError:
                pass
            return signals
        except Exception:
            return {}

    # -----------------------------------------------------------------------
    # G8: Image feature signals helper
    # -----------------------------------------------------------------------

    @staticmethod
    def _extract_image_feature_signals(df, image_cols):
        """Compute image-modality feature signals for preprocessing context (G8)."""
        if not image_cols:
            return {}
        try:
            from PIL import Image as _PILImage
            import os as _os

            widths = []
            heights = []
            channels_set = set()
            blur_proxies = []

            for col in image_cols:
                if col not in df.columns:
                    continue
                paths = df[col].dropna().astype(str).head(30).tolist()
                for p in paths:
                    try:
                        if not _os.path.isfile(p):
                            continue
                        img = _PILImage.open(p)
                        w, h = img.size
                        widths.append(float(w))
                        heights.append(float(h))
                        mode = img.mode
                        if mode == "RGB":
                            channels_set.add("rgb")
                        elif mode == "L":
                            channels_set.add("grayscale")
                        elif mode == "RGBA":
                            channels_set.add("rgba")
                        else:
                            channels_set.add(mode.lower())
                        try:
                            _arr = np.array(img.convert("L"))
                            lap_var = float(np.var(np.diff(np.diff(_arr, axis=0), axis=1)))
                            blur_proxies.append(lap_var)
                        except Exception:
                            pass
                    except Exception:
                        continue

            signals = {}
            if widths and heights:
                signals["mean_resolution"] = round(
                    float(sum(widths) / len(widths)) * float(sum(heights) / len(heights)), 1
                )
                ratios = [w / h for w, h in zip(widths, heights) if h > 0]
                signals["aspect_ratio_variance"] = round(float(np.var(ratios)), 4) if ratios else 0.0
            if channels_set:
                signals["channels"] = sorted(channels_set)
            if blur_proxies:
                signals["blur_proxy_variance_of_laplacian"] = round(float(np.mean(blur_proxies)), 2)
            if widths:
                signals["object_count_proxy"] = "high" if float(sum(widths) / len(widths)) > 512 else "low"
            return signals
        except ImportError:
            return {}
        except Exception:
            return {}

    # -----------------------------------------------------------------------
    # Combined feature signals builder (G7/G8)
    # -----------------------------------------------------------------------

    def _build_feature_signals(self, df, detected):
        """Aggregate text and image feature signals into the IndividualSchema."""
        signals = {}
        text_cols = detected.get("text", [])
        image_cols = detected.get("image", [])
        if text_cols:
            signals["text"] = self._extract_text_feature_signals(df, text_cols)
        if image_cols:
            signals["image"] = self._extract_image_feature_signals(df, image_cols)
        return signals

    # -----------------------------------------------------------------------
    # G9: Cross-modality alignment signals
    # -----------------------------------------------------------------------

    @staticmethod
    def _compute_cross_modality_signals(per_dataset_results):
        """Compute cross-modality complementarity and alignment signals (G9)."""
        try:
            if len(per_dataset_results) < 2:
                return {}
            MODALITY_ORDER = ["tabular", "text", "image", "timeseries"]
            vectors = []
            confidence_gaps = []
            for ds in per_dataset_results:
                mods = set(ds.get("modalities", []))
                vec = [1.0 if m in mods else 0.0 for m in MODALITY_ORDER]
                vectors.append(vec)
                reasoning = ds.get("reasoning", {})
                gap = float(reasoning.get("xs3_confidence_gap", 0.0) or 0.0)
                confidence_gaps.append(gap)

            # complementarity_score = variance of per-dataset confidence_gap
            complementarity_score = float(np.var(confidence_gaps)) if confidence_gaps else 0.0

            # alignment_strength = mean pairwise cosine of modality-presence vectors
            cosine_sims = []
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    a = np.array(vectors[i], dtype=float)
                    b = np.array(vectors[j], dtype=float)
                    denom = (np.linalg.norm(a) * np.linalg.norm(b))
                    if denom > 0:
                        cosine_sims.append(float(np.dot(a, b) / denom))
            alignment_strength = float(np.mean(cosine_sims)) if cosine_sims else 0.0

            return {
                "complementarity_score": round(complementarity_score, 4),
                "alignment_strength": round(alignment_strength, 4),
            }
        except Exception:
            return {}


    # -----------------------------------------------------------------------
    # G10: Text target override validator
    # -----------------------------------------------------------------------

    @staticmethod
    def _validate_text_target_override(col: str, series, declared_task: str) -> tuple:
        """Validate text target override against declared task type (G10).
        Returns (valid: bool, reason: str).
        """
        import re as _re
        n_unique = int(series.dropna().nunique())
        if declared_task == "text_classification":
            if 2 <= n_unique <= 200:
                return True, "text_classification valid"
            return False, f"text_classification requires 2-200 unique labels, got {n_unique}"
        if declared_task == "ner_sequence":
            iob_re = _re.compile(r"^(B|I|O)-[A-Z]{1,10}$")
            sample = series.dropna().astype(str).head(100)
            iob_hits = sum(
                1 for val in sample
                if any(iob_re.match(tok) for tok in str(val).split())
            )
            if iob_hits / max(1, len(sample)) >= 0.5:
                return True, "ner_sequence: >=50% IOB tokens detected"
            return False, f"ner_sequence: only {iob_hits}/{len(sample)} IOB tokens found"
        if declared_task == "seq2seq":
            # Caller must verify paired source column; we allow it
            return True, "seq2seq: caller-verified"
        return True, f"unknown declared_task '{declared_task}' â€” allowed"

    # -----------------------------------------------------------------------
    # G11: Image unsupervised fallback detector
    # -----------------------------------------------------------------------

    @staticmethod
    def _check_image_label_validity(df, target_col: str) -> tuple:
        """Check image target col for folder-based classification validity (G11).
        Returns (valid: bool, problem_type: str)
        """
        if target_col not in df.columns:
            return False, "unsupervised_vision"
        col = df[target_col].dropna()
        if len(col) == 0:
            return False, "unsupervised_vision"
        cardinality_ratio = col.nunique() / max(1, len(col))
        if cardinality_ratio > 0.3:
            return False, "unsupervised_vision"
        return True, "classification_multiclass"

    # -----------------------------------------------------------------------
    # Column-type checks
    # -----------------------------------------------------------------------

    def _is_image(self, series: pd.Series, col_name: str = "") -> bool:
        """
        Detect image path / URL columns.

        Three signals (any one is sufficient with high confidence):
        1. File-extension match — values contain .jpg/.png/etc.
        2. URL pattern — values look like HTTP/S3/GCS image URLs even without an extension
        3. Column-name hint — name contains "img", "image", "photo", "thumbnail", etc.
           combined with path-like / URL-like values

        This fixes silent failure when images are referenced by extensionless CDN
        URLs (e.g. https://cdn.example.com/media/abc123) or by S3 keys.
        """
        if series.dtype != "object":
            return False
        sample = series.dropna().astype(str).head(50)
        if len(sample) == 0:
            return False

        # Signal 1: known image file extensions in values
        ext_hits = sum(any(ext in v.lower() for ext in self.IMAGE_EXTENSIONS) for v in sample)
        if ext_hits > max(3, len(sample) * 0.30):
            return True

        # Signal 2: HTTP/S3/GCS URLs that look like media references
        # (e.g. "https://cdn.example.com/img/abc123" — no extension but clearly a media URL)
        _IMAGE_URL_RE = _re_import.compile(
            r'^(https?://|s3://|gs://|azure://)[^\s]+$', _re_import.IGNORECASE
        )
        url_hits = sum(1 for v in sample if _IMAGE_URL_RE.match(v.strip()))
        # Threshold: at least 2 hits and ≥30% of sample (≥ not >; handles tiny samples)
        if url_hits >= max(2, len(sample) * 0.30):
            # Additional guard: values must not look like free text (no whitespace)
            ws_ratio = float(sample.str.contains(r'\s', regex=True).mean())
            if ws_ratio < 0.1:
                return True

        # Signal 3: column name word-parts (split on non-alpha) suggest images
        # + values look like paths.  Using word-splitting avoids \b failing on
        # underscore-separated names like "image_url", "photo_path", "img_id".
        _IMAGE_COL_WORDS = {"img", "image", "photo", "pic", "thumbnail",
                            "screenshot", "frame", "pixel"}
        _col_parts = set(_re_import.split(r'[^a-zA-Z0-9]', (col_name or "").lower()))
        if _col_parts & _IMAGE_COL_WORDS:
            # Values must look like paths (contain / or \\ or a dot, no whitespace)
            path_hits = sum(
                1 for v in sample
                if ('/' in v or '\\' in v or '.' in v) and ' ' not in v
            )
            if path_hits >= max(2, len(sample) * 0.20):
                return True

        return False

    def _is_text(self, series: pd.Series) -> bool:
        """
        Return True when the column looks like free-form text.
        Uses mean length > 30 chars OR mean length > 15 with whitespace present.
        Lower threshold (was 50) catches short-form text like product reviews and medical notes.
        """
        if series.dtype != "object":
            return False
        sample = series.dropna().astype(str).head(50)
        if len(sample) == 0:
            return False
        avg_len = float(sample.str.len().mean())
        if avg_len > 30:
            return True
        if avg_len > 15:
            whitespace_ratio = float(sample.str.contains(r"\s", regex=True).mean())
            return whitespace_ratio > 0.5
        return False

    def _is_timeseries(self, series: pd.Series) -> bool:
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        if series.dtype != "object":
            return False
        sample = series.dropna().astype(str).head(50)
        if len(sample) == 0:
            return False
        # Array/list-encoded timeseries (e.g. ECG: "[1.2, 3.4, ...]")
        if sample.str.contains(r"^\s*\[.*\]\s*$", na=False, regex=True).mean() > 0.5:
            return True
        # Timestamp strings (ISO dates, datetimes)
        looks_temporal = sample.str.contains(
            r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}:\d{2}:\d{2}|T\d{2}:\d{2}",
            na=False, regex=True,
        ).mean()
        if looks_temporal > 0.6:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(sample, errors="coerce", utc=False)
            return float(parsed.notna().mean()) > 0.85
        return False

    # -----------------------------------------------------------------------
    # Target detection — 3-Layer system
    # -----------------------------------------------------------------------

    _ID_RE = _re_import.compile(r'(?:^|_)id(?:$|_)')

    def detect_semantic_target(self, col: pd.Series) -> bool:
        if col.dtype != "object":
            return False
        sample = col.dropna().astype(str).head(50)
        if len(sample) == 0:
            return False
        json_like = sample.str.contains(r"\{.*\}", na=False).mean()
        list_like = sample.str.contains(r"\[.*\]", na=False).mean()
        avg_length = sample.str.len().mean()

        return bool(
            (json_like > 0.3 or list_like > 0.3)
            and avg_length > 10
        )

    def classify_text_column(self, series: pd.Series) -> str:
        if series.dtype != "object":
            return "not_text"
        sample = series.dropna().astype(str).head(50)
        if len(sample) == 0:
            return "not_text"

        if sample.str.contains(r"\{.*\}").mean() > 0.3:
            return "structured_label"
        if sample.str.contains(",").mean() > 0.3:
            return "multi_label"
        if series.nunique() < 20:
            return "categorical_text"
        return "free_text"

    def _semantic_analysis(self, col: pd.Series) -> Dict[str, Any]:
        n = len(col)
        unique = col.nunique(dropna=True)
        try:
            nan_ratio = float(col.isna().mean())
        except Exception:
            nan_ratio = 1.0

        info = {}
        if nan_ratio > 0.5:
            info["valid"] = False
            info["reason"] = "Too many NaNs"
            return info

        if unique <= 1:
            info["valid"] = False
            info["reason"] = "Constant column"
            return info

        info["semantic_target"] = self.detect_semantic_target(col)
        info["text_type"] = self.classify_text_column(col)

        if unique / max(n, 1) > 0.98 and col.dtype != "object":
            info["role"] = "id"
        elif col.dtype == "object" and 2 <= unique <= 20:
            info["role"] = "target_candidate"
        else:
            info["role"] = "feature"

        if col.dtype == "object":
            info["structure"] = info["text_type"]
        else:
            info["structure"] = "numeric"

        info["valid"] = True
        return info

    # -----------------------------------------------------------------------
    # PATCH 1, 2, 3, 6 — Advanced ML probes & multilabel detection
    # -----------------------------------------------------------------------
    def _compute_learnability(self, df: pd.DataFrame, target_col: str, modality_map: Any = None) -> float:
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            from sklearn.ensemble import RandomForestClassifier

            y = df[target_col].fillna("__NA__")
            X = df.drop(columns=[target_col], errors="ignore")
            X = X.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan).fillna(0)

            if X.shape[1] == 0 or len(X) < 10:
                return 0.0

            from sklearn.preprocessing import LabelEncoder
            y_enc = LabelEncoder().fit_transform(y.astype(str))

            X_train, X_val, y_train, y_val = train_test_split(X, y_enc, test_size=0.2, random_state=42)

            model = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)

            return float(accuracy_score(y_val, preds))
        except Exception:
            return 0.0

    def _text_signal_score(self, s: pd.Series) -> float:
        score = 0.0
        sample = s.dropna().astype(str).head(50)
        is_structured_label = sample.str.contains(r"\{.*\}", na=False).mean() > 0.3

        if is_structured_label:
            score += 0.5
        if 2 <= s.nunique() <= 50:
            score += 0.3

        avg_len = sample.str.len().mean() if len(sample) > 0 else 0
        if avg_len > 100:
            score -= 0.2  # long text = feature, not target

        return max(0.0, min(1.0, score))

    def _image_label_quality(self, lazy_data: Any) -> float:
        classes = getattr(lazy_data, "classes", None)
        if classes and len(classes) > 1:
            return 1.0
        return 0.0

    def _cross_modal_boost(self, modalities: List[str]) -> float:
        score = 0.0
        if "text" in modalities and "tabular" in modalities:
            score += 0.2
        if "image" in modalities and "tabular" in modalities:
            score += 0.2
        return score

    @staticmethod
    def _text_predictability_score(series: pd.Series, df: pd.DataFrame, max_features: int = 500) -> float:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score

            texts = series.fillna("").astype(str)
            if texts.str.len().mean() < 3:
                return 0.0

            vectorizer = TfidfVectorizer(max_features=max_features)
            X = vectorizer.fit_transform(texts)
            y = texts.astype("category").cat.codes
            if y.nunique() < 2:
                return 0.0

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = LogisticRegression(max_iter=200)
            clf.fit(X_train, y_train)
            return float(accuracy_score(y_val, clf.predict(X_val)))
        except Exception:
            return 0.0

    @staticmethod
    def _text_information_density(series: pd.Series) -> float:
        try:
            lengths = series.fillna("").astype(str).str.len()
            return float(lengths.std() / (lengths.mean() + 1e-5))
        except Exception:
            return 0.0

    @staticmethod
    def _image_label_separability_score(df: pd.DataFrame, label_col: str) -> float:
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split

            X = df.drop(columns=[label_col], errors="ignore")
            y = df[label_col]
            if y.nunique() < 2:
                return 0.0

            X = X.select_dtypes(include=["number"]).fillna(0)
            if X.shape[1] == 0:
                return 0.0

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = RandomForestClassifier(n_estimators=50)
            clf.fit(X_train, y_train)
            return float(clf.score(X_val, y_val))
        except Exception:
            return 0.0

    @staticmethod
    def _gini_class_balance(series: pd.Series) -> float:
        """Return a diversity-style class balance score in [0, 1)."""
        try:
            probs = series.dropna().value_counts(normalize=True)
            if len(probs) == 0:
                return 0.0
            return float(1.0 - np.square(probs.to_numpy(dtype=np.float64)).sum())
        except Exception:
            return 0.0

    @staticmethod
    def _infer_text_task_type(target_col: str, series: pd.Series) -> Optional[str]:
        """Infer whether the chosen text task is classification, NER, or seq2seq."""
        import re

        if series is None:
            return None

        sample = series.dropna().astype(str).head(100)
        if len(sample) == 0:
            return None

        iob_pattern = re.compile(r"^(B|I|O)-[A-Z]{1,10}$")
        seq2seq_tokens = {"</s>", "[SEP]", "<eos>", "[CLS]", "<pad>"}

        iob_hits = sum(
            1 for value in sample
            if any(iob_pattern.match(tok) for tok in str(value).split())
        )
        if iob_hits / max(1, len(sample)) > 0.3:
            return "ner_sequence"

        avg_len = float(sample.str.len().mean())
        target_name = str(target_col or "").lower()
        special_tok_hits = sum(
            1 for value in sample
            if any(tok in str(value) for tok in seq2seq_tokens)
        )
        if avg_len > 30 and (
            special_tok_hits > 0
            or "output" in target_name
            or "response" in target_name
            or "summary" in target_name
        ):
            return "seq2seq"

        n_unique = int(series.dropna().nunique(dropna=True))
        if 2 <= n_unique <= 200:
            return "text_classification"
        return None

    @staticmethod
    def _infer_semantic_role(col: str, series: pd.Series) -> str:
        if series.dtype == "object":
            avg_len = series.dropna().astype(str).str.len().mean()
            if avg_len > 30:
                return "text"
        if series.nunique(dropna=True) > 0.9 * max(1, len(series)):
            return "identifier"
        if series.nunique(dropna=True) < 20:
            return "categorical_label"
        return "numeric_feature"

    @staticmethod
    def _compute_interaction_score(df: pd.DataFrame, col: str) -> float:
        scores = []
        for other in df.columns:
            if other == col:
                continue
            try:
                with np.errstate(invalid="ignore", divide="ignore"):
                    corr = abs(df[col].corr(df[other]))
                if not np.isnan(corr):
                    scores.append(corr)
            except Exception:
                continue
        return np.mean(scores) if scores else 0.0

    @staticmethod
    def _compute_uncertainty(series: pd.Series) -> float:
        try:
            if series.dtype == "object" or series.nunique(dropna=True) < 20:
                return float(1.0 - series.dropna().value_counts(normalize=True).max())

            mean_val = series.mean()
            return float(series.std() / (mean_val + 1e-5))
        except Exception:
            return 1.0

    @staticmethod
    def _is_multilabel(series: pd.Series) -> bool:
        """True when the column carries multi-label targets (JSON dicts or comma-separated label lists)."""
        sample = series.dropna().astype(str).head(100)
        if len(sample) == 0:
            return False
        # Must look like a complete dict/list structure, not just contain a stray "{" or ":"
        structured = float(sample.str.match(r'^\s*[\[{].*[}\]]\s*$').mean())
        if structured > 0.3:
            return True
        # Comma-separated label lists: multiple tokens, no path separators
        has_comma = float(sample.str.contains(r'[^,]+,[^,]+', regex=True).mean())
        no_path = float((~sample.str.contains(r'[/\\]', regex=True)).mean())
        return has_comma > 0.3 and no_path > 0.8

    def _get_valid_candidates(self, df: pd.DataFrame) -> List[str]:
        candidates = []
        for col in df.columns:
            if df[col].isna().mean() > 0.5:
                continue
            if df[col].nunique(dropna=True) <= 1:
                continue
            candidates.append(col)
        return candidates

    def _validate_target(self, score: TargetScore) -> Tuple[bool, str]:
        # 🔴 PATCH 1 — HARD CONSTRAINT FILTER
        if score.nan_ratio > 0.5:
            return False, "Too many NaNs"
        if score.final_score == 0.0:
            is_semantic = score.semantics.get("semantic_target", False) or score.semantics.get("is_multilabel", False)
            if not is_semantic:
                return False, "No predictive signal"
        if score.uniqueness_score > 0.99:
            return False, "Looks like ID"
        if score.uniqueness_score < 0.01:
            return False, "Constant / near constant"
        return True, "Valid"

    def _score_column(
        self,
        df: pd.DataFrame,
        col: str,
        n_rows: int,
        col_index: int = 0,
        total_cols: int = 1,
        cross_dataset_counts: Optional[Dict[str, int]] = None,
        total_datasets: int = 1,
    ) -> TargetScore:
        """
        X-S³ Engine — Unified CMTI Scoring.
        """
        series = df[col]
        name = col.lower()
        parts = _re_import.split(r'[_\s]+', name)
        last_part = parts[-1] if parts else ""

        any_match = float(any(k in name for k in self.TARGET_KEYWORDS))
        suffix_score_val = float(last_part in self.TARGET_SUFFIX_KEYWORDS)
        keyword_score = min(any_match + suffix_score_val * 0.5, 1.0)
        json_score = 0.0
        binary_penalty = 0.0
        n_unique = 0
        try:
            n_unique = series.nunique(dropna=True)
            if series.dtype == "object":
                sample = series.dropna().astype(str).head(50)
                if len(sample) > 0:
                    json_score = float(sample.str.contains(r"\{.*\}", na=False).mean())
                    vals = set(sample.str.lower().unique())
                    if vals <= self._BINARY_ATTRIBUTE_VALUES and n_unique == 2:
                        binary_penalty = 0.20
        except Exception:
            pass

        try:
            nan_ratio = float(series.isna().mean())
        except Exception:
            nan_ratio = 1.0

        uniqueness_score = max(0.0, 1.0 - (n_unique / max(n_rows, 1)))

        sem = self._semantic_analysis(series)
        is_mlabel = sem.get("text_type") == "multi_label" or json_score > 0.4

        predictability = self._predictability_score(df, col)
        complementarity = self._complementarity_score(df, col)

        cross_dataset = 0.0
        if cross_dataset_counts and total_datasets > 1:
            col_norm = name.strip()
            count = cross_dataset_counts.get(col_norm, 1)
            cross_dataset = (count - 1) / (total_datasets - 1) if count > 1 else 0.0

        degeneracy = self._degeneracy_penalty(series)
        predictability = predictability * (1.0 - degeneracy)

        dtype = "text" if self._is_text(series) else "image" if self._is_image(series, col_name=str(col)) else "tabular"

        if dtype == "text":
            text_pred = self._text_predictability_score(series, df)
            density = self._text_information_density(series)
            predictability = max(predictability, text_pred)
            complementarity = (complementarity + density) / 2

        if dtype == "image":
            img_score = self._image_label_separability_score(df, col)
            predictability = max(predictability, img_score)

        if is_mlabel:
            predictability += 0.2

        quality = 0.0
        if (n_unique / max(n_rows, 1)) < 0.95:
            quality += 0.3
        if nan_ratio < 0.5:
            quality += 0.2

        is_categorical = False
        if str(series.dtype) in ("object", "category") or n_unique < 20:
            is_categorical = True
        if is_categorical:
            quality += 0.3
        if keyword_score > 0:
            quality += 0.2

        semantic_score = 0.0
        if any(k in name for k in ["code", "label", "class", "target", "diagnosis", "sentiment", "scp_codes"]):
            semantic_score += 0.5
        try:
            sample_val = series.dropna().iloc[0] if len(series.dropna()) > 0 else ""
            if isinstance(sample_val, dict) or str(sample_val).startswith("{"):
                semantic_score += 0.5
            if series.astype(str).str.contains(r"\{.*\}", na=False).mean() > 0.3:
                semantic_score += 0.5
        except Exception:
            pass
        semantic_score = min(semantic_score, 1.0)

        if any(name == k for k in ["scp_codes", "diagnosis", "label", "target", "class", "category", "sentiment"]):
            semantic_score = 1.0
            predictability += 0.25

        if dtype == "text":
            semantic_score = max(semantic_score, self._text_signal_score(series))

        # Verb-phrase / suffix name pattern (human-level name understanding)
        verb_score = self._verb_phrase_score(col)
        if verb_score > 0:
            semantic_score = min(1.0, semantic_score + verb_score * 0.5)

        if quality < 0.3:
            final_score = 0.0
            interaction_score = 0.0
            uncertainty = 1.0
            semantic_role = self._infer_semantic_role(col, series)
        else:
            interaction_score = self._compute_interaction_score(df, col)
            uncertainty = self._compute_uncertainty(series)
            semantic_role = self._infer_semantic_role(col, series)

            # MI-based predictability blended with RF probe (handles mixed types)
            mi_pred = self._mi_predictability_score(df, col)
            predictability = max(predictability, mi_pred * 0.85)

            # Dataset-level understanding signals
            directionality = self._directionality_score(df, col, forward_predictability=predictability)
            fingerprint = self._value_set_fingerprint(series)
            leakage_pen = self._leakage_penalty(predictability)

            # Small bonus for last column — many ML datasets put the target last
            position_bonus = 0.04 if col_index == total_cols - 1 else 0.0

            final_score = (
                0.20 * predictability +
                0.17 * complementarity +
                0.12 * semantic_score +
                0.12 * interaction_score +
                0.08 * cross_dataset +
                0.12 * (1.0 - uncertainty) +
                0.12 * directionality +
                0.07 * fingerprint
                + position_bonus
            ) - leakage_pen

            final_score += self._cross_modal_boost([dtype])

        reasons = []
        if semantic_score > 0.4:
            reasons.append("Structured / semantic label detected")
        if predictability < 0.1:
            reasons.append("Low predictability (complex or latent target)")
        if complementarity > 0.5:
            reasons.append("Provides unique signal")
        if is_categorical:
            reasons.append("Categorical target structure")
        if not reasons:
            reasons.append("General predictive target")

        explanation = [" | ".join(reasons)]
        final_score = max(0.0, final_score)

        return TargetScore(
            column=col,
            keyword_score=keyword_score,
            uniqueness_score=uniqueness_score,
            regression_score=float(pd.api.types.is_float_dtype(series) or (pd.api.types.is_numeric_dtype(series) and n_unique > 20)),
            json_score=json_score,
            predictability_score=predictability,
            complementarity_score=complementarity,
            cross_dataset_score=cross_dataset,
            degeneracy_penalty=degeneracy,
            final_score=round(final_score, 4),
            nan_ratio=nan_ratio,
            valid=True,
            reason="Valid",
            quality=quality,
            semantic_score=semantic_score,
            semantics=sem,
            explanation=explanation,
            semantic_role=semantic_role,
            interaction_score=interaction_score,
            uncertainty=uncertainty,
        )

    # ------------------------------------------------------------------
    # Layer 2 — Predictability score (Random-Forest cross-validation)
    # ------------------------------------------------------------------

    @staticmethod
    def _predictability_score(
        df: pd.DataFrame,
        column: str,
        max_rows: int = 500,
    ) -> float:
        """
        Estimate how well *other* columns predict *column* using a shallow
        RandomForest (max_depth=3) with 3-fold cross-validation.
        """
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.model_selection import cross_val_score
            from sklearn.preprocessing import LabelEncoder
            import warnings

            sub = df.head(max_rows).copy()
            y_raw = sub[column].fillna("__NA__")

            # Constant / near-constant columns are trivially predictable — return 0
            # to avoid inflating scores for content features (repeated review text, etc.)
            if y_raw.nunique() <= 1:
                return 0.0
            unique_ratio = y_raw.nunique() / max(len(y_raw), 1)
            if unique_ratio < 0.01:
                return 0.0

            X = sub.drop(columns=[column]).select_dtypes(include=["number"])
            if len(X.columns) == 0:
                logger.debug(f"No numeric features for predictability score of {column}")
                return 0.0

            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

            le = LabelEncoder()
            y = le.fit_transform(y_raw.astype(str))

            if len(le.classes_) <= 20:
                clf = RandomForestClassifier(n_estimators=50, random_state=42)
                scoring = "accuracy"
            else:
                clf = RandomForestRegressor(n_estimators=50, random_state=42)
                scoring = "r2"

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_scores = cross_val_score(clf, X.fillna(0), y, cv=3, scoring=scoring)

            score = float(max(0.0, cv_scores.mean()))
            logger.debug(f"RandomForest predictability score for {column}: {score:.4f}")
            return score

        except Exception as e:
            logger.warning(
                f"RandomForest predictability probe failed for column '{column}': {type(e).__name__}: {e}. "
                f"Falling back to correlation-based estimate."
            )

            try:
                X_numeric = df.drop(columns=[column]).select_dtypes(include=["number"])
                y_numeric = df[column]

                if X_numeric.shape[1] == 0:
                    return 0.0

                if not pd.api.types.is_numeric_dtype(y_numeric):
                    y_numeric = pd.factorize(y_numeric)[0]
                else:
                    y_numeric = y_numeric.fillna(y_numeric.mean())

                corrs = X_numeric.corrwith(pd.Series(y_numeric)).abs()
                max_corr = corrs.max()

                if np.isnan(max_corr):
                    logger.debug(f"Correlation fallback also failed for {column}, returning 0")
                    return 0.0

                fallback_score = float(max(0.0, max_corr))
                logger.debug(f"Correlation-based fallback for {column}: {fallback_score:.4f}")
                return fallback_score

            except Exception as e2:
                logger.error(
                    f"Both RandomForest and correlation fallback failed for {column}: {type(e2).__name__}: {e2}",
                    exc_info=False
                )
                return 0.0

    @staticmethod
    def _cross_modal_convergence(
        df: pd.DataFrame,
        col: str,
        detected: Dict[str, List[str]],
        threshold: float = 0.10,
    ) -> float:
        """
        How many independent modalities can each predict this column?

        A genuine multimodal target is predictable from text features ALONE,
        tabular features ALONE, and image metadata ALONE — convergent evidence
        from every view of the data.  A spurious feature might correlate with
        one modality but not others.

        Returns a score in [0, 1]:
          0.0 → no modality predicts this column
          0.5 → one modality predicts it
          1.0 → all present modalities predict it
        """
        try:
            modality_scores: List[float] = []
            for modality, cols in detected.items():
                feature_cols = [c for c in cols if c != col]
                if not feature_cols:
                    continue
                sub = df[feature_cols + [col]].head(500)

                if modality == "text":
                    # Use TF-IDF probe for text features predicting col
                    for fc in feature_cols:
                        try:
                            from sklearn.feature_extraction.text import TfidfVectorizer
                            from sklearn.linear_model import LogisticRegression
                            from sklearn.model_selection import cross_val_score

                            texts = sub[fc].fillna("").astype(str)
                            y_raw = sub[col].fillna("__NA__")
                            if y_raw.nunique() <= 1 or y_raw.nunique() > 50:
                                continue
                            y_enc = pd.factorize(y_raw.astype(str))[0]
                            vec = TfidfVectorizer(max_features=200)
                            X_t = vec.fit_transform(texts)
                            scores = cross_val_score(
                                LogisticRegression(max_iter=100), X_t, y_enc,
                                cv=3, scoring="accuracy",
                            )
                            modality_scores.append(float(max(0.0, scores.mean())))
                        except Exception:
                            pass
                else:
                    # Numeric/tabular and image metadata: RF probe on those columns only
                    try:
                        X_mod = sub[feature_cols].select_dtypes(include=["number"]).fillna(0)
                        if X_mod.shape[1] == 0:
                            continue
                        y_raw = sub[col].fillna("__NA__")
                        if y_raw.nunique() <= 1:
                            continue
                        from sklearn.ensemble import RandomForestClassifier
                        from sklearn.model_selection import cross_val_score
                        from sklearn.preprocessing import LabelEncoder
                        import warnings

                        y_enc = LabelEncoder().fit_transform(y_raw.astype(str))
                        clf = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            s = cross_val_score(clf, X_mod, y_enc, cv=3, scoring="accuracy")
                        modality_scores.append(float(max(0.0, s.mean())))
                    except Exception:
                        pass

            if not modality_scores:
                return 0.0

            n_modalities = len([s for s in modality_scores if s > threshold])
            total = len(modality_scores)
            return round(float(n_modalities) / max(1, total), 3)
        except Exception:
            return 0.0

    @staticmethod
    def _directionality_score(
        df: pd.DataFrame,
        col: str,
        forward_predictability: float = 0.0,
    ) -> float:
        """
        Causal direction signal: targets are predicted BY features, not predictors OF them.

        forward  = how well do all other columns predict `col`?  (passed in — already computed)
        backward = mean |correlation| of `col` with every other column
                   (proxy for how much information `col` leaks about other columns)

        directionality = max(0, forward - backward), scaled to [0, 1]

        High → col is a downstream outcome (likely target)
        Near 0 → col contributes equally in both directions (feature)
        Negative (clamped to 0) → col strongly predicts others (upstream feature / ID)
        """
        try:
            sample = df.head(500)
            col_s = sample[col]
            if col_s.dtype == "object" or str(col_s.dtype) == "category":
                col_enc = pd.Series(
                    pd.factorize(col_s.fillna("__NA__"))[0], dtype=float
                )
            else:
                col_enc = col_s.fillna(float(col_s.mean()) if len(col_s.dropna()) else 0).astype(float)

            backward: list = []
            for other in sample.columns:
                if other == col:
                    continue
                try:
                    other_s = sample[other]
                    if other_s.dtype == "object":
                        other_enc = pd.Series(
                            pd.factorize(other_s.fillna("__NA__"))[0], dtype=float
                        )
                    else:
                        other_enc = other_s.fillna(
                            float(other_s.mean()) if len(other_s.dropna()) else 0
                        ).astype(float)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        c = float(col_enc.corr(other_enc))
                    if not pd.isna(c):
                        backward.append(abs(c))
                except Exception:
                    continue

            mean_backward = float(np.mean(backward)) if backward else 0.0
            raw = max(0.0, forward_predictability - mean_backward)
            return round(min(1.0, raw * 2.0), 4)
        except Exception:
            return 0.0

    @staticmethod
    def _value_set_fingerprint(series: pd.Series) -> float:
        """
        Read the actual cell values to understand whether a column 'looks like a target'.

        This is dataset-level understanding rather than name heuristics:
        - "survived"/"died", "yes"/"no", "spam"/"ham" → binary classification target → 0.90
        - Balanced readable string labels (≤10 classes, short text) → 0.75
        - Dense integer codes 0…N-1 (MNIST-style) → 0.80
        - UUID / hash strings → ID column → 0.0
        - Long floating-point values → likely a continuous feature → 0.15
        """
        try:
            sample = series.dropna()
            if len(sample) == 0:
                return 0.0

            n_unique = int(sample.nunique())

            # ── UUID / hash: penalise immediately ────────────────────────
            if series.dtype == "object":
                str_s = sample.astype(str).head(20)
                uuid_hit = float(str_s.str.match(
                    r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-', na=False
                ).mean())
                hash_hit = float(str_s.str.match(r'^[0-9a-fA-F]{20,}$', na=False).mean())
                if uuid_hit > 0.3 or hash_hit > 0.3:
                    return 0.0

            # ── Binary human-readable pairs ───────────────────────────────
            if n_unique == 2 and series.dtype == "object":
                val_set = frozenset(sample.astype(str).str.strip().str.lower().unique())
                KNOWN_BINARY_TARGETS = [
                    {"yes", "no"}, {"true", "false"}, {"1", "0"},
                    {"positive", "negative"}, {"survived", "died"},
                    {"spam", "ham"}, {"fraud", "genuine"},
                    {"malignant", "benign"}, {"churn", "retain"},
                    {"churned", "retained"}, {"approved", "rejected"},
                    {"pass", "fail"}, {"win", "loss"},
                    {"up", "down"}, {"buy", "sell"},
                    {"success", "failure"}, {"default", "paid"},
                ]
                if any(val_set <= frozenset(pair) for pair in KNOWN_BINARY_TARGETS):
                    return 0.90

            # ── Dense integer codes starting at 0 or 1 ───────────────────
            if pd.api.types.is_integer_dtype(series) and 2 <= n_unique <= 20:
                min_v = int(sample.min())
                max_v = int(sample.max())
                if min_v in (0, 1) and max_v == min_v + n_unique - 1:
                    return 0.80

            # ── Balanced readable multi-class string labels ───────────────
            if n_unique <= 10 and series.dtype == "object":
                vc = sample.value_counts(normalize=True)
                gini = 1.0 - float(np.square(vc.values).sum())
                avg_len = float(sample.astype(str).str.len().mean())
                if gini > 0.35 and avg_len < 25:
                    return 0.75

            # ── Continuous float (regression target candidate) ────────────
            if pd.api.types.is_float_dtype(series) and n_unique > 20:
                try:
                    skew = abs(float(sample.skew()))
                    return max(0.0, round(0.40 - skew * 0.05, 3))
                except Exception:
                    return 0.20

            return 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _leakage_penalty(predictability: float) -> float:
        """
        Penalise columns that are suspiciously easy to predict.

        A genuine target is hard but possible (predictability ≈ 0.5 – 0.85).
        A derived / leaky column is trivially predictable (> 0.95).

        Examples of leaky columns:
          - "loan_paid"   derived from "loan_status"
          - "age_group"   bucketed from "age"
          - "price_usd"   converted copy of "price_eur"

        Returns penalty ∈ [0, 0.30] to subtract from final_score.
        """
        if predictability > 0.97:
            return 0.30
        if predictability > 0.93:
            return 0.15
        return 0.0

    @staticmethod
    def _verb_phrase_score(col_name: str) -> float:
        """
        Score a column name based on verb-prefix and target-suffix patterns.
        "is_fraud", "will_churn", "has_disease", "purchase_flag" → score 0.4–0.6.
        These patterns indicate the column was purpose-built as a prediction target.
        """
        name = str(col_name).strip().lower()
        if _VERB_PREFIX_RE.match(name):
            return 0.60
        if _TARGET_SUFFIX_RE.search(name):
            return 0.50
        return 0.0

    @staticmethod
    def _mi_predictability_score(df: pd.DataFrame, column: str) -> float:
        """
        Estimate predictability via Mutual Information (MI) using sklearn.
        Unlike the RF probe, MI handles mixed feature types (categorical + numeric)
        by encoding object columns via factorize before computing MI.
        Returns MI score normalised to approximately [0, 1].
        Falls back to 0.0 on any failure.
        """
        try:
            from sklearn.feature_selection import (
                mutual_info_classif,
                mutual_info_regression,
            )

            sub = df.head(500).copy()
            y_raw = sub[column].fillna("__NA__")
            X = sub.drop(columns=[column])

            # Encode mixed-type features: object → ordinal codes
            X_enc = pd.DataFrame(index=X.index)
            for c in X.columns:
                col_s = X[c]
                if col_s.dtype == "object" or str(col_s.dtype) == "category":
                    X_enc[c] = pd.factorize(col_s.fillna("__NA__"))[0].astype(float)
                else:
                    X_enc[c] = pd.to_numeric(col_s, errors="coerce").fillna(0)

            X_enc = X_enc.replace([np.inf, -np.inf], 0).fillna(0)
            if X_enc.shape[1] == 0:
                return 0.0

            y_enc = pd.factorize(y_raw.astype(str))[0]
            n_unique = int(np.unique(y_enc).size)

            if n_unique <= 20:
                mi = mutual_info_classif(X_enc, y_enc, random_state=42)
            else:
                y_num = pd.to_numeric(y_raw, errors="coerce").fillna(0)
                mi = mutual_info_regression(X_enc, y_num.values, random_state=42)

            # MI is in nats; scale so ~0.3 nats (moderate signal) → ~0.6
            raw = float(mi.max()) if len(mi) > 0 else 0.0
            return float(min(1.0, raw * 2.0))

        except Exception:
            return 0.0

    @staticmethod
    def _complementarity_score(df: pd.DataFrame, target_col: str) -> float:
        """
        How much unique signal does this column carry that isn't captured by other features?
        - Numeric target: 1 - max(abs_correlation_with_features)
        - Object/text target with few unique values (categorical label): moderate fixed score
        - Object/text target with many unique values (free text): type-token ratio as richness proxy
        """
        try:
            y = df[target_col]
            if y.dtype == "object" or str(y.dtype) == "category":
                n_unique = int(y.nunique(dropna=True))
                if n_unique <= 2:
                    return 0.55  # binary label: moderate, not trivially unique
                if n_unique <= 20:
                    return 0.65  # multi-class label: standard target structure
                # Free-form text: type-token ratio as information richness
                sample = y.dropna().astype(str).head(500)
                tokens = sample.str.split().explode()
                if len(tokens) > 0:
                    ttr = float(tokens.nunique()) / max(1, len(tokens))
                    return min(1.0, ttr * 1.5)
                return 0.35

            X = df.drop(columns=[target_col]).select_dtypes(include=["number"])
            if X.shape[1] == 0:
                return 0.5  # no numeric features to correlate → neutral

            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
            y_num = y if pd.api.types.is_numeric_dtype(y) else pd.Series(pd.factorize(y)[0])
            corrs = X.corrwith(y_num).abs()
            max_corr = corrs.max()
            if np.isnan(max_corr):
                return 0.5
            return float(max(0.0, 1.0 - max_corr))
        except Exception:
            return 0.0

    @staticmethod
    def _degeneracy_penalty(series: pd.Series) -> float:
        """
        Penalize targets with suspicious cardinality patterns.
        Returns: penalty in [0, 1] where 1 = completely degenerate.
        """
        try:
            n = len(series)
            unique_count = series.nunique(dropna=True)
            unique_ratio = unique_count / max(n, 1)

            if unique_count <= 1:
                return 1.0
            if unique_ratio > 0.95:
                return 0.8
            if unique_ratio < 0.02:
                return 0.7

            value_counts = series.value_counts()
            if len(value_counts) > 1:
                proportions = value_counts.values / n
                max_prop = proportions.max()
                if max_prop > 0.99:
                    return 0.6
                if max_prop > 0.95:
                    return 0.3

            return 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _cross_dataset_score(col: str, per_dataset: List[Dict[str, Any]]) -> float:
        try:
            count = sum(col in ds.get("detected_columns", {}).get("tabular", []) for ds in per_dataset)
            return float(count / max(len(per_dataset), 1))
        except Exception:
            return 0.0

    @staticmethod
    def _explain(c: TargetScore) -> Dict[str, float]:
        return {
            "semantic": round(0.2 * c.keyword_score, 3),
            "predictability": round(0.3 * c.predictability_score, 3),
            "complementarity": round(0.2 * c.complementarity_score, 3),
            "uniqueness": round(0.2 * c.uniqueness_score, 3),
            "penalty": round(-0.1 * c.degeneracy_penalty, 3)
        }

    @staticmethod
    def _counterfactual(candidates: List[TargetScore]) -> Dict[str, Any]:
        sorted_c = sorted(candidates, key=lambda x: x.final_score, reverse=True)
        if len(sorted_c) < 2:
            return {}
        return {
            "runner_up": sorted_c[1].column,
            "gap": round(sorted_c[0].final_score - sorted_c[1].final_score, 3)
        }

    def _detect_target(self, df: pd.DataFrame) -> Tuple[str, float, List[TargetScore]]:
        """
        Unified X-S³ Engine target inference with Hard Validation.
        Returns (best_column_name, confidence_score, candidates_list).
        """
        n_rows = max(len(df), 1)
        total_cols = len(df.columns)

        valid_candidate_names = self._get_valid_candidates(df)
        scored = []
        rejected = []

        for i, col in enumerate(df.columns):
            ts = self._score_column(df, col, n_rows, col_index=i, total_cols=total_cols)

            if col not in valid_candidate_names:
                ts.valid = False
                ts.reason = "Filtered by preliminary scan (constant or NaN-heavy)"
                rejected.append(ts)
                continue

            valid, reason = self._validate_target(ts)
            ts.valid = valid
            ts.reason = reason

            if valid:
                scored.append(ts)
            else:
                rejected.append(ts)

        if not scored:
            raise ValueError("No valid target found. Please select manually.")

        scored.sort(key=lambda s: s.final_score, reverse=True)
        rejected.sort(key=lambda s: s.final_score, reverse=True)

        all_candidates = scored + rejected
        self.last_target_candidates = all_candidates  # type: List[TargetScore]

        best = scored[0]

        top_score = scored[0].final_score
        second_score = scored[1].final_score if len(scored) > 1 else 0.0
        gap = max(0.0, top_score - second_score)
        confidence = XS3TargetSelector._calibrate_confidence(gap, top_score)

        return best.column, confidence, all_candidates

    # -----------------------------------------------------------------------
    # Problem-type inference
    # -----------------------------------------------------------------------

    def _infer_problem(self, df: pd.DataFrame, target: str) -> str:
        """
        Infer the ML problem type from the target column.

        Order of checks:
          1. Unknown → unsupervised
          2. Object with JSON structure → multilabel
          3. Float dtype → regression (checked before unique-count to avoid n_unique≤20 false classification)
          4. Binary → classification_binary
          5. Integers with very sparse value range → regression (e.g. 10 values over 0–200K)
          6. 3–20 unique → classification_multiclass
          7. >20 unique integers: dense range → classification_multiclass, sparse → regression
          8. Object/string → classification_multiclass
        """
        if target == "Unknown":
            return "unsupervised"

        s = df[target]

        if s.dtype == "object":
            sample = s.dropna().astype(str).head(50)
            if len(sample) > 0 and sample.str.contains(r"\{.*\}", na=False).mean() > 0.3:
                return "multilabel_classification"

        # Float columns → regression regardless of unique count
        if pd.api.types.is_float_dtype(s):
            return "regression"

        n_unique: int = int(s.nunique(dropna=True))

        if n_unique == 2:
            return "classification_binary"

        if 3 <= n_unique <= 20:
            # Sparse integer range (e.g. prices [0, 100, 5000, 200000]) → regression
            if pd.api.types.is_integer_dtype(s):
                try:
                    min_v, max_v = int(s.min()), int(s.max())
                    range_size = max(1, max_v - min_v + 1)
                    if n_unique / range_size < 0.05:
                        return "regression"
                except Exception:
                    pass
            return "classification_multiclass"

        if pd.api.types.is_numeric_dtype(s):
            # Dense sequential integer range → ordinal/class code
            try:
                min_v, max_v = int(s.min()), int(s.max())
                range_size = max(1, max_v - min_v + 1)
                if n_unique / range_size > 0.6:
                    return "classification_multiclass"
            except Exception:
                pass
            return "regression"

        return "classification_multiclass"

    # -----------------------------------------------------------------------
    # Tier-2 aggregation helpers
    # -----------------------------------------------------------------------

    def _aggregate_modalities(self, results: List[Dict[str, Any]]) -> List[str]:
        """Union all per-dataset modalities into a sorted list."""
        mods: set = set()
        for r in results:
            mods.update(r.get("modalities", []))
        return sorted(mods)

    def _aggregate_problem_type(self, results: List[Dict[str, Any]]) -> str:
        """
        Resolve a single global problem type.
        Regression takes priority when mixed; otherwise majority vote.
        """
        types: List[str] = [
            r["problem_type"]
            for r in results
            if r.get("problem_type", "unsupervised") != "unsupervised"
        ]

        if not types:
            return "unsupervised"

        if "regression" in types:
            return "regression"

        return max(set(types), key=types.count)

    def _collect_all_candidates(
        self,
        results: List[Dict[str, Any]],
    ) -> List[TargetScore]:
        """Merge all candidate signals across datasets into a unified ranking."""
        cmap: Dict[str, TargetScore] = {}
        for r in results:
            for c_dict in r.get("candidates", []):
                col = c_dict.get("column", "")
                if not col:
                    continue
                ts = TargetScore(
                    column=col,
                    final_score=float(c_dict.get("final_score", 0.0)),
                    nan_ratio=float(c_dict.get("nan_ratio", 0.0)),
                    valid=bool(c_dict.get("valid", True)),
                    reason=str(c_dict.get("reason", "Valid")),
                    quality=float(c_dict.get("quality_score", c_dict.get("quality", 0.0))),
                    semantic_score=float(c_dict.get("semantic_score", 0.0)),
                    semantics=c_dict.get("semantics", {}),
                    explanation=c_dict.get("explanation", []),
                    semantic_role=str(c_dict.get("semantic_role", "")),
                    interaction_score=float(c_dict.get("interaction_score", 0.0)),
                    uncertainty=float(c_dict.get("uncertainty", 0.0)),
                    predictability_score=float(c_dict.get("predictability_score", 0.0)),
                    complementarity_score=float(c_dict.get("complementarity_score", 0.0)),
                )
                ts.cross_dataset_score = self._cross_dataset_score(col, results)
                if col not in cmap or ts.final_score > cmap[col].final_score:
                    cmap[col] = ts
        return list(cmap.values())

    # -------------------------------------------------------------------
    # Cross-dataset relatedness
    # -------------------------------------------------------------------

    def _check_relatedness(
        self,
        per_dataset_results: List[Dict[str, Any]],
    ) -> Tuple[List[List[int]], Dict[str, Any]]:
        """
        Check pairwise relatedness of datasets and return groups.
        """
        n = len(per_dataset_results)
        if n <= 1:
            return [list(range(n))], {"single_dataset": True, "n_groups": 1}

        scores: Dict[Tuple[int, int], float] = {}
        for i in range(n):
            for j in range(i + 1, n):
                a = per_dataset_results[i]
                b = per_dataset_results[j]

                cols_a: set = set()
                cols_b: set = set()
                for mod_cols in a.get("detected_columns", {}).values():
                    cols_a.update(mod_cols)
                for mod_cols in b.get("detected_columns", {}).values():
                    cols_b.update(mod_cols)
                union = cols_a | cols_b
                col_jaccard = len(cols_a & cols_b) / len(union) if union else 0.0

                target_match = 1.0 if (
                    a.get("target_column", "X") == b.get("target_column", "Y")
                    and a.get("target_column") != "Unknown"
                ) else 0.0

                mods_a = set(a.get("modalities", []))
                mods_b = set(b.get("modalities", []))
                mod_union = mods_a | mods_b
                mod_jaccard = (
                    len(mods_a & mods_b) / len(mod_union) if mod_union else 0.0
                )

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

    @staticmethod
    def _compute_within_dataset_signals(individual_schema: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute cross-modality signals for a single multimodal dataset (e.g. a CSV with
        both text and tabular columns).  Used when only one file is uploaded so the
        pairwise multi-dataset method is unavailable.

        complementarity_score — spread of per-modality signal strength (high = modalities
                                carry different information → good for fusion)
        alignment_strength    — mean per-modality informativeness (high = all modalities
                                are strong → attention/graph fusion preferred)
        """
        try:
            mods = individual_schema.get("modalities", [])
            if len(mods) < 2:
                return {}

            feature_signals = individual_schema.get("feature_signals", {})
            candidates = individual_schema.get("candidates", [])

            strength: Dict[str, float] = {}

            if "text" in mods:
                text_sig = feature_signals.get("text", {})
                avg_tokens = float(text_sig.get("avg_tokens_per_sample", 0) or 0)
                strength["text"] = min(1.0, avg_tokens / 80.0)

            if "tabular" in mods:
                tab_scores = [
                    float(c.get("predictability_score", 0))
                    for c in candidates
                    if c.get("dtype") == "tabular" and c.get("valid")
                ]
                strength["tabular"] = float(np.mean(tab_scores)) if tab_scores else 0.3

            if "image" in mods:
                img_sig = feature_signals.get("image", {})
                blur = float(img_sig.get("blur_proxy_variance_of_laplacian", 100) or 100)
                strength["image"] = 0.8 if blur > 50 else 0.4

            if "timeseries" in mods:
                strength["timeseries"] = 0.6  # conservative neutral estimate

            values = list(strength.values())
            if len(values) < 2:
                return {}

            alignment_strength = float(np.mean(values))
            complementarity_score = float(np.std(values))

            return {
                "complementarity_score": round(complementarity_score, 4),
                "alignment_strength": round(alignment_strength, 4),
            }
        except Exception:
            return {}

    def detect_global_schema(
        self,
        lazy_datasets: Dict[str, Any],
        target_overrides: Optional[Dict[str, str]] = None,
    ) -> GlobalSchema:
        """Public API: infer per-dataset schema and aggregate to global schema."""
        if not lazy_datasets:
            raise ValueError("detect_global_schema requires at least one dataset")

        overrides = target_overrides or {}
        per_dataset_results: List[Dict[str, Any]] = []

        for dataset_id, lazy_ref in lazy_datasets.items():
            override = overrides.get(dataset_id)
            try:
                individual = self._detect_single(
                    dataset_id=dataset_id,
                    lazy_data=lazy_ref,
                    target_override=override,
                )
                per_dataset_results.append(asdict(individual))
            except Exception as exc:
                logger.warning("Schema detection failed for %s: %s", dataset_id, exc)

        if not per_dataset_results:
            raise RuntimeError("No dataset produced a valid schema")

        global_modalities = self._aggregate_modalities(per_dataset_results)
        global_problem_type = self._aggregate_problem_type(per_dataset_results)
        primary_target = self._select_primary_target(per_dataset_results)
        if not primary_target or primary_target == "Unknown":
            primary_target = max(
                per_dataset_results,
                key=lambda r: float(r.get("confidence", 0.0)),
            ).get("target_column", "Unknown")

        detection_confidence = float(np.mean([
            float(r.get("confidence", 0.0)) for r in per_dataset_results
        ]))
        fusion_ready = len(global_modalities) > 1
        _, relatedness_report = self._check_relatedness(per_dataset_results)

        semantic_enrichment = {
            "semantic_roles_by_dataset": {
                r.get("dataset_id", f"dataset_{idx}"): r.get("semantic_roles", {})
                for idx, r in enumerate(per_dataset_results)
            },
            "business_patterns_by_dataset": {
                r.get("dataset_id", f"dataset_{idx}"): r.get("business_patterns", {})
                for idx, r in enumerate(per_dataset_results)
            },
        }

        # Cross-modality signals: use pairwise for multi-dataset; within-dataset for single-file multimodal
        if len(per_dataset_results) >= 2:
            multimodal_signals = self._compute_cross_modality_signals(per_dataset_results)
        elif fusion_ready:
            multimodal_signals = self._compute_within_dataset_signals(per_dataset_results[0])
        else:
            multimodal_signals = {}

        return GlobalSchema(
            global_problem_type=global_problem_type,
            global_modalities=global_modalities,
            primary_target=primary_target,
            fusion_ready=fusion_ready,
            detection_confidence=round(detection_confidence, 3),
            per_dataset=per_dataset_results,
            relatedness_report=relatedness_report,
            semantic_enrichment=semantic_enrichment,
            multimodal_signals=multimodal_signals,
        )


class MultiDatasetSchemaDetector(COGMASchemaDetector):
    """Backward-compatible public detector class used by orchestrator/API."""

    def __init__(self, use_fix4_target_detection: bool = True, fix4_engine=None):
        super().__init__(fix4_engine=fix4_engine if use_fix4_target_detection else None)


class SchemaDetector(MultiDatasetSchemaDetector):
    """Legacy compatibility shim used by older integration tests/callers."""

    def detect_schema_from_dataframe(
        self,
        df: pd.DataFrame,
        dataset_id: str = "dataset_0",
    ) -> Dict[str, Any]:
        """Return a serializable single-dataset schema payload."""
        result = asdict(self._inspect_dataframe(dataset_id=dataset_id, df=df))
        # Legacy consumers expect these aliases.
        result.setdefault("columns", result.get("detected_columns", {}))
        result.setdefault("target", result.get("target_column"))
        return result

    def detect_schema(
        self,
        df: pd.DataFrame,
        dataset_id: str = "dataset_0",
    ) -> Dict[str, Any]:
        """Alias for backward compatibility with old call sites."""
        return self.detect_schema_from_dataframe(df=df, dataset_id=dataset_id)
