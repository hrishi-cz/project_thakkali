"""
Data schema definitions for the APEX pipeline.

Two-tier architecture:
  IndividualSchema  – Tier-1 output: per-file column/target/problem inference
  GlobalSchema      – Tier-2 output: aggregated schema that feeds Phase 3+

Legacy classes (ColumnSchema, DataSchema) are preserved for backward
compatibility with any existing callers.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Legacy classes – backward compatibility preserved
# ---------------------------------------------------------------------------

@dataclass
class ColumnSchema:
    """Schema for a single data column."""
    name: str
    dtype: str
    modality: str  # 'image' | 'tabular' | 'text' | 'timeseries'
    optional: bool = False


class DataSchema:
    """Legacy schema definition – kept for backward compatibility."""

    def __init__(self, columns: List[ColumnSchema]) -> None:
        self.columns = columns
        self.column_map: Dict[str, ColumnSchema] = {
            col.name: col for col in columns
        }

    def get_column(self, name: str) -> Optional[ColumnSchema]:
        """Return column schema by name, or None."""
        return self.column_map.get(name)

    def get_modalities(self) -> List[str]:
        """Return the unique modalities present in this schema."""
        return list({col.modality for col in self.columns})

    def validate(self, data: Dict[str, Any]) -> bool:
        """Check that all non-optional columns are present in *data*."""
        return all(
            col.name in data
            for col in self.columns
            if not col.optional
        )


# ---------------------------------------------------------------------------
# Tier-1 – per-dataset schema
# ---------------------------------------------------------------------------

@dataclass
class IndividualSchema:
    """
    Schema produced by Tier-1 detection for a **single** dataset.

    Attributes:
        dataset_id:        The SHA-256 hash (or name) identifying the dataset.
        detected_columns:  Mapping from modality label to the list of column
                           names assigned to that modality.
                           Keys: "image", "text", "tabular", "timeseries".
        target_column:     Best-guess prediction target column name, or
                           "Unknown" when detection fails.
        problem_type:      One of:  "classification_binary",
                                    "classification_multiclass",
                                    "regression",
                                    "multilabel_classification",
                                    "unsupervised".
        modalities:        Sorted list of modality keys that have at least one
                           column assigned to them.
        confidence:        Detection confidence score in [0, 1].
    """

    dataset_id: str
    detected_columns: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "image": [],
            "text": [],
            "tabular": [],
            "timeseries": [],
        }
    )
    target_column: str = "Unknown"
    problem_type: str = "unsupervised"
    modalities: List[str] = field(default_factory=list)
    confidence: float = 0.0
    target_profile: Dict[str, Any] = field(default_factory=dict)
    reasoning: Dict[str, Any] = field(default_factory=dict)
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    rejected_candidates: List[Dict[str, Any]] = field(default_factory=list)
    preprocessing_hints: Dict[str, Any] = field(default_factory=dict)
    selection_mode: str = "auto"
    semantic_summary: Dict[str, Any] = field(default_factory=dict)
    interaction_summary: Dict[str, Any] = field(default_factory=dict)
    uncertainty_summary: Dict[str, Any] = field(default_factory=dict)
    semantic_roles: Dict[str, List[str]] = field(default_factory=dict)
    business_patterns: Dict[str, Any] = field(default_factory=dict)
    text_task_type: Optional[str] = None
    num_features: int = 0
    has_relational_columns: bool = False
    image_label_separability: float = 0.0
    image_class_balance: float = 0.0
    image_dataset_size: int = 0
    # G7/G8: per-modality feature signals (text vocab/tokens, image resolution/blur, etc.)
    feature_signals: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tier-2 – global / aggregated schema
# ---------------------------------------------------------------------------

@dataclass
class GlobalSchema:
    """
    Unified schema produced by Tier-2 aggregation across **all** datasets
    ingested in the current session.

    This is the canonical object consumed by Phase 3 (Preprocessing) and
    Phase 4 (Model Selection).

    Attributes:
        global_problem_type:   Single resolved problem type for the run.
        global_modalities:     Union of all per-dataset modalities, sorted.
        primary_target:        The single target column chosen from the
                               per-dataset inferences to drive training.
        fusion_ready:          True when more than one modality is present,
                               triggering multimodal fusion in Phase 4.
        detection_confidence:  Mean confidence across all per-dataset
                               IndividualSchema inferences.
        per_dataset:           List of serialised IndividualSchema dicts
                               (one per ingested dataset), giving the
                               frontend full per-file visibility.
    """

    global_problem_type: str
    global_modalities: List[str]
    primary_target: str
    fusion_ready: bool
    detection_confidence: float
    per_dataset: List[Dict[str, Any]] = field(default_factory=list)
    relatedness_report: Dict[str, Any] = field(default_factory=dict)
    semantic_enrichment: Dict[str, Any] = field(default_factory=dict)
    # Architecture routing signals
    total_feature_count: int = 0
    has_relational_columns: bool = False
    # G9: cross-modality alignment signals (complementarity_score, alignment_strength)
    multimodal_signals: Dict[str, float] = field(default_factory=dict)
