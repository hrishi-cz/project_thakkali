"""
Unified Execution Context - Single Source of Truth
Merged from api/execution_context.py and pipeline/execution_context.py

This is the central intelligence graph that flows through all 8 pipeline phases:
  Phase 1: Data Ingestion
  Phase 2: Schema Detection
  Phase 3: Target Detection
  Phase 4: Global Aggregation
  Phase 5: Preprocessing Planning
  Phase 6: Model Selection
  Phase 7: Training
  Phase 8: Monitoring & Drift Detection

All phases read and write to this unified context to ensure coherent intelligence.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import json
import hashlib
import logging
import uuid

logger = logging.getLogger(__name__)


@dataclass
class DatasetProfile:
    """
    Intelligence unit for one dataset (all phases).
    Stores all decisions made about this dataset.
    """
    dataset_id: str
    source_url: Optional[str] = None
    file_path: Optional[str] = None
    
    # Phase 2: Schema detection
    schema_detected: bool = False
    schema_result: Optional[Dict[str, Any]] = None
    schema_confidence: float = 0.0
    schema_evidence: Optional[str] = None
    
    # Phase 3: Target detection
    target_detected: bool = False
    target_candidates: List[Dict[str, Any]] = field(default_factory=list)
    chosen_target: Optional[str] = None
    target_locked: bool = False
    target_override_reason: Optional[str] = None
    
    # Modality breakdown (from Integrator)
    modality_breakdown: Dict[str, float] = field(default_factory=dict)
    
    # Phase 4: Global compatibility
    global_compatible: bool = False
    compatibility_score: float = 0.0
    compatibility_notes: Optional[str] = None
    
    # Phase 5: Preprocessing plan
    preprocessing_plan: Optional[Dict[str, Any]] = None
    text_task_type: Optional[str] = None
    
    # Phase 6: Embeddings cache
    embeddings_cached: bool = False
    embedding_refs: Optional[Dict[str, str]] = None
    
    # User overrides
    user_overrides: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetProfile':
        """Deserialize from dict."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ExecutionContext:
    """
    Unified execution context for all 8 pipeline phases.
    
    Single source of truth that makes intelligence transferable:
    - Upstream decisions are stored here
    - Downstream stages read from this context instead of recomputing
    - Overrides are tracked and propagated
    - Every stage can see the full reasoning history
    
    Merged from:
    - api/execution_context.py (frontend phases 1-5)
    - pipeline/execution_context.py (backend phases 6-8)
    """
    
    # Session identification
    session_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    revision: int = 0  # Optimistic-lock token persisted with each session write.
    status: str = "active"
    user_id: Optional[str] = None
    project_name: Optional[str] = None
    description: Optional[str] = None
    
    # Active datasets for this run
    active_dataset_ids: List[str] = field(default_factory=list)
    dataset_profiles: Dict[str, DatasetProfile] = field(default_factory=dict)
    
    # Phase 4: Global intelligence (aggregated across datasets)
    global_schema: Optional[Dict[str, Any]] = None
    global_schema_confidence: float = 0.0
    global_target: Optional[str] = None
    global_target_confidence: float = 0.0
    global_target_candidates: List[Dict[str, Any]] = field(default_factory=list)
    
    # Phase 4: Global compatibility
    datasets_compatible: bool = False
    compatibility_matrix: Optional[Dict[str, Any]] = None
    primary_dataset_id: Optional[str] = None  # Fallback when incompatible
    
    # Phase 4: Modality routing
    modality_map: Dict[str, List[str]] = field(default_factory=dict)  # {modality: [dataset_ids]}
    modality_presence: Dict[str, bool] = field(default_factory=dict)  # From pipeline context
    fusion_mode: Optional[str] = None  # "late", "cross_attention", "graph", etc.
    
    # Phase 5: Preprocessing context
    preprocessing_plan: Dict[str, Dict] = field(default_factory=dict)  # Per-modality plans
    preprocessing_choices: Dict[str, Dict] = field(default_factory=dict)  # Extracted choices
    preprocessing_context: Dict[str, Any] = field(default_factory=dict)  # Additional context
    preprocess_plan_version: Optional[str] = None
    artifact_versions: Dict[str, str] = field(default_factory=dict)
    
    # Phase 6: Model selection (from both contexts)
    model_candidates: List[Dict[str, Any]] = field(default_factory=list)
    model_choices: List[Any] = field(default_factory=list)  # From pipeline context
    selected_model: Optional[str] = None
    model_selection_reason: str = ""
    probe_scores_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    ranked_candidates: Dict[str, List[Any]] = field(default_factory=dict)
    
    # Phase 6: Fusion strategy (from pipeline context)
    fusion_strategy: Optional[str] = None
    fusion_policy_locked: bool = False
    fusion_policy_source: Optional[str] = None
    modality_importance: Dict[str, float] = field(default_factory=dict)
    eligible_modalities: List[str] = field(default_factory=list)
    excluded_modalities: Dict[str, str] = field(default_factory=dict)
    
    # Phase 7: Training state
    trial_history_refs: List[str] = field(default_factory=list)
    warm_start_params: Optional[Dict[str, Any]] = None
    training_signals: Dict[str, Any] = field(default_factory=dict)
    active_modalities: List[str] = field(default_factory=list)
    
    # Phase 8: Drift detection & monitoring
    drift_detected: bool = False
    drift_severity: float = 0.0
    drift_details: Dict[str, Any] = field(default_factory=dict)
    drift_adjusted_predictability: Dict[str, float] = field(default_factory=dict)
    drifted_features: List[str] = field(default_factory=list)
    drift_feedback_applied: bool = False
    
    # Model registry integration
    registered_model_ids: List[str] = field(default_factory=list)
    active_prediction_model_id: Optional[str] = None
    # G13: per-modality target overrides (set via /override-target-per-modality)
    per_modality_target_override: Dict[str, str] = field(default_factory=dict)
    
    # Explainability (XAI)
    xai_config: Dict[str, Any] = field(default_factory=dict)
    
    # Performance constraints
    constraints: Dict[str, Any] = field(default_factory=dict)
    latency_budget_ms: Optional[float] = None
    memory_budget_mb: Optional[float] = None
    
    # User overrides (global)
    user_overrides: Dict[str, Any] = field(default_factory=dict)
    override_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Execution history (audit trail)
    execution_log: List[Dict[str, Any]] = field(default_factory=list)
    pipeline_stage: Optional[str] = None
    
    # Confidence tracking
    confidence_map: Dict[str, float] = field(default_factory=dict)
    predictability_scores: Dict[str, float] = field(default_factory=dict)  # From pipeline
    target_confidence: float = 0.0  # From pipeline
    
    # Version control
    version: Optional[str] = None
    version_timestamp: Optional[str] = None
    phase_timings: Dict[str, float] = field(default_factory=dict)
    training_fit_analysis: Dict[str, Any] = field(default_factory=dict)

    # Architecture routing (set by run_architecture_selection before model selection)
    head_architecture_type: str = "mlp"   # "mlp" | "attention" | "graph"
    head_hidden_dim: int = 256
    head_num_layers: int = 3
    encoder_output_dims: Dict[str, int] = field(
        default_factory=lambda: {"tabular": 16, "text": 768, "image": 512}
    )
    encoder_plan: Dict[str, Any] = field(default_factory=dict)

    # Drift policy (set by drift detector)
    retraining_depth_required: str = "none"  # "none"|"calibration_only"|"head_only"|"full"

    # Aggregated schema intelligence (semantic/interaction/uncertainty summaries
    # extracted from IndividualSchema per-dataset results during Phase 2).
    # Keys: dataset_id → {semantic_roles, business_patterns, semantic_summary,
    #                      interaction_summary, uncertainty_summary, avg_text_len,
    #                      high_missing_cols, id_columns, n_features,
    #                      image_label_separability, image_class_balance,
    #                      image_dataset_size, text_task_type}
    feature_intelligence: Dict[str, Any] = field(default_factory=dict)

    # Label noise detection results (set by LabelNoiseDetector in training phase)
    suspicious_label_indices: List[int] = field(default_factory=list)
    
    # ===== Dataset Management Methods =====
    
    def add_dataset_profile(self, profile: DatasetProfile) -> None:
        """Add a dataset profile to this context."""
        self.dataset_profiles[profile.dataset_id] = profile
        if profile.dataset_id not in self.active_dataset_ids:
            self.active_dataset_ids.append(profile.dataset_id)
        logger.info("Added dataset profile to context: %s", profile.dataset_id)
    
    def get_dataset_profile(self, dataset_id: str) -> Optional[DatasetProfile]:
        """Get a dataset profile by ID."""
        return self.dataset_profiles.get(dataset_id)
    
    def get_active_profiles(self) -> List[DatasetProfile]:
        """Get all active dataset profiles."""
        return [self.dataset_profiles[did] for did in self.active_dataset_ids 
                if did in self.dataset_profiles]
    
    # ===== Global Intelligence Methods (Phase 4) =====
    
    def set_global_schema(self, schema: Dict[str, Any], confidence: float, evidence: Optional[str] = None) -> None:
        """Set global schema (Phase 4 aggregation)."""
        self.global_schema = schema
        self.global_schema_confidence = confidence
        self.confidence_map['global_schema'] = confidence
        self.log_decision('global_schema', f"Set global schema with confidence {confidence:.2f}", evidence)
        self._update_timestamp()
    
    def set_global_target(self, target: str, confidence: float, candidates: List[Dict[str, Any]]) -> None:
        """Set global target (Phase 4 aggregation)."""
        self.global_target = target
        self.global_target_confidence = confidence
        self.global_target_candidates = candidates
        self.confidence_map['global_target'] = confidence
        self.target_confidence = confidence  # Sync with pipeline field
        self.log_decision('global_target', f"Set global target: {target} (confidence: {confidence:.2f})")
        self._update_timestamp()
    
    def override_global_target(self, new_target: str, reason: str) -> None:
        """Override global target with user choice."""
        old_target = self.global_target
        self.global_target = new_target
        self.user_overrides['global_target'] = new_target
        self.override_history.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'field': 'global_target',
            'old_value': old_target,
            'new_value': new_target,
            'reason': reason
        })
        self.log_decision('override', f"User overrode global target: {old_target} -> {new_target}", reason)
        self._update_timestamp()

    def override_fusion_strategy(self, strategy: str, reason: str) -> None:
        """
        Override fusion strategy with user choice and immediately propagate it.

        Unlike storing in user_overrides alone, this calls update_fusion() so
        the training orchestrator sees the change reflected in ctx.fusion_strategy
        without any manual follow-up.
        """
        old_strategy = self.fusion_strategy
        self.user_overrides['fusion_strategy'] = strategy
        self.override_history.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'field': 'fusion_strategy',
            'old_value': old_strategy,
            'new_value': strategy,
            'reason': reason,
        })
        # Propagate immediately — unlock and apply
        self.fusion_policy_locked = False
        self.update_fusion(
            strategy=strategy,
            importance_weights=dict(self.modality_importance or {}),
        )
        self.fusion_policy_source = "user_override"
        self.log_decision(
            'override',
            f"User overrode fusion strategy: {old_strategy} → {strategy}",
            reason,
        )
        self._update_timestamp()

    def set_feature_intelligence(self, per_dataset_results: list) -> None:
        """
        Aggregate semantic intelligence from per-dataset IndividualSchema dicts
        (produced by asdict(individual_schema) in schema_detector) into a
        per-dataset lookup that ALL downstream phases can read.

        Called once during Phase 2 after global schema is assembled.
        """
        aggregated: Dict[str, Any] = {}
        for ds in (per_dataset_results or []):
            if not isinstance(ds, dict):
                continue
            dataset_id = str(ds.get("dataset_id") or "")
            if not dataset_id:
                continue

            # Text length heuristic — average across all text columns
            semantic_summary = dict(ds.get("semantic_summary") or {})
            interaction_summary = dict(ds.get("interaction_summary") or {})
            uncertainty_summary = dict(ds.get("uncertainty_summary") or {})
            semantic_roles: Dict[str, list] = dict(ds.get("semantic_roles") or {})
            business_patterns: Dict[str, Any] = dict(ds.get("business_patterns") or {})

            avg_text_len: float = float(
                ds.get("reasoning", {}).get("avg_text_len", 0.0) or 0.0
            )
            n_features: int = int(ds.get("num_features", 0) or 0)
            image_label_separability: float = float(
                ds.get("image_label_separability", 0.0) or 0.0
            )
            image_class_balance: float = float(
                ds.get("image_class_balance", 0.0) or 0.0
            )
            image_dataset_size: int = int(ds.get("image_dataset_size", 0) or 0)
            text_task_type: Optional[str] = ds.get("text_task_type")
            id_columns: list = list(semantic_roles.get("id_columns", []))
            high_missing_cols: list = list(
                business_patterns.get("high_missing_columns", [])
            )
            long_tail_cats: list = list(
                business_patterns.get("long_tail_categoricals", [])
            )
            preprocessing_hints: Dict[str, Any] = dict(ds.get("preprocessing_hints") or {})

            aggregated[dataset_id] = {
                "semantic_summary": semantic_summary,
                "interaction_summary": interaction_summary,
                "uncertainty_summary": uncertainty_summary,
                "semantic_roles": semantic_roles,
                "business_patterns": business_patterns,
                "avg_text_len": avg_text_len,
                "n_features": n_features,
                "id_columns": id_columns,
                "high_missing_cols": high_missing_cols,
                "long_tail_cats": long_tail_cats,
                "image_label_separability": image_label_separability,
                "image_class_balance": image_class_balance,
                "image_dataset_size": image_dataset_size,
                "text_task_type": text_task_type,
                "preprocessing_hints": preprocessing_hints,
            }

        self.feature_intelligence = aggregated
        self.log_decision(
            "feature_intelligence",
            f"Semantic intelligence aggregated for {len(aggregated)} dataset(s)",
            evidence=f"datasets={list(aggregated.keys())}",
        )
        self._update_timestamp()
    
    # ===== Phase Update Methods (from pipeline context) =====
    
    def update_from_schema(self, schema: Dict[str, Any]) -> None:
        """Called by schema detector after schema detection."""
        self.global_schema = schema  # Note: might be overwritten by global aggregation
        self.modality_presence = schema.get("modality_presence", {})
        self.predictability_scores = schema.get("predictability_scores", {})
        self.target_confidence = schema.get("target_confidence", 0.0)
        _mm = schema.get("multimodal_signals", {})
        if _mm:
            self.preprocessing_context["multimodal_signals"] = dict(_mm)

        for ds in schema.get("per_dataset", []):
            if not isinstance(ds, dict):
                continue
            did = ds.get("dataset_id", "")
            pred = ds.get("target_profile", {}).get("predictability_score", 0.0)
            if did and pred:
                self.predictability_scores[did] = float(pred)

        self._compute_version()
        self._update_timestamp()
        logger.info(
            "ExecutionContext: updated from schema. Modalities: %s",
            list(self.modality_presence.keys())
        )
    
    def update_preprocessing(self, preprocessing_plan: Dict[str, Dict]) -> None:
        """Called by preprocessing orchestrator after Phase 5 setup."""
        self.preprocessing_plan = preprocessing_plan
        self.preprocessing_choices = self._extract_preprocessing_choices(preprocessing_plan)
        plan_str = json.dumps(preprocessing_plan or {}, sort_keys=True, default=str)
        self.preprocess_plan_version = hashlib.sha256(plan_str.encode()).hexdigest()[:8]
        self.artifact_versions["preprocessing_plan"] = self.preprocess_plan_version
        self._compute_version()
        self._update_timestamp()
        self.log_decision(
            "preprocessing",
            (
                "Preprocessing plan updated: "
                f"version={self.preprocess_plan_version}, "
                f"choices={list(self.preprocessing_choices.keys())}"
            ),
            evidence=f"plan_keys={list((self.preprocessing_plan or {}).keys())}",
        )
        logger.info(
            "ExecutionContext: updated preprocessing plan. Modalities configured: %s",
            list(preprocessing_plan.keys())
        )

    def update_preprocessing_contract(
        self,
        preprocessing_plan: Dict[str, Dict],
        preprocessing_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist preprocessing plan plus the adaptive context contract."""
        self.update_preprocessing(preprocessing_plan)

        contract = dict(preprocessing_context or {})
        if not contract:
            return

        runtime = contract.get("runtime")
        if isinstance(runtime, dict):
            self.preprocessing_context["runtime"] = dict(runtime)

        for key in ("weak_modalities", "strong_modalities", "drifted_features"):
            values = contract.get(key)
            if values is not None:
                self.preprocessing_context[key] = [str(value) for value in list(values or [])]

        modality_predictability = contract.get("modality_predictability")
        if isinstance(modality_predictability, dict):
            self.preprocessing_context["modality_predictability"] = {
                str(k): float(v)
                for k, v in modality_predictability.items()
                if isinstance(v, (int, float))
            }

        context_signals = contract.get("context_signals")
        if isinstance(context_signals, dict):
            self.preprocessing_context["context_signals"] = dict(context_signals)

        validation = contract.get("validation")
        if isinstance(validation, dict):
            self.preprocessing_context["validation"] = dict(validation)

        if "dataset_total_samples" in contract and contract.get("dataset_total_samples") is not None:
            try:
                self.preprocessing_context["dataset_total_samples"] = int(contract["dataset_total_samples"])
            except Exception:
                pass

        fusion_recommendation = contract.get("fusion_recommendation")
        if fusion_recommendation is not None:
            self.preprocessing_context["fusion_recommendation"] = str(fusion_recommendation)

        adaptive_tabular_config = contract.get("adaptive_tabular_config")
        if isinstance(adaptive_tabular_config, dict):
            self.preprocessing_context["adaptive_tabular_config"] = dict(adaptive_tabular_config)

        dataset_plans = contract.get("dataset_plans")
        if isinstance(dataset_plans, dict):
            self.preprocessing_context["dataset_plans"] = dict(dataset_plans)

        self._update_timestamp()
    
    def update_model_selection(self, candidates: List[Any], reason: str) -> None:
        """Called by model selector after probing.

        Raises ValueError when the candidate list is empty so callers get an
        explicit, actionable error rather than a silent None `selected_model`.
        """
        if not candidates:
            logger.warning(
                "update_model_selection: empty candidates list (reason=%s). "
                "All model candidates were filtered out — check VRAM budget, "
                "predictability thresholds, or modality configuration.",
                reason,
            )
        self.model_choices = candidates
        self.model_selection_reason = reason

        # Populate previously-empty fields: model_candidates (serialisable list)
        # and selected_model (name of the top candidate).
        self.model_candidates = [
            (c if isinstance(c, dict) else {"name": getattr(c, "name", str(c))})
            for c in (candidates or [])
        ]
        if candidates:
            first = candidates[0]
            top_name = (
                first.get("name") if isinstance(first, dict)
                else getattr(first, "name", str(first))
            )
            self.selected_model = str(top_name) if top_name else None

        self.eligible_modalities = []
        self.excluded_modalities = {}
        if candidates:
            first = candidates[0]
            if isinstance(first, dict):
                self.eligible_modalities = list(first.get("eligible_modalities", []) or [])
                self.excluded_modalities = dict(first.get("excluded_modalities", {}) or {})

        model_selection_state = {
            "top": str(candidates[0]) if candidates else "none",
            "count": len(candidates or []),
            "eligible": list(self.eligible_modalities),
            "excluded": dict(self.excluded_modalities),
        }
        self.artifact_versions["model_selection"] = hashlib.sha256(
            json.dumps(model_selection_state, sort_keys=True, default=str).encode()
        ).hexdigest()[:8]
        self._compute_version()
        self._update_timestamp()
        self.log_decision(
            "model_selection",
            (
                "Model candidates evaluated: "
                f"n={len(candidates or [])}, "
                f"reason={(self.model_selection_reason or '')[:200]}"
            ),
            evidence=(
                f"selected={self.selected_model}; "
                f"eligible_modalities={self.eligible_modalities}; "
                f"excluded={self.excluded_modalities}"
            ),
        )
        logger.info(
            "ExecutionContext: model selection updated. Candidates: %s",
            [getattr(c, 'name', str(c)) for c in candidates[:3]]
        )
    
    def update_fusion(self, strategy: Optional[str], importance_weights: Dict[str, float]) -> None:
        """Called by fusion layer after strategy selection."""
        self.fusion_strategy = strategy
        self.fusion_mode = strategy  # Sync with frontend field
        self.modality_importance = importance_weights
        self.fusion_policy_locked = bool(strategy)
        self.fusion_policy_source = "context_update"
        fusion_state = {
            "strategy": strategy,
            "importance": dict(importance_weights or {}),
            "locked": self.fusion_policy_locked,
        }
        self.artifact_versions["fusion_policy"] = hashlib.sha256(
            json.dumps(fusion_state, sort_keys=True, default=str).encode()
        ).hexdigest()[:8]
        self._compute_version()
        self._update_timestamp()
        self.log_decision(
            "fusion",
            f"Fusion strategy selected: {self.fusion_strategy}",
            evidence=(
                f"modality_importance={self.modality_importance}; "
                f"policy_locked={self.fusion_policy_locked}; "
                f"source={self.fusion_policy_source}"
            ),
        )
        logger.info(
            "ExecutionContext: fusion updated. Strategy: %s",
            strategy
        )
    
    def update_training(self, signals: Dict[str, Any]) -> None:
        """Called by trainer after trial completion."""
        self.training_signals = signals
        self.active_modalities = self._extract_active_modalities()
        self._compute_version()
        self._update_timestamp()
        self.log_decision(
            "training",
            (
                "Training completed: "
                f"signals_keys={list((self.training_signals or {}).keys())}"
            ),
            evidence=f"active_modalities={self.active_modalities}",
        )
        logger.info(
            "ExecutionContext: training signals updated. Fit type: %s",
            signals.get("fit_type", "unknown")
        )

    def record_phase_timing(self, phase_name: str, duration_s: float) -> None:
        """Record per-phase duration for API/UX diagnostics."""
        self.phase_timings[str(phase_name)] = round(float(duration_s), 3)
        self._update_timestamp()

    def update_drift(
        self,
        detected: bool,
        severity: float,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist drift summary from monitoring/training Phase 6."""
        self.drift_detected = bool(detected)
        self.drift_severity = float(severity)
        self.drift_details = dict(details or {})
        self.log_decision(
            "drift_detection",
            f"drift_detected={self.drift_detected}",
            f"composite={self.drift_severity:.4f}",
        )
        self._update_timestamp()

    def update_fit_analysis(self, analysis: Optional[Dict[str, Any]]) -> None:
        """Persist latest TrialIntelligence analysis from LossWeightScheduler."""
        latest = dict(analysis or {})
        existing = dict(self.training_fit_analysis or {})
        for key in (
            "predictability_factors",
            "training_adjusted_predictability",
            "feedback_applied",
            "next_run_feedback",
        ):
            if key not in latest and key in existing:
                latest[key] = existing[key]

        self.training_fit_analysis = latest
        self._compute_version()
        self._update_timestamp()
        self.log_decision(
            "training_fit_analysis",
            (
                "Fit analysis updated: "
                f"{(self.training_fit_analysis or {}).get('fit_type', 'unknown')}"
            ),
            evidence=(
                f"train_slope={(self.training_fit_analysis or {}).get('train_slope')}; "
                f"val_slope={(self.training_fit_analysis or {}).get('val_slope')}; "
                f"gap={(self.training_fit_analysis or {}).get('generalization_gap')}"
            ),
        )

    @staticmethod
    def _resolve_predictability_factor(
        score_key: str,
        factors: Dict[str, float],
    ) -> float:
        """Resolve modality-level factor for a predictability score key."""
        if not factors:
            return 1.0

        normalized_key = str(score_key).lower()
        direct = factors.get(score_key)
        if direct is not None:
            try:
                return float(direct)
            except Exception:
                return 1.0

        direct_norm = factors.get(normalized_key)
        if direct_norm is not None:
            try:
                return float(direct_norm)
            except Exception:
                return 1.0

        for modality, factor in factors.items():
            modality_key = str(modality).lower()
            if not modality_key:
                continue
            if normalized_key == modality_key or modality_key in normalized_key:
                try:
                    return float(factor)
                except Exception:
                    return 1.0

        return 1.0

    def apply_training_feedback(
        self,
        analysis: Optional[Dict[str, Any]],
        predictability_factors: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Persist trial feedback and map it into predictability adjustments.

        This keeps later preprocessing/model-selection phases context-driven
        without requiring those consumers to know trial-level details.
        """
        payload = dict(analysis or {})

        raw_factors = predictability_factors
        if raw_factors is None:
            next_run = payload.get("next_run_feedback")
            if isinstance(next_run, dict):
                raw_factors = next_run.get("predictability_factors")

        sanitized_factors: Dict[str, float] = {}
        for key, value in dict(raw_factors or {}).items():
            try:
                factor = float(value)
            except Exception:
                continue
            sanitized_factors[str(key)] = max(0.30, min(1.50, factor))

        base_scores = dict(self.predictability_scores or {})
        if self.drift_feedback_applied and self.drift_adjusted_predictability:
            base_scores = dict(self.drift_adjusted_predictability)

        adjusted_scores: Dict[str, float] = {}
        for key, value in base_scores.items():
            try:
                base_value = float(value)
            except Exception:
                continue
            factor = self._resolve_predictability_factor(str(key), sanitized_factors)
            adjusted_scores[str(key)] = round(max(0.0, base_value * factor), 4)

        payload["predictability_factors"] = sanitized_factors
        payload["training_adjusted_predictability"] = adjusted_scores
        payload["feedback_applied"] = bool(sanitized_factors and adjusted_scores)
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()

        self.training_fit_analysis = payload
        self._compute_version()
        self._update_timestamp()

    def apply_drift_feedback(
        self,
        drift_report_dict: Dict[str, Any],
        decay: float = 0.5,
    ) -> None:
        """
        Apply closed-loop drift feedback to predictability scores.

        Uses schema-importance weighting: features with high interaction scores
        in feature_intelligence cause a larger predictability reduction when they
        drift, because they matter more to the model's signal.  Features with
        low or unknown importance apply only the base uniform decay.
        """
        per_feature_ks: Dict[str, float] = dict(
            (drift_report_dict or {}).get("per_feature_ks", {}) or {}
        )
        drifted = [feat for feat, ks in per_feature_ks.items() if float(ks) > 0.30]
        self.drifted_features = drifted

        # Build a schema importance map: feature → interaction score (0–1)
        # Aggregated across all datasets in feature_intelligence.
        importance_map: Dict[str, float] = {}
        for _ds_intel in (self.feature_intelligence or {}).values():
            for feat, score in (_ds_intel.get("interaction_summary") or {}).items():
                try:
                    existing = importance_map.get(str(feat), 0.0)
                    importance_map[str(feat)] = max(existing, float(score))
                except Exception:
                    pass

        current_scores = dict(self.predictability_scores or {})
        if drifted and current_scores:
            # Compute a weighted drift fraction: each drifted feature contributes
            # proportionally to its schema importance (default weight 1.0 if unknown).
            total_weight = sum(
                importance_map.get(f, 1.0) for f in per_feature_ks
            ) or 1.0
            drifted_weight = sum(importance_map.get(f, 1.0) for f in drifted)
            weighted_drift_fraction = drifted_weight / total_weight

            factor = max(0.0, 1.0 - float(decay) * weighted_drift_fraction)
            for key, value in list(current_scores.items()):
                key_str = str(key).lower()
                if "tabular" in key_str:
                    try:
                        current_scores[key] = round(float(value) * factor, 4)
                    except Exception:
                        continue

        self.drift_adjusted_predictability = current_scores
        self.drift_feedback_applied = bool(drifted)
        self.log_decision(
            "drift_feedback",
            f"{len(drifted)} drifted features (schema-weighted)",
            f"adjusted_predictability={current_scores}; "
            f"importance_map_size={len(importance_map)}",
        )
        self._update_timestamp()

    def get_effective_predictability_scores(self) -> Dict[str, float]:
        """
        Return predictability scores after drift feedback when available.

        This keeps downstream planning/model-selection consumers aligned on
        one authoritative predictability view.
        """
        source = dict(self.predictability_scores or {})
        if self.drift_feedback_applied and self.drift_adjusted_predictability:
            source = dict(self.drift_adjusted_predictability)

        fit_analysis = dict(self.training_fit_analysis or {})
        raw_factors = fit_analysis.get("predictability_factors")
        factor_map: Dict[str, float] = {}
        if isinstance(raw_factors, dict):
            for key, value in raw_factors.items():
                try:
                    factor_map[str(key)] = float(value)
                except Exception:
                    continue

        if factor_map:
            adjusted: Dict[str, float] = {}
            for key, value in source.items():
                try:
                    score = float(value)
                except Exception:
                    continue
                factor = self._resolve_predictability_factor(str(key), factor_map)
                adjusted[str(key)] = max(0.0, score * factor)
            if adjusted:
                source = adjusted

        normalized: Dict[str, float] = {}
        for key, value in source.items():
            try:
                normalized[str(key)] = float(value)
            except Exception:
                continue
        return normalized

    def get_preprocessing_signals(self) -> Dict[str, Any]:
        """Return context signals required by adaptive preprocessing planners."""
        return {
            "global_schema": dict(self.global_schema or {}),
            "modality_presence": dict(self.modality_presence or {}),
            "predictability_scores": self.get_effective_predictability_scores(),
            "drift_adjusted_predictability": dict(self.drift_adjusted_predictability or {}),
            "drifted_features": list(self.drifted_features or []),
            "drift_feedback_applied": bool(self.drift_feedback_applied),
            "training_fit_analysis": dict(self.training_fit_analysis or {}),
            "feature_intelligence": dict(self.feature_intelligence or {}),
            "encoder_plan": dict(self.encoder_plan or {}),
        }
    
    # ===== Query Methods =====
    
    def get_active_modalities(self) -> List[str]:
        """Return list of active modalities.

        Prefers modality_presence (set from schema) over predictability_scores,
        since predictability_scores are keyed by dataset_id (not modality name)
        and only reflect tabular-feature signal — text/image modalities never
        appear there even when they are fully active.
        """
        if self.active_modalities:
            return [
                str(m) for m in self.active_modalities
                if str(m) not in set(self.excluded_modalities or {})
            ]
        if self.eligible_modalities:
            return [
                str(m) for m in self.eligible_modalities
                if str(m) not in set(self.excluded_modalities or {})
            ]
        if self.modality_presence:
            return [
                m for m, present in self.modality_presence.items()
                if present and m not in set(self.excluded_modalities or {})
            ]
        # Final fallback: predictability threshold (tabular-only signal)
        threshold = 0.4
        return [
            mod for mod, score in self.predictability_scores.items()
            if score is not None and score > threshold
        ]

    def should_include_fusion(self) -> bool:
        """Return True when 2+ modalities are present (regardless of predictability)."""
        # Count modalities from presence map — text and image are always active
        # when detected, even without tabular predictability scores.
        active = self.get_active_modalities()
        if len(active) >= 2:
            return True
        # Fallback: check modality_presence directly
        present_count = sum(1 for v in self.modality_presence.values() if v)
        return present_count >= 2
    
    def get_preprocessing_config(self, modality: str) -> Dict[str, Any]:
        """Get preprocessing config for specific modality."""
        return self.preprocessing_plan.get(modality, {})
    
    def get_modality_importance(self, modality: str) -> float:
        """Get importance weight for modality (0-1)."""
        return self.modality_importance.get(modality, 0.0)
    
    def get_preferred_model(self) -> Optional[Any]:
        """Get top-ranked model choice."""
        return self.model_choices[0] if self.model_choices else None
    
    def get_ranked_candidates(self, modality: str) -> List[Any]:
        """Retrieve ranked candidates for modality from Phase 6 cache."""
        return self.ranked_candidates.get(modality, [])
    
    def get_probe_scores(self, modality: str) -> Dict[str, Any]:
        """Retrieve probe scores for modality from Phase 6 cache."""
        return self.probe_scores_cache.get(modality, {})
    
    # ===== Logging & Audit Methods =====
    
    def log_decision(self, stage: str, decision: str, evidence: Optional[str] = None) -> None:
        """Log a pipeline decision for explainability."""
        entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'stage': stage,
            'decision': decision,
            'evidence': evidence
        }
        self.execution_log.append(entry)
        logger.info("[%s] %s", stage, decision)
    
    def set_pipeline_stage(self, stage: str) -> None:
        """Update current pipeline stage."""
        self.pipeline_stage = stage
        self.log_decision('pipeline', f"Entered stage: {stage}")
        self._update_timestamp()
    
    # ===== Serialization Methods =====
    
    def compute_hash(self) -> str:
        """Compute hash of current context state (for version control)."""
        state = {
            'session_id': self.session_id,
            'active_dataset_ids': sorted(self.active_dataset_ids),
            'global_schema': self.global_schema,
            'global_target': self.global_target,
            'user_overrides': self.user_overrides,
            'fusion_mode': self.fusion_mode
        }
        state_str = json.dumps(state, sort_keys=True)
        return hashlib.sha256(state_str.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for storage/transmission."""
        data = asdict(self)
        # Convert datetime
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        # Convert dataset profiles
        data['dataset_profiles'] = {
            k: v.to_dict() if isinstance(v, DatasetProfile) else v 
            for k, v in self.dataset_profiles.items()
        }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionContext':
        """Deserialize from dict."""
        # Convert datetime
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if isinstance(data.get('updated_at'), str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])

        try:
            data['revision'] = int(data.get('revision', 0) or 0)
        except Exception:
            data['revision'] = 0
        
        # Convert dataset profiles
        if 'dataset_profiles' in data:
            data['dataset_profiles'] = {
                k: DatasetProfile.from_dict(v) if isinstance(v, dict) else v
                for k, v in data['dataset_profiles'].items()
            }
        
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def snapshot(self) -> Dict[str, Any]:
        """Create a snapshot of current state for checkpointing."""
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'session_id': self.session_id,
            'revision': int(self.revision or 0),
            'pipeline_stage': self.pipeline_stage,
            'active_datasets': len(self.active_dataset_ids),
            'global_schema': bool(self.global_schema),
            'global_target': self.global_target,
            'decisions': len(self.execution_log),
            'overrides': len(self.override_history),
            'hash': self.compute_hash()
        }
    
    # ===== Internal Methods =====
    
    def _extract_preprocessing_choices(self, preprocessing_plan: Dict[str, Dict]) -> Dict[str, Dict]:
        """Extract config choices from full preprocessing plan."""
        choices = {}
        for modality, plan in preprocessing_plan.items():
            choices[modality] = {
                k: v for k, v in plan.items()
                if k in ["tokenizer", "max_length", "pooling", "target_size", "augmentation", "scaling", "scaler"]
            }
        return choices
    
    def _extract_active_modalities(self) -> List[str]:
        """Extract which modalities were actually used."""
        if not self.training_signals:
            return self.get_active_modalities()
        return self.training_signals.get("active_modalities", self.get_active_modalities())
    
    def _compute_version(self) -> None:
        """Hash current context for reproducibility tracking."""
        fit_summary = dict(self.training_fit_analysis or {})
        state_str = (
            str(sorted(self.global_schema.keys() if self.global_schema else [])) +
            str(sorted(self.predictability_scores.items())) +
            str(self.preprocessing_choices) +
            str(self.preprocess_plan_version) +
            str(sorted(self.artifact_versions.items())) +
            str(self.fusion_strategy) +
            str(self.fusion_policy_locked) +
            str(self.fusion_policy_source) +
            str(sorted(dict(self.encoder_plan or {}).items())) +
            str(sorted(self.eligible_modalities)) +
            str(sorted(self.excluded_modalities.items())) +
            str(sorted(self.modality_importance.items())) +
            str(fit_summary.get("fit_type")) +
            str(sorted(dict(fit_summary.get("predictability_factors", {}) or {}).items()))
        )
        self.version = hashlib.sha256(state_str.encode()).hexdigest()[:8]
        self.version_timestamp = datetime.now(timezone.utc).isoformat()

    def update_timestamp(self) -> None:
        """Public timestamp update helper for external callers."""
        self._update_timestamp()
    
    def _update_timestamp(self) -> None:
        """Update modification timestamp."""
        self.updated_at = datetime.now(timezone.utc)


# ===== Context Validation =====

def validate_context(ctx: ExecutionContext, stage: str) -> List[str]:
    """
    Validate that context has required intelligence for a given stage.
    Returns list of missing/invalid elements.
    """
    errors = []
    
    if stage == 'preprocessing':
        # Preprocessing requires schema and target for each active dataset
        for dataset_id in ctx.active_dataset_ids:
            profile = ctx.get_dataset_profile(dataset_id)
            if not profile:
                errors.append(f"Missing profile for dataset {dataset_id}")
            elif not profile.schema_detected:
                errors.append(f"Schema not detected for dataset {dataset_id}")
            elif not profile.chosen_target and not ctx.global_target:
                errors.append(f"No target chosen for dataset {dataset_id} and no global target")
    
    elif stage == 'model_selection':
        # Model selection requires preprocessing plans
        for dataset_id in ctx.active_dataset_ids:
            profile = ctx.get_dataset_profile(dataset_id)
            if profile and not profile.preprocessing_plan:
                errors.append(f"No preprocessing plan for dataset {dataset_id}")
    
    elif stage == 'training':
        # Training requires model selection
        if not ctx.selected_model and not ctx.model_candidates and not ctx.model_choices:
            errors.append("No model selected or candidates available")
        
        # Check fusion requirements
        if ctx.should_include_fusion() and not ctx.fusion_strategy and not ctx.fusion_mode:
            errors.append("Multimodal context requires fusion strategy")
    
    elif stage == 'monitoring':
        # Monitoring requires trained model
        if not ctx.registered_model_ids and not ctx.active_prediction_model_id:
            errors.append("No trained model registered for monitoring")
    
    return errors


def create_execution_context(
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ExecutionContext:
    """Factory function to create new execution context."""
    resolved_session_id = session_id
    if not resolved_session_id:
        resolved_session_id = f"session_{uuid.uuid4().hex[:12]}"
    metadata = metadata or {}
    ctx = ExecutionContext(
        session_id=resolved_session_id,
        user_id=metadata.get("user_id"),
        project_name=metadata.get("project_name"),
        description=metadata.get("description"),
        status="active",
    )
    ctx.log_decision('initialization', f"Created new execution context for session {resolved_session_id}")
    return ctx
