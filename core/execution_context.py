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
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import json
import hashlib
import logging

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
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
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
    
    # Phase 6: Model selection (from both contexts)
    model_candidates: List[Dict[str, Any]] = field(default_factory=list)
    model_choices: List[Any] = field(default_factory=list)  # From pipeline context
    selected_model: Optional[str] = None
    model_selection_reason: str = ""
    probe_scores_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    ranked_candidates: Dict[str, List[Any]] = field(default_factory=dict)
    
    # Phase 6: Fusion strategy (from pipeline context)
    fusion_strategy: Optional[str] = None
    modality_importance: Dict[str, float] = field(default_factory=dict)
    
    # Phase 7: Training state
    trial_history_refs: List[str] = field(default_factory=list)
    warm_start_params: Optional[Dict[str, Any]] = None
    training_signals: Dict[str, Any] = field(default_factory=dict)
    active_modalities: List[str] = field(default_factory=list)
    
    # Phase 8: Drift detection & monitoring
    drift_detected: bool = False
    drift_severity: Optional[str] = None
    drift_details: Optional[Dict[str, Any]] = None
    
    # Model registry integration
    registered_model_ids: List[str] = field(default_factory=list)
    active_prediction_model_id: Optional[str] = None
    
    # Explainability (XAI)
    xai_config: Dict[str, Any] = field(default_factory=dict)
    
    # Performance constraints
    constraints: Dict[str, Any] = field(default_factory=dict)
    
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
            'timestamp': datetime.utcnow().isoformat(),
            'field': 'global_target',
            'old_value': old_target,
            'new_value': new_target,
            'reason': reason
        })
        self.log_decision('override', f"User overrode global target: {old_target} -> {new_target}", reason)
        self._update_timestamp()
    
    # ===== Phase Update Methods (from pipeline context) =====
    
    def update_from_schema(self, schema: Dict[str, Any]) -> None:
        """Called by schema detector after schema detection."""
        self.global_schema = schema  # Note: might be overwritten by global aggregation
        self.modality_presence = schema.get("modality_presence", {})
        self.predictability_scores = schema.get("predictability_scores", {})
        self.target_confidence = schema.get("target_confidence", 0.0)
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
        self._compute_version()
        self._update_timestamp()
        logger.info(
            "ExecutionContext: updated preprocessing plan. Modalities configured: %s",
            list(preprocessing_plan.keys())
        )
    
    def update_model_selection(self, candidates: List[Any], reason: str) -> None:
        """Called by model selector after probing."""
        self.model_choices = candidates
        self.model_selection_reason = reason
        self._compute_version()
        self._update_timestamp()
        logger.info(
            "ExecutionContext: model selection updated. Candidates: %s",
            [getattr(c, 'name', str(c)) for c in candidates[:3]]
        )
    
    def update_fusion(self, strategy: Optional[str], importance_weights: Dict[str, float]) -> None:
        """Called by fusion layer after strategy selection."""
        self.fusion_strategy = strategy
        self.fusion_mode = strategy  # Sync with frontend field
        self.modality_importance = importance_weights
        self._compute_version()
        self._update_timestamp()
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
        logger.info(
            "ExecutionContext: training signals updated. Fit type: %s",
            signals.get("fit_type", "unknown")
        )
    
    # ===== Query Methods =====
    
    def get_active_modalities(self) -> List[str]:
        """Return list of modalities with predictability > threshold."""
        threshold = 0.4
        return [
            mod for mod, score in self.predictability_scores.items()
            if score is not None and score > threshold
        ]
    
    def should_include_fusion(self) -> bool:
        """Return True only if 2+ modalities are strong enough."""
        active = self.get_active_modalities()
        return len(active) >= 2
    
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
            'timestamp': datetime.utcnow().isoformat(),
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
            'timestamp': datetime.utcnow().isoformat(),
            'session_id': self.session_id,
            'pipeline_stage': self.pipeline_stage,
            'active_datasets': len(self.active_dataset_ids),
            'global_schema': bool(self.global_schema),
            'global_target': self.global_target,
            'decisions': len(self.execution_log),
            'overrides': len(self.override_history),
            'hash': self.compute_hash()
        }
    
    def compress(self) -> Dict[str, Any]:
        """Return lightweight version for API responses."""
        return {
            "version": self.version,
            "modalities": list(self.modality_presence.keys()),
            "predictability_scores": self.predictability_scores,
            "preprocessing_choices": self.preprocessing_choices,
            "fusion_strategy": self.fusion_strategy,
            "model_name": getattr(self.get_preferred_model(), 'name', None) 
                         if self.get_preferred_model() else None,
            "is_fallback": self.training_signals.get("fallback_activated", False),
        }
    
    # ===== Internal Methods =====
    
    def _extract_preprocessing_choices(self, preprocessing_plan: Dict[str, Dict]) -> Dict[str, Dict]:
        """Extract config choices from full preprocessing plan."""
        choices = {}
        for modality, plan in preprocessing_plan.items():
            choices[modality] = {
                k: v for k, v in plan.items()
                if k in ["tokenizer", "max_length", "augmentation", "scaling"]
            }
        return choices
    
    def _extract_active_modalities(self) -> List[str]:
        """Extract which modalities were actually used."""
        if not self.training_signals:
            return self.get_active_modalities()
        return self.training_signals.get("active_modalities", self.get_active_modalities())
    
    def _compute_version(self) -> None:
        """Hash current context for reproducibility tracking."""
        state_str = (
            str(sorted(self.global_schema.keys() if self.global_schema else [])) +
            str(sorted(self.predictability_scores.items())) +
            str(self.preprocessing_choices) +
            str(self.fusion_strategy) +
            str(sorted(self.modality_importance.items()))
        )
        self.version = hashlib.sha256(state_str.encode()).hexdigest()[:8]
        self.version_timestamp = datetime.utcnow().isoformat()
    
    def _update_timestamp(self) -> None:
        """Update modification timestamp."""
        self.updated_at = datetime.utcnow()


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


def create_execution_context(session_id: str) -> ExecutionContext:
    """Factory function to create new execution context."""
    ctx = ExecutionContext(session_id=session_id)
    ctx.log_decision('initialization', f"Created new execution context for session {session_id}")
    return ctx
