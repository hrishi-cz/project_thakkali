"""
Execution Context - Phase 6
The intelligence transfer object that flows through all pipeline stages.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import json
import hashlib
import logging

logger = logging.getLogger(__name__)


@dataclass
class DatasetProfile:
    """
    Intelligence unit for one dataset (Phases 3-5).
    Stores all decisions made about this dataset.
    """
    dataset_id: str
    source_url: Optional[str] = None
    file_path: Optional[str] = None
    
    # Schema detection (Phase 3)
    schema_detected: bool = False
    schema_result: Optional[Dict[str, Any]] = None
    schema_confidence: float = 0.0
    schema_evidence: Optional[str] = None
    
    # Target detection (Phase 4)
    target_detected: bool = False
    target_candidates: List[Dict[str, Any]] = field(default_factory=list)
    chosen_target: Optional[str] = None
    target_locked: bool = False
    target_override_reason: Optional[str] = None
    
    # Modality breakdown
    modality_breakdown: Dict[str, float] = field(default_factory=dict)
    
    # Compatibility (Phase 5)
    global_compatible: bool = False
    compatibility_score: float = 0.0
    compatibility_notes: Optional[str] = None
    
    # Preprocessing plan (Phase 7)
    preprocessing_plan: Optional[Dict[str, Any]] = None
    
    # Embeddings (Phase 8)
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
    The intelligence graph that flows through every pipeline stage (Phase 6).
    
    This is the core mechanism that makes intelligence transferable:
    - Upstream decisions are stored here
    - Downstream stages read from this context instead of recomputing
    - Overrides are tracked and propagated
    - Every stage can see the full reasoning history
    """
    
    # Session identification
    session_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Active datasets for this run
    active_dataset_ids: List[str] = field(default_factory=list)
    dataset_profiles: Dict[str, DatasetProfile] = field(default_factory=dict)
    
    # Global intelligence (Phase 5)
    global_schema: Optional[Dict[str, Any]] = None
    global_schema_confidence: float = 0.0
    global_target: Optional[str] = None
    global_target_confidence: float = 0.0
    global_target_candidates: List[Dict[str, Any]] = field(default_factory=list)
    
    # Global compatibility
    datasets_compatible: bool = False
    compatibility_matrix: Optional[Dict[str, Any]] = None
    primary_dataset_id: Optional[str] = None  # Fallback when incompatible
    
    # Modality routing
    modality_map: Dict[str, List[str]] = field(default_factory=dict)  # {modality: [dataset_ids]}
    fusion_mode: Optional[str] = None  # "late", "cross_attention", "graph", etc.
    
    # Preprocessing (Phase 7)
    preprocessing_context: Dict[str, Any] = field(default_factory=dict)
    
    # Model selection (Phase 10)
    model_candidates: List[Dict[str, Any]] = field(default_factory=list)
    selected_model: Optional[str] = None
    
    # Trial memory (Phase 10)
    trial_history_refs: List[str] = field(default_factory=list)
    warm_start_params: Optional[Dict[str, Any]] = None
    
    # Drift state (Phase 13)
    drift_detected: bool = False
    drift_severity: Optional[str] = None
    drift_details: Optional[Dict[str, Any]] = None
    
    # Registry references (Phase 11)
    registered_model_ids: List[str] = field(default_factory=list)
    active_prediction_model_id: Optional[str] = None
    
    # User overrides (all phases)
    user_overrides: Dict[str, Any] = field(default_factory=dict)
    override_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Execution history
    execution_log: List[Dict[str, Any]] = field(default_factory=list)
    pipeline_stage: Optional[str] = None
    
    # Confidence tracking
    confidence_map: Dict[str, float] = field(default_factory=dict)
    
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
    
    def set_global_schema(self, schema: Dict[str, Any], confidence: float, evidence: Optional[str] = None) -> None:
        """Set global schema (Phase 5)."""
        self.global_schema = schema
        self.global_schema_confidence = confidence
        self.confidence_map['global_schema'] = confidence
        self.log_decision('global_schema', f"Set global schema with confidence {confidence:.2f}", evidence)
    
    def set_global_target(self, target: str, confidence: float, candidates: List[Dict[str, Any]]) -> None:
        """Set global target (Phase 5)."""
        self.global_target = target
        self.global_target_confidence = confidence
        self.global_target_candidates = candidates
        self.confidence_map['global_target'] = confidence
        self.log_decision('global_target', f"Set global target: {target} (confidence: {confidence:.2f})")
    
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
    
    def compute_hash(self) -> str:
        """Compute hash of current context state (for version control)."""
        # Serialize key fields
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


# Context validation helper
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
        if not ctx.selected_model and not ctx.model_candidates:
            errors.append("No model selected or candidates available")
    
    return errors
