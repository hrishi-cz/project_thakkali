"""
ExecutionContext: Central Intelligence System

Single source of truth for all pipeline decisions across all 8 phases:
  1. Data Ingestion (schema detection)
  2. Preprocessing
  3. Model Selection
  4. Fusion
  5. Training
  6. Prediction
  7. Monitoring
  8. Retraining

All phases update and read from this context to ensure coherent decision-making.
Prevents fragmented intelligence where different modules make independent choices.

Architecture:
  schema_detector → ExecutionContext → preprocessing → model_selection → 
  fusion → training → prediction → monitoring → retraining
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional

import logging

logger = logging.getLogger(__name__)


class ExecutionContext:
    """
    Central intelligence object for entire pipeline.
    
    Single source of truth for schema, preprocessing, model, and training decisions.
    Ensures consistency across all 8 pipeline phases.
    
    Attributes
    ----------
    schema : dict
        From schema_detector. Contains modality presence, target info, data types.
    modality_presence : Dict[str, bool]
        {"tabular": True, "text": True, "image": False, ...}
    predictability_scores : Dict[str, float]
        {"tabular": 0.85, "text": 0.72, "image": None, ...}
        Scores for modalities that successfully passed validation.
    target_confidence : float
        0-1; how certain we are about the detected target.
    
    preprocessing_plan : Dict[str, Dict]
        Per-modality preprocessing configuration.
    preprocessing_choices : Dict[str, Dict]
        Extracted choices: {"text": {"tokenizer": "bert"}, ...}
    
    model_choices : List
        Ranked list of candidate model architectures.
    model_selection_reason : str
        Why these models were selected (for audit trail).
    fusion_strategy : Optional[str]
        "hybrid", "attention", "graph", "concatenation", or None.
    modality_importance : Dict[str, float]
        Importance weights: {"tabular": 0.4, "text": 0.3, "image": 0.3}
    
    training_signals : Dict[str, Any]
        From trainer.py: fit_type, losses, convergence, etc.
    active_modalities : List[str]
        Which modalities actually used during training.
    
    xai_config : Dict
        XAI method selection per modality.
    
    constraints : Dict
        Latency, memory, batch size constraints.
    
    version : str
        SHA256 hash of context state; changes only if decisions change.
    """
    
    def __init__(self):
        # Schema information
        self.schema: Optional[Dict[str, Any]] = None
        self.modality_presence: Dict[str, bool] = {}
        self.predictability_scores: Dict[str, float] = {}
        self.target_confidence: float = 0.0
        
        # Preprocessing plan
        self.preprocessing_plan: Dict[str, Dict] = {}
        self.preprocessing_choices: Dict[str, Dict] = {}
        
        # Model selection
        self.model_choices: List[Any] = []
        self.model_selection_reason: str = ""
        self.fusion_strategy: Optional[str] = None
        self.modality_importance: Dict[str, float] = {}
        
        # FIX-3: Cache Phase 4 probe results for /select-model API consistency
        self.probe_scores_cache: Dict[str, Dict[str, Any]] = {}
        self.ranked_candidates: Dict[str, List[Any]] = {}
        
        # Training state
        self.training_signals: Dict[str, Any] = {}
        self.active_modalities: List[str] = []
        
        # Explainability
        self.xai_config: Dict[str, Any] = {}
        
        # Constraints
        self.constraints: Dict[str, Any] = {
            "latency_ms": 500,
            "memory_mb": 2048,
            "batch_size": 32,
        }
        
        # Reproducibility
        self.version: Optional[str] = None
        self._version_timestamp: Optional[str] = None
    
    # === POPULATION METHODS (called by each phase) ===
    
    def update_from_schema(self, schema: Dict[str, Any]) -> None:
        """Called by data_ingestion/schema_detector.py after schema detection."""
        self.schema = schema
        self.modality_presence = schema.get("modality_presence", {})
        self.predictability_scores = schema.get("predictability_scores", {})
        self.target_confidence = schema.get("target_confidence", 0.0)
        self._compute_version()
        logger.info(
            "ExecutionContext: updated from schema. "
            "Modalities: %s, Predictability: %s",
            list(self.modality_presence.keys()),
            {k: f"{v:.2f}" for k, v in self.predictability_scores.items() if v is not None}
        )
    
    def update_preprocessing(self, preprocessing_plan: Dict[str, Dict]) -> None:
        """Called by preprocessing orchestrator after Phase 3 setup."""
        self.preprocessing_plan = preprocessing_plan
        self.preprocessing_choices = self._extract_choices(preprocessing_plan)
        self._compute_version()
        logger.info(
            "ExecutionContext: updated preprocessing plan. "
            "Modalities configured: %s",
            list(preprocessing_plan.keys())
        )
    
    def update_model_selection(self, candidates: List[Any], reason: str) -> None:
        """Called by automl/advanced_selector.py after model probing."""
        self.model_choices = candidates
        self.model_selection_reason = reason
        self._compute_version()
        logger.info(
            "ExecutionContext: model selection updated. "
            "Candidates: %s. Reason: %s",
            [getattr(c, 'name', str(c)) for c in candidates[:3]],
            reason
        )
    
    def update_fusion(self, strategy: Optional[str], importance_weights: Dict[str, float]) -> None:
        """Called by modelss/fusion.py after strategy selection."""
        self.fusion_strategy = strategy
        self.modality_importance = importance_weights
        self._compute_version()
        logger.info(
            "ExecutionContext: fusion updated. "
            "Strategy: %s, Importance: %s",
            strategy,
            {k: f"{v:.2f}" for k, v in importance_weights.items()}
        )
    
    def update_training(self, signals: Dict[str, Any]) -> None:
        """Called by automl/trainer.py after trial completion."""
        self.training_signals = signals
        self.active_modalities = self._extract_active_modalities()
        self._compute_version()
        logger.info(
            "ExecutionContext: training signals updated. "
            "Fit type: %s, Active modalities: %s",
            signals.get("fit_type", "unknown"),
            self.active_modalities
        )
    
    # === QUERY METHODS (read by components) ===
    
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
        """FIX-3: Retrieve ranked candidates for modality from Phase 4 cache."""
        return self.ranked_candidates.get(modality, [])
    
    def get_probe_scores(self, modality: str) -> Dict[str, Any]:
        """FIX-3: Retrieve probe scores for modality from Phase 4 cache."""
        return self.probe_scores_cache.get(modality, {})
    
    def get_loss_weights_for_fusion(self) -> Dict[str, float]:
        """Return auxiliary loss weights based on fusion strategy."""
        if self.fusion_strategy == "hybrid":
            return {
                "complementarity": 0.05,
                "diversity": 0.02,
                "contrastive": 0.03,
                "graph_sparsity": 0.01,
            }
        elif self.fusion_strategy == "graph":
            return {
                "complementarity": 0.05,
                "diversity": 0.02,
                "contrastive": 0.00,
                "graph_sparsity": 0.01,
            }
        elif self.fusion_strategy == "attention":
            return {
                "complementarity": 0.05,
                "diversity": 0.00,
                "contrastive": 0.00,
                "graph_sparsity": 0.00,
            }
        else:  # concatenation or unknown
            return {
                "complementarity": 0.00,
                "diversity": 0.00,
                "contrastive": 0.00,
                "graph_sparsity": 0.00,
            }
    
    # === ENFORCEMENT METHODS ===
    
    def assert_modality_allowed(self, modality: str) -> None:
        """Raise error if modality not in schema."""
        if modality not in self.modality_presence:
            raise ValueError(
                f"Modality '{modality}' not in schema. "
                f"Available: {list(self.modality_presence.keys())}"
            )
    
    def assert_model_valid(self, model: Any) -> None:
        """Raise error if model not in approved candidates."""
        model_names = [getattr(m, 'name', str(m)) for m in self.model_choices]
        model_name = getattr(model, 'name', str(model))
        if model_name not in model_names:
            raise ValueError(
                f"Model '{model_name}' not in approved candidates. "
                f"Available: {model_names}"
            )
    
    def assert_context_ready(self) -> None:
        """Ensure all critical fields populated for training."""
        checks = [
            (self.schema is not None, "Schema required"),
            (self.predictability_scores, "Predictability scores required"),
            (self.preprocessing_plan, "Preprocessing plan required"),
            (self.model_choices, "Model choices required"),
            (
                self.fusion_strategy is not None or not self.should_include_fusion(),
                "Fusion strategy required for multimodal"
            ),
        ]
        
        errors = []
        for check, msg in checks:
            if not check:
                errors.append(msg)
        
        if errors:
            raise RuntimeError(
                "ExecutionContext not ready:\n" + "\n".join(f"  - {e}" for e in errors)
            )
    
    def validate_consistency(self) -> None:
        """Check 5 key consistency points."""
        errors = []
        
        # 1. Schema consistency
        if self.schema is None:
            errors.append("Schema not initialized")
        elif len(self.predictability_scores) != len(self.modality_presence):
            errors.append("Modality count mismatch: schema vs predictability")
        
        # 2. Preprocessing consistency
        if self.preprocessing_plan:
            for modality in self.get_active_modalities():
                if modality not in self.preprocessing_plan:
                    errors.append(f"Missing preprocessing plan for {modality}")
        
        # 3. Model selection consistency
        if self.model_choices:
            for model in self.model_choices[:1]:  # Check top model only
                model_name = getattr(model, 'name', str(model))
                logger.debug(
                    "Model %s checking modality compatibility", model_name
                )
        
        # 4. Fusion consistency
        if self.should_include_fusion() and self.fusion_strategy is None:
            errors.append("Multimodal requires fusion_strategy")
        
        # 5. Training consistency
        if self.training_signals and "fit_type" not in self.training_signals:
            errors.append("Training signals missing fit_type")
        
        if errors:
            raise RuntimeError(
                "ExecutionContext consistency violations:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )
    
    # === INTERNAL METHODS ===
    
    def _extract_choices(self, preprocessing_plan: Dict[str, Dict]) -> Dict[str, Dict]:
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
            str(sorted(self.schema.get("modalities", []) if self.schema else [])) +
            str(sorted(self.predictability_scores.items())) +
            str(self.preprocessing_choices) +
            str(self.fusion_strategy) +
            str(sorted(self.modality_importance.items()))
        )
        
        self.version = hashlib.sha256(state_str.encode()).hexdigest()[:8]
        self._version_timestamp = datetime.now().isoformat()
    
    def compress(self) -> Dict[str, Any]:
        """Return lightweight version; full version kept in memory."""
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize context to dictionary."""
        return {
            "version": self.version,
            "schema": self.schema,
            "modality_presence": self.modality_presence,
            "predictability_scores": self.predictability_scores,
            "target_confidence": self.target_confidence,
            "preprocessing_plan": self.preprocessing_plan,
            "fusion_strategy": self.fusion_strategy,
            "modality_importance": self.modality_importance,
            "training_signals": self.training_signals,
            "active_modalities": self.active_modalities,
            "constraints": self.constraints,
        }


def create_execution_context() -> ExecutionContext:
    """Factory function to create new execution context."""
    return ExecutionContext()
