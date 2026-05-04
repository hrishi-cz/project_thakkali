"""
Integrator: Unified modality pipeline orchestrator.

PURPOSE (FIX-4 Part 3):
  Single entry point for:
    1. Detect modalities (auto) + encode (embeddings)
    2. Validate predictability (RF scoring)
    3. Return consolidated metadata for inference
  
  Replaces separate SIFT/OCAM/TF-IDF paths with unified
  modality detection → encoding → validation pipeline.

INTEGRATION:
  Called by: pipeline/orchestrator.py or data_ingestion/schema_detector.py
  Calls: {ModalityEncoder, UniversalTargetValidator, }
  Output: ModalityMetadata (structured output with embeddings + scores)

EXPECTED FLOW:
  1. User provides raw data (images, text, numbers)
  2. Integrator auto-detects modalities
  3. Each modality → embeddings (via ModalityEncoder)
  4. Validate each modality (via UniversalTargetValidator)
  5. Return consolidated ModalityMetadata with all info
  6. downstream modules (fusion, inference) use metadata
"""

from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .modality_encoder import ModalityEncoder
from .target_validator import UniversalTargetValidator

logger = logging.getLogger(__name__)


@dataclass
class ModalityMetadata:
    """
    Structured output of Integrator: embeddings + validation scores.
    
    Attributes
    ----------
    modality_name : str
        "text", "image", "tabular", etc.
    embeddings : np.ndarray, shape (N, D)
        Learned embeddings (D = encoder output dim).
    predictability_score : float
        RF 3-fold CV score (0-1).
    complementarity_score : float
        Uniqueness vs other modalities (0-1).
    degeneracy_score : float
        Non-constant feature fraction (0-1).
    noise_robustness_score : float
        Stability under input perturbation (0-1).
    feature_importance_score : float
        Signal concentration (0-1).
    encoder_name : str
        "SIFT", "OCAM", "sentence-transformers/all-mpnet-base-v2", etc.
    raw_shape : Tuple[int, ...]
        Original data shape before encoding.
    is_valid : bool
        Overall validity (all scores ≥ threshold).
    detection_method : str
        How modality was detected ("shape", "dtype", "filetype").
    """
    modality_name: str
    embeddings: np.ndarray
    predictability_score: float = 0.0
    complementarity_score: float = 0.0
    degeneracy_score: float = 0.0
    noise_robustness_score: float = 0.0
    feature_importance_score: float = 0.0
    encoder_name: str = ""
    raw_shape: Tuple[int, ...] = field(default_factory=tuple)
    is_valid: bool = False
    detection_method: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def final_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Recompute final predictability score (same as UniversalTargetValidator).
        
        Parameters
        ----------
        weights : Optional[Dict[str, float]]
            Override default weights.
        
        Returns
        -------
        float
            Weighted score (0-1).
        """
        w = weights or {
            "predictability": 0.40,
            "complementarity": 0.20,
            "degeneracy": 0.15,
            "noise_robustness": 0.15,
            "feature_importance": 0.10,
        }
        
        score = (
            w["predictability"] * self.predictability_score +
            w["complementarity"] * self.complementarity_score +
            w["degeneracy"] * self.degeneracy_score +
            w["noise_robustness"] * self.noise_robustness_score +
            w["feature_importance"] * self.feature_importance_score
        )
        
        return float(np.clip(score, 0.0, 1.0))
    
    def __repr__(self) -> str:
        """Pretty print ModalityMetadata."""
        return (
            f"ModalityMetadata("
            f"modality={self.modality_name}, "
            f"embeddings_shape={self.embeddings.shape}, "
            f"final_score={self.final_score():.3f}, "
            f"encoder={self.encoder_name})"
        )


class Integrator:
    """
    Unified modality pipeline: detect + encode + validate.
    
    This is the orchestrator that:
      1. Auto-detects modalities from raw data
      2. Encodes each modality into embeddings
      3. Validates each modality's predictability
      4. Returns consolidated metadata
    
    Attributes
    ----------
    encoder : ModalityEncoder
        Encodes raw data → embeddings.
    validator : UniversalTargetValidator
        Validates embeddings → scores.
    min_predictability : float
        Min threshold for is_valid (default 0.3).
    """
    
    def __init__(
        self,
        min_predictability: float = 0.3,
        custom_encoders: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Integrator.
        
        Parameters
        ----------
        min_predictability : float
            Modality is_valid if final_score ≥ this threshold.
        custom_encoders : Optional[Dict[str, Any]]
            Custom encoder overrides (modality → encoder object).
        """
        try:
            self.encoder = ModalityEncoder(custom_encoders=custom_encoders)
        except TypeError:
            # Backward-compat fallback for ModalityEncoder variants that do
            # not expose the custom_encoders keyword.
            text_encoder = None
            image_encoder = None
            if isinstance(custom_encoders, dict):
                text_encoder = custom_encoders.get("text")
                image_encoder = custom_encoders.get("image")
            self.encoder = ModalityEncoder(
                text_encoder=text_encoder,
                image_encoder=image_encoder,
            )
            logger.debug(
                "Integrator: ModalityEncoder initialized via legacy signature fallback"
            )
        self.validator = UniversalTargetValidator()
        self.min_predictability = min_predictability
        logger.info(
            "Integrator initialized: min_predictability=%.3f",
            min_predictability,
        )
    
    def detect_modality(
        self,
        data: Any,
        field_name: str = "unknown_field",
    ) -> Optional[str]:
        """
        Auto-detect modality from raw data.
        
        Parameters
        ----------
        data : Any
            Raw data (list of strings, PIL images, numpy arrays, etc.).
        field_name : str
            Name of field (optional context).
        
        Returns
        -------
        Optional[str]
            Detected modality ("text", "image", "tabular", "categorical")
            or None if unknown.
        """
        # Delegate to encoder
        return self.encoder.detect_modality(data, field_name=field_name)
    
    def process_single_modality(
        self,
        raw_data: Any,
        modality: Optional[str] = None,
        y: Optional[np.ndarray] = None,
        task_type: str = "regression",
        field_name: str = "unknown_field",
    ) -> ModalityMetadata:
        """
        End-to-end: detect + encode + validate a single modality.
        
        Parameters
        ----------
        raw_data : Any
            Raw data (list of strings, PIL images, numpy array, etc.).
        modality : Optional[str]
            Force modality (if None, auto-detect).
        y : Optional[np.ndarray]
            Target variable for validation (optional).
        task_type : str
            "regression" or "classification"
        
        Returns
        -------
        ModalityMetadata
            Complete metadata: embeddings + scores.
        
        Raises
        ------
        ValueError
            If modality detection/encoding fails.
        """
        # 1. Detect modality
        forced_modality = modality
        if modality is None:
            modality = self.detect_modality(raw_data, field_name=field_name)
            if modality is None:
                raise ValueError(
                    "process_single_modality: could not auto-detect modality"
                )
            logger.info("Auto-detected modality: %s", modality)
        
        # 2. Encode to embeddings
        try:
            try:
                encode_result = self.encoder.encode(
                    raw_data,
                    modality=modality,
                    return_metadata=True,
                    field_name=field_name,
                )
            except TypeError:
                # Backward-compat for older ModalityEncoder variants.
                encode_result = self.encoder.encode(modality, raw_data)

            if isinstance(encode_result, tuple) and len(encode_result) == 3:
                embeddings, encoder_name, raw_shape = encode_result
            else:
                embeddings = np.asarray(encode_result)
                encoder_name = str(modality)
                raw_shape = tuple(getattr(np.asarray(raw_data), "shape", tuple()))
        except Exception as e:
            msg = str(e)
            if modality in {"text", "image"} and "encoder not initialized" in msg.lower():
                logger.warning(
                    "process_single_modality: %s capability not scored because encoder is not initialized",
                    modality,
                )
                n_rows = len(raw_data) if hasattr(raw_data, "__len__") else 0
                return ModalityMetadata(
                    modality_name=str(modality),
                    embeddings=np.zeros((int(n_rows), 0), dtype=np.float32),
                    encoder_name="not_scored",
                    raw_shape=tuple(getattr(np.asarray(raw_data), "shape", tuple())),
                    is_valid=False,
                    detection_method="capability_unavailable",
                    metadata={
                        "field_name": field_name,
                        "forced_modality": forced_modality,
                        "resolved_modality": modality,
                        "status": "not_scored",
                        "reason": msg,
                    },
                )
            logger.error("process_single_modality: encoding failed: %s", e)
            raise ValueError(f"Encoding failed for {modality}: {e}")
        
        logger.info(
            "Encoded %s: embeddings shape = %s, encoder = %s",
            modality, embeddings.shape, encoder_name,
        )
        
        # 3. Validate (if y provided)
        scores = {}
        is_valid = True
        
        if y is not None and len(y) == embeddings.shape[0]:
            try:
                # Use validator's components directly
                pred = self.validator._predict(embeddings, y, task_type=task_type)
                comp = self.validator._check_complementarity(embeddings)
                degen = self.validator._check_degeneracy(embeddings)
                noise = self.validator._check_noise_robustness(
                    embeddings, y, task_type=task_type
                )
                feat = self.validator._check_feature_importance(
                    embeddings, y, task_type=task_type
                )
                
                scores = {
                    "predictability": pred,
                    "complementarity": comp,
                    "degeneracy": degen,
                    "noise_robustness": noise,
                    "feature_importance": feat,
                }
                
                # Final score
                final = (
                    0.40 * pred + 0.20 * comp + 0.15 * degen +
                    0.15 * noise + 0.10 * feat
                )
                
                is_valid = final >= self.min_predictability
                
                logger.info(
                    "Validation %s: final_score=%.3f, is_valid=%s",
                    modality, final, is_valid,
                )
            
            except Exception as e:
                logger.warning(
                    "process_single_modality: validation failed: %s", e
                )
                is_valid = False
        
        # 4. Build metadata
        detection_method = (
            "forced"
            if forced_modality is not None
            else self._resolve_detection_method(raw_data, field_name=field_name)
        )

        metadata = ModalityMetadata(
            modality_name=modality,
            embeddings=embeddings,
            predictability_score=scores.get("predictability", 0.0),
            complementarity_score=scores.get("complementarity", 0.0),
            degeneracy_score=scores.get("degeneracy", 0.0),
            noise_robustness_score=scores.get("noise_robustness", 0.0),
            feature_importance_score=scores.get("feature_importance", 0.0),
            encoder_name=encoder_name,
            raw_shape=raw_shape or tuple(),
            is_valid=is_valid,
            detection_method=detection_method,
            metadata={
                "field_name": field_name,
                "forced_modality": forced_modality,
                "resolved_modality": modality,
            },
        )
        
        return metadata

    @staticmethod
    def _resolve_detection_method(raw_data: Any, field_name: str) -> str:
        """Best-effort tag describing how modality auto-detection was inferred."""
        lowered_field = str(field_name or "").lower()
        if any(
            tok in lowered_field
            for tok in (
                "image",
                "img",
                "photo",
                "picture",
                "pixel",
                "text",
                "report",
                "note",
                "description",
                "content",
                "caption",
                "table",
                "tabular",
                "feature",
            )
        ):
            return "field_hint"

        if isinstance(raw_data, np.ndarray):
            return "shape"

        if isinstance(raw_data, (list, tuple)):
            sample = next((item for item in raw_data if item is not None), None)
            if isinstance(sample, str):
                suffix = Path(sample).suffix.lower()
                if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}:
                    return "filetype"
                return "dtype"
            if sample is not None:
                return "dtype"

        if hasattr(raw_data, "dtypes"):
            return "dtype"

        return "auto"
    
    def process_multimodal(
        self,
        raw_data_dict: Dict[str, Any],
        modalities: Optional[Dict[str, str]] = None,
        y: Optional[np.ndarray] = None,
        task_type: str = "regression",
    ) -> Dict[str, ModalityMetadata]:
        """
        Process all modalities from a multimodal dataset.
        
        Parameters
        ----------
        raw_data_dict : Dict[str, Any]
            Field name → raw data: {"images": [...], "text": [...], "features": [...]}
        modalities : Optional[Dict[str, str]]
            Field name → forced modality (if None, auto-detect).
        y : Optional[np.ndarray]
            Target variable for validation.
        task_type : str
            "regression" or "classification"
        
        Returns
        -------
        Dict[str, ModalityMetadata]
            Field name → ModalityMetadata.
        """
        modalities = modalities or {}
        results = {}
        
        for field_name, raw_data in raw_data_dict.items():
            modality = modalities.get(field_name)
            
            try:
                meta = self.process_single_modality(
                    raw_data=raw_data,
                    modality=modality,
                    y=y,
                    task_type=task_type,
                    field_name=field_name,
                )
                results[field_name] = meta
                logger.info(
                    "Processed field %s → modality %s, score %.3f",
                    field_name, meta.modality_name, meta.final_score(),
                )
            
            except Exception as e:
                logger.error(
                    "process_multimodal failed for field %s: %s", field_name, e
                )
                # Skip this modality
                continue
        
        return results


def process_dataset(
    raw_data_dict: Dict[str, Any],
    y: Optional[np.ndarray] = None,
    task_type: str = "regression",
) -> Dict[str, ModalityMetadata]:
    """
    Convenience function: process entire dataset at once.
    
    Parameters
    ----------
    raw_data_dict : Dict[str, Any]
        Field name → raw data.
    y : Optional[np.ndarray]
        Target variable.
    task_type : str
        "regression" or "classification"
    
    Returns
    -------
    Dict[str, ModalityMetadata]
        Field name → metadata with embeddings + scores.
    """
    integrator = Integrator()
    return integrator.process_multimodal(raw_data_dict, y=y, task_type=task_type)
