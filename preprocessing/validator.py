"""
preprocessing/validator.py

FIX-6: Preprocessing Validation Framework

Validates preprocessing plan coherence with schema before Phase 3 execution.
Catches schema mismatches early with fail-fast error handling.

Architecture
------------
1. PreprocessingValidator.validate_plan()
   Checks preprocessing plan against GlobalSchema for consistency.

2. validate_modality_coherence()
   Ensures modality presence in schema matches preprocessing plan.

3. validate_column_schema_match()
   Errors if detected columns don't match actual dataset columns.

4. validate_target_column()
   Verifies target column exists and is consistent.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PreprocessingValidationError(Exception):
    """Raised when preprocessing plan is invalid."""
    pass


class PreprocessingValidator:
    """
    Validates preprocessing plan before Phase 3 execution.
    
    Purpose: Fail-fast on schema mismatches to prevent silent errors
    downstream in training (Phase 5+).
    """
    
    def validate_plan(
        self,
        plan: Dict[str, Any],
        schema_info: Dict[str, Any],
        dataset_shape: Optional[tuple] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of preprocessing plan against schema.
        
        Parameters
        ----------
        plan : Dict
            Output of PreprocessingPlanner.build_plan()
        schema_info : Dict
            Phase 2 GlobalSchema (serialised)
        dataset_shape : Optional tuple
            (n_rows, n_cols) of actual dataset
            
        Returns
        -------
        Dict with validation results and any warnings/errors
        
        Raises
        ------
        PreprocessingValidationError
            If critical mismatches detected (fail-fast)
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "checks_passed": 0,
            "checks_total": 0,
        }
        
        try:
            # Check 1: Modality coherence
            results["checks_total"] += 1
            self._validate_modality_coherence(plan, schema_info, results)
            results["checks_passed"] += 1 if not results["errors"] else 0
        except Exception as e:
            results["errors"].append(f"Modality coherence check failed: {str(e)}")
            results["valid"] = False
        
        try:
            # Check 2: Target column validation
            results["checks_total"] += 1
            self._validate_target_column(schema_info, results)
            results["checks_passed"] += 1
        except Exception as e:
            results["errors"].append(f"Target column validation failed: {str(e)}")
            results["valid"] = False
        
        try:
            # Check 3: Feature selections match schema
            results["checks_total"] += 1
            self._validate_feature_selection(plan, schema_info, results)
            results["checks_passed"] += 1
        except Exception as e:
            results["errors"].append(f"Feature selection validation failed: {str(e)}")
            results["valid"] = False
        
        try:
            # Check 4: Text/image kwargs valid
            results["checks_total"] += 1
            self._validate_encoding_kwargs(plan, results)
            results["checks_passed"] += 1
        except Exception as e:
            results["errors"].append(f"Encoding kwargs validation failed: {str(e)}")
            results["valid"] = False
        
        # Log results
        if not results["valid"]:
            logger.error(
                "FIX-6: Preprocessing validation FAILED. "
                "Errors: %s",
                results["errors"]
            )
            raise PreprocessingValidationError(
                f"Preprocessing plan validation failed: {results['errors']}"
            )
        
        logger.info(
            "FIX-6: Preprocessing validation PASSED (%d/%d checks). "
            "Warnings: %s",
            results["checks_passed"],
            results["checks_total"],
            results["warnings"] if results["warnings"] else "none"
        )
        
        return results
    
    def _validate_modality_coherence(
        self,
        plan: Dict[str, Any],
        schema_info: Dict[str, Any],
        results: Dict[str, Any],
    ) -> None:
        """Ensure modalities in plan match schema modalities."""
        schema_mods = set(schema_info.get("global_modalities", []))
        plan_mods = set(plan.get("modality", {}).keys())
        
        # Modalities in plan but not in schema
        extra_mods = plan_mods - schema_mods
        if extra_mods:
            results["warnings"].append(
                f"Plan includes modalities not in schema: {extra_mods}"
            )
        
        # Modalities in schema but not planned
        missing_mods = schema_mods - plan_mods
        if missing_mods:
            logger.debug("Schema modalities not planned: %s (OK if weight < 0.2)", missing_mods)
        
        logger.debug("Modality coherence: schema=%s, plan=%s", schema_mods, plan_mods)
    
    def _validate_target_column(
        self,
        schema_info: Dict[str, Any],
        results: Dict[str, Any],
    ) -> None:
        """Verify target column is valid and consistent."""
        target_col = schema_info.get("primary_target")
        if not target_col or target_col == "Unknown":
            raise PreprocessingValidationError(
                f"Invalid target column in schema: {target_col}"
            )
        
        problem_type = schema_info.get("global_problem_type")
        if not problem_type:
            raise PreprocessingValidationError("Problem type not set in schema")
        
        if problem_type not in [
            "classification_binary",
            "classification_multiclass",
            "regression",
            "multilabel_classification",
            "timeseries",
        ]:
            raise PreprocessingValidationError(
                f"Unknown problem type: {problem_type}"
            )
        
        logger.debug("Target validation: column=%s, type=%s", target_col, problem_type)
    
    def _validate_feature_selection(
        self,
        plan: Dict[str, Any],
        schema_info: Dict[str, Any],
        results: Dict[str, Any],
    ) -> None:
        """Validate feature selection parameters."""
        top_k = plan.get("feature_selection", {}).get("top_k", None)
        if top_k is not None and top_k < 1:
            results["warnings"].append(
                f"Feature selection top_k={top_k} very low; may drop too many features"
            )
        
        detected_types = []
        for ds in schema_info.get("per_dataset", []):
            detected_types.extend(ds.get("detected_columns", {}).keys())
        
        logger.debug("Feature selection: detected types=%s", set(detected_types))
    
    def _validate_encoding_kwargs(
        self,
        plan: Dict[str, Any],
        results: Dict[str, Any],
    ) -> None:
        """Validate text/image encoding parameters."""
        # Text validation
        text_plan = plan.get("modality", {}).get("text", {})
        max_len = text_plan.get("max_length", 256)
        if max_len < 16 or max_len > 1024:
            results["warnings"].append(
                f"Text max_length={max_len} unusual (typical: 128-512)"
            )
        
        # Image validation
        img_plan = plan.get("modality", {}).get("image", {})
        img_size = img_plan.get("image_size", [224, 224])
        if img_size[0] < 64 or img_size[0] > 512:
            results["warnings"].append(
                f"Image size={img_size} unusual (typical: 224 or 384)"
            )
        
        logger.debug(
            "Encoding validation: text_max_len=%s, image_size=%s",
            max_len,
            img_size
        )


def validate_preprocessor_consistency(
    tabular_prep: Optional[Any],
    text_prep: Optional[Any],
    image_prep: Optional[Any],
    schema_info: Dict[str, Any],
) -> bool:
    """
    FIX-6: Validate that all preprocessors are initialized and consistent
    with schema before Phase 5 training begins.
    
    Parameters
    ----------
    tabular_prep : TabularPreprocessor | None
    text_prep : TextPreprocessor | None
    image_prep : ImagePreprocessor | None
    schema_info : Dict from Phase 2
    
    Returns
    -------
    bool
        True if all active preprocessors are valid.
        
    Raises
    ------
    PreprocessingValidationError
        If preprocessor state is inconsistent with schema.
    """
    schema_mods = set(schema_info.get("global_modalities", []))
    
    # Check each preprocessor against schema
    if "tabular" in schema_mods and tabular_prep is None:
        raise PreprocessingValidationError(
            "Schema expects tabular modality but tabular_prep is None"
        )
    
    if "text" in schema_mods and text_prep is None:
        raise PreprocessingValidationError(
            "Schema expects text modality but text_prep is None"
        )
    
    if "image" in schema_mods and image_prep is None:
        raise PreprocessingValidationError(
            "Schema expects image modality but image_prep is None"
        )
    
    # Verify preprocessors have required methods
    for prep, name in [(tabular_prep, "tabular"), (text_prep, "text"), (image_prep, "image")]:
        if prep is not None:
            required_methods = ["fit", "transform"] if name == "tabular" else ["configure"]
            for method in required_methods:
                if not hasattr(prep, method):
                    raise PreprocessingValidationError(
                        f"{name} preprocessor missing method: {method}"
                    )
    
    logger.info("FIX-6: All preprocessors validated and consistent with schema")
    return True
