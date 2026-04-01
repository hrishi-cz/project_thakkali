"""
Enhanced Schema Detection Service - Phase 3
Context-aware schema detection that stores results in DatasetProfile.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

from api.execution_context import ExecutionContext, DatasetProfile
from database.dataset_profile_db import dataset_profile_db
from data_ingestion.schema_detector import COGMASchemaDetector

logger = logging.getLogger(__name__)


class ContextAwareSchemaDetector:
    """
    Phase 3: Per-dataset schema detection with context awareness.
    
    Wraps existing COGMASchemaDetector and stores results in ExecutionContext.
    """
    
    def __init__(self):
        self.cogma_detector = COGMASchemaDetector()
    
    def detect_dataset_schema(
        self,
        dataset_id: str,
        data: pd.DataFrame,
        ctx: ExecutionContext
    ) -> Tuple[Dict[str, Any], float]:
        """
        Detect schema for one dataset and store in context.
        
        Returns:
            (schema_result, confidence)
        """
        logger.info("Detecting schema for dataset: %s", dataset_id)
        
        # Get or create profile
        profile = ctx.get_dataset_profile(dataset_id)
        if not profile:
            profile = DatasetProfile(dataset_id=dataset_id)
            ctx.add_dataset_profile(profile)
        
        # Check if already detected and locked
        if profile.schema_detected and profile.user_overrides.get('schema_locked'):
            logger.info("Schema locked for dataset %s, using existing result", dataset_id)
            return profile.schema_result, profile.schema_confidence
        
        # Use COGMA detector
        try:
            schema_result = self.cogma_detector.detect_schema(data)
            
            # Calculate confidence based on schema quality
            confidence = self._calculate_schema_confidence(schema_result, data)
            
            # Build evidence string
            evidence = self._build_evidence(schema_result, data)
            
            # Update profile
            profile.schema_detected = True
            profile.schema_result = schema_result
            profile.schema_confidence = confidence
            profile.schema_evidence = evidence
            
            # Persist to database
            dataset_profile_db.update_schema(
                dataset_id,
                schema_result,
                confidence,
                evidence
            )
            
            # Log decision
            ctx.log_decision(
                'schema_detection',
                f"Dataset {dataset_id}: detected {len(schema_result.get('columns', []))} columns",
                evidence
            )
            
            logger.info("Schema detected for %s with confidence %.2f", dataset_id, confidence)
            return schema_result, confidence
            
        except Exception as e:
            logger.error("Schema detection failed for %s: %s", dataset_id, e, exc_info=True)
            # Store partial result
            profile.schema_detected = False
            profile.schema_result = {"error": str(e)}
            profile.schema_confidence = 0.0
            return {"error": str(e)}, 0.0
    
    def _calculate_schema_confidence(self, schema: Dict[str, Any], data: pd.DataFrame) -> float:
        """Calculate confidence score for schema detection."""
        if not schema or 'columns' not in schema:
            return 0.0
        
        scores = []
        columns = schema.get('columns', {})
        
        for col_name, col_info in columns.items():
            if col_name not in data.columns:
                continue
            
            # Type consistency score
            if col_info.get('detected_type'):
                type_score = 1.0 if col_info.get('type_confidence', 0) > 0.8 else 0.5
                scores.append(type_score)
            
            # Null handling score
            if 'null_count' in col_info:
                null_ratio = col_info['null_count'] / len(data)
                null_score = 1.0 - (null_ratio * 0.5)  # Penalty for high null %
                scores.append(null_score)
        
        return np.mean(scores) if scores else 0.5
    
    def _build_evidence(self, schema: Dict[str, Any], data: pd.DataFrame) -> str:
        """Build human-readable evidence string."""
        if not schema or 'columns' not in schema:
            return "Schema detection failed"
        
        columns = schema.get('columns', {})
        evidence_parts = [
            f"Detected {len(columns)} columns:",
        ]
        
        # Count by type
        type_counts = {}
        for col_info in columns.values():
            dtype = col_info.get('detected_type', 'unknown')
            type_counts[dtype] = type_counts.get(dtype, 0) + 1
        
        for dtype, count in sorted(type_counts.items()):
            evidence_parts.append(f"  - {count} {dtype} column(s)")
        
        # Identify potential issues
        issues = []
        for col_name, col_info in columns.items():
            if col_info.get('null_count', 0) / len(data) > 0.5:
                issues.append(f"{col_name} has >50% nulls")
            if col_info.get('is_identifier'):
                issues.append(f"{col_name} appears to be an ID column")
        
        if issues:
            evidence_parts.append("Potential issues:")
            for issue in issues[:5]:  # Limit to 5
                evidence_parts.append(f"  - {issue}")
        
        return "\n".join(evidence_parts)
    
    def detect_for_session(
        self,
        ctx: ExecutionContext,
        data_map: Dict[str, pd.DataFrame]
    ) -> Dict[str, Tuple[Dict[str, Any], float]]:
        """
        Detect schemas for all active datasets in a session.
        
        Args:
            ctx: ExecutionContext
            data_map: {dataset_id: DataFrame}
        
        Returns:
            {dataset_id: (schema_result, confidence)}
        """
        results = {}
        
        for dataset_id in ctx.active_dataset_ids:
            if dataset_id not in data_map:
                logger.warning("Dataset %s not in data_map, skipping schema detection", dataset_id)
                continue
            
            data = data_map[dataset_id]
            schema, conf = self.detect_dataset_schema(dataset_id, data, ctx)
            results[dataset_id] = (schema, conf)
        
        logger.info("Schema detection complete for %d datasets", len(results))
        return results
    
    def override_schema(
        self,
        dataset_id: str,
        ctx: ExecutionContext,
        schema_override: Dict[str, Any],
        reason: str
    ) -> bool:
        """Allow user to override detected schema."""
        profile = ctx.get_dataset_profile(dataset_id)
        if not profile:
            logger.error("No profile found for dataset %s", dataset_id)
            return False
        
        old_schema = profile.schema_result
        profile.schema_result = schema_override
        profile.user_overrides['schema'] = schema_override
        profile.user_overrides['schema_locked'] = True
        profile.user_overrides['schema_override_reason'] = reason
        
        # Log override
        ctx.override_history.append({
            'dataset_id': dataset_id,
            'field': 'schema',
            'old_value': old_schema,
            'new_value': schema_override,
            'reason': reason
        })
        
        # Persist
        dataset_profile_db.update_schema(
            dataset_id,
            schema_override,
            1.0,  # User override = 100% confidence
            f"User override: {reason}"
        )
        
        ctx.log_decision(
            'schema_override',
            f"User overrode schema for dataset {dataset_id}",
            reason
        )
        
        logger.info("Schema overridden for dataset %s: %s", dataset_id, reason)
        return True


# Global singleton
schema_detector = ContextAwareSchemaDetector()
