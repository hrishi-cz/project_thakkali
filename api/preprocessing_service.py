"""
Context-Aware Preprocessing Service - Phase 7
Preprocessing that uses intelligence from ExecutionContext instead of recomputing.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

from api.execution_context import ExecutionContext, DatasetProfile
from database.dataset_profile_db import dataset_profile_db

logger = logging.getLogger(__name__)


class ContextAwarePreprocessor:
    """
    Phase 7: Preprocessing planner that uses context intelligence.
    
    Rules:
    - Use schema from ExecutionContext (don't re-detect)
    - Use target from ExecutionContext (don't re-detect)
    - Use global schema/target when available
    - Store preprocessing plan in DatasetProfile
    - Make decisions based on task type, modality, and overrides
    """
    
    def __init__(self):
        self.tabular_strategies = ['standard_scale', 'minmax_scale', 'robust_scale']
        self.text_strategies = ['tfidf', 'bert', 'word2vec']
        self.image_strategies = ['resnet', 'efficientnet', 'vit']
    
    def plan_preprocessing(
        self,
        dataset_id: str,
        ctx: ExecutionContext,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Create preprocessing plan for one dataset using context intelligence.
        
        Returns preprocessing plan dict.
        """
        logger.info("Planning preprocessing for dataset: %s", dataset_id)
        
        # Get profile with schema and target
        profile = ctx.get_dataset_profile(dataset_id)
        if not profile:
            logger.error("No profile found for dataset %s", dataset_id)
            return {"error": "No profile found"}
        
        # Check overrides
        if profile.user_overrides.get('preprocessing_plan'):
            logger.info("Using user-overridden preprocessing plan for %s", dataset_id)
            return profile.user_overrides['preprocessing_plan']
        
        # Use cached plan if exists and not stale
        if profile.preprocessing_plan and not self._is_plan_stale(profile, ctx):
            logger.info("Using cached preprocessing plan for %s", dataset_id)
            return profile.preprocessing_plan
        
        # Build new plan based on schema and target
        plan = self._build_preprocessing_plan(profile, ctx, data)
        
        # Store in profile
        profile.preprocessing_plan = plan
        dataset_profile_db.update_preprocessing_plan(dataset_id, plan)
        
        ctx.log_decision(
            'preprocessing_planning',
            f"Created preprocessing plan for dataset {dataset_id}",
            f"Modality: {profile.modality_breakdown}, Target: {profile.chosen_target}"
        )
        
        logger.info("Preprocessing plan created for %s", dataset_id)
        return plan
    
    def _build_preprocessing_plan(
        self,
        profile: DatasetProfile,
        ctx: ExecutionContext,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Build preprocessing plan based on context intelligence."""
        
        # Determine task type from target
        task_type = self._infer_task_type(profile, data)
        
        # Get modality breakdown
        modality = profile.modality_breakdown or {'tabular': 1.0}
        
        plan = {
            'dataset_id': profile.dataset_id,
            'task_type': task_type,
            'target_column': profile.chosen_target or ctx.global_target,
            'modality_breakdown': modality,
            'steps': [],
            'influenced_by': {
                'local_schema': bool(profile.schema_result),
                'local_target': bool(profile.chosen_target),
                'global_schema': bool(ctx.global_schema),
                'global_target': bool(ctx.global_target),
                'user_override': bool(profile.user_overrides)
            }
        }
        
        # Add modality-specific steps
        if modality.get('tabular', 0) > 0:
            plan['steps'].extend(self._plan_tabular_preprocessing(profile, task_type, data))
        
        if modality.get('text', 0) > 0:
            plan['steps'].extend(self._plan_text_preprocessing(profile, task_type))
        
        if modality.get('image', 0) > 0:
            plan['steps'].extend(self._plan_image_preprocessing(profile, task_type))
        
        # Add validation step
        plan['steps'].append({
            'name': 'validate_preprocessed_data',
            'modality': 'all',
            'params': {
                'check_nulls': True,
                'check_shapes': True,
                'check_dtypes': True
            }
        })
        
        return plan
    
    def _plan_tabular_preprocessing(
        self,
        profile: DatasetProfile,
        task_type: str,
        data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Plan tabular preprocessing steps."""
        steps = []
        
        schema = profile.schema_result or {}
        columns = schema.get('columns', {})
        
        # Handle missing values
        null_columns = [col for col, info in columns.items() 
                       if info.get('null_count', 0) > 0]
        if null_columns:
            steps.append({
                'name': 'handle_missing_values',
                'modality': 'tabular',
                'params': {
                    'strategy': 'median' if task_type == 'regression' else 'mode',
                    'columns': null_columns
                }
            })
        
        # Encode categorical features
        categorical_cols = [col for col, info in columns.items()
                           if info.get('detected_type') == 'categorical']
        if categorical_cols:
            steps.append({
                'name': 'encode_categorical',
                'modality': 'tabular',
                'params': {
                    'strategy': 'onehot' if len(categorical_cols) < 10 else 'target_encode',
                    'columns': categorical_cols
                }
            })
        
        # Scale numeric features
        numeric_cols = [col for col, info in columns.items()
                       if info.get('detected_type') in ['numeric', 'float', 'integer']]
        if numeric_cols:
            steps.append({
                'name': 'scale_numeric',
                'modality': 'tabular',
                'params': {
                    'strategy': 'standard',
                    'columns': numeric_cols
                }
            })
        
        # Feature engineering
        if task_type == 'classification' and len(data) > 1000:
            steps.append({
                'name': 'feature_engineering',
                'modality': 'tabular',
                'params': {
                    'create_interactions': True,
                    'polynomial_features': False,
                    'max_features': 50
                }
            })
        
        return steps
    
    def _plan_text_preprocessing(
        self,
        profile: DatasetProfile,
        task_type: str
    ) -> List[Dict[str, Any]]:
        """Plan text preprocessing steps."""
        return [
            {
                'name': 'clean_text',
                'modality': 'text',
                'params': {
                    'lowercase': True,
                    'remove_punctuation': False,
                    'remove_stopwords': task_type == 'classification'
                }
            },
            {
                'name': 'tokenize',
                'modality': 'text',
                'params': {
                    'tokenizer': 'bert' if task_type == 'classification' else 'simple',
                    'max_length': 512
                }
            },
            {
                'name': 'embed_text',
                'modality': 'text',
                'params': {
                    'embedding': 'bert-base-uncased',
                    'pooling': 'cls'
                }
            }
        ]
    
    def _plan_image_preprocessing(
        self,
        profile: DatasetProfile,
        task_type: str
    ) -> List[Dict[str, Any]]:
        """Plan image preprocessing steps."""
        return [
            {
                'name': 'resize_images',
                'modality': 'image',
                'params': {
                    'target_size': (224, 224),
                    'interpolation': 'bilinear'
                }
            },
            {
                'name': 'normalize_images',
                'modality': 'image',
                'params': {
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]
                }
            },
            {
                'name': 'augment_images',
                'modality': 'image',
                'params': {
                    'enabled': task_type == 'classification',
                    'rotation': 15,
                    'flip_horizontal': True,
                    'brightness': 0.2
                }
            }
        ]
    
    def _infer_task_type(self, profile: DatasetProfile, data: pd.DataFrame) -> str:
        """Infer task type from target column."""
        target_col = profile.chosen_target
        if not target_col or target_col not in data.columns:
            return 'classification'  # Default
        
        target_series = data[target_col].dropna()
        n_unique = target_series.nunique()
        
        if n_unique <= 20:
            return 'classification'
        elif pd.api.types.is_numeric_dtype(target_series):
            return 'regression'
        else:
            return 'classification'
    
    def _is_plan_stale(self, profile: DatasetProfile, ctx: ExecutionContext) -> bool:
        """Check if preprocessing plan is stale and needs regeneration."""
        if not profile.preprocessing_plan:
            return True
        
        # If schema or target changed, plan is stale
        if profile.user_overrides.get('schema') or profile.user_overrides.get('target'):
            return True
        
        # If global schema/target changed, might be stale
        if ctx.global_schema or ctx.global_target:
            influenced = profile.preprocessing_plan.get('influenced_by', {})
            if influenced.get('global_schema') != bool(ctx.global_schema):
                return True
            if influenced.get('global_target') != bool(ctx.global_target):
                return True
        
        return False
    
    def plan_for_session(
        self,
        ctx: ExecutionContext,
        data_map: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Plan preprocessing for all active datasets in session.
        
        Returns:
            {dataset_id: preprocessing_plan}
        """
        plans = {}
        
        for dataset_id in ctx.active_dataset_ids:
            if dataset_id not in data_map:
                logger.warning("Dataset %s not in data_map, skipping preprocessing planning", dataset_id)
                continue
            
            data = data_map[dataset_id]
            plan = self.plan_preprocessing(dataset_id, ctx, data)
            plans[dataset_id] = plan
        
        logger.info("Preprocessing plans created for %d datasets", len(plans))
        return plans
    
    def override_preprocessing_plan(
        self,
        dataset_id: str,
        ctx: ExecutionContext,
        plan_override: Dict[str, Any],
        reason: str
    ) -> bool:
        """Override preprocessing plan with user-provided plan."""
        profile = ctx.get_dataset_profile(dataset_id)
        if not profile:
            return False
        
        old_plan = profile.preprocessing_plan
        profile.preprocessing_plan = plan_override
        profile.user_overrides['preprocessing_plan'] = plan_override
        profile.user_overrides['preprocessing_override_reason'] = reason
        
        # Log override
        ctx.override_history.append({
            'dataset_id': dataset_id,
            'field': 'preprocessing_plan',
            'old_value': old_plan,
            'new_value': plan_override,
            'reason': reason
        })
        
        # Persist
        dataset_profile_db.update_preprocessing_plan(dataset_id, plan_override)
        
        ctx.log_decision(
            'preprocessing_override',
            f"User overrode preprocessing plan for dataset {dataset_id}",
            reason
        )
        
        logger.info("Preprocessing plan overridden for dataset %s", dataset_id)
        return True


# Global singleton
context_aware_preprocessor = ContextAwarePreprocessor()
