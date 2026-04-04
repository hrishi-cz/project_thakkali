"""
Global Schema and Target Aggregation Service - Phase 5
Infers common schema/target across multiple datasets or handles incompatibility.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from collections import Counter

from api.execution_context import ExecutionContext, DatasetProfile

logger = logging.getLogger(__name__)


class GlobalAggregationService:
    """
    Phase 5: Global schema and target inference across multiple datasets.
    
    Rules:
    - If datasets share meaningful structure/semantics, infer global schema/target
    - If incompatible, ask user to select primary dataset
    - Both global schema and global target must be overrideable
    - Individual targets remain available even when global target exists
    """
    
    def __init__(self):
        self.compatibility_threshold = 0.5  # Min score for compatibility
    
    def infer_global_schema(
        self,
        ctx: ExecutionContext,
        data_map: Dict[str, pd.DataFrame]
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Infer global schema from active datasets.
        
        Returns:
            (global_schema, confidence)
        """
        logger.info("Inferring global schema for session %s", ctx.session_id)
        
        profiles = ctx.get_active_profiles()
        if not profiles:
            return None, 0.0
        
        if len(profiles) == 1:
            # Single dataset: global schema = dataset schema
            profile = profiles[0]
            if profile.schema_result:
                ctx.set_global_schema(profile.schema_result, profile.schema_confidence)
                return profile.schema_result, profile.schema_confidence
            return None, 0.0
        
        # Multiple datasets: find common schema
        common_columns = self._find_common_columns(profiles, data_map)
        
        if not common_columns:
            logger.warning("No common columns found across datasets")
            ctx.set_global_schema({"error": "No common columns"}, 0.0)
            ctx.datasets_compatible = False
            return None, 0.0
        
        # Build global schema from common columns
        global_schema = {
            'columns': common_columns,
            'source': 'global_aggregation',
            'num_datasets': len(profiles),
            'common_column_count': len(common_columns)
        }
        
        confidence = len(common_columns) / max(len(p.schema_result.get('columns', {})) 
                                                for p in profiles if p.schema_result)
        confidence = min(1.0, max(0.0, confidence))
        
        evidence = f"Found {len(common_columns)} common columns across {len(profiles)} datasets"
        ctx.set_global_schema(global_schema, confidence, evidence)
        ctx.datasets_compatible = True
        
        logger.info("Global schema inferred with %d columns (confidence: %.2f)", 
                   len(common_columns), confidence)
        
        return global_schema, confidence
    
    def _find_common_columns(
        self,
        profiles: List[DatasetProfile],
        data_map: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Find columns that exist in all datasets with compatible types."""
        if not profiles:
            return {}
        
        # Get column names from each dataset
        all_column_sets = []
        for profile in profiles:
            if profile.schema_result and 'columns' in profile.schema_result:
                all_column_sets.append(set(profile.schema_result['columns'].keys()))
            elif profile.dataset_id in data_map:
                all_column_sets.append(set(data_map[profile.dataset_id].columns))
        
        if not all_column_sets:
            return {}
        
        # Find intersection
        common_names = set.intersection(*all_column_sets)
        
        if not common_names:
            return {}
        
        # Build merged column info
        common_columns = {}
        for col_name in common_names:
            # Aggregate type info from all datasets
            types = []
            for profile in profiles:
                if profile.schema_result and col_name in profile.schema_result.get('columns', {}):
                    col_info = profile.schema_result['columns'][col_name]
                    if 'detected_type' in col_info:
                        types.append(col_info['detected_type'])
            
            # Use most common type
            if types:
                most_common_type = Counter(types).most_common(1)[0][0]
                common_columns[col_name] = {
                    'detected_type': most_common_type,
                    'type_consensus': types.count(most_common_type) / len(types),
                    'present_in_datasets': len(types)
                }
        
        return common_columns
    
    def infer_global_target(
        self,
        ctx: ExecutionContext
    ) -> Tuple[Optional[str], float, List[Dict[str, Any]]]:
        """
        Infer global target from dataset targets.
        
        Returns:
            (global_target, confidence, candidates)
        """
        logger.info("Inferring global target for session %s", ctx.session_id)
        
        profiles = ctx.get_active_profiles()
        if not profiles:
            return None, 0.0, []
        
        # Collect targets from all datasets
        targets = []
        for profile in profiles:
            if profile.chosen_target:
                targets.append(profile.chosen_target)
        
        if not targets:
            logger.warning("No targets detected in any dataset")
            return None, 0.0, []
        
        # Single dataset or unanimous target
        if len(set(targets)) == 1:
            target = targets[0]
            confidence = 1.0
            candidates = [{'name': target, 'score': 1.0, 'reason': 'Unanimous across datasets'}]
            ctx.set_global_target(target, confidence, candidates)
            return target, confidence, candidates
        
        # Multiple different targets: rank by frequency
        target_counts = Counter(targets)
        total = len(targets)
        
        candidates = []
        for target_name, count in target_counts.most_common():
            score = count / total
            reason = f"Present in {count}/{total} datasets"
            candidates.append({
                'name': target_name,
                'score': score,
                'reason': reason,
                'dataset_count': count
            })
        
        # Choose most common target
        if candidates:
            chosen_target = candidates[0]['name']
            confidence = candidates[0]['score']
            ctx.set_global_target(chosen_target, confidence, candidates)
            logger.info("Global target inferred: %s (confidence: %.2f)", chosen_target, confidence)
            return chosen_target, confidence, candidates
        
        return None, 0.0, []
    
    def calculate_compatibility_matrix(
        self,
        ctx: ExecutionContext,
        data_map: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Calculate pairwise compatibility scores between datasets.
        
        Returns compatibility matrix and overall compatibility flag.
        """
        profiles = ctx.get_active_profiles()
        if len(profiles) < 2:
            ctx.datasets_compatible = True
            ctx.compatibility_matrix = {"compatible": True, "reason": "Single dataset"}
            return ctx.compatibility_matrix
        
        matrix = {}
        for i, p1 in enumerate(profiles):
            for j, p2 in enumerate(profiles):
                if i >= j:
                    continue
                
                score = self._calculate_pairwise_compatibility(p1, p2, data_map)
                key = f"{p1.dataset_id}_{p2.dataset_id}"
                matrix[key] = score
        
        # Overall compatibility: average of pairwise scores
        if matrix:
            avg_score = np.mean(list(matrix.values()))
            compatible = avg_score >= self.compatibility_threshold
        else:
            avg_score = 1.0
            compatible = True
        
        result = {
            'compatible': compatible,
            'average_score': float(avg_score),
            'pairwise_scores': matrix,
            'threshold': self.compatibility_threshold
        }
        
        ctx.datasets_compatible = compatible
        ctx.compatibility_matrix = result
        
        # Update individual dataset compatibility scores
        for profile in profiles:
            # Average score of this dataset with all others
            dataset_scores = [
                score for key, score in matrix.items()
                if profile.dataset_id in key
            ]
            if dataset_scores:
                profile.compatibility_score = np.mean(dataset_scores)
                profile.global_compatible = profile.compatibility_score >= self.compatibility_threshold
                from database.dataset_profile_db import dataset_profile_db
                dataset_profile_db.update_compatibility(
                    profile.dataset_id,
                    profile.global_compatible,
                    profile.compatibility_score
                )
        
        logger.info("Compatibility matrix calculated: compatible=%s, avg_score=%.2f", 
                   compatible, avg_score)
        
        return result
    
    def _calculate_pairwise_compatibility(
        self,
        p1: DatasetProfile,
        p2: DatasetProfile,
        data_map: Dict[str, pd.DataFrame]
    ) -> float:
        """Calculate compatibility score between two datasets."""
        score = 0.0
        factors = 0
        
        # 1. Column overlap
        if p1.schema_result and p2.schema_result:
            cols1 = set(p1.schema_result.get('columns', {}).keys())
            cols2 = set(p2.schema_result.get('columns', {}).keys())
            
            if cols1 and cols2:
                overlap = len(cols1 & cols2) / max(len(cols1), len(cols2))
                score += overlap
                factors += 1
        
        # 2. Target agreement
        if p1.chosen_target and p2.chosen_target:
            if p1.chosen_target == p2.chosen_target:
                score += 1.0
            else:
                score += 0.0
            factors += 1
        
        # 3. Modality similarity
        if p1.modality_breakdown and p2.modality_breakdown:
            # Cosine similarity of modality vectors
            m1 = p1.modality_breakdown
            m2 = p2.modality_breakdown
            all_modalities = set(m1.keys()) | set(m2.keys())
            
            v1 = np.array([m1.get(m, 0.0) for m in all_modalities])
            v2 = np.array([m2.get(m, 0.0) for m in all_modalities])
            
            if v1.sum() > 0 and v2.sum() > 0:
                cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                score += cosine
                factors += 1
        
        # 4. Row count similarity (order of magnitude)
        if p1.dataset_id in data_map and p2.dataset_id in data_map:
            rows1 = len(data_map[p1.dataset_id])
            rows2 = len(data_map[p2.dataset_id])
            if rows1 > 0 and rows2 > 0:
                ratio = min(rows1, rows2) / max(rows1, rows2)
                score += ratio
                factors += 1
        
        return score / factors if factors > 0 else 0.0
    
    def override_global_target(
        self,
        ctx: ExecutionContext,
        new_target: str,
        reason: str = "User override"
    ) -> bool:
        """Override global target selection."""
        ctx.override_global_target(new_target, reason)
        logger.info("Global target overridden: %s", new_target)
        return True
    
    def choose_primary_dataset(
        self,
        ctx: ExecutionContext,
        dataset_id: str,
        reason: str = "User selection for incompatible datasets"
    ) -> bool:
        """
        Choose primary dataset when datasets are incompatible.
        
        This allows the pipeline to proceed with one dataset's schema/target.
        """
        if dataset_id not in ctx.active_dataset_ids:
            logger.error("Dataset %s not in active datasets", dataset_id)
            return False
        
        ctx.primary_dataset_id = dataset_id
        
        # Use primary dataset's schema and target as global
        profile = ctx.get_dataset_profile(dataset_id)
        if profile:
            if profile.schema_result:
                ctx.set_global_schema(profile.schema_result, profile.schema_confidence)
            if profile.chosen_target:
                candidates = profile.target_candidates or [
                    {'name': profile.chosen_target, 'score': 1.0, 'reason': 'Primary dataset target'}
                ]
                ctx.set_global_target(profile.chosen_target, 1.0, candidates)
        
        ctx.log_decision('primary_dataset', f"Selected primary dataset: {dataset_id}", reason)
        logger.info("Primary dataset set: %s", dataset_id)
        return True


# Global singleton
global_aggregation = GlobalAggregationService()
