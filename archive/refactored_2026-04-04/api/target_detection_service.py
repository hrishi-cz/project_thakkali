"""
Enhanced Target Detection Service - Phase 4
Context-aware target detection with ranking, override, and lock support.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy.stats import entropy

from api.execution_context import ExecutionContext, DatasetProfile
from database.dataset_profile_db import dataset_profile_db

logger = logging.getLogger(__name__)


class ContextAwareTargetDetector:
    """
    Phase 4: Per-dataset target detection with intelligent ranking.
    
    Ranks potential targets based on:
    - Uniqueness (ID columns = bad targets)
    - Entropy / class balance
    - Mutual information with other features
    - Label keyword cues ("label", "target", "class", "y")
    - Leakage risk
    """
    
    LABEL_KEYWORDS = {'label', 'target', 'class', 'y', 'output', 'response', 
                      'outcome', 'dependent', 'prediction', 'result'}
    
    LEAKAGE_KEYWORDS = {'id', 'index', 'timestamp', 'date', 'time', 'key',
                        'uuid', 'guid', '_id', 'created', 'updated'}
    
    def __init__(self):
        pass
    
    def detect_dataset_target(
        self,
        dataset_id: str,
        data: pd.DataFrame,
        ctx: ExecutionContext,
        schema: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Detect and rank target candidates for one dataset.
        
        Returns:
            (target_candidates, chosen_target)
            where target_candidates = [{name, score, reason}, ...]
        """
        logger.info("Detecting target for dataset: %s", dataset_id)
        
        # Get or create profile
        profile = ctx.get_dataset_profile(dataset_id)
        if not profile:
            profile = DatasetProfile(dataset_id=dataset_id)
            ctx.add_dataset_profile(profile)
        
        # Check if target is locked
        if profile.target_locked:
            logger.info("Target locked for dataset %s: %s", dataset_id, profile.chosen_target)
            return profile.target_candidates, profile.chosen_target
        
        # Rank all columns as potential targets
        candidates = self._rank_target_candidates(data, schema)
        
        # Choose top candidate (if score > threshold)
        chosen_target = None
        if candidates and candidates[0]['score'] > 0.3:
            chosen_target = candidates[0]['name']
        
        # Update profile
        profile.target_detected = True
        profile.target_candidates = candidates
        profile.chosen_target = chosen_target
        
        # Persist
        dataset_profile_db.update_target(
            dataset_id,
            candidates,
            chosen_target,
            locked=False
        )
        
        # Log
        reason_str = candidates[0]['reason'] if candidates else "No strong candidate"
        ctx.log_decision(
            'target_detection',
            f"Dataset {dataset_id}: chosen target = {chosen_target or 'None'}",
            reason_str
        )
        
        logger.info("Target detected for %s: %s (score: %.2f)", 
                   dataset_id, chosen_target, candidates[0]['score'] if candidates else 0.0)
        
        return candidates, chosen_target
    
    def _rank_target_candidates(
        self,
        data: pd.DataFrame,
        schema: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Rank all columns as potential targets.
        
        Scoring factors:
        - Label keyword match: +0.4
        - Categorical with 2-50 classes: +0.3
        - Low uniqueness (not ID): +0.2
        - High entropy: +0.1
        - Leakage keyword: -0.5
        - Too many unique values: -0.3
        """
        candidates = []
        
        for col in data.columns:
            score = 0.0
            reasons = []
            
            col_lower = col.lower()
            series = data[col].dropna()
            
            if len(series) == 0:
                continue
            
            # 1. Leakage detection (disqualify immediately)
            if any(kw in col_lower for kw in self.LEAKAGE_KEYWORDS):
                score -= 0.6
                reasons.append("Likely ID/leakage column")
            
            # 2. Label keyword match
            if any(kw in col_lower for kw in self.LABEL_KEYWORDS):
                score += 0.5
                reasons.append("Column name suggests label")
            
            # 3. Uniqueness check
            unique_ratio = series.nunique() / len(series)
            if unique_ratio > 0.95:
                score -= 0.4
                reasons.append(f"Too unique ({unique_ratio:.1%})")
            elif unique_ratio < 0.01:
                score -= 0.3
                reasons.append(f"Almost constant ({unique_ratio:.3%})")
            else:
                score += 0.2
                reasons.append(f"Good uniqueness ({unique_ratio:.1%})")
            
            # 4. Categorical suitability
            n_unique = series.nunique()
            if 2 <= n_unique <= 50:
                score += 0.3
                reasons.append(f"Good class count ({n_unique})")
            elif n_unique > 100:
                score -= 0.2
                reasons.append(f"Too many classes ({n_unique})")
            
            # 5. Entropy (for categorical)
            if pd.api.types.is_categorical_dtype(series) or n_unique < 100:
                try:
                    value_counts = series.value_counts(normalize=True)
                    col_entropy = entropy(value_counts)
                    if col_entropy > 0.5:
                        score += 0.1
                        reasons.append(f"Good entropy ({col_entropy:.2f})")
                except:
                    pass
            
            # 6. Numeric but looks like class (small int range)
            if pd.api.types.is_numeric_dtype(series):
                if series.min() >= 0 and series.max() <= 10 and n_unique <= 10:
                    score += 0.2
                    reasons.append("Numeric but class-like")
            
            candidates.append({
                'name': col,
                'score': max(0.0, min(1.0, score)),  # Clamp to [0, 1]
                'reason': '; '.join(reasons) if reasons else 'Neutral column',
                'unique_values': int(n_unique),
                'unique_ratio': float(unique_ratio)
            })
        
        # Sort by score descending
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return candidates
    
    def override_target(
        self,
        dataset_id: str,
        ctx: ExecutionContext,
        new_target: str,
        lock: bool = True,
        reason: str = "User override"
    ) -> bool:
        """Override target selection for a dataset."""
        profile = ctx.get_dataset_profile(dataset_id)
        if not profile:
            logger.error("No profile found for dataset %s", dataset_id)
            return False
        
        old_target = profile.chosen_target
        profile.chosen_target = new_target
        profile.target_locked = lock
        profile.target_override_reason = reason
        profile.user_overrides['target'] = new_target
        profile.user_overrides['target_locked'] = lock
        
        # Log override
        ctx.override_history.append({
            'dataset_id': dataset_id,
            'field': 'target',
            'old_value': old_target,
            'new_value': new_target,
            'reason': reason
        })
        
        # Persist
        dataset_profile_db.update_target(
            dataset_id,
            profile.target_candidates,
            new_target,
            lock,
            reason
        )
        
        ctx.log_decision(
            'target_override',
            f"User overrode target for dataset {dataset_id}: {old_target} -> {new_target}",
            reason
        )
        
        logger.info("Target overridden for dataset %s: %s (locked=%s)", dataset_id, new_target, lock)
        return True
    
    def lock_target(
        self,
        dataset_id: str,
        ctx: ExecutionContext
    ) -> bool:
        """Lock current target to prevent automatic changes."""
        profile = ctx.get_dataset_profile(dataset_id)
        if not profile:
            return False
        
        profile.target_locked = True
        profile.user_overrides['target_locked'] = True
        
        dataset_profile_db.update_target(
            dataset_id,
            profile.target_candidates,
            profile.chosen_target,
            locked=True
        )
        
        ctx.log_decision('target_lock', f"Locked target for dataset {dataset_id}: {profile.chosen_target}")
        return True
    
    def unlock_target(
        self,
        dataset_id: str,
        ctx: ExecutionContext
    ) -> bool:
        """Unlock target to allow re-detection."""
        profile = ctx.get_dataset_profile(dataset_id)
        if not profile:
            return False
        
        profile.target_locked = False
        profile.user_overrides['target_locked'] = False
        
        dataset_profile_db.update_target(
            dataset_id,
            profile.target_candidates,
            profile.chosen_target,
            locked=False
        )
        
        ctx.log_decision('target_unlock', f"Unlocked target for dataset {dataset_id}")
        return True
    
    def detect_for_session(
        self,
        ctx: ExecutionContext,
        data_map: Dict[str, pd.DataFrame],
        schema_map: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Tuple[List[Dict[str, Any]], Optional[str]]]:
        """
        Detect targets for all active datasets in a session.
        
        Returns:
            {dataset_id: (candidates, chosen_target)}
        """
        results = {}
        
        for dataset_id in ctx.active_dataset_ids:
            if dataset_id not in data_map:
                logger.warning("Dataset %s not in data_map, skipping target detection", dataset_id)
                continue
            
            data = data_map[dataset_id]
            schema = schema_map.get(dataset_id) if schema_map else None
            candidates, chosen = self.detect_dataset_target(dataset_id, data, ctx, schema)
            results[dataset_id] = (candidates, chosen)
        
        logger.info("Target detection complete for %d datasets", len(results))
        return results


# Global singleton
target_detector = ContextAwareTargetDetector()
