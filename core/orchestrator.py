"""
Pipeline Orchestrator - Coordinates All 8 Pipeline Phases

This is the central coordinator that manages ExecutionContext lifecycle
and orchestrates phase execution across the entire pipeline.

Usage:
    orchestrator = PipelineOrchestrator()
    ctx = orchestrator.load_or_create_context(session_id)
    orchestrator.execute_phase_2_schema(ctx, data_map)
    orchestrator.execute_phase_3_target(ctx, data_map)
    # ... etc
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import logging

from core.execution_context import ExecutionContext, DatasetProfile, create_execution_context
from database.context_db import context_db
from data_ingestion.schema_detector import COGMASchemaDetector
from data_ingestion.integrator import Integrator

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Coordinates execution of all 8 pipeline phases.
    
    Responsibilities:
    - Load/save ExecutionContext from database
    - Orchestrate phase execution
    - Call core modules (schema detector, target validator, etc.)
    - Update context with results
    - Ensure context validation before next phase
    """
    
    def __init__(self):
        """Initialize orchestrator with core components."""
        self.schema_detector = COGMASchemaDetector()
        # Lazy initialization - only create when needed to avoid import errors
        self._integrator = None
        logger.info("PipelineOrchestrator initialized")
    
    @property
    def integrator(self):
        """Lazy-load integrator to avoid import issues."""
        if self._integrator is None:
            self._integrator = Integrator()
        return self._integrator
    
    # ===== Context Lifecycle =====
    
    def load_or_create_context(self, session_id: str) -> ExecutionContext:
        """Load context from DB or create new."""
        data = context_db.load_context(session_id)
        if data:
            logger.info("Loaded context for session %s", session_id)
            return ExecutionContext.from_dict(data)
        
        logger.info("Creating new context for session %s", session_id)
        ctx = create_execution_context(session_id)
        self.save_context(ctx)
        return ctx
    
    def save_context(self, ctx: ExecutionContext) -> None:
        """Persist context to database."""
        context_db.save_context(ctx.to_dict())
        logger.info("Saved context for session %s", ctx.session_id)
    
    def save_profile(self, profile: DatasetProfile, session_id: str) -> None:
        """Persist dataset profile to database."""
        context_db.save_profile(profile.to_dict(), session_id)
        logger.info("Saved profile for dataset %s", profile.dataset_id)
    
    # ===== Phase 1: Data Ingestion (Post-Processing) =====
    
    def register_ingested_datasets(
        self,
        ctx: ExecutionContext,
        ingested_hashes: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Called after DataIngestionManager completes.
        Creates DatasetProfile for each ingested dataset.
        
        Args:
            ctx: Current execution context
            ingested_hashes: {hash: {source, file_path, ...}}
        """
        for dataset_id, metadata in ingested_hashes.items():
            # Check if already exists
            if ctx.get_dataset_profile(dataset_id):
                logger.info("Dataset %s already registered, skipping", dataset_id)
                continue
            
            # Create profile
            profile = DatasetProfile(
                dataset_id=dataset_id,
                source_url=metadata.get('source'),
                file_path=metadata.get('file_path')
            )
            
            ctx.add_dataset_profile(profile)
            self.save_profile(profile, ctx.session_id)
        
        ctx.set_pipeline_stage('ingestion_complete')
        self.save_context(ctx)
        logger.info("Registered %d datasets to context", len(ingested_hashes))
    
    # ===== Phase 2: Schema Detection =====
    
    def execute_phase_2_schema(
        self,
        ctx: ExecutionContext,
        data_map: Dict[str, pd.DataFrame]
    ) -> None:
        """
        Execute Phase 2: Schema Detection.
        
        Updates each DatasetProfile with schema information.
        """
        ctx.set_pipeline_stage('schema_detection')
        
        for dataset_id, data in data_map.items():
            profile = ctx.get_dataset_profile(dataset_id)
            if not profile:
                logger.warning("No profile for dataset %s, creating one", dataset_id)
                profile = DatasetProfile(dataset_id=dataset_id)
                ctx.add_dataset_profile(profile)
            
            # Skip if already detected
            if profile.schema_detected and not profile.user_overrides.get('force_redetect'):
                logger.info("Schema already detected for %s, skipping", dataset_id)
                continue
            
            # Use COGMA detector
            try:
                schema = self.schema_detector.detect_schema(data)
                
                # Update profile
                profile.schema_detected = True
                profile.schema_result = {
                    'columns': schema.detected_columns,
                    'target_column': schema.target_column,
                    'problem_type': schema.problem_type,
                    'modalities': schema.modalities,
                    'confidence': schema.confidence
                }
                profile.schema_confidence = schema.confidence
                profile.schema_evidence = f"Detected {len(schema.modalities)} modalities"
                
                # Store modality breakdown
                profile.modality_breakdown = {
                    mod: 1.0 / len(schema.modalities) if schema.modalities else 0.0
                    for mod in schema.modalities
                }
                
                self.save_profile(profile, ctx.session_id)
                ctx.log_decision('schema_detection', f"Detected schema for {dataset_id}")
                
            except Exception as e:
                logger.error("Schema detection failed for %s: %s", dataset_id, e, exc_info=True)
                profile.schema_detected = False
                profile.schema_result = {"error": str(e)}
        
        self.save_context(ctx)
        logger.info("Phase 2 (Schema Detection) complete for session %s", ctx.session_id)
    
    # ===== Phase 3: Target Detection =====
    
    def execute_phase_3_target(
        self,
        ctx: ExecutionContext,
        data_map: Dict[str, pd.DataFrame]
    ) -> None:
        """
        Execute Phase 3: Target Detection.
        
        Updates each DatasetProfile with target candidates.
        """
        ctx.set_pipeline_stage('target_detection')
        
        for dataset_id, data in data_map.items():
            profile = ctx.get_dataset_profile(dataset_id)
            if not profile:
                logger.warning("No profile for dataset %s, skipping target detection", dataset_id)
                continue
            
            # Skip if target locked
            if profile.target_locked:
                logger.info("Target locked for %s, skipping detection", dataset_id)
                continue
            
            # Simple target ranking (using schema detector's logic)
            try:
                # Get target from schema if available
                if profile.schema_result and 'target_column' in profile.schema_result:
                    chosen_target = profile.schema_result['target_column']
                    
                    profile.target_detected = True
                    profile.target_candidates = [
                        {
                            'name': chosen_target,
                            'score': profile.schema_confidence,
                            'reason': 'Detected by COGMA schema detector'
                        }
                    ]
                    profile.chosen_target = chosen_target
                    
                    self.save_profile(profile, ctx.session_id)
                    ctx.log_decision('target_detection', f"Selected target {chosen_target} for {dataset_id}")
                
            except Exception as e:
                logger.error("Target detection failed for %s: %s", dataset_id, e, exc_info=True)
        
        self.save_context(ctx)
        logger.info("Phase 3 (Target Detection) complete for session %s", ctx.session_id)
    
    # ===== Phase 4: Global Aggregation =====
    
    def execute_phase_4_aggregation(
        self,
        ctx: ExecutionContext,
        data_map: Dict[str, pd.DataFrame]
    ) -> None:
        """
        Execute Phase 4: Global Aggregation.
        
        Infers global schema and target across all datasets.
        """
        ctx.set_pipeline_stage('global_aggregation')
        
        profiles = ctx.get_active_profiles()
        if not profiles:
            logger.warning("No active profiles for aggregation")
            return
        
        # Simple aggregation: use most common target
        targets = [p.chosen_target for p in profiles if p.chosen_target]
        if targets:
            from collections import Counter
            most_common_target = Counter(targets).most_common(1)[0][0]
            confidence = targets.count(most_common_target) / len(targets)
            
            ctx.set_global_target(
                most_common_target,
                confidence,
                [{'name': most_common_target, 'score': confidence, 'reason': 'Most common across datasets'}]
            )
        
        # Mark as compatible if same target
        ctx.datasets_compatible = len(set(targets)) == 1 if targets else False
        
        self.save_context(ctx)
        logger.info("Phase 4 (Global Aggregation) complete for session %s", ctx.session_id)
    
    # ===== Phase 5: Preprocessing Planning =====
    
    def execute_phase_5_preprocessing(
        self,
        ctx: ExecutionContext,
        data_map: Dict[str, pd.DataFrame]
    ) -> None:
        """
        Execute Phase 5: Preprocessing Planning.
        
        Creates preprocessing plan for each dataset.
        """
        ctx.set_pipeline_stage('preprocessing_planning')
        
        for dataset_id, data in data_map.items():
            profile = ctx.get_dataset_profile(dataset_id)
            if not profile or not profile.schema_detected:
                logger.warning("Skipping preprocessing for %s (no schema)", dataset_id)
                continue
            
            # Simple preprocessing plan based on modality
            plan = {
                'dataset_id': dataset_id,
                'target_column': profile.chosen_target or ctx.global_target,
                'steps': []
            }
            
            # Add steps based on modality
            if profile.modality_breakdown.get('tabular', 0) > 0:
                plan['steps'].append({
                    'name': 'handle_missing_values',
                    'modality': 'tabular',
                    'params': {'strategy': 'median'}
                })
                plan['steps'].append({
                    'name': 'scale_numeric',
                    'modality': 'tabular',
                    'params': {'strategy': 'standard'}
                })
            
            profile.preprocessing_plan = plan
            self.save_profile(profile, ctx.session_id)
            ctx.log_decision('preprocessing', f"Created preprocessing plan for {dataset_id}")
        
        self.save_context(ctx)
        logger.info("Phase 5 (Preprocessing) complete for session %s", ctx.session_id)


# Global singleton
orchestrator = PipelineOrchestrator()
