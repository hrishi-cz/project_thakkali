"""
Pipeline Orchestrator - Metadata and context coordination layer.

This orchestrator manages ExecutionContext lifecycle and metadata-centric
pipeline stages (registration, schema, target, aggregation, preprocessing plan).

Execution-heavy training stages are handled by
`pipeline/training_orchestrator.py`.

Usage:
    orchestrator = PipelineOrchestrator()
    ctx = orchestrator.load_or_create_context(session_id)
    orchestrator.execute_phase_2_schema(ctx, data_map)
    orchestrator.execute_phase_3_target(ctx, data_map)
    # ... etc
"""

from dataclasses import asdict
from typing import Dict, Any, List, Optional
import pandas as pd
import logging

from core.execution_context import (
    ExecutionContext,
    DatasetProfile,
    create_execution_context,
    validate_context,
)
from database.context_db import context_db
from data_ingestion.schema_detector import MultiDatasetSchemaDetector
from data_ingestion.integrator import Integrator
from preprocessing.preprocessing_planner import PreprocessingPlanner

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Coordinates metadata-centric orchestration phases.
    
    Responsibilities:
    - Load/save ExecutionContext from database
    - Orchestrate metadata phase execution
    - Call core modules (schema detector, target validator, etc.)
    - Update context with results
    - Ensure context validation before next phase
    """
    
    def __init__(self):
        """Initialize orchestrator with core components."""
        self.schema_detector = MultiDatasetSchemaDetector()
        self.preprocessing_planner = PreprocessingPlanner()
        self._integrator = Integrator()
        logger.info("PipelineOrchestrator initialized")

    def _warn_context_issues(self, ctx: ExecutionContext, stage: str) -> List[str]:
        """Log context validation issues without hard-failing phase execution."""
        issues = validate_context(ctx, stage)
        if not issues:
            return []

        for issue in issues:
            logger.warning("Context validation warning (%s): %s", stage, issue)

        ctx.log_decision(
            "validation_warning",
            f"{len(issues)} context issue(s) detected before stage '{stage}'",
            "; ".join(issues[:5]),
        )
        return issues

    @staticmethod
    def _safe_len(data: Any) -> int:
        """Best-effort row count helper for planner sizing heuristics."""
        try:
            return int(len(data))
        except Exception:
            return 0

    def _run_integrator_scoring(
        self,
        df: pd.DataFrame,
        target_col: str,
        task_type: str = "classification",
    ) -> Optional[Dict[str, Any]]:
        """
        Lightweight multimodal scoring using Integrator.

        Extracted to preserve modularity and allow reuse by downstream
        representation/fusion orchestration without duplicating phase logic.
        """
        if not isinstance(df, pd.DataFrame) or not target_col or target_col not in df.columns:
            return None

        modalities: List[str] = ["tabular"]

        text_cols = [
            col for col in df.columns
            if col != target_col and str(df[col].dtype) in ("object", "string", "category")
        ]
        if text_cols:
            modalities.append("text")

        image_cols = [
            col for col in df.columns
            if col != target_col
            and any(kw in str(col).lower() for kw in ("image", "img", "path", "photo", "pic"))
        ]
        if image_cols:
            modalities.append("image")

        tabular_cols = [
            col for col in df.columns
            if col != target_col and pd.api.types.is_numeric_dtype(df[col])
        ]
        if not tabular_cols and "tabular" in modalities:
            modalities.remove("tabular")

        if not modalities:
            return None

        sample_df = df.head(2000)
        raw_data_dict: Dict[str, Any] = {}
        forced_modalities: Dict[str, str] = {}

        if "tabular" in modalities:
            raw_data_dict["tabular"] = sample_df[tabular_cols]
            forced_modalities["tabular"] = "tabular"

        if "text" in modalities:
            raw_data_dict["text"] = (
                sample_df[text_cols].fillna("").astype(str).agg(" ".join, axis=1).tolist()
            )
            forced_modalities["text"] = "text"

        if "image" in modalities:
            image_col = image_cols[0]
            raw_data_dict["image"] = sample_df[image_col].fillna("").astype(str).tolist()
            forced_modalities["image"] = "image"

        if not raw_data_dict:
            return None

        try:
            target_values = sample_df[target_col].to_numpy()
            return self._integrator.process_multimodal(
                raw_data_dict=raw_data_dict,
                modalities=forced_modalities,
                y=target_values,
                task_type=task_type,
            )
        except Exception as exc:
            logger.warning(
                "PipelineOrchestrator: Integrator scoring failed for target %s: %s",
                target_col,
                exc,
            )
            return None
    
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
        expected_revision = int(getattr(ctx, "revision", 0) or 0)
        ctx.revision = context_db.save_context(
            ctx.to_dict(),
            expected_revision=expected_revision,
        )
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

        if not data_map:
            logger.warning("No datasets provided for schema detection")
            self.save_context(ctx)
            return

        pending_data: Dict[str, Any] = {}
        for dataset_id, data in data_map.items():
            profile = ctx.get_dataset_profile(dataset_id)
            if not profile:
                logger.warning("No profile for dataset %s, creating one", dataset_id)
                profile = DatasetProfile(dataset_id=dataset_id)
                ctx.add_dataset_profile(profile)

            if profile.schema_detected and not profile.user_overrides.get('force_redetect'):
                logger.info("Schema already detected for %s, skipping", dataset_id)
                continue

            pending_data[dataset_id] = data

        if not pending_data:
            logger.info("No datasets required schema redetection")
            self.save_context(ctx)
            return

        try:
            global_schema = self.schema_detector.detect_global_schema(pending_data)

            # Populate architecture routing signals from per-dataset results
            _total_feats = sum(
                int(ds.get("num_features", 0) or 0) for ds in global_schema.per_dataset
            )
            _has_relational = any(
                bool(ds.get("has_relational_columns") or ds.get("foreign_keys"))
                for ds in global_schema.per_dataset
            )
            global_schema.total_feature_count = _total_feats
            global_schema.has_relational_columns = _has_relational

            global_schema_dict = asdict(global_schema)
            ctx.set_global_schema(
                schema=global_schema_dict,
                confidence=float(global_schema.detection_confidence),
                evidence=f"Aggregated {len(global_schema.per_dataset)} dataset schema(s)",
            )
            ctx.datasets_compatible = bool(
                global_schema.relatedness_report.get("n_groups", 1) == 1
            )
            ctx.compatibility_matrix = global_schema.relatedness_report
            ctx.log_decision(
                "compatibility",
                f"Datasets compatible: {bool(ctx.datasets_compatible)}",
                evidence=f"matrix_rows={len(ctx.compatibility_matrix or [])}",
            )

            for dataset_result in global_schema.per_dataset:
                dataset_id = str(dataset_result.get("dataset_id", ""))
                if not dataset_id:
                    continue

                profile = ctx.get_dataset_profile(dataset_id)
                if not profile:
                    profile = DatasetProfile(dataset_id=dataset_id)
                    ctx.add_dataset_profile(profile)

                modalities = list(dataset_result.get("modalities", []))
                reasoning = dataset_result.get("reasoning", {})
                xs3_gap = float(
                    reasoning.get(
                        "xs3_confidence_gap",
                        reasoning.get(
                            "confidence_gap",
                            dataset_result.get("confidence", 0.0),
                        ),
                    )
                )

                profile.schema_detected = True
                profile.schema_result = dataset_result
                profile.schema_confidence = float(dataset_result.get("confidence", 0.0))
                profile.schema_evidence = (
                    f"Detected {len(modalities)} modalities; "
                    f"X-S3 confidence gap {xs3_gap:.3f}"
                )
                profile.text_task_type = dataset_result.get("text_task_type")
                profile.modality_breakdown = {
                    mod: 1.0 / len(modalities) if modalities else 0.0
                    for mod in modalities
                }

                # FIX-21: score all detectable modalities (tabular/text/image)
                # through a reusable Integrator boundary.
                _df = pending_data.get(dataset_id)
                if _df is None:
                    _df = data_map.get(dataset_id)

                if isinstance(_df, pd.DataFrame):
                    _target_col = str(
                        dataset_result.get("target_column")
                        or profile.chosen_target
                        or ctx.global_target
                        or ""
                    )
                    if _target_col and _target_col in _df.columns:
                        _task_type = (
                            "regression"
                            if "regression" in str(
                                global_schema_dict.get("global_problem_type", "")
                            ).lower()
                            else "classification"
                        )
                        _meta = self._run_integrator_scoring(
                            _df,
                            _target_col,
                            task_type=_task_type,
                        )
                        if _meta:
                            _tab_meta = _meta.get("tabular")
                            if _tab_meta is not None:
                                _tab_score = float(
                                    getattr(_tab_meta, "predictability_score", 0.0)
                                )
                                _tab_prev = ctx.predictability_scores.get(dataset_id)
                                ctx.predictability_scores[dataset_id] = _tab_score
                                ctx.log_decision(
                                    "predictability",
                                    f"Predictability[{dataset_id}] = {_tab_score:.3f}",
                                    evidence=f"prev={_tab_prev}",
                                )
                                logger.info(
                                    "PipelineOrchestrator: Integrator scored tabular predictability=%.3f for dataset %s",
                                    _tab_score,
                                    dataset_id,
                                )

                            # Extract scores for detected non-tabular modalities.
                            # Bug 8 fix: ModalityEncoder starts with text/image encoders
                            # absent (heavy models not loaded at startup).  The integrator
                            # returns predictability_score=0.0 for those modalities with a
                            # warning "encoder is not initialized".  Storing 0.0 causes
                            # AdvancedModelSelector to treat them as "low quality" and
                            # exclude them (threshold < 0.25).  The correct interpretation:
                            # score=0.0 from absent encoder means "unscored", not "bad".
                            # Only store scores when the encoder actually ran (score > 0).
                            for _mod_name in ("text", "image"):
                                _mod_meta = _meta.get(_mod_name)
                                if _mod_meta is None:
                                    continue
                                _mod_score = float(
                                    getattr(_mod_meta, "predictability_score", 0.0)
                                )
                                if _mod_score <= 0.0:
                                    # Encoder absent or failed — don't store 0.0.
                                    # _resolve_modality_predictability returns None for
                                    # missing keys → modality treated as eligible.
                                    logger.info(
                                        "PipelineOrchestrator: %s encoder absent — "
                                        "predictability not stored (modality stays eligible)",
                                        _mod_name,
                                    )
                                    continue
                                _mod_prev = ctx.predictability_scores.get(_mod_name)
                                ctx.predictability_scores[_mod_name] = _mod_score
                                ctx.log_decision(
                                    "predictability",
                                    f"Predictability[{_mod_name}] = {_mod_score:.3f}",
                                    evidence=f"prev={_mod_prev}",
                                )
                                logger.info(
                                    "PipelineOrchestrator: Integrator scored %s predictability=%.3f for dataset %s",
                                    _mod_name,
                                    _mod_score,
                                    dataset_id,
                                )
                        else:
                            logger.info(
                                "PipelineOrchestrator: no active modalities detected for %s - skipping Integrator",
                                dataset_id,
                            )

                self.save_profile(profile, ctx.session_id)
                ctx.log_decision(
                    'schema_detection',
                    f"Detected schema for {dataset_id}",
                    f"xs3_confidence_gap={xs3_gap:.3f}",
                )

        except Exception as e:
            logger.error("Schema detection failed: %s", e, exc_info=True)
            for dataset_id in pending_data:
                profile = ctx.get_dataset_profile(dataset_id)
                if not profile:
                    continue
                profile.schema_detected = False
                profile.schema_result = {"error": str(e)}
                self.save_profile(profile, ctx.session_id)
        
        # Aggregate semantic intelligence from all per-dataset IndividualSchema dicts
        # into ctx.feature_intelligence so every downstream phase can read it.
        try:
            _per_dataset = list((global_schema_dict or {}).get("per_dataset", []))
            if _per_dataset:
                ctx.set_feature_intelligence(_per_dataset)
        except Exception as _fi_exc:
            logger.debug("set_feature_intelligence failed (non-fatal): %s", _fi_exc)

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

        for dataset_id in data_map:
            profile = ctx.get_dataset_profile(dataset_id)
            if not profile:
                logger.warning("No profile for dataset %s, skipping target detection", dataset_id)
                continue
            
            # Skip if target locked
            if profile.target_locked:
                logger.info("Target locked for %s, skipping detection", dataset_id)
                continue
            
            try:
                schema_result = profile.schema_result or {}
                chosen_target = schema_result.get('target_column')
                if not chosen_target or chosen_target == 'Unknown':
                    logger.warning(
                        "No detected target for %s; skipping target selection",
                        dataset_id,
                    )
                    continue

                ranked_candidates: List[Dict[str, Any]] = []
                for candidate in schema_result.get('candidates', []):
                    if not isinstance(candidate, dict):
                        continue
                    name = candidate.get('column')
                    if not name:
                        continue
                    score = float(candidate.get('final_score', candidate.get('score', 0.0)))
                    ranked_candidates.append(
                        {
                            'name': name,
                            'score': score,
                            'final_score': score,
                            'reason': candidate.get('reason', 'Detected by schema detector'),
                        }
                    )

                if not ranked_candidates:
                    ranked_candidates = [
                        {
                            'name': chosen_target,
                            'score': float(profile.schema_confidence),
                            'final_score': float(profile.schema_confidence),
                            'reason': 'Detected by schema detector',
                        }
                    ]

                reasoning = schema_result.get('reasoning', {})
                xs3_gap = float(
                    reasoning.get(
                        'xs3_confidence_gap',
                        reasoning.get('confidence_gap', profile.schema_confidence),
                    )
                )

                profile.target_detected = True
                profile.target_candidates = ranked_candidates
                profile.chosen_target = chosen_target
                ctx.log_decision(
                    "target_per_dataset",
                    f"Target for {profile.dataset_id}: {profile.chosen_target}",
                    evidence=f"candidates={profile.target_candidates}",
                )

                self.save_profile(profile, ctx.session_id)
                ctx.log_decision(
                    'target_detection',
                    f"Selected target {chosen_target} for {dataset_id}",
                    f"xs3_confidence_gap={xs3_gap:.3f}",
                )

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
        
        targets = [
            p.chosen_target
            for p in profiles
            if p.chosen_target and p.chosen_target != 'Unknown'
        ]
        if targets:
            from collections import Counter
            counts = Counter(targets)
            ordered = counts.most_common()
            most_common_target, top_count = ordered[0]
            confidence = top_count / len(targets)
            global_candidates = [
                {
                    'name': target,
                    'score': count / len(targets),
                    'reason': 'Most common across datasets',
                }
                for target, count in ordered
            ]
            
            ctx.set_global_target(
                most_common_target,
                confidence,
                global_candidates,
            )
        
        # Mark as compatible if same target
        ctx.datasets_compatible = len(set(targets)) == 1 if targets else False
        ctx.compatibility_matrix = {
            'target_counts': {t: targets.count(t) for t in sorted(set(targets))},
            'datasets_compatible': ctx.datasets_compatible,
        }
        ctx.log_decision(
            "aggregation",
            f"Post-aggregation compatibility: {bool(ctx.datasets_compatible)}",
            evidence=f"primary_dataset_id={ctx.primary_dataset_id}",
        )
        
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
        self._warn_context_issues(ctx, 'preprocessing')

        global_modalities: List[str] = []
        if isinstance(ctx.global_schema, dict):
            global_modalities = list(ctx.global_schema.get('global_modalities', []))

        context_signals: Dict[str, Any] = {}
        if hasattr(ctx, "get_preprocessing_signals"):
            try:
                context_signals = dict(ctx.get_preprocessing_signals() or {})
            except Exception:
                context_signals = {}
        if not context_signals:
            context_signals = {
                "global_schema": dict(getattr(ctx, "global_schema", {}) or {}),
                "modality_presence": {
                    str(k): bool(v)
                    for k, v in dict(getattr(ctx, "modality_presence", {}) or {}).items()
                },
                "predictability_scores": {
                    str(k): float(v)
                    for k, v in dict(getattr(ctx, "predictability_scores", {}) or {}).items()
                    if isinstance(v, (int, float))
                },
                "drift_adjusted_predictability": {
                    str(k): float(v)
                    for k, v in dict(getattr(ctx, "drift_adjusted_predictability", {}) or {}).items()
                    if isinstance(v, (int, float))
                },
                "drifted_features": [
                    str(col) for col in list(getattr(ctx, "drifted_features", []) or [])
                ],
                "drift_feedback_applied": bool(getattr(ctx, "drift_feedback_applied", False)),
                "training_fit_analysis": dict(getattr(ctx, "training_fit_analysis", {}) or {}),
            }

        context_predictability: Dict[str, float] = {}
        context_predictability = {
            str(k): float(v)
            for k, v in dict(context_signals.get("predictability_scores", {}) or {}).items()
            if isinstance(v, (int, float))
        }
        context_modality_presence: Dict[str, bool] = {
            str(k): bool(v)
            for k, v in dict(context_signals.get("modality_presence", {}) or {}).items()
        }
        context_drift_adjusted: Dict[str, float] = {
            str(k): float(v)
            for k, v in dict(context_signals.get("drift_adjusted_predictability", {}) or {}).items()
            if isinstance(v, (int, float))
        }
        context_drifted_features: List[str] = [
            str(col) for col in list(context_signals.get("drifted_features", []) or [])
        ]
        global_schema_context: Dict[str, Any] = dict(context_signals.get("global_schema", {}) or {})

        dataset_plans: Dict[str, Dict[str, Any]] = {}
        
        for dataset_id, data in data_map.items():
            profile = ctx.get_dataset_profile(dataset_id)
            if not profile or not profile.schema_detected:
                logger.warning("Skipping preprocessing for %s (no schema)", dataset_id)
                continue

            schema_result = profile.schema_result or {}
            dataset_modalities = list(schema_result.get('modalities', [])) or global_modalities
            schema_info = {
                'global_modalities': dataset_modalities,
            }
            total_samples = self._safe_len(data)
            plan = self.preprocessing_planner.create_plan(
                schema_info=schema_info,
                total_samples=total_samples,
                predictability_scores=context_predictability,
                modality_presence=context_modality_presence,
                drift_adjusted_predictability=context_drift_adjusted,
                drifted_features=context_drifted_features,
                global_schema=global_schema_context,
                preprocessing_hints=dict(schema_result.get("preprocessing_hints", {}) or {}),
                feature_intelligence=dict((ctx.feature_intelligence or {}).get(dataset_id, {}) or {}),
            )
            plan['dataset_id'] = dataset_id
            plan['target_column'] = profile.chosen_target or ctx.global_target

            # Inject feature_intelligence signals into the modality sub-plans
            _fi_ds = (ctx.feature_intelligence or {}).get(dataset_id, {})
            if _fi_ds:
                # Text: set max_length from avg token length (cap 16–512)
                _avg_tl = _fi_ds.get("avg_text_len")
                if _avg_tl:
                    _dynamic_max = min(512, max(16, int(_avg_tl * 2)))
                    _text_plan = plan.get("text")
                    if isinstance(_text_plan, dict):
                        _text_plan.setdefault("max_length", _dynamic_max)
                    elif not _text_plan:
                        plan["text"] = {"max_length": _dynamic_max}

                # Tabular: register id_columns and high_missing_cols for auto-drop
                _id_cols = list(_fi_ds.get("id_columns") or [])
                _hi_miss = list(_fi_ds.get("high_missing_cols") or [])
                _detected_cols = dict(schema_result.get("detected_columns") or {})
                _carrier_cols = set((_detected_cols.get("text") or []) + (_detected_cols.get("image") or []))
                _id_cols = [c for c in _id_cols if c not in _carrier_cols]
                _hi_miss = [c for c in _hi_miss if c not in _carrier_cols]
                _tab_plan = plan.get("tabular")
                if isinstance(_tab_plan, dict) and (_id_cols or _hi_miss):
                    _existing_drop = list(_tab_plan.get("columns_to_drop") or [])
                    _combined = list(dict.fromkeys(_existing_drop + _id_cols + _hi_miss))
                    _tab_plan["columns_to_drop"] = _combined
                    logger.info(
                        "  feature_intelligence: scheduling %d id/high-missing cols for drop: %s",
                        len(_combined), _combined[:10],
                    )

            reasoning = schema_result.get('reasoning', {})
            xs3_gap = reasoning.get('xs3_confidence_gap', reasoning.get('confidence_gap'))
            if xs3_gap is not None:
                plan['xs3_confidence_gap'] = float(xs3_gap)
            
            profile.preprocessing_plan = plan
            dataset_plans[dataset_id] = dict(plan)
            self.save_profile(profile, ctx.session_id)
            ctx.log_decision('preprocessing', f"Created preprocessing plan for {dataset_id}")

        if dataset_plans:
            first_plan = next(iter(dataset_plans.values()))
            context_plan = {
                modality: dict(first_plan.get(modality, {}) or {})
                for modality in ("tabular", "text", "image")
                if isinstance(first_plan.get(modality), dict)
            }
            if context_plan:
                ctx.update_preprocessing(context_plan)

            ctx.preprocessing_context["runtime"] = dict(first_plan.get("runtime", {}) or {})
            ctx.preprocessing_context["weak_modalities"] = list(first_plan.get("weak_modalities", []) or [])
            ctx.preprocessing_context["strong_modalities"] = list(first_plan.get("strong_modalities", []) or [])
            ctx.preprocessing_context["modality_predictability"] = dict(first_plan.get("modality_predictability", {}) or {})
            ctx.preprocessing_context["context_signals"] = dict(first_plan.get("context_signals", {}) or {})
            ctx.preprocessing_context["dataset_plans"] = dataset_plans

        self._warn_context_issues(ctx, 'model_selection')

        self.run_architecture_selection(ctx)
        self.save_context(ctx)
        logger.info("Phase 5 (Preprocessing) complete for session %s", ctx.session_id)

    def run_architecture_selection(self, ctx: ExecutionContext) -> None:
        """
        Derive head architecture type and encoder output dims from schema + modality signals.

        Called between preprocessing and model selection so TrainingOrchestrator
        can read head_architecture_type when constructing the model head.

        Uses feature_intelligence cross-modality interaction scores (when available)
        to refine fusion/head strategy beyond simple modality count heuristics.
        """
        schema = ctx.global_schema or {}
        modalities = list(
            getattr(ctx, "active_modalities", []) or ctx.eligible_modalities
            or schema.get("active_modalities", [])
            or schema.get("global_modalities", [])
            or []
        )
        n_mod = len(modalities)
        problem_type = str(
            schema.get("global_problem_type")
            or getattr(ctx, "global_problem_type", "")
            or ""
        )

        # --- Compute cross-modality interaction strength from feature_intelligence ---
        mean_interaction_score: float = 0.0
        max_interaction_score: float = 0.0
        avg_text_len: float = 0.0
        avg_image_dataset_size: float = 0.0
        mean_image_separability: float = 0.0
        mean_image_class_balance: float = 0.0
        mean_uncertainty: float = 0.0
        n_id_cols: int = 0
        n_high_missing: int = 0
        fi = ctx.feature_intelligence or {}
        if fi:
            all_scores: list = []
            text_len_vals: list = []
            image_size_vals: list = []
            image_sep_vals: list = []
            image_balance_vals: list = []
            uncertainty_vals: list = []
            for ds_intel in fi.values():
                inter = ds_intel.get("interaction_summary") or {}
                if isinstance(inter, dict):
                    all_scores.extend(float(v) for v in inter.values() if v is not None)
                unc = ds_intel.get("uncertainty_summary") or {}
                if isinstance(unc, dict):
                    uncertainty_vals.extend(
                        float(v) for v in unc.values()
                        if isinstance(v, (int, float))
                    )
                tl = ds_intel.get("avg_text_len")
                if tl:
                    text_len_vals.append(float(tl))
                img_size = ds_intel.get("image_dataset_size")
                if img_size:
                    image_size_vals.append(float(img_size))
                img_sep = ds_intel.get("image_label_separability")
                if img_sep is not None:
                    image_sep_vals.append(float(img_sep))
                img_bal = ds_intel.get("image_class_balance")
                if img_bal is not None:
                    image_balance_vals.append(float(img_bal))
                n_id_cols += len(ds_intel.get("id_columns") or [])
                n_high_missing += len(ds_intel.get("high_missing_cols") or [])
            if all_scores:
                mean_interaction_score = sum(all_scores) / len(all_scores)
                max_interaction_score = max(all_scores)
            if text_len_vals:
                avg_text_len = sum(text_len_vals) / len(text_len_vals)
            if image_size_vals:
                avg_image_dataset_size = sum(image_size_vals) / len(image_size_vals)
            if image_sep_vals:
                mean_image_separability = sum(image_sep_vals) / len(image_sep_vals)
            if image_balance_vals:
                mean_image_class_balance = sum(image_balance_vals) / len(image_balance_vals)
            if uncertainty_vals:
                mean_uncertainty = sum(uncertainty_vals) / len(uncertainty_vals)

        ctx.head_hidden_dim = 256
        ctx.head_num_layers = 3
        ctx.encoder_plan = {}

        # --- Head type selection (interaction-aware) ---
        relational = bool(schema.get("has_relational_columns"))
        high_interaction = mean_interaction_score > 0.4 or max_interaction_score > 0.6
        low_interaction = mean_interaction_score < 0.15 and max_interaction_score < 0.25

        if relational or n_mod >= 3:
            ctx.head_architecture_type = "graph"
        elif n_mod >= 2 and high_interaction:
            # Strong inter-modality signal → attention fusion captures cross-modal deps
            ctx.head_architecture_type = "attention"
            ctx.head_hidden_dim = 512
        elif "text" in modalities and n_mod >= 2 and not low_interaction:
            ctx.head_architecture_type = "attention"
            ctx.head_hidden_dim = 512
        elif n_mod >= 2 and low_interaction:
            # Modalities are nearly independent → simple concat is sufficient
            ctx.head_architecture_type = "mlp"
            if not hasattr(ctx, "_fusion_override_active"):
                ctx.fusion_strategy = "concatenation"
        elif "regression" in problem_type and n_mod == 1:
            ctx.head_architecture_type = "mlp"
            ctx.head_num_layers = 4
        else:
            ctx.head_architecture_type = "mlp"

        # Scale head hidden dim by interaction strength (capped between 128 and 1024)
        if high_interaction and ctx.head_architecture_type in ("attention", "graph"):
            ctx.head_hidden_dim = min(1024, max(256, int(ctx.head_hidden_dim * (1.0 + mean_interaction_score))))

        # Regression tasks benefit from a wider representation
        if "regression" in problem_type:
            ctx.head_hidden_dim = max(ctx.head_hidden_dim, 512)

        total_features = int(schema.get("total_feature_count", 0) or 0)

        # --- Image-aware routing ---
        if "image" in modalities:
            if 0 < avg_image_dataset_size < 5_000:
                ctx.encoder_plan["image"] = "mobilenet"
            elif mean_image_separability < 0.35 and avg_image_dataset_size >= 10_000:
                ctx.encoder_plan["image"] = "efficientnet"
            elif avg_image_dataset_size >= 5_000:
                ctx.encoder_plan["image"] = "resnet50"
            else:
                ctx.encoder_plan["image"] = "mobilenet"

            if mean_image_separability >= 0.70:
                ctx.head_num_layers = 2
                ctx.head_hidden_dim = max(192, min(ctx.head_hidden_dim, 384))
                ctx.constraints["dropout_floor"] = max(
                    0.05, float(ctx.constraints.get("dropout_floor", 0.05) or 0.05)
                )
            elif mean_image_separability <= 0.35:
                ctx.head_num_layers = max(ctx.head_num_layers, 4)
                ctx.head_hidden_dim = max(ctx.head_hidden_dim, 512)
                ctx.constraints["dropout_floor"] = max(
                    0.20, float(ctx.constraints.get("dropout_floor", 0.20) or 0.20)
                )

        # --- Text-aware routing ---
        if "text" in modalities:
            if avg_text_len and avg_text_len < 50:
                ctx.encoder_plan["text"] = "distilbert"
            elif avg_text_len and avg_text_len > 200:
                ctx.encoder_plan["text"] = "deberta"
            else:
                ctx.encoder_plan["text"] = "bert"

        # --- Tabular-aware routing ---
        if "tabular" in modalities:
            ctx.encoder_plan["tabular"] = (
                "grn" if (total_features > 32 or mean_uncertainty > 0.25) else "mlp"
            )

        # --- Encoder output dims ---
        if total_features > 200:
            ctx.encoder_output_dims["tabular"] = 64
        elif total_features < 10 and total_features > 0:
            ctx.encoder_output_dims["tabular"] = 8
        else:
            ctx.encoder_output_dims["tabular"] = 16

        # Scale text encoder dim by average sequence length
        if "text" in modalities:
            if avg_text_len > 256:
                ctx.encoder_output_dims["text"] = 1024
            elif avg_text_len > 64:
                ctx.encoder_output_dims["text"] = 768
            else:
                ctx.encoder_output_dims["text"] = 512

        ctx.log_decision(
            "architecture",
            f"Head={ctx.head_architecture_type}, hidden={ctx.head_hidden_dim}, "
            f"layers={ctx.head_num_layers}, encoder_dims={ctx.encoder_output_dims}",
            evidence=(
                f"modalities={modalities}, relational={relational}, "
                f"n_features={total_features}, problem_type={problem_type}, "
                f"mean_interaction={mean_interaction_score:.3f}, "
                f"max_interaction={max_interaction_score:.3f}, "
                f"avg_text_len={avg_text_len:.1f}, "
                f"avg_image_dataset_size={avg_image_dataset_size:.1f}, "
                f"image_separability={mean_image_separability:.3f}, "
                f"image_class_balance={mean_image_class_balance:.3f}, "
                f"mean_uncertainty={mean_uncertainty:.3f}, "
                f"encoder_plan={ctx.encoder_plan}, "
                f"high_missing_cols={n_high_missing}, id_cols={n_id_cols}"
            ),
        )
        logger.info(
            "Architecture selection: head=%s, hidden=%d, tabular_dim=%d, "
            "interaction=%.3f, text_dim=%s, encoder_plan=%s",
            ctx.head_architecture_type,
            ctx.head_hidden_dim,
            ctx.encoder_output_dims.get("tabular", 16),
            mean_interaction_score,
            ctx.encoder_output_dims.get("text", "n/a"),
            ctx.encoder_plan,
        )
        self.save_context(ctx)


# Global singleton
orchestrator = PipelineOrchestrator()
