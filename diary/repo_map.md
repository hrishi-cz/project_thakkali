final*worktree/
|
|-- api/
| |-- **init**.py
| |-- run_api.py
| |-- run_server.py
| |-- session_manager.py
|
|-- automl/
| |-- **init**.py
| |-- # Comprehensive Multimodal Pipeline Audi.prompt.md
| |-- advanced_selector.py
| |-- candidate_selector.py
| |-- jit_encoder_selector.py
| |-- meta_learning.py
| |-- model_selector.py
| |-- optuna_adaptive.py
| |-- trainer.py
| |-- trial_intelligence.py
|
|-- config/
| |-- **init**.py
| |-- encoder_plugins.py
| |-- hyperparameters.py
| |-- paths.py
|
|-- core/
| |-- **init**.py
| |-- execution_context.py
| |-- orchestrator.py
| |-- types.py
|
|-- data/
| |-- dataset_cache/ # runtime cache
| |-- embedding_cache/ # runtime cache
| |-- sessions.db # sqlite context database
|
|-- data_ingestion/
| |-- **init**.py
| |-- data_bridge.py
| |-- dataset_object.py
| |-- ingestion_manager.py
| |-- integrator.py
| |-- loader.py
| |-- modality_encoder.py
| |-- sampling.py
| |-- schema.py
| |-- schema_detector.py
| |-- semantic_analyzer.py
| |-- target_validator.py
| |-- xs3_target_selector.py
| |-- adapters/
| | |-- **init**.py
| | |-- ecg_adapter.py
|
|-- database/
| |-- **init**.py
| |-- context_db.py
|
|-- frontend/
| |-- **init**.py
| |-- app_enhanced.py
|
|-- model_registry_pkg/
| |-- **init**.py
| |-- model_registry.py
|
|-- models/ # compatibility mirror package
| |-- **init**.py
| |-- fusion.py
| |-- multimodal_alignment.py
| |-- predictor.py
| |-- encoders/
| | |-- **init**.py
| | |-- image.py
| | |-- tabular.py
| | |-- text.py
| |-- registry/ # model artifact folders
|
|-- modelss/ # primary model implementation package
| |-- **init**.py
| |-- fusion.py
| |-- multimodal_alignment.py
| |-- predictor.py
| |-- encoders/
| | |-- **init**.py
| | |-- image.py
| | |-- tabular.py
| | |-- text.py
|
|-- monitoring/
| |-- **init**.py
| |-- drift_detector.py
| |-- performance_tracker.py
|
|-- pipeline/
| |-- **init**.py
| |-- dataset_manager.py
| |-- drift_adapter.py
| |-- embedding_cache.py
| |-- evaluation.py
| |-- inference_engine.py
| |-- monitoring.py
| |-- representation_layer.py
| |-- research_metrics.py
| |-- retraining_orchestrator.py
| |-- retraining_pipeline.py
| |-- retrain_executor.py
| |-- state.py
| |-- training_orchestrator.py
| |-- xai_engine.py
|
|-- preprocessing/
| |-- **init**.py
| |-- image_preprocessor.py
| |-- preprocessing_planner.py
| |-- tabular_preprocessor.py
| |-- text_preprocessor.py
| |-- validator.py
|
|-- registry/
| |-- **init**.py
| |-- model_registry.py
|
|-- research/
| |-- ablation.py
| |-- experiment_collector.py
| |-- paper_generator.py
| |-- paper_service.py
| |-- plots.py
|
|-- tests/
| |-- system_validation.py
| |-- test_api_context_intelligence_contracts.py
| |-- test_api_model_registry_actions.py
| |-- test_calibration.py
| |-- test_candidate_selector_edge_cases.py
| |-- test_context_enforcer.py
| |-- test_embedding_cache_policy.py
| |-- test_experiment_collector_alignment.py
| |-- test_fix1_trial_intelligence_wiring.py
| |-- test_fusion_comprehensive.py
| |-- test_integration_e2e.py
| |-- test_integrator_field_tagging.py
| |-- test_monitoring_engine.py
| |-- test_paper_generator.py
| |-- test_performance_tracker.py
| |-- test_phase2_sessions.py
| |-- test_trainer_dummy_fill_fallback.py
| |-- test_xai_engine.py
|
|-- archive/ # historical scripts and snapshots
| |-- orchestrator_stub.py
| |-- validate_kaggle.py
| |-- verify_fix1_wiring.py
| |-- verify_fix1_wiring_clean.py
| |-- verify_fix4_integration.py
| |-- refactored_2026-04-04/
| | |-- api/
| | |-- database/
| | |-- pipeline/
|
|-- claude/ # assistant notes
|-- diary/ # architecture and audit notes
| |-- apex_architecture_analysis \_5_4_26*
| |-- repo_map.md
| |-- fixes/
|-- docs/ # archived docs/examples
| |-- archive/
| |-- examples/
|-- logs/ # runtime logs and meta-learning store
| |-- system_validation_meta.json
|
|-- **init**.py
|-- pytest.ini
|-- requirements.txt
|-- task_store.py
|-- tasks.db

Notes:

- Runtime and cache folders are environment-dependent and may be empty.
- Top-level audit/guide markdown files are intentionally omitted for brevity.
- Tooling folders such as .git, .venv, .pytest_cache, and **pycache** are intentionally omitted from this map.
- Model registry API contract now includes:
  - GET /model-registry
  - PATCH /model-registry/{model_id}/rename (alias-only; model_id and folder remain unchanged)
  - GET /model-registry/{model_id}/download (standard zip bundle with model_weights.pth + metadata)
- Decision-trace category mapping now treats drift_detection and drift_feedback stages as monitoring events.
- Embedding cache write-skip policy is modality-aware via metadata (not hashed cache key text matching).
- Root package create_app resolves app from api/run_api.py via .api.run_api import.
- Utility context/intelligence endpoints now enforce session-backed context contracts via require_context(..., require_session=True).
- Retired MultimodalPredictor is no longer advertised in root/model packages exports (tombstone module remains for explicit legacy imports).
- registry/model_registry.py is the active registry reader used by API routes; model_registry_pkg is retained as legacy compatibility surface.
- Added regression coverage for model-registry actions, v2 context intelligence contracts, and embedding-cache modality skip policy.
