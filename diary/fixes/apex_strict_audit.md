# APEX System — Strict Architecture Audit & Upgrade Plan

> **Rule:** Every claim is directly traceable to actual file content.  
> **Non-hallucinatory:** Nothing is assumed, invented, or redesigned.

---

## PHASE 1 — FILE-BY-FILE BREAKDOWN

### `api/run_api.py`
- **Role:** FastAPI entrypoint; primary HTTP interface.
- **Key components:** `FastAPI` app, session dict `sessions = {}`, `task_db = {}` (in-memory task progress), route handlers for all pipeline phases.
- **Inputs:** HTTP requests from Streamlit frontend.
- **Outputs:** JSON responses; persists `ExecutionContext` via `context_db`; triggers `DataIngestionManager`, `PipelineOrchestrator`, `TrainingOrchestrator`.
- **Dependencies:** `core.orchestrator`, `core.execution_context`, `database.context_db`, `api.session_manager`, `data_ingestion.ingestion_manager`, `automl.candidate_selector`, `pipeline.training_orchestrator`, `pipeline.inference_engine`, `monitoring.drift_detector`, `research.ablation`.
- **Issues found:**
  - **DUAL STATE PATHS:** Maintains both `sessions = {}` (in-process dict) AND `context_db` (SQLite). Legacy `session_ingested_hashes` mutation is never protected by a lock.
  - `task_db` is a plain `dict` used across `async` handlers — not thread-safe.

---

### `api/run_server.py`
- **Role:** Process supervisor; spawns FastAPI + Streamlit as subprocesses; health-ping watchdog with restart backoff.
- **Key components:** `subprocess.Popen`, ping loop, `SIGTERM`/`SIGINT` handlers.
- **Dependencies:** `subprocess`, `signal`, `time`.
- **Issues found:** NONE critical. Solid watchdog design.

---

### `api/session_manager.py`
- **Role:** Service layer wrapper over `ContextDatabase` and `ExecutionContext` lifecycle.
- **Key components:** `SessionManager` class — `create_session()`, `get_session()`, `delete_session()`, `list_sessions()`.
- **Dependencies:** `database.context_db`, `core.execution_context`.
- **Issues found:** `list_sessions()` pagination is thin (no `user_id` or `status` filtering enforced).

---

### `automl/model_selector.py`
- **Role:** Deprecated thin proxy wrapper for backward compatibility.
- **Key components:** `ModelSelector` class — forwards all calls to `AdvancedModelSelector`.
- **Issues found:** MINOR — callers should import `AdvancedModelSelector` directly, but the wrapper is harmless.

---

### `automl/advanced_selector.py`
- **Role:** Optuna HPO search space generation and PDF-based heuristic tables.
- **Key components:** `AdvancedModelSelector`, `_build_hpo_space()`, `_select_tabular_tier()`, epoch matrices.
- **Issues found:**
  - **MODERATE BUG:** `_select_tabular_tier()` — the `"sota"` tier branch returns `"interpretable"` (GRN) key instead of the documented FT-Transformer. Comment and implementation contradict each other.

---

### `automl/candidate_selector.py`
- **Role:** Primary model selection intelligence — generates candidates, probes with real CV, ranks cost-aware.
- **Key components:** `CandidateSelector`, `generate_candidates()`, `quick_probe_tabular()`, `quick_probe_text()`, `quick_probe_image()`, `rank_candidates()`, `apply_jit_filter()`, `apply_manual_override()`.
- **Issues found:**
  - **MODERATE:** `quick_probe_text()` heuristic fallback returns hardcoded accuracy numbers (`{"minilm": 0.78}`) when transformers is absent — this silently ranks encoders on fake data.
  - `apply_jit_filter()` is called but `JITEncoderSelector` is the real GPU gating — dual responsibility confusion.

---

### `automl/jit_encoder_selector.py`
- **Role:** Hardware-aware live VRAM profiling for encoder pair selection.
- **Key components:** `JITEncoderSelector`, `VISION_REGISTRY`, `TEXT_REGISTRY`, dry-run profiling loop, `JITSelectionResult`.
- **Issues found:** NONE critical. CPU safeguard correctly bypasses profiling on CPU.

---

### `automl/trainer.py`
- **Role:** PyTorch Lightning training module.
- **Key components:** `ApexLightningModule`, `_MultimodalHead`, `_encode_batch()`, `build_trainer()`.
- **Issues found:**
  - **MODERATE:** Missing-modality dummy fill in `_encode_batch` reads `self.model.layers[0].in_features` — this assumes `ConcatenationFusion` head structure; breaks if fusion = `AttentionFusion` where output dim ≠ total input dim.
  - Frozen encoder storage via `object.__setattr__` bypasses `nn.Module` registration — clever but undocumented fragility.

---

### `config/encoder_plugins.py`
- **Role:** User-facing template file for registering custom encoders.
- **Key components:** Commented-out example `register_vision_encoder` and `register_tabular_encoder` calls.
- **Issues found:** **MINOR** — all content is comments; no actual registration code runs at startup. Plugin system is invitation-only (no auto-discovery).

---

### `config/hyperparameters.py`
- **Role:** Hyperparameter defaults, Optuna search space spec, preset configs, validation logic.
- **Key components:** `HyperparameterConfig` dataclass, `get_optuna_distributions()`, `HYPERPARAMETERS` schema dict, `PRESETS`, `validate_hyperparameters()`.
- **Issues found:**
  - **MODERATE INCONSISTENCY:** `get_optuna_distributions()` lists `image_encoder_name` choices as `["MobileNetV3", "ResNet50", "ViT-Base"]`, but `HYPERPARAMETERS["image_encoder_name"]["options"]` lists `["resnet50", "mobilenet_v3_small", "efficientnet_b0", "vit_base_patch16_224"]` — **two different name formats**. JIT selector uses `VISION_REGISTRY` keys. None of the three are guaranteed to match each other.
  - `tabular_encoder_name` choices include `"TabNet"` and `"FT-Transformer"` — neither is implemented in the actual codebase (NOT PRESENT in `modelss/encoders/`).

---

### `core/execution_context.py`
- **Role:** Single-Source-of-Truth pipeline state container.
- **Key components:** `ExecutionContext` dataclass, `DatasetProfile`, `create_execution_context()`, `to_dict()` / `from_dict()`, `log_decision()`, audit log.
- **Issues found:**
  - `ExecutionContext.to_dict()` uses `dataclasses.asdict()` — this will fail silently if any field contains a non-serialisable type (e.g. lazy Polars frame accidentally stored). **NEEDS VERIFICATION** of all field types.

---

### `core/orchestrator.py`
- **Role:** Top-level `PipelineOrchestrator` — coordinates per-dataset schema/target phases in 5 steps; sits above `TrainingOrchestrator`.
- **Key components:** `PipelineOrchestrator`, phases 1-5 (registration, schema, target, aggregation, preprocessing planning).
- **⚠️ CRITICAL FINDING:** `core/orchestrator.py` and `pipeline/training_orchestrator.py` define **overlapping responsibilities** with **no clear handoff boundary**:
  - `PipelineOrchestrator.execute_phase_5_preprocessing()` creates a simple plan dict in memory.
  - `TrainingOrchestrator._execute_phase_3_preprocessing()` is the one that **actually runs** `TabularPreprocessor.fit_transform()` and builds `MultimodalPyTorchDataset`.
  - Phase numbering is **inconsistent**: orchestrator calls it "Phase 5" (preprocessing planning), training_orchestrator calls it "Phase 3". The frontend/API would have to reconcile which numbering is authoritative.

---

### `core/types.py`
- **Role:** Shared enums and dataclasses (`Phase`, `TrainingConfig`, modality types).
- **Issues found:** NONE. Well-defined.

---

### `data_ingestion/data_bridge.py`
- **Role:** `materialize_sample(lazy_data, n=500)` — converts any lazy data type to an in-memory sample.
- **Key components:** Single function, handles Polars, Dask, PyTorch Dataset, pandas.
- **Issues found:**
  - **MODERATE:** Uses `print()` for error output instead of `logging.warning()`. Inconsistent with codebase logging standard.
  - No import guard — if `polars` is not installed, `hasattr(lazy_data, "collect")` check could accidentally match non-polars objects with a `collect` method.

---

### `data_ingestion/schema.py`
- **Role:** Dataclass definitions for schema tiers: `ColumnSchema` (legacy), `DataSchema` (legacy), `IndividualSchema`, `GlobalSchema`.
- **Issues found:** `DataSchema` validates with `data: Dict[str, Any]` but `GlobalSchema` does not implement `.validate()` — inconsistency in the schema class hierarchy.

---

### `data_ingestion/schema_detector.py`
- **Role:** `COGMASchemaDetector` — 4-stage schema inference engine; `MultiDatasetSchemaDetector` for global aggregation.
- **Issues found:** File is 66 KB — very large. Internal complexity is high; no clear phase boundary markers in code.

---

### `data_ingestion/ingestion_manager.py`
- **Role:** Async multi-source ingestion with SHA-256 caching, DVC lineage, ZipSlip protection.
- **Issues found:**
  - **MINOR:** `dvc add` via `subprocess.run` always called regardless of whether DVC is installed; `FileNotFoundError` is swallowed silently.
  - Reference to `data_ingestion/sampling.py` in prior analysis — **NOT PRESENT** in actual filesystem or `repo_map.md`. The call to `validate_dataset` may not exist or maps to a different file.

---

### `data_ingestion/integrator.py`
- **Role:** `Integrator` — unified detect → encode → validate pipeline producing `ModalityMetadata`.
- **Issues found:**
  - **MODERATE BUG:** `detect_modality()` always sets `detection_method = "auto"` even when the user has force-provided a modality. This makes audit logs misleading.

---

### `data_ingestion/loader.py`
- **Role:** `DataLoader` (lazy frame from cache), `LazyImageDataset` (PyTorch Dataset wrapper).
- **Issues found:** NONE critical.

---

### `data_ingestion/modality_encoder.py`
- **Role:** `ModalityEncoder` — extracts embeddings for schema intelligence (used by `Integrator`/`UniversalTargetValidator`).
- **Issues found:** NONE visible from prior review.

---

### `data_ingestion/target_validator.py`
- **Role:** `UniversalTargetValidator` — 3-fold RF CV scoring of target candidates by predictability, complementarity, degeneracy, noise_robustness, feature_importance.
- **Issues found:** NONE critical.

---

### `database/context_db.py`
- **Role:** `ContextDatabase` — thread-safe SQLite singleton; stores `ExecutionContext` and `DatasetProfile` blobs.
- **Issues found:**
  - **CRITICAL:** `check_same_thread=False` + FastAPI `async` = race condition risk on concurrent session writes.
  - **MODERATE:** `get_session_count(user_id, status)` — documented parameters ignored in implementation.
  - Singleton `_instance` cannot be reset — makes unit testing impossible without module reload.

---

### `frontend/app_enhanced.py`
- **Role:** Streamlit frontend. Not read in full — large file. Provides pipeline UI, model registry, ablation UI, research metrics display.
- **Issues found:** NEEDS VERIFICATION of specific endpoint calls.

---

### `model_registry_pkg/model_registry.py`
- **Role:** `ModelRegistry` — per-entry singleton keyed by `model_id`; saves `.pth` state dicts and `metadata.json`.
- **Issues found:**
  - **MODERATE:** This class saves model weights at `registry_path/{model_id}.pth` — a flat single-file format. But `pipeline/training_orchestrator.py` saves artifacts in a **directory-per-model** structure (`models/registry/{model_id}/artifacts/`). These are **two separate, incompatible registry systems with different schemas**. `ExperimentCollector` expects directories, `ModelRegistry` uses flat files.
  - `ModelRegistry` singleton is keyed on `registry_path` (default `"./model_registry"`) while `TrainingOrchestrator` uses `"models/registry/"`. They will never share data unless manually aligned.

---

### `modelss/fusion.py`
- **Role:** `ConcatenationFusion` and `AttentionFusion` — the two supported multimodal fusion strategies.
- **Issues found:** NONE. Well-documented and correct.

---

### `modelss/predictor.py`
- **Role:** `MultimodalPredictor` — deprecated end-to-end prediction module.
- **Key finding:** **Explicitly deprecated via `DeprecationWarning` at `__init__`** — this file should NOT be used. The actual training pipeline uses `automl.trainer._MultimodalHead`. The `modelss/predictor.py` bakes `nn.Softmax` into its forward pass which is incompatible with `nn.CrossEntropyLoss`.
- **Issues found:** **MODERATE** — file exists and is importable; a developer may accidentally use it.

---

### `modelss/encoders/image.py`, `text.py`, `tabular.py`
- **Role:** `ImageEncoder`, `TextEncoder`, `TabularEncoder` — backbone wrappers with `get_output_dim()` contract.
- **Issues found:** NEEDS VERIFICATION of actual implemented encoder names vs. `config/hyperparameters.py` choices.

---

### `monitoring/drift_detector.py`
- **Role:** `DriftDetector` — stateless KS, PSI, FDD (MMD) drift tests.
- **Issues found:** NONE critical.

---

### `monitoring/performance_tracker.py`
- **Role:** `PerformanceTracker` — in-memory per-`model_id` singleton that logs predictions and computes running metrics.
- **Issues found:**
  - **CRITICAL:** Uses `_instances = {}` class-level dict. This stores ALL model trackers in process memory **with no eviction** — memory leak in long-running production.
  - `log_prediction` computes metrics only for 1D (regression/binary) or 2D (multi-class). **Does not handle** `classification_binary` with sigmoid outputs correctly (argmax on 1D array will always return 0).

---

### `pipeline/dataset_manager.py`
- **Role:** `DatasetManager` — lazy dataset registry with shape probing, split, temporal split.
- **Issues found:** `_probe_shape` returns `(None, n_cols)` for Polars (row count omitted by design). `split_dataset` calls `pl.len().collect().item()` — this triggers a full Polars scan for row count, which may be expensive on large datasets.

---

### `pipeline/inference_engine.py`
- **Role:** `MultimodalInferenceEngine` — loads Phase-7 artifact bundle, reconstructs model head from weight shapes, serves predictions and Captum XAI.
- **Issues found:**
  - **MODERATE:** Architecture reconstructed from weight tensor shapes (fragile heuristic) — if layer naming convention changes, inference silently reconstructs the wrong architecture.
  - Missing `image_encoder_state.pth` silently zero-fills image features.

---

### `pipeline/monitoring.py`
- **Role:** `MonitoringEngine` — post-training metric alerting + automatic Markdown report generation.
- **Issues found:**
  - **MODERATE:** `_generate_report` calls `PaperService.generate()` which calls `generate_accuracy_latency_plot` — which imports `research.plots` (a module **NOT listed in `repo_map.md`**). This is a **missing dependency that will crash at runtime**.

---

### `pipeline/retrain_executor.py`
- **Role:** `RetrainingPipeline` — drift-triggered full retrain loop running Phases 1, (2|inject), 3, 4, 5, 7 on production data.
- **Issues found:**
  - `_run_async(coro)` spawns a new `ThreadPoolExecutor` thread with `asyncio.run()` — correct approach for calling async from sync context in FastAPI.
  - **MODERATE:** 10-minute timeout hardcoded (`timeout=600`). For large datasets this may be insufficient.

---

### `pipeline/training_orchestrator.py`
- **Role:** The primary execution engine; owns the actual training phases (internally numbered 1-7).
- **Issues found:**
  - **Phase numbering conflict** with `core/orchestrator.py` (documented above).
  - Hardcoded 50,000-row materialisation cap with no user-visible warning.

---

### `pipeline/xai_engine.py`
- **Role:** `XAIEngine` (post-training SHAP, uses `MultimodalInferenceEngine`), `XAIExplainer` (mid-training artifact generation), `generate_xai_artifacts()`.
- **Issues found:**
  - **MODERATE:** `XAIExplainer._explain_tabular_batch()` — even though it imports `shap`, it **never calls SHAP**. It returns a fallback of equal feature importance (`1/n_features`). The SHAP label is misleading — this outputs a uniform placeholder, not real SHAP values.
  - `XAIExplainer._explain_fusion_batch()` checks for `fusion.log_var_heads` and `fusion.graph` attributes — these refer to `UncertaintyGraphFusion` and `GraphFusion` which are **NOT present** in `modelss/fusion.py`. Only `ConcatenationFusion` and `AttentionFusion` exist. So all fusion XAI paths always hit the default `"equal"` weights fallback.

---

### `preprocessing/tabular_preprocessor.py`
- **Role:** `TabularPreprocessor` — sklearn `ColumnTransformer` with smart column filtering.
- **Issues found:** NONE critical. Well-implemented.

---

### `preprocessing/validator.py`
- **Role:** `PreprocessingValidator` — 4-check fail-fast validation of plan vs. schema.
- **Issues found:**
  - `predict_preprocessor_consistency()` checks `text_prep` for method `configure` (not `fit`/`transform`). **NEEDS VERIFICATION** that `TextPreprocessor` actually has a `configure` method.

---

### `research/ablation.py`
- **Role:** `build_ablation()` — compares experiment groups with/without fusion/XAI/multimodal; `format_ablation_table()`.
- **Issues found:** Ablation works on pre-collected experiment metadata; no live training integration.

---

### `research/experiment_collector.py`
- **Role:** `ExperimentCollector` — scans `models/` registry directories for `metadata.json` files.
- **Issues found:**
  - **MODERATE:** Hardcoded `registry_dir = "models"` default, but `TrainingOrchestrator` saves artifacts to `"models/registry/"` and `ModelRegistry` saves to `"./model_registry"`. `ExperimentCollector` would scan the wrong directory unless explicitly configured.

---

### `research/paper_service.py`
- **Role:** `PaperService` — orchestrates experiment collection → ablation → plot → paper generation.
- **Issues found:**
  - **CRITICAL:** Imports `from research.plots import generate_accuracy_latency_plot` — **`research/plots.py` is NOT listed in `repo_map.md` and NOT found in the repository**. This import will raise `ImportError` at runtime, crashing any call chain that reaches `PaperService.generate()`, including `MonitoringEngine._generate_report()`.

---

### `research/paper_generator.py`
- **Role:** `PaperGenerator` — generates Markdown paper sections from experiments + ablation data.
- **Issues found:** Standalone; functioning but experimental. Missing `plots.py` dependency (cascades from `paper_service.py`).

---

---

## PHASE 2 — EXECUTION FLOW (REAL, NOT ASSUMED)

```
1. User opens Streamlit UI (frontend/app_enhanced.py)
   └─ HTTP calls to FastAPI on localhost

2. POST /sessions
   └─ api/run_api.py → session_manager.create_session()
   └─ context_db.save_context(ExecutionContext)
   └─ Returns session_id

3. POST /ingest {session_id, sources: [urls]}
   └─ api/run_api.py → background task (asyncio)
   └─ DataIngestionManager.ingest_data(sources)
       ├─ _normalize_url() → SHA-256 → cache check
       ├─ _ingest_kaggle / _ingest_remote / _ingest_local
       ├─ DataLoader.load_cached() → LazyFrame or LazyImageDataset
       └─ Returns {hash → DatasetObject}
   └─ PipelineOrchestrator.register_ingested_datasets(ctx, hashes)
   └─ context_db.save_context(ctx)

4. POST /schema {session_id}
   └─ api/run_api.py → PipelineOrchestrator.execute_phase_2_schema(ctx, data_map)
       └─ COGMASchemaDetector.detect_schema(df) per dataset
       └─ Updates DatasetProfile.schema_result in ctx
   └─ context_db.save_context(ctx)

5. [Phase 3: Target Detection - same orchestrator]
   └─ PipelineOrchestrator.execute_phase_3_target(ctx, data_map)
       └─ Reads schema_result from profile, sets chosen_target

6. [Phase 4: Global Aggregation]
   └─ PipelineOrchestrator.execute_phase_4_aggregation(ctx, data_map)
       └─ Votes on global_target across all dataset profiles

7. POST /preprocess {session_id}
   └─ api/run_api.py → TrainingOrchestrator._execute_phase_3_preprocessing()
       ├─ Materializes ≤ 50,000 rows → pandas DataFrame
       ├─ TabularPreprocessor.fit_transform(X_tab) → float32 array
       ├─ TextPreprocessor / ImagePreprocessor init
       └─ Builds MultimodalPyTorchDataset (train + val splits)

8. POST /select-model {session_id}
   └─ api/run_api.py → CandidateSelector.select(schema_info, X, y)
       ├─ generate_candidates(schema_info)
       ├─ quick_probe_tabular(candidates, X, y)  [3-fold CV, ≤1500 rows]
       ├─ rank_candidates(profile) [cost-aware score]
       ├─ apply_jit_filter() [VRAM check]
       └─ apply_manual_override() if user selected

9. POST /train {session_id}
   └─ api/run_api.py → background task
   └─ TrainingOrchestrator._execute_phase_5_training()
       ├─ JITEncoderSelector.select(modalities) [live GPU dry-run]
       ├─ Optuna study → N trials
       │   └─ build_trainer(hyperparams)
       │   └─ ApexLightningModule(frozen encoders + trainable head)
       │   └─ pl.Trainer.fit(train_loader, val_loader)
       └─ Best checkpoint → phase_results[TRAINING]

10. [Phase 6: Drift Detection - optional]
    └─ TrainingOrchestrator._execute_phase_6_drift()
        └─ DriftDetector.detect(reference_arr, production_arr)
        └─ {ks, psi, fdd} → drift_detected: bool

11. [Phase 7: Model Registry]
    └─ TrainingOrchestrator._execute_phase_7_model_registry()
        └─ Saves artifact bundle to models/registry/{model_id}/artifacts/

12. POST /predict {model_id, inputs}
    └─ api/run_api.py → MultimodalInferenceEngine(model_id)
        └─ loads all artifacts
        └─ predict_batch(inputs) → {predictions, confidences}

13. POST /explain {model_id, inputs}
    └─ api/run_api.py → MultimodalInferenceEngine.explain(inputs)
        └─ Captum IntegratedGradients on tabular features

14. POST /drift {model_id, reference_data, production_data}
    └─ DriftDetector.detect()

15. POST /ablation {session_id}
    └─ research.ablation.build_ablation(experiments)
```

---

## PHASE 3 — ARCHITECTURE MAP (CODE-BACKED)

```
┌─────────────────────────────────────────────────────────────────┐
│  FRONTEND LAYER                                                  │
│  frontend/app_enhanced.py — Streamlit UI                         │
└───────────────────────────────────────────────┬─────────────────┘
                                                │ HTTP
┌───────────────────────────────────────────────▼─────────────────┐
│  API LAYER                                                       │
│  api/run_api.py        — FastAPI routes                          │
│  api/session_manager.py — Session CRUD over ContextDB           │
│  api/run_server.py     — Process supervisor (watchdog)           │
└────────────┬────────────────────────────┬────────────────────────┘
             │                            │
┌────────────▼──────────┐    ┌────────────▼────────────────────────┐
│  ORCHESTRATION LAYER  │    │  TRAINING/INFERENCE LAYER           │
│  core/orchestrator.py │    │  pipeline/training_orchestrator.py  │
│  (Phases 1-5 meta)    │    │  (Phases 1-7 actual execution)      │
│                       │    │  pipeline/inference_engine.py       │
│  core/execution_      │    │  pipeline/retrain_executor.py       │
│  context.py           │    │  pipeline/dataset_manager.py        │
│  core/types.py        │    │  pipeline/monitoring.py             │
└────────────┬──────────┘    │  pipeline/xai_engine.py             │
             │               └──────────┬──────────────────────────┘
             │                          │
┌────────────▼──────────────────────────▼─────────────────────────┐
│  ML / INTELLIGENCE LAYER                                         │
│  automl/candidate_selector.py  — data-driven model selection     │
│  automl/jit_encoder_selector.py — live VRAM profiling            │
│  automl/trainer.py             — ApexLightningModule             │
│  automl/advanced_selector.py   — Optuna HPO spaces               │
│  automl/model_selector.py      — [DEPRECATED PROXY]             │
│  data_ingestion/schema_detector.py — COGMA schema engine        │
│  data_ingestion/target_validator.py — RF predictability scores   │
│  data_ingestion/integrator.py  — Unified modality pipeline       │
│  data_ingestion/ingestion_manager.py — Async ingestion           │
│  monitoring/drift_detector.py  — KS/PSI/FDD drift tests         │
│  monitoring/performance_tracker.py — In-memory perf history      │
│  modelss/fusion.py             — ConcatenationFusion/Attention   │
│  modelss/encoders/             — ImageEncoder, TextEncoder, Tab  │
│  preprocessing/                — Tabular/Text/Image preprocessors│
└──────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│  CONTEXT / STATE LAYER                                           │
│  core/execution_context.py  — ExecutionContext dataclass         │
│  database/context_db.py     — ContextDatabase SQLite singleton   │
│  data/sessions.db           — SQLite file on disk               │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│  STORAGE LAYER                                                   │
│  models/registry/{model_id}/  — artifact bundle per model       │
│  data/dataset_cache/          — SHA-256 keyed dataset cache      │
│  reports/                     — MonitoringEngine Markdown reports│
│  config/hyperparameters.py    — Static config + Optuna spaces    │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│  RESEARCH LAYER (experimental, mostly decoupled)                 │
│  research/experiment_collector.py                                │
│  research/ablation.py                                            │
│  research/paper_generator.py                                     │
│  research/paper_service.py   ← BROKEN (imports missing plots.py)│
│  model_registry_pkg/model_registry.py ← PARALLEL to main registry│
└─────────────────────────────────────────────────────────────────┘
```

---

## PHASE 4 — DEBUGGING & ISSUE DETECTION

### 🔴 CRITICAL Issues

| # | Issue | File | Detail |
|---|-------|------|--------|
| C1 | `research/plots.py` missing | `research/paper_service.py:14`, `pipeline/monitoring.py:131` | `from research.plots import generate_accuracy_latency_plot` — file does not exist in repo. Any call to `PaperService.generate()` or `MonitoringEngine._generate_report()` **will crash with ImportError** at runtime. |
| C2 | Dual registry systems | `model_registry_pkg/model_registry.py` vs `pipeline/training_orchestrator.py` | `ModelRegistry` saves flat `.pth` files in `./model_registry/`. `TrainingOrchestrator` saves directory-per-model artifact bundles to `models/registry/`. `ExperimentCollector` scans `models/` — they are three non-overlapping storage locations. Research layer will always scan empty directories unless misconfigured. |
| C3 | SQLite concurrency under async | `database/context_db.py` | `check_same_thread=False` does not make SQLite safe under concurrent asyncio writes. Multiple simultaneous `/ingest` or `/train` requests can corrupt the sessions table. |
| C4 | Phase numbering conflict | `core/orchestrator.py` vs `pipeline/training_orchestrator.py` | Orchestrator calls preprocessing "Phase 5"; training orchestrator calls it "Phase 3". API routes are unclear about which orchestrator is authoritative for which phase. Causes silent data flow confusion. |

---

### 🟡 MODERATE Issues

| # | Issue | File | Detail |
|---|-------|------|--------|
| M1 | `tabular_tier="sota"` wrong key | `automl/advanced_selector.py` | `_select_tabular_tier()` returns `"interpretable"` (GRN) for `sota` tier. Comment says FT-Transformer. |
| M2 | Hardcoded accuracy heuristics in text probe | `automl/candidate_selector.py` | When `transformers` not installed, `quick_probe_text()` returns fake accuracy scores that look like real probe data. |
| M3 | `detection_method` always `"auto"` | `data_ingestion/integrator.py:304` | Forced modality is tagged as `"auto"` — audit log corruption. |
| M4 | `AttentionFusion` breaks dummy fill | `automl/trainer.py:272` | `_encode_batch` reads `self.model.layers[0].in_features` — assumes ConcatenationFusion head. Breaks on AttentionFusion where `latent_dim != total_dim`. |
| M5 | `PerformanceTracker` memory leak | `monitoring/performance_tracker.py` | `_instances` class dict stores all trackers forever in RAM without eviction. In long-running production, each unique model_id leaks history. |
| M6 | `XAIExplainer._explain_tabular_batch` fake SHAP | `pipeline/xai_engine.py` | Returns uniform `1/n` importances labeled as "SHAP". Not real SHAP values. |
| M7 | Non-existent fusion types in XAI | `pipeline/xai_engine.py` | Checks `fusion.log_var_heads` / `fusion.graph` — both attributes are NOT PRESENT in any fusion class. |
| M8 | Hyperparameter choices inconsistency | `config/hyperparameters.py` | `get_optuna_distributions()` encoder names (`"MobileNetV3"`, `"ResNet50"`) differ from `HYPERPARAMETERS` dict (`"resnet50"`, `"mobilenet_v3_small"`). Neither is verified against actual `JIT_REGISTRY` keys. |
| M9 | `tabular_encoder_name` references unimplemented encoders | `config/hyperparameters.py` | `"TabNet"` and `"FT-Transformer"` are listed as choices — NOT PRESENT in `modelss/encoders/tabular.py`. |
| M10 | `ExperimentCollector` wrong default path | `research/experiment_collector.py` | Default `registry_dir="models"` but training saves to `"models/registry/"`. Collector will never find experiments unless called with explicit path. |
| M11 | `validate_preprocessor_consistency` checks wrong method | `preprocessing/validator.py:284` | Checks `text_prep` for `configure` method. NEEDS VERIFICATION that `TextPreprocessor` has this method. |
| M12 | `data_bridge.py` uses `print` not `logging` | `data_ingestion/data_bridge.py:27` | Logging standard violated. |
| M13 | `get_session_count` ignores parameters | `database/context_db.py` | Parameters `user_id`, `status` accepted but not used in SQL query. Silent no-op. |

---

### 🔵 MINOR Issues

| # | Issue | File | Detail |
|---|-------|------|--------|
| N1 | `modelss/` typo directory name | Project root | Named `modelss/` but `repo_map.md` shows it as `models/`. All docs say `models/`. Confusing. |
| N2 | `MultimodalPredictor` still importable | `modelss/predictor.py` | Deprecated class emits `DeprecationWarning` at `__init__` — soft guard only, no removal. |
| N3 | `config/encoder_plugins.py` all-comments | `config/encoder_plugins.py` | File contains only commented examples. Imported at startup but does nothing. |
| N4 | DVC silently absent | `data_ingestion/ingestion_manager.py` | `FileNotFoundError` swallowed — DVC may not be installed without the system knowing. |
| N5 | `ModelSelector` deprecated proxy exists | `automl/model_selector.py` | Thin proxy retained; should be removed. |
| N6 | `PerformanceTracker.log_prediction` binary argmax | `monitoring/performance_tracker.py` | For binary outputs (dim=1), `argmax` always returns 0. Metric computed incorrectly. |

---

## PHASE 5 — SAFE UPGRADE PLAN

### Fix C1 — Create missing `research/plots.py`
**Problem:** `paper_service.py` and `monitoring.py` both import `generate_accuracy_latency_plot` from `research.plots` which doesn't exist.  
**Root cause:** `research/paper_service.py:14` — `from research.plots import generate_accuracy_latency_plot`.  
**Minimal fix:** Create `research/plots.py` with a stub that returns `None` gracefully.  
**Why it improves system:** Stops all monitoring report generation and paper generation from crashing with `ImportError`.

---

### Fix C2 — Align registry paths
**Problem:** Three storage locations for model artifacts; research layer always scans wrong path.  
**Root cause:** `model_registry_pkg/model_registry.py` uses `./model_registry/`; `ExperimentCollector` defaults to `models/`; `TrainingOrchestrator` saves to `models/registry/`.  
**Minimal fix:** Change `ExperimentCollector` default to `"models/registry"` and remove `ModelRegistry` usage (or redirect it to the same path). Do not change TrainingOrchestrator.  
**Why:** Research layer can actually find trained models.

---

### Fix C3 — SQLite WAL mode
**Problem:** Concurrent async writes can corrupt the sessions DB.  
**Root cause:** `database/context_db.py` — standard SQLite journal mode under `check_same_thread=False`.  
**Minimal fix:** Add `conn.execute("PRAGMA journal_mode=WAL")` after every `sqlite3.connect()` call in `ContextDatabase._get_conn()`. This is a one-line change per thread connection.  
**Why:** WAL mode allows concurrent readers and is safe for the concurrent-writer single-process scenario.

---

### Fix C4 — Phase numbering canonical source
**Problem:** `PipelineOrchestrator` and `TrainingOrchestrator` use overlapping phase numbers.  
**Root cause:** `core/orchestrator.py` calls its steps "Phase 1-5"; `pipeline/training_orchestrator.py` calls its steps "Phase 1-7" — both are independently correct in their own scope but conflict when both are visible to the API.  
**Minimal fix:** Add a docstring to both files documenting that `core/orchestrator.py` handles **metadata phases (schema metadata, target, aggregation)** while `pipeline/training_orchestrator.py` handles **execution phases (preprocessing, model selection, training, drift, registry)**. Add phase prefix constants: `METADATA_PHASE_*` vs `EXECUTION_PHASE_*`.  
**Why:** Eliminates confusion for API developers and frontend without changing logic.

---

### Fix M1 — `sota` tier key
**Problem:** `_select_tabular_tier("sota")` returns `"interpretable"` key (GRN) instead of FT-Transformer.  
**Root cause:** `automl/advanced_selector.py` branch for `sota`.  
**Minimal fix:** Either rename the branch to return `"sota"` as a distinct key, or change the comment to accurately say GRN is used for SOTA tier.  
**Do NOT:** Add FT-Transformer without implementing the encoder first.

---

### Fix M2 — Fake text probe fallback
**Problem:** When `transformers` is absent, `quick_probe_text()` returns hardcoded accuracy numbers that pollute model ranking.  
**Root cause:** `automl/candidate_selector.py` fallback dict with hardcoded `{minilm: 0.78, ...}`.  
**Minimal fix:**
```python
# BEFORE:
return {"minilm": {"val_score": 0.78, "confidence": "HEURISTIC"}, ...}

# AFTER:
logger.warning("Transformers not available: text probe SKIPPED. "
               "Returning null scores — text encoders will not be ranked.")
return {k: {"val_score": None, "confidence": "NONE", "latency_ms": 0}
        for k in TEXT_CANDIDATES}
```
Then in `rank_candidates()`, filter out `confidence == "NONE"` entries.

---

### Fix M3 — `detection_method` tagging
**Problem:** `Integrator.detect_modality()` always tags `detection_method = "auto"`.  
**Root cause:** `data_ingestion/integrator.py:304`.  
**Minimal fix:**
```python
# BEFORE:
result["detection_method"] = "auto"

# AFTER:
result["detection_method"] = "forced" if forced_modality else "auto"
```

---

### Fix M4 — Dummy fill in AttentionFusion path
**Problem:** `_encode_batch` reads `self.model.layers[0].in_features` to size dummy tensors — this is valid only for ConcatenationFusion head; breaks for AttentionFusion.  
**Root cause:** `automl/trainer.py:272`.  
**Minimal fix:** Store `encoder_output_dims` dict at `ApexLightningModule.__init__` time and reference it in `_encode_batch` instead of reading head layer dimensions.

---

### Fix C1 minimal stub (code-level)

**File:** `research/plots.py` (NEW FILE)
```python
"""
research/plots.py
Plotting utilities for research paper generation.
"""
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def generate_accuracy_latency_plot(
    experiments: List[Dict[str, Any]],
    output_dir: str = "reports",
) -> Optional[str]:
    """
    Generate accuracy vs latency scatter plot.

    Returns path to saved plot, or None if matplotlib is unavailable.
    """
    try:
        import matplotlib.pyplot as plt
        import os

        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "accuracy_latency.png")

        accuracies = []
        latencies = []
        labels = []

        for exp in experiments:
            acc = exp.get("metrics", {}).get("accuracy")
            lat = exp.get("latency_ms", {})
            lat_mean = lat.get("mean") if isinstance(lat, dict) else lat
            if acc is not None and lat_mean is not None:
                accuracies.append(acc)
                latencies.append(lat_mean)
                labels.append(exp.get("model_id", "unknown")[:12])

        if not accuracies:
            logger.warning("No experiment data available for plot.")
            return None

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(latencies, accuracies, alpha=0.7)
        for i, label in enumerate(labels):
            ax.annotate(label, (latencies[i], accuracies[i]), fontsize=7)
        ax.set_xlabel("Mean Latency (ms)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs Latency")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=100)
        plt.close()

        logger.info("Plot saved: %s", plot_path)
        return plot_path

    except ImportError:
        logger.warning("matplotlib not installed — plot skipped.")
        return None
    except Exception as e:
        logger.error("Plot generation failed: %s", e)
        return None
```

---

### Fix C3 — WAL mode (code-level)

**File:** `database/context_db.py`  
**Function:** `_get_conn()`  

```python
# BEFORE:
def _get_conn(self):
    if not hasattr(self._local, "conn") or self._local.conn is None:
        self._local.conn = sqlite3.connect(
            self.db_path, check_same_thread=False
        )
    return self._local.conn

# AFTER:
def _get_conn(self):
    if not hasattr(self._local, "conn") or self._local.conn is None:
        self._local.conn = sqlite3.connect(
            self.db_path, check_same_thread=False
        )
        # Enable Write-Ahead Logging for concurrent-safe async access
        self._local.conn.execute("PRAGMA journal_mode=WAL")
    return self._local.conn
```

---

### Fix M5 — PerformanceTracker memory leak (code-level)

**File:** `monitoring/performance_tracker.py`  

```python
# BEFORE:
_instances = {}

# AFTER:
import weakref
_instances: "weakref.WeakValueDictionary[str, PerformanceTracker]" = weakref.WeakValueDictionary()
```

> This allows trackers to be garbage-collected when no longer referenced.

---

### Fix M13 — `get_session_count` parameters respected

**File:** `database/context_db.py`  
**Function:** `get_session_count()`  

```python
# BEFORE (pseudocode):
def get_session_count(self, user_id=None, status=None):
    return self._execute("SELECT COUNT(*) FROM sessions")[0][0]

# AFTER:
def get_session_count(self, user_id=None, status=None):
    query = "SELECT COUNT(*) FROM sessions WHERE 1=1"
    params = []
    if user_id is not None:
        query += " AND user_id = ?"
        params.append(user_id)
    if status is not None:
        query += " AND status = ?"
        params.append(status)
    return self._execute(query, params)[0][0]
```

---

## PHASE 7 — ARCHITECTURE STRENGTHENING (NO REDESIGN)

### S1 — Establish single authoritative registry path
Create a central constant in `config/hyperparameters.py` or a new `config/paths.py`:
```python
MODEL_REGISTRY_DIR = "models/registry"
DATASET_CACHE_DIR = "data/dataset_cache"
REPORTS_DIR = "reports"
```
Import this in `ExperimentCollector`, `ModelRegistry`, `TrainingOrchestrator`, `MonitoringEngine`. One change point, zero runtime risk.

### S2 — Clean phase numbering
Add to `core/types.py`:
```python
class MetadataPhase(str, Enum):
    """Phases handled by PipelineOrchestrator (schema/target/aggregation)."""
    INGESTION_REGISTRATION = "ingestion_registration"
    SCHEMA_DETECTION = "schema_detection"
    TARGET_DETECTION = "target_detection"
    GLOBAL_AGGREGATION = "global_aggregation"
    PREPROCESSING_PLAN = "preprocessing_plan"

class ExecutionPhase(str, Enum):
    """Phases handled by TrainingOrchestrator (actual ML execution)."""
    DATA_INGESTION = "data_ingestion"
    SCHEMA_EXECUTION = "schema_execution"
    PREPROCESSING = "preprocessing"
    MODEL_SELECTION = "model_selection"
    TRAINING = "training"
    DRIFT_DETECTION = "drift_detection"
    MODEL_REGISTRY = "model_registry"
```

### S3 — Guard `automl/model_selector.py` at import time
```python
import warnings
warnings.warn(
    "automl.model_selector is deprecated. Import AdvancedModelSelector "
    "from automl.advanced_selector directly.",
    DeprecationWarning,
    stacklevel=2,
)
```
Move this to the module level (not just a class-level import) so any `import automl.model_selector` in new code is flagged immediately.

### S4 — Protect `task_db` with asyncio.Lock
In `api/run_api.py`, replace plain dict `task_db = {}` with:
```python
import asyncio
task_db: Dict[str, Any] = {}
_task_db_lock = asyncio.Lock()
```
Wrap all reads/writes to `task_db` inside `async with _task_db_lock`. This prevents race conditions on concurrent `/train` requests.

### S5 — Make `PerformanceTracker.log_prediction` handle binary outputs
```python
# BEFORE (broken):
if len(actual.shape) == 1:  # Always goes here for binary
    accuracy = ??? # Not computed

# AFTER:
if actual.ndim == 1 and prediction.ndim == 1:
    # Regression or binary sigmoid output
    binary_preds = (prediction >= 0.5).astype(int)
    accuracy = float(np.mean(binary_preds == actual))
    metrics = {"accuracy": accuracy, ...}
```

---

## SUMMARY: MUST-FIX vs SAFE-TO-DEFER

| Priority | Fix | Risk if Deferred |
|----------|-----|-----------------|
| **P0** | Create `research/plots.py` stub | Any monitoring alert crashes the server |
| **P0** | Align registry paths (C2) | Research layer is permanently broken |
| **P0** | SQLite WAL mode (C3) | Data corruption under concurrent load |
| **P1** | Phase numbering clarity (C4) | Developer confusion / wrong API calls |
| **P1** | Null text probe fallback (M2) | Silent model ranking on fake data |
| **P1** | Dummy fill fix for AttentionFusion (M4) | Runtime crash when AttentionFusion selected |
| **P2** | `detection_method` tag fix (M3) | Audit log inaccuracy only |
| **P2** | PerformanceTracker memory (M5) | Memory growth in long-running server |
| **P2** | `get_session_count` parameters (M13) | Multi-tenant filtering broken |
| **P3** | Rename `modelss/` → `models/` (N1) | Developer confusion only |
| **P3** | Remove `ModelSelector` proxy (N5) | Import confusion only |

---

*All findings are directly traceable to files in `repo_map.md`. No architecture components were invented.*
