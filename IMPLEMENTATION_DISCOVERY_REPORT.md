# AutoVision+ Implementation Discovery Report

**Date:** 2026-03-12
**Scope:** Full implementation audit of backend, frontend, and pipeline modules
**Method:** Source code analysis across all modules, endpoint tracing, data flow mapping

---

## 1. Pipeline Implementation Table

| # | Pipeline Stage | Backend Status | Frontend Status | Files Involved |
|---|---|---|---|---|
| 1 | Multi-Dataset Ingestion | **Implemented** | **Implemented** | `data_ingestion/ingestion_manager.py`, `data_ingestion/loader.py`, `run_api.py`, `frontend/app_enhanced.py` |
| 2 | Multi-Dataset Schema Detection | **Implemented** | **Implemented** | `data_ingestion/schema_detector.py`, `data_ingestion/schema.py`, `run_api.py`, `frontend/app_enhanced.py` |
| 3 | Preprocessing Pipeline | **Implemented** | **Implemented** | `preprocessing/tabular_preprocessor.py`, `preprocessing/text_preprocessor.py`, `preprocessing/image_preprocessor.py`, `run_api.py`, `frontend/app_enhanced.py` |
| 4 | Encoder Selection | **Implemented** | **Implemented** | `automl/jit_encoder_selector.py`, `automl/advanced_selector.py`, `automl/model_selector.py`, `run_api.py`, `frontend/app_enhanced.py` |
| 5 | Training Behavior | **Implemented** | **Implemented** | `pipeline/training_orchestrator.py`, `automl/trainer.py`, `run_api.py`, `frontend/app_enhanced.py` |
| 6 | Multimodal Encoding & Fusion | **Implemented** | **Partial** | `modelss/encoders/image.py`, `modelss/encoders/text.py`, `modelss/encoders/tabular.py`, `modelss/fusion.py`, `automl/trainer.py` |
| 7 | Drift Detection & Monitoring | **Implemented** | **Implemented** | `monitoring/drift_detector.py`, `monitoring/performance_tracker.py`, `run_api.py`, `pipeline/training_orchestrator.py`, `frontend/app_enhanced.py` |
| 8 | Model Registry & Management | **Partial** | **Partial** | `model_registry_pkg/model_registry.py`, `pipeline/training_orchestrator.py`, `run_api.py`, `frontend/app_enhanced.py` |
| 9 | Prediction Pipeline | **Implemented** | **Implemented** | `pipeline/inference_engine.py`, `run_api.py`, `frontend/app_enhanced.py` |

---

## 2. Actual System Pipeline (Real Data Flow)

This section describes the **real implemented pipeline** as traced through source code, not the documented pipeline.

```
1. User enters dataset URLs in Streamlit text_area
      |
   POST /ingest/datasets {dataset_urls, session_id}
      |
2. DataIngestionManager.ingest_data()
   - SHA-256 normalization + cache check (16-char truncated digest)
   - On cache miss: aiohttp download / kaggle CLI / shutil.copy2
   - Atomic metadata persistence (mkstemp + os.replace)
   - Returns Polars LazyFrame / Dask DataFrame / LazyImageDataset
   - Registered in DatasetManager (lazy, no materialization)
      |
3. POST /detect-schema {}
   - MultiDatasetSchemaDetector.detect_global_schema()
   - Materializes 500-row sample per dataset
   - Per-column classification: image/text/timeseries/tabular
   - 4-signal target detection + problem type inference
   - Cross-dataset relatedness via union-find (Jaccard threshold 0.5)
   - Returns GlobalSchema (problem_type, modalities, target, confidence)
      |
4. POST /preprocess {}
   - Materializes all datasets into single pandas DataFrame (cap: 50K rows)
   - Drops >50% NaN columns
   - Target separation + encoding (LabelEncoder / StandardScaler / multilabel)
   - Fits TabularPreprocessor (ID/date filtering, ColumnTransformer)
   - Initializes TextPreprocessor (lazy tokenizer) + ImagePreprocessor
   - Builds train/val MultimodalPyTorchDatasets (augmented vs clean)
      |
5. POST /select-model {dataset_size, modalities, problem_type}
   - AdvancedModelSelector.select_models()
   - Tier selection: image/text/tabular encoders from catalogs
   - GPU memory probing for batch size
   - HPO search space generation from PDF epoch/LR/dropout bounds
      |
6. POST /train-pipeline {problem_type, modalities, hp_overrides?}
   - Background task executes 7 phases sequentially:

   Phase 1: Re-register cached datasets
   Phase 2: Schema detection
   Phase 3: Preprocessing (fits transformers)
   Phase 4: AdvancedModelSelector + JIT Encoder Selection
      - JITEncoderSelector.select(): VRAM budget = 0.85 * free
      - Cartesian product search (3x3=9 combos), sorted by capacity desc
      - Dummy-forward profiling per combo, first feasible wins
   Phase 5: Optuna HPO Study
      - Pre-computes frozen text embeddings (train + val)
      - Pre-computes frozen image embeddings (val only)
      - Per trial: fresh GRN/MLP tabular encoder, shared frozen img/text
      - PyTorch Lightning trainer: FP16, EarlyStopping(patience=5),
        CosineAnnealingLR, HyperbandPruner, Windows TDR sync
      - GPU cleanup in finally: non-best -> CPU, gc.collect, empty_cache
   Phase 6: Drift Detection
      - 70/30 temporal split, PSI/KS/MMD tests
      - Saves drift_ref_snapshot (up to 5000 rows)
      - If drift detected: spawns daemon retrain thread
   Phase 7: Model Registry Serialization
      - model_weights.pth, encoder state dicts, tabular_scaler.joblib,
        target_encoder.joblib, text_tokenizer/, encoder_config.json,
        schema.json, drift_reference.npy, metadata.json
      |
7. POST /predict-async or WS /ws/predict {model_id, inputs}
   - Load/cache MultimodalInferenceEngine (LRU, max 5)
   - Engine reconstructs: head from state_dict, frozen encoders,
     preprocessors, tabular encoder (GRN/MLP)
   - _build_batch(): tabular -> scaler -> encoder, text -> BERT CLS,
     image -> _resolve_image() -> ResNet-50 -> 512-dim
   - _MultimodalHead: fusion -> Linear -> GELU -> Dropout -> Linear
   - _decode_logits(): sigmoid/softmax/raw -> predictions + confidences
   - Optional Captum IntegratedGradients XAI
   - Buffers numeric inputs for production drift monitoring
```

---

## 3. Per-Stage Evaluation

### Stage 1 — Multi-Dataset Ingestion

**Backend: Implemented**

- `DataIngestionManager` (`data_ingestion/ingestion_manager.py`) provides async multi-URL ingestion via `asyncio.gather()` with per-URL fault isolation
- SHA-256 source-addressed caching with URL normalization (`_normalize_url`: strips whitespace, lowercases host, removes `www.`, canonicalizes Kaggle slugs)
- Cache hit path checks normalized hash first, falls back to legacy (raw-string) hash, then transparently migrates to normalized key
- Supported sources: HTTP/HTTPS (aiohttp, 2-hour timeout), Kaggle CLI (1800s timeout, ZipSlip protection), local file paths (shutil.copy2)
- Atomic metadata persistence via `tempfile.mkstemp()` + `os.replace()`
- DVC lineage integration: `dvc add` called on cache directories (fails silently if DVC not installed)
- Mendeley URLs explicitly rejected with `ValueError`
- `DataLoader` (`data_ingestion/loader.py`) supports CSV, Parquet, JSON, and image directories. Uses Polars `scan_csv`/`scan_parquet` for lazy loading, with Dask fallback. Image datasets use `LazyImageDataset` (PyTorch Dataset storing only paths, loading pixels in `__getitem__`)

**Frontend: Implemented**

- Multi-URL text area input (Phase 1 in 7-phase workflow)
- Per-dataset status display (ingesting/cached/failed) via polling `GET /ingest/status/{task_id}`
- Session ID generated from `datetime.now().isoformat()`

**Gap:** Cached/ingesting status indicators exist but rely on task polling, not direct cache-hit detection. The frontend does not display whether a dataset was served from cache vs freshly downloaded.

---

### Stage 2 — Multi-Dataset Schema Detection

**Backend: Implemented**

- `MultiDatasetSchemaDetector` (`data_ingestion/schema_detector.py`) implements two-tier detection
- Tier-1: Per-dataset column classification (image/text/timeseries/tabular) via heuristics (image extension matching, text mean-length >50 chars, timeseries bracket patterns)
- Target detection: 4-signal scoring (keyword +0.35, unique-ratio +0.30, structural pattern +0.60, regression signal +0.25, ID penalty -0.45)
- Problem type inference: multilabel > binary > multiclass > regression > unsupervised
- Tier-2: Cross-dataset aggregation with union-find relatedness grouping (4 signals: column Jaccard 0.40, target match 0.30, modality Jaccard 0.20, problem match 0.10; threshold 0.50)
- Returns `GlobalSchema` dataclass with `global_problem_type`, `global_modalities`, `primary_target`, `fusion_ready`, `detection_confidence`, `per_dataset`, `relatedness_report`

**Frontend: Implemented**

- Displays detected schema: problem type, target column, modalities, confidence score
- Per-dataset schema summary

**Gap: Dataset selection for incompatible schemas is NOT implemented.** When multiple unrelated groups are detected, the system silently selects the largest related group. No user interaction for dataset selection or conflict resolution.

---

### Stage 3 — Preprocessing Pipeline

**Backend: Implemented**

- **Tabular** (`preprocessing/tabular_preprocessor.py`): scikit-learn `ColumnTransformer` — numeric: `SimpleImputer(median)` + `StandardScaler`; categorical: `SimpleImputer(most_frequent)` + `OneHotEncoder(sparse_output=False, max_cardinality=50)`. Automatic filtering of: ID columns (regex + uniqueness ratio), datetime columns (dtype + string probing), path/URL columns (extension regex), constant columns, near-unique integers
- **Text** (`preprocessing/text_preprocessor.py`): `bert-base-uncased` tokenizer via HuggingFace `AutoTokenizer`, max_length=128, NaN/None sanitization, squeeze from `[1,128]` to `[128]` for correct DataLoader batching
- **Image** (`preprocessing/image_preprocessor.py`): `Resize(224,224)` + `ToTensor()` + `Normalize(ImageNet)`. Separate augmentation pipeline: `RandomHorizontalFlip` + `RandomRotation(10)` + `ColorJitter(0.2, 0.2)`

**Frontend: Implemented**

- Phase 3 displays preprocessing status and completion

---

### Stage 4 — Encoder Selection

**Backend: Implemented**

- `AdvancedModelSelector` (`automl/advanced_selector.py`): Tier-based selection (lightweight/balanced/sota) per modality based on dataset size and GPU memory. Produces HPO search space from PDF epoch/LR/dropout/fusion bounds
- `JITEncoderSelector` (`automl/jit_encoder_selector.py`): Runtime VRAM-constrained selection. Probes `torch.cuda.mem_get_info()`, budget = 0.85 × free VRAM, exhaustive Cartesian search over 3×3=9 vision×text combos sorted by capacity descending, dummy-forward profiling, first feasible wins
- Plugin registration API: `register_vision_encoder()`, `register_text_encoder()`, `register_tabular_encoder()` for hot-loading custom encoders
- Vision registry: ConvNeXt-Tiny (28.6M) > ResNet-50 (25.6M) > MobileNetV3-Small (2.5M)
- Text registry: DeBERTa-v3-base (183.8M) > BERT-base (109.5M) > MiniLM-L6-v2 (22.7M)
- Tabular registry: GRN (12K) > MLP (5K)
- CPU fallback: lightest encoders unconditionally

**Frontend: Implemented**

- Phase 4 displays model recommendations with primary and fallback options
- HPO search space visualization

---

### Stage 5 — Training Behavior

**Backend: Implemented**

- `ApexLightningModule` (`automl/trainer.py`): PyTorch Lightning wrapper with FP16 mixed precision, CosineAnnealingLR, AdamW
- Frozen encoder storage via `object.__setattr__` (excluded from state_dict/parameters)
- Trainable tabular encoder as proper `nn.Module` submodule
- Auto-selected loss functions: `BCEWithLogitsLoss` (binary/multilabel), `CrossEntropyLoss` (multiclass), `MSELoss` (regression)
- Automatic class imbalance correction via inverse-frequency weights
- Windows WDDM TDR safety: `torch.cuda.synchronize()` after every step
- torchmetrics: Accuracy + F1 (classification), RMSE + R2 (regression)
- Optuna HPO with HyperbandPruner, EarlyStopping (patience=5)
- Per-trial GPU cleanup in `finally` blocks
- Embedding pre-computation: frozen text embeddings cached for both splits, image embeddings cached for val only (training needs stochastic augmentation)

**Frontend: Implemented**

- Phase 5 displays per-trial metrics, epoch progress, best trial tracking
- Real-time polling via `GET /train-pipeline/status/{task_id}`

**Adaptive Training Behavior:**
- EarlyStopping detects convergence/flatline (patience=5 epochs, delta=0.001)
- HyperbandPruner detects underfitting/overfitting (prunes bottom trials at intermediate epochs)
- Epoch count is searchable via HPO (bounded by PDF matrix: dataset-size × modality)

---

### Stage 6 — Multimodal Encoding & Fusion

**Backend: Implemented**

- `_MultimodalHead` (`automl/trainer.py:34-93`): Accepts per-modality embedding dicts, applies fusion, projects through `Linear(fused_dim, hidden_dim)` → `GELU` → `Dropout` → `Linear(hidden_dim, num_outputs)`
- Fusion strategies: `ConcatenationFusion` (concatenate all embeddings) and `AttentionFusion` (learned attention weights) from `modelss/fusion.py`
- `_encode_batch()` transforms raw batch keys into pooled embeddings: tabular → trainable GRN/MLP, text → frozen BERT CLS pooling → `text_pooled [N, 768]`, image → frozen ImageEncoder → `image_pooled [N, 512]`
- Missing-modality safety: injects `1e-7` dummy tensors for expected keys not present in batch

**Frontend: Partial**

- No direct visualization of fusion mechanism or per-modality embedding contributions
- Fusion strategy is shown as a hyperparameter choice but not as a live pipeline component

---

### Stage 7 — Drift Detection & Monitoring

**Backend: Implemented**

- `DriftDetector` (`monitoring/drift_detector.py`): Stateless detector with PSI (threshold >0.25), KS test (threshold >0.3), MMD/FDD (threshold >0.5)
- Phase 6 in training pipeline: 70/30 temporal split, runs PSI/KS/MMD on tabular features, saves `drift_ref_snapshot` (up to 5000 rows)
- Autonomous retraining: if drift detected, spawns daemon thread via `RetrainingPipeline.retrain()` (runs Phases 1-5,7; skips Phase 6 to prevent infinite recursion)
- `drift_reference.npy` persisted in model artifacts during Phase 7
- Production drift monitoring: `POST /predict/drift-check` compares buffered prediction inputs against saved reference
- `POST /monitor/drift` re-runs full pipeline drift detection on session data
- `GET /predict/buffer-stats/{model_id}` returns buffer count

**Frontend: Implemented**

- Phase 6 drift monitoring dashboard with 3 tabs (Metrics, Distribution, History)
- Production drift monitor in prediction section: buffer stats display, "Check Drift" button (disabled when <10 buffered), PSI/KS/FDD metrics

**Gap:** Autonomous retrain runs silently with no user notification. No task_id, no status tracking, no UI indication.

---

### Stage 8 — Model Registry & Management

**Backend: Partial**

- `ModelRegistry` (`model_registry_pkg/model_registry.py`): Singleton with `register_model()`, `get_model()`, `list_models()`, `unregister_model()`
- Training orchestrator Phase 7: serializes weights, encoder state dicts, tabular_scaler.joblib, target_encoder.joblib, text_tokenizer/, encoder_config.json, schema.json, drift_reference.npy, metadata.json
- `GET /model-registry` scans filesystem (not ModelRegistry singleton), reads metadata.json + MLflow SQLite for val_loss
- `GET /model-info/{model_id}` loads preprocessor and returns effective features, dropped columns, class labels
- **Missing endpoints**: No model delete, no model rename, no model download

**Frontend: Partial**

- Model list display with metadata
- Model download button exists but is **disabled** (no backend endpoint)
- Deploy button exists but is **disabled** (no backend endpoint)
- No delete or rename UI controls

---

### Stage 9 — Prediction Pipeline

**Backend: Implemented**

- `MultimodalInferenceEngine` (`pipeline/inference_engine.py`): Loads model artifacts, reconstructs head + frozen encoders + preprocessors + tabular encoder
- Three prediction endpoints: `POST /predict` (sync), `POST /predict-async` (fire-and-poll), `WS /ws/predict` (WebSocket streaming)
- Input handling: `_resolve_image()` supports data URIs, raw base64, raw bytes, file paths
- `_build_batch()`: tabular → scaler → encoder, text → BERT CLS, image → ResNet-50 → 512-dim
- `_decode_logits()`: sigmoid (binary/multilabel), softmax (multiclass), raw (regression) → predictions + per-class confidences
- Optional Captum XAI: IntegratedGradients with two attribution paths (real BERT via inputs_embeds, approximate fallback)
- `POST /upload/image` multipart endpoint (10MB limit, MIME validation, UUID filenames)
- Prediction input buffering for production drift monitoring

**Frontend: Implemented**

- Image upload with base64 data URI encoding (bypasses shared filesystem)
- CSV batch upload with schema validation
- Manual tabular input with per-field controls
- Confidence visualization: per-class bar charts, batch confidence histograms
- Prediction history (last 20 entries, session-scoped)

**Gap:** Frontend uses only `POST /predict-async` + polling. Sync `/predict` and WebSocket `/ws/predict` are not used by the frontend. Image upload endpoint (`POST /upload/image`) exists but frontend uses inline base64 instead.

---

## 4. Documentation vs Reality

### 4.1 Documented Features Missing or Partially Implemented

| Feature | README/Doc Claim | Reality |
|---|---|---|
| Model Rename | Implied by "model management" | **Not implemented**. No endpoint or UI. |
| Model Download | Expected by UI button | **Not implemented**. Frontend button disabled. No download endpoint. |
| Model Delete | Expected by registry management | **Partially implemented**. `ModelRegistry.unregister_model()` exists but no API endpoint. No UI. |
| Autonomous Retrain Visibility | README: "autonomous retraining triggers" | **Runs silently**. Daemon thread with no user notification, no task tracking, no UI status. |
| MLflow UI Integration | README tech stack: "MLflow (experiment tracking)" | **Partial**. MLflow logs params/val_loss during Optuna trials. No MLflow UI integration. `/model-registry` reads MLflow SQLite directly but doesn't expose MLflow experiments. |
| ViT-Base / RoBERTa-large / TabNet / FT-Transformer | Listed in `config/hyperparameters.py` `get_optuna_distributions()` | **Not implemented**. Appear only as categorical option strings. No factory, no model code. |
| DistilBERT | Referenced in `config/hyperparameters.py` presets | **Not implemented**. Actual smallest text encoder is MiniLM-L6-v2. |
| Dataset Selection for Incompatible Schemas | Expected user interaction | **Not implemented**. System silently selects largest related group. |
| Port 8000 | README: "FastAPI server starts on http://localhost:8000" | **Incorrect**. Actual port is `8001` (`run_api.py` bottom). |
| Out-of-core Exclusion | README "Excludes" section: datasets exceeding RAM will fail | **Partially incorrect**. `AutoVisionIterableDataset` streaming class exists in `training_orchestrator.py:292-577` but is not wired into the standard training path. |

### 4.2 Features Implemented but Undocumented

| Feature | Location | Description |
|---|---|---|
| ECG Domain Adapter | `data_ingestion/adapters/ecg_adapter.py` | Full PTB-XL ECG waveform adapter with SCP code expansion, image path resolution, ECG-specific preprocessing config (landscape 224x448). Not in README. |
| AutoVisionIterableDataset | `training_orchestrator.py:292-577` | Out-of-core streaming dataset for 100GB+ CSVs with multi-worker chunk sharding, Parquet row-group iteration. Not in README. |
| DVC Lineage Integration | `ingestion_manager.py:121` | `dvc add` on cached datasets for version control. Fails silently if not installed. Not in README. |
| Mendeley URL Rejection | `ingestion_manager.py:283` | Explicit `ValueError` for Mendeley download URLs. Not documented. |
| Legacy Cache Hash Migration | `ingestion_manager.py:229-245` | Transparent migration from pre-normalization hashes. Undocumented. |
| Image Upload Endpoint | `run_api.py` | `POST /upload/image` multipart file upload (UUID naming, MIME validation, 10MB limit). Not in README. |
| Prediction Input Buffer + Drift Check | `run_api.py` | In-memory buffer + `/predict/drift-check` and `/predict/buffer-stats` endpoints. Not in README. |
| Base64 Image Support in Inference | `inference_engine.py` | `_resolve_image()` supports data URIs, raw base64, raw bytes, and file paths. Not in README. |
| Hyperparameter Presets | `config/hyperparameters.py:230` | Four named presets (small/medium/large/fast). Not in README. |
| Config YAML/JSON loading | `config/hyperparameters.py:70-90` | `HyperparameterConfig.from_json()`, `from_yaml()`, `save_json()`, `save_yaml()`. Not in README. |
| PerformanceTracker | `monitoring/performance_tracker.py` | Per-model prediction tracking with MSE/MAE/RMSE/accuracy, time-windowed trends. Fully implemented but no API endpoint or UI uses it. |
| MultimodalPredictor (Deprecated) | `modelss/predictor.py` | Legacy predictor with own fusion MLP (3 blocks, 512->256->128). Deprecated with warning. |
| Confidence Visualization | `frontend/app_enhanced.py` | Per-class probability bar charts, batch confidence histograms. Not in README. |
| Prediction History | `frontend/app_enhanced.py` | Rolling 20-entry prediction log in session state. Not in README. |

---

## 5. Additional Capabilities Discovered

### 5.1 Multi-Layer Caching Architecture (3 layers)

1. **L1: Dataset Ingestion Cache** — SHA-256 source-addressed, atomic metadata writes, legacy hash migration (`ingestion_manager.py`)
2. **L2: Embedding Pre-computation Cache** — Frozen encoder outputs cached as CPU tensors before HPO. Text: both splits (deterministic). Image: val only (augmentation stochasticity preserved) (`training_orchestrator.py`)
3. **L3: LRU Inference Engine Cache** — `OrderedDict` + `threading.Lock`, max 5 engines, shared across REST/WebSocket (`run_api.py`)

### 5.2 GPU Memory Lifecycle Management

- Unconditional cleanup in `finally` blocks: non-best trial models moved to CPU, `torch.cuda.empty_cache()`, `gc.collect()`
- Windows WDDM TDR safety: `torch.cuda.synchronize()` after every training/validation step (`trainer.py`)

### 5.3 Automatic Class Imbalance Correction

- Inverse-frequency class weights computed from training split target distribution
- Injected into `CrossEntropyLoss` (multiclass) or `BCEWithLogitsLoss` `pos_weight` (binary)

### 5.4 Hot-Loadable Encoder Plugin System

- `register_vision_encoder()`, `register_text_encoder()`, `register_tabular_encoder()` in `jit_encoder_selector.py`
- `config/encoder_plugins.py` loaded at startup
- Currently a template with commented-out examples

### 5.5 Background Task Orchestration (3 patterns)

1. `asyncio.create_task()` for ingestion and training
2. `BackgroundTasks.add_task()` for async inference
3. `asyncio.to_thread()` for CPU-bound operations within async endpoints

### 5.6 Schema Contract Propagation (3 layers)

- Preprocessing layer: `_feature_names_in` persisted in serialized preprocessor
- API layer: `/model-info/{model_id}` returns effective features + dropped columns
- Frontend layer: dynamic input fields from effective features, downloadable CSV template

### 5.7 Cross-Dataset Relatedness via Union-Find

- 4-signal weighted scoring (column Jaccard 0.40, target match 0.30, modality Jaccard 0.20, problem match 0.10)
- Union-find with path compression groups related datasets
- Largest group selected when unrelated groups detected

### 5.8 Autonomous Drift-Triggered Retraining

- Phase 6 drift detection -> `should_retrain()` -> daemon thread spawns `RetrainingPipeline.retrain()`
- Retrain runs Phases 1-5,7 (Phase 6 omitted to prevent infinite recursion)
- Schema validation: detects schema drift between original and production data

### 5.9 Dual Prediction-Time Drift Monitoring

- **Phase 6 drift** (`POST /monitor/drift`): re-runs full pipeline on session data with temporal split
- **Production drift** (`POST /predict/drift-check`): compares buffered prediction inputs against saved `drift_reference.npy`

### 5.10 Captum XAI with Two Attribution Paths

- **Real BERT path**: `inputs_embeds` injection for full differentiability through transformer -> fusion head
- **Approximate path**: random embedding fallback when BERT unavailable
- Tabular IG: per-feature mean absolute attribution

### 5.11 Out-of-Core Streaming Dataset

- `AutoVisionIterableDataset` for datasets exceeding RAM
- Fixed-size chunk reading from CSV/Parquet with multi-worker sharding
- **Not currently used** by the main training path (standard map-style dataset used instead)

### 5.12 Security Hardening

| Vulnerability | Location | Mechanism |
|---|---|---|
| Directory Traversal | `run_api.py` | Regex validation (`^[\w\-.:]+$`) + `..` rejection on all `model_id` parameters |
| ZipSlip | `ingestion_manager.py` | Member path resolution + `startswith()` check before `extractall()` |
| XSS | `app_enhanced.py` | `html.escape()` on user-derived token text before HTML embedding |
| CORS Misconfiguration | `run_api.py` | Origin whitelist instead of wildcard `*` |
| Session State Races | `run_api.py` | `threading.Lock` protecting shared dicts |
| SQLite TOCTOU | `task_store.py` | `BEGIN IMMEDIATE` transactions |
| Unsafe Deserialization | `model_registry.py` | `torch.load(..., weights_only=True)` |

---

## 6. Frontend <-> Backend Synchronization

### 6.1 Mismatches

| Issue | Frontend | Backend | Impact |
|---|---|---|---|
| Model Delete | No UI control | `ModelRegistry.unregister_model()` exists but no endpoint | Users cannot delete models |
| Model Rename | No UI control | No endpoint | Feature entirely absent |
| Model Download | Disabled button | No endpoint | Non-functional UI element |
| Deploy to Production | Disabled button | No endpoint | Non-functional UI element |
| Sync Predict | Not used | `POST /predict` exposed | Frontend always uses async, ignoring faster sync path |
| WebSocket Predict | Not used | `WS /ws/predict` with chunked streaming | No real-time streaming in UI |
| Image Upload Endpoint | Uses base64 data URIs | `POST /upload/image` multipart exposed | Upload endpoint bypassed entirely |
| PerformanceTracker | Not used | Fully implemented class | Prediction tracking with metrics goes unused |
| Autonomous Retrain | No notification | Daemon thread runs silently | Users unaware of auto-retraining |
| Phase 6 Naming | "Phase 6: Monitoring" with 3 tabs | "Phase 6: Drift Detection" in pipeline | Semantic mismatch |
| `/config` Endpoint | Not called | Returns `HyperparameterConfig().to_dict()` | HP form controls hardcoded in frontend |
| `/health` Endpoint | Checks `status_code == 200` only | Returns GPU info, CUDA version | Detailed health info ignored |

### 6.2 Properly Synchronized

| Endpoint | Status |
|---|---|
| `POST /ingest/datasets` | Aligned |
| `GET /ingest/status/{task_id}` | Aligned |
| `POST /detect-schema` | Aligned |
| `POST /preprocess` | Aligned |
| `POST /select-model` | Aligned |
| `POST /train-pipeline` | Aligned |
| `GET /train-pipeline/status/{task_id}` | Aligned |
| `POST /predict-async` | Aligned |
| `GET /task/{task_id}` | Aligned |
| `GET /model-registry` | Aligned |
| `GET /model-info/{model_id}` | Aligned |
| `POST /monitor/drift` | Aligned |
| `POST /predict/drift-check` | Aligned |
| `GET /predict/buffer-stats/{model_id}` | Aligned |

---

## 7. Missing Capabilities

| Capability | Expected By | Status | Detail |
|---|---|---|---|
| Model Download/Export | Frontend UI + user workflow | **Missing** | Button disabled. No endpoint. No ONNX/TorchScript export. |
| Model Rename | User workflow | **Missing** | Neither backend nor frontend. |
| Model Delete via UI | User workflow | **Missing endpoint** | Registry method exists but not exposed via API. |
| Dataset Selection for Incompatible Schemas | Schema detection UX | **Missing** | System silently selects largest related group. |
| Production Deployment | README acknowledges exclusion | **Confirmed absent** | No Kubernetes, Docker, or serving infra. |
| Test Suite | Engineering practice | **Missing** | No `tests/` directory. No test files. |
| Rate Limiting | API security | **Missing** | No rate limiting on any endpoint. |
| Authentication/Authorization | API security | **Missing** | All endpoints fully open. |
| Task Cleanup | README roadmap P2 | **Missing** | No TTL-based cleanup. Completed tasks persist indefinitely. |
| Distributed Training | README roadmap P1 | **Missing** | Single-device only. |
| Time-Series Modality | README roadmap P1 | **Missing** | Column detection exists (`_is_timeseries`) but no encoder or processing pipeline. |

---

## 8. Architectural Risks

### 8.1 Session State Volatility

`_session_store`, `_engine_cache`, `_prediction_buffer` are all in-memory Python dicts. Server restart loses all state. The global `session_ingested_hashes` is overwritten on every new ingestion, meaning only the latest session's data is available through the fallback path.

### 8.2 Materialization Memory Cliff

Phase 3 materializes all datasets into a single pandas DataFrame, capped at 50,000 rows. For datasets near this cap with wide feature spaces (high-cardinality OHE), memory usage can spike dramatically. The `AutoVisionIterableDataset` streaming alternative exists but is not wired into the standard training path.

### 8.3 Engine Cache Race Condition

Between releasing the lock after a cache miss and re-acquiring it after model load, another thread can load the same model. Result: duplicate model loads wasting VRAM, though the last write wins without data corruption.

### 8.4 Autonomous Retrain Opacity

Retrain runs as a daemon thread with no user notification, no task_id tracking, no status polling. If retrain fails, the error is only logged. If retrain succeeds, a new model appears in the registry with no indication it was auto-triggered.

### 8.5 ModelRegistry Singleton Conflict

`ModelRegistry` uses a class-level singleton. If two different `registry_path` values are passed, the second is silently ignored. The training orchestrator writes artifacts directly to `models/registry/` (bypassing `ModelRegistry`), while `ModelRegistry` has its own `metadata.json`. These two systems are disconnected — `/model-registry` scans the filesystem, not `ModelRegistry.metadata`.

### 8.6 Schema Drift Between Training and Serving

Schema contract propagation works for tabular features via `_feature_names_in`. However, text and image columns are identified by column name heuristics (fallback to common names like "text", "report", "description"). If column names differ between training and prediction input dicts, text/image features may be silently missed.

### 8.7 SQLite Under Multi-Worker Uvicorn

`task_store.py` uses `BEGIN IMMEDIATE` for safety, but Uvicorn runs with a single worker. If deployed with `--workers N`, the `_session_store` in-memory dict would be process-local, causing cross-worker failures for session-based endpoints.

### 8.8 Hardcoded Port Mismatch

Backend serves on port `8001` (`run_api.py`). Frontend connects to `http://localhost:8001`. README says "starts on `http://localhost:8000`" — this is incorrect.

### 8.9 Hyperparameter Config Divergence

`config/hyperparameters.py` references encoders (ViT-Base, RoBERTa-large, DistilBERT, TabNet, FT-Transformer) that have no implementations. `get_optuna_distributions()` includes these as categorical options. If selected via the config API, they would produce `KeyError` or `ImportError` at runtime.

### 8.10 No Input Validation on Prediction Payload Structure

`/predict` endpoints check batch size and model_id but do not validate input dict keys against the model's expected schema. Mismatched or missing columns are handled by zero-filling, which can produce incorrect predictions without any warning.

---

## 9. Endpoint Inventory

Complete list of API endpoints discovered in `run_api.py`:

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | System health + GPU info |
| GET | `/config` | Hyperparameter config |
| POST | `/ingest/datasets` | Multi-URL dataset ingestion |
| GET | `/ingest/status/{task_id}` | Ingestion progress polling |
| POST | `/detect-schema` | Schema detection |
| POST | `/preprocess` | Preprocessing pipeline |
| POST | `/select-model` | Model/encoder recommendation |
| POST | `/train-pipeline` | Launch 7-phase training |
| GET | `/train-pipeline/status/{task_id}` | Training progress polling |
| POST | `/monitor/drift` | Session-data drift check |
| GET | `/model-registry` | List registered models |
| GET | `/model-info/{model_id}` | Model metadata + schema |
| POST | `/predict` | Synchronous prediction |
| POST | `/predict-async` | Async prediction (fire-and-poll) |
| GET | `/task/{task_id}` | Generic task result polling |
| WS | `/ws/predict` | WebSocket streaming prediction |
| POST | `/upload/image` | Image file upload |
| POST | `/predict/drift-check` | Production drift check |
| GET | `/predict/buffer-stats/{model_id}` | Prediction buffer stats |

**Total: 19 endpoints (17 REST + 1 WebSocket + 1 GET generic)**

---

## 10. Summary Statistics

| Metric | Value |
|---|---|
| Total Python source files analyzed | ~25 |
| Total lines of code (estimated) | ~10,500 |
| API endpoints | 19 |
| Pipeline stages implemented | 9/9 (2 partial) |
| Documented features missing | 10 |
| Undocumented features found | 14 |
| Frontend-backend mismatches | 12 |
| Architectural risks identified | 10 |
| Test files | 0 |

---

*Report generated by implementation discovery audit, 2026-03-12*
