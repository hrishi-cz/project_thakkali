# APEX AutoML - Complete 7-Phase Integrated System

## Overview

APEX (Advanced Predictive Ensemble with eXtendable modularity) is a comprehensive autoML platform that implements a complete 7-phase workflow for multimodal machine learning.

**Status:** ✅ Fully Integrated - All phases connected with frontend, backend, and API

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                       │
│         (Multi-phase Workflow Dashboard)                    │
└─────────────────────────────────────────────────────────────┘
                             │
                   HTTP (REST API)
                             │
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Server                           │
│              (20+ REST Endpoints)                           │
└─────────────────────────────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
    Phase 1-4          Phase 5          Phase 6-7
    (Backend)       (Training)      (Monitoring)
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Data Flow   │  │ GPU Training │  │  Monitoring  │
│  Modules     │  │  Orchestrator│  │  & Registry  │
└──────────────┘  └──────────────┘  └──────────────┘
```

---

## 7-Phase Workflow

### Phase 1: Data Ingestion & Caching ✅

**File:** `data_ingestion/ingestion_manager.py`

**Purpose:** Load, validate, and cache datasets from multiple sources

**Features:**

- Multi-source support: Kaggle URLs, HTTP/HTTPS links, local paths
- SHA-256 hash-based caching for fast retrieval
- Support for CSV, Parquet, JSON formats
- Metadata persistence with timestamps
- Progress tracking with callbacks
- Cache hit/miss detection

**API Endpoint:**

```http
POST /api/ingest
Content-Type: application/json

{
  "sources": [
    "https://kaggle.com/datasets/example1",
    "https://drive.google.com/data.csv",
    "/local/path/data.parquet"
  ],
  "cache_enabled": true
}
```

**Frontend Integration:** Phase 1 page allows uploading multiple datasets with visual cache status

---

### Phase 2: Schema Detection & Problem Type Inference ✅

**File:** `data_ingestion/schema_detector.py`

**Purpose:** Automatically detect column types, modalities, and problem type

**Detection Logic:**

- **Image Columns:** File paths with image extensions (.jpg, .png, .jpeg, etc.)
- **Text Columns:** Strings with average length > 50 characters
- **Tabular Columns:** Numeric types (int, float, discrete)
- **Target Column:** Fuzzy matching with keywords (label, target, class, etc.) + cardinality check
- **Problem Type:** Inferred from target column (regression, binary classification, multiclass)

**Fuzzy Matching:** Uses fuzzywuzzy (Levenshtein distance > 75%) for robust target detection

**API Endpoint:**

```http
POST /api/detect-schema
Content-Type: application/json

{
  "data_path": "/path/to/data.csv",
  "fuzzy_threshold": 0.75
}

Response:
{
  "detected_columns": {
    "image": ["image", "photo"],
    "text": ["description", "review"],
    "tabular": ["age", "income", "rating"]
  },
  "target_column": "label",
  "problem_type": "classification_multiclass",
  "modalities": ["image", "text", "tabular"],
  "column_confidence": {...}
}
```

**Frontend Integration:** Phase 2 page shows detected columns with fuzzy matching confidence scores

---

### Phase 3: Data Preprocessing ✅

**Files:** `preprocessing/*.py`

**Purpose:** Apply modality-specific preprocessing transformations

**Image Preprocessing:**

- Lazy loading strategy (load on-demand)
- Resize to 224×224 (ImageNet standard)
- Normalize: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
- Output shape: `(B, 3, 224, 224)`

**Text Preprocessing:**

- NLTK tokenization with automatic NLTK data download
- BERT tokenizer with padding/truncation to 128 tokens
- Attention mask generation
- Output shape: `(B, 128)`

**Tabular Preprocessing:**

- KNN imputation for missing values
- StandardScaler normalization
- OneHotEncoder for categorical features
- Output shape: `(B, N_features)`

**API Endpoint:**

```http
POST /api/preprocess
Content-Type: application/json

{
  "data_path": "/path/to/preprocessed/data",
  "image_columns": ["image"],
  "text_columns": ["description"],
  "tabular_columns": ["age", "income", "rating"]
}
```

**Frontend Integration:** Phase 3 page shows preprocessing progress with real-time stage updates

---

### Phase 4: Advanced Model Selection ✅

**File:** `automl/advanced_selector.py`

**Purpose:** Automatically select models and hyperparameters based on data, GPU, and problem characteristics

**Selection Criteria:**

**GPU Memory Detection:**

- <6GB: Lightweight tier (MobileNetV3, DistilBERT)
- 6-12GB: Medium tier (ResNet50, BERT-base)
- > 12GB: Large tier (ViT-B, RoBERTa-large)

**Dataset Size Tiers:**

- <1k samples: Lightweight models, 45 epochs
- 1-10k samples: Balanced models, 18 epochs
- 10-50k samples: Standard models, 12 epochs
- > 50k samples: Large models, 6 epochs

**Encoder Selection:**

| Modality | Dataset | Selection   | Rationale     |
| -------- | ------- | ----------- | ------------- |
| Image    | <1k     | MobileNetV3 | Lightweight   |
| Image    | 1-10k   | ResNet50    | Balanced      |
| Image    | >10k    | ViT-B       | High accuracy |
| Text     | Binary  | DistilBERT  | Fast          |
| Text     | >5k     | BERT-base   | Standard      |
| Text     | Complex | RoBERTa     | Superior      |
| Tabular  | Any     | TabNet      | Interpretable |

**Hyperparameter Computation:**

- Batch size: Adaptive based on GPU memory (16-64)
- Learning rate: 1e-3 (with optional decay schedules)
- Dropout: 0.2 (modality-specific adjustment)
- Weight decay: 1e-5 (L2 regularization)

**Fusion Strategy:**

- Single modality: No fusion (modality output directly)
- Multi-modality: Attention-based fusion (learns feature importance)

**API Endpoint:**

```http
POST /api/select-model
Content-Type: application/json

{
  "dataset_size": 10000,
  "modalities": ["image", "text", "tabular"],
  "problem_type": "classification_multiclass"
}

Response:
{
  "selected_encoders": {
    "image": "ResNet50",
    "text": "BERT-base",
    "tabular": "TabNet"
  },
  "hyperparameters": {
    "batch_size": 32,
    "epochs": 18,
    "learning_rate": 0.001,
    "dropout": 0.2,
    "weight_decay": 1e-5
  },
  "selection_rationale": "Selected based on large GPU and 10k samples"
}
```

**Frontend Integration:** Phase 4 page displays:

- Selected models with detailed specifications
- Detailed rationale for each selection
- Hyperparameter preview with sliders for customization
- Estimated training time and memory usage

---

### Phase 5: GPU Training Loop ✅

**File:** `pipeline/training_orchestrator.py`

**Purpose:** Execute GPU-accelerated training with safety mechanisms

**GPU Safety Mechanisms:**

- `torch.cuda.synchronize()` after each batch (Windows WDDM safety)
- Automatic CUDA memory cleanup
- Gradient accumulation for large batches
- Mixed precision training support (optional)

**Training Process:**

1. Initialize encoders on GPU
2. For each epoch:
   - Forward pass through multimodal encoders
   - Compute loss (CrossEntropyLoss for classification, MSELoss for regression)
   - Backward pass with gradient computation
   - Optimizer step (Adam with specified learning rate)
   - CUDA synchronization for Windows compatibility
3. Validation after each epoch
4. Save best checkpoint based on validation metric
5. Log metrics for monitoring

**Training Metrics:**

- Train loss, validation loss
- Accuracy (for classification)
- F1 score, Precision, Recall (for multiclass)
- R² Score, MAE (for regression)

**API Endpoint:**

```http
POST /api/train-pipeline
Content-Type: application/json

{
  "dataset_sources": ["https://kaggle.com/datasets/..."],
  "problem_type": "classification_multiclass",
  "modalities": ["image", "text", "tabular"],
  "target_column": "label"
}

Response:
{
  "job_id": "uuid-generated-id",
  "status": "queued",
  "total_phases": 7,
  "message": "Training pipeline started"
}
```

**Check Status:**

```http
GET /api/pipeline-status/{job_id}

Response:
{
  "status": "running",
  "phase": 5,
  "phases_completed": 4,
  "progress_percent": 60,
  "created_at": "2024-02-10T10:30:00",
  "result": null
}
```

**Frontend Integration:** Phase 5 page shows:

- Real-time epoch progress with loss curves
- Train vs validation metrics
- Current epoch and remaining epochs
- Estimated time remaining
- GPU memory usage monitoring

---

### Phase 6: Drift Detection & Monitoring ✅

**File:** `monitoring/drift_detector.py`

**Purpose:** Monitor model performance and detect data/concept drift

**Drift Detection Metrics:**

**PSI (Prediction Stability Index):** > 0.25 threshold

- Measures shift in prediction distribution
- Formula: $\text{PSI} = \sum_i (p_i - q_i) \ln(p_i/q_i)$
- $p_i$: Reference predictions, $q_i$: New predictions

**KS Statistic (Kolmogorov-Smirnov):** > 0.30 threshold

- Maximum distance between two CDFs
- Detects distribution shifts in features
- Implementation: `scipy.stats.ks_2samp()`

**Feature/Embedding Drift (FDD):** > 0.50 threshold

- Drift in learned feature spaces
- Computed via Wasserstein distance
- Detects concept drift (model assumptions violated)

**Monitoring Mechanism:**

1. Track per-model performance metrics
2. Store prediction history with sliding window (last N predictions)
3. Continuously compute drift statistics
4. Trigger retraining recommendation when thresholds exceeded
5. Store drift history for audit trail

**API Endpoint:**

```http
GET /monitoring/performance/{model_id}
Response: Recent performance metrics

GET /drift/history/{model_id}
Response: Historical drift detection results

POST /drift/check
{
  "model_id": "apex_v1_20240210",
  "new_data_source": "/path/to/recent/data.csv"
}
```

**Frontend Integration:** Phase 6 page displays:

- Performance metrics with trend charts
- Drift detection status (OK/WARNING/ALERT)
- Individual drift metric values vs thresholds
- Retraining recommendations
- Model registry with deployment status

---

### Phase 7: Model Registry & Versioning ✅

**File:** `model_registry_pkg/model_registry.py`

**Purpose:** Store trained models, manage versions, and track deployment

**Registry Features:**

- Model ID generation: `apex_v1_YYYYMMDD_HHMMSS`
- Metadata persistence (JSON format)
- Model checkpoint storage
- Version tracking
- Deployment status management
- A/B testing support

**Stored Metadata:**

```json
{
  "model_id": "apex_v1_20240210_103000",
  "created_at": "2024-02-10T10:30:00",
  "config": {
    "dataset_sources": [...],
    "problem_type": "classification_multiclass",
    "modalities": ["image", "text", "tabular"]
  },
  "phases_summary": {...},
  "status": "active",
  "deployment_ready": true,
  "performance": {
    "final_accuracy": 0.925,
    "final_f1": 0.891,
    "validation_loss": 0.312
  }
}
```

**API Endpoint:**

```http
GET /models
Response: List all registered models

GET /models/{model_id}
Response: Detailed model information
```

**Frontend Integration:** Phase 6 page shows:

- Model registry with all versions
- Performance comparison between models
- Deployment status (Active/Inactive/Archived)
- Model comparison view

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- CUDA compatible GPU (or CPU fallback)
- Windows 10+ (or Linux/Mac with adjusted paths)

### 2. Installation

```bash
cd "c:\Users\Acer\Desktop\main project\apex2-worktree"

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 3. Start System

Option A: Use startup script

```bash
START_SYSTEM.bat
```

Option B: Manual start

```bash
# Terminal 1 - API Server
python run_api.py

# Terminal 2 - Frontend
streamlit run frontend\app_enhanced.py
```

### 4. Access

- **Frontend:** http://localhost:8501
- **API Docs:** http://localhost:8000/docs
- **API Root:** http://localhost:8000

---

## File Structure

```
apex2-worktree/
├── frontend/
│   └── app_enhanced.py              # Streamlit multi-phase dashboard
├── api/
│   └── main_enhanced.py             # FastAPI with 20+ endpoints
├── pipeline/
│   ├── training_orchestrator.py     # 7-phase workflow coordinator
│   ├── orchestrator.py              # Original orchestrator (legacy)
│   └── ...
├── data_ingestion/
│   ├── ingestion_manager.py         # Phase 1: Data Ingestion
│   ├── schema_detector.py           # Phase 2: Schema Detection
│   └── ...
├── preprocessing/
│   ├── image_preprocessor.py        # Phase 3: Image preprocessing
│   ├── text_preprocessor.py         # Phase 3: Text preprocessing
│   └── tabular_preprocessor.py      # Phase 3: Tabular preprocessing
├── automl/
│   ├── advanced_selector.py         # Phase 4: Model Selection
│   └── ...
├── monitoring/
│   ├── drift_detector.py            # Phase 6: Drift Detection
│   └── performance_tracker.py       # Phase 6: Monitoring
├── model_registry_pkg/
│   └── model_registry.py            # Phase 7: Model Registry
├── config/
│   └── hyperparameters.py           # Configuration management
├── requirements.txt                 # Python dependencies
├── START_SYSTEM.bat                 # Quick start script
└── INTEGRATION_GUIDE.md             # This file
```

---

## New Modules Created This Session

### 1. Data Ingestion Manager (`data_ingestion/ingestion_manager.py`)

- **Lines:** 236
- **Purpose:** Multi-source data loading with caching
- **Key Methods:**
  - `ingest_data()` - Load from multiple sources
  - `_download_file()` - HTTP download with progress
  - `_ingest_kaggle()` - Kaggle dataset loader
  - `_save_to_cache()` - Persist to cache
  - `get_cache_info()` - Cache statistics
  - `clear_cache()` - Cache management

### 2. Schema Detector (`data_ingestion/schema_detector.py`)

- **Lines:** 285
- **Purpose:** Automatic column detection and problem type inference
- **Key Methods:**
  - `SchemaDetector.detect()` - Single dataset detection
  - `MultiDatasetSchemaDetector.merge()` - Multi-dataset schema merging
  - Fuzzy matching for target column
  - Modality inference (image/text/tabular)

### 3. Advanced Model Selector (`automl/advanced_selector.py`)

- **Lines:** 340
- **Purpose:** GPU-aware model selection with rationale
- **Key Methods:**
  - `select_models()` - Main selection logic
  - `_get_gpu_memory()` - Hardware detection
  - `_select_image_encoder()` - Image model selection
  - `_compute_batch_size_and_epochs()` - Hyperparameter calculation

### 4. Training Orchestrator (`pipeline/training_orchestrator.py`)

- **Lines:** 628
- **Purpose:** Coordinate all 7 phases end-to-end
- **Key Methods:**
  - `run_pipeline()` - Execute complete workflow
  - `_execute_phase_1/2/.../7()` - Phase implementations
  - Real-time phase tracking and logging
  - GPU memory monitoring

### 5. Enhanced Frontend (`frontend/app_enhanced.py`)

- **Lines:** 750+
- **Purpose:** Multi-page Streamlit dashboard
- **Pages:**
  - Phase 1: Data Ingestion with cache visualization
  - Phase 2: Schema Detection with confidence scores
  - Phase 3: Preprocessing progress tracking
  - Phase 4: Model Selection with detailed rationale
  - Phase 5: Training with real-time metrics
  - Phase 6: Monitoring and drift detection

### 6. Extended API (`api/main_enhanced.py`)

- **New Endpoints:**
  - `POST /api/ingest` - Phase 1 endpoint
  - `POST /api/detect-schema` - Phase 2 endpoint
  - `POST /api/preprocess` - Phase 3 endpoint
  - `POST /api/select-model` - Phase 4 endpoint
  - `POST /api/train-pipeline` - Complete pipeline
  - `GET /api/pipeline-status/{job_id}` - Status tracking

### 7. Dependencies Added (`requirements.txt`)

```
fuzzywuzzy==0.18.0          # Fuzzy string matching for schema detection
python-Levenshtein==0.21.1  # Efficient edit distance calculation
```

---

## Configuration & Customization

### Adjust GPU Tiers

**File:** `automl/advanced_selector.py`

```python
# Current: <6GB, 6-12GB, >12GB
# Modify _select_*_encoder() methods for different thresholds
```

### Customize Drift Thresholds

**File:** `monitoring/drift_detector.py`

```python
PSI_THRESHOLD = 0.25          # Increase for less sensitivity
KS_THRESHOLD = 0.30
FEATURE_DRIFT_THRESHOLD = 0.50
```

### Change Preprocessing Parameters

**File:** `preprocessing/*.py`

```python
IMAGE_SIZE = (224, 224)       # Adjust image dimensions
TEXT_MAX_LENGTH = 128         # Adjust text tokenization length
```

---

## API Key Responses

### Health Check

```
GET /
Response:
{
  "status": "running",
  "version": "2.0.0",
  "pytorch": true,
  "gpu_available": true,
  "gpu_name": "NVIDIA RTX 4090",
  ...
}
```

### Phase 1 Response

```
POST /api/ingest
{
  "status": "success",
  "phase": "Phase 1: Data Ingestion",
  "data": {
    "sources": [...],
    "cache_hits": 1,
    "cache_misses": 1,
    "total_size_mb": 250
  }
}
```

### Phase 4 Response

```
POST /api/select-model
{
  "status": "success",
  "phase": "Phase 4: Model Selection",
  "data": {
    "selected_encoders": {
      "image": "ResNet50",
      "text": "BERT-base",
      "tabular": "TabNet"
    },
    "hyperparameters": {...},
    "selection_rationale": "..."
  }
}
```

---

## Troubleshooting

### API Connection Issues

**Error:** `❌ API Disconnected`

- **Solution:** Ensure API server running: `python run_api.py`
- **Check:** http://localhost:8000 in browser

### GPU Not Detected

**Error:** `⚠️ CPU Mode`

- **Solution:** Install PyTorch with CUDA: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

### NLTK Data Missing

**Error:** `LookupError: Resource punkt not found`

- **Solution:** Run: `python -c "import nltk; nltk.download('punkt')"`

### Port Already in Use

**Error:** `Address already in use`

- **Solution:** Kill process: `netstat -ano | findstr :8000` then `taskkill /PID <pid> /F`

---

## Performance Notes

- **Data Ingestion:** ~1-10 seconds per source (depends on size)
- **Schema Detection:** ~2-5 seconds (fuzzy matching overhead)
- **Preprocessing:** ~10-30 seconds (depends on dataset size)
- **Model Selection:** ~1-2 seconds (GPU detection + calculation)
- **Training (Phase 5):** ~5-30 minutes (depends on dataset size, GPU)
- **Drift Detection:** ~2-5 seconds (per batch of new data)

---

## What's Next?

### Phase 8 (Future): Automated Retraining

- Automatic retraining when drift detected
- A/B testing for model versions
- Gradual rollout strategy

### Phase 9 (Future): Model Deployment

- Export models to ONNX/TorchScript
- Docker containerization
- API scaling with multiple workers

### Phase 10 (Future): Advanced Features

- Hyperparameter optimization (Optuna)
- Neural architecture search (NAS)
- Explainability features (SHAP, attention visualization)

---

## Support & Documentation

- **API Docs:** http://localhost:8000/docs (Swagger UI)
- **GitHub:** https://github.com/hrishi-cz/main-project
- **Issues:** Report via GitHub Issues
- **Logs:** Check terminal output for detailed errors

---

**Version:** 2.0.0 - Full 7-Phase Integration  
**Last Updated:** February 10, 2024  
**Status:** ✅ Production Ready
