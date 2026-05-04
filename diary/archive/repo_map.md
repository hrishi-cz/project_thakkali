# AutoVision+ Repository Topology & Data Flow

This map defines the spatial layout, data flow, and dependency rules of the AutoVision+ platform.

## I. System Data Flow

AutoVision+ processes multimodal datasets through a strictly sequential pipeline. Do not bypass these layers.

Raw Sources
↓
data_ingestion/
↓
preprocessing/
↓
automl/jit_encoder_selector.py
↓
pipeline/training_orchestrator.py
↓
models/encoders + fusion
↓
automl/trainer.py
↓
model_registry_pkg/model_registry.py
↓
run_api.py
↓
frontend/app_enhanced.py

## II. Dependency Direction

Modules must only depend on lower layers. Upstream imports are strictly forbidden to prevent architecture erosion.
`data_ingestion` → `preprocessing` → `automl` → `pipeline` → `run_api` → `frontend`

## III. Repository Structure

```text
main-project/
├── run_api.py                     ⭐ System entrypoint (FastAPI / WebSocket)
├── task_store.py                  ⭐ Cross-worker task persistence (SQLite WAL)
├── requirements.txt
│
├── frontend/
│   └── app_enhanced.py
│
├── pipeline/
│   ├── training_orchestrator.py   ⭐ Core orchestration engine
│   ├── inference_engine.py
│   ├── dataset_manager.py
│   └── retraining_pipeline.py
│
├── automl/
│   ├── trainer.py
│   ├── jit_encoder_selector.py
│   ├── advanced_selector.py
│   └── model_selector.py
│
├── models/                        # (Note: Safely renamed from legacy 'modelss/')
│   ├── encoders/
│   │   ├── image.py
│   │   ├── text.py
│   │   └── tabular.py
│   ├── fusion.py                  ⭐ Modality fusion layer
│   └── predictor.py
│
├── preprocessing/
│   ├── tabular_preprocessor.py
│   ├── text_preprocessor.py
│   └── image_preprocessor.py
│
├── data_ingestion/
│   ├── ingestion_manager.py
│   ├── schema_detector.py
│   ├── loader.py
│   └── adapters/
│
├── monitoring/
│   ├── drift_detector.py
│   └── performance_tracker.py
│
├── config/
│   ├── hyperparameters.py
│   └── encoder_plugins.py
│
└── model_registry_pkg/
    └── model_registry.py
```
