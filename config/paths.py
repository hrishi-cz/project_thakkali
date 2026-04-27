"""Canonical project paths shared across modules."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_CACHE_DIR = DATA_DIR / "dataset_cache"
EMBEDDING_CACHE_DIR = DATA_DIR / "embedding_cache"
SESSION_DB_PATH = DATA_DIR / "sessions.db"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_REGISTRY_DIR = MODELS_DIR / "registry"
RETRAIN_HISTORY_PATH = MODEL_REGISTRY_DIR / "_retrain_history.jsonl"

LOGS_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"

TASK_DB_PATH = PROJECT_ROOT / "tasks.db"
META_LEARNING_STORE = LOGS_DIR / "meta_learning.json"


def ensure_runtime_dirs() -> None:
    """Create runtime directories used by ingestion, registry, and reporting."""
    for path in (
        DATA_DIR,
        DATASET_CACHE_DIR,
        EMBEDDING_CACHE_DIR,
        MODELS_DIR,
        MODEL_REGISTRY_DIR,
        LOGS_DIR,
        REPORTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
