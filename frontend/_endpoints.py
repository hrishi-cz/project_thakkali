"""Centralized API endpoint URL constants for the APEX frontend.

Usage::

    from frontend._endpoints import ep
    resp = requests.get(ep.intelligence(sid), timeout=5)
    resp = requests.post(ep.TRAIN_PIPELINE, json={...})
"""

from __future__ import annotations

import os
import streamlit as st


# ---------------------------------------------------------------------------
# Base URL (resolved once at import time)
# ---------------------------------------------------------------------------
def _secrets_get(key: str, default=None):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

API_BASE_URL: str = (
    os.getenv("APEX_API_BASE_URL")
    or _secrets_get("apex_api_base_url")
    or "http://localhost:8001"
)


# ---------------------------------------------------------------------------
# Static endpoints (no session ID needed)
# ---------------------------------------------------------------------------
HEALTH = f"{API_BASE_URL}/health"
ROOT = f"{API_BASE_URL}/"
TRAIN_PIPELINE = f"{API_BASE_URL}/train-pipeline"
MODEL_REGISTRY = f"{API_BASE_URL}/model-registry"


# ---------------------------------------------------------------------------
# Session-scoped endpoint builders
# ---------------------------------------------------------------------------

def session(sid: str, path: str = "") -> str:
    """Build ``/v2/sessions/{sid}/{path}`` URL."""
    base = f"{API_BASE_URL}/v2/sessions/{sid}"
    return f"{base}/{path}" if path else base


def intelligence(sid: str, sub: str = "") -> str:
    """Build ``/v2/sessions/{sid}/intelligence/{sub}`` URL."""
    base = f"{API_BASE_URL}/v2/sessions/{sid}/intelligence"
    return f"{base}/{sub}" if sub else base


def datasets(sid: str, ds_id: str = "") -> str:
    """Build ``/v2/sessions/{sid}/datasets/{ds_id}`` URL."""
    base = f"{API_BASE_URL}/v2/sessions/{sid}/datasets"
    return f"{base}/{ds_id}" if ds_id else base


def context_drift(sid: str) -> str:
    """Build ``/context/{sid}/drift-status`` URL."""
    return f"{API_BASE_URL}/context/{sid}/drift-status"


def train_status(task_id: str) -> str:
    """Build ``/train-pipeline/status/{task_id}`` URL."""
    return f"{API_BASE_URL}/train-pipeline/status/{task_id}"


def model_info(model_id: str) -> str:
    """Build ``/model-info/{model_id}`` URL."""
    return f"{API_BASE_URL}/model-info/{model_id}"


def predict() -> str:
    """Build ``/predict`` URL.  Pass ``model_id`` in the request body, not the URL."""
    return f"{API_BASE_URL}/predict"


def registered_models(sid: str) -> str:
    """Build ``/v2/sessions/{sid}/registered-models`` URL."""
    return f"{API_BASE_URL}/v2/sessions/{sid}/registered-models"


# ---------------------------------------------------------------------------
# Context endpoints (no V2 prefix)
# ---------------------------------------------------------------------------

def context_fit_analysis(sid: str) -> str:
    """Build ``/context/{sid}/fit-analysis`` URL."""
    return f"{API_BASE_URL}/context/{sid}/fit-analysis"


def context_phase_timings(sid: str) -> str:
    """Build ``/context/{sid}/phase-timings`` URL."""
    return f"{API_BASE_URL}/context/{sid}/phase-timings"


def context_probe_sample_save(sid: str) -> str:
    """Build ``/context/{sid}/probe-sample`` URL (POST — persists a probe sample)."""
    return f"{API_BASE_URL}/context/{sid}/probe-sample"


def context_probe_sample_status(sid: str) -> str:
    """Build ``/context/{sid}/probe-sample/status`` URL (GET — check if sample exists)."""
    return f"{API_BASE_URL}/context/{sid}/probe-sample/status"


def decision_trace(sid: str) -> str:
    """Build ``/v2/sessions/{sid}/decision-trace`` URL."""
    return f"{API_BASE_URL}/v2/sessions/{sid}/decision-trace"


def global_target(sid: str) -> str:
    """Build ``/v2/sessions/{sid}/global-target`` URL."""
    return f"{API_BASE_URL}/v2/sessions/{sid}/global-target"


def global_schema(sid: str) -> str:
    """Build ``/v2/sessions/{sid}/global-schema`` URL."""
    return f"{API_BASE_URL}/v2/sessions/{sid}/global-schema"


def dataset_target_candidates(dataset_id: str) -> str:
    """Build ``/v2/datasets/{dataset_id}/target-candidates`` URL."""
    return f"{API_BASE_URL}/v2/datasets/{dataset_id}/target-candidates"


def dataset_lock_target(dataset_id: str) -> str:
    """Build ``/v2/datasets/{dataset_id}/lock-target`` URL."""
    return f"{API_BASE_URL}/v2/datasets/{dataset_id}/lock-target"


def dataset_unlock_target(dataset_id: str) -> str:
    """Build ``/v2/datasets/{dataset_id}/unlock-target`` URL."""
    return f"{API_BASE_URL}/v2/datasets/{dataset_id}/unlock-target"


def dataset_override_schema(dataset_id: str) -> str:
    """Build ``/v2/datasets/{dataset_id}/override-schema`` URL."""
    return f"{API_BASE_URL}/v2/datasets/{dataset_id}/override-schema"


def model_stats(model_id: str) -> str:
    """Build ``/models/{model_id}/stats`` URL."""
    return f"{API_BASE_URL}/models/{model_id}/stats"


def override_target_per_modality(sid: str) -> str:
    """Build ``/v2/sessions/{sid}/override-target-per-modality`` URL."""
    return f"{API_BASE_URL}/v2/sessions/{sid}/override-target-per-modality"


# Static endpoints for experiments
RUN_ABLATIONS = f"{API_BASE_URL}/experiments/run-ablations"
ABLATION_RESULTS = f"{API_BASE_URL}/experiments/ablation-results"
RETRAIN_HISTORY = f"{API_BASE_URL}/retrain-history"
CACHE_STATS = f"{API_BASE_URL}/cache/stats"
CACHE_METADATA = f"{API_BASE_URL}/cache/metadata"
CACHE_CLEAR = f"{API_BASE_URL}/cache/clear"
CONFIG = f"{API_BASE_URL}/config"
PREDICT_ASYNC = f"{API_BASE_URL}/predict-async"

# ---------------------------------------------------------------------------
# Previously missing static endpoints (added to fix audit gaps)
# ---------------------------------------------------------------------------
SESSIONS              = f"{API_BASE_URL}/v2/sessions"
MONITOR               = f"{API_BASE_URL}/monitor"
MONITOR_DRIFT         = f"{API_BASE_URL}/monitor/drift"
SCHEMA_DETECT         = f"{API_BASE_URL}/api/schema/detect"
SCHEMA_OVERRIDE       = f"{API_BASE_URL}/api/schema/override"
PREPROCESS            = f"{API_BASE_URL}/preprocess"
SELECT_MODEL          = f"{API_BASE_URL}/select-model"
EMBEDDING_CACHE_STATS = f"{API_BASE_URL}/embedding-cache/stats"
META_LEARNING_INSIGHTS= f"{API_BASE_URL}/meta-learning/insights"


# Previously missing dynamic builders (added to fix audit gaps)
def ingest_status(task_id: str) -> str:
    """Build ``/ingest/status/{task_id}`` URL."""
    return f"{API_BASE_URL}/ingest/status/{task_id}"


def task_status(task_id: str) -> str:
    """Build ``/task/{task_id}`` URL (polls async task results)."""
    return f"{API_BASE_URL}/task/{task_id}"


def encoder_overrides(sid: str) -> str:
    """Build ``/v2/sessions/{sid}/encoder-overrides`` URL."""
    return f"{API_BASE_URL}/v2/sessions/{sid}/encoder-overrides"


def override_fusion(sid: str) -> str:
    """Build ``/v2/sessions/{sid}/override-fusion`` URL."""
    return f"{API_BASE_URL}/v2/sessions/{sid}/override-fusion"


def override_global_target(sid: str) -> str:
    """Build ``/v2/sessions/{sid}/override-global-target`` URL."""
    return f"{API_BASE_URL}/v2/sessions/{sid}/override-global-target"


def active_model(sid: str) -> str:
    """Build ``/v2/sessions/{sid}/active-model`` URL."""
    return f"{API_BASE_URL}/v2/sessions/{sid}/active-model"


def choose_primary_dataset(sid: str) -> str:
    """Build ``/v2/sessions/{sid}/choose-primary-dataset`` URL."""
    return f"{API_BASE_URL}/v2/sessions/{sid}/choose-primary-dataset"


# ---------------------------------------------------------------------------
# Convenience namespace (use as ep.intelligence(sid) etc.)
# ---------------------------------------------------------------------------

class _Endpoints:
    """Namespace for all endpoint helpers, importable as ``from frontend._endpoints import ep``."""
    API_BASE_URL           = API_BASE_URL
    HEALTH                 = HEALTH
    ROOT                   = ROOT
    TRAIN_PIPELINE         = TRAIN_PIPELINE
    MODEL_REGISTRY         = MODEL_REGISTRY
    RUN_ABLATIONS          = RUN_ABLATIONS
    ABLATION_RESULTS       = ABLATION_RESULTS
    RETRAIN_HISTORY        = RETRAIN_HISTORY
    CACHE_STATS            = CACHE_STATS
    CACHE_METADATA         = CACHE_METADATA
    CACHE_CLEAR            = CACHE_CLEAR
    CONFIG                 = CONFIG
    PREDICT_ASYNC          = PREDICT_ASYNC
    # Previously missing statics
    SESSIONS               = SESSIONS
    MONITOR                = MONITOR
    MONITOR_DRIFT          = MONITOR_DRIFT
    SCHEMA_DETECT          = SCHEMA_DETECT
    SCHEMA_OVERRIDE        = SCHEMA_OVERRIDE
    PREPROCESS             = PREPROCESS
    SELECT_MODEL           = SELECT_MODEL
    EMBEDDING_CACHE_STATS  = EMBEDDING_CACHE_STATS
    META_LEARNING_INSIGHTS = META_LEARNING_INSIGHTS

    session                      = staticmethod(session)
    intelligence                 = staticmethod(intelligence)
    datasets                     = staticmethod(datasets)
    context_drift                = staticmethod(context_drift)
    context_fit_analysis         = staticmethod(context_fit_analysis)
    context_phase_timings        = staticmethod(context_phase_timings)
    context_probe_sample_save    = staticmethod(context_probe_sample_save)
    context_probe_sample_status  = staticmethod(context_probe_sample_status)
    train_status                 = staticmethod(train_status)
    ingest_status                = staticmethod(ingest_status)
    task_status                  = staticmethod(task_status)
    model_info                   = staticmethod(model_info)
    model_stats                  = staticmethod(model_stats)
    predict                      = staticmethod(predict)
    registered_models            = staticmethod(registered_models)
    decision_trace               = staticmethod(decision_trace)
    global_target                = staticmethod(global_target)
    global_schema                = staticmethod(global_schema)
    dataset_target_candidates    = staticmethod(dataset_target_candidates)
    dataset_lock_target          = staticmethod(dataset_lock_target)
    dataset_unlock_target        = staticmethod(dataset_unlock_target)
    dataset_override_schema      = staticmethod(dataset_override_schema)
    override_target_per_modality = staticmethod(override_target_per_modality)
    encoder_overrides            = staticmethod(encoder_overrides)
    override_fusion              = staticmethod(override_fusion)
    override_global_target       = staticmethod(override_global_target)
    active_model                 = staticmethod(active_model)
    choose_primary_dataset       = staticmethod(choose_primary_dataset)


ep = _Endpoints()

