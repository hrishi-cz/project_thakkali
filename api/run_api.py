"""AutoVision API – production-grade FastAPI entrypoint."""

import os
# Must be set before any torch/CUDA import. Suppresses CuBLAS non-determinism
# warnings when torch.use_deterministic_algorithms(True) is active on CUDA >= 10.2.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import asyncio
import io
import json
import collections
import logging
import sys
import threading
import uuid
import time
import zipfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Import path setup (must run before local package imports)
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load .env from project root — must happen before any os.getenv calls
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(project_root / ".env", override=False)
except ImportError:
    pass

from contextlib import asynccontextmanager
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
try:
    from pydantic import ConfigDict as _PydanticConfigDict
    _PYDANTIC_V2 = True
except ImportError:
    _PYDANTIC_V2 = False
import uvicorn
import pandas as pd
import numpy as np
import torch
import os
import re as _re

from config.paths import DATASET_CACHE_DIR, MODEL_REGISTRY_DIR
from core.context_enforcer import (
    ContextValidationError,
    ContextValidator,
    ensure_session_context,
    require_context,
)
from guardrails.latency_guard import LatencyGuard
from guardrails.memory_guard import MemoryGuard
from guardrails.session_isolator import SessionIsolator
from task_store import task_db, IngestionProgressTracker, TrainingProgressTracker

# Suppress numpy divide/invalid warnings globally — these come from Pearson
# correlation on zero-variance columns in the schema detector and are expected.
import warnings as _warnings
_warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
_warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in true_divide")
# Suppress HuggingFace sentencepiece byte-fallback tokenizer conversion noise
_warnings.filterwarnings("ignore", category=UserWarning, message=".*sentencepiece tokenizer.*byte fallback.*")

# Load user-registered encoder plugins (safe import -- file may be empty)
try:
    import config.encoder_plugins  # noqa: F401
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------
@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Startup/shutdown lifecycle handler."""
    try:
        from database.context_db import ContextDatabase as _CDB
        _cdb = _CDB()
        deleted = _cdb.cleanup_stale_sessions(max_age_hours=48)
        if deleted:
            logger.info("Startup cleanup: removed %d stale sessions", deleted)
    except Exception as _exc:
        logger.debug("Startup session cleanup skipped: %s", _exc)
    yield  # application runs here


app = FastAPI(title="AutoVision API", version="2.0.0", lifespan=_lifespan)


_cors_origins_env = os.getenv(
    "APEX_CORS_ORIGINS",
    "http://localhost:8501,http://127.0.0.1:8501",
).strip()
_cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------
GPU_AVAILABLE: bool = torch.cuda.is_available()
GPU_DEVICE: str = torch.cuda.get_device_name(0) if GPU_AVAILABLE else "CPU"
_session_isolator = SessionIsolator()

# ---------------------------------------------------------------------------
# Input sanitization
# ---------------------------------------------------------------------------
_SAFE_MODEL_ID = _re.compile(r"^[\w\-.:]+$")
_SAFE_MODEL_ALIAS = _re.compile(r"^[A-Za-z0-9_-]{1,128}$")
ALLOW_LEGACY_SESSION_FALLBACK: bool = (
    os.getenv("APEX_ALLOW_LEGACY_SESSION_FALLBACK", "0").strip() == "1"
)
# G6: hard-block legacy fallback in production. Operator must explicitly set
# APEX_MODE=development to use legacy in-memory session state.
APEX_MODE: str = os.getenv("APEX_MODE", "production").strip().lower()
if ALLOW_LEGACY_SESSION_FALLBACK and APEX_MODE == "production":
    raise RuntimeError(
        "Legacy session fallback (APEX_ALLOW_LEGACY_SESSION_FALLBACK=1) is "
        "disabled in production mode. Either unset the env var or set "
        "APEX_MODE=development."
    )

# ---------------------------------------------------------------------------
# Session dataset loader (FIX: remove duplication)
# ---------------------------------------------------------------------------

def get_session_datasets(session_id: str) -> Dict[str, Any]:
    """
    Load lazy datasets for a session (single source of truth).
    
    Uses context_db (thread-safe, no race conditions).
    """
    from data_ingestion.loader import DataLoader

    # Get session context from DB (thread-safe)
    ctx = session_manager.get_session(session_id)
    if not ctx or not ctx.active_dataset_ids:
        return {}

    cache_dir = DATASET_CACHE_DIR
    loader = DataLoader()

    datasets = {}

    for dataset_id in ctx.active_dataset_ids:
        # Get profile to find hash/path
        profile = context_db.load_profile(dataset_id)
        if not profile:
            logger.warning("Profile not found for dataset %s", dataset_id)
            continue
        
        # Load from cache using hash or path
        hash_id = profile.get('dataset_id')  # dataset_id is the hash
        cache_path = cache_dir / hash_id
        lazy_ref = loader.load_cached(cache_path)

        if lazy_ref is not None:
            datasets[hash_id] = lazy_ref
        else:
            logger.warning("Cache miss for hash %s", hash_id)

    return datasets

def _to_json_safe(obj):
    """Recursively convert numpy scalars/bools/arrays to Python-native JSON-serializable types.

    FastAPI's jsonable_encoder cannot handle numpy.bool_, numpy.int64, etc.
    Call this on any dict/list that may contain values derived from pandas/numpy
    before returning it from an endpoint.
    """
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _sanitize_model_id(model_id: str) -> str:
    """Validate model_id to prevent directory traversal attacks."""
    if not _SAFE_MODEL_ID.match(model_id) or ".." in model_id:
        raise HTTPException(
            status_code=400,
            detail="Invalid model_id: contains disallowed characters.",
        )
    return model_id


def _sanitize_model_alias(alias: str) -> str:
    """Validate alias used by model-registry rename endpoint."""
    cleaned = str(alias or "").strip()
    if not _SAFE_MODEL_ALIAS.match(cleaned):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid alias. Use 1-128 characters from: "
                "letters, digits, hyphen, underscore."
            ),
        )
    return cleaned


def _resolve_xai_target(target_class: int, result: Dict[str, Any]) -> int:
    """Resolve -1 sentinel to argmax of first prediction's confidence."""
    if target_class >= 0:
        return target_class
    preds = result.get("predictions", [])
    confs = result.get("confidences", [])
    if not preds:
        return 0
    first_pred = preds[0]
    if isinstance(first_pred, int):
        return first_pred
    if isinstance(first_pred, list) and confs:
        first_conf = confs[0] if isinstance(confs[0], list) else confs
        return int(max(range(len(first_conf)), key=lambda i: first_conf[i]))
    return 0


def _get_session_hashes(session_id: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Return ingested hashes for a session.

    Uses context_db (thread-safe, no race conditions).
    If session_id is provided, returns datasets from that session.
    If session_id is missing, returns an empty mapping.
    """
    if not session_id:
        return {}

    result: Dict[str, Dict[str, Any]] = {}
    ctx = session_manager.get_session(session_id)

    # Session-scoped access is context-first by design.
    # If context is missing, do not leak legacy in-memory state.
    if ctx is None:
        logger.warning(
            "_get_session_hashes: missing ExecutionContext for session %s",
            session_id,
        )
        return {}

    # Build dict from persisted dataset profiles.
    for dataset_id in ctx.active_dataset_ids:
        profile = context_db.load_profile(dataset_id)
        if profile:
            result[dataset_id] = profile

    if result:
        return _session_isolator.validate(session_id, result, ctx)

    # Compatibility fallback for legacy sessions can be enabled explicitly.
    if ALLOW_LEGACY_SESSION_FALLBACK:
        with _session_lock:
            legacy = _session_store.get(session_id, {}).get("datasets", {})
            if isinstance(legacy, dict) and legacy:
                logger.warning(
                    "_get_session_hashes: using legacy in-memory dataset map for session %s",
                    session_id,
                )
                return _session_isolator.validate(session_id, dict(legacy), ctx)
    else:
        logger.info(
            "_get_session_hashes: persisted dataset profiles unavailable for %s; "
            "legacy fallback disabled",
            session_id,
        )

    return {}


def _write_legacy_session_cache(session_id: Optional[str], key: str, value: Any) -> None:
    """Write to in-memory session cache only when compatibility mode is enabled."""
    if not session_id or not ALLOW_LEGACY_SESSION_FALLBACK:
        return
    with _session_lock:
        bucket = _session_store.setdefault(session_id, {"datasets": {}, "schema": None})
        bucket[key] = value


def _read_from_context_or_store(
    session_id: Optional[str],
    key: str,
    ctx_attr: Optional[str] = None,
) -> Any:
    """
    Read state from ExecutionContext first, then fall back to session store.

    ExecutionContext is the canonical source for persistent session state.
    _session_store remains a best-effort in-memory cache.
    """
    if not session_id:
        return None

    attr_name = ctx_attr or key

    try:
        ctx = session_manager.get_session(session_id)
        if ctx is not None:
            value = getattr(ctx, attr_name, None)
            if value is not None and (
                not isinstance(value, (dict, list, tuple, set)) or len(value) > 0
            ):
                return value
    except Exception:
        pass

    if not ALLOW_LEGACY_SESSION_FALLBACK:
        return None

    with _session_lock:
        return _session_store.get(session_id, {}).get(key)


def _read_context_artifact(ctx: Any, artifact_key: str) -> Any:
    """Read dotted artifact keys from ExecutionContext-like objects."""
    current: Any = ctx
    for part in str(artifact_key).split("."):
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part)
            continue
        current = getattr(current, part, None)
    return current


def _is_missing_context_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, bytes)):
        return len(value) == 0
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    return False


def _get_session_context_or_422(session_id: Optional[str], stage: str) -> Any:
    """Load ExecutionContext for a request stage.

    Raises 422 if session_id is absent (malformed request).
    Raises 404 if the session doesn't exist in the DB yet (not ingested).
    Raises 422 if the session exists but has no ExecutionContext (pipeline not started).
    """
    if not session_id:
        raise HTTPException(status_code=422, detail="session_id is required.")
    ctx = session_manager.get_session(str(session_id))
    if ctx is None:
        # Check whether the session row exists at all — 404 vs 422 distinction
        try:
            from database.context_db import ContextDatabase as _CDB
            _row_exists = _CDB().session_exists(session_id)
        except Exception:
            _row_exists = False
        if not _row_exists:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{session_id}' not found. Ingest a dataset first.",
            )
        raise HTTPException(
            status_code=422,
            detail=(
                f"No ExecutionContext for session {session_id} at stage '{stage}'. "
                "Run ingestion and schema detection first."
            ),
        )
    return ctx


def _require_context_artifact(session_id: Optional[str], artifact_key: str, stage: str) -> Any:
    """Require a non-empty artifact from session ExecutionContext."""
    ctx = _get_session_context_or_422(session_id, stage)
    value = _read_context_artifact(ctx, artifact_key)
    if _is_missing_context_value(value):
        raise HTTPException(
            status_code=422,
            detail=(
                f"ExecutionContext missing required artifact '{artifact_key}' "
                f"for stage '{stage}'."
            ),
        )
    return value


def _context_contract_payload(ctx: Optional[Any]) -> Dict[str, Any]:
    """Return standard context contract fields for API responses."""
    if ctx is None:
        return {
            "context_stage": None,
            "context_version": None,
            "artifact_versions": {},
            "fusion_policy_locked": False,
            "fusion_policy_source": None,
        }

    artifact_versions = dict(getattr(ctx, "artifact_versions", {}) or {})
    preprocess_plan_version = getattr(ctx, "preprocess_plan_version", None)
    if preprocess_plan_version and "preprocessing_plan" not in artifact_versions:
        artifact_versions["preprocessing_plan"] = str(preprocess_plan_version)

    return {
        "context_stage": getattr(ctx, "pipeline_stage", None),
        "context_version": getattr(ctx, "version", None),
        "artifact_versions": artifact_versions,
        "fusion_policy_locked": bool(getattr(ctx, "fusion_policy_locked", False)),
        "fusion_policy_source": getattr(ctx, "fusion_policy_source", None),
    }


_MODALITY_ORDER = ("image", "text", "tabular", "timeseries")
_IMAGE_REQUEST_KEYS = ("img", "image", "image_path", "img_path", "file_path", "url")


def _ordered_modalities(modalities: Any) -> List[str]:
    """Return stable, de-duplicated modality names in frontend-friendly order."""
    seen = {str(m).strip().lower() for m in list(modalities or []) if str(m).strip()}
    ordered = [m for m in _MODALITY_ORDER if m in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def _canonical_fusion_strategy(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "concat": "concatenation",
        "concatenationfusion": "concatenation",
        "attentionfusion": "attention",
        "unifiedlatentfusion": "ula",
        "unified_latent": "ula",
        "unified_latent_alignment": "ula",
        "omnimodal": "ula",
        "gatedfusion": "gated",
        "gated_fusion": "gated",
    }
    return aliases.get(raw, raw)


def _is_probably_id_or_path_column(name: Any) -> bool:
    lowered = str(name or "").strip().lower()
    if lowered in {"id", "idx", "index", "row_id", "uuid", "guid"}:
        return True
    return any(tok in lowered for tok in ("path", "file", "filename", "url", "uri"))


def _infer_problem_type_from_values(values: Any, default: str = "classification_binary") -> str:
    """Infer binary/multiclass/regression from concrete target values."""
    try:
        series = pd.Series(values).dropna()
        if series.empty:
            return default
        n_unique = int(series.nunique(dropna=True))
        if n_unique == 2:
            return "classification_binary"
        dtype = str(series.dtype).lower()
        if ("float" in dtype or "double" in dtype) and n_unique > 20:
            return "regression"
        if n_unique <= 50:
            return "classification_multiclass"
        return "regression"
    except Exception:
        return default


def _materialize_session_head(session_id: Optional[str], max_rows: int = 2000) -> pd.DataFrame:
    """Best-effort materialisation of the current session into a small pandas frame."""
    if not session_id:
        return pd.DataFrame()
    frames: List[pd.DataFrame] = []
    try:
        for lazy_ref in (get_session_datasets(str(session_id)) or {}).values():
            try:
                import polars as _pl
                if isinstance(lazy_ref, _pl.LazyFrame):
                    frames.append(lazy_ref.head(max_rows).collect().to_pandas())
                    continue
                if isinstance(lazy_ref, _pl.DataFrame):
                    frames.append(lazy_ref.head(max_rows).to_pandas())
                    continue
            except Exception:
                pass
            try:
                import dask.dataframe as _dd
                if isinstance(lazy_ref, _dd.DataFrame):
                    frames.append(lazy_ref.head(max_rows, compute=True))
                    continue
            except Exception:
                pass
            if isinstance(lazy_ref, pd.DataFrame):
                frames.append(lazy_ref.head(max_rows))
                continue
            if hasattr(lazy_ref, "head"):
                head = lazy_ref.head(max_rows)
                frames.append(head if isinstance(head, pd.DataFrame) else pd.DataFrame(head))
    except Exception as exc:
        logger.debug("Session head materialisation skipped: %s", exc)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _refresh_problem_type_from_target(
    session_id: Optional[str],
    ctx: Optional[Any],
    target_col: Optional[str],
    default: str = "classification_binary",
) -> str:
    """Persist a corrected problem type based on the final chosen target."""
    if ctx is None or not target_col:
        return default
    schema = dict(getattr(ctx, "global_schema", {}) or {})
    current = str(schema.get("global_problem_type") or default)
    frame = _materialize_session_head(session_id)
    inferred = current or default
    if not frame.empty and target_col in frame.columns:
        inferred = _infer_problem_type_from_values(frame[target_col], default=current or default)
    schema["primary_target"] = target_col
    schema["global_problem_type"] = inferred
    schema["problem_type"] = inferred
    for ds in schema.get("per_dataset", []) or []:
        if isinstance(ds, dict):
            ds["target_column"] = target_col
            ds["problem_type"] = inferred
    try:
        ctx.global_schema = schema
        ctx.global_target = target_col
        setattr(ctx, "global_problem_type", inferred)
        if hasattr(ctx, "log_decision"):
            ctx.log_decision(
                "target_sanity",
                f"Target '{target_col}' problem type resolved as {inferred}",
                "Final target values inspected after overrides.",
            )
        ctx.update_timestamp()
    except Exception:
        pass
    return inferred


def _resolve_active_modality_contract(ctx: Optional[Any], schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Resolve detected/active/excluded modalities from schema + preprocessing context."""
    schema = dict(schema or getattr(ctx, "global_schema", {}) or {})
    detected = _ordered_modalities(schema.get("global_modalities", []))
    per_ds = list(schema.get("per_dataset", []) or [])
    text_cols: List[str] = []
    image_cols: List[str] = []
    tabular_cols: List[str] = []
    id_like: set[str] = set()
    target_col = str(
        schema.get("primary_target")
        or schema.get("target_column")
        or getattr(ctx, "global_target", "")
        or ""
    )
    for ds in per_ds:
        if not isinstance(ds, dict):
            continue
        det = dict(ds.get("detected_columns", {}) or {})
        text_cols.extend([str(c) for c in det.get("text", []) or []])
        image_cols.extend([str(c) for c in det.get("image", []) or []])
        tabular_cols.extend([str(c) for c in det.get("tabular", []) or []])
        id_like.update(str(c) for c in ds.get("id_like_columns", []) or [])
    text_cols = list(dict.fromkeys(text_cols))
    image_cols = list(dict.fromkeys(image_cols))
    tabular_cols = list(dict.fromkeys(tabular_cols))

    if not detected:
        detected = _ordered_modalities(
            ["image"] * bool(image_cols)
            + ["text"] * bool(text_cols)
            + ["tabular"] * bool(tabular_cols)
        )

    excluded = dict(getattr(ctx, "excluded_modalities", {}) or {}) if ctx is not None else {}
    active = _ordered_modalities(getattr(ctx, "active_modalities", []) if ctx is not None else [])
    if not active:
        active = list(detected)

    if "tabular" in active or "tabular" in detected:
        usable_tab = [
            c for c in tabular_cols
            if c != target_col and c not in id_like and not _is_probably_id_or_path_column(c)
        ]
        if tabular_cols and not usable_tab:
            active = [m for m in active if m != "tabular"]
            excluded["tabular"] = "id_only_or_target_only"

    if text_cols and "text" not in active:
        active.append("text")
    if image_cols and "image" not in active:
        active.append("image")

    return {
        "detected_modalities": _ordered_modalities(detected),
        "active_modalities": _ordered_modalities(active),
        "excluded_modalities": excluded,
        "target_column": target_col,
        "text_columns": text_cols,
        "image_columns": image_cols,
        "tabular_columns": tabular_cols,
    }


def _apply_active_modality_contract(ctx: Optional[Any]) -> Dict[str, Any]:
    """Persist resolved active/excluded modality fields into ExecutionContext."""
    contract = _resolve_active_modality_contract(ctx)
    if ctx is not None:
        try:
            ctx.active_modalities = list(contract["active_modalities"])
            ctx.eligible_modalities = list(contract["active_modalities"])
            ctx.excluded_modalities = dict(contract["excluded_modalities"])
            if ctx.global_schema:
                ctx.global_schema["active_modalities"] = list(contract["active_modalities"])
                ctx.global_schema["excluded_modalities"] = dict(contract["excluded_modalities"])
            ctx.update_timestamp()
        except Exception:
            pass
    return contract


def _load_model_registry_metadata(model_id: str) -> Dict[str, Any]:
    metadata_path = MODEL_REGISTRY_DIR / str(model_id) / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        with open(metadata_path, encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _require_prediction_ready_model(ctx: Optional[Any], model_id: str, endpoint: str) -> Dict[str, Any]:
    """Require a registered deployment-ready model before monitoring/prediction."""
    metadata = _load_model_registry_metadata(model_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"{endpoint}: model '{model_id}' not found.")
    if not bool(metadata.get("deployment_ready", False)):
        raise HTTPException(
            status_code=409,
            detail=f"{endpoint}: model '{model_id}' is not deployment-ready.",
        )
    if ctx is not None:
        registered = {str(m) for m in list(getattr(ctx, "registered_model_ids", []) or [])}
        active = str(getattr(ctx, "active_prediction_model_id", "") or "")
        if registered and model_id not in registered and model_id != active:
            raise HTTPException(
                status_code=409,
                detail=f"{endpoint}: model '{model_id}' is not registered to this session.",
            )
    return metadata


def _not_available_monitor_payload(reason: str, ctx: Optional[Any], model_id: Optional[str] = None) -> Dict[str, Any]:
    contract = _context_contract_payload(ctx)
    return {
        "status": "not_available",
        "context_stage": contract["context_stage"],
        "context_version": contract["context_version"],
        "artifact_versions": contract["artifact_versions"],
        "data": {
            "availability": {
                "status": "not_available",
                "reason": reason,
                "model_id": model_id,
                "registered_model_ids": list(getattr(ctx, "registered_model_ids", []) or []) if ctx is not None else [],
                "active_model_id": getattr(ctx, "active_prediction_model_id", None) if ctx is not None else None,
            },
            "monitor": {
                "status": "not_available",
                "reason": reason,
                "severity": "unavailable",
                "breached_metrics": [],
                "retrain_recommendation": "unavailable",
                "model_id": model_id,
            },
            "drift": {
                "drift_detected": None,
                "metrics": {},
                "thresholds": {},
                "model_id": model_id,
            },
        },
    }


def _build_prediction_contract(
    *,
    model_id: str,
    metadata: Dict[str, Any],
    schema: Dict[str, Any],
    ctx: Optional[Any] = None,
    class_labels: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    modality_contract = _resolve_active_modality_contract(ctx, schema)
    training_signals = dict(metadata.get("training_signals", {}) or {})
    active = _ordered_modalities(
        training_signals.get("active_modalities")
        or modality_contract["active_modalities"]
        or metadata.get("modalities")
        or schema.get("global_modalities", [])
    )
    excluded = dict(modality_contract["excluded_modalities"])
    if "tabular" not in active and ("tabular" in modality_contract["detected_modalities"] or "tabular" in excluded):
        excluded.setdefault("tabular", "id_only_or_target_only")

    text_fields = list(modality_contract["text_columns"] or ["text"] if "text" in active else [])
    image_fields = list(modality_contract["image_columns"] or ["img"] if "image" in active else [])
    problem_type = str(
        schema.get("global_problem_type")
        or metadata.get("config", {}).get("problem_type")
        or training_signals.get("problem_type")
        or ""
    )
    fusion_meta = dict(metadata.get("fusion", {}) or {})
    phases_summary = dict(metadata.get("phases_summary", {}) or {})
    training_summary = dict(phases_summary.get("TRAINING", {}) or {})
    calibration_meta = dict(
        metadata.get("calibration")
        or training_summary.get("calibration")
        or training_signals.get("calibration")
        or {}
    )
    fusion_strategy = (
        fusion_meta.get("strategy")
        or training_signals.get("fusion_strategy_used")
        or getattr(ctx, "fusion_strategy", None)
    )
    fusion_strategy = _canonical_fusion_strategy(fusion_strategy)
    xai_meta = dict(metadata.get("xai", {}) or {})
    return {
        "model_id": model_id,
        "target": modality_contract["target_column"] or schema.get("primary_target", ""),
        "problem_type": problem_type,
        "class_labels": [str(v) for v in list(class_labels or [])],
        "active_modalities": active,
        "detected_modalities": modality_contract["detected_modalities"],
        "excluded_modalities": excluded,
        "input_columns": {
            "tabular": [] if "tabular" not in active else [
                c for c in modality_contract["tabular_columns"]
                if c != modality_contract["target_column"] and not _is_probably_id_or_path_column(c)
            ],
            "text": text_fields,
            "image": image_fields,
        },
        "accepted_image_request_keys": list(_IMAGE_REQUEST_KEYS),
        "required_modalities": [m for m in active if m in {"text", "image", "tabular"}],
        "fusion": {
            "strategy": str(fusion_strategy or ""),
            "metadata": fusion_meta,
        },
        "calibration": calibration_meta,
        "artifact_versions": dict(metadata.get("artifact_versions", {}) or {}),
        "xai_availability": {
            "tabular": bool(xai_meta.get("tabular")) or "tabular" in active,
            "text": bool(xai_meta.get("text")) or "text" in active,
            "image": bool(xai_meta.get("image")) or "image" in active,
            "fusion": str(fusion_strategy or "").lower() == "ula",
        },
    }


def _load_model_class_labels(model_id: str) -> List[Any]:
    enc_path = MODEL_REGISTRY_DIR / str(model_id) / "artifacts" / "target_encoder.joblib"
    if not enc_path.exists():
        return []
    try:
        import joblib
        enc = joblib.load(enc_path)
        if isinstance(enc, dict):
            return list(enc.get("all_labels", []) or [])
        if hasattr(enc, "classes_"):
            return list(enc.classes_)
    except Exception:
        return []
    return []


def _load_model_schema(model_id: str) -> Dict[str, Any]:
    schema_path = MODEL_REGISTRY_DIR / str(model_id) / "artifacts" / "schema.json"
    if not schema_path.exists():
        return {}
    try:
        with open(schema_path, encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _normalise_prediction_inputs(
    raw_inputs: List[Dict[str, Any]],
    prediction_contract: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Validate required inputs and add aliases consumed by inference_engine."""
    active = set(prediction_contract.get("active_modalities", []) or [])
    cols = prediction_contract.get("input_columns", {}) or {}
    text_cols = list(cols.get("text") or ["text"])
    image_cols = list(cols.get("image") or ["img"])
    tab_cols = list(cols.get("tabular") or [])
    warnings: List[str] = []
    consumed: List[Dict[str, Any]] = []
    normalised: List[Dict[str, Any]] = []
    missing: List[str] = []

    def _usable(value: Any) -> bool:
        if value is None:
            return False
        try:
            if pd.isna(value):
                return False
        except Exception:
            pass
        return str(value).strip().lower() not in {"", "nan", "none", "null", "<na>"}

    for idx, row in enumerate(raw_inputs):
        out = dict(row or {})
        row_used: Dict[str, Any] = {}
        if "text" in active:
            text_key = next((key for key in text_cols + ["text"] if key in out and _usable(out.get(key))), None)
            if not text_key:
                missing.append(f"row {idx}: required text field {text_cols or ['text']}")
            else:
                row_used["text"] = text_key
                out.setdefault("text", out[text_key])

        if "image" in active:
            image_candidates = image_cols + list(_IMAGE_REQUEST_KEYS)
            image_key = next((key for key in image_candidates if key in out and _usable(out.get(key))), None)
            if not image_key:
                missing.append(f"row {idx}: required image field {image_cols or list(_IMAGE_REQUEST_KEYS)}")
            else:
                row_used["image"] = image_key
                image_value = out[image_key]
                if image_cols:
                    out.setdefault(image_cols[0], image_value)
                out.setdefault("image_path", image_value)

        if "tabular" in active:
            missing_tab = [key for key in tab_cols if key not in out]
            if missing_tab:
                warnings.append(
                    f"row {idx}: missing optional tabular fields zero-filled: {missing_tab[:5]}"
                )
            row_used["tabular"] = [key for key in tab_cols if key in out]

        normalised.append(out)
        consumed.append(row_used)

    if missing:
        raise HTTPException(
            status_code=400,
            detail="Missing required prediction inputs: " + "; ".join(missing[:10]),
        )
    return normalised, {"consumed_inputs": consumed, "warnings": warnings}


def _prediction_output_payload(
    *,
    model_id: str,
    result: Dict[str, Any],
    explanations: Optional[Dict[str, Any]],
    prediction_contract: Dict[str, Any],
    io_audit: Dict[str, Any],
    ctx: Optional[Any],
) -> Dict[str, Any]:
    contract = _context_contract_payload(ctx)
    labels = list(prediction_contract.get("class_labels", []) or [])
    confidences = result.get("confidences", []) or []
    predictions = result.get("predictions", []) or []
    per_class: List[Any] = []
    if result.get("problem_type") == "classification_binary":
        for conf, pred in zip(confidences, predictions):
            if isinstance(conf, list):
                per_class.append(conf)
                continue
            try:
                p_conf = float(conf)
                pred_str = str(pred)
                positive = pred_str in {"1", "true", "True"} or (labels and pred_str == str(labels[-1]))
                p1 = p_conf if positive else 1.0 - p_conf
                per_class.append([round(1.0 - p1, 6), round(p1, 6)])
            except Exception:
                per_class.append(None)
    elif confidences and isinstance(confidences[0], list):
        per_class = confidences

    xai_payload = explanations
    if xai_payload is None:
        xai_payload = {
            "status": "not_requested",
            "availability": prediction_contract.get("xai_availability", {}),
        }
    else:
        xai_payload = dict(xai_payload)
        availability = {
            modality: bool(xai_payload.get(modality))
            for modality in ("tabular", "text", "image", "fusion")
        }
        reasons = {
            modality: "not returned by explainer"
            for modality, available in availability.items()
            if not available and modality in set(prediction_contract.get("active_modalities", []) or [])
        }
        xai_payload["status"] = "available" if any(availability.values()) else "unavailable"
        if reasons and any(availability.values()):
            xai_payload["status"] = "partial"
        xai_payload["availability"] = availability
        xai_payload["unavailable_reasons"] = reasons

    return {
        "status": "success",
        "model_id": model_id,
        "session_id": getattr(ctx, "session_id", None) if ctx is not None else None,
        "predictions": predictions,
        "predicted_class_ids": [str(p) for p in predictions],
        "predicted_class_labels": [str(p) for p in predictions],
        "confidences": confidences,
        "per_class_probabilities": per_class,
        "class_labels": labels,
        "problem_type": result.get("problem_type"),
        "n_samples": result.get("n_samples"),
        "active_modalities": prediction_contract.get("active_modalities", []),
        "fusion_strategy": (prediction_contract.get("fusion", {}) or {}).get("strategy"),
        "calibration": prediction_contract.get("calibration", {}),
        "calibration_applied": bool(prediction_contract.get("calibration")),
        "input_contract": prediction_contract,
        "consumed_inputs": io_audit.get("consumed_inputs", []),
        "warnings": io_audit.get("warnings", []),
        "explanations": xai_payload,
        "context_stage": contract["context_stage"],
        "context_version": contract["context_version"],
        "artifact_versions": contract["artifact_versions"],
    }


def _build_guardrail_snapshot(session_id: str, ctx: Optional[Any]) -> Dict[str, Any]:
    """Return a truthful guardrail snapshot for the current session."""
    memory_guard = MemoryGuard()
    vram_status = memory_guard.check_vram()
    ram_status = memory_guard.check_ram()

    active_dataset_ids = list(getattr(ctx, "active_dataset_ids", []) or []) if ctx is not None else []
    validated_hashes = _get_session_hashes(session_id)
    validated_count = len(validated_hashes)
    active_count = len(active_dataset_ids)

    if active_count == 0:
        isolation_status = "inactive"
    elif validated_count == active_count:
        isolation_status = "ok"
    else:
        isolation_status = "partial"

    overall_status = "ok"
    if isolation_status == "partial" or bool(vram_status.get("critical")) or bool(ram_status.get("critical")):
        overall_status = "attention_needed"

    return {
        "overall_status": overall_status,
        "latency": {
            "status": "configured",
            "implementation": "LatencyGuard.timed",
            "protected_endpoints": ["/predict-async", "/ws/predict"],
            "budgets_s": {
                "predict_async": 30.0,
                "ws_predict": 30.0,
            },
        },
        "memory": {
            "vram": vram_status,
            "ram": ram_status,
        },
        "session_isolation": {
            "status": isolation_status,
            "active_dataset_count": active_count,
            "validated_dataset_count": validated_count,
        },
    }


def _build_monitor_retraining_orchestrator(
    session_id: str,
    ctx: Any,
    snapshot: Dict[str, Dict[str, Any]],
) -> Optional[Any]:
    """Build a retraining orchestrator from session-backed drift inputs."""
    from pipeline.retraining_orchestrator import RetrainingOrchestrator

    production_sources: List[str] = []
    seen_sources: set[str] = set()

    for dataset_id, profile in dict(snapshot or {}).items():
        source_url = None
        if isinstance(profile, dict):
            source_url = profile.get("source_url") or profile.get("file_path") or profile.get("source")
        if not source_url and ctx is not None:
            dataset_profile = getattr(ctx, "dataset_profiles", {}).get(dataset_id)
            if dataset_profile is not None:
                source_url = getattr(dataset_profile, "source_url", None) or getattr(dataset_profile, "file_path", None)

        source_url = str(source_url or "").strip()
        if not source_url or source_url in seen_sources:
            continue

        seen_sources.add(source_url)
        production_sources.append(source_url)

    schema_info: Optional[Dict[str, Any]] = None
    if ctx is not None and getattr(ctx, "global_schema", None):
        schema_info = dict(getattr(ctx, "global_schema", {}) or {})

    problem_type = "classification_binary"
    if isinstance(schema_info, dict):
        problem_type = str(
            schema_info.get("global_problem_type")
            or schema_info.get("problem_type")
            or problem_type
        )

    training_signals = dict(getattr(ctx, "training_signals", {}) or {}) if ctx is not None else {}
    if not problem_type or problem_type == "classification_binary":
        problem_type = str(training_signals.get("problem_type") or problem_type)

    modalities: List[str] = []
    if ctx is not None:
        active_modalities = list(getattr(ctx, "active_modalities", []) or [])
        if active_modalities:
            modalities = active_modalities
        else:
            try:
                modalities = list(ctx.get_active_modalities())
            except Exception:
                modalities = []
    if not modalities and isinstance(schema_info, dict):
        modalities = list(schema_info.get("global_modalities", []) or [])
    if not modalities and ctx is not None:
        modalities = list(getattr(ctx, "modality_presence", {}).keys())
    if not modalities:
        modalities = ["tabular"]

    if not production_sources:
        return None

    return RetrainingOrchestrator(
        production_sources=production_sources,
        problem_type=problem_type,
        modalities=modalities,
        schema_info=schema_info,
        cooldown_seconds=3600,
        session_id=session_id,
        execution_context=ctx,
    )


def _sync_monitor_drift_to_context(
    session_id: Optional[str],
    ctx: Optional[Any],
    drift_data: Dict[str, Any],
) -> None:
    """Persist monitor/drift output into the active ExecutionContext."""
    if ctx is None:
        return

    try:
        metrics = dict(drift_data.get("metrics", {}) or {})
        retrain_info = dict(drift_data.get("retrain_info") or {})
        retrain_result = dict(retrain_info.get("result") or {}) if isinstance(retrain_info.get("result"), dict) else {}
        retrain_event = dict(retrain_info.get("event") or {}) if isinstance(retrain_info.get("event"), dict) else {}
        retrain_model_id = str(
            retrain_result.get("model_id")
            or retrain_event.get("model_id")
            or ""
        ).strip()
        retrain_deployment_ready = bool(
            retrain_result.get("deployment_ready")
            if retrain_result
            else retrain_event.get("deployment_ready", False)
        )

        ctx.update_drift(
            detected=bool(drift_data.get("drift_detected", False)),
            severity=float(drift_data.get("composite_score", 0.0) or 0.0),
            details={
                "ks": float(metrics.get("ks_statistic", 0.0) or 0.0),
                "psi": float(metrics.get("psi", 0.0) or 0.0),
                "mmd": float(metrics.get("fdd", 0.0) or 0.0),
                "composite": float(drift_data.get("composite_score", 0.0) or 0.0),
                "retrain_triggered": bool(drift_data.get("retrain_triggered", False)),
                "model_id": drift_data.get("model_id"),
                "retrain_info": retrain_info,
                "monitor": dict(drift_data.get("monitor", {}) or {}),
                "breached_metrics": list((drift_data.get("monitor", {}) or {}).get("breached_metrics", []) or []),
                "retrain_recommendation": (drift_data.get("monitor", {}) or {}).get("retrain_recommendation"),
            },
        )

        if hasattr(ctx, "apply_drift_feedback"):
            ctx.apply_drift_feedback(dict(drift_data or {}), decay=0.5)

        if retrain_model_id:
            if retrain_model_id not in getattr(ctx, "registered_model_ids", []):
                ctx.registered_model_ids.append(retrain_model_id)
            if retrain_deployment_ready:
                ctx.active_prediction_model_id = retrain_model_id
            if hasattr(ctx, "log_decision"):
                ctx.log_decision(
                    "model_registry",
                    f"Retrained model registered: {retrain_model_id}",
                    evidence=(
                        "active_prediction_model_id="
                        f"{getattr(ctx, 'active_prediction_model_id', None)}"
                    ),
                )

        ctx.set_pipeline_stage("monitoring")

        if session_id:
            session_manager.update_session_context(session_id, ctx)
    except OptimisticLockError:
        raise
    except Exception as exc:
        logger.warning("/monitor/drift: context sync failed: %s", exc)

# ---------------------------------------------------------------------------
# Pydantic models – defined BEFORE any endpoint that references them
# ---------------------------------------------------------------------------

class IngestionRequest(BaseModel):
    """Frontend contract for ingestion request."""
    dataset_urls: List[str]
    session_id: str


class IngestionResponse(BaseModel):
    """
    Frontend contract for ingestion response (Streamlit-compatible).

    The ``ingestion_progress`` dict MUST contain exactly:
      - status   : "success" | "partial" | "failed"
      - progress : int  0-100
      - message  : str  human-readable status
      - datasets : List[Dict] each with keys source, hash, shape, columns, status
    """
    ingestion_progress: Dict[str, Any]


class HPOverridesModel(BaseModel):
    """Validated hyperparameter overrides for /train-pipeline."""

    lr: Optional[float] = Field(None, gt=0.0, le=1.0, description="Learning rate (0, 1]")
    epochs: Optional[int] = Field(None, ge=1, le=500, description="Max training epochs [1, 500]")
    batch_size: Optional[int] = Field(None, ge=1, le=4096, description="Batch size [1, 4096]")
    dropout: Optional[float] = Field(None, ge=0.0, le=0.95, description="Dropout rate [0, 0.95]")
    weight_decay: Optional[float] = Field(None, ge=0.0, le=1.0, description="Weight decay [0, 1]")
    n_trials: Optional[int] = Field(None, ge=1, le=200, description="Optuna trials [1, 200]")

    if _PYDANTIC_V2:
        model_config = _PydanticConfigDict(extra="allow")
    else:
        class Config:
            extra = "allow"


def _validate_hp_overrides(raw: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Validate hp_overrides dict and raise HTTPException on invalid values."""
    if not raw:
        return raw
    try:
        validated = HPOverridesModel(**raw)
        return {k: v for k, v in validated.dict().items() if v is not None}
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid hp_overrides: {exc}",
        ) from exc


SUPPORTED_MODALITIES: frozenset = frozenset({"tabular", "text", "image", "timeseries"})


def _update_session_context_with_retry(
    session_id: str,
    ctx,
    max_retries: int = 3,
    backoff_s: float = 0.05,
):
    """Write session context with optimistic-lock retry.

    On OptimisticLockError, refresh the context from the DB, re-apply the
    caller's mutations via a shallow merge, and retry up to max_retries times.

    Returns the effective (possibly refreshed) ExecutionContext on success,
    or the original ctx on exhausted retries (best-effort — does not raise).
    Callers that need the canonical post-write context should use the return value.
    """
    import time as _time

    for attempt in range(max_retries):
        try:
            session_manager.update_session_context(session_id, ctx)
            return ctx
        except OptimisticLockError:
            if attempt >= max_retries - 1:
                logger.warning(
                    "_update_session_context_with_retry: gave up after %d attempts "
                    "for session=%s", max_retries, session_id,
                )
                return ctx
            _time.sleep(backoff_s * (2 ** attempt))
            # Refresh context from DB and re-apply the staged fields
            try:
                fresh = session_manager.get_session(session_id)
                if fresh is not None:
                    for _field in vars(ctx):
                        if _field == "revision":
                            continue
                        try:
                            setattr(fresh, _field, getattr(ctx, _field))
                        except Exception:
                            pass
                    ctx = fresh
            except Exception:
                pass
    return ctx


def _validate_modalities(modalities: List[str], endpoint: str) -> List[str]:
    """Reject unknown modality names, return validated list."""
    unknown = [m for m in modalities if m not in SUPPORTED_MODALITIES]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{endpoint}: unknown modalities {unknown}. "
                f"Supported: {sorted(SUPPORTED_MODALITIES)}"
            ),
        )
    return modalities


class ModelRegistryAliasRenameRequest(BaseModel):
    """Request body for alias-only rename in model registry."""

    new_name: str


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

# Session state is primarily handled by ContextDatabase.
# The in-memory store below is retained only for legacy endpoints that still
# expect ephemeral session schema/hash snapshots.
_session_store: Dict[str, Dict[str, Any]] = {}
_session_lock = threading.RLock()
_model_registry_lock = threading.RLock()
# session_id → task_id for the currently active training task (volatile, cleared on API restart)
_active_training_tasks: Dict[str, str] = {}
_active_training_lock = threading.Lock()

# Backward-compatible alias (used in /train-pipeline and other endpoints)
# NOTE:
# Do not keep a global cross-session ingestion hash cache here. Session state
# must remain strictly scoped to a provided session_id.

# Inference engine cache – avoids re-loading model weights on every /predict call.
# LRU eviction: oldest entry is dropped when the cache exceeds _MAX_ENGINES.
_MAX_ENGINES: int = 5
_engine_cache: collections.OrderedDict[str, Any] = collections.OrderedDict()
_engine_cache_lock = threading.Lock()






# ---------------------------------------------------------------------------
# Basic routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "message": "AutoVision API",
        "version": "2.0.0",
        "status": "running",
        "gpu_available": GPU_AVAILABLE,
        "device": GPU_DEVICE,
    }


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "service": "AutoVision API",
        "gpu_available": GPU_AVAILABLE,
        "device": GPU_DEVICE,
        "cuda_version": torch.version.cuda,
    }


@app.get("/config")
async def get_config() -> Dict[str, Any]:
    from config.hyperparameters import HyperparameterConfig
    return HyperparameterConfig().to_dict()


# ---------------------------------------------------------------------------
# Cache management endpoints
# ---------------------------------------------------------------------------

@app.get("/cache/stats")
async def cache_stats() -> Dict[str, Any]:
    """Return cache statistics for the dataset cache directory."""
    def _stats_sync() -> Dict[str, Any]:
        from data_ingestion.ingestion_manager import DataIngestionManager
        mgr = DataIngestionManager()
        info = mgr.get_cache_info()
        cache_dir = Path(info["cache_dir"])
        items: List[Dict[str, Any]] = []
        total_size_bytes: int = 0
        for entry in cache_dir.iterdir():
            if entry.name == "cache_metadata.json":
                continue
            if entry.is_dir():
                dir_size = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
            else:
                dir_size = entry.stat().st_size
            total_size_bytes += dir_size
            items.append({
                "name": entry.name,
                "size_mb": round(dir_size / (1024 * 1024), 2),
            })
        return {
            "cache_location": str(cache_dir),
            "total_items": info["total_cached"],
            "total_size_mb": round(total_size_bytes / (1024 * 1024), 2),
            "items": items,
        }
    return await asyncio.to_thread(_stats_sync)


@app.post("/cache/clear")
async def cache_clear() -> Dict[str, Any]:
    """Clear all cached datasets."""
    def _clear_sync() -> Dict[str, Any]:
        from data_ingestion.ingestion_manager import DataIngestionManager
        mgr = DataIngestionManager()
        mgr.clear_cache()
        return {"message": "Cache cleared successfully", "status": "success"}
    return await asyncio.to_thread(_clear_sync)


@app.get("/cache/metadata")
async def cache_metadata_endpoint() -> Dict[str, Any]:
    """Return the raw cache metadata mapping hash -> {source, timestamp, size_mb, ...}.
    Used by the frontend to display human-readable source URLs instead of bare hashes.
    """
    def _meta_sync() -> Dict[str, Any]:
        from data_ingestion.ingestion_manager import DataIngestionManager
        mgr = DataIngestionManager()
        return mgr.cache_metadata  # already a dict[hash -> {source, ...}]
    return await asyncio.to_thread(_meta_sync)


@app.get("/embedding-cache/stats")
async def embedding_cache_stats() -> Dict[str, Any]:
    """Return embedding cache filesystem stats plus latest training cache counters."""

    def _embedding_stats_sync() -> Dict[str, Any]:
        from config.paths import EMBEDDING_CACHE_DIR

        cache_dir = Path(EMBEDDING_CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)

        files = list(cache_dir.glob("*.pt"))
        total_size_bytes = sum(f.stat().st_size for f in files if f.is_file())

        latest_model_id: Optional[str] = None
        latest_created_at: str = ""
        latest_training_stats: Dict[str, Any] = {}

        if MODEL_REGISTRY_DIR.exists():
            for model_dir in MODEL_REGISTRY_DIR.iterdir():
                if not model_dir.is_dir():
                    continue
                meta_file = model_dir / "metadata.json"
                if not meta_file.exists():
                    continue
                try:
                    with open(meta_file, encoding="utf-8") as fh:
                        meta = json.load(fh)
                    created_at = str(meta.get("created_at") or "")
                    training = (meta.get("phases_summary", {}) or {}).get("TRAINING", {}) or {}
                    emb_stats = training.get("embedding_cache")
                    if isinstance(emb_stats, dict) and created_at >= latest_created_at:
                        latest_created_at = created_at
                        latest_model_id = str(meta.get("model_id") or model_dir.name)
                        latest_training_stats = emb_stats
                except Exception:
                    continue

        return {
            "status": "success",
            "cache_dir": str(cache_dir),
            "cache_file_count": len(files),
            "cache_size_mb": round(total_size_bytes / (1024 * 1024), 2),
            "latest_model_id": latest_model_id,
            "latest_training_embedding_cache": latest_training_stats,
        }

    return await asyncio.to_thread(_embedding_stats_sync)


@app.get("/meta-learning/insights")
async def meta_learning_insights(
    session_id: Optional[str] = None,
    dataset_size: Optional[int] = None,
    num_cols: Optional[int] = None,
    problem_type: Optional[str] = None,
    modalities: Optional[str] = None,
    top_k: int = Query(default=5, ge=1, le=20),
) -> Dict[str, Any]:
    """Return meta-learning suggestions and predicted config for current context."""

    def _build_dataset_meta_sync() -> Dict[str, Any]:
        inferred_problem = str(problem_type or "classification_binary")
        inferred_modalities = [
            m.strip()
            for m in str(modalities or "").split(",")
            if m and m.strip()
        ]
        inferred_rows = int(dataset_size or 0)
        inferred_cols = int(num_cols or 0)

        if session_id:
            ctx = session_manager.get_session(session_id)
            if ctx is not None:
                if not inferred_modalities:
                    inferred_modalities = (
                        list(ctx.active_modalities or [])
                        or list((ctx.modality_presence or {}).keys())
                        or list(ctx.get_active_modalities())
                    )

                if not problem_type:
                    global_schema = ctx.global_schema or {}
                    if isinstance(global_schema, dict):
                        inferred_problem = str(
                            global_schema.get("global_problem_type", inferred_problem)
                        )

                global_schema = ctx.global_schema or {}
                if isinstance(global_schema, dict):
                    if inferred_rows <= 0:
                        inferred_rows = int(
                            global_schema.get("total_samples")
                            or global_schema.get("n_rows")
                            or 0
                        )

                    if inferred_cols <= 0:
                        per_dataset = global_schema.get("per_dataset", [])
                        if isinstance(per_dataset, list) and per_dataset:
                            detected = per_dataset[0].get("detected_columns", {}) if isinstance(per_dataset[0], dict) else {}
                            if isinstance(detected, dict):
                                inferred_cols = sum(
                                    len(cols) for cols in detected.values() if isinstance(cols, list)
                                )

        if inferred_rows <= 0:
            inferred_rows = 10_000
        if inferred_cols <= 0:
            inferred_cols = max(1, len(inferred_modalities) * 8)

        target_type = (
            "regression"
            if "regression" in inferred_problem.lower()
            else "classification"
        )

        return {
            "num_rows": int(inferred_rows),
            "num_cols": int(inferred_cols),
            "modalities": list(inferred_modalities),
            "target_type": target_type,
        }

    try:
        from automl.meta_learning import MetaLearningStore

        store = MetaLearningStore()
        dataset_meta = await asyncio.to_thread(_build_dataset_meta_sync)
        suggestions = await asyncio.to_thread(store.suggest, dataset_meta, top_k)
        predicted = await asyncio.to_thread(store.predict_best_config, dataset_meta, top_k)
        record_count = len(await asyncio.to_thread(store.load))

        return {
            "status": "success",
            "session_id": session_id,
            "dataset_meta": dataset_meta,
            "records_available": record_count,
            "suggestions": suggestions,
            "predicted_config": predicted,
        }
    except Exception as exc:
        logger.error("/meta-learning/insights error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Ingest datasets endpoint
# ---------------------------------------------------------------------------

@app.post("/ingest/datasets")
async def ingest_datasets_endpoint(
    request: IngestionRequest,
) -> Dict[str, Any]:
    """
    Start dataset ingestion as a background task.

    Returns a ``task_id`` immediately.  The frontend polls
    ``GET /ingest/status/{task_id}`` for real-time progress.

    FRONTEND CONTRACT
    -----------------
    Immediate response::

        {"status": "started", "task_id": "<8-char hex>"}

    Poll response (GET /ingest/status/{task_id})::

        {
          "task_id":            str,
          "status":             "running" | "completed" | "failed",
          "progress_pct":       int (0-100),
          "message":            str,
          "total_sources":      int,
          "completed_sources":  int,
          "datasets":           [{source, hash, shape, columns, status}, ...],
          "result":             null | {ingestion_progress dict},
          "error":              null | str
        }

    Backward-compatible: when polling returns ``status: "completed"``,
    ``result.ingestion_progress`` has the same shape as the old synchronous
    response so the frontend can handle both patterns.
    """
    session_id: str = request.session_id
    dataset_urls: List[str] = request.dataset_urls

    logger.info("[%s] Ingestion request for %d dataset(s)", session_id, len(dataset_urls))

    task_id = uuid.uuid4().hex[:8]
    tracker = IngestionProgressTracker(task_id, dataset_urls, task_db)

    # Per-session in-memory cache is legacy-only and must stay opt-in.
    session_hashes: Dict[str, Any] = {}
    if ALLOW_LEGACY_SESSION_FALLBACK:
        with _session_lock:
            if session_id not in _session_store:
                _session_store[session_id] = {
                    "datasets": {},
                    "schema": None
                }
            session_hashes = _session_store[session_id]["datasets"]

    async def _run_ingestion() -> None:
        try:
            tracker.set_progress(5, "Initializing ingestion pipeline...")

            # Guarantee the sessions row exists before FK-constrained save_profile runs.
            # Without this, dataset_profiles INSERT fails (FOREIGN KEY constraint).
            session_manager.get_or_create_session(session_id)

            from data_ingestion.ingestion_manager import DataIngestionManager
            manager = DataIngestionManager()

            for idx, source_url in enumerate(dataset_urls):
                _base_pct = 5 + int((idx / max(len(dataset_urls), 1)) * 85)
                tracker.set_progress(
                    _base_pct,
                    f"Downloading dataset {idx + 1}/{len(dataset_urls)}: {source_url[:60]}...",
                )

                # Wire live download progress into the tracker so the frontend
                # slider updates during long Kaggle / HuggingFace downloads.
                def _make_progress_cb(base: int, total_sources: int):
                    _slot = 85 // max(total_sources, 1)
                    def _cb(pct: int, msg: str) -> None:
                        tracker.set_progress(base + int(pct * _slot / 100), msg)
                    return _cb

                try:
                    lazy_datasets, ingest_meta = await manager.ingest_data(
                        [source_url],
                        force_download=False,
                        progress_callback=_make_progress_cb(_base_pct, len(dataset_urls)),
                    )

                    if lazy_datasets:
                        for source_hash, dataset_obj in lazy_datasets.items():
                            lazy_ref = (
                                dataset_obj.lazy_data
                                if hasattr(dataset_obj, "lazy_data")
                                else dataset_obj
                            )
                            # ── Shape + columns from the lazy ref directly ───────────
                            shape: Optional[List[int]] = None
                            columns: List[str] = []
                            try:
                                import polars as pl
                                if isinstance(lazy_ref, pl.LazyFrame):
                                    schema = lazy_ref.collect_schema()
                                    columns = schema.names()
                                    try:
                                        n_rows = int(
                                            lazy_ref.select(pl.len()).collect().item()
                                        )
                                        shape = [n_rows, len(columns)]
                                    except Exception:
                                        shape = [None, len(columns)]  # type: ignore[list-item]
                            except ImportError:
                                pass

                            if not columns:
                                # Dask / pandas fallback
                                try:
                                    if hasattr(lazy_ref, "columns"):
                                        columns = list(lazy_ref.columns)
                                        try:
                                            shape = [len(lazy_ref), len(columns)]
                                        except Exception:
                                            shape = [None, len(columns)]  # type: ignore[list-item]
                                except Exception:
                                    pass

                            ds_info = {
                                "source": source_url,
                                "hash": source_hash,
                                "shape": shape,
                                "columns": columns,
                                "status": "success",
                            }
                            tracker.report_dataset(ds_info)

                            if ALLOW_LEGACY_SESSION_FALLBACK:
                                with _session_lock:
                                    session_hashes[source_hash] = {
                                        "source_url": source_url,
                                        "hash": source_hash,
                                        "timestamp": ingest_meta["ingestion_time"],
                                    }

                            # Keep session/context DB state in sync so v2 and
                            # monitor endpoints can discover active datasets.
                            try:
                                profile = context_db.load_profile(source_hash) or {"dataset_id": source_hash}
                                profile["dataset_id"] = source_hash
                                profile["source_url"] = source_url
                                if os.path.exists(source_url):
                                    profile["file_path"] = source_url
                                context_db.save_profile(profile, session_id)

                                ctx_obj = session_manager.get_session(session_id)
                                if ctx_obj and source_hash not in ctx_obj.active_dataset_ids:
                                    ctx_obj.active_dataset_ids.append(source_hash)
                                    ctx_obj.log_decision(
                                        "ingestion",
                                        f"Registered dataset {source_hash}",
                                        source_url,
                                    )
                                    session_manager.update_session_context(session_id, ctx_obj)
                            except Exception as sync_exc:
                                logger.error(
                                    "Ingest sync failed for %s in session %s: %s",
                                    source_hash,
                                    session_id,
                                    sync_exc,
                                    exc_info=True,
                                )
                                tracker.report_dataset({
                                    "source": source_url,
                                    "hash": source_hash,
                                    "shape": None,
                                    "columns": [],
                                    "status": f"context_sync_failed: {sync_exc}",
                                })
                                # G4: do not count this as a successful ingestion.
                                continue
                    else:
                        failed = ingest_meta.get("failed", {})
                        err_msg = next(iter(failed.values()), "Unknown error") if failed else "No data returned"
                        tracker.report_dataset({
                            "source": source_url,
                            "hash": None,
                            "shape": None,
                            "columns": [],
                            "status": f"failed: {err_msg}",
                        })

                except Exception as ds_exc:
                    logger.error("Dataset ingestion failed for %s: %s", source_url, ds_exc)
                    tracker.report_dataset({
                        "source": source_url,
                        "hash": None,
                        "shape": None,
                        "columns": [],
                        "status": f"failed: {ds_exc}",
                    })

            # Build final result in the old response format for compatibility
            _payload = task_db.get_payload(task_id)
            datasets_info = _payload.get("datasets", [])

            success_count = sum(1 for d in datasets_info if d["status"] == "success")
            failed_count = len(datasets_info) - success_count

            if failed_count == 0:
                overall_status = "success"
                message = f"Successfully ingested {success_count} dataset(s)"
            elif success_count > 0:
                overall_status = "partial"
                message = f"Ingested {success_count}/{len(datasets_info)}; {failed_count} failed"
            else:
                overall_status = "failed"
                message = f"All {len(datasets_info)} dataset(s) failed to ingest"

            final_result = {
                "ingestion_progress": {
                    "status": overall_status,
                    "progress": 100,
                    "message": message,
                    "datasets": datasets_info,
                }
            }
            tracker.complete(final_result)
            logger.info("[%s] Ingestion complete: %s", session_id, message)

        except Exception as exc:
            logger.error("Ingestion task %s failed: %s", task_id, exc, exc_info=True)
            tracker.fail(str(exc))

    # Launch as background asyncio task
    asyncio.create_task(_run_ingestion())

    return {
        "status": "started",
        "task_id": task_id,
    }


@app.get("/ingest/status/{task_id}")
async def ingest_status(task_id: str) -> Dict[str, Any]:
    """Poll ingestion progress for a given task_id."""
    task = task_db.get_task(task_id)
    if task is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown ingestion task_id: {task_id}",
        )
    payload = task.get("payload", {})
    return {
        "task_id":            task["task_id"],
        "status":             task["status"],
        "progress_pct":       payload.get("progress_pct", 0),
        "message":            payload.get("message", ""),
        "total_sources":      payload.get("total_sources", 0),
        "completed_sources":  payload.get("completed_sources", 0),
        "datasets":           payload.get("datasets", []),
        "result":             task.get("result"),
        "error":              task.get("error"),
    }


# ---------------------------------------------------------------------------
# Schema detection
# ---------------------------------------------------------------------------
@app.post("/api/schema/detect")
async def detect_schema(request: Request):

    try:
        from dataclasses import asdict
        from data_ingestion.schema_detector import COGMASchemaDetector

        body = await request.json()
        session_id = body.get("session_id")
        target_override = body.get("target_override")

        if not session_id:
            raise HTTPException(400, "session_id required")

        lazy_datasets = get_session_datasets(session_id)

        if not lazy_datasets:
            raise HTTPException(400, "No datasets found. Run ingestion first.")

        detector = COGMASchemaDetector()

        per_dataset_results = []

        for dataset_id, lazy_data in lazy_datasets.items():

            result = detector._detect_single(
                dataset_id,
                lazy_data,
                target_override=target_override
            )

            per_dataset_results.append(asdict(result))

        # 🔥 IMPORTANT FIX: use correct global builder
        global_schema = detector._build_global_schema(per_dataset_results)

        schema_dict = asdict(global_schema)

        # Activate PipelineOrchestrator: persist profiles + preprocessing plans to SQLite.
        # get_or_create_session guarantees context exists regardless of whether ingestion
        # previously created it — fixes the silent skip when session was missing.
        try:
            from core.orchestrator import orchestrator as _po

            _ctx = session_manager.get_or_create_session(session_id)
            _data_map_orch: Dict[str, Any] = {}
            _session_lazy_map = get_session_datasets(session_id)

            for _ds_res in per_dataset_results:
                _did = _ds_res.get("dataset_id", "")
                if not _did:
                    continue
                _lazy = _session_lazy_map.get(_did)
                if _lazy is None:
                    continue

                try:
                    import polars as _pl
                    if isinstance(_lazy, _pl.LazyFrame):
                        _data_map_orch[_did] = _lazy.head(500).collect().to_pandas()
                        continue
                except ImportError:
                    pass

                try:
                    import dask.dataframe as _dd
                    if isinstance(_lazy, _dd.DataFrame):
                        _data_map_orch[_did] = _lazy.head(500, compute=True)
                        continue
                except ImportError:
                    pass

                if isinstance(_lazy, pd.DataFrame):
                    _data_map_orch[_did] = _lazy.head(500)
                    continue

                if hasattr(_lazy, "head"):
                    _head = _lazy.head(500)
                    if isinstance(_head, pd.DataFrame):
                        _data_map_orch[_did] = _head

            if _data_map_orch:
                _po.execute_phase_2_schema(_ctx, _data_map_orch)
                _po.execute_phase_3_target(_ctx, _data_map_orch)
                _po.execute_phase_4_aggregation(_ctx, _data_map_orch)
                _po.execute_phase_5_preprocessing(_ctx, _data_map_orch)
            # Always persist context so pipeline_stage is set even if data_map was empty
            session_manager.update_session_context(session_id, _ctx)
            logger.info(
                "/api/schema/detect: PipelineOrchestrator phases 2-5 complete for %s",
                session_id,
            )
        except Exception as _po_exc:
            logger.error(
                "/api/schema/detect: PipelineOrchestrator delegation failed: %s",
                _po_exc,
                exc_info=True,
            )
            # Still persist context with whatever state was reached
            try:
                session_manager.update_session_context(session_id, _ctx)
            except Exception:
                pass

        # Optional compatibility mirror for legacy consumers.
        _write_legacy_session_cache(session_id, "schema", schema_dict)

        # Update ExecutionContext with schema + xs3_confidence_gap
        try:
            ctx = session_manager.get_session(session_id)
            if ctx:
                per_ds = schema_dict.get("per_dataset", [])
                xs3_gaps = [
                    float(ds.get("reasoning", {}).get("xs3_confidence_gap", 0.0))
                    for ds in per_ds
                    if isinstance(ds, dict)
                ]
                max_xs3_gap = max(xs3_gaps, default=0.0)

                # Keys are dataset_ids (UUIDs), values are the tabular-feature → target
                # RandomForest CV score from schema detection. This is NOT per-modality;
                # it measures how well tabular columns alone predict the target.
                predictability_scores = {
                    ds.get("dataset_id", f"ds_{i}"): float(
                        ds.get("target_profile", {}).get("predictability_score", 0.0)
                    )
                    for i, ds in enumerate(per_ds)
                    if isinstance(ds, dict)
                }

                schema_for_context = dict(schema_dict)
                schema_for_context["_xs3_max_gap"] = max_xs3_gap
                schema_for_context["modality_presence"] = {
                    str(mod): True
                    for mod in schema_dict.get("global_modalities", [])
                }
                schema_for_context["predictability_scores"] = predictability_scores
                schema_for_context["target_confidence"] = float(
                    schema_dict.get("detection_confidence", 0.0)
                )

                ctx.update_from_schema(schema_for_context)
                ctx.confidence_map["xs3_target_gap"] = max_xs3_gap
                ctx.predictability_scores = predictability_scores
                session_manager.update_session_context(session_id, ctx)
        except OptimisticLockError:
            raise
        except Exception as _ctx_exc:
            logger.warning(
                "/api/schema/detect: ExecutionContext update failed: %s",
                _ctx_exc,
            )

        return _to_json_safe({
            "status": "success",
            "phase": "Schema Detection",
            "data": schema_dict,
            "candidates": [asdict(c) for c in detector.last_target_candidates],
        })

    except OptimisticLockError:
        raise
    except HTTPException:
        raise  # Don't convert 400/422 to 500 — let FastAPI return the correct status
    except Exception as e:
        logger.error("Schema detection failed: %s", e, exc_info=True)
        raise HTTPException(500, "Schema detection failed")



# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
@app.post("/api/schema/override")
async def override_target(request: Request):

    body = await request.json()
    session_id = body.get("session_id")
    new_target = body.get("target")
    reason = body.get("reason", "User override")

    if not session_id or not new_target:
        raise HTTPException(400, "session_id and target required")

    ctx = _get_session_context_or_422(session_id, "/api/schema/override")

    # Keep schema and target override in one authoritative context update.
    schema = dict(getattr(ctx, "global_schema", {}) or {})
    if schema:
        schema["primary_target"] = new_target
        for ds in schema.get("per_dataset", []):
            if isinstance(ds, dict):
                ds["target_column"] = new_target
        ctx.update_from_schema(schema)

    if ALLOW_LEGACY_SESSION_FALLBACK:
        _write_legacy_session_cache(session_id, "target_override", new_target)
        if schema:
            _write_legacy_session_cache(session_id, "schema", schema)

    try:
        ctx.override_global_target(new_target, reason)
        ctx.log_decision(
            "schema_override",
            f"Target override via /api/schema/override: {new_target}",
            reason,
        )
        session_manager.update_session_context(session_id, ctx)
    except OptimisticLockError:
        raise
    except Exception as _ov_exc:
        logger.error("/api/schema/override: context update failed: %s", _ov_exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to apply target override") from _ov_exc

    return {"status": "override_applied", "target": new_target}
@app.post("/preprocess")
@require_context("schema_detection", required_fields=["global_schema"], require_session=True)
async def preprocess_data(request: Request) -> Dict[str, Any]:
    """
    Run Phase 3 preprocessing on the current session's datasets.

    STRICT SESSION ISOLATION
    ------------------------
    Requires an active ingestion session scoped to ``session_id``.
    Returns HTTP 400 if ``/ingest/datasets`` has not been called first.

    FRONTEND CONTRACT
    -----------------
    Returns::

        {
          "status": "success",
          "phase": "Phase 3: Preprocessing",
          "data": {
            "preprocessing_stages": [
              {"stage": "<name>", "status": "success", "output_shape": "<shape>"},
              ...
            ],
            "total_samples": <int>,
            "output_shapes": {
              "tabular": "(N, <D>)",
              "text":    "(N, 128) per key",
              "image":   "(N, 3, 224, 224)"
            }
          }
        }
    """
    try:
        from data_ingestion.loader import DataLoader
        from data_ingestion.schema_detector import MultiDatasetSchemaDetector
        from preprocessing.image_preprocessor import ImagePreprocessor
        from preprocessing.text_preprocessor import TextPreprocessor
        from preprocessing.tabular_preprocessor import TabularPreprocessor
        from preprocessing.validator import (
            PreprocessingValidationError,
            PreprocessingValidator,
            validate_preprocessor_consistency,
        )

        body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
        _sid = body.get("session_id") if isinstance(body, dict) else None
        _ctx = _get_session_context_or_422(_sid, "/preprocess")
        global_schema = _require_context_artifact(_sid, "global_schema", "/preprocess")

        with _session_lock:
            _session_snapshot = _get_session_hashes(_sid)
            if not _session_snapshot:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "No active ingestion session. "
                        "Call POST /ingest/datasets first to register datasets."
                    ),
                )
            _snapshot_hashes_pp = list(_session_snapshot.keys())

        # ----------------------------------------------------------------
        # Load cached lazy refs for current session
        # ----------------------------------------------------------------
        cache_dir = DATASET_CACHE_DIR  # was hardcoded Path("./data/dataset_cache")
        loader = DataLoader()
        lazy_datasets: Dict[str, Any] = {}
        for hash_id in _snapshot_hashes_pp:
            cache_path = cache_dir / hash_id
            lazy_ref = loader.load_cached(cache_path)
            if lazy_ref is not None:
                lazy_datasets[hash_id] = lazy_ref
            else:
                logger.warning("/preprocess: cache miss for session hash %s", hash_id)

        if not lazy_datasets:
            raise HTTPException(
                status_code=400,
                detail="Active session has no valid data files in cache.",
            )

        # ----------------------------------------------------------------
        # All heavy work (schema detection, materialisation, sklearn fit)
        # is offloaded to a worker thread to avoid blocking the event loop.
        # ----------------------------------------------------------------
        def _preprocess_sync() -> Dict[str, Any]:

            detected: Dict[str, List[str]] = {}
            modalities_raw: Any = []
            target_col: str = "Unknown"

            if isinstance(global_schema, dict):
                per_ds = global_schema.get("per_dataset", [])
                if per_ds and isinstance(per_ds[0], dict):
                    detected = per_ds[0].get("detected_columns", {}) or {}
                modalities_raw = global_schema.get("global_modalities", []) or []
                target_col = global_schema.get("primary_target", "Unknown")
            else:
                detected = getattr(global_schema, "detected_columns", {}) or {}
                modalities_raw = getattr(
                    global_schema,
                    "modality_presence",
                    getattr(global_schema, "global_modalities", []),
                )
                target_col = getattr(global_schema, "primary_target", "Unknown")

            if isinstance(modalities_raw, dict):
                modalities = {str(k) for k, v in modalities_raw.items() if bool(v)}
            elif isinstance(modalities_raw, list):
                modalities = {str(m) for m in modalities_raw}
            else:
                modalities = set()

            _cache_sid = str(_sid) if _sid else "default"
            _cache_session_dir = Path(f"./data/session_cache/{_cache_sid}")
            _cache_session_dir.mkdir(parents=True, exist_ok=True)
            _scaler_path = _cache_session_dir / "tabular_scaler.joblib"
            
            # Sample frames for preprocessing — materialise every lazy type to pandas.
            # Priority order mirrors load_cached: LazyFrame > DataFrame > LazyImageDataset > dask.
            # For multimodal CSV datasets (Hateful Memes, MMIMDB) load_cached returns a LazyFrame
            # from the CSV (which contains img_path string columns) — LazyImageDataset only
            # appears when the cache has raw image files with no CSV manifest.
            MAX_SAMPLE = 10_000
            frames: List[pd.DataFrame] = []
            for lazy_ref in lazy_datasets.values():
                try:
                    import polars as _pl
                    from data_ingestion.loader import LazyImageDataset as _LID
                    if isinstance(lazy_ref, _pl.LazyFrame):
                        frames.append(lazy_ref.head(MAX_SAMPLE).collect().to_pandas())
                    elif isinstance(lazy_ref, _pl.DataFrame):
                        frames.append(lazy_ref.head(MAX_SAMPLE).to_pandas())
                    elif isinstance(lazy_ref, pd.DataFrame):
                        frames.append(lazy_ref.head(MAX_SAMPLE))
                    elif isinstance(lazy_ref, _LID):
                        # Pure image directory: expose paths as a single-column DataFrame
                        paths = lazy_ref._paths[:MAX_SAMPLE]
                        frames.append(pd.DataFrame({"image_path": paths}))
                    else:
                        # dask or unknown lazy type — try head() then wrap
                        try:
                            head = lazy_ref.head(MAX_SAMPLE)
                            frames.append(head if isinstance(head, pd.DataFrame) else pd.DataFrame(head))
                        except Exception:
                            pass
                except Exception as e:
                    logger.warning("Failed to materialise dataset sample: %s", e)

            total_samples = sum(len(f) for f in frames)
            full_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

            feature_df = (
                full_df.drop(columns=[target_col])
                if target_col != "Unknown" and target_col in full_df.columns
                else full_df
            )

            preprocessing_stages: List[Dict[str, Any]] = []
            output_shapes: Dict[str, Any] = {}
            samples: Dict[str, Any] = {}

            text_cols = [c for c in detected.get("text", []) if c in feature_df.columns]
            image_cols = [c for c in detected.get("image", []) if c in feature_df.columns]
            tabular_cols = [
                c for c in feature_df.columns
                if c not in text_cols and c not in image_cols
            ]

            # Pre-filter: drop tabular when all remaining cols are IDs or the target.
            # Mirrors the same logic in /select-model and training_orchestrator.py.
            _prefilter_per_ds = (
                global_schema.get("per_dataset", []) if isinstance(global_schema, dict) else []
            )
            _tab_id_like: set = set()
            for _pds in _prefilter_per_ds:
                if isinstance(_pds, dict):
                    _tab_id_like.update(_pds.get("id_like_columns") or [])

            # Fallback: if the schema didn't populate id_like_columns, detect by
            # uniqueness ratio directly — any col with >90% unique values is ID-like.
            if not _tab_id_like and tabular_cols and not feature_df.empty:
                _n = max(len(feature_df), 1)
                for _col in tabular_cols:
                    if _col == target_col:
                        continue
                    try:
                        if feature_df[_col].nunique() / _n > 0.90:
                            _tab_id_like.add(_col)
                    except Exception:
                        pass

            _tab_target = str(target_col) if target_col != "Unknown" else ""
            _tab_usable = set(tabular_cols) - {_tab_target} - _tab_id_like
            if tabular_cols and not _tab_usable:
                logger.info(
                    "/preprocess: tabular auto-skipped — cols %s are all IDs or target '%s' "
                    "(id_like detected: %s).",
                    tabular_cols, _tab_target, sorted(_tab_id_like),
                )
                tabular_cols = []
                modalities.discard("tabular")

            # Bug #1 + #5: Validate that image paths resolve to real files.
            # Probe the first 10 non-null values; warn (do NOT fail) if none resolve.
            if image_cols and not feature_df.empty:
                _img_invalid: List[str] = []
                for _img_col in image_cols:
                    _probe_series = feature_df[_img_col].dropna().astype(str).str.strip()
                    _probe_series = _probe_series[~_probe_series.str.lower().isin(
                        {"", "nan", "none", "null", "<na>"}
                    )]
                    _probe_paths = _probe_series.head(10).tolist()
                    for _p in _probe_paths:
                        _path_obj = Path(_p)
                        if not _path_obj.is_absolute():
                            # Bug #5: try resolving relative path against CWD
                            _resolved = Path.cwd() / _p
                            if _resolved.is_file():
                                continue
                        if not _path_obj.is_file():
                            _img_invalid.append(_p)
                if _img_invalid:
                    logger.warning(
                        "/preprocess: %d image path(s) could not be resolved "
                        "(first: %s). Relative paths are resolved against CWD=%s",
                        len(_img_invalid), _img_invalid[0], Path.cwd(),
                    )

            schema_for_validation = (
                dict(global_schema)
                if isinstance(global_schema, dict)
                else {
                    "global_modalities": list(modalities),
                    "primary_target": target_col,
                    "per_dataset": [{"detected_columns": detected}],
                }
            )

            validation_plan = {
                "modality": {
                    "tabular": {
                        "columns": tabular_cols,
                        "imputer_strategy": "median",
                    },
                    "text": {
                        "columns": text_cols,
                        "max_length": 128,
                    },
                    "image": {
                        "columns": image_cols,
                        "image_size": [224, 224],
                    },
                },
                "feature_selection": {
                    "top_k": max(1, min(256, len(tabular_cols) or 1)),
                },
            }

            validator = PreprocessingValidator()
            try:
                validation_report = validator.validate_plan(
                    validation_plan,
                    schema_for_validation,
                    dataset_shape=feature_df.shape,
                )
            except PreprocessingValidationError as validation_exc:
                raise RuntimeError(
                    f"PREPROCESS_VALIDATION_ERROR: {validation_exc}"
                ) from validation_exc

            adaptive_tabular_config: Dict[str, Any] = {}
            adaptive_context: Dict[str, Any] = {}
            drifted_features: List[str] = []
            if tabular_cols and _sid:
                try:
                    from preprocessing.adaptive_engine import AdaptivePreprocessingEngine

                    _ctx_local = session_manager.get_session(_sid)
                    if _ctx_local is not None:
                        adaptive_engine = AdaptivePreprocessingEngine(_ctx_local)
                        adaptive_context = adaptive_engine.build_context_contract(feature_df[tabular_cols])
                        adaptive_tabular_config = dict(adaptive_context.get("adaptive_tabular_config", {}) or {})
                        drifted_features = list(adaptive_context.get("drifted_features", []) or [])
                except Exception as adaptive_exc:
                    logger.warning("/preprocess: adaptive tabular config unavailable: %s", adaptive_exc)

            # Read preprocessing plan from ExecutionContext (set during schema detection Phase 5)
            preprocessing_plan: Dict[str, Any] = {}
            if _ctx is not None:
                _raw_plan = getattr(_ctx, "preprocessing_plan", None)
                if isinstance(_raw_plan, dict):
                    preprocessing_plan = _raw_plan

            tabular_prep_obj = None
            text_prep_obj = None
            image_prep_obj = None

            if tabular_cols and "tabular" in modalities:
                out_dim = len(tabular_cols)
                try:
                    import joblib

                    if _scaler_path.exists():
                        tabular_prep_obj = joblib.load(str(_scaler_path))
                        if adaptive_tabular_config and hasattr(tabular_prep_obj, "configure"):
                            tabular_prep_obj.configure(adaptive_tabular_config)
                        if drifted_features:
                            setattr(tabular_prep_obj, "_drifted_features", list(drifted_features))
                        logger.info(
                            "/preprocess: using cached TabularPreprocessor from %s",
                            _scaler_path,
                        )
                    else:
                        tabular_prep_obj = TabularPreprocessor(
                            adaptive_config=adaptive_tabular_config,
                            drifted_features=drifted_features,
                        )
                        tabular_prep_obj.fit(feature_df[tabular_cols])

                        # FIX-20: cache representative transformed sample so
                        # /select-model can perform data-driven probing.
                        try:
                            _n_sample = min(1000, len(feature_df))
                            if _sid and _n_sample > 0:
                                _sample_df = feature_df[tabular_cols].sample(
                                    _n_sample,
                                    random_state=42,
                                    replace=False,
                                )
                                _target_series = (
                                    full_df[target_col].iloc[_sample_df.index]
                                    if target_col != "Unknown" and target_col in full_df.columns
                                    else None
                                )
                                if _target_series is not None:
                                    _sample_X = tabular_prep_obj.transform(_sample_df)
                                    _sample_y = _target_series.to_numpy()
                                    try:
                                        context_db.save_probe_sample(_sid, _sample_X, _sample_y)
                                        logger.info(
                                            "/preprocess: persisted probe sample to SQLite for session %s",
                                            _sid,
                                        )
                                    except Exception as _persist_exc:
                                        if ALLOW_LEGACY_SESSION_FALLBACK:
                                            logger.warning(
                                                "/preprocess: probe sample persist failed, using in-memory fallback: %s",
                                                _persist_exc,
                                            )
                                            _write_legacy_session_cache(
                                                _sid,
                                                "tabular_sample",
                                                {
                                                    "X": _sample_X,
                                                    "y": _sample_y,
                                                },
                                            )
                                        else:
                                            logger.warning(
                                                "/preprocess: probe sample persist failed and legacy fallback is disabled: %s",
                                                _persist_exc,
                                            )
                                    logger.info(
                                        "/preprocess: cached tabular probe sample "
                                        "(n=%d, features=%d) for session %s",
                                        _n_sample,
                                        _sample_X.shape[1] if hasattr(_sample_X, "shape") and len(_sample_X.shape) > 1 else 0,
                                        _sid,
                                    )
                        except Exception as _sample_exc:
                            logger.warning(
                                "/preprocess: could not cache tabular sample (non-fatal): %s",
                                _sample_exc,
                            )

                        joblib.dump(tabular_prep_obj, str(_scaler_path))
                        logger.info(
                            "/preprocess: persisted TabularPreprocessor to %s",
                            _scaler_path,
                        )

                    if tabular_prep_obj is not None and hasattr(tabular_prep_obj, "get_output_dim"):
                        out_dim = tabular_prep_obj.get_output_dim()

                    if tabular_prep_obj is not None and hasattr(tabular_prep_obj, "transform"):
                        # --- Sample: raw vs transformed (first 3 rows) ---
                        raw_sample_df = feature_df[tabular_cols].head(3)
                        transformed_arr = tabular_prep_obj.transform(raw_sample_df)

                        # Feature names from the ColumnTransformer
                        try:
                            feat_names = list(tabular_prep_obj._transformer.get_feature_names_out())
                        except Exception:
                            feat_names = [f"f{i}" for i in range(transformed_arr.shape[1])]

                        samples["tabular"] = {
                            "raw_columns": list(raw_sample_df.columns),
                            "raw_rows": raw_sample_df.fillna("").astype(str).values.tolist(),
                            "transformed_columns": feat_names,
                            "transformed_rows": transformed_arr.tolist(),
                            "dropped_columns": list(getattr(tabular_prep_obj, "_dropped_cols", [])),
                        }
                except Exception as tab_exc:
                    logger.warning("/preprocess tabular sample failed: %s", tab_exc)
                    out_dim = len(tabular_cols)

                output_shapes["tabular"] = f"(N, {out_dim})"
                preprocessing_stages.append({
                    "stage": "tabular_preprocessing",
                    "status": "success",
                    "output_shape": output_shapes["tabular"],
                })

            if text_cols and "text" in modalities:
                # --- Sample: original text + tokenized ids (first row) ---
                # Build TextPreprocessor first so we can report the actual max_length
                try:
                    first_text = str(feature_df[text_cols[0]].dropna().iloc[0])
                    text_prep_obj = TextPreprocessor()
                    # Wire feature_intelligence from context so the preprocessor
                    # adapts tokenizer, max_length, and pooling to the dataset's signals.
                    _text_plan: Dict[str, Any] = dict(
                        (preprocessing_plan or {}).get("text") or {}
                    )
                    _text_plan["feature_intelligence"] = getattr(_ctx, "feature_intelligence", {}) or {}
                    _text_plan["text_task_type"] = (
                        _text_plan["feature_intelligence"].get("text_task_type")
                        or _text_plan.get("text_task_type")
                    )
                    try:
                        text_prep_obj.configure(_text_plan)
                    except Exception as _cfg_exc:
                        logger.debug("/preprocess TextPreprocessor.configure failed: %s", _cfg_exc)
                    tok_out = text_prep_obj.preprocess(first_text)
                    samples["text"] = {
                        "column": text_cols[0],
                        "original": first_text[:500],
                        "input_ids": tok_out["input_ids"].tolist(),
                        "attention_mask": tok_out["attention_mask"].tolist(),
                        "tokenizer": getattr(text_prep_obj, "_pretrained_model", "bert-base-uncased"),
                        "max_length": getattr(text_prep_obj, "max_length", 128),
                    }
                    _actual_max_len = getattr(text_prep_obj, "max_length", 128)
                except Exception as txt_exc:
                    logger.warning("/preprocess text sample failed: %s", txt_exc)
                    _actual_max_len = 128
                output_shapes["text"] = f"(N, {_actual_max_len}) per key"
                preprocessing_stages.append({
                    "stage": "text_preprocessing",
                    "status": "success",
                    "output_shape": output_shapes["text"],
                })

            if image_cols and "image" in modalities:
                # Validate a sample of image paths exist — catch broken paths
                # before training starts (cheap: just os.path.exists, no pixel load)
                _img_col = image_cols[0]
                _img_paths = feature_df[_img_col].dropna().astype(str).head(10).tolist()
                _missing = [p for p in _img_paths if p and not os.path.exists(p)]

                # Path repair: if paths are missing, the CSV was generated with wrong
                # base dir. Search progressively wider: current session cache →
                # all session caches → full data dir.
                if _missing:
                    _mp0 = Path(_missing[0])
                    _img_name = _mp0.name
                    _img_parent_name = _mp0.parent.name  # e.g. "img"
                    # Search order: current session → all sessions → data dir →
                    # Windows common locations (Downloads, Desktop, Documents)
                    _search_roots = [
                        Path(f"./data/session_cache/{_sid}") if _sid else None,
                        Path("./data/session_cache"),
                        Path("./data"),
                        Path.home() / "Downloads",
                        Path.home() / "Desktop",
                        Path.home() / "Documents",
                    ]
                    _repair_base: Optional[Path] = None
                    for _sroot in _search_roots:
                        if _sroot is None or not _sroot.exists():
                            continue
                        try:
                            for _found_img in _sroot.rglob(_img_name):
                                if _found_img.parent.name == _img_parent_name:
                                    _repair_base = _found_img.parent.parent
                                    break
                        except (PermissionError, OSError):
                            continue
                        if _repair_base:
                            break
                    if _repair_base:
                        def _repair_img(p: str, _base: Path = _repair_base) -> str:
                            if not p or os.path.exists(p):
                                return p
                            _pp = Path(p)
                            _new = _base / _pp.parent.name / _pp.name
                            return str(_new) if _new.exists() else p
                        feature_df = feature_df.copy()
                        feature_df[_img_col] = feature_df[_img_col].apply(_repair_img)
                        full_df = full_df.copy()
                        full_df[_img_col] = full_df[_img_col].apply(_repair_img)
                        _img_paths = feature_df[_img_col].dropna().astype(str).head(10).tolist()
                        _missing = [p for p in _img_paths if p and not os.path.exists(p)]
                        logger.info(
                            "/preprocess: repaired image paths using base=%s, still_missing=%d",
                            _repair_base, len(_missing),
                        )
                        # Persist corrected paths back to the CSV that contains the image column.
                        # Search _repair_base.parent (the dataset root) and data/ broadly.
                        _csv_search_roots = [_repair_base.parent, Path("./data")]
                        for _csv_root in _csv_search_roots:
                            if not _csv_root.exists():
                                continue
                            for _csv_cand in _csv_root.rglob("*.csv"):
                                try:
                                    _hdr = pd.read_csv(_csv_cand, nrows=0)
                                    if _img_col in _hdr.columns:
                                        full_df.to_csv(_csv_cand, index=False)
                                        logger.info(
                                            "/preprocess: repaired CSV persisted to %s", _csv_cand
                                        )
                                        break
                                except Exception:
                                    pass
                            else:
                                continue
                            break

                _img_status = "success" if not _missing else f"warning: {len(_missing)}/10 sample paths missing"
                # Build ImagePreprocessor BEFORE output_shapes so we can report the actual size
                image_prep_obj = ImagePreprocessor()
                _image_plan: Dict[str, Any] = dict((preprocessing_plan or {}).get("image") or {})
                _image_plan["feature_intelligence"] = getattr(_ctx, "feature_intelligence", {}) or {}
                try:
                    image_prep_obj.configure(_image_plan)
                except Exception as _icfg_exc:
                    logger.debug("/preprocess ImagePreprocessor.configure failed: %s", _icfg_exc)
                _img_ts = list(getattr(image_prep_obj, "target_size", [224, 224]))
                output_shapes["image"] = f"(N, 3, {_img_ts[0]}, {_img_ts[1]})"
                preprocessing_stages.append({
                    "stage": "image_preprocessing",
                    "status": _img_status,
                    "output_shape": output_shapes["image"],
                    "sample_paths_checked": len(_img_paths),
                    "missing_paths": len(_missing),
                })
                # Find first valid image path for frontend preview
                _first_valid_img: Optional[str] = None
                for _pp in feature_df[_img_col].dropna().astype(str).tolist():
                    if _pp and os.path.exists(_pp):
                        _first_valid_img = _pp
                        break

                # Generate raw + augmented preview images as base64 PNG
                _preview_raw_b64: Optional[str] = None
                _preview_aug_b64: Optional[str] = None
                _raw_wh: Optional[list] = None
                if _first_valid_img:
                    try:
                        import io as _io
                        import base64 as _b64
                        import torch as _torch_prev
                        from PIL import Image as _PILPrev
                        from torchvision import transforms as _tvt

                        def _pil_to_b64(img: "_PILPrev.Image") -> str:
                            _buf = _io.BytesIO()
                            img.save(_buf, format="PNG")
                            return _b64.b64encode(_buf.getvalue()).decode()

                        _raw_pil = _PILPrev.open(_first_valid_img).convert("RGB")
                        _raw_wh = list(_raw_pil.size)        # [W, H]
                        _ts = list(image_prep_obj.target_size)

                        # Resize-only (raw preview at target size)
                        _resized = _raw_pil.resize((_ts[0], _ts[1]), _PILPrev.LANCZOS)
                        _preview_raw_b64 = _pil_to_b64(_resized)

                        # Augmented preview — spatial/colour ops only (no ToTensor/Normalize)
                        # so the result is still a displayable PIL image.
                        _aug_intensity = getattr(image_prep_obj, "augment_intensity", "medium")
                        _sharp = bool(getattr(image_prep_obj, "_apply_sharpening", False))
                        _aug_ops: list = [_tvt.RandomHorizontalFlip(p=1.0)]  # always flip for demo
                        if _aug_intensity in ("medium", "strong"):
                            _aug_ops.append(
                                _tvt.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)
                            )
                            _aug_ops.append(_tvt.RandomRotation(15))
                        if _aug_intensity == "strong":
                            _aug_ops.append(
                                _tvt.RandomPerspective(distortion_scale=0.2, p=1.0)
                            )
                        if _sharp:
                            _aug_ops.append(_tvt.RandomAdjustSharpness(sharpness_factor=2.5, p=1.0))

                        _torch_prev.manual_seed(42)       # deterministic for display
                        _aug_pil = _tvt.Compose(_aug_ops)(_resized)
                        _preview_aug_b64 = _pil_to_b64(_aug_pil)

                    except Exception as _prev_exc:
                        logger.debug("/preprocess: image preview generation failed: %s", _prev_exc)

                # Expose what configure() chose so the frontend can show a dynamic pipeline
                samples["image"] = {
                    "column": image_cols[0] if image_cols else "?",
                    "target_size": list(image_prep_obj.target_size),
                    "augment_intensity": getattr(image_prep_obj, "augment_intensity", "medium"),
                    "grayscale": bool(getattr(image_prep_obj, "_force_grayscale", False)),
                    "sharpening": bool(getattr(image_prep_obj, "_apply_sharpening", False)),
                    "normalize_mean": [0.485, 0.456, 0.406],
                    "normalize_std": [0.229, 0.224, 0.225],
                    "missing_paths": len(_missing),
                    "total_paths_checked": len(_img_paths),
                    "first_valid_path": _first_valid_img,
                    "sample_path_raw": _img_paths[0] if _img_paths else None,
                    "preview_raw_b64": _preview_raw_b64,
                    "preview_aug_b64": _preview_aug_b64,
                    "raw_size_wh": _raw_wh,
                }

            schema_modalities_for_consistency: List[str] = []
            if tabular_cols:
                schema_modalities_for_consistency.append("tabular")
            if text_cols:
                schema_modalities_for_consistency.append("text")
            if image_cols:
                schema_modalities_for_consistency.append("image")
            schema_for_consistency = dict(schema_for_validation)
            schema_for_consistency["global_modalities"] = schema_modalities_for_consistency

            try:
                validate_preprocessor_consistency(
                    tabular_prep_obj,
                    text_prep_obj,
                    image_prep_obj,
                    schema_for_consistency,
                )
            except PreprocessingValidationError as validation_exc:
                raise RuntimeError(
                    f"PREPROCESS_VALIDATION_ERROR: {validation_exc}"
                ) from validation_exc

            try:
                _ctx_local = session_manager.get_session(str(_sid)) if _sid else None
                if _ctx_local is not None:
                    _ctx_local.update_preprocessing_contract(
                        validation_plan.get("modality", {}),
                        {
                            "runtime": {
                                "use_embedding_cache": bool(len(feature_df) >= 2_000 and len(modalities) >= 2),
                                "high_volume_mode": bool(len(feature_df) >= 100_000),
                            },
                            "weak_modalities": list(adaptive_context.get("weak_modalities", []) or []),
                            "strong_modalities": list(adaptive_context.get("strong_modalities", []) or []),
                            "modality_predictability": dict(adaptive_context.get("modality_predictability", {}) or {}),
                            "context_signals": {
                                "validation": dict(validation_report or {}),
                                "adaptive": {
                                    key: value
                                    for key, value in adaptive_context.items()
                                    if key != "adaptive_tabular_config"
                                },
                            },
                            "validation": dict(validation_report or {}),
                            "dataset_total_samples": int(len(feature_df)),
                            "fusion_recommendation": adaptive_context.get("fusion_recommendation"),
                            "adaptive_tabular_config": dict(adaptive_tabular_config),
                            "drifted_features": list(drifted_features),
                        },
                    )
                    _active_contract = _apply_active_modality_contract(_ctx_local)
                    if isinstance(_ctx_local.global_schema, dict):
                        _ctx_local.global_schema["global_modalities"] = list(
                            _active_contract["active_modalities"]
                        )
                        _ctx_local.global_schema["modality_presence"] = {
                            mod: True for mod in _active_contract["active_modalities"]
                        }
                    _ctx_local.set_pipeline_stage("preprocessing_planning")
                    _update_session_context_with_retry(str(_sid), _ctx_local)
            except OptimisticLockError:
                logger.warning("/preprocess: giving up after max retries on optimistic lock")
            except Exception as _ctx_pre_exc:
                logger.warning("/preprocess: context preprocessing update failed: %s", _ctx_pre_exc)

            if not preprocessing_stages and not feature_df.empty:
                output_shapes["tabular"] = f"(N, {len(feature_df.columns)})"
                preprocessing_stages.append({
                    "stage": "tabular_preprocessing",
                    "status": "success",
                    "output_shape": output_shapes["tabular"],
                })

            _ctx_after = session_manager.get_session(str(_sid)) if _sid else _ctx
            _contract = _context_contract_payload(_ctx_after)

            # Probe availability: a tabular probe sample is required for data-driven
            # model selection. If tabular was skipped (no usable tabular cols), no probe
            # can be saved — model selection will fall back to heuristic ranking.
            _probe_saved = bool(tabular_cols and _sid)
            _probe_note = (
                "Probe sample saved — model selection will use data-driven ranking."
                if _probe_saved
                else (
                    "No tabular features → no probe sample. "
                    "Model selection uses heuristic ranking (hardware tier + architecture rules). "
                    "This is expected for text/image-only datasets."
                )
            )

            return {
                "status": "success",
                "phase": "Phase 3: Preprocessing",
                "context_stage": _contract["context_stage"],
                "context_version": _contract["context_version"],
                "artifact_versions": _contract["artifact_versions"],
                "data": {
                    "preprocessing_stages": preprocessing_stages,
                    "total_samples": total_samples,
                    "output_shapes": output_shapes,
                    "preprocessor_cached": _scaler_path.exists(),
                    "preprocessor_path": str(_scaler_path),
                    "validation": validation_report,
                    "validation_report": validation_report,
                    "samples": samples,
                    "text_columns": text_cols,
                    "image_columns": image_cols,
                    "tabular_columns": tabular_cols,
                    "probe_available": _probe_saved,
                    "probe_note": _probe_note,
                },
            }

        try:
            return await asyncio.to_thread(_preprocess_sync)
        except RuntimeError as exc:
            msg = str(exc)
            _prefix = "PREPROCESS_VALIDATION_ERROR:"
            if msg.startswith(_prefix):
                raise HTTPException(status_code=422, detail=msg[len(_prefix):].strip())
            raise

    except HTTPException:
        raise
    except OptimisticLockError:
        raise
    except Exception as exc:
        logger.error("/preprocess error: %s", exc, exc_info=True)
        return JSONResponse(
            {"error": "Preprocessing failed. Check server logs for details."},
            status_code=500,
        )


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

@app.post("/select-model")
@require_context("preprocessing_planning", required_fields=["global_schema"], require_session=True)
async def select_model(request: Request) -> Dict[str, Any]:
    """
    Run Phase 4 model selection using ``AdvancedModelSelector``.

    Request body (all fields optional)
    -----------------------------------
    ``problem_type``  : str   – e.g. ``"classification_binary"`` (default ``"unsupervised"``)
    ``modalities``    : list  – subset of ``["image","text","tabular"]`` (default ``[]``)
    ``dataset_size``  : int   – total training samples (default ``10000``)
    ``avg_tokens``    : int   – mean tokens/text sample, drives text batch-size rule
                                (default ``128``)

    FRONTEND CONTRACT
    -----------------
    Returns the exact structure Streamlit expects::

        {
          "status":             "success",
          "problem_type":       <str>,
          "modalities":         <list[str]>,
          "recommended_models": <list[dict]>,
          "best_model":         <dict>   ← first entry in recommended_models
        }

    Each model dict contains::

        {
          "name":            "<ViT-Base + BERT-base + TabNet>",
          "image_encoder":   "<name or null>",
          "text_encoder":    "<name or null>",
          "tabular_encoder": "<name or null>",
          "fusion_strategy": "<attention|concatenation>",
          "batch_size":      <int>,
          "hpo_space":       { "<param>": {"type":..,"low":..,"high":..}, ... },
          "rationale":       { "<component>": "<reason>", ... },
          "hardware_info":   { "gpu_available": bool, "gpu_memory_gb": float, ... },
          "tier":            "primary" | "fallback"
        }
    """
    try:
        body = await request.json()
        _sid = body.get("session_id") if isinstance(body, dict) else None
        _ctx_for_budget = _get_session_context_or_422(_sid, "/select-model")
        _require_context_artifact(_sid, "global_schema", "/select-model")
        _active_contract = _apply_active_modality_contract(_ctx_for_budget)
        _body_modalities = _validate_modalities(
            list(body.get("modalities") or []), "/select-model"
        )
        modalities: List[str] = _ordered_modalities(
            _active_contract.get("active_modalities") or _body_modalities
        )
        if not modalities:
            modalities = list(_body_modalities)
        # After pre-filtering (tabular removed as all-ID, etc.), modalities may be empty.
        # An empty list passed to recommend_models() produces nonsensical results —
        # guard it here before any downstream processing.
        if not modalities:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No usable modalities remain after pre-filtering. "
                    "All detected tabular columns appear to be IDs/targets, "
                    "and no text/image columns were found. "
                    "Check your dataset schema or re-run /preprocess."
                ),
            )
        problem_type: str = str(
            body.get("problem_type")
            or (getattr(_ctx_for_budget, "global_schema", {}) or {}).get("global_problem_type")
            or "classification_binary"
        )
        _target_for_problem = (
            body.get("target_column")
            or _active_contract.get("target_column")
            or getattr(_ctx_for_budget, "global_target", None)
        )
        if _target_for_problem:
            problem_type = _refresh_problem_type_from_target(
                _sid,
                _ctx_for_budget,
                str(_target_for_problem),
                default=problem_type,
            )

        # Pre-filter: remove "tabular" when the schema shows no usable tabular columns
        # (all detected tabular cols are the target or ID/path columns). This prevents
        # the JIT selector and Optuna from wasting budget on a modality that Phase 3
        # will auto-skip anyway.
        _excluded_modalities_pre: Dict[str, str] = {}
        if "tabular" in modalities and _ctx_for_budget is not None:
            _gs = getattr(_ctx_for_budget, "global_schema", {}) or {}
            _per_ds = _gs.get("per_dataset", [{}]) or [{}]
            _tab_detected: set = set()
            _id_like: set = set()
            for _ds_entry in _per_ds:
                _tab_detected.update(
                    (_ds_entry.get("detected_columns") or {}).get("tabular") or []
                )
                _id_like.update(_ds_entry.get("id_like_columns") or [])
            _target_col = str(
                _gs.get("primary_target")
                or _gs.get("target_column")
                or _gs.get("global_target_column")
                or getattr(_ctx_for_budget, "global_target", "")
                or ""
            )
            _usable_tab = {
                c for c in _tab_detected
                if c != _target_col
                and c not in _id_like
                and not _is_probably_id_or_path_column(c)
            }
            if _tab_detected and not _usable_tab:
                modalities = [m for m in modalities if m != "tabular"]
                _excluded_modalities_pre["tabular"] = (
                    f"All detected tabular columns ({sorted(_tab_detected)}) are "
                    f"either the target ('{_target_col}') or ID/path columns — "
                    f"no usable features remain after filtering."
                )
                logger.info(
                    "/select-model: 'tabular' pre-filtered — detected=%s id_like=%s target='%s'",
                    sorted(_tab_detected), sorted(_id_like & _tab_detected), _target_col,
                )

        dataset_size: int = int(body.get("dataset_size") or 10_000)
        avg_tokens: int = int(body.get("avg_tokens") or 128)
        latency_budget_ms = body.get("latency_budget_ms")
        memory_budget_mb = body.get("memory_budget_mb")

        if latency_budget_ms is None and _ctx_for_budget is not None:
            latency_budget_ms = getattr(_ctx_for_budget, "latency_budget_ms", None)
        if memory_budget_mb is None and _ctx_for_budget is not None:
            memory_budget_mb = getattr(_ctx_for_budget, "memory_budget_mb", None)

        # FIX-20: use /preprocess cached sample for data-driven probing.
        _probe_X: Optional[Any] = None
        _probe_y: Optional[Any] = None
        if _sid:
            _probe_sample = None
            try:
                _probe_sample = context_db.load_probe_sample(_sid)
            except Exception as _probe_db_exc:
                logger.warning(
                    "/select-model: failed loading persisted probe sample for %s: %s",
                    _sid,
                    _probe_db_exc,
                )

            if _probe_sample is not None:
                _probe_X, _probe_y = _probe_sample
                logger.info(
                    "/select-model: loaded persisted probe sample from SQLite (n=%d)",
                    len(_probe_X) if _probe_X is not None else 0,
                )
            elif ALLOW_LEGACY_SESSION_FALLBACK:
                with _session_lock:
                    _probe_cache = _session_store.get(_sid, {}).get("tabular_sample")
                if _probe_cache:
                    _probe_X = _probe_cache.get("X")
                    _probe_y = _probe_cache.get("y")
                    logger.info(
                        "/select-model: using in-memory tabular probe sample "
                        "(n=%d)",
                        len(_probe_X) if _probe_X is not None else 0,
                    )
                else:
                    logger.info(
                        "/select-model: no probe cache for session %s - "
                        "heuristic ranking will be used",
                        _sid,
                    )
            else:
                logger.info(
                    "/select-model: no persisted probe sample for session %s "
                    "(legacy fallback disabled) - heuristic ranking will be used",
                    _sid,
                )

        from automl.advanced_selector import AdvancedModelSelector

        selector = AdvancedModelSelector()
        predictability_scores = None
        if _ctx_for_budget is not None:
            if hasattr(_ctx_for_budget, "get_effective_predictability_scores"):
                try:
                    predictability_scores = dict(
                        _ctx_for_budget.get_effective_predictability_scores()
                    )
                except Exception:
                    predictability_scores = None
            if not predictability_scores:
                predictability_scores = dict(
                    getattr(_ctx_for_budget, "predictability_scores", {}) or {}
                )
        recommendations: List[Dict[str, Any]] = await asyncio.to_thread(
            selector.recommend_models,
            problem_type=problem_type,
            modalities=modalities,
            dataset_size=dataset_size,
            avg_tokens=avg_tokens,
            tabular_X=_probe_X,
            tabular_y=_probe_y,
            latency_budget_ms=latency_budget_ms,
            memory_budget_mb=memory_budget_mb,
            predictability_scores=predictability_scores,
        )
        if recommendations and isinstance(recommendations[0], dict):
            recommendations[0]["eligible_modalities"] = list(modalities)
            recommendations[0]["active_modalities"] = list(modalities)
            if _excluded_modalities_pre:
                _rec_excluded = dict(recommendations[0].get("excluded_modalities", {}) or {})
                _rec_excluded.update(_excluded_modalities_pre)
                recommendations[0]["excluded_modalities"] = _rec_excluded

        probe_scores: Dict[str, Any] = {}
        selection_metadata: Dict[str, Any] = {}
        fusion_probe: Dict[str, Any] = {}

        tabular_probe_scores: Dict[str, Dict[str, Any]] = {}
        ranked_tabular_candidates: List[Dict[str, Any]] = []

        if recommendations and isinstance(recommendations[0], dict):
            _embedded_probe = recommendations[0].get("tabular_probe_scores")
            if not _embedded_probe:
                _embedded_probe = (
                    (recommendations[0].get("probe_scores") or {}).get("tabular")
                )
            if isinstance(_embedded_probe, dict) and _embedded_probe:
                tabular_probe_scores = _embedded_probe
                probe_scores["tabular"] = tabular_probe_scores

            _embedded_meta = recommendations[0].get("selection_metadata")
            if isinstance(_embedded_meta, dict):
                selection_metadata = dict(_embedded_meta)
            elif tabular_probe_scores:
                top_probe_model = recommendations[0].get("tabular_probe_top_model")
                top_probe_score = recommendations[0].get("quick_probe_score")
                selection_metadata = {
                    "probe_method": "tabular_3fold_cv",
                    "top_probe_model": top_probe_model,
                    "top_probe_score": top_probe_score,
                    "probe_scores": tabular_probe_scores,
                }

            _embedded_ranked = (recommendations[0].get("ranked_candidates") or {}).get("tabular")
            if isinstance(_embedded_ranked, list) and _embedded_ranked:
                ranked_tabular_candidates = list(_embedded_ranked)

        if not tabular_probe_scores and _probe_X is not None and _probe_y is not None:
            try:
                from automl.candidate_selector import CandidateSelector, TABULAR_CANDIDATE_POOL

                probe_selector = CandidateSelector()
                probe_X = _probe_X.toarray() if hasattr(_probe_X, "toarray") else np.asarray(_probe_X)
                probe_y_raw = np.asarray(_probe_y)

                if probe_y_raw.ndim > 1 and probe_y_raw.shape[1] > 1:
                    probe_y = np.argmax(probe_y_raw, axis=1)
                else:
                    probe_y_flat = probe_y_raw.ravel()
                    if probe_y_flat.dtype.kind in ("U", "S", "O"):
                        probe_y = pd.factorize(probe_y_flat)[0]
                    else:
                        try:
                            probe_y = probe_y_flat.astype(int)
                        except Exception:
                            probe_y = pd.factorize(probe_y_flat)[0]

                tabular_probe_scores = probe_selector.quick_probe_tabular(
                    list(TABULAR_CANDIDATE_POOL),
                    probe_X,
                    probe_y,
                    problem_type,
                )
                probe_scores["tabular"] = tabular_probe_scores

                score_map = {
                    model_name: float(details.get("val_score", 0.0) or 0.0)
                    for model_name, details in tabular_probe_scores.items()
                    if isinstance(details, dict)
                }

                top_probe_model: Optional[str] = None
                top_probe_score: Optional[float] = None
                if score_map:
                    top_probe_model = max(score_map, key=score_map.get)
                    top_probe_score = float(score_map[top_probe_model])
                    if recommendations:
                        recommendations[0]["tabular_probe_top_model"] = top_probe_model
                        recommendations[0]["quick_probe_score"] = top_probe_score
                        recommendations[0]["probe_score"] = top_probe_score

                complexity = probe_selector.compute_data_complexity(probe_X, np.asarray(probe_y))
                confidence = (
                    probe_selector.compute_selection_confidence(score_map)
                    if score_map
                    else None
                )

                selection_metadata = {
                    "probe_method": "tabular_3fold_cv",
                    "top_probe_model": top_probe_model,
                    "top_probe_score": top_probe_score,
                    "probe_scores": tabular_probe_scores,
                    "data_complexity": complexity,
                    "selection_confidence": (
                        round(float(confidence), 4)
                        if isinstance(confidence, (int, float))
                        else None
                    ),
                }

                if len(modalities) >= 2 and score_map:
                    try:
                        probed_fusion, fusion_scores = probe_selector.probe_fusion(
                            {"tabular": top_probe_model},
                            {
                                "tabular": probe_X,
                                "labels": np.asarray(probe_y),
                            },
                            max_samples=min(int(len(probe_y)), 500),
                        )
                        fusion_probe = {
                            "selected_strategy": probed_fusion,
                            "scores": fusion_scores,
                            "method": "joint_probe",
                        }
                    except Exception as fusion_exc:
                        logger.warning(
                            "/select-model: fusion probe failed (non-fatal): %s",
                            fusion_exc,
                        )
            except Exception as probe_exc:
                logger.warning(
                    "/select-model: probe diagnostics unavailable (non-fatal): %s",
                    probe_exc,
                )

        if not ranked_tabular_candidates and tabular_probe_scores:
            ranked_tabular_candidates = sorted(
                [
                    {
                        "name": model_name,
                        "val_score": float(details.get("val_score", 0.0) or 0.0),
                        "latency_ms": float(details.get("latency_ms", 0.0) or 0.0),
                        "uncertainty": float(details.get("uncertainty", 0.0) or 0.0),
                        "confidence": details.get("confidence"),
                    }
                    for model_name, details in tabular_probe_scores.items()
                    if isinstance(details, dict)
                ],
                key=lambda row: row.get("val_score", 0.0),
                reverse=True,
            )

        if recommendations and _probe_X is not None and "probe_score" not in recommendations[0]:
            quick_probe = recommendations[0].get("quick_probe_score")
            if isinstance(quick_probe, (int, float)):
                recommendations[0]["probe_score"] = float(quick_probe)

        # VRAM filtering transparency (derived from real candidate memory specs).
        if recommendations:
            jit_dry_run: Dict[str, Any] = {}
            try:
                from automl.jit_encoder_selector import JITEncoderSelector

                _jit_batch_size = int(recommendations[0].get("batch_size", 16) or 16)
                _jit_selector = JITEncoderSelector(batch_size=max(1, _jit_batch_size))
                _jit_result = _jit_selector.select(
                    modalities=modalities,
                    device=torch.device("cuda:0") if torch.cuda.is_available() else None,
                )
                jit_dry_run = {
                    "selected_image_encoder": _jit_result.image_encoder_name,
                    "selected_text_encoder": _jit_result.text_encoder_name,
                    "vram_budget_bytes": int(_jit_result.vram_budget_bytes),
                    "peak_memory_bytes": int(_jit_result.total_peak_memory_bytes),
                    "dry_run_attempts": None,
                    "rationale": dict(_jit_result.rationale or {}),
                }
            except Exception as jit_exc:
                logger.warning("/select-model: JIT dry-run rationale unavailable: %s", jit_exc)

            try:
                from automl.candidate_selector import (
                    IMAGE_CANDIDATE_POOL,
                    TABULAR_CANDIDATE_POOL,
                    TEXT_CANDIDATE_POOL,
                )

                pool_map = {
                    "tabular": list(TABULAR_CANDIDATE_POOL),
                    "text": list(TEXT_CANDIDATE_POOL),
                    "image": list(IMAGE_CANDIDATE_POOL),
                }
                best_model = recommendations[0]
                hw_info = best_model.get("hardware_info", {}) if isinstance(best_model, dict) else {}
                gpu_mem = float(hw_info.get("gpu_memory_gb", 0.0) or 0.0)
                budget_mb = int(gpu_mem * 1024 * 0.70) if gpu_mem > 0 else None

                vram_excluded: Dict[str, List[Dict[str, Any]]] = {}
                vram_kept: Dict[str, List[Dict[str, Any]]] = {}

                for mod in modalities:
                    if mod not in pool_map:
                        continue
                    pool = pool_map[mod]
                    if budget_mb is None:
                        vram_kept[mod] = [
                            {
                                "name": str(item.get("name")),
                                "required_mb": int(item.get("vram_mb", 0) or 0),
                            }
                            for item in pool
                        ]
                        vram_excluded[mod] = []
                        continue

                    kept_mod: List[Dict[str, Any]] = []
                    excluded_mod: List[Dict[str, Any]] = []
                    for item in pool:
                        required_mb = int(item.get("vram_mb", 0) or 0)
                        candidate_entry = {
                            "name": str(item.get("name")),
                            "required_mb": required_mb,
                        }
                        if required_mb <= budget_mb:
                            kept_mod.append(candidate_entry)
                        else:
                            candidate_entry["reason"] = (
                                f"required {required_mb}MB > budget {budget_mb}MB"
                            )
                            excluded_mod.append(candidate_entry)

                    if not kept_mod and pool:
                        lightest = min(pool, key=lambda x: int(x.get("vram_mb", 0) or 0))
                        kept_mod = [
                            {
                                "name": str(lightest.get("name")),
                                "required_mb": int(lightest.get("vram_mb", 0) or 0),
                                "forced_keep": True,
                            }
                        ]

                    vram_kept[mod] = kept_mod
                    vram_excluded[mod] = excluded_mod

                vram_filter_report = {
                    "gpu_memory_gb": gpu_mem,
                    "vram_budget_mb": budget_mb,
                    "kept": vram_kept,
                    "excluded": vram_excluded,
                    "excluded_counts": {
                        mod: len(items) for mod, items in vram_excluded.items()
                    },
                    "method": "jit_vram_budget_70pct",
                }
            except Exception as vram_exc:
                logger.warning(
                    "/select-model: VRAM transparency build failed (non-fatal): %s",
                    vram_exc,
                )
                vram_filter_report = {}

            if not fusion_probe:
                fusion_choices = (
                    ((recommendations[0].get("hpo_space") or {}).get("fusion_strategy") or {}).get("choices")
                    or []
                )
                fusion_probe = {
                    "selected_strategy": recommendations[0].get("fusion_strategy"),
                    "candidate_strategies": fusion_choices,
                    "scores": {},  # empty — not measured; use priority_weights below
                    "priority_weights": {
                        str(name): round(
                            1.0 - (idx / max(1, len(fusion_choices) - 1)),
                            3,
                        )
                        for idx, name in enumerate(fusion_choices)
                    },
                    "method": "hpo_priority_order",
                    "is_measured": False,
                }
            if (
                {"image", "text"}.issubset(set(modalities))
                and str(recommendations[0].get("fusion_strategy", "")).lower() == "ula"
            ):
                fusion_choices = list(
                    ((recommendations[0].get("hpo_space") or {}).get("fusion_strategy") or {}).get("choices")
                    or fusion_probe.get("candidate_strategies")
                    or []
                )
                if "ula" in fusion_choices:
                    fusion_choices = ["ula"] + [f for f in fusion_choices if f != "ula"]
                    recommendations[0].setdefault("hpo_space", {}).setdefault(
                        "fusion_strategy", {"type": "categorical"}
                    )["choices"] = fusion_choices
                fusion_probe.update(
                    {
                        "selected_strategy": "ula",
                        "candidate_strategies": fusion_choices,
                        "method": "policy_text_image_gpu",
                        "is_measured": False,
                    }
                )

            recommendations[0]["probe_scores"] = probe_scores
            recommendations[0]["selection_metadata"] = selection_metadata
            recommendations[0]["fusion_probe"] = fusion_probe
            recommendations[0]["vram_filter_report"] = vram_filter_report
            recommendations[0]["selection_contract_version"] = "model_selection.v2"
            if ranked_tabular_candidates:
                recommendations[0]["ranked_candidates"] = {
                    "tabular": ranked_tabular_candidates
                }
            if jit_dry_run:
                recommendations[0]["jit_dry_run"] = jit_dry_run
                # Rebuild model name from actual JIT-selected encoders — the tier template
                # names (ResNet50, BERT-base, GRN) are rule-based placeholders that JIT
                # always overrides with hardware-profiled selections.
                _jit_img_name = jit_dry_run.get("selected_image_encoder")
                _jit_txt_name = jit_dry_run.get("selected_text_encoder")
                if _jit_img_name:
                    recommendations[0]["tier_template_image_encoder"] = recommendations[0].get("image_encoder")
                    recommendations[0]["image_encoder"] = _jit_img_name
                if _jit_txt_name:
                    recommendations[0]["tier_template_text_encoder"] = recommendations[0].get("text_encoder")
                    recommendations[0]["text_encoder"] = _jit_txt_name
                _jit_fusion = recommendations[0].get("fusion_strategy", "")
                _fusion_labels = {
                    "ula": "[ULA]", "attention": "[Attention]",
                    "graph": "[RGAT]", "uncertainty": "[Uncertainty]",
                    "uncertainty_graph": "[UncertaintyGraph]",
                }
                _fusion_label = _fusion_labels.get(str(_jit_fusion).lower(), "")
                _jit_name_parts = [p for p in [_jit_img_name, _jit_txt_name] if p]
                if _jit_name_parts:
                    recommendations[0]["name"] = (
                        " + ".join(_jit_name_parts)
                        + (f" {_fusion_label}" if _fusion_label else "")
                    )

        # Store best selection so /train-pipeline can consume it
        if _sid and recommendations:
            _write_legacy_session_cache(_sid, "model_selection", recommendations[0])

        # Update ExecutionContext with model selection
        try:
            if _sid and recommendations:
                ctx = session_manager.get_session(_sid)
                if ctx:
                    _apply_active_modality_contract(ctx)
                    best = recommendations[0]
                    top_probe = best.get("selection_metadata", {}).get("top_probe_model")
                    model_reason = "AdvancedModelSelector recommendations"
                    if isinstance(top_probe, str) and top_probe:
                        model_reason += f" + tabular probe winner: {top_probe}"
                    ctx.update_model_selection(
                        recommendations,
                        model_reason,
                    )
                    ctx.fusion_strategy = best.get("fusion_strategy")
                    ctx.fusion_policy_source = "selector_recommendation"
                    ctx.fusion_policy_locked = bool(best.get("fusion_strategy"))
                    ctx.selected_model = best.get("name")
                    ctx.set_pipeline_stage("model_selection")
                    if probe_scores:
                        ctx.probe_scores_cache = probe_scores
                    if ranked_tabular_candidates:
                        ctx.ranked_candidates = {
                            "tabular": list(ranked_tabular_candidates)
                        }
                    _warm = best.get("warm_start_params", {}) if isinstance(best, dict) else {}
                    if isinstance(_warm, dict) and _warm:
                        ctx.warm_start_params = dict(_warm)
                    if latency_budget_ms is not None:
                        try:
                            ctx.latency_budget_ms = float(latency_budget_ms)
                        except Exception:
                            pass
                    if memory_budget_mb is not None:
                        try:
                            ctx.memory_budget_mb = float(memory_budget_mb)
                        except Exception:
                            pass
                    ctx.update_fusion(
                        best.get("fusion_strategy"),
                        dict(getattr(ctx, "modality_importance", {}) or {}),
                    )
                    _apply_active_modality_contract(ctx)
                    ctx.fusion_policy_source = "selector_recommendation"
                    session_manager.update_session_context(_sid, ctx)
        except OptimisticLockError:
            raise
        except Exception as _ctx_exc:
            logger.warning("/select-model: ExecutionContext update failed: %s", _ctx_exc)

        _ctx_after = session_manager.get_session(_sid) if _sid else _ctx_for_budget
        _return_modality_contract = _resolve_active_modality_contract(_ctx_after)
        _contract = _context_contract_payload(_ctx_after)
        _best_model = recommendations[0] if recommendations else None
        _fusion_policy = (
            _best_model.get("fusion_strategy")
            if isinstance(_best_model, dict)
            else None
        )

        return {
            "status":             "success",
            "selection_contract_version": "model_selection.v2",
            "problem_type":       problem_type,
            "modalities":         modalities,
            "detected_modalities": _return_modality_contract["detected_modalities"],
            "active_modalities": _return_modality_contract["active_modalities"],
            "context_stage":      _contract["context_stage"],
            "context_version":    _contract["context_version"],
            "artifact_versions":  _contract["artifact_versions"],
            "eligible_modalities": (
                _return_modality_contract["active_modalities"]
                or recommendations[0].get("eligible_modalities", modalities)
                if recommendations
                else modalities
            ),
            "excluded_modalities": {
                **{
                    k: ("id_only_or_target_only" if k == "tabular" else v)
                    for k, v in _excluded_modalities_pre.items()
                },
                **_return_modality_contract["excluded_modalities"],
                **(
                    recommendations[0].get("excluded_modalities", {})
                    if recommendations
                    else {}
                ),
            },
            "fusion_policy": _fusion_policy,
            "policy_source": (
                _contract["fusion_policy_source"]
                or "selector_recommendation"
            ),
            "recommended_models": recommendations,
            "best_model":         _best_model,
        }

    except OptimisticLockError:
        raise
    except Exception as exc:
        logger.error("/select-model error: %s", exc, exc_info=True)
        return JSONResponse(
            {"error": "Model selection failed. Check server logs for details."},
            status_code=500,
        )


# ---------------------------------------------------------------------------
# Train pipeline  (replaces the defunct /train-model stub)
# ---------------------------------------------------------------------------

@app.post("/train-pipeline")
@require_context("preprocessing_planning", required_fields=["global_schema"], require_session=True)
async def train_pipeline(request: Request) -> Dict[str, Any]:
    """
    Start the training pipeline as a background task.

    Returns a ``task_id`` immediately.  The frontend polls
    ``GET /train-pipeline/status/{task_id}`` for real-time progress.

    FRONTEND CONTRACT
    -----------------
    Immediate response::

        {"status": "started", "task_id": "<8-char hex>"}

    Poll response (GET /train-pipeline/status/{task_id})::

        {
          "task_id":            str,
          "status":             "running" | "completed" | "failed",
          "current_phase":      int (1-5),
          "current_phase_name": str,
          "progress_pct":       int (0-100),
          "messages":           [{phase, type, text, timestamp}, ...],
          "result":             null | {final metrics dict},
          "error":              null | str
        }
    """
    try:
        from pipeline.training_orchestrator import (
            TrainingOrchestrator,
            TrainingConfig,
            Phase,
        )

        body = await request.json()
        _sid = body.get("session_id") if isinstance(body, dict) else None
        _training_ctx = _get_session_context_or_422(_sid, "/train-pipeline")
        _session_cached_schema = _require_context_artifact(
            _sid,
            "global_schema",
            "/train-pipeline",
        )
        # Bug #3: prefer explicit target override from request body (user schema override)
        _body_target_override: Optional[str] = (
            str(body.get("target_column")).strip()
            if body.get("target_column")
            else None
        ) if isinstance(body, dict) else None
        _session_target_override = (
            _body_target_override
            or getattr(_training_ctx, "global_target", None)
        )
        # Write override back into session context so other endpoints see it
        if _body_target_override and _training_ctx is not None:
            try:
                if hasattr(_training_ctx, "override_global_target"):
                    _training_ctx.override_global_target(
                        _body_target_override, "/train-pipeline body"
                    )
                else:
                    _training_ctx.global_target = _body_target_override
                _update_session_context_with_retry(str(_sid), _training_ctx)
                # G3: Persist target override to context_db so it survives restart.
                try:
                    for _ds_id in list(_training_ctx.active_dataset_ids or []):
                        _prof = context_db.load_profile(_ds_id) or {"dataset_id": _ds_id}
                        _prof["chosen_target"] = _body_target_override
                        _prof["target_locked"] = True
                        _prof["target_override_reason"] = "/train-pipeline body override"
                        context_db.save_profile(_prof, str(_sid))
                except Exception as _persist_exc:
                    logger.warning(
                        "/train-pipeline: target persist to context_db failed: %s",
                        _persist_exc,
                    )
            except Exception as _tgt_exc:
                logger.warning("/train-pipeline: target cascade failed: %s", _tgt_exc)

        # STRICT SESSION ISOLATION — snapshot under lock to prevent TOCTOU race
        with _session_lock:
            _snapshot = _get_session_hashes(_sid)
            if not _snapshot:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "No active ingestion session. "
                        "Call POST /ingest/datasets first to register datasets."
                    ),
                )

        problem_type: str = body.get("problem_type", "classification_binary")
        modalities: List[str] = _validate_modalities(
            list(body.get("modalities") or ["tabular"]), "/train-pipeline"
        )
        hp_overrides: Optional[Dict[str, Any]] = _validate_hp_overrides(body.get("hp_overrides"))
        # Bug 12: extract retraining_depth sent by frontend ("full" | "head_only" | "calibration_only")
        _retraining_depth: str = str(body.get("retraining_depth", "full") or "full")
        if _retraining_depth not in ("full", "head_only", "calibration_only"):
            _retraining_depth = "full"
        logger.info("/train-pipeline: retraining_depth=%s", _retraining_depth)
        # Explicit column-type assignments from the preprocessing step
        _body_text_columns: List[str] = list(body.get("text_columns") or [])
        _body_image_columns: List[str] = list(body.get("image_columns") or [])

        sources: List[str] = [
            meta.get("source_url", hid)
            for hid, meta in _snapshot.items()
        ]

        _session_lazy_datasets: Dict[str, Any] = {}
        if _sid:
            try:
                _session_lazy_datasets = get_session_datasets(_sid) or {}
            except Exception as _ds_exc:
                logger.warning(
                    "_run_training: failed to read session datasets: %s",
                    _ds_exc,
                )

        if _sid:
            try:
                _ctx_latest = session_manager.get_session(_sid)
                if _ctx_latest is not None:
                    _training_ctx = _ctx_latest
            except Exception:
                pass

        if _training_ctx is not None:
            _training_contract = _apply_active_modality_contract(_training_ctx)
            if _session_target_override:
                problem_type = _refresh_problem_type_from_target(
                    _sid,
                    _training_ctx,
                    str(_session_target_override),
                    default=problem_type,
                )
            elif _training_contract.get("target_column"):
                problem_type = _refresh_problem_type_from_target(
                    _sid,
                    _training_ctx,
                    str(_training_contract["target_column"]),
                    default=problem_type,
                )
            _active_modalities_for_training = list(_training_contract.get("active_modalities") or [])
            if _active_modalities_for_training:
                modalities = _active_modalities_for_training

        try:
            context_validation = ensure_session_context(
                _training_ctx,
                session_id=_sid,
                dataset_snapshot=_snapshot,
            )
            for warning in context_validation.warnings:
                logger.warning("/train-pipeline context warning: %s", warning)
        except ContextValidationError as ctx_exc:
            raise HTTPException(
                status_code=400,
                detail=f"ExecutionContext validation failed: {ctx_exc}",
            ) from ctx_exc

        if _training_ctx is not None:
            try:
                ContextValidator.require_schema(
                    _training_ctx,
                    phase="/train-pipeline preflight",
                )
                ContextValidator.require_modality_consistency(
                    _training_ctx,
                    modalities,
                    phase="/train-pipeline preflight",
                )
                ContextValidator.require_fusion_consistency(
                    fusion_strategy=str(
                        getattr(_training_ctx, "fusion_strategy", None)
                        or "ula"      # ULA is system default for text+image; concatenation was wrong default
                    ),
                    modalities=modalities,
                    phase="/train-pipeline preflight",
                )
            except ValueError as ctx_contract_exc:
                raise HTTPException(status_code=422, detail=str(ctx_contract_exc)) from ctx_contract_exc

        if not sources and not _session_lazy_datasets:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No datasets available for training. "
                    "Call POST /ingest/datasets first."
                ),
            )

        # Concurrent training guard: reject if a training task is already running for this session.
        # _active_training_tasks is volatile (cleared on API restart) so stale entries are
        # automatically evicted. Check task status to confirm it's truly still running.
        if _sid:
            with _active_training_lock:
                _existing_task_id = _active_training_tasks.get(_sid)
            if _existing_task_id:
                _existing_task = task_db.get_task(_existing_task_id)
                if _existing_task and _existing_task.get("status") == "running":
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            f"Training already in progress for session {_sid} "
                            f"(task_id={_existing_task_id}). "
                            "Wait for it to complete or cancel it first."
                        ),
                    )
                else:
                    # Old task finished or doesn't exist — evict stale entry
                    with _active_training_lock:
                        _active_training_tasks.pop(_sid, None)

        task_id = uuid.uuid4().hex[:8]
        if _sid:
            with _active_training_lock:
                _active_training_tasks[_sid] = task_id
        tracker = TrainingProgressTracker(task_id, task_db)
        tracker.set_phase(0, "Context Preflight", 1)
        tracker.set_substage("context_preflight")
        tracker.add_message(
            0,
            "info",
            f"Validating context: problem={problem_type}, modalities={modalities}, "
            f"fusion={getattr(_training_ctx, 'fusion_strategy', None) if _training_ctx is not None else 'unknown'}",
        )
        task_db.merge_payload(task_id, {
            "selected_fusion": getattr(_training_ctx, "fusion_strategy", None) if _training_ctx is not None else None,
            "active_modalities": list(modalities),
            "excluded_modalities": dict(getattr(_training_ctx, "excluded_modalities", {}) or {}) if _training_ctx is not None else {},
        })

        logger.info(
            "/train-pipeline: task=%s  problem=%s  modalities=%s  sources=%d  hp_overrides=%s",
            task_id, problem_type, modalities, len(sources), hp_overrides,
        )

        async def _run_training() -> None:
            nonlocal _training_ctx
            try:
                if _sid:
                    ensure_session_context(
                        _training_ctx,
                        session_id=_sid,
                        dataset_snapshot=_snapshot,
                    )

                # Persist retraining depth to context so orchestrator can skip phases
                if _training_ctx is not None and _retraining_depth != "full":
                    try:
                        object.__setattr__(_training_ctx, "retraining_depth", _retraining_depth)
                    except Exception:
                        pass

                orchestrator = TrainingOrchestrator(
                    TrainingConfig(
                        dataset_sources=sources,
                        problem_type=problem_type,
                        modalities=modalities,
                    ),
                    execution_context=_training_ctx,
                )

                if _session_lazy_datasets:
                    orchestrator.inject_external_datasets(_session_lazy_datasets)

                _canonical_target_override = _session_target_override
                if _training_ctx is not None:
                    _ctx_target = getattr(_training_ctx, "global_target", None)
                    if _ctx_target:
                        _canonical_target_override = _ctx_target

                # Inject pre-computed schema so Phase 2 is skipped in the orchestrator.
                # Merge explicit text/image column overrides from the request body into
                # the per_dataset detected_columns so Phase 3 uses the right columns.
                if isinstance(_session_cached_schema, dict) and _session_cached_schema:
                    _schema_for_inject = dict(_session_cached_schema)
                    if _body_text_columns or _body_image_columns:
                        _per_ds = list(_schema_for_inject.get("per_dataset") or [])
                        for _ds in _per_ds:
                            if not isinstance(_ds, dict):
                                continue
                            _det = dict(_ds.get("detected_columns") or {})
                            if _body_text_columns:
                                _det["text"] = _body_text_columns
                            if _body_image_columns:
                                _det["image"] = _body_image_columns
                            _ds["detected_columns"] = _det
                        _schema_for_inject["per_dataset"] = _per_ds
                    orchestrator.inject_external_schema(
                        _schema_for_inject,
                        target_override=_canonical_target_override,
                    )

                # Inject pre-fitted scaler if /preprocess was called
                _scaler_path = Path(f"./data/session_cache/{_sid}/tabular_scaler.joblib") if _sid else None
                if _scaler_path and _scaler_path.exists():
                    orchestrator.inject_external_preprocessors(str(_scaler_path))

                # Inject model selection if /select-model was called
                _session_model_sel_raw = (
                    getattr(_training_ctx, "model_choices", None)
                    if _training_ctx is not None
                    else None
                )
                _session_model_sel = None
                if isinstance(_session_model_sel_raw, dict):
                    _session_model_sel = _session_model_sel_raw
                elif isinstance(_session_model_sel_raw, list) and _session_model_sel_raw:
                    if isinstance(_session_model_sel_raw[0], dict):
                        _session_model_sel = _session_model_sel_raw[0]
                if _session_model_sel:
                    orchestrator.inject_external_model_selection(_session_model_sel)

                # Phase 1 — Data Ingestion
                tracker.set_phase(1, "Data Ingestion", 5)
                if _session_lazy_datasets:
                    tracker.add_message(
                        1,
                        "info",
                        f"Reusing {len(_session_lazy_datasets)} pre-ingested session dataset(s)...",
                    )
                else:
                    tracker.add_message(1, "info", "Registering datasets for training...")

                phase1_res = await orchestrator._execute_phase_1_data_ingestion(sources=sources)
                tracker.add_message(
                    1,
                    "result",
                    f"Registered {len(phase1_res.get('registered_datasets', []))} dataset(s)",
                )

                # Phase 2 — Schema Detection
                tracker.set_phase(2, "Schema Detection", 15)
                tracker.add_message(2, "info", "Analyzing column types and problem type...")
                await asyncio.to_thread(orchestrator._execute_phase_2_schema_detection)
                schema = orchestrator.phase_results.get(Phase.SCHEMA_DETECTION, {})
                tracker.add_message(2, "result",
                    f"Problem: {schema.get('global_problem_type', '?')}")
                tracker.add_message(2, "detail",
                    f"Modalities: {', '.join(schema.get('global_modalities', []))}")
                tracker.add_message(2, "detail",
                    f"Target: {schema.get('primary_target', '?')}")
                n_classes = schema.get("n_classes")
                if n_classes:
                    tracker.add_message(2, "detail", f"Classes: {n_classes}")

                # Phase 3 — Preprocessing
                tracker.set_phase(3, "Preprocessing", 30)
                tracker.add_message(3, "info", "Materializing datasets and fitting transformers...")
                await asyncio.to_thread(orchestrator._execute_phase_3_preprocessing)
                prep = orchestrator.phase_results.get(Phase.PREPROCESSING, {})
                tracker.add_message(3, "result",
                    f"Total samples: {prep.get('total_samples', '?')}")
                # Report smart-filtered columns
                tab_prep = orchestrator.fitted_transformers.get("tabular")
                if tab_prep and hasattr(tab_prep, "_dropped_cols") and tab_prep._dropped_cols:
                    tracker.add_message(3, "detail",
                        f"Dropped {len(tab_prep._dropped_cols)} useless columns: "
                        f"{tab_prep._dropped_cols}")
                if tab_prep and hasattr(tab_prep, "get_output_dim"):
                    tracker.add_message(3, "detail",
                        f"Tabular output dim: {tab_prep.get_output_dim()}")
                for stage in prep.get("preprocessing_stages", []):
                    tracker.add_message(3, "detail",
                        f"{stage.get('stage', '?')}: {stage.get('output_shape', '?')}")

                # Phase 4 — Model Selection
                tracker.set_phase(4, "Model Selection", 45)
                tracker.add_message(4, "info", "Running AdvancedModelSelector...")
                await asyncio.to_thread(orchestrator._execute_phase_4_model_selection)
                model_sel = orchestrator.phase_results.get(Phase.MODEL_SELECTION, {})
                parts = []
                for key in ("text_encoder_name", "tabular_encoder_name", "image_encoder_name"):
                    val = model_sel.get(key)
                    if val:
                        parts.append(val)
                tracker.add_message(4, "result",
                    f"Selected: {' + '.join(parts) or 'Default'}")
                tracker.add_message(4, "detail",
                    f"Fusion: {model_sel.get('fusion_strategy', '?')}, "
                    f"Batch: {model_sel.get('batch_size', '?')}")

                # Phase 5 — Training
                tracker.set_phase(5, "Training", 55)
                tracker.set_substage("phase5_bootstrap")
                if orchestrator.fitted_transformers.get("text") is not None:
                    tracker.add_message(5, "info",
                        "Loading BERT encoder (~440MB on first run)...")
                if hp_overrides:
                    tracker.add_message(5, "info",
                        f"Using manual HP overrides: {hp_overrides}")
                else:
                    tracker.add_message(5, "info", "Starting Optuna HPO study...")

                # Attach log handler to capture trial-level messages
                class _ProgressHandler(logging.Handler):
                    def emit(self, record: logging.LogRecord) -> None:
                        msg = record.getMessage()
                        if any(kw in msg for kw in ("Trial", "val_loss", "Epoch", "trial")):
                            tracker.add_message(5, "detail", msg.strip())

                _handler = _ProgressHandler()
                _handler.setLevel(logging.INFO)
                _orch_logger = logging.getLogger("pipeline.training_orchestrator")
                _orch_logger.addHandler(_handler)
                try:
                    await asyncio.to_thread(orchestrator._execute_phase_5_training,
                                            hp_overrides, progress_callback=tracker)
                finally:
                    _orch_logger.removeHandler(_handler)

                phase5 = orchestrator.phase_results.get(Phase.TRAINING, {})
                training_time = f"{phase5.get('duration_seconds', 0):.1f}s"
                tracker.add_message(5, "result",
                    f"Best val_loss: {phase5.get('best_val_loss', 0):.4f}")
                tracker.add_message(5, "detail",
                    f"Trials: {phase5.get('n_trials', '?')}, "
                    f"Best: #{phase5.get('best_trial', '?')}, "
                    f"Time: {training_time}")

                try:
                    if _sid and orchestrator.execution_context is not None:
                        tracker.set_substage("phase5_context_sync")
                        _training_ctx = _update_session_context_with_retry(
                            _sid,
                            orchestrator.execution_context,
                        )
                        orchestrator.execution_context = _training_ctx
                except Exception as _ctx_exc:
                    logger.warning(
                        "_run_training: ExecutionContext training update failed: %s",
                        _ctx_exc,
                    )

                # Phase 6 — Drift Detection
                tracker.set_phase(6, "Drift Detection", 96)
                tracker.set_substage("drift_detection")
                tracker.add_message(6, "info", "Computing KS / PSI / MMD drift statistics...")
                try:
                    await asyncio.to_thread(orchestrator._execute_phase_6_drift_detection)
                    drift_res = orchestrator.phase_results.get(Phase.DRIFT_DETECTION, {})
                    drift_m = drift_res.get("metrics", {})
                    tracker.add_message(6, "result",
                        f"PSI={drift_m.get('psi', 0):.4f}  "
                        f"KS={drift_m.get('ks_statistic', 0):.4f}  "
                        f"MMD={drift_m.get('fdd', 0):.4f}")
                    if drift_res.get("drift_detected"):
                        tracker.add_message(6, "detail", "Drift detected above threshold")
                    else:
                        tracker.add_message(6, "detail", "No significant drift detected")
                except Exception as drift_exc:
                    logger.warning("Phase 6 drift detection failed (non-fatal): %s", drift_exc)
                    tracker.add_message(6, "detail",
                        f"Drift detection skipped: {drift_exc}")

                # Phase 7 — Model Registry (save weights + artifacts)
                tracker.set_phase(7, "Model Registry", 98)
                tracker.set_substage("registry_save")
                tracker.add_message(7, "info", "Saving model weights and artifacts...")
                await asyncio.to_thread(orchestrator._execute_phase_7_model_registry)
                phase7 = orchestrator.phase_results.get(Phase.MODEL_REGISTRY, {})
                model_id = phase7.get("model_id", "unknown")
                deployment_ready = phase7.get("deployment_ready", False)
                tracker.add_message(7, "result", f"Model ID: {model_id}")
                tracker.add_message(7, "detail",
                    f"Deployment ready: {deployment_ready}  "
                    f"Artifacts: {len(phase7.get('artifact_paths', {}))}")

                final_result = {
                    "status": "success",
                    "data": {
                        "model_id": model_id,
                        "context_stage": getattr(orchestrator.execution_context, "pipeline_stage", None),
                        "context_version": getattr(orchestrator.execution_context, "version", None),
                        "artifact_versions": dict(
                            getattr(orchestrator.execution_context, "artifact_versions", {}) or {}
                        ),
                        "fusion_policy": getattr(orchestrator.execution_context, "fusion_strategy", None),
                        "policy_source": getattr(orchestrator.execution_context, "fusion_policy_source", None),
                        "active_modalities": list(
                            getattr(orchestrator.execution_context, "active_modalities", []) or modalities
                        ),
                        "excluded_modalities": dict(
                            getattr(orchestrator.execution_context, "excluded_modalities", {}) or {}
                        ),
                        "metrics": {
                            "final_loss":    phase5.get("best_val_loss", 0.0),
                            "best_val_loss": phase5.get("best_val_loss", 0.0),
                            "best_val_acc":  phase5.get("best_val_acc", 0.0),
                            "best_val_f1":   phase5.get("best_val_f1", 0.0),
                            "best_train_acc": phase5.get("best_train_acc", 0.0),
                            "training_time": training_time,
                            "n_trials":      phase5.get("n_trials", 0),
                            "n_pruned":      phase5.get("n_pruned", 0),
                            "n_complete":    phase5.get("n_complete", 0),
                            "best_trial":    phase5.get("best_trial", 0),
                            "best_params":   phase5.get("best_params", {}),
                            "problem_type":  phase5.get("problem_type", problem_type),
                            "gpu_enabled":   GPU_AVAILABLE,
                            "data_split":    phase5.get("data_split", {}),
                            "fit_type":      phase5.get("fit_type", "unknown"),
                            "trial_diagnostics": phase5.get("trial_diagnostics", []),
                            "lw_schedule_history": phase5.get("lw_schedule_history", []),
                            "trial_feedback_events": phase5.get("trial_feedback_events", []),
                            "trial_feedback_summary": phase5.get("trial_feedback_summary", {}),
                            "next_run_feedback": phase5.get("next_run_feedback", {}),
                            "alignment_summary": phase5.get("alignment_summary", {}),
                            "alignment_loss_history": phase5.get("alignment_loss_history", []),
                            "contrastive_loss_history": phase5.get("contrastive_loss_history", []),
                            "fusion_summary": phase5.get("fusion_summary", {}),
                            "fusion_aux_weights": phase5.get("fusion_aux_weights", {}),
                            "embedding_cache": phase5.get("embedding_cache", {}),
                            "calibration": phase5.get("calibration", {}),
                            "phase_timings": orchestrator.state.snapshot().get("phase_timings", {}),
                        },
                        "deployment_ready": deployment_ready,
                    },
                }

                try:
                    if _sid and orchestrator.execution_context is not None:
                        tracker.set_substage("final_context_sync")
                        _training_ctx = _update_session_context_with_retry(
                            _sid,
                            orchestrator.execution_context,
                        )
                        orchestrator.execution_context = _training_ctx
                except Exception as _ctx_final_exc:
                    logger.warning(
                        "_run_training: final ExecutionContext persistence failed: %s",
                        _ctx_final_exc,
                    )

                tracker.complete(final_result)

            except Exception as exc:
                # Bug 14: prefix error with current phase name for diagnosability
                _phase_name = tracker.current_phase_name if hasattr(tracker, "current_phase_name") else "unknown"
                _err_msg = f"[Phase {_phase_name}] {exc}"
                logger.error("Training task %s failed: %s", task_id, _err_msg, exc_info=True)
                tracker.fail(_err_msg)

            finally:
                # Deregister from active training map so next run can start
                if _sid:
                    with _active_training_lock:
                        if _active_training_tasks.get(_sid) == task_id:
                            _active_training_tasks.pop(_sid, None)

        # Bug 3: training task timeout — wrap in daemon thread that marks FAILED if hung
        _MAX_TRAINING_S = int(os.environ.get("APEX_TRAINING_TIMEOUT_S", "3600"))

        async def _run_training_with_timeout() -> None:
            try:
                await asyncio.wait_for(_run_training(), timeout=_MAX_TRAINING_S)
            except asyncio.TimeoutError:
                _msg = f"Training timed out after {_MAX_TRAINING_S}s (APEX_TRAINING_TIMEOUT_S)"
                logger.error("Training task %s: %s", task_id, _msg)
                try:
                    tracker.fail(_msg)
                except Exception:
                    pass

        # Launch as background asyncio task
        asyncio.create_task(_run_training_with_timeout())

        return {
            "status": "started",
            "task_id": task_id,
            **_context_contract_payload(_training_ctx),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("/train-pipeline error: %s", exc, exc_info=True)
        return JSONResponse(
            {"error": "Training pipeline failed to start. Check server logs for details."},
            status_code=500,
        )


@app.get("/train-pipeline/status/{task_id}")
async def train_pipeline_status(task_id: str) -> Dict[str, Any]:
    """Poll training progress for a given task_id."""
    task = task_db.get_task(task_id)
    if task is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown task_id: {task_id}",
        )
    payload = task.get("payload", {})
    result_payload = task.get("result") or {}
    phase_timings = {}
    context_stage = None
    context_version = None
    artifact_versions: Dict[str, Any] = {}
    if isinstance(result_payload, dict):
        phase_timings = (
            ((result_payload.get("data") or {}).get("metrics") or {}).get("phase_timings")
            or {}
        )
        context_stage = ((result_payload.get("data") or {}).get("context_stage"))
        context_version = ((result_payload.get("data") or {}).get("context_version"))
        artifact_versions = ((result_payload.get("data") or {}).get("artifact_versions") or {})
    _model_id_from_result = (result_payload.get("data") or {}).get("model_id") if isinstance(result_payload, dict) else None
    return {
        "task_id":            task["task_id"],
        "status":             task["status"],
        "current_phase":      payload.get("current_phase", 0),
        "current_phase_name": payload.get("current_phase_name", "Initializing"),
        "progress_pct":       payload.get("progress_pct", 0),
        "substage":           payload.get("substage"),
        "current_trial":      payload.get("current_trial"),
        "best_so_far":        payload.get("best_so_far"),
        "trial_events":       payload.get("trial_events", []),
        "pruning_status":     payload.get("pruning_status"),
        "next_trial_plan":    payload.get("next_trial_plan"),
        "context_stage":      context_stage,
        "context_version":    context_version,
        "artifact_versions":  artifact_versions,
        "model_id":           _model_id_from_result,
        "selected_fusion":    payload.get("selected_fusion") or ((result_payload.get("data") or {}).get("fusion_policy") if isinstance(result_payload, dict) else None),
        "active_modalities":  payload.get("active_modalities") or ((result_payload.get("data") or {}).get("active_modalities") if isinstance(result_payload, dict) else None),
        "excluded_modalities": payload.get("excluded_modalities") or ((result_payload.get("data") or {}).get("excluded_modalities") if isinstance(result_payload, dict) else {}),
        "messages":           payload.get("messages", []),
        "epoch_metrics":      payload.get("epoch_metrics", []),
        "trial_progress":     payload.get("trial_progress"),
        "data_split":         payload.get("data_split"),
        "phase_timings":      phase_timings,
        "result":             result_payload,
        "error":              task.get("error"),
    }


# ---------------------------------------------------------------------------
# Drift monitoring  (replaces the defunct /monitor-model stub)
# ---------------------------------------------------------------------------

@app.post("/monitor/drift")
@require_context("ingestion_complete", require_session=True)
async def monitor_drift(request: Request) -> Dict[str, Any]:
    """
    Drift detection against cached session data.
    Retraining is triggered when the session has a retraining-capable context.
    """
    try:
        body = await request.json()
        _sid = body.get("session_id") if isinstance(body, dict) else None
        _ctx_monitor = _get_session_context_or_422(_sid, "/monitor/drift")
        model_id: Optional[str] = body.get("model_id")
        if model_id:
            model_id = _sanitize_model_id(model_id)
        else:
            active_id = getattr(_ctx_monitor, "active_prediction_model_id", None)
            model_id = _sanitize_model_id(str(active_id)) if active_id else None

        if not model_id:
            return _not_available_monitor_payload(
                "no_deployment_ready_model_registered",
                _ctx_monitor,
                None,
            )
        _monitor_metadata = _load_model_registry_metadata(model_id)
        if not _monitor_metadata:
            return _not_available_monitor_payload("model_metadata_missing", _ctx_monitor, model_id)
        if not bool(_monitor_metadata.get("deployment_ready", False)):
            return _not_available_monitor_payload("model_not_deployment_ready", _ctx_monitor, model_id)
        # Bug fix: do NOT return NOT_AVAILABLE when reference_sample.npy is missing.
        # The _drift_sync() closure below has a 70/30 split fallback that produces a
        # valid reference distribution from the production data — that fallback is the
        # correct behaviour when Phase 6 didn't persist a reference artifact.
        # Early-returning here made that fallback permanently unreachable.

        # Snapshot under lock to prevent TOCTOU race
        with _session_lock:
            _snapshot = _get_session_hashes(_sid)
            if not _snapshot:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "No active ingestion session. "
                        "Call POST /ingest/datasets first to register datasets."
                    ),
                )

        ensure_session_context(
            _ctx_monitor,
            session_id=_sid,
            dataset_snapshot=_snapshot,
        )

        def _materialize_head(lazy_ref: Any, n_rows: int = 5000) -> Optional[pd.DataFrame]:
            try:
                import polars as pl

                if isinstance(lazy_ref, pl.LazyFrame):
                    return lazy_ref.head(n_rows).collect().to_pandas()
            except ImportError:
                pass

            try:
                import dask.dataframe as dd

                if isinstance(lazy_ref, dd.DataFrame):
                    return lazy_ref.head(n_rows, compute=True)
            except ImportError:
                pass

            if isinstance(lazy_ref, pd.DataFrame):
                return lazy_ref.head(n_rows)

            if hasattr(lazy_ref, "head"):
                try:
                    head_obj = lazy_ref.head(n_rows)
                    if isinstance(head_obj, pd.DataFrame):
                        return head_obj
                    if hasattr(head_obj, "to_pandas"):
                        return head_obj.to_pandas()
                except Exception:
                    return None
            return None

        def _to_numeric_array(df: pd.DataFrame) -> np.ndarray:
            if df.empty:
                return np.zeros((0, 1), dtype=np.float64)
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                return np.zeros((len(df), 1), dtype=np.float64)
            return numeric_df.fillna(0.0).to_numpy(dtype=np.float64)

        def _drift_sync() -> Dict[str, Any]:
            from monitoring.drift_detector import DriftDetector
            from pipeline.drift_adapter import DriftAdapter
            from data_ingestion.loader import DataLoader

            loader = DataLoader()
            cache_dir = DATASET_CACHE_DIR  # was hardcoded Path("./data/dataset_cache")

            prod_frames: List[pd.DataFrame] = []
            for hash_id in _snapshot:
                lazy_ref = loader.load_cached(cache_dir / hash_id)
                if lazy_ref is None:
                    continue

                frame = _materialize_head(lazy_ref, n_rows=5000)
                if frame is not None and not frame.empty:
                    prod_frames.append(frame)

            if not prod_frames:
                raise RuntimeError("No cached data found for session.")

            prod_df = pd.concat(prod_frames, ignore_index=True)

            # Load reference distribution from registry if model_id provided
            ref_df: Optional[pd.DataFrame] = None
            stored_drift: Dict[str, Any] = {}
            if model_id:
                ref_npy = MODEL_REGISTRY_DIR / model_id / "artifacts" / "reference_sample.npy"
                if ref_npy.exists():
                    ref_arr = np.load(str(ref_npy))
                    ref_df = pd.DataFrame(ref_arr)

                meta_file = MODEL_REGISTRY_DIR / model_id / "metadata.json"
                if meta_file.exists():
                    with open(meta_file, encoding="utf-8") as fh:
                        stored_meta = json.load(fh)
                    stored_drift = (
                        stored_meta.get("phases_summary", {})
                        .get("DRIFT_DETECTION", {})
                    )

            # Fallback: 70/30 split of production data as reference/production
            if ref_df is None:
                n = len(prod_df)
                if n < 2:
                    raise RuntimeError("Not enough cached rows for drift detection.")
                split = max(1, min(n - 1, int(n * 0.7)))
                ref_df = prod_df.iloc[:split].copy()
                prod_df = prod_df.iloc[split:].copy()

            retraining_orchestrator = None
            retrain_setup_error: Optional[str] = None
            try:
                retraining_orchestrator = _build_monitor_retraining_orchestrator(_sid, _ctx_monitor, _snapshot)
            except Exception as retrain_setup_exc:
                retrain_setup_error = str(retrain_setup_exc)
                logger.warning(
                    "/monitor/drift: retraining orchestrator setup failed for %s: %s",
                    _sid,
                    retrain_setup_exc,
                )

            detector = DriftDetector(retraining_orchestrator=retraining_orchestrator)
            report = detector.detect(
                _to_numeric_array(ref_df),
                _to_numeric_array(prod_df),
                dataset_id=str(_sid or "default"),
            )

            result = {
                "drift_detected": report.drift_detected,
                "metrics": {
                    "psi": report.psi,
                    "ks_statistic": report.ks_statistic,
                    "fdd": report.fdd,
                },
                "thresholds": {
                    "psi": 0.25,
                    "ks_statistic": 0.30,
                    "fdd": 0.50,
                },
                "status": report.status,
                "per_feature_ks": dict(report.per_feature_ks or {}),
                "per_feature_psi": dict(report.per_feature_psi or {}),
                "n_reference": report.n_reference,
                "n_production": report.n_production,
                "n_features": report.n_features,
                "composite_score": report.composite_score,
                "retrain_triggered": report.retrain_triggered,
                "retrain_info": dict(report.retrain_info or {}),
                "model_id": model_id,
                "stored_phase6_summary": stored_drift,
            }

            if retrain_setup_error:
                retrain_info = dict(result.get("retrain_info", {}) or {})
                if not retrain_info:
                    retrain_info = {"triggered": False}
                retrain_info.update(
                    {
                        "status": "error",
                        "error": retrain_setup_error,
                    }
                )
                result["retrain_info"] = retrain_info

            adapter = DriftAdapter()
            result["monitor"] = adapter.build_monitor_payload(result)
            return result

        data = await asyncio.to_thread(_drift_sync)

        _sync_monitor_drift_to_context(_sid, _ctx_monitor, data)

        _contract = _context_contract_payload(_ctx_monitor)
        return {
            "status": "success",
            "context_stage": _contract["context_stage"],
            "context_version": _contract["context_version"],
            "artifact_versions": _contract["artifact_versions"],
            "data": {
                "drift_detected": data["drift_detected"],
                "metrics": data["metrics"],
                "thresholds": data["thresholds"],
                "monitor": data.get("monitor", {}),
                "availability": {
                    "status": "available",
                    "model_id": data.get("model_id"),
                    "reference_artifact": "reference_sample.npy",
                },
                "status_per_metric": data["status"],
                "n_reference": data["n_reference"],
                "n_production": data["n_production"],
                "n_features": data["n_features"],
                "composite_score": data.get("composite_score", 0.0),
                "retrain_triggered": data.get("retrain_triggered", False),
                "retrain_info": data.get("retrain_info", {}),
                "model_id": data.get("model_id"),
                "stored_phase6_summary": data.get("stored_phase6_summary", {}),
            },
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("/monitor/drift error: %s", exc, exc_info=True)
        return JSONResponse(
            {"error": "Drift detection failed. Check server logs for details."},
            status_code=500,
        )


@app.post("/monitor")
@require_context("ingestion_complete", require_session=True)
async def monitor_overview(request: Request) -> Dict[str, Any]:
    """Composite monitoring endpoint that enriches Phase-6 output."""
    try:
        body = await request.json()
        if not isinstance(body, dict):
            body = {}

        _sid = body.get("session_id") if isinstance(body, dict) else None
        _ctx_monitor = _get_session_context_or_422(_sid, "/monitor")

        with _session_lock:
            _snapshot = _get_session_hashes(_sid)
            if not _snapshot:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "No active ingestion session. "
                        "Call POST /ingest/datasets first to register datasets."
                    ),
                )

        ensure_session_context(
            _ctx_monitor,
            session_id=_sid,
            dataset_snapshot=_snapshot,
        )

        drift_response = await monitor_drift(request)
        if isinstance(drift_response, JSONResponse):
            return drift_response

        drift_data: Dict[str, Any] = drift_response.get("data", {})

        from pipeline.drift_adapter import DriftAdapter

        monitor_payload = (
            drift_data.get("monitor")
            if isinstance(drift_data.get("monitor"), dict)
            else DriftAdapter().build_monitor_payload(drift_data)
        )

        model_stats: Optional[Dict[str, Any]] = None
        model_id = body.get("model_id")
        if model_id:
            model_id = _sanitize_model_id(str(model_id))
            from registry.model_registry import ThreadSafeModelRegistry

            try:
                model_stats = ThreadSafeModelRegistry().get_model_stats(model_id)
            except FileNotFoundError:
                model_stats = None

        _contract = _context_contract_payload(_ctx_monitor)
        return {
            "status": drift_response.get("status", "success"),
            "context_stage": _contract["context_stage"],
            "context_version": _contract["context_version"],
            "artifact_versions": _contract["artifact_versions"],
            "data": {
                "monitor": monitor_payload,
                "drift": drift_data,
                "model_stats": model_stats,
                "availability": drift_data.get("availability", {}),
            },
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("/monitor error: %s", exc, exc_info=True)
        return JSONResponse(
            {"error": "Monitor query failed. Check server logs for details."},
            status_code=500,
        )


@app.get("/models/{model_id}/stats")
async def model_stats(model_id: str, refresh: bool = False) -> Dict[str, Any]:
    """Return compact training/drift/research stats for one registered model."""
    try:
        model_id = _sanitize_model_id(model_id)
        from registry.model_registry import ThreadSafeModelRegistry

        stats = ThreadSafeModelRegistry().get_model_stats(model_id, refresh=refresh)
        return {"status": "success", "data": stats}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("/models/{model_id}/stats error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/experiments/run-ablations")
@require_context("preprocessing_planning", required_fields=["global_schema"], require_session=True)
async def run_ablations(request: Request) -> Dict[str, Any]:
    """Run structured ablation experiments asynchronously."""
    try:
        from core.types import TrainingConfig
        from pipeline.experiment_engine import ExperimentManager, PREDEFINED_ABLATIONS

        body = await request.json()
        if not isinstance(body, dict):
            body = {}

        session_id = body.get("session_id") if isinstance(body, dict) else None
        ctx = _get_session_context_or_422(session_id, "/experiments/run-ablations")
        _require_context_artifact(session_id, "global_schema", "/experiments/run-ablations")

        with _session_lock:
            snapshot = _get_session_hashes(session_id)
            if not snapshot:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "No active ingestion session. "
                        "Call POST /ingest/datasets first to register datasets."
                    ),
                )

        ensure_session_context(
            ctx,
            session_id=session_id,
            dataset_snapshot=snapshot,
        )

        requested_sources = body.get("sources") or []
        allowed_sources = {
            str(meta.get("source_url"))
            for meta in snapshot.values()
            if isinstance(meta, dict) and meta.get("source_url")
        }
        allowed_sources.update(str(dataset_id) for dataset_id in snapshot.keys())

        if requested_sources:
            invalid_sources = [
                str(src)
                for src in requested_sources
                if str(src) not in allowed_sources
            ]
            if invalid_sources:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Ablation sources must be session-scoped and present in the "
                        f"active session snapshot. Invalid sources: {invalid_sources}"
                    ),
                )
            sources = [str(src) for src in requested_sources]
        else:
            sources = [
                str(meta.get("source_url") or dataset_id)
                for dataset_id, meta in snapshot.items()
                if isinstance(meta, dict)
            ]

        if not sources:
            raise HTTPException(
                status_code=400,
                detail="No dataset sources available. Provide 'sources' or an active session_id.",
            )

        problem_type = str(body.get("problem_type", "classification_binary"))
        modalities = body.get("modalities") or ["tabular"]

        requested_conditions = body.get("conditions") or []
        available_map = {condition.name: condition for condition in PREDEFINED_ABLATIONS}
        if requested_conditions:
            selected_conditions = [
                available_map[name]
                for name in requested_conditions
                if isinstance(name, str) and name in available_map
            ]
            if not selected_conditions:
                raise HTTPException(
                    status_code=400,
                    detail="No valid ablation condition names were provided.",
                )
        else:
            selected_conditions = list(PREDEFINED_ABLATIONS)

        base_cfg = TrainingConfig(
            dataset_sources=[str(src) for src in sources],
            problem_type=problem_type,
            modalities=[str(mod) for mod in modalities],
        )
        manager = ExperimentManager(base_cfg, execution_context=ctx)

        task_id = uuid.uuid4().hex[:8]
        task_db.insert_task(
            task_id=task_id,
            task_type="ablation",
            status="PENDING",
            payload={
                "session_id": session_id,
                "context_stage": getattr(ctx, "pipeline_stage", None),
                "conditions": [condition.name for condition in selected_conditions],
                "n_conditions": len(selected_conditions),
            },
        )

        async def _run() -> None:
            task_db.update_status(task_id, "PROCESSING")
            try:
                await asyncio.to_thread(manager.run_ablations, selected_conditions)
                rows = manager.to_rows()
                task_db.update_result(
                    task_id,
                    "COMPLETED",
                    {
                        "results": rows,
                        "count": len(rows),
                    },
                )
            except Exception as exc:
                logger.error("/experiments/run-ablations task %s failed: %s", task_id, exc, exc_info=True)
                task_db.update_error(task_id, "FAILED", str(exc))

        asyncio.create_task(_run())

        _contract = _context_contract_payload(ctx)
        return {
            "status": "started",
            "task_id": task_id,
            "n_conditions": len(selected_conditions),
            "context_stage": _contract["context_stage"],
            "context_version": _contract["context_version"],
            "artifact_versions": _contract["artifact_versions"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("/experiments/run-ablations error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/experiments/ablation-results")
async def get_ablation_results() -> Dict[str, Any]:
    """Return latest persisted ablation results."""
    try:
        from pipeline.experiment_engine import EXPERIMENT_STORE

        if not EXPERIMENT_STORE.exists():
            return {
                "results": [],
                "count": 0,
                "message": "No ablation results yet. Run /experiments/run-ablations first.",
            }

        with open(EXPERIMENT_STORE, "r", encoding="utf-8") as fh:
            rows = json.load(fh)

        return {
            "results": rows,
            "count": len(rows) if isinstance(rows, list) else 0,
        }
    except Exception as exc:
        logger.error("/experiments/ablation-results error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/retrain-history")
async def retrain_history(
    limit: int = Query(default=100, ge=1, le=500),
    dataset_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Return persisted retraining events from drift-triggered retrain cycles."""
    try:
        from pipeline.retraining_pipeline import AdaptiveRetrainingPipeline

        history = AdaptiveRetrainingPipeline().get_history(
            limit=limit,
            dataset_id=dataset_id,
            session_id=session_id,
        )
        return {
            "status": "success",
            "count": len(history),
            "history": history,
        }
    except Exception as exc:
        logger.error("/retrain-history error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@app.get("/model-registry")
async def model_registry() -> Dict[str, Any]:
    """
    List all models currently stored in ``models/registry/``.

    Scans each sub-directory for a ``metadata.json`` file and returns its
    contents as a structured list.  Also optionally includes the MLflow
    run-level ``val_loss`` if the SQLite database is present.

    FRONTEND CONTRACT
    -----------------
    Returns::

        {
          "status": "success",
          "count": <int>,
          "models": [
            {
              "model_id":         "<str>",
              "created_at":       "<ISO-8601>",
              "status":           "active",
              "deployment_ready": bool,
              "artifact_paths":   { "model_weights": "...", ... },
              "phases_summary":   { ... }
            },
            ...
          ]
        }
    """
    try:
        import json as _json

        registry_root = MODEL_REGISTRY_DIR
        models: List[Dict[str, Any]] = []

        if registry_root.exists():
            for model_dir in sorted(registry_root.iterdir()):
                if not model_dir.is_dir():
                    continue
                meta_file = model_dir / "metadata.json"
                if not meta_file.exists():
                    continue
                try:
                    with open(meta_file, encoding="utf-8") as fh:
                        meta: Dict[str, Any] = _json.load(fh)
                    alias_name = str(
                        meta.get("display_name_alias")
                        or meta.get("display_name")
                        or meta.get("model_id")
                        or model_dir.name
                    )
                    meta["display_name_alias"] = alias_name
                    meta["display_name"] = alias_name
                    meta["rename_mode"] = "alias_only"
                    meta.setdefault("artifact_versions", {})
                    meta.setdefault("training_signals", {})
                    meta.setdefault("training_fit_analysis", {})
                    meta.setdefault("xai_config", {})
                    meta.setdefault("xai", {})
                    # Add computed artifact existence flags
                    artifact_paths: Dict[str, str] = meta.get("artifact_paths", {})
                    artifact_status: Dict[str, bool] = {
                        name: Path(path).exists()
                        for name, path in artifact_paths.items()
                    }
                    meta["artifact_exists"] = artifact_status
                    models.append(meta)
                except Exception as parse_exc:
                    logger.warning(
                        "/model-registry: could not parse %s: %s", meta_file, parse_exc
                    )

        # Optionally append last-known val_loss from MLflow SQLite
        mlflow_db = Path("mlruns") / "mlflow.db"
        mlflow_val_losses: Dict[str, float] = {}
        if mlflow_db.exists():
            try:
                import sqlite3
                conn = sqlite3.connect(str(mlflow_db))
                rows = conn.execute(
                    "SELECT run_uuid, value FROM metrics WHERE key='val_loss' "
                    "ORDER BY timestamp DESC"
                ).fetchall()
                conn.close()
                for run_uuid, val in rows:
                    if run_uuid not in mlflow_val_losses:
                        mlflow_val_losses[run_uuid] = val
            except Exception as mlflow_exc:
                logger.debug("/model-registry: MLflow DB query failed: %s", mlflow_exc)

        # Attach MLflow metrics where run_id matches best_params
        for model in models:
            phases = model.get("phases_summary", {})
            training = phases.get("TRAINING", {})
            best_params = training.get("best_params", {})
            run_id_hint = best_params.get("mlflow_run_id")
            if run_id_hint and run_id_hint in mlflow_val_losses:
                model["mlflow_best_val_loss"] = mlflow_val_losses[run_id_hint]

        return {
            "status": "success",
            "count":  len(models),
            "models": models,
        }

    except Exception as exc:
        logger.error("/model-registry error: %s", exc, exc_info=True)
        return JSONResponse(
            {"error": "Model registry query failed. Check server logs for details."},
            status_code=500,
        )


@app.api_route("/model-registry/{model_id}/rename", methods=["PATCH", "POST"])
async def model_registry_alias_rename(
    model_id: str,
    payload: ModelRegistryAliasRenameRequest,
) -> Dict[str, Any]:
    """
    Alias-only rename for model registry entries.
    Accepts both PATCH (REST-correct) and POST (frontend compatibility).

    This does not rename directories or model_id. It only persists a
    display alias in metadata.json.
    """
    model_id = _sanitize_model_id(model_id)
    alias_name = _sanitize_model_alias(payload.new_name)

    registry_root = MODEL_REGISTRY_DIR / model_id
    metadata_path = registry_root / "metadata.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")

    try:
        with _model_registry_lock:
            with open(metadata_path, "r", encoding="utf-8") as fh:
                metadata = json.load(fh)

            previous_alias = str(
                metadata.get("display_name_alias")
                or metadata.get("display_name")
                or metadata.get("model_id")
                or model_id
            )
            metadata["display_name_alias"] = alias_name
            metadata["display_name"] = alias_name
            metadata["rename_mode"] = "alias_only"
            metadata["alias_updated_at"] = datetime.now(timezone.utc).isoformat()

            with open(metadata_path, "w", encoding="utf-8") as fh:
                json.dump(metadata, fh, indent=2, default=str)

        return {
            "status": "success",
            "model_id": model_id,
            "display_name_alias": alias_name,
            "previous_alias": previous_alias,
            "rename_mode": "alias_only",
            "note": "Model ID and artifact directory are unchanged.",
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("/model-registry/%s/rename error: %s", model_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/model-registry/{model_id}/download")
async def model_registry_download(model_id: str) -> StreamingResponse:
    """
    Download a standard model bundle containing model weights and metadata.

    Bundle format:
    - metadata.json
    - bundle_manifest.json
    - README.txt
    - model_weights.pth (if available)
    - artifacts/* (all available artifact files)
    """
    model_id = _sanitize_model_id(model_id)
    registry_root = MODEL_REGISTRY_DIR / model_id
    metadata_path = registry_root / "metadata.json"

    if not registry_root.exists() or not metadata_path.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")

    try:
        with _model_registry_lock:
            with open(metadata_path, "r", encoding="utf-8") as fh:
                metadata: Dict[str, Any] = json.load(fh)

        alias_name = str(
            metadata.get("display_name_alias")
            or metadata.get("display_name")
            or metadata.get("model_id")
            or model_id
        )
        artifact_paths = dict(metadata.get("artifact_paths", {}) or {})

        root_resolved = registry_root.resolve()

        def _resolve_registry_path(raw_path: str) -> Optional[Path]:
            raw_clean = str(raw_path or "").strip()
            if not raw_clean:
                return None
            candidate = Path(raw_clean)
            if not candidate.is_absolute():
                candidate = (project_root / candidate)
            try:
                resolved = candidate.resolve()
                resolved.relative_to(root_resolved)
            except Exception:
                return None
            return resolved

        bundle_manifest: Dict[str, Any] = {
            "model_id": model_id,
            "display_name_alias": alias_name,
            "rename_mode": "alias_only",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "included_files": [],
        }

        zip_buffer = io.BytesIO()
        added_paths: set[str] = set()

        def _add_file(zf: zipfile.ZipFile, source: Path, arcname: str) -> None:
            arc = str(arcname).replace("\\", "/")
            if arc in added_paths:
                return
            zf.write(source, arc)
            added_paths.add(arc)
            bundle_manifest["included_files"].append(arc)

        with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("metadata.json", json.dumps(metadata, indent=2, default=str))
            bundle_manifest["included_files"].append("metadata.json")

            # G24/G28: rich README with usage instructions, input formats, calibration, glossary
            _trained_modalities = metadata.get("modalities") or metadata.get("active_modalities") or []
            _calibration_method = metadata.get("calibration_method") or metadata.get("calibration", {}).get("method") or "none"
            _encoder_config = metadata.get("encoder_config") or {}
            _input_shapes = _encoder_config.get("input_shapes") or {}
            _head_arch = metadata.get("head_architecture") or {}

            _shapes_text = "\n".join(
                f"  {mod}: {shape}" for mod, shape in _input_shapes.items()
            ) if _input_shapes else "  (see metadata.json for encoder_config)"

            _glossary = (
                "### Glossary\n"
                "- AutoML: Automated Machine Learning — the system selects and tunes models automatically\n"
                "- Fusion: How tabular/text/image representations are combined into one vector\n"
                "- Calibration: Post-training step that aligns model confidence with true accuracy\n"
                "- XAI (Explainable AI): Methods that explain which features drove each prediction\n"
                "- Optuna: Hyperparameter optimisation framework used to tune model parameters\n"
                "- FTTransformer: Feature Tokeniser Transformer — state-of-the-art tabular encoder (NeurIPS 2021)\n"
                "- Focal Loss: Loss function that down-weights easy examples, used for class-imbalanced data\n"
                "- SWA: Stochastic Weight Averaging — averages weights over last 10%% of training for better generalisation\n"
                "- PCGrad: Gradient surgery that prevents destructive interference between modality gradients\n"
            )

            readme = (
                "# APEX Model Bundle\n\n"
                f"model_id: {model_id}\n"
                f"display_name: {alias_name}\n"
                f"generated_at: {datetime.now(timezone.utc).isoformat()}\n\n"
                "---\n\n"
                "## 1. Quick Start\n\n"
                "```python\n"
                "import torch\n"
                "# Load model state dict\n"
                "state = torch.load('model_weights.pth', map_location='cpu')\n"
                "# Expected keys: head.*, encoders.*, fusion.*\n"
                "# Reconstruct via APEX TrainingOrchestrator.load_from_bundle() or\n"
                "# pass state to your own _MultimodalHead instance.\n"
                "```\n\n"
                "## 2. Trained Modalities\n\n"
                f"{', '.join(_trained_modalities) if _trained_modalities else 'see metadata.json'}\n\n"
                "## 3. Input Formats\n\n"
                f"{_shapes_text}\n\n"
                "Each batch element should be a dict:\n"
                "  `{'tabular': tensor(N, D_tab), 'text': {'input_ids': tensor(N, L), ...}, 'image': tensor(N, 3, H, W)}`\n"
                "Absent modalities are zero-padded automatically by the _MultimodalHead.\n\n"
                "## 4. Calibration\n\n"
                f"Method: {_calibration_method}\n"
                "Load calibration parameters from metadata.json → 'calibration' key.\n\n"
                "## 5. Head Architecture\n\n"
                f"Type: {_head_arch.get('type', 'mlp')}  "
                f"Hidden dim: {_head_arch.get('hidden_dim', '?')}  "
                f"Outputs: {_head_arch.get('num_outputs', '?')}\n\n"
                + _glossary
            )
            zf.writestr("README.txt", readme)
            bundle_manifest["included_files"].append("README.txt")

            weight_path = None
            raw_weight = artifact_paths.get("model_weights")
            if isinstance(raw_weight, str) and raw_weight.strip():
                weight_path = _resolve_registry_path(raw_weight)
            if weight_path is None:
                fallback_weight = registry_root / "artifacts" / "model_weights.pth"
                if fallback_weight.exists():
                    weight_path = fallback_weight.resolve()
            if weight_path is not None and weight_path.exists() and weight_path.is_file():
                _add_file(zf, weight_path, "model_weights.pth")

            for artifact_name, raw_path in artifact_paths.items():
                if not isinstance(raw_path, str) or not raw_path.strip():
                    continue
                resolved = _resolve_registry_path(raw_path)
                if resolved is None or not resolved.exists():
                    continue

                if resolved.is_file():
                    _add_file(zf, resolved, f"artifacts/{resolved.name}")
                    continue

                if resolved.is_dir():
                    for nested in resolved.rglob("*"):
                        if not nested.is_file():
                            continue
                        rel = nested.relative_to(resolved).as_posix()
                        _add_file(
                            zf,
                            nested,
                            f"artifacts/{resolved.name}/{rel}",
                        )

            # inference_example.py — self-contained usage script bundled with every model
            _modalities_list = metadata.get("config", {}).get("modalities", ["tabular"])
            _prob_type = metadata.get("config", {}).get("problem_type", "classification_binary")
            _example_input_lines = []
            if "tabular" in _modalities_list:
                _example_input_lines.append('    # Tabular: replace with your actual column names and values\n    "age": 35, "fare": 7.25, "pclass": 3,')
            if "text" in _modalities_list:
                _example_input_lines.append('    # Text: raw string for the text column\n    "text": "Sample input text here.",')
            if "image" in _modalities_list:
                _example_input_lines.append('    # Image: absolute or relative path to image file\n    "image_path": "/path/to/image.jpg",')
            _example_input = "\n".join(_example_input_lines) or '    "feature_0": 0.5,'
            _inference_script = (
                '"""AutoVision model inference — generated for model: ' + model_id + '\n\n'
                'Requirements: pip install torch transformers Pillow joblib scikit-learn\n'
                'The AutoVision package must be installed or on your PYTHONPATH.\n"""\n\n'
                'import sys\n'
                'from pathlib import Path\n\n'
                '# Add AutoVision to path if not installed as a package\n'
                '# sys.path.insert(0, "/path/to/autovision")\n\n'
                'from pipeline.inference_engine import MultimodalInferenceEngine\n\n'
                '# Load the model (point MODEL_ID to the directory under models/registry/)\n'
                'MODEL_ID = "' + model_id + '"\n\n'
                'engine = MultimodalInferenceEngine(model_id=MODEL_ID)\n\n'
                'sample = {\n' + _example_input + '\n}\n\n'
                'result = engine.predict_batch([sample])\n\n'
                'print("Prediction :", result["predictions"])\n'
                'print("Confidence :", result["confidences"])\n'
                'print("Problem type:", result["problem_type"])\n\n'
                '# Batch inference from CSV\n'
                '# import pandas as pd\n'
                '# df = pd.read_csv("my_data.csv")\n'
                '# result = engine.predict_batch(df)\n\n'
                '# XAI (IntegratedGradients) — requires: pip install captum\n'
                '# explanations = engine.generate_explanations([sample], target_class=0)\n'
                '# print(explanations["tabular"])  # feature attributions\n'
                '# print(explanations["text"])     # token attributions\n'
            )
            zf.writestr("inference_example.py", _inference_script)
            bundle_manifest["included_files"].append("inference_example.py")

            bundle_manifest["included_files"].append("bundle_manifest.json")
            zf.writestr(
                "bundle_manifest.json",
                json.dumps(bundle_manifest, indent=2, default=str),
            )

        zip_buffer.seek(0)
        filename = f"{model_id}_weights_bundle.zip"
        headers = {
            "Content-Disposition": f'attachment; filename="{filename}"',
        }
        return StreamingResponse(zip_buffer, media_type="application/zip", headers=headers)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("/model-registry/%s/download error: %s", model_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/model-registry/{model_id}/export-onnx")
async def model_registry_export_onnx(model_id: str, request: Request) -> Dict[str, Any]:
    """
    Export a trained model's fusion head to ONNX format.

    Exports the post-encoding portion of the pipeline (tabular/text/image pooled
    vectors → logits).  The returned artifact can be run with onnxruntime without
    PyTorch installed.

    Returns
    -------
    JSON with ``onnx_path``, ``input_names``, ``output_names``, ``opset``.
    """
    model_id = _sanitize_model_id(model_id)
    try:
        import io
        import torch as _torch
        from pipeline.inference_engine import MultimodalInferenceEngine
        from config.paths import MODEL_REGISTRY_DIR as _MRD
        from pathlib import Path as _Path

        engine = MultimodalInferenceEngine(model_id=model_id)
        head = engine._head
        head.eval()

        input_dims = engine.input_dims
        if not input_dims:
            raise HTTPException(status_code=400, detail="No input_dims found for model.")

        # Part C.4 — ULA token-mode: produce 3D dummy inputs (N, seq_len, dim)
        # for keys that belong to token-sequence modalities.
        _ula_config_path = _Path(_MRD) / model_id / "artifacts" / "ula_config.json"
        _ula_token_mode = False
        if _ula_config_path.exists():
            try:
                import json as _json
                _ula_cfg = _json.loads(_ula_config_path.read_text())
                _ula_token_mode = bool(_ula_cfg.get("token_mode", False))
            except Exception:
                pass

        dummy_inputs: Dict[str, _torch.Tensor] = {}
        for k, v in input_dims.items():
            if _ula_token_mode:
                if "text" in k:
                    dummy_inputs[k] = _torch.zeros(1, 128, v, dtype=_torch.float32)
                elif "image" in k:
                    dummy_inputs[k] = _torch.zeros(1, 196, v, dtype=_torch.float32)
                elif "tabular" in k:
                    dummy_inputs[k] = _torch.zeros(1, v, 256, dtype=_torch.float32)
                else:
                    dummy_inputs[k] = _torch.zeros(1, v, dtype=_torch.float32)
            else:
                dummy_inputs[k] = _torch.zeros(1, v, dtype=_torch.float32)

        export_dir = _Path(_MRD) / model_id / "onnx"
        export_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = export_dir / "fusion_head.onnx"

        input_names = sorted(dummy_inputs.keys())

        class _HeadWrapper(_torch.nn.Module):
            def __init__(self, h): super().__init__(); self.h = h
            def forward(self, *args):
                return self.h({k: v for k, v in zip(input_names, args)})

        wrapper = _HeadWrapper(head)
        dummy_tuple = tuple(dummy_inputs[k] for k in input_names)
        dynamic_axes = {}
        for k in input_names:
            t = dummy_inputs[k]
            if t.dim() == 3:
                dynamic_axes[k] = {0: "batch", 1: "seq_len"}
            else:
                dynamic_axes[k] = {0: "batch"}
        dynamic_axes["logits"] = {0: "batch"}

        try:
            _torch.onnx.export(
                wrapper,
                dummy_tuple,
                str(onnx_path),
                input_names=input_names,
                output_names=["logits"],
                dynamic_axes=dynamic_axes,
                opset_version=17,
                do_constant_folding=True,
            )
            logger.info("ONNX export: %s → %s", model_id, onnx_path)
        except Exception as onnx_exc:
            raise HTTPException(
                status_code=500,
                detail=f"ONNX export failed: {onnx_exc}. "
                       "Ensure the fusion head uses only ONNX-compatible ops."
            )

        return {
            "onnx_path": str(onnx_path),
            "input_names": input_names,
            "output_names": ["logits"],
            "opset": 17,
            "input_dims": input_dims,
            "note": (
                "This ONNX graph covers tabular+text+image pooled vectors → logits. "
                "Run tokenization and image encoding separately before this graph. "
                "Load with: import onnxruntime; sess = onnxruntime.InferenceSession(path)"
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("/model-registry/%s/export-onnx error: %s", model_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/model-info/{model_id}")
async def model_info(model_id: str) -> Dict[str, Any]:
    """Return class labels and expected feature columns for a registered model."""
    import json as _json

    model_id = _sanitize_model_id(model_id)
    registry_root = MODEL_REGISTRY_DIR / model_id
    if not registry_root.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")

    def _load_model_info_sync() -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        metadata_path = registry_root / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, encoding="utf-8") as fh:
                    metadata = _json.load(fh)
            except Exception:
                metadata = {}

        # Load schema for feature columns
        schema: Dict[str, Any] = {}
        schema_path = registry_root / "artifacts" / "schema.json"
        if schema_path.exists():
            with open(schema_path, encoding="utf-8") as fh:
                schema = _json.load(fh)

        # Load target encoder for class labels
        class_labels: List[str] = []
        enc_path = registry_root / "artifacts" / "target_encoder.joblib"
        if enc_path.exists():
            try:
                import joblib
                enc = joblib.load(enc_path)
                if isinstance(enc, dict):
                    class_labels = enc.get("all_labels", [])
                elif hasattr(enc, "classes_"):
                    class_labels = list(enc.classes_)
            except Exception:
                pass

        # Load fitted preprocessor for effective features
        effective_features: List[str] = []
        dropped_columns: List[str] = []
        scaler_path = registry_root / "artifacts" / "tabular_scaler.joblib"
        if scaler_path.exists():
            try:
                import joblib
                prep = joblib.load(scaler_path)
                effective_features = list(getattr(prep, "_feature_names_in", []))
                dropped_columns = list(getattr(prep, "_dropped_cols", []))
            except Exception:
                pass

        # Extract expected feature columns from schema
        per_ds = schema.get("per_dataset", [{}])
        detected = per_ds[0].get("detected_columns", {}) if per_ds else {}
        tabular_cols = detected.get("tabular", [])
        text_cols = detected.get("text", [])
        image_cols = detected.get("image", [])
        target_col = schema.get("primary_target", "")

        # Use effective_features (post-preprocessing) when available,
        # fall back to raw schema columns minus target
        input_tabular = (
            effective_features
            if effective_features
            else [c for c in tabular_cols if c != target_col]
        )

        phases_summary: Dict[str, Any] = dict(metadata.get("phases_summary", {}) or {})
        training_summary: Dict[str, Any] = dict(phases_summary.get("TRAINING", {}) or {})
        evaluation_summary: Dict[str, Any] = dict(training_summary.get("evaluation", {}) or {})
        calibration_summary: Dict[str, Any] = dict(training_summary.get("calibration", {}) or {})

        # Part C.1 — Load ULA and LoRA configs when present in artifacts
        ula_config: Dict[str, Any] = {}
        ula_cfg_path = registry_root / "artifacts" / "ula_config.json"
        if ula_cfg_path.exists():
            try:
                with open(ula_cfg_path, encoding="utf-8") as _fh:
                    ula_config = _json.load(_fh)
            except Exception:
                pass

        lora_config_info: Dict[str, Any] = {}
        enc_cfg_path = registry_root / "artifacts" / "encoder_config.json"
        if enc_cfg_path.exists():
            try:
                with open(enc_cfg_path, encoding="utf-8") as _fh:
                    _enc_cfg = _json.load(_fh)
                _txt_enc = _enc_cfg.get("text_encoder", {}) or {}
                _img_enc = _enc_cfg.get("image_encoder", {}) or {}
                if _txt_enc.get("lora_r") or _img_enc.get("lora_r"):
                    lora_config_info = {
                        "r": _txt_enc.get("lora_r", _img_enc.get("lora_r")),
                        "alpha": _txt_enc.get("lora_alpha", 16.0),
                        "text_lora_saved": (registry_root / "artifacts" / "lora_text.pth").exists(),
                        "image_lora_saved": (registry_root / "artifacts" / "lora_image.pth").exists(),
                    }
            except Exception:
                pass

        prediction_contract = _build_prediction_contract(
            model_id=model_id,
            metadata={
                **metadata,
                "calibration": calibration_summary,
            },
            schema=schema,
            ctx=None,
            class_labels=class_labels,
        )

        return {
            "model_id": model_id,
            "problem_type": prediction_contract.get("problem_type") or schema.get("global_problem_type", ""),
            "modalities": prediction_contract.get("active_modalities") or schema.get("global_modalities", []),
            "active_modalities": prediction_contract.get("active_modalities", []),
            "detected_modalities": prediction_contract.get("detected_modalities", []),
            "excluded_modalities": prediction_contract.get("excluded_modalities", {}),
            "class_labels": class_labels,
            "target_column": target_col,
            "input_columns": {
                "tabular": prediction_contract.get("input_columns", {}).get("tabular", input_tabular),
                "text": prediction_contract.get("input_columns", {}).get("text", text_cols),
                "image": prediction_contract.get("input_columns", {}).get("image", image_cols),
            },
            "prediction_contract": prediction_contract,
            "accepted_image_request_keys": prediction_contract.get("accepted_image_request_keys", []),
            "effective_features": effective_features,
            "dropped_columns": dropped_columns,
            "artifact_versions": dict(metadata.get("artifact_versions", {}) or {}),
            "training_signals": dict(metadata.get("training_signals", {}) or {}),
            "training_fit_analysis": dict(metadata.get("training_fit_analysis", {}) or {}),
            "xai_config": dict(metadata.get("xai_config", {}) or {}),
            "training": training_summary,
            "evaluation": evaluation_summary,
            "calibration": calibration_summary,
            "research_metrics": dict(metadata.get("research_metrics", {}) or {}),
            "fusion": dict(metadata.get("fusion", {}) or {}),
            "xai": dict(metadata.get("xai", {}) or {}),
            "ula_config": ula_config,
            "lora_config": lora_config_info,
        }

    return await asyncio.to_thread(_load_model_info_sync)


# ---------------------------------------------------------------------------
# Part C.3 — Fusion diagnostics endpoint
# ---------------------------------------------------------------------------

@app.get("/intelligence/fusion-diagnostics/{model_id}")
async def fusion_diagnostics(model_id: str) -> Dict[str, Any]:
    """
    Return ULA fusion diagnostics for a registered model.

    When the model uses UnifiedLatentFusion, reports:
    - fusion strategy, latent_dim, token_mode
    - last_token_count (number of cross-modal tokens in final forward pass)
    - LoRA artifact presence
    - attention_rollout_compatible flag
    """
    import json as _json

    model_id = _sanitize_model_id(model_id)
    registry_root = MODEL_REGISTRY_DIR / model_id
    if not registry_root.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")

    artifacts_dir = registry_root / "artifacts"
    result: Dict[str, Any] = {"model_id": model_id, "status": "ok"}

    # Load ULA config if present
    ula_path = artifacts_dir / "ula_config.json"
    if ula_path.exists():
        try:
            with open(ula_path, encoding="utf-8") as fh:
                result["ula_config"] = _json.load(fh)
            result["fusion_strategy"] = "ula"
        except Exception as exc:
            result["ula_config"] = {}
            result["ula_config_error"] = str(exc)
    else:
        # Read fusion strategy from metadata
        meta_path = registry_root / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, encoding="utf-8") as fh:
                    _meta = _json.load(fh)
                _training = _meta.get("phases_summary", {}).get("TRAINING", {})
                result["fusion_strategy"] = _training.get("fusion_strategy", "unknown")
            except Exception:
                result["fusion_strategy"] = "unknown"
        else:
            result["fusion_strategy"] = "unknown"

    result["lora_text_saved"] = (artifacts_dir / "lora_text.pth").exists()
    result["lora_image_saved"] = (artifacts_dir / "lora_image.pth").exists()
    result["attention_rollout_compatible"] = result.get("ula_config", {}).get(
        "attention_rollout_compatible", False
    )
    return result


# ---------------------------------------------------------------------------
# Pass A — Compute budget endpoint (publication transparency)
# ---------------------------------------------------------------------------

@app.get("/intelligence/compute-budget/{model_id}")
async def compute_budget(model_id: str) -> Dict[str, Any]:
    """
    Return per-trial compute budget records (FLOPs / VRAM / GPU-hours / params).

    Reads `diary/results/*_compute.json` files written by `pipeline.compute_tracker.ComputeTracker.save()`.
    Filters records by model_id when present in the run_id, otherwise returns all.
    """
    model_id = _sanitize_model_id(model_id)
    try:
        from pipeline.compute_tracker import ComputeTracker
        records = ComputeTracker.load_all()
    except Exception as exc:
        logger.warning("/intelligence/compute-budget: ComputeTracker.load_all failed: %s", exc)
        records = []

    matching = [r for r in records if model_id in str(r.get("run_id", ""))]
    if not matching:
        matching = records  # fallback: return all if no run_id filter matches

    if not matching:
        return {
            "model_id": model_id,
            "n_trials": 0,
            "total_gpu_hours": 0.0,
            "peak_vram_mb": 0.0,
            "records": [],
            "note": "No compute budget records found. Trials must call ComputeTracker.save() to populate diary/results/*_compute.json.",
        }

    total_gpu_hours = sum(r.get("gpu_hours", 0.0) or 0.0 for r in matching)
    peak_vram = max((r.get("peak_vram_mb", 0.0) or 0.0 for r in matching), default=0.0)
    total_flops = sum((r.get("flops", 0.0) or 0.0) for r in matching)

    return {
        "model_id": model_id,
        "n_trials": len(matching),
        "total_gpu_hours": round(total_gpu_hours, 4),
        "peak_vram_mb": round(peak_vram, 2),
        "total_flops": total_flops,
        "records": matching,
    }


# ---------------------------------------------------------------------------
# Pass A — Aggregated research results endpoint (publication transparency)
# ---------------------------------------------------------------------------

@app.get("/research/aggregated-results")
async def aggregated_results() -> Dict[str, Any]:
    """
    Return aggregated multi-seed results with Wilcoxon p-values and bootstrap CIs.

    Reads `diary/results/aggregated_results.json` produced by `scripts/aggregate_results.py`.
    """
    import json as _json
    from pathlib import Path as _Path

    _path = _Path("diary/results/aggregated_results.json")
    if not _path.exists():
        return {
            "status": "missing",
            "note": "Run `python scripts/aggregate_results.py` to generate diary/results/aggregated_results.json",
            "expected_path": str(_path),
        }
    try:
        with open(_path, encoding="utf-8") as fh:
            data = _json.load(fh)
        data["status"] = "ok"
        return data
    except Exception as exc:
        logger.warning("/research/aggregated-results read failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to read aggregated results: {exc}")


# ---------------------------------------------------------------------------
# Async inference  (fire-and-poll pattern – eliminates 504 timeouts)
# ---------------------------------------------------------------------------

def _run_inference_task(
    task_id: str,
    model_id: str,
    raw_inputs: List[Dict[str, Any]],
    explain: bool,
    target_class: int,
    n_steps: int,
    latency_budget_s: float = 30.0,
    session_id: Optional[str] = None,
) -> None:
    """
    Background worker that wraps the existing synchronous inference path.

    Runs in a thread spawned by FastAPI BackgroundTasks.  Updates
    the SQLite task store with PROCESSING -> COMPLETED | FAILED so the
    frontend can poll ``GET /task/{task_id}`` without blocking.
    """
    task_db.update_status(task_id, "PROCESSING")

    try:
        from pipeline.inference_engine import MultimodalInferenceEngine

        _ctx = session_manager.get_session(str(session_id)) if session_id else None
        _contract = _context_contract_payload(_ctx)
        _metadata = _require_prediction_ready_model(_ctx, model_id, "/predict-async")
        _schema = _load_model_schema(model_id) or dict(getattr(_ctx, "global_schema", {}) or {})
        _prediction_contract = _build_prediction_contract(
            model_id=model_id,
            metadata=_metadata,
            schema=_schema,
            ctx=_ctx,
            class_labels=_load_model_class_labels(model_id),
        )
        raw_inputs, _io_audit = _normalise_prediction_inputs(raw_inputs, _prediction_contract)

        # Re-use cached engine (same LRU logic as /predict)
        with _engine_cache_lock:
            if model_id in _engine_cache:
                _engine_cache.move_to_end(model_id)
                engine = _engine_cache[model_id]
            else:
                engine = None

        if engine is None:
            engine = MultimodalInferenceEngine(model_id=model_id)
            with _engine_cache_lock:
                _engine_cache[model_id] = engine
                while len(_engine_cache) > _MAX_ENGINES:
                    _engine_cache.popitem(last=False)

        df: pd.DataFrame = pd.DataFrame(raw_inputs)

        # Core inference – identical to the synchronous /predict path
        _budget_s = float(max(1.0, latency_budget_s))
        t0 = time.perf_counter()
        result: Dict[str, Any] = LatencyGuard.timed(
            engine.predict_batch,
            _budget_s,
            df,
            execution_context=_ctx,
        )

        # XAI (optional)
        explanations: Optional[Dict[str, Any]] = None
        if explain:
            effective_target = _resolve_xai_target(target_class, result)

            elapsed = time.perf_counter() - t0
            remaining_budget = max(1.0, _budget_s - elapsed)
            try:
                explanations = LatencyGuard.timed(
                    engine.generate_explanations,
                    remaining_budget,
                    df,
                    target_class=effective_target,
                    n_steps=n_steps,
                )
            except Exception as xai_exc:
                explanations = {
                    "status": "unavailable",
                    "unavailable_reasons": {"global": str(xai_exc)},
                    "availability": _prediction_contract.get("xai_availability", {}),
                }

        # Bug 13: include calibration metadata so frontend can confirm calibration was active
        _cal = getattr(engine, "probability_calibrator", None)
        payload = _prediction_output_payload(
            model_id=model_id,
            result=result,
            explanations=explanations,
            prediction_contract=_prediction_contract,
            io_audit=_io_audit,
            ctx=_ctx,
        )
        payload["calibration_applied"] = _cal is not None or payload.get("calibration_applied", False)
        payload["calibration_method"] = getattr(_cal, "method", None) if _cal else None
        payload["context_stage"] = _contract["context_stage"]
        payload["context_version"] = _contract["context_version"]
        payload["artifact_versions"] = _contract["artifact_versions"]

        task_db.update_result(task_id, "COMPLETED", payload)

    except Exception as exc:
        logger.error("Background inference task %s failed: %s", task_id, exc, exc_info=True)
        task_db.update_error(task_id, "FAILED", str(exc))


@app.post("/predict-async")
@require_context("training", required_fields=["global_schema"], require_session=True)
async def predict_async(request: Request, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Fire-and-return async inference.

    Immediately returns ``{"task_id": "<uuid>"}``; the frontend polls
    ``GET /task/{task_id}`` until status is COMPLETED or FAILED.
    """

    body: Dict[str, Any] = await request.json()
    _sid = body.get("session_id") if isinstance(body, dict) else None
    _ctx = _get_session_context_or_422(_sid, "/predict-async")
    _require_context_artifact(_sid, "global_schema", "/predict-async")
    _contract = _context_contract_payload(_ctx)

    model_id: Optional[str] = body.get("model_id")
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required.")
    model_id = _sanitize_model_id(model_id)

    raw_inputs: List[Dict[str, Any]] = body.get("inputs", [])
    if not raw_inputs:
        raise HTTPException(status_code=400, detail="inputs list is empty.")
    _MAX_BATCH: int = 10_000
    if len(raw_inputs) > _MAX_BATCH:
        raise HTTPException(
            status_code=400,
            detail=f"Batch too large ({len(raw_inputs)} rows). Maximum is {_MAX_BATCH}.",
        )

    _metadata = _require_prediction_ready_model(_ctx, model_id, "/predict-async")
    _schema = _load_model_schema(model_id) or dict(getattr(_ctx, "global_schema", {}) or {})
    _prediction_contract = _build_prediction_contract(
        model_id=model_id,
        metadata=_metadata,
        schema=_schema,
        ctx=_ctx,
        class_labels=_load_model_class_labels(model_id),
    )
    raw_inputs, _io_audit = _normalise_prediction_inputs(raw_inputs, _prediction_contract)

    explain: bool     = bool(body.get("explain", False))
    target_class: int = int(body.get("target_class", -1))
    n_steps: int      = int(body.get("n_steps", 50))
    latency_budget_s: float = float(body.get("latency_budget_s", 30.0) or 30.0)

    task_id: str = str(uuid.uuid4())
    task_db.insert_task(
        task_id=task_id,
        task_type="inference",
        status="PENDING",
        payload={
            "model_id": model_id,
            "n_samples": len(raw_inputs),
            "latency_budget_s": latency_budget_s,
            "session_id": _sid,
            "context_stage": _contract["context_stage"],
            "input_contract": _prediction_contract,
            "consumed_inputs": _io_audit.get("consumed_inputs", []),
        },
    )

    background_tasks.add_task(
        _run_inference_task,
        task_id, model_id, raw_inputs, explain, target_class, n_steps, latency_budget_s, _sid,
    )

    return {
        "task_id": task_id,
        "status": "PENDING",
        "context_stage": _contract["context_stage"],
        "context_version": _contract["context_version"],
        "artifact_versions": _contract["artifact_versions"],
    }


@app.get("/task/{task_id}")
async def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    Poll the status of an async inference task.

    Returns the full prediction payload once COMPLETED, or an error
    message if FAILED.
    """
    task = task_db.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")

    response: Dict[str, Any] = {
        "task_id":    task_id,
        "status":     task["status"],
        "created_at": task.get("created_at"),
    }
    if task["status"] == "COMPLETED":
        response["result"] = task["result"]
    elif task["status"] == "FAILED":
        response["error"] = task["error"]
    return response


# ---------------------------------------------------------------------------
# WebSocket streaming inference
# ---------------------------------------------------------------------------

@app.websocket("/ws/predict")
async def ws_predict(websocket: WebSocket) -> None:
    """
    WebSocket inference endpoint with real-time status streaming.

    Protocol
    --------
    1. Client connects, server sends ``{"type": "status", "status": "CONNECTED"}``.
    2. Client sends a JSON message with ``model_id``, ``inputs``, etc.
    3. Server streams status updates as processing progresses.
    4. Server sends ``{"type": "complete", "result": {...}}`` with the
       full prediction payload.
    5. On error: ``{"type": "error", "error": "..."}``.
    6. Connection closes after the result is sent.
    """
    await websocket.accept()

    try:
        # 1. Acknowledge connection
        await websocket.send_json({"type": "status", "status": "CONNECTED"})

        # 2. Receive the inference request
        raw_message: Optional[str] = None
        while True:
            try:
                raw_message = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=float(os.getenv("APEX_WS_IDLE_TIMEOUT_SEC", "30.0")),
                )
                break
            except asyncio.TimeoutError:
                try:
                    await websocket.send_json({"type": "ping", "ts": datetime.now(timezone.utc).isoformat()})
                except Exception:
                    break
                continue

        if raw_message is None:
            return

        body: Dict[str, Any] = json.loads(raw_message)
        _sid = body.get("session_id") if isinstance(body, dict) else None
        if not _sid:
            await websocket.send_json({"type": "error", "error": "session_id is required."})
            return

        _ctx = session_manager.get_session(str(_sid))
        if _ctx is None:
            await websocket.send_json(
                {
                    "type": "error",
                    "error": f"No ExecutionContext for session {_sid}.",
                }
            )
            return
        if _is_missing_context_value(getattr(_ctx, "global_schema", None)):
            await websocket.send_json(
                {
                    "type": "error",
                    "error": "Session ExecutionContext is missing global_schema.",
                }
            )
            return
        _contract = _context_contract_payload(_ctx)

        model_id: Optional[str] = body.get("model_id")
        if not model_id:
            await websocket.send_json({"type": "error", "error": "model_id is required."})
            return
        try:
            model_id = _sanitize_model_id(model_id)
        except HTTPException as exc:
            await websocket.send_json({"type": "error", "error": exc.detail})
            return

        raw_inputs: List[Dict[str, Any]] = body.get("inputs", [])
        if not raw_inputs:
            await websocket.send_json({"type": "error", "error": "inputs list is empty."})
            return

        _MAX_BATCH: int = 10_000
        if len(raw_inputs) > _MAX_BATCH:
            await websocket.send_json({
                "type": "error",
                "error": f"Batch too large ({len(raw_inputs)} rows). Maximum is {_MAX_BATCH}.",
            })
            return

        try:
            _metadata = _require_prediction_ready_model(_ctx, model_id, "/ws/predict")
            _schema = _load_model_schema(model_id) or dict(getattr(_ctx, "global_schema", {}) or {})
            _prediction_contract = _build_prediction_contract(
                model_id=model_id,
                metadata=_metadata,
                schema=_schema,
                ctx=_ctx,
                class_labels=_load_model_class_labels(model_id),
            )
            raw_inputs, _io_audit = _normalise_prediction_inputs(raw_inputs, _prediction_contract)
        except HTTPException as exc:
            await websocket.send_json({"type": "error", "error": exc.detail})
            return

        explain: bool = bool(body.get("explain", False))
        target_class: int = int(body.get("target_class", -1))
        n_steps: int = int(body.get("n_steps", 50))

        # 3. Load or retrieve inference engine
        await websocket.send_json({"type": "status", "status": "LOADING_MODEL"})

        from pipeline.inference_engine import MultimodalInferenceEngine

        with _engine_cache_lock:
            if model_id in _engine_cache:
                _engine_cache.move_to_end(model_id)
                engine = _engine_cache[model_id]
            else:
                engine = None

        if engine is None:
            engine = await asyncio.to_thread(
                MultimodalInferenceEngine, model_id=model_id,
            )
            with _engine_cache_lock:
                _engine_cache[model_id] = engine
                while len(_engine_cache) > _MAX_ENGINES:
                    _engine_cache.popitem(last=False)

        # 4. Run inference — chunk large batches for progress streaming
        await websocket.send_json({
            "type": "status",
            "status": "PROCESSING",
            "n_samples": len(raw_inputs),
        })

        CHUNK_SIZE: int = 100
        df_full: pd.DataFrame = pd.DataFrame(raw_inputs)

        # Bug 15: wrap inference in per-call timeout — prevents indefinite hang on OOM/deadlock
        _WS_INFER_TIMEOUT = float(os.environ.get("APEX_WS_INFERENCE_TIMEOUT_S", "120"))

        async def _predict_with_timeout(df_chunk: pd.DataFrame) -> Dict[str, Any]:
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(engine.predict_batch, df_chunk, execution_context=_ctx),
                    timeout=_WS_INFER_TIMEOUT,
                )
            except asyncio.TimeoutError:
                await websocket.send_json({
                    "type": "error",
                    "error": f"Inference timed out after {_WS_INFER_TIMEOUT:.0f}s (APEX_WS_INFERENCE_TIMEOUT_S)",
                })
                raise

        if len(raw_inputs) <= CHUNK_SIZE:
            try:
                result = await _predict_with_timeout(df_full)
            except asyncio.TimeoutError:
                return
        else:
            all_predictions: List[Any] = []
            all_confidences: List[Any] = []
            n_chunks = (len(raw_inputs) + CHUNK_SIZE - 1) // CHUNK_SIZE
            problem_type = ""

            for chunk_idx in range(n_chunks):
                start = chunk_idx * CHUNK_SIZE
                end = min(start + CHUNK_SIZE, len(raw_inputs))
                chunk_df = df_full.iloc[start:end]

                try:
                    chunk_result = await _predict_with_timeout(chunk_df)
                except asyncio.TimeoutError:
                    return
                all_predictions.extend(chunk_result["predictions"])
                all_confidences.extend(chunk_result["confidences"])
                problem_type = chunk_result["problem_type"]

                await websocket.send_json({
                    "type": "progress",
                    "chunk": chunk_idx + 1,
                    "total_chunks": n_chunks,
                    "samples_completed": end,
                    "samples_total": len(raw_inputs),
                })

            result = {
                "predictions": all_predictions,
                "confidences": all_confidences,
                "problem_type": problem_type,
                "n_samples": len(all_predictions),
            }

        # 5. Optional XAI explanations
        explanations: Optional[Dict[str, Any]] = None
        if explain:
            await websocket.send_json({
                "type": "status",
                "status": "GENERATING_EXPLANATIONS",
            })

            effective_target = _resolve_xai_target(target_class, result)

            try:
                explanations = await asyncio.to_thread(
                    engine.generate_explanations,
                    df_full,
                    target_class=effective_target,
                    n_steps=n_steps,
                )
            except Exception as xai_exc:
                explanations = {
                    "status": "unavailable",
                    "unavailable_reasons": {"global": str(xai_exc)},
                    "availability": _prediction_contract.get("xai_availability", {}),
                }

        # 6. Send complete result
        payload = _prediction_output_payload(
            model_id=model_id,
            result=result,
            explanations=explanations,
            prediction_contract=_prediction_contract,
            io_audit=_io_audit,
            ctx=_ctx,
        )
        _cal = getattr(engine, "probability_calibrator", None)
        payload["calibration_applied"] = _cal is not None or payload.get("calibration_applied", False)
        payload["calibration_method"] = getattr(_cal, "method", None) if _cal else None
        payload["context_stage"] = _contract["context_stage"]
        payload["context_version"] = _contract["context_version"]
        payload["artifact_versions"] = _contract["artifact_versions"]
        await websocket.send_json({"type": "complete", "result": payload})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected during inference")
    except json.JSONDecodeError as e:
        try:
            await websocket.send_json({"type": "error", "error": f"Invalid JSON: {e}"})
        except Exception:
            pass
    except FileNotFoundError:
        try:
            await websocket.send_json({"type": "error", "error": "Model not found."})
        except Exception:
            pass
    except Exception as exc:
        logger.error("WebSocket /ws/predict error: %s", exc, exc_info=True)
        try:
            await websocket.send_json({"type": "error", "error": "Internal inference error."})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Inference  (synchronous -- kept for backward compat, small batches)
# ---------------------------------------------------------------------------

@app.post("/predict")
@require_context("training", required_fields=["global_schema"], require_session=True)
async def predict_multimodal(request: Request) -> Dict[str, Any]:
    """
    Run batch inference through a registered model's artifacts and optionally
    generate Captum IntegratedGradients explanations.

    REQUEST BODY
    ------------
    ``model_id``     : str          – required; must match a directory under
                                      ``models/registry/``
    ``inputs``       : List[Dict]   – list of feature dicts (one per sample)
    ``explain``      : bool         – default false; triggers Captum IG
    ``target_class`` : int          – default -1 (auto: explain predicted class); XAI target class index
    ``n_steps``      : int          – default 50; IG integration steps

    RESPONSE CONTRACT
    -----------------
    Returns::

        {
          "status":       "success",
          "predictions":  [int | float, ...],
          "confidences":  [float, ...],
          "problem_type": str,
          "n_samples":    int,
          "explanations": null | {
            "method":       "IntegratedGradients",
            "target_class": int,
            "tabular": {
              "feature_names":  [str, ...],
              "attributions":   [float, ...],
              "raw_attributions": [[float, ...], ...]
            } | null,
            "text": {
              "tokens":       [str, ...],
              "attributions": [float, ...],
              "note":         str
            } | null
          }
        }
    """
    try:
        body: Dict[str, Any] = await request.json()
        _sid = body.get("session_id") if isinstance(body, dict) else None
        _ctx = _get_session_context_or_422(_sid, "/predict")
        _require_context_artifact(_sid, "global_schema", "/predict")
        _contract = _context_contract_payload(_ctx)

        model_id: Optional[str] = body.get("model_id")
        if not model_id:
            raise HTTPException(
                status_code=400,
                detail=(
                    "model_id is required. "
                    "Retrieve available model IDs from GET /model-registry."
                ),
            )
        model_id = _sanitize_model_id(model_id)

        raw_inputs: List[Dict[str, Any]] = body.get("inputs", [])
        if not raw_inputs:
            raise HTTPException(
                status_code=400,
                detail="inputs list is empty. Provide at least one feature dict.",
            )
        _MAX_BATCH: int = 10_000
        if len(raw_inputs) > _MAX_BATCH:
            raise HTTPException(
                status_code=400,
                detail=f"Batch too large ({len(raw_inputs)} rows). Maximum is {_MAX_BATCH}.",
            )

        _metadata = _require_prediction_ready_model(_ctx, model_id, "/predict")
        _schema = _load_model_schema(model_id) or dict(getattr(_ctx, "global_schema", {}) or {})
        _prediction_contract = _build_prediction_contract(
            model_id=model_id,
            metadata=_metadata,
            schema=_schema,
            ctx=_ctx,
            class_labels=_load_model_class_labels(model_id),
        )
        raw_inputs, _io_audit = _normalise_prediction_inputs(raw_inputs, _prediction_contract)

        explain: bool      = bool(body.get("explain", False))
        target_class: int  = int(body.get("target_class", -1))
        n_steps: int        = int(body.get("n_steps", 50))

        from pipeline.inference_engine import MultimodalInferenceEngine

        # Re-use cached engine with LRU eviction (thread-safe)
        with _engine_cache_lock:
            if model_id in _engine_cache:
                _engine_cache.move_to_end(model_id)
                engine = _engine_cache[model_id]
            else:
                engine = None

        if engine is None:
            engine = await asyncio.to_thread(
                MultimodalInferenceEngine, model_id=model_id,
            )
            with _engine_cache_lock:
                _engine_cache[model_id] = engine
                while len(_engine_cache) > _MAX_ENGINES:
                    _engine_cache.popitem(last=False)

        df: pd.DataFrame = pd.DataFrame(raw_inputs)

        # Offload blocking inference to worker thread (CRIT-1 fix)
        result: Dict[str, Any] = await asyncio.to_thread(
            engine.predict_batch,
            df,
            execution_context=_ctx,
        )

        # Captum XAI – gradients enabled only inside generate_explanations
        explanations: Optional[Dict[str, Any]] = None
        if explain:
            effective_target = _resolve_xai_target(target_class, result)

            try:
                explanations = await asyncio.to_thread(
                    engine.generate_explanations,
                    df,
                    target_class=effective_target,
                    n_steps=n_steps,
                )
            except Exception as xai_exc:
                explanations = {
                    "status": "unavailable",
                    "unavailable_reasons": {"global": str(xai_exc)},
                    "availability": _prediction_contract.get("xai_availability", {}),
                }

        return _prediction_output_payload(
            model_id=model_id,
            result=result,
            explanations=explanations,
            prediction_contract=_prediction_contract,
            io_audit=_io_audit,
            ctx=_ctx,
        )

    except HTTPException:
        raise
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except Exception as exc:
        logger.error("/predict error: %s", exc, exc_info=True)
        return JSONResponse(
            {"error": "Prediction failed. Check server logs for details."},
            status_code=500,
        )


# ---------------------------------------------------------------------------
# V2 Session Management Endpoints (Phase 2)
# ---------------------------------------------------------------------------

from api.session_manager import session_manager
from core.orchestrator import orchestrator
from database.context_db import OptimisticLockError, context_db


@app.exception_handler(OptimisticLockError)
async def optimistic_lock_exception_handler(
    _request: Request,
    exc: OptimisticLockError,
) -> JSONResponse:
    """Return explicit 409 conflicts for stale session writes."""
    return JSONResponse(
        status_code=409,
        content={
            "error": "session_conflict",
            "detail": (
                "Session state was updated by another request. "
                "Retry with a fresh context snapshot."
            ),
            "message": str(exc),
        },
    )

class SessionCreateRequest(BaseModel):
    """Request body for creating a new session."""
    user_id: Optional[str] = None
    project_name: Optional[str] = None
    description: Optional[str] = None


class SessionDatasetRequest(BaseModel):
    """Request body for adding datasets to session."""
    dataset_urls: List[str]
    force_redownload: bool = False


@app.post("/v2/sessions")
async def create_session_v2(request: SessionCreateRequest) -> Dict[str, Any]:
    """
    Create a new session (Phase 2).
    
    Returns:
        {
            "session_id": str,
            "created_at": ISO8601,
            "status": "active"
        }
    """
    try:
        ctx = session_manager.create_session(
            user_id=request.user_id,
            project_name=request.project_name,
            description=request.description
        )
        
        return {
            "session_id": ctx.session_id,
            "created_at": ctx.created_at.isoformat(),
            "status": ctx.status
        }
    
    except Exception as exc:
        logger.error("/v2/sessions error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/v2/sessions/{session_id}")
async def get_session_v2(session_id: str) -> Dict[str, Any]:
    """
    Get a session by ID (Phase 2).
    
    Returns full SessionContext as dict.
    """
    try:
        ctx = session_manager.get_session(session_id)
        if not ctx:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        return ctx.to_dict()
    
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("/v2/sessions/{session_id} error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/context/{session_id}/phase-timings")
@require_context("ingestion_complete", require_session=True)
async def get_phase_timings(session_id: str) -> Dict[str, Any]:
    """Return per-phase timing breakdown from ExecutionContext."""
    ctx = session_manager.get_session(session_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    timings = dict(getattr(ctx, "phase_timings", {}) or {})
    total_duration_s = round(sum(float(v) for v in timings.values()), 3) if timings else 0.0
    return {
        "session_id": session_id,
        "phase_timings": timings,
        "total_duration_s": total_duration_s,
    }


@app.get("/context/{session_id}/drift-status")
@require_context("ingestion_complete", require_session=True)
async def get_drift_status(session_id: str) -> Dict[str, Any]:
    """Return latest drift state persisted in ExecutionContext."""
    ctx = session_manager.get_session(session_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    active_model_id = getattr(ctx, "active_prediction_model_id", None)
    if not active_model_id:
        unavailable = _not_available_monitor_payload(
            "no_deployment_ready_model_registered",
            ctx,
            None,
        )
        return {
            "session_id": session_id,
            **unavailable["data"],
        }

    details = dict(getattr(ctx, "drift_details", {}) or {})
    return {
        "session_id": session_id,
        "drift_detected": bool(getattr(ctx, "drift_detected", False)),
        "drift_severity": float(getattr(ctx, "drift_severity", 0.0) or 0.0),
        "drift_details": details,
        "monitor": {
            "status": "available",
            "model_id": active_model_id,
            "severity": float(getattr(ctx, "drift_severity", 0.0) or 0.0),
            "breached_metrics": details.get("breached_metrics", []),
            "retrain_recommendation": details.get("retrain_recommendation"),
        },
    }


@app.get("/context/{session_id}/fit-analysis")
@require_context("ingestion_complete", require_session=True)
async def get_fit_analysis(session_id: str) -> Dict[str, Any]:
    """Return latest TrialIntelligence fit analysis from ExecutionContext."""
    ctx = session_manager.get_session(session_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    return {
        "session_id": session_id,
        "training_fit_analysis": dict(getattr(ctx, "training_fit_analysis", {}) or {}),
        "training_signals": dict(getattr(ctx, "training_signals", {}) or {}),
    }


@app.post("/context/{session_id}/probe-sample")
@require_context("ingestion_complete", require_session=True)
async def save_probe_sample(session_id: str, request: Request) -> Dict[str, Any]:
    """Persist a tabular probe sample for a session."""
    body = await request.json()
    X = body.get("X")
    y = body.get("y")
    if X is None or y is None:
        raise HTTPException(status_code=400, detail="Both 'X' and 'y' are required")

    try:
        context_db.save_probe_sample(session_id, X, y)
        return {"status": "saved", "session_id": session_id}
    except Exception as exc:
        logger.error("/context/%s/probe-sample save failed: %s", session_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/context/{session_id}/probe-sample/status")
@require_context("ingestion_complete", require_session=True)
async def probe_sample_status(session_id: str) -> Dict[str, Any]:
    """Check if a persisted probe sample exists for a session."""
    try:
        sample = context_db.load_probe_sample(session_id)
    except Exception as exc:
        logger.error("/context/%s/probe-sample/status failed: %s", session_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if sample is None:
        return {"session_id": session_id, "exists": False}

    X, y = sample
    n_rows = int(len(X)) if hasattr(X, "__len__") else 0
    n_cols = int(X.shape[1]) if hasattr(X, "shape") and len(X.shape) > 1 else 0
    return {
        "session_id": session_id,
        "exists": True,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "target_size": int(len(y)) if hasattr(y, "__len__") else 0,
    }


def _load_registry_metadata(model_id: str) -> Dict[str, Any]:
    """Best-effort metadata loader for a registered model ID."""
    try:
        metadata_path = MODEL_REGISTRY_DIR / str(model_id) / "metadata.json"
        if not metadata_path.exists():
            return {}
        with open(metadata_path, encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


@app.get("/v2/sessions/{session_id}/intelligence/calibration")
@require_context("ingestion_complete", require_session=True)
async def get_intelligence_calibration(session_id: str) -> Dict[str, Any]:
    """Return calibration metrics from execution context and registry metadata."""
    ctx = session_manager.get_session(session_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    active_model_id = getattr(ctx, "active_prediction_model_id", None)
    per_model_by_id: Dict[str, Dict[str, Any]] = {}

    training_signals = dict(getattr(ctx, "training_signals", {}) or {})
    session_calibration = training_signals.get("calibration")
    if isinstance(session_calibration, dict):
        metrics_after = dict(session_calibration.get("metrics_after", {}) or {})
        synthetic_model_id = str(active_model_id or "active")
        per_model_by_id[synthetic_model_id] = {
            "model_id": synthetic_model_id,
            "method": session_calibration.get("mode") or session_calibration.get("method"),
            "nll": metrics_after.get("nll", session_calibration.get("nll")),
            "brier": metrics_after.get("brier", session_calibration.get("brier")),
            "ece": metrics_after.get("ece", session_calibration.get("ece")),
        }

    for model_id in list(getattr(ctx, "registered_model_ids", []) or []):
        metadata = _load_registry_metadata(str(model_id))
        if not metadata:
            continue

        training_summary = dict(metadata.get("training", {}) or {})
        calibration = metadata.get("calibration")
        if not isinstance(calibration, dict):
            calibration = training_summary.get("calibration")
        if not isinstance(calibration, dict):
            calibration = dict(metadata.get("training_signals", {}) or {}).get("calibration")
        if not isinstance(calibration, dict):
            continue

        metrics_after = dict(calibration.get("metrics_after", {}) or {})
        per_model_by_id[str(model_id)] = {
            "model_id": str(model_id),
            "method": calibration.get("mode") or calibration.get("method"),
            "nll": metrics_after.get("nll", calibration.get("nll")),
            "brier": metrics_after.get("brier", calibration.get("brier")),
            "ece": metrics_after.get("ece", calibration.get("ece")),
        }

    return {
        "per_model": list(per_model_by_id.values()),
        "active_model_id": active_model_id,
    }


@app.get("/v2/sessions/{session_id}/intelligence/xai")
@require_context("ingestion_complete", require_session=True)
async def get_intelligence_xai(session_id: str) -> Dict[str, Any]:
    """Return per-model XAI summaries when available."""
    ctx = session_manager.get_session(session_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    per_model: List[Dict[str, Any]] = []

    for model_id in list(getattr(ctx, "registered_model_ids", []) or []):
        payload: Dict[str, Any] = {}
        summary_path = MODEL_REGISTRY_DIR / str(model_id) / "xai_summary.json"
        if summary_path.exists():
            try:
                with open(summary_path, encoding="utf-8") as fh:
                    raw_payload = json.load(fh)
                if isinstance(raw_payload, dict):
                    payload = raw_payload
            except Exception:
                payload = {}
        if not payload:
            payload = dict(_load_registry_metadata(str(model_id)).get("xai", {}) or {})
        if not payload:
            continue

        entry: Dict[str, Any] = {"model_id": str(model_id)}
        if payload.get("method") is not None:
            entry["method"] = payload.get("method")

        tabular_payload = payload.get("tabular")
        if isinstance(tabular_payload, dict):
            ranking = tabular_payload.get("feature_ranking", []) or []
            normalized_ranking: List[List[Any]] = []
            for item in ranking:
                if isinstance(item, dict):
                    feat = item.get("feature")
                    imp = item.get("importance")
                    if feat is not None and imp is not None:
                        normalized_ranking.append([feat, imp])
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    normalized_ranking.append([item[0], item[1]])
            entry["tabular"] = {"feature_ranking": normalized_ranking}

        text_payload = payload.get("text")
        if isinstance(text_payload, dict):
            entry["text"] = text_payload

        image_payload = payload.get("image")
        if isinstance(image_payload, dict):
            entry["image"] = image_payload

        per_model.append(entry)

    return {"per_model": per_model}


@app.get("/v2/sessions/{session_id}/intelligence/guardrails")
@require_context("ingestion_complete", require_session=True)
async def get_intelligence_guardrails(session_id: str) -> Dict[str, Any]:
    """Return guardrail snapshot for the session context."""
    ctx = session_manager.get_session(session_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return _build_guardrail_snapshot(session_id, ctx)


@app.get("/v2/sessions/{session_id}/intelligence/ranked-candidates")
@require_context("ingestion_complete", require_session=True)
async def get_intelligence_ranked_candidates(session_id: str) -> Dict[str, Any]:
    """Return ranked candidates with optional probe scores."""
    ctx = session_manager.get_session(session_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    candidate_scores: Dict[str, float] = {}
    ranked_raw = getattr(ctx, "ranked_candidates", {}) or {}

    def _update_candidate(name: Any, score: Any) -> None:
        if not name:
            return
        try:
            score_val = float(score)
        except Exception:
            score_val = 0.0
        key = str(name)
        current = candidate_scores.get(key)
        if current is None or score_val > current:
            candidate_scores[key] = score_val

    if isinstance(ranked_raw, dict):
        for key, value in ranked_raw.items():
            if isinstance(value, (int, float)):
                _update_candidate(key, value)
                continue
            if isinstance(value, dict):
                _update_candidate(
                    value.get("name") or value.get("model") or key,
                    value.get("score", value.get("final_score", 0.0)),
                )
                continue
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        _update_candidate(
                            item.get("name") or item.get("model") or item.get("model_name") or key,
                            item.get("score", item.get("final_score", item.get("val_score", 0.0))),
                        )
                    else:
                        _update_candidate(item, 0.0)

    if not candidate_scores:
        for item in list(getattr(ctx, "model_choices", []) or []):
            if isinstance(item, dict):
                _update_candidate(item.get("name") or item.get("model") or item.get("model_name"), item.get("score", 0.0))
            else:
                _update_candidate(str(item), 0.0)

    probe_cache = dict(getattr(ctx, "probe_scores_cache", {}) or {})

    def _probe_score_for(name: str) -> Optional[float]:
        direct = probe_cache.get(name)
        if isinstance(direct, (int, float)):
            return float(direct)
        if isinstance(direct, dict) and isinstance(direct.get("score"), (int, float)):
            return float(direct.get("score"))
        for value in probe_cache.values():
            if isinstance(value, dict):
                nested = value.get(name)
                if isinstance(nested, (int, float)):
                    return float(nested)
                if isinstance(nested, dict) and isinstance(nested.get("score"), (int, float)):
                    return float(nested.get("score"))
        return None

    ranked: List[Dict[str, Any]] = []
    for name, score in sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True):
        ranked.append(
            {
                "name": name,
                "score": float(score),
                "probe_score": _probe_score_for(name),
            }
        )

    return {
        "selected_model": getattr(ctx, "selected_model", None),
        "reason": getattr(ctx, "model_selection_reason", ""),
        "ranked": ranked,
    }


@app.get("/v2/sessions/{session_id}/intelligence/trial-intelligence")
@require_context("ingestion_complete", require_session=True)
async def get_intelligence_trial_intelligence(session_id: str) -> Dict[str, Any]:
    """Return fit-analysis, adaptive LR guidance, and recent trial diagnostics."""
    ctx = session_manager.get_session(session_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    fit_analysis = dict(getattr(ctx, "training_fit_analysis", {}) or {})
    training_signals = dict(getattr(ctx, "training_signals", {}) or {})

    recent_trials: List[Dict[str, Any]] = []
    trial_diagnostics = list(training_signals.get("trial_diagnostics", []) or [])
    for row in trial_diagnostics[-10:]:
        if not isinstance(row, dict):
            continue
        recent_trials.append(
            {
                "trial_id": row.get("trial_id", row.get("trial", row.get("number"))),
                "lr": row.get("learning_rate", row.get("lr")),
                "val_loss": row.get("val_loss"),
                "decision": row.get("decision", row.get("fit_type", "keep")),
            }
        )

    if not recent_trials:
        try:
            from automl.meta_learning import MetaLearningStore

            history = list(MetaLearningStore().load() or [])
            for index, record in enumerate(history[-5:], start=1):
                params = dict(record.get("best_params", {}) or {}) if isinstance(record, dict) else {}
                recent_trials.append(
                    {
                        "trial_id": index,
                        "lr": params.get("learning_rate", params.get("lr")),
                        "val_loss": record.get("performance") if isinstance(record, dict) else None,
                        "decision": "historical",
                    }
                )
        except Exception:
            pass

    adaptive_lr: Dict[str, Any] = {}
    next_feedback = fit_analysis.get("next_run_feedback")
    if isinstance(next_feedback, dict):
        lr_factor = next_feedback.get("recommended_lr_factor")
        if lr_factor is None:
            lr_factor = next_feedback.get("lr_factor", next_feedback.get("learning_rate_factor"))
        trigger_reason = next_feedback.get("trigger_reason", next_feedback.get("reason"))

        if lr_factor is not None:
            try:
                adaptive_lr["recommended_lr_factor"] = float(lr_factor)
            except Exception:
                pass
        if trigger_reason is not None:
            adaptive_lr["trigger_reason"] = str(trigger_reason)

    if not adaptive_lr:
        fit_type = str(fit_analysis.get("fit_type", "unknown"))
        if fit_type == "overfitting":
            adaptive_lr = {
                "recommended_lr_factor": 0.7,
                "trigger_reason": "validation degradation",
            }
        elif fit_type == "underfitting":
            adaptive_lr = {
                "recommended_lr_factor": 1.2,
                "trigger_reason": "slow convergence",
            }
        else:
            adaptive_lr = {
                "recommended_lr_factor": 1.0,
                "trigger_reason": "stable fit",
            }

    return {
        "fit_analysis": fit_analysis,
        "adaptive_lr": adaptive_lr,
        "recent_trials": recent_trials,
    }


@app.get("/v2/sessions/{session_id}/intelligence/preprocessing-plan")
@require_context("ingestion_complete", require_session=True)
async def get_intelligence_preprocessing_plan(session_id: str) -> Dict[str, Any]:
    """Return preprocessing plan and context details for frontend transparency."""
    ctx = session_manager.get_session(session_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    preprocessing_context = dict(getattr(ctx, "preprocessing_context", {}) or {})
    dataset_plans_payload = preprocessing_context.get("dataset_plans")
    per_dataset_plans: List[Dict[str, Any]] = []

    if isinstance(dataset_plans_payload, dict):
        for dataset_id, dataset_plan in dataset_plans_payload.items():
            row: Dict[str, Any] = {"dataset_id": str(dataset_id)}
            if isinstance(dataset_plan, dict):
                row.update(dataset_plan)
            per_dataset_plans.append(row)

    if not per_dataset_plans:
        for dataset_id, profile in dict(getattr(ctx, "dataset_profiles", {}) or {}).items():
            profile_plan = getattr(profile, "preprocessing_plan", None)
            if isinstance(profile_plan, dict) and profile_plan:
                row = {"dataset_id": str(dataset_id)}
                row.update(profile_plan)
                per_dataset_plans.append(row)

    return {
        "version": getattr(ctx, "preprocess_plan_version", None),
        "plan": dict(getattr(ctx, "preprocessing_plan", {}) or {}),
        "choices": dict(getattr(ctx, "preprocessing_choices", {}) or {}),
        "context": preprocessing_context,
        "per_dataset_plans": per_dataset_plans,
    }


@app.get("/v2/sessions/{session_id}/intelligence/drift")
@require_context("ingestion_complete", require_session=True)
async def get_intelligence_drift(session_id: str) -> Dict[str, Any]:
    """Return drift detection status, embedding drift, concept drift, and retraining depth."""
    ctx = session_manager.get_session(session_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    from monitoring.drift_detector import DriftDetector
    from monitoring.performance_tracker import PerformanceTracker

    dd = DriftDetector()
    pt = PerformanceTracker(model_id=getattr(ctx, "active_prediction_model_id", None) or session_id)

    # ── Covariate drift snapshot from context ────────────────────────────────
    covariate_drift: Dict[str, Any] = {}
    try:
        drift_state = getattr(ctx, "drift_state", None) or {}
        if isinstance(drift_state, dict) and drift_state:
            report = drift_state.get("last_report") or {}
            covariate_drift = {
                "detected": bool(report.get("drift_detected", False)),
                "composite_score": float(report.get("composite_score", 0.0) or 0.0),
                "per_feature": dict(report.get("per_feature_ks", {})),
                "ks_statistic": float(report.get("ks_statistic", 0.0) or 0.0),
                "psi": float(report.get("psi", 0.0) or 0.0),
                "fdd": float(report.get("fdd", 0.0) or 0.0),
            }
        elif hasattr(ctx, "latest_drift_report"):
            raw = getattr(ctx, "latest_drift_report", None) or {}
            if isinstance(raw, dict):
                covariate_drift = {
                    "detected": bool(raw.get("drift_detected", False)),
                    "composite_score": float(raw.get("composite_score", 0.0) or 0.0),
                    "per_feature": dict(raw.get("per_feature_ks", {})),
                }
    except Exception:
        pass

    # ── Concept drift from prediction distribution history ───────────────────
    concept_drift: Dict[str, Any] = {}
    try:
        pred_dist = getattr(pt, "_pred_dist_history", {})
        key = f"{getattr(ctx, 'active_prediction_model_id', None) or session_id}_pred_dist"
        if key in pred_dist and len(pred_dist[key]) >= 2:
            from scipy import stats as _stats
            ref = pred_dist[key][0]
            cur = pred_dist[key][-1]
            ks_stat, p_value = _stats.ks_2samp(ref, cur)
            concept_drift = {
                "detected": bool(p_value < 0.05),
                "ks_stat": float(ks_stat),
                "p_value": float(p_value),
                "n_batches": int(len(pred_dist[key])),
            }
    except Exception:
        pass

    # ── Embedding drift from registry metadata ───────────────────────────────
    embedding_drift: Dict[str, Any] = {}
    try:
        from config.paths import MODEL_REGISTRY_DIR

        active_id = getattr(ctx, "active_prediction_model_id", None)
        if active_id:
            model_dir = MODEL_REGISTRY_DIR / active_id
            for modality in ("text", "image", "timeseries"):
                ref_path = model_dir / f"reference_embeddings_{modality}.npy"
                cur_path = model_dir / f"current_embeddings_{modality}.npy"
                if ref_path.exists() and cur_path.exists():
                    try:
                        ref_emb = __import__("numpy").load(str(ref_path))
                        cur_emb = __import__("numpy").load(str(cur_path))
                        embedding_drift[modality] = dd.detect_embedding_drift(
                            ref_emb, cur_emb, modality_name=modality
                        )
                    except Exception:
                        pass
    except Exception:
        pass

    # ── Retraining depth ─────────────────────────────────────────────────────
    retraining_depth = getattr(ctx, "retraining_depth_required", "none")

    # ── Last-checked timestamp ───────────────────────────────────────────────
    from datetime import datetime, timezone
    last_checked = datetime.now(timezone.utc).isoformat()

    return {
        "covariate_drift": covariate_drift,
        "concept_drift": concept_drift,
        "embedding_drift": embedding_drift,
        "retraining_depth": retraining_depth,
        "last_checked": last_checked,
    }


@app.get("/v2/sessions/{session_id}/intelligence/feature-intelligence")
@require_context("ingestion_complete", require_session=True)
async def get_intelligence_feature_intelligence(session_id: str) -> Dict[str, Any]:
    """
    Return per-dataset feature intelligence: semantic roles, interaction scores,
    business patterns, uncertainty signals, id/high-missing columns, avg text length.
    """
    ctx = session_manager.get_session(session_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    fi = dict(getattr(ctx, "feature_intelligence", {}) or {})
    # Summarise across datasets for the top-level summary
    all_roles: Dict[str, int] = {}
    all_patterns: Dict[str, int] = {}
    total_id_cols: int = 0
    total_high_missing: int = 0
    avg_text_lengths: list = []

    for ds_intel in fi.values():
        for role, cols in (ds_intel.get("semantic_roles") or {}).items():
            all_roles[role] = all_roles.get(role, 0) + (len(cols) if isinstance(cols, list) else 1)
        for pattern, info in (ds_intel.get("business_patterns") or {}).items():
            all_patterns[pattern] = all_patterns.get(pattern, 0) + 1
        total_id_cols += len(ds_intel.get("id_columns") or [])
        total_high_missing += len(ds_intel.get("high_missing_cols") or [])
        tl = ds_intel.get("avg_text_len")
        if tl:
            avg_text_lengths.append(float(tl))

    return {
        "per_dataset": fi,
        "summary": {
            "semantic_role_counts": all_roles,
            "business_pattern_counts": all_patterns,
            "total_id_columns": total_id_cols,
            "total_high_missing_columns": total_high_missing,
            "avg_text_len": (sum(avg_text_lengths) / len(avg_text_lengths)) if avg_text_lengths else None,
            "n_datasets": len(fi),
        },
    }


@app.get("/v2/sessions/{session_id}/intelligence")
@require_context("ingestion_complete", require_session=True)
async def get_session_intelligence(session_id: str) -> Dict[str, Any]:
    """
    Return a structured, UI-facing snapshot of ExecutionContext intelligence.
    """
    ctx = session_manager.get_session(session_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    contract = _context_contract_payload(ctx)

    fusion_source: Optional[str] = None
    if ctx.fusion_strategy:
        fusion_source = (
            "user_override"
            if any(e.get("field") == "fusion_strategy" for e in ctx.override_history)
            else "auto"
        )

    xs3_gap = float(ctx.confidence_map.get("xs3_target_gap", 0.0) or 0.0)
    active_modalities = ctx.active_modalities or ctx.get_active_modalities()
    updated_at = ctx.updated_at.isoformat() if hasattr(ctx.updated_at, "isoformat") else str(ctx.updated_at)

    return {
        "session_id": ctx.session_id,
        "pipeline_stage": ctx.pipeline_stage,
        "context_version": ctx.version,
        "active_modalities": active_modalities,
        "modality_presence": {k: bool(v) for k, v in ctx.modality_presence.items()},
        "predictability_scores": {
            k: round(float(v), 4)
            for k, v in ctx.predictability_scores.items()
            if v is not None
        },
        "artifact_versions": contract["artifact_versions"],
        "fusion_strategy": ctx.fusion_strategy,
        "fusion_source": fusion_source,
        "fusion_mode": ctx.fusion_mode,
        "modality_importance": {
            k: round(float(v), 4)
            for k, v in ctx.modality_importance.items()
            if v is not None
        },
        "selected_model": ctx.selected_model,
        "model_selection_reason": ctx.model_selection_reason,
        "global_target": ctx.global_target,
        "global_target_confidence": round(float(ctx.global_target_confidence or 0.0), 4),
        "target_confidence": round(float(ctx.target_confidence or 0.0), 4),
        "xs3_confidence_gap": round(xs3_gap, 4),
        "override_applied": bool(ctx.user_overrides),
        "override_history": ctx.override_history[-10:],
        "preprocessing_choices": ctx.preprocessing_choices,
        "drift_detected": bool(ctx.drift_detected),
        "drift_severity": ctx.drift_severity,
        "drift_details": dict(ctx.drift_details or {}),
        "datasets_compatible": bool(ctx.datasets_compatible),
        "should_include_fusion": ctx.should_include_fusion(),
        "probe_scores_cache": ctx.probe_scores_cache,
        "ranked_candidates": ctx.ranked_candidates,
        "registered_model_ids": list(ctx.registered_model_ids or []),
        "active_prediction_model_id": ctx.active_prediction_model_id,
        "training_signals": ctx.training_signals,
        "training_fit_analysis": dict(getattr(ctx, "training_fit_analysis", {}) or {}),
        "xai_config": dict(getattr(ctx, "xai_config", {}) or {}),
        "fusion_policy_locked": contract["fusion_policy_locked"],
        "fusion_policy_source": contract["fusion_policy_source"],
        "phase_timings": dict(getattr(ctx, "phase_timings", {}) or {}),
        "guardrails": _build_guardrail_snapshot(session_id, ctx),
        "execution_log_count": len(ctx.execution_log),
        "updated_at": updated_at,
    }


@app.get("/v2/sessions/{session_id}/decision-trace")
@require_context("ingestion_complete", require_session=True)
async def get_decision_trace(session_id: str, limit: int = 50) -> Dict[str, Any]:
    """
    Return structured decision trace from ExecutionContext.execution_log.
    """
    ctx = session_manager.get_session(session_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    category_map = {
        "ingestion": "ingestion",
        "schema_detection": "schema",
        "schema": "schema",
        "target": "target",
        "global_target": "target",
        "global_schema": "schema",
        "preprocessing": "preprocessing",
        "model_selection": "model_selection",
        "model": "model_selection",
        "training": "training",
        "drift": "monitoring",
        "drift_detection": "monitoring",
        "drift_feedback": "monitoring",
        "monitoring": "monitoring",
        "override": "override",
        "pipeline": "pipeline",
        "initialization": "pipeline",
    }

    limit = max(1, min(int(limit), 500))
    raw_log = ctx.execution_log[-limit:] if len(ctx.execution_log) > limit else ctx.execution_log
    trace: List[Dict[str, Any]] = []
    for seq, entry in enumerate(raw_log, start=1):
        stage = str(entry.get("stage", "other"))
        trace.append({
            "seq": seq,
            "timestamp": entry.get("timestamp", ""),
            "stage": stage,
            "decision": entry.get("decision", ""),
            "evidence": entry.get("evidence"),
            "category": category_map.get(stage.lower(), "other"),
        })

    curated: List[str] = []
    if ctx.global_target:
        gap = float(ctx.confidence_map.get("xs3_target_gap", 0.0) or 0.0)
        src = (
            "user override"
            if "global_target" in ctx.user_overrides
            else f"XS3 gap={gap:.3f} (highest among candidates)"
        )
        curated.append(f"Target selected: '{ctx.global_target}' - {src}")

    if ctx.preprocessing_choices:
        mods = [m for m in ("tabular", "text", "image") if m in ctx.preprocessing_choices]
        curated.append(f"Preprocessing configured for: {', '.join(mods) if mods else 'none'}")

    if ctx.selected_model:
        probe_cache = ctx.probe_scores_cache or {}
        top_probe = max(
            (s.get("score", 0) for s in probe_cache.values() if isinstance(s, dict)),
            default=None,
        )
        curated.append(
            f"Model selected: '{ctx.selected_model}'"
            + (f" - probe score={float(top_probe):.3f}" if isinstance(top_probe, (int, float)) and top_probe else " (heuristic)")
        )

    if ctx.fusion_strategy:
        n_active = len(ctx.get_active_modalities())
        was_override = any(e.get("field") == "fusion_strategy" for e in ctx.override_history)
        curated.append(
            f"Fusion strategy: '{ctx.fusion_strategy}' "
            + ("(user override)" if was_override else f"(auto - {n_active} modalities active)")
        )

    if ctx.training_signals:
        acc = ctx.training_signals.get("best_val_acc")
        ttime = ctx.training_signals.get("training_time", "?")
        if isinstance(acc, (int, float)):
            curated.append(f"Training completed - best val_acc={float(acc):.2%}, time={ttime}")

    if ctx.active_prediction_model_id:
        model_count = len(ctx.registered_model_ids or [])
        curated.append(
            f"Retraining completed - active model='{ctx.active_prediction_model_id}'"
            + (f", registered models={model_count}" if model_count else "")
        )

    if ctx.drift_detected:
        sev = ctx.drift_severity or "unknown"
        curated.append(f"Drift detected - severity={sev}. Retraining recommended.")

    if ctx.override_history:
        for ov in ctx.override_history[-3:]:
            curated.append(
                f"Override: '{ov.get('field', '?')}' changed "
                f"'{ov.get('old_value', '?')}' -> '{ov.get('new_value', '?')}' "
                f"({ov.get('reason', 'no reason')})"
            )

    return {
        "session_id": ctx.session_id,
        "total_decisions": len(ctx.execution_log),
        "trace": trace,
        "curated_summary": curated,
    }


@app.get("/v2/sessions")
async def list_sessions_v2(
    user_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> Dict[str, Any]:
    """
    List sessions with optional filtering (Phase 2).
    
    Query params:
        user_id: Filter by user ID
        status: Filter by status (active, closed, error)
        limit: Max sessions to return
        offset: Pagination offset
    
    Returns:
        {
            "sessions": [session_summary, ...],
            "total": int
        }
    """
    try:
        sessions = session_manager.list_sessions(user_id, status, limit, offset)
        total = session_manager.db.get_session_count(user_id, status)
        
        return {
            "sessions": sessions,
            "total": total
        }
    
    except Exception as exc:
        logger.error("/v2/sessions list error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/v2/sessions/{session_id}/close")
async def close_session_v2(session_id: str) -> Dict[str, Any]:
    """
    Close a session (Phase 2).
    
    Returns:
        {
            "session_id": str,
            "status": "closed",
            "closed_at": ISO8601
        }
    """
    try:
        success = session_manager.close_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        return {
            "session_id": session_id,
            "status": "closed",
            "closed_at": datetime.now(timezone.utc).isoformat()
        }
    
    except HTTPException:
        raise
    except OptimisticLockError:
        raise
    except Exception as exc:
        logger.error("/v2/sessions/{session_id}/close error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/v2/sessions/{session_id}/datasets")
async def add_datasets_to_session_v2(
    session_id: str,
    request: SessionDatasetRequest
) -> Dict[str, Any]:
    """
    Add datasets to a session (Phase 2).
    
    Starts async ingestion and associates datasets with session.
    
    Returns:
        {
            "task_id": str,
            "status": "processing",
            "datasets": [{"dataset_id": str, "source": str, "status": str}, ...]
        }
    """
    try:
        # Ensure session exists
        ctx = session_manager.get_session(session_id)
        if not ctx:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        # Use existing ingestion endpoint logic
        ingestion_req = IngestionRequest(
            dataset_urls=request.dataset_urls,
            session_id=session_id
        )
        
        # Call existing ingestion endpoint
        result = await ingest_datasets_endpoint(ingestion_req)
        
        # Extract task_id
        task_id = result.get("task_id")
        
        return {
            "task_id": task_id,
            "status": "processing",
            "datasets": [
                {
                    "dataset_id": None,  # Will be assigned after ingestion
                    "source": url,
                    "status": "ingesting"
                }
                for url in request.dataset_urls
            ]
        }
    
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("/v2/sessions/{session_id}/datasets error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/v2/sessions/{session_id}/datasets/{dataset_id}")
async def remove_dataset_from_session_v2(
    session_id: str,
    dataset_id: str
) -> Dict[str, Any]:
    """
    Remove a dataset from a session's active list (Phase 2).
    
    Dataset remains in cache but is not used for training.
    
    Returns:
        {
            "session_id": str,
            "dataset_id": str,
            "status": "removed",
            "cache_preserved": true
        }
    """
    try:
        success = session_manager.remove_dataset_from_session(session_id, dataset_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found in session")
        
        return {
            "session_id": session_id,
            "dataset_id": dataset_id,
            "status": "removed",
            "cache_preserved": True
        }
    
    except HTTPException:
        raise
    except OptimisticLockError:
        raise
    except Exception as exc:
        logger.error("/v2/sessions/{session_id}/datasets/{dataset_id} DELETE error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# V2 Schema Detection Endpoints (Phase 3)
# ---------------------------------------------------------------------------

# Services removed - using orchestrator directly (no more wrapper services)

@app.get("/v2/sessions/{session_id}/datasets")
async def list_session_datasets_v2(session_id: str) -> Dict[str, Any]:
    """
    List all datasets in a session with their status (Phase 3).
    
    Returns:
        {
            "session_id": str,
            "datasets": [
                {
                    "dataset_id": str,
                    "source_url": str,
                    "status": "active" | "cached",
                    "schema_detected": bool,
                    "target_detected": bool
                }
            ],
            "active_datasets": [{"dataset_id": str, "source": str, "in_session": true}, ...],
            "cached_datasets": [{"dataset_id": str, "source": str, "in_session": false}, ...]
        }
    """
    try:
        ctx = session_manager.get_session(session_id)
        if not ctx:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        profiles = context_db.get_session_profiles(session_id)
        
        datasets = []
        active_datasets: List[Dict[str, Any]] = []
        cached_datasets: List[Dict[str, Any]] = []
        for profile in profiles:
            dataset_id = profile['dataset_id']
            in_session = dataset_id in ctx.active_dataset_ids
            source_url = profile.get('source_url')
            dataset_row = {
                "dataset_id": profile['dataset_id'],
                "source_url": source_url,
                "status": "active" if in_session else "cached",
                "schema_detected": profile.get('schema_detected', False),
                "target_detected": profile.get('target_detected', False),
                "created_at": profile.get('created_at')
            }
            datasets.append(dataset_row)

            compact = {
                "dataset_id": dataset_id,
                "source": source_url,
                "in_session": in_session,
                "cached_at": profile.get('created_at'),
            }
            if in_session:
                active_datasets.append(compact)
            else:
                cached_datasets.append(compact)
        
        return {
            "session_id": session_id,
            "datasets": datasets,
            "active_datasets": active_datasets,
            "cached_datasets": cached_datasets,
        }
    
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("/v2/sessions/{session_id}/datasets error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/v2/datasets/{dataset_id}/schema")
async def get_dataset_schema_v2(dataset_id: str) -> Dict[str, Any]:
    """
    Get schema detection results for a dataset (Phase 3).
    
    Returns:
        {
            "dataset_id": str,
            "schema_detected": bool,
            "schema_result": {...},
            "confidence": float,
            "evidence": str
        }
    """
    try:
        profile = context_db.load_profile(dataset_id)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
        
        return {
            "dataset_id": dataset_id,
            "schema_detected": profile.get('schema_detected', False),
            "schema_result": profile.get('schema_result'),
            "confidence": profile.get('schema_confidence', 0.0),
            "evidence": profile.get('schema_evidence')
        }
    
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("/v2/datasets/{dataset_id}/schema error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


class SchemaOverrideRequest(BaseModel):
    """Request body for schema override."""
    schema_override: Dict[str, Any]
    reason: str


@app.post("/v2/datasets/{dataset_id}/override-schema")
async def override_dataset_schema_v2(
    dataset_id: str,
    request: SchemaOverrideRequest,
    session_id: str = Query(...)
) -> Dict[str, Any]:
    """
    Override detected schema for a dataset (Phase 3).
    
    Returns:
        {
            "dataset_id": str,
            "schema_overridden": bool,
            "new_schema": {...}
        }
    """
    try:
        ctx = session_manager.get_session(session_id)
        if not ctx:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        with _session_lock:
            profile = context_db.load_profile(dataset_id)
            if not profile:
                raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
            if dataset_id not in ctx.active_dataset_ids:
                raise HTTPException(
                    status_code=400,
                    detail=f"Dataset {dataset_id} is not active in session {session_id}",
                )

            profile["schema_detected"] = True
            profile["schema_result"] = request.schema_override
            profile["schema_confidence"] = 1.0
            profile["schema_evidence"] = f"Manual override: {request.reason}"

            user_overrides = profile.get("user_overrides") or {}
            user_overrides["schema_override"] = {
                "value": request.schema_override,
                "reason": request.reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            profile["user_overrides"] = user_overrides

            context_db.save_profile(profile, session_id)
            ctx.log_decision(
                "schema_override",
                f"Schema overridden for dataset {dataset_id}",
                request.reason,
            )
            session_manager.update_session_context(session_id, ctx)
        
        return {
            "dataset_id": dataset_id,
            "schema_overridden": True,
            "new_schema": request.schema_override
        }
    
    except HTTPException:
        raise
    except OptimisticLockError:
        raise
    except Exception as exc:
        logger.error("/v2/datasets/{dataset_id}/override-schema error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# V2 Target Detection Endpoints (Phase 4)
# ---------------------------------------------------------------------------

# Target detection service removed - using orchestrator directly


@app.get("/v2/datasets/{dataset_id}/target-candidates")
async def get_target_candidates_v2(dataset_id: str) -> Dict[str, Any]:
    """
    Get target candidates for a dataset (Phase 4).
    
    Returns:
        {
            "dataset_id": str,
            "candidates": [{name, score, reason}, ...],
            "chosen_target": str | null,
            "target_locked": bool
        }
    """
    try:
        profile = context_db.load_profile(dataset_id)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

        candidates = profile.get('target_candidates', [])
        ranked_scores = []
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            raw_score = cand.get('score', cand.get('final_score', 0.0))
            if isinstance(raw_score, (int, float)):
                ranked_scores.append(float(raw_score))
        ranked_scores.sort(reverse=True)
        xs3_confidence_gap = (
            ranked_scores[0] - ranked_scores[1]
            if len(ranked_scores) > 1 else
            (ranked_scores[0] if ranked_scores else 0.0)
        )
        
        return {
            "dataset_id": dataset_id,
            "candidates": candidates,
            "chosen_target": profile.get('chosen_target'),
            "target_locked": profile.get('target_locked', False),
            "xs3_confidence_gap": float(max(0.0, xs3_confidence_gap)),
        }
    
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("/v2/datasets/{dataset_id}/target-candidates error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


class TargetOverrideRequest(BaseModel):
    """Request body for target override."""
    new_target: str
    lock: bool = True
    reason: str = "User override"
    declared_task: Optional[str] = None  # G10: "text_classification"|"ner_sequence"|"seq2seq"


@app.post("/v2/datasets/{dataset_id}/override-target")
async def override_target_v2(
    dataset_id: str,
    request: TargetOverrideRequest,
    session_id: str = Query(...)
) -> Dict[str, Any]:
    """
    Override target selection for a dataset (Phase 4).
    
    Returns:
        {
            "dataset_id": str,
            "target_overridden": bool,
            "new_target": str,
            "locked": bool
        }
    """
    try:
        ctx = session_manager.get_session(session_id)
        if not ctx:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        with _session_lock:
            profile = context_db.load_profile(dataset_id)
            if not profile:
                raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
            if dataset_id not in ctx.active_dataset_ids:
                raise HTTPException(
                    status_code=400,
                    detail=f"Dataset {dataset_id} is not active in session {session_id}",
                )

            # G10/G11: per-modality validation before writing.
            _active_modalities = list(
                (getattr(ctx, "eligible_modalities", None) or [])
                or (getattr(ctx, "active_modalities", None) or [])
            )
            if "text" in _active_modalities and request.declared_task:
                try:
                    from data_ingestion.schema_detector import MultiDatasetSchemaDetector as _MSD
                    _cached_data = get_session_datasets(session_id)
                    _df_sample = None
                    for _lazy in _cached_data.values():
                        try:
                            _df_sample = _lazy.to_pandas().head(5_000) if hasattr(_lazy, "to_pandas") else _lazy
                            break
                        except Exception:
                            pass
                    if _df_sample is not None and request.new_target in _df_sample.columns:
                        _valid, _reason = _MSD._validate_text_target_override(
                            request.new_target,
                            _df_sample[request.new_target],
                            request.declared_task,
                        )
                        if not _valid:
                            raise HTTPException(
                                status_code=422,
                                detail=f"Text target validation failed: {_reason}",
                            )
                except HTTPException:
                    raise
                except Exception as _val_exc:
                    logger.warning("Text target validation skipped: %s", _val_exc)

            if "image" in _active_modalities:
                try:
                    from data_ingestion.schema_detector import MultiDatasetSchemaDetector as _MSD2
                    _cached_data2 = get_session_datasets(session_id)
                    _df_img = None
                    for _lazy2 in _cached_data2.values():
                        try:
                            _df_img = _lazy2.to_pandas().head(5_000) if hasattr(_lazy2, "to_pandas") else _lazy2
                            break
                        except Exception:
                            pass
                    if _df_img is not None:
                        _img_valid, _img_problem = _MSD2._check_image_label_validity(
                            _df_img, request.new_target
                        )
                        if not _img_valid:
                            profile["problem_type"] = _img_problem  # unsupervised_vision
                            profile["chosen_target"] = None
                            context_db.save_profile(profile, session_id)
                            raise HTTPException(
                                status_code=422,
                                detail=(
                                    f"Image target '{request.new_target}' has invalid label structure "
                                    f"(problem_type set to '{_img_problem}'). "
                                    "Provide a low-cardinality label column or remove the target for unsupervised learning."
                                ),
                            )
                except HTTPException:
                    raise
                except Exception as _img_exc:
                    logger.warning("Image target validation skipped: %s", _img_exc)

            profile["target_detected"] = True
            profile["chosen_target"] = request.new_target
            profile["target_locked"] = bool(request.lock)
            profile["target_override_reason"] = request.reason

            candidates = profile.get("target_candidates") or []
            candidates = [c for c in candidates if c.get("name") != request.new_target]
            candidates.insert(
                0,
                {
                    "name": request.new_target,
                    "score": 1.0,
                    "reason": f"User override: {request.reason}",
                },
            )
            profile["target_candidates"] = candidates

            user_overrides = profile.get("user_overrides") or {}
            user_overrides["target_override"] = {
                "value": request.new_target,
                "locked": bool(request.lock),
                "reason": request.reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            profile["user_overrides"] = user_overrides

            context_db.save_profile(profile, session_id)
            ctx.log_decision(
                "target_override",
                f"Target overridden for dataset {dataset_id}: {request.new_target}",
                request.reason,
            )

            # ── Intelligence invalidation on target change ─────────────────
            # The target column determines problem_type, interaction scores, and
            # predictability estimates. When the target changes these become stale.
            # Clear the per-dataset intelligence so Phase 3/4 recompute from the
            # new target rather than using regression scores for a classification
            # task (or vice versa).
            try:
                _old_fi = dict(ctx.feature_intelligence or {})
                if dataset_id in _old_fi:
                    _old_fi.pop(dataset_id)
                    ctx.feature_intelligence = _old_fi

                # Infer new problem_type from the overridden target's column stats
                _cached = get_session_datasets(session_id)
                for _lazy in _cached.values():
                    try:
                        _df_snap = _lazy.to_pandas().head(2000) if hasattr(_lazy, "to_pandas") else _lazy
                        if request.new_target in _df_snap.columns:
                            _col = _df_snap[request.new_target].dropna()
                            _n_uniq = _col.nunique()
                            _dtype = str(_col.dtype)
                            if "float" in _dtype and _n_uniq > 20:
                                _new_prob = "regression"
                            elif _n_uniq == 2:
                                _new_prob = "classification_binary"
                            elif _n_uniq <= 20:
                                _new_prob = "classification_multiclass"
                            else:
                                _new_prob = "regression"
                            # Update context so Phase 4 uses the correct type
                            ctx.global_problem_type = _new_prob  # type: ignore[attr-defined]
                            # Invalidate old predictability scores (regression → classification gap)
                            ctx.predictability_scores = {}
                            ctx.log_decision(
                                "target_intelligence_reset",
                                f"feature_intelligence and predictability_scores invalidated "
                                f"after target override to '{request.new_target}' "
                                f"(new problem_type={_new_prob})",
                                "Prevents stale regression scores from biasing classification model selection.",
                            )
                    except Exception:
                        pass
                    break
            except Exception as _inv_exc:
                logger.warning("target_override: intelligence invalidation failed: %s", _inv_exc)

            session_manager.update_session_context(session_id, ctx)

        return {
            "dataset_id": dataset_id,
            "target_overridden": True,
            "new_target": request.new_target,
            "locked": request.lock,
            "intelligence_reset": True,
        }

    except HTTPException:
        raise
    except OptimisticLockError:
        raise
    except Exception as exc:
        logger.error("/v2/datasets/{dataset_id}/override-target error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/v2/datasets/{dataset_id}/lock-target")
async def lock_target_v2(dataset_id: str, session_id: str = Query(...)) -> Dict[str, Any]:
    """Lock target to prevent automatic changes (Phase 4)."""
    try:
        ctx = session_manager.get_session(session_id)
        if not ctx:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        with _session_lock:
            profile = context_db.load_profile(dataset_id)
            if not profile:
                raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
            if dataset_id not in ctx.active_dataset_ids:
                raise HTTPException(
                    status_code=400,
                    detail=f"Dataset {dataset_id} is not active in session {session_id}",
                )
            profile["target_locked"] = True
            context_db.save_profile(profile, session_id)

        return {"dataset_id": dataset_id, "locked": True}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("/v2/datasets/{dataset_id}/lock-target error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/v2/datasets/{dataset_id}/unlock-target")
async def unlock_target_v2(dataset_id: str, session_id: str = Query(...)) -> Dict[str, Any]:
    """Unlock target to allow re-detection (Phase 4)."""
    try:
        ctx = session_manager.get_session(session_id)
        if not ctx:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        with _session_lock:
            profile = context_db.load_profile(dataset_id)
            if not profile:
                raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
            if dataset_id not in ctx.active_dataset_ids:
                raise HTTPException(
                    status_code=400,
                    detail=f"Dataset {dataset_id} is not active in session {session_id}",
                )
            profile["target_locked"] = False
            context_db.save_profile(profile, session_id)

        return {"dataset_id": dataset_id, "unlocked": True}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("/v2/datasets/{dataset_id}/unlock-target error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# V2 Global Schema/Target Endpoints (Phase 5)
# ---------------------------------------------------------------------------

# Global aggregation service removed - using orchestrator directly


@app.get("/v2/sessions/{session_id}/global-schema")
async def get_global_schema_v2(session_id: str) -> Dict[str, Any]:
    """
    Get global schema for a session (Phase 5).
    
    Returns:
        {
            "session_id": str,
            "global_schema": {...} | null,
            "confidence": float,
            "datasets_compatible": bool,
            "compatibility_matrix": {...}
        }
    """
    try:
        ctx = session_manager.get_session(session_id)
        if not ctx:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        return {
            "session_id": session_id,
            "global_schema": ctx.global_schema,
            "confidence": ctx.global_schema_confidence,
            "datasets_compatible": ctx.datasets_compatible,
            "compatibility_matrix": ctx.compatibility_matrix
        }
    
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("/v2/sessions/{session_id}/global-schema error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/v2/sessions/{session_id}/global-target")
async def get_global_target_v2(session_id: str) -> Dict[str, Any]:
    """
    Get global target for a session (Phase 5).
    
    Returns:
        {
            "session_id": str,
            "global_target": str | null,
            "confidence": float,
            "candidates": [{name, score, reason}, ...]
        }
    """
    try:
        ctx = session_manager.get_session(session_id)
        if not ctx:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        ranked_scores = []
        for cand in ctx.global_target_candidates:
            if not isinstance(cand, dict):
                continue
            raw_score = cand.get("score", cand.get("final_score", 0.0))
            if isinstance(raw_score, (int, float)):
                ranked_scores.append(float(raw_score))
        ranked_scores.sort(reverse=True)
        xs3_confidence_gap = (
            ranked_scores[0] - ranked_scores[1]
            if len(ranked_scores) > 1 else
            (ranked_scores[0] if ranked_scores else 0.0)
        )
        
        return {
            "session_id": session_id,
            "global_target": ctx.global_target,
            "confidence": ctx.global_target_confidence,
            "candidates": ctx.global_target_candidates,
            "xs3_confidence_gap": float(max(0.0, xs3_confidence_gap)),
        }
    
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("/v2/sessions/{session_id}/global-target error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


class GlobalTargetOverrideRequest(BaseModel):
    """Request body for global target override."""
    new_target: str
    reason: str = "User override"


@app.post("/v2/sessions/{session_id}/override-global-target")
async def override_global_target_v2(
    session_id: str,
    request: GlobalTargetOverrideRequest
) -> Dict[str, Any]:
    """Override global target (Phase 5)."""
    try:
        ctx = session_manager.get_session(session_id)
        if not ctx:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        with _session_lock:
            ctx.override_global_target(request.new_target, request.reason)
            session_manager.update_session_context(session_id, ctx)
        
        return {
            "session_id": session_id,
            "global_target": request.new_target,
            "overridden": True
        }
    
    except HTTPException:
        raise
    except OptimisticLockError:
        raise
    except Exception as exc:
        logger.error("/v2/sessions/{session_id}/override-global-target error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


class PrimaryDatasetRequest(BaseModel):
    """Request body for choosing primary dataset."""
    dataset_id: str
    reason: str = "User selection for incompatible datasets"


@app.post("/v2/sessions/{session_id}/choose-primary-dataset")
async def choose_primary_dataset_v2(
    session_id: str,
    request: PrimaryDatasetRequest
) -> Dict[str, Any]:
    """Choose primary dataset when datasets are incompatible (Phase 5)."""
    try:
        ctx = session_manager.get_session(session_id)
        if not ctx:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        if request.dataset_id not in ctx.active_dataset_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset {request.dataset_id} is not active in session {session_id}",
            )

        with _session_lock:
            ctx.primary_dataset_id = request.dataset_id
            ctx.datasets_compatible = False
            ctx.log_decision(
                "primary_dataset",
                f"Primary dataset set to {request.dataset_id}",
                request.reason,
            )
            session_manager.update_session_context(session_id, ctx)
        
        return {
            "session_id": session_id,
            "primary_dataset_id": request.dataset_id,
            "reason": request.reason
        }
    
    except HTTPException:
        raise
    except OptimisticLockError:
        raise
    except Exception as exc:
        logger.error("/v2/sessions/{session_id}/choose-primary-dataset error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))



# ---------------------------------------------------------------------------
# G13: Per-modality target override endpoint
# ---------------------------------------------------------------------------

class PerModalityTargetOverrideRequest(BaseModel):
    modality: str          # "text" | "image" | "tabular"
    target_column: str
    task_type: Optional[str] = None   # e.g. "text_classification", "ner_sequence"
    reason: str = ""


@app.post("/v2/sessions/{session_id}/override-target-per-modality")
async def override_target_per_modality(
    session_id: str, request: PerModalityTargetOverrideRequest
):
    """
    G13: Override the target column for a specific modality within a session.

    Stores in ``ctx.per_modality_target_override[modality] = target_column``
    and runs the appropriate per-modality validator (G10 for text, G11 for image).
    """
    try:
        _require_session(session_id)
        ctx = _load_session_required(session_id)

        modality = str(request.modality or "").strip().lower()
        if modality not in {"text", "image", "tabular"}:
            raise HTTPException(
                status_code=422,
                detail=f"modality must be one of text|image|tabular, got '{modality}'"
            )

        target_col = str(request.target_column or "").strip()
        if not target_col:
            raise HTTPException(status_code=422, detail="target_column must be non-empty")

        validation_result: Dict[str, Any] = {"valid": True, "reason": "no validation performed"}

        # G10: text target validation
        if modality == "text" and request.task_type:
            try:
                from data_ingestion.schema_detector import COGMASchemaDetector
                datasets = get_session_datasets(session_id)
                for ds in datasets.values():
                    df = ds.get("df") if isinstance(ds, dict) else None
                    if df is not None and target_col in df.columns:
                        valid, reason = COGMASchemaDetector._validate_text_target_override(
                            target_col, df[target_col], request.task_type
                        )
                        validation_result = {"valid": valid, "reason": reason}
                        if not valid:
                            raise HTTPException(status_code=422, detail=f"Text target validation: {reason}")
                        break
            except HTTPException:
                raise
            except Exception as val_exc:
                logger.warning("G10 text validation error (non-fatal): %s", val_exc)

        # G11: image target validation
        if modality == "image":
            try:
                from data_ingestion.schema_detector import COGMASchemaDetector
                datasets = get_session_datasets(session_id)
                for ds in datasets.values():
                    df = ds.get("df") if isinstance(ds, dict) else None
                    if df is not None:
                        valid, fallback_pt = COGMASchemaDetector._check_image_label_validity(df, target_col)
                        validation_result = {"valid": valid, "reason": fallback_pt}
                        if not valid:
                            raise HTTPException(
                                status_code=422,
                                detail=f"Image label validation failed: {fallback_pt}. Use unsupervised mode instead."
                            )
                        break
            except HTTPException:
                raise
            except Exception as val_exc:
                logger.warning("G11 image validation error (non-fatal): %s", val_exc)

        if not hasattr(ctx, "per_modality_target_override"):
            ctx.per_modality_target_override = {}
        ctx.per_modality_target_override[modality] = target_col
        ctx.user_overrides[f"target_{modality}"] = target_col
        ctx.log_decision(
            "per_modality_target_override",
            f"G13: {modality} target overridden to '{target_col}'",
            evidence=request.reason or f"task_type={request.task_type}",
        )
        session_manager.update_session_context(session_id, ctx)

        return {
            "session_id": session_id,
            "modality": modality,
            "target_column": target_col,
            "task_type": request.task_type,
            "validation": validation_result,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("/v2/sessions/%s/override-target-per-modality: %s", session_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# G12: Fusion strategy override endpoint
# ---------------------------------------------------------------------------

class FusionOverrideRequest(BaseModel):
    strategy: str
    reason: str = "User override"


_VALID_FUSION_STRATEGIES = {
    "concatenation", "attention", "graph",
    "uncertainty", "uncertainty_graph",
    "structural_semantic", "complementarity",
    # New strategies added in ULA/GatedFusion/FuseMoE upgrade
    "gated", "gated_fusion",
    "ula", "unified_latent", "unified_latent_alignment", "omnimodal",
    "fusemoe", "moe", "mixture_of_experts",
}


@app.post("/v2/sessions/{session_id}/override-fusion")
async def override_fusion_strategy(session_id: str, request: FusionOverrideRequest) -> Dict[str, Any]:
    """G12: Override fusion strategy for the session."""
    try:
        ctx = session_manager.get_session(session_id)
        if not ctx:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        strategy = str(request.strategy or "").strip().lower()
        if strategy not in _VALID_FUSION_STRATEGIES:
            raise HTTPException(
                status_code=422,
                detail=f"strategy must be one of {sorted(_VALID_FUSION_STRATEGIES)}, got '{strategy}'",
            )
        with _session_lock:
            ctx.override_fusion_strategy(strategy, request.reason)
            session_manager.update_session_context(session_id, ctx)
        return {
            "session_id": session_id,
            "fusion_strategy": strategy,
            "reason": request.reason,
            "applied": True,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("/v2/sessions/%s/override-fusion: %s", session_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Encoder override — lets the frontend pin specific encoders per modality
# ---------------------------------------------------------------------------

class EncoderOverridesRequest(BaseModel):
    preferred_image_encoder: Optional[str] = None
    preferred_text_encoder: Optional[str] = None
    preferred_tabular_encoder: Optional[str] = None
    reason: str = "UI encoder override"


@app.post("/v2/sessions/{session_id}/encoder-overrides")
async def set_encoder_overrides(
    session_id: str, request: EncoderOverridesRequest
) -> Dict[str, Any]:
    """
    Store preferred encoder names for the next model-selection / training run.

    Writes to ``ctx.encoder_overrides`` (a simple dict) so Phase 4 / Phase 5
    can read them when calling ``JITEncoderSelector.select()``.  Does NOT
    immediately trigger re-selection; the overrides take effect on the next
    ``/select-model`` or ``/train-pipeline`` call.
    """
    try:
        ctx = session_manager.get_session(session_id)
        if ctx is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

        overrides: Dict[str, Any] = {}
        if request.preferred_image_encoder:
            overrides["preferred_image_encoder"] = str(request.preferred_image_encoder).strip()
        if request.preferred_text_encoder:
            overrides["preferred_text_encoder"] = str(request.preferred_text_encoder).strip()
        if request.preferred_tabular_encoder:
            overrides["preferred_tabular_encoder"] = str(request.preferred_tabular_encoder).strip()

        # Store inside encoder_plan which IS a proper ExecutionContext dict field
        existing_plan = dict(getattr(ctx, "encoder_plan", {}) or {})
        existing_enc_ovr = dict(existing_plan.get("_encoder_overrides", {}) or {})
        existing_enc_ovr.update(overrides)
        existing_plan["_encoder_overrides"] = existing_enc_ovr
        existing = existing_enc_ovr  # what we return to caller

        try:
            ctx.encoder_plan = existing_plan
        except Exception:
            try:
                object.__setattr__(ctx, "encoder_plan", existing_plan)
            except Exception:
                pass

        ctx.log_decision(
            "encoder_override",
            f"Encoder overrides set: {overrides}",
            evidence=f"reason={request.reason}",
        )
        session_manager.update_session_context(session_id, ctx)

        logger.info("/v2/sessions/%s/encoder-overrides: %s", session_id, overrides)
        return {
            "status": "ok",
            "encoder_overrides": existing,
            "message": (
                "Encoder overrides stored. They will be applied on the next "
                "model selection or training run."
            ),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("/v2/sessions/%s/encoder-overrides: %s", session_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# G22: Active-model setter and registered-models lister
# ---------------------------------------------------------------------------

class SetActiveModelRequest(BaseModel):
    model_id: str
    reason: str = ""


@app.post("/v2/sessions/{session_id}/active-model")
async def set_active_prediction_model(
    session_id: str, request: SetActiveModelRequest
):
    """
    G22: Set the active model for the prediction playground in a session.
    Writes to ``ctx.active_prediction_model_id`` and logs the decision.
    """
    try:
        _require_session(session_id)
        ctx = _load_session_required(session_id)

        model_id = str(request.model_id or "").strip()
        if not model_id or not _SAFE_MODEL_ID.match(model_id):
            raise HTTPException(status_code=422, detail=f"Invalid model_id: '{model_id}'")

        # Validate that model is registered in session
        registered = list(getattr(ctx, "registered_model_ids", []) or [])
        if registered and model_id not in registered:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_id}' not found in session registered_model_ids: {registered}"
            )

        ctx.active_prediction_model_id = model_id
        ctx.log_decision(
            "active_model",
            f"G22: active prediction model set to '{model_id}'",
            evidence=request.reason,
        )
        session_manager.update_session_context(session_id, ctx)

        return {
            "session_id": session_id,
            "active_prediction_model_id": model_id,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("/v2/sessions/%s/active-model: %s", session_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/v2/sessions/{session_id}/registered-models")
async def list_registered_models(session_id: str):
    """
    G22: List all models registered in a session, with the active model marked.
    """
    try:
        _require_session(session_id)
        ctx = _load_session_required(session_id)

        registered = list(getattr(ctx, "registered_model_ids", []) or [])
        active = getattr(ctx, "active_prediction_model_id", None)

        # Enrich with any registry metadata available
        enriched = []
        try:
            from modelss.model_registry import ModelRegistry
            reg = ModelRegistry()
            for mid in registered:
                try:
                    meta = reg.get_model(mid) or {}
                    enriched.append({
                        "model_id": mid,
                        "active": (mid == active),
                        "metadata": meta,
                    })
                except Exception:
                    enriched.append({"model_id": mid, "active": (mid == active), "metadata": {}})
        except Exception:
            enriched = [
                {"model_id": mid, "active": (mid == active), "metadata": {}}
                for mid in registered
            ]

        return {
            "session_id": session_id,
            "registered_models": enriched,
            "active_model_id": active,
            "count": len(enriched),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("/v2/sessions/%s/registered-models: %s", session_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting APEX Framework API Server...")
    print(f"API: http://localhost:8001")
    print(f"GPU: {GPU_DEVICE if GPU_AVAILABLE else 'CPU (no GPU)'}")

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
