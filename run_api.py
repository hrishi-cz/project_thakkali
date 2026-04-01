"""APEX Framework API – production-grade FastAPI entrypoint."""

import asyncio
import json
import collections
import logging
import sys
import threading
import uuid
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import pandas as pd
import torch
import os
import re as _re

from task_store import task_db, IngestionProgressTracker, TrainingProgressTracker

# Load user-registered encoder plugins (safe import -- file may be empty)
try:
    import config.encoder_plugins  # noqa: F401
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
apex_dir = Path(__file__).parent
sys.path.insert(0, str(apex_dir))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------
app = FastAPI(title="APEX Framework API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------
GPU_AVAILABLE: bool = torch.cuda.is_available()
GPU_DEVICE: str = torch.cuda.get_device_name(0) if GPU_AVAILABLE else "CPU"

# ---------------------------------------------------------------------------
# Input sanitization
# ---------------------------------------------------------------------------
_SAFE_MODEL_ID = _re.compile(r"^[\w\-.:]+$")

# ---------------------------------------------------------------------------
# Session dataset loader (FIX: remove duplication)
# ---------------------------------------------------------------------------

def get_session_datasets(session_id: str) -> Dict[str, Any]:
    """Load lazy datasets for a session (single source of truth)."""
    from data_ingestion.loader import DataLoader

    with _session_lock:
        session = _session_store.get(session_id)
        if not session or "datasets" not in session:
            return {}
        session_hashes = session["datasets"]

    cache_dir = Path("./data/dataset_cache")
    loader = DataLoader()

    datasets = {}

    for source, hash_id in session_hashes.items():
        cache_path = cache_dir / hash_id
        lazy_ref = loader.load_cached(cache_path)

        if lazy_ref is not None:
            datasets[hash_id] = lazy_ref
        else:
            logger.warning("Cache miss for hash %s", hash_id)

    return datasets

def _sanitize_model_id(model_id: str) -> str:
    """Validate model_id to prevent directory traversal attacks."""
    if not _SAFE_MODEL_ID.match(model_id) or ".." in model_id:
        raise HTTPException(
            status_code=400,
            detail="Invalid model_id: contains disallowed characters.",
        )
    return model_id


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
    """Return ingested hashes for a session, with backward-compatible fallback.

    If *session_id* is provided and present in ``_session_store``, that
    per-session dict is returned (correct under concurrency).  Otherwise
    the legacy global ``session_ingested_hashes`` is returned.

    Must be called under ``_session_lock``.
    """
    if session_id and session_id in _session_store:
        return dict(_session_store[session_id])
    # Backward-compatible: use legacy global alias
    return dict(session_ingested_hashes)

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


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

# Maps session_id -> {dataset_id -> metadata} for concurrent session isolation.
# Each session_id is independent; concurrent requests don't interfere.
# session_id -> {
#   "datasets": {hash: metadata},
#   "schema": Optional[dict]
# }
_session_store: Dict[str, Dict[str, Any]] = {}
_session_lock = threading.Lock()

# Backward-compatible alias (used in /train-pipeline and other endpoints)
session_ingested_hashes: Dict[str, Dict[str, Any]] = {}

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
        "message": "APEX Framework API",
        "version": "2.0.0",
        "status": "running",
        "gpu_available": GPU_AVAILABLE,
        "device": GPU_DEVICE,
    }


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "service": "APEX API",
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
    global session_ingested_hashes

    session_id: str = request.session_id
    dataset_urls: List[str] = request.dataset_urls

    logger.info("[%s] Ingestion request for %d dataset(s)", session_id, len(dataset_urls))

    task_id = uuid.uuid4().hex[:8]
    tracker = IngestionProgressTracker(task_id, dataset_urls, task_db)

    # Per-session isolation: use setdefault so re-ingesting into the same
    # session APPENDS new hashes instead of wiping previously ingested datasets.
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

            from data_ingestion.ingestion_manager import DataIngestionManager
            manager = DataIngestionManager()

            for idx, source_url in enumerate(dataset_urls):
                tracker.set_progress(
                    5 + int((idx / max(len(dataset_urls), 1)) * 85),
                    f"Downloading dataset {idx + 1}/{len(dataset_urls)}: {source_url[:60]}...",
                )

                try:
                    lazy_datasets, ingest_meta = await manager.ingest_data(
                        [source_url], force_download=False
                    )

                    if lazy_datasets:
                        for source_hash, lazy_ref in lazy_datasets.items():
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

                            with _session_lock:
                                session_hashes[source_hash] = {
                                    "source_url": source_url,
                                    "hash": source_hash,
                                    "timestamp": ingest_meta["ingestion_time"],
                                }
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

            # Sync session-scoped hashes to global alias for backward compatibility
            with _session_lock:
                session_ingested_hashes.clear()
                session_ingested_hashes.update(session_hashes)

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

        # 🔥 store schema
        with _session_lock:
            _session_store[session_id]["schema"] = schema_dict

        return {
            "status": "success",
            "phase": "Schema Detection",
            "data": schema_dict,

            # 🔥 expose intelligence
            "candidates": [
                asdict(c) for c in detector.last_target_candidates
            ]
        }

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

    if not session_id or not new_target:
        raise HTTPException(400, "session_id and target required")

    with _session_lock:
        if session_id not in _session_store:
            raise HTTPException(400, "Session not found")

        _session_store[session_id]["target_override"] = new_target

    return {"status": "override_applied", "target": new_target}
@app.post("/preprocess")
async def preprocess_data(request: Request) -> Dict[str, Any]:
    """
    Run Phase 3 preprocessing on the current session's datasets.

    STRICT SESSION ISOLATION
    ------------------------
    Requires an active ingestion session (``session_ingested_hashes``).
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

        body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
        _sid = body.get("session_id") if isinstance(body, dict) else None

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
        cache_dir = Path("./data/dataset_cache")
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
            with _session_lock:
                session = _session_store.get(_sid, {})
                global_schema = session.get("schema")

            if not global_schema:
                raise HTTPException(400, "Run schema detection first")
            
            detected = global_schema.detected_columns
            modalities = global_schema.modality_presence
            
            # Sample frames for preprocessing
            MAX_SAMPLE = 10_000
            frames = []
            for lazy_ref in lazy_datasets.values():
                try:
                    frames.append(lazy_ref.head(MAX_SAMPLE))
                except Exception as e:
                    logger.warning("Failed to load sample from dataset: %s", e)

            total_samples = sum(len(f) for f in frames)
            full_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

            target_col = global_schema.primary_target
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

            if tabular_cols and "tabular" in modalities:
                tp = TabularPreprocessor()
                try:
                    tp.fit(feature_df[tabular_cols])
                    out_dim = tp.get_output_dim()

                    # --- Sample: raw vs transformed (first 3 rows) ---
                    raw_sample_df = feature_df[tabular_cols].head(3)
                    transformed_arr = tp.transform(raw_sample_df)

                    # Feature names from the ColumnTransformer
                    try:
                        feat_names = list(tp._transformer.get_feature_names_out())
                    except Exception:
                        feat_names = [f"f{i}" for i in range(transformed_arr.shape[1])]

                    samples["tabular"] = {
                        "raw_columns": list(raw_sample_df.columns),
                        "raw_rows": raw_sample_df.fillna("").astype(str).values.tolist(),
                        "transformed_columns": feat_names,
                        "transformed_rows": transformed_arr.tolist(),
                        "dropped_columns": list(getattr(tp, "_dropped_cols", [])),
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
                output_shapes["text"] = "(N, 128) per key"
                preprocessing_stages.append({
                    "stage": "text_preprocessing",
                    "status": "success",
                    "output_shape": output_shapes["text"],
                })

                # --- Sample: original text + tokenized ids (first row) ---
                try:
                    first_text = str(feature_df[text_cols[0]].dropna().iloc[0])
                    from preprocessing.text_preprocessor import TextPreprocessor
                    text_prep = TextPreprocessor()
                    tok_out = text_prep.preprocess(first_text)
                    samples["text"] = {
                        "column": text_cols[0],
                        "original": first_text[:500],
                        "input_ids": tok_out["input_ids"].tolist(),
                        "attention_mask": tok_out["attention_mask"].tolist(),
                        "tokenizer": "bert-base-uncased",
                        "max_length": 128,
                    }
                except Exception as txt_exc:
                    logger.warning("/preprocess text sample failed: %s", txt_exc)

            if image_cols and "image" in modalities:
                output_shapes["image"] = "(N, 3, 224, 224)"
                preprocessing_stages.append({
                    "stage": "image_preprocessing",
                    "status": "success",
                    "output_shape": output_shapes["image"],
                })

            if not preprocessing_stages and not feature_df.empty:
                output_shapes["tabular"] = f"(N, {len(feature_df.columns)})"
                preprocessing_stages.append({
                    "stage": "tabular_preprocessing",
                    "status": "success",
                    "output_shape": output_shapes["tabular"],
                })

            return {
                "status": "success",
                "phase": "Phase 3: Preprocessing",
                "data": {
                    "preprocessing_stages": preprocessing_stages,
                    "total_samples": total_samples,
                    "output_shapes": output_shapes,
                    "samples": samples,
                },
            }

        return await asyncio.to_thread(_preprocess_sync)

    except HTTPException:
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
        problem_type: str = body.get("problem_type", "unsupervised")
        modalities: List[str] = body.get("modalities", [])
        dataset_size: int = int(body.get("dataset_size") or 10_000)
        avg_tokens: int = int(body.get("avg_tokens") or 128)

        from automl.advanced_selector import AdvancedModelSelector

        selector = AdvancedModelSelector()
        recommendations: List[Dict[str, Any]] = await asyncio.to_thread(
            selector.recommend_models,
            problem_type=problem_type,
            modalities=modalities,
            dataset_size=dataset_size,
            avg_tokens=avg_tokens,
        )

        return {
            "status":             "success",
            "problem_type":       problem_type,
            "modalities":         modalities,
            "recommended_models": recommendations,
            "best_model":         recommendations[0] if recommendations else None,
        }

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
        modalities: List[str] = body.get("modalities", ["tabular"])
        hp_overrides: Optional[Dict[str, Any]] = body.get("hp_overrides")

        sources: List[str] = [
            meta.get("source_url", hid)
            for hid, meta in _snapshot.items()
        ]

        task_id = uuid.uuid4().hex[:8]
        tracker = TrainingProgressTracker(task_id, task_db)

        logger.info(
            "/train-pipeline: task=%s  problem=%s  modalities=%s  sources=%d  hp_overrides=%s",
            task_id, problem_type, modalities, len(sources), hp_overrides,
        )

        async def _run_training() -> None:
            try:
                orchestrator = TrainingOrchestrator(
                    TrainingConfig(
                        dataset_sources=sources,
                        problem_type=problem_type,
                        modalities=modalities,
                    )
                )

                # Phase 1 — Data Ingestion
                tracker.set_phase(1, "Data Ingestion", 5)
                tracker.add_message(1, "info", "Re-registering datasets from cache...")
                await orchestrator._execute_phase_1_data_ingestion(sources=sources)
                tracker.add_message(1, "result", f"Registered {len(sources)} dataset(s)")

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
                                            hp_overrides, tracker)
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

                # Phase 6 — Drift Detection
                tracker.set_phase(6, "Drift Detection", 96)
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
                        },
                        "deployment_ready": deployment_ready,
                    },
                }
                tracker.complete(final_result)

            except Exception as exc:
                logger.error("Training task %s failed: %s", task_id, exc, exc_info=True)
                tracker.fail(str(exc))

        # Launch as background asyncio task
        asyncio.create_task(_run_training())

        return {
            "status": "started",
            "task_id": task_id,
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
    return {
        "task_id":            task["task_id"],
        "status":             task["status"],
        "current_phase":      payload.get("current_phase", 0),
        "current_phase_name": payload.get("current_phase_name", "Initializing"),
        "progress_pct":       payload.get("progress_pct", 0),
        "messages":           payload.get("messages", []),
        "epoch_metrics":      payload.get("epoch_metrics", []),
        "trial_progress":     payload.get("trial_progress"),
        "data_split":         payload.get("data_split"),
        "result":             task.get("result"),
        "error":              task.get("error"),
    }


# ---------------------------------------------------------------------------
# Drift monitoring  (replaces the defunct /monitor-model stub)
# ---------------------------------------------------------------------------

@app.post("/monitor/drift")
async def monitor_drift(request: Request) -> Dict[str, Any]:
    """
    Run Phase 6 drift detection on the current session's datasets and
    optionally cross-reference a previously registered model's stored metrics.

    REQUEST BODY (all optional)
    ---------------------------
    ``model_id``    : str  – if provided, also loads drift metrics from
                             ``models/registry/{model_id}/metadata.json``
    ``problem_type``: str  – default ``"classification_binary"``
    ``modalities``  : list – default ``["tabular"]``

    FRONTEND CONTRACT
    -----------------
    Returns::

        {
          "status": "success",
          "data": {
            "drift_detected":  bool,
            "metrics": {
              "psi":          float,
              "ks_statistic": float,
              "fdd":          float
            },
            "thresholds": {
              "psi": 0.25,
              "ks_statistic": 0.30,
              "fdd": 0.50
            },
            "status_per_metric": {"psi": bool, "ks_statistic": bool, "fdd": bool},
            "n_reference":  int,
            "n_production": int,
            "n_features":   int,
            "model_id":     str | null
          }
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

        problem_type: str = body.get("problem_type", "classification_binary")
        modalities: List[str] = body.get("modalities", ["tabular"])
        model_id: Optional[str] = body.get("model_id")
        if model_id:
            model_id = _sanitize_model_id(model_id)

        sources: List[str] = [
            meta.get("source_url", hid)
            for hid, meta in _snapshot.items()
        ]

        orchestrator = TrainingOrchestrator(
            TrainingConfig(
                dataset_sources=sources,
                problem_type=problem_type,
                modalities=modalities,
            )
        )

        # Re-register from cache, then run Phase 2–3 to fit preprocessors
        await orchestrator._execute_phase_1_data_ingestion(sources=sources)
        await asyncio.to_thread(orchestrator._execute_phase_2_schema_detection)
        await asyncio.to_thread(orchestrator._execute_phase_3_preprocessing)

        # Phase 6 – real KS / PSI / MMD computation
        await asyncio.to_thread(orchestrator._execute_phase_6_drift_detection)

        p6: Dict[str, Any] = orchestrator.phase_results[Phase.DRIFT_DETECTION]

        # Optionally enrich with stored model metadata
        stored_drift: Dict[str, Any] = {}
        if model_id:
            meta_file = Path("models") / "registry" / model_id / "metadata.json"
            if meta_file.exists():
                try:
                    import json as _json
                    with open(meta_file, encoding="utf-8") as fh:
                        stored_meta = _json.load(fh)
                    phases_sum = stored_meta.get("phases_summary", {})
                    stored_drift = phases_sum.get("DRIFT_DETECTION", {})
                except Exception as load_exc:
                    logger.warning(
                        "/monitor/drift: could not load model metadata for '%s': %s",
                        model_id, load_exc,
                    )
            else:
                logger.warning(
                    "/monitor/drift: model_id '%s' not found at %s", model_id, meta_file
                )

        return {
            "status": "success",
            "data": {
                "drift_detected":   p6["drift_detected"],
                "metrics":          p6["metrics"],
                "thresholds":       p6["thresholds"],
                "status_per_metric": p6["status"],
                "n_reference":      p6["n_reference"],
                "n_production":     p6["n_production"],
                "n_features":       p6["n_features"],
                "model_id":         model_id,
                "stored_phase6_summary": stored_drift,
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

        registry_root = Path("models") / "registry"
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


@app.get("/model-info/{model_id}")
async def model_info(model_id: str) -> Dict[str, Any]:
    """Return class labels and expected feature columns for a registered model."""
    import json as _json

    model_id = _sanitize_model_id(model_id)
    registry_root = Path("models") / "registry" / model_id
    if not registry_root.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")

    def _load_model_info_sync() -> Dict[str, Any]:
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
        target_col = schema.get("primary_target", "")

        # Use effective_features (post-preprocessing) when available,
        # fall back to raw schema columns minus target
        input_tabular = (
            effective_features
            if effective_features
            else [c for c in tabular_cols if c != target_col]
        )

        return {
            "model_id": model_id,
            "problem_type": schema.get("global_problem_type", ""),
            "modalities": schema.get("global_modalities", []),
            "class_labels": class_labels,
            "target_column": target_col,
            "input_columns": {
                "tabular": input_tabular,
                "text": text_cols,
            },
            "effective_features": effective_features,
            "dropped_columns": dropped_columns,
        }

    return await asyncio.to_thread(_load_model_info_sync)


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
        result: Dict[str, Any] = engine.predict_batch(df)

        # XAI (optional)
        explanations: Optional[Dict[str, Any]] = None
        if explain:
            effective_target = _resolve_xai_target(target_class, result)

            explanations = engine.generate_explanations(
                df, target_class=effective_target, n_steps=n_steps,
            )

        payload = {
            "status":       "success",
            "predictions":  result["predictions"],
            "confidences":  result["confidences"],
            "problem_type": result["problem_type"],
            "n_samples":    result["n_samples"],
            "explanations": explanations,
        }

        task_db.update_result(task_id, "COMPLETED", payload)

    except Exception as exc:
        logger.error("Background inference task %s failed: %s", task_id, exc, exc_info=True)
        task_db.update_error(task_id, "FAILED", str(exc))


@app.post("/predict-async")
async def predict_async(request: Request, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Fire-and-return async inference.

    Immediately returns ``{"task_id": "<uuid>"}``; the frontend polls
    ``GET /task/{task_id}`` until status is COMPLETED or FAILED.
    """

    body: Dict[str, Any] = await request.json()

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

    explain: bool     = bool(body.get("explain", False))
    target_class: int = int(body.get("target_class", -1))
    n_steps: int      = int(body.get("n_steps", 50))

    task_id: str = str(uuid.uuid4())
    task_db.insert_task(
        task_id=task_id,
        task_type="inference",
        status="PENDING",
        payload={"model_id": model_id, "n_samples": len(raw_inputs)},
    )

    background_tasks.add_task(
        _run_inference_task,
        task_id, model_id, raw_inputs, explain, target_class, n_steps,
    )

    return {"task_id": task_id, "status": "PENDING"}


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
        raw_message = await websocket.receive_text()
        body: Dict[str, Any] = json.loads(raw_message)

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

        if len(raw_inputs) <= CHUNK_SIZE:
            result = await asyncio.to_thread(engine.predict_batch, df_full)
        else:
            all_predictions: List[Any] = []
            all_confidences: List[Any] = []
            n_chunks = (len(raw_inputs) + CHUNK_SIZE - 1) // CHUNK_SIZE
            problem_type = ""

            for chunk_idx in range(n_chunks):
                start = chunk_idx * CHUNK_SIZE
                end = min(start + CHUNK_SIZE, len(raw_inputs))
                chunk_df = df_full.iloc[start:end]

                chunk_result = await asyncio.to_thread(engine.predict_batch, chunk_df)
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

            explanations = await asyncio.to_thread(
                engine.generate_explanations,
                df_full,
                target_class=effective_target,
                n_steps=n_steps,
            )

        # 6. Send complete result
        payload = {
            "status": "success",
            "predictions": result["predictions"],
            "confidences": result["confidences"],
            "problem_type": result["problem_type"],
            "n_samples": result["n_samples"],
            "explanations": explanations,
        }
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
        result: Dict[str, Any] = await asyncio.to_thread(engine.predict_batch, df)

        # Captum XAI – gradients enabled only inside generate_explanations
        explanations: Optional[Dict[str, Any]] = None
        if explain:
            effective_target = _resolve_xai_target(target_class, result)

            explanations = await asyncio.to_thread(
                engine.generate_explanations,
                df,
                target_class=effective_target,
                n_steps=n_steps,
            )

        return {
            "status":       "success",
            "predictions":  result["predictions"],
            "confidences":  result["confidences"],
            "problem_type": result["problem_type"],
            "n_samples":    result["n_samples"],
            "explanations": explanations,
        }

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

from api.session_manager import session_manager, SessionContext

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
            "closed_at": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
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
    except Exception as exc:
        logger.error("/v2/sessions/{session_id}/datasets/{dataset_id} DELETE error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# V2 Schema Detection Endpoints (Phase 3)
# ---------------------------------------------------------------------------

from api.schema_detection_service import schema_detector
from api.execution_context import ExecutionContext

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
            ]
        }
    """
    try:
        from database.dataset_profile_db import dataset_profile_db
        
        ctx = session_manager.get_session(session_id)
        if not ctx:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        profiles = dataset_profile_db.get_session_profiles(session_id)
        
        datasets = []
        for profile in profiles:
            datasets.append({
                "dataset_id": profile['dataset_id'],
                "source_url": profile.get('source_url'),
                "status": "active" if profile['dataset_id'] in ctx.active_dataset_ids else "cached",
                "schema_detected": profile.get('schema_detected', False),
                "target_detected": profile.get('target_detected', False),
                "created_at": profile.get('created_at')
            })
        
        return {
            "session_id": session_id,
            "datasets": datasets
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
        from database.dataset_profile_db import dataset_profile_db
        
        profile = dataset_profile_db.get_profile(dataset_id)
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
        
        # Convert SessionContext to ExecutionContext
        exec_ctx = ExecutionContext(session_id=ctx.session_id)
        exec_ctx.active_dataset_ids = ctx.active_dataset_ids
        
        success = schema_detector.override_schema(
            dataset_id,
            exec_ctx,
            request.schema_override,
            request.reason
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to override schema")
        
        return {
            "dataset_id": dataset_id,
            "schema_overridden": True,
            "new_schema": request.schema_override
        }
    
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("/v2/datasets/{dataset_id}/override-schema error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# V2 Target Detection Endpoints (Phase 4)
# ---------------------------------------------------------------------------

from api.target_detection_service import target_detector


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
        from database.dataset_profile_db import dataset_profile_db
        
        profile = dataset_profile_db.get_profile(dataset_id)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
        
        return {
            "dataset_id": dataset_id,
            "candidates": profile.get('target_candidates', []),
            "chosen_target": profile.get('chosen_target'),
            "target_locked": profile.get('target_locked', False)
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
        
        exec_ctx = ExecutionContext(session_id=ctx.session_id)
        exec_ctx.active_dataset_ids = ctx.active_dataset_ids
        
        success = target_detector.override_target(
            dataset_id,
            exec_ctx,
            request.new_target,
            request.lock,
            request.reason
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to override target")
        
        return {
            "dataset_id": dataset_id,
            "target_overridden": True,
            "new_target": request.new_target,
            "locked": request.lock
        }
    
    except HTTPException:
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
        
        exec_ctx = ExecutionContext(session_id=ctx.session_id)
        success = target_detector.lock_target(dataset_id, exec_ctx)
        
        return {"dataset_id": dataset_id, "locked": success}
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
        
        exec_ctx = ExecutionContext(session_id=ctx.session_id)
        success = target_detector.unlock_target(dataset_id, exec_ctx)
        
        return {"dataset_id": dataset_id, "unlocked": success}
    except Exception as exc:
        logger.error("/v2/datasets/{dataset_id}/unlock-target error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# V2 Global Schema/Target Endpoints (Phase 5)
# ---------------------------------------------------------------------------

from api.global_aggregation_service import global_aggregation


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
        
        return {
            "session_id": session_id,
            "global_target": ctx.global_target,
            "confidence": ctx.global_target_confidence,
            "candidates": ctx.global_target_candidates
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
        
        exec_ctx = ExecutionContext(session_id=ctx.session_id)
        success = global_aggregation.override_global_target(
            exec_ctx,
            request.new_target,
            request.reason
        )
        
        # Update session context
        ctx.global_target = request.new_target
        session_manager.update_session_context(session_id, ctx)
        
        return {
            "session_id": session_id,
            "global_target": request.new_target,
            "overridden": success
        }
    
    except HTTPException:
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
        
        exec_ctx = ExecutionContext(session_id=ctx.session_id)
        exec_ctx.active_dataset_ids = ctx.active_dataset_ids
        
        success = global_aggregation.choose_primary_dataset(
            exec_ctx,
            request.dataset_id,
            request.reason
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to set primary dataset")
        
        # Update session context
        ctx.primary_dataset_id = request.dataset_id
        session_manager.update_session_context(session_id, ctx)
        
        return {
            "session_id": session_id,
            "primary_dataset_id": request.dataset_id,
            "reason": request.reason
        }
    
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("/v2/sessions/{session_id}/choose-primary-dataset error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    try:
        success = session_manager.remove_dataset_from_session(session_id, dataset_id)
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found or dataset {dataset_id} not in session"
            )
        
        return {
            "session_id": session_id,
            "dataset_id": dataset_id,
            "status": "removed",
            "cache_preserved": True
        }
    
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("/v2/sessions/{session_id}/datasets/{dataset_id} error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/v2/sessions/{session_id}/datasets")
async def list_session_datasets_v2(session_id: str) -> Dict[str, Any]:
    """
    List datasets for a session (Phase 2).
    
    Returns:
        {
            "active_datasets": [{"dataset_id": str, "source": str, "in_session": true}, ...],
            "cached_datasets": [{"dataset_id": str, "source": str, "in_session": false}, ...]
        }
    """
    try:
        ctx = session_manager.get_session(session_id)
        if not ctx:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        # Get cache metadata
        from data_ingestion.ingestion_manager import DataIngestionManager
        mgr = DataIngestionManager()
        cache_meta = mgr.cache_metadata
        
        # Build active datasets
        active_datasets = []
        for dataset_id in ctx.active_dataset_ids:
            meta = cache_meta.get(dataset_id, {})
            active_datasets.append({
                "dataset_id": dataset_id,
                "source": meta.get("source", "unknown"),
                "in_session": True,
                "cached_at": meta.get("timestamp", "unknown")
            })
        
        # Build cached datasets (not active)
        all_cached_ids = set(cache_meta.keys())
        inactive_ids = all_cached_ids - set(ctx.active_dataset_ids)
        
        cached_datasets = []
        for dataset_id in inactive_ids:
            meta = cache_meta.get(dataset_id, {})
            cached_datasets.append({
                "dataset_id": dataset_id,
                "source": meta.get("source", "unknown"),
                "in_session": False,
                "cached_at": meta.get("timestamp", "unknown")
            })
        
        return {
            "active_datasets": active_datasets,
            "cached_datasets": cached_datasets
        }
    
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("/v2/sessions/{session_id}/datasets error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting APEX Framework API Server...")
    print(f"API: http://localhost:8001")
    print(f"GPU: {GPU_DEVICE if GPU_AVAILABLE else 'CPU (no GPU)'}")

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
