"""Thread-safe registry reader utilities for model metadata and stats."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List

from config.paths import MODEL_REGISTRY_DIR
from pipeline.evaluation import EvaluationAdapter


class ThreadSafeModelRegistry:
    """Read-only metadata registry with lock-guarded refresh and cached scans."""

    def __init__(self, registry_dir: Path | None = None, cache_ttl_seconds: float = 5.0) -> None:
        self.registry_dir = Path(registry_dir or MODEL_REGISTRY_DIR)
        self.cache_ttl_seconds = float(max(0.0, cache_ttl_seconds))
        self._lock = RLock()
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ts = 0.0
        self._evaluator = EvaluationAdapter()

    def _refresh(self) -> None:
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        updated: Dict[str, Dict[str, Any]] = {}

        for model_dir in sorted(self.registry_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            meta_file = model_dir / "metadata.json"
            if not meta_file.exists():
                continue
            try:
                with open(meta_file, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
            except Exception:
                continue
            model_id = str(meta.get("model_id") or model_dir.name)
            updated[model_id] = meta

        self._cache = updated
        self._cache_ts = time.time()

    def _ensure_cache(self, refresh: bool = False) -> None:
        with self._lock:
            stale = (time.time() - self._cache_ts) > self.cache_ttl_seconds
            if refresh or stale or not self._cache:
                self._refresh()

    def list_models(self, refresh: bool = False) -> List[Dict[str, Any]]:
        self._ensure_cache(refresh=refresh)
        with self._lock:
            models = []
            for model_id, meta in self._cache.items():
                models.append(
                    {
                        "model_id": model_id,
                        "created_at": meta.get("created_at"),
                        "status": meta.get("status", "unknown"),
                        "deployment_ready": bool(meta.get("deployment_ready", False)),
                    }
                )
            return models

    def get_model_metadata(self, model_id: str, refresh: bool = False) -> Dict[str, Any]:
        self._ensure_cache(refresh=refresh)
        with self._lock:
            if model_id not in self._cache:
                raise FileNotFoundError(f"Model '{model_id}' not found in registry")
            return deepcopy(self._cache[model_id])

    def get_model_stats(self, model_id: str, refresh: bool = False) -> Dict[str, Any]:
        meta = self.get_model_metadata(model_id, refresh=refresh)
        phases = meta.get("phases_summary", {}) or {}
        training = phases.get("TRAINING", {}) or {}
        drift = phases.get("DRIFT_DETECTION", {}) or {}

        artifact_versions: Dict[str, Any] = dict(meta.get("artifact_versions", {}) or {})
        training_signals: Dict[str, Any] = dict(meta.get("training_signals", {}) or {})
        training_fit_analysis: Dict[str, Any] = dict(meta.get("training_fit_analysis", {}) or {})
        xai_config: Dict[str, Any] = dict(meta.get("xai_config", {}) or {})
        fusion_payload: Dict[str, Any] = dict(meta.get("fusion", {}) or {})
        xai_payload: Dict[str, Any] = dict(meta.get("xai", {}) or {})

        fusion_attention_summary: Dict[str, Any] = {}
        if fusion_payload:
            fusion_attention_summary = {
                "summary": fusion_payload.get("summary", {}),
                "auxiliary_loss_weights": fusion_payload.get("auxiliary_loss_weights", {}),
                "alignment_summary": fusion_payload.get("alignment_summary", {}),
            }
        elif isinstance(xai_payload, dict):
            legacy_fusion_payload = xai_payload.get("fusion", {})
            if isinstance(legacy_fusion_payload, dict) and legacy_fusion_payload:
                fusion_attention_summary = {
                    "strategy": legacy_fusion_payload.get("strategy"),
                    "method": legacy_fusion_payload.get("method"),
                    "weights": legacy_fusion_payload.get("weights", {}),
                }

        evaluation = self._evaluator.evaluate_from_summary(training, drift, metadata=meta)

        return {
            "model_id": model_id,
            "created_at": meta.get("created_at"),
            "deployment_ready": bool(meta.get("deployment_ready", False)),
            "status": meta.get("status", "unknown"),
            "artifact_count": len(meta.get("artifact_paths", {}) or {}),
            "training": training,
            "drift": drift,
            "research_metrics": meta.get("research_metrics", {}),
            "fusion_attention_summary": fusion_attention_summary,
            "artifact_versions": artifact_versions,
            "training_signals": training_signals,
            "training_fit_analysis": training_fit_analysis,
            "xai_config": xai_config,
            "fusion_summary": fusion_payload,
            "xai": xai_payload,
            "evaluation": evaluation,
        }
