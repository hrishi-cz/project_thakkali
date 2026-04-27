"""Disk-backed embedding cache for frozen encoder outputs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Dict, Optional

import torch

from config.paths import DATA_DIR


class EmbeddingCache:
    """Store and retrieve embedding tensors using deterministic cache keys."""

    _CACHE_SKIP_THRESHOLD: float = 0.20

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        modality_priorities: Optional[Dict[str, float]] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir or (DATA_DIR / "embedding_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._hits = 0
        self._misses = 0
        self._writes = 0
        self._skipped_writes = 0
        self._modality_priorities: Dict[str, float] = dict(modality_priorities or {})

    @staticmethod
    def build_key(parts: Dict[str, Any]) -> str:
        payload = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _path_for_key(self, key: str) -> Path:
        safe = hashlib.sha1(str(key).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{safe}.pt"

    def get(self, key: str) -> Optional[torch.Tensor]:
        path = self._path_for_key(key)
        with self._lock:
            if not path.exists():
                self._misses += 1
                return None
            try:
                payload = torch.load(path, map_location="cpu")
                tensor = payload.get("tensor") if isinstance(payload, dict) else payload
                if isinstance(tensor, torch.Tensor):
                    self._hits += 1
                    return tensor
            except Exception:
                pass
            self._misses += 1
            return None

    def set(self, key: str, tensor: torch.Tensor, meta: Optional[Dict[str, Any]] = None) -> None:
        if not self._should_cache(key, meta=meta):
            with self._lock:
                self._skipped_writes += 1
            return

        path = self._path_for_key(key)
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "meta": dict(meta or {}),
            "tensor": tensor.detach().cpu(),
        }
        with self._lock:
            torch.save(payload, path)
            self._writes += 1

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], torch.Tensor],
        meta: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        cached = self.get(key)
        if cached is not None:
            return cached
        value = compute_fn()
        self.set(key, value, meta=meta)
        return value

    def set_modality_priorities(self, priorities: Dict[str, float]) -> None:
        """Update modality caching priorities from ExecutionContext signals."""
        with self._lock:
            self._modality_priorities = {
                str(k): float(v)
                for k, v in dict(priorities or {}).items()
            }

    def _should_cache(self, key: str, meta: Optional[Dict[str, Any]] = None) -> bool:
        if not self._modality_priorities:
            return True

        modality_hint = ""
        if isinstance(meta, dict):
            modality_hint = str(meta.get("modality", "")).lower().strip()

        if modality_hint:
            for modality, priority in self._modality_priorities.items():
                if modality.lower() in modality_hint and float(priority) < self._CACHE_SKIP_THRESHOLD:
                    return False
            return True

        key_lower = str(key).lower()
        for modality, priority in self._modality_priorities.items():
            if modality.lower() in key_lower and float(priority) < self._CACHE_SKIP_THRESHOLD:
                return False
        return True

    def evict_weak_modality(self, modality: str) -> int:
        """Delete cached entries for a modality after severe drift/low predictability."""
        removed = 0
        modality_lower = str(modality).lower()
        with self._lock:
            for path in self.cache_dir.glob("*.pt"):
                try:
                    payload = torch.load(path, map_location="cpu")
                    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
                    if modality_lower in str(meta.get("modality", "")).lower():
                        path.unlink(missing_ok=True)
                        removed += 1
                except Exception:
                    continue
        return removed

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "writes": self._writes,
                "skipped_writes": self._skipped_writes,
            }
