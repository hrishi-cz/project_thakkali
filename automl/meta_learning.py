from __future__ import annotations

import json
import logging
import hashlib
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List

from config.paths import META_LEARNING_STORE

logger = logging.getLogger(__name__)


class MetaLearningStore:
    """
    Persist model-selection experience across runs in a local JSON file.

    Record format:
    {
      "dataset_meta": {
        "num_rows": int,
        "num_cols": int,
        "modalities": [...],
        "target_type": "classification" | "regression"
      },
      "best_params": dict,
      "fusion_strategy": str,
      "loss_weights": dict,
      "performance": float
    }
    """

    def __init__(self, storage_path: str = str(META_LEARNING_STORE)) -> None:
        self.storage_path = Path(storage_path)
        self._lock = Lock()

    def load(self) -> List[Dict[str, Any]]:
        with self._lock:
            if not self.storage_path.exists():
                return []
            try:
                with open(self.storage_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    return data
                logger.warning("MetaLearningStore: invalid JSON root, expected list")
                return []
            except Exception as exc:
                logger.warning("MetaLearningStore.load failed: %s", exc)
                return []

    def save(self, records: List[Dict[str, Any]]) -> None:
        with self._lock:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as fh:
                json.dump(records, fh, indent=2)

    def add_experiment(self, record: Dict[str, Any]) -> None:
        required = {
            "dataset_meta",
            "best_params",
            "fusion_strategy",
            "loss_weights",
            "performance",
        }
        missing = required - set(record.keys())
        if missing:
            raise ValueError(f"MetaLearningStore.add_experiment missing keys: {sorted(missing)}")

        # Hold lock across the full read-modify-write so concurrent callers
        # don't interleave their load/save calls and lose each other's records.
        with self._lock:
            if not self.storage_path.exists():
                records: List[Dict[str, Any]] = []
            else:
                try:
                    with open(self.storage_path, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    records = data if isinstance(data, list) else []
                except Exception as exc:
                    logger.warning("MetaLearningStore.add_experiment load failed: %s", exc)
                    records = []

            records.append(record)

            # Keep file bounded for fast lookup.
            if len(records) > 500:
                records = records[-500:]

            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as fh:
                json.dump(records, fh, indent=2)

    def get_similar_context(
        self,
        dataset_meta: Dict[str, Any],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        records = self.load()
        if not records:
            return []

        scored: List[Dict[str, Any]] = []
        for rec in records:
            rec_meta = rec.get("dataset_meta", {})
            score = self._similarity_score(dataset_meta, rec_meta)
            if score <= 0.0:
                continue
            scored.append({**rec, "similarity": round(score, 4)})

        scored.sort(key=lambda item: item.get("similarity", 0.0), reverse=True)
        return scored[:top_k]

    def _similarity_score(self, query: Dict[str, Any], rec: Dict[str, Any]) -> float:
        q_target = str(query.get("target_type", "")).lower()
        r_target = str(rec.get("target_type", "")).lower()
        if not q_target or q_target != r_target:
            return 0.0

        q_cols = max(1, int(query.get("num_cols", 1)))
        r_cols = max(1, int(rec.get("num_cols", 1)))
        col_distance = abs(q_cols - r_cols)
        col_score = max(0.0, 1.0 - (col_distance / max(q_cols, r_cols)))

        q_mods = set(query.get("modalities", []) or [])
        r_mods = set(rec.get("modalities", []) or [])
        if not q_mods or not r_mods:
            modality_score = 0.0
        else:
            modality_score = len(q_mods & r_mods) / max(1, len(q_mods | r_mods))

        # target_type is a hard gate; combine column and modality similarity.
        return 0.55 * col_score + 0.45 * modality_score

    def build_fingerprint(self, dataset_meta: Dict[str, Any]) -> str:
        """Create a stable fingerprint for dataset meta used by recommendation APIs."""
        normalized = {
            "num_rows": int(dataset_meta.get("num_rows", 0)),
            "num_cols": int(dataset_meta.get("num_cols", 0)),
            "modalities": sorted(list(dataset_meta.get("modalities", []) or [])),
            "target_type": str(dataset_meta.get("target_type", "classification")).lower(),
        }
        payload = json.dumps(normalized, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def suggest(
        self,
        dataset_meta: Dict[str, Any],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return top similar historical contexts as direct suggestions."""
        similar = self.get_similar_context(dataset_meta=dataset_meta, top_k=top_k)
        suggestions: List[Dict[str, Any]] = []
        for rec in similar:
            suggestions.append(
                {
                    "fingerprint": self.build_fingerprint(rec.get("dataset_meta", {})),
                    "similarity": float(rec.get("similarity", 0.0)),
                    "performance": float(rec.get("performance", 0.0)),
                    "fusion_strategy": rec.get("fusion_strategy"),
                    "best_params": dict(rec.get("best_params", {})),
                    "loss_weights": dict(rec.get("loss_weights", {})),
                }
            )
        return suggestions

    def predict_best_config(
        self,
        dataset_meta: Dict[str, Any],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Predict a single recommended configuration using weighted similar runs.

        The output is intentionally conservative: when no historical data matches,
        it returns a default-friendly skeleton instead of making up values.
        """
        suggestions = self.suggest(dataset_meta=dataset_meta, top_k=top_k)
        if not suggestions:
            return {
                "fingerprint": self.build_fingerprint(dataset_meta),
                "confidence": 0.0,
                "fusion_strategy": "concatenation",
                "best_params": {},
                "loss_weights": {},
                "source_count": 0,
            }

        total_weight = 0.0
        fusion_votes: Dict[str, float] = {}
        numeric_params: Dict[str, float] = {}
        numeric_counts: Dict[str, float] = {}
        loss_weights: Dict[str, float] = {}
        loss_counts: Dict[str, float] = {}

        for item in suggestions:
            similarity = float(item.get("similarity", 0.0))
            performance = float(item.get("performance", 0.0))
            weight = max(1e-6, similarity) * max(1e-6, performance if performance > 0 else 0.1)
            total_weight += weight

            fusion = item.get("fusion_strategy")
            if isinstance(fusion, str) and fusion:
                fusion_votes[fusion] = fusion_votes.get(fusion, 0.0) + weight

            for k, v in item.get("best_params", {}).items():
                if isinstance(v, (int, float)):
                    numeric_params[k] = numeric_params.get(k, 0.0) + float(v) * weight
                    numeric_counts[k] = numeric_counts.get(k, 0.0) + weight

            for k, v in item.get("loss_weights", {}).items():
                if isinstance(v, (int, float)):
                    loss_weights[k] = loss_weights.get(k, 0.0) + float(v) * weight
                    loss_counts[k] = loss_counts.get(k, 0.0) + weight

        agg_params = {
            k: (numeric_params[k] / numeric_counts[k])
            for k in numeric_params
            if numeric_counts.get(k, 0.0) > 0
        }
        agg_loss = {
            k: (loss_weights[k] / loss_counts[k])
            for k in loss_weights
            if loss_counts.get(k, 0.0) > 0
        }

        best_fusion = "concatenation"
        if fusion_votes:
            best_fusion = max(fusion_votes.items(), key=lambda x: x[1])[0]

        confidence = 0.0
        if suggestions:
            confidence = float(sum(float(s.get("similarity", 0.0)) for s in suggestions) / len(suggestions))

        return {
            "fingerprint": self.build_fingerprint(dataset_meta),
            "confidence": round(confidence, 4),
            "fusion_strategy": best_fusion,
            "best_params": agg_params,
            "loss_weights": agg_loss,
            "source_count": len(suggestions),
        }
