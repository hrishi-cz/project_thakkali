"""
research/experiment_collector.py

Collects experiment metadata from trained models in the registry.
Aggregates metrics, schema, hyperparameters, and XAI artifacts for paper generation.
"""

import os
import json
import logging
from typing import List, Dict, Any

from config.paths import MODEL_REGISTRY_DIR

logger = logging.getLogger(__name__)


class ExperimentCollector:
    """
    Scans model registry and collects all experiment metadata.
    
    Usage:
        collector = ExperimentCollector(registry_dir="models/registry")
        experiments = collector.collect()
        # Returns list of experiment dicts with metrics, modalities, fusion_type, xai, etc.
    """

    def __init__(self, registry_dir: str = str(MODEL_REGISTRY_DIR)):
        """
        Parameters
        ----------
        registry_dir : str
            Path to model registry directory (e.g., "models/registry").
            Supports both:
            - directory-per-model with `metadata.json`
            - legacy flat `*_metadata.json` files
        """
        self.registry_dir = registry_dir

    @staticmethod
    def _normalize_metrics(meta: Dict[str, Any]) -> Dict[str, Any]:
        """Support both nested `metrics` and legacy flat metric keys."""
        metrics = meta.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}

        for key in ("accuracy", "f1", "auc_roc", "ece", "brier", "loss"):
            if key in meta and key not in metrics:
                metrics[key] = meta.get(key)

        phases_summary = meta.get("phases_summary", {})
        if not isinstance(phases_summary, dict):
            phases_summary = {}

        training = phases_summary.get("TRAINING", {})
        if not isinstance(training, dict):
            training = {}

        calibration = training.get("calibration", {})
        if not isinstance(calibration, dict):
            calibration = {}

        evaluation = training.get("evaluation", {})
        if not isinstance(evaluation, dict):
            evaluation = {}

        fallback_map = {
            "accuracy": training.get("best_val_acc"),
            "f1": training.get("best_val_f1"),
            "loss": training.get("best_val_loss"),
            "ece": calibration.get("ece"),
            "brier": calibration.get("brier"),
            "auc_roc": evaluation.get("auc_roc"),
        }
        for key, value in fallback_map.items():
            if key in metrics:
                continue
            try:
                if value is not None:
                    metrics[key] = float(value)
            except (TypeError, ValueError):
                continue
        return metrics

    @staticmethod
    def _normalize_modalities(meta: Dict[str, Any]) -> List[str]:
        """Support modalities from direct metadata or newer config payloads."""
        modalities = meta.get("modalities")

        if isinstance(modalities, dict):
            return [str(k) for k, v in modalities.items() if bool(v)]
        if isinstance(modalities, (list, tuple, set)):
            return [str(m) for m in modalities if m]
        if isinstance(modalities, str) and modalities.strip():
            return [modalities.strip()]

        config = meta.get("config", {})
        if isinstance(config, dict):
            cfg_modalities = config.get("modalities")
            if isinstance(cfg_modalities, (list, tuple, set)):
                return [str(m) for m in cfg_modalities if m]

        return []

    @staticmethod
    def _normalize_target_type(meta: Dict[str, Any]) -> Any:
        """Resolve target type across legacy and current metadata shapes."""
        for value in (
            meta.get("target_type"),
            meta.get("problem_type"),
            (meta.get("config", {}) or {}).get("problem_type") if isinstance(meta.get("config", {}), dict) else None,
        ):
            if value is not None:
                return value
        return None

    @staticmethod
    def _normalize_fusion_type(meta: Dict[str, Any]) -> str:
        """Resolve fusion type from legacy and Phase-4/7 metadata contracts."""
        direct = meta.get("fusion_type") or meta.get("fusion_strategy")
        if isinstance(direct, str) and direct.strip():
            return direct

        phases_summary = meta.get("phases_summary", {})
        if isinstance(phases_summary, dict):
            model_selection = phases_summary.get("MODEL_SELECTION", {})
            if isinstance(model_selection, dict):
                fusion = model_selection.get("fusion_strategy")
                if isinstance(fusion, str) and fusion.strip():
                    return fusion

        state_slots = meta.get("state_slots", {})
        if isinstance(state_slots, dict):
            model_selection_slot = state_slots.get("model_selection", {})
            if isinstance(model_selection_slot, dict):
                fusion = model_selection_slot.get("fusion_strategy")
                if isinstance(fusion, str) and fusion.strip():
                    return fusion

        return "concatenation"

    @staticmethod
    def _normalize_latency_ms(meta: Dict[str, Any]) -> Dict[str, Any]:
        """Support scalar/dict latency plus training-summary fallback."""
        latency_val = meta.get("latency_ms", {})

        if not latency_val:
            phases_summary = meta.get("phases_summary", {})
            if isinstance(phases_summary, dict):
                training = phases_summary.get("TRAINING", {})
                if isinstance(training, dict):
                    evaluation = training.get("evaluation", {})
                    if isinstance(evaluation, dict):
                        latency_val = evaluation.get("latency_ms", {})

        if isinstance(latency_val, dict):
            return latency_val

        try:
            return {"mean": float(latency_val)}
        except (TypeError, ValueError):
            return {"mean": None}

    def collect(self) -> List[Dict[str, Any]]:
        """
        Scan registry and collect all experiment metadata.
        
        Returns
        -------
        List[Dict] with entries:
            {
                "model_id": "apex_v1_...",
                "metrics": {"accuracy": 0.85, "f1": 0.82, ...},
                "modalities": ["tabular", "image"],
                "target": "disease",
                "target_type": "binary",
                "fusion_type": "uncertainty_graph",
                "latency_ms": {"mean": 45.2, "p95": 120.5},
                "memory_mb": 2048,
                "xai": {...},
                "preprocessing_plan": {...},
                "hyperparameters": {...}
            }
        """
        experiments = []

        if not os.path.exists(self.registry_dir):
            logger.warning(f"Registry directory not found: {self.registry_dir}")
            return experiments

        # Iterate through either model directories or legacy metadata files
        for entry in os.listdir(self.registry_dir):
            model_path = os.path.join(self.registry_dir, entry)
            if os.path.isdir(model_path):
                model_id = entry
                meta_path = os.path.join(model_path, "metadata.json")
                if not os.path.exists(meta_path):
                    continue
            elif entry.endswith("_metadata.json"):
                model_id = entry[: -len("_metadata.json")]
                meta_path = model_path
            else:
                continue

            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                metrics = self._normalize_metrics(meta)
                modalities = self._normalize_modalities(meta)
                target_type = self._normalize_target_type(meta)
                fusion_type = self._normalize_fusion_type(meta)
                latency_ms = self._normalize_latency_ms(meta)

                experiment = {
                    "model_id": meta.get("model_id", model_id),
                    "metrics": metrics,
                    "modalities": modalities,
                    "target": meta.get("target"),
                    "target_type": target_type,
                    "fusion_type": fusion_type,
                    "latency_ms": latency_ms,
                    "memory_mb": meta.get("memory_mb", None),
                    "xai": meta.get("xai", {}),
                    "preprocessing_plan": meta.get("preprocessing_plan", {}),
                    "hyperparameters": meta.get("hyperparameters", {}),
                    "schema": meta.get("schema", {}),
                    "created_at": meta.get("created_at") or meta.get("timestamp"),
                    # Legacy compatibility fields used by some tests/tools.
                    "accuracy": metrics.get("accuracy"),
                    "f1": metrics.get("f1"),
                }

                experiments.append(experiment)
                logger.info(f"  ✓ Collected {model_id}")

            except Exception as e:
                logger.warning(f"  Failed to read {model_id}: {e}")

        logger.info(f"Total experiments collected: {len(experiments)}")
        return experiments

    def get_best_experiment(
        self,
        experiments: List[Dict[str, Any]] | None = None,
        metric: str = "accuracy",
    ) -> Dict[str, Any]:
        """
        Find the best-performing experiment by a given metric.
        
        Parameters
        ----------
        experiments : List[Dict] | None
            Output from collect(). If omitted, collect() is called.
        metric : str
            Metric key to optimize (default "accuracy").
        
        Returns
        -------
        Dict : Best experiment or empty dict if none found.
        """
        if experiments is None:
            experiments = self.collect()

        valid = [e for e in experiments if metric in e.get("metrics", {})]
        if not valid:
            return {}
        return max(valid, key=lambda e: e["metrics"][metric])

    def get_experiments_by_modality(
        self,
        experiments_or_modality: List[Dict[str, Any]] | str,
        modality: str | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Filter experiments that use a specific modality.
        """
        if isinstance(experiments_or_modality, list):
            experiments = experiments_or_modality
            target_modality = modality
        else:
            experiments = self.collect()
            target_modality = experiments_or_modality

        if not target_modality:
            return []

        return [e for e in experiments if target_modality in e.get("modalities", [])]
