from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.paths import RETRAIN_HISTORY_PATH

logger = logging.getLogger(__name__)


class RetrainingOrchestrator:
    """
    Orchestrates drift-triggered retraining decisions and execution.

    This class keeps orchestration separate from drift detection logic:
    - DriftDetector computes metrics.
    - RetrainingOrchestrator decides and triggers retraining.
    """

    def __init__(
        self,
        production_sources: Optional[List[str]] = None,
        problem_type: str = "classification_binary",
        modalities: Optional[List[str]] = None,
        schema_info: Optional[Dict[str, Any]] = None,
        cooldown_seconds: int = 3600,
        session_id: Optional[str] = None,
        history_path: Optional[Path] = None,
        execution_context: Optional[Any] = None,
    ) -> None:
        self.production_sources = list(production_sources or [])
        self.problem_type = problem_type
        self.modalities = list(modalities or ["tabular"])
        self.schema_info = schema_info
        self.cooldown_seconds = max(0, int(cooldown_seconds))
        self.session_id = session_id
        self.history_path = Path(history_path or RETRAIN_HISTORY_PATH)
        self.execution_context = execution_context
        self._last_trigger_by_dataset: Dict[str, float] = {}
        self._event_log: List[Dict[str, Any]] = []

    def should_retrain(self, drift_report: Dict[str, Any]) -> bool:
        """Return True when drift is confirmed or composite risk is high."""
        trigger_reason: Optional[str] = None
        if bool(drift_report.get("drift_detected", False)):
            trigger_reason = "drift_detected"
        else:
            composite = float(drift_report.get("composite_score", 0.0) or 0.0)
            if composite >= 1.0:
                trigger_reason = f"composite_score={composite:.4f}"

        if trigger_reason is not None:
            if self.execution_context is not None and hasattr(self.execution_context, "log_decision"):
                self.execution_context.log_decision(
                    "retraining",
                    "Retraining triggered",
                    evidence=f"trigger_reason={trigger_reason}",
                )
            return True

        return False

    def trigger_retraining(
        self,
        dataset_id: str,
        drift_report: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run a retraining cycle if cooldown has elapsed."""
        from pipeline.retraining_pipeline import AdaptiveRetrainingPipeline

        history = AdaptiveRetrainingPipeline(
            model_id=f"drift_retrain_{dataset_id}",
            history_path=self.history_path,
            execution_context=self.execution_context,
        )
        now = time.time()
        last_ts = self._last_trigger_by_dataset.get(dataset_id)
        if last_ts is not None and (now - last_ts) < self.cooldown_seconds:
            remaining = self.cooldown_seconds - (now - last_ts)
            event = {
                "dataset_id": dataset_id,
                "status": "cooldown_blocked",
                "cooldown_remaining_seconds": round(max(0.0, remaining), 2),
                "timestamp": now,
            }
            if self.session_id is not None:
                event["session_id"] = self.session_id
            self.log_retraining_event(event)
            history.log_event(event)
            return {"triggered": False, **event}

        if not self.production_sources:
            event = {
                "dataset_id": dataset_id,
                "status": "skipped_no_sources",
                "timestamp": now,
            }
            if self.session_id is not None:
                event["session_id"] = self.session_id
            self.log_retraining_event(event)
            history.log_event(event)
            return {"triggered": False, **event}

        # Read depth from context if available; default to "full"
        depth = "full"
        if self.execution_context is not None:
            ctx_depth = getattr(self.execution_context, "retraining_depth_required", "full")
            if ctx_depth in ("calibration_only", "head_only", "full"):
                depth = ctx_depth

        logger.info("RetrainingOrchestrator: retraining depth=%s for dataset=%s", depth, dataset_id)

        if depth == "calibration_only":
            # Only re-calibrate the existing model; skip full training
            try:
                from pipeline.calibration import run_calibration_only
                retrain_result = run_calibration_only(
                    execution_context=self.execution_context,
                    production_sources=self.production_sources,
                    problem_type=self.problem_type,
                )
            except Exception as exc:
                logger.warning("calibration_only path failed (%s); falling back to full retrain", exc)
                retrain_result = history.retrain(
                    production_sources=self.production_sources,
                    problem_type=self.problem_type,
                    modalities=self.modalities,
                    schema_info=self.schema_info,
                )
        elif depth == "head_only":
            # Freeze encoders; only retrain the head
            retrain_result = history.retrain(
                production_sources=self.production_sources,
                problem_type=self.problem_type,
                modalities=self.modalities,
                schema_info=self.schema_info,
                freeze_encoders=True,
            )
        else:
            retrain_result = history.retrain(
                production_sources=self.production_sources,
                problem_type=self.problem_type,
                modalities=self.modalities,
                schema_info=self.schema_info,
            )

        self._last_trigger_by_dataset[dataset_id] = now
        event = {
            "dataset_id": dataset_id,
            "status": "triggered",
            "timestamp": now,
            "model_id": retrain_result.get("model_id"),
            "drift_report": drift_report or {},
        }
        if self.session_id is not None:
            event["session_id"] = self.session_id
        self.log_retraining_event(event)
        history.log_event(event)

        return {"triggered": True, "event": event, "result": retrain_result}

    def log_retraining_event(self, event: Dict[str, Any]) -> None:
        self._event_log.append(dict(event))
        logger.info("RetrainingOrchestrator event: %s", event)

    def get_events(self) -> List[Dict[str, Any]]:
        return list(self._event_log)
