"""Persistent retraining wrapper with history tracking."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional

from config.paths import RETRAIN_HISTORY_PATH

logger = logging.getLogger(__name__)


class AdaptiveRetrainingPipeline:
    """
    Wrap legacy retraining execution and persist retraining history.

    EWC (Elastic Weight Consolidation, Kirkpatrick et al. 2017 / IEEE TNNLS [8])
    is applied automatically when an existing model is being retrained on new
    data.  The EWC penalty prevents catastrophic forgetting of the previous
    task's optimal weights while adapting to the new distribution.

    Set ``lambda_ewc=0`` to disable EWC (equivalent to plain retraining).
    """

    def __init__(
        self,
        model_id: str = "retrain",
        history_path: Optional[Path] = None,
        execution_context: Optional[Any] = None,
        lambda_ewc: float = 400.0,
    ) -> None:
        self.model_id = str(model_id)
        self.history_path = Path(history_path or RETRAIN_HISTORY_PATH)
        self.execution_context = execution_context
        self.lambda_ewc = float(lambda_ewc)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def retrain(
        self,
        production_sources: List[str],
        problem_type: str,
        modalities: List[str],
        schema_info: Optional[Dict[str, Any]] = None,
        previous_model: Optional[Any] = None,
        ewc_dataloader: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute a retraining cycle.

        Parameters
        ----------
        production_sources : list of str
            Paths to new data files used for retraining.
        problem_type : str
            "classification_binary", "classification_multi", or "regression".
        modalities : list of str
            Active modalities for the retrained model.
        schema_info : dict | None
            Optional schema overrides forwarded to the executor.
        previous_model : nn.Module | None
            The model *before* retraining.  When supplied together with
            ``ewc_dataloader``, EWC is computed on the previous model's
            reference data to protect its weights during the new training run.
        ewc_dataloader : DataLoader | None
            Reference dataloader used to compute the diagonal Fisher
            information matrix for EWC.  When None, EWC is disabled even
            if ``previous_model`` is supplied.
        """
        ewc = None
        if previous_model is not None and ewc_dataloader is not None and self.lambda_ewc > 0:
            try:
                from guardrails.ewc import EWC
                ewc = EWC(
                    model=previous_model,
                    dataloader=ewc_dataloader,
                    lambda_ewc=self.lambda_ewc,
                )
                logger.info(
                    "AdaptiveRetrainingPipeline: EWC initialised "
                    "(λ=%.1f, model_id=%s)",
                    self.lambda_ewc, self.model_id,
                )
            except Exception as ewc_exc:
                logger.warning(
                    "AdaptiveRetrainingPipeline: EWC init failed (%s); "
                    "continuing without EWC regularisation.",
                    ewc_exc,
                )

        from pipeline.retrain_executor import RetrainingPipeline

        runner = RetrainingPipeline(model_id=self.model_id)
        result = runner.retrain(
            production_sources=production_sources,
            problem_type=problem_type,
            modalities=modalities,
            schema_info=schema_info,
            ewc=ewc,
        )

        if self.execution_context is not None:
            try:
                model_id = str(result.get("model_id") or "").strip()
                if model_id:
                    registered_model_ids = getattr(self.execution_context, "registered_model_ids", None)
                    if isinstance(registered_model_ids, list) and model_id not in registered_model_ids:
                        registered_model_ids.append(model_id)

                    if bool(result.get("deployment_ready", False)):
                        self.execution_context.active_prediction_model_id = model_id

                    if hasattr(self.execution_context, "log_decision"):
                        self.execution_context.log_decision(
                            "pipeline",
                            f"Retraining completed: model_id={model_id}",
                            f"deployment_ready={bool(result.get('deployment_ready', False))}",
                        )

                update_timestamp = getattr(self.execution_context, "_update_timestamp", None)
                if callable(update_timestamp):
                    update_timestamp()
            except Exception as ctx_exc:
                logger.warning("AdaptiveRetrainingPipeline context sync failed: %s", ctx_exc)

        return result

    def log_event(self, event: Dict[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        line = json.dumps(payload, default=str)
        with self._lock:
            with open(self.history_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def get_history(
        self,
        limit: int = 100,
        dataset_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not self.history_path.exists():
            return []

        records: List[Dict[str, Any]] = []
        with self._lock:
            with open(self.history_path, "r", encoding="utf-8") as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        item = json.loads(raw)
                    except Exception:
                        continue
                    if dataset_id and str(item.get("dataset_id")) != str(dataset_id):
                        continue
                    if session_id and str(item.get("session_id")) != str(session_id):
                        continue
                    records.append(item)

        records = list(reversed(records))
        return records[: max(1, int(limit))]
