"""
Retraining pipeline for autonomous drift-triggered model improvement.

When Phase 6 confirms distributional drift, ``RetrainingPipeline.retrain()``
instantiates a fresh ``TrainingOrchestrator`` seeded with the same dataset
sources (served from the on-disk cache – no network I/O) and executes
Phases 1, 3, 4, 5, and 7, producing a newly-optimised model artifact.

Phase 2 (schema detection) is re-used from the parent orchestrator's result
dict when *schema_info* is supplied, which avoids redundant column inference
on data whose schema is already known.

Phase 6 is deliberately omitted from the retrain loop to prevent an
infinite drift → retrain recursion.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RetrainingPipeline:
    """
    Autonomous retraining pipeline triggered by Phase 6 drift detection.

    Parameters
    ----------
    model_id : str
        Human-readable label for retrain runs.  Phase 7 appends a timestamp
        suffix to guarantee global uniqueness of the generated model ID.
    """

    def __init__(self, model_id: str = "retrain") -> None:
        self.model_id: str = model_id
        self.retrain_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Trigger check
    # ------------------------------------------------------------------ #

    def should_retrain(self, drift_report: Dict[str, Any]) -> bool:
        """
        Return ``True`` when Phase 6 confirms distributional drift.

        Parameters
        ----------
        drift_report : dict
            Phase 6 result dict (key ``"drift_detected"`` : bool).
        """
        return bool(drift_report.get("drift_detected", False))

    # ------------------------------------------------------------------ #
    # Core retrain loop
    # ------------------------------------------------------------------ #

    def retrain(
        self,
        production_sources: List[str],
        problem_type: str,
        modalities: List[str],
        schema_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute Phases 1, (2|inject), 3, 4, 5, 7 on the production dataset.

        Phase 2 is skipped when *schema_info* is supplied — the caller passes
        the schema already detected during the drift run since the production
        split shares the same column structure as the full dataset.

        Phase 1 is always executed so the new orchestrator's
        ``DatasetManager`` is populated from the on-disk cache without any
        network I/O (cache-hit fast path).

        Parameters
        ----------
        production_sources : List[str]
            Dataset source URLs or local paths that were ingested by the
            parent pipeline.  Cache hits guarantee no re-download.
        problem_type : str
            E.g. ``"classification_binary"`` or ``"regression"``.
        modalities : List[str]
            Subset of ``["tabular", "text", "image"]``.
        schema_info : dict | None
            Phase 2 result dict from the parent orchestrator.  When provided,
            Phase 2 schema detection is skipped entirely.

        Returns
        -------
        dict
            Phase 7 result dict containing ``model_id``, ``deployment_ready``,
            ``artifact_paths``, and ``phases_summary``.

        Raises
        ------
        RuntimeError
            Propagated from any phase that fails internally.
        """
        from pipeline.training_orchestrator import (
            TrainingOrchestrator,
            TrainingConfig,
            Phase,
        )

        logger.info(
            "RetrainingPipeline.retrain: sources=%d  problem=%s  modalities=%s",
            len(production_sources), problem_type, modalities,
        )

        cfg = TrainingConfig(
            dataset_sources=production_sources,
            problem_type=problem_type,
            modalities=modalities,
        )
        orchestrator = TrainingOrchestrator(cfg)

        # ── Phase 1: re-register from cache (no network I/O on cache hits) ─
        self._run_async(
            orchestrator._execute_phase_1_data_ingestion(sources=production_sources)
        )

        # ── Phase 2: inject existing schema or detect from scratch ─────────
        if schema_info is not None:
            # Validate that production data columns still match the schema.
            # New or removed columns indicate schema drift which requires
            # a fresh Phase 2 detection rather than blind injection.
            try:
                production_cols = set()
                for name in orchestrator.dataset_registry.list_datasets():
                    meta = orchestrator.dataset_registry.get_metadata(name)
                    production_cols.update(meta.get("columns", []))

                per_ds = schema_info.get("per_dataset", [{}])
                detected = per_ds[0].get("detected_columns", {}) if per_ds else {}
                schema_cols = set()
                for col_list in detected.values():
                    schema_cols.update(col_list)
                target = schema_info.get("primary_target", "")
                if target and target != "Unknown":
                    schema_cols.add(target)

                if production_cols and schema_cols:
                    new_cols = production_cols - schema_cols
                    removed_cols = schema_cols - production_cols
                    if new_cols or removed_cols:
                        logger.warning(
                            "  Schema drift detected: new_cols=%s  removed_cols=%s "
                            "– running fresh Phase 2 instead of injecting stale schema.",
                            new_cols, removed_cols,
                        )
                        orchestrator._execute_phase_2_schema_detection()
                    else:
                        orchestrator.phase_results[Phase.SCHEMA_DETECTION] = dict(schema_info)
                        logger.info("  Schema validated and injected – Phase 2 skipped.")
                else:
                    # Could not probe columns; inject as-is (best-effort)
                    orchestrator.phase_results[Phase.SCHEMA_DETECTION] = dict(schema_info)
                    logger.info("  Schema injected from parent orchestrator – Phase 2 skipped.")
            except Exception as exc:
                logger.warning(
                    "  Schema validation failed (%s) – running Phase 2 fresh.", exc
                )
                orchestrator._execute_phase_2_schema_detection()
        else:
            orchestrator._execute_phase_2_schema_detection()

        # ── Phase 3: fit preprocessors + build augmented/clean datasets ────
        orchestrator._execute_phase_3_preprocessing()

        # ── Phase 4: model selection + HPO search space ─────────────────────
        orchestrator._execute_phase_4_model_selection()

        # ── Phase 5: Optuna study + Lightning Trainer + MLflow ──────────────
        orchestrator._execute_phase_5_training()

        # ── Phase 7: serialise artifacts to models/registry/ ────────────────
        # (Phase 6 deliberately omitted to prevent infinite retrain recursion)
        orchestrator._execute_phase_7_model_registry()

        result: Dict[str, Any] = orchestrator.phase_results.get(
            Phase.MODEL_REGISTRY, {}
        )
        self.retrain_history.append(result)

        logger.info(
            "RetrainingPipeline: new model_id=%s  deployment_ready=%s",
            result.get("model_id"), result.get("deployment_ready"),
        )
        return result

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    def get_retrain_history(self) -> List[Dict[str, Any]]:
        """Return a list of all Phase 7 result dicts from past retrain runs."""
        return self.retrain_history

    # ------------------------------------------------------------------ #
    # Internal: safe async runner
    # ------------------------------------------------------------------ #

    @staticmethod
    def _run_async(coro: Any) -> Any:
        """
        Run an async coroutine from synchronous code safely.

        Uses a dedicated ``ThreadPoolExecutor`` thread with its own event loop
        so that ``asyncio.run()`` never contends with a running loop on the
        calling thread (e.g. FastAPI / uvicorn).

        Parameters
        ----------
        coro : coroutine
            The coroutine to execute (e.g. ``_execute_phase_1_data_ingestion``).

        Returns
        -------
        Any
            Whatever the coroutine returns.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result(timeout=600)
