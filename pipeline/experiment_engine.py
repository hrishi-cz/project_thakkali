"""Structured ablation experiment runner for research and paper generation."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.types import Phase, TrainingConfig
from pipeline.training_orchestrator import TrainingOrchestrator

logger = logging.getLogger(__name__)

EXPERIMENT_STORE = Path("data") / "experiments" / "ablation_results.json"


@dataclass
class AblationCondition:
    """One ablation condition with config overrides and resulting metrics."""

    name: str
    description: str
    config_overrides: Dict[str, Any]
    best_val_loss: float = 0.0
    best_val_acc: float = 0.0
    best_val_f1: float = 0.0
    n_trials: int = 0
    duration_s: float = 0.0
    ece: Optional[float] = None
    brier: Optional[float] = None
    status: str = "pending"
    error: Optional[str] = None


PREDEFINED_ABLATIONS: List[AblationCondition] = [
    AblationCondition(
        name="baseline_concat",
        description="Concatenation fusion, no auxiliary losses",
        config_overrides={"fusion_strategy": "concatenation", "alignment_weight": 0.0},
    ),
    AblationCondition(
        name="attention_fusion",
        description="Attention fusion, no auxiliary losses",
        config_overrides={"fusion_strategy": "attention", "alignment_weight": 0.0},
    ),
    AblationCondition(
        name="graph_fusion",
        description="Graph fusion with auxiliary regularizers",
        config_overrides={"fusion_strategy": "graph", "alignment_weight": 0.0},
    ),
    AblationCondition(
        name="uncertainty_fusion",
        description="Uncertainty-weighted fusion",
        config_overrides={"fusion_strategy": "uncertainty", "alignment_weight": 0.0},
    ),
    AblationCondition(
        name="full_system",
        description="Uncertainty graph + alignment + modality dropout",
        config_overrides={
            "fusion_strategy": "uncertainty_graph",
            "alignment_weight": 0.1,
            "modality_dropout_prob": 0.15,
        },
    ),
]


class ExperimentManager:
    """Run and persist ablation experiments against the existing pipeline."""

    def __init__(
        self,
        base_training_config: TrainingConfig,
        execution_context: Optional[Any] = None,
        store_path: Path = EXPERIMENT_STORE,
    ) -> None:
        self._base_config = base_training_config
        self._ctx = execution_context
        self._store_path = Path(store_path)
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        self._results: List[AblationCondition] = []

    @staticmethod
    def _clone_condition(condition: AblationCondition) -> AblationCondition:
        return AblationCondition(
            name=condition.name,
            description=condition.description,
            config_overrides=dict(condition.config_overrides),
        )

    def run_ablations(self, conditions: Optional[List[AblationCondition]] = None) -> List[AblationCondition]:
        selected = [self._clone_condition(c) for c in (conditions or PREDEFINED_ABLATIONS)]

        for condition in selected:
            logger.info("ExperimentManager: running ablation '%s'", condition.name)
            t0 = time.perf_counter()
            try:
                cfg_dict = asdict(self._base_config)
                config_fields = set(TrainingConfig.__dataclass_fields__.keys())

                cfg_overrides = {
                    k: v for k, v in condition.config_overrides.items() if k in config_fields
                }
                cfg_dict.update(cfg_overrides)

                run_config = TrainingConfig(**cfg_dict)
                orchestrator = TrainingOrchestrator(run_config, execution_context=self._ctx)

                asyncio.run(orchestrator._execute_phase_1_data_ingestion())
                orchestrator._execute_phase_2_schema_detection()
                orchestrator._execute_phase_3_preprocessing()
                orchestrator._execute_phase_4_model_selection()
                orchestrator._execute_phase_5_training(hp_overrides=condition.config_overrides)

                phase5 = orchestrator.phase_results.get(Phase.TRAINING, {})
                condition.best_val_loss = float(phase5.get("best_val_loss", 0.0) or 0.0)
                condition.best_val_acc = float(phase5.get("best_val_acc", 0.0) or 0.0)
                condition.best_val_f1 = float(phase5.get("best_val_f1", 0.0) or 0.0)
                condition.n_trials = int(phase5.get("n_trials", 0) or 0)

                calibration = phase5.get("calibration", {}) if isinstance(phase5, dict) else {}
                if isinstance(calibration, dict):
                    if calibration.get("ece_after") is not None:
                        condition.ece = float(calibration.get("ece_after"))
                    if calibration.get("brier_after") is not None:
                        condition.brier = float(calibration.get("brier_after"))

                condition.status = "completed"
            except Exception as exc:
                logger.warning("Ablation '%s' failed: %s", condition.name, exc)
                condition.status = "failed"
                condition.error = str(exc)
            finally:
                condition.duration_s = round(time.perf_counter() - t0, 2)

            self._results.append(condition)

        self.save()
        return list(self._results)

    def to_rows(self) -> List[Dict[str, Any]]:
        return [asdict(condition) for condition in self._results]

    def save(self) -> None:
        payload = self.to_rows()
        with open(self._store_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        logger.info(
            "ExperimentManager: saved %d ablation results to %s",
            len(payload),
            self._store_path,
        )
