"""Full 7-phase pipeline run on a tiny synthetic dataset.

Drives the REAL PipelineOrchestrator + TrainingOrchestrator. No mocks beyond
what is necessary to fit on a laptop CPU in <60 seconds. Acts as the
regression net for intelligence-propagation work.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.execution_context import create_execution_context
from core.orchestrator import PipelineOrchestrator
from core.types import Phase, TrainingConfig
from pipeline import retrain_executor
from pipeline.inference_engine import MultimodalInferenceEngine
from pipeline.retraining_pipeline import AdaptiveRetrainingPipeline
from pipeline.training_orchestrator import TrainingOrchestrator


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def tiny_workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("apex_e2e")
    (root / "data").mkdir()
    (root / "models").mkdir()
    (root / "registry").mkdir()

    rng = np.random.default_rng(42)
    n = 64
    df = pd.DataFrame({
        "feat_a": rng.normal(size=n),
        "feat_b": rng.normal(size=n),
        "feat_c": rng.normal(size=n),
        "target": (rng.normal(size=n) > 0).astype(int),
    })
    csv_path = root / "data" / "tiny.csv"
    df.to_csv(csv_path, index=False)

    return {
        "root": root,
        "csv": csv_path,
        "frame": df,
        "dataset_id": "tiny_dataset",
    }


@pytest.fixture(scope="module")
def trained_artifacts(tiny_workspace):
    session_id = f"e2e_full_{uuid.uuid4().hex[:8]}"
    ctx = create_execution_context(session_id=session_id)
    dataset_id = tiny_workspace["dataset_id"]
    csv_path = tiny_workspace["csv"]
    frame = tiny_workspace["frame"]

    pipeline = PipelineOrchestrator()
    pipeline.save_context(ctx)
    pipeline.register_ingested_datasets(
        ctx,
        {
            dataset_id: {
                "source": str(csv_path),
                "file_path": str(csv_path),
            }
        },
    )
    pipeline.execute_phase_2_schema(ctx, {dataset_id: frame})
    pipeline.execute_phase_3_target(ctx, {dataset_id: frame})
    pipeline.execute_phase_4_aggregation(ctx, {dataset_id: frame})
    pipeline.execute_phase_5_preprocessing(ctx, {dataset_id: frame})

    if ctx.global_target != "target":
        ctx.override_global_target("target", "E2E test stabilization")

    schema = dict(ctx.global_schema or {})
    problem_type = str(schema.get("global_problem_type", "classification_binary"))
    modalities = list(schema.get("global_modalities", ["tabular"])) or ["tabular"]

    training = TrainingOrchestrator(
        TrainingConfig(
            dataset_sources=[str(csv_path)],
            problem_type=problem_type,
            modalities=modalities,
            target_column=ctx.global_target,
        ),
        execution_context=ctx,
    )
    training.inject_external_datasets({dataset_id: frame})
    training.inject_external_schema(schema, target_override=ctx.global_target)
    training.run_phase(Phase.PREPROCESSING)
    training.run_phase(Phase.MODEL_SELECTION)
    training.run_phase(
        Phase.TRAINING,
        hp_overrides={"epochs": 1, "batch_size": 8, "learning_rate": 1e-3},
    )
    training.run_phase(Phase.DRIFT_DETECTION)
    training.run_phase(Phase.MODEL_REGISTRY)

    if ctx.training_fit_analysis:
        ctx.update_fit_analysis(dict(ctx.training_fit_analysis))

    model_id = ctx.active_prediction_model_id or (ctx.registered_model_ids[-1] if ctx.registered_model_ids else None)
    assert model_id is not None

    return {
        "ctx": ctx,
        "model_id": model_id,
        "frame": frame,
        "csv_path": csv_path,
        "dataset_id": dataset_id,
        "session_id": session_id,
    }


def test_full_pipeline_populates_context(trained_artifacts):
    """All 7 phases run and every expected context field is populated."""
    ctx = trained_artifacts["ctx"]

    assert ctx.active_dataset_ids, "ingestion must register datasets"
    assert ctx.global_schema is not None
    assert ctx.global_target == "target"
    assert ctx.datasets_compatible is True or len(ctx.active_dataset_ids) == 1
    assert ctx.preprocessing_plan, "preprocessing plan must be populated"
    assert ctx.preprocess_plan_version is not None
    assert ctx.selected_model, "a model must be selected"
    assert ctx.ranked_candidates, "ranked candidates must be populated"
    assert ctx.fusion_strategy, "fusion strategy must be set"
    assert ctx.training_signals, "training must emit signals"
    assert ctx.training_fit_analysis, "fit analysis must be populated"
    assert ctx.registered_model_ids, "at least one model must register"
    assert trained_artifacts["model_id"] in ctx.registered_model_ids
    assert ctx.pipeline_stage == "model_registry"


def test_full_pipeline_decision_trace_coverage(trained_artifacts):
    """Every expected decision-trace category fires at least once."""
    ctx = trained_artifacts["ctx"]

    categories_seen = {entry["stage"] for entry in ctx.execution_log}
    expected_categories = {
        "global_schema",
        "global_target",
        "preprocessing",
        "model_selection",
        "fusion",
        "training",
        "training_fit_analysis",
        "drift_detection",
        "drift_feedback",
        "model_registry",
    }
    missing = expected_categories - categories_seen
    assert not missing, f"decision-trace categories missing: {missing}"


def test_full_pipeline_inference_round_trip(trained_artifacts):
    """After training, inference must succeed and log an 'inference' decision."""
    ctx = create_execution_context(session_id="e2e_infer")
    model_id = trained_artifacts["model_id"]
    sample = trained_artifacts["frame"].head(4).drop(columns=["target"])

    ctx.active_prediction_model_id = model_id
    engine = MultimodalInferenceEngine(model_id=model_id)
    result = engine.predict_batch(sample, execution_context=ctx)

    assert "predictions" in result
    assert len(result["predictions"]) == 4
    assert any(entry["stage"] == "inference" for entry in ctx.execution_log)


def test_retraining_propagates_to_context(trained_artifacts, monkeypatch):
    """Adaptive retraining should update the attached context and history."""

    def _mock_retrain(self, production_sources, problem_type, modalities, schema_info=None):
        return {
            "model_id": "pipeline_model_001",
            "deployment_ready": True,
            "sources": production_sources,
            "problem_type": problem_type,
            "modalities": modalities,
        }

    monkeypatch.setattr(retrain_executor.RetrainingPipeline, "retrain", _mock_retrain)

    ctx = create_execution_context(session_id="e2e_retrain")
    pipeline = AdaptiveRetrainingPipeline(
        model_id="e2e_retrain_model",
        execution_context=ctx,
    )

    result = pipeline.retrain(
        production_sources=[str(trained_artifacts["csv_path"])],
        problem_type="classification_binary",
        modalities=["tabular"],
    )

    assert result["model_id"] == "pipeline_model_001"
    assert ctx.registered_model_ids == ["pipeline_model_001"]
    assert ctx.active_prediction_model_id == "pipeline_model_001"
    assert any(
        entry.get("decision", "").startswith("Retraining completed: model_id=pipeline_model_001")
        for entry in ctx.execution_log
    )