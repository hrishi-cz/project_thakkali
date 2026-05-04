import pandas as pd
import pytest

from core.execution_context import ExecutionContext
from core.types import TrainingConfig
from data_ingestion.dataset_object import DatasetObject
from pipeline.training_orchestrator import TrainingOrchestrator


def _make_orchestrator(session_id: str = "test-session"):
    ctx = ExecutionContext(session_id=session_id)
    config = TrainingConfig(
        dataset_sources=["https://example.com/dataset.csv"],
        problem_type="classification_binary",
        modalities=["tabular"],
    )
    return TrainingOrchestrator(config, execution_context=ctx), ctx


def test_inject_external_schema_mirrors_execution_context():
    orchestrator, ctx = _make_orchestrator("schema-session")

    schema = {
        "primary_target": "label",
        "global_problem_type": "classification_binary",
        "global_modalities": ["tabular", "text"],
        "modality_presence": {"tabular": True, "text": True},
        "predictability_scores": {"dataset-1": 0.91},
        "per_dataset": [
            {
                "dataset_id": "dataset-1",
                "target_column": "label",
                "modalities": ["tabular", "text"],
                "confidence": 0.91,
                "reasoning": {"xs3_confidence_gap": 0.27},
                "detected_columns": {"tabular": ["feature_a"], "text": ["notes"]},
            }
        ],
    }

    orchestrator.inject_external_schema(schema, target_override="label")

    assert ctx.global_schema["primary_target"] == "label"
    assert ctx.global_target == "label"
    assert ctx.pipeline_stage == "schema_detection"
    assert ctx.predictability_scores["dataset-1"] == pytest.approx(0.91)

    profile = ctx.get_dataset_profile("dataset-1")
    assert profile is not None
    assert profile.schema_detected is True
    assert profile.chosen_target == "label"
    assert profile.schema_confidence == pytest.approx(0.91)
    assert profile.schema_result["dataset_id"] == "dataset-1"
    assert set(profile.modality_breakdown) == {"tabular", "text"}


def test_inject_external_model_selection_mirrors_execution_context():
    orchestrator, ctx = _make_orchestrator("selection-session")

    model_selection = {
        "selected_model": "fusion-a",
        "fusion_strategy": "late",
        "modality_importance": {"tabular": 0.8},
        "recommended_models": [
            {
                "name": "fusion-a",
                "eligible_modalities": ["tabular"],
                "excluded_modalities": {"image": "not needed"},
                "fusion_strategy": "late",
            }
        ],
    }

    orchestrator.inject_external_model_selection(model_selection)

    assert ctx.pipeline_stage == "model_selection"
    assert ctx.selected_model == "fusion-a"
    assert ctx.fusion_strategy == "late"
    assert ctx.model_choices
    assert ctx.eligible_modalities == ["tabular"]
    assert ctx.excluded_modalities == {"image": "not needed"}
    assert "inject_external_model_selection" in ctx.model_selection_reason


def test_sync_preprocessing_contract_mirrors_execution_context():
    orchestrator, ctx = _make_orchestrator("preprocess-session")

    preprocessing_plan = {
        "tabular": {"scaler": "robust"},
        "runtime": {"use_embedding_cache": True},
        "weak_modalities": ["image"],
        "strong_modalities": ["tabular"],
        "modality_predictability": {"tabular": 0.88},
        "fusion_recommendation": "attention",
        "adaptive_tabular_config": {"scaler": "robust"},
    }

    orchestrator._sync_preprocessing_contract_to_context(  # noqa: SLF001
        preprocessing_plan=preprocessing_plan,
        validation_report={"valid": True, "checks_passed": 3},
        context_signals={"global_schema": {"primary_target": "label"}},
        total_samples=42,
        adaptive_tabular_config={"scaler": "robust"},
        drifted_features=["f1", "f2"],
    )

    assert ctx.preprocessing_plan["tabular"]["scaler"] == "robust"
    assert ctx.preprocessing_context["runtime"]["use_embedding_cache"] is True
    assert ctx.preprocessing_context["weak_modalities"] == ["image"]
    assert ctx.preprocessing_context["strong_modalities"] == ["tabular"]
    assert ctx.preprocessing_context["modality_predictability"]["tabular"] == pytest.approx(0.88)
    assert ctx.preprocessing_context["context_signals"]["global_schema"]["primary_target"] == "label"
    assert ctx.preprocessing_context["validation"]["checks_passed"] == 3
    assert ctx.preprocessing_context["dataset_total_samples"] == 42
    assert ctx.preprocessing_context["fusion_recommendation"] == "attention"
    assert ctx.preprocessing_context["adaptive_tabular_config"]["scaler"] == "robust"
    assert ctx.preprocessing_context["drifted_features"] == ["f1", "f2"]
    assert ctx.pipeline_stage == "preprocessing_planning"


def test_sync_training_results_mirrors_execution_context():
    orchestrator, ctx = _make_orchestrator("training-session")

    results = {
        "best_val_loss": 0.123,
        "best_val_acc": 0.91,
        "best_val_f1": 0.88,
        "best_trial": 7,
        "n_trials": 12,
        "fit_type": "classification",
        "trial_diagnostics": [{"trial": 7, "val_loss": 0.123}],
        "trial_feedback_summary": {"best_fit_type": "classification"},
        "next_run_feedback": {"predictability_factors": {"tabular": 1.05}},
        "alignment_summary": {"tabular": 0.7},
        "fusion_summary": {"fusion": "late"},
        "fusion_aux_weights": {"tabular": 0.4},
        "duration_seconds": 9.75,
    }

    orchestrator._sync_training_results_to_context(  # noqa: SLF001
        results=results,
        active_modalities=["tabular", "text"],
    )

    assert ctx.training_signals["best_val_loss"] == pytest.approx(0.123)
    assert ctx.training_signals["best_val_acc"] == pytest.approx(0.91)
    assert ctx.training_signals["best_val_f1"] == pytest.approx(0.88)
    assert ctx.training_signals["best_trial"] == 7
    assert ctx.training_signals["n_trials"] == 12
    assert ctx.training_signals["training_time"] == "9.8s"
    assert ctx.training_signals["fit_type"] == "classification"
    assert ctx.training_signals["trial_feedback_summary"]["best_fit_type"] == "classification"
    assert ctx.training_signals["next_run_feedback"]["predictability_factors"]["tabular"] == pytest.approx(1.05)
    assert ctx.training_signals["alignment_summary"]["tabular"] == pytest.approx(0.7)
    assert ctx.training_signals["fusion_summary"]["fusion"] == "late"
    assert ctx.training_signals["fusion_aux_weights"]["tabular"] == pytest.approx(0.4)
    assert ctx.training_signals["active_modalities"] == ["tabular", "text"]
    assert ctx.active_modalities == ["tabular", "text"]


def test_sync_model_registry_mirrors_execution_context():
    orchestrator, ctx = _make_orchestrator("registry-session")

    orchestrator._sync_model_registry_to_context(  # noqa: SLF001
        model_id="apex_v1_20260418_120000",
        deployment_ready=True,
    )

    assert ctx.registered_model_ids == ["apex_v1_20260418_120000"]
    assert ctx.active_prediction_model_id == "apex_v1_20260418_120000"
    assert ctx.pipeline_stage == "model_registry"


def test_sync_drift_results_mirrors_execution_context():
    orchestrator, ctx = _make_orchestrator("drift-session")
    ctx.predictability_scores = {"tabular": 0.8, "text": 0.5}

    results = {
        "drift_detected": True,
        "metrics": {"psi": 0.31, "ks_statistic": 0.42, "fdd": 0.18},
        "composite_score": 0.56,
        "per_feature_ks": {"f1": 0.41, "f2": 0.12},
        "retrain_triggered": True,
        "retrain_info": {
            "triggered": True,
            "event": {"status": "triggered"},
            "result": {"model_id": "retrained_model_001", "deployment_ready": True},
        },
    }

    orchestrator._sync_drift_results_to_context(  # noqa: SLF001
        results=results,
        modality_drift={"tabular": {"drift_detected": True}},
    )

    assert ctx.drift_detected is True
    assert ctx.drift_severity == pytest.approx(0.56)
    assert ctx.drift_details["psi"] == pytest.approx(0.31)
    assert ctx.drift_details["ks"] == pytest.approx(0.42)
    assert ctx.drift_details["mmd"] == pytest.approx(0.18)
    assert ctx.drift_details["retrain_triggered"] is True
    assert ctx.drift_details["retrain_info"]["result"]["model_id"] == "retrained_model_001"
    assert ctx.drift_feedback_applied is True
    assert ctx.drifted_features == ["f1"]
    assert ctx.drift_adjusted_predictability["tabular"] == pytest.approx(0.6)
    assert ctx.registered_model_ids[-1] == "retrained_model_001"
    assert ctx.active_prediction_model_id == "retrained_model_001"
    assert ctx.pipeline_stage == "drift_detection"


def test_inject_external_datasets_mirrors_execution_context():
    orchestrator, ctx = _make_orchestrator("ingestion-session")

    dataset_obj = DatasetObject(
        dataset_id="dataset-1",
        lazy_data=pd.DataFrame({"f1": [1, 2], "target": [0, 1]}),
        metadata={
            "source_url": "https://example.com/dataset-1.csv",
            "cache_path": "cache/dataset-1",
        },
    )

    orchestrator.inject_external_datasets({"dataset-1": dataset_obj})

    assert ctx.active_dataset_ids == ["dataset-1"]
    profile = ctx.get_dataset_profile("dataset-1")
    assert profile is not None
    assert profile.dataset_id == "dataset-1"
    assert profile.source_url == "https://example.com/dataset-1.csv"
    assert profile.file_path == "cache/dataset-1"
    assert ctx.pipeline_stage == "ingestion_complete"
