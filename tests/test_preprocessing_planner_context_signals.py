from __future__ import annotations

import pytest

from core.execution_context import create_execution_context
from preprocessing.preprocessing_planner import PreprocessingPlanner
from preprocessing.tabular_preprocessor import TabularPreprocessor


def test_planner_uses_drift_adjusted_scores_for_tabular_policy() -> None:
    planner = PreprocessingPlanner()

    plan = planner.create_plan(
        schema_info={"global_modalities": ["tabular", "text"]},
        total_samples=20_000,
        predictability_scores={"tabular": 0.90, "text": 0.80},
        drift_adjusted_predictability={"tabular": 0.20},
        drifted_features=["age_shift"],
    )

    assert plan["modality_predictability"]["tabular"] == pytest.approx(0.20)
    assert "tabular" in plan["weak_modalities"]
    assert plan["tabular"]["scaler"] == "robust"
    assert plan["tabular"]["drifted_features"] == ["age_shift"]


def test_planner_merges_modality_presence_into_modalities() -> None:
    planner = PreprocessingPlanner()

    plan = planner.create_plan(
        schema_info={"global_modalities": ["tabular"]},
        total_samples=512,
        modality_presence={"text": True, "image": False},
    )

    assert "tabular" in plan["modalities"]
    assert "text" in plan["modalities"]
    assert "image" not in plan["modalities"]
    assert plan["context_signals"]["modality_presence"]["text"] is True


def test_execution_context_effective_predictability_prefers_drift_feedback() -> None:
    ctx = create_execution_context("session_preproc_test")
    ctx.predictability_scores = {"tabular": 0.70}
    ctx.drift_adjusted_predictability = {"tabular": 0.35}
    ctx.drift_feedback_applied = True

    scores = ctx.get_effective_predictability_scores()
    assert scores["tabular"] == pytest.approx(0.35)


def test_execution_context_applies_training_feedback_factors() -> None:
    ctx = create_execution_context("session_training_feedback_test")
    ctx.predictability_scores = {"tabular": 0.80, "text": 0.60}

    ctx.apply_training_feedback(
        {
            "fit_type": "overfitting",
            "adaptive_penalty": 0.22,
            "next_run_feedback": {
                "predictability_factors": {"tabular": 0.75, "text": 0.90}
            },
        },
        predictability_factors={"tabular": 0.75, "text": 0.90},
    )

    scores = ctx.get_effective_predictability_scores()
    assert scores["tabular"] == pytest.approx(0.60)
    assert scores["text"] == pytest.approx(0.54)

    signals = ctx.get_preprocessing_signals()
    assert signals["training_fit_analysis"].get("feedback_applied") is True


def test_execution_context_persists_rich_preprocessing_contract() -> None:
    ctx = create_execution_context("session_preprocessing_contract_test")

    ctx.update_preprocessing_contract(
        {"tabular": {"scaler": "robust"}, "text": {"max_length": 128}},
        {
            "runtime": {"use_embedding_cache": True, "high_volume_mode": False},
            "weak_modalities": ["tabular"],
            "strong_modalities": ["text"],
            "modality_predictability": {"tabular": 0.20, "text": 0.91},
            "context_signals": {"validation": {"status": "ok"}},
            "validation": {"status": "ok"},
            "dataset_total_samples": 1234,
            "fusion_recommendation": "attention",
            "adaptive_tabular_config": {"scaler": "robust"},
            "drifted_features": ["age_shift"],
        },
    )

    assert ctx.preprocessing_plan["tabular"]["scaler"] == "robust"
    assert ctx.preprocessing_context["runtime"]["use_embedding_cache"] is True
    assert ctx.preprocessing_context["weak_modalities"] == ["tabular"]
    assert ctx.preprocessing_context["strong_modalities"] == ["text"]
    assert ctx.preprocessing_context["modality_predictability"]["tabular"] == pytest.approx(0.20)
    assert ctx.preprocessing_context["validation"]["status"] == "ok"
    assert ctx.preprocessing_context["dataset_total_samples"] == 1234
    assert ctx.preprocessing_context["fusion_recommendation"] == "attention"
    assert ctx.preprocessing_context["adaptive_tabular_config"]["scaler"] == "robust"
    assert ctx.preprocessing_context["drifted_features"] == ["age_shift"]


def test_tabular_preprocessor_accepts_drifted_features_from_plan() -> None:
    preprocessor = TabularPreprocessor()
    preprocessor.configure({"drifted_features": ["f1", "f2"], "scaler": "robust"})

    assert preprocessor._drifted_features == ["f1", "f2"]
