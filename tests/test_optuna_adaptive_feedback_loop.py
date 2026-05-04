from __future__ import annotations

from automl.optuna_adaptive import AdaptiveOptunaController


def test_build_next_run_feedback_includes_overrides_and_factors() -> None:
    controller = AdaptiveOptunaController()

    summary = {
        "fit_type": "overfitting",
        "adaptive_penalty": 0.28,
    }
    feedback_state = {
        "last_fit_type": "overfitting",
        "history": ["overfitting", "overfitting"],
    }
    hpo_space = {
        "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-3},
        "weight_decay": {"type": "float", "low": 1e-6, "high": 1e-3},
        "dropout": {"type": "float", "low": 0.05, "high": 0.40},
    }

    out = controller.build_next_run_feedback(
        trial_summary=summary,
        feedback_state=feedback_state,
        hpo_space=hpo_space,
        modalities=["tabular", "text"],
        modality_importance={"tabular": 0.8, "text": 0.2},
    )

    assert out["fit_type"] == "overfitting"
    assert "next_trial_overrides" in out
    assert "predictability_factors" in out
    assert out["next_trial_overrides"]["learning_rate"] < 5.5e-4
    assert out["next_trial_overrides"]["weight_decay"] > 5e-4
    assert out["next_trial_overrides"]["dropout"] > 0.25
    assert out["predictability_factors"]["tabular"] < 1.0
    assert out["predictability_factors"]["text"] < 1.0


def test_derive_predictability_factors_underfitting_biases_upward() -> None:
    controller = AdaptiveOptunaController()

    factors = controller.derive_predictability_factors(
        trial_summary={"fit_type": "underfitting", "adaptive_penalty": 0.20},
        modalities=["tabular", "image"],
        modality_importance={"tabular": 0.7, "image": 0.1},
    )

    assert factors["tabular"] > 1.0
    assert factors["image"] > 1.0
