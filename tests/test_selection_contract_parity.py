"""Regression tests for model-selection contract parity across selector paths."""

from __future__ import annotations

from automl.advanced_selector import AdvancedModelSelector
from automl.candidate_selector import CandidateSelector


def test_advanced_selector_recommendations_include_contract_fields() -> None:
    selector = AdvancedModelSelector()

    recs = selector.recommend_models(
        problem_type="classification_binary",
        modalities=["tabular"],
        dataset_size=512,
    )

    assert isinstance(recs, list) and recs
    best = recs[0]
    expected = {
        "selection_contract_version",
        "probe_scores",
        "selection_metadata",
        "ranked_candidates",
        "eligible_modalities",
        "excluded_modalities",
    }
    assert expected.issubset(set(best.keys()))
    assert best["selection_contract_version"] == "model_selection.v2"


def test_advanced_selector_filters_weak_modalities_consistently() -> None:
    selector = AdvancedModelSelector()

    recs = selector.recommend_models(
        problem_type="classification_binary",
        modalities=["tabular", "text"],
        dataset_size=1024,
        predictability_scores={"tabular": 0.9, "text": 0.1},
    )

    best = recs[0]
    fallback = recs[1]

    assert "tabular" in list(best.get("eligible_modalities") or [])
    assert "text" in dict(best.get("excluded_modalities") or {})
    assert best.get("text_encoder") is None
    assert fallback.get("text_encoder") is None


def test_selector_contract_keys_are_parity_aligned() -> None:
    candidate_best = CandidateSelector().recommend_models(
        problem_type="classification_binary",
        modalities=["tabular"],
        dataset_size=256,
    )[0]
    advanced_best = AdvancedModelSelector().recommend_models(
        problem_type="classification_binary",
        modalities=["tabular"],
        dataset_size=256,
    )[0]

    parity_keys = {
        "selection_contract_version",
        "probe_scores",
        "selection_metadata",
        "ranked_candidates",
        "eligible_modalities",
        "excluded_modalities",
    }
    assert parity_keys.issubset(set(candidate_best.keys()))
    assert parity_keys.issubset(set(advanced_best.keys()))
