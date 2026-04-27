"""Regression tests for CandidateSelector edge cases."""

import numpy as np

from automl.candidate_selector import CandidateSelector, TABULAR_CANDIDATE_POOL


def test_apply_jit_filter_handles_empty_candidate_lists() -> None:
    selector = CandidateSelector()

    filtered = selector.apply_jit_filter({"text": []}, vram_gb=8.0)

    assert "text" in filtered
    assert filtered["text"] == []


def test_rank_candidates_marks_unprobed_models() -> None:
    selector = CandidateSelector()

    ranked = selector.rank_candidates(
        probe_scores={"tabular": {}},
        candidates={"tabular": [dict(TABULAR_CANDIDATE_POOL[0])]},
        schema_info={"total_samples": 500},
        hardware_info={"gpu_memory_gb": 0.0},
    )

    assert ranked["tabular"][0].probed is False


def test_quick_probe_tabular_handles_stratification_edge_case() -> None:
    selector = CandidateSelector()
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.array([0, 0, 0, 1], dtype=int)

    scores = selector.quick_probe_tabular(
        candidates=list(TABULAR_CANDIDATE_POOL),
        X=X,
        y=y,
        problem_type="classification_binary",
        max_rows=4,
    )

    assert isinstance(scores, dict)


def test_recommend_models_returns_normalized_contract_fields() -> None:
    selector = CandidateSelector()

    recs = selector.recommend_models(
        problem_type="classification_binary",
        modalities=["tabular"],
        dataset_size=256,
    )

    assert isinstance(recs, list) and recs
    best = recs[0]
    assert best.get("selection_contract_version") == "model_selection.v2"
    assert isinstance(best.get("probe_scores"), dict)
    assert isinstance(best.get("selection_metadata"), dict)
    assert isinstance(best.get("ranked_candidates"), dict)


def test_recommend_models_filters_weak_modalities_by_predictability() -> None:
    selector = CandidateSelector()

    recs = selector.recommend_models(
        problem_type="classification_binary",
        modalities=["tabular", "text"],
        dataset_size=1024,
        predictability_scores={"tabular": 0.8, "text": 0.1},
    )

    best = recs[0]
    fallback = recs[1]

    assert "tabular" in list(best.get("eligible_modalities") or [])
    assert "text" in dict(best.get("excluded_modalities") or {})
    assert best.get("text_encoder") is None
    assert fallback.get("text_encoder") is None
