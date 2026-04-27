"""Regression tests for ExperimentCollector metadata normalization."""

import json

from research.experiment_collector import ExperimentCollector


def test_collect_normalizes_phase_summary_metadata(tmp_path) -> None:
    registry_dir = tmp_path / "registry"
    model_dir = registry_dir / "apex_demo_model"
    model_dir.mkdir(parents=True)

    metadata = {
        "model_id": "apex_demo_model",
        "created_at": "2026-04-15T00:00:00",
        "config": {
            "problem_type": "classification_binary",
            "modalities": ["tabular", "text"],
        },
        "phases_summary": {
            "MODEL_SELECTION": {
                "fusion_strategy": "attention",
            },
            "TRAINING": {
                "best_val_acc": 0.88,
                "best_val_f1": 0.85,
                "best_val_loss": 0.42,
                "calibration": {
                    "ece": 0.04,
                    "brier": 0.08,
                },
            },
        },
    }

    (model_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    collector = ExperimentCollector(registry_dir=str(registry_dir))
    experiments = collector.collect()

    assert len(experiments) == 1
    exp = experiments[0]
    assert exp["model_id"] == "apex_demo_model"
    assert exp["metrics"]["accuracy"] == 0.88
    assert exp["metrics"]["f1"] == 0.85
    assert exp["metrics"]["loss"] == 0.42
    assert exp["metrics"]["ece"] == 0.04
    assert exp["fusion_type"] == "attention"
    assert exp["modalities"] == ["tabular", "text"]
    assert exp["target_type"] == "classification_binary"
