"""API contract tests for model-registry actions used by the frontend."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

import api.run_api as run_api
import registry.model_registry as registry_model_registry


def _seed_registry_model(registry_root: Path, model_id: str = "demo_model") -> Path:
    model_dir = registry_root / model_id
    artifacts_dir = model_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    weights_path = artifacts_dir / "model_weights.pth"
    weights_path.write_bytes(b"dummy-weights")

    metadata = {
        "model_id": model_id,
        "display_name": model_id,
        "artifact_paths": {
            "model_weights": str(weights_path.resolve()),
        },
        "artifact_versions": {
            "preprocessing_plan": "plan-v1",
            "model_selection": "sel-v2",
        },
        "training_signals": {
            "best_val_loss": 0.123,
            "best_val_acc": 0.91,
            "best_val_f1": 0.88,
            "fit_type": "classification",
        },
        "training_fit_analysis": {
            "fit_type": "classification",
            "feedback_applied": True,
            "predictability_factors": {"tabular": 1.05},
        },
        "xai_config": {
            "enabled": True,
            "method": "shap",
        },
        "fusion": {
            "summary": {
                "fusion": "late",
                "strategy": "attention",
            },
            "auxiliary_loss_weights": {
                "tabular": 0.4,
            },
            "alignment_summary": {
                "tabular": 0.7,
            },
        },
        "xai": {
            "fusion": {
                "strategy": "late",
                "method": "attention",
                "weights": {
                    "tabular": 0.4,
                },
            }
        },
        "research_metrics": {
            "fusion_ablation_delta": 0.12,
            "xai_coverage": 1.0,
        },
        "phases_summary": {
            "TRAINING": {
                "best_val_loss": 0.123,
                "best_val_acc": 0.91,
                "best_val_f1": 0.88,
                "n_trials": 12,
                "calibration": {
                    "enabled": True,
                    "mode": "isotonic",
                    "ece_before": 0.11,
                    "ece_after": 0.07,
                    "brier_before": 0.21,
                    "brier_after": 0.18,
                },
                "evaluation": {
                    "training": {
                        "performance": 0.9,
                        "loss_score": 0.89,
                        "generalization_gap": 0.03,
                        "stability": 0.97,
                        "overall_score": 0.92,
                    },
                    "monitoring": {
                        "drift_detected": False,
                        "risk_score": 0.1,
                        "health_score": 0.9,
                        "retrain_triggered": False,
                    },
                    "combined_score": 0.84,
                },
            }
        },
    }
    with open(model_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    return model_dir


def test_model_registry_rename_and_download_contract(tmp_path, monkeypatch) -> None:
    registry_root = tmp_path / "models" / "registry"
    _seed_registry_model(registry_root, model_id="demo_model")

    monkeypatch.setattr(run_api, "MODEL_REGISTRY_DIR", registry_root)
    client = TestClient(run_api.app)

    list_resp = client.get("/model-registry")
    assert list_resp.status_code == 200
    listed = list_resp.json()
    assert listed["status"] == "success"
    assert listed["count"] == 1
    assert listed["models"][0]["model_id"] == "demo_model"
    assert listed["models"][0]["rename_mode"] == "alias_only"
    assert listed["models"][0]["artifact_exists"]["model_weights"] is True

    rename_resp = client.patch(
        "/model-registry/demo_model/rename",
        json={"new_name": "demo_alias"},
    )
    assert rename_resp.status_code == 200
    rename_payload = rename_resp.json()
    assert rename_payload["status"] == "success"
    assert rename_payload["model_id"] == "demo_model"
    assert rename_payload["display_name_alias"] == "demo_alias"
    assert rename_payload["rename_mode"] == "alias_only"

    meta_path = registry_root / "demo_model" / "metadata.json"
    with open(meta_path, "r", encoding="utf-8") as fh:
        saved_meta = json.load(fh)
    assert saved_meta["display_name_alias"] == "demo_alias"
    assert saved_meta["rename_mode"] == "alias_only"

    download_resp = client.get("/model-registry/demo_model/download")
    assert download_resp.status_code == 200
    assert download_resp.headers.get("content-type", "").startswith("application/zip")

    zip_bytes = io.BytesIO(download_resp.content)
    with zipfile.ZipFile(zip_bytes, "r") as zf:
        names = set(zf.namelist())
        assert "metadata.json" in names
        assert "bundle_manifest.json" in names
        assert "README.txt" in names
        assert "model_weights.pth" in names


def test_model_registry_actions_return_404_for_missing_model(tmp_path, monkeypatch) -> None:
    registry_root = tmp_path / "models" / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(run_api, "MODEL_REGISTRY_DIR", registry_root)
    client = TestClient(run_api.app)

    rename_resp = client.patch(
        "/model-registry/missing_model/rename",
        json={"new_name": "alias"},
    )
    assert rename_resp.status_code == 404

    download_resp = client.get("/model-registry/missing_model/download")
    assert download_resp.status_code == 404


def test_model_registry_richer_metadata_contract(tmp_path, monkeypatch) -> None:
    registry_root = tmp_path / "models" / "registry"
    _seed_registry_model(registry_root, model_id="demo_model")

    monkeypatch.setattr(run_api, "MODEL_REGISTRY_DIR", registry_root)
    monkeypatch.setattr(registry_model_registry, "MODEL_REGISTRY_DIR", registry_root)
    client = TestClient(run_api.app)

    info_resp = client.get("/model-info/demo_model")
    assert info_resp.status_code == 200
    info_payload = info_resp.json()
    assert info_payload["artifact_versions"]["preprocessing_plan"] == "plan-v1"
    assert info_payload["training_signals"]["best_val_loss"] == 0.123
    assert info_payload["training_fit_analysis"]["fit_type"] == "classification"
    assert info_payload["xai_config"]["enabled"] is True
    assert info_payload["fusion"]["summary"]["fusion"] == "late"
    assert info_payload["training"]["calibration"]["mode"] == "isotonic"
    assert info_payload["evaluation"]["combined_score"] == 0.84
    assert info_payload["research_metrics"]["xai_coverage"] == 1.0

    stats_resp = client.get("/models/demo_model/stats")
    assert stats_resp.status_code == 200
    stats_payload = stats_resp.json()["data"]
    assert stats_payload["artifact_versions"]["model_selection"] == "sel-v2"
    assert stats_payload["training_signals"]["best_val_acc"] == 0.91
    assert stats_payload["training_fit_analysis"]["feedback_applied"] is True
    assert stats_payload["xai_config"]["method"] == "shap"
    assert stats_payload["fusion_summary"]["summary"]["strategy"] == "attention"
    assert stats_payload["training"]["calibration"]["ece_after"] == 0.07
    assert stats_payload["evaluation"]["monitoring"]["health_score"] == 1.0
    assert stats_payload["research_metrics"]["fusion_ablation_delta"] == 0.12
