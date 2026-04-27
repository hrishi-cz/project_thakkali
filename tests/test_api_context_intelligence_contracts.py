"""Contract tests for v2 intelligence/decision-trace and phase-timings APIs."""

from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

import api.run_api as run_api
from core.execution_context import DatasetProfile


def _create_session(client: TestClient) -> str:
    resp = client.post(
        "/v2/sessions",
        json={
            "user_id": "contract_test_user",
            "project_name": "contract_test_project",
            "description": "contract validation",
        },
    )
    assert resp.status_code == 200
    sid = resp.json().get("session_id")
    assert isinstance(sid, str) and sid
    return sid


def test_v2_intelligence_contract_has_expected_fields() -> None:
    client = TestClient(run_api.app)
    session_id = _create_session(client)

    ctx = run_api.session_manager.get_session(session_id)
    assert ctx is not None
    ctx.registered_model_ids = ["model-alpha", "model-beta"]
    ctx.active_prediction_model_id = "model-beta"
    run_api.session_manager.update_session_context(session_id, ctx)

    resp = client.get(f"/v2/sessions/{session_id}/intelligence")
    assert resp.status_code == 200
    payload = resp.json()

    expected_keys = {
        "session_id",
        "pipeline_stage",
        "context_version",
        "active_modalities",
        "artifact_versions",
        "predictability_scores",
        "fusion_strategy",
        "phase_timings",
        "drift_detected",
        "registered_model_ids",
        "active_prediction_model_id",
        "training_signals",
        "xai_config",
        "guardrails",
        "fusion_policy_locked",
        "fusion_policy_source",
        "execution_log_count",
    }
    assert expected_keys.issubset(set(payload.keys()))
    assert payload["session_id"] == session_id
    assert payload["registered_model_ids"] == ["model-alpha", "model-beta"]
    assert payload["active_prediction_model_id"] == "model-beta"

    guardrails = payload.get("guardrails", {})
    assert isinstance(guardrails, dict)
    assert guardrails.get("session_isolation", {}).get("status") == "inactive"
    assert guardrails.get("latency", {}).get("implementation") == "LatencyGuard.timed"
    assert guardrails.get("memory", {}).get("vram", {}).get("used_pct") is not None
    assert guardrails.get("memory", {}).get("ram", {}).get("used_pct") is not None
    assert guardrails.get("overall_status") in {"ok", "attention_needed"}


def test_decision_trace_maps_drift_detection_to_monitoring() -> None:
    client = TestClient(run_api.app)
    session_id = _create_session(client)

    ctx = run_api.session_manager.get_session(session_id)
    assert ctx is not None
    ctx.log_decision("drift_detection", "drift check complete", "psi=0.31")
    ctx.registered_model_ids = ["model-alpha"]
    ctx.active_prediction_model_id = "model-alpha"
    run_api.session_manager.update_session_context(session_id, ctx)

    resp = client.get(f"/v2/sessions/{session_id}/decision-trace", params={"limit": 20})
    assert resp.status_code == 200
    payload = resp.json()
    trace = payload.get("trace", [])

    drift_entries = [e for e in trace if str(e.get("stage", "")).lower() == "drift_detection"]
    assert drift_entries, "Expected at least one drift_detection trace entry"
    assert all(e.get("category") == "monitoring" for e in drift_entries)
    assert any("Retraining completed" in str(line) for line in payload.get("curated_summary", []))


def test_phase_timings_contract_returns_summary_shape() -> None:
    client = TestClient(run_api.app)
    session_id = _create_session(client)

    ctx = run_api.session_manager.get_session(session_id)
    assert ctx is not None
    ctx.record_phase_timing("schema_detection", 1.25)
    ctx.record_phase_timing("training", 7.75)
    run_api.session_manager.update_session_context(session_id, ctx)

    resp = client.get(f"/context/{session_id}/phase-timings")
    assert resp.status_code == 200
    payload = resp.json()

    assert payload["session_id"] == session_id
    assert isinstance(payload.get("phase_timings"), dict)
    assert "total_duration_s" in payload
    assert payload["phase_timings"].get("schema_detection") == 1.25
    assert payload["phase_timings"].get("training") == 7.75


def test_monitor_drift_sync_updates_execution_context() -> None:
    client = TestClient(run_api.app)
    session_id = _create_session(client)

    ctx = run_api.session_manager.get_session(session_id)
    assert ctx is not None
    ctx.predictability_scores = {"tabular": 0.8, "text": 0.5}
    run_api.session_manager.update_session_context(session_id, ctx)

    drift_data = {
        "drift_detected": True,
        "metrics": {"psi": 0.31, "ks_statistic": 0.42, "fdd": 0.18},
        "composite_score": 0.56,
        "per_feature_ks": {"f1": 0.41, "f2": 0.12},
        "per_feature_psi": {"f1": 0.28, "f2": 0.09},
        "retrain_triggered": True,
        "model_id": "apex_v1_20260418_120000",
        "retrain_info": {
            "triggered": True,
            "event": {"status": "triggered", "model_id": "apex_v1_20260418_120000"},
            "result": {"model_id": "apex_v1_20260418_120000", "deployment_ready": True},
        },
    }

    run_api._sync_monitor_drift_to_context(session_id, ctx, drift_data)

    updated = run_api.session_manager.get_session(session_id)
    assert updated is not None
    assert updated.drift_detected is True
    assert updated.drift_severity == pytest.approx(0.56)
    assert updated.drift_details["psi"] == pytest.approx(0.31)
    assert updated.drift_details["ks"] == pytest.approx(0.42)
    assert updated.drift_details["mmd"] == pytest.approx(0.18)
    assert updated.drift_details["retrain_triggered"] is True
    assert updated.drift_details["model_id"] == "apex_v1_20260418_120000"
    assert updated.drift_details["retrain_info"]["result"]["deployment_ready"] is True
    assert updated.drift_feedback_applied is True
    assert updated.drifted_features == ["f1"]
    assert updated.drift_adjusted_predictability["tabular"] == pytest.approx(0.6)
    assert updated.pipeline_stage == "monitoring"
    assert updated.registered_model_ids[-1] == "apex_v1_20260418_120000"
    assert updated.active_prediction_model_id == "apex_v1_20260418_120000"


def test_monitor_drift_endpoint_wires_retraining_orchestrator(monkeypatch) -> None:
    client = TestClient(run_api.app)
    session_id = _create_session(client)

    fake_dataset_id = "dataset-hash-1"
    ctx = run_api.session_manager.get_session(session_id)
    assert ctx is not None
    ctx.active_dataset_ids = [fake_dataset_id]
    ctx.dataset_profiles[fake_dataset_id] = DatasetProfile(
        dataset_id=fake_dataset_id,
        source_url="memory://dataset.csv",
    )
    ctx.global_schema = {
        "global_problem_type": "classification_binary",
        "global_modalities": ["tabular"],
    }
    ctx.set_pipeline_stage("ingestion_complete")
    run_api.session_manager.update_session_context(session_id, ctx)

    monkeypatch.setattr(
        run_api,
        "_get_session_hashes",
        lambda _session_id=None: {fake_dataset_id: {"source_url": "memory://dataset.csv"}},
    )

    captured: dict[str, object] = {}

    class FakeDriftDetector:
        def __init__(self, retraining_orchestrator=None, cooldown_seconds: int = 3600) -> None:
            captured["retraining_orchestrator"] = retraining_orchestrator
            self.retraining_orchestrator = retraining_orchestrator

        def detect(self, reference, production, feature_names=None, dataset_id="default"):
            captured["dataset_id"] = dataset_id
            assert self.retraining_orchestrator is not None
            from types import SimpleNamespace

            return SimpleNamespace(
                drift_detected=True,
                psi=0.31,
                ks_statistic=0.42,
                fdd=0.18,
                status={"psi": True, "ks_statistic": True, "fdd": False},
                per_feature_ks={"f1": 0.41},
                per_feature_psi={"f1": 0.28},
                n_features=2,
                n_reference=len(reference),
                n_production=len(production),
                composite_score=0.56,
                retrain_triggered=True,
                retrain_info={"triggered": True, "status": "triggered", "model_id": "model-from-drift"},
                reference_sample=None,
            )

    class FakeDataLoader:
        def load_cached(self, _path):
            import pandas as pd

            return pd.DataFrame(
                {
                    "f1": [0.1, 0.2, 0.3, 0.4],
                    "f2": [1.0, 1.1, 1.2, 1.3],
                }
            )

    monkeypatch.setattr("monitoring.drift_detector.DriftDetector", FakeDriftDetector)
    monkeypatch.setattr("data_ingestion.loader.DataLoader", FakeDataLoader)

    resp = client.post("/monitor/drift", json={"session_id": session_id})
    assert resp.status_code == 200

    payload = resp.json()
    data = payload.get("data", {})
    assert data.get("retrain_triggered") is True
    assert data.get("retrain_info", {}).get("status") == "triggered"
    assert data.get("retrain_info", {}).get("model_id") == "model-from-drift"
    assert captured.get("retraining_orchestrator") is not None
    assert captured.get("dataset_id") == session_id


def test_intelligence_calibration_shape() -> None:
    client = TestClient(run_api.app)
    session_id = _create_session(client)

    resp = client.get(f"/v2/sessions/{session_id}/intelligence/calibration")
    assert resp.status_code == 200
    body = resp.json()
    assert "per_model" in body
    assert isinstance(body["per_model"], list)


def test_intelligence_xai_shape() -> None:
    client = TestClient(run_api.app)
    session_id = _create_session(client)

    resp = client.get(f"/v2/sessions/{session_id}/intelligence/xai")
    assert resp.status_code == 200
    body = resp.json()
    assert "per_model" in body
    assert isinstance(body["per_model"], list)


def test_intelligence_guardrails_shape() -> None:
    client = TestClient(run_api.app)
    session_id = _create_session(client)

    resp = client.get(f"/v2/sessions/{session_id}/intelligence/guardrails")
    assert resp.status_code == 200
    body = resp.json()
    assert "overall_status" in body
    assert "latency" in body
    assert "memory" in body
    assert "session_isolation" in body


def test_intelligence_ranked_candidates_shape() -> None:
    client = TestClient(run_api.app)
    session_id = _create_session(client)

    resp = client.get(f"/v2/sessions/{session_id}/intelligence/ranked-candidates")
    assert resp.status_code == 200
    body = resp.json()
    assert "ranked" in body
    assert isinstance(body["ranked"], list)


def test_intelligence_trial_intelligence_shape() -> None:
    client = TestClient(run_api.app)
    session_id = _create_session(client)

    resp = client.get(f"/v2/sessions/{session_id}/intelligence/trial-intelligence")
    assert resp.status_code == 200
    body = resp.json()
    assert "fit_analysis" in body
    assert "adaptive_lr" in body
    assert "recent_trials" in body
    assert isinstance(body["recent_trials"], list)


def test_intelligence_preprocessing_plan_shape() -> None:
    client = TestClient(run_api.app)
    session_id = _create_session(client)

    resp = client.get(f"/v2/sessions/{session_id}/intelligence/preprocessing-plan")
    assert resp.status_code == 200
    body = resp.json()
    assert "version" in body
    assert "plan" in body
    assert "choices" in body
    assert "context" in body
    assert "per_dataset_plans" in body
    assert isinstance(body["per_dataset_plans"], list)
