from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api.run_api as run_api
from core.execution_context import create_execution_context
from pipeline import retrain_executor
from pipeline.retraining_orchestrator import RetrainingOrchestrator
from pipeline.retraining_pipeline import AdaptiveRetrainingPipeline


def test_retraining_orchestrator_tags_session_history(tmp_path, monkeypatch):
    def _mock_retrain(self, production_sources, problem_type, modalities, schema_info=None, ewc=None):
        return {
            "model_id": "mock_model_001",
            "deployment_ready": True,
            "sources": production_sources,
            "problem_type": problem_type,
            "modalities": modalities,
        }

    monkeypatch.setattr(retrain_executor.RetrainingPipeline, "retrain", _mock_retrain)

    history_path = tmp_path / "retrain-history.jsonl"
    ctx = create_execution_context(session_id="session-123")
    orchestrator = RetrainingOrchestrator(
        production_sources=["mock_source"],
        problem_type="classification_binary",
        modalities=["tabular"],
        cooldown_seconds=0,
        session_id="session-123",
        history_path=history_path,
        execution_context=ctx,
    )

    result = orchestrator.trigger_retraining("dataset_a", {"drift_detected": True})

    assert result["triggered"] is True
    assert result["event"]["session_id"] == "session-123"
    assert ctx.registered_model_ids == ["mock_model_001"]
    assert ctx.active_prediction_model_id == "mock_model_001"

    history = AdaptiveRetrainingPipeline(history_path=history_path).get_history(
        session_id="session-123"
    )
    assert len(history) == 1
    assert history[0]["session_id"] == "session-123"
    assert history[0]["model_id"] == "mock_model_001"


def test_adaptive_retraining_pipeline_updates_execution_context(tmp_path, monkeypatch):
    def _mock_retrain(self, production_sources, problem_type, modalities, schema_info=None, ewc=None):
        return {
            "model_id": "pipeline_model_001",
            "deployment_ready": True,
            "sources": production_sources,
            "problem_type": problem_type,
            "modalities": modalities,
        }

    monkeypatch.setattr(retrain_executor.RetrainingPipeline, "retrain", _mock_retrain)

    ctx = create_execution_context(session_id="session-456")
    pipeline = AdaptiveRetrainingPipeline(
        model_id="pipeline-model",
        history_path=tmp_path / "retrain-history.jsonl",
        execution_context=ctx,
    )

    result = pipeline.retrain(
        production_sources=["mock_source"],
        problem_type="classification_binary",
        modalities=["tabular"],
    )

    assert result["model_id"] == "pipeline_model_001"
    assert ctx.registered_model_ids == ["pipeline_model_001"]
    assert ctx.active_prediction_model_id == "pipeline_model_001"
    assert any(
        entry.get("decision", "") == "Retraining completed: model_id=pipeline_model_001"
        for entry in ctx.execution_log
    )


def test_retrain_history_route_passes_session_id(monkeypatch):
    captured = {}

    def _fake_get_history(self, limit=100, dataset_id=None, session_id=None):
        captured["limit"] = limit
        captured["dataset_id"] = dataset_id
        captured["session_id"] = session_id
        return [
            {
                "dataset_id": dataset_id,
                "session_id": session_id,
                "model_id": "mock_model_001",
            }
        ]

    monkeypatch.setattr(
        "pipeline.retraining_pipeline.AdaptiveRetrainingPipeline.get_history",
        _fake_get_history,
    )

    client = TestClient(run_api.app)
    response = client.get(
        "/retrain-history",
        params={"limit": 5, "dataset_id": "dataset_a", "session_id": "session-123"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert captured["limit"] == 5
    assert captured["dataset_id"] == "dataset_a"
    assert captured["session_id"] == "session-123"
    assert payload["history"][0]["session_id"] == "session-123"
