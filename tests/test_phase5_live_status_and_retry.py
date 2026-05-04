from __future__ import annotations

import uuid
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import api.run_api as run_api
from task_store import TrainingProgressTracker, task_db


def _create_session(client: TestClient) -> str:
    resp = client.post(
        "/v2/sessions",
        json={
            "user_id": "phase5-test-user",
            "project_name": "phase5-test-project",
            "description": "phase5 live status contract",
        },
    )
    assert resp.status_code == 200
    session_id = resp.json().get("session_id")
    assert isinstance(session_id, str) and session_id
    return session_id


def test_update_session_context_with_retry_returns_fresh_revision() -> None:
    client = TestClient(run_api.app)
    session_id = _create_session(client)

    stale_ctx = run_api.session_manager.get_session(session_id)
    assert stale_ctx is not None

    fresh_ctx = run_api.session_manager.get_session(session_id)
    assert fresh_ctx is not None
    fresh_ctx.pipeline_stage = "schema_detection"
    run_api.session_manager.update_session_context(session_id, fresh_ctx)
    revision_after_external_write = fresh_ctx.revision

    stale_ctx.pipeline_stage = "training"
    stale_ctx.active_modalities = ["text", "image"]

    merged_ctx = run_api._update_session_context_with_retry(
        session_id,
        stale_ctx,
        max_retries=2,
        backoff_s=0.0,
    )

    assert merged_ctx.revision > revision_after_external_write
    assert merged_ctx.pipeline_stage == "training"
    assert merged_ctx.active_modalities == ["text", "image"]

    persisted = run_api.session_manager.get_session(session_id)
    assert persisted is not None
    assert persisted.revision == merged_ctx.revision
    assert persisted.pipeline_stage == "training"


def test_train_pipeline_status_exposes_live_cockpit_fields() -> None:
    client = TestClient(run_api.app)
    task_id = f"phase5-{uuid.uuid4().hex[:10]}"
    tracker = TrainingProgressTracker(task_id, task_db)

    tracker.set_phase(5, "Training", 72)
    tracker.set_substage("trial_fit")
    tracker.set_trial(2, 3)
    tracker.set_current_trial(
        number=2,
        total=3,
        fusion="attention",
        lr=1e-4,
        epochs=10,
        status="running",
        current_epoch=4,
        max_epoch=10,
    )
    tracker.set_best_so_far(trial=1, val_loss=0.1234, val_acc=0.91, val_f1=0.89)
    tracker.push_trial_event(2, "trial_start", "Trial 2 started", {"fusion": "attention"})
    tracker.push_trial_event(2, "epoch", "Epoch 4/10 val_loss=0.2231", {"epoch": 4})
    tracker.set_pruning_status(
        available=True,
        backend="inline_optuna_callback",
        reason="Fallback pruning bridge active.",
        pruned_count=1,
        completed_count=1,
    )
    tracker.set_next_trial_plan({"learning_rate_max": 5e-4, "ula_n_layers_max": 2})
    tracker.log_epoch(
        trial=1,
        epoch=4,
        max_epoch=10,
        train_loss=0.301,
        val_loss=0.2231,
        train_acc=0.82,
        val_acc=0.79,
        train_f1=0.81,
        val_f1=0.78,
        val_auroc=0.88,
    )

    resp = client.get(f"/train-pipeline/status/{task_id}")
    assert resp.status_code == 200
    payload = resp.json()

    assert payload["status"] == "running"
    assert payload["current_phase"] == 5
    assert payload["substage"] == "trial_fit"
    assert payload["current_trial"]["number"] == 2
    assert payload["current_trial"]["current_epoch"] == 4
    assert payload["best_so_far"]["trial"] == 1
    assert payload["best_so_far"]["val_loss"] == 0.1234
    assert payload["pruning_status"]["available"] is True
    assert payload["pruning_status"]["backend"] == "inline_optuna_callback"
    assert payload["next_trial_plan"]["learning_rate_max"] == 5e-4
    assert [event["event"] for event in payload["trial_events"][-2:]] == ["trial_start", "epoch"]
    assert payload["epoch_metrics"][-1]["epoch"] == 4
