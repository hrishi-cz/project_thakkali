"""Phase 2 session-scoped ingestion API tests."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict

import pytest
import requests

BASE_URL = os.getenv("APEX_API_BASE_URL", "http://127.0.0.1:8001")


def _request_json(method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
    response = requests.request(method, f"{BASE_URL}{path}", timeout=60, **kwargs)
    assert response.status_code < 500, response.text
    payload = response.json()
    return {"status_code": response.status_code, "payload": payload}


def _wait_for_ingestion(task_id: str, timeout_seconds: int = 120) -> Dict[str, Any]:
    deadline = time.time() + timeout_seconds
    latest: Dict[str, Any] = {}
    while time.time() < deadline:
        latest = _request_json("GET", f"/ingest/status/{task_id}")
        state = latest["payload"].get("status")
        if state in {"completed", "failed"}:
            return latest
        time.sleep(1)
    raise AssertionError(f"ingestion task {task_id} did not complete in {timeout_seconds}s")


@pytest.fixture
def session_payload() -> Dict[str, Any]:
    return {
        "user_id": "test_user_1",
        "project_name": "AutoVision+ Test",
        "description": "Testing session-scoped ingestion",
    }


@pytest.fixture
def session_id(session_payload: Dict[str, Any]) -> str:
    created = _request_json("POST", "/v2/sessions", json=session_payload)
    assert created["status_code"] == 200
    sid = created["payload"].get("session_id")
    assert isinstance(sid, str) and sid
    try:
        yield sid
    finally:
        try:
            requests.post(f"{BASE_URL}/v2/sessions/{sid}/close", timeout=30)
        except Exception:
            pass


@pytest.fixture
def local_dataset_path(tmp_path: Path) -> str:
    csv_path = tmp_path / "phase2_dataset.csv"
    csv_path.write_text(
        "f1,f2,target\n"
        "1.0,2.0,0\n"
        "1.2,2.1,1\n"
        "0.9,1.8,0\n"
        "1.4,2.4,1\n",
        encoding="utf-8",
    )
    return str(csv_path)


def test_create_session(session_id: str) -> None:
    assert session_id.startswith("session_")


def test_get_session(session_id: str) -> None:
    response = _request_json("GET", f"/v2/sessions/{session_id}")
    data = response["payload"]

    assert response["status_code"] == 200
    assert data["session_id"] == session_id
    assert data["status"] == "active"
    assert isinstance(data.get("active_dataset_ids"), list)


def test_list_sessions_contains_created_session(session_id: str) -> None:
    response = _request_json("GET", "/v2/sessions")
    data = response["payload"]

    assert response["status_code"] == 200
    assert "sessions" in data
    assert "total" in data
    ids = {item.get("session_id") for item in data["sessions"]}
    assert session_id in ids


def test_add_datasets(session_id: str, local_dataset_path: str) -> None:
    start = _request_json(
        "POST",
        f"/v2/sessions/{session_id}/datasets",
        json={
            "dataset_urls": [local_dataset_path],
            "force_redownload": False,
        },
    )
    data = start["payload"]

    assert start["status_code"] == 200
    assert data["status"] == "processing"
    task_id = data.get("task_id")
    assert isinstance(task_id, str) and task_id

    final = _wait_for_ingestion(task_id)
    assert final["payload"].get("status") == "completed"
    progress = (final["payload"].get("result") or {}).get("ingestion_progress", {})
    assert progress.get("status") in {"success", "partial"}


def test_list_session_datasets(session_id: str, local_dataset_path: str) -> None:
    start = _request_json(
        "POST",
        f"/v2/sessions/{session_id}/datasets",
        json={
            "dataset_urls": [local_dataset_path],
            "force_redownload": False,
        },
    )
    task_id = start["payload"].get("task_id")
    assert isinstance(task_id, str) and task_id
    _wait_for_ingestion(task_id)

    response = _request_json("GET", f"/v2/sessions/{session_id}/datasets")
    data = response["payload"]

    assert response["status_code"] == 200
    assert "datasets" in data
    assert "active_datasets" in data
    assert "cached_datasets" in data
    assert isinstance(data["datasets"], list)
    assert len(data["datasets"]) >= 1


def test_close_session(session_payload: Dict[str, Any]) -> None:
    created = _request_json("POST", "/v2/sessions", json=session_payload)
    session_id_value = created["payload"]["session_id"]

    response = _request_json("POST", f"/v2/sessions/{session_id_value}/close")
    data = response["payload"]

    assert response["status_code"] == 200
    assert data["status"] == "closed"


def test_database_persistence_basic(session_id: str) -> None:
    # Persistence across process restarts is an operational test; this verifies
    # that newly persisted sessions are retrievable from the database now.
    response = _request_json("GET", f"/v2/sessions/{session_id}")
    assert response["status_code"] == 200
    assert response["payload"]["session_id"] == session_id
