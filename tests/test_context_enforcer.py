"""Unit tests for hard ExecutionContext enforcement helpers."""

from __future__ import annotations

import pytest

from core.context_enforcer import (
    ContextValidationError,
    ensure_session_context,
    validate_session_context,
)
from core.execution_context import create_execution_context


def test_validate_session_context_requires_context_for_session() -> None:
    result = validate_session_context(
        None,
        session_id="session_abc",
        dataset_snapshot={"d1": {}},
    )

    assert result.ok is False
    assert any("ExecutionContext not found" in err for err in result.errors)


def test_validate_session_context_accepts_dataset_snapshot() -> None:
    ctx = create_execution_context("session_abc")

    result = validate_session_context(
        ctx,
        session_id="session_abc",
        dataset_snapshot={"hash_1": {"source": "x.csv"}},
    )

    assert result.ok is True
    assert result.errors == []


def test_validate_session_context_rejects_closed_session() -> None:
    ctx = create_execution_context("session_closed")
    ctx.status = "closed"

    result = validate_session_context(
        ctx,
        session_id="session_closed",
        dataset_snapshot={"hash_1": {}},
    )

    assert result.ok is False
    assert any("not active" in err for err in result.errors)


def test_validate_session_context_rejects_session_mismatch() -> None:
    ctx = create_execution_context("session_a")

    result = validate_session_context(
        ctx,
        session_id="session_b",
        dataset_snapshot={"hash_1": {}},
    )

    assert result.ok is False
    assert any("mismatch" in err for err in result.errors)


def test_ensure_session_context_raises_validation_error() -> None:
    ctx = create_execution_context("session_test")

    with pytest.raises(ContextValidationError):
        ensure_session_context(ctx, session_id="session_test", dataset_snapshot={})
