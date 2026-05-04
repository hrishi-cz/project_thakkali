"""
Context enforcement helpers for session-backed training flows.

This module centralizes hard validation for ExecutionContext integrity so
API handlers and orchestrators fail fast with actionable messages.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, List

from core.execution_context import ExecutionContext


@dataclass
class ContextValidationResult:
    """Structured result for context validation checks."""

    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ContextValidationError(RuntimeError):
    """Raised when a required session context contract is violated."""

    def __init__(self, result: ContextValidationResult):
        self.result = result
        message = "; ".join(result.errors) if result.errors else "ExecutionContext validation failed"
        super().__init__(message)


def validate_session_context(
    ctx: Optional[ExecutionContext],
    *,
    session_id: Optional[str],
    dataset_snapshot: Optional[Mapping[str, Any]] = None,
) -> ContextValidationResult:
    """
    Validate session-backed context invariants used by training endpoints.

    Rules are intentionally strict for session-backed runs:
    - context must exist when a session_id is supplied
    - context/session IDs must match
    - context status must be active
    - at least one dataset must be present (context or snapshot)
    - optional schema payload must be well-formed when present
    """
    errors: List[str] = []
    warnings: List[str] = []

    if session_id and ctx is None:
        errors.append(
            "ExecutionContext not found for session_id. "
            "Create or restore the session before training."
        )
        return ContextValidationResult(ok=False, errors=errors, warnings=warnings)

    if ctx is None:
        return ContextValidationResult(ok=True, errors=errors, warnings=warnings)

    if session_id and ctx.session_id != session_id:
        errors.append(
            f"ExecutionContext/session mismatch (context={ctx.session_id}, request={session_id})."
        )

    if str(getattr(ctx, "status", "active")).lower() != "active":
        errors.append(
            f"Session {ctx.session_id} is not active (status={ctx.status})."
        )

    snapshot_ids = set(dataset_snapshot.keys()) if isinstance(dataset_snapshot, Mapping) else set()
    context_ids = set(ctx.active_dataset_ids or [])
    if not context_ids and not snapshot_ids:
        errors.append(
            "No active datasets found in session context or ingestion snapshot."
        )

    if context_ids and snapshot_ids and not (context_ids & snapshot_ids):
        warnings.append(
            "Session context dataset IDs and ingestion snapshot IDs do not overlap."
        )

    if ctx.global_schema is not None:
        if not isinstance(ctx.global_schema, dict):
            errors.append("ExecutionContext.global_schema must be a dict when provided.")
        else:
            modalities = ctx.global_schema.get("global_modalities")
            if modalities is not None and not isinstance(modalities, list):
                errors.append("global_schema.global_modalities must be a list when provided.")
            per_dataset = ctx.global_schema.get("per_dataset")
            if per_dataset is not None and not isinstance(per_dataset, list):
                errors.append("global_schema.per_dataset must be a list when provided.")

    if ctx.global_target is not None and not isinstance(ctx.global_target, str):
        errors.append("ExecutionContext.global_target must be a string when provided.")

    return ContextValidationResult(ok=len(errors) == 0, errors=errors, warnings=warnings)


def ensure_session_context(
    ctx: Optional[ExecutionContext],
    *,
    session_id: Optional[str],
    dataset_snapshot: Optional[Mapping[str, Any]] = None,
) -> ContextValidationResult:
    """Validate and raise ContextValidationError on hard failures."""
    result = validate_session_context(
        ctx,
        session_id=session_id,
        dataset_snapshot=dataset_snapshot,
    )
    if not result.ok:
        raise ContextValidationError(result)
    return result


class ContextValidator:
    """Hard runtime assertions for orchestrator and API preflight checks."""

    @staticmethod
    def require_schema(ctx: Optional[Any], phase: str = "model_selection") -> None:
        if ctx is None:
            raise ValueError(
                f"ContextValidator [{phase}]: ExecutionContext is None. "
                "Run schema detection before this phase."
            )
        schema = getattr(ctx, "global_schema", None)
        if not isinstance(schema, dict) or not schema:
            raise ValueError(
                f"ContextValidator [{phase}]: ctx.global_schema missing/invalid. "
                "Run POST /api/schema/detect first."
            )

    @staticmethod
    def require_modality_consistency(
        ctx: Optional[Any],
        requested_modalities: List[str],
        phase: str = "training",
    ) -> None:
        if ctx is None:
            return
        schema = getattr(ctx, "global_schema", {}) or {}
        schema_modalities = set(schema.get("global_modalities", []) or [])
        requested = set(requested_modalities or [])
        unknown = requested - schema_modalities - {"tabular"}
        if unknown:
            raise ValueError(
                f"ContextValidator [{phase}]: requested modalities {sorted(unknown)} "
                f"not present in schema modalities {sorted(schema_modalities)}."
            )

    @staticmethod
    def require_fusion_consistency(
        fusion_strategy: str,
        modalities: List[str],
        phase: str = "training",
    ) -> None:
        multi_only = {"attention", "graph", "uncertainty", "uncertainty_graph"}
        if fusion_strategy in multi_only and len(modalities or []) < 2:
            raise ValueError(
                f"ContextValidator [{phase}]: fusion '{fusion_strategy}' requires >=2 modalities; "
                f"got {modalities}."
            )

    @staticmethod
    def require_model_selection(
        ctx: Optional[Any],
        phase: str = "training",
    ) -> None:
        if ctx is None:
            return
        model_choices = getattr(ctx, "model_choices", None)
        if not model_choices:
            import logging as _logging

            _logging.getLogger(__name__).warning(
                "ContextValidator [%s]: ctx.model_choices empty; training will rely on heuristics.",
                phase,
            )


def require_context(
    min_stage: str = "ingestion_complete",
    required_fields: Optional[List[str]] = None,
    require_session: bool = False,
):
    """Decorator for API handlers requiring an existing ExecutionContext stage.

    Parameters
    ----------
    min_stage:
        Lowest pipeline stage required for endpoint execution.
    required_fields:
        Optional list of context fields that must be present and non-empty.
        Supports dotted paths (for example: ``"global_schema.per_dataset"``).
    require_session:
        When ``True``, reject requests that do not provide ``session_id``.
    """

    required_fields = list(required_fields or [])

    stage_order = [
        "ingestion_complete",
        "schema_detection",
        "target_detection",
        "global_aggregation",
        "preprocessing_planning",
        "model_selection",
        "training",
        "monitoring",
    ]

    def _read_dotted(payload: Mapping[str, Any], dotted_key: str) -> Any:
        current: Any = payload
        for part in str(dotted_key).split("."):
            if not isinstance(current, Mapping):
                return None
            current = current.get(part)
        return current

    def _is_missing(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, (str, bytes)):
            return len(value) == 0
        if isinstance(value, (list, tuple, dict, set)):
            return len(value) == 0
        return False

    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            session_id = kwargs.get("session_id")
            if session_id is None:
                body_obj = None
                if args:
                    # FastAPI Request often sits first in endpoint signatures.
                    body_obj = args[0]
                else:
                    for value in kwargs.values():
                        if hasattr(value, "json") and (hasattr(value, "path_params") or hasattr(value, "query_params")):
                            body_obj = value
                            break

                if body_obj is None:
                    body_obj = kwargs.get("request")

                if body_obj is None:
                    body_obj = kwargs.get("body")

                if body_obj is not None:
                    session_id = getattr(body_obj, "session_id", None)

                if session_id is None and body_obj is not None:
                    try:
                        path_params = getattr(body_obj, "path_params", None)
                        if isinstance(path_params, Mapping):
                            session_id = path_params.get("session_id")
                    except Exception:
                        session_id = None

                if session_id is None and body_obj is not None:
                    try:
                        query_params = getattr(body_obj, "query_params", None)
                        if query_params is not None:
                            session_id = query_params.get("session_id")
                    except Exception:
                        session_id = None

                if session_id is None and body_obj is not None and hasattr(body_obj, "json"):
                    try:
                        payload = await body_obj.json()
                        if isinstance(payload, Mapping):
                            session_id = payload.get("session_id")
                    except Exception:
                        session_id = None

            if require_session and not session_id:
                from fastapi import HTTPException

                raise HTTPException(
                    status_code=422,
                    detail="session_id is required for this endpoint.",
                )

            if session_id:
                from fastapi import HTTPException

                from database.context_db import context_db

                ctx_data = context_db.load_context(str(session_id))
                if ctx_data is None:
                    raise HTTPException(
                        status_code=422,
                        detail=f"No ExecutionContext for session {session_id}. Run ingestion first.",
                    )

                stage = str(ctx_data.get("pipeline_stage", ""))
                if min_stage in stage_order and stage in stage_order:
                    if stage_order.index(stage) < stage_order.index(min_stage):
                        raise HTTPException(
                            status_code=422,
                            detail=(
                                f"Pipeline stage '{stage}' is before required '{min_stage}'. "
                                "Complete earlier phases first."
                            ),
                        )

                missing_fields: List[str] = []
                for field_name in required_fields:
                    value = _read_dotted(ctx_data, field_name)
                    if _is_missing(value):
                        missing_fields.append(field_name)

                if missing_fields:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            "Session context is missing required artifact(s): "
                            + ", ".join(missing_fields)
                        ),
                    )

            return await fn(*args, **kwargs)

        return wrapper

    return decorator
