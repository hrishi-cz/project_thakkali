"""Regression tests for monitoring.performance_tracker."""

import uuid

import numpy as np
import pytest

from monitoring.performance_tracker import PerformanceTracker


def test_history_is_bounded_by_max_history() -> None:
    tracker = PerformanceTracker(model_id=f"tracker_{uuid.uuid4().hex}", max_history=3)

    for idx in range(5):
        tracker.log_prediction(
            prediction=np.array([0.2, 0.8]),
            actual=np.array([0.0, 1.0]),
            timestamp=f"2026-01-01T00:00:0{idx}",
        )

    assert len(tracker.history) == 3


def test_log_prediction_raises_for_row_mismatch() -> None:
    tracker = PerformanceTracker(model_id=f"tracker_{uuid.uuid4().hex}", max_history=10)

    with pytest.raises(ValueError):
        tracker.log_prediction(
            prediction=np.array([0.2, 0.8]),
            actual=np.array([1.0]),
        )
