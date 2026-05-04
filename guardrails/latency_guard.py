"""Latency guard utilities for bounded runtime calls."""

from __future__ import annotations

import concurrent.futures
import time
from typing import Any, Callable, Optional


class LatencyGuard:
    """Context manager and helper methods for wall-clock budgets."""

    def __init__(self, budget_s: float, fallback_fn: Optional[Callable[[], Any]] = None) -> None:
        self.budget_s = float(max(0.0, budget_s))
        self._fallback_fn = fallback_fn
        self._start: Optional[float] = None

    def __enter__(self) -> "LatencyGuard":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def check(self) -> Any:
        """Validate elapsed time against the configured budget."""
        if self._start is None:
            return None
        elapsed = time.perf_counter() - self._start
        if elapsed <= self.budget_s:
            return None
        if self._fallback_fn is not None:
            return self._fallback_fn()
        raise TimeoutError(f"LatencyGuard: exceeded {self.budget_s:.2f}s budget (elapsed={elapsed:.2f}s)")

    @staticmethod
    def timed(func: Callable[..., Any], budget_s: float, *args: Any, **kwargs: Any) -> Any:
        """Execute a callable with a strict timeout budget."""
        timeout = float(max(0.0, budget_s))
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError as exc:
                raise TimeoutError(f"Function exceeded latency budget ({timeout:.2f}s)") from exc
