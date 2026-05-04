"""Thread-safe pipeline state slots for cross-phase coordination."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Dict, List, Optional


@dataclass
class PipelineState:
    """Mutable state container with lock-guarded slot operations."""

    slots: Dict[str, Any] = field(default_factory=dict)
    phase_timings: Dict[str, float] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def set_slot(self, key: str, value: Any) -> None:
        with self._lock:
            self.slots[key] = value

    def merge_slot(self, key: str, value: Dict[str, Any]) -> None:
        with self._lock:
            current = self.slots.get(key, {})
            if not isinstance(current, dict):
                current = {}
            current.update(value)
            self.slots[key] = current

    def get_slot(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if key not in self.slots:
                return default
            return deepcopy(self.slots[key])

    def set_phase_timing(self, phase_name: str, duration_seconds: float) -> None:
        with self._lock:
            self.phase_timings[str(phase_name)] = float(duration_seconds)

    def add_event(
        self,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
        phase: Optional[str] = None,
    ) -> None:
        record = {
            "type": str(event_type),
            "phase": str(phase) if phase is not None else None,
            "payload": dict(payload or {}),
        }
        with self._lock:
            self.events.append(record)
            self.events = self.events[-500:]

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "slots": deepcopy(self.slots),
                "phase_timings": dict(self.phase_timings),
                "events": deepcopy(self.events),
            }
