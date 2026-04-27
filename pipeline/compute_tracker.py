"""Compute budget tracking for AutoVision training runs.

Tracks FLOPs, peak VRAM, wall-clock GPU-hours, and trainable parameter counts
(LoRA-aware) for each Optuna trial. Persists to diary/results/{run_id}_compute.json.

Usage::

    tracker = ComputeTracker(run_id="trial_42")
    tracker.start()
    # ... training ...
    tracker.stop()
    tracker.log_model(model)
    tracker.save()
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_RESULTS_DIR = Path(__file__).resolve().parent.parent / "diary" / "results"


class ComputeTracker:
    """Per-trial compute budget tracker.

    Measures:
    - Wall-clock seconds (always)
    - Peak VRAM in MB (when CUDA available)
    - Trainable parameter count (LoRA-aware: separates backbone vs adapter params)
    - FLOPs estimate via fvcore when available
    """

    def __init__(self, run_id: str = "unknown") -> None:
        self.run_id = run_id
        self._start_time: Optional[float] = None
        self._elapsed_s: float = 0.0
        self._peak_vram_mb: float = 0.0
        self._total_params: int = 0
        self._trainable_params: int = 0
        self._lora_params: int = 0
        self._backbone_params: int = 0
        self._flops: Optional[float] = None
        self._extra: Dict[str, Any] = {}

    def start(self) -> None:
        self._start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def stop(self) -> None:
        if self._start_time is not None:
            self._elapsed_s = time.time() - self._start_time
        if torch.cuda.is_available():
            self._peak_vram_mb = (
                torch.cuda.max_memory_allocated() / 1024 / 1024
            )

    def log_model(self, model: nn.Module, sample_input: Optional[Any] = None) -> None:
        """Count parameters and optionally estimate FLOPs."""
        try:
            from modelss.adapters.lora import lora_parameters as _lp
            lora_ps = set(id(p) for p in _lp(model))
        except Exception:
            lora_ps = set()

        total = 0
        trainable = 0
        lora = 0
        for p in model.parameters():
            n = p.numel()
            total += n
            if p.requires_grad:
                trainable += n
                if id(p) in lora_ps:
                    lora += n

        self._total_params = total
        self._trainable_params = trainable
        self._lora_params = lora
        self._backbone_params = total - trainable

        if sample_input is not None:
            try:
                from fvcore.nn import FlopCountAnalysis
                flop_analysis = FlopCountAnalysis(model, sample_input)
                self._flops = float(flop_analysis.total())
            except Exception as _flop_exc:
                logger.debug("FLOPs estimation skipped: %s", _flop_exc)

    def add(self, key: str, value: Any) -> None:
        """Store an arbitrary extra metric."""
        self._extra[key] = value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "wall_clock_seconds": round(self._elapsed_s, 3),
            "gpu_hours": round(self._elapsed_s / 3600, 6),
            "peak_vram_mb": round(self._peak_vram_mb, 2),
            "total_params": self._total_params,
            "trainable_params": self._trainable_params,
            "lora_params": self._lora_params,
            "backbone_frozen_params": self._backbone_params,
            "flops": self._flops,
            **self._extra,
        }

    def save(self) -> Path:
        """Write compute budget to diary/results/{run_id}_compute.json."""
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = _RESULTS_DIR / f"{self.run_id}_compute.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)
        logger.info("Compute budget saved: %s", out_path)
        return out_path

    @staticmethod
    def load_all() -> list:
        """Load all compute budget files from diary/results/."""
        records = []
        for p in sorted(_RESULTS_DIR.glob("*_compute.json")):
            try:
                with open(p, encoding="utf-8") as fh:
                    records.append(json.load(fh))
            except Exception:
                pass
        return records
