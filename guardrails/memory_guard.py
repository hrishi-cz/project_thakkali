"""Memory guard for GPU and system RAM pressure checks."""

from __future__ import annotations

import logging
from typing import Dict

import torch

logger = logging.getLogger(__name__)


class MemoryGuard:
    """Monitor VRAM and RAM usage and trigger non-fatal mitigation."""

    def __init__(self, vram_threshold_pct: float = 0.92, ram_threshold_pct: float = 0.90) -> None:
        self.vram_threshold = float(max(0.0, min(1.0, vram_threshold_pct)))
        self.ram_threshold = float(max(0.0, min(1.0, ram_threshold_pct)))

    def check_vram(self) -> Dict[str, float]:
        if not torch.cuda.is_available():
            return {"available": 0.0, "used_pct": 0.0, "critical": 0.0}

        total = float(torch.cuda.get_device_properties(0).total_memory)
        reserved = float(torch.cuda.memory_reserved(0))
        used_pct = (reserved / total) if total > 0 else 0.0
        critical = 1.0 if used_pct > self.vram_threshold else 0.0
        return {
            "available": 1.0,
            "used_pct": used_pct,
            "critical": critical,
        }

    def check_ram(self) -> Dict[str, float]:
        try:
            import psutil

            vm = psutil.virtual_memory()
            used_pct = float(vm.percent) / 100.0
            critical = 1.0 if used_pct > self.ram_threshold else 0.0
            return {
                "available": 1.0,
                "used_pct": used_pct,
                "critical": critical,
            }
        except Exception:
            return {"available": 0.0, "used_pct": 0.0, "critical": 0.0}

    def maybe_clear_cache(self) -> bool:
        """Clear CUDA cache when VRAM pressure is above threshold."""
        status = self.check_vram()
        if status.get("available") and status.get("critical"):
            try:
                torch.cuda.empty_cache()
                logger.warning(
                    "MemoryGuard: VRAM pressure %.1f%% exceeded threshold %.1f%%; cache cleared",
                    float(status.get("used_pct", 0.0)) * 100.0,
                    self.vram_threshold * 100.0,
                )
                return True
            except Exception:
                return False
        return False
