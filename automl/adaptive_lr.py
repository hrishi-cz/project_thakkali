"""Adaptive learning-rate scheduler driven by TrialIntelligence feedback."""

from __future__ import annotations

import math
from typing import Any, Optional

import torch


class AdaptiveLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing baseline with fit-aware multiplier.

    Every epoch, reads LossWeightScheduler.last_analysis and applies:
    - overfitting  -> LR * 0.70
    - underfitting -> LR * 1.05
    - good/unknown -> LR * 1.00
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_max: int,
        loss_weight_scheduler: Optional[Any] = None,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.T_max = max(1, int(T_max))
        self.eta_min = float(eta_min)
        self._loss_weight_scheduler = loss_weight_scheduler
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        progress = min(max(self.last_epoch, 0), self.T_max)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress / self.T_max))

        analysis = {}
        if self._loss_weight_scheduler is not None:
            analysis = getattr(self._loss_weight_scheduler, "last_analysis", {}) or {}
        fit_type = str(analysis.get("fit_type", "good")).lower()
        multiplier = {"overfitting": 0.70, "underfitting": 1.05}.get(fit_type, 1.0)

        lrs: list[float] = []
        for base_lr in self.base_lrs:
            cosine_lr = self.eta_min + (base_lr - self.eta_min) * cosine_factor
            lrs.append(max(self.eta_min, float(cosine_lr) * multiplier))
        return lrs
