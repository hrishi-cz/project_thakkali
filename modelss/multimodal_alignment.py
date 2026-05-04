"""Utilities for multimodal representation alignment diagnostics and loss."""

from __future__ import annotations

from typing import Dict, List

import torch


class MultimodalAligner:
    """Compute alignment loss and pairwise cosine summaries across modalities."""

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = float(eps)

    def alignment_loss(self, features: List[torch.Tensor]) -> torch.Tensor:
        if len(features) < 2:
            ref = features[0] if features else torch.zeros(1)
            return torch.zeros((), device=ref.device, dtype=ref.dtype)

        losses: List[torch.Tensor] = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                a = features[i].float()
                b = features[j].float()
                # Pool token sequences to (N, D) before element-wise ops.
                # ULA token_mode passes (N, T, D) / (N, P, D) — mean-pool for alignment.
                if a.ndim == 3:
                    a = a.mean(dim=1)
                if b.ndim == 3:
                    b = b.mean(dim=1)
                dim = min(a.shape[-1], b.shape[-1])
                if dim <= 0:
                    continue
                a = a[..., :dim]
                b = b[..., :dim]
                a = a / (a.norm(dim=-1, keepdim=True) + self.eps)
                b = b / (b.norm(dim=-1, keepdim=True) + self.eps)
                cosine = (a * b).sum(dim=-1)
                losses.append((1.0 - cosine).mean())

        if not losses:
            return torch.zeros((), device=features[0].device, dtype=features[0].dtype)
        return torch.stack(losses).mean()

    def alignment_report(self, named_features: Dict[str, torch.Tensor]) -> Dict[str, float]:
        keys = [k for k, v in named_features.items() if isinstance(v, torch.Tensor) and v.ndim >= 2]
        report: Dict[str, float] = {}
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a = named_features[keys[i]].detach().float()
                b = named_features[keys[j]].detach().float()
                if a.ndim == 3:
                    a = a.mean(dim=1)
                if b.ndim == 3:
                    b = b.mean(dim=1)
                dim = min(a.shape[-1], b.shape[-1])
                if dim <= 0:
                    continue
                a = a[..., :dim]
                b = b[..., :dim]
                a = a / (a.norm(dim=-1, keepdim=True) + self.eps)
                b = b / (b.norm(dim=-1, keepdim=True) + self.eps)
                score = float((a * b).sum(dim=-1).mean().item())
                report[f"{keys[i]}__{keys[j]}"] = score
        return report
