from __future__ import annotations

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import automl.trainer as trainer_mod
from automl.trainer import ApexLightningModule, _MultimodalHead


def test_multiclass_label_smoothing_uses_device_aligned_class_weights(monkeypatch) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    captured: dict[str, torch.device | None] = {"weight_device": None}

    def fake_cross_entropy(logits, targets, weight=None, label_smoothing=0.0):
        captured["weight_device"] = None if weight is None else weight.device
        return logits.sum() * 0 + torch.tensor(0.25, device=logits.device)

    monkeypatch.setattr(trainer_mod.F, "cross_entropy", fake_cross_entropy)

    module = ApexLightningModule(
        model=_MultimodalHead(input_dims={"tabular": 4}, num_outputs=3),
        problem_type="classification_multiclass",
        num_classes=3,
        class_weights=torch.tensor([1.0, 2.0, 3.0]),
        label_smoothing=0.1,
    )

    logits = torch.randn(4, 3, device=device)
    targets = torch.tensor([0, 1, 2, 1], device=device)
    loss = module._compute_loss(logits, targets)

    assert captured["weight_device"] == logits.device
    assert loss.device == logits.device


def test_binary_label_smoothing_uses_device_aligned_pos_weight(monkeypatch) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    captured: dict[str, torch.device | None] = {"pos_weight_device": None}

    def fake_bce_with_logits(logits, targets, pos_weight=None, reduction="mean"):
        captured["pos_weight_device"] = None if pos_weight is None else pos_weight.device
        return logits.sum() * 0 + torch.tensor(0.5, device=logits.device)

    monkeypatch.setattr(trainer_mod.F, "binary_cross_entropy_with_logits", fake_bce_with_logits)

    module = ApexLightningModule(
        model=_MultimodalHead(input_dims={"tabular": 4}, num_outputs=1),
        problem_type="classification_binary",
        num_classes=2,
        class_weights=torch.tensor([0.75, 1.35]),
        label_smoothing=0.1,
    )

    logits = torch.randn(4, 1, device=device)
    targets = torch.tensor([0, 1, 1, 0], device=device)
    loss = module._compute_loss(logits, targets)

    assert module._binary_pos_weight is not None
    assert captured["pos_weight_device"] == logits.device
    assert loss.device == logits.device
