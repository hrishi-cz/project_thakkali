"""
Elastic Weight Consolidation (EWC) for continual learning.

Prevents catastrophic forgetting when retraining a model on new data by
penalising large deviations from the previous task's optimal weights,
weighted by the Fisher information matrix (which measures how important
each weight is for the previous task).

Theory (Kirkpatrick et al., 2017; surveyed in [8] IEEE TNNLS 2024):
    L_total = L_new(θ) + λ/2 · Σ_i F_i (θ_i - θ*_i)²

Where:
  θ*   = optimal weights from previous training run
  F_i  = diagonal Fisher information for parameter θ_i
  λ    = EWC regularisation strength (default 400.0 — Kirkpatrick et al.)

Usage in retraining:
    ewc = EWC(model, dataloader, criterion, device, lambda_ewc=400.0)
    # ... during new training ...
    loss = criterion(logits, targets) + ewc.penalty(model)

Paper reference:
    [8] "Continual learning: A comprehensive survey",
        IEEE Transactions on Neural Networks, 2024.
    Kirkpatrick et al. (2017), "Overcoming catastrophic forgetting in
        neural networks", PNAS.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class EWC:
    """
    Elastic Weight Consolidation.

    Computes the diagonal Fisher information matrix on a reference dataset
    after initial training.  Provides a regularisation penalty term to be
    added to the loss during subsequent retraining on new data.

    Parameters
    ----------
    model : nn.Module
        The trained model whose weights we want to protect.
    dataloader : DataLoader | None
        Reference data used to compute Fisher.  If None, Fisher is all-zeros
        (EWC penalty = 0, effectively disabled).
    criterion : callable | None
        Loss function ``criterion(logits, targets) → scalar``.
        If None, CrossEntropyLoss is used for classification.
    device : torch.device | str
        Device for computation.
    lambda_ewc : float
        Regularisation strength λ (Kirkpatrick et al. use 400.0).
    n_fisher_samples : int
        Number of batches to use for Fisher estimation (fewer = faster).
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: Optional[Any] = None,
        criterion: Optional[Any] = None,
        device: Optional[Any] = None,
        lambda_ewc: float = 400.0,
        n_fisher_samples: int = 200,
    ) -> None:
        self.lambda_ewc = float(lambda_ewc)
        self._params_star: Dict[str, torch.Tensor] = {}   # θ*
        self._fisher: Dict[str, torch.Tensor] = {}        # F_i

        if device is None:
            device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
        self.device = torch.device(device)

        # Snapshot optimal weights θ*
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._params_star[name] = param.data.clone().to(self.device)

        # Compute diagonal Fisher if dataloader is provided
        if dataloader is not None:
            try:
                self._compute_fisher(model, dataloader, criterion, n_fisher_samples)
                n_params = sum(f.numel() for f in self._fisher.values())
                logger.info(
                    "EWC: Fisher computed over %d parameter entries from %d samples",
                    n_params, n_fisher_samples,
                )
            except Exception as exc:
                logger.warning("EWC: Fisher computation failed (%s); penalty = 0", exc)
        else:
            logger.debug("EWC: no dataloader — Fisher not computed; penalty = 0")

    def _compute_fisher(
        self,
        model: nn.Module,
        dataloader: Any,
        criterion: Optional[Any],
        n_fisher_samples: int,
    ) -> None:
        """Compute diagonal Fisher via empirical Fisher (squared gradients)."""
        model = model.to(self.device)
        model.eval()

        _criterion = criterion or nn.CrossEntropyLoss()

        # Initialise Fisher accumulators
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._fisher[name] = torch.zeros_like(param.data)

        n_batches = 0
        for batch in dataloader:
            if n_batches >= n_fisher_samples:
                break

            try:
                # Support both dict-style and tuple-style batches
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0], batch[-1]
                elif isinstance(batch, dict):
                    targets = batch.pop("target", batch.pop("label", None))
                    inputs = batch
                else:
                    continue

                if targets is None:
                    continue

                # Move to device
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(self.device)
                elif isinstance(inputs, dict):
                    inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                              for k, v in inputs.items()}

                if isinstance(targets, torch.Tensor):
                    targets = targets.to(self.device)

                model.zero_grad()

                # Forward pass
                if isinstance(inputs, dict):
                    logits = model(**inputs)
                else:
                    logits = model(inputs)

                # Handle regression vs classification
                if logits.dim() == 1 or logits.shape[-1] == 1:
                    loss = F.mse_loss(logits.squeeze(), targets.float())
                else:
                    loss = _criterion(logits, targets.long())

                loss.backward()

                # Accumulate squared gradients (diagonal Fisher)
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        self._fisher[name] += param.grad.data.pow(2).detach()

                n_batches += 1

            except Exception as batch_exc:
                logger.debug("EWC Fisher: batch %d failed: %s", n_batches, batch_exc)
                continue

        # Normalise by number of batches
        if n_batches > 0:
            for name in self._fisher:
                self._fisher[name] /= float(n_batches)

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute the EWC penalty term: λ/2 · Σ_i F_i (θ_i - θ*_i)².

        Returns a scalar tensor on the model's device.
        Gracefully returns 0.0 if Fisher was not computed.
        """
        if not self._fisher or not self._params_star:
            return torch.tensor(0.0, device=self.device)

        penalty = torch.tensor(0.0, device=self.device)
        for name, param in model.named_parameters():
            if name in self._fisher and name in self._params_star:
                fisher = self._fisher[name].to(param.device)
                star = self._params_star[name].to(param.device)
                penalty = penalty + (fisher * (param - star).pow(2)).sum()

        return (self.lambda_ewc / 2.0) * penalty

    def state_dict(self) -> Dict[str, Any]:
        """Serialise EWC state for checkpointing."""
        return {
            "lambda_ewc": self.lambda_ewc,
            "params_star": {k: v.cpu() for k, v in self._params_star.items()},
            "fisher": {k: v.cpu() for k, v in self._fisher.items()},
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any], device: Optional[Any] = None) -> "EWC":
        """Restore from a serialised state dict."""
        obj = cls.__new__(cls)
        obj.lambda_ewc = float(state.get("lambda_ewc", 400.0))
        obj.device = torch.device(device or "cpu")
        obj._params_star = {k: v.to(obj.device) for k, v in state.get("params_star", {}).items()}
        obj._fisher = {k: v.to(obj.device) for k, v in state.get("fisher", {}).items()}
        return obj
