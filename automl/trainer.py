"""
APEX Framework – PyTorch Lightning trainer module.

Implements a ``LightningModule`` that:
- Selects loss/metrics automatically from ``problem_type``
- Integrates ``torchmetrics`` (Accuracy+F1 for classification; RMSE+R2 for
  regression)
- Calls ``torch.cuda.synchronize()`` at the end of every training step for
  Windows WDDM TDR safety
- Accepts Optuna-derived hyperparameters (learning_rate, dropout,
  weight_decay) at construction time
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import pytorch_lightning as pl
import torchmetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight multimodal head
# ---------------------------------------------------------------------------

class _MultimodalHead(nn.Module):
    """
    Fusion head that accepts one or more modality embeddings,
    fuses them via the chosen strategy, and projects to ``num_outputs``.

    ``fusion_strategy`` selects the fusion layer:

    - ``"concatenation"`` (default): horizontal ``torch.cat`` via
      :class:`~modelss.fusion.ConcatenationFusion`.
    - ``"attention"``: learned attention-weighted fusion via
      :class:`~modelss.fusion.AttentionFusion`.

    Each modality key in ``input_dims`` must match the keys that
    ``MultimodalPyTorchDataset.__getitem__`` returns (``"tabular"``,
    ``"input_ids"`` / ``"image"`` are handled by their respective encoders
    upstream; this head operates on already-pooled embeddings).
    """

    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_outputs: int = 2,
        dropout: float = 0.1,
        fusion_strategy: str = "concatenation",
    ) -> None:
        super().__init__()
        self._keys = sorted(input_dims.keys())  # only consume declared modalities

        feature_dims = [input_dims[k] for k in self._keys]

        if fusion_strategy == "attention":
            from models.fusion import AttentionFusion
            self.fusion = AttentionFusion(feature_dims)
        else:
            from models.fusion import ConcatenationFusion
            self.fusion = ConcatenationFusion(feature_dims)

        fused_dim = self.fusion.get_output_dim()
        self.layers = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_outputs),
        )

    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        missing = [k for k in self._keys if k not in embeddings]
        if missing:
            raise KeyError(
                f"_MultimodalHead: missing modality keys {missing}. "
                f"Expected all of {self._keys}."
            )
        parts = [embeddings[k].float() for k in self._keys]
        x = self.fusion(parts)
        return self.layers(x)


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------

class ApexLightningModule(pl.LightningModule):
    """
    PyTorch Lightning wrapper for the APEX multimodal head.

    Parameters
    ----------
    model : nn.Module
        The forward model.  Must accept a dict of modality tensors and return
        raw logits (classification) or raw scalar predictions (regression).
    problem_type : str
        One of ``"classification_binary"``, ``"classification_multiclass"``,
        ``"regression"``.
    num_classes : int
        Number of target classes (ignored for regression).
    learning_rate : float
        Initial learning rate passed to AdamW.
    weight_decay : float
        L2 regularisation coefficient for AdamW.
    max_epochs : int
        Total training epochs – used to parameterise CosineAnnealingLR.

    Windows WDDM Safety
    -------------------
    ``torch.cuda.synchronize()`` is called at the end of every
    ``training_step`` to prevent TDR (Timeout Detection & Recovery) stalls
    on WDDM GPU drivers.  On Linux / CPU this is a no-op.
    """

    def __init__(
        self,
        model: nn.Module,
        problem_type: str = "classification_binary",
        num_classes: int = 2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        max_epochs: int = 10,
        image_encoder: Optional[nn.Module] = None,
        text_encoder: Optional[nn.Module] = None,
        tabular_encoder: Optional[nn.Module] = None,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model", "image_encoder", "text_encoder", "tabular_encoder", "class_weights"])

        self.model = model
        self.problem_type = problem_type
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        # Store frozen encoders WITHOUT nn.Module registration so they
        # stay out of state_dict(), parameters(), and the optimizer.
        # This prevents ~540 MB (BERT + ResNet) bloating every checkpoint.
        object.__setattr__(self, "_image_encoder", image_encoder)
        object.__setattr__(self, "_text_encoder", text_encoder)

        # TRAINABLE tabular encoder: registered as a proper nn.Module
        # submodule so its parameters ARE included in self.parameters()
        # and the optimizer.  Created fresh per Optuna trial.
        self.tabular_encoder = tabular_encoder

        # ── Loss function (with optional class weights for imbalanced data) ──
        if problem_type == "classification_binary":
            if class_weights is not None and len(class_weights) >= 2:
                pos_weight = (class_weights[1] / class_weights[0]).unsqueeze(0)
                self.loss_fn: nn.Module = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                logger.info("  BCEWithLogitsLoss pos_weight=%.3f", pos_weight.item())
            else:
                self.loss_fn: nn.Module = nn.BCEWithLogitsLoss()
        elif problem_type == "multilabel_classification":
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif problem_type.startswith("classification"):
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            if class_weights is not None:
                logger.info("  CrossEntropyLoss class weights: %s", class_weights.tolist())
        else:
            self.loss_fn = nn.MSELoss()

        # ── torchmetrics ───────────────────────────────────────────────────
        if problem_type == "multilabel_classification":
            ml_kwargs: Dict[str, Any] = {"task": "multilabel", "num_labels": num_classes}
            self.train_acc = torchmetrics.Accuracy(**ml_kwargs)
            self.val_acc   = torchmetrics.Accuracy(**ml_kwargs)
            self.train_f1  = torchmetrics.F1Score(**ml_kwargs)
            self.val_f1    = torchmetrics.F1Score(**ml_kwargs)
        elif problem_type.startswith("classification"):
            task = "binary" if problem_type == "classification_binary" else "multiclass"
            metric_kwargs: Dict[str, Any] = (
                {"task": task}
                if task == "binary"
                else {"task": task, "num_classes": num_classes}
            )
            self.train_acc = torchmetrics.Accuracy(**metric_kwargs)
            self.val_acc   = torchmetrics.Accuracy(**metric_kwargs)
            self.train_f1  = torchmetrics.F1Score(**metric_kwargs)
            self.val_f1    = torchmetrics.F1Score(**metric_kwargs)
        else:
            self.train_rmse = torchmetrics.MeanSquaredError(squared=False)
            self.val_rmse   = torchmetrics.MeanSquaredError(squared=False)
            self.train_r2   = torchmetrics.R2Score()
            self.val_r2     = torchmetrics.R2Score()

    # ------------------------------------------------------------------
    # Encode raw batch → pooled embeddings
    # ------------------------------------------------------------------

    def _encode_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Transform raw dataset keys into the pooled-embedding keys that
        ``_MultimodalHead`` expects.

        Mapping
        -------
        ``"input_ids"`` + ``"attention_mask"`` → BERT CLS → ``"text_pooled"``  [N, 768]
        ``"image"``                            → ImageEncoder → ``"image_pooled"`` [N, 512]
        ``"tabular"``                          → pass-through  ``"tabular"``    [N, D]

        Missing-modality safety
        -----------------------
        When the head expects a modality key (it's in ``self.model._keys``)
        but the encoder is absent or the raw data key is missing, a 1e-7
        dummy tensor is injected to preserve the concatenated dimension
        and prevent ``KeyError`` in ``_MultimodalHead.forward()``.
        """
        encoded: Dict[str, torch.Tensor] = {}

        # Determine batch size from any available tensor
        N: int = 1
        for v in batch.values():
            if isinstance(v, torch.Tensor) and v.ndim >= 1:
                N = v.shape[0]
                break

        head_keys = getattr(self.model, "_keys", [])

        # ── Tabular: route through trainable encoder (WITH gradient flow) ──
        if "tabular" in batch:
            if self.tabular_encoder is not None:
                encoded["tabular"] = self.tabular_encoder(batch["tabular"])
            else:
                encoded["tabular"] = batch["tabular"]

        # ── Text: use pre-computed embedding or route through frozen BERT ──
        if "text_pooled" in batch:
            encoded["text_pooled"] = batch["text_pooled"]
        elif "input_ids" in batch and self._text_encoder is not None:
            with torch.no_grad():
                outputs = self._text_encoder.transformer(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                )
                cls_token = outputs.last_hidden_state[:, 0, :]
                # Apply projection when hidden_size != 768 (e.g. bert-large)
                if self._text_encoder._projection is not None:
                    cls_token = self._text_encoder._projection(cls_token)
                encoded["text_pooled"] = cls_token

        # ── Image: use pre-computed embedding or route through frozen encoder ─
        if "image_pooled" in batch:
            encoded["image_pooled"] = batch["image_pooled"]
        elif "image" in batch and self._image_encoder is not None:
            with torch.no_grad():
                encoded["image_pooled"] = self._image_encoder(batch["image"])

        # ── Dummy-fill any head-expected keys still missing ───────────
        # Determines reference device from existing encoded tensors or batch
        ref_device = next(
            (t.device for t in encoded.values() if isinstance(t, torch.Tensor)),
            next((t.device for t in batch.values() if isinstance(t, torch.Tensor)), None),
        )
        for key in head_keys:
            if key not in encoded and key != "target":
                # Infer dim from head's input_dims (stored in forward's _keys order)
                dim = self.model.layers[0].in_features  # total_dim fallback
                if key == "text_pooled":
                    dim = 768
                elif key == "image_pooled":
                    dim = 512
                elif key == "tabular":
                    # Use encoder output dim if available, else derive from head
                    if self.tabular_encoder is not None:
                        dim = self.tabular_encoder.get_output_dim()
                    else:
                        other = sum(768 if k == "text_pooled" else 512 if k == "image_pooled" else 0 for k in head_keys if k != "tabular")
                        dim = self.model.layers[0].in_features - other
                encoded[key] = torch.full(
                    (N, dim), 1e-7,
                    dtype=torch.float32,
                    device=ref_device,
                )

        return encoded

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        encoded = self._encode_batch(batch)
        return self.model(encoded)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        targets = batch["target"]
        logits = self(batch)

        loss = self._compute_loss(logits, targets)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        if self.problem_type.startswith("classification") or self.problem_type == "multilabel_classification":
            preds = self._to_preds(logits)
            metric_targets = targets.long() if self.problem_type == "multilabel_classification" else targets
            self.train_acc(preds, metric_targets)
            self.train_f1(preds, metric_targets)
            self.log("train_acc", self.train_acc, prog_bar=True, on_epoch=True)
            self.log("train_f1",  self.train_f1,  prog_bar=False, on_epoch=True)
        else:
            preds_float = logits.squeeze(-1)
            self.train_rmse(preds_float, targets.float())
            self.train_r2(preds_float, targets.float())
            self.log("train_rmse", self.train_rmse, prog_bar=True,  on_epoch=True)
            self.log("train_r2",   self.train_r2,   prog_bar=False, on_epoch=True)

        # Windows WDDM TDR safety – synchronise after each step
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        return loss

    # ------------------------------------------------------------------
    # Validation step
    # ------------------------------------------------------------------

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        targets = batch["target"]
        logits = self(batch)

        loss = self._compute_loss(logits, targets)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

        if self.problem_type.startswith("classification") or self.problem_type == "multilabel_classification":
            preds = self._to_preds(logits)
            metric_targets = targets.long() if self.problem_type == "multilabel_classification" else targets
            self.val_acc(preds, metric_targets)
            self.val_f1(preds, metric_targets)
            self.log("val_acc", self.val_acc, prog_bar=True,  on_epoch=True)
            self.log("val_f1",  self.val_f1,  prog_bar=False, on_epoch=True)
        else:
            preds_float = logits.squeeze(-1)
            self.val_rmse(preds_float, targets.float())
            self.val_r2(preds_float, targets.float())
            self.log("val_rmse", self.val_rmse, prog_bar=True,  on_epoch=True)
            self.log("val_r2",   self.val_r2,   prog_bar=False, on_epoch=True)

        # Windows WDDM TDR safety – synchronise after each step
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        return loss

    # ------------------------------------------------------------------
    # Optimiser + scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    # ------------------------------------------------------------------
    # Device placement override for frozen encoders
    # ------------------------------------------------------------------

    def to(self, *args, **kwargs):
        """Move frozen encoders alongside the registered parameters.

        ``object.__setattr__`` bypasses ``nn.Module.register_module``,
        so Lightning's ``.to(device)`` won't reach them.  This override
        ensures that frozen BERT / ResNet encoders follow the module to
        the correct device.
        """
        result = super().to(*args, **kwargs)
        if self._text_encoder is not None:
            self._text_encoder.to(*args, **kwargs)
        if self._image_encoder is not None:
            self._image_encoder.to(*args, **kwargs)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        if self.problem_type == "classification_binary":
            return self.loss_fn(logits.squeeze(-1).float(), targets.float())
        elif self.problem_type == "multilabel_classification":
            return self.loss_fn(logits.float(), targets.float())
        elif self.problem_type.startswith("classification"):
            return self.loss_fn(logits, targets.long())
        else:
            return self.loss_fn(logits.squeeze(-1).float(), targets.float())

    def _to_preds(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert raw logits to predictions."""
        if self.problem_type == "classification_binary":
            return (torch.sigmoid(logits.squeeze(-1)) >= 0.5).long()
        elif self.problem_type == "multilabel_classification":
            return (torch.sigmoid(logits) >= 0.5).long()
        return logits.argmax(dim=-1)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_trainer(
    problem_type: str,
    num_classes: int,
    input_dims: Dict[str, int],
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    dropout: float = 0.1,
    max_epochs: int = 10,
    hidden_dim: int = 256,
    image_encoder: Optional[nn.Module] = None,
    text_encoder: Optional[nn.Module] = None,
    tabular_encoder: Optional[nn.Module] = None,
    class_weights: Optional[torch.Tensor] = None,
    fusion_strategy: str = "concatenation",
) -> ApexLightningModule:
    """
    Build an :class:`ApexLightningModule` wrapping a fresh multimodal head.

    Parameters
    ----------
    input_dims : dict
        Mapping ``{modality_key: embedding_dim}`` – keys present in each
        sample dict produced by ``MultimodalPyTorchDataset``.
    image_encoder : nn.Module | None
        Frozen ``ImageEncoder`` instance shared across Optuna trials.
    text_encoder : nn.Module | None
        Frozen ``TextEncoder`` instance shared across Optuna trials.
    tabular_encoder : nn.Module | None
        Trainable tabular encoder, freshly instantiated per trial.
    fusion_strategy : str
        ``"concatenation"`` (default) or ``"attention"``.

    Returns
    -------
    ApexLightningModule ready to be handed to ``pytorch_lightning.Trainer``.
    """
    # Binary classification uses BCEWithLogitsLoss which expects scalar
    # logits [N] (squeezed from [N,1]), NOT [N,2].  Only multiclass needs
    # num_classes outputs for CrossEntropyLoss.
    if problem_type in ("regression", "classification_binary"):
        num_outputs = 1
    else:
        num_outputs = num_classes
    head = _MultimodalHead(
        input_dims=input_dims,
        hidden_dim=hidden_dim,
        num_outputs=num_outputs,
        dropout=dropout,
        fusion_strategy=fusion_strategy,
    )
    return ApexLightningModule(
        model=head,
        problem_type=problem_type,
        num_classes=num_classes,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        tabular_encoder=tabular_encoder,
        class_weights=class_weights,
    )
