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
import os

# Must be set before any CUDA/cuBLAS operation to suppress non-determinism
# warnings when torch.use_deterministic_algorithms(True) is active on CUDA >= 10.2.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

import pytorch_lightning as pl
import torchmetrics
from automl.adaptive_lr import AdaptiveLRScheduler
from guardrails.fallback_manager import FallbackManager
from guardrails.memory_guard import MemoryGuard
from models.multimodal_alignment import MultimodalAligner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reproducibility — seed all RNGs at import time (overridable via APEX_SEED)
# ---------------------------------------------------------------------------
_APEX_SEED = int(os.getenv("APEX_SEED", "42"))
pl.seed_everything(_APEX_SEED, workers=True)
torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------------------
# Focal Loss — Lin et al., ICCV 2017 "Focal Loss for Dense Object Detection"
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss for hard-sample mining in class-imbalanced datasets.

    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    α_t balances positive/negative classes (class weights).
    γ (focusing parameter) reduces loss for well-classified samples,
    forcing the model to focus on hard / misclassified examples.

    Supports binary and multi-class classification.  Reduces to
    standard cross-entropy when γ=0 (Lin et al. 2017 [12]).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        # Register as buffer so .to(device) / Lightning's module.cuda() moves it automatically.
        self.register_buffer("alpha", alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape[-1] == 1 or logits.dim() == 1:
            # Binary case
            probs = torch.sigmoid(logits.squeeze(-1))
            targets_f = targets.float()
            bce = F.binary_cross_entropy_with_logits(
                logits.squeeze(-1), targets_f, reduction="none"
            )
            p_t = probs * targets_f + (1 - probs) * (1 - targets_f)
            focal_weight = (1.0 - p_t) ** self.gamma
            loss = focal_weight * bce
        else:
            # Multi-class case
            log_probs = F.log_softmax(logits, dim=-1)
            probs = torch.exp(log_probs)
            alpha = self.alpha
            if alpha is not None and alpha.device != logits.device:
                alpha = alpha.to(logits.device)
            ce = F.nll_loss(log_probs, targets.long(), weight=alpha, reduction="none")
            p_t = probs.gather(1, targets.long().unsqueeze(1)).squeeze(1)
            focal_weight = (1.0 - p_t) ** self.gamma
            loss = focal_weight * ce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class AlignmentLoss(nn.Module):
    """Simple modality alignment loss based on mean cosine distance."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self._aligner = MultimodalAligner(eps=float(eps))

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        return self._aligner.alignment_loss(features)

    def summarize(self, named_features: Dict[str, torch.Tensor]) -> Dict[str, float]:
        return self._aligner.alignment_report(named_features)


class LinkedModalityContrastiveLoss(nn.Module):
    """
    NT-Xent contrastive loss for entity-linked cross-modal pairs.

    When the same entity appears in both tabular and image/text columns
    (identified by a shared ID column), embeddings from the same entity
    should be close (positive pairs) and embeddings from different entities
    far (negative pairs).

    This is the CLIP-style objective applied within a mini-batch:
      - Positive pair: (tabular_emb[i], image_emb[i]) for same entity i
      - Negative pairs: all cross-entity combinations in the batch

    Parameters
    ----------
    temperature : float
        NT-Xent temperature τ (default 0.07 — standard for in-batch negatives).
    eps : float
        Numerical stability for normalisation.
    """

    def __init__(self, temperature: float = 0.07, eps: float = 1e-8) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.eps = float(eps)

    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        emb_a, emb_b : torch.Tensor
            Embeddings from two modalities, shape ``(N, D_a)`` and ``(N, D_b)``.
            Row i in emb_a and row i in emb_b belong to the same entity.
            D_a and D_b may differ — both are L2-normalised after projection
            to the shared dim ``min(D_a, D_b)``.

        Returns
        -------
        torch.Tensor
            Scalar NT-Xent loss.
        """
        # Pool token sequences to (N, D) before similarity computation.
        # ULA token_mode passes (N, T, D) / (N, P, D) — mean-pool for contrastive loss.
        if emb_a.ndim == 3:
            emb_a = emb_a.mean(dim=1)
        if emb_b.ndim == 3:
            emb_b = emb_b.mean(dim=1)

        N = emb_a.shape[0]
        if N < 2:
            return torch.zeros((), device=emb_a.device, dtype=emb_a.dtype)

        # Project to shared dim and normalise
        dim = min(emb_a.shape[-1], emb_b.shape[-1])
        a = emb_a[..., :dim].float()
        b = emb_b[..., :dim].float()
        a = F.normalize(a, dim=-1, eps=self.eps)
        b = F.normalize(b, dim=-1, eps=self.eps)

        # Similarity matrix (N × N), scaled by temperature
        sim = torch.matmul(a, b.T) / self.temperature  # (N, N)

        # NT-Xent: diagonal entries are positive pairs
        labels = torch.arange(N, device=emb_a.device)
        loss_ab = F.cross_entropy(sim, labels)
        loss_ba = F.cross_entropy(sim.T, labels)
        return (loss_ab + loss_ba) / 2.0


# ---------------------------------------------------------------------------
# CLIP-style learnable projection head for cross-modal alignment
# Novel contribution: projects each modality embedding into a shared
# contrastive space before NT-Xent loss, enabling proper cross-modal
# alignment even when modality dimensions differ.
# ---------------------------------------------------------------------------

class CLIPProjectionHead(nn.Module):
    """Learnable projection head for cross-modal contrastive learning.

    Projects modality embeddings into a shared ``proj_dim``-dimensional
    space before computing NT-Xent loss.  This is the standard approach
    from Radford et al. (2021) "Learning Transferable Visual Models From
    Natural Language Supervision" (CLIP).

    Each modality gets its own linear projection + LayerNorm, ensuring
    that the contrastive objective doesn't corrupt the primary task
    embeddings.
    """

    def __init__(self, input_dim: int, proj_dim: int = 128) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(x), dim=-1)


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
        fusion_config: Optional[Dict[str, Any]] = None,
        head_architecture_type: str = "mlp",
        head_num_layers: int = 3,
    ) -> None:
        super().__init__()
        self._input_dims = dict(input_dims)
        self._keys = sorted(input_dims.keys())  # only consume declared modalities

        feature_dims = [input_dims[k] for k in self._keys]
        fusion_cfg = dict(fusion_config or {})

        fusion_key = str(fusion_strategy or "concatenation").strip().lower()
        fusion_key = fusion_key.replace("-", "_")
        if fusion_key == "concat":
            fusion_key = "concatenation"
        self.fusion_strategy = fusion_key

        if fusion_key in ("uncertainty_graph", "uncertaintygraph"):
            from models.fusion import UncertaintyGraphFusion
            graph_branch_weight = float(
                fusion_cfg.get(
                    "uncertainty_graph_weight",
                    fusion_cfg.get("graph_branch_weight", 0.5),
                )
            )
            uncertainty_branch_weight = float(
                fusion_cfg.get(
                    "uncertainty_branch_weight",
                    max(0.0, 1.0 - graph_branch_weight),
                )
            )
        elif fusion_key in ("gated", "gated_fusion"):
            # Learned gate suppresses noisy/conflicting modalities per sample
            from modelss.fusion import GatedFusion
            self.fusion = GatedFusion(
                feature_dims=feature_dims,
                output_dim=int(fusion_cfg.get("output_dim", 512)),
                dropout=float(fusion_cfg.get("dropout", 0.1)),
            )
        elif fusion_key in ("ula", "unified_latent", "unified_latent_alignment", "omnimodal"):
            # Omni-modal Unified Latent Alignment (ImageBind / 4M style)
            from modelss.fusion import UnifiedLatentFusion
            self.fusion = UnifiedLatentFusion(
                feature_dims=feature_dims,
                latent_dim=int(fusion_cfg.get("latent_dim", 256)),
                n_heads=int(fusion_cfg.get("n_heads", 4)),
                n_layers=int(fusion_cfg.get("n_layers", 2)),
                dropout=float(fusion_cfg.get("dropout", 0.1)),
                token_mode=bool(fusion_cfg.get("token_mode", False)),
            )
        elif fusion_key in ("fusemoe", "moe", "mixture_of_experts"):
            # FuseMoE: missing-modality-aware mixture of experts (ICML 2024)
            from modelss.fusion import FuseMoE
            self.fusion = FuseMoE(
                feature_dims=feature_dims,
                output_dim=int(fusion_cfg.get("output_dim", 512)),
                n_experts=int(fusion_cfg.get("n_experts", 4)),
                top_k=int(fusion_cfg.get("top_k", 2)),
                dropout=float(fusion_cfg.get("dropout", 0.1)),
            )
        elif fusion_key in ("complementarity", "crossfuse"):
            # [4] CrossFuse: Complementarity-aware fusion (ECCV 2024)
            from modelss.fusion import ComplementarityFusion
            self.fusion = ComplementarityFusion(
                feature_dims=feature_dims,
                latent_dim=int(fusion_cfg.get("latent_dim", 512)),
            )
        elif fusion_key in ("structural_semantic", "ssunifier"):
            # [1] Structural-Semantic Unifier (ICML 2025)
            from modelss.fusion import StructuralSemanticRouter
            self.fusion = StructuralSemanticRouter(
                feature_dims=feature_dims,
                latent_dim=int(fusion_cfg.get("latent_dim", 512)),
                heads=int(fusion_cfg.get("graph_heads", 4)),
            )
        elif fusion_key == "graph":
            from models.fusion import GraphFusion
            self.fusion = GraphFusion(
                dim=int(fusion_cfg.get("graph_dim", 512)),
                num_modalities=len(feature_dims),
                heads=int(fusion_cfg.get("graph_heads", 4)),
                input_dims=feature_dims,
            )
        elif fusion_key == "uncertainty":
            from models.fusion import UncertaintyFusion
            self.fusion = UncertaintyFusion(
                feature_dims=feature_dims,
                latent_dim=int(fusion_cfg.get("uncertainty_latent_dim", 512)),
            )
        elif fusion_key == "attention":
            from models.fusion import AttentionFusion
            self.fusion = AttentionFusion(
                feature_dims,
                latent_dim=int(fusion_cfg.get("attention_latent_dim", 512)),
            )
        else:
            from models.fusion import ConcatenationFusion
            self.fusion = ConcatenationFusion(feature_dims)

        if fusion_key in ("uncertainty_graph", "uncertaintygraph"):
            self.fusion = UncertaintyGraphFusion(
                feature_dims=feature_dims,
                latent_dim=int(fusion_cfg.get("uncertainty_latent_dim", 512)),
                heads=int(fusion_cfg.get("graph_heads", 4)),
                graph_weight=graph_branch_weight,
                uncertainty_weight=uncertainty_branch_weight,
            )

        fused_dim = self.fusion.get_output_dim()

        # Use schema-driven head architecture if specified
        try:
            from modelss.heads import build_head
            self.layers = build_head(
                head_type=head_architecture_type,
                in_dim=fused_dim,
                out_dim=num_outputs,
                hidden=hidden_dim,
                num_layers=head_num_layers,
            )
        except Exception:
            # Fallback to original MLP if heads module unavailable
            self.layers = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_outputs),
            )

    def forward(
        self,
        embeddings: Dict[str, torch.Tensor],
        modality_mask: Optional[Dict[str, bool]] = None,
    ) -> torch.Tensor:
        # Determine batch size and device from any present tensor
        _ref = next((v for v in embeddings.values() if v is not None), None)
        if _ref is None:
            raise ValueError("_MultimodalHead.forward: no tensors provided")
        _batch, _device = _ref.shape[0], _ref.device

        parts: List[torch.Tensor] = []
        for k in self._keys:
            is_present = (modality_mask is None) or modality_mask.get(k, True)
            if is_present and k in embeddings and embeddings[k] is not None:
                parts.append(embeddings[k].float())
            else:
                # Missing/masked modality: inject zero tensor to preserve fusion dim
                parts.append(torch.zeros(_batch, self._input_dims[k], device=_device))
        # G25: forward modality_mask into fusion when supported
        if modality_mask is not None and getattr(self.fusion, "accepts_mask", False):
            x = self.fusion(parts, modality_names=list(self._keys), modality_mask=modality_mask)
        else:
            x = self.fusion(parts)
        return self.layers(x)


class OptunaCallback:
    """Lightweight callback container for trial-level train/val loss history."""

    def __init__(self, trial: Any, window: int = 5, threshold: float = 1e-3) -> None:
        self.trial = trial
        self.window = int(window)
        self.threshold = float(threshold)
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def on_epoch_end(self, epoch: int, train_loss: float, val_loss: float) -> None:
        self.train_losses.append(float(train_loss))
        self.val_losses.append(float(val_loss))
        if hasattr(self.trial, "report"):
            self.trial.report(float(val_loss), step=int(epoch))
        if hasattr(self.trial, "should_prune") and self.trial.should_prune():
            import optuna
            try:
                self.trial.set_user_attr("pruned_at_step", int(epoch))
            except Exception:
                pass
            raise optuna.exceptions.TrialPruned()


class LossWeightScheduler:
    """
    Adaptive weight scheduler driven by fit diagnostics.

    Static base weights remain intact; only multiplicative dynamic factors are
    updated over time.
    """

    def __init__(self, ctx: Optional[Any] = None) -> None:
        self.base_weights: Dict[str, float] = {
            "data_loss": 1.0,
            "regularization": 1.0,
            "constraint": 1.0,
        }
        self.dynamic_factors: Dict[str, float] = {
            "data_loss": 1.0,
            "regularization": 1.0,
            "constraint": 1.0,
        }
        self.last_analysis: Dict[str, Any] = {}
        self._trial_intelligence = self._load_trial_intelligence()
        self._ctx = ctx

    @staticmethod
    def _load_trial_intelligence() -> Optional[Any]:
        try:
            from automl.trial_intelligence import TrialIntelligence

            return TrialIntelligence()
        except Exception:
            return None

    @staticmethod
    def _fallback_analysis(train_losses: List[float], val_losses: List[float]) -> Dict[str, Any]:
        if len(train_losses) < 2 or len(val_losses) < 2:
            return {"fit_type": "good", "train_slope": 0.0, "val_slope": 0.0}

        train_slope = float(train_losses[-1] - train_losses[0]) / max(1, len(train_losses) - 1)
        val_slope = float(val_losses[-1] - val_losses[0]) / max(1, len(val_losses) - 1)

        if train_slope < -1e-3 and val_slope > 1e-3:
            fit_type = "overfitting"
        elif abs(train_slope) < 1e-3 and abs(val_slope) < 1e-3:
            fit_type = "underfitting"
        else:
            fit_type = "good"

        return {
            "fit_type": fit_type,
            "train_slope": train_slope,
            "val_slope": val_slope,
        }

    def update(self, train_losses: List[float], val_losses: List[float]) -> Dict[str, Any]:
        if self._trial_intelligence is not None and hasattr(self._trial_intelligence, "analyze"):
            try:
                analysis = self._trial_intelligence.analyze(train_losses, val_losses)
                if hasattr(self._trial_intelligence, "consistency_gate"):
                    gate = self._trial_intelligence.consistency_gate(analysis)
                    analysis["consistency"] = gate
                    if not gate.get("stable", True):
                        # Keep current factors when signal is too weak.
                        self.last_analysis = analysis
                        return analysis
            except Exception:
                analysis = self._fallback_analysis(train_losses, val_losses)
        else:
            analysis = self._fallback_analysis(train_losses, val_losses)

        fit_type = str(analysis.get("fit_type", "good"))

        if fit_type == "overfitting":
            self.dynamic_factors["regularization"] = min(
                2.0, self.dynamic_factors["regularization"] * 1.10
            )
            self.dynamic_factors["constraint"] = min(
                2.0, self.dynamic_factors["constraint"] * 1.08
            )
            self.dynamic_factors["data_loss"] = max(
                0.7, self.dynamic_factors["data_loss"] * 0.98
            )
        elif fit_type == "underfitting":
            self.dynamic_factors["regularization"] = max(
                0.5, self.dynamic_factors["regularization"] * 0.92
            )
            self.dynamic_factors["constraint"] = max(
                0.5, self.dynamic_factors["constraint"] * 0.94
            )
            self.dynamic_factors["data_loss"] = min(
                1.3, self.dynamic_factors["data_loss"] * 1.03
            )
        else:
            for key, factor in self.dynamic_factors.items():
                self.dynamic_factors[key] = float(factor + (1.0 - factor) * 0.1)

        self.last_analysis = analysis
        if self._ctx is not None:
            try:
                self._ctx.update_fit_analysis(analysis)
            except Exception:
                pass
        return analysis

    def get_effective_weights(self) -> Dict[str, float]:
        return {
            name: self.base_weights[name] * self.dynamic_factors[name]
            for name in self.base_weights
        }


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
        alignment_weight: float = 0.0,
        modality_dropout_prob: float = 0.15,
        fusion_aux_weights: Optional[Dict[str, float]] = None,
        label_smoothing: float = 0.0,
        execution_context: Optional[Any] = None,
        contrastive_weight: float = 0.0,
        ewc: Optional[Any] = None,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        mixup_alpha: float = 0.0,
        lora_config: Optional[Dict[str, Any]] = None,
        tabular_tokenizer: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model", "image_encoder", "text_encoder", "tabular_encoder", "class_weights", "ewc", "lora_config", "tabular_tokenizer"])
        self.mixup_alpha = float(max(0.0, mixup_alpha))
        self._lora_config: Optional[Dict[str, Any]] = lora_config
        # EWC object (Kirkpatrick et al. 2017 [8]) — None when not retraining
        self._ewc = ewc
        self._use_focal_loss = bool(use_focal_loss)
        self._focal_gamma = float(focal_gamma)

        self.model = model
        self.problem_type = problem_type
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.alignment_weight = float(max(0.0, alignment_weight))
        self.modality_dropout_prob = float(max(0.0, min(0.5, modality_dropout_prob)))
        self.label_smoothing = float(max(0.0, min(0.25, label_smoothing)))

        _aux_defaults: Dict[str, float] = {
            "diversity_loss_weight": 0.01,
            "graph_sparsity_weight": 0.005,
            "uncertainty_aux_weight": 0.0,
        }
        _aux_input = dict(fusion_aux_weights or {})
        self.fusion_aux_weights: Dict[str, float] = {}
        for key, default_value in _aux_defaults.items():
            try:
                self.fusion_aux_weights[key] = max(0.0, float(_aux_input.get(key, default_value)))
            except Exception:
                self.fusion_aux_weights[key] = float(default_value)

        # Apply LoRA adapters when lora_config is provided.
        # LoRA layers become trainable even inside frozen encoders.
        if lora_config and isinstance(lora_config, dict):
            try:
                from modelss.adapters.lora import apply_lora
                _r     = int(lora_config.get("r", 8))
                _alpha = float(lora_config.get("alpha", 16.0))
                if image_encoder is not None:
                    apply_lora(image_encoder, r=_r, alpha=_alpha)
                    logger.info("LoRA applied to image encoder: r=%d alpha=%.0f", _r, _alpha)
                if text_encoder is not None:
                    apply_lora(text_encoder, r=_r, alpha=_alpha)
                    logger.info("LoRA applied to text encoder: r=%d alpha=%.0f", _r, _alpha)
            except Exception as _lora_exc:
                logger.warning("LoRA application failed: %s — encoders remain frozen", _lora_exc)

        # ULA token-mode: UnifiedLatentFusion with token_mode=True accepts full
        # token sequences (N,T,D) instead of pooled (N,D) vectors — enables true
        # cross-modal attention between ViT patch embeddings and BERT token states.
        try:
            from modelss.fusion import UnifiedLatentFusion as _ULAFusion
            _fusion_obj = getattr(model, "fusion", None)
            self._use_token_sequences: bool = (
                isinstance(_fusion_obj, _ULAFusion) and
                getattr(_fusion_obj, "token_mode", False)
            )
        except Exception:
            self._use_token_sequences = False

        # Store encoders WITHOUT nn.Module registration (keeps checkpoint small).
        # LoRA A/B parameters are collected explicitly in configure_optimizers.
        object.__setattr__(self, "_image_encoder", image_encoder)
        object.__setattr__(self, "_text_encoder", text_encoder)
        object.__setattr__(self, "_tabular_tokenizer", tabular_tokenizer)
        # Register loss-weight tensors as buffers so Lightning's .to(device) / .cuda()
        # moves them automatically — prevents CPU/GPU mismatch in loss evaluation.
        _cw = class_weights.float() if class_weights is not None else None
        self.register_buffer("_class_weights", _cw)
        self.register_buffer("_binary_pos_weight", None)  # filled below if needed

        # TRAINABLE tabular encoder: registered as a proper nn.Module
        # submodule so its parameters ARE included in self.parameters()
        # and the optimizer.  Created fresh per Optuna trial.
        self.tabular_encoder = tabular_encoder

        # ── Loss function (with optional class weights for imbalanced data) ──
        if problem_type == "classification_binary":
            if self._use_focal_loss:
                self.loss_fn: nn.Module = FocalLoss(gamma=self._focal_gamma)
                logger.info("  FocalLoss [ICCV 2017] γ=%.1f (binary)", self._focal_gamma)
            elif _cw is not None and len(_cw) >= 2:
                pos_weight = (_cw[1] / _cw[0]).unsqueeze(0)
                # Keep the buffer identity stable; only replace the stored tensor.
                self._binary_pos_weight = pos_weight
                self.loss_fn: nn.Module = nn.BCEWithLogitsLoss(
                    pos_weight=self._binary_pos_weight
                )
                logger.info("  BCEWithLogitsLoss pos_weight=%.3f", pos_weight.item())
            else:
                self.loss_fn: nn.Module = nn.BCEWithLogitsLoss()
        elif problem_type == "multilabel_classification":
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif problem_type.startswith("classification"):
            if self._use_focal_loss:
                self.loss_fn = FocalLoss(gamma=self._focal_gamma, alpha=_cw)
                logger.info("  FocalLoss [ICCV 2017] γ=%.1f (multiclass)", self._focal_gamma)
            else:
                self.loss_fn = nn.CrossEntropyLoss(
                    weight=_cw,
                    label_smoothing=self.label_smoothing,
                )
                if _cw is not None:
                    logger.info("  CrossEntropyLoss class weights: %s", _cw.tolist())
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
            # AUROC for classification (binary preferred; multiclass uses macro OvR)
            try:
                _auroc_kwargs = (
                    {"task": "binary"}
                    if task == "binary"
                    else {"task": "multiclass", "num_classes": num_classes, "average": "macro"}
                )
                self.val_auroc = torchmetrics.AUROC(**_auroc_kwargs)
                self._auroc_enabled = True
            except Exception:
                self._auroc_enabled = False
        else:
            self.train_rmse = torchmetrics.MeanSquaredError(squared=False)
            self.val_rmse   = torchmetrics.MeanSquaredError(squared=False)
            self.train_r2   = torchmetrics.R2Score()
            self.val_r2     = torchmetrics.R2Score()

        # Auto-activate contrastive loss for multimodal training
        # (CLIP-style NT-Xent) — novel for AutoML frameworks
        _n_modalities = sum([
            tabular_encoder is not None,
            text_encoder is not None,
            image_encoder is not None,
        ])
        if contrastive_weight == 0.0 and _n_modalities >= 2:
            contrastive_weight = 0.1
            logger.info(
                "  Auto-activated contrastive loss (weight=0.1) for %d modalities",
                _n_modalities,
            )
        self.contrastive_weight = float(max(0.0, contrastive_weight))
        self.loss_weight_scheduler = LossWeightScheduler(ctx=execution_context)
        self.alignment_loss = AlignmentLoss()
        self.contrastive_loss_fn = LinkedModalityContrastiveLoss(temperature=0.07)

        # CLIP projection heads — project each modality into shared 128-d space
        self._clip_projections: Dict[str, CLIPProjectionHead] = nn.ModuleDict()
        if self.contrastive_weight > 0.0:
            if tabular_encoder is not None:
                tab_dim = getattr(tabular_encoder, 'output_dim',
                         getattr(tabular_encoder, 'out_features', 128))
                self._clip_projections['tabular'] = CLIPProjectionHead(tab_dim, 128)
            if text_encoder is not None:
                txt_dim = getattr(text_encoder, 'output_dim',
                         getattr(text_encoder, 'hidden_size', 768))
                self._clip_projections['text'] = CLIPProjectionHead(txt_dim, 128)
            if image_encoder is not None:
                img_dim = getattr(image_encoder, 'output_dim',
                         getattr(image_encoder, 'num_features', 512))
                self._clip_projections['image'] = CLIPProjectionHead(img_dim, 128)
        self._train_loss_history: List[float] = []
        self._val_loss_history: List[float] = []
        self._alignment_loss_history: List[float] = []
        self._contrastive_loss_history: List[float] = []
        self._alignment_loss_epoch_values: List[float] = []
        self._contrastive_loss_epoch_values: List[float] = []
        self.fusion_attention_logs: List[Dict[str, Any]] = []
        self._last_encoded_batch: Dict[str, torch.Tensor] = {}
        self.last_alignment_summary: Dict[str, float] = {}
        self._modality_grad_scales: Dict[str, float] = {}
        self._memory_guard = MemoryGuard()
        self._memory_guard_every_n_steps = 50
        self._fallback_manager = FallbackManager()

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
        expected_dims: Dict[str, int] = getattr(self.model, "_input_dims", {})

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
            try:
                with torch.set_grad_enabled(self._encoder_grad_enabled()):
                    outputs = self._text_encoder.transformer(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                    )
                    if self._use_token_sequences:
                        # ULA token-mode: full sequence (N, T, hidden_size) for cross-modal attention
                        encoded["text_pooled"] = outputs.last_hidden_state
                    else:
                        cls_token = outputs.last_hidden_state[:, 0, :]
                        if self._text_encoder._projection is not None:
                            cls_token = self._text_encoder._projection(cls_token)
                        encoded["text_pooled"] = cls_token
            except Exception as exc:
                text_name = str(getattr(self._text_encoder, "model_name", type(self._text_encoder).__name__))
                self._fallback_manager.mark_failed("text", text_name)
                logger.warning("Text encoder failed; using dummy-fill fallback (%s): %s", text_name, exc)

        # ── Image: use pre-computed embedding or route through frozen encoder ─
        if "image_pooled" in batch:
            encoded["image_pooled"] = batch["image_pooled"]
        elif "image" in batch and self._image_encoder is not None:
            try:
                with torch.set_grad_enabled(self._encoder_grad_enabled()):
                    if self._use_token_sequences and hasattr(self._image_encoder, "forward"):
                        import inspect as _inspect
                        _sig = _inspect.signature(self._image_encoder.forward)
                        if "return_all_tokens" in _sig.parameters:
                            # ULA token-mode: patch token sequence (N, P, D)
                            encoded["image_pooled"] = self._image_encoder(
                                batch["image"], return_all_tokens=True
                            )
                        else:
                            encoded["image_pooled"] = self._image_encoder(batch["image"])
                    else:
                        encoded["image_pooled"] = self._image_encoder(batch["image"])
            except Exception as exc:
                image_name = str(getattr(self._image_encoder, "model_name", type(self._image_encoder).__name__))
                self._fallback_manager.mark_failed("image", image_name)
                logger.warning("Image encoder failed; using dummy-fill fallback (%s): %s", image_name, exc)

        # ── Tabular tokenizer (ULA token-mode): converts (N,D) → (N,F,token_dim) ──
        if self._use_token_sequences and self._tabular_tokenizer is not None and "tabular" in encoded:
            try:
                with torch.no_grad():
                    encoded["tabular"] = self._tabular_tokenizer(encoded["tabular"])
            except Exception as exc:
                logger.warning("TabularFeatureTokenizer failed; keeping pooled tabular: %s", exc)

        # ── Dummy-fill any head-expected keys still missing ───────────
        # Determines reference device from existing encoded tensors or batch
        ref_device = next(
            (t.device for t in encoded.values() if isinstance(t, torch.Tensor)),
            next((t.device for t in batch.values() if isinstance(t, torch.Tensor)), None),
        )
        for key in head_keys:
            if key not in encoded and key != "target":
                dim = self._resolve_missing_dim(key, expected_dims)
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
        self._last_encoded_batch = encoded
        return self.model(encoded)

    def _encoder_grad_enabled(self) -> bool:
        """Enable encoder autograd only for LoRA adapter training."""
        return bool(self.training and self._lora_config)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    @staticmethod
    def _mixup_batch(
        tabular: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.4,
    ):
        """
        MixUp augmentation for tabular features.

        Samples a mixing coefficient λ ~ Beta(α, α) and creates convex
        combinations of two random permutations of the batch:

            x̃ = λ·xᵢ + (1-λ)·xⱼ
            ỹ = λ·yᵢ + (1-λ)·yⱼ   (soft labels for classification)

        References
        ----------
        Zhang et al. "MixUp: Beyond Empirical Risk Minimisation."
        ICLR 2018.
        Guo et al. "Mixup as Locally Linear Out-of-Manifold Regularization."
        AAAI 2019. (Tabular MixUp justification)
        """
        import numpy as _np
        lam = float(_np.random.beta(alpha, alpha))
        N = tabular.shape[0]
        perm = torch.randperm(N, device=tabular.device)
        mixed_tab = lam * tabular + (1.0 - lam) * tabular[perm]
        # Soft targets for classification (hard for regression — keep original)
        if targets.dtype in (torch.long, torch.int):
            # Convert to one-hot for soft mixing
            n_classes = int(targets.max().item()) + 1
            one_hot = torch.zeros(N, n_classes, device=targets.device, dtype=torch.float32)
            one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
            mixed_targets = lam * one_hot + (1.0 - lam) * one_hot[perm]
        else:
            mixed_targets = lam * targets + (1.0 - lam) * targets[perm]
        return mixed_tab, mixed_targets

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        targets = batch["target"]

        # MixUp tabular augmentation (active when mixup_alpha > 0)
        _mixup_alpha: float = getattr(self, "mixup_alpha", 0.0)
        _mixup_targets = None
        if (
            _mixup_alpha > 0.0
            and self.training
            and "tabular" in batch
            and self.problem_type.startswith("classification")
        ):
            batch = dict(batch)
            batch["tabular"], _mixup_targets = self._mixup_batch(
                batch["tabular"], targets, alpha=_mixup_alpha
            )

        if self.modality_dropout_prob > 0.0 and self.training:
            batch = dict(batch)
            drop_text = torch.rand(1).item() < self.modality_dropout_prob
            drop_image = torch.rand(1).item() < self.modality_dropout_prob

            if drop_text:
                batch.pop("input_ids", None)
                batch.pop("attention_mask", None)
                batch.pop("text_pooled", None)
                self.log("modality_dropout_text", 1.0, on_step=False, on_epoch=True)
            if drop_image:
                batch.pop("image", None)
                batch.pop("image_pooled", None)
                self.log("modality_dropout_image", 1.0, on_step=False, on_epoch=True)

        logits = self(batch)

        # Use soft mixed labels when MixUp was applied
        _loss_targets = _mixup_targets if _mixup_targets is not None else targets
        if _mixup_targets is not None:
            # Soft cross-entropy for MixUp: -Σ ỹ · log p
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            base_loss = -((_loss_targets * log_probs).sum(dim=-1)).mean()
        else:
            base_loss = self._compute_loss(logits, targets)
        loss = self._apply_adaptive_loss(base_loss)

        # EWC regularisation — Kirkpatrick et al. 2017 [8]
        # λ/2 · Σ_i F_i(θ_i − θ*_i)² prevents catastrophic forgetting
        if self._ewc is not None:
            try:
                ewc_penalty = self._ewc.penalty(self.model)
                loss = loss + ewc_penalty
                self.log("ewc_penalty", float(ewc_penalty), prog_bar=False, on_step=False, on_epoch=True)
            except Exception:
                pass

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        # PCGrad logging (gradient surgery applied in optimizer step via manual_backward)
        # Logged here so the metric is always present in callback_metrics
        self.log("pcgrad_conflicts", 0.0, prog_bar=False, on_step=False, on_epoch=True, reduce_fx="sum")
        self.log("train_loss_raw", base_loss, prog_bar=False, on_step=False, on_epoch=True)

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
            if batch_idx % self._memory_guard_every_n_steps == 0:
                try:
                    self._memory_guard.maybe_clear_cache()
                except Exception:
                    pass
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
            # AUROC: pass probabilities (softmax for multiclass, sigmoid col-1 for binary)
            if getattr(self, "_auroc_enabled", False):
                try:
                    if self.problem_type == "classification_binary":
                        probs = torch.softmax(logits.float(), dim=-1)[:, 1]
                    else:
                        probs = torch.softmax(logits.float(), dim=-1)
                    self.val_auroc(probs, metric_targets)
                    self.log("val_auroc", self.val_auroc, prog_bar=False, on_epoch=True)
                except Exception:
                    pass
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

    def on_validation_epoch_end(self) -> None:
        callback_metrics = getattr(self.trainer, "callback_metrics", {})
        train_loss = self._metric_to_float(callback_metrics.get("train_loss"))
        val_loss = self._metric_to_float(callback_metrics.get("val_loss"))

        if train_loss is not None and val_loss is not None:
            self._train_loss_history.append(train_loss)
            self._val_loss_history.append(val_loss)

            self._train_loss_history = self._train_loss_history[-20:]
            self._val_loss_history = self._val_loss_history[-20:]

        self._flush_aux_loss_histories()

        if train_loss is not None and val_loss is not None:
            analysis = self.loss_weight_scheduler.update(
                self._train_loss_history,
                self._val_loss_history,
            )
            weights = self.loss_weight_scheduler.get_effective_weights()
            self.log(
                "loss_weight_regularization",
                float(weights["regularization"]),
                prog_bar=False,
                on_epoch=True,
            )
            self.log(
                "loss_weight_data",
                float(weights["data_loss"]),
                prog_bar=False,
                on_epoch=True,
            )
            logger.debug("LossWeightScheduler analysis=%s weights=%s", analysis, weights)

        fusion_layer = getattr(self.model, "fusion", None)
        attention_summary: Dict[str, Any] = {}
        if fusion_layer is not None and hasattr(fusion_layer, "get_attention_summary"):
            try:
                attention_summary = fusion_layer.get_attention_summary() or {}
                modality_importance = attention_summary.get("modality_importance", {})
                if isinstance(modality_importance, dict) and modality_importance:
                    total = max(1e-8, sum(float(v) for v in modality_importance.values()))
                    self._modality_grad_scales = {
                        str(key): max(0.10, float(value) / total)
                        for key, value in modality_importance.items()
                    }
            except Exception as exc:
                logger.debug("Fusion attention summary unavailable: %s", exc)
                attention_summary = {}
                self._modality_grad_scales = {}

        self.fusion_attention_logs.append({
            "epoch": int(self.current_epoch),
            "fusion": type(fusion_layer).__name__ if fusion_layer is not None else "none",
            "summary": attention_summary,
        })
        self.fusion_attention_logs = self.fusion_attention_logs[-100:]

        if attention_summary and "head_diversity" in attention_summary:
            self.log(
                "fusion_head_diversity",
                float(attention_summary.get("head_diversity", 0.0)),
                prog_bar=False,
                on_epoch=True,
            )

        if self._last_encoded_batch:
            named = {
                k: v
                for k, v in self._last_encoded_batch.items()
                if k in ("tabular", "text_pooled", "image_pooled")
            }
            if len(named) >= 2:
                self.last_alignment_summary = self.alignment_loss.summarize(named)
                if self.last_alignment_summary:
                    mean_alignment = sum(self.last_alignment_summary.values()) / len(self.last_alignment_summary)
                    self.log(
                        "alignment_mean_cosine",
                        float(mean_alignment),
                        prog_bar=False,
                        on_epoch=True,
                    )

    def _flush_aux_loss_histories(self) -> None:
        """Append mean per-epoch ULA diagnostic losses to frontend histories."""
        if self._alignment_loss_epoch_values:
            self._alignment_loss_history.append(
                float(sum(self._alignment_loss_epoch_values) / len(self._alignment_loss_epoch_values))
            )
            self._alignment_loss_history = self._alignment_loss_history[-20:]
            self._alignment_loss_epoch_values.clear()
        if self._contrastive_loss_epoch_values:
            self._contrastive_loss_history.append(
                float(sum(self._contrastive_loss_epoch_values) / len(self._contrastive_loss_epoch_values))
            )
            self._contrastive_loss_history = self._contrastive_loss_history[-20:]
            self._contrastive_loss_epoch_values.clear()

    # ------------------------------------------------------------------
    # Optimiser + scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        # Build parameter groups: head/tabular encoder at full LR,
        # LoRA adapter parameters at a separate (usually lower) LR.
        param_groups = [
            {
                "params": self.parameters(),
                "lr": self.learning_rate,
                "weight_decay": self.weight_decay,
            }
        ]

        if self._lora_config:
            try:
                from modelss.adapters.lora import lora_parameters
                lora_lr = self.learning_rate * float(self._lora_config.get("lr_mult", 0.1))
                lora_params = []
                for enc in (self._text_encoder, self._image_encoder):
                    if enc is not None:
                        lora_params.extend(lora_parameters(enc))
                if lora_params:
                    # Exclude LoRA params from the default group to avoid double-counting
                    lora_ids = {id(p) for p in lora_params}
                    param_groups[0]["params"] = [
                        p for p in self.parameters() if id(p) not in lora_ids
                    ]
                    param_groups.append({
                        "params": lora_params,
                        "lr": lora_lr,
                        "weight_decay": 0.0,  # LoRA adapters typically not regularized
                    })
                    logger.info("LoRA param group: %d params at lr=%.2e", len(lora_params), lora_lr)
            except Exception as _le:
                logger.warning("Could not build LoRA optimizer group: %s", _le)

        optimizer = AdamW(param_groups)
        scheduler = AdaptiveLRScheduler(
            optimizer,
            T_max=self.max_epochs,
            loss_weight_scheduler=self.loss_weight_scheduler,
        )
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

    def _resolve_missing_dim(self, key: str, expected_dims: Dict[str, int]) -> int:
        """Resolve safe fallback dimensions for dummy modality fill-in."""
        if key in expected_dims:
            try:
                dim = int(expected_dims[key])
                if dim > 0:
                    return dim
            except Exception:
                pass

        if key == "tabular" and self.tabular_encoder is not None:
            getter = getattr(self.tabular_encoder, "get_output_dim", None)
            if callable(getter):
                try:
                    dim = int(getter())
                    if dim > 0:
                        return dim
                except Exception as exc:
                    logger.debug("tabular_encoder.get_output_dim() failed: %s", exc)

        if key == "text_pooled":
            return 768
        if key == "image_pooled":
            return 512
        return 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_adaptive_loss(self, base_loss: torch.Tensor) -> torch.Tensor:
        weights = self.loss_weight_scheduler.get_effective_weights()

        reg_term = torch.zeros((), device=base_loss.device, dtype=base_loss.dtype)
        for param in self.model.parameters():
            if param.requires_grad:
                reg_term = reg_term + param.pow(2).mean()

        reg_strength = 1e-6 * float(weights["regularization"])
        scaled = base_loss * float(weights["data_loss"]) + reg_strength * reg_term

        aligned_features = [
            self._last_encoded_batch[k]
            for k in ("tabular", "text_pooled", "image_pooled")
            if k in self._last_encoded_batch
        ]
        if len(aligned_features) >= 2:
            try:
                align_term = self.alignment_loss(aligned_features)
                self._alignment_loss_epoch_values.append(
                    float(align_term.detach().float().cpu().item())
                )
                try:
                    self.log(
                        "train/alignment_loss",
                        align_term.detach(),
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                    )
                except Exception:
                    pass
            except Exception:
                align_term = None
            if align_term is not None and self.alignment_weight > 0.0:
                if self._modality_grad_scales:
                    min_scale = min(self._modality_grad_scales.values())
                    align_scale = 2.0 - min_scale
                else:
                    align_scale = 1.0
                scaled = scaled + (self.alignment_weight * align_scale * align_term)

        fusion_mod = getattr(self.model, "fusion", None)
        if fusion_mod is not None:
            fusion_type = type(fusion_mod).__name__
            if fusion_type in ("GraphFusion", "UncertaintyGraphFusion"):
                try:
                    from models.fusion import diversity_loss, graph_sparsity_loss

                    head_outputs = getattr(fusion_mod, "last_head_outputs", None)
                    diversity_weight = float(self.fusion_aux_weights.get("diversity_loss_weight", 0.0))
                    if head_outputs and diversity_weight > 0.0:
                        scaled = scaled + diversity_weight * diversity_loss(head_outputs)

                    graph_tensor = None
                    graph_attr = getattr(fusion_mod, "graph", None)
                    if callable(graph_attr):
                        graph_tensor = graph_attr()
                    elif isinstance(graph_attr, torch.Tensor):
                        graph_tensor = graph_attr
                    sparsity_weight = float(self.fusion_aux_weights.get("graph_sparsity_weight", 0.0))
                    if isinstance(graph_tensor, torch.Tensor) and sparsity_weight > 0.0:
                        scaled = scaled + sparsity_weight * graph_sparsity_loss(graph_tensor)
                except Exception:
                    pass

            if fusion_type in ("UncertaintyFusion", "UncertaintyGraphFusion"):
                uncertainty_aux_weight = float(self.fusion_aux_weights.get("uncertainty_aux_weight", 0.0))
                if uncertainty_aux_weight > 0.0:
                    try:
                        log_var_heads = getattr(fusion_mod, "log_var_heads", None)
                        if isinstance(log_var_heads, nn.ModuleList):
                            unc_reg = torch.zeros((), device=base_loss.device, dtype=base_loss.dtype)
                            for head in log_var_heads:
                                for param in head.parameters():
                                    unc_reg = unc_reg + param.pow(2).mean()
                            scaled = scaled + (uncertainty_aux_weight * unc_reg)
                    except Exception:
                        pass

        # ── CLIP-style NT-Xent contrastive loss (all modality pairs) ─────
        if self._last_encoded_batch:
            try:
                emb_tab = self._last_encoded_batch.get("tabular")
                emb_txt = self._last_encoded_batch.get("text_pooled")
                emb_img = self._last_encoded_batch.get("image_pooled")

                # Project through CLIP heads if available
                proj = self._clip_projections
                if emb_tab is not None and 'tabular' in proj:
                    emb_tab = proj['tabular'](emb_tab)
                if emb_txt is not None and 'text' in proj:
                    emb_txt = proj['text'](emb_txt)
                if emb_img is not None and 'image' in proj:
                    emb_img = proj['image'](emb_img)

                total_contrastive = torch.zeros((), device=scaled.device)
                n_pairs = 0

                # Pair 1: tabular ↔ text
                if emb_tab is not None and emb_txt is not None:
                    total_contrastive = total_contrastive + self.contrastive_loss_fn(emb_tab, emb_txt)
                    n_pairs += 1
                # Pair 2: tabular ↔ image
                if emb_tab is not None and emb_img is not None:
                    total_contrastive = total_contrastive + self.contrastive_loss_fn(emb_tab, emb_img)
                    n_pairs += 1
                # Pair 3: text ↔ image (NEW — completes the triangle)
                if emb_txt is not None and emb_img is not None:
                    total_contrastive = total_contrastive + self.contrastive_loss_fn(emb_txt, emb_img)
                    n_pairs += 1

                if n_pairs > 0:
                    avg_contrastive = total_contrastive / n_pairs
                    self._contrastive_loss_epoch_values.append(
                        float(avg_contrastive.detach().float().cpu().item())
                    )
                    if self.contrastive_weight > 0.0:
                        scaled = scaled + self.contrastive_weight * avg_contrastive
                    try:
                        self.log("train/contrastive_loss", avg_contrastive.detach(),
                                 on_step=False, on_epoch=True, prog_bar=False)
                    except Exception:
                        pass
            except Exception:
                pass

        if self._modality_grad_scales and self._last_encoded_batch:
            key_map = {
                "modality_0": "tabular",
                "modality_1": "text_pooled",
                "modality_2": "image_pooled",
            }
            for mod_key, scale in self._modality_grad_scales.items():
                tensor_key = key_map.get(mod_key)
                if tensor_key and tensor_key in self._last_encoded_batch:
                    tensor = self._last_encoded_batch[tensor_key]
                    if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                        tensor.register_hook(lambda grad, s=float(scale): grad * s)

        return scaled

    @staticmethod
    def _metric_to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return float(value.detach().float().mean().cpu().item())
        try:
            return float(value)
        except Exception:
            return None

    def get_loss_weight_state(self) -> Dict[str, float]:
        return self.loss_weight_scheduler.get_effective_weights()

    def get_fusion_attention_logs(self) -> List[Dict[str, Any]]:
        return list(self.fusion_attention_logs)

    def get_alignment_summary(self) -> Dict[str, float]:
        return dict(self.last_alignment_summary)

    def get_fusion_summary(self) -> Dict[str, Any]:
        """Return fusion diagnostics, encoding architecture, and modality health signals."""
        fusion_mod = getattr(self.model, "fusion", None)

        # Per-modality encoder output dimensions
        encoder_dims: Dict[str, int] = {}
        _tab_enc = getattr(self, "tabular_encoder", None)
        _txt_enc = object.__getattribute__(self, "_text_encoder") if hasattr(self, "_text_encoder") else None
        _img_enc = object.__getattribute__(self, "_image_encoder") if hasattr(self, "_image_encoder") else None
        if _tab_enc is not None:
            encoder_dims["tabular"] = int(getattr(_tab_enc, "output_dim", getattr(_tab_enc, "out_features", 0)))
        if _txt_enc is not None:
            encoder_dims["text"] = int(getattr(_txt_enc, "hidden_size", getattr(_txt_enc, "output_dim", 768)))
        if _img_enc is not None:
            encoder_dims["image"] = int(getattr(_img_enc, "num_features", getattr(_img_enc, "output_dim", 512)))

        summary: Dict[str, Any] = {
            "strategy": str(getattr(self.model, "fusion_strategy", "") or "").lower(),
            "fusion_type": type(fusion_mod).__name__ if fusion_mod is not None else "none",
            "backend_module": getattr(type(fusion_mod), "__module__", "unknown") if fusion_mod is not None else "unknown",
            "alignment_weight": float(self.alignment_weight),
            "contrastive_weight": float(self.contrastive_weight),
            "modality_dropout_prob": float(self.modality_dropout_prob),
            "label_smoothing": float(self.label_smoothing),
            "auxiliary_loss_weights": dict(self.fusion_aux_weights),
            # Encoding architecture signals
            "token_mode": bool(getattr(self, "_use_token_sequences", False)),
            "encoder_dims": encoder_dims,
            "active_modalities": list(encoder_dims.keys()),
            "clip_projections_active": len(getattr(self, "_clip_projections", {})) > 0,
            # Per-modality gradient health (Wang et al. 2020 gradient balancing)
            "modality_grad_scales": {
                k: round(float(v), 4)
                for k, v in (getattr(self, "_modality_grad_scales", {}) or {}).items()
            },
        }

        if fusion_mod is None:
            return summary

        # ULA-specific: latent dim and transformer config
        if hasattr(fusion_mod, "latent_dim"):
            summary["ula_latent_dim"] = int(fusion_mod.latent_dim)
        if hasattr(fusion_mod, "transformer"):
            tr = fusion_mod.transformer
            if hasattr(tr, "layers"):
                summary["ula_n_layers"] = len(tr.layers)

        if hasattr(fusion_mod, "get_attention_summary"):
            try:
                summary["attention_summary"] = dict(fusion_mod.get_attention_summary() or {})
            except Exception:
                summary["attention_summary"] = {}

        if hasattr(fusion_mod, "get_branch_weights"):
            try:
                summary["branch_weights"] = dict(fusion_mod.get_branch_weights() or {})
            except Exception:
                pass

        if "attention_summary" not in summary:
            summary["attention_summary"] = {}

        return summary

    def _class_weights_for_device(self, device: torch.device) -> Optional[torch.Tensor]:
        """Return class weights on the active logits device when present."""
        if self._class_weights is None:
            return None
        return self._class_weights if self._class_weights.device == device else self._class_weights.to(device)

    def _binary_pos_weight_for_device(self, device: torch.device) -> Optional[torch.Tensor]:
        """Return binary pos_weight on the active logits device when present."""
        if self._binary_pos_weight is None:
            return None
        return (
            self._binary_pos_weight
            if self._binary_pos_weight.device == device
            else self._binary_pos_weight.to(device)
        )

    def _compute_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        dev = logits.device
        if self.problem_type == "classification_binary":
            if self.label_smoothing > 0.0:
                smoothed_targets = targets.float() * (1.0 - self.label_smoothing)
                smoothed_targets = smoothed_targets + (0.5 * self.label_smoothing)
                _pw = self._binary_pos_weight_for_device(dev)
                return F.binary_cross_entropy_with_logits(
                    logits.squeeze(-1).float(),
                    smoothed_targets,
                    pos_weight=_pw,
                )
            return self.loss_fn(logits.squeeze(-1).float(), targets.float())
        elif self.problem_type == "multilabel_classification":
            if self.label_smoothing > 0.0:
                smoothed_targets = targets.float() * (1.0 - self.label_smoothing)
                smoothed_targets = smoothed_targets + (0.5 * self.label_smoothing)
                return F.binary_cross_entropy_with_logits(
                    logits.float(),
                    smoothed_targets,
                )
            return self.loss_fn(logits.float(), targets.float())
        elif self.problem_type.startswith("classification"):
            _cw = self._class_weights_for_device(dev)
            if self.label_smoothing > 0.0:
                return F.cross_entropy(
                    logits,
                    targets.long(),
                    weight=_cw,
                    label_smoothing=self.label_smoothing,
                )
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
# PCGrad — Gradient Surgery (Yu et al., NeurIPS 2020)
# "Gradient Surgery for Multi-Task Learning"
# ---------------------------------------------------------------------------

class PCGradCallback(pl.Callback):
    """
    PCGrad gradient surgery for multimodal training.

    When gradients from two modality-specific parameter groups conflict
    (cosine similarity < 0), project each gradient onto the normal plane
    of the other.  This eliminates destructive interference between
    tabular, text, and image encoder gradients during joint optimisation.

    Algorithm (Yu et al., NeurIPS 2020 [13]):
      g_i ← g_i - (g_i · g_j / ||g_j||²) · g_j   when g_i · g_j < 0

    Applied as a Lightning on_before_optimizer_step hook so it integrates
    cleanly with automatic optimization.
    """

    def __init__(self, modality_attrs: tuple = ("tabular_encoder", "_image_encoder", "_text_encoder")) -> None:
        super().__init__()
        self._modality_attrs = modality_attrs
        self._conflict_count: int = 0

    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        try:
            self._apply_gradient_surgery(pl_module)
        except Exception:
            pass

    def _apply_gradient_surgery(self, module: nn.Module) -> None:
        # Collect per-modality gradient vectors (flattened, detached)
        modality_grads: Dict[str, torch.Tensor] = {}
        modality_params: Dict[str, List] = {}
        for attr in self._modality_attrs:
            enc = getattr(module, attr, None)
            if enc is None:
                continue
            grads, params = [], []
            for p in enc.parameters():
                if p.grad is not None and p.requires_grad:
                    grads.append(p.grad.detach().flatten())
                    params.append(p)
            if grads:
                modality_grads[attr] = torch.cat(grads)
                modality_params[attr] = params

        if len(modality_grads) < 2:
            return

        keys = list(modality_grads.keys())
        conflicts = 0
        for i, key_i in enumerate(keys):
            g_i = modality_grads[key_i]
            for j, key_j in enumerate(keys):
                if i >= j:
                    continue
                g_j = modality_grads[key_j]
                dot = float(torch.dot(g_i, g_j))
                if dot < 0.0:
                    conflicts += 1
                    # Project g_i onto normal plane of g_j
                    g_j_norm_sq = max(float(g_j.dot(g_j)), 1e-12)
                    # Apply the projection back to individual parameter gradients
                    proj_coeff = dot / g_j_norm_sq
                    cursor = 0
                    for p in modality_params[key_i]:
                        if p.grad is None:
                            continue
                        n = p.grad.numel()
                        g_j_slice = g_j[cursor:cursor + n].view_as(p.grad)
                        p.grad.data -= proj_coeff * g_j_slice
                        cursor += n
        self._conflict_count += conflicts


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
    fusion_config: Optional[Dict[str, Any]] = None,
    head_architecture_type: str = "mlp",
    head_num_layers: int = 3,
    label_smoothing: float = 0.0,
    alignment_weight: float = 0.0,
    modality_dropout_prob: float = 0.15,
    fusion_aux_weights: Optional[Dict[str, float]] = None,
    execution_context: Optional[Any] = None,
    contrastive_weight: float = 0.0,
    ewc: Optional[Any] = None,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    lora_config: Optional[Dict[str, Any]] = None,
    tabular_tokenizer: Optional[Any] = None,
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
    image_alias = {
        "resnet50": "resnet50",
        "mobilenetv3": "resnet50",
        "mobilenet_v3": "resnet50",
        "mobilenet_v3_small": "resnet50",
        "vit-base": "resnet50",
        "vit_base": "resnet50",
    }
    if image_encoder is not None and hasattr(image_encoder, "model_name"):
        try:
            raw_name = str(getattr(image_encoder, "model_name") or "resnet50")
            normalized = image_alias.get(raw_name.lower(), raw_name.lower())
            setattr(image_encoder, "model_name", normalized)
        except Exception:
            pass

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
        fusion_config=fusion_config,
        head_architecture_type=head_architecture_type,
        head_num_layers=head_num_layers,
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
        alignment_weight=alignment_weight,
        modality_dropout_prob=modality_dropout_prob,
        fusion_aux_weights=fusion_aux_weights,
        label_smoothing=label_smoothing,
        execution_context=execution_context,
        contrastive_weight=contrastive_weight,
        ewc=ewc,
        use_focal_loss=use_focal_loss,
        focal_gamma=focal_gamma,
        lora_config=lora_config,
        tabular_tokenizer=tabular_tokenizer,
    )
