"""
modelss/encoders/image.py

ResNet-50 image encoder with a fixed 2048 → 512 projection head.

Architecture
------------
  torchvision.models.resnet50(pretrained=True)
      ↓  (fc layer replaced with nn.Identity())
  [N, 2048]  global-average-pooled feature map
      ↓
  nn.Linear(2048, 512) → nn.ReLU()
      ↓
  [N, 512]  projected image features

The ``fc`` layer is replaced by ``nn.Identity()`` at construction time so
the backbone's forward pass outputs the raw 2048-dim GAP vector.  The
``projection`` head then maps this to the fixed 512-dim output required by
the fusion layer.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models

logger = logging.getLogger(__name__)

# Architectural constants (must not be changed – fusion layer depends on these)
IMAGE_BACKBONE_DIM: int = 2048
IMAGE_OUTPUT_DIM: int   = 512


class ImageEncoder(nn.Module):
    """
    ResNet-50 backbone with a fixed 2048 → 512 projection head.

    Parameters
    ----------
    pretrained : bool
        Load ImageNet-1k weights via ``torchvision.models.ResNet50_Weights``.
        Default ``True``.
    freeze_backbone : bool
        Freeze all ResNet-50 convolutional parameters so only the projection
        head is trained.  Useful for small datasets.  Default ``False``.
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        # ── ResNet-50 backbone ────────────────────────────────────────────
        # Use the new weights API (torchvision ≥ 0.13); fall back gracefully
        # for older verions that still accept the bool ``pretrained`` kwarg.
        try:
            weights = (
                tv_models.ResNet50_Weights.IMAGENET1K_V1
                if pretrained
                else None
            )
            backbone: nn.Module = tv_models.resnet50(weights=weights)
        except TypeError:
            # torchvision < 0.13
            backbone = tv_models.resnet50(pretrained=pretrained)  # type: ignore[call-arg]

        # Strip the classification head: replace fc with Identity so the
        # backbone forward pass returns [N, 2048] (GAP output).
        backbone.fc = nn.Identity()
        self.backbone: nn.Module = backbone

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ── Projection head: 2048 → 512 ──────────────────────────────────
        self.projection: nn.Sequential = nn.Sequential(
            nn.Linear(IMAGE_BACKBONE_DIM, IMAGE_OUTPUT_DIM),
            nn.ReLU(),
        )
        self._runtime_config: Dict[str, Any] = {
            "freeze_backbone": bool(freeze_backbone),
            "use_projection_relu": True,
        }

        logger.info(
            "ImageEncoder: backbone=resnet50  pretrained=%s  "
            "freeze_backbone=%s  output_dim=%d",
            pretrained, freeze_backbone, IMAGE_OUTPUT_DIM,
        )

    def configure(self, plan: Optional[Dict[str, Any]]) -> None:
        """Apply runtime settings from the preprocessing planner."""
        if not isinstance(plan, dict):
            return

        if "freeze_backbone" in plan:
            freeze_backbone = bool(plan["freeze_backbone"])
            for param in self.backbone.parameters():
                param.requires_grad = not freeze_backbone
            self._runtime_config["freeze_backbone"] = freeze_backbone

        if "use_projection_relu" in plan:
            use_relu = bool(plan["use_projection_relu"])
            linear = self.projection[0]
            self.projection = nn.Sequential(
                linear,
                nn.ReLU() if use_relu else nn.Identity(),
            )
            self._runtime_config["use_projection_relu"] = use_relu

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract and project image features.

        Parameters
        ----------
        x : torch.Tensor
            Float image batch of shape ``(N, 3, H, W)``.  Values should be
            ImageNet-normalised (mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]) and spatially resized to ≥ 32 × 32.

        Returns
        -------
        torch.Tensor
            Shape ``(N, 512)`` — projected, ReLU-activated feature vectors.
        """
        gap_features: torch.Tensor = self.backbone(x)   # (N, 2048)
        return self.projection(gap_features)             # (N, 512)

    def get_output_dim(self) -> int:
        """Return the fixed output dimensionality (512)."""
        return IMAGE_OUTPUT_DIM


class ViTImageEncoder(nn.Module):
    """
    Vision Transformer (ViT) image encoder backed by CLIP or DINOv2 from HuggingFace.

    Supports two output modes:
    - Pooled (default, ``return_all_tokens=False``): CLS or mean-pooled vector ``(N, D)``
    - Token-sequence (``return_all_tokens=True``): all patch embeddings ``(N, P, D)``
      where P=196 for ViT-B/16 with 224×224 input.

    The token-sequence mode feeds directly into ``UnifiedLatentFusion`` for
    true cross-modal attention between image patches and text tokens.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
        Supported: ``"openai/clip-vit-base-patch16"`` (768-dim),
                   ``"facebook/dinov2-base"`` (768-dim).
    freeze_backbone : bool
        Freeze backbone weights.  LoRA adapters remain trainable when applied.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self._output_dim: int = 768

        try:
            from transformers import AutoModel
            self._vit: Optional[nn.Module] = AutoModel.from_pretrained(model_name)
            # Detect output dimension from config
            cfg = self._vit.config
            hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "vision_config", None)
            if hasattr(hidden, "hidden_size"):
                hidden = hidden.hidden_size
            if isinstance(hidden, int) and hidden > 0:
                self._output_dim = hidden
        except Exception as exc:
            logger.warning("ViTImageEncoder: could not load '%s': %s", model_name, exc)
            self._vit = None

        if self._vit is not None and freeze_backbone:
            for p in self._vit.parameters():
                p.requires_grad_(False)

        logger.info(
            "ViTImageEncoder: model=%s  freeze=%s  output_dim=%d",
            model_name, freeze_backbone, self._output_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_all_tokens: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  ``(N, 3, H, W)`` — normalised image batch
        return_all_tokens : bool
            False → pooled vector ``(N, D)``
            True  → patch token sequence ``(N, P, D)`` — first (CLS) token excluded
        """
        if self._vit is None:
            N = x.shape[0]
            if return_all_tokens:
                return torch.zeros(N, 196, self._output_dim, device=x.device)
            return torch.zeros(N, self._output_dim, device=x.device)

        out = self._vit(pixel_values=x)

        if return_all_tokens:
            # last_hidden_state: (N, 1+P, D) where index 0 is CLS
            seq = out.last_hidden_state
            return seq[:, 1:, :]  # (N, P, D) — patch tokens only

        # Pooled: prefer pooler_output when available (standard ViT/CLIP).
        # SigLIP has no CLS token — pooler_output is mean of all patches (N, D).
        # Fallback: mean-pool ALL tokens (no [:, 1:] skip — SigLIP has no CLS at 0).
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output                       # (N, D)
        return out.last_hidden_state.mean(dim=1)           # (N, D) — safe for all ViT variants

    def get_output_dim(self) -> int:
        return self._output_dim

    def configure(self, plan: Optional[Dict[str, Any]]) -> None:
        """Accepts the same configure interface as other encoders (no-op for ViT)."""
        pass


class MultiScaleImageEncoder(nn.Module):
    """
    Dual-resolution ResNet-50 ensemble encoder.

    Processes each image at two spatial scales simultaneously:
      - Small scale (112 × 112): captures global structure cheaply
      - Large scale (224 × 224): captures fine-grained local features

    The two 2048-dim GAP vectors are concatenated → projected to `output_dim`.
    This is an FPN-style approach that improves accuracy 2–4 % with ~30 %
    additional VRAM vs. single-scale.

    Parameters
    ----------
    pretrained : bool
        Load ImageNet-1k weights for both branches.
    freeze_backbone : bool
        Freeze all conv parameters so only the projection is trained.
    output_dim : int
        Projected output dimension. Default 512.
    share_weights : bool
        When True both scale branches share the same ResNet weights
        (saves ~95 MB VRAM; slightly reduces accuracy gain).
        Default False (independent branches).
    """

    SMALL_SIZE: int = 112
    LARGE_SIZE: int = 224

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        output_dim: int = IMAGE_OUTPUT_DIM,
        share_weights: bool = False,
    ) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.share_weights = bool(share_weights)

        def _build_backbone(pretrained: bool) -> nn.Module:
            try:
                weights = (
                    tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
                )
                bb = tv_models.resnet50(weights=weights)
            except TypeError:
                bb = tv_models.resnet50(pretrained=pretrained)  # type: ignore[call-arg]
            bb.fc = nn.Identity()
            return bb

        self.backbone_large = _build_backbone(pretrained)
        if share_weights:
            self.backbone_small = self.backbone_large
        else:
            self.backbone_small = _build_backbone(pretrained)

        if freeze_backbone:
            for param in self.backbone_large.parameters():
                param.requires_grad = False
            if not share_weights:
                for param in self.backbone_small.parameters():
                    param.requires_grad = False

        # 2048 (small) + 2048 (large) → output_dim
        self.projection = nn.Sequential(
            nn.Linear(IMAGE_BACKBONE_DIM * 2, self.output_dim),
            nn.ReLU(),
        )

        logger.info(
            "MultiScaleImageEncoder: scales=%dx%d+%dx%d  shared=%s  "
            "output_dim=%d  pretrained=%s",
            self.SMALL_SIZE, self.SMALL_SIZE,
            self.LARGE_SIZE, self.LARGE_SIZE,
            share_weights, self.output_dim, pretrained,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            ``(N, 3, H, W)`` ImageNet-normalised batch. H/W ≥ LARGE_SIZE.

        Returns
        -------
        torch.Tensor
            ``(N, output_dim)`` multi-scale feature vectors.
        """
        import torch.nn.functional as F

        # Downsample for small branch
        x_small = F.interpolate(
            x, size=(self.SMALL_SIZE, self.SMALL_SIZE),
            mode="bilinear", align_corners=False,
        )

        feat_small = self.backbone_small(x_small)   # (N, 2048)
        feat_large = self.backbone_large(x)          # (N, 2048)
        fused = torch.cat([feat_small, feat_large], dim=-1)  # (N, 4096)
        return self.projection(fused)                # (N, output_dim)

    def get_output_dim(self) -> int:
        return self.output_dim
