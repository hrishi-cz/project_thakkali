"""
modelss/decoders/unet.py

U-Net segmentation architecture with a ResNet-50 encoder backbone.

Architecture
------------
  ResNetSegBackbone (encoder)
      conv1 + bn1 + relu + maxpool  →  c0   [N, 64,   H/4,  W/4]
      layer1                        →  c1   [N, 256,  H/4,  W/4]
      layer2                        →  c2   [N, 512,  H/8,  W/8]
      layer3                        →  c3   [N, 1024, H/16, W/16]
      layer4                        →  c4   [N, 2048, H/32, W/32]

  UNetDecoder (decoder with skip connections)
      up4: upsample c4 → concat(c3) → conv block  →  d3  [N, 256, H/16, W/16]
      up3: upsample d3 → concat(c2) → conv block  →  d2  [N, 128, H/8,  W/8]
      up2: upsample d2 → concat(c1) → conv block  →  d1  [N, 64,  H/4,  W/4]
      up1: upsample d1               → conv block  →  d0  [N, 32,  H/2,  W/2]
      final: upsample d0 → 1×1 conv                →  out [N, C,   H,    W]

The decoder restores spatial resolution through bilinear upsampling followed
by skip-connection concatenation (standard U-Net pattern).  The final output
has shape ``(N, num_classes, H, W)`` — raw logits for ``CrossEntropyLoss``.

This module does NOT modify ``modelss.encoders.image.ImageEncoder``, which
keeps its flat 512-dim output for classification tasks.

GPU Safety
----------
All modules are standard ``nn.Module`` subclasses.  GPU lifecycle is managed
by the caller (training orchestrator / inference engine) via the standard
``try / except / finally`` pattern with ``torch.cuda.empty_cache()``.
"""

from __future__ import annotations

import logging
from typing import List

import torch
import torch.nn as nn
import torchvision.models as tv_models

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Double-conv building block (Conv → BN → ReLU → Conv → BN → ReLU)
# ---------------------------------------------------------------------------

class _DoubleConv(nn.Module):
    """Two sequential 3×3 conv layers with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# ResNet-50 segmentation backbone (exposes intermediate feature maps)
# ---------------------------------------------------------------------------

class ResNetSegBackbone(nn.Module):
    """
    ResNet-50 backbone that exposes multi-scale feature maps for the decoder.

    Unlike ``modelss.encoders.image.ImageEncoder`` (which applies GAP and
    returns a flat 512-dim vector), this backbone returns a list of spatial
    feature tensors from each ResNet stage — exactly what the U-Net decoder
    needs for skip connections.

    Parameters
    ----------
    pretrained : bool
        Load ImageNet-1k pretrained weights.  Default ``True``.
    freeze : bool
        Freeze all backbone parameters (encoder is not fine-tuned).
        Default ``True``.
    """

    def __init__(self, pretrained: bool = True, freeze: bool = True) -> None:
        super().__init__()

        try:
            weights = (
                tv_models.ResNet50_Weights.IMAGENET1K_V1
                if pretrained else None
            )
            backbone = tv_models.resnet50(weights=weights)
        except TypeError:
            backbone = tv_models.resnet50(pretrained=pretrained)

        # Decompose ResNet into stages for skip-connection access
        self.layer0 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
        )
        self.layer1 = backbone.layer1  # stride 4,   256 channels
        self.layer2 = backbone.layer2  # stride 8,   512 channels
        self.layer3 = backbone.layer3  # stride 16, 1024 channels
        self.layer4 = backbone.layer4  # stride 32, 2048 channels

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

        logger.info(
            "ResNetSegBackbone: pretrained=%s  freeze=%s", pretrained, freeze,
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features.

        Returns
        -------
        list of torch.Tensor
            ``[c1, c2, c3, c4]`` with shapes:
            - c1: ``(N, 256,  H/4,  W/4)``
            - c2: ``(N, 512,  H/8,  W/8)``
            - c3: ``(N, 1024, H/16, W/16)``
            - c4: ``(N, 2048, H/32, W/32)``
        """
        c0 = self.layer0(x)
        c1 = self.layer1(c0)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return [c1, c2, c3, c4]


# ---------------------------------------------------------------------------
# U-Net decoder
# ---------------------------------------------------------------------------

class UNetDecoder(nn.Module):
    """
    Standard U-Net decoder with skip connections.

    Takes multi-scale encoder features and progressively upsamples back to
    the original spatial resolution.

    Parameters
    ----------
    encoder_channels : list of int
        Channel counts from the encoder stages, in order of increasing depth.
        Default ``[256, 512, 1024, 2048]`` (ResNet-50).
    decoder_channels : list of int
        Channel counts for each decoder stage.
        Default ``[256, 128, 64, 32]``.
    num_classes : int
        Number of output segmentation classes (including background).
    """

    def __init__(
        self,
        encoder_channels: List[int] | None = None,
        decoder_channels: List[int] | None = None,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [256, 512, 1024, 2048]
        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32]

        # up4: c4 (2048) upsampled + c3 (1024) skip → 256
        self.up4 = _DoubleConv(encoder_channels[3] + encoder_channels[2],
                               decoder_channels[0])
        # up3: d3 (256) upsampled + c2 (512) skip → 128
        self.up3 = _DoubleConv(decoder_channels[0] + encoder_channels[1],
                               decoder_channels[1])
        # up2: d2 (128) upsampled + c1 (256) skip → 64
        self.up2 = _DoubleConv(decoder_channels[1] + encoder_channels[0],
                               decoder_channels[2])
        # up1: d1 (64) upsampled (no skip) → 32
        self.up1 = _DoubleConv(decoder_channels[2], decoder_channels[3])

        # Final 1×1 conv to produce class logits
        self.final_conv = nn.Conv2d(decoder_channels[3], num_classes,
                                    kernel_size=1)

        self._upsample = nn.Upsample(scale_factor=2, mode="bilinear",
                                     align_corners=False)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Decode encoder features into dense class logits.

        Parameters
        ----------
        features : list of torch.Tensor
            ``[c1, c2, c3, c4]`` from ``ResNetSegBackbone``.

        Returns
        -------
        torch.Tensor
            Shape ``(N, num_classes, H, W)`` — raw logits (no softmax).
            ``H, W`` match the input image spatial dimensions.
        """
        c1, c2, c3, c4 = features

        # Stage 4 → 3: upsample c4 to c3's spatial size, concat, conv
        d3 = self.up4(torch.cat([self._upsample(c4), c3], dim=1))
        # Stage 3 → 2
        d2 = self.up3(torch.cat([self._upsample(d3), c2], dim=1))
        # Stage 2 → 1
        d1 = self.up2(torch.cat([self._upsample(d2), c1], dim=1))
        # Stage 1 → 0 (no skip — upsample to H/2)
        d0 = self.up1(self._upsample(d1))
        # Final upsample to full resolution + 1×1 class projection
        out = self.final_conv(self._upsample(d0))
        return out


# ---------------------------------------------------------------------------
# Complete segmentation model (backbone + decoder)
# ---------------------------------------------------------------------------

class SegmentationModel(nn.Module):
    """
    End-to-end segmentation model: ResNet-50 encoder + U-Net decoder.

    Input:  ``(N, 3, H, W)``  — RGB image batch
    Output: ``(N, num_classes, H, W)``  — per-pixel class logits

    Parameters
    ----------
    num_classes : int
        Segmentation classes including background.  Default ``2``.
    pretrained : bool
        Use ImageNet-pretrained backbone.  Default ``True``.
    freeze_backbone : bool
        Freeze encoder weights (only decoder is trained).  Default ``True``.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = ResNetSegBackbone(
            pretrained=pretrained, freeze=freeze_backbone,
        )
        self.decoder = UNetDecoder(num_classes=num_classes)
        self.num_classes = num_classes

        logger.info(
            "SegmentationModel: num_classes=%d  pretrained=%s  "
            "freeze_backbone=%s",
            num_classes, pretrained, freeze_backbone,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Image batch ``(N, 3, H, W)``.  Should be ImageNet-normalised.

        Returns
        -------
        torch.Tensor
            Raw logits ``(N, num_classes, H, W)``.
        """
        features = self.backbone(x)
        return self.decoder(features)
