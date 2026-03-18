"""Decoder architectures for dense prediction tasks (segmentation)."""

from modelss.decoders.unet import SegmentationModel, UNetDecoder, ResNetSegBackbone

__all__ = ["SegmentationModel", "UNetDecoder", "ResNetSegBackbone"]
