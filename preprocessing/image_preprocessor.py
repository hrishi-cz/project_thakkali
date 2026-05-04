"""Image preprocessing – callable torchvision transform pipeline."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
import torchvision.transforms as transforms

_APEX_SEED = int(os.environ.get("APEX_SEED", "42"))
# Seed moved to ImagePreprocessor.__init__ so parallel test workers with different
# APEX_SEED values don't stomp on each other via a module-level global side-effect.


class ImagePreprocessor:
    """
    ImageNet-normalised torchvision transform pipeline.

    The instance is callable (implements ``__call__``) so it can be passed
    directly as the ``transform`` argument to any ``torch.utils.data.Dataset``
    or ``torchvision.datasets.*`` class.

    Usage
    -----
    >>> ip = ImagePreprocessor()
    >>> dataset = MyDataset(transform=ip)   # pass as callable transform
    >>> tensor = ip(pil_image)              # direct call → torch.Tensor [3,224,224]
    """

    def __init__(self, target_size: Tuple[int, int] = (224, 224)) -> None:
        self.target_size = target_size
        self.normalize_mode = "imagenet"
        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]
        self.augment_enabled = True
        self._force_grayscale: bool = False   # set by channels signal
        self._apply_sharpening: bool = False  # set by blur_proxy signal
        torch.manual_seed(_APEX_SEED)         # seed per-instance, not at module import
        self._build_transforms()

    def _build_transforms(self) -> None:
        # Convert mode: grayscale → single channel, else RGB
        convert_op = (
            transforms.Grayscale(num_output_channels=1)
            if self._force_grayscale
            else transforms.Lambda(lambda img: img.convert("RGB"))
        )
        normalize = transforms.Normalize(
            mean=self.normalize_mean, std=self.normalize_std
        )
        self.transforms = transforms.Compose([
            convert_op,
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            normalize,
        ])
        # Augmentation pipeline — append sharpening if blur_proxy signals low quality
        aug_ops = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
        if self._apply_sharpening:
            # Blurry dataset: unsharp-mask via RandomAdjustSharpness
            aug_ops.append(transforms.RandomAdjustSharpness(sharpness_factor=2.5, p=0.5))
        self._augment_transforms = transforms.Compose(aug_ops)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Apply the full transform pipeline to a PIL Image."""
        return self.transforms(image)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """
        Make the preprocessor callable – required for use as a PyTorch
        ``transform`` argument (``DataLoader`` calls ``transform(sample)``).
        """
        return self.preprocess(image)

    def batch_preprocess(self, images: List[Image.Image]) -> torch.Tensor:
        """Process a list of PIL images and stack into a batch tensor ``[N, 3, H, W]``."""
        return torch.stack([self.preprocess(img) for img in images])

    def augment(self, image: Image.Image) -> Image.Image:
        """Apply data augmentation (training only – returns PIL Image)."""
        if not self.augment_enabled:
            return image
        return self._augment_transforms(image)

    def configure(self, plan: Dict[str, Any] | None) -> None:
        """Apply optional runtime overrides from the preprocessing planner (G17 enhanced)."""
        if not isinstance(plan, dict):
            return

        target_size = plan.get("target_size")
        if isinstance(target_size, (list, tuple)) and len(target_size) == 2:
            try:
                h, w = int(target_size[0]), int(target_size[1])
                if h > 0 and w > 0:
                    self.target_size = (h, w)
            except Exception:
                pass

        normalize_spec = plan.get("normalize")
        # Part D.2 — CLIP normalization: detect when CLIP/SigLIP encoder is selected
        _selected_enc = str(plan.get("selected_image_encoder", "") or "").lower()
        if any(k in _selected_enc for k in ("clip", "siglip")):
            self.normalize_mode = "clip"
            self.normalize_mean = [0.481, 0.458, 0.408]
            self.normalize_std = [0.269, 0.261, 0.276]
        elif isinstance(normalize_spec, str):
            if normalize_spec.strip().lower() == "imagenet":
                self.normalize_mode = "imagenet"
                self.normalize_mean = [0.485, 0.456, 0.406]
                self.normalize_std = [0.229, 0.224, 0.225]
            elif normalize_spec.strip().lower() == "clip":
                self.normalize_mode = "clip"
                self.normalize_mean = [0.481, 0.458, 0.408]
                self.normalize_std = [0.269, 0.261, 0.276]
        elif isinstance(normalize_spec, dict):
            mean = normalize_spec.get("mean")
            std = normalize_spec.get("std")
            if isinstance(mean, (list, tuple)) and isinstance(std, (list, tuple)):
                if len(mean) == 3 and len(std) == 3:
                    try:
                        self.normalize_mean = [float(v) for v in mean]
                        self.normalize_std = [float(v) for v in std]
                        self.normalize_mode = "custom"
                    except Exception:
                        pass

        self.augment_enabled = bool(plan.get("augment_train", self.augment_enabled))

        # G17: context-driven augmentation intensity + target_size capping
        sig = plan.get("feature_intelligence") or {}
        img_sig = sig.get("image") or {}
        dataset_size = int(plan.get("dataset_size", 0) or img_sig.get("dataset_size", 0) or 0)
        label_separability = float(plan.get("label_separability", 0.5) or img_sig.get("label_separability", 0.5) or 0.5)
        class_balance = float(plan.get("class_balance", 0.5) or img_sig.get("class_balance", 0.5) or 0.5)
        mean_resolution = float(img_sig.get("mean_resolution", 0.0) or 0.0)

        if dataset_size > 0:
            if dataset_size < 5000 or label_separability < 0.4:
                self.augment_intensity = "strong"
            elif dataset_size > 100_000 and label_separability > 0.7:
                self.augment_intensity = "light"
            else:
                self.augment_intensity = "medium"

            if dataset_size < 1000 and mean_resolution > 0:
                cap = min(224, int(mean_resolution ** 0.5))
                cap = max(32, cap)
                h_cur, w_cur = self.target_size
                if h_cur > cap or w_cur > cap:
                    self.target_size = (cap, cap)

        if class_balance < 0.5:
            self.class_weight_dropout = float(0.5 - class_balance)
        else:
            self.class_weight_dropout = 0.0

        # channels signal: switch normalization stats for grayscale images
        _channels = img_sig.get("channels")
        if isinstance(_channels, list):
            _channels = _channels[0] if _channels else None
        if _channels == "grayscale":
            # ImageNet stats are RGB; grayscale needs single-channel equivalent
            self.normalize_mean = [0.449]   # mean of RGB ImageNet means
            self.normalize_std  = [0.226]   # mean of RGB ImageNet stds
            self.normalize_mode = "grayscale"
            self._force_grayscale = True

        # blur_proxy signal: blurry images → add sharpening augmentation,
        # not additional geometric augmentations that worsen blur further
        _blur = float(img_sig.get("blur_proxy_variance_of_laplacian", -1) or -1)
        if 0 <= _blur < 50:
            # Low Laplacian variance = blurry dataset → mark for sharpening
            self._apply_sharpening = True
        else:
            self._apply_sharpening = False

        self._build_transforms()

    # ------------------------------------------------------------------
    # Config helper (used by /preprocess API endpoint)
    # ------------------------------------------------------------------

    def get_default_config(self) -> Dict[str, Any]:
        h, w = self.target_size
        return {
            "target_size": list(self.target_size),
            "normalize": self.normalize_mode,
            "normalize_mean": list(self.normalize_mean),
            "normalize_std": list(self.normalize_std),
            "output_shape": f"[3, {h}, {w}]",
        }
