"""
JIT Hardware Profiler & Constrained Encoder Optimizer.

Replaces brittle IF/THEN dataset-size heuristics with mathematically sound
hardware-aware model selection.  All decisions are driven by live VRAM
measurement — zero magic numbers.

Architecture
------------
1. **Encoder Registry** — catalogue of candidate encoders per modality with
   factory functions that return instantiated ``nn.Module`` objects matching
   the ``build_trainer(image_encoder=…, text_encoder=…)`` interface contract.
   Each entry carries a *capacity score* (total trainable + frozen parameter
   count) used by the constrained optimizer.

2. **JIT Dry-Run Memory Estimator** — measures peak GPU footprint of a
   candidate encoder by running a dummy batch through it inside
   ``torch.no_grad()`` and recording the delta of
   ``torch.cuda.max_memory_allocated()``.  Cleans up with
   ``torch.cuda.empty_cache()`` after every probe.

3. **Constrained Optimizer** — maximizes total capacity score C(m) subject
   to the constraint::

       F_peak(m, b) ≤ η · V_avail

   where:
     - F_peak = sum of per-encoder peak footprints (from dry-run)
     - η = 0.85 (safety margin — reserves 15% VRAM for activations/optimizer)
     - V_avail = free VRAM from ``torch.cuda.mem_get_info()``
     - b = batch_size from Phase 4

   Selection is exhaustive over the Cartesian product of candidate tiers
   (at most 3 × 2 × 2 = 12 combinations), sorted descending by capacity.
   The first feasible combination wins.

CPU Safeguard
-------------
When ``torch.cuda.is_available()`` returns ``False``, the profiler is
bypassed entirely.  The lightest encoder in every modality is selected
unconditionally — no dry-runs, no memory probes.

Interface Contract
------------------
``JITEncoderSelector.select()`` returns a ``JITSelectionResult`` whose
``image_encoder`` / ``text_encoder`` fields are instantiated, frozen,
eval-mode ``nn.Module`` objects ready to be passed directly to
``build_trainer()``.  The ``tabular_encoder`` field is unused by the
current ``build_trainer`` but included for forward compatibility.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# 1. ENCODER REGISTRY
# ───────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EncoderSpec:
    """
    Immutable specification for a single encoder candidate.

    Attributes
    ----------
    name : str
        Human-readable name for logging and UI display.
    factory : Callable[[], nn.Module]
        Zero-argument callable that returns a fresh, frozen, eval-mode
        ``nn.Module``.  Called exactly once per selection.
    output_dim : int
        Dimensionality of the encoder's output tensor.  Must match the
        fusion layer's expectation (512 for image, 768 for text).
    capacity : int
        Total parameter count (trainable + frozen) — the objective value
        that the constrained optimizer maximizes.
    dummy_input_fn : Callable[[int, torch.device], torch.Tensor | list]
        Returns a dummy input suitable for ``encoder.forward()`` with
        batch size N on the given device.  Used by the dry-run profiler.
    """
    name: str
    factory: Callable[[], nn.Module]
    output_dim: int
    capacity: int
    dummy_input_fn: Callable


@dataclass(frozen=True)
class TabularEncoderSpec:
    """
    Specification for a tabular encoder candidate.

    Unlike image/text ``EncoderSpec``, this stores a **class reference**
    (not a factory returning a frozen instance) because tabular encoders
    are trainable and must be freshly instantiated per Optuna trial.
    """
    name: str
    encoder_class: type       # Must accept ``(input_dim: int)`` as first arg
    output_dim: int           # Fixed output dimensionality (16)
    capacity: int             # Approximate parameter count for ranking


def _count_params(module: nn.Module) -> int:
    """Total parameter count (trainable + frozen)."""
    return sum(p.numel() for p in module.parameters())


def _freeze_and_eval(module: nn.Module) -> nn.Module:
    """Freeze all parameters and set to eval mode."""
    module.eval()
    for p in module.parameters():
        p.requires_grad = False
    return module


# ── Vision encoder factories ──────────────────────────────────────────────

class _VisionEncoderWrapper(nn.Module):
    """
    Thin wrapper exposing ``get_output_dim()`` for non-ImageEncoder
    backbones.  Ensures all JIT-selected vision encoders share the same
    minimal interface consumed by ``_encode_batch()`` and Phase 7.
    """

    def __init__(self, backbone: nn.Module, projection: nn.Module, output_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.projection = projection
        self._output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.projection(features)

    def get_output_dim(self) -> int:
        return self._output_dim


def _make_mobilenet_v3() -> nn.Module:
    """MobileNetV3-Small backbone with projection to 512-dim output."""
    import torchvision.models as tv

    try:
        weights = tv.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        backbone = tv.mobilenet_v3_small(weights=weights)
    except (TypeError, AttributeError):
        backbone = tv.mobilenet_v3_small(pretrained=True)

    # MobileNetV3-Small classifier[0].in_features == 576
    in_features = backbone.classifier[0].in_features
    backbone.classifier = nn.Identity()

    projection = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
    )
    encoder = _VisionEncoderWrapper(backbone, projection, output_dim=512)
    return _freeze_and_eval(encoder)


def _make_efficientnet_b0() -> nn.Module:
    """EfficientNet-B0 backbone with projection to 512-dim output."""
    import torchvision.models as tv

    try:
        weights = tv.EfficientNet_B0_Weights.IMAGENET1K_V1
        backbone = tv.efficientnet_b0(weights=weights)
    except (TypeError, AttributeError):
        backbone = tv.efficientnet_b0(pretrained=True)

    in_features = backbone.classifier[-1].in_features
    backbone.classifier = nn.Identity()

    projection = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
    )
    encoder = _VisionEncoderWrapper(backbone, projection, output_dim=512)
    return _freeze_and_eval(encoder)


def _make_resnet50() -> nn.Module:
    """Standard ResNet-50 — delegates to the existing ImageEncoder."""
    from models.encoders.image import ImageEncoder
    enc = ImageEncoder(pretrained=True, freeze_backbone=True)
    return _freeze_and_eval(enc)


def _make_multiscale_resnet50() -> nn.Module:
    """Dual-resolution ResNet-50 (112×112 + 224×224) — higher accuracy, ~30% VRAM overhead."""
    from modelss.encoders.image import MultiScaleImageEncoder
    enc = MultiScaleImageEncoder(pretrained=True, freeze_backbone=True, share_weights=False)
    return _freeze_and_eval(enc)


def _make_convnext_tiny() -> nn.Module:
    """ConvNeXt-Tiny backbone with projection to 512-dim output.

    ConvNeXt's original classifier is ``Sequential(LayerNorm2d, Flatten,
    Linear)``.  We cannot simply replace it with ``nn.Identity()`` because
    the ``Flatten`` step would be lost, producing ``(N, 768, H', W')``
    instead of ``(N, 768)``.  Instead, we compose ``features + avgpool +
    Flatten`` to get a clean ``(N, 768)`` feature vector.
    """
    import torchvision.models as tv

    try:
        weights = tv.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        backbone = tv.convnext_tiny(weights=weights)
    except (TypeError, AttributeError):
        backbone = tv.convnext_tiny(pretrained=True)

    # ConvNeXt feature dim = 768 (last stage output channels)
    in_features = backbone.classifier[-1].in_features  # 768

    # Compose feature extractor: features → avgpool → flatten
    # This avoids the shape mismatch from removing the classifier wholesale.
    flat_backbone = nn.Sequential(
        backbone.features,
        backbone.avgpool,       # AdaptiveAvgPool2d(1)
        nn.Flatten(1),          # (N, 768, 1, 1) → (N, 768)
    )

    projection = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
    )
    encoder = _VisionEncoderWrapper(flat_backbone, projection, output_dim=512)
    return _freeze_and_eval(encoder)


# ── Text encoder factories ────────────────────────────────────────────────

def _make_minilm() -> nn.Module:
    """all-MiniLM-L6-v2 — lightweight, 384-dim → projected to 768."""
    from models.encoders.text import TextEncoder
    enc = TextEncoder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        max_length=128,
        freeze_backbone=True,
    )
    # The projection Linear(384, 768) is initialized with random weights.
    # Use Xavier uniform initialization before freezing so the projection
    # produces a meaningful (non-random) mapping.
    if enc._projection is not None:
        nn.init.xavier_uniform_(enc._projection.weight)
        nn.init.zeros_(enc._projection.bias)
    return _freeze_and_eval(enc)


def _make_bert_base() -> nn.Module:
    """Standard BERT-base-uncased — delegates to existing TextEncoder."""
    from models.encoders.text import TextEncoder
    enc = TextEncoder(
        model_name="bert-base-uncased",
        max_length=128,
        freeze_backbone=True,
    )
    return _freeze_and_eval(enc)


def _make_distilbert() -> nn.Module:
    """DistilBERT â€” lighter 768-dim text encoder."""
    from models.encoders.text import TextEncoder
    enc = TextEncoder(
        model_name="distilbert-base-uncased",
        max_length=128,
        freeze_backbone=True,
    )
    return _freeze_and_eval(enc)


def _make_deberta() -> nn.Module:
    """DeBERTa-v3-base — stronger text understanding, heavier."""
    from models.encoders.text import TextEncoder
    enc = TextEncoder(
        model_name="microsoft/deberta-v3-base",
        max_length=128,
        freeze_backbone=True,
    )
    return _freeze_and_eval(enc)


# ── Dummy input generators ────────────────────────────────────────────────

def _dummy_image(batch_size: int, device: torch.device) -> torch.Tensor:
    """[N, 3, 224, 224] random image tensor."""
    return torch.randn(batch_size, 3, 224, 224, device=device)


def _dummy_text(batch_size: int, device: torch.device) -> list:
    """List of N dummy text strings (TextEncoder.forward accepts strings)."""
    return ["dummy text input for profiling"] * batch_size


# ── Registry construction ─────────────────────────────────────────────────
# Capacity scores are approximate parameter counts used for ranking only.
# They do NOT need to be precise — they just order candidates correctly.

VISION_REGISTRY: List[EncoderSpec] = [
    EncoderSpec(
        name="MobileNetV3-Small",
        factory=_make_mobilenet_v3,
        output_dim=512,
        capacity=2_500_000,       # ~2.5M params
        dummy_input_fn=_dummy_image,
    ),
    EncoderSpec(
        name="EfficientNet-B0",
        factory=_make_efficientnet_b0,
        output_dim=512,
        capacity=5_300_000,       # ~5.3M params
        dummy_input_fn=_dummy_image,
    ),
    EncoderSpec(
        name="ResNet-50",
        factory=_make_resnet50,
        output_dim=512,
        capacity=25_600_000,      # ~25.6M params
        dummy_input_fn=_dummy_image,
    ),
    EncoderSpec(
        name="ConvNeXt-Tiny",
        factory=_make_convnext_tiny,
        output_dim=512,
        capacity=28_600_000,      # ~28.6M params
        dummy_input_fn=_dummy_image,
    ),
    EncoderSpec(
        name="MultiScale-ResNet50",
        factory=_make_multiscale_resnet50,
        output_dim=512,
        capacity=51_200_000,      # ~2× ResNet-50 params (two branches)
        dummy_input_fn=_dummy_image,
    ),
]

# ── ViT encoders (optional — require 'transformers' package) ─────────────
def _make_clip_vit_b16() -> nn.Module:
    """CLIP ViT-B/16 — 86M params, 768-dim, supports patch tokens for ULA."""
    from modelss.encoders.image import ViTImageEncoder
    return ViTImageEncoder("openai/clip-vit-base-patch16", freeze_backbone=True)

def _make_dinov2_vitb14() -> nn.Module:
    """DINOv2 ViT-B/14 — 86M params, 768-dim, superior dense patch features."""
    from modelss.encoders.image import ViTImageEncoder
    return ViTImageEncoder("facebook/dinov2-base", freeze_backbone=True)

try:
    # Register ViT encoders only if transformers is importable
    import transformers as _tf_check  # noqa: F401
    VISION_REGISTRY.append(EncoderSpec(
        name="CLIP-ViT-B/16",
        factory=_make_clip_vit_b16,
        output_dim=768,
        capacity=86_000_000,
        dummy_input_fn=_dummy_image,
    ))
    VISION_REGISTRY.append(EncoderSpec(
        name="DINOv2-ViT-B/14",
        factory=_make_dinov2_vitb14,
        output_dim=768,
        capacity=86_000_000,
        dummy_input_fn=_dummy_image,
    ))
    logger.info("JIT registry: CLIP-ViT-B/16 and DINOv2-ViT-B/14 added")
except ImportError:
    logger.debug("JIT registry: transformers not installed — ViT encoders unavailable")

TEXT_REGISTRY: List[EncoderSpec] = [
    EncoderSpec(
        name="MiniLM-L6-v2",
        factory=_make_minilm,
        output_dim=768,
        capacity=22_700_000,      # ~22.7M params
        dummy_input_fn=_dummy_text,
    ),
    EncoderSpec(
        name="DistilBERT-base-uncased",
        factory=_make_distilbert,
        output_dim=768,
        capacity=66_400_000,      # ~66.4M params
        dummy_input_fn=_dummy_text,
    ),
    EncoderSpec(
        name="BERT-base-uncased",
        factory=_make_bert_base,
        output_dim=768,
        capacity=109_500_000,     # ~109.5M params
        dummy_input_fn=_dummy_text,
    ),
    EncoderSpec(
        name="DeBERTa-v3-base",
        factory=_make_deberta,
        output_dim=768,
        capacity=183_800_000,     # ~183.8M params
        dummy_input_fn=_dummy_text,
    ),
]

# Sorted descending by capacity for the constrained optimizer
VISION_REGISTRY.sort(key=lambda s: s.capacity, reverse=True)
TEXT_REGISTRY.sort(key=lambda s: s.capacity, reverse=True)


# ── Tabular encoder registry ─────────────────────────────────────────────
# Tabular encoders are tiny (~5-12K params) and trainable (not frozen), so
# no VRAM profiling is needed.  The JIT selector picks the highest-capacity
# type; the orchestrator instantiates a fresh copy per Optuna trial.

from models.encoders.tabular import TabularEncoder as _TabularMLPClass
from models.encoders.tabular import GRNTabularEncoder as _TabularGRNClass
from models.encoders.tabular import FTTransformerEncoder as _TabularFTTClass


class _FTTWrapper:
    """Wrapper so FTTransformerEncoder is constructed with only input_dim (like MLP/GRN)."""
    def __new__(cls, input_dim: int):
        return _TabularFTTClass(input_dim=input_dim, output_dim=64)


TABULAR_REGISTRY: List[TabularEncoderSpec] = [
    TabularEncoderSpec(
        name="FTTransformer",
        encoder_class=_FTTWrapper,
        output_dim=64,
        capacity=150_000,     # ~150K params (d_token=96, 3 layers, 8 heads)
    ),
    TabularEncoderSpec(
        name="GRN",
        encoder_class=_TabularGRNClass,
        output_dim=16,
        capacity=12_000,      # ~12K params (with hidden_dim=64)
    ),
    TabularEncoderSpec(
        name="MLP",
        encoder_class=_TabularMLPClass,
        output_dim=16,
        capacity=5_000,       # ~5K params
    ),
]
TABULAR_REGISTRY.sort(key=lambda s: s.capacity, reverse=True)


# ───────────────────────────────────────────────────────────────────────────
# 1b. PUBLIC REGISTRATION API  (hot-loadable encoder plugins)
# ───────────────────────────────────────────────────────────────────────────

def _validate_encoder_factory(
    name: str, factory: Callable, output_dim: int, modality: str,
) -> None:
    """Validate that a factory is callable and output_dim is positive."""
    if not callable(factory):
        raise TypeError(
            f"Encoder '{name}': factory must be callable, got {type(factory)}"
        )
    if output_dim <= 0:
        raise ValueError(
            f"Encoder '{name}': output_dim must be positive, got {output_dim}"
        )


def _validate_tabular_encoder_class(
    name: str, encoder_class: type, output_dim: int,
) -> None:
    """Validate that a tabular encoder class accepts (input_dim: int)."""
    if not isinstance(encoder_class, type):
        raise TypeError(
            f"Tabular encoder '{name}': encoder_class must be a class, "
            f"got {type(encoder_class)}"
        )
    if output_dim <= 0:
        raise ValueError(
            f"Tabular encoder '{name}': output_dim must be positive"
        )
    import inspect
    sig = inspect.signature(encoder_class.__init__)
    params = list(sig.parameters.keys())
    if "input_dim" not in params and len(params) < 2:
        raise TypeError(
            f"Tabular encoder '{name}': class must accept (input_dim: int) "
            f"as first positional arg"
        )


def register_vision_encoder(
    name: str,
    factory: Callable[[], nn.Module],
    output_dim: int,
    capacity: int,
    dummy_input_fn: Callable,
) -> None:
    """
    Register a custom vision encoder into the JIT selection pool.

    Parameters
    ----------
    name : str
        Human-readable name for logging.
    factory : callable
        Zero-arg callable returning a frozen, eval-mode ``nn.Module``
        with a ``get_output_dim()`` method.
    output_dim : int
        Dimensionality of the encoder's output tensor.
    capacity : int
        Total parameter count (used for ranking).
    dummy_input_fn : callable
        ``(batch_size, device) -> Tensor`` for VRAM profiling dry-runs.
    """
    _validate_encoder_factory(name, factory, output_dim, "vision")
    VISION_REGISTRY.append(EncoderSpec(
        name=name, factory=factory, output_dim=output_dim,
        capacity=capacity, dummy_input_fn=dummy_input_fn,
    ))
    VISION_REGISTRY.sort(key=lambda s: s.capacity, reverse=True)
    logger.info("Registered custom vision encoder: %s (capacity=%d)", name, capacity)


def register_text_encoder(
    name: str,
    factory: Callable[[], nn.Module],
    output_dim: int,
    capacity: int,
    dummy_input_fn: Callable,
) -> None:
    """
    Register a custom text encoder into the JIT selection pool.

    Same contract as ``register_vision_encoder`` but for text modality.
    """
    _validate_encoder_factory(name, factory, output_dim, "text")
    TEXT_REGISTRY.append(EncoderSpec(
        name=name, factory=factory, output_dim=output_dim,
        capacity=capacity, dummy_input_fn=dummy_input_fn,
    ))
    TEXT_REGISTRY.sort(key=lambda s: s.capacity, reverse=True)
    logger.info("Registered custom text encoder: %s (capacity=%d)", name, capacity)


def register_tabular_encoder(
    name: str,
    encoder_class: type,
    output_dim: int,
    capacity: int,
) -> None:
    """
    Register a custom tabular encoder class into the JIT selection pool.

    Parameters
    ----------
    name : str
        Human-readable name for logging.
    encoder_class : type
        A class whose constructor accepts ``(input_dim: int)`` and returns
        an ``nn.Module`` with a ``get_output_dim()`` method.
    output_dim : int
        Fixed output dimensionality of the encoder.
    capacity : int
        Approximate parameter count (used for ranking).
    """
    _validate_tabular_encoder_class(name, encoder_class, output_dim)
    TABULAR_REGISTRY.append(TabularEncoderSpec(
        name=name, encoder_class=encoder_class,
        output_dim=output_dim, capacity=capacity,
    ))
    TABULAR_REGISTRY.sort(key=lambda s: s.capacity, reverse=True)
    logger.info("Registered custom tabular encoder: %s (capacity=%d)", name, capacity)


# ───────────────────────────────────────────────────────────────────────────
# 2. JIT DRY-RUN MEMORY ESTIMATOR
# ───────────────────────────────────────────────────────────────────────────

def estimate_peak_memory(
    encoder: nn.Module,
    dummy_input: Any,
    device: torch.device,
) -> int:
    """
    Measure peak GPU memory footprint of a single forward pass.

    Steps
    -----
    1. Move encoder to ``device``.
    2. Reset ``torch.cuda.max_memory_allocated()``.
    3. Run one forward pass inside ``torch.no_grad()``.
    4. Record the delta of ``max_memory_allocated()``.
    5. Move encoder back to CPU and call ``torch.cuda.empty_cache()``.

    Parameters
    ----------
    encoder : nn.Module
        The encoder to profile (already frozen and in eval mode).
    dummy_input : Tensor | list
        Input matching the encoder's ``forward()`` signature.
    device : torch.device
        The CUDA device to profile on.

    Returns
    -------
    int
        Peak memory in bytes consumed by the encoder + activations.
    """
    encoder.to(device)

    # Record baseline BEFORE the forward pass
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)

    try:
        with torch.no_grad():
            if isinstance(dummy_input, torch.Tensor):
                dummy_input = dummy_input.to(device)
            _ = encoder(dummy_input)

        peak = torch.cuda.max_memory_allocated(device)
        footprint = peak - baseline

    finally:
        # ── DRY-RUN SAFETY: unconditional cleanup ─────────────────────
        encoder.cpu()
        # Delete intermediate activations
        if isinstance(dummy_input, torch.Tensor):
            del dummy_input
        gc.collect()
        torch.cuda.empty_cache()

    return max(footprint, 0)


# ───────────────────────────────────────────────────────────────────────────
# 3. JIT ENCODER SELECTOR (Constrained Optimizer)
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class JITSelectionResult:
    """
    Output of the constrained encoder optimizer.

    Fields match the ``build_trainer()`` interface contract:
      - ``image_encoder``: instantiated, frozen ``nn.Module`` or ``None``
      - ``text_encoder``:  instantiated, frozen ``nn.Module`` or ``None``
      - ``tabular_encoder_class``: class reference for per-trial instantiation
        (tabular encoders are trainable, not frozen, so no instance is shared)

    Ancillary fields provide transparency for logging and the API response.
    """
    image_encoder: Optional[nn.Module] = None
    text_encoder: Optional[nn.Module] = None
    image_encoder_name: Optional[str] = None
    text_encoder_name: Optional[str] = None
    tabular_encoder_name: Optional[str] = None
    tabular_encoder_class: Optional[type] = None
    tabular_encoder_output_dim: int = 16
    total_capacity: int = 0
    total_peak_memory_bytes: int = 0
    vram_available_bytes: int = 0
    vram_budget_bytes: int = 0          # η · V_avail
    selection_method: str = "jit_profiler"
    rationale: Dict[str, str] = field(default_factory=dict)


class JITEncoderSelector:
    """
    Hardware-aware encoder selector using live VRAM profiling.

    Maximizes total model capacity subject to the peak-memory budget
    constraint.  Falls back to lightest encoders on CPU or when no
    feasible combination fits in VRAM.

    Parameters
    ----------
    safety_margin : float
        Fraction of available VRAM to use as budget.  Default 0.85
        (reserves 15% for optimizer states, gradient accumulation buffers,
        and DataLoader prefetch memory).
    batch_size : int
        Batch size for the dry-run forward pass.  Should match the Phase-4
        selected batch size so the memory estimate is realistic.
    """

    # Safety margin — fraction of free VRAM available for encoders.
    # The remaining 15% covers optimizer state + activation peaks that
    # exceed the single-batch dry-run estimate.
    ETA: float = 0.85

    def __init__(
        self,
        safety_margin: float = 0.85,
        batch_size: int = 16,
    ) -> None:
        self.ETA = safety_margin
        self._batch_size = batch_size

    def _resolve_tabular_spec(self, preferred_tabular: Optional[str]) -> Optional[TabularEncoderSpec]:
        if not TABULAR_REGISTRY:
            return None

        pref = str(preferred_tabular or "").strip().lower()
        if not pref:
            return TABULAR_REGISTRY[0]

        if pref in {"simple", "mlp"}:
            target_name = "mlp"
        elif pref in {"interpretable", "sota", "grn", "xgboost", "lightgbm"}:
            target_name = "grn"
        else:
            target_name = pref

        for spec in TABULAR_REGISTRY:
            if spec.name.strip().lower() == target_name:
                return spec

        return TABULAR_REGISTRY[0]

    @staticmethod
    def _prioritize_encoder_specs(
        candidates: List[EncoderSpec],
        preferred_name: Optional[str],
    ) -> List[EncoderSpec]:
        """Move a preferred encoder to the front while retaining fallbacks."""
        if not candidates:
            return []

        pref = str(preferred_name or "").strip().lower()
        if not pref:
            return list(candidates)

        alias_map = {
            "mobilenet": "mobilenetv3-small",
            "mobilenetv3": "mobilenetv3-small",
            "efficientnet": "efficientnet-b0",
            "efficientnet_b0": "efficientnet-b0",
            "resnet": "resnet-50",
            "resnet50": "resnet-50",
            "convnext": "convnext-tiny",
            "distilbert": "distilbert-base-uncased",
            "bert": "bert-base-uncased",
            "deberta": "deberta-v3-base",
        }
        target = alias_map.get(pref, pref)

        preferred: List[EncoderSpec] = []
        fallback: List[EncoderSpec] = []
        for spec in candidates:
            spec_name = spec.name.strip().lower()
            if spec_name == target or target in spec_name or spec_name in target:
                preferred.append(spec)
            else:
                fallback.append(spec)

        return preferred + fallback if preferred else list(candidates)

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def select(
        self,
        modalities: List[str],
        device: Optional[torch.device] = None,
        preferred_tabular: Optional[str] = None,
        preferred_text: Optional[str] = None,
        preferred_image: Optional[str] = None,
    ) -> JITSelectionResult:
        """
        Select the highest-capacity encoder combination that fits in VRAM.

        Parameters
        ----------
        modalities : list of str
            Active modalities for this training run.  Subset of
            ``["image", "text", "tabular"]``.
        device : torch.device or None
            Target CUDA device.  Auto-detected when ``None``.
        preferred_tabular : str or None
            Optional preferred tabular encoder key/name derived from
            model-selection or probe results (e.g. ``"GRN"``, ``"MLP"``,
            ``"interpretable"``).

        Returns
        -------
        JITSelectionResult
            Contains instantiated, frozen encoder modules ready for
            ``build_trainer(image_encoder=…, text_encoder=…)``.
        """
        need_image = "image" in modalities
        need_text = "text" in modalities
        need_tabular = "tabular" in modalities

        # ── Tabular encoder selection (no VRAM profiling needed) ───────
        # Tabular encoders are tiny (~5-12K params) and trainable, so we
        # simply pick the highest-capacity type.  The orchestrator will
        # instantiate a fresh copy per Optuna trial.
        _tab_spec: Optional[TabularEncoderSpec] = None
        if need_tabular and TABULAR_REGISTRY:
            _tab_spec = self._resolve_tabular_spec(preferred_tabular)
            logger.info(
                "JITEncoderSelector: tabular encoder type = %s "
                "(capacity=%d, trainable per-trial)",
                _tab_spec.name, _tab_spec.capacity,
            )

        # ── CPU SAFEGUARD ─────────────────────────────────────────────
        if not torch.cuda.is_available():
            logger.info(
                "JITEncoderSelector: CUDA unavailable -- selecting "
                "lightest encoders without profiling."
            )
            return self._cpu_fallback(need_image, need_text, _tab_spec)

        if device is None:
            device = torch.device("cuda:0")

        # ── Probe available VRAM ──────────────────────────────────────
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        budget_bytes = int(free_bytes * self.ETA)

        logger.info(
            "JITEncoderSelector: VRAM total=%.2f GB  free=%.2f GB  "
            "budget (eta=%.2f)=%.2f GB",
            total_bytes / 1e9, free_bytes / 1e9,
            self.ETA, budget_bytes / 1e9,
        )

        # ── Build candidate lists per modality ────────────────────────
        vision_candidates = VISION_REGISTRY if need_image else []
        text_candidates = TEXT_REGISTRY if need_text else []
        vision_candidates = self._prioritize_encoder_specs(vision_candidates, preferred_image)
        text_candidates = self._prioritize_encoder_specs(text_candidates, preferred_text)

        # ── Exhaustive search over Cartesian product ──────────────────
        # Registry lists are pre-sorted descending by capacity, so the
        # first feasible combination maximizes total capacity.
        best = self._constrained_search(
            vision_candidates=vision_candidates,
            text_candidates=text_candidates,
            budget_bytes=budget_bytes,
            device=device,
        )

        if best is not None:
            best.vram_available_bytes = free_bytes
            best.vram_budget_bytes = budget_bytes
            # Attach tabular selection (independent of VRAM profiling)
            if _tab_spec is not None:
                best.tabular_encoder_name = _tab_spec.name
                best.tabular_encoder_class = _tab_spec.encoder_class
                best.tabular_encoder_output_dim = _tab_spec.output_dim
            return best

        # ── All combinations exceeded budget — absolute fallback ──────
        logger.warning(
            "JITEncoderSelector: no feasible combination found within "
            "VRAM budget (%.2f GB).  Falling back to lightest encoders.",
            budget_bytes / 1e9,
        )
        return self._cpu_fallback(need_image, need_text, _tab_spec)

    # ------------------------------------------------------------------ #
    #  Constrained search
    # ------------------------------------------------------------------ #

    def _constrained_search(
        self,
        vision_candidates: List[EncoderSpec],
        text_candidates: List[EncoderSpec],
        budget_bytes: int,
        device: torch.device,
    ) -> Optional[JITSelectionResult]:
        """
        Enumerate {vision × text} combinations in descending capacity
        order and return the first whose total peak memory fits within
        ``budget_bytes``.

        Each candidate encoder is instantiated, profiled via dry-run,
        then destroyed if not selected — preventing VRAM leaks.
        """
        # Generate all combinations sorted by total capacity (descending)
        combos: List[Tuple[Optional[EncoderSpec], Optional[EncoderSpec], int]] = []

        # Case: both modalities active
        if vision_candidates and text_candidates:
            for v in vision_candidates:
                for t in text_candidates:
                    combos.append((v, t, v.capacity + t.capacity))
        # Case: vision only
        elif vision_candidates:
            for v in vision_candidates:
                combos.append((v, None, v.capacity))
        # Case: text only
        elif text_candidates:
            for t in text_candidates:
                combos.append((None, t, t.capacity))
        else:
            # Tabular only — no encoders to select
            return JITSelectionResult(
                selection_method="jit_profiler",
                rationale={"tabular": "Tabular-only — no encoder selection needed"},
            )

        combos.sort(key=lambda c: c[2], reverse=True)

        # ── Profile each combination until one fits ───────────────────
        for v_spec, t_spec, total_cap in combos:
            combo_name = (
                f"[{v_spec.name if v_spec else '-'} + "
                f"{t_spec.name if t_spec else '-'}]"
            )
            logger.info("  Profiling %s (capacity=%d) ...", combo_name, total_cap)

            total_peak = 0
            img_enc = None
            txt_enc = None

            try:
                # Profile vision encoder
                if v_spec is not None:
                    img_enc = v_spec.factory()
                    dummy = v_spec.dummy_input_fn(self._batch_size, device)
                    v_peak = estimate_peak_memory(img_enc, dummy, device)
                    total_peak += v_peak
                    logger.info(
                        "    %s peak: %.2f MB",
                        v_spec.name, v_peak / 1e6,
                    )

                # Profile text encoder
                if t_spec is not None:
                    txt_enc = t_spec.factory()
                    dummy = t_spec.dummy_input_fn(self._batch_size, device)
                    t_peak = estimate_peak_memory(txt_enc, dummy, device)
                    total_peak += t_peak
                    logger.info(
                        "    %s peak: %.2f MB",
                        t_spec.name, t_peak / 1e6,
                    )

                # ── Check constraint: F_peak ≤ η · V_avail ───────────
                if total_peak <= budget_bytes:
                    logger.info(
                        "  >> Selected %s -- peak %.2f MB <= budget %.2f MB",
                        combo_name, total_peak / 1e6, budget_bytes / 1e6,
                    )

                    # Move selected encoders to GPU for training
                    if img_enc is not None:
                        img_enc.to(device)
                    if txt_enc is not None:
                        txt_enc.to(device)

                    rationale = {}
                    if v_spec:
                        rationale["image_encoder"] = (
                            f"JIT selected {v_spec.name} "
                            f"(capacity={v_spec.capacity:,}, "
                            f"peak={total_peak / 1e6:.1f}MB)"
                        )
                    if t_spec:
                        rationale["text_encoder"] = (
                            f"JIT selected {t_spec.name} "
                            f"(capacity={t_spec.capacity:,}, "
                            f"peak={total_peak / 1e6:.1f}MB)"
                        )

                    return JITSelectionResult(
                        image_encoder=img_enc,
                        text_encoder=txt_enc,
                        image_encoder_name=v_spec.name if v_spec else None,
                        text_encoder_name=t_spec.name if t_spec else None,
                        total_capacity=total_cap,
                        total_peak_memory_bytes=total_peak,
                        selection_method="jit_profiler",
                        rationale=rationale,
                    )

                else:
                    logger.info(
                        "  xx Rejected %s -- peak %.2f MB > budget %.2f MB",
                        combo_name, total_peak / 1e6, budget_bytes / 1e6,
                    )
                    # Clean up rejected encoders
                    self._cleanup_encoder(img_enc)
                    self._cleanup_encoder(txt_enc)
                    img_enc = None
                    txt_enc = None

            except Exception as exc:
                logger.warning(
                    "  xx Profiling failed for %s: %s", combo_name, exc,
                )
                self._cleanup_encoder(img_enc)
                self._cleanup_encoder(txt_enc)
                img_enc = None
                txt_enc = None
                continue

        return None

    # ------------------------------------------------------------------ #
    #  CPU fallback — lightest encoders, no profiling
    # ------------------------------------------------------------------ #

    def _cpu_fallback(
        self,
        need_image: bool,
        need_text: bool,
        tab_spec: Optional[TabularEncoderSpec] = None,
    ) -> JITSelectionResult:
        """
        Instantiate the lightest encoder per modality without any VRAM
        probing.  Used on CPU-only systems or when the constrained search
        finds no feasible GPU combination.
        """
        img_enc = None
        txt_enc = None
        img_name = None
        txt_name = None
        total_cap = 0
        rationale: Dict[str, str] = {}

        if need_image:
            # Lightest = last in descending-sorted list
            lightest = VISION_REGISTRY[-1]
            img_enc = lightest.factory()
            img_name = lightest.name
            total_cap += lightest.capacity
            rationale["image_encoder"] = (
                f"CPU fallback: {lightest.name} "
                f"(lightest, capacity={lightest.capacity:,})"
            )
            logger.info("  CPU fallback image: %s", lightest.name)

        if need_text:
            lightest = TEXT_REGISTRY[-1]
            txt_enc = lightest.factory()
            txt_name = lightest.name
            total_cap += lightest.capacity
            rationale["text_encoder"] = (
                f"CPU fallback: {lightest.name} "
                f"(lightest, capacity={lightest.capacity:,})"
            )
            logger.info("  CPU fallback text: %s", lightest.name)

        # Tabular: only set if tab_spec was provided (meaning tabular modality
        # was requested). Don't unconditionally fallback to a tabular encoder
        # when the pipeline doesn't need one.
        tab_name = tab_spec.name if tab_spec else None
        tab_class = tab_spec.encoder_class if tab_spec else None
        tab_out_dim = tab_spec.output_dim if tab_spec else 16

        return JITSelectionResult(
            image_encoder=img_enc,
            text_encoder=txt_enc,
            image_encoder_name=img_name,
            text_encoder_name=txt_name,
            tabular_encoder_name=tab_name,
            tabular_encoder_class=tab_class,
            tabular_encoder_output_dim=tab_out_dim,
            total_capacity=total_cap,
            selection_method="cpu_fallback",
            rationale=rationale,
        )

    # ------------------------------------------------------------------ #
    #  Cleanup helper
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cleanup_encoder(encoder: Optional[nn.Module]) -> None:
        """
        Move an encoder to CPU and release all GPU tensors.

        Called after a candidate is rejected by the constrained optimizer
        to prevent VRAM leaks across profiling iterations.
        """
        if encoder is None:
            return
        try:
            encoder.cpu()
        except Exception:
            pass
        del encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
