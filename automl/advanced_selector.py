"""
Advanced Model Selector – Optuna HPO search spaces + PDF heuristic tables.

Epoch bounds (PDF matrix)
-------------------------
Dataset size   | Image    | Text     | Tabular
<5K            | 45-50    | 40-45    | 30-35
5K-50K         | 18-25    | 15-20    | 12-15
50K-500K       | 12-18    | 10-15    | 8-12
>500K          | 10-15    | 8-12     | 6-10

Batch size rules (PDF)
----------------------
Image   : GPU <4 GB → 4 | <8 GB → 8 | <12 GB → 16 | ≥12 GB → 32
Text    : avg_tokens > 512 → 8 | ≤512 → GPU-scaled (same tiers as image, ×2)
Tabular : min(256, dataset_size // 100)

Returns Optuna-compatible search space dicts for every tuneable parameter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optuna space type aliases
# ---------------------------------------------------------------------------

# int  : {"type": "int",         "low": lo,  "high": hi}
# float: {"type": "float",       "low": lo,  "high": hi, "log": bool}
# cat  : {"type": "categorical", "choices": [...]}
OptunaDist = Dict[str, Any]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SelectionResult:
    """
    Model selection outcome combining static encoder choices and Optuna HPO
    search spaces that Phase 5 will consume via ``optuna.trial.suggest_*``.

    Attributes
    ----------
    image_encoder   : tier key into IMAGE_ENCODERS (``None`` when not needed).
    text_encoder    : tier key into TEXT_ENCODERS.
    tabular_encoder : tier key into TABULAR_ENCODERS.
    fusion_strategy : ``"attention"`` | ``"concatenation"`` (static choice).
    batch_size      : Fixed value derived from PDF heuristics (not tuned).
    max_epochs      : Fixed epoch ceiling from PDF matrix (early stopping
                      handles actual duration — epochs are NOT an HPO param).
    hpo_space       : Per-parameter Optuna search space specs.  Keys:
                        ``learning_rate``, ``dropout``,
                        ``weight_decay``, and optionally ``fusion_strategy``
                        when more than one modality is active.
    rationale       : Human-readable selection rationale per component.
    hardware_info   : GPU/CPU environment snapshot.
    """

    image_encoder: Optional[str]
    text_encoder: Optional[str]
    tabular_encoder: Optional[str]
    fusion_strategy: str
    batch_size: int
    max_epochs: int = 20
    hpo_space: Dict[str, OptunaDist] = field(default_factory=dict)
    rationale: Dict[str, str] = field(default_factory=dict)
    hardware_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Encoder catalogue (static metadata only)
# ---------------------------------------------------------------------------

IMAGE_ENCODERS: Dict[str, Dict[str, Any]] = {
    "lightweight": {"name": "MobileNetV3-Small", "output_dim": 512,  "params": "2.5M"},
    "efficient":   {"name": "EfficientNet-B0",   "output_dim": 512,  "params": "5.3M"},
    "balanced":    {"name": "ResNet-50",          "output_dim": 512,  "params": "25M"},
    "sota":        {"name": "ConvNeXt-Tiny",      "output_dim": 512,  "params": "28.6M"},
    "premium":     {"name": "ViT-B-16",           "output_dim": 512,  "params": "86M"},
}

TEXT_ENCODERS: Dict[str, Dict[str, Any]] = {
    "ultralight": {"name": "ALBERT-base-v2",     "output_dim": 768,  "params": "11.7M"},
    "fast":       {"name": "MiniLM-L6-v2",       "output_dim": 768,  "params": "22.7M"},
    "efficient":  {"name": "DistilBERT",          "output_dim": 768,  "params": "66M"},
    "balanced":   {"name": "BERT-base-uncased",   "output_dim": 768,  "params": "110M"},
    "sota":       {"name": "DeBERTa-v3-base",     "output_dim": 768,  "params": "183.8M"},
}

TABULAR_ENCODERS: Dict[str, Dict[str, Any]] = {
    "simple":       {"name": "MLP",  "output_dim": 16},
    "interpretable":{"name": "GRN",  "output_dim": 16},
    "sota":         {"name": "GRN",  "output_dim": 16},
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AdvancedModelSelector:
    """
    Stateless model selector that maps hardware + data context to encoder
    choices and Optuna HPO search bounds.

    Public API
    ----------
    select_models(problem_type, modalities, dataset_size, avg_tokens, gpu_memory_gb)
        → SelectionResult  (primary method, consumed by orchestrator Phase 4)

    recommend_models(problem_type, modalities, dataset_size, avg_tokens)
        → List[Dict]       (legacy/API shim, consumed by /select-model endpoint)
    """

    # ------------------------------------------------------------------
    # Primary selection method
    # ------------------------------------------------------------------

    def select_models(
        self,
        problem_type: str,
        modalities: List[str],
        dataset_size: int,
        avg_tokens: int = 128,
        gpu_memory_gb: Optional[float] = None,
    ) -> SelectionResult:
        """
        Select encoder tier, fixed batch size, and Optuna HPO search bounds.

        Parameters
        ----------
        problem_type  : e.g. ``"classification_binary"``.
        modalities    : Subset of ``["image", "text", "tabular"]``.
        dataset_size  : Total number of training samples.
        avg_tokens    : Mean number of tokens per text sample (default 128).
                        Used exclusively for the text batch-size rule.
        gpu_memory_gb : GPU RAM in GB.  Auto-detected when ``None``.

        Returns
        -------
        SelectionResult
        """
        gpu_mem: float = (
            gpu_memory_gb if gpu_memory_gb is not None
            else self._probe_gpu_memory()
        )
        hardware_info = self._build_hardware_info(gpu_mem)

        # ── Segmentation: fixed ResNet-50 + U-Net decoder ─────────────
        if problem_type == "segmentation":
            logger.info(
                "AdvancedModelSelector: segmentation → ResNet-50 + UNet decoder"
            )
            return SelectionResult(
                image_encoder="balanced",  # ResNet-50
                text_encoder=None,
                tabular_encoder=None,
                fusion_strategy="none",
                batch_size=self._pdf_batch_size(["image"], gpu_mem, dataset_size, avg_tokens),
                max_epochs=self._pdf_epoch_bounds(dataset_size, ["image"])[1],
                hpo_space=self._build_hpo_space(
                    modalities=["image"],
                    dataset_size=dataset_size,
                    problem_type=problem_type,
                    gpu_mem=gpu_mem,
                    fusion_static="none",
                ),
                rationale={"image_encoder": "ResNet-50 backbone for U-Net segmentation"},
                hardware_info=hardware_info,
            )

        # ── encoder tier selection ──────────────────────────────────────
        image_tier, img_rationale = (
            self._select_image_tier(dataset_size, gpu_mem)
            if "image" in modalities
            else (None, "")
        )
        text_tier, txt_rationale = (
            self._select_text_tier(problem_type, dataset_size, gpu_mem)
            if "text" in modalities
            else (None, "")
        )
        tabular_tier, tab_rationale = (
            self._select_tabular_tier(dataset_size, gpu_mem)
            if "tabular" in modalities
            else (None, "")
        )

        # ── batch size (PDF – not tuned) ────────────────────────────────
        batch_size = self._pdf_batch_size(modalities, gpu_mem, dataset_size, avg_tokens)

        # ── max epochs (PDF ceiling – early stopping handles actual duration)
        _epoch_lo, _epoch_hi = self._pdf_epoch_bounds(dataset_size, modalities)
        max_epochs = _epoch_hi

        # ── fusion strategy (modality-aware, also offered as HPO when multimodal)
        # image+text benefits from attention cross-modal weighting;
        # tabular combinations work well with simple concatenation.
        has_image = "image" in modalities
        has_text  = "text"  in modalities
        if len(modalities) > 1 and has_image and has_text:
            fusion = "attention" if gpu_mem >= 8 else "concatenation"
        elif len(modalities) > 1:
            fusion = "concatenation"
        else:
            fusion = "concatenation"

        # ── Optuna HPO search spaces ────────────────────────────────────
        hpo_space = self._build_hpo_space(
            modalities=modalities,
            dataset_size=dataset_size,
            problem_type=problem_type,
            gpu_mem=gpu_mem,
            fusion_static=fusion,
        )

        rationale: Dict[str, str] = {}
        if img_rationale:
            rationale["image_encoder"] = img_rationale
        if txt_rationale:
            rationale["text_encoder"] = txt_rationale
        if tab_rationale:
            rationale["tabular_encoder"] = tab_rationale
        rationale["batch_size"] = (
            f"dataset_size={dataset_size}, gpu_mem={gpu_mem:.1f}GB, "
            f"avg_tokens={avg_tokens}"
        )

        logger.info(
            "AdvancedModelSelector: image=%s  text=%s  tabular=%s  "
            "batch=%d  gpu=%.1f GB",
            image_tier, text_tier, tabular_tier, batch_size, gpu_mem,
        )
        return SelectionResult(
            image_encoder=image_tier,
            text_encoder=text_tier,
            tabular_encoder=tabular_tier,
            fusion_strategy=fusion,
            batch_size=batch_size,
            max_epochs=max_epochs,
            hpo_space=hpo_space,
            rationale=rationale,
            hardware_info=hardware_info,
        )

    # ------------------------------------------------------------------
    # API / frontend shim
    # ------------------------------------------------------------------

    def recommend_models(
        self,
        problem_type: str,
        modalities: List[str],
        dataset_size: int = 10_000,
        avg_tokens: int = 128,
    ) -> List[Dict[str, Any]]:
        """
        Return a ranked list of model recommendation dicts suitable for the
        Streamlit frontend JSON contract.

        The first entry is always the primary (highest-quality) recommendation
        derived from ``select_models()``.  The remaining entries enumerate
        cheaper alternatives for each tier.
        """
        primary: SelectionResult = self.select_models(
            problem_type=problem_type,
            modalities=modalities,
            dataset_size=dataset_size,
            avg_tokens=avg_tokens,
        )

        def _encoder_name(catalogue: Dict, tier: Optional[str]) -> Optional[str]:
            return catalogue[tier]["name"] if tier and tier in catalogue else None

        primary_rec: Dict[str, Any] = {
            "name": self._build_model_name(
                primary.image_encoder,
                primary.text_encoder,
                primary.tabular_encoder,
            ),
            "image_encoder":   _encoder_name(IMAGE_ENCODERS, primary.image_encoder),
            "text_encoder":    _encoder_name(TEXT_ENCODERS,  primary.text_encoder),
            "tabular_encoder": _encoder_name(TABULAR_ENCODERS, primary.tabular_encoder),
            "fusion_strategy": primary.fusion_strategy,
            "batch_size":      primary.batch_size,
            "hpo_space":       primary.hpo_space,
            "rationale":       primary.rationale,
            "hardware_info":   primary.hardware_info,
            "tier":            "primary",
        }

        # Lightweight fallback alternative
        alt_hpo = dict(primary.hpo_space)  # same search bounds
        alt_rec: Dict[str, Any] = {
            "name": "Lightweight Fallback",
            "image_encoder":   IMAGE_ENCODERS["lightweight"]["name"] if "image" in modalities else None,
            "text_encoder":    TEXT_ENCODERS["fast"]["name"]         if "text"  in modalities else None,
            "tabular_encoder": TABULAR_ENCODERS["simple"]["name"]    if "tabular" in modalities else None,
            "fusion_strategy": "concatenation",
            "batch_size":      min(primary.batch_size, 8),
            "hpo_space":       alt_hpo,
            "rationale":       {"general": "Lightweight fallback for memory-constrained environments"},
            "hardware_info":   primary.hardware_info,
            "tier":            "fallback",
        }

        return [primary_rec, alt_rec]

    # ------------------------------------------------------------------
    # PDF epoch bounds
    # ------------------------------------------------------------------

    @staticmethod
    def _pdf_epoch_bounds(
        dataset_size: int,
        modalities: List[str],
    ) -> Tuple[int, int]:
        """
        Return ``(low, high)`` epoch bounds from the PDF matrix.

        Priority: image > text > tabular when multiple modalities are active.
        """
        has_image   = "image"   in modalities
        has_text    = "text"    in modalities
        # has_tabular = "tabular" in modalities  # lowest priority

        if dataset_size < 5_000:
            if has_image:   return 45, 50
            if has_text:    return 40, 45
            return 30, 35
        elif dataset_size < 50_000:
            if has_image:   return 18, 25
            if has_text:    return 15, 20
            return 12, 15
        elif dataset_size < 500_000:
            if has_image:   return 12, 18
            if has_text:    return 10, 15
            return 8, 12
        else:
            if has_image:   return 10, 15
            if has_text:    return 8,  12
            return 6, 10

    # ------------------------------------------------------------------
    # PDF batch-size rules
    # ------------------------------------------------------------------

    @staticmethod
    def _pdf_batch_size(
        modalities: List[str],
        gpu_mem: float,
        dataset_size: int,
        avg_tokens: int,
    ) -> int:
        """
        Compute the fixed batch size from PDF heuristic rules.

        Image  : <4 GB→4 | <8 GB→8 | <12 GB→16 | ≥12 GB→32
        Text   : avg_tokens>512→8 | ≤512 scaled by GPU tier (×2 vs image)
        Tabular: min(256, dataset_size // 100)
        """
        if "image" in modalities:
            if gpu_mem < 4:   return 4
            if gpu_mem < 8:   return 8
            if gpu_mem < 12:  return 16
            return 32

        if "text" in modalities:
            if avg_tokens > 512:
                return 8
            # ≤512: same GPU tiers but doubled (as per PDF)
            if gpu_mem < 4:   return 8
            if gpu_mem < 8:   return 16
            if gpu_mem < 12:  return 32
            return 64

        # Pure tabular
        return min(256, max(1, dataset_size // 100))

    # ------------------------------------------------------------------
    # Encoder tier selection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _select_image_tier(dataset_size: int, gpu_mem: float) -> Tuple[str, str]:
        if dataset_size < 1_000:
            return "lightweight", "Dataset <1k: MobileNetV3 for efficiency"
        if dataset_size < 5_000:
            if gpu_mem >= 8:
                return "efficient", f"Dataset 1-5k + GPU {gpu_mem:.1f}GB: EfficientNet-B0"
            return "lightweight", f"Dataset 1-5k but GPU {gpu_mem:.1f}GB: MobileNetV3"
        if dataset_size < 10_000:
            if gpu_mem >= 8:
                return "balanced", f"Dataset 5-10k + GPU {gpu_mem:.1f}GB: ResNet50"
            return "efficient", f"Dataset 5-10k + GPU {gpu_mem:.1f}GB: EfficientNet-B0"
        # dataset_size >= 10_000
        if gpu_mem >= 16:
            return "premium", f"Dataset >10k + GPU {gpu_mem:.1f}GB: ViT-B-16"
        if gpu_mem >= 12:
            return "sota",    f"Dataset >10k + GPU {gpu_mem:.1f}GB: ConvNeXt-Tiny"
        if gpu_mem >= 8:
            return "balanced", f"Dataset >10k + GPU {gpu_mem:.1f}GB: ResNet50"
        return "lightweight", f"Dataset >10k but GPU {gpu_mem:.1f}GB: MobileNetV3"

    @staticmethod
    def _select_text_tier(
        problem_type: str,
        dataset_size: int,
        gpu_mem: float,
    ) -> Tuple[str, str]:
        if dataset_size < 1_000:
            return "ultralight", "Tiny dataset: ALBERT-base-v2 for minimal overhead"
        if "binary" in problem_type or dataset_size < 5_000:
            return "fast", "Binary/small dataset: MiniLM-L6-v2 for speed"
        if dataset_size < 10_000:
            if gpu_mem >= 8:
                return "efficient", f"Mid dataset + GPU {gpu_mem:.1f}GB: DistilBERT"
            return "fast", f"Mid dataset + GPU {gpu_mem:.1f}GB: MiniLM"
        # dataset_size >= 10_000
        if "multiclass" in problem_type and gpu_mem >= 12 and dataset_size > 50_000:
            return "sota", f"Multiclass + GPU {gpu_mem:.1f}GB + large dataset: DeBERTa-v3"
        if gpu_mem >= 8:
            return "balanced", "Default: BERT-base"
        return "fast", f"GPU {gpu_mem:.1f}GB: MiniLM"

    @staticmethod
    def _select_tabular_tier(dataset_size: int, gpu_mem: float) -> Tuple[str, str]:
        if dataset_size < 5_000 and gpu_mem < 8:
            return "simple", "Small dataset + limited GPU: MLP"
        if gpu_mem >= 12:
            return "sota",    f"GPU {gpu_mem:.1f}GB: GRN (highest capacity)"
        if gpu_mem >= 8:
            return "interpretable", f"GPU {gpu_mem:.1f}GB: GRN"
        return "simple", f"GPU {gpu_mem:.1f}GB: MLP"

    # ------------------------------------------------------------------
    # HPO space builder
    # ------------------------------------------------------------------

    def _build_hpo_space(
        self,
        modalities: List[str],
        dataset_size: int,
        problem_type: str,
        gpu_mem: float,
        fusion_static: str,
    ) -> Dict[str, OptunaDist]:
        """
        Build a fully-specified Optuna search space based on data context.

        Each entry is serialisable to JSON and can be consumed by Phase 5 as::

            value = trial.suggest_float(name, low, high, log=log)  # type == "float"
            value = trial.suggest_categorical(name, choices)  # type == "categorical"

        Epochs are NOT part of the search space — ``max_epochs`` is set as a
        fixed ceiling on ``SelectionResult`` and early stopping controls
        actual training duration.
        """
        # Learning rate: tight range suitable for transformer/embedding models
        lr_low  = 1e-5
        lr_high = 5e-4

        space: Dict[str, OptunaDist] = {
            "learning_rate": {
                "type": "float",
                "low":  lr_low,
                "high": lr_high,
                "log":  True,
            },
            "dropout": {
                "type": "float",
                "low":  0.1,
                "high": 0.4,
            },
            "weight_decay": {
                "type": "float",
                "low":  1e-6,
                "high": 1e-3,
                "log":  True,
            },
        }

        # Fusion strategy is also tunable when multiple modalities exist
        if len(modalities) > 1:
            choices = (
                ["attention", "concatenation"]
                if gpu_mem >= 8
                else ["concatenation"]
            )
            space["fusion_strategy"] = {
                "type":    "categorical",
                "choices": choices,
            }

        return space

    # ------------------------------------------------------------------
    # Hardware probing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _probe_gpu_memory() -> float:
        """Return total GPU memory in GB; 0.0 on CPU-only systems."""
        if not torch.cuda.is_available():
            return 0.0
        try:
            return round(
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2
            )
        except Exception:
            return 0.0

    @staticmethod
    def _build_hardware_info(gpu_mem: float) -> Dict[str, Any]:
        return {
            "gpu_available":  torch.cuda.is_available(),
            "gpu_memory_gb":  gpu_mem,
            "device":         "GPU" if torch.cuda.is_available() else "CPU",
            "cuda_device":    (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else "None"
            ),
        }

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_model_name(
        image_tier: Optional[str],
        text_tier:  Optional[str],
        tabular_tier: Optional[str],
    ) -> str:
        parts: List[str] = []
        if image_tier   and image_tier   in IMAGE_ENCODERS:
            parts.append(IMAGE_ENCODERS[image_tier]["name"])
        if text_tier    and text_tier    in TEXT_ENCODERS:
            parts.append(TEXT_ENCODERS[text_tier]["name"])
        if tabular_tier and tabular_tier in TABULAR_ENCODERS:
            parts.append(TABULAR_ENCODERS[tabular_tier]["name"])
        return " + ".join(parts) if parts else "Unsupervised"
