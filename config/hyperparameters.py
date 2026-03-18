"""
Hyperparameter configuration for models.

``HyperparameterConfig`` stores concrete default values for a single run.
``get_optuna_distributions()`` returns Optuna-compatible search space specs
that Phase 5 uses to call ``trial.suggest_*`` during HPO.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Concrete configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class HyperparameterConfig:
    """
    Default hyperparameter values for a single training run.

    These are the *baseline* settings used when no HPO trial is active.
    During Phase 5 HPO, ``get_optuna_distributions()`` provides the search
    bounds that Optuna uses to propose improved values.
    """

    # Encoder names (canonical JIT registry identifiers)
    image_encoder_name: str = "ResNet-50"
    image_output_dim: int = 2048
    text_model_name: str = "BERT-base-uncased"
    text_output_dim: int = 768
    tabular_hidden_dims: Optional[List[int]] = None
    tabular_output_dim: int = 128

    # Fusion
    fusion_strategy: str = "attention"
    fusion_output_dim: int = 256

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 10
    dropout: float = 0.2
    weight_decay: float = 1e-5

    # Segmentation-specific
    seg_decoder: str = "unet"
    seg_num_classes: int = 2
    seg_input_size: int = 256

    def __post_init__(self) -> None:
        if self.tabular_hidden_dims is None:
            self.tabular_hidden_dims = [256, 128]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HyperparameterConfig":
        """Create config from dict, ignoring unknown keys."""
        valid_keys = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: str) -> "HyperparameterConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_yaml(cls, path: str) -> "HyperparameterConfig":
        if yaml is None:
            raise ImportError("pyyaml is required for YAML config support.")
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_yaml(self, path: str) -> None:
        if yaml is None:
            raise ImportError("pyyaml is required for YAML config support.")
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f)

    def merge(self, override: Dict[str, Any]) -> None:
        """Merge override dict into config (in-place)."""
        for k, v in override.items():
            if hasattr(self, k):
                setattr(self, k, v)


# ---------------------------------------------------------------------------
# Optuna search space definitions
# ---------------------------------------------------------------------------

def get_optuna_distributions() -> Dict[str, Dict[str, Any]]:
    """
    Return a mapping of hyperparameter name → Optuna distribution spec.

    Each spec is a plain dict serialisable to JSON.  Phase 5 consumes it as::

        for name, dist in get_optuna_distributions().items():
            if dist["type"] == "float":
                value = trial.suggest_float(
                    name, dist["low"], dist["high"],
                    log=dist.get("log", False)
                )
            elif dist["type"] == "int":
                value = trial.suggest_int(name, dist["low"], dist["high"])
            elif dist["type"] == "categorical":
                value = trial.suggest_categorical(name, dist["choices"])

    Epoch bounds are intentionally wide here; ``AdvancedModelSelector`` narrows
    them based on dataset size and modality at Phase 4 runtime.
    """
    return {
        # ── Training dynamics ──────────────────────────────────────────
        "learning_rate": {
            "type": "float",
            "low":  1e-5,
            "high": 1e-2,
            "log":  True,
            "description": "AdamW / Adam LR – log-uniform over four decades",
        },
        "weight_decay": {
            "type": "float",
            "low":  1e-6,
            "high": 1e-2,
            "log":  True,
            "description": "L2 weight decay – log-uniform",
        },
        "dropout": {
            "type": "float",
            "low":  0.0,
            "high": 0.5,
            "description": "Dropout probability applied after each encoder block",
        },
        "num_epochs": {
            "type": "int",
            "low":  6,
            "high": 50,
            "description": (
                "Number of training epochs.  AdvancedModelSelector narrows "
                "this range at phase-4 time based on the PDF epoch matrix."
            ),
        },

        # ── Architecture ──────────────────────────────────────────────
        "image_encoder_name": {
            "type":    "categorical",
            "choices": ["MobileNetV3-Small", "EfficientNet-B0", "ResNet-50",
                        "ConvNeXt-Tiny", "ViT-B-16"],
            "description": "Image backbone (matched to JIT encoder registry)",
        },
        "text_model_name": {
            "type":    "categorical",
            "choices": ["ALBERT-base-v2", "MiniLM-L6-v2", "DistilBERT",
                        "BERT-base-uncased", "DeBERTa-v3-base"],
            "description": "Text transformer backbone (matched to JIT encoder registry)",
        },
        "tabular_encoder_name": {
            "type":    "categorical",
            "choices": ["MLP", "GRN"],
            "description": "Tabular encoder architecture (matched to JIT encoder registry)",
        },
        "fusion_strategy": {
            "type":    "categorical",
            "choices": ["attention", "concatenation"],
            "description": "Multimodal fusion head strategy",
        },
        "fusion_output_dim": {
            "type": "int",
            "low":  64,
            "high": 512,
            "description": "Dimension of the shared fusion head output",
        },

        # ── Optimiser extras ─────────────────────────────────────────
        "warmup_steps": {
            "type": "int",
            "low":  0,
            "high": 1000,
            "description": "Linear LR warm-up steps (0 = disabled)",
        },
    }


# ---------------------------------------------------------------------------
# Static schema (kept for /config endpoint backward-compat)
# ---------------------------------------------------------------------------

HYPERPARAMETERS: Dict[str, Dict[str, Any]] = {
    "image_encoder_name": {
        "type":    "string",
        "default": "ResNet-50",
        "options": ["MobileNetV3-Small", "EfficientNet-B0", "ResNet-50",
                    "ConvNeXt-Tiny", "ViT-B-16"],
    },
    "image_output_dim":  {"type": "integer", "default": 2048, "min": 64,  "max": 2048},
    "tabular_output_dim":{"type": "integer", "default": 128,  "min": 32,  "max": 512},
    "text_model_name": {
        "type":    "string",
        "default": "BERT-base-uncased",
        "options": ["ALBERT-base-v2", "MiniLM-L6-v2", "DistilBERT",
                    "BERT-base-uncased", "DeBERTa-v3-base"],
    },
    "text_output_dim":   {"type": "integer", "default": 768,  "min": 64,  "max": 1024},
    "fusion_strategy": {
        "type":    "string",
        "default": "attention",
        "options": ["concatenation", "attention"],
    },
    "learning_rate": {"type": "float",   "default": 1e-3,  "min": 1e-6, "max": 1e-1},
    "batch_size":    {"type": "integer", "default": 32,
                      "options": [4, 8, 16, 32, 64, 128, 256]},
    "num_epochs":    {"type": "integer", "default": 10,    "min": 1,    "max": 100},
    "dropout":       {"type": "float",   "default": 0.2,   "min": 0.0,  "max": 0.9},
    "weight_decay":  {"type": "float",   "default": 1e-5,  "min": 1e-7, "max": 1e-1},
    # Segmentation
    "seg_decoder":   {"type": "string",  "default": "unet", "options": ["unet"]},
    "seg_num_classes":{"type": "integer", "default": 2,     "min": 2,   "max": 256},
    "seg_input_size":{"type": "integer", "default": 256,    "min": 64,  "max": 1024},
}

# Canonical display name → HuggingFace / torchvision model identifier.
# Used by code that needs to load pretrained weights from a canonical name.
CANONICAL_TO_HF: Dict[str, str] = {
    # Vision
    "MobileNetV3-Small": "mobilenet_v3_small",
    "EfficientNet-B0":   "efficientnet_b0",
    "ResNet-50":         "resnet50",
    "ConvNeXt-Tiny":     "convnext_tiny",
    "ViT-B-16":          "vit_b_16",
    # Text
    "ALBERT-base-v2":    "albert-base-v2",
    "MiniLM-L6-v2":      "sentence-transformers/all-MiniLM-L6-v2",
    "DistilBERT":        "distilbert-base-uncased",
    "BERT-base-uncased":  "bert-base-uncased",
    "DeBERTa-v3-base":   "microsoft/deberta-v3-base",
    # Tabular (no HF counterpart — class-based)
    "MLP":               "MLP",
    "GRN":               "GRN",
}


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------

PRESETS: Dict[str, Dict[str, Any]] = {
    "small": {
        "image_encoder_name": "MobileNetV3-Small",
        "image_output_dim":   128,
        "tabular_output_dim": 64,
        "text_model_name":    "MiniLM-L6-v2",
        "text_output_dim":    128,
        "fusion_strategy":    "concatenation",
        "learning_rate":      5e-4,
        "batch_size":         16,
        "num_epochs":         5,
        "dropout":            0.1,
        "weight_decay":       1e-5,
    },
    "medium": {
        "image_encoder_name": "ResNet-50",
        "image_output_dim":   2048,
        "tabular_output_dim": 128,
        "text_model_name":    "BERT-base-uncased",
        "text_output_dim":    768,
        "fusion_strategy":    "attention",
        "learning_rate":      1e-3,
        "batch_size":         32,
        "num_epochs":         10,
        "dropout":            0.2,
        "weight_decay":       1e-5,
    },
    "large": {
        "image_encoder_name": "ConvNeXt-Tiny",
        "image_output_dim":   2048,
        "tabular_output_dim": 256,
        "text_model_name":    "DeBERTa-v3-base",
        "text_output_dim":    768,
        "fusion_strategy":    "attention",
        "learning_rate":      1e-3,
        "batch_size":         64,
        "num_epochs":         15,
        "dropout":            0.3,
        "weight_decay":       1e-4,
    },
    "fast": {
        "image_encoder_name": "MobileNetV3-Small",
        "image_output_dim":   128,
        "tabular_output_dim": 64,
        "text_model_name":    "MiniLM-L6-v2",
        "text_output_dim":    128,
        "fusion_strategy":    "concatenation",
        "learning_rate":      1e-2,
        "batch_size":         128,
        "num_epochs":         3,
        "dropout":            0.1,
        "weight_decay":       1e-5,
    },
    "premium": {
        "image_encoder_name": "ViT-B-16",
        "image_output_dim":   2048,
        "tabular_output_dim": 256,
        "text_model_name":    "DeBERTa-v3-base",
        "text_output_dim":    768,
        "fusion_strategy":    "attention",
        "learning_rate":      5e-4,
        "batch_size":         16,
        "num_epochs":         20,
        "dropout":            0.3,
        "weight_decay":       1e-4,
    },
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def load_dynamic_config(config_path: Optional[str] = None) -> HyperparameterConfig:
    """Load config from JSON/YAML file if provided, else use defaults."""
    if config_path:
        ext = Path(config_path).suffix.lower()
        if ext in (".yaml", ".yml"):
            return HyperparameterConfig.from_yaml(config_path)
        if ext == ".json":
            return HyperparameterConfig.from_json(config_path)
        raise ValueError(f"Unsupported config file type: {ext}")
    return HyperparameterConfig()


def get_presets() -> Dict[str, Dict[str, Any]]:
    return PRESETS.copy()


def get_preset(preset_name: str) -> Dict[str, Any]:
    presets = get_presets()
    if preset_name not in presets:
        raise ValueError(
            f"Preset '{preset_name}' not found. Available: {list(presets.keys())}"
        )
    return presets[preset_name].copy()


def get_default_config() -> HyperparameterConfig:
    return HyperparameterConfig()


def validate_hyperparameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and clip hyperparameters against the static schema."""
    validated: Dict[str, Any] = {}
    for key, value in params.items():
        if key not in HYPERPARAMETERS:
            validated[key] = value
            continue
        schema = HYPERPARAMETERS[key]
        param_type = schema.get("type", "string")
        try:
            if param_type == "integer":
                validated[key] = int(value)
                if "min" in schema:
                    validated[key] = max(validated[key], schema["min"])
                if "max" in schema:
                    validated[key] = min(validated[key], schema["max"])
            elif param_type == "float":
                validated[key] = float(value)
                if "min" in schema:
                    validated[key] = max(validated[key], schema["min"])
                if "max" in schema:
                    validated[key] = min(validated[key], schema["max"])
            else:
                validated[key] = str(value)
                if "options" in schema and validated[key] not in schema["options"]:
                    validated[key] = schema.get("default", value)
        except Exception:
            validated[key] = schema.get("default", value)
    return validated
