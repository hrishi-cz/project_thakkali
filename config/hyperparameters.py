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

    # Encoder names
    image_encoder_name: str = "resnet50"
    image_output_dim: int = 2048
    text_model_name: str = "bert-base-uncased"
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
            "choices": ["resnet50"],
            "description": "Image backbone",
        },
        "text_model_name": {
            "type":    "categorical",
            "choices": ["DistilBERT", "BERT-base", "RoBERTa-large"],
            "description": "Text transformer backbone",
        },
        "tabular_encoder_name": {
            "type":    "categorical",
            "choices": ["xgboost", "lightgbm", "grn", "mlp"],
            "description": "Tabular encoder architecture",
        },
        "fusion_strategy": {
            "type":    "categorical",
            "choices": [
                "attention",
                "concatenation",
                "graph",
                "uncertainty",
                "uncertainty_graph",
            ],
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
        "default": "resnet50",
        "options": ["resnet50"],
    },
    "image_output_dim":  {"type": "integer", "default": 2048, "min": 64,  "max": 2048},
    "tabular_output_dim":{"type": "integer", "default": 128,  "min": 32,  "max": 512},
    "text_model_name": {
        "type":    "string",
        "default": "bert-base-uncased",
        "options": ["distilbert-base-uncased", "bert-base-uncased",
                    "roberta-base", "roberta-large"],
    },
    "text_output_dim":   {"type": "integer", "default": 768,  "min": 64,  "max": 1024},
    "fusion_strategy": {
        "type":    "string",
        "default": "attention",
        "options": [
            "attention",
            "concatenation",
            "graph",
            "uncertainty",
            "uncertainty_graph",
        ],
    },
    "learning_rate": {"type": "float",   "default": 1e-3,  "min": 1e-6, "max": 1e-1},
    "batch_size":    {"type": "integer", "default": 32,
                      "options": [4, 8, 16, 32, 64, 128, 256]},
    "num_epochs":    {"type": "integer", "default": 10,    "min": 1,    "max": 100},
    "dropout":       {"type": "float",   "default": 0.2,   "min": 0.0,  "max": 0.9},
    "weight_decay":  {"type": "float",   "default": 1e-5,  "min": 1e-7, "max": 1e-1},
}


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------

PRESETS: Dict[str, Dict[str, Any]] = {
    "small": {
        "image_encoder_name": "mobilenet_v3_small",
        "image_output_dim":   128,
        "tabular_output_dim": 64,
        "text_model_name":    "distilbert-base-uncased",
        "text_output_dim":    128,
        "fusion_strategy":    "concatenation",
        "learning_rate":      5e-4,
        "batch_size":         16,
        "num_epochs":         5,
        "dropout":            0.1,
        "weight_decay":       1e-5,
    },
    "medium": {
        "image_encoder_name": "resnet50",
        "image_output_dim":   2048,
        "tabular_output_dim": 128,
        "text_model_name":    "bert-base-uncased",
        "text_output_dim":    768,
        "fusion_strategy":    "attention",
        "learning_rate":      1e-3,
        "batch_size":         32,
        "num_epochs":         10,
        "dropout":            0.2,
        "weight_decay":       1e-5,
    },
    "large": {
        "image_encoder_name": "resnet50",
        "image_output_dim":   2048,
        "tabular_output_dim": 256,
        "text_model_name":    "roberta-large",
        "text_output_dim":    1024,
        "fusion_strategy":    "attention",
        "learning_rate":      1e-3,
        "batch_size":         64,
        "num_epochs":         15,
        "dropout":            0.3,
        "weight_decay":       1e-4,
    },
    "fast": {
        "image_encoder_name": "mobilenet_v3_small",
        "image_output_dim":   128,
        "tabular_output_dim": 64,
        "text_model_name":    "distilbert-base-uncased",
        "text_output_dim":    128,
        "fusion_strategy":    "concatenation",
        "learning_rate":      1e-2,
        "batch_size":         128,
        "num_epochs":         3,
        "dropout":            0.1,
        "weight_decay":       1e-5,
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


def get_modality_optuna_distributions(
    active_modalities: List[str],
    n_tabular_features: int = 0,
    avg_text_len: float = 0.0,
    problem_type: str = "classification",
) -> Dict[str, Dict[str, Any]]:
    """
    Return Optuna distribution specs adapted to the active modalities and
    dataset characteristics gathered during schema detection.

    Rules (additive — later rules override earlier ones):
    ─────────────────────────────────────────────────────
    Text-only / text-dominant (text in modalities, no image):
      • LR ceiling drops to 5e-4 (transformer fine-tuning needs small LR)
      • Warmup steps min raised to 100

    Image-present:
      • Batch size minimum raised to 16 (vision nets need bigger batches)
      • Dropout ceiling raised to 0.6 (ResNet is over-parameterised)

    Tabular-only (no text, no image):
      • LR range widened to [1e-4, 5e-2] (tree/MLP can handle higher LR)
      • Warmup steps max lowered to 0 (no transformer, warmup irrelevant)

    Large tabular feature space (n_tabular_features > 100):
      • Dropout minimum raised to 0.1 (regularise wide networks)

    Regression:
      • Fusion output dim ceiling raised to 1024 (richer representation)

    Parameters
    ----------
    active_modalities : list of "tabular", "text", "image", "timeseries"
    n_tabular_features : int   — from feature_intelligence / global_schema
    avg_text_len : float       — average token count per text sample
    problem_type : str         — "regression", "classification_*", etc.

    Returns
    -------
    Same dict shape as ``get_optuna_distributions()``.
    """
    base = get_optuna_distributions()

    has_text = "text" in active_modalities
    has_image = "image" in active_modalities
    has_tabular = "tabular" in active_modalities or "timeseries" in active_modalities
    tabular_only = has_tabular and not has_text and not has_image

    # ── Text modality ────────────────────────────────────────────────────────
    if has_text:
        base["learning_rate"] = {**base["learning_rate"], "high": 5e-4}
        base["warmup_steps"] = {**base["warmup_steps"], "low": 100}
        # Long texts benefit from larger batches to stabilise gradient estimates
        if avg_text_len > 128:
            base["warmup_steps"] = {**base["warmup_steps"], "high": 2000}

    # ── Image modality ───────────────────────────────────────────────────────
    if has_image:
        base["dropout"] = {**base["dropout"], "high": 0.6}

    # ── Tabular-only ─────────────────────────────────────────────────────────
    if tabular_only:
        base["learning_rate"] = {
            **base["learning_rate"],
            "low": 1e-4,
            "high": 5e-2,
        }
        base["warmup_steps"] = {**base["warmup_steps"], "low": 0, "high": 0}

    # ── Large feature space ───────────────────────────────────────────────────
    if n_tabular_features > 100:
        base["dropout"] = {**base["dropout"], "low": 0.1}
        base["weight_decay"] = {**base["weight_decay"], "low": 1e-5}

    # ── Regression ────────────────────────────────────────────────────────────
    if "regression" in str(problem_type).lower():
        base["fusion_output_dim"] = {**base["fusion_output_dim"], "high": 1024}

    return base


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
