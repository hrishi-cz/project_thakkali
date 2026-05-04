"""AutoML package for automatic model selection and training."""

from .model_selector import ModelSelector

# Lazy imports – trainer requires pytorch_lightning which may not be
# installed yet during lightweight operations like model selection.


def __getattr__(name: str):
    if name in ("ApexLightningModule", "build_trainer"):
        from .trainer import ApexLightningModule, build_trainer
        return {"ApexLightningModule": ApexLightningModule, "build_trainer": build_trainer}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ModelSelector", "ApexLightningModule", "build_trainer"]
