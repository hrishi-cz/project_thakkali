from __future__ import annotations

import importlib
import os
import tempfile
from pathlib import Path

import pytest


def _patch_registry_constants(registry_root: Path) -> None:
    module_names = [
        "config.paths",
        "api.run_api",
        "pipeline.training_orchestrator",
        "pipeline.inference_engine",

        "registry.model_registry",
    ]

    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        if hasattr(module, "MODEL_REGISTRY_DIR"):
            setattr(module, "MODEL_REGISTRY_DIR", registry_root)


@pytest.fixture(autouse=True, scope="session")
def _isolate_registry() -> None:
    original = os.environ.get("MODEL_REGISTRY_DIR")
    with tempfile.TemporaryDirectory(prefix="apex_registry_") as tmp:
        registry_root = Path(tmp)
        os.environ["MODEL_REGISTRY_DIR"] = str(registry_root)
        _patch_registry_constants(registry_root)
        yield

    if original is not None:
        os.environ["MODEL_REGISTRY_DIR"] = original
    else:
        os.environ.pop("MODEL_REGISTRY_DIR", None)