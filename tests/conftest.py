from __future__ import annotations

import importlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


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
    tmp_parent = WORKSPACE_ROOT / ".tmp"
    tmp_parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="apex_registry_", dir=str(tmp_parent))
    registry_root = Path(tmp)
    os.environ["MODEL_REGISTRY_DIR"] = str(registry_root)
    _patch_registry_constants(registry_root)
    yield

    if original is not None:
        os.environ["MODEL_REGISTRY_DIR"] = original
    else:
        os.environ.pop("MODEL_REGISTRY_DIR", None)
    shutil.rmtree(registry_root, ignore_errors=True)
