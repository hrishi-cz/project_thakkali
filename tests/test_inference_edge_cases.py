"""Inference engine edge case tests — missing encoders, invalid image paths, input_dims."""

import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_head_and_weights(input_dims: dict, hidden_dim: int = 32, num_outputs: int = 2):
    """Build a _MultimodalHead and save its state dict + head_architecture JSON."""
    from automl.trainer import _MultimodalHead
    head = _MultimodalHead(
        input_dims=input_dims,
        hidden_dim=hidden_dim,
        num_outputs=num_outputs,
        fusion_strategy="concatenation",
    )
    head.eval()
    return head


# ---------------------------------------------------------------------------
# Test: input_dims.json loading in _load_head()
# ---------------------------------------------------------------------------

class TestInputDimsJson:
    """_load_head() uses input_dims.json when present instead of heuristic."""

    def _make_artifacts(self, tmpdir: Path, input_dims: dict, hidden_dim: int = 32, num_outputs: int = 2):
        """Write a minimal artifacts directory for inference engine loading."""
        from automl.trainer import _MultimodalHead
        import torch

        artifacts = tmpdir / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)

        head = _MultimodalHead(
            input_dims=input_dims,
            hidden_dim=hidden_dim,
            num_outputs=num_outputs,
            fusion_strategy="concatenation",
        )
        head.eval()

        # Wrap in a fake LightningModule state dict (prefix "model.")
        state = {"model." + k: v for k, v in head.state_dict().items()}
        torch.save(state, artifacts / "model_weights.pth")

        # Persist input_dims.json (the Bug 1 fix)
        (artifacts / "input_dims.json").write_text(json.dumps(input_dims))

        # Write schema.json so the engine can read modalities
        schema = {
            "global_modalities": [k.replace("_pooled", "").replace("tabular", "tabular")
                                   for k in input_dims],
            "global_problem_type": "classification_binary",
            "per_dataset": [],
        }
        (artifacts / "schema.json").write_text(json.dumps(schema))

        # Write metadata.json
        meta = {
            "config": {"problem_type": "classification_binary", "modalities": ["tabular"]},
            "head_architecture": {
                "hidden_dim": hidden_dim,
                "total_dim": sum(input_dims.values()),
                "num_outputs": num_outputs,
            },
        }
        (tmpdir / "metadata.json").write_text(json.dumps(meta))

        return tmpdir

    def test_input_dims_json_loaded_correctly(self, tmp_path):
        """When input_dims.json present, _load_head() uses it without heuristic reconstruction."""
        from pipeline.inference_engine import MultimodalInferenceEngine

        input_dims = {"tabular": 16, "text_pooled": 64}
        self._make_artifacts(tmp_path, input_dims, hidden_dim=32, num_outputs=2)

        engine = object.__new__(MultimodalInferenceEngine)
        object.__setattr__(engine, "artifacts_dir", tmp_path / "artifacts")
        object.__setattr__(engine, "metadata", json.loads((tmp_path / "metadata.json").read_text()))
        object.__setattr__(engine, "schema", {"global_modalities": ["tabular", "text"]})
        object.__setattr__(engine, "modalities", ["tabular", "text"])
        object.__setattr__(engine, "problem_type", "classification_binary")

        head, loaded_dims = engine._load_head()
        assert loaded_dims == input_dims, f"Expected {input_dims}, got {loaded_dims}"
        assert isinstance(head, torch.nn.Module)

    def test_input_dims_json_with_768_image(self, tmp_path):
        """Non-standard image dim (768 from DINOv2) loads correctly via persisted file."""
        from pipeline.inference_engine import MultimodalInferenceEngine

        input_dims = {"tabular": 32, "image_pooled": 768}  # DINOv2, not 512
        self._make_artifacts(tmp_path, input_dims, hidden_dim=64, num_outputs=2)

        engine = object.__new__(MultimodalInferenceEngine)
        object.__setattr__(engine, "artifacts_dir", tmp_path / "artifacts")
        object.__setattr__(engine, "metadata", json.loads((tmp_path / "metadata.json").read_text()))
        object.__setattr__(engine, "schema", {"global_modalities": ["tabular", "image"]})
        object.__setattr__(engine, "modalities", ["tabular", "image"])
        object.__setattr__(engine, "problem_type", "classification_binary")

        head, loaded_dims = engine._load_head()
        assert loaded_dims["image_pooled"] == 768
        assert loaded_dims["tabular"] == 32


# ---------------------------------------------------------------------------
# Test: missing encoder → WARNING logged (Bug 10)
# ---------------------------------------------------------------------------

class TestMissingEncoderWarning:
    """When head expects text_pooled but no text encoder → logger.warning emitted."""

    def _minimal_engine(self, tmp_path: Path, input_dims: dict):
        from pipeline.inference_engine import MultimodalInferenceEngine
        engine = object.__new__(MultimodalInferenceEngine)
        object.__setattr__(engine, "input_dims", input_dims)
        object.__setattr__(engine, "_text_encoder", None)
        object.__setattr__(engine, "_image_encoder", None)
        object.__setattr__(engine, "_tabular_encoder", None)
        object.__setattr__(engine, "_tabular_tokenizer", None)
        object.__setattr__(engine, "_use_token_sequences", False)
        object.__setattr__(engine, "tabular_prep", None)
        object.__setattr__(engine, "tokenizer", None)
        object.__setattr__(engine, "_image_preprocessor", None)
        object.__setattr__(engine, "schema", {"per_dataset": []})
        object.__setattr__(engine, "device", torch.device("cpu"))
        object.__setattr__(engine, "TEXT_DIM", 768)
        object.__setattr__(engine, "IMAGE_DIM", 512)
        return engine

    def test_missing_text_encoder_logs_warning(self, tmp_path, caplog):
        engine = self._minimal_engine(tmp_path, {"text_pooled": 768, "tabular": 8})
        import pandas as pd
        inputs = [{"tabular_0": 1.0, "tabular_1": 0.0}]
        with caplog.at_level(logging.WARNING, logger="pipeline.inference_engine"):
            engine._build_batch(inputs)
        assert any("text_pooled" in r.message or "TextEncoder" in r.message
                   for r in caplog.records if r.levelno >= logging.WARNING), (
            "Expected WARNING about missing text encoder, got: "
            f"{[r.message for r in caplog.records]}"
        )

    def test_missing_image_encoder_logs_warning(self, tmp_path, caplog):
        engine = self._minimal_engine(tmp_path, {"image_pooled": 512, "tabular": 8})
        inputs = [{"tabular_0": 1.0}]
        with caplog.at_level(logging.WARNING, logger="pipeline.inference_engine"):
            engine._build_batch(inputs)
        assert any("image_pooled" in r.message or "ImageEncoder" in r.message
                   for r in caplog.records if r.levelno >= logging.WARNING), (
            "Expected WARNING about missing image encoder"
        )


# ---------------------------------------------------------------------------
# Test: image path failure rate tracking (Bug 11)
# ---------------------------------------------------------------------------

class TestImagePathFailureRate:
    """High image path failure rate emits logger.warning."""

    def _minimal_engine_with_image_prep(self):
        from pipeline.inference_engine import MultimodalInferenceEngine
        from preprocessing.image_preprocessor import ImagePreprocessor

        engine = object.__new__(MultimodalInferenceEngine)
        _prep = ImagePreprocessor()
        object.__setattr__(engine, "_image_preprocessor", _prep)
        object.__setattr__(engine, "schema", {
            "per_dataset": [{"detected_columns": {"image": ["image_path"]}}]
        })
        return engine

    def test_all_invalid_paths_emits_warning(self, caplog):
        """When all image paths are invalid, a WARNING should be logged."""
        engine = self._minimal_engine_with_image_prep()
        import pandas as pd
        inputs = [
            {"image_path": "/nonexistent/fake1.jpg"},
            {"image_path": "/nonexistent/fake2.jpg"},
            {"image_path": "/nonexistent/fake3.jpg"},
            {"image_path": "/nonexistent/fake4.jpg"},
        ]
        with caplog.at_level(logging.WARNING, logger="pipeline.inference_engine"):
            engine._extract_image_tensors(inputs)
        warn_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("failed" in m.lower() or "path" in m.lower() for m in warn_messages), (
            f"Expected WARNING about image failures, got: {warn_messages}"
        )

    def test_partial_failure_below_threshold_is_info(self, caplog, tmp_path):
        """When <30% paths fail, only INFO is logged (no WARNING)."""
        from PIL import Image as PILImage

        engine = self._minimal_engine_with_image_prep()
        # Create 3 valid images and 0 invalid ones
        valid_paths = []
        for i in range(3):
            p = tmp_path / f"test_{i}.jpg"
            PILImage.new("RGB", (32, 32), color=(i * 80, 100, 200)).save(str(p))
            valid_paths.append(str(p))

        inputs = [{"image_path": p} for p in valid_paths]
        with caplog.at_level(logging.WARNING, logger="pipeline.inference_engine"):
            result = engine._extract_image_tensors(inputs)
        # All valid — no WARNING
        warn_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warn_messages) == 0, f"Unexpected WARNINGs: {warn_messages}"
        assert result is not None
        assert result.shape[0] == 3


# ---------------------------------------------------------------------------
# Test: input_dims.json absent → RuntimeError with clear message
# ---------------------------------------------------------------------------

class TestInputDimsMismatchError:
    """When input_dims.json absent and heuristic fails → RuntimeError (not silent fallback)."""

    def test_mismatch_raises_runtime_error(self, tmp_path):
        """total_dim=1312 (768+512+32) but encoder absent → reconstruction gives 512+768=1280 ≠ 1312."""
        from pipeline.inference_engine import MultimodalInferenceEngine
        from automl.trainer import _MultimodalHead
        import torch

        # Build head with real dims (includes tabular=32 not in heuristic)
        input_dims = {"text_pooled": 768, "image_pooled": 512, "tabular": 32}
        head = _MultimodalHead(input_dims=input_dims, hidden_dim=64, num_outputs=2,
                               fusion_strategy="concatenation")
        head.eval()

        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        state = {"model." + k: v for k, v in head.state_dict().items()}
        torch.save(state, artifacts / "model_weights.pth")
        # Intentionally do NOT write input_dims.json

        schema = {"global_modalities": ["tabular", "text", "image"],
                  "global_problem_type": "classification_binary", "per_dataset": []}
        (artifacts / "schema.json").write_text(json.dumps(schema))
        meta = {"config": {"problem_type": "classification_binary", "modalities": ["tabular", "text", "image"]},
                "head_architecture": {"hidden_dim": 64, "total_dim": 1312, "num_outputs": 2}}
        (tmp_path / "metadata.json").write_text(json.dumps(meta))

        engine = object.__new__(MultimodalInferenceEngine)
        object.__setattr__(engine, "artifacts_dir", artifacts)
        object.__setattr__(engine, "metadata", json.loads((tmp_path / "metadata.json").read_text()))
        object.__setattr__(engine, "schema", schema)
        object.__setattr__(engine, "modalities", ["tabular", "text", "image"])
        object.__setattr__(engine, "problem_type", "classification_binary")
        object.__setattr__(engine, "_tabular_encoder", None)
        object.__setattr__(engine, "tokenizer", None)

        # Provide a wrong tabular prep so reconstruction gives wrong total: 9+768+512=1289 ≠ 1312
        class _WrongPrep:
            _feature_names_in = None
            def get_output_dim(self): return 9  # wrong dim → mismatch

        object.__setattr__(engine, "tabular_prep", _WrongPrep())

        with pytest.raises(RuntimeError, match="input_dims reconstruction mismatch"):
            engine._load_head()
