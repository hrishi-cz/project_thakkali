from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from fastapi.testclient import TestClient


def test_phase5_trainer_disables_sanity_validation() -> None:
    import pipeline.training_orchestrator as orchestrator_mod

    source = inspect.getsource(orchestrator_mod.TrainingOrchestrator._execute_phase_5_training)
    assert "num_sanity_val_steps=0" in source


def test_ula_auxiliary_histories_flush_from_training_buffers() -> None:
    from automl.trainer import build_trainer

    module = build_trainer(
        problem_type="classification_binary",
        num_classes=2,
        input_dims={"tabular": 4, "text_pooled": 4},
        fusion_strategy="ula",
        fusion_config={"latent_dim": 8, "n_layers": 1, "n_heads": 2},
        contrastive_weight=0.0,
    )
    module.train()
    module._last_encoded_batch = {
        "tabular": torch.randn(4, 4),
        "text_pooled": torch.randn(4, 4),
    }

    loss = module._apply_adaptive_loss(torch.tensor(0.5, requires_grad=True))
    assert torch.is_tensor(loss)
    assert module._alignment_loss_epoch_values
    assert module._contrastive_loss_epoch_values

    module._flush_aux_loss_histories()
    assert len(module._alignment_loss_history) == 1
    assert len(module._contrastive_loss_history) == 1
    assert not module._alignment_loss_epoch_values
    assert not module._contrastive_loss_epoch_values


def test_lora_encoder_path_keeps_gradients_enabled() -> None:
    from automl.trainer import build_trainer
    from modelss.adapters.lora import lora_parameters

    class _FakeTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.query = nn.Linear(4, 4, bias=False)

        def forward(self, input_ids, attention_mask=None):
            x = torch.nn.functional.one_hot(input_ids.clamp(0, 3), num_classes=4).float()
            return SimpleNamespace(last_hidden_state=self.query(x))

    class _FakeTextEncoder(nn.Module):
        output_dim = 4
        hidden_size = 4
        model_name = "fake-text"
        max_length = 2

        def __init__(self) -> None:
            super().__init__()
            self.transformer = _FakeTransformer()
            self._projection = None

    text_encoder = _FakeTextEncoder()
    module = build_trainer(
        problem_type="classification_binary",
        num_classes=2,
        input_dims={"tabular": 4, "text_pooled": 4},
        text_encoder=text_encoder,
        lora_config={"r": 2, "alpha": 4.0, "lr_mult": 0.1},
    )
    module.train()

    batch = {
        "tabular": torch.randn(3, 4),
        "input_ids": torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long),
        "attention_mask": torch.ones(3, 2, dtype=torch.long),
    }
    logits = module(batch)
    logits.sum().backward()

    grads = [p.grad for p in lora_parameters(text_encoder)]
    assert grads
    assert any(g is not None and float(g.abs().sum()) > 0.0 for g in grads)


def test_lora_trials_can_clear_and_restore_embedding_caches() -> None:
    from pipeline.training_orchestrator import (
        _clear_embedding_caches,
        _restore_embedding_caches,
        _snapshot_embedding_caches,
    )

    dataset = SimpleNamespace(
        _precomputed_text=torch.ones(2, 4),
        _precomputed_image=torch.ones(2, 4),
    )
    snapshot = _snapshot_embedding_caches(dataset)
    _clear_embedding_caches(dataset)
    assert dataset._precomputed_text is None
    assert dataset._precomputed_image is None

    _restore_embedding_caches(snapshot)
    assert torch.equal(dataset._precomputed_text, torch.ones(2, 4))
    assert torch.equal(dataset._precomputed_image, torch.ones(2, 4))


def test_phase7_fusion_payload_canonicalizes_ula() -> None:
    from pipeline.training_orchestrator import _phase7_fusion_payload

    payload = _phase7_fusion_payload(
        {
            "fusion_summary": {"fusion_type": "UnifiedLatentFusion"},
            "fusion_aux_weights": {"graph_sparsity_weight": 0.0},
            "alignment_summary": {"tabular_text": 0.1},
        }
    )
    assert payload["strategy"] == "ula"
    assert payload["summary"]["fusion_type"] == "UnifiedLatentFusion"


def test_inference_head_reconstructs_ula_from_metadata(tmp_path: Path) -> None:
    from automl.trainer import _MultimodalHead
    from pipeline.inference_engine import MultimodalInferenceEngine

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    input_dims = {"tabular": 4, "text_pooled": 4}
    ula_config = {"latent_dim": 8, "n_layers": 1, "n_heads": 2, "token_mode": False}
    head = _MultimodalHead(
        input_dims=input_dims,
        hidden_dim=16,
        num_outputs=1,
        fusion_strategy="ula",
        fusion_config=ula_config,
    )
    torch.save(head.state_dict(), artifacts / "model_weights.pth")
    (artifacts / "input_dims.json").write_text(json.dumps(input_dims), encoding="utf-8")
    (artifacts / "ula_config.json").write_text(json.dumps(ula_config), encoding="utf-8")

    engine = object.__new__(MultimodalInferenceEngine)
    object.__setattr__(engine, "artifacts_dir", artifacts)
    object.__setattr__(
        engine,
        "metadata",
        {
            "fusion": {"strategy": "ula"},
            "head_architecture": {"hidden_dim": 16, "total_dim": 8, "num_outputs": 1},
        },
    )
    object.__setattr__(engine, "schema", {"global_modalities": ["tabular", "text"]})
    object.__setattr__(engine, "modalities", ["tabular", "text"])
    object.__setattr__(engine, "problem_type", "classification_binary")
    object.__setattr__(engine, "tabular_prep", None)
    object.__setattr__(engine, "_tabular_encoder", None)

    loaded, loaded_dims = engine._load_head()
    assert loaded_dims == input_dims
    assert getattr(loaded, "fusion_strategy") == "ula"
    assert type(getattr(loaded, "fusion")).__name__ == "UnifiedLatentFusion"


class _FakePredictionEngine:
    probability_calibrator = None

    def predict_batch(self, inputs, execution_context=None):
        return {
            "predictions": [1 for _ in range(len(inputs))],
            "confidences": [0.91 for _ in range(len(inputs))],
            "problem_type": "classification_binary",
            "n_samples": len(inputs),
        }

    def generate_explanations(self, inputs, target_class=0, n_steps=50):
        return {"tabular": {"feature_names": ["feat"], "attributions": [1.0]}}


def _prepare_prediction_case(run_api):
    client = TestClient(run_api.app)
    session_resp = client.post("/v2/sessions", json={"project_name": "ula-predict"})
    assert session_resp.status_code == 200
    session_id = session_resp.json()["session_id"]
    model_id = "ula_prediction_test_model"

    model_root = run_api.MODEL_REGISTRY_DIR / model_id
    artifacts = model_root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    schema = {
        "global_modalities": ["tabular"],
        "active_modalities": ["tabular"],
        "global_problem_type": "classification_binary",
        "primary_target": "label",
        "per_dataset": [
            {"detected_columns": {"tabular": ["feat"]}, "id_like_columns": []}
        ],
    }
    metadata = {
        "model_id": model_id,
        "deployment_ready": True,
        "config": {"problem_type": "classification_binary", "modalities": ["tabular"]},
        "training_signals": {
            "active_modalities": ["tabular"],
            "problem_type": "classification_binary",
        },
        "fusion": {"strategy": "ula"},
    }
    (artifacts / "schema.json").write_text(json.dumps(schema), encoding="utf-8")
    (model_root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    ctx = run_api.session_manager.get_session(session_id)
    assert ctx is not None
    ctx.global_schema = dict(schema)
    ctx.active_modalities = ["tabular"]
    ctx.registered_model_ids = [model_id]
    ctx.active_prediction_model_id = model_id
    ctx.set_pipeline_stage("training")
    run_api.session_manager.update_session_context(session_id, ctx)

    with run_api._engine_cache_lock:
        run_api._engine_cache.clear()
        run_api._engine_cache[model_id] = _FakePredictionEngine()

    return client, session_id, model_id


def test_predict_http_surfaces_return_canonical_ula() -> None:
    import api.run_api as run_api

    client, session_id, model_id = _prepare_prediction_case(run_api)
    payload = {"session_id": session_id, "model_id": model_id, "inputs": [{"feat": 1.0}]}

    sync_resp = client.post("/predict", json=payload)
    assert sync_resp.status_code == 200
    sync_body = sync_resp.json()
    assert sync_body["fusion_strategy"] == "ula"
    assert sync_body["input_contract"]["fusion"]["strategy"] == "ula"
    assert sync_body["active_modalities"] == ["tabular"]

    async_resp = client.post("/predict-async", json=payload)
    assert async_resp.status_code == 200
    task_id = async_resp.json()["task_id"]
    task_resp = client.get(f"/task/{task_id}")
    assert task_resp.status_code == 200
    task_body = task_resp.json()
    assert task_body["status"] == "COMPLETED"
    assert task_body["result"]["fusion_strategy"] == "ula"
    assert task_body["result"]["input_contract"]["fusion"]["strategy"] == "ula"


def test_predict_websocket_returns_canonical_ula() -> None:
    import api.run_api as run_api

    client, session_id, model_id = _prepare_prediction_case(run_api)
    with client.websocket_connect("/ws/predict") as websocket:
        assert websocket.receive_json()["status"] == "CONNECTED"
        websocket.send_json(
            {
                "session_id": session_id,
                "model_id": model_id,
                "inputs": [{"feat": 1.0}],
            }
        )
        result = None
        for _ in range(10):
            message = websocket.receive_json()
            if message.get("type") == "complete":
                result = message["result"]
                break
        assert result is not None
        assert result["fusion_strategy"] == "ula"
        assert result["input_contract"]["fusion"]["strategy"] == "ula"
        assert result["active_modalities"] == ["tabular"]
