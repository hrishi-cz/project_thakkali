from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable, List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


workspace_root = Path(__file__).parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))


def _run_test(name: str, fn: Callable[[], None]) -> Tuple[str, bool, str]:
    try:
        fn()
        print(f"PASS - {name}")
        return (name, True, "")
    except Exception as exc:
        print(f"FAIL - {name}: {exc}")
        return (name, False, str(exc))


def test_meta_learning_store_persistence() -> None:
    from automl.meta_learning import MetaLearningStore

    storage = Path("logs") / "system_validation_meta.json"
    if storage.exists():
        storage.unlink()

    store = MetaLearningStore(storage_path=str(storage))
    record = {
        "dataset_meta": {
            "num_rows": 1200,
            "num_cols": 24,
            "modalities": ["tabular", "text"],
            "target_type": "classification",
        },
        "best_params": {"learning_rate": 1e-3},
        "fusion_strategy": "attention",
        "loss_weights": {"data_loss": 1.0, "regularization": 1.1},
        "performance": 0.87,
    }
    store.add_experiment(record)

    loaded = store.load()
    assert len(loaded) >= 1

    similar = store.get_similar_context(
        {
            "num_rows": 1500,
            "num_cols": 22,
            "modalities": ["tabular", "text"],
            "target_type": "classification",
        }
    )
    assert similar and similar[0]["fusion_strategy"] == "attention"


def test_retraining_orchestrator_trigger() -> None:
    from pipeline.retraining_orchestrator import RetrainingOrchestrator
    from pipeline import retrain_executor

    original_retrain = retrain_executor.RetrainingPipeline.retrain

    def _mock_retrain(self, production_sources, problem_type, modalities, schema_info=None):
        return {
            "model_id": "mock_model_001",
            "deployment_ready": True,
            "sources": production_sources,
            "problem_type": problem_type,
            "modalities": modalities,
        }

    retrain_executor.RetrainingPipeline.retrain = _mock_retrain
    try:
        orchestrator = RetrainingOrchestrator(
            production_sources=["mock_source"],
            problem_type="classification_binary",
            modalities=["tabular"],
            cooldown_seconds=3600,
        )
        assert orchestrator.should_retrain({"drift_detected": True, "composite_score": 1.2})

        first = orchestrator.trigger_retraining("dataset_a", {"drift_detected": True})
        assert first.get("triggered") is True

        second = orchestrator.trigger_retraining("dataset_a", {"drift_detected": True})
        assert second.get("triggered") is False
        assert second.get("status") == "cooldown_blocked"
    finally:
        retrain_executor.RetrainingPipeline.retrain = original_retrain


def test_graph_fusion_attention_logging() -> None:
    from modelss.fusion import GraphFusion

    fusion = GraphFusion(dim=64, num_modalities=2, heads=2, input_dims=[16, 12])
    x1 = torch.randn(8, 16)
    x2 = torch.randn(8, 12)
    y = fusion.forward([x1, x2])

    assert y.shape == (8, 64)
    summary = fusion.get_attention_summary()
    assert "modality_importance" in summary
    assert "head_diversity" in summary
    assert len(summary["modality_importance"]) == 2



def test_semantic_analyzer_roles() -> None:
    from data_ingestion.semantic_analyzer import SemanticAnalyzer

    df = pd.DataFrame(
        {
            "customer_id": [f"id_{i}" for i in range(30)],
            "category": ["a", "b", "c"] * 10,
            "amount": [float(i) for i in range(30)],
            "review_text": ["great product and fast delivery"] * 30,
            "event_time": pd.date_range("2024-01-01", periods=30, freq="D"),
        }
    )

    analyzer = SemanticAnalyzer()
    roles = analyzer.infer_column_roles(df)

    assert "customer_id" in roles["id_columns"]
    assert "amount" in roles["numeric"]
    assert "review_text" in roles["text"]
    assert "event_time" in roles["time_series"]


class _TinyTabularDataset(Dataset):
    def __init__(self, n: int = 16, dim: int = 8) -> None:
        self.x = torch.randn(n, dim)
        self.y = torch.randint(0, 2, (n,), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return {
            "tabular": self.x[idx],
            "target": self.y[idx],
        }



def test_training_minimal_e2e() -> None:
    import pytorch_lightning as pl

    from automl.trainer import build_trainer

    module = build_trainer(
        problem_type="classification_binary",
        num_classes=2,
        input_dims={"tabular": 8},
        learning_rate=1e-3,
        max_epochs=1,
        fusion_strategy="concatenation",
    )

    ds = _TinyTabularDataset(n=16, dim=8)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        limit_train_batches=2,
        limit_val_batches=1,
    )
    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)

    loss_state = module.get_loss_weight_state()
    assert "regularization" in loss_state
    assert isinstance(module.get_fusion_attention_logs(), list)


if __name__ == "__main__":
    tests: List[Tuple[str, Callable[[], None]]] = [
        ("MetaLearningStore persists data", test_meta_learning_store_persistence),
        ("Retraining orchestrator triggers and respects cooldown", test_retraining_orchestrator_trigger),
        ("GraphFusion logs attention summary", test_graph_fusion_attention_logging),
        ("SemanticAnalyzer infers roles", test_semantic_analyzer_roles),
        ("Training runs end-to-end", test_training_minimal_e2e),
    ]

    outcomes = [_run_test(name, fn) for name, fn in tests]
    passed = sum(1 for _, ok, _ in outcomes if ok)
    total = len(outcomes)

    print("-" * 70)
    print(f"RESULT: {passed}/{total} tests passed")
    print("PASS" if passed == total else "FAIL")
