"""LoRA rank ablation study.

Sweeps lora_r ∈ {0, 4, 8, 16, 32, 64} where 0 = frozen baseline.
Reports trainable param count, val accuracy, training time per epoch.

Usage::

    python scripts/run_lora_ablation.py
    python scripts/run_lora_ablation.py --dataset data/fixtures/titanic.csv --seeds 42 123
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

logger = logging.getLogger("apex.lora_ablation")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_LORA_RANKS = [0, 4, 8, 16, 32, 64]


def _run_one(
    df_path: str,
    seed: int,
    lora_r: int,
    max_epochs: int,
) -> Dict[str, Any]:
    import torch
    import pytorch_lightning as pl

    pl.seed_everything(seed, workers=True)

    try:
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        df = pd.read_csv(df_path)
        target_col = None
        for c in ["label", "target", "Survived", "survived", df.columns[-1]]:
            if c in df.columns:
                target_col = c
                break

        feature_cols = [c for c in df.columns if c != target_col]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            numeric_cols = feature_cols[:1]

        X = df[numeric_cols].fillna(0).values.astype(np.float32)
        le = LabelEncoder()
        y = le.fit_transform(df[target_col].astype(str))

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, random_state=seed,
            stratify=y if len(np.unique(y)) > 1 else None
        )

        from automl.trainer import build_trainer
        from torch.utils.data import DataLoader, TensorDataset

        n_classes = len(np.unique(y))
        problem_type = "classification_binary" if n_classes == 2 else "classification_multiclass"
        lora_config = {"r": lora_r, "alpha": lora_r * 2.0} if lora_r > 0 else None

        # Build a text-like encoder with Linear layers so LoRA has something to adapt
        text_enc = _make_mock_text_encoder(lora_r) if lora_r > 0 else None

        module = build_trainer(
            problem_type=problem_type,
            num_classes=n_classes,
            input_dims={"tabular": X_tr.shape[1]},
            learning_rate=1e-3,
            max_epochs=max_epochs,
            hidden_dim=64,
            fusion_strategy="concatenation",
            lora_config=lora_config,
            text_encoder=text_enc,
        )

        train_dl = DataLoader(
            TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.long)),
            batch_size=32, shuffle=True,
        )
        val_dl = DataLoader(
            TensorDataset(torch.tensor(X_val), torch.tensor(y_val, dtype=torch.long)),
            batch_size=64, shuffle=False,
        )

        class _DL:
            def __init__(self, dl, has_text=False):
                self._dl = dl; self._has_text = has_text
            def __iter__(self):
                for x, y in self._dl:
                    d = {"tabular": x.float(), "target": y}
                    yield d
            def __len__(self): return len(self._dl)

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )

        t0 = time.time()
        trainer.fit(module, _DL(train_dl), _DL(val_dl))
        elapsed = time.time() - t0

        # Count trainable and LoRA params
        total_p = sum(p.numel() for p in module.parameters())
        train_p = sum(p.numel() for p in module.parameters() if p.requires_grad)
        lora_p = 0
        if text_enc is not None:
            try:
                from modelss.adapters.lora import lora_parameters
                lora_p = sum(p.numel() for p in lora_parameters(text_enc))
            except Exception:
                pass

        metrics = trainer.callback_metrics
        return {
            "seed": seed,
            "lora_r": lora_r,
            "val_acc": float(metrics.get("val_acc", 0.0)),
            "val_loss": float(metrics.get("val_loss", float("inf"))),
            "elapsed_s": round(elapsed, 2),
            "time_per_epoch_s": round(elapsed / max_epochs, 3),
            "total_params": total_p,
            "trainable_params": train_p,
            "lora_params": lora_p,
            "status": "ok",
        }

    except Exception as exc:
        logger.warning("Trial failed (seed=%d, r=%d): %s", seed, lora_r, exc)
        return {"seed": seed, "lora_r": lora_r, "status": "failed", "error": str(exc)}


def _make_mock_text_encoder(lora_r: int):
    """Create a small nn.Module with query/value Linears so LoRA has real targets."""
    import torch.nn as nn

    class _MockTextEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(64, 64)
            self.value = nn.Linear(64, 64)
            for p in self.parameters():
                p.requires_grad_(False)

        def forward(self, x):
            return x

    return _MockTextEncoder()


def run_lora_ablation(
    dataset: str,
    seeds: List[int],
    max_epochs: int,
    output: str,
) -> None:
    results = []
    for r in _LORA_RANKS:
        logger.info("== LoRA r=%d ==", r)
        for seed in seeds:
            logger.info("  seed=%d ...", seed)
            results.append(_run_one(dataset, seed, r, max_epochs))

    from scripts.aggregate_results import compute_statistics  # type: ignore
    summary: Dict[str, Any] = {}
    for r in _LORA_RANKS:
        ok = [x for x in results if x.get("lora_r") == r and x.get("status") == "ok"]
        if ok:
            summary[f"r={r}"] = {
                "lora_r": r,
                "val_acc": compute_statistics([x["val_acc"] for x in ok]),
                "trainable_params": compute_statistics([float(x["trainable_params"]) for x in ok]),
                "lora_params": compute_statistics([float(x["lora_params"]) for x in ok]),
                "time_per_epoch_s": compute_statistics([x["time_per_epoch_s"] for x in ok]),
                "n_seeds": len(ok),
            }

    out = {
        "description": "LoRA rank ablation: r ∈ {0, 4, 8, 16, 32, 64}",
        "dataset": dataset,
        "seeds": seeds,
        "ranks_tested": _LORA_RANKS,
        "results": results,
        "summary": summary,
    }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)

    logger.info("LoRA ablation results written to %s", out_path)
    logger.info("\n%-8s  %8s ± %6s  %12s  %10s", "r", "val_acc", "std", "trainable_params", "lora_params")
    logger.info("-" * 60)
    for k, v in summary.items():
        acc = v["val_acc"]
        tp = v["trainable_params"]
        lp = v["lora_params"]
        logger.info(
            "%-8s  %8.4f ± %6.4f  %12.0f  %10.0f",
            k, acc.get("mean") or 0.0, acc.get("std") or 0.0,
            tp.get("mean") or 0.0, lp.get("mean") or 0.0,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA rank ablation")
    parser.add_argument("--dataset", default="data/fixtures/titanic_n200.csv")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output", default="diary/results/lora_ablation.json")
    args = parser.parse_args()
    run_lora_ablation(args.dataset, args.seeds, args.epochs, args.output)
