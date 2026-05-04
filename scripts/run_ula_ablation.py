"""ULA fusion strategy ablation study.

Compares 7 fusion configurations on the same dataset across N seeds:
  1. concatenation (baseline)
  2. attention
  3. structural_semantic
  4. gated
  5. ula (latent_dim=256, n_layers=2)
  6. ula-large (latent_dim=512, n_layers=4)
  7. ula+lora r=8

Usage::

    python scripts/run_ula_ablation.py --dataset data/fixtures/hateful_memes/hateful_memes_n1000.csv
    python scripts/run_ula_ablation.py --seeds 42 123 456 --epochs 5 --output diary/results/ula_ablation.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

logger = logging.getLogger("apex.ula_ablation")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_CONDITIONS: List[Dict[str, Any]] = [
    {"name": "concatenation", "fusion_strategy": "concatenation", "lora_config": None},
    {"name": "attention", "fusion_strategy": "attention", "lora_config": None},
    {"name": "structural_semantic", "fusion_strategy": "structural_semantic", "lora_config": None},
    {"name": "gated", "fusion_strategy": "gated", "lora_config": None},
    {"name": "ula_256_2", "fusion_strategy": "ula",
     "fusion_config": {"latent_dim": 256, "n_layers": 2, "n_heads": 4}, "lora_config": None},
    {"name": "ula_512_4", "fusion_strategy": "ula",
     "fusion_config": {"latent_dim": 512, "n_layers": 4, "n_heads": 8}, "lora_config": None},
    {"name": "ula_lora_r8", "fusion_strategy": "ula",
     "fusion_config": {"latent_dim": 256, "n_layers": 2, "n_heads": 4},
     "lora_config": {"r": 8, "alpha": 16}},
]


def _run_one_trial(
    df_path: str,
    seed: int,
    condition: Dict[str, Any],
    max_epochs: int,
    n_trials: int,
) -> Dict[str, Any]:
    """Run a single (seed, condition) trial and return metric dict."""
    import torch
    import pytorch_lightning as pl

    pl.seed_everything(seed, workers=True)

    try:
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        df = pd.read_csv(df_path)
        if df.empty:
            raise ValueError(f"Empty dataset: {df_path}")

        # Detect target and feature columns heuristically
        target_col = None
        for candidate in ["label", "target", "y", df.columns[-1]]:
            if candidate in df.columns:
                target_col = candidate
                break

        if target_col is None:
            raise ValueError("Could not detect target column in dataset")

        feature_cols = [c for c in df.columns if c != target_col]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            logger.warning("No numeric features — using random features for ablation")
            numeric_cols = feature_cols[:1]

        X = df[numeric_cols].fillna(0).values.astype(np.float32)
        y_raw = df[target_col].values

        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y if len(np.unique(y)) > 1 else None
        )

        from automl.trainer import build_trainer
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        input_dims = {"tabular": X_tr.shape[1]}
        n_classes = len(np.unique(y))
        problem_type = "classification_binary" if n_classes == 2 else "classification_multiclass"

        fusion_config = condition.get("fusion_config", {})
        lora_config = condition.get("lora_config")

        module = build_trainer(
            problem_type=problem_type,
            num_classes=n_classes,
            input_dims=input_dims,
            learning_rate=1e-3,
            max_epochs=max_epochs,
            hidden_dim=128,
            fusion_strategy=condition["fusion_strategy"],
            fusion_config=fusion_config,
            lora_config=lora_config,
        )

        X_tr_t = torch.tensor(X_tr)
        y_tr_t = torch.tensor(y_tr, dtype=torch.long)
        X_val_t = torch.tensor(X_val)
        y_val_t = torch.tensor(y_val, dtype=torch.long)

        train_ds = TensorDataset(X_tr_t, y_tr_t)
        val_ds = TensorDataset(X_val_t, y_val_t)
        train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)

        # Wrap dataset to return dict-style batches
        class _DictDataLoader:
            def __init__(self, dl):
                self._dl = dl
            def __iter__(self):
                for x, y in self._dl:
                    yield {"tabular": x.float(), "target": y}
            def __len__(self):
                return len(self._dl)

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )

        t0 = time.time()
        trainer.fit(module, _DictDataLoader(train_dl), _DictDataLoader(val_dl))
        elapsed = time.time() - t0

        # Collect metrics from the last validation epoch
        metrics = trainer.callback_metrics
        val_acc = float(metrics.get("val_acc", 0.0))
        val_loss = float(metrics.get("val_loss", float("inf")))
        val_f1 = float(metrics.get("val_f1", 0.0))

        # Count trainable params
        total_p = sum(p.numel() for p in module.parameters())
        train_p = sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "seed": seed,
            "condition": condition["name"],
            "fusion_strategy": condition["fusion_strategy"],
            "lora_config": lora_config,
            "val_acc": val_acc,
            "val_loss": val_loss,
            "val_f1": val_f1,
            "elapsed_s": round(elapsed, 2),
            "total_params": total_p,
            "trainable_params": train_p,
            "status": "ok",
        }

    except Exception as exc:
        logger.warning("Trial failed (seed=%d, cond=%s): %s", seed, condition["name"], exc)
        return {
            "seed": seed,
            "condition": condition["name"],
            "status": "failed",
            "error": str(exc),
        }


def run_ablation(
    dataset: str,
    seeds: List[int],
    max_epochs: int,
    n_trials: int,
    output: str,
) -> None:
    results: List[Dict[str, Any]] = []

    for cond in _CONDITIONS:
        logger.info("== Condition: %s ==", cond["name"])
        for seed in seeds:
            logger.info("  seed=%d ...", seed)
            r = _run_one_trial(dataset, seed, cond, max_epochs, n_trials)
            results.append(r)

    # Compute per-condition summary statistics
    from scripts.aggregate_results import compute_statistics  # type: ignore
    summary: Dict[str, Any] = {}
    for cond in _CONDITIONS:
        name = cond["name"]
        cond_results = [r for r in results if r.get("condition") == name and r.get("status") == "ok"]
        if cond_results:
            accs = [r["val_acc"] for r in cond_results]
            f1s = [r["val_f1"] for r in cond_results]
            summary[name] = {
                "val_acc": compute_statistics(accs),
                "val_f1": compute_statistics(f1s),
                "n_seeds": len(cond_results),
                "fusion_strategy": cond["fusion_strategy"],
            }

    out = {
        "description": "ULA fusion strategy ablation study",
        "dataset": dataset,
        "seeds": seeds,
        "max_epochs": max_epochs,
        "conditions": [c["name"] for c in _CONDITIONS],
        "results": results,
        "summary": summary,
    }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)

    logger.info("ULA ablation results written to %s", out_path)
    _print_summary_table(summary)


def _print_summary_table(summary: Dict[str, Any]) -> None:
    logger.info("\n%-22s  %8s ± %6s  %8s ± %6s", "Condition", "val_acc", "std", "val_f1", "std")
    logger.info("-" * 65)
    for name, stats in summary.items():
        acc = stats["val_acc"]
        f1 = stats["val_f1"]
        logger.info(
            "%-22s  %8.4f ± %6.4f  %8.4f ± %6.4f",
            name,
            acc.get("mean") or 0.0, acc.get("std") or 0.0,
            f1.get("mean") or 0.0,  f1.get("std") or 0.0,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ULA fusion ablation")
    parser.add_argument("--dataset", default="data/fixtures/hateful_memes/hateful_memes_n1000.csv")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 2026])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--n_trials", type=int, default=1)
    parser.add_argument("--output", default="diary/results/ula_ablation.json")
    args = parser.parse_args()
    run_ablation(
        dataset=args.dataset,
        seeds=args.seeds,
        max_epochs=args.epochs,
        n_trials=args.n_trials,
        output=args.output,
    )
