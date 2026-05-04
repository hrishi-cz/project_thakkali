#!/usr/bin/env python
"""Run baseline models for comparison with APEX pipeline results.

Usage:
    APEX_SEED=42 python scripts/run_baselines.py [--dataset PATH] [--output PATH]

Trains XGBoost (if installed) and a plain sklearn MLP on the same
dataset used by APEX ablations.  Outputs structured JSON for the
paper generator to produce comparison tables.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_SEED = int(os.getenv("APEX_SEED", "42"))


def _load_dataset(path: str) -> pd.DataFrame:
    """Load CSV or Parquet."""
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _prepare_tabular(df: pd.DataFrame) -> tuple:
    """Simple tabular prep: drop non-numeric, impute, split X/y (last col = target)."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    target_col = df.columns[-1]
    y_raw = df[target_col]
    X = df.drop(columns=[target_col])

    # Keep only numeric columns for baselines
    X_numeric = X.select_dtypes(include=[np.number]).fillna(0)

    # Encode target if categorical
    le = None
    if y_raw.dtype == object or str(y_raw.dtype) == "category":
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
    else:
        y = y_raw.values

    X_train, X_test, y_train, y_test = train_test_split(
        X_numeric.values, y, test_size=0.2, random_state=_SEED, stratify=y if len(np.unique(y)) > 1 else None,
    )
    return X_train, X_test, y_train, y_test, target_col, le


def _run_sklearn_mlp(X_train, X_test, y_train, y_test) -> Dict[str, Any]:
    """Train a plain sklearn MLP baseline."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, f1_score

    t0 = time.perf_counter()
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        max_iter=200,
        random_state=_SEED,
        early_stopping=True,
        validation_fraction=0.15,
    )
    clf.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    y_pred = clf.predict(X_test)
    return {
        "model": "sklearn_MLP",
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "train_time_s": round(elapsed, 2),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def _run_xgboost(X_train, X_test, y_train, y_test) -> Dict[str, Any] | None:
    """Train XGBoost baseline (skipped if not installed)."""
    try:
        import xgboost as xgb
    except ImportError:
        logger.warning("xgboost not installed — skipping XGBoost baseline")
        return None

    from sklearn.metrics import accuracy_score, f1_score

    n_classes = len(np.unique(y_train))
    t0 = time.perf_counter()
    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective="binary:logistic" if n_classes == 2 else "multi:softmax",
        num_class=n_classes if n_classes > 2 else None,
        random_state=_SEED,
        eval_metric="logloss",
        verbosity=0,
    )
    clf.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    y_pred = clf.predict(X_test)
    return {
        "model": "XGBoost",
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "train_time_s": round(elapsed, 2),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def _run_autogluon_multimodal(dataset_path: str, target_col: str, time_limit: int = 120) -> Dict[str, Any] | None:
    """
    Run AutoGluon-Multimodal baseline (skipped if autogluon not installed).

    AutoGluon-Multimodal v1.x (Shi et al., AutoML Conf 2024) is the direct
    AutoML competitor. We run it with default settings + time_limit to produce
    a fair comparison point for the paper.
    """
    try:
        from autogluon.multimodal import MultiModalPredictor
    except ImportError:
        logger.info("autogluon.multimodal not installed — skipping AutoGluon-Multimodal baseline")
        return None

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score

    try:
        df = pd.read_csv(dataset_path)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=_SEED)
        t0 = time.perf_counter()
        predictor = MultiModalPredictor(label=target_col)
        predictor.fit(train_df, time_limit=time_limit)
        elapsed = time.perf_counter() - t0

        preds = predictor.predict(test_df.drop(columns=[target_col]))
        y_true = test_df[target_col].values

        acc = float(accuracy_score(y_true, preds))
        f1 = float(f1_score(y_true, preds, average="weighted", zero_division=0))

        logger.info("AutoGluon-Multimodal: acc=%.4f f1=%.4f time=%.1fs", acc, f1, elapsed)
        return {
            "model": "AutoGluon-Multimodal",
            "accuracy": acc,
            "f1": f1,
            "train_time_s": round(elapsed, 2),
            "time_limit_s": time_limit,
            "citation": "Shi et al., AutoML Conf 2024",
        }
    except Exception as exc:
        logger.warning("AutoGluon-Multimodal baseline failed: %s", exc)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline models for APEX comparison")
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Path to dataset CSV/Parquet. Defaults to first data/fixtures/*.csv.",
    )
    parser.add_argument(
        "--output", type=str, default="diary/results/baselines.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    # Resolve dataset
    if args.dataset:
        ds_path = args.dataset
    else:
        fixtures = sorted((_ROOT / "data" / "fixtures").glob("*.csv"))
        if not fixtures:
            logger.error("No dataset specified and no fixtures found.")
            sys.exit(1)
        ds_path = str(fixtures[0])

    logger.info("Loading dataset: %s", ds_path)
    df = _load_dataset(ds_path)
    logger.info("Shape: %s", df.shape)

    X_train, X_test, y_train, y_test, target_col, le = _prepare_tabular(df)
    logger.info(
        "Prepared: train=%d, test=%d, features=%d, target=%s",
        len(X_train), len(X_test), X_train.shape[1], target_col,
    )

    results: List[Dict[str, Any]] = []

    # sklearn MLP
    mlp_result = _run_sklearn_mlp(X_train, X_test, y_train, y_test)
    results.append(mlp_result)
    logger.info("MLP: acc=%.4f f1=%.4f time=%.1fs", mlp_result["accuracy"], mlp_result["f1"], mlp_result["train_time_s"])

    # XGBoost
    xgb_result = _run_xgboost(X_train, X_test, y_train, y_test)
    if xgb_result:
        results.append(xgb_result)
        logger.info("XGBoost: acc=%.4f f1=%.4f time=%.1fs", xgb_result["accuracy"], xgb_result["f1"], xgb_result["train_time_s"])

    # AutoGluon-Multimodal (optional — skipped if not installed)
    ag_result = _run_autogluon_multimodal(ds_path, target_col, time_limit=120)
    if ag_result:
        results.append(ag_result)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "seed": _SEED,
            "dataset": ds_path,
            "target_column": target_col,
            "baselines": results,
        }, f, indent=2)
    logger.info("Results saved to: %s", output_path)


if __name__ == "__main__":
    main()
