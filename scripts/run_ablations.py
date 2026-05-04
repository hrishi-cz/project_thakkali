#!/usr/bin/env python
"""Execute the predefined ablation battery.

Usage:
    APEX_SEED=42 python scripts/run_ablations.py [--dataset PATH] [--output PATH]

Runs every ablation condition defined in
``pipeline.experiment_engine.PREDEFINED_ABLATIONS`` against a given dataset
and saves structured results to JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.types import TrainingConfig
from pipeline.experiment_engine import ExperimentManager, PREDEFINED_ABLATIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run APEX ablation battery")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset CSV/Parquet. Defaults to data/fixtures/*.csv.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="diary/results/ablations.json",
        help="Output path for ablation results JSON.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if __import__("torch").cuda.is_available() else "cpu",
        help="Compute device (cuda or cpu).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,123,456,789,2026",
        help="Comma-separated seeds for multi-seed reproducibility.",
    )
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    # Resolve dataset sources
    if args.dataset:
        sources = [args.dataset]
    else:
        fixtures = sorted((_ROOT / "data" / "fixtures").glob("*.csv"))
        sources = [str(f) for f in fixtures]
        if not sources:
            logger.error("No dataset specified and no fixtures found in data/fixtures/")
            sys.exit(1)
    logger.info("Dataset sources: %s", sources)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_seed_results = {}

    for seed in seeds:
        logger.info("=" * 60)
        logger.info("Running ablation battery with SEED=%d", seed)
        logger.info("=" * 60)

        os.environ["APEX_SEED"] = str(seed)

        # Detect modalities from dataset: if text column present, add text modality
        import pandas as _pd
        _sample = _pd.read_csv(sources[0], nrows=5) if sources else _pd.DataFrame()
        _has_text = any(
            c.lower() in ("text", "review", "caption", "plot", "sentence")
            for c in _sample.columns
        )
        _modalities = ["text", "tabular"] if _has_text else ["tabular"]

        # Build base training config per seed (TrainingConfig only accepts its declared fields)
        base_config = TrainingConfig(
            dataset_sources=sources,
            problem_type="classification_binary",
            modalities=_modalities,
            device=args.device,
            seed=seed,
        )

        # Per-seed output
        seed_output = output_path.parent / f"ablations_seed{seed}.json"
        mgr = ExperimentManager(
            base_training_config=base_config,
            store_path=seed_output,
        )
        results = mgr.run_ablations(PREDEFINED_ABLATIONS)

        # Summary per seed
        completed = [r for r in results if r.status == "completed"]
        failed = [r for r in results if r.status == "failed"]
        logger.info(
            "Seed %d: %d/%d succeeded, %d failed",
            seed, len(completed), len(results), len(failed),
        )
        for r in results:
            logger.info(
                "  %-25s  status=%-10s  val_acc=%.4f  val_loss=%.4f  time=%.1fs",
                r.name, r.status, r.best_val_acc, r.best_val_loss, r.duration_s,
            )
        all_seed_results[seed] = {
            "completed": len(completed),
            "failed": len(failed),
            "total": len(results),
            "output": str(seed_output),
        }

    # Aggregate summary
    agg_path = output_path.parent / "ablation_aggregate.json"
    with open(agg_path, "w") as f:
        json.dump(all_seed_results, f, indent=2)
    logger.info("Multi-seed aggregate saved to: %s", agg_path)
    logger.info("Individual results: %s", [v["output"] for v in all_seed_results.values()])


if __name__ == "__main__":
    main()
