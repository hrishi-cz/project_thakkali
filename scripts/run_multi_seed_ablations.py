"""
scripts/run_multi_seed_ablations.py
=====================================
Run all PREDEFINED_ABLATIONS across 5 seeds and compute paired Wilcoxon
signed-rank tests. Uses EXISTING repo fixtures — no real dataset download needed.

Default dataset: data/fixtures/adult_income_smoke.csv (2,000 rows, tabular)

Usage
-----
# Fast (1 trial, 2 epochs per run — ~10-15 min total on GPU):
python scripts/run_multi_seed_ablations.py --quick

# Full (respects ExperimentManager epochs/trials settings — ~1-2 hours):
python scripts/run_multi_seed_ablations.py

# Custom dataset:
python scripts/run_multi_seed_ablations.py --dataset path/to/data.csv

Output
------
diary/results/ablations_seed{N}.json   — per-condition results per seed (via ExperimentManager)
diary/results/wilcoxon_results.json    — Wilcoxon p-values + effect sizes + bootstrap CI
diary/results/aggregated_results.json  — updated statistical_tests block

Scientific references
---------------------
Wilcoxon signed-rank test: Wilcoxon 1945 (Biometrics 1:80–83)
Comparison framework: Demsar 2006 (JMLR 7:1–30)
Effect size: rank-biserial correlation r = Z/sqrt(N) (Kerby 2014)
Bootstrap CI: Efron 1979 (Ann. Stat. 7(1):1–26), B=2000 resamples
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = ROOT / "diary" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WILCOXON_OUT   = RESULTS_DIR / "wilcoxon_results.json"
AGGREGATED_OUT = RESULTS_DIR / "aggregated_results.json"

DEFAULT_FIXTURE = ROOT / "data" / "fixtures" / "adult_income_smoke.csv"
DEFAULT_SEEDS   = [42, 123, 456, 789, 2026]


# ─── Statistical utilities ─────────────────────────────────────────────────

def wilcoxon_test(x: List[float], y: List[float],
                  alpha: float = 0.05) -> Dict[str, Any]:
    """
    Paired Wilcoxon signed-rank test (two-sided, Wilcoxon 1945).
    x, y: matched paired observations (same seed, different condition).
    Effect size: rank-biserial correlation r (Kerby 2014).
    """
    try:
        from scipy.stats import wilcoxon as _wt
    except ImportError:
        return {"p_value": 1.0, "significant": False, "note": "scipy not installed"}

    diffs = [xi - yi for xi, yi in zip(x, y)]
    non_zero = [d for d in diffs if abs(d) > 1e-12]
    if len(non_zero) < 3:
        return {
            "p_value": 1.0, "statistic": 0.0, "significant": False,
            "effect_size_r": 0.0,
            "note": f"Only {len(non_zero)} non-zero diffs — need ≥3 for valid test",
        }
    stat, p = _wt(x, y, alternative="two-sided", zero_method="wilcox")
    n = len(non_zero)
    # Z from normal approximation of Wilcoxon statistic
    mu    = n * (n + 1) / 4.0
    sigma = ((n * (n + 1) * (2 * n + 1)) / 24.0) ** 0.5
    z     = abs(stat - mu) / (sigma + 1e-12)
    r     = min(1.0, z / (n ** 0.5 + 1e-12))   # rank-biserial correlation
    return {
        "p_value": float(p), "statistic": float(stat),
        "significant": bool(p < alpha),
        "effect_size_r": float(r), "n_pairs": int(n),
    }


def bootstrap_ci(values: List[float], n_boot: int = 2000,
                 ci: float = 0.95) -> Tuple[float, float]:
    """Bootstrap percentile CI (Efron 1979)."""
    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(42)
    boot = [rng.choice(arr, size=len(arr), replace=True).mean()
            for _ in range(n_boot)]
    lo = float(np.percentile(boot, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot, (1 + ci) / 2 * 100))
    return lo, hi


# ─── Load per-seed ablation results ────────────────────────────────────────

def _load_seed_results(seed: int) -> Dict[str, Dict[str, float]]:
    """
    Load ablation results for a single seed from
    diary/results/ablations_seed{seed}.json.
    Returns {condition_name: {val_f1, val_acc, val_loss}}.
    """
    path = RESULTS_DIR / f"ablations_seed{seed}.json"
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return {
                item["name"]: {
                    "val_f1":   float(item.get("best_val_f1",  0.0)),
                    "val_acc":  float(item.get("best_val_acc", 0.0)),
                    "val_loss": float(item.get("best_val_loss", float("inf"))),
                }
                for item in data if item.get("status") == "completed"
            }
        # dict-of-dicts format
        if isinstance(data, dict):
            return {
                k: {
                    "val_f1":   float(v.get("best_val_f1",  0.0)),
                    "val_acc":  float(v.get("best_val_acc", 0.0)),
                    "val_loss": float(v.get("best_val_loss", float("inf"))),
                }
                for k, v in data.items() if isinstance(v, dict)
            }
    except Exception as exc:
        logger.warning("Could not load %s: %s", path, exc)
    return {}


# ─── Run ablations via existing ExperimentManager ─────────────────────────

def _run_seed(seed: int, dataset_path: Path, quick: bool) -> None:
    """
    Run PREDEFINED_ABLATIONS for one seed via ExperimentManager.
    Writes diary/results/ablations_seed{seed}.json.
    """
    import pandas as _pd
    from core.types import TrainingConfig
    from pipeline.experiment_engine import ExperimentManager, PREDEFINED_ABLATIONS

    os.environ["APEX_SEED"] = str(seed)

    _sample = _pd.read_csv(dataset_path, nrows=5)
    _has_text = any(
        c.lower() in ("text", "review", "caption", "plot", "sentence")
        for c in _sample.columns
    )
    _modalities = ["text", "tabular"] if _has_text else ["tabular"]

    base_config = TrainingConfig(
        dataset_sources=[str(dataset_path)],
        problem_type="classification_binary",
        modalities=_modalities,
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
        seed=seed,
    )

    conditions = PREDEFINED_ABLATIONS
    if quick:
        # Override to 1 trial / 2 epochs for speed
        from pipeline.experiment_engine import AblationCondition
        conditions = [
            AblationCondition(
                name=c.name,
                description=c.description,
                config_overrides={
                    **c.config_overrides,
                    "n_trials": 1,
                    "epochs": 2,
                },
            )
            for c in PREDEFINED_ABLATIONS
        ]

    out_path = RESULTS_DIR / f"ablations_seed{seed}.json"
    mgr = ExperimentManager(base_training_config=base_config, store_path=out_path)
    results = mgr.run_ablations(conditions)

    completed = sum(1 for r in results if r.status == "completed")
    logger.info("Seed %d: %d/%d completed → %s", seed, completed, len(results), out_path)
    for r in results:
        logger.info("  %-25s  %-10s  val_f1=%.4f  val_acc=%.4f",
                    r.name, r.status, r.best_val_f1, r.best_val_acc)


# ─── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-seed ablations + Wilcoxon tests on repo fixtures"
    )
    parser.add_argument("--dataset", default=str(DEFAULT_FIXTURE),
                        help=f"Path to CSV fixture (default: {DEFAULT_FIXTURE.name})")
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
                        help="Comma-separated seeds (default: 42,123,456,789,2026)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Wilcoxon significance level (default: 0.05)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 1 trial, 2 epochs — ~10-15 min total on GPU")
    parser.add_argument("--wilcoxon-only", action="store_true",
                        help="Skip runs — only compute Wilcoxon from existing seed JSONs")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    dataset_path = Path(args.dataset)

    if not dataset_path.exists():
        logger.error("Dataset not found: %s", dataset_path)
        sys.exit(1)

    logger.info("Dataset: %s", dataset_path)
    logger.info("Seeds:   %s", seeds)
    logger.info("Quick:   %s", args.quick)

    # ── Step 1: run ablations (skip if --wilcoxon-only) ────────────────────
    if not args.wilcoxon_only:
        t0 = time.time()
        for seed in seeds:
            seed_json = RESULTS_DIR / f"ablations_seed{seed}.json"
            if seed_json.exists() and not args.quick:
                logger.info("Seed %d: results already exist at %s — skipping",
                            seed, seed_json)
                continue
            logger.info("=" * 60)
            logger.info("Running seed %d ...", seed)
            try:
                _run_seed(seed, dataset_path, quick=args.quick)
            except Exception as exc:
                logger.error("Seed %d failed: %s", seed, exc)
        logger.info("All seeds done in %.1fs", time.time() - t0)

    # ── Step 2: load all seed results ──────────────────────────────────────
    logger.info("Loading results for seeds: %s", seeds)
    # condition → seed → {val_f1, val_acc, val_loss}
    by_condition: Dict[str, Dict[int, Dict[str, float]]] = {}
    for seed in seeds:
        results = _load_seed_results(seed)
        for cname, metrics in results.items():
            by_condition.setdefault(cname, {})[seed] = metrics

    if not by_condition:
        logger.error("No results found. Run without --wilcoxon-only first.")
        sys.exit(1)

    # ── Step 3: Wilcoxon — each condition vs baseline_concat ───────────────
    baseline_name = "baseline_concat"
    baseline_by_seed = by_condition.get(baseline_name, {})

    wilcoxon_tests: List[Dict[str, Any]] = []
    logger.info("Computing Wilcoxon tests (α=%.2f) ...", args.alpha)

    for cname, seed_metrics in sorted(by_condition.items()):
        if cname == baseline_name:
            continue
        # Align seeds: only use seeds present in BOTH baseline and this condition
        common_seeds = sorted(
            set(baseline_by_seed.keys()) & set(seed_metrics.keys())
        )
        if len(common_seeds) < 3:
            logger.warning("  %s: only %d matched seeds — Wilcoxon requires ≥3",
                           cname, len(common_seeds))
            continue

        x_base = [baseline_by_seed[s]["val_f1"] for s in common_seeds]
        x_cond = [seed_metrics[s]["val_f1"]     for s in common_seeds]

        wtest = wilcoxon_test(x_base, x_cond, alpha=args.alpha)
        mean_diff = float(np.mean(x_cond) - np.mean(x_base))
        diffs = [c - b for c, b in zip(x_cond, x_base)]
        ci_lo, ci_hi = (bootstrap_ci(diffs) if len(diffs) >= 3
                        else (mean_diff, mean_diff))

        wtest.update({
            "comparison":       f"{cname} vs {baseline_name}",
            "metric":           "val_f1",
            "baseline_mean":    float(np.mean(x_base)),
            "condition_mean":   float(np.mean(x_cond)),
            "mean_diff":        mean_diff,
            "bootstrap_ci95_lo": ci_lo,
            "bootstrap_ci95_hi": ci_hi,
            "seeds_used":       common_seeds,
            "n_seeds":          len(common_seeds),
        })
        wilcoxon_tests.append(wtest)
        sig = "✓ SIGNIFICANT" if wtest["significant"] else "✗ n.s."
        logger.info("  %-30s  p=%.4f  %s  r=%.3f  Δ=%+.4f  CI=[%.4f,%.4f]",
                    cname, wtest["p_value"], sig,
                    wtest.get("effect_size_r", 0.0), mean_diff, ci_lo, ci_hi)

    # ── Step 4: write Wilcoxon JSON ─────────────────────────────────────────
    wilcoxon_out = {
        "description": (
            "Paired Wilcoxon signed-rank tests comparing each fusion strategy "
            "vs baseline_concat on val_F1. "
            "Dataset: adult_income_smoke (2K rows, tabular). "
            "Null hypothesis: no difference in val_F1 distributions."
        ),
        "reference": (
            "Wilcoxon 1945 (Biometrics 1:80-83); "
            "Demsar 2006 (JMLR 7:1-30); "
            "Effect size: Kerby 2014 rank-biserial r"
        ),
        "alpha": args.alpha,
        "dataset": str(dataset_path.name),
        "seeds_attempted": seeds,
        "tests": wilcoxon_tests,
    }
    with open(WILCOXON_OUT, "w", encoding="utf-8") as f:
        json.dump(wilcoxon_out, f, indent=2, allow_nan=False, default=str)
    logger.info("Wilcoxon results → %s", WILCOXON_OUT)

    # ── Step 5: update aggregated_results.json ──────────────────────────────
    try:
        with open(AGGREGATED_OUT, encoding="utf-8", errors="replace") as f:
            aggregated = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        aggregated = {}

    # Per-condition summary with bootstrap CI
    per_cond_summary: Dict[str, Any] = {}
    for cname, seed_metrics in by_condition.items():
        for metric in ("val_f1", "val_acc", "val_loss"):
            vals = [v[metric] for v in seed_metrics.values()
                    if not np.isnan(v[metric]) and not np.isinf(v[metric])]
            if not vals:
                continue
            arr = np.array(vals)
            ci_lo, ci_hi = (bootstrap_ci(vals) if len(vals) >= 3
                            else (float(arr.mean()), float(arr.mean())))
            per_cond_summary.setdefault(cname, {})[metric] = {
                "mean": float(arr.mean()), "std": float(arr.std()),
                "ci95_lo": ci_lo, "ci95_hi": ci_hi,
                "n": len(vals), "seeds": sorted(seed_metrics.keys()),
            }

    aggregated["statistical_tests"] = {
        "method":      "Paired Wilcoxon signed-rank (Wilcoxon 1945, two-sided)",
        "effect_size": "Rank-biserial correlation r = Z/sqrt(N) (Kerby 2014)",
        "ci_method":   "Bootstrap percentile 95% CI (Efron 1979, B=2000)",
        "alpha":       args.alpha,
        "tests":       wilcoxon_tests,
    }
    aggregated["ablations_multiseed"] = {
        "n_seeds": len(seeds),
        "seeds_attempted": seeds,
        "conditions": per_cond_summary,
    }
    with open(AGGREGATED_OUT, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2, allow_nan=False, default=str)
    logger.info("Updated %s", AGGREGATED_OUT)

    # ── Print summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("WILCOXON SIGNIFICANCE SUMMARY  (val_F1, α=%.2f)" % args.alpha)
    print("=" * 70)
    if wilcoxon_tests:
        for t in wilcoxon_tests:
            sig = "✓ SIGNIFICANT" if t["significant"] else "✗ n.s."
            print(f"  {t['comparison']:40s} p={t['p_value']:.4f}  {sig:<14}  "
                  f"Δ={t['mean_diff']:+.4f}  r={t.get('effect_size_r',0):.3f}")
    else:
        print("  No tests computed. Check that ≥3 seed results exist per condition.")
    print("=" * 70)
    print(f"\nTo use in paper: {WILCOXON_OUT}")
    print("paper_generator.py reads aggregated_results.json automatically.\n")


if __name__ == "__main__":
    main()
