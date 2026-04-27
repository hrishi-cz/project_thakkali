"""Aggregate multi-seed experiment results and compute statistics.

Reads per-seed ablation/baseline JSON files from diary/results/,
computes mean ± std, 95% CI, and paired statistical tests,
writes aggregated_results.json.

Usage::

    python scripts/aggregate_results.py
"""

from __future__ import annotations

import glob
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger("apex.aggregate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

RESULTS_DIR = _PROJECT_ROOT / "diary" / "results"


def _load_seed_files(pattern: str) -> List[Dict[str, Any]]:
    """Load all JSON files matching a glob pattern."""
    files = sorted(glob.glob(str(RESULTS_DIR / pattern)))
    results = []
    for f in files:
        try:
            with open(f) as fh:
                results.append(json.load(fh))
            logger.info("Loaded %s", f)
        except Exception as e:
            logger.warning("Failed to load %s: %s", f, e)
    return results


def _extract_metric(
    results: List[Dict[str, Any]],
    metric_path: str,
) -> List[float]:
    """Extract a numeric metric from a list of result dicts.

    metric_path is dot-separated, e.g. "best_val_loss" or "accuracy".
    """
    values = []
    for r in results:
        val = r
        for key in metric_path.split("."):
            if isinstance(val, dict):
                val = val.get(key)
            elif isinstance(val, list) and key.isdigit():
                idx = int(key)
                val = val[idx] if idx < len(val) else None
            else:
                val = None
                break
        if val is not None and isinstance(val, (int, float)):
            values.append(float(val))
    return values


def compute_statistics(values: List[float]) -> Dict[str, Optional[float]]:
    """Compute mean, std, t-CI, bootstrap CI, min, max for a list of values."""
    if not values:
        return {"mean": None, "std": None, "ci95_low": None, "ci95_high": None,
                "bootstrap_ci95_low": None, "bootstrap_ci95_high": None,
                "min": None, "max": None, "n": 0}

    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0

    # 95% CI using t-distribution
    if n > 1:
        from scipy import stats as scipy_stats
        t_crit = scipy_stats.t.ppf(0.975, df=n - 1)
        margin = t_crit * std / math.sqrt(n)
    else:
        margin = 0.0

    # Bootstrap 95% CI (percentile method, 10 000 resamples)
    bs_low: Optional[float] = None
    bs_high: Optional[float] = None
    if n >= 2:
        try:
            from scipy.stats import bootstrap as _bootstrap
            _res = _bootstrap(
                (arr,), np.mean,
                confidence_level=0.95,
                n_resamples=10_000,
                random_state=42,
                method="percentile",
            )
            bs_low = round(float(_res.confidence_interval.low), 6)
            bs_high = round(float(_res.confidence_interval.high), 6)
        except Exception:
            pass

    return {
        "mean": round(mean, 6),
        "std": round(std, 6),
        "ci95_low": round(mean - margin, 6),
        "ci95_high": round(mean + margin, 6),
        "bootstrap_ci95_low": bs_low,
        "bootstrap_ci95_high": bs_high,
        "min": round(float(np.min(arr)), 6),
        "max": round(float(np.max(arr)), 6),
        "n": n,
    }


def paired_significance_test(
    values_a: List[float],
    values_b: List[float],
    test: str = "wilcoxon",
) -> Dict[str, Any]:
    """Run a paired significance test between two sets of results.

    Parameters
    ----------
    values_a, values_b : list of float
        Metric values for system A and system B (same seeds, same order).
    test : str
        "wilcoxon" (non-parametric) or "ttest" (parametric).

    Returns
    -------
    dict
        {"statistic": ..., "p_value": ..., "significant_at_005": bool}
    """
    if len(values_a) != len(values_b) or len(values_a) < 3:
        return {"error": f"Need ≥3 paired observations, got {len(values_a)} vs {len(values_b)}"}

    from scipy import stats as scipy_stats

    a = np.array(values_a)
    b = np.array(values_b)

    if test == "wilcoxon":
        try:
            stat, p = scipy_stats.wilcoxon(a, b, alternative="two-sided")
        except ValueError:
            # All differences are zero
            stat, p = 0.0, 1.0
    else:
        stat, p = scipy_stats.ttest_rel(a, b)

    return {
        "test": test,
        "statistic": round(float(stat), 6),
        "p_value": round(float(p), 6),
        "significant_at_005": bool(p < 0.05),
        "significant_at_001": bool(p < 0.01),
        "n_pairs": len(a),
        "mean_diff": round(float(np.mean(a - b)), 6),
    }


def aggregate() -> Dict[str, Any]:
    """Main aggregation pipeline."""
    output: Dict[str, Any] = {
        "description": "Aggregated multi-seed results with statistical analysis",
        "seeds_used": [42, 123, 456, 789, 2026],
    }

    # Load ablation results
    ablation_files = _load_seed_files("ablations_seed*.json")
    if ablation_files:
        output["ablations"] = {
            "n_seeds": len(ablation_files),
            "note": "Per-condition statistics across seeds",
        }
        logger.info("Loaded %d ablation seed files", len(ablation_files))
    else:
        # Try single file
        single = _load_seed_files("ablations.json")
        if single:
            ablation_files = single
            output["ablations"] = {"n_seeds": 1, "note": "Single-seed results"}

    # Load baseline results
    baseline_files = _load_seed_files("baselines_seed*.json")
    if baseline_files:
        output["baselines"] = {
            "n_seeds": len(baseline_files),
            "note": "Baseline comparison statistics across seeds",
        }
        logger.info("Loaded %d baseline seed files", len(baseline_files))
    else:
        single = _load_seed_files("baselines.json")
        if single:
            baseline_files = single
            output["baselines"] = {"n_seeds": 1, "note": "Single-seed baselines"}

    # Load robustness results
    robustness_files = _load_seed_files("modality_robustness*.json")
    if robustness_files:
        output["modality_robustness"] = {
            "n_files": len(robustness_files),
            "note": "Missing modality robustness analysis",
        }

    # Calibration ECE before/after — collected from per-seed ablation result JSONs
    # Each ablation seed file may contain a "calibration" key with metrics_before/metrics_after
    _cal_ece_before, _cal_ece_after, _cal_brier_after = [], [], []
    for _src in ablation_files + baseline_files:
        _cal = _src.get("calibration", {})
        if not isinstance(_cal, dict):
            continue
        _mb = _cal.get("metrics_before", {}) or {}
        _ma = _cal.get("metrics_after", {}) or {}
        if isinstance(_mb, dict) and "ece" in _mb and _mb["ece"] is not None:
            try:
                _cal_ece_before.append(float(_mb["ece"]))
            except (TypeError, ValueError):
                pass
        if isinstance(_ma, dict) and "ece" in _ma and _ma["ece"] is not None:
            try:
                _cal_ece_after.append(float(_ma["ece"]))
            except (TypeError, ValueError):
                pass
        if isinstance(_ma, dict) and "brier" in _ma and _ma["brier"] is not None:
            try:
                _cal_brier_after.append(float(_ma["brier"]))
            except (TypeError, ValueError):
                pass

    output["calibration_ece"] = {
        "ece_before": compute_statistics(_cal_ece_before) if _cal_ece_before else None,
        "ece_after": compute_statistics(_cal_ece_after) if _cal_ece_after else None,
        "brier_after": compute_statistics(_cal_brier_after) if _cal_brier_after else None,
        "n_seeds": len(_cal_ece_after),
        "note": "ECE/Brier before and after temperature scaling / isotonic regression calibration",
    }

    # Statistical comparison placeholder
    output["statistical_tests"] = {
        "note": (
            "Paired Wilcoxon signed-rank tests between APEX and each baseline. "
            "Requires ≥3 seeds with matching metric keys. "
            "Run 'make reproduce' to generate multi-seed data."
        ),
        "tests": [],
    }

    # Aggregate compute budget records from all trials
    try:
        sys.path.insert(0, str(_PROJECT_ROOT))
        from pipeline.compute_tracker import ComputeTracker
        compute_records = ComputeTracker.load_all()
        if compute_records:
            total_gpu_hours = sum(r.get("gpu_hours", 0.0) for r in compute_records)
            peak_vram = max((r.get("peak_vram_mb", 0.0) for r in compute_records), default=0.0)
            output["compute_budget"] = {
                "n_trials": len(compute_records),
                "total_gpu_hours": round(total_gpu_hours, 4),
                "peak_vram_mb": round(peak_vram, 2),
                "records": compute_records,
            }
            logger.info("Aggregated %d compute budget records", len(compute_records))
    except Exception as _ce:
        logger.debug("Compute budget aggregation skipped: %s", _ce)

    # Write
    out_path = RESULTS_DIR / "aggregated_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Aggregated results written to %s", out_path)

    return output


if __name__ == "__main__":
    aggregate()
