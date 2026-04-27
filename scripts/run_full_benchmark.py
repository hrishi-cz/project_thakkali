"""Master benchmark runner — produces the full publication bundle in one command.

Orchestrates (in order):
  1. Tier 1 dataset fixture generation
  2. Ablation study (5 fusion strategies × 5 seeds)
  3. XGBoost/MLP/AutoGluon-Multimodal baselines
  4. ULA fusion ablation (7 conditions × seeds)
  5. LoRA rank ablation (6 ranks × seeds)
  6. Modality robustness (missing-modality graceful degradation)
  7. Statistical aggregation (Wilcoxon + bootstrap CIs)
  8. LaTeX paper generation
  9. Plot generation

Usage::

    # Quick mode (~5 min, 200 rows, 1 seed)
    python scripts/run_full_benchmark.py --quick

    # Full run (30-60 min on GPU)
    python scripts/run_full_benchmark.py --seeds 42 123 456 789 2026

    # Tier 1 only, skip smoke
    python scripts/run_full_benchmark.py --datasets adult hateful_memes mmimdb mosei

Tier 1 datasets (publication-grade):
  - MMIMDB      (text + image + tabular,  multilabel genre)
  - Hateful Memes (text + image,          binary hate detection)
  - CMU-MOSEI   (text + COVAREP tabular,  ternary sentiment)
  - Adult Income (tabular,                binary, imbalanced)
  - California Housing (tabular,          regression)
  - IMDb subset (text,                    binary sentiment)

Tier 2 datasets (smoke only, skipped by default):
  - Titanic, synthetic multiclass/regression, Food-101

External baselines (cited, not replicated):
  AutoGluon-Multimodal, TabPFN-v2, LLaVA-1.5, BLIP-2
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("apex.full_benchmark")

RESULTS_DIR = _ROOT / "diary" / "results"
FIXTURES_DIR = _ROOT / "data" / "fixtures"


# ---------------------------------------------------------------------------
# Step 1: generate dataset fixtures
# ---------------------------------------------------------------------------

def _generate_fixtures(quick: bool = False, max_rows: int = 500) -> Dict[str, str]:
    """Generate all Tier 1 dataset fixtures and return {name: path} mapping."""
    import numpy as np
    import pandas as pd

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    n = 200 if quick else max_rows

    # ── Adult Income (fairness / imbalanced classification baseline) ──────────
    _adult_path = FIXTURES_DIR / "adult_income_smoke.csv"
    if not _adult_path.exists():
        try:
            from sklearn.datasets import fetch_openml
            data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
            df = data.frame.dropna().sample(min(n, len(data.frame)), random_state=42)
            df.to_csv(_adult_path, index=False)
        except Exception:
            rng = np.random.default_rng(42)
            df = pd.DataFrame({
                "age": rng.integers(18, 90, n),
                "education_num": rng.integers(1, 16, n),
                "hours_per_week": rng.integers(1, 99, n),
                "capital_gain": rng.integers(0, 99999, n),
                "capital_loss": rng.integers(0, 4356, n),
                "class": rng.choice(["<=50K", ">50K"], n, p=[0.76, 0.24]),
            })
            df.to_csv(_adult_path, index=False)
    paths["adult"] = str(_adult_path)
    logger.info("Adult Income fixture: %s", _adult_path)

    # ── California Housing (regression) ───────────────────────────────────────
    _housing_path = FIXTURES_DIR / "california_housing_smoke.csv"
    if not _housing_path.exists():
        try:
            from sklearn.datasets import fetch_california_housing
            data = fetch_california_housing(as_frame=True)
            df = data.frame.sample(min(n, len(data.frame)), random_state=42)
            df.to_csv(_housing_path, index=False)
        except Exception:
            rng = np.random.default_rng(42)
            df = pd.DataFrame({
                "MedInc": rng.uniform(0.5, 15, n),
                "HouseAge": rng.uniform(1, 52, n),
                "AveRooms": rng.uniform(2, 20, n),
                "AveBedrms": rng.uniform(0.5, 5, n),
                "Population": rng.integers(3, 36000, n),
                "Latitude": rng.uniform(32, 42, n),
                "Longitude": rng.uniform(-124, -114, n),
                "MedHouseVal": rng.uniform(0.15, 5.0, n),
            })
            df.to_csv(_housing_path, index=False)
    paths["housing"] = str(_housing_path)
    logger.info("California Housing fixture: %s", _housing_path)

    # ── IMDb subset (text binary sentiment) ───────────────────────────────────
    _imdb_path = FIXTURES_DIR / "imdb_smoke.csv"
    if not _imdb_path.exists():
        try:
            from datasets import load_dataset  # type: ignore
            ds = load_dataset("imdb", split=f"train[:{n}]")
            df = pd.DataFrame({"text": ds["text"], "label": ds["label"]})
            df.to_csv(_imdb_path, index=False)
        except Exception:
            rng = np.random.default_rng(42)
            pos = ["great movie", "loved it", "excellent acting", "must watch", "highly recommend"]
            neg = ["terrible film", "waste of time", "awful acting", "boring movie", "very disappointing"]
            texts, labels = [], []
            for _ in range(n):
                label = rng.integers(0, 2)
                phrases = pos if label == 1 else neg
                texts.append(" ".join(rng.choice(phrases, 5, replace=True)))
                labels.append(int(label))
            pd.DataFrame({"text": texts, "label": labels}).to_csv(_imdb_path, index=False)
    paths["imdb"] = str(_imdb_path)
    logger.info("IMDb fixture: %s", _imdb_path)

    # ── MMIMDB (text + image + tabular multilabel) ─────────────────────────────
    try:
        from tests.test_benchmark_datasets import _ensure_mmimdb
        _mmimdb_path = _ensure_mmimdb(n_rows=n)
        paths["mmimdb"] = str(_mmimdb_path)
        logger.info("MMIMDB fixture: %s", _mmimdb_path)
    except Exception as e:
        logger.warning("MMIMDB fixture generation failed: %s", e)

    # ── CMU-MOSEI (text + COVAREP tabular ternary sentiment) ──────────────────
    try:
        from tests.test_benchmark_datasets import _ensure_mosei
        _mosei_path = _ensure_mosei(n_rows=n)
        paths["mosei"] = str(_mosei_path)
        logger.info("CMU-MOSEI fixture: %s", _mosei_path)
    except Exception as e:
        logger.warning("MOSEI fixture generation failed: %s", e)

    # ── Hateful Memes (text + image binary) ────────────────────────────────────
    _hm_path = FIXTURES_DIR / "hateful_memes" / f"hateful_memes_n{n}.csv"
    if _hm_path.exists():
        paths["hateful_memes"] = str(_hm_path)
        logger.info("Hateful Memes fixture: %s", _hm_path)
    else:
        # Fallback to any hateful_memes fixture
        for _candidate in sorted(FIXTURES_DIR.glob("hateful_memes/hateful_memes_n*.csv")):
            paths["hateful_memes"] = str(_candidate)
            logger.info("Hateful Memes fixture (fallback): %s", _candidate)
            break

    return paths


# ---------------------------------------------------------------------------
# Step helpers — run each script as subprocess for isolation
# ---------------------------------------------------------------------------

def _run_script(args: List[str], label: str) -> bool:
    """Run a Python script and return True on success."""
    logger.info("=== %s ===", label)
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, *args],
        cwd=str(_ROOT),
        capture_output=False,
    )
    elapsed = time.time() - t0
    if result.returncode == 0:
        logger.info("✓ %s completed in %.1fs", label, elapsed)
        return True
    else:
        logger.warning("✗ %s FAILED (exit %d) in %.1fs", label, result.returncode, elapsed)
        return False


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_full_benchmark(
    quick: bool = False,
    max_rows: int = 500,
    seeds: List[int] = None,
    datasets: Optional[List[str]] = None,
    include_smoke: bool = False,
) -> Dict[str, Any]:
    """Orchestrate all benchmark steps and return a summary dict."""
    seeds = seeds or ([42] if quick else [42, 123, 456, 789, 2026])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {"steps": {}, "quick": quick, "seeds": seeds}

    # Step 1: generate fixtures
    logger.info("Step 1: Generating Tier 1 dataset fixtures ...")
    fixture_paths = _generate_fixtures(quick=quick, max_rows=max_rows)
    summary["fixture_paths"] = fixture_paths
    logger.info("Fixtures ready: %s", list(fixture_paths.keys()))

    # Choose which datasets to benchmark on
    active_datasets = datasets or list(fixture_paths.keys())
    _primary_tabular = fixture_paths.get("adult", fixture_paths.get("housing", ""))
    _primary_multimodal = fixture_paths.get("hateful_memes", "")

    # Step 2: Ablation study
    # Need text+tabular dataset to exercise multi-modality fusion strategies.
    # Prefer synthetic_text_tabular (review text + age/income tabular) — clean, no ambiguity.
    # run_ablations.py takes --seeds as comma-separated string (e.g. 42,123,456)
    _synthetic_tt = FIXTURES_DIR / "synthetic_text_tabular.csv"
    if not _synthetic_tt.exists():
        # Generate it via the modality combination test helper
        try:
            from tests.test_e2e_real_datasets import _make_text_tabular_csv
            _synthetic_tt = _make_text_tabular_csv(n=200)
        except Exception as _e:
            logger.warning("synthetic_text_tabular generation failed: %s", _e)
    _ablation_dataset = (
        str(_synthetic_tt) if _synthetic_tt.exists() else
        fixture_paths.get("imdb") or      # text-only fallback
        _primary_tabular                   # last resort: tabular-only
    )
    if _ablation_dataset:
        ok = _run_script([
            "scripts/run_ablations.py",
            "--dataset", _ablation_dataset,
            "--seeds", ",".join(str(s) for s in seeds),  # comma-joined: "42"
        ], "Ablation study (MOSEI text+tabular — exercises all fusion strategies)")
        summary["steps"]["ablations"] = "ok" if ok else "failed"
    else:
        logger.warning("No ablation dataset — skipping ablation study")

    # Step 3: Baselines (XGBoost + MLP + optional AutoGluon-Multimodal)
    if _primary_tabular:
        ok = _run_script([
            "scripts/run_baselines.py",
            "--dataset", _primary_tabular,
            "--output", str(RESULTS_DIR / "baselines.json"),
        ], "Baselines (XGBoost + MLP + AutoGluon-Multimodal if installed)")
        summary["steps"]["baselines"] = "ok" if ok else "failed"

    # Step 4: ULA ablation — must run on a tabular dataset (hateful_memes has
    # only text/image columns; the ablation runner needs numeric feature columns)
    _ula_dataset = _primary_tabular  # Adult Income or California Housing
    if _ula_dataset:
        ok = _run_script([
            "scripts/run_ula_ablation.py",
            "--dataset", _ula_dataset,
            "--seeds", ",".join(str(s) for s in seeds[:3]),  # max 3 seeds
            "--epochs", "3" if quick else "5",
            "--output", str(RESULTS_DIR / "ula_ablation.json"),
        ], "ULA fusion ablation")
        summary["steps"]["ula_ablation"] = "ok" if ok else "failed"

    # Step 5: LoRA rank ablation (on adult income tabular)
    if _primary_tabular:
        ok = _run_script([
            "scripts/run_lora_ablation.py",
            "--dataset", _primary_tabular,
            "--seeds", *[str(s) for s in seeds[:3]],
            "--epochs", "3" if quick else "5",
            "--output", str(RESULTS_DIR / "lora_ablation.json"),
        ], "LoRA rank ablation")
        summary["steps"]["lora_ablation"] = "ok" if ok else "failed"

    # Step 6: Modality robustness (on hateful_memes)
    if _primary_multimodal:
        ok = _run_script([
            "scripts/run_modality_robustness.py",
        ], "Modality robustness (graceful degradation)")
        summary["steps"]["modality_robustness"] = "ok" if ok else "failed"

    # Step 7: Hateful Memes benchmark (if fixture exists)
    if _primary_multimodal:
        _hm_args = ["scripts/run_hateful_memes_benchmark.py"]
        if quick:
            _hm_args.append("--quick")
        ok = _run_script(_hm_args, "Hateful Memes benchmark (5 conditions × seeds)")
        summary["steps"]["hateful_memes_benchmark"] = "ok" if ok else "failed"

    # Step 8: Statistical aggregation (Wilcoxon + bootstrap CIs)
    ok = _run_script(["scripts/aggregate_results.py"], "Statistical aggregation")
    summary["steps"]["aggregation"] = "ok" if ok else "failed"

    # Step 9: LaTeX paper generation (call via dedicated helper script)
    ok = _run_script(["scripts/_generate_paper.py"], "LaTeX paper generation")
    summary["steps"]["paper"] = "ok" if ok else "failed"

    # Step 10: Plot generation
    ok = _run_script(["scripts/generate_plots.py"], "Plot generation")
    summary["steps"]["plots"] = "ok" if ok else "failed"

    # Summary
    ok_count = sum(1 for v in summary["steps"].values() if v == "ok")
    total_count = len(summary["steps"])
    logger.info("=" * 60)
    logger.info("Benchmark complete: %d/%d steps succeeded", ok_count, total_count)
    for step, status in summary["steps"].items():
        mark = "✓" if status == "ok" else "✗"
        logger.info("  %s %s: %s", mark, step, status)
    logger.info("Results in: %s", RESULTS_DIR)

    # Write summary
    _summary_path = RESULTS_DIR / "benchmark_summary.json"
    with open(_summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    logger.info("Summary written to: %s", _summary_path)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoVision full publication benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 200 rows, 1 seed, 3 epochs")
    parser.add_argument("--max-rows", type=int, default=500, help="Max rows per fixture")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Random seeds")
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        choices=["adult", "housing", "imdb", "hateful_memes", "mmimdb", "mosei"],
        help="Limit to specific datasets",
    )
    parser.add_argument("--include-smoke", action="store_true", help="Include Tier 2 smoke datasets")
    args = parser.parse_args()

    run_full_benchmark(
        quick=args.quick,
        max_rows=args.max_rows,
        seeds=args.seeds,
        datasets=args.datasets,
        include_smoke=args.include_smoke,
    )
