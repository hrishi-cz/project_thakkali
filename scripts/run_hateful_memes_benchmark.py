"""
AutoVision Multimodal AutoML — Hateful Memes Benchmark
=======================================================

Simulates the Hateful Memes multimodal classification task.

Since the original Hateful Memes dataset (Kiela et al., NeurIPS 2020)
requires a Facebook DLC, this uses a structurally equivalent synthetic dataset:
  - N rows with img_path (64×64 PNG), caption (meme text), label (0/1)
  - Correlation: text+image both partially correlated with label but with noise
  - Class balance: ~58:42 (not-hateful : hateful)

Conditions compared
-------------------
1. AutoVision Multimodal — text + image, attention fusion
2. AutoVision Text-Only  — caption encoder only
3. AutoVision Image-Only — image encoder only
4. TF-IDF + LR           — strong non-neural text baseline
5. Pixel MLP             — flatten + MLP image baseline

Metrics: Accuracy, F1 (macro). 3 seeds.
Results → diary/results/hateful_memes_benchmark.json
LaTeX  → diary/results/hateful_memes_table.tex

Usage
-----
    python scripts/run_hateful_memes_benchmark.py [--quick]
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

FIXTURES_DIR = ROOT / "data" / "fixtures" / "hateful_memes"
RESULTS_PATH = ROOT / "diary" / "results" / "hateful_memes_benchmark.json"
SEEDS = [42, 123, 456]

# ── 1. Synthetic data generation ─────────────────────────────────────────────

_HATEFUL_TEXTS = [
    "these people are destroying our neighborhood and causing all the problems",
    "they should go back to where they came from nobody wants them here",
    "disgusting creatures ruining everything that we worked so hard to build",
    "criminals invading our country and taking jobs that belong to real people",
    "subhuman behavior from people who do not belong in civilized society",
    "dangerous parasites corrupting the youth and spreading disease everywhere",
    "vermin flooding our borders making our beautiful country unsafe for us",
    "violent savages who cannot control their barbaric primitive instincts",
    "look at these animals destroying property and attacking innocent civilians",
    "the infestation must be stopped before they completely take over everything",
]

_BENIGN_TEXTS = [
    "adorable puppy playing in the park on a beautiful sunny afternoon",
    "delicious homemade pizza fresh from the oven with extra cheese toppings",
    "spectacular mountain view at sunrise with colorful clouds in the sky",
    "friends laughing and having fun at an outdoor summer festival together",
    "cozy reading nook with a warm cup of coffee on a rainy day",
    "baby animals learning to walk for the first time in the spring meadow",
    "amazing street art mural painted by local artists in the city center",
    "vintage car show featuring classic vehicles from the nineteen fifties era",
    "the cat knocked over the coffee mug again and looked absolutely unrepentant",
    "monday morning energy levels are exactly zero send help and caffeine please",
]

_NOISE_WORDS = ["literally", "basically", "honestly", "actually", "definitely",
                "truly", "absolutely", "incredibly", "remarkably", "surprisingly"]


def _make_text(rng: np.random.RandomState, label: int) -> str:
    """Generate text that is partially correlated with label + noise."""
    # 70% chance of using a label-correlated template; 30% noise → harder task
    if rng.rand() < 0.70:
        pool = _HATEFUL_TEXTS if label == 1 else _BENIGN_TEXTS
    else:
        pool = _BENIGN_TEXTS if label == 1 else _HATEFUL_TEXTS  # flipped → hard cases
    base = rng.choice(pool)
    # Add random noise words to increase difficulty
    for _ in range(rng.randint(1, 4)):
        base = rng.choice(_NOISE_WORDS) + " " + base
    return base.strip()


def _make_image(rng: np.random.RandomState, label: int, path: Path) -> None:
    """64×64 PNG partially correlated with label (colour bias + texture noise)."""
    from PIL import Image
    arr = (rng.rand(64, 64, 3) * 200 + 28).astype(np.uint8)
    if label == 1:
        arr[:, :, 0] = np.clip(arr[:, :, 0].astype(int) + rng.randint(30, 80), 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2].astype(int) - rng.randint(20, 50), 0, 255)
    else:
        arr[:, :, 2] = np.clip(arr[:, :, 2].astype(int) + rng.randint(30, 80), 0, 255)
        arr[:, :, 0] = np.clip(arr[:, :, 0].astype(int) - rng.randint(20, 50), 0, 255)
    # Add noise to make it harder
    noise = (rng.rand(64, 64, 3) * 60).astype(int)
    arr = np.clip(arr.astype(int) + noise - 30, 0, 255).astype(np.uint8)
    Image.fromarray(arr, "RGB").save(path)


def ensure_fixture(n_rows: int = 1000, seed: int = 42) -> Path:
    csv_path = FIXTURES_DIR / f"hateful_memes_n{n_rows}.csv"
    if csv_path.is_file():
        df = pd.read_csv(csv_path)
        if len(df) == n_rows:
            logger.info("Fixture exists: %s (%d rows)", csv_path, n_rows)
            return csv_path
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    (FIXTURES_DIR / "images").mkdir(exist_ok=True)

    rng = np.random.RandomState(seed)
    labels = (rng.rand(n_rows) > 0.58).astype(int)
    rows = []
    for i, lbl in enumerate(labels):
        img_path = FIXTURES_DIR / "images" / f"meme_{i:05d}.png"
        _make_image(rng, int(lbl), img_path)
        rows.append({
            "caption": _make_text(rng, int(lbl)),
            "img_path": str(img_path),
            "label": int(lbl),
        })
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    logger.info("Generated synthetic Hateful Memes: %d rows → %s", n_rows, csv_path)
    return csv_path


# ── 2. Schema injection helper ────────────────────────────────────────────────

def _build_schema(
    dataset_id: str,
    text_cols: List[str],
    image_cols: List[str],
    tabular_cols: List[str],
    target: str,
    modalities: List[str],
) -> dict:
    """Build a fully-specified GlobalSchema dict for inject_external_schema."""
    detected = {
        "image": [c for c in image_cols if "image" in modalities],
        "text": [c for c in text_cols if "text" in modalities],
        "tabular": [c for c in tabular_cols if "tabular" in modalities],
        "timeseries": [],
    }
    active_mods = [m for m in modalities if detected.get(m)]
    # target_profile must have a high xs3_confidence_gap so Phase 4 does NOT
    # fall back to tabular-only mode ("xs3_confidence_gap < 0.15" guard).
    target_profile = {
        "column": target,
        "score": 0.92,
        "final_score": 0.92,
        "xs3_score": 0.92,
        "confidence_gap": 0.80,      # xs3_confidence_gap → must be > 0.15
        "xs3_confidence_gap": 0.80,
        "reason": "manual_override",
        "dtype": "int64",
        "n_unique": 2,
    }
    return {
        "global_problem_type": "classification_binary",
        "global_modalities": active_mods,
        "primary_target": target,
        "fusion_ready": len(active_mods) >= 2,
        "detection_confidence": 0.92,
        "per_dataset": [{
            "dataset_id": dataset_id,
            "target_column": target,
            "problem_type": "classification_binary",
            "modalities": active_mods,
            "detected_columns": detected,
            "confidence": 0.92,
            "target_profile": target_profile,
            "candidates": [target_profile],
            "reasoning": {
                "confidence_gap": 0.80,
                "xs3_confidence_gap": 0.80,
                "selected": target_profile,
            },
            "preprocessing_hints": {},
            "selection_mode": "manual_override",
        }],
        "relatedness_report": {"n_groups": 1},
    }


# ── 3. APEX pipeline runner ───────────────────────────────────────────────────

def _filtered_csv(csv_path: str, modalities: List[str]) -> str:
    """
    Return a path to a CSV that only contains columns relevant to the given
    modalities. This prevents unrelated columns (e.g., img_path during a
    text-only run) from being silently treated as tabular features, which
    would break Phase 3 preprocessor consistency validation.
    """
    keep_cols = {"label"}
    if "text"  in modalities: keep_cols.add("caption")
    if "image" in modalities: keep_cols.add("img_path")

    df = pd.read_csv(csv_path)
    df_filtered = df[[c for c in df.columns if c in keep_cols]]

    suffix = "_".join(modalities)
    out_path = Path(csv_path).with_suffix(f".{suffix}.csv")
    df_filtered.to_csv(out_path, index=False)
    return str(out_path)


def _run_apex(
    csv_path: str,
    modalities: List[str],
    seed: int,
    label: str,
    n_epochs: int,
) -> Dict[str, Any]:
    import asyncio
    os.environ["APEX_SEED"] = str(seed)
    # Use a per-modality filtered CSV so unrelated columns don't become tabular
    csv_path = _filtered_csv(csv_path, modalities)

    from core.types import TrainingConfig, Phase
    from pipeline.training_orchestrator import TrainingOrchestrator

    # Force text-image or single-modal fusion
    if len(modalities) >= 2 and "text" in modalities and "image" in modalities:
        fusion = "structural_semantic"  # [1] for vision+language
    else:
        fusion = "concatenation"

    config = TrainingConfig(
        dataset_sources=[csv_path],
        problem_type="classification_binary",
        modalities=modalities,
        target_column="label",
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
    )
    orchestrator = TrainingOrchestrator(config)

    t0 = time.time()
    try:
        asyncio.run(orchestrator._execute_phase_1_data_ingestion())

        # Skip schema detection — inject fully-specified schema so modality
        # column assignments are unambiguous (schema detector can't handle
        # img_path + short caption + integer label without hints).
        dataset_id = list(orchestrator.dataset_registry.list_datasets())[0]
        schema = _build_schema(
            dataset_id=dataset_id,
            text_cols=["caption"] if "text" in modalities else [],
            image_cols=["img_path"] if "image" in modalities else [],
            tabular_cols=[],
            target="label",
            modalities=modalities,
        )
        orchestrator.inject_external_schema(schema, target_override="label")

        orchestrator._execute_phase_3_preprocessing()
        orchestrator._execute_phase_4_model_selection()
        orchestrator._execute_phase_5_training(hp_overrides={
            "fusion_strategy": fusion,
            "learning_rate": 1e-3,
            "epochs": n_epochs,
        })

    except Exception as exc:
        logger.error("APEX [%s] seed=%d FAILED: %s", label, seed, exc, exc_info=True)
        return {"condition": label, "seed": seed, "error": str(exc),
                "best_val_acc": float("nan"), "best_val_f1": float("nan")}

    elapsed = time.time() - t0
    p5 = orchestrator.phase_results.get(Phase.TRAINING, {})

    return {
        "condition": label,
        "seed": seed,
        "best_val_loss": float(p5.get("best_val_loss", float("nan"))),
        "best_val_acc":  float(p5.get("best_val_acc",  float("nan")) or float("nan")),
        "best_val_f1":   float(p5.get("best_val_f1",   float("nan")) or float("nan")),
        "n_trials":      int(p5.get("n_trials", 0) or 0),
        "modalities":    modalities,
        "fusion":        fusion,
        "elapsed_s":     round(elapsed, 1),
    }


# ── 4. Non-neural baselines ───────────────────────────────────────────────────

def _run_tfidf_lr(csv_path: str, seed: int) -> Dict[str, Any]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(csv_path)
    X, y = df["caption"].astype(str).values, df["label"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    Xtr_v = vec.fit_transform(Xtr)
    Xte_v = vec.transform(Xte)

    lr = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
    lr.fit(Xtr_v, ytr)
    preds = lr.predict(Xte_v)
    proba = lr.predict_proba(Xte_v)[:, 1]

    return {
        "condition": "TF-IDF + LR (text)",
        "seed": seed,
        "best_val_acc": float(accuracy_score(yte, preds)),
        "best_val_f1":  float(f1_score(yte, preds, average="macro")),
        "auroc":        float(roc_auc_score(yte, proba)),
        "modalities":   ["text"],
        "elapsed_s":    0.0,
    }


def _run_pixel_mlp(csv_path: str, seed: int, quick: bool = False) -> Dict[str, Any]:
    import torch
    import torch.nn as nn
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    from PIL import Image
    from torch.utils.data import DataLoader, TensorDataset

    np.random.seed(seed)
    torch.manual_seed(seed)

    df = pd.read_csv(csv_path)
    imgs, labels = [], []
    for _, row in df.iterrows():
        try:
            img = np.array(
                Image.open(str(row["img_path"])).convert("RGB").resize((32, 32))
            ).flatten() / 255.0
        except Exception:
            img = np.zeros(32 * 32 * 3, dtype=np.float32)
        imgs.append(img)
        labels.append(int(row["label"]))

    X = torch.FloatTensor(np.array(imgs))
    y_all = torch.LongTensor(labels)
    idx = np.arange(len(X))
    split = int(0.8 * len(idx))
    np.random.shuffle(idx)
    tr, te = idx[:split], idx[split:]

    dl_tr = DataLoader(TensorDataset(X[tr], y_all[tr]), batch_size=64, shuffle=True)
    dl_te = DataLoader(TensorDataset(X[te], y_all[te]), batch_size=128)

    model = nn.Sequential(
        nn.Linear(32 * 32 * 3, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 2),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n_ep = 5 if quick else 15
    for _ in range(n_ep):
        model.train()
        for xb, yb in dl_tr:
            opt.zero_grad(); nn.CrossEntropyLoss()(model(xb), yb).backward(); opt.step()

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in dl_te:
            all_preds.extend(model(xb).argmax(1).numpy())
            all_labels.extend(yb.numpy())

    return {
        "condition": "Pixel MLP (image)",
        "seed": seed,
        "best_val_acc": float(accuracy_score(all_labels, all_preds)),
        "best_val_f1":  float(f1_score(all_labels, all_preds, average="macro")),
        "modalities":   ["image"],
        "elapsed_s":    0.0,
    }


# ── 5. Aggregate + output ─────────────────────────────────────────────────────

def _aggregate(results: List[Dict]) -> Dict[str, Dict]:
    from collections import defaultdict
    by_cond = defaultdict(list)
    for r in results:
        if "error" not in r and not np.isnan(r.get("best_val_acc", float("nan"))):
            by_cond[r["condition"]].append(r)
    agg = {}
    for cond, runs in by_cond.items():
        accs   = [r["best_val_acc"] for r in runs]
        f1s    = [r["best_val_f1"] for r in runs]
        aurocs = [r.get("auroc", float("nan")) for r in runs]
        agg[cond] = {
            "acc_mean":   float(np.nanmean(accs)),
            "acc_std":    float(np.nanstd(accs)),
            "f1_mean":    float(np.nanmean(f1s)),
            "f1_std":     float(np.nanstd(f1s)),
            "auroc_mean": float(np.nanmean(aurocs)),
            "n_seeds":    len(runs),
            "modalities": runs[0].get("modalities", []),
        }
    return agg


_ORDER = [
    "AutoVision Multimodal (text+img)",
    "AutoVision Text-Only",
    "AutoVision Image-Only",
    "TF-IDF + LR (text)",
    "Pixel MLP (image)",
]


def _print_table(agg: Dict) -> str:
    header = f"\n{'='*72}\n  AutoVision — Hateful Memes Multimodal Benchmark\n{'='*72}"
    fmt = "  {:<34} {:>12} {:>12}  {:>8}  {}"
    rows = [header, fmt.format("Condition", "Acc (%)", "F1 (mac)", "AUROC", "Modality"), "  " + "-"*70]
    best_acc = max((v["acc_mean"] for v in agg.values()), default=0)
    for cond in _ORDER:
        if cond not in agg: continue
        r = agg[cond]
        acc_s  = f"{r['acc_mean']*100:.1f}+/-{r['acc_std']*100:.1f}"
        f1_s   = f"{r['f1_mean']:.3f}+/-{r['f1_std']:.3f}"
        au_s   = f"{r['auroc_mean']:.3f}" if not np.isnan(r.get("auroc_mean", float("nan"))) else "  N/A "
        mods   = "+".join(r["modalities"])
        marker = " << BEST" if abs(r["acc_mean"] - best_acc) < 1e-4 else ""
        rows.append(fmt.format(cond[:34], acc_s, f1_s, au_s, mods + marker))
    rows.append("=" * 72)
    table = "\n".join(rows)
    print(table)
    return table


def _latex_table(agg: Dict) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{AutoVision vs.\ Baselines on Synthetic Hateful Memes "
        r"(N\,=\,1000, 3\,seeds, mean\,$\pm$\,std)}",
        r"\label{tab:hateful-memes}",
        r"\begin{tabular}{lcccl}",
        r"\toprule",
        r"Method & Acc.\ (\%) & F1\textsubscript{mac} & AUROC & Modality \\",
        r"\midrule",
    ]
    best_acc = max((v["acc_mean"] for v in agg.values()), default=0)
    for cond in _ORDER:
        if cond not in agg: continue
        r = agg[cond]
        acc_s  = f"{r['acc_mean']*100:.1f} $\\pm$ {r['acc_std']*100:.1f}"
        f1_s   = f"{r['f1_mean']:.3f} $\\pm$ {r['f1_std']:.3f}"
        au_s   = f"{r['auroc_mean']:.3f}" if not np.isnan(r.get("auroc_mean", float("nan"))) else "---"
        mods   = "+".join(r["modalities"])
        if abs(r["acc_mean"] - best_acc) < 1e-4:
            acc_s = r"\textbf{" + acc_s + "}"
        lines.append(f"{cond} & {acc_s} & {f1_s} & {au_s} & {mods} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ── 6. Main ───────────────────────────────────────────────────────────────────

def main(quick: bool = False) -> dict:
    n_rows  = 200 if quick else 1000
    n_ep    = 3   if quick else 5
    seeds   = [42] if quick else SEEDS

    logger.info("=== AutoVision Hateful Memes Benchmark  quick=%s ===", quick)
    csv_path = str(ensure_fixture(n_rows=n_rows))

    all_results: List[Dict] = []

    logger.info("\n[1/5] AutoVision Multimodal (text + image, structural_semantic fusion)")
    for seed in seeds:
        r = _run_apex(csv_path, ["text", "image"], seed,
                      "AutoVision Multimodal (text+img)", n_ep)
        all_results.append(r)
        logger.info("  seed=%d  acc=%.3f  f1=%.3f  elapsed=%.0fs",
                    seed, r.get("best_val_acc", 0), r.get("best_val_f1", 0), r.get("elapsed_s", 0))

    logger.info("\n[2/5] AutoVision Text-Only")
    for seed in seeds:
        r = _run_apex(csv_path, ["text"], seed, "AutoVision Text-Only", n_ep)
        all_results.append(r)
        logger.info("  seed=%d  acc=%.3f  f1=%.3f", seed,
                    r.get("best_val_acc", 0), r.get("best_val_f1", 0))

    logger.info("\n[3/5] AutoVision Image-Only")
    for seed in seeds:
        r = _run_apex(csv_path, ["image"], seed, "AutoVision Image-Only", n_ep)
        all_results.append(r)
        logger.info("  seed=%d  acc=%.3f  f1=%.3f", seed,
                    r.get("best_val_acc", 0), r.get("best_val_f1", 0))

    logger.info("\n[4/5] TF-IDF + Logistic Regression (text baseline)")
    for seed in seeds:
        r = _run_tfidf_lr(csv_path, seed)
        all_results.append(r)
        logger.info("  seed=%d  acc=%.3f  f1=%.3f  auroc=%.3f", seed,
                    r["best_val_acc"], r["best_val_f1"], r.get("auroc", 0))

    logger.info("\n[5/5] Pixel MLP (image baseline)")
    for seed in seeds:
        r = _run_pixel_mlp(csv_path, seed, quick=quick)
        all_results.append(r)
        logger.info("  seed=%d  acc=%.3f  f1=%.3f", seed,
                    r["best_val_acc"], r["best_val_f1"])

    agg   = _aggregate(all_results)
    table = _print_table(agg)
    latex = _latex_table(agg)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "benchmark": "hateful_memes_synthetic",
        "n_rows": n_rows, "n_seeds": len(seeds), "n_epochs": n_ep,
        "quick_mode": quick,
        "raw_results": all_results,
        "aggregated": agg,
        "latex_table": latex,
        "ascii_table": table,
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2, default=str))
    (RESULTS_PATH.parent / "hateful_memes_table.tex").write_text(latex)
    logger.info("\n✅  Results → %s", RESULTS_PATH)
    print("\n" + latex)
    return output


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="200 rows, 1 seed, 3 epochs")
    main(quick=ap.parse_args().quick)
