"""
Stage 0a — BLIP Image Captioning
=================================
Generates descriptive captions for images in a dataset CSV.
Output adds a `blip_caption` column and saves to a new CSV.

MUST run before generate_reasoning_augmentation.py.

Usage
-----
    python scripts/generate_image_captions.py --input data/dataset.csv \
        --image-col img_path --output data/dataset_captioned.csv

    # Dry-run (3 samples only):
    python scripts/generate_image_captions.py --input data/dataset.csv --dry-run 3

    # Skip if already done:
    python scripts/generate_image_captions.py --input data/dataset.csv \
        --skip-if-exists

Env vars
--------
    APEX_SKIP_BLIP=1   Skip this stage entirely (pass-through with empty blip_caption)
    APEX_BLIP_MODEL    HuggingFace model ID (default: Salesforce/blip-image-captioning-large)
    APEX_BLIP_BATCH    Batch size for BLIP inference (default: 16)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

BLIP_MODEL_DEFAULT = "Salesforce/blip-image-captioning-large"


def _caption_batch(paths: List[str], captioner) -> List[str]:
    from PIL import Image, UnidentifiedImageError

    images = []
    valid_idx = []
    for i, p in enumerate(paths):
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
            valid_idx.append(i)
        except (FileNotFoundError, UnidentifiedImageError, Exception) as e:
            logger.warning("Cannot open image %s: %s", p, e)

    if not images:
        return [""] * len(paths)

    try:
        results = captioner(images, max_new_tokens=64)
        captions_valid = [r[0]["generated_text"] if r else "" for r in results]
    except Exception as e:
        logger.warning("BLIP inference failed for batch: %s", e)
        captions_valid = [""] * len(images)

    out = [""] * len(paths)
    for vi, cap in zip(valid_idx, captions_valid):
        out[vi] = cap
    return out


def run(
    input_csv: str,
    image_col: str,
    output_csv: str,
    dry_run: Optional[int],
    batch_size: int,
    skip_if_exists: bool,
) -> None:
    import pandas as pd

    out_path = Path(output_csv)
    if skip_if_exists and out_path.exists():
        logger.info("Output already exists at %s — skipping.", out_path)
        return

    if os.environ.get("APEX_SKIP_BLIP", "0") == "1":
        logger.info("APEX_SKIP_BLIP=1 — copying input to output with empty blip_caption.")
        df = pd.read_csv(input_csv)
        df["blip_caption"] = ""
        df.to_csv(output_csv, index=False)
        return

    df = pd.read_csv(input_csv)

    if image_col not in df.columns:
        logger.error("Column %r not found. Available: %s", image_col, list(df.columns))
        sys.exit(1)

    if dry_run is not None:
        df = df.head(dry_run)
        logger.info("Dry-run: processing first %d rows.", dry_run)

    model_id = os.environ.get("APEX_BLIP_MODEL", BLIP_MODEL_DEFAULT)
    logger.info("Loading BLIP model: %s", model_id)

    import torch
    from transformers import pipeline as hf_pipeline

    device = 0 if torch.cuda.is_available() else -1
    logger.info("BLIP running on %s", "GPU" if device == 0 else "CPU")

    captioner = hf_pipeline(
        "image-to-text",
        model=model_id,
        device=device,
    )

    captions: List[str] = []
    n = len(df)
    input_dir = Path(input_csv).parent

    for start in range(0, n, batch_size):
        batch_rows = df.iloc[start : start + batch_size]
        paths = []
        for _, row in batch_rows.iterrows():
            p = str(row[image_col])
            # Resolve relative paths against input CSV directory
            if not Path(p).is_absolute():
                p = str(input_dir / p)
            paths.append(p)

        batch_caps = _caption_batch(paths, captioner)
        captions.extend(batch_caps)
        logger.info("  Captioned %d/%d samples", min(start + batch_size, n), n)

    df["blip_caption"] = captions
    df.to_csv(output_csv, index=False)
    logger.info("Saved %d rows with blip_caption to %s", len(df), output_csv)

    # Stats
    empty = sum(1 for c in captions if not c.strip())
    logger.info("Stats: %d successful, %d failed/empty", n - empty, empty)


def main() -> None:
    parser = argparse.ArgumentParser(description="BLIP image captioning for APEX datasets")
    parser.add_argument("--input",          required=True, help="Input CSV path")
    parser.add_argument("--image-col",      default="img_path", help="Image path column name")
    parser.add_argument("--output",         default=None, help="Output CSV (default: input + _captioned.csv)")
    parser.add_argument("--dry-run",        type=int, default=None, metavar="N", help="Process first N rows only")
    parser.add_argument("--batch-size",     type=int, default=int(os.environ.get("APEX_BLIP_BATCH", "16")))
    parser.add_argument("--skip-if-exists", action="store_true", help="Skip if output already exists")
    args = parser.parse_args()

    output = args.output or str(Path(args.input).with_suffix("")) + "_captioned.csv"
    run(
        input_csv=args.input,
        image_col=args.image_col,
        output_csv=output,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
        skip_if_exists=args.skip_if_exists,
    )


if __name__ == "__main__":
    main()
