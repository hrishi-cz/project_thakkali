"""
Stage 0b — LLM Reasoning Augmentation
======================================
Generates semantic reasoning text for each sample using GPT-4o-mini (or any
configured LLM).  Requires Stage 0a (`generate_image_captions.py`) when image
columns are present — the blip_caption column is used in the prompt.

Output adds a `reasoning_text` column.  No confidence/soft labels are generated
(benchmark contamination risk — see plan for rationale).

Usage
-----
    # After generate_image_captions.py:
    python scripts/generate_reasoning_augmentation.py \
        --input data/dataset_captioned.csv \
        --caption-col caption --blip-col blip_caption \
        --output data/dataset_with_reasoning.csv

    # Dry-run:
    python scripts/generate_reasoning_augmentation.py \
        --input data/dataset_captioned.csv --dry-run 5

    # Text-only (no image column):
    python scripts/generate_reasoning_augmentation.py \
        --input data/dataset.csv --caption-col caption --no-image

    # With tabular context:
    python scripts/generate_reasoning_augmentation.py \
        --input data/dataset_captioned.csv \
        --tabular-cols age,income,category

Env vars
--------
    APEX_OPENAI_API_KEY       OpenAI API key (falls back to OPENAI_API_KEY)
    APEX_REASONING_MODEL      Model to use (default: gpt-4o-mini)
    APEX_REASONING_PROMPT_FILE  YAML file with domain-specific system prompt override
    APEX_SKIP_REASONING=1     Skip this stage entirely (pass-through with empty reasoning_text)
    APEX_REASONING_MAX_TOKENS Maximum tokens for reasoning output (default: 200)
    APEX_REASONING_BATCH      API calls per concurrent batch (default: 5)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_TOKENS = 200

SYSTEM_PROMPT_DEFAULT = (
    "You are a content moderation expert. Given information about a meme or post, "
    "provide semantic context useful for classification. Output exactly:\n"
    "REASONING: [2-3 sentences: identify target group if any, cultural/ironic reference "
    "invoked, and what juxtaposition between image and text creates harm, if applicable. "
    "Be concise and specific.]"
)

USER_TEMPLATE_FULL = (
    'Caption: "{caption}"\n'
    'Image shows: "{blip_caption}"{tabular_part}'
)
USER_TEMPLATE_TEXT_ONLY = (
    'Caption: "{caption}"{tabular_part}'
)
USER_TEMPLATE_IMAGE_ONLY = (
    'Image shows: "{blip_caption}"{tabular_part}'
)


def _load_system_prompt() -> str:
    prompt_file = os.environ.get("APEX_REASONING_PROMPT_FILE")
    if prompt_file and Path(prompt_file).exists():
        try:
            import yaml
            with open(prompt_file, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return str(data.get("system_prompt", SYSTEM_PROMPT_DEFAULT))
        except Exception as e:
            logger.warning("Failed to load prompt file %s: %s. Using default.", prompt_file, e)
    return SYSTEM_PROMPT_DEFAULT


def _serialize_tabular(row: Dict[str, Any], tabular_cols: List[str]) -> str:
    if not tabular_cols:
        return ""
    parts = []
    for col in tabular_cols[:8]:  # cap at 8 features to stay within token budget
        val = row.get(col)
        if val is not None and str(val) not in ("nan", "None", ""):
            parts.append(f"{col}: {val}")
    return ("\nAdditional data: " + ", ".join(parts)) if parts else ""


def _build_user_message(
    row: Dict[str, Any],
    caption_col: Optional[str],
    blip_col: Optional[str],
    tabular_cols: List[str],
) -> str:
    caption = str(row.get(caption_col, "")).strip() if caption_col else ""
    blip = str(row.get(blip_col, "")).strip() if blip_col else ""
    tabular_part = _serialize_tabular(row, tabular_cols)

    has_caption = bool(caption)
    has_blip = bool(blip)

    if has_caption and has_blip:
        return USER_TEMPLATE_FULL.format(
            caption=caption, blip_caption=blip, tabular_part=tabular_part
        )
    elif has_caption:
        return USER_TEMPLATE_TEXT_ONLY.format(caption=caption, tabular_part=tabular_part)
    elif has_blip:
        return USER_TEMPLATE_IMAGE_ONLY.format(blip_caption=blip, tabular_part=tabular_part)
    else:
        # Nothing to reason about
        return ""


def _call_api(
    client: Any,
    system_prompt: str,
    user_message: str,
    model: str,
    max_tokens: int,
    retries: int = 3,
) -> str:
    if not user_message.strip():
        return ""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            text = response.choices[0].message.content or ""
            # Strip REASONING: prefix
            if text.startswith("REASONING:"):
                text = text[len("REASONING:"):].strip()
            return text[:450]  # hard cap at 450 chars
        except Exception as e:
            logger.warning("API call failed (attempt %d/%d): %s", attempt + 1, retries, e)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return ""


def run(
    input_csv: str,
    caption_col: Optional[str],
    blip_col: Optional[str],
    tabular_cols: List[str],
    output_csv: str,
    dry_run: Optional[int],
    skip_if_exists: bool,
    no_image: bool,
) -> None:
    import pandas as pd

    out_path = Path(output_csv)
    if skip_if_exists and out_path.exists():
        logger.info("Output already exists at %s — skipping.", out_path)
        return

    if os.environ.get("APEX_SKIP_REASONING", "0") == "1":
        logger.info("APEX_SKIP_REASONING=1 — copying input with empty reasoning_text.")
        df = pd.read_csv(input_csv)
        df["reasoning_text"] = ""
        df.to_csv(output_csv, index=False)
        return

    api_key = os.environ.get("APEX_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning(
            "No API key found (APEX_OPENAI_API_KEY or OPENAI_API_KEY). "
            "Set APEX_SKIP_REASONING=1 to skip, or provide a key. "
            "Writing empty reasoning_text."
        )
        df = pd.read_csv(input_csv)
        if dry_run:
            df = df.head(dry_run)
        df["reasoning_text"] = ""
        df.to_csv(output_csv, index=False)
        return

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except ImportError:
        logger.error("openai package not installed. Run: pip install openai")
        sys.exit(1)

    df = pd.read_csv(input_csv)
    if dry_run is not None:
        df = df.head(dry_run)
        logger.info("Dry-run: processing first %d rows.", dry_run)

    model = os.environ.get("APEX_REASONING_MODEL", DEFAULT_MODEL)
    max_tokens = int(os.environ.get("APEX_REASONING_MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))
    system_prompt = _load_system_prompt()

    # Resolve blip_col
    effective_blip_col = None if no_image else blip_col
    if effective_blip_col and effective_blip_col not in df.columns:
        logger.warning(
            "blip_caption column %r not found — run generate_image_captions.py first. "
            "Falling back to text-only reasoning.",
            effective_blip_col,
        )
        effective_blip_col = None

    logger.info(
        "Generating reasoning with model=%s, max_tokens=%d, %d rows",
        model, max_tokens, len(df),
    )

    reasoning_texts: List[str] = []
    n = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        msg = _build_user_message(row, caption_col, effective_blip_col, tabular_cols)
        text = _call_api(client, system_prompt, msg, model, max_tokens)
        reasoning_texts.append(text)

        if (i + 1) % 50 == 0 or (i + 1) == n:
            logger.info("  Processed %d/%d", i + 1, n)

        # Small delay to avoid rate limits
        if (i + 1) % 20 == 0:
            time.sleep(0.5)

    df["reasoning_text"] = reasoning_texts
    df.to_csv(output_csv, index=False)

    empty = sum(1 for t in reasoning_texts if not t.strip())
    avg_len = (
        sum(len(t) for t in reasoning_texts if t.strip()) / max(1, n - empty)
    )
    logger.info(
        "Done. %d/%d successful. Avg reasoning length: %.0f chars.",
        n - empty, n, avg_len,
    )
    logger.info("Output saved to %s", output_csv)

    # Show 2 sample outputs for dry-run validation
    if dry_run:
        for i, (idx, row) in enumerate(df.iterrows()):
            if i >= 2:
                break
            caption = row.get(caption_col or "caption", "")
            reasoning = reasoning_texts[i]
            logger.info("\n--- Sample %d ---\nCaption: %s\nREASONING: %s", i, caption, reasoning)


def main() -> None:
    parser = argparse.ArgumentParser(description="GPT-4o-mini reasoning augmentation for APEX datasets")
    parser.add_argument("--input",          required=True, help="Input CSV (with blip_caption if images present)")
    parser.add_argument("--caption-col",    default="caption", help="Text caption column name")
    parser.add_argument("--blip-col",       default="blip_caption", help="BLIP image caption column name")
    parser.add_argument("--tabular-cols",   default="", help="Comma-separated tabular column names to include")
    parser.add_argument("--output",         default=None, help="Output CSV path")
    parser.add_argument("--dry-run",        type=int, default=None, metavar="N")
    parser.add_argument("--no-image",       action="store_true", help="Skip image/BLIP context even if available")
    parser.add_argument("--skip-if-exists", action="store_true")
    args = parser.parse_args()

    output = args.output or str(Path(args.input).with_suffix("")) + "_reasoned.csv"
    tabular_cols = [c.strip() for c in args.tabular_cols.split(",") if c.strip()]

    run(
        input_csv=args.input,
        caption_col=args.caption_col if args.caption_col else None,
        blip_col=args.blip_col,
        tabular_cols=tabular_cols,
        output_csv=output,
        dry_run=args.dry_run,
        skip_if_exists=args.skip_if_exists,
        no_image=args.no_image,
    )


if __name__ == "__main__":
    main()
