"""
Lazy data loader for memory-safe multimodal dataset handling.

Tabular:  polars.scan_csv / scan_parquet  -> LazyFrame  (zero data in RAM)
          dask.dataframe                  -> DataFrame   (fallback if polars absent)
Images:   LazyImageDataset (PyTorch)      -> only paths in RAM; pixels read
          strictly inside __getitem__ via PIL
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy-import guard: polars preferred, dask as fallback
# ---------------------------------------------------------------------------
try:
    import polars as pl
    _POLARS_AVAILABLE = True
except ImportError:
    _POLARS_AVAILABLE = False

if not _POLARS_AVAILABLE:
    try:
        import dask.dataframe as dd
        _DASK_AVAILABLE = True
    except ImportError:
        _DASK_AVAILABLE = False
else:
    _DASK_AVAILABLE = False

import torch
from torch.utils.data import Dataset
from PIL import Image


def detect_image_structure(cache_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Backward-compatible helper for schema detection of image directories.

    Returns a compact summary with at least a ``type`` key:
      - ``classification`` when class-folder patterns are detected
      - ``unsupervised`` otherwise
    """
    root = Path(cache_path)
    if not root.exists():
        return {
            "type": "unsupervised",
            "n_images": 0,
            "n_classes": 0,
            "classes": [],
            "class_distribution": {},
        }

    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}
    image_paths = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in image_suffixes
    ]

    if not image_paths:
        return {
            "type": "unsupervised",
            "n_images": 0,
            "n_classes": 0,
            "classes": [],
            "class_distribution": {},
        }

    # Infer class labels from first path segment under root.
    class_counts: Counter[str] = Counter()
    for img_path in image_paths:
        try:
            rel_parts = img_path.relative_to(root).parts
        except Exception:
            rel_parts = img_path.parts

        if not rel_parts:
            continue

        label = rel_parts[0]
        # Common split roots: train/<class>/... or val/<class>/...
        if label.lower() in {"train", "training", "val", "valid", "validation", "test"} and len(rel_parts) >= 2:
            label = rel_parts[1]

        if label and label not in {".", ".."}:
            class_counts[str(label)] += 1

    if len(class_counts) >= 2:
        return {
            "type": "classification",
            "n_images": len(image_paths),
            "n_classes": len(class_counts),
            "classes": sorted(class_counts.keys()),
            "class_distribution": dict(class_counts),
        }

    return {
        "type": "unsupervised",
        "n_images": len(image_paths),
        "n_classes": len(class_counts),
        "classes": sorted(class_counts.keys()),
        "class_distribution": dict(class_counts),
    }


# ---------------------------------------------------------------------------
# Lazy image dataset
# ---------------------------------------------------------------------------

class LazyImageDataset(Dataset):
    """
    PyTorch Dataset that stores ONLY file paths in memory.

    The actual image bytes are read from disk strictly inside ``__getitem__``.
    No pixel data is held in RAM during ``__init__`` or between ``__getitem__``
    calls.

    Args:
        image_paths: Sequence of absolute or relative paths to image files.
        transform:   Optional torchvision transform applied after PIL open.
    """

    def __init__(
        self,
        image_paths: List[str],
        transform: Optional[Any] = None,
    ) -> None:
        self._paths: List[str] = image_paths
        self._transform = transform

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self._paths[idx]
        # Image bytes are read here – not before, not after.
        # If the file is missing (e.g. Hateful Memes without downloaded images),
        # return a black RGB placeholder so training degrades gracefully instead
        # of crashing with FileNotFoundError mid-epoch.
        try:
            image = Image.open(path).convert("RGB")
        except (FileNotFoundError, OSError, Exception):
            # Determine size from transform if possible, otherwise default 224×224
            _size = (224, 224)
            if self._transform is not None:
                try:
                    import torchvision.transforms as _tv
                    for _t in getattr(self._transform, "transforms", []):
                        if hasattr(_t, "size"):
                            _s = _t.size
                            _size = (_s, _s) if isinstance(_s, int) else tuple(_s[:2])
                            break
                except Exception:
                    pass
            image = Image.new("RGB", _size, color=0)
            logger.warning("LazyImageDataset: path not found, using black placeholder: %s", path)
        if self._transform is not None:
            image = self._transform(image)
        return {"image": image, "path": path}

    def __repr__(self) -> str:
        return f"LazyImageDataset(n_paths={len(self._paths)})"


# ---------------------------------------------------------------------------
# Internal helper: lazy tabular scan
# ---------------------------------------------------------------------------

# Union type for the lazy tabular references this module can return
LazyTabular = Union["pl.LazyFrame", "dd.DataFrame"]  # type: ignore[name-defined]


def _lazy_scan(filepath: Path) -> LazyTabular:
    """
    Return a fully lazy tabular reference – never materialises data.

    Preference order:
      1. Polars LazyFrame  (scan_parquet / scan_csv)
      2. Dask DataFrame    (read_parquet / read_csv)
    """
    if _POLARS_AVAILABLE:
        if filepath.suffix == ".parquet":
            return pl.scan_parquet(str(filepath))
        return pl.scan_csv(str(filepath))

    if _DASK_AVAILABLE:
        if filepath.suffix == ".parquet":
            return dd.read_parquet(str(filepath))  # type: ignore[union-attr]
        return dd.read_csv(str(filepath))  # type: ignore[union-attr]

    raise ImportError(
        "Neither polars nor dask is installed. "
        "Install one with: pip install polars  OR  pip install dask[dataframe]"
    )


# ---------------------------------------------------------------------------
# Public DataLoader class
# ---------------------------------------------------------------------------

class DataLoader:
    """
    Universal lazy data loader for multimodal datasets.

    All tabular methods return a Polars ``LazyFrame`` or Dask ``DataFrame`` –
    no rows are loaded into RAM until an explicit ``.collect()`` / ``.compute()``
    call downstream.

    Images are wrapped in a ``LazyImageDataset`` (PyTorch ``Dataset``); pixels
    are read on demand inside ``__getitem__``.
    """

    # ------------------------------------------------------------------
    # Tabular lazy loaders
    # ------------------------------------------------------------------

    def load_csv(self, filepath: str) -> LazyTabular:
        """Lazily scan a CSV file.  No rows are read into RAM."""
        return _lazy_scan(Path(filepath))

    def load_parquet(self, filepath: str) -> LazyTabular:
        """Lazily scan a Parquet file.  No rows are read into RAM."""
        return _lazy_scan(Path(filepath))

    # ------------------------------------------------------------------
    # Tiny config-style loader (not lazy – these are never large)
    # ------------------------------------------------------------------

    def load_json(self, filepath: str) -> Dict[str, Any]:
        """Load a JSON file fully into memory (config/metadata use-case only)."""
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Image lazy loader
    # ------------------------------------------------------------------

    def load_images(
        self,
        image_paths: List[str],
        transform: Optional[Any] = None,
    ) -> LazyImageDataset:
        """
        Return a lazy PyTorch Dataset.

        Only the file *paths* are held in memory; pixels are read from disk
        strictly inside ``__getitem__`` on each access.
        """
        return LazyImageDataset(image_paths, transform=transform)

    # ------------------------------------------------------------------
    # Cache-directory auto-detect
    # ------------------------------------------------------------------

    def load_cached(
        self,
        cache_path: Path,
    ) -> Optional[Union[LazyTabular, LazyImageDataset]]:
        """
        Return a lazy reference to whatever data lives in *cache_path*.

        Detection priority:
          1. ``*.parquet`` – tabular lazy scan
          2. ``*.csv``     – tabular lazy scan
          3. ``*.jpg / *.jpeg / *.png`` (recursive) – LazyImageDataset

        Returns ``None`` if the directory is empty or contains no recognised
        file types.
        """
        if not cache_path.exists():
            return None

        # Parquet preferred — check top-level first, then recursive (nested zip archives)
        parquet_files = sorted(cache_path.glob("*.parquet")) or sorted(cache_path.rglob("*.parquet"))
        if parquet_files:
            return _lazy_scan(parquet_files[0])

        # CSV fallback — recursive so nested-directory zips (PTB-XL, Kaggle multi-folder) are found
        csv_files = sorted(cache_path.glob("*.csv")) or sorted(cache_path.rglob("*.csv"))
        if csv_files:
            return _lazy_scan(csv_files[0])

        # JSONL — convert to CSV (covers Hateful Memes, HuggingFace exports, etc.)
        # Prefer train split; fall back to any .jsonl file.
        jsonl_files = sorted(cache_path.rglob("*.jsonl")) + sorted(cache_path.rglob("*.json"))
        if jsonl_files:
            import pandas as _pd
            # Prefer train split over dev/test
            _preferred = [f for f in jsonl_files if "train" in f.name.lower()]
            _src = _preferred[0] if _preferred else jsonl_files[0]
            try:
                df = _pd.read_json(_src, lines=True)
            except Exception:
                df = _pd.read_json(_src)
            # Resolve relative image paths to absolute paths so downstream
            # image validators and preprocessors can find the files.
            # Hateful Memes uses `img/42953.png` relative to the JSONL file's
            # directory — use _src.parent, not cache_path, so nested archives
            # (e.g. cache/facebook-hateful-meme-dataset/train.jsonl) resolve
            # to cache/facebook-hateful-meme-dataset/img/42953.png.
            _jsonl_dir = _src.parent
            for _col in df.columns:
                if df[_col].dtype == object:
                    _sample = df[_col].dropna().astype(str).head(3)
                    _is_img = _sample.apply(lambda v: (
                        "/" in v or "\\" in v) and any(v.lower().endswith(ext)
                        for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif")
                    ))
                    if _is_img.any():
                        def _resolve_img_path(p, jd=_jsonl_dir, cd=cache_path):
                            if not isinstance(p, str) or Path(p).is_absolute():
                                return p
                            # Prefer JSONL-relative resolution (correct for nested archives)
                            _candidate = (jd / p).resolve()
                            if _candidate.exists():
                                return str(_candidate)
                            # Fallback to cache_path-relative (flat archive layout)
                            return str((cd / p).resolve())
                        df[_col] = df[_col].apply(_resolve_img_path)
                        break
            csv_path = cache_path / (_src.stem + ".csv")
            df.to_csv(csv_path, index=False)
            logger.info("JSONL converted to CSV: %s -> %s (%d rows)", _src, csv_path, len(df))
            return _lazy_scan(csv_path)

        # Excel — convert to CSV so the rest of the pipeline sees a uniform format
        xlsx_files = sorted(cache_path.glob("*.xlsx")) + sorted(cache_path.glob("*.xls"))
        if xlsx_files:
            import pandas as _pd
            try:
                df = _pd.read_excel(xlsx_files[0], engine="openpyxl")
            except Exception:
                df = _pd.read_excel(xlsx_files[0])
            csv_path = xlsx_files[0].with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            return _lazy_scan(csv_path)

        # Image directory – use itertools.islice to cap memory usage.
        # Only the first 500_000 paths are collected; for larger datasets
        # callers should use load_images() with an explicit manifest.
        import itertools
        image_iter = itertools.chain(
            cache_path.rglob("*.jpg"),
            cache_path.rglob("*.jpeg"),
            cache_path.rglob("*.png"),
        )
        image_files = list(itertools.islice(image_iter, 500_000))
        if image_files:
            return LazyImageDataset([str(p) for p in image_files])

        return None

    # ------------------------------------------------------------------
    # Pass-through merge
    # ------------------------------------------------------------------

    def merge_datasets(
        self,
        datasets: Dict[str, Union[LazyTabular, LazyImageDataset]],
    ) -> Dict[str, Union[LazyTabular, LazyImageDataset]]:
        """
        Pass-through: downstream phases handle modality-specific merging.
        Kept for API compatibility with callers that call this method.
        """
        return datasets
