"""
Comprehensive Training Orchestrator - Coordinates all 7 phases of ML pipeline.

This module is the execution-phase orchestrator (preprocessing, selection,
training, drift, registry). Metadata-centric context orchestration is handled
by `core/orchestrator.py`.

Workflow:
Phase 1: Data Ingestion - Load, validate, and cache datasets from multiple sources
Phase 2: Schema Detection - Detect columns, infer problem type, identify modalities
Phase 3: Preprocessing - Apply modality-specific preprocessing (images, text, tabular)
Phase 4: Model Selection - Auto-select models and hyperparameters based on data/GPU
Phase 5: Training - Execute GPU training loop with safety mechanisms
Phase 6: Drift Detection - Monitor performance and detect data drift
Phase 7: Model Registry - Store models, versioning, and deployment tracking
"""

import asyncio
import logging
import os
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import json
import hashlib

import torch
import numpy as np
import pandas as pd

from dataclasses import asdict

from data_ingestion.ingestion_manager import DataIngestionManager
from data_ingestion.schema import GlobalSchema
from data_ingestion.schema_detector import MultiDatasetSchemaDetector
from pipeline.dataset_manager import DatasetManager
from preprocessing.image_preprocessor import ImagePreprocessor
from preprocessing.text_preprocessor import TextPreprocessor
from preprocessing.adaptive_engine import AdaptivePreprocessingEngine
from preprocessing.tabular_preprocessor import TabularPreprocessor
from preprocessing.preprocessing_planner import PreprocessingPlanner
from preprocessing.validator import (
    PreprocessingValidationError,
    PreprocessingValidator,
    validate_preprocessor_consistency,
)
from automl.optuna_adaptive import AdaptiveOptunaController
from pipeline.embedding_cache import EmbeddingCache
from pipeline.representation_layer import RepresentationLayer
from pipeline.evaluation import EvaluationAdapter
from pipeline.drift_adapter import DriftAdapter
from pipeline.research_metrics import ResearchMetrics
from pipeline.state import PipelineState
from pipeline.xai_engine import generate_xai_artifacts
from pipeline.calibration import ProbabilityCalibrator
from config.paths import MODEL_REGISTRY_DIR
from core.execution_context import DatasetProfile
from core.context_enforcer import (
    ContextValidationError,
    ContextValidator,
    ensure_session_context,
)
from core.types import Phase, TrainingConfig, ModelSelectionResult, TrainingMetrics


# Configure logging
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reproducibility — seed all RNGs at import time (overridable via APEX_SEED)
# ---------------------------------------------------------------------------
import pytorch_lightning as _pl
_APEX_SEED = int(os.getenv("APEX_SEED", "42"))
_pl.seed_everything(_APEX_SEED, workers=True)


def _canonical_fusion_strategy(value: Any) -> str:
    """Normalize fusion identifiers to stable metadata/API values."""
    raw = str(value or "").strip().lower().replace("-", "_")
    raw = raw.replace(" ", "_")
    aliases = {
        "concat": "concatenation",
        "concatenationfusion": "concatenation",
        "attentionfusion": "attention",
        "unifiedlatentfusion": "ula",
        "unified_latent": "ula",
        "unified_latent_alignment": "ula",
        "omnimodal": "ula",
        "gated_fusion": "gated",
        "gatedfusion": "gated",
        "fusemoe": "fusemoe",
    }
    return aliases.get(raw, raw or "concatenation")


def _snapshot_embedding_caches(*datasets: Any) -> List[Tuple[Any, str, Any]]:
    """Capture dataset embedding-cache attributes before a trial mutates them."""
    snapshot: List[Tuple[Any, str, Any]] = []
    seen: set[int] = set()
    for dataset in datasets:
        if dataset is None or id(dataset) in seen:
            continue
        seen.add(id(dataset))
        for attr in ("_precomputed_text", "_precomputed_image"):
            if hasattr(dataset, attr):
                snapshot.append((dataset, attr, getattr(dataset, attr)))
    return snapshot


def _clear_embedding_caches(*datasets: Any) -> None:
    """Force raw text/image encoder execution for adapter-training trials."""
    seen: set[int] = set()
    for dataset in datasets:
        if dataset is None or id(dataset) in seen:
            continue
        seen.add(id(dataset))
        for attr in ("_precomputed_text", "_precomputed_image"):
            if hasattr(dataset, attr):
                setattr(dataset, attr, None)


def _restore_embedding_caches(snapshot: List[Tuple[Any, str, Any]]) -> None:
    """Restore dataset embedding caches after a trial completes."""
    for dataset, attr, value in snapshot:
        try:
            setattr(dataset, attr, value)
        except Exception:
            pass


def _phase7_fusion_payload(phase5_training: Dict[str, Any]) -> Dict[str, Any]:
    """Build canonical Phase 7 fusion metadata from Phase 5 outputs."""
    phase5_training = dict(phase5_training or {})
    fusion_summary = dict(phase5_training.get("fusion_summary", {}) or {})
    strategy = _canonical_fusion_strategy(
        phase5_training.get("fusion_strategy")
        or fusion_summary.get("strategy")
        or fusion_summary.get("fusion_type")
        or "concatenation"
    )
    return {
        "strategy": strategy,
        "summary": fusion_summary,
        "auxiliary_loss_weights": dict(phase5_training.get("fusion_aux_weights", {}) or {}),
        "alignment_summary": dict(phase5_training.get("alignment_summary", {}) or {}),
    }


# ---------------------------------------------------------------------------
# Unified PyTorch Dataset produced by Phase 3
# ---------------------------------------------------------------------------

class MultimodalPyTorchDataset(torch.utils.data.Dataset):
    """
    Unified PyTorch Dataset that applies modality-specific preprocessors
    in ``__getitem__``.

    All heavy transformations happen lazily on demand — the full dataset is
    never materialised at once.

    Parameters
    ----------
    df : pd.DataFrame
        Fully materialised feature frame (target column excluded).
    targets : torch.Tensor
        1-D target tensor (``torch.long`` for classification,
        ``torch.float32`` for regression).
    schema_info : dict
        Output of Phase 2 (asdict-serialised ``GlobalSchema``).
    tabular_preprocessor : TabularPreprocessor | None
        Fitted tabular transformer.  ``None`` when no tabular columns exist.
    text_preprocessor : TextPreprocessor | None
        Callable BERT tokeniser.  ``None`` when no text columns exist.
    image_preprocessor : ImagePreprocessor | None
        Callable torchvision pipeline.  ``None`` when no image columns exist.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        targets: torch.Tensor,
        schema_info: dict,
        tabular_preprocessor=None,
        text_preprocessor=None,
        image_preprocessor=None,
        apply_augmentation: bool = False,
        precomputed_text_embeddings: Optional[torch.Tensor] = None,
        precomputed_image_embeddings: Optional[torch.Tensor] = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.targets = targets
        self.schema_info = schema_info
        self.tabular_preprocessor = tabular_preprocessor
        self.text_preprocessor = text_preprocessor
        self.image_preprocessor = image_preprocessor
        # When True, image_preprocessor.augment() is applied before the
        # standard Resize+Normalize pipeline (training split only).
        self.apply_augmentation: bool = apply_augmentation
        # Pre-computed frozen encoder embeddings (set after JIT selection)
        self._precomputed_text: Optional[torch.Tensor] = precomputed_text_embeddings
        self._precomputed_image: Optional[torch.Tensor] = precomputed_image_embeddings

        # Pre-compute column groupings from schema
        per_ds = schema_info.get("per_dataset", [{}])
        all_text_cols: set = set()
        all_image_cols: set = set()
        for ds_entry in per_ds:
            detected = ds_entry.get("detected_columns", {}) if isinstance(ds_entry, dict) else {}
            all_text_cols.update(detected.get("text", []))
            all_image_cols.update(detected.get("image", []))
        self._text_cols = [c for c in all_text_cols if c in df.columns]
        self._image_cols = [c for c in all_image_cols if c in df.columns]
        # Track image load failures for aggregate reporting
        self._image_load_attempts: int = 0
        self._image_load_failures: int = 0
        self._tabular_cols = [
            c for c in df.columns
            if c not in self._text_cols and c not in self._image_cols
        ]

        # Pre-transform tabular block once (cheap since it's already a float32 array)
        if self.tabular_preprocessor is not None and self._tabular_cols:
            self._tabular_array = torch.tensor(
                self.tabular_preprocessor.transform(self.df[self._tabular_cols]),
                dtype=torch.float32,
            )
        else:
            self._tabular_array = None

    @staticmethod
    def _clean_text_value(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        return "" if text.lower() in {"", "nan", "none", "null", "<na>"} else text

    def _compose_text(self, row: pd.Series, columns: List[str]) -> str:
        parts = [self._clean_text_value(row[col]) for col in columns if col in row.index]
        parts = [p for p in parts if p]
        return " ".join(parts)

    @staticmethod
    def _resolve_image_path(row: pd.Series, columns: List[str], base_dir: Optional[str] = None) -> Optional[str]:
        for col in columns:
            if col not in row.index:
                continue
            raw = row[col]
            if pd.isna(raw):
                continue
            path_val = str(raw).strip()
            if path_val.lower() in {"", "nan", "none", "null", "<na>"}:
                continue
            # Fast path: absolute + exists
            p = Path(path_val)
            if p.is_absolute() and p.is_file():
                return path_val
            # Relative path: resolve against base_dir then CWD
            if base_dir:
                resolved = Path(base_dir) / path_val
                if resolved.is_file():
                    return str(resolved)
            cwd_resolved = Path.cwd() / path_val
            if cwd_resolved.is_file():
                return str(cwd_resolved)
            # Fall back: return raw and let the caller handle FileNotFoundError
            return path_val
        return None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        sample: dict = {"target": self.targets[idx]}

        # Tabular
        if self._tabular_array is not None:
            sample["tabular"] = self._tabular_array[idx]

        # Text: use pre-computed embedding if available, else tokenize
        if self._precomputed_text is not None and self._text_cols:
            sample["text_pooled"] = self._precomputed_text[idx]
        elif self.text_preprocessor is not None and self._text_cols:
            text_val = self._compose_text(row, self._text_cols)
            enc = self.text_preprocessor(text_val)
            sample["input_ids"] = enc["input_ids"]
            sample["attention_mask"] = enc["attention_mask"]

        # Image: use pre-computed embedding if available, else load + preprocess
        # Training datasets apply RandomFlip/Rotate/ColorJitter augmentation
        # before Resize+Normalize.  Validation/test datasets skip augmentation
        # so metrics are computed on deterministic, unperturbed inputs.
        if self._precomputed_image is not None and self._image_cols:
            sample["image_pooled"] = self._precomputed_image[idx]
        elif self.image_preprocessor is not None and self._image_cols:
            img_path = self._resolve_image_path(row, self._image_cols)
            self._image_load_attempts += 1
            try:
                from PIL import Image as PILImage
                if not img_path:
                    raise ValueError("no image path found in row")
                pil_img = PILImage.open(img_path).convert("RGB")
                if self.apply_augmentation and hasattr(self.image_preprocessor, "augment"):
                    pil_img = self.image_preprocessor.augment(pil_img)
                sample["image"] = self.image_preprocessor(pil_img)
            except Exception as exc:
                self._image_load_failures += 1
                # Warn only at first failure and then every 50th to avoid log spam
                if self._image_load_failures == 1 or self._image_load_failures % 50 == 0:
                    _rate = self._image_load_failures / max(1, self._image_load_attempts)
                    logger.warning(
                        "Image load failed (failure %d/%d attempts = %.0f%%): "
                        "idx=%d path=%s err=%s. "
                        "Check that image paths are accessible from the server.",
                        self._image_load_failures, self._image_load_attempts,
                        _rate * 100, idx, img_path, exc,
                    )
                h, w = self.image_preprocessor.target_size
                sample["image"] = torch.zeros(3, h, w, dtype=torch.float32)

        return sample


# ---------------------------------------------------------------------------
# Frozen encoder pre-computation helpers
# ---------------------------------------------------------------------------

def _precompute_text_embeddings(
    dataset: MultimodalPyTorchDataset,
    text_encoder,
    device: torch.device,
    batch_size: int = 32,
    progress_fn: Optional[Any] = None,
) -> torch.Tensor:
    """Run frozen text encoder over all samples once.

    Returns a ``[N, output_dim]`` float32 CPU tensor aligned to the
    dataset's row indices so that ``Subset`` indexing works correctly.
    """
    text_encoder.eval()
    all_embeds: List[torch.Tensor] = []
    n = len(dataset)
    total_batches = max(1, (n + batch_size - 1) // batch_size)

    for batch_idx, start in enumerate(range(0, n, batch_size)):
        end = min(start + batch_size, n)
        input_ids_list = []
        attn_mask_list = []
        for i in range(start, end):
            sample = dataset[i]
            input_ids_list.append(sample["input_ids"])
            attn_mask_list.append(sample["attention_mask"])

        input_ids = torch.stack(input_ids_list).to(device)
        attn_mask = torch.stack(attn_mask_list).to(device)

        with torch.no_grad():
            outputs = text_encoder.transformer(
                input_ids=input_ids,
                attention_mask=attn_mask,
            )
            cls_token = outputs.last_hidden_state[:, 0, :]
            if text_encoder._projection is not None:
                cls_token = text_encoder._projection(cls_token)
            all_embeds.append(cls_token.cpu())

        if progress_fn is not None and batch_idx % 10 == 0:
            pct = int(batch_idx / total_batches * 100)
            progress_fn(pct, f"Pre-computing text embeddings: {end}/{n} samples")

    if progress_fn is not None:
        progress_fn(100, f"Text embeddings done: {n} samples")
    return torch.cat(all_embeds, dim=0)


def _precompute_image_embeddings(
    dataset: MultimodalPyTorchDataset,
    image_encoder,
    device: torch.device,
    batch_size: int = 32,
    progress_fn: Optional[Any] = None,
) -> torch.Tensor:
    """Run frozen image encoder over all samples once.

    Returns a ``[N, output_dim]`` float32 CPU tensor aligned to the
    dataset's row indices.  Must only be used on datasets **without**
    augmentation (validation split) to ensure deterministic embeddings.
    """
    image_encoder.eval()
    all_embeds: List[torch.Tensor] = []
    n = len(dataset)
    total_batches = max(1, (n + batch_size - 1) // batch_size)

    for batch_idx, start in enumerate(range(0, n, batch_size)):
        end = min(start + batch_size, n)
        img_list = []
        for i in range(start, end):
            sample = dataset[i]
            img_list.append(sample["image"])

        images = torch.stack(img_list).to(device)

        with torch.no_grad():
            embeds = image_encoder(images)
            # Ensure pooled (N, D) — guard against encoders returning patch sequences
            if embeds.ndim == 3:
                embeds = embeds.mean(dim=1)
            all_embeds.append(embeds.cpu())

        if progress_fn is not None and batch_idx % 10 == 0:
            pct = int(batch_idx / total_batches * 100)
            progress_fn(pct, f"Pre-computing image embeddings: {end}/{n} samples")

    if progress_fn is not None:
        progress_fn(100, f"Image embeddings done: {n} samples")
    return torch.cat(all_embeds, dim=0)


# ---------------------------------------------------------------------------
# Out-of-core streaming dataset for 100 GB+ datasets
# ---------------------------------------------------------------------------

class AutoVisionIterableDataset(torch.utils.data.IterableDataset):
    """
    Out-of-core streaming :class:`IterableDataset` for datasets that
    exceed available RAM.

    Instead of materialising the full DataFrame, this class reads the
    backing CSV / Parquet file(s) in fixed-size chunks via
    ``pd.read_csv(chunksize=…)`` or ``pd.read_parquet`` with row-group
    slicing.  Preprocessing (tabular scaling, BERT tokenization, image
    resize+normalize) is applied lazily **inside the generator yield
    loop** so only ``chunksize`` rows are ever resident in memory.

    Multi-worker safety
    -------------------
    When ``DataLoader(num_workers > 0)``, each worker receives a copy of
    this object.  ``__iter__`` inspects ``torch.utils.data.get_worker_info()``
    and mathematically partitions chunks across workers so that no two
    workers process the same row.

    Batch dictionary contract
    -------------------------
    Each yielded sample is a ``dict`` with the **exact same keys** as
    ``MultimodalPyTorchDataset.__getitem__``::

        {
            "target":          torch.Tensor,          # always
            "tabular":         torch.Tensor [D],      # when tabular_preprocessor is set
            "input_ids":       torch.LongTensor [128], # when text_preprocessor is set
            "attention_mask":  torch.LongTensor [128], # when text_preprocessor is set
            "image":           torch.Tensor [3,224,224] # when image_preprocessor is set
        }

    Parameters
    ----------
    file_paths : list[str | Path]
        One or more CSV or Parquet file paths to stream from.
    target_column : str
        Name of the target column in the underlying files.
    schema_info : dict
        Phase 2 schema output (``asdict(GlobalSchema)``).
    target_encoder : object | None
        Fitted ``LabelEncoder``, ``StandardScaler``, or multilabel dict
        from Phase 3 target encoding.
    tabular_preprocessor : TabularPreprocessor | None
        Fitted tabular transformer.
    text_preprocessor : TextPreprocessor | None
        Callable BERT tokeniser.
    image_preprocessor : ImagePreprocessor | None
        Callable torchvision pipeline.
    chunksize : int
        Number of rows to read per chunk.  Controls peak memory usage.
        Default 4096 balances I/O throughput against RAM.
    apply_augmentation : bool
        When True, image augmentation is applied (training split).
    indices : list[int] | None
        Optional subset of global row indices to yield.  Used by the
        orchestrator to implement train/val splits without requiring a
        data-duplicating Subset wrapper.
    """

    def __init__(
        self,
        file_paths: "List[Union[str, Path]]",
        target_column: str,
        schema_info: dict,
        target_encoder=None,
        tabular_preprocessor=None,
        text_preprocessor=None,
        image_preprocessor=None,
        chunksize: int = 4096,
        apply_augmentation: bool = False,
        indices: "Optional[List[int]]" = None,
    ) -> None:
        super().__init__()
        self._file_paths = [Path(p) for p in file_paths]
        self._target_column = target_column
        self._schema_info = schema_info
        self._target_encoder = target_encoder
        self._tabular_preprocessor = tabular_preprocessor
        self._text_preprocessor = text_preprocessor
        self._image_preprocessor = image_preprocessor
        self._chunksize = chunksize
        self._apply_augmentation = apply_augmentation
        self._indices = set(indices) if indices is not None else None

        # Pre-compute column groupings from schema (same logic as map-style)
        per_ds = schema_info.get("per_dataset", [{}])
        all_text_cols: set = set()
        all_image_cols: set = set()
        for ds_entry in per_ds:
            detected = ds_entry.get("detected_columns", {}) if isinstance(ds_entry, dict) else {}
            all_text_cols.update(detected.get("text", []))
            all_image_cols.update(detected.get("image", []))
        self._text_cols = [str(c) for c in all_text_cols]
        self._image_cols = [str(c) for c in all_image_cols]

    @staticmethod
    def _clean_text_value(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        return "" if text.lower() in {"", "nan", "none", "null", "<na>"} else text

    def _compose_text(self, row: pd.Series, columns: List[str]) -> str:
        parts = [self._clean_text_value(row[col]) for col in columns if col in row.index]
        parts = [p for p in parts if p]
        return " ".join(parts)

    @staticmethod
    def _resolve_image_path(row: pd.Series, columns: List[str], base_dir: Optional[str] = None) -> Optional[str]:
        for col in columns:
            if col not in row.index:
                continue
            raw = row[col]
            if pd.isna(raw):
                continue
            path_val = str(raw).strip()
            if path_val.lower() in {"", "nan", "none", "null", "<na>"}:
                continue
            p = Path(path_val)
            if p.is_absolute() and p.is_file():
                return path_val
            if base_dir:
                resolved = Path(base_dir) / path_val
                if resolved.is_file():
                    return str(resolved)
            cwd_resolved = Path.cwd() / path_val
            if cwd_resolved.is_file():
                return str(cwd_resolved)
            return path_val
        return None

    # ------------------------------------------------------------------ #
    #  Target encoding (mirrors Phase 3 logic)
    # ------------------------------------------------------------------ #

    def _encode_target(self, y_series: "pd.Series") -> torch.Tensor:
        """Encode a chunk's target column using the fitted Phase 3 encoder."""
        enc = self._target_encoder
        if enc is None:
            # Fallback: raw float
            return torch.tensor(y_series.values.astype(float), dtype=torch.float32)

        if isinstance(enc, dict) and enc.get("type") == "multilabel":
            import ast
            label_to_idx = enc["label_to_idx"]
            n_classes = len(enc["all_labels"])
            parsed = y_series.astype(str).apply(
                lambda v: ast.literal_eval(v) if v.startswith("{") else {v: 1.0}
            )
            multi_hot = np.zeros((len(parsed), n_classes), dtype=np.float32)
            for row_i, d in enumerate(parsed):
                for lbl, val in d.items():
                    if lbl in label_to_idx:
                        multi_hot[row_i, label_to_idx[lbl]] = float(val) / 100.0
            return torch.tensor(multi_hot, dtype=torch.float32)

        if hasattr(enc, "transform"):
            # LabelEncoder or StandardScaler
            try:
                encoded = enc.transform(y_series.astype(str))
                if hasattr(enc, "classes_"):
                    return torch.tensor(encoded, dtype=torch.long)
                # StandardScaler for regression
                return torch.tensor(
                    encoded.ravel() if encoded.ndim > 1 else encoded,
                    dtype=torch.float32,
                )
            except Exception:
                return torch.tensor(y_series.values.astype(float), dtype=torch.float32)

        return torch.tensor(y_series.values.astype(float), dtype=torch.float32)

    # ------------------------------------------------------------------ #
    #  Chunk reader generators
    # ------------------------------------------------------------------ #

    def _read_chunks(self, filepath: Path):
        """
        Yield ``(chunk_df, global_start_row)`` tuples from a single file.

        CSV → ``pd.read_csv(chunksize=…)`` (native streaming).
        Parquet → row-group-aligned slicing via ``pd.read_parquet``.
        """
        if filepath.suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(str(filepath))
                global_offset = 0
                for batch in pf.iter_batches(batch_size=self._chunksize):
                    chunk = batch.to_pandas()
                    yield chunk, global_offset
                    global_offset += len(chunk)
                return
            except ImportError:
                # Fall back to reading full file (degrades gracefully)
                df = pd.read_parquet(str(filepath))
                for start in range(0, len(df), self._chunksize):
                    chunk = df.iloc[start:start + self._chunksize]
                    yield chunk, start
                return

        # CSV: native chunked reader — O(chunksize) RAM
        reader = pd.read_csv(str(filepath), chunksize=self._chunksize)
        global_offset = 0
        for chunk in reader:
            yield chunk, global_offset
            global_offset += len(chunk)

    # ------------------------------------------------------------------ #
    #  Core iterator with worker sharding
    # ------------------------------------------------------------------ #

    def __iter__(self):
        """
        Yield one sample dict per row, applying preprocessing lazily.

        Multi-worker sharding
        ---------------------
        ``torch.utils.data.get_worker_info()`` returns ``None`` in the
        main process and a ``WorkerInfo(id, num_workers)`` in workers.
        We assign each chunk to ``worker_id = chunk_index % num_workers``
        so that no two workers process the same data and every chunk is
        covered exactly once across the worker pool.
        """
        from torch.utils.data import get_worker_info

        worker_info = get_worker_info()
        worker_id = 0
        num_workers = 1
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        global_row_idx = 0
        chunk_counter = 0

        for filepath in self._file_paths:
            if not filepath.exists():
                logger.warning(
                    "AutoVisionIterableDataset: file not found: %s", filepath
                )
                continue

            for chunk_df, _offset in self._read_chunks(filepath):
                # Worker sharding: this worker only processes its assigned chunks
                if chunk_counter % num_workers != worker_id:
                    global_row_idx += len(chunk_df)
                    chunk_counter += 1
                    continue
                chunk_counter += 1

                # Separate target from features
                if self._target_column in chunk_df.columns:
                    y_chunk = chunk_df[self._target_column]
                    feature_chunk = chunk_df.drop(columns=[self._target_column])
                else:
                    y_chunk = chunk_df.iloc[:, -1]
                    feature_chunk = chunk_df.iloc[:, :-1]

                # Encode targets for this chunk
                targets = self._encode_target(y_chunk)

                # Identify column groups within this chunk
                text_cols = [c for c in self._text_cols if c in feature_chunk.columns]
                image_cols = [c for c in self._image_cols if c in feature_chunk.columns]
                tabular_cols = [
                    c for c in feature_chunk.columns
                    if c not in text_cols and c not in image_cols
                ]

                # Pre-transform tabular block for the entire chunk (vectorised)
                tabular_array = None
                if self._tabular_preprocessor is not None and tabular_cols:
                    try:
                        tabular_array = torch.tensor(
                            self._tabular_preprocessor.transform(
                                feature_chunk[tabular_cols]
                            ),
                            dtype=torch.float32,
                        )
                    except Exception as tab_exc:
                        logger.warning(
                            "AutoVisionIterableDataset: tabular transform "
                            "failed on chunk: %s", tab_exc,
                        )

                # Yield individual samples from the chunk
                for i in range(len(chunk_df)):
                    row_global = global_row_idx + i

                    # Index filtering (train/val split)
                    if self._indices is not None and row_global not in self._indices:
                        continue

                    sample: dict = {"target": targets[i]}

                    # Tabular
                    if tabular_array is not None:
                        sample["tabular"] = tabular_array[i]

                    # Text (first text column, lazy BERT tokenization)
                    if self._text_preprocessor is not None and text_cols:
                        text_val = self._compose_text(feature_chunk.iloc[i], text_cols)
                        enc = self._text_preprocessor(text_val)
                        sample["input_ids"] = enc["input_ids"]
                        sample["attention_mask"] = enc["attention_mask"]

                    # Image (lazy PIL load + resize + normalize)
                    if self._image_preprocessor is not None and image_cols:
                        try:
                            from PIL import Image as PILImage
                            img_path = self._resolve_image_path(feature_chunk.iloc[i], image_cols)
                            if not img_path:
                                raise ValueError("no image path found in row")
                            pil_img = PILImage.open(img_path).convert("RGB")
                            if (self._apply_augmentation
                                    and hasattr(self._image_preprocessor, "augment")):
                                pil_img = self._image_preprocessor.augment(pil_img)
                            sample["image"] = self._image_preprocessor(pil_img)
                        except Exception:
                            h, w = self._image_preprocessor.target_size
                            sample["image"] = torch.zeros(
                                3, h, w, dtype=torch.float32
                            )

                    yield sample

                global_row_idx += len(chunk_df)


class TrainingOrchestrator:
    """
    Orchestrates complete 7-phase training pipeline.

    Usage:
        config = TrainingConfig(
            dataset_sources=["https://..."],
            problem_type="classification_multiclass",
            modalities=["image", "text", "tabular"]
        )
        orchestrator = TrainingOrchestrator(config)
        result = asyncio.run(orchestrator.run_pipeline())
    """

    def __init__(
        self,
        config: TrainingConfig,
        execution_context: Optional[Any] = None,
    ):
        """Initialize orchestrator."""
        self.config = config
        self.execution_context = execution_context
        # FIX-19: enforce context integrity only for session-backed runs.
        # Standalone runs (execution_context=None at init) remain unaffected.
        self._require_context: bool = execution_context is not None
        self.current_phase = Phase.DATA_INGESTION
        self.phase_results = {}
        self.state = PipelineState()
        self.start_time = None
        self.metrics_history = []
        # Lazy dataset registry – populated by Phase 1
        self.dataset_registry = DatasetManager()
        # Phase 3 outputs – set by _execute_phase_3_preprocessing
        self.torch_dataset = None
        # Augmentation-aware variants: train receives augmented images,
        # val/test receive only Resize+Normalize (no random transforms).
        self.train_torch_dataset = None
        self.val_torch_dataset = None
        self.fitted_transformers = {}
        # Phase 5 output – best trained LightningModule captured by Optuna closure
        self.best_lightning_module: Optional[Any] = None
        self.probability_calibrator: Optional[ProbabilityCalibrator] = None
        self.preprocessing_planner = PreprocessingPlanner()
        self.optuna_adaptive = AdaptiveOptunaController()
        self.embedding_cache = EmbeddingCache()
        self.representation_layer = RepresentationLayer()
        self.evaluation_adapter = EvaluationAdapter()
        self.drift_adapter = DriftAdapter()
        self.research_metrics = ResearchMetrics()
        self._phase6_reference_sample: Optional[Any] = None
        self.state.set_slot("config", asdict(config))

        # Setup device
        self.device = torch.device(config.device)
        logger.info(f"Using device: {self.device}")

        if self.device.type == "cuda":
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        if self.execution_context is not None:
            logger.info(
                "TrainingOrchestrator: ExecutionContext attached "
                "(session=%s)",
                getattr(self.execution_context, "session_id", "?"),
            )

    def _get_ctx(self) -> Any:
        """
        Return attached ExecutionContext.

        Raises when context was expected (session-backed run) but has become
        None, which indicates unexpected state mutation.
        """
        if self._require_context and self.execution_context is None:
            raise RuntimeError(
                "ExecutionContext was provided at construction time but is now None. "
                "This indicates unexpected state mutation. "
                "All session-backed phases require a valid ExecutionContext."
            )
        return self.execution_context

    def _filter_modalities_by_drift(self) -> None:
        """
        Drop modalities whose drift-adjusted predictability has fallen below
        the candidate-selector threshold (0.25).  Called before Phase 4 on
        retrain runs so that a modality that has degraded to noise does not
        drag down the multimodal model.
        """
        ctx = self._get_ctx()
        if ctx is None:
            return
        if not getattr(ctx, "drift_feedback_applied", False):
            return
        adjusted = dict(getattr(ctx, "drift_adjusted_predictability", {}) or {})
        if not adjusted:
            return
        _THRESHOLD = 0.25
        weak = [mod for mod, score in adjusted.items() if float(score or 0) < _THRESHOLD]
        if weak and self.config.modalities:
            before = list(self.config.modalities)
            self.config.modalities = [m for m in before if m not in weak]
            if not self.config.modalities:
                self.config.modalities = before  # safety: never drop all modalities
            else:
                ctx.log_decision(
                    "drift_modality_filter",
                    f"Dropped low-predictability modalities after drift: {weak}",
                    f"drift_adjusted_predictability={adjusted}",
                )
                logger.info(
                    "Drift feedback: dropped modalities %s (predictability < %.2f)",
                    weak, _THRESHOLD,
                )

    def _filter_to_primary_dataset(self) -> None:
        """G14: restrict dataset_sources to primary_dataset_id when datasets are incompatible."""
        ctx = self._get_ctx()
        if ctx is None:
            return
        if getattr(ctx, "datasets_compatible", True):
            return
        primary = getattr(ctx, "primary_dataset_id", None)
        if not primary:
            return
        filtered = [
            s for s in (self.config.dataset_sources or [])
            if primary in s or str(s).endswith(primary)
        ]
        if filtered and len(filtered) < len(self.config.dataset_sources or []):
            self.config.dataset_sources = filtered
            ctx.log_decision(
                "dataset_filter",
                f"Restricted to primary_dataset_id={primary}",
                "datasets_compatible=False",
            )
            logger.info(
                "G14: filtered dataset_sources to primary %s (%d source(s))",
                primary,
                len(filtered),
            )

    def _enforce_session_context(self, stage: str) -> None:
        """
        Validate session-backed context invariants before critical phases.

        Standalone orchestrator runs (no session context attached) bypass this.
        """
        if not self._require_context:
            return

        ctx = self._get_ctx()
        dataset_snapshot = {
            name: {"source": "dataset_registry"}
            for name in self.dataset_registry.list_datasets()
        }
        try:
            validation = ensure_session_context(
                ctx,
                session_id=getattr(ctx, "session_id", None),
                dataset_snapshot=dataset_snapshot,
            )
            for warning in validation.warnings:
                logger.warning("%s context warning: %s", stage, warning)
        except ContextValidationError as exc:
            raise RuntimeError(f"{stage}: {exc}") from exc

    def _record_phase_timing_in_context(self, phase_name: str, duration_s: float) -> None:
        """Mirror PipelineState phase timing into ExecutionContext when present."""
        ctx = self._get_ctx()
        if ctx is None:
            return
        try:
            ctx.record_phase_timing(phase_name, duration_s)
        except Exception as exc:
            logger.debug("Could not record phase timing in ExecutionContext: %s", exc)

    def _sync_preprocessing_contract_to_context(
        self,
        preprocessing_plan: Dict[str, Any],
        validation_report: Dict[str, Any],
        context_signals: Dict[str, Any],
        total_samples: int,
        adaptive_tabular_config: Dict[str, Any],
        drifted_features: List[str],
    ) -> None:
        """Persist preprocessing planning output into the attached ExecutionContext."""
        ctx = self._get_ctx()
        if ctx is None:
            return

        try:
            context_plan = {
                modality: dict(preprocessing_plan.get(modality, {}) or {})
                for modality in ("tabular", "text", "image")
                if isinstance(preprocessing_plan.get(modality), dict)
            }

            preprocessing_context = {
                "runtime": dict(preprocessing_plan.get("runtime", {}) or {}),
                "weak_modalities": list(preprocessing_plan.get("weak_modalities", []) or []),
                "strong_modalities": list(preprocessing_plan.get("strong_modalities", []) or []),
                "modality_predictability": dict(preprocessing_plan.get("modality_predictability", {}) or {}),
                "context_signals": dict(preprocessing_plan.get("context_signals", {}) or context_signals),
                "validation": dict(validation_report or {}),
                "dataset_total_samples": int(total_samples),
                "fusion_recommendation": preprocessing_plan.get("fusion_recommendation"),
                "adaptive_tabular_config": dict(adaptive_tabular_config or {}),
                "drifted_features": list(drifted_features or []),
            }

            ctx.update_preprocessing_contract(context_plan, preprocessing_context)
            ctx.set_pipeline_stage("preprocessing_planning")
        except Exception as ctx_pre_exc:
            logger.warning(
                "Phase 3: failed to update ExecutionContext preprocessing contract: %s",
                ctx_pre_exc,
            )

    def _sync_training_results_to_context(
        self,
        results: Dict[str, Any],
        active_modalities: Optional[List[str]] = None,
    ) -> None:
        """Persist phase 5 training summary into the attached ExecutionContext."""
        ctx = self._get_ctx()
        if ctx is None:
            return

        try:
            modalities = list(active_modalities or results.get("active_modalities", []) or self.config.modalities)
            training_signals = {
                "best_val_loss": results.get("best_val_loss", 0.0),
                "best_val_acc": results.get("best_val_acc", 0.0),
                "best_val_f1": results.get("best_val_f1", 0.0),
                "best_trial": results.get("best_trial", 0),
                "n_trials": results.get("n_trials", 0),
                "training_time": f"{float(results.get('duration_seconds', 0.0) or 0.0):.1f}s",
                "fit_type": results.get("fit_type", "unknown"),
                "trial_diagnostics": list(results.get("trial_diagnostics", []) or []),
                "trial_feedback_summary": dict(results.get("trial_feedback_summary", {}) or {}),
                "next_run_feedback": dict(results.get("next_run_feedback", {}) or {}),
                "alignment_summary": dict(results.get("alignment_summary", {}) or {}),
                "fusion_summary": dict(results.get("fusion_summary", {}) or {}),
                "fusion_aux_weights": dict(results.get("fusion_aux_weights", {}) or {}),
                "active_modalities": modalities,
                # Explicit fusion audit field — used by monitoring and next-run decisions
                "fusion_strategy_used": (
                    results.get("fusion_strategy")
                    or dict(results.get("fusion_summary", {}) or {}).get("strategy")
                    or getattr(ctx, "fusion_strategy", None)
                    or "unknown"
                ),
                # Calibration placeholder — populated after Phase 7 calibration
                "calibration": dict(results.get("calibration", {}) or {}),
            }

            ctx.update_training(training_signals)
        except Exception as ctx_train_exc:
            logger.warning(
                "Phase 5: failed to update ExecutionContext training summary: %s",
                ctx_train_exc,
            )

    def _sync_drift_results_to_context(
        self,
        results: Dict[str, Any],
        modality_drift: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist phase 6 drift summary into the attached ExecutionContext."""
        ctx = self._get_ctx()
        if ctx is None:
            return

        try:
            retrain_info = dict(results.get("retrain_info") or {})
            retrain_result = dict(retrain_info.get("result") or {}) if isinstance(retrain_info.get("result"), dict) else {}
            retrain_event = dict(retrain_info.get("event") or {}) if isinstance(retrain_info.get("event"), dict) else {}
            retrain_model_id = (
                str(
                    retrain_result.get("model_id")
                    or retrain_event.get("model_id")
                    or ""
                ).strip()
            )
            retrain_deployment_ready = bool(
                retrain_result.get("deployment_ready")
                if retrain_result
                else retrain_event.get("deployment_ready", False)
            )

            ctx.update_drift(
                detected=bool(results.get("drift_detected", False)),
                severity=float(results.get("composite_score", 0.0) or 0.0),
                details={
                    "ks": float(results.get("metrics", {}).get("ks_statistic", 0.0) or 0.0),
                    "psi": float(results.get("metrics", {}).get("psi", 0.0) or 0.0),
                    "mmd": float(results.get("metrics", {}).get("fdd", 0.0) or 0.0),
                    "composite": float(results.get("composite_score", 0.0) or 0.0),
                    "modality_drift": dict(modality_drift or {}),
                    "retrain_triggered": bool(results.get("retrain_triggered", False)),
                    "retrain_info": retrain_info,
                },
            )
            ctx.apply_drift_feedback(results, decay=0.5)

            if retrain_model_id:
                if retrain_model_id not in ctx.registered_model_ids:
                    ctx.registered_model_ids.append(retrain_model_id)
                if retrain_deployment_ready:
                    ctx.active_prediction_model_id = retrain_model_id
                ctx.log_decision(
                    "model_registry",
                    f"Retrained model registered: {retrain_model_id}",
                    evidence=(
                        "active_prediction_model_id="
                        f"{ctx.active_prediction_model_id}"
                    ),
                )

            ctx.set_pipeline_stage("drift_detection")
        except Exception as ctx_drift_exc:
            logger.warning(
                "Phase 6: failed to update ExecutionContext drift state: %s",
                ctx_drift_exc,
            )

    def _sync_ingestion_results_to_context(
        self,
        registered_datasets: List[str],
        dataset_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """Persist phase 1 dataset registration into the attached ExecutionContext."""
        ctx = self._get_ctx()
        if ctx is None:
            return

        try:
            metadata_map = dict(dataset_metadata or {})
            for dataset_id in list(registered_datasets or []):
                dataset_id = str(dataset_id)
                profile = ctx.get_dataset_profile(dataset_id)
                if profile is None:
                    profile = DatasetProfile(dataset_id=dataset_id)

                metadata = dict(metadata_map.get(dataset_id, {}) or {})
                source_url = metadata.get("source_url") or metadata.get("source")
                cache_path = metadata.get("cache_path")

                if source_url and not profile.source_url:
                    profile.source_url = str(source_url)
                if cache_path and not profile.file_path:
                    profile.file_path = str(cache_path)

                ctx.add_dataset_profile(profile)

            ctx.set_pipeline_stage("ingestion_complete")
        except Exception as ctx_ingest_exc:
            logger.warning(
                "Phase 1: failed to update ExecutionContext ingestion state: %s",
                ctx_ingest_exc,
            )

    def _sync_model_registry_to_context(
        self,
        model_id: str,
        deployment_ready: bool,
    ) -> None:
        """Persist phase 7 registry output into the attached ExecutionContext."""
        ctx = self._get_ctx()
        if ctx is None:
            return

        try:
            model_id = str(model_id)
            if model_id and model_id not in ctx.registered_model_ids:
                ctx.registered_model_ids.append(model_id)

            if deployment_ready:
                ctx.active_prediction_model_id = model_id

            if model_id:
                ctx.log_decision(
                    "model_registry",
                    f"Retrained model registered: {model_id}",
                    evidence=(
                        "active_prediction_model_id="
                        f"{ctx.active_prediction_model_id}"
                    ),
                )

            ctx.set_pipeline_stage("model_registry")
        except Exception as ctx_registry_exc:
            logger.warning(
                "Phase 7: failed to update ExecutionContext model registry state: %s",
                ctx_registry_exc,
            )

    def _collect_validation_logits(
        self,
        lightning_module: Any,
        val_loader: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Collect validation logits and targets for post-hoc calibration."""
        lightning_module.eval()
        try:
            device = next(lightning_module.parameters()).device
        except Exception:
            device = self.device

        logits_batches: List[torch.Tensor] = []
        target_batches: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in val_loader:
                batch_on_device: Dict[str, Any] = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch_on_device[key] = value.to(device)
                    else:
                        batch_on_device[key] = value

                logits = lightning_module(batch_on_device)
                targets = batch_on_device.get("target")
                if not isinstance(logits, torch.Tensor) or not isinstance(targets, torch.Tensor):
                    continue

                logits_batches.append(logits.detach().cpu())
                target_batches.append(targets.detach().cpu())

        if not logits_batches or not target_batches:
            raise RuntimeError("Could not collect validation logits/targets for calibration")

        logits_np = torch.cat(logits_batches, dim=0).numpy()
        targets_np = torch.cat(target_batches, dim=0).numpy()
        return logits_np, targets_np

    # ------------------------------------------------------------------
    # External result injection methods (Tree A -> Tree B unification)
    # ------------------------------------------------------------------

    def inject_external_schema(
        self,
        schema_dict: Dict[str, Any],
        target_override: Optional[str] = None,
    ) -> None:
        """
        Inject a pre-computed GlobalSchema dict from /api/schema/detect.
        _execute_phase_2_schema_detection() will be a no-op when this is set.
        """
        if not isinstance(schema_dict, dict):
            return

        if target_override:
            schema_dict = dict(schema_dict)
            schema_dict["primary_target"] = target_override
            for ds in schema_dict.get("per_dataset", []):
                if isinstance(ds, dict):
                    ds["target_column"] = target_override

        self.phase_results[Phase.SCHEMA_DETECTION] = schema_dict
        self.state.set_slot("phase2_schema", schema_dict)

        ctx = self._get_ctx()
        if ctx is not None:
            try:
                ctx.update_from_schema(schema_dict)

                if target_override:
                    current_target = getattr(ctx, "global_target", None)
                    if current_target != target_override:
                        if hasattr(ctx, "override_global_target"):
                            ctx.override_global_target(
                                target_override,
                                "TrainingOrchestrator.inject_external_schema",
                            )
                        else:
                            ctx.global_target = target_override

                per_dataset = schema_dict.get("per_dataset", [])
                if isinstance(per_dataset, list):
                    for ds_result in per_dataset:
                        if not isinstance(ds_result, dict):
                            continue
                        dataset_id = str(ds_result.get("dataset_id", "") or "").strip()
                        if not dataset_id:
                            continue

                        profile = ctx.get_dataset_profile(dataset_id)
                        if profile is None:
                            profile = DatasetProfile(dataset_id=dataset_id)
                            ctx.add_dataset_profile(profile)

                        modalities = list(ds_result.get("modalities", []) or [])
                        reasoning = ds_result.get("reasoning", {})
                        xs3_gap = 0.0
                        if isinstance(reasoning, dict):
                            xs3_gap = float(
                                reasoning.get(
                                    "xs3_confidence_gap",
                                    reasoning.get(
                                        "confidence_gap",
                                        ds_result.get("confidence", 0.0),
                                    ),
                                )
                                or 0.0
                            )

                        profile.schema_detected = True
                        profile.schema_result = dict(ds_result)
                        profile.schema_confidence = float(
                            ds_result.get("confidence", ds_result.get("schema_confidence", 0.0)) or 0.0
                        )
                        profile.schema_evidence = (
                            f"Detected {len(modalities)} modalities; "
                            f"X-S3 confidence gap {xs3_gap:.3f}"
                        )
                        profile.modality_breakdown = {
                            modality: (1.0 / len(modalities)) if modalities else 0.0
                            for modality in modalities
                        }

                        target_column = (
                            ds_result.get("target_column")
                            or schema_dict.get("primary_target")
                            or ctx.global_target
                        )
                        if target_column:
                            profile.chosen_target = str(target_column)

                        ctx.dataset_profiles[dataset_id] = profile

                ctx.set_pipeline_stage("schema_detection")
            except Exception as ctx_exc:
                logger.warning(
                    "inject_external_schema: failed to mirror schema into ExecutionContext: %s",
                    ctx_exc,
                )
        logger.info(
            "TrainingOrchestrator: injected external schema "
            "(primary_target=%s, %d datasets)",
            schema_dict.get("primary_target", "?"),
            len(schema_dict.get("per_dataset", [])),
        )

    def inject_external_preprocessors(
        self,
        tabular_scaler_path: Optional[str] = None,
    ) -> None:
        """
        Load pre-fitted TabularPreprocessor from disk.
        _execute_phase_3_preprocessing() will reuse it instead of re-fitting.
        """
        if not tabular_scaler_path:
            return

        scaler_path = Path(tabular_scaler_path)
        if not scaler_path.exists():
            logger.warning(
                "inject_external_preprocessors: path not found: %s "
                "- Phase 3 will re-fit.",
                scaler_path,
            )
            return

        try:
            import joblib

            tab_prep = joblib.load(str(scaler_path))
            self.fitted_transformers["tabular"] = tab_prep
            logger.info(
                "TrainingOrchestrator: loaded TabularPreprocessor from %s "
                "(output_dim=%d)",
                scaler_path,
                tab_prep.get_output_dim() if hasattr(tab_prep, "get_output_dim") else -1,
            )
        except Exception as exc:
            logger.warning(
                "inject_external_preprocessors: failed to load %s: %s "
                "- Phase 3 will re-fit.",
                scaler_path,
                exc,
            )

    def inject_external_model_selection(
        self,
        model_sel_dict: Dict[str, Any],
    ) -> None:
        """
        Inject pre-computed model selection from /select-model.
        _execute_phase_4_model_selection() will be a no-op when this is set.
        """
        if not isinstance(model_sel_dict, dict):
            return

        self.phase_results[Phase.MODEL_SELECTION] = model_sel_dict
        self.state.set_slot("phase4_model_selection", model_sel_dict)

        ctx = self._get_ctx()
        if ctx is not None:
            try:
                recommendations = model_sel_dict.get("recommended_models")
                if not isinstance(recommendations, list) or not recommendations:
                    recommendations = [model_sel_dict]

                ctx.update_model_selection(
                    recommendations,
                    "TrainingOrchestrator.inject_external_model_selection",
                )

                selected_model = model_sel_dict.get("selected_model") or model_sel_dict.get("name")
                if selected_model:
                    ctx.selected_model = str(selected_model)

                fusion_strategy = model_sel_dict.get("fusion_strategy")
                if fusion_strategy:
                    modality_importance = model_sel_dict.get("modality_importance", {})
                    if not isinstance(modality_importance, dict):
                        modality_importance = {}
                    ctx.update_fusion(str(fusion_strategy), dict(modality_importance))

                ctx.set_pipeline_stage("model_selection")
            except Exception as ctx_exc:
                logger.warning(
                    "inject_external_model_selection: failed to mirror selection into ExecutionContext: %s",
                    ctx_exc,
                )
        logger.info(
            "TrainingOrchestrator: injected external model selection (%s)",
            model_sel_dict.get("selected_model", model_sel_dict.get("name", "?")),
        )

    def inject_external_datasets(
        self,
        session_datasets: Dict[str, Any],
    ) -> None:
        """
        Register pre-ingested session datasets into DatasetManager.

        This allows Phase 1 ingestion to short-circuit and avoid duplicate
        download/cache work for session-backed training runs.
        """
        if not isinstance(session_datasets, dict) or not session_datasets:
            return

        registered: List[str] = []
        registered_metadata: Dict[str, Dict[str, Any]] = {}
        for source_hash, dataset_obj in session_datasets.items():
            try:
                lazy_ref = (
                    dataset_obj.lazy_data
                    if hasattr(dataset_obj, "lazy_data")
                    else dataset_obj
                )
                metadata = dict(getattr(dataset_obj, "metadata", {}) or {})
                registered_metadata[str(source_hash)] = {
                    "source_url": metadata.get("source_url") or metadata.get("source"),
                    "cache_path": metadata.get("cache_path"),
                }
                self.dataset_registry.register_dataset(
                    source_hash,
                    lazy_ref,
                    metadata={"source": "session_cache", "hash": source_hash},
                )
                registered.append(source_hash)
            except Exception as reg_exc:
                logger.warning(
                    "inject_external_datasets: failed to register %s: %s",
                    source_hash,
                    reg_exc,
                )

        if not registered:
            return

        phase1_result: Dict[str, Any] = {
            "sources": list(session_datasets.keys()),
            "registered_datasets": registered,
            "failed_urls": {},
            "success_count": len(registered),
            "failed_count": 0,
            "ingestion_time": "session_cached",
            "duration_seconds": 0.0,
            "injected": True,
        }
        self.phase_results[Phase.DATA_INGESTION] = phase1_result
        self.state.set_slot("phase1_ingestion", phase1_result)
        self.state.set_phase_timing("DATA_INGESTION", 0.0)
        self._record_phase_timing_in_context("DATA_INGESTION", 0.0)
        self._sync_ingestion_results_to_context(registered, registered_metadata)
        logger.info(
            "TrainingOrchestrator: injected %d session datasets - Phase 1 will be skipped",
            len(registered),
        )
    
    async def run_pipeline(self) -> Dict[str, Any]:
        """Execute complete 7-phase pipeline (async – Phase 1 is truly async)."""
        self.start_time = time.time()
        logger.info("=" * 80)
        logger.info("APEX AutoML Training Pipeline Starting")
        logger.info("=" * 80)
        
        try:
            # Phase 1: Data Ingestion
            await self._execute_phase_1_data_ingestion()
            
            # Phase 2: Schema Detection
            self._execute_phase_2_schema_detection()
            
            # Phase 3: Preprocessing
            self._execute_phase_3_preprocessing()
            
            # Phase 4: Model Selection
            self._execute_phase_4_model_selection()
            
            # Phase 5: Training
            self._execute_phase_5_training()
            
            # Phase 6: Drift Detection (fault-isolated – failure here
            # must not prevent Phase 7 model registration)
            try:
                self._execute_phase_6_drift_detection()
            except Exception as phase6_err:
                logger.warning(
                    "Phase 6 (drift detection) failed – continuing to Phase 7: %s",
                    phase6_err,
                )
                self.phase_results[Phase.DRIFT_DETECTION] = {
                    "status": "error",
                    "error": str(phase6_err),
                    "drift_detected": False,
                    "retrain_triggered": False,
                }

            # Phase 7: Model Registry
            self._execute_phase_7_model_registry()
            
            elapsed = time.time() - self.start_time
            logger.info("=" * 80)
            logger.info(f"✅ PIPELINE COMPLETE - Total time: {elapsed:.2f}s")
            logger.info("=" * 80)
            
            return self._compile_results(elapsed)
            
        except Exception as e:
            logger.error(f"❌ Pipeline execution failed: {str(e)}")
            raise

    def run_phase(self, phase: Any, **kwargs: Any) -> Dict[str, Any]:
        """Legacy phase dispatcher kept for backward compatibility.

        New code should call explicit phase methods (``_execute_phase_*``) or
        ``run_pipeline()``.
        """
        if isinstance(phase, Phase):
            resolved = phase
        elif isinstance(phase, int):
            resolved = Phase(phase)
        elif isinstance(phase, str):
            normalized = phase.strip().upper()
            # Accept either enum names (e.g. "MODEL_SELECTION") or the
            # legacy "phase_4" style labels.
            if normalized.startswith("PHASE_"):
                normalized = normalized.replace("PHASE_", "")
            if normalized.isdigit():
                resolved = Phase(int(normalized))
            else:
                resolved = Phase[normalized]
        else:
            raise ValueError(f"Unsupported phase identifier: {phase!r}")

        if resolved == Phase.DATA_INGESTION:
            raise RuntimeError(
                "DATA_INGESTION is async-only in this orchestrator. "
                "Call _execute_phase_1_data_ingestion() from an async context."
            )
        if resolved == Phase.SCHEMA_DETECTION:
            self._execute_phase_2_schema_detection()
        elif resolved == Phase.PREPROCESSING:
            self._execute_phase_3_preprocessing()
        elif resolved == Phase.MODEL_SELECTION:
            self._execute_phase_4_model_selection()
        elif resolved == Phase.TRAINING:
            self._execute_phase_5_training(
                hp_overrides=kwargs.get("hp_overrides"),
                early_stop_patience=kwargs.get("early_stop_patience", 5),
            )
        elif resolved == Phase.DRIFT_DETECTION:
            self._execute_phase_6_drift_detection()
        elif resolved == Phase.MODEL_REGISTRY:
            self._execute_phase_7_model_registry()

        phase_result = self.phase_results.get(resolved, {})
        if isinstance(phase_result, dict):
            return phase_result
        return {"result": phase_result}
    
    async def _execute_phase_1_data_ingestion(
        self,
        sources: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Phase 1: Async Data Ingestion – download, cache, and register datasets.

        All np.random simulation logic has been removed.  This method now calls
        the real DataIngestionManager and stores lazy references in the
        DatasetManager registry so no data is materialised into RAM.

        Args:
            sources: Override list of dataset URLs/paths.  Falls back to
                     ``self.config.dataset_sources`` when not provided.

        Returns:
            Phase-1 results dict with keys:
              registered_datasets, failed_urls, success_count, failed_count, …
        """
        if (
            Phase.DATA_INGESTION in self.phase_results
            and self.phase_results[Phase.DATA_INGESTION].get("injected")
        ):
            logger.info(
                "Phase 1: SKIPPED - using %d injected session dataset(s)",
                len(self.phase_results[Phase.DATA_INGESTION].get("registered_datasets", [])),
            )
            self.current_phase = Phase.SCHEMA_DETECTION
            return self.phase_results[Phase.DATA_INGESTION]

        logger.info("\n" + "=" * 80)
        logger.info("PHASE 1: DATA INGESTION")
        logger.info("=" * 80)

        phase_start = time.time()
        active_sources: List[str] = sources or self.config.dataset_sources

        try:
            manager = DataIngestionManager()
            lazy_datasets, ingest_meta = await manager.ingest_data(active_sources)

            registered: List[str] = []
            failed_urls: Dict[str, str] = ingest_meta.get("failed", {})

            for source_hash, dataset_obj in lazy_datasets.items():
                lazy_ref = (
                    dataset_obj.lazy_data
                    if hasattr(dataset_obj, "lazy_data")
                    else dataset_obj
                )
                source_url: str = next(
                    (s for s, h in ingest_meta["cached_hashes"].items()
                     if h == source_hash),
                    source_hash,
                )
                self.dataset_registry.register_dataset(
                    source_hash,
                    lazy_ref,
                    metadata={
                        "source_url": source_url,
                        "hash": source_hash,
                        "timestamp": ingest_meta["ingestion_time"],
                    },
                )
                registered.append(source_hash)
                logger.info("  Registered [%s] from %s", source_hash, source_url)

            for url, err in failed_urls.items():
                logger.warning("  FAILED [%s]: %s", url, err)

            elapsed = time.time() - phase_start
            results: Dict[str, Any] = {
                "sources": active_sources,
                "registered_datasets": registered,
                "failed_urls": failed_urls,
                "success_count": len(registered),
                "failed_count": len(failed_urls),
                "ingestion_time": ingest_meta["ingestion_time"],
                "duration_seconds": elapsed,
            }

            logger.info("\nPhase 1 Summary:")
            logger.info("  Registered : %d", len(registered))
            logger.info("  Failed     : %d", len(failed_urls))
            logger.info("  Duration   : %.2fs", elapsed)

            self.phase_results[Phase.DATA_INGESTION] = results
            self.state.set_slot("phase1_ingestion", results)
            self.state.set_phase_timing("DATA_INGESTION", elapsed)
            self._record_phase_timing_in_context("DATA_INGESTION", elapsed)
            self._sync_ingestion_results_to_context(
                registered,
                {
                    str(source_hash): {
                        "source_url": next(
                            (s for s, h in ingest_meta.get("cached_hashes", {}).items() if h == source_hash),
                            source_hash,
                        ),
                    }
                    for source_hash in registered
                },
            )
            self.current_phase = Phase.SCHEMA_DETECTION
            return results

        except Exception as e:
            logger.error("Phase 1 failed: %s", str(e))
            raise
    
    def _execute_phase_2_schema_detection(self) -> None:
        """
        Phase 2: Schema Detection – infer column types, target, and problem type.

        Fetches every lazy dataset reference registered by Phase 1 from the
        DatasetManager, passes them to MultiDatasetSchemaDetector.detect_global_schema(),
        and stores the resulting GlobalSchema in phase_results.

        All hardcoded mock dictionaries have been removed.
        """
        # Short-circuit if schema was injected via inject_external_schema()
        if Phase.SCHEMA_DETECTION in self.phase_results:
            logger.info(
                "Phase 2: SKIPPED - using injected schema (primary_target=%s)",
                self.phase_results[Phase.SCHEMA_DETECTION].get("primary_target", "?"),
            )
            self.current_phase = Phase.PREPROCESSING
            return

        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2: SCHEMA DETECTION")
        logger.info("=" * 80)

        phase_start = time.time()

        try:
            # Retrieve every lazy ref registered in Phase 1
            lazy_datasets: Dict[str, Any] = {
                name: self.dataset_registry.get(name)
                for name in self.dataset_registry.list_datasets()
            }

            if not lazy_datasets:
                raise RuntimeError(
                    "Phase 2 requires datasets from Phase 1. "
                    "dataset_registry is empty – run Phase 1 first."
                )

            detector = MultiDatasetSchemaDetector()
            global_schema: GlobalSchema = detector.detect_global_schema(lazy_datasets)

            elapsed = time.time() - phase_start
            results: Dict[str, Any] = asdict(global_schema)
            results["duration_seconds"] = elapsed

            logger.info("\nPhase 2 Summary:")
            logger.info("  Global Problem Type : %s", global_schema.global_problem_type)
            logger.info("  Global Modalities   : %s", global_schema.global_modalities)
            logger.info("  Primary Target      : %s", global_schema.primary_target)
            logger.info("  Fusion Ready        : %s", global_schema.fusion_ready)
            logger.info("  Confidence          : %.3f", global_schema.detection_confidence)
            logger.info("  Duration            : %.2fs", elapsed)

            self.phase_results[Phase.SCHEMA_DETECTION] = results
            self.state.set_slot("phase2_schema", results)
            self.state.set_phase_timing("SCHEMA_DETECTION", elapsed)
            self._record_phase_timing_in_context("SCHEMA_DETECTION", elapsed)
            self.current_phase = Phase.PREPROCESSING

        except Exception as e:
            logger.error("Phase 2 failed: %s", str(e))
            raise
    
    def _execute_phase_3_preprocessing(self) -> None:
        """
        Phase 3: Preprocessing – materialise datasets, fit transformers, build
        a ``MultimodalPyTorchDataset`` that Phase 5 can hand to a DataLoader.

        Steps
        -----
        1. Materialise all registered lazy datasets into a single pandas
           DataFrame (≤ MAX_ROWS rows to stay memory-safe).
        2. Separate feature columns from the target column (from Phase 2).
        3. Encode the target: ``LabelEncoder`` → ``torch.long`` for
           classification; ``StandardScaler`` → ``torch.float32`` for
           regression.
        4. Identify modality columns from Phase 2 schema.
        5. Fit tabular ``ColumnTransformer`` on feature columns that are
           neither text nor image.
        6. Construct ``MultimodalPyTorchDataset`` with fitted preprocessors.
        7. Store fitted transformers in ``self.fitted_transformers`` and the
           dataset in ``self.torch_dataset``.
        """
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 3: PREPROCESSING")
        logger.info("=" * 80)

        phase_start = time.time()
        MAX_ROWS = 50_000  # memory-safe cap for materialisation

        try:
            # ----------------------------------------------------------------
            # 1  Collect Phase 2 schema info
            # ----------------------------------------------------------------
            schema_info: Dict[str, Any] = self.phase_results.get(
                Phase.SCHEMA_DETECTION, {}
            )
            target_col: str = schema_info.get("primary_target", "Unknown")
            problem_type: str = schema_info.get(
                "global_problem_type", self.config.problem_type
            )

            # ----------------------------------------------------------------
            # 2  Materialise registered lazy datasets
            # ----------------------------------------------------------------
            frames: list = []
            image_datasets: list = []  # Standalone PyTorch image datasets
            for name in self.dataset_registry.list_datasets():
                lazy_ref = self.dataset_registry.get(name)
                if lazy_ref is None:
                    continue
                try:
                    import polars as pl
                    if isinstance(lazy_ref, pl.LazyFrame):
                        frames.append(lazy_ref.head(MAX_ROWS).collect().to_pandas())
                        continue
                except ImportError:
                    pass
                try:
                    import dask.dataframe as dd
                    if isinstance(lazy_ref, dd.DataFrame):
                        frames.append(lazy_ref.head(MAX_ROWS, compute=True))
                        continue
                except ImportError:
                    pass
                if isinstance(lazy_ref, pd.DataFrame):
                    frames.append(lazy_ref.head(MAX_ROWS))
                    continue
                # PyTorch Dataset (e.g. image-only datasets) — stash separately
                from torch.utils.data import Dataset as TorchDataset
                if isinstance(lazy_ref, TorchDataset):
                    image_datasets.append(lazy_ref)
                    logger.info(
                        "  Dataset '%s' is a PyTorch Dataset (%d samples) — "
                        "will be handled as image-only modality.",
                        name, len(lazy_ref),
                    )

            if not frames and not image_datasets:
                raise RuntimeError(
                    "Phase 3: no materialisable datasets found in registry."
                )

            image_only_mode = False
            full_df: pd.DataFrame = (
                pd.concat(frames, ignore_index=True) if frames
                else pd.DataFrame()
            )

            if full_df.empty and image_datasets:
                def _materialize_image_dataset(ds: Any, max_rows: int) -> pd.DataFrame:
                    paths: List[str] = []

                    raw_paths = getattr(ds, "_paths", None)
                    if isinstance(raw_paths, list) and raw_paths:
                        paths = [str(p) for p in raw_paths[:max_rows]]
                    else:
                        upper = min(max_rows, len(ds))
                        for idx in range(upper):
                            try:
                                sample = ds[idx]
                            except Exception:
                                continue
                            path_val = None
                            if isinstance(sample, dict):
                                path_val = sample.get("path")
                            elif isinstance(sample, (tuple, list)) and len(sample) >= 2:
                                path_val = sample[1]
                            if path_val is not None:
                                paths.append(str(path_val))

                    if not paths:
                        return pd.DataFrame()

                    out = pd.DataFrame({"image_path": paths})
                    parent_labels = [Path(p).parent.name for p in paths]
                    unique_labels = sorted({lbl for lbl in parent_labels if lbl})
                    if len(unique_labels) > 1:
                        out["target"] = parent_labels
                    return out

                image_frames = []
                for image_ds in image_datasets:
                    image_frame = _materialize_image_dataset(image_ds, MAX_ROWS)
                    if not image_frame.empty:
                        image_frames.append(image_frame)

                if image_frames:
                    full_df = pd.concat(image_frames, ignore_index=True, sort=False)
                    image_only_mode = True
                    logger.info(
                        "  Materialised %d rows from %d image dataset(s) for image-only preprocessing",
                        len(full_df),
                        len(image_frames),
                    )
                else:
                    raise RuntimeError(
                        "Phase 3: image datasets found but no image paths could be materialised."
                    )

            # Drop columns with >50% NaN (artifacts of non-overlapping schemas)
            nan_ratio = full_df.isna().mean()
            high_nan_cols = nan_ratio[nan_ratio > 0.5].index.tolist()
            if high_nan_cols:
                logger.warning(
                    "  Dropping %d columns with >50%% NaN (non-overlapping schemas): %s",
                    len(high_nan_cols), high_nan_cols[:10],
                )
                full_df = full_df.drop(columns=high_nan_cols)

            total_samples: int = len(full_df)
            logger.info("  Materialised %d rows from %d dataset(s)", total_samples, len(frames))

            ctx = self._get_ctx()
            context_signals: Dict[str, Any] = {}
            predictability_scores: Dict[str, float] = {}
            drift_adjusted_predictability: Dict[str, float] = {}
            modality_presence: Dict[str, bool] = {}
            planner_drifted_features: List[str] = []
            global_schema_context: Dict[str, Any] = {}
            feature_intelligence_context: Dict[str, Any] = {}
            if ctx is not None:
                if hasattr(ctx, "get_preprocessing_signals"):
                    try:
                        context_signals = dict(ctx.get_preprocessing_signals() or {})
                    except Exception:
                        context_signals = {}
                if not context_signals:
                    context_signals = {
                        "global_schema": dict(getattr(ctx, "global_schema", {}) or {}),
                        "modality_presence": {
                            str(k): bool(v)
                            for k, v in dict(getattr(ctx, "modality_presence", {}) or {}).items()
                        },
                        "predictability_scores": {
                            str(k): float(v)
                            for k, v in dict(getattr(ctx, "predictability_scores", {}) or {}).items()
                            if isinstance(v, (int, float))
                        },
                        "drift_adjusted_predictability": {
                            str(k): float(v)
                            for k, v in dict(getattr(ctx, "drift_adjusted_predictability", {}) or {}).items()
                            if isinstance(v, (int, float))
                        },
                        "drifted_features": [
                            str(col) for col in list(getattr(ctx, "drifted_features", []) or [])
                        ],
                        "drift_feedback_applied": bool(getattr(ctx, "drift_feedback_applied", False)),
                        "training_fit_analysis": dict(getattr(ctx, "training_fit_analysis", {}) or {}),
                        "feature_intelligence": dict(getattr(ctx, "feature_intelligence", {}) or {}),
                        "encoder_plan": dict(getattr(ctx, "encoder_plan", {}) or {}),
                    }

                predictability_scores = {
                    str(k): float(v)
                    for k, v in dict(context_signals.get("predictability_scores", {}) or {}).items()
                    if isinstance(v, (int, float))
                }
                drift_adjusted_predictability = {
                    str(k): float(v)
                    for k, v in dict(context_signals.get("drift_adjusted_predictability", {}) or {}).items()
                    if isinstance(v, (int, float))
                }
                modality_presence = {
                    str(k): bool(v)
                    for k, v in dict(context_signals.get("modality_presence", {}) or {}).items()
                }
                planner_drifted_features = [
                    str(col) for col in list(context_signals.get("drifted_features", []) or [])
                ]
                global_schema_context = dict(context_signals.get("global_schema", {}) or {})
                feature_intelligence_context = dict(context_signals.get("feature_intelligence", {}) or {})

            per_dataset_hints: List[Dict[str, Any]] = []
            per_dataset_entries = list(schema_info.get("per_dataset", []) or [])
            for entry in per_dataset_entries:
                if isinstance(entry, dict):
                    raw_hints = entry.get("preprocessing_hints")
                    if isinstance(raw_hints, dict) and raw_hints:
                        per_dataset_hints.append(dict(raw_hints))

            merged_preprocessing_hints: Dict[str, Any] = {}
            for raw_hint in per_dataset_hints:
                for key, value in raw_hint.items():
                    if isinstance(value, dict):
                        bucket = dict(merged_preprocessing_hints.get(key, {}) or {})
                        bucket.update(dict(value))
                        merged_preprocessing_hints[key] = bucket
                    else:
                        merged_preprocessing_hints[key] = value

            merged_feature_intelligence: Dict[str, Any] = {}
            if isinstance(feature_intelligence_context, dict) and feature_intelligence_context:
                fi_entries = [
                    dict(value) for value in feature_intelligence_context.values()
                    if isinstance(value, dict)
                ]
                if fi_entries:
                    text_lengths = [
                        float(entry.get("avg_text_len"))
                        for entry in fi_entries
                        if isinstance(entry.get("avg_text_len"), (int, float))
                    ]
                    image_sizes = [
                        int(entry.get("image_dataset_size"))
                        for entry in fi_entries
                        if isinstance(entry.get("image_dataset_size"), (int, float))
                    ]
                    image_seps = [
                        float(entry.get("image_label_separability"))
                        for entry in fi_entries
                        if isinstance(entry.get("image_label_separability"), (int, float))
                    ]
                    image_balances = [
                        float(entry.get("image_class_balance"))
                        for entry in fi_entries
                        if isinstance(entry.get("image_class_balance"), (int, float))
                    ]
                    all_uncertainty: Dict[str, Any] = {}
                    all_hints: Dict[str, Any] = {}
                    text_task_type = None
                    for entry in fi_entries:
                        all_uncertainty.update(dict(entry.get("uncertainty_summary", {}) or {}))
                        all_hints.update(dict(entry.get("preprocessing_hints", {}) or {}))
                        if not text_task_type and entry.get("text_task_type"):
                            text_task_type = entry.get("text_task_type")
                    merged_feature_intelligence = {
                        "avg_text_len": max(text_lengths) if text_lengths else 0.0,
                        "image_dataset_size": max(image_sizes) if image_sizes else 0,
                        "image_label_separability": (
                            sum(image_seps) / len(image_seps) if image_seps else 0.0
                        ),
                        "image_class_balance": (
                            sum(image_balances) / len(image_balances) if image_balances else 0.0
                        ),
                        "uncertainty_summary": all_uncertainty,
                        "text_task_type": text_task_type,
                        "preprocessing_hints": all_hints,
                        "n_features": int(schema_info.get("total_feature_count", 0) or 0),
                        # Bug 8: propagate rich schema signals to preprocessors
                        "vocab_sizes": [
                            int((e.get("feature_signals") or {}).get("vocab_size", 0))
                            for e in fi_entries
                        ],
                        "language_ids": [
                            (e.get("feature_signals") or {}).get("language_id")
                            for e in fi_entries
                            if (e.get("feature_signals") or {}).get("language_id")
                        ],
                        "avg_tokens": [
                            float((e.get("feature_signals") or {}).get("avg_tokens_per_sample", 0))
                            for e in fi_entries
                        ],
                        "image_channels": list({
                            ch for e in fi_entries
                            for ch in ((e.get("feature_signals") or {}).get("channels") or [])
                        }),
                        "aspect_ratio_variances": [
                            float((e.get("feature_signals") or {}).get("aspect_ratio_variance", 0))
                            for e in fi_entries
                        ],
                    }

            preprocessing_plan = self.preprocessing_planner.create_plan(
                schema_info=schema_info,
                total_samples=total_samples,
                predictability_scores=predictability_scores,
                modality_presence=modality_presence,
                drift_adjusted_predictability=drift_adjusted_predictability,
                drifted_features=planner_drifted_features,
                global_schema=global_schema_context,
                preprocessing_hints=merged_preprocessing_hints,
                feature_intelligence=merged_feature_intelligence,
            )
            self.state.set_slot("preprocessing_plan", preprocessing_plan)

            # ----------------------------------------------------------------
            # 3  Target separation and encoding
            # ----------------------------------------------------------------
            from sklearn.preprocessing import LabelEncoder, StandardScaler as SS

            if image_only_mode and target_col == "Unknown" and "target" in full_df.columns:
                target_col = "target"

            if target_col != "Unknown" and target_col in full_df.columns:
                y_raw = full_df[target_col]
                feature_df = full_df.drop(columns=[target_col])
            else:
                if full_df.shape[1] < 2:
                    raise RuntimeError(
                        "Phase 3: target column is missing. "
                        "For image-only datasets provide labels (e.g. class-folder structure) "
                        "or include an explicit target column."
                    )
                logger.warning(
                    "  Target column '%s' not found – using last column as target", target_col
                )
                feature_df = full_df.iloc[:, :-1]
                y_raw = full_df.iloc[:, -1]
                target_col = full_df.columns[-1]

            if image_only_mode and "image_path" in feature_df.columns:
                global_modalities = list(schema_info.get("global_modalities", []) or [])
                if "image" not in global_modalities:
                    schema_info["global_modalities"] = sorted(global_modalities + ["image"])

                per_dataset = schema_info.get("per_dataset", [])
                if not isinstance(per_dataset, list) or not per_dataset:
                    per_dataset = [{}]
                    schema_info["per_dataset"] = per_dataset
                first_entry = per_dataset[0]
                if not isinstance(first_entry, dict):
                    first_entry = {}
                    per_dataset[0] = first_entry
                detected_cols = first_entry.setdefault("detected_columns", {})
                image_bucket = detected_cols.setdefault("image", [])
                if "image_path" not in image_bucket:
                    image_bucket.append("image_path")

            # Drop rows with NaN targets (from non-overlapping dataset concat)
            nan_target_mask = y_raw.isna()
            if nan_target_mask.any():
                n_nan = int(nan_target_mask.sum())
                logger.warning(
                    "  Dropping %d rows with NaN target values", n_nan
                )
                valid_idx = ~nan_target_mask
                y_raw = y_raw[valid_idx].reset_index(drop=True)
                feature_df = feature_df[valid_idx].reset_index(drop=True)

            if problem_type == "multilabel_classification":
                # Target values are dict-like strings, e.g. "{'NORM': 100.0, 'SR': 0.0}"
                # Parse into multi-hot float vectors.
                import ast

                parsed = y_raw.astype(str).apply(
                    lambda v: ast.literal_eval(v) if v.startswith("{") else {v: 1.0}
                )
                all_labels = sorted({k for d in parsed for k in d})
                label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}
                n_classes = len(all_labels)

                multi_hot = np.zeros((len(parsed), n_classes), dtype=np.float32)
                for row_i, d in enumerate(parsed):
                    for lbl, val in d.items():
                        if lbl in label_to_idx:
                            multi_hot[row_i, label_to_idx[lbl]] = float(val) / 100.0

                targets = torch.tensor(multi_hot, dtype=torch.float32)
                self.fitted_transformers["target_encoder"] = {
                    "type": "multilabel",
                    "label_to_idx": label_to_idx,
                    "all_labels": all_labels,
                }
                logger.info("  Target encoder: Multilabel  classes=%d  labels=%s", n_classes, all_labels)

            elif problem_type.startswith("classification"):
                le = LabelEncoder()
                y_encoded = le.fit_transform(y_raw.astype(str))
                targets = torch.tensor(y_encoded, dtype=torch.long)
                self.fitted_transformers["target_encoder"] = le
                n_classes = len(le.classes_)
                logger.info("  Target encoder: LabelEncoder  classes=%d", n_classes)
            else:
                ss = SS()
                y_values = y_raw.values.reshape(-1, 1).astype(float)
                y_scaled = ss.fit_transform(y_values).ravel()
                targets = torch.tensor(y_scaled, dtype=torch.float32)
                self.fitted_transformers["target_encoder"] = ss
                logger.info("  Target encoder: StandardScaler (regression)")

            # ----------------------------------------------------------------
            # 4  Identify column groups from schema
            #    Union detected columns across ALL per-dataset entries so that
            #    text/image columns from datasets 2+ are not silently treated
            #    as tabular features.
            # ----------------------------------------------------------------
            per_ds = schema_info.get("per_dataset", [{}])
            all_text_cols: set = set()
            all_image_cols: set = set()
            for ds_entry in per_ds:
                detected = ds_entry.get("detected_columns", {})
                all_text_cols.update(detected.get("text", []))
                all_image_cols.update(detected.get("image", []))
            text_cols = [c for c in all_text_cols if c in feature_df.columns]
            image_cols = [c for c in all_image_cols if c in feature_df.columns]
            tabular_cols = [
                c for c in feature_df.columns
                if c not in text_cols and c not in image_cols
            ]
            id_like_cols = {
                str(c)
                for ds_entry in per_ds
                if isinstance(ds_entry, dict)
                for c in (ds_entry.get("id_like_columns") or [])
            }
            if tabular_cols:
                usable_tabular_cols = [
                    c for c in tabular_cols
                    if c != target_col
                    and c not in id_like_cols
                    and str(c).lower() not in {"id", "idx", "index", "row_id", "uuid", "guid"}
                    and not any(tok in str(c).lower() for tok in ("path", "file", "filename", "url", "uri"))
                ]
                if not usable_tabular_cols:
                    logger.info(
                        "Phase 3: tabular auto-skipped - cols %s are all IDs/path-like or target '%s'",
                        tabular_cols,
                        target_col,
                    )
                    tabular_cols = []
                    schema_info["global_modalities"] = [
                        m for m in list(schema_info.get("global_modalities", []) or [])
                        if m != "tabular"
                    ]
                    if self.config.modalities:
                        self.config.modalities = [m for m in self.config.modalities if m != "tabular"]
                    if ctx is not None:
                        ctx.active_modalities = [
                            m for m in list(getattr(ctx, "active_modalities", []) or schema_info.get("global_modalities", []))
                            if m != "tabular"
                        ]
                        ctx.eligible_modalities = list(ctx.active_modalities)
                        ctx.excluded_modalities = {
                            **dict(getattr(ctx, "excluded_modalities", {}) or {}),
                            "tabular": "id_only_or_target_only",
                        }
                        if isinstance(getattr(ctx, "global_schema", None), dict):
                            ctx.global_schema["global_modalities"] = list(schema_info.get("global_modalities", []))
                            ctx.global_schema["active_modalities"] = list(ctx.active_modalities)
                            ctx.global_schema["excluded_modalities"] = dict(ctx.excluded_modalities)

            adaptive_tabular_config: Dict[str, Any] = {}
            drifted_features: List[str] = list(planner_drifted_features)
            if ctx is not None and tabular_cols:
                try:
                    adaptive_engine = AdaptivePreprocessingEngine(ctx)
                    adaptive_tabular_config = adaptive_engine.build_tabular_config(feature_df[tabular_cols])
                    if not drifted_features:
                        drifted_features = list(getattr(ctx, "drifted_features", []) or [])

                    weak_modalities = adaptive_engine.get_weak_modalities()
                    if weak_modalities:
                        preprocessing_plan["weak_modalities"] = weak_modalities
                    preprocessing_plan["fusion_recommendation"] = adaptive_engine.get_fusion_recommendation()

                    tab_plan = dict(preprocessing_plan.get("tabular", {}) or {})
                    tab_plan.update(adaptive_tabular_config)
                    preprocessing_plan["tabular"] = tab_plan
                except Exception as adaptive_exc:
                    logger.warning("Phase 3: adaptive preprocessing engine unavailable: %s", adaptive_exc)

            validation_plan = {
                "modality": {
                    "tabular": {
                        "columns": tabular_cols,
                        "imputer_strategy": (preprocessing_plan.get("tabular", {}) or {}).get(
                            "imputer_strategy", "median"
                        ),
                    },
                    "text": {
                        "columns": text_cols,
                        "max_length": (preprocessing_plan.get("text", {}) or {}).get(
                            "max_length", 128
                        ),
                    },
                    "image": {
                        "columns": image_cols,
                        "image_size": (preprocessing_plan.get("image", {}) or {}).get(
                            "target_size", [224, 224]
                        ),
                    },
                },
                "feature_selection": {
                    "top_k": max(1, min(512, len(tabular_cols) or 1)),
                },
            }

            validation_report: Dict[str, Any] = {
                "valid": True,
                "warnings": [],
                "errors": [],
                "checks_passed": 0,
                "checks_total": 0,
            }
            try:
                validation_report = PreprocessingValidator().validate_plan(
                    validation_plan,
                    schema_info,
                    dataset_shape=feature_df.shape,
                )
            except PreprocessingValidationError as validation_exc:
                raise RuntimeError(
                    f"Phase 3 preprocessing validation failed: {validation_exc}"
                ) from validation_exc

            # ----------------------------------------------------------------
            # 5  Fit modality preprocessors
            # ----------------------------------------------------------------
            text_prep = None
            image_prep = None
            tabular_prep = None
            output_shapes: Dict[str, Any] = {}
            preprocessing_stages = []

            # Guard condition: use cols presence only — not schema_info.global_modalities.
            # schema_info can be stale or incorrect after overrides; the column arrays
            # (tabular_cols/text_cols/image_cols) are built directly from detected_columns
            # so they are the authoritative source.
            if tabular_cols:
                existing_tabular = self.fitted_transformers.get("tabular")
                tabular_plan = dict(preprocessing_plan.get("tabular", {}) or {})
                if existing_tabular is not None:
                    tabular_prep = existing_tabular
                    tabular_prep.configure(tabular_plan)
                    if drifted_features:
                        setattr(tabular_prep, "_drifted_features", list(drifted_features))
                    _ = tabular_prep.transform(feature_df[tabular_cols])
                    logger.info(
                        "  Phase 3: reusing pre-fitted TabularPreprocessor (output_dim=%d)",
                        tabular_prep.get_output_dim() if hasattr(tabular_prep, "get_output_dim") else -1,
                    )
                else:
                    try:
                        tabular_prep = TabularPreprocessor(
                            adaptive_config=tabular_plan,
                            drifted_features=drifted_features,
                        )
                        tabular_prep.configure(tabular_plan)
                        _ = tabular_prep.fit_transform(feature_df[tabular_cols])
                        self.fitted_transformers["tabular"] = tabular_prep
                    except ValueError as _no_cols_exc:
                        _remaining_mods = [
                            m for m in schema_info.get("global_modalities", []) if m != "tabular"
                        ]
                        logger.warning(
                            "Phase 3: tabular modality auto-skipped — candidate cols %s were all "
                            "identified as IDs/paths/high-cardinality and dropped. "
                            "Reason: %s. "
                            "Continuing with modalities: %s. "
                            "If tabular features are expected, verify no genuine feature columns "
                            "share names with ID/path columns.",
                            tabular_cols, _no_cols_exc, _remaining_mods,
                        )
                        tabular_prep = None
                        tabular_cols = []
                        # Prune "tabular" from both modality trackers so Phase 4/5 don't
                        # waste JIT VRAM budget or Optuna HPO trials on a missing modality.
                        _gm = list(schema_info.get("global_modalities", []) or [])
                        if "tabular" in _gm:
                            schema_info["global_modalities"] = [m for m in _gm if m != "tabular"]
                        if self.config.modalities and "tabular" in self.config.modalities:
                            self.config.modalities = [
                                m for m in self.config.modalities if m != "tabular"
                            ]
                if tabular_prep is not None:
                    output_dim = tabular_prep.get_output_dim()
                    # Guard against all-columns-dropped silent failure
                    if output_dim == 0:
                        raise RuntimeError(
                            "Phase 3: TabularPreprocessor produced 0 output features. "
                            "All tabular columns were filtered out (too many unique values, "
                            "path-like strings, or excessive missing values). "
                            "Suggestions: (1) Check your dataset for ID or path columns, "
                            "(2) Reduce _NEAR_UNIQUE_RATIO threshold via /configure, "
                            "(3) Manually select feature columns before ingestion."
                        )
                    output_shapes["tabular"] = f"(N, {output_dim})"
                    preprocessing_stages.append({
                        "stage": "tabular_preprocessing",
                        "status": "success",
                        "output_shape": output_shapes["tabular"],
                    })
                    logger.info("  Tabular preprocessor fitted: output_dim=%d", output_dim)

                # Part A.4 — when ULA fusion is selected, build a TabularFeatureTokenizer
                # so tabular features become per-feature token sequences (N,F,token_dim)
                # instead of a flat pooled vector, enabling cross-modal attention in ULA.
                _recommended_fusion = (
                    self.state.get_slot("schema_derived_fusion")
                    if hasattr(self, "state") and hasattr(self.state, "get_slot")
                    else None
                ) or self.config.__dict__.get("fusion_strategy", "concatenation")
                if tabular_prep is not None and str(_recommended_fusion).lower() in ("ula", "unified_latent", "unified_latent_alignment", "omnimodal"):
                    try:
                        from preprocessing.tabular_preprocessor import TabularFeatureTokenizer
                        _ula_token_dim = int(
                            self.config.__dict__.get("ula_latent_dim", None) or 256
                        )
                        _tab_tok = TabularFeatureTokenizer(
                            n_features=output_dim, token_dim=_ula_token_dim
                        )
                        self.fitted_transformers["tabular_tokenizer"] = _tab_tok
                        logger.info(
                            "  ULA TabularFeatureTokenizer built: n_features=%d token_dim=%d",
                            output_dim, _ula_token_dim,
                        )
                    except Exception as _tok_exc:
                        logger.warning("TabularFeatureTokenizer init failed: %s", _tok_exc)

            if text_cols:
                text_prep = TextPreprocessor()
                _text_plan = dict(preprocessing_plan.get("text") or {})
                _text_plan["feature_intelligence"] = merged_feature_intelligence
                _text_plan["text_task_type"] = merged_feature_intelligence.get("text_task_type")
                text_prep.configure(_text_plan)
                self.fitted_transformers["text"] = text_prep
                output_shapes["text"] = f"(N, {text_prep.max_length}) per key"
                preprocessing_stages.append({
                    "stage": "text_preprocessing",
                    "status": "success",
                    "output_shape": output_shapes["text"],
                })
                logger.info("  Text preprocessor initialised (lazy tokeniser)")

            if image_cols:
                image_prep = ImagePreprocessor()
                _img_plan = dict(preprocessing_plan.get("image") or {})
                _img_plan["feature_intelligence"] = merged_feature_intelligence
                _img_plan["dataset_size"] = merged_feature_intelligence.get("image_dataset_size", 0)
                _img_plan["label_separability"] = merged_feature_intelligence.get("image_label_separability", 0.5)
                _img_plan["class_balance"] = merged_feature_intelligence.get("image_class_balance", 0.5)
                image_prep.configure(_img_plan)
                self.fitted_transformers["image"] = image_prep
                _h, _w = image_prep.target_size
                output_shapes["image"] = f"(N, 3, {_h}, {_w})"
                preprocessing_stages.append({
                    "stage": "image_preprocessing",
                    "status": "success",
                    "output_shape": output_shapes["image"],
                })
                logger.info("  Image preprocessor initialised")

            consistency_schema = dict(schema_info)
            consistency_schema["global_modalities"] = [
                modality
                for modality, columns in (
                    ("tabular", tabular_cols),
                    ("text", text_cols),
                    ("image", image_cols),
                )
                if columns
            ]
            schema_info["global_modalities"] = list(consistency_schema["global_modalities"])
            if ctx is not None:
                ctx.active_modalities = list(consistency_schema["global_modalities"])
                ctx.eligible_modalities = list(consistency_schema["global_modalities"])

            try:
                validate_preprocessor_consistency(
                    tabular_prep,
                    text_prep,
                    image_prep,
                    consistency_schema,
                )
            except PreprocessingValidationError as consistency_exc:
                raise RuntimeError(
                    f"Phase 3 preprocessor consistency failed: {consistency_exc}"
                ) from consistency_exc

            # ----------------------------------------------------------------
            # 6  Build MultimodalPyTorchDataset
            #    Two variants are constructed on the same underlying DataFrame:
            #    - train_torch_dataset: apply_augmentation=True → training rows
            #      receive RandomHorizontalFlip / RandomRotation / ColorJitter
            #      before Resize+Normalize.
            #    - val_torch_dataset:   apply_augmentation=False → validation
            #      rows receive only Resize+Normalize (deterministic inputs).
            #    Phase 5 routes train row indices to the augmented dataset and
            #    val row indices to the clean dataset via torch.utils.data.Subset.
            # ----------------------------------------------------------------
            # ── Pre-flight image path validation ─────────────────────────
            # Check that image files are actually accessible before building
            # the dataset.  Silent zero-fill during training is impossible to
            # diagnose — fail loudly here instead.
            if image_cols and image_prep is not None:
                _probe_paths: list = []
                for _ic in image_cols:
                    if _ic not in feature_df.columns:
                        continue
                    _probe_paths.extend(
                        feature_df[_ic].dropna().astype(str).head(20).tolist()
                    )
                _probe_paths = [p for p in _probe_paths if p.strip() and
                                p.strip().lower() not in {"nan","none","null",""}]
                if _probe_paths:
                    from pathlib import Path as _Path
                    _n_ok = sum(1 for p in _probe_paths if _Path(p).is_file())
                    _load_rate = _n_ok / len(_probe_paths)
                    if _load_rate == 0.0:
                        raise RuntimeError(
                            f"Phase 3: Image pre-flight check FAILED — 0/{len(_probe_paths)} "
                            f"sampled image paths are accessible on this server.\n"
                            f"  Sample path: {_probe_paths[0]}\n"
                            "Image columns reference local paths on your machine. "
                            "Copy the images to the server or provide an accessible path. "
                            "To train without images, remove the image column from your CSV."
                        )
                    elif _load_rate < 0.5:
                        logger.warning(
                            "Phase 3: Image pre-flight WARNING — only %d/%d (%.0f%%) sampled "
                            "image paths are accessible. Training will proceed but image "
                            "embeddings will be zero for missing files. "
                            "Check that image paths are accessible from the server.",
                            _n_ok, len(_probe_paths), _load_rate * 100,
                        )
                    else:
                        logger.info(
                            "  Image pre-flight: %d/%d paths accessible (%.0f%%)",
                            _n_ok, len(_probe_paths), _load_rate * 100,
                        )

            _dataset_kwargs = dict(
                df=feature_df,
                targets=targets,
                schema_info=schema_info,
                tabular_preprocessor=tabular_prep,
                text_preprocessor=text_prep,
                image_preprocessor=image_prep,
            )
            self.train_torch_dataset = MultimodalPyTorchDataset(
                **_dataset_kwargs, apply_augmentation=True
            )
            self.val_torch_dataset = MultimodalPyTorchDataset(
                **_dataset_kwargs, apply_augmentation=False
            )
            # Backward-compat alias points to the clean (no-aug) variant.
            self.torch_dataset = self.val_torch_dataset
            logger.info(
                "  MultimodalPyTorchDataset created: %d samples "
                "(train=augmented, val/test=clean)",
                len(self.train_torch_dataset),
            )

            # ----------------------------------------------------------------
            # 7  Store results
            # ----------------------------------------------------------------
            elapsed = time.time() - phase_start
            results: Dict[str, Any] = {
                "preprocessing_stages": preprocessing_stages,
                "total_samples": total_samples,
                "output_shapes": output_shapes,
                "target_column": target_col,
                "problem_type": problem_type,
                "active_modalities": [
                    modality
                    for modality, columns in (
                        ("tabular", tabular_cols),
                        ("text", text_cols),
                        ("image", image_cols),
                    )
                    if columns
                ],
                "excluded_modalities": (
                    dict(getattr(ctx, "excluded_modalities", {}) or {})
                    if ctx is not None else {}
                ),
                "text_columns": text_cols,
                "image_columns": image_cols,
                "tabular_columns": tabular_cols,
                "preprocessing_plan": preprocessing_plan,
                "validation": validation_report,
                "duration_seconds": elapsed,
            }

            if ctx is not None:
                self._sync_preprocessing_contract_to_context(
                    preprocessing_plan=preprocessing_plan,
                    validation_report=validation_report,
                    context_signals=context_signals,
                    total_samples=total_samples,
                    adaptive_tabular_config=adaptive_tabular_config,
                    drifted_features=drifted_features,
                )

            logger.info("\nPhase 3 Summary:")
            logger.info("  Stages     : %d", len(preprocessing_stages))
            logger.info("  Samples    : %d", total_samples)
            logger.info("  Duration   : %.2fs", elapsed)

            self.phase_results[Phase.PREPROCESSING] = results
            self.state.set_slot("phase3_preprocessing", results)
            self.state.set_phase_timing("PREPROCESSING", elapsed)
            self._record_phase_timing_in_context("PREPROCESSING", elapsed)
            self.current_phase = Phase.MODEL_SELECTION

        except Exception as exc:
            logger.error("Phase 3 failed: %s", str(exc))
            raise
    
    def _execute_phase_4_model_selection(self) -> None:
        """
        Phase 4: Model Selection – delegate to ``AdvancedModelSelector`` and
        store Optuna HPO search spaces for Phase 5.

        Inputs (from earlier phases)
        ----------------------------
        Phase 2 results : ``global_modalities``, ``global_problem_type``
        Phase 3 results : ``total_samples``, ``text_columns``

        Outputs (stored in ``phase_results[Phase.MODEL_SELECTION]``)
        -------------------------------------------------------------
        ``image_encoder``   : selected tier key or None
        ``text_encoder``    : selected tier key or None
        ``tabular_encoder`` : selected tier key or None
        ``fusion_strategy`` : static choice
        ``batch_size``      : fixed PDF heuristic value
        ``epochs``          : midpoint of HPO epoch range (Phase 5 refines)
        ``learning_rate``   : midpoint of HPO LR range  (Phase 5 refines)
        ``hpo_space``       : full Optuna search space dict
        ``rationale``       : human-readable selection notes
        ``hardware_info``   : GPU/CPU snapshot
        """
        # Short-circuit if model selection was injected via inject_external_model_selection()
        if Phase.MODEL_SELECTION in self.phase_results:
            injected = self.phase_results[Phase.MODEL_SELECTION]
            logger.info(
                "Phase 4: SKIPPED - using injected model selection (%s)",
                injected.get("selected_model", injected.get("name", "?")),
            )
            self.current_phase = Phase.TRAINING
            return

        logger.info("\n" + "=" * 80)
        logger.info("PHASE 4: MODEL SELECTION")
        logger.info("=" * 80)

        self._enforce_session_context("Phase 4")

        _ctx_for_validation = self._get_ctx()
        if _ctx_for_validation is not None:
            ContextValidator.require_schema(_ctx_for_validation, phase="model_selection")
            ContextValidator.require_modality_consistency(
                _ctx_for_validation,
                self.config.modalities,
                phase="model_selection",
            )

        phase_start = time.time()

        try:
            from automl.advanced_selector import (
                AdvancedModelSelector,
                IMAGE_ENCODERS,
                TEXT_ENCODERS,
                TABULAR_ENCODERS,
            )

            # ----------------------------------------------------------------
            # 1  Pull context from upstream phases
            # ----------------------------------------------------------------
            schema_info: Dict[str, Any] = self.phase_results.get(
                Phase.SCHEMA_DETECTION, {}
            )
            prep_info: Dict[str, Any] = self.phase_results.get(
                Phase.PREPROCESSING, {}
            )

            from models.fusion import select_fusion_strategy

            schema_derived_fusion = select_fusion_strategy(schema_info)

            # Use multimodal_signals (complementarity_score, alignment_strength)
            # to upgrade the fusion strategy when signals are informative.
            _mm_signals = schema_info.get("multimodal_signals", {}) or {}
            _complementarity = float(_mm_signals.get("complementarity_score", 0.0))
            _alignment = float(_mm_signals.get("alignment_strength", 0.0))
            _n_mods = len(schema_info.get("global_modalities", []))

            if _n_mods >= 2 and schema_derived_fusion not in ("ula", "gated", "fusemoe"):
                if _complementarity > 0.6 and _alignment > 0.45:
                    # Diverse, well-aligned modalities — ULA cross-modal attention
                    schema_derived_fusion = "ula"
                    logger.info(
                        "Phase 4: upgrading fusion → ULA "
                        "(complementarity=%.3f alignment=%.3f)",
                        _complementarity, _alignment,
                    )
                elif _complementarity < 0.25:
                    # Conflicting modalities — use gates to suppress noise
                    schema_derived_fusion = "gated"
                    logger.info(
                        "Phase 4: upgrading fusion → GatedFusion "
                        "(complementarity=%.3f — modalities likely conflict)",
                        _complementarity,
                    )

            # Derive ULA latent_dim from alignment_strength
            if schema_derived_fusion == "ula":
                if _alignment > 0.7:
                    _ula_latent_dim = 128
                elif _alignment > 0.4:
                    _ula_latent_dim = 256
                else:
                    _ula_latent_dim = 512
                logger.info("Phase 4: ULA latent_dim=%d (alignment=%.3f)", _ula_latent_dim, _alignment)
                self.state.set_slot("ula_latent_dim", _ula_latent_dim)
            else:
                _ula_latent_dim = 256  # default

            logger.info(
                "Phase 4: schema-derived fusion strategy = %s",
                schema_derived_fusion,
            )
            self.state.set_slot("schema_derived_fusion", schema_derived_fusion)

            # Respect user fusion overrides: if the user explicitly locked the
            # fusion strategy via /override-fusion, honour that over any
            # schema-derived routing. Otherwise apply schema intelligence.
            ctx = self._get_ctx()
            if ctx is not None:
                ctx_fusion = getattr(ctx, "fusion_strategy", None)
                ctx_locked = bool(getattr(ctx, "fusion_policy_locked", False))
                ctx_source = getattr(ctx, "fusion_policy_source", "")
                _is_user_override = ctx_locked and ctx_source == "user_override"
                if ctx_fusion and _is_user_override:
                    schema_derived_fusion = str(ctx_fusion)
                    self.state.set_slot("schema_derived_fusion", schema_derived_fusion)
                    logger.info(
                        "Phase 4: user fusion override LOCKED to '%s' -- "
                        "schema-derived routing suppressed",
                        schema_derived_fusion,
                    )
                # Bug 3 fix: removed soft-suggestion `elif ctx_fusion and not _is_user_override`
                # block. That block let stale ctx values (e.g. "concatenation" from a prior run)
                # silently overwrite the fresh schema-derived fusion (e.g. "ula"). Schema
                # intelligence already computed the correct strategy — don't corrupt it with
                # historical state that wasn't explicitly locked by the user.

            # xs3_confidence_gap gate: ambiguous target -> restrict to tabular for safety
            xs3_gap = 0.0
            for ds in schema_info.get("per_dataset", []):
                if not isinstance(ds, dict):
                    continue
                reasoning = ds.get("reasoning", {})
                if isinstance(reasoning, dict):
                    g = float(reasoning.get("xs3_confidence_gap", 0.0) or 0.0)
                    xs3_gap = max(xs3_gap, g)

            if xs3_gap < 0.15:
                logger.warning(
                    "Phase 4: xs3_confidence_gap=%.3f < 0.15 - "
                    "target is ambiguous; restricting modalities to tabular.",
                    xs3_gap,
                )
                self.config.modalities = ["tabular"]

            modalities: List[str] = schema_info.get(
                "global_modalities", self.config.modalities
            )

            if ctx is not None:
                try:
                    ctx_active = ctx.get_active_modalities()
                except Exception:
                    ctx_active = []
                valid_modalities = [
                    m for m in (ctx_active or []) if m in {"tabular", "text", "image"}
                ]
                if valid_modalities:
                    modalities = valid_modalities
                    self.config.modalities = list(valid_modalities)
                    logger.info(
                        "Phase 4: using ExecutionContext active modalities: %s",
                        valid_modalities,
                    )

            if xs3_gap < 0.15:
                modalities = ["tabular"]
            problem_type: str = schema_info.get(
                "global_problem_type", self.config.problem_type
            )
            _user_fusion_override_active = bool(
                ctx is not None
                and bool(getattr(ctx, "fusion_policy_locked", False))
                and getattr(ctx, "fusion_policy_source", "") == "user_override"
            )
            if (
                "image" in set(modalities)
                and "text" in set(modalities)
                and not _user_fusion_override_active
            ):
                schema_derived_fusion = "ula"
                self.state.set_slot("schema_derived_fusion", schema_derived_fusion)
                logger.info("Phase 4: active text+image contract forces ULA fusion")
            dataset_size: int = prep_info.get("total_samples", 10_000)

            # Estimate avg_tokens from column count (no materialisation needed)
            text_cols: List[str] = prep_info.get("text_columns", [])
            image_cols: List[str] = prep_info.get("image_columns", [])
            tabular_cols: List[str] = prep_info.get("tabular_columns", [])
            avg_tokens: int = 128  # conservative default; Phase 5 can override
            if ctx is not None:
                try:
                    fi = dict(getattr(ctx, "feature_intelligence", {}) or {})
                    text_lengths = [
                        float(ds.get("avg_text_len"))
                        for ds in fi.values()
                        if isinstance(ds, dict) and isinstance(ds.get("avg_text_len"), (int, float))
                    ]
                    if text_lengths:
                        approx_tokens = int(max(8, min(4096, round(max(text_lengths) / 4.0))))
                        avg_tokens = approx_tokens
                except Exception:
                    pass

            dataset_meta = {
                "num_rows": int(dataset_size),
                "num_cols": int(len(text_cols) + len(image_cols) + len(tabular_cols)),
                "modalities": list(modalities),
                "target_type": "regression" if "regression" in problem_type else "classification",
            }

            logger.info(
                "  modalities=%s  problem=%s  dataset_size=%d",
                modalities, problem_type, dataset_size,
            )

            latency_budget_ms = None
            memory_budget_mb = None
            if ctx is not None:
                latency_budget_ms = getattr(ctx, "latency_budget_ms", None)
                memory_budget_mb = getattr(ctx, "memory_budget_mb", None)
                try:
                    from core.orchestrator import orchestrator as _metadata_orchestrator
                    _metadata_orchestrator.run_architecture_selection(ctx)
                except Exception as arch_exc:
                    logger.debug("Phase 4: architecture routing refresh skipped: %s", arch_exc)

            # ----------------------------------------------------------------
            # 2  Run AdvancedModelSelector
            # ----------------------------------------------------------------
            selector = AdvancedModelSelector()

            def _name(catalogue: Dict[str, Dict[str, Any]], tier: Optional[str]) -> Optional[str]:
                return catalogue[tier]["name"] if tier and tier in catalogue else None

            def _tier_from_name(catalogue: Dict[str, Dict[str, Any]], encoder_name: Optional[str]) -> Optional[str]:
                if not encoder_name:
                    return None
                target = str(encoder_name).strip().lower()
                for tier_key, spec in catalogue.items():
                    name = str(spec.get("name", "")).strip().lower()
                    if name == target:
                        return str(tier_key)
                return None

            def _extract_tabular_probe_arrays() -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
                dataset_obj = self.train_torch_dataset or self.torch_dataset
                if dataset_obj is None:
                    return None, None

                tab_block = getattr(dataset_obj, "_tabular_array", None)
                targets = getattr(dataset_obj, "targets", None)
                if tab_block is None or targets is None:
                    return None, None

                try:
                    probe_x = (
                        tab_block.detach().cpu().numpy()
                        if hasattr(tab_block, "detach")
                        else np.asarray(tab_block)
                    )
                    probe_y = (
                        targets.detach().cpu().numpy()
                        if hasattr(targets, "detach")
                        else np.asarray(targets)
                    )

                    if probe_x.ndim == 1:
                        probe_x = probe_x.reshape(-1, 1)

                    n_rows = min(len(probe_x), len(probe_y))
                    if n_rows <= 1:
                        return None, None

                    return probe_x[:n_rows], probe_y[:n_rows]
                except Exception as probe_exc:
                    logger.warning(
                        "Phase 4: failed to materialise tabular probe arrays: %s",
                        probe_exc,
                    )
                    return None, None

            probe_x, probe_y = _extract_tabular_probe_arrays()
            predictability_scores = None
            if ctx is not None:
                if hasattr(ctx, "get_effective_predictability_scores"):
                    try:
                        predictability_scores = dict(
                            ctx.get_effective_predictability_scores()
                        )
                    except Exception:
                        predictability_scores = None
                if not predictability_scores:
                    predictability_scores = dict(
                        getattr(ctx, "predictability_scores", {}) or {}
                    )
            recommendations = selector.recommend_models(
                problem_type=problem_type,
                modalities=modalities,
                dataset_size=dataset_size,
                avg_tokens=avg_tokens,
                tabular_X=probe_x,
                tabular_y=probe_y,
                latency_budget_ms=latency_budget_ms,
                memory_budget_mb=memory_budget_mb,
                predictability_scores=predictability_scores,
            )

            if recommendations and isinstance(recommendations[0], dict):
                best_model: Dict[str, Any] = dict(recommendations[0])
            else:
                # Defensive fallback: keep legacy selector path if API-style
                # recommendations are unavailable for any reason.
                result = selector.select_models(
                    problem_type=problem_type,
                    modalities=modalities,
                    dataset_size=dataset_size,
                    avg_tokens=avg_tokens,
                    dataset_meta=dataset_meta,
                    latency_budget_ms=latency_budget_ms,
                    memory_budget_mb=memory_budget_mb,
                    predictability_scores=predictability_scores,
                )
                best_model = {
                    "name": selector._build_model_name(
                        result.image_encoder,
                        result.text_encoder,
                        result.tabular_encoder,
                    ),
                    "image_encoder": _name(IMAGE_ENCODERS, result.image_encoder),
                    "text_encoder": _name(TEXT_ENCODERS, result.text_encoder),
                    "tabular_encoder": _name(TABULAR_ENCODERS, result.tabular_encoder),
                    "fusion_strategy": result.fusion_strategy,
                    "batch_size": result.batch_size,
                    "hpo_space": dict(result.hpo_space),
                    "rationale": dict(result.rationale),
                    "hardware_info": dict(result.hardware_info),
                    "meta_context": list(result.meta_context),
                    "eligible_modalities": list(result.eligible_modalities),
                    "excluded_modalities": dict(result.excluded_modalities),
                    "tier": "primary",
                }
                recommendations = [best_model]

            # Keep schema-derived/context-derived fusion as final authority.
            resolved_fusion = schema_derived_fusion or best_model.get("fusion_strategy")
            if resolved_fusion:
                best_model["fusion_strategy"] = str(resolved_fusion)

            # Normalize probe diagnostics into the same fields used by the API
            # /select-model contract.
            tabular_probe_scores = best_model.get("tabular_probe_scores")
            if not tabular_probe_scores:
                tabular_probe_scores = (
                    (best_model.get("probe_scores") or {}).get("tabular")
                )
            if isinstance(tabular_probe_scores, dict) and tabular_probe_scores:
                score_map: Dict[str, float] = {
                    model_name: float(details.get("val_score", 0.0) or 0.0)
                    for model_name, details in tabular_probe_scores.items()
                    if isinstance(details, dict)
                }
                top_probe_model = max(score_map, key=score_map.get) if score_map else None
                top_probe_score = float(score_map[top_probe_model]) if top_probe_model else None

                if top_probe_model is not None:
                    best_model.setdefault("tabular_probe_top_model", str(top_probe_model))
                if isinstance(top_probe_score, (int, float)):
                    best_model.setdefault("quick_probe_score", float(top_probe_score))
                    best_model.setdefault("probe_score", float(top_probe_score))

                complexity = None
                confidence = None
                if probe_x is not None and probe_y is not None:
                    try:
                        from automl.candidate_selector import CandidateSelector

                        _probe_selector = CandidateSelector()
                        probe_y_arr = np.asarray(probe_y)
                        if probe_y_arr.ndim > 1 and probe_y_arr.shape[1] > 1:
                            probe_y_norm = np.argmax(probe_y_arr, axis=1)
                        else:
                            probe_y_flat = probe_y_arr.ravel()
                            if probe_y_flat.dtype.kind in ("U", "S", "O"):
                                probe_y_norm = pd.factorize(probe_y_flat)[0]
                            else:
                                try:
                                    probe_y_norm = probe_y_flat.astype(int)
                                except Exception:
                                    probe_y_norm = pd.factorize(probe_y_flat)[0]

                        complexity = _probe_selector.compute_data_complexity(
                            np.asarray(probe_x),
                            np.asarray(probe_y_norm),
                        )
                        confidence = _probe_selector.compute_selection_confidence(score_map)
                    except Exception as probe_meta_exc:
                        logger.debug(
                            "Phase 4: probe metadata computation skipped: %s",
                            probe_meta_exc,
                        )

                selection_metadata = dict(best_model.get("selection_metadata") or {})
                selection_metadata.update({
                    "probe_method": "tabular_3fold_cv",
                    "top_probe_model": top_probe_model,
                    "top_probe_score": top_probe_score,
                    "probe_scores": tabular_probe_scores,
                    "data_complexity": complexity,
                    "selection_confidence": (
                        round(float(confidence), 4)
                        if isinstance(confidence, (int, float))
                        else None
                    ),
                })
                best_model["selection_metadata"] = selection_metadata
                best_model["probe_scores"] = {"tabular": tabular_probe_scores}
                existing_ranked_tabular = (
                    (best_model.get("ranked_candidates") or {}).get("tabular")
                )
                if isinstance(existing_ranked_tabular, list) and existing_ranked_tabular:
                    best_model["ranked_candidates"] = {
                        "tabular": list(existing_ranked_tabular)
                    }
                else:
                    best_model["ranked_candidates"] = {
                        "tabular": sorted(
                            [
                                {
                                    "name": model_name,
                                    "val_score": float(details.get("val_score", 0.0) or 0.0),
                                    "latency_ms": float(details.get("latency_ms", 0.0) or 0.0),
                                    "uncertainty": float(details.get("uncertainty", 0.0) or 0.0),
                                    "confidence": details.get("confidence"),
                                }
                                for model_name, details in tabular_probe_scores.items()
                                if isinstance(details, dict)
                            ],
                            key=lambda row: row.get("val_score", 0.0),
                            reverse=True,
                        )
                    }

            best_model["selection_contract_version"] = "model_selection.v2"
            recommendations[0] = best_model

            best_hpo_space: Dict[str, Any] = dict(best_model.get("hpo_space") or {})
            batch_size = int(best_model.get("batch_size") or 16)
            meta_context = list(best_model.get("meta_context") or [])

            img_name = best_model.get("image_encoder")
            txt_name = best_model.get("text_encoder")
            tab_name = best_model.get("tabular_encoder")

            img_tier = _tier_from_name(IMAGE_ENCODERS, img_name)
            txt_tier = _tier_from_name(TEXT_ENCODERS, txt_name)
            tab_tier = _tier_from_name(TABULAR_ENCODERS, tab_name)

            # ----------------------------------------------------------------
            # 3  Derive phase-5-compatible scalar defaults from HPO space
            #    (midpoint of epoch range; geometric mean of LR range)
            # ----------------------------------------------------------------
            epoch_space = best_hpo_space.get("epochs", {})
            epoch_default: int = (
                (epoch_space.get("low", 10) + epoch_space.get("high", 10)) // 2
                if epoch_space else 10
            )

            lr_space = best_hpo_space.get("learning_rate", {})
            import math
            lr_default: float = (
                math.sqrt(lr_space.get("low", 1e-4) * lr_space.get("high", 1e-3))
                if lr_space else 1e-3
            )

            # ----------------------------------------------------------------
            # 4  Resolve human-readable encoder names for logging
            # ----------------------------------------------------------------
            logger.info("  Image encoder   : %s (%s)", img_name, img_tier or "name-based")
            logger.info("  Text encoder    : %s (%s)", txt_name, txt_tier or "name-based")
            logger.info("  Tabular encoder : %s (%s)", tab_name, tab_tier or "name-based")
            logger.info("  Fusion strategy : %s", best_model.get("fusion_strategy"))
            logger.info("  Batch size      : %d  (PDF heuristic, not tuned)", batch_size)
            logger.info("  Meta priors     : %d similar experiments", len(meta_context))
            logger.info(
                "  Epoch range     : [%d, %d]  → default %d",
                epoch_space.get("low", "?"), epoch_space.get("high", "?"), epoch_default,
            )

            # ----------------------------------------------------------------
            # 5  Store results dict
            # ----------------------------------------------------------------
            elapsed = time.time() - phase_start
            phase_result: Dict[str, Any] = {
                "selection_contract_version": "model_selection.v2",
                "selected_model": best_model.get("name"),
                "recommended_models": recommendations,
                "best_model": best_model,
                "image_encoder":   img_tier or img_name,
                "text_encoder":    txt_tier or txt_name,
                "tabular_encoder": tab_tier or tab_name,
                "image_encoder_name":   img_name,
                "text_encoder_name":    txt_name,
                "tabular_encoder_name": tab_name,
                "fusion_strategy": best_model.get("fusion_strategy"),
                "schema_derived_fusion": schema_derived_fusion,
                "batch_size":      batch_size,
                # Scalar defaults used by Phase 5 before HPO narrows them:
                "epochs":          epoch_default,
                "learning_rate":   lr_default,
                # Full Optuna search bounds:
                "hpo_space":       best_hpo_space,
                "rationale":       best_model.get("rationale", {}),
                "hardware_info":   best_model.get("hardware_info", {}),
                "dataset_meta":    dataset_meta,
                "meta_context":    meta_context,
                "eligible_modalities": list(best_model.get("eligible_modalities") or modalities),
                "excluded_modalities": dict(best_model.get("excluded_modalities") or {}),
                "probe_score": best_model.get("probe_score"),
                "probe_scores": dict(best_model.get("probe_scores") or {}),
                "selection_metadata": dict(best_model.get("selection_metadata") or {}),
                "ranked_candidates": dict(best_model.get("ranked_candidates") or {}),
                "duration_seconds": elapsed,
            }

            if ctx is not None:
                _ctx_ranked = getattr(ctx, "ranked_candidates", None)
                if _ctx_ranked:
                    phase_result["ranked_candidates_from_ctx"] = _ctx_ranked

                if meta_context:
                    first_similar = meta_context[0]
                    warm_params = first_similar.get("best_params") if isinstance(first_similar, dict) else None
                    if isinstance(warm_params, dict) and warm_params:
                        ctx.warm_start_params = dict(warm_params)
                        logger.info(
                            "Phase 4: warm_start_params populated from meta-learning (%d keys)",
                            len(ctx.warm_start_params),
                        )

                try:
                    if not getattr(ctx, "fusion_strategy", None):
                        ctx.update_fusion(phase_result["fusion_strategy"], {})
                    if not getattr(ctx, "model_choices", None):
                        ctx.update_model_selection(
                            recommendations,
                            "Phase 4 orchestrator selection",
                        )
                    ctx.selected_model = best_model.get("name")
                    if phase_result.get("probe_scores"):
                        ctx.probe_scores_cache = dict(phase_result["probe_scores"])
                    if phase_result.get("ranked_candidates"):
                        ctx.ranked_candidates = dict(phase_result["ranked_candidates"])
                except Exception as ctx_exc:
                    logger.debug("Phase 4: context update skipped: %s", ctx_exc)

            logger.info("\nPhase 4 Summary:")
            logger.info("  Selected model: %s", best_model.get("name", "N/A"))
            logger.info("  Batch size    : %d", batch_size)
            logger.info("  Epoch default : %d (HPO will tune in range)", epoch_default)
            logger.info("  LR default    : %.2e (HPO will tune in range)", lr_default)
            logger.info("  HPO params    : %d", len(best_hpo_space))
            logger.info("  Duration      : %.2fs", elapsed)

            self.phase_results[Phase.MODEL_SELECTION] = phase_result
            self.state.set_slot("phase4_model_selection", phase_result)
            self.state.set_phase_timing("MODEL_SELECTION", elapsed)
            self._record_phase_timing_in_context("MODEL_SELECTION", elapsed)
            self.current_phase = Phase.TRAINING

        except Exception as exc:
            logger.error("Phase 4 failed: %s", str(exc))
            raise
    
    def _execute_phase_5_training(self, hp_overrides: Optional[Dict[str, Any]] = None,
                                    early_stop_patience: int = 5,
                                    progress_callback: Optional[Any] = None) -> None:
        """
        Phase 5: GPU Training – Optuna HPO study with MLflow tracking.

        Steps
        -----
        1. Require ``self.torch_dataset`` and ``self.fitted_transformers``
           from Phase 3 (raises ``RuntimeError`` if absent).
        2. Derive ``input_dims`` from the fitted tabular transformer; extend
           with image/text embedding dims when those preprocessors are present.
        3. Determine ``num_classes`` from Phase 3's target encoder.
        4. Split the dataset 80/20 into train / validation subsets.
        5. Create an Optuna ``minimize`` study and run ``N_TRIALS`` trials.
           Each trial:
             a. Samples hyperparams from Phase 4 ``hpo_space`` bounds.
             b. Builds an ``ApexLightningModule`` via ``build_trainer()``.
             c. Trains with ``pytorch_lightning.Trainer`` (GPU if available,
                ``enable_checkpointing=False`` for speed).
             d. Wraps the run in ``mlflow.start_run()`` and logs all params
                and the best ``val_loss`` returned by the trainer.
             e. Returns ``best_val_loss`` as the Optuna objective value.
        6. Extracts the best trial's parameters and reported metric.
        7. Stores a ``phase_results`` dict compatible with Phase 6 and the
           ``/train-pipeline`` API endpoint.

        Windows WDDM Safety
        -------------------
        ``torch.cuda.synchronize()`` is delegated to
        ``ApexLightningModule.training_step`` (called after every batch).
        No additional synchronisation is needed here.
        """
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 5: GPU TRAINING (Optuna + MLflow + Lightning)")
        logger.info("=" * 80)

        self._enforce_session_context("Phase 5")
        self._filter_to_primary_dataset()   # G14: restrict to primary when datasets incompatible
        self._filter_modalities_by_drift()  # drop modalities degraded by drift feedback

        # Scale N_TRIALS with dataset size — small datasets need fewer trials because
        # each trial trains fast and the HP landscape is simpler.
        # HyperbandPruner kills ~60% of trials at epoch 3-5, so effective compute
        # is ~5× a single run regardless of N_TRIALS.
        # Set APEX_N_TRIALS env var to override.
        import os as _os_hpo
        _dataset_n = len(self.torch_dataset) if self.torch_dataset is not None else 10_000
        _default_trials = 30 if _dataset_n > 50_000 else 20 if _dataset_n > 10_000 else 12
        N_TRIALS: int = int(_os_hpo.environ.get("APEX_N_TRIALS", str(_default_trials)))
        VAL_SPLIT: float = 0.2

        if hp_overrides:
            # Use override as trial 0, then explore nearby with Optuna.
            # Default 5 trials: trial 0 = exact override, trials 1-4 = TPE search around it.
            # TPE learns from trial 0's val_loss and biases subsequent samples toward
            # HP regions that performed well. Set APEX_N_TRIALS_MANUAL to override.
            N_TRIALS = min(N_TRIALS, int(_os_hpo.environ.get("APEX_N_TRIALS_MANUAL", "5")))
            logger.info(
                "  HP overrides provided — trial 0 uses exact values, "
                "then %d total trials with Optuna TPE exploring nearby. "
                "Set APEX_N_TRIALS_MANUAL env var to change total (min 1).",
                N_TRIALS,
            )

        phase_start = time.time()

        try:
            import optuna
            import mlflow
            import pytorch_lightning as pl
            from torch.utils.data import DataLoader
            from automl.trainer import OptunaCallback, build_trainer

            optuna.logging.set_verbosity(optuna.logging.WARNING)

            # ----------------------------------------------------------------
            # 1  Require Phase 3 outputs
            # ----------------------------------------------------------------
            if self.torch_dataset is None:
                raise RuntimeError(
                    "Phase 5 requires self.torch_dataset from Phase 3. "
                    "Run _execute_phase_3_preprocessing() first."
                )

            # ----------------------------------------------------------------
            # 2  Derive input_dims from fitted tabular transformer
            # ----------------------------------------------------------------
            input_dims: Dict[str, int] = {}
            tabular_prep = self.fitted_transformers.get("tabular")
            if tabular_prep is not None:
                input_dims["tabular"] = tabular_prep.get_output_dim()

            # Text/image encoders would need full pretrained models to be
            # loaded here; include their default output dims only when the
            # corresponding preprocessor is present so the head is sized
            # correctly if those encoders are wired in later.
            if self.fitted_transformers.get("text") is not None:
                input_dims["text_pooled"] = 768    # default; updated after encoder init
            if self.fitted_transformers.get("image") is not None:
                input_dims["image_pooled"] = 512   # default; updated after encoder init

            if not input_dims:
                raise RuntimeError(
                    "Phase 5: no fitted preprocessors found in "
                    "self.fitted_transformers; cannot determine input_dims."
                )

            # ----------------------------------------------------------------
            # 3  Determine num_classes and problem_type
            # ----------------------------------------------------------------
            schema_info: Dict[str, Any] = self.phase_results.get(
                Phase.SCHEMA_DETECTION, {}
            )
            problem_type: str = schema_info.get(
                "global_problem_type", self.config.problem_type
            )
            target_enc = self.fitted_transformers.get("target_encoder")
            if isinstance(target_enc, dict) and target_enc.get("type") == "multilabel":
                num_classes: int = len(target_enc["all_labels"])
            elif problem_type.startswith("classification") and hasattr(target_enc, "classes_"):
                num_classes: int = len(target_enc.classes_)
            else:
                num_classes = 2   # safe default

            # ----------------------------------------------------------------
            # 4  Train / val split
            #    A shared random permutation splits the row indices once.
            #    Train indices → augmented dataset (RandomFlip/Rotate/Jitter).
            #    Val   indices → clean dataset     (Resize+Normalize only).
            # ----------------------------------------------------------------
            _aug_ds   = self.train_torch_dataset or self.torch_dataset
            _clean_ds = self.val_torch_dataset   or self.torch_dataset
            n_total: int = len(_aug_ds)
            n_val: int = max(1, int(n_total * VAL_SPLIT))
            n_train: int = n_total - n_val

            from torch.utils.data import Subset as _Subset

            # Stratified split for classification; random for regression.
            # Preserves class balance in both train and val subsets.
            all_indices = list(range(n_total))
            _use_stratify = problem_type.startswith("classification") or problem_type == "multilabel_classification"
            _use_stratified_kfold = False
            ctx = self._get_ctx()  # may be None for standalone runs
            if _use_stratify and ctx is not None:
                try:
                    for _ds_intel in dict(getattr(ctx, "feature_intelligence", {}) or {}).values():
                        if not isinstance(_ds_intel, dict):
                            continue
                        long_tail_cats = [str(v) for v in list(_ds_intel.get("long_tail_cats", []) or [])]
                        if target_col and str(target_col) in long_tail_cats:
                            _use_stratified_kfold = True
                            break
                except Exception:
                    _use_stratified_kfold = False
            if _use_stratify:
                try:
                    from sklearn.model_selection import StratifiedKFold, train_test_split as _split
                    # Extract target labels for stratification
                    _targets_for_strat = []
                    for _idx in range(n_total):
                        _sample = _aug_ds[_idx]
                        _t = _sample.get("target") if isinstance(_sample, dict) else _sample[-1]
                        if hasattr(_t, "item"):
                            _targets_for_strat.append(_t.item())
                        elif hasattr(_t, "argmax"):
                            _targets_for_strat.append(int(_t.argmax()))
                        else:
                            _targets_for_strat.append(_t)
                    if _use_stratified_kfold:
                        n_splits = max(3, int(round(1.0 / max(VAL_SPLIT, 1e-6))))
                        n_splits = min(n_splits, max(2, len(set(_targets_for_strat))))
                        splitter = StratifiedKFold(
                            n_splits=n_splits,
                            shuffle=True,
                            random_state=self.config.seed,
                        )
                        train_indices, val_indices = next(
                            splitter.split(np.zeros(n_total), _targets_for_strat)
                        )
                        train_indices = list(train_indices)
                        val_indices = list(val_indices)
                        logger.info(
                            "  StratifiedKFold split (long-tail target): train=%d  val=%d",
                            len(train_indices),
                            len(val_indices),
                        )
                    else:
                        train_indices, val_indices = _split(
                            all_indices,
                            test_size=VAL_SPLIT,
                            random_state=self.config.seed,
                            stratify=_targets_for_strat,
                        )
                        logger.info("  Stratified split: train=%d  val=%d", len(train_indices), len(val_indices))
                except Exception as strat_exc:
                    logger.warning(
                        "  Stratified split failed (%s), falling back to random split",
                        strat_exc,
                    )
                    perm = torch.randperm(
                        n_total,
                        generator=torch.Generator().manual_seed(self.config.seed),
                    ).tolist()
                    train_indices = perm[:n_train]
                    val_indices   = perm[n_train:]
            else:
                perm = torch.randperm(
                    n_total,
                    generator=torch.Generator().manual_seed(self.config.seed),
                ).tolist()
                train_indices = perm[:n_train]
                val_indices   = perm[n_train:]

            n_train = len(train_indices)
            n_val   = len(val_indices)

            train_ds = _Subset(_aug_ds,   train_indices)
            val_ds   = _Subset(_clean_ds, val_indices)

            # Report data split to progress callback
            if progress_callback is not None:
                progress_callback.set_data_split(n_train, n_val, n_total)

            # Resolve execution context for class-weight / encoder-plan reads
            ctx = self._get_ctx()

            # ── Compute class weights for imbalanced classification ──────
            _class_weights: Optional[torch.Tensor] = None
            _use_focal_loss: bool = False
            if (problem_type.startswith("classification")
                    and problem_type != "multilabel_classification"):
                try:
                    from sklearn.utils.class_weight import compute_class_weight as _ccw
                    _train_targets = _aug_ds.targets[train_indices]
                    _train_labels = _train_targets.numpy().astype(int)
                    _unique_classes = np.sort(np.unique(_train_labels))
                    _raw_weights = _ccw("balanced", classes=_unique_classes, y=_train_labels)
                    _class_weights = torch.tensor(_raw_weights, dtype=torch.float32)
                    # Activate Focal Loss when imbalance ratio > 3:1 [Lin et al. ICCV 2017]
                    _imbalance_ratio = float(_raw_weights.max() / max(_raw_weights.min(), 1e-8))
                    _use_focal_loss = _imbalance_ratio > 3.0
                    logger.info(
                        "  Class weights (balanced): %s  imbalance_ratio=%.2f  focal_loss=%s",
                        {int(c): round(float(w), 3) for c, w in zip(_unique_classes, _raw_weights)},
                        _imbalance_ratio,
                        _use_focal_loss,
                    )
                except Exception as cw_exc:
                    logger.warning("  Class weight computation failed: %s", cw_exc)

            # ── Label noise detection ─────────────────────────────────────
            _noise_sample_weights: Optional[np.ndarray] = None
            try:
                from pipeline.label_noise_detector import LabelNoiseDetector
                _tabular_prep = self.fitted_transformers.get("tabular")
                if (
                    _tabular_prep is not None
                    and problem_type.startswith("classification")
                    and hasattr(_aug_ds, "targets")
                ):
                    _all_targets = _aug_ds.targets.numpy()
                    _X_tab_full = None
                    try:
                        _X_tab_full = _tabular_prep.transform(
                            getattr(_aug_ds, "df", None)
                        )
                    except Exception:
                        pass

                    if _X_tab_full is not None and len(_X_tab_full) == n_total:
                        _noise_detector = LabelNoiseDetector(
                            n_folds=5,
                            noise_threshold=0.8,
                            min_samples_to_run=200,
                            weight_floor=0.2,
                        )
                        _noise_result = _noise_detector.detect(
                            _X_tab_full,
                            _all_targets,
                            problem_type=problem_type,
                        )
                        if not _noise_result["skipped"] and _noise_result["n_suspicious"] > 0:
                            _noise_sample_weights = _noise_result["sample_weights"]
                            if ctx is not None and hasattr(ctx, "__setattr__"):
                                ctx.suspicious_label_indices = _noise_result["suspicious_indices"]
                            logger.info(
                                "  Label noise: %d/%d samples down-weighted (w=%.2f)",
                                _noise_result["n_suspicious"], n_total, 0.2,
                            )
            except Exception as _lnd_exc:
                logger.debug("Label noise detection failed (non-fatal): %s", _lnd_exc)

            model_sel: Dict[str, Any] = self.phase_results.get(Phase.MODEL_SELECTION, {})
            batch_size: int = model_sel.get("batch_size", 32)
            hpo_space: Dict[str, Any] = model_sel.get("hpo_space", {})
            preprocessing_plan = self.state.get_slot("preprocessing_plan", {})
            encoder_plan = preprocessing_plan.get("encoder_config", {}) if isinstance(preprocessing_plan, dict) else {}
            ctx_encoder_plan: Dict[str, Any] = {}
            if ctx is not None:
                ctx_encoder_plan = dict(getattr(ctx, "encoder_plan", {}) or {})
            if ctx_encoder_plan:
                encoder_plan = {
                    **dict(encoder_plan or {}),
                    **{
                        modality: {
                            **dict((encoder_plan or {}).get(modality, {}) or {}),
                            "preferred_model": value,
                        }
                        for modality, value in ctx_encoder_plan.items()
                    },
                }

            mean_uncertainty: float = 0.0
            if ctx is not None:
                try:
                    uncertainty_vals: List[float] = []
                    for _ds_intel in dict(getattr(ctx, "feature_intelligence", {}) or {}).values():
                        if not isinstance(_ds_intel, dict):
                            continue
                        uncertainty_vals.extend(
                            float(v)
                            for v in dict(_ds_intel.get("uncertainty_summary", {}) or {}).values()
                            if isinstance(v, (int, float))
                        )
                    if uncertainty_vals:
                        mean_uncertainty = sum(uncertainty_vals) / len(uncertainty_vals)
                except Exception:
                    mean_uncertainty = 0.0

            context_dropout_floor = 0.0
            if ctx is not None:
                try:
                    context_dropout_floor = float(
                        dict(getattr(ctx, "constraints", {}) or {}).get("dropout_floor", 0.0) or 0.0
                    )
                except Exception:
                    context_dropout_floor = 0.0

            hpo_space = self.optuna_adaptive.adapt_search_space(
                base_space=hpo_space,
                dataset_size=n_total,
                modalities=schema_info.get("global_modalities", self.config.modalities),
                problem_type=problem_type,
            )

            # Merge modality-specific Optuna bounds from feature_intelligence
            try:
                from config.hyperparameters import get_modality_optuna_distributions
                _active_mods = schema_info.get("global_modalities", self.config.modalities) or []
                _fi = getattr(self.execution_context, "feature_intelligence", None) or {}
                _n_tabular = int(schema_info.get("total_feature_count", 0) or 0)
                _avg_text_len: float = 0.0
                for _ds_intel in _fi.values():
                    _tl = _ds_intel.get("avg_text_len")
                    if _tl:
                        _avg_text_len = max(_avg_text_len, float(_tl))
                _modality_dists = get_modality_optuna_distributions(
                    active_modalities=list(_active_mods),
                    n_tabular_features=_n_tabular,
                    avg_text_len=_avg_text_len,
                    problem_type=problem_type,
                )
                # Merge: modality-specific bounds override the static adapt_search_space output
                for _param, _dist in _modality_dists.items():
                    if _param in hpo_space:
                        hpo_space[_param] = {**hpo_space[_param], **_dist}
                    else:
                        hpo_space[_param] = _dist
                logger.info(
                    "  Modality-specific HPO bounds applied: mods=%s, "
                    "n_tabular=%d, avg_text_len=%.1f",
                    list(_active_mods), _n_tabular, _avg_text_len,
                )
            except Exception as _hpo_exc:
                logger.debug("Modality-specific HPO merge failed (non-fatal): %s", _hpo_exc)

            if not hp_overrides:
                N_TRIALS = self.optuna_adaptive.suggest_trial_count(
                    dataset_size=n_total,
                    gpu_available=(self.device.type == "cuda"),
                )
                logger.info("  Adaptive HPO policy: trials=%d", N_TRIALS)

            # Allow hp_overrides to change batch_size and fusion_strategy
            if hp_overrides:
                if "batch_size" in hp_overrides:
                    batch_size = int(hp_overrides["batch_size"])
                    logger.info("  batch_size overridden to %d", batch_size)
                if "fusion_strategy" in hp_overrides:
                    model_sel["fusion_strategy"] = hp_overrides["fusion_strategy"]
                    logger.info("  fusion_strategy overridden to %s", hp_overrides["fusion_strategy"])

            import os as _os
            import sys as _sys
            # Cap workers: spawn (Windows) has high per-worker overhead; fork
            # (Linux) can share memory.  Never exceed physical core count.
            _max_safe_workers = 0 if _sys.platform == "win32" else min(4, _os.cpu_count() or 1)
            _n_workers: int = _max_safe_workers
            # On Windows (spawn start method), persistent_workers causes each
            # worker to pickle the entire Subset + parent dataset on every
            # epoch boundary.  Disable to avoid OOM on large datasets.
            _persistent = _n_workers > 0 and _sys.platform != "win32"
            _pin = self.device.type == "cuda" and _n_workers > 0

            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True,
                num_workers=_n_workers, pin_memory=_pin,
                persistent_workers=_persistent,
            )
            val_loader = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False,
                num_workers=_n_workers, pin_memory=_pin,
                persistent_workers=_persistent,
            )

            logger.info(
                "  dataset=%d  train=%d  val=%d  batch_size=%d  "
                "input_dims=%s  num_classes=%d",
                n_total, n_train, n_val, batch_size, input_dims, num_classes,
            )

            # ----------------------------------------------------------------
            # 5  Optuna study
            # ----------------------------------------------------------------
            accelerator: str = "gpu" if self.device.type == "cuda" else "cpu"
            mlflow.set_experiment("apex_phase5")

            # Load user-registered encoder plugins before JIT selection
            try:
                import config.encoder_plugins  # noqa: F401
            except ImportError:
                pass

            # Instantiate frozen encoders ONCE via JIT hardware profiler —
            # selects the highest-capacity encoders that fit within the
            # available VRAM budget (eta=0.85 safety margin).  Falls back to
            # lightest encoders on CPU or when no combination fits.
            from automl.jit_encoder_selector import JITEncoderSelector

            if progress_callback is not None:
                progress_callback.set_phase(5, "Profiling encoders on GPU (JIT dry-run)...", 10)
                progress_callback.set_substage("jit_profiling")
                progress_callback.push_trial_event(
                    None,
                    "substage",
                    "Profiling encoders on GPU",
                    {"substage": "jit_profiling"},
                )

            _jit_selector = JITEncoderSelector(
                safety_margin=0.85,
                batch_size=batch_size,
            )
            _preferred_tabular = (
                model_sel.get("tabular_probe_top_model")
                or model_sel.get("tabular_encoder_name")
                or model_sel.get("tabular_encoder")
            )
            # Seed preferred encoders from Phase 4 JIT dry-run result so Phase 5
            # honours the same hardware-fit selection that was shown in the UI.
            # User overrides (below) and encoder_plan take precedence if set.
            _p4_jit = model_sel.get("jit_dry_run", {}) or {}
            _preferred_text = _p4_jit.get("selected_text_encoder") or None
            _preferred_image = _p4_jit.get("selected_image_encoder") or None
            if _preferred_text or _preferred_image:
                logger.info(
                    "Phase 5: seeding JIT preferred encoders from Phase 4 dry-run — "
                    "text=%s image=%s",
                    _preferred_text, _preferred_image,
                )

            # Override with encoder_plan (from preprocessing context) if present
            if isinstance(encoder_plan, dict):
                _preferred_text = (
                    dict(encoder_plan.get("text", {}) or {}).get("preferred_model")
                    or dict(encoder_plan.get("text", {}) or {}).get("model_name")
                ) or _preferred_text
                _preferred_image = (
                    dict(encoder_plan.get("image", {}) or {}).get("preferred_model")
                ) or _preferred_image

            # Apply user encoder overrides from context (set via /encoder-overrides UI)
            # Stored inside encoder_plan["_encoder_overrides"] for persistence
            _ctx_for_enc = self._get_ctx()
            _enc_plan = dict(getattr(_ctx_for_enc, "encoder_plan", {}) or {})
            _enc_overrides = dict(_enc_plan.get("_encoder_overrides", {}) or {})
            if _enc_overrides:
                _preferred_image  = _enc_overrides.get("preferred_image_encoder") or _preferred_image
                _preferred_text   = _enc_overrides.get("preferred_text_encoder")  or _preferred_text
                _preferred_tabular= _enc_overrides.get("preferred_tabular_encoder") or _preferred_tabular
                logger.info("Phase 5: user encoder overrides applied: %s", _enc_overrides)
            # Read schema-derived output dims from ExecutionContext so the JIT
            # selector can pick an encoder whose output_dim matches the head.
            _ctx_enc_dims = {}
            try:
                _ctx_enc_dims = dict(getattr(self._get_ctx(), "encoder_output_dims", {}) or {})
            except Exception:
                pass

            _jit_result = _jit_selector.select(
                modalities=schema_info.get("global_modalities", self.config.modalities),
                device=self.device if self.device.type == "cuda" else None,
                preferred_tabular=_preferred_tabular,
                preferred_text=_preferred_text,
                preferred_image=_preferred_image,
            )
            if progress_callback is not None:
                progress_callback.push_trial_event(
                    None,
                    "jit_selected",
                    f"Selected image={_jit_result.image_encoder_name} text={_jit_result.text_encoder_name}",
                    {
                        "image_encoder": _jit_result.image_encoder_name,
                        "text_encoder": _jit_result.text_encoder_name,
                        "peak_mb": float(getattr(_jit_result, "peak_memory_mb", 0.0) or 0.0),
                    },
                )

            # Apply ctx encoder_output_dims as output-dim overrides when the
            # selected encoder supports runtime reconfiguration.
            for _mod, _dim in _ctx_enc_dims.items():
                try:
                    enc = getattr(_jit_result, f"{_mod}_encoder", None)
                    if enc is not None and hasattr(enc, "output_dim"):
                        enc.output_dim = int(_dim)
                except Exception:
                    pass

            _image_encoder = _jit_result.image_encoder
            _text_encoder = _jit_result.text_encoder

            if _text_encoder is not None:
                _text_plan = dict(encoder_plan.get("text", {}))
                if "text_max_length" in _text_plan and "max_length" not in _text_plan:
                    _text_plan["max_length"] = _text_plan["text_max_length"]
                try:
                    _text_encoder.configure(_text_plan)
                except Exception as cfg_exc:
                    logger.debug("  Text encoder configure failed: %s", cfg_exc)

            # ── Bug 1 fix: sync text tokenizer to the JIT-selected encoder ──
            # Phase 3 always tokenizes with bert-base-uncased.  When JIT selects
            # DeBERTa (or any other encoder), the vocab is completely different.
            # Feeding BERT token IDs to DeBERTa produces garbage embeddings.
            # We update the fitted TextPreprocessor here, before precomputation,
            # so _precompute_text_embeddings sees the correct tokenizer.
            _TEXT_TOKENIZER_MAP = {
                "deberta":    "microsoft/deberta-v3-base",
                "roberta":    "roberta-base",
                "distilbert": "distilbert-base-uncased",
                "bert":       "bert-base-uncased",
                "electra":    "google/electra-base-discriminator",
                "mpnet":      "sentence-transformers/all-mpnet-base-v2",
                "minilm":     "sentence-transformers/all-MiniLM-L6-v2",
                "xlm":        "xlm-roberta-base",
            }
            _text_enc_name = (_jit_result.text_encoder_name or "").lower()
            _text_prep_fitted = self.fitted_transformers.get("text")
            if _text_prep_fitted is not None and _text_enc_name:
                _matched_tok = None
                for _tok_key, _tok_id in _TEXT_TOKENIZER_MAP.items():
                    if _tok_key in _text_enc_name:
                        _matched_tok = _tok_id
                        break
                if _matched_tok and _matched_tok != getattr(_text_prep_fitted, "_pretrained_model", None):
                    _old_tok = getattr(_text_prep_fitted, "_pretrained_model", "?")
                    _text_prep_fitted._pretrained_model = _matched_tok
                    _text_prep_fitted._tokenizer = None  # force lazy reload
                    # Use wider context window for encoders that support it
                    _natural_max = 512 if any(k in _text_enc_name for k in ("deberta", "roberta", "xlm")) else 128
                    _text_prep_fitted.max_length = max(_text_prep_fitted.max_length, min(256, _natural_max))
                    logger.info(
                        "Phase 5: text tokenizer synced '%s' → '%s'  max_length=%d",
                        _old_tok, _matched_tok, _text_prep_fitted.max_length,
                    )

            if _image_encoder is not None:
                _image_plan = dict(encoder_plan.get("image", {}))
                if "freeze_image_backbone" in _image_plan and "freeze_backbone" not in _image_plan:
                    _image_plan["freeze_backbone"] = _image_plan["freeze_image_backbone"]
                # ── Bug 2 fix: pass JIT-selected encoder name so the preprocessor
                # switches to CLIP normalization for SigLIP/CLIP encoders ──
                _image_plan["selected_image_encoder"] = _jit_result.image_encoder_name or ""
                try:
                    _image_encoder.configure(_image_plan)
                except Exception as cfg_exc:
                    logger.debug("  Image encoder configure failed: %s", cfg_exc)

            # Also update the fitted ImagePreprocessor normalization to match the
            # JIT-selected encoder (SigLIP/CLIP need CLIP stats, not ImageNet).
            _image_prep_fitted = self.fitted_transformers.get("image")
            if _image_prep_fitted is not None and _jit_result.image_encoder_name:
                _old_norm = getattr(_image_prep_fitted, "normalize_mode", "imagenet")
                _image_prep_fitted.configure({"selected_image_encoder": _jit_result.image_encoder_name})
                _new_norm = getattr(_image_prep_fitted, "normalize_mode", "imagenet")
                if _old_norm != _new_norm:
                    logger.info(
                        "Phase 5: image normalization updated '%s' → '%s' for encoder '%s'",
                        _old_norm, _new_norm, _jit_result.image_encoder_name,
                    )

            # Update input_dims from selected encoders' actual output dims
            if _text_encoder is not None and hasattr(_text_encoder, "get_output_dim"):
                input_dims["text_pooled"] = _text_encoder.get_output_dim()
            if _image_encoder is not None and hasattr(_image_encoder, "get_output_dim"):
                input_dims["image_pooled"] = _image_encoder.get_output_dim()
                logger.info("  input_dims['image_pooled'] updated to %d from encoder", input_dims["image_pooled"])

            # Log selection results
            logger.info(
                "  JIT Encoder Selection: method=%s  "
                "image=%s  text=%s  capacity=%s  peak=%.2f MB",
                _jit_result.selection_method,
                _jit_result.image_encoder_name or "—",
                _jit_result.text_encoder_name or "—",
                f"{_jit_result.total_capacity:,}",
                _jit_result.total_peak_memory_bytes / 1e6,
            )
            for component, reason in _jit_result.rationale.items():
                logger.info("    %s: %s", component, reason)

            if progress_callback is not None:
                _img_name = _jit_result.image_encoder_name or "image encoder"
                _txt_name = _jit_result.text_encoder_name or "text encoder"
                progress_callback.set_phase(
                    5,
                    f"Encoders selected: {_img_name} + {_txt_name}. "
                    f"Loading weights (~440MB first run)...",
                    20,
                )

            # Extract tabular encoder class from JIT result for per-trial
            # instantiation.  Tabular encoders are trainable (not frozen),
            # so we store only the class reference here and create fresh
            # instances inside objective().
            _tabular_encoder_class = _jit_result.tabular_encoder_class
            _tabular_input_dim: Optional[int] = None
            tabular_prep = self.fitted_transformers.get("tabular")
            if tabular_prep is not None:
                _tabular_input_dim = tabular_prep.get_output_dim()

            # Update input_dims["tabular"] to encoder OUTPUT dim (not raw dim)
            # so _MultimodalHead is sized to the encoded representation.
            if _tabular_encoder_class is not None and "tabular" in input_dims:
                input_dims["tabular"] = _jit_result.tabular_encoder_output_dim
                logger.info(
                    "  Tabular encoder: %s  raw_input_dim=%s  output_dim=%d",
                    _jit_result.tabular_encoder_name,
                    _tabular_input_dim,
                    _jit_result.tabular_encoder_output_dim,
                )

            # ================================================================
            # Pre-compute frozen encoder embeddings (one-time forward pass)
            # ================================================================
            # Text embeddings are deterministic (no augmentation) so we cache
            # for both train and val splits.  Image embeddings are cached for
            # val only because training images go through random augmentation.
            dataset_fingerprint = {
                "sources": sorted(self.config.dataset_sources),
                "problem_type": problem_type,
                "modalities": sorted(schema_info.get("global_modalities", self.config.modalities)),
                "rows": n_total,
                "seed": self.config.seed,
            }

            _ctx_for_cache = self._get_ctx()
            if _ctx_for_cache is not None:
                _scores = dict(getattr(_ctx_for_cache, "predictability_scores", {}) or {})
                _priority_map: Dict[str, float] = {}
                for key, score in _scores.items():
                    try:
                        score_val = float(score)
                    except Exception:
                        continue
                    key_lower = str(key).lower()
                    if "image" in key_lower:
                        _priority_map["image"] = score_val
                    elif "text" in key_lower:
                        _priority_map["text"] = score_val
                    elif "tabular" in key_lower:
                        _priority_map["tabular"] = score_val
                if _priority_map:
                    self.embedding_cache.set_modality_priorities(_priority_map)

            if _text_encoder is not None and hasattr(_aug_ds, '_text_cols') and _aug_ds._text_cols:
                if progress_callback is not None:
                    progress_callback.set_substage("text_embedding_cache")
                    progress_callback.push_trial_event(
                        None,
                        "substage",
                        "Preparing text embedding cache",
                        {"substage": "text_embedding_cache"},
                    )
                text_cache_key = self.embedding_cache.build_key({
                    **dataset_fingerprint,
                    "modality": "text",
                    "encoder": _jit_result.text_encoder_name,
                    "max_length": getattr(_text_encoder, "max_length", 128),
                })
                _precomputed_text = self.embedding_cache.get(text_cache_key)
                if _precomputed_text is None or _precomputed_text.shape[0] != n_total:
                    logger.info("  Pre-computing text embeddings (%d samples)...", n_total)
                    if progress_callback is not None:
                        progress_callback.set_phase(5, f"Pre-computing text embeddings (0/{n_total})...", 25)
                    _precomputed_text = _precompute_text_embeddings(
                        _clean_ds, _text_encoder, self.device, batch_size=batch_size,
                        progress_fn=lambda pct, msg, _cb=progress_callback: _cb.set_phase(
                            5, msg, 25 + int(pct * 0.20)
                        ) if progress_callback is not None else None,
                    )
                    self.embedding_cache.set(
                        text_cache_key,
                        _precomputed_text,
                        meta={"modality": "text", "rows": n_total},
                    )
                else:
                    logger.info("  Loaded text embeddings from cache: shape=%s", list(_precomputed_text.shape))

                _aug_ds._precomputed_text = _precomputed_text
                _clean_ds._precomputed_text = _precomputed_text
                logger.info("  Text embeddings ready: shape=%s", list(_precomputed_text.shape))
                if progress_callback is not None:
                    progress_callback.add_message(
                        5, "info",
                        f"Text embeddings ready: {_precomputed_text.shape[0]}×{_precomputed_text.shape[1]}d"
                    )
                    progress_callback.push_trial_event(
                        None,
                        "embedding_ready",
                        "Text embeddings ready",
                        {
                            "modality": "text",
                            "rows": int(_precomputed_text.shape[0]),
                            "dim": int(_precomputed_text.shape[1]),
                        },
                    )
            else:
                _precomputed_text = None

            if _image_encoder is not None and hasattr(_clean_ds, '_image_cols') and _clean_ds._image_cols:
                if progress_callback is not None:
                    progress_callback.set_substage("image_embedding_cache")
                    progress_callback.push_trial_event(
                        None,
                        "substage",
                        "Preparing image embedding cache",
                        {"substage": "image_embedding_cache"},
                    )
                image_cache_key = self.embedding_cache.build_key({
                    **dataset_fingerprint,
                    "modality": "image_val",
                    "encoder": _jit_result.image_encoder_name,
                    "split": "val_clean",
                })
                _precomputed_image_val = self.embedding_cache.get(image_cache_key)
                if _precomputed_image_val is None or _precomputed_image_val.shape[0] != n_total:
                    logger.info("  Pre-computing image embeddings for val (%d samples)...", n_total)
                    if progress_callback is not None:
                        progress_callback.set_phase(5, f"Pre-computing image embeddings (0/{n_total})...", 45)
                    _precomputed_image_val = _precompute_image_embeddings(
                        _clean_ds, _image_encoder, self.device, batch_size=batch_size,
                        progress_fn=lambda pct, msg, _cb=progress_callback: _cb.set_phase(
                            5, msg, 45 + int(pct * 0.20)
                        ) if progress_callback is not None else None,
                    )
                    self.embedding_cache.set(
                        image_cache_key,
                        _precomputed_image_val,
                        meta={"modality": "image_val", "rows": n_total},
                    )
                else:
                    logger.info(
                        "  Loaded image-val embeddings from cache: shape=%s",
                        list(_precomputed_image_val.shape),
                    )

                # Train images: NOT cached (random augmentation must be preserved)
                _aug_ds._precomputed_image = None
                # Val images: cached (deterministic preprocessing only)
                _clean_ds._precomputed_image = _precomputed_image_val
                logger.info("  Image embeddings ready (val only): shape=%s", list(_precomputed_image_val.shape))
                if progress_callback is not None:
                    progress_callback.add_message(
                        5, "info",
                        f"Image embeddings ready: {_precomputed_image_val.shape[0]}×{_precomputed_image_val.shape[1]}d — "
                        "starting Optuna HPO..."
                    )
                    progress_callback.push_trial_event(
                        None,
                        "embedding_ready",
                        "Image embeddings ready",
                        {
                            "modality": "image",
                            "rows": int(_precomputed_image_val.shape[0]),
                            "dim": int(_precomputed_image_val.shape[1]),
                        },
                    )
            else:
                _precomputed_image_val = None

            def _sample(trial: optuna.Trial, key: str, default: Any) -> Any:
                """Sample a value from hpo_space or return the default."""
                spec = hpo_space.get(key)
                if spec is None:
                    return default
                t = spec.get("type")
                if t == "int":
                    return trial.suggest_int(key, spec["low"], spec["high"])
                if t == "float":
                    return trial.suggest_float(
                        key, spec["low"], spec["high"],
                        log=spec.get("log", False),
                    )
                if t == "categorical":
                    return trial.suggest_categorical(key, spec["choices"])
                return default

            def _calibration_proxy_from_gap(gap: Any) -> Optional[float]:
                try:
                    gap_value = float(gap)
                except Exception:
                    return None
                proxy = 1.0 - (abs(gap_value) / (1.0 + abs(gap_value)))
                return round(max(0.0, min(1.0, proxy)), 6)

            def _adaptive_penalty_from_diag(diag: Dict[str, Any]) -> float:
                dynamic = dict(diag.get("dynamic_factors", {}) or {})
                penalties: List[float] = []
                for value in dynamic.values():
                    try:
                        penalties.append(max(0.0, float(value) - 1.0))
                    except Exception:
                        continue

                dynamic_penalty = sum(penalties) / max(1, len(penalties)) if penalties else 0.0
                try:
                    gap = abs(float(diag.get("generalization_gap", 0.0) or 0.0))
                except Exception:
                    gap = 0.0
                gap_penalty = gap / (1.0 + gap)
                return round(min(0.8, dynamic_penalty + 0.30 * gap_penalty), 6)

            # Mutable containers capture the best trained LightningModule
            # inside the closure without needing 'nonlocal' on a scalar.
            _best_val: List[float] = [float("inf")]
            _best_module_ref: List[Any] = []
            _best_metrics: Dict[str, float] = {"val_acc": 0.0, "val_f1": 0.0, "train_acc": 0.0}
            _best_lora_config: Dict[str, Any] = {}
            _best_fusion_strategy: List[str] = [""]
            adaptive_feedback_state: Dict[str, Any] = {}
            adaptive_trial_overrides: Dict[str, Any] = {}
            trial_feedback_events: List[Dict[str, Any]] = []
            pruning_runtime: Dict[str, Any] = {
                "available": False,
                "backend": "disabled",
                "reason": "",
            }

            def objective(trial: optuna.Trial) -> float:
                # Report trial number and status to progress callback
                if progress_callback is not None:
                    progress_callback.set_trial(trial.number + 1, N_TRIALS)
                    progress_callback.set_substage("trial_sampling")
                    progress_callback.set_phase(
                        5,
                        f"Trial {trial.number + 1}/{N_TRIALS}: sampling hyperparameters...",
                        65 + int(trial.number / max(1, N_TRIALS) * 30),
                    )

                # Unified sampling path — works for all trials:
                # • Trial 0 (enqueued): _sample returns the enqueued hp_overrides values.
                # • Trials 1+: _sample uses Optuna TPE (learned from trial 0's result).
                # • Params NOT in hpo_space: use hp_overrides value if provided, else default.
                _hp = hp_overrides or {}
                trial_lr      = _sample(trial, "learning_rate", _hp.get("learning_rate", model_sel.get("learning_rate", 1e-3)))
                trial_wd      = _sample(trial, "weight_decay",  _hp.get("weight_decay", 1e-5))
                trial_dropout = _sample(trial, "dropout",       _hp.get("dropout", 0.1))
                trial_label_smoothing = _hp.get("label_smoothing", 0.0)
                trial_epochs  = _sample(trial, "epochs",        int(_hp.get("epochs", model_sel.get("epochs", 10))))
                # Bug 6 fix: use schema_derived_fusion (set by Phase 4 intelligence) as
                # the fallback, not a hardcoded "concatenation". For text+image datasets
                # this ensures ULA is the default when hp_overrides don't specify a strategy.
                _schema_fusion_default = (
                    self.state.get_slot("schema_derived_fusion")
                    or model_sel.get("fusion_strategy")
                    or "concatenation"
                )
                trial_fusion  = _sample(trial, "fusion_strategy", _hp.get("fusion_strategy", _schema_fusion_default))
                trial_alignment = _sample(trial, "alignment_weight", _hp.get("alignment_weight", 0.0))
                # Params not in hpo_space: fixed from override or default for all trials
                trial_modality_dropout = _hp.get("modality_dropout_prob", 0.15)
                trial_graph_sparsity = _hp.get("graph_sparsity_weight", 0.005)
                trial_diversity_weight = _hp.get("diversity_loss_weight", 0.01)
                trial_uncertainty_aux = _hp.get("uncertainty_aux_weight", 0.0)
                trial_graph_branch_weight = _hp.get("uncertainty_graph_weight", 0.5)
                trial_uncertainty_branch_weight = _hp.get(
                    "uncertainty_branch_weight",
                    max(0.0, 1.0 - float(trial_graph_branch_weight)),
                )

                # Adaptive overrides apply to trials beyond trial 0 (don't overwrite user's exact values)
                if adaptive_trial_overrides and trial.number > 0:
                    if "learning_rate" in adaptive_trial_overrides:
                        trial_lr = float(adaptive_trial_overrides["learning_rate"])
                    if "weight_decay" in adaptive_trial_overrides:
                        trial_wd = float(adaptive_trial_overrides["weight_decay"])
                    if "dropout" in adaptive_trial_overrides:
                        trial_dropout = float(adaptive_trial_overrides["dropout"])
                    if "epochs" in adaptive_trial_overrides:
                        _ep_override = int(adaptive_trial_overrides["epochs"])
                        _ep_min = int(hpo_space.get("epochs", {}).get("low", 3))
                        _ep_max = int(hpo_space.get("epochs", {}).get("high", 40))
                        trial_epochs = max(_ep_min, min(_ep_max, _ep_override))
                    logger.info(
                        "  Trial %d adaptive overrides applied: %s",
                        trial.number,
                        adaptive_trial_overrides,
                    )

                if trial_fusion == "auto":
                    trial_fusion = self.state.get_slot("schema_derived_fusion") or "attention"

                # ── ULA-specific HPO sampling ─────────────────────────────
                _ula_fusion_cfg: Dict[str, Any] = {}
                _trial_lora_config: Optional[Dict[str, Any]] = None

                if trial_fusion in ("ula", "unified_latent", "unified_latent_alignment"):
                    _base_latent = int(self.state.get_slot("ula_latent_dim") or 256)
                    _latent_choices = sorted(set([max(64, _base_latent // 2), _base_latent, min(512, _base_latent * 2)]))
                    # Always sample ULA params via Optuna — trial 0 uses enqueued/default values,
                    # subsequent trials explore via TPE.
                    _ula_latent_dim = trial.suggest_categorical("ula_latent_dim", _latent_choices)
                    _ula_n_layers   = trial.suggest_int("ula_n_layers", 1, 4)
                    _ula_n_heads    = trial.suggest_categorical("ula_n_heads", [2, 4, 8])
                    _lora_r         = trial.suggest_categorical("lora_r", [4, 8, 16])
                    _lora_alpha     = int(_lora_r * 2)
                    # For trial 0 with hp_overrides: apply user-specified ULA values on top
                    if hp_overrides and trial.number == 0:
                        _ula_latent_dim = int(_hp.get("ula_latent_dim", _ula_latent_dim))
                        _ula_n_layers   = int(_hp.get("ula_n_layers", _ula_n_layers))
                        _ula_n_heads    = int(_hp.get("ula_n_heads", _ula_n_heads))
                        _lora_r         = int(_hp.get("lora_r", _lora_r))
                        _lora_alpha     = int(_hp.get("lora_alpha", _lora_r * 2))

                    _ula_fusion_cfg = {
                        "latent_dim": _ula_latent_dim,
                        "n_layers":   _ula_n_layers,
                        "n_heads":    _ula_n_heads,
                    }
                    _trial_lora_config = {
                        "r":       _lora_r,
                        "alpha":   float(_lora_alpha),
                        "lr_mult": 0.1,
                    }
                    logger.info(
                        "  Trial %d ULA config: latent=%d layers=%d heads=%d lora_r=%d",
                        trial.number, _ula_latent_dim, _ula_n_layers, _ula_n_heads, _lora_r,
                    )

                trial_graph_sparsity = max(0.0, float(trial_graph_sparsity))
                trial_diversity_weight = max(0.0, float(trial_diversity_weight))
                trial_uncertainty_aux = max(0.0, float(trial_uncertainty_aux))
                trial_graph_branch_weight = min(1.0, max(0.0, float(trial_graph_branch_weight)))
                trial_uncertainty_branch_weight = max(0.0, float(trial_uncertainty_branch_weight))
                trial_dropout = float(max(context_dropout_floor, float(trial_dropout)))
                if mean_uncertainty > 0.0:
                    uncertainty_dropout = min(
                        0.5,
                        max(0.1, 0.1 + 0.2 * float(mean_uncertainty)),
                    )
                    trial_dropout = max(trial_dropout, uncertainty_dropout)
                    trial_label_smoothing = max(
                        float(trial_label_smoothing),
                        float(min(0.25, max(0.0, 0.05 * float(mean_uncertainty)))),
                    )

                head_architecture_type = "mlp"
                head_hidden_dim = int(model_sel.get("hidden_dim", 256) or 256)
                head_num_layers = 3
                _ctx_current = self._get_ctx()
                if _ctx_current is not None:
                    head_architecture_type = str(
                        getattr(_ctx_current, "head_architecture_type", head_architecture_type)
                        or head_architecture_type
                    )
                    head_hidden_dim = int(
                        getattr(_ctx_current, "head_hidden_dim", head_hidden_dim)
                        or head_hidden_dim
                    )
                    head_num_layers = int(
                        getattr(_ctx_current, "head_num_layers", head_num_layers)
                        or head_num_layers
                    )

                fusion_config = {
                    "uncertainty_graph_weight": trial_graph_branch_weight,
                    "uncertainty_branch_weight": trial_uncertainty_branch_weight,
                }
                # Merge ULA-specific config when ULA fusion is selected
                if _ula_fusion_cfg:
                    fusion_config.update(_ula_fusion_cfg)
                fusion_aux_weights = {
                    "graph_sparsity_weight": trial_graph_sparsity,
                    "diversity_loss_weight": trial_diversity_weight,
                    "uncertainty_aux_weight": trial_uncertainty_aux,
                }

                ctx = self._get_ctx()
                if ctx is not None:
                    ctx_fusion = getattr(ctx, "fusion_strategy", None)
                    _locked = bool(getattr(ctx, "fusion_policy_locked", False))
                    _src    = getattr(ctx, "fusion_policy_source", "")
                    if ctx_fusion and (_locked and _src == "user_override"):
                        # Hard lock: only when user explicitly overrode fusion via the UI.
                        # Model-selection recommendations (selector_recommendation) do NOT
                        # lock — Optuna must freely sample fusion strategies so ULA, concat,
                        # attention etc. are all explored across trials.
                        trial_fusion = str(ctx_fusion)
                        logger.debug("Phase 5: user fusion lock enforced → %s", trial_fusion)
                    # ← removed soft-suggestion block: Optuna's trial.suggest_categorical()
                    # already samples from the correct priority-ordered choices (ULA first).

                # Bug 9: ULA cross-modal attention requires ≥2 active modalities
                _active_mods_ula = [k for k in input_dims if (input_dims.get(k) or 0) > 0]
                if (
                    str(trial_fusion).lower() in ("ula", "unified_latent", "unified_latent_alignment", "omnimodal")
                    and len(_active_mods_ula) < 2
                ):
                    logger.warning(
                        "Trial %d: ULA requires ≥2 modalities but only %s active — "
                        "falling back to concatenation",
                        trial.number, _active_mods_ula,
                    )
                    trial_fusion = "concatenation"

                if _canonical_fusion_strategy(trial_fusion) == "ula" and not _ula_fusion_cfg:
                    _base_latent = int(self.state.get_slot("ula_latent_dim") or 256)
                    _latent_choices = sorted(set([max(64, _base_latent // 2), _base_latent, min(512, _base_latent * 2)]))
                    _ula_latent_dim = trial.suggest_categorical("ula_latent_dim", _latent_choices)
                    _ula_n_layers = trial.suggest_int("ula_n_layers", 1, 4)
                    _ula_n_heads = trial.suggest_categorical("ula_n_heads", [2, 4, 8])
                    _lora_r = trial.suggest_categorical("lora_r", [4, 8, 16])
                    _lora_alpha = int(_lora_r * 2)
                    _ula_fusion_cfg = {
                        "latent_dim": _ula_latent_dim,
                        "n_layers": _ula_n_layers,
                        "n_heads": _ula_n_heads,
                    }
                    fusion_config.update(_ula_fusion_cfg)
                    _trial_lora_config = {
                        "r": _lora_r,
                        "alpha": float(_lora_alpha),
                        "lr_mult": 0.1,
                    }
                elif _canonical_fusion_strategy(trial_fusion) != "ula":
                    _trial_lora_config = None

                if progress_callback is not None:
                    progress_callback.set_phase(
                        5,
                        f"Trial {trial.number + 1}/{N_TRIALS}: "
                        f"fusion={trial_fusion}, lr={trial_lr:.2e}, epochs={trial_epochs}",
                        67 + int(trial.number / max(1, N_TRIALS) * 28),
                    )
                    progress_callback.add_message(
                        5, "info",
                        f"▶ Trial {trial.number+1}/{N_TRIALS}: "
                        f"fusion={trial_fusion}  lr={trial_lr:.2e}  "
                        f"dropout={trial_dropout:.2f}  epochs={trial_epochs}"
                    )
                    progress_callback.set_substage("trial_fit")
                    progress_callback.set_current_trial(
                        number=trial.number + 1,
                        total=N_TRIALS,
                        fusion=str(trial_fusion),
                        lr=float(trial_lr),
                        epochs=int(trial_epochs),
                        status="running",
                        current_epoch=0,
                        max_epoch=int(trial_epochs),
                    )
                    progress_callback.push_trial_event(
                        trial.number + 1,
                        "trial_start",
                        f"Trial {trial.number + 1}/{N_TRIALS} started",
                        {
                            "fusion": str(trial_fusion),
                            "learning_rate": float(trial_lr),
                            "weight_decay": float(trial_wd),
                            "dropout": float(trial_dropout),
                            "epochs": int(trial_epochs),
                            "label_smoothing": float(trial_label_smoothing),
                        },
                    )

                ContextValidator.require_fusion_consistency(
                    fusion_strategy=str(trial_fusion),
                    modalities=self.config.modalities,
                    phase="training",
                )
                ContextValidator.require_model_selection(ctx, phase="training")

                # Create a FRESH tabular encoder for this trial (trainable,
                # random init).  Image/text encoders are shared (frozen).
                _trial_tabular_encoder = None
                if _tabular_encoder_class is not None and _tabular_input_dim is not None:
                    _trial_tabular_encoder = _tabular_encoder_class(
                        input_dim=_tabular_input_dim,
                    )
                    try:
                        _trial_tabular_encoder.configure(encoder_plan.get("tabular", {}))
                    except Exception as cfg_exc:
                        logger.debug("  Trial tabular encoder configure failed: %s", cfg_exc)

                # Contrastive weight: activate when ≥2 modalities AND id_columns
                # detected by feature_intelligence (entity-linking signal)
                _contrastive_weight: float = 0.0
                _ctx_for_cw = self._get_ctx()
                if (
                    len(input_dims) >= 2
                    and _ctx_for_cw is not None
                ):
                    _fi_for_cw = getattr(_ctx_for_cw, "feature_intelligence", {}) or {}
                    _any_id_cols = any(
                        len(ds.get("id_columns") or []) > 0
                        for ds in _fi_for_cw.values()
                    )
                    if _any_id_cols:
                        _contrastive_weight = 0.05

                # ULA: use smaller, simpler head (ULA CLS token already cross-modal)
                if trial_fusion in ("ula", "unified_latent", "unified_latent_alignment"):
                    _ula_ld = int(fusion_config.get("latent_dim", 256))
                    head_hidden_dim  = _ula_ld * 2
                    head_num_layers  = 2
                    head_architecture_type = "mlp"

                _trial_uses_lora = bool(_trial_lora_config) and _canonical_fusion_strategy(trial_fusion) == "ula"
                _trial_text_encoder = _text_encoder
                _trial_image_encoder = _image_encoder
                _cache_snapshot: List[Tuple[Any, str, Any]] = []
                _fit_train_loader = train_loader
                _fit_val_loader = val_loader
                if _trial_uses_lora:
                    try:
                        import copy as _copy
                        _trial_text_encoder = _copy.deepcopy(_text_encoder) if _text_encoder is not None else None
                        _trial_image_encoder = _copy.deepcopy(_image_encoder) if _image_encoder is not None else None
                        _cache_snapshot = _snapshot_embedding_caches(_aug_ds, _clean_ds)
                        _clear_embedding_caches(_aug_ds, _clean_ds)
                        _fit_train_loader = DataLoader(
                            train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=False,
                            persistent_workers=False,
                        )
                        _fit_val_loader = DataLoader(
                            val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False,
                            persistent_workers=False,
                        )
                        logger.info(
                            "Trial %d: ULA+LoRA using raw encoder path; precomputed embedding caches disabled",
                            trial.number,
                        )
                    except Exception as _lora_iso_exc:
                        logger.warning(
                            "Trial %d: could not isolate LoRA encoders/cache (%s); disabling LoRA for this trial",
                            trial.number,
                            _lora_iso_exc,
                        )
                        _trial_lora_config = None
                        _trial_uses_lora = False
                        _trial_text_encoder = _text_encoder
                        _trial_image_encoder = _image_encoder
                        _restore_embedding_caches(_cache_snapshot)
                        _cache_snapshot = []

                lightning_module = build_trainer(
                    problem_type=problem_type,
                    num_classes=num_classes,
                    input_dims=input_dims,
                    learning_rate=trial_lr,
                    weight_decay=trial_wd,
                    dropout=trial_dropout,
                    max_epochs=trial_epochs,
                    hidden_dim=head_hidden_dim,
                    image_encoder=_trial_image_encoder,
                    text_encoder=_trial_text_encoder,
                    tabular_encoder=_trial_tabular_encoder,
                    class_weights=_class_weights,
                    fusion_strategy=trial_fusion,
                    fusion_config=fusion_config,
                    head_architecture_type=head_architecture_type,
                    head_num_layers=head_num_layers,
                    label_smoothing=float(trial_label_smoothing),
                    alignment_weight=float(trial_alignment),
                    modality_dropout_prob=float(trial_modality_dropout),
                    fusion_aux_weights=fusion_aux_weights,
                    execution_context=_ctx_for_cw,
                    contrastive_weight=_contrastive_weight,
                    ewc=getattr(self, "_ewc", None),
                    use_focal_loss=_use_focal_loss,
                    lora_config=_trial_lora_config,
                    tabular_tokenizer=self.fitted_transformers.get("tabular_tokenizer"),
                )

                # Part A.3 — LoRA warm-start: load previous best LoRA A/B weights
                # into encoders before training begins, giving each trial a head start.
                try:
                    _ctx_warm = self._get_ctx()
                    _lora_warm = dict(getattr(_ctx_warm, "lora_warm_start_state", {}) or {})
                    if _lora_warm and _trial_lora_config:
                        from modelss.adapters.lora import load_lora_state_dict as _llsd
                        if "_text_encoder" in _lora_warm and lightning_module._text_encoder is not None:
                            _llsd(lightning_module._text_encoder, _lora_warm["_text_encoder"])
                            logger.info("Trial %d: LoRA warm-start loaded for text encoder", trial.number)
                        if "_image_encoder" in _lora_warm and lightning_module._image_encoder is not None:
                            _llsd(lightning_module._image_encoder, _lora_warm["_image_encoder"])
                            logger.info("Trial %d: LoRA warm-start loaded for image encoder", trial.number)
                except Exception as _warm_exc:
                    logger.debug("LoRA warm-start load skipped: %s", _warm_exc)

                # Build a Lightning callback to push epoch metrics in real-time
                _pl_callbacks = []
                if progress_callback is not None:
                    class _EpochReporter(pl.Callback):
                        """Forwards per-epoch metrics to the API progress tracker."""
                        def on_validation_epoch_end(self, trainer, pl_module):
                            m = trainer.callback_metrics
                            _epoch = trainer.current_epoch + 1
                            _train_loss = float(m.get("train_loss", 0))
                            _val_loss = float(m.get("val_loss", 0))
                            _train_acc = float(m.get("train_acc", 0))
                            _val_acc = float(m.get("val_acc", 0))
                            _train_f1 = float(m.get("train_f1", 0))
                            _val_f1 = float(m.get("val_f1", 0))
                            _val_auroc = float(m.get("val_auroc", 0))
                            # ULA auxiliary losses — logged as "train/alignment_loss"
                            # and "train/contrastive_loss" with on_epoch=True
                            _align_loss = float(
                                m.get("train/alignment_loss",
                                      m.get("alignment_loss", 0)) or 0
                            )
                            _contrast_loss = float(
                                m.get("train/contrastive_loss",
                                      m.get("contrastive_loss", 0)) or 0
                            )
                            # Also pull from the module's internal history as fallback
                            if _align_loss == 0 and hasattr(pl_module, "_alignment_loss_history"):
                                _h = pl_module._alignment_loss_history
                                if _h:
                                    _align_loss = float(_h[-1])
                            if _contrast_loss == 0 and hasattr(pl_module, "_contrastive_loss_history"):
                                _h = pl_module._contrastive_loss_history
                                if _h:
                                    _contrast_loss = float(_h[-1])
                            progress_callback.log_epoch(
                                trial=trial.number,
                                epoch=_epoch,
                                max_epoch=trial_epochs,
                                train_loss=_train_loss,
                                val_loss=_val_loss,
                                train_acc=_train_acc,
                                val_acc=_val_acc,
                                train_f1=_train_f1,
                                val_f1=_val_f1,
                                val_auroc=_val_auroc,
                                alignment_loss=_align_loss,
                                contrastive_loss=_contrast_loss,
                            )
                            progress_callback.push_trial_event(
                                trial.number + 1,
                                "epoch",
                                f"Epoch {_epoch}/{trial_epochs} val_loss={_val_loss:.4f}",
                                {
                                    "epoch": int(_epoch),
                                    "max_epoch": int(trial_epochs),
                                    "train_loss": _train_loss,
                                    "val_loss": _val_loss,
                                    "val_acc": _val_acc,
                                    "val_f1": _val_f1,
                                    "val_auroc": _val_auroc,
                                },
                            )
                    _pl_callbacks.append(_EpochReporter())

                # EarlyStopping: halt training if val_loss stalls for 5 epochs.
                # restore_best_weights via ModelCheckpoint is implicit — Lightning
                # keeps the in-memory model at the last epoch, and we track the
                # best module ourselves via _best_val / _best_module_ref.
                from pytorch_lightning.callbacks import EarlyStopping as _EarlyStopping
                _pl_callbacks.append(_EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    patience=max(1, int(early_stop_patience)),
                    verbose=False,
                ))

                # PCGrad — Gradient Surgery (Yu et al., NeurIPS 2020)
                # Prevents destructive gradient interference between modality encoders.
                # Only activate when 2+ encoder types are present.
                _active_modalities = set(input_dims.keys())
                if len(_active_modalities) >= 2:
                    from automl.trainer import PCGradCallback as _PCGradCallback
                    _pl_callbacks.append(_PCGradCallback())

                # Optuna pruning: kill unpromising trials early (skip for manual HP runs).
                # Requires optuna-integration[pytorch_lightning]; falls back gracefully
                # if not installed — HyperbandPruner still prunes between trials.
                _pruning_available = False
                _pruning_backend = "disabled"
                _pruning_reason = ""
                if not hp_overrides:
                    try:
                        from optuna.integration import PyTorchLightningPruningCallback
                        _pl_callbacks.append(PyTorchLightningPruningCallback(
                            trial, monitor="val_loss",
                        ))
                        _pruning_available = True
                        _pruning_backend = "optuna_integration[pytorch_lightning]"
                    except (ModuleNotFoundError, ImportError):
                        _optuna_callback = OptunaCallback(trial)

                        class _InlineOptunaPruningCallback(pl.Callback):
                            """Fallback Optuna pruning bridge when optuna-integration is unavailable."""

                            def on_validation_epoch_end(self, trainer, pl_module):
                                m = trainer.callback_metrics
                                _optuna_callback.on_epoch_end(
                                    epoch=trainer.current_epoch + 1,
                                    train_loss=float(m.get("train_loss", 0.0)),
                                    val_loss=float(m.get("val_loss", float("inf"))),
                                )

                        _pl_callbacks.append(_InlineOptunaPruningCallback())
                        _pruning_available = True
                        _pruning_backend = "inline_optuna_callback"
                        _pruning_reason = (
                            "optuna-integration[pytorch_lightning] unavailable; "
                            "using internal epoch-end pruning bridge."
                        )
                        logger.warning(_pruning_reason)
                else:
                    _pruning_reason = "Manual hyperparameter override run; intra-trial pruning disabled."

                if progress_callback is not None:
                    pruning_runtime.update(
                        {
                            "available": _pruning_available,
                            "backend": _pruning_backend,
                            "reason": _pruning_reason,
                        }
                    )
                    progress_callback.set_pruning_status(
                        available=_pruning_available,
                        backend=_pruning_backend,
                        reason=_pruning_reason,
                        pruned_count=sum(
                            1
                            for t in getattr(study, "trials", [])
                            if getattr(t, "state", None) == optuna.trial.TrialState.PRUNED
                        ),
                        completed_count=sum(
                            1
                            for t in getattr(study, "trials", [])
                            if getattr(t, "state", None) == optuna.trial.TrialState.COMPLETE
                        ),
                    )

                # SWA — Stochastic Weight Averaging (Izmailov et al., UAI 2018)
                # Averages weights along SGD trajectory for flatter minima and
                # better generalization, especially on small/medium datasets.
                # Activates in the last 10% of epochs (min 1 epoch).
                try:
                    from pytorch_lightning.callbacks import StochasticWeightAveraging as _SWA
                    _swa_start = max(0, int(trial_epochs * 0.9) - 1)  # last 10% of epochs
                    if trial_epochs >= 5:
                        _pl_callbacks.append(_SWA(
                            swa_lrs=float(trial_lr) * 0.5,
                            swa_epoch_start=_swa_start,
                            annealing_epochs=max(1, trial_epochs - _swa_start),
                            device=None,
                        ))
                        logger.info(
                            "  SWA [UAI 2018] enabled: start_epoch=%d  swa_lr=%.6f",
                            _swa_start, float(trial_lr) * 0.5,
                        )
                except Exception as _swa_exc:
                    logger.debug("SWA not available: %s", _swa_exc)

                pl_trainer = pl.Trainer(
                    max_epochs=trial_epochs,
                    accelerator=accelerator,
                    devices=1,
                    # AMP 16-bit mixed precision for GPU tensor core saturation;
                    # falls back to 32-bit on CPU where float16 is unsupported.
                    precision="16-mixed" if accelerator == "gpu" else "32-true",
                    enable_checkpointing=False,
                    enable_progress_bar=False,
                    num_sanity_val_steps=0,
                    logger=False,
                    callbacks=_pl_callbacks,
                )

                best_val_loss = float("inf")
                trial_val_acc = 0.0
                trial_val_f1  = 0.0
                trial_train_acc = 0.0

                try:
                    with mlflow.start_run(nested=True):
                        mlflow.log_params({
                            "trial":        trial.number,
                            "learning_rate": trial_lr,
                            "weight_decay":  trial_wd,
                            "dropout":       trial_dropout,
                            "label_smoothing": float(trial_label_smoothing),
                            "epochs":        trial_epochs,
                            "fusion_strategy": str(trial_fusion),
                            "alignment_weight": float(trial_alignment),
                            "modality_dropout_prob": float(trial_modality_dropout),
                            "graph_sparsity_weight": trial_graph_sparsity,
                            "diversity_loss_weight": trial_diversity_weight,
                            "uncertainty_aux_weight": trial_uncertainty_aux,
                            "uncertainty_graph_weight": trial_graph_branch_weight,
                            "uncertainty_branch_weight": trial_uncertainty_branch_weight,
                            "problem_type":  problem_type,
                            "num_classes":   num_classes,
                        })

                        try:
                            pl_trainer.fit(lightning_module, _fit_train_loader, _fit_val_loader)
                            # Retrieve best val_loss logged during training
                            cb_metrics = pl_trainer.callback_metrics
                            best_val_loss = float(cb_metrics.get("val_loss", float("inf")))
                            trial_val_acc = float(cb_metrics.get("val_acc", 0))
                            trial_val_f1  = float(cb_metrics.get("val_f1", 0))
                            trial_train_acc = float(cb_metrics.get("train_acc", 0))
                        except optuna.exceptions.TrialPruned:
                            logger.info("  Trial %d PRUNED by Optuna", trial.number)
                            _pruned_step = len(trial.intermediate_values)
                            if progress_callback is not None:
                                progress_callback.add_message(
                                    5, "pruned",
                                    f"Trial {trial.number + 1}/{N_TRIALS} pruned at epoch "
                                    f"{_pruned_step} — val_loss above median threshold",
                                )
                                # Sentinel record so the chart draws a prune marker
                                progress_callback.log_epoch(
                                    trial=trial.number,
                                    epoch=_pruned_step,
                                    max_epoch=trial_epochs,
                                    train_loss=-1.0,
                                    val_loss=-1.0,
                                    pruned=True,
                                )
                                progress_callback.set_current_trial(
                                    number=trial.number + 1,
                                    total=N_TRIALS,
                                    fusion=str(trial_fusion),
                                    lr=float(trial_lr),
                                    epochs=int(trial_epochs),
                                    status="pruned",
                                    current_epoch=int(_pruned_step),
                                    max_epoch=int(trial_epochs),
                                )
                                progress_callback.push_trial_event(
                                    trial.number + 1,
                                    "pruned",
                                    f"Trial pruned at epoch {_pruned_step}",
                                    {
                                        "epoch": int(_pruned_step),
                                        "fusion": str(trial_fusion),
                                        "learning_rate": float(trial_lr),
                                    },
                                )
                                progress_callback.set_pruning_status(
                                    available=_pruning_available,
                                    backend=_pruning_backend,
                                    reason=_pruning_reason,
                                    pruned_count=(
                                        sum(
                                            1
                                            for t in getattr(study, "trials", [])
                                            if getattr(t, "state", None) == optuna.trial.TrialState.PRUNED
                                        )
                                        + 1
                                    ),
                                    completed_count=sum(
                                        1
                                        for t in getattr(study, "trials", [])
                                        if getattr(t, "state", None) == optuna.trial.TrialState.COMPLETE
                                    ),
                                )
                            # Capture pruning metadata for warm-start guidance
                            _best_inter  = (
                                min(trial.intermediate_values.values())
                                if trial.intermediate_values else float("inf")
                            )
                            try:
                                trial.set_user_attr("pruned_at_step", _pruned_step)
                                trial.set_user_attr("best_intermediate_loss", _best_inter)
                                trial.set_user_attr("lr_at_prune", trial_lr)
                                trial.set_user_attr("fusion_at_prune", trial_fusion)
                            except Exception:
                                pass
                            raise  # re-raise so Optuna marks the trial as pruned
                        except Exception as trial_exc:
                            try:
                                trial.set_user_attr("fail_reason", str(trial_exc))
                            except Exception:
                                pass
                            logger.warning("  Trial %d error: %s", trial.number, trial_exc)
                            if progress_callback is not None:
                                progress_callback.set_current_trial(
                                    number=trial.number + 1,
                                    total=N_TRIALS,
                                    fusion=str(trial_fusion),
                                    lr=float(trial_lr),
                                    epochs=int(trial_epochs),
                                    status="failed",
                                    current_epoch=0,
                                    max_epoch=int(trial_epochs),
                                )
                                progress_callback.push_trial_event(
                                    trial.number + 1,
                                    "warning",
                                    f"Trial failed: {trial_exc}",
                                    {"error": str(trial_exc)},
                                )

                        mlflow.log_metric("val_loss", best_val_loss)

                    if progress_callback is not None:
                        progress_callback.set_substage("trial_analysis")

                    # Persist TrialIntelligence diagnostics on the Optuna trial.
                    try:
                        trial_diag: Dict[str, Any] = {}
                        if hasattr(lightning_module, "loss_weight_scheduler"):
                            trial_diag = dict(
                                getattr(
                                    lightning_module.loss_weight_scheduler,
                                    "last_analysis",
                                    {},
                                )
                                or {}
                            )
                            dynamic_factors = dict(
                                getattr(
                                    lightning_module.loss_weight_scheduler,
                                    "dynamic_factors",
                                    {},
                                )
                                or {}
                            )
                            if dynamic_factors:
                                trial_diag["dynamic_factors"] = dynamic_factors
                        if trial_diag:
                            calibration_proxy = _calibration_proxy_from_gap(
                                trial_diag.get("generalization_gap")
                            )
                            if calibration_proxy is not None:
                                trial_diag["calibration_proxy"] = calibration_proxy

                            trial_diag["adaptive_penalty"] = _adaptive_penalty_from_diag(trial_diag)

                            trial.set_user_attr(
                                "fit_type",
                                str(trial_diag.get("fit_type", "unknown")),
                            )
                            if trial_diag.get("calibration_proxy") is not None:
                                trial.set_user_attr(
                                    "calibration_proxy",
                                    float(trial_diag.get("calibration_proxy", 0.0)),
                                )
                            trial.set_user_attr(
                                "adaptive_penalty",
                                float(trial_diag.get("adaptive_penalty", 0.0)),
                            )
                            trial.set_user_attr("trial_diagnostics", trial_diag)

                            if not hp_overrides:
                                updated_state = self.optuna_adaptive.update_from_trial_diagnostics(
                                    adaptive_feedback_state,
                                    trial_diag,
                                )
                                adaptive_feedback_state.clear()
                                adaptive_feedback_state.update(updated_state)

                                # G20: pass recent finished trials so prune-step cap can apply
                                _recent_trials_g20 = [
                                    t for t in study.trials if t.state.is_finished()
                                ][-6:]
                                next_overrides = self.optuna_adaptive.next_trial_overrides(
                                    adaptive_feedback_state,
                                    hpo_space,
                                    recent_trials=_recent_trials_g20,
                                )
                                adaptive_trial_overrides.clear()
                                adaptive_trial_overrides.update(next_overrides)

                                if next_overrides:
                                    event = {
                                        "after_trial": int(trial.number),
                                        "fit_type": str(trial_diag.get("fit_type", "unknown")),
                                        "adaptive_penalty": float(
                                            trial_diag.get("adaptive_penalty", 0.0)
                                        ),
                                        "calibration_proxy": trial_diag.get("calibration_proxy"),
                                        "overrides": dict(next_overrides),
                                    }
                                    trial_feedback_events.append(event)
                                    if progress_callback is not None:
                                        progress_callback.set_next_trial_plan(dict(next_overrides))
                                        progress_callback.push_trial_event(
                                            trial.number + 1,
                                            "next_trial_plan",
                                            "Adaptive overrides prepared for the next trial",
                                            dict(next_overrides),
                                        )
                    except Exception as diag_exc:
                        logger.debug("  Trial diagnostics capture failed: %s", diag_exc)

                    # Capture best module – mutable list avoids 'nonlocal'
                    _prev_best = _best_val[0]   # capture BEFORE possible update
                    if best_val_loss < _best_val[0]:
                        _best_val[0] = best_val_loss
                        _best_module_ref.clear()
                        _best_module_ref.append(lightning_module)
                        _best_metrics["val_acc"]   = trial_val_acc
                        _best_metrics["val_f1"]    = trial_val_f1
                        _best_metrics["train_acc"] = trial_train_acc
                        _best_lora_config.clear()
                        if _trial_lora_config:
                            _best_lora_config.update(dict(_trial_lora_config))
                        _best_fusion_strategy[0] = _canonical_fusion_strategy(trial_fusion)

                    logger.info(
                        "  Trial %d: lr=%.2e  wd=%.2e  dropout=%.2f  "
                        "label_smoothing=%.3f  epochs=%d  val_loss=%.4f  fusion=%s",
                        trial.number, trial_lr, trial_wd, trial_dropout,
                        trial_label_smoothing, trial_epochs, best_val_loss, trial_fusion,
                    )
                    if progress_callback is not None:
                        _is_new_best = best_val_loss < _prev_best
                        progress_callback.add_message(
                            5, "success" if _is_new_best else "info",
                            f"Trial {trial.number+1}/{N_TRIALS} done — "
                            f"val_loss={best_val_loss:.4f}  "
                            f"val_acc={trial_val_acc:.3f}  "
                            f"val_f1={trial_val_f1:.3f}"
                            + ("  ★ new best" if _is_new_best else ""),
                        )
                        progress_callback.set_current_trial(
                            number=trial.number + 1,
                            total=N_TRIALS,
                            fusion=str(trial_fusion),
                            lr=float(trial_lr),
                            epochs=int(trial_epochs),
                            status="completed",
                            current_epoch=int(trial_epochs),
                            max_epoch=int(trial_epochs),
                        )
                        progress_callback.push_trial_event(
                            trial.number + 1,
                            "trial_complete",
                            f"Trial completed with val_loss={best_val_loss:.4f}",
                            {
                                "val_loss": float(best_val_loss),
                                "val_acc": float(trial_val_acc),
                                "val_f1": float(trial_val_f1),
                                "fusion": str(trial_fusion),
                            },
                        )
                        progress_callback.set_pruning_status(
                            available=_pruning_available,
                            backend=_pruning_backend,
                            reason=_pruning_reason,
                            pruned_count=sum(
                                1
                                for t in getattr(study, "trials", [])
                                if getattr(t, "state", None) == optuna.trial.TrialState.PRUNED
                            ),
                            completed_count=(
                                sum(
                                    1
                                    for t in getattr(study, "trials", [])
                                    if getattr(t, "state", None) == optuna.trial.TrialState.COMPLETE
                                )
                                + 1
                            ),
                        )
                        if _is_new_best:
                            progress_callback.set_best_so_far(
                                trial=trial.number + 1,
                                val_loss=float(best_val_loss),
                                val_acc=float(trial_val_acc),
                                val_f1=float(trial_val_f1),
                            )
                            progress_callback.push_trial_event(
                                trial.number + 1,
                                "new_best",
                                f"New best trial with val_loss={best_val_loss:.4f}",
                                {
                                    "val_loss": float(best_val_loss),
                                    "val_acc": float(trial_val_acc),
                                    "val_f1": float(trial_val_f1),
                                },
                            )

                    return best_val_loss

                finally:
                    # ── GPU cleanup: runs for success, error, AND TrialPruned ──
                    # Previously only in the normal return path, meaning pruned
                    # trials leaked GPU memory ("zombie trial" compute leak).
                    _restore_embedding_caches(_cache_snapshot)
                    del pl_trainer
                    _is_best = bool(_best_module_ref) and _best_module_ref[0] is lightning_module
                    if not _is_best:
                        lightning_module.cpu()
                        del lightning_module
                    import gc; gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            with mlflow.start_run(run_name="phase5_optuna"):
                # HyperbandPruner: aggressively prunes underperforming trials
                # at intermediate epochs, saving GPU compute.
                _pruner = optuna.pruners.HyperbandPruner(
                    min_resource=1,
                    max_resource=hpo_space.get("epochs", {}).get("high", model_sel.get("epochs", 10)),
                    reduction_factor=3,
                ) if N_TRIALS > 1 else optuna.pruners.NopPruner()
                study = optuna.create_study(
                    direction="minimize",
                    pruner=_pruner,
                    sampler=optuna.samplers.TPESampler(seed=_APEX_SEED),
                )

                # If manual overrides provided, enqueue them as trial 0 so TPE can
                # learn from the user's starting point and explore around it.
                if hp_overrides and hpo_space:
                    _enqueue_vals = {k: v for k, v in hp_overrides.items() if k in hpo_space}
                    if _enqueue_vals:
                        try:
                            study.enqueue_trial(_enqueue_vals)
                            logger.info("  Enqueued override as trial 0: %s", _enqueue_vals)
                        except Exception as _eq_exc:
                            logger.warning("enqueue_trial failed (non-fatal): %s", _eq_exc)

                _ctx_for_warm = self._get_ctx()
                _warm_params = (
                    getattr(_ctx_for_warm, "warm_start_params", None)
                    if _ctx_for_warm is not None
                    else None
                )
                if isinstance(_warm_params, dict) and _warm_params:
                    self.optuna_adaptive.seed_from_warm_start(
                        study=study,
                        warm_params=_warm_params,
                        hpo_space=hpo_space,
                    )

                study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
                if progress_callback is not None:
                    progress_callback.set_substage("phase5_summary")
                    progress_callback.set_pruning_status(
                        available=bool(pruning_runtime.get("available", False)),
                        backend=str(pruning_runtime.get("backend", "disabled") or "disabled"),
                        reason=str(pruning_runtime.get("reason", "") or "Phase 5 study completed."),
                        pruned_count=sum(
                            1
                            for t in study.trials
                            if getattr(t, "state", None) == optuna.trial.TrialState.PRUNED
                        ),
                        completed_count=sum(
                            1
                            for t in study.trials
                            if getattr(t, "state", None) == optuna.trial.TrialState.COMPLETE
                        ),
                    )
                    progress_callback.push_trial_event(
                        None,
                        "study_complete",
                        f"Optuna study finished: {N_TRIALS} trial(s)",
                        {
                            "n_trials": int(N_TRIALS),
                            "pruned_count": int(
                                sum(
                                    1
                                    for t in study.trials
                                    if getattr(t, "state", None) == optuna.trial.TrialState.PRUNED
                                )
                            ),
                            "completed_count": int(
                                sum(
                                    1
                                    for t in study.trials
                                    if getattr(t, "state", None) == optuna.trial.TrialState.COMPLETE
                                )
                            ),
                        },
                    )

            # Derive next-trial overrides from pruning patterns and store on context
            try:
                import statistics as _stats
                _pruned_trials = [
                    t for t in study.trials
                    if getattr(t, "state", None) and "PRUNED" in str(t.state)
                ]
                _completed_trials = [
                    t for t in study.trials
                    if getattr(t, "state", None) and "COMPLETE" in str(t.state)
                ]
                _next_overrides: Dict[str, Any] = {}

                if _pruned_trials:
                    _prune_steps = [
                        t.user_attrs.get("pruned_at_step", 5) for t in _pruned_trials
                    ]
                    _med_step = _stats.median(_prune_steps)
                    if _med_step <= 2:
                        # Very early pruning → reduce complexity next run
                        _next_overrides["ula_n_layers_max"] = 2
                        _next_overrides["ula_latent_dim_choices"] = [128, 256]
                        _next_overrides["learning_rate_max"] = 5e-4
                    elif _med_step >= max(8, N_TRIALS * 0.8):
                        # Late pruning → need more capacity
                        _next_overrides["ula_latent_dim_choices"] = [256, 512]
                        _next_overrides["ula_n_layers_max"] = 4

                if _completed_trials:
                    _best_done = min(_completed_trials, key=lambda t: t.value or float("inf"))
                    _next_overrides["lr_center"] = _best_done.params.get("learning_rate", 1e-3)
                    _next_overrides["lora_r_hint"] = _best_done.params.get("lora_r", 8)

                _ctx_for_next = self._get_ctx()
                if _ctx_for_next is not None and _next_overrides:
                    try:
                        object.__setattr__(
                            _ctx_for_next, "next_trial_overrides",
                            {**dict(getattr(_ctx_for_next, "next_trial_overrides", {}) or {}),
                             **_next_overrides},
                        )
                        logger.info("Phase 5: next_trial_overrides derived: %s", _next_overrides)
                        if progress_callback is not None:
                            progress_callback.set_next_trial_plan(dict(_next_overrides))
                            progress_callback.push_trial_event(
                                None,
                                "next_trial_plan",
                                "Derived next-trial guidance for the next run",
                                dict(_next_overrides),
                            )
                    except Exception:
                        pass

                # LoRA warm-start: save best trial's adapter weights for next training run
                if (_best_module_ref and _ctx_for_next is not None):
                    try:
                        from modelss.adapters.lora import lora_state_dict as _lsd
                        _lora_warm = {}
                        _best_for_warm = _best_module_ref[0]
                        for _enc_name in ("_text_encoder", "_image_encoder"):
                            _enc = getattr(_best_for_warm, _enc_name, None)
                            if _enc is not None:
                                _enc_state = _lsd(_enc)
                                if _enc_state:
                                    _lora_warm[_enc_name] = _enc_state
                        if _lora_warm:
                            object.__setattr__(_ctx_for_next, "lora_warm_start_state", _lora_warm)
                            logger.info("Phase 5: LoRA warm-start state saved (%d encoder(s))", len(_lora_warm))
                    except Exception as _ws_exc:
                        logger.debug("LoRA warm-start save failed: %s", _ws_exc)
            except Exception as _post_exc:
                logger.debug("Post-study analysis failed: %s", _post_exc)

            # Persist best model reference so Phase 7 can serialise weights
            if _best_module_ref:
                self.best_lightning_module = _best_module_ref[0]

            # ── Validity guard: fail fast when every trial errored or was pruned ──
            # Proceeding into the registry with best_val_loss=inf and no trained
            # module would silently register a non-deployable "successful" model.
            _n_complete = sum(
                1 for t in study.trials
                if getattr(t, "value", None) is not None and np.isfinite(float(t.value))
            )
            if _n_complete == 0 or not _best_module_ref:
                _trial_errors = [
                    str((getattr(t, "user_attrs", {}) or {}).get("fail_reason", "")).strip()
                    for t in study.trials
                    if str((getattr(t, "user_attrs", {}) or {}).get("fail_reason", "")).strip()
                ]
                if progress_callback is not None:
                    progress_callback.add_message(
                        5, "warning",
                        f"All {N_TRIALS} trials failed or were pruned before completing. "
                        "Cannot register a model. Check errors above.",
                    )
                    progress_callback.push_trial_event(
                        None,
                        "warning",
                        "All Phase 5 trials were invalid; training cannot continue to registry.",
                        {"n_trials": int(N_TRIALS), "errors": list(_trial_errors[:3])},
                    )
                raise RuntimeError(
                    f"Phase 5: no valid training trial completed "
                    f"({N_TRIALS} trials run, 0 finished). "
                    f"First trial error hint: {_trial_errors[0] if _trial_errors else 'see logs'}. "
                    "Fix the underlying error and retry."
                )

            best = study.best_trial
            best_val_loss: float = best.value if best.value is not None else float("inf")

            best_fit_type = str(best.user_attrs.get("fit_type", "unknown"))
            if best_fit_type == "unknown" and self.best_lightning_module is not None:
                try:
                    _best_diag = dict(
                        getattr(
                            self.best_lightning_module.loss_weight_scheduler,
                            "last_analysis",
                            {},
                        )
                        or {}
                    )
                    best_fit_type = str(_best_diag.get("fit_type", "unknown"))
                except Exception:
                    best_fit_type = "unknown"

            trial_diagnostics: List[Dict[str, Any]] = []
            for t in study.trials:
                if t.state != optuna.trial.TrialState.COMPLETE:
                    continue
                _diag = t.user_attrs.get("trial_diagnostics", {})
                trial_diagnostics.append(
                    {
                        "trial": int(t.number),
                        "fit_type": t.user_attrs.get("fit_type", "unknown"),
                        "val_loss": float(t.value) if t.value is not None else None,
                        "train_slope": _diag.get("train_slope") if isinstance(_diag, dict) else None,
                        "val_slope": _diag.get("val_slope") if isinstance(_diag, dict) else None,
                        "generalization_gap": (
                            _diag.get("generalization_gap") if isinstance(_diag, dict) else None
                        ),
                        "dynamic_factors": (
                            _diag.get("dynamic_factors") if isinstance(_diag, dict) else None
                        ),
                        "calibration_proxy": (
                            _diag.get("calibration_proxy") if isinstance(_diag, dict) else t.user_attrs.get("calibration_proxy")
                        ),
                        "adaptive_penalty": (
                            _diag.get("adaptive_penalty") if isinstance(_diag, dict) else t.user_attrs.get("adaptive_penalty")
                        ),
                    }
                )

            lw_schedule_history = [
                {
                    "trial": row.get("trial"),
                    "fit_type": row.get("fit_type"),
                    "dynamic_factors": row.get("dynamic_factors") or {},
                }
                for row in trial_diagnostics
            ]

            trial_feedback_summary: Dict[str, Any] = {}
            try:
                from automl.trial_intelligence import TrialIntelligence

                _trial_intelligence = TrialIntelligence()
                trial_feedback_summary = _trial_intelligence.summarize_trials(trial_diagnostics)
            except Exception as summary_exc:
                logger.debug("  Trial feedback summary unavailable: %s", summary_exc)

            ctx_for_feedback = self._get_ctx()
            feedback_modalities = schema_info.get("global_modalities", self.config.modalities)
            feedback_importance: Dict[str, float] = {}
            if ctx_for_feedback is not None:
                feedback_importance = dict(
                    getattr(ctx_for_feedback, "modality_importance", {}) or {}
                )

            next_run_feedback = self.optuna_adaptive.build_next_run_feedback(
                trial_summary=trial_feedback_summary,
                feedback_state=adaptive_feedback_state,
                hpo_space=hpo_space,
                modalities=feedback_modalities,
                modality_importance=feedback_importance,
            )

            meta_saved = False
            try:
                from automl.advanced_selector import AdvancedModelSelector

                selector_for_memory = AdvancedModelSelector()
                best_fusion = best.params.get(
                    "fusion_strategy",
                    model_sel.get(
                        "fusion_strategy",
                        self.state.get_slot("schema_derived_fusion") or "concatenation",
                    ),
                )

                loss_weights: Dict[str, float] = {}
                if self.best_lightning_module is not None and hasattr(
                    self.best_lightning_module, "get_loss_weight_state"
                ):
                    loss_weights = self.best_lightning_module.get_loss_weight_state()

                performance = (
                    float(_best_metrics["val_acc"])
                    if problem_type.startswith("classification")
                    or problem_type == "multilabel_classification"
                    else float(-best_val_loss)
                )

                selector_for_memory.record_experiment(
                    dataset_meta=model_sel.get(
                        "dataset_meta",
                        {
                            "num_rows": n_total,
                            "num_cols": len(input_dims),
                            "modalities": schema_info.get("global_modalities", self.config.modalities),
                            "target_type": "regression" if "regression" in problem_type else "classification",
                        },
                    ),
                    best_params=dict(best.params),
                    fusion_strategy=str(best_fusion),
                    loss_weights=loss_weights,
                    performance=performance,
                )
                meta_saved = True
            except Exception as meta_exc:
                logger.warning("  Meta-learning persistence skipped: %s", meta_exc)

            representation_summary: Dict[str, Any] = {}
            alignment_summary: Dict[str, float] = {}
            fusion_summary: Dict[str, Any] = {}
            fusion_aux_weights: Dict[str, float] = {}
            if self.best_lightning_module is not None:
                try:
                    representation_summary = self.representation_layer.summarize(
                        getattr(self.best_lightning_module, "_last_encoded_batch", {}),
                    )
                except Exception as rep_exc:
                    logger.debug("  Representation summary unavailable: %s", rep_exc)
                if hasattr(self.best_lightning_module, "get_alignment_summary"):
                    try:
                        alignment_summary = self.best_lightning_module.get_alignment_summary()
                    except Exception as align_exc:
                        logger.debug("  Alignment summary unavailable: %s", align_exc)
                if hasattr(self.best_lightning_module, "get_fusion_summary"):
                    try:
                        fusion_summary = dict(self.best_lightning_module.get_fusion_summary() or {})
                        fusion_aux_weights = dict(
                            fusion_summary.get("auxiliary_loss_weights", {}) or {}
                        )
                    except Exception as fusion_exc:
                        logger.debug("  Fusion summary unavailable: %s", fusion_exc)

            best_fusion_strategy = _canonical_fusion_strategy(
                _best_fusion_strategy[0]
                or fusion_summary.get("strategy")
                or best.params.get("fusion_strategy")
                or model_sel.get("fusion_strategy")
                or self.state.get_slot("schema_derived_fusion")
                or "concatenation"
            )
            if fusion_summary:
                fusion_summary["strategy"] = best_fusion_strategy
            best_lora_config = dict(_best_lora_config or {})

            # ----------------------------------------------------------------
            # XAI Artifacts — generated from one val batch after best trial
            # Fault-isolated: failure here does not abort Phase 5.
            # ----------------------------------------------------------------
            xai_artifacts: Dict[str, Any] = {}
            if self.best_lightning_module is not None:
                try:
                    _xai_batch = next(iter(val_loader))
                    _active_modalities = schema_info.get(
                        "global_modalities", self.config.modalities
                    )
                    xai_artifacts = generate_xai_artifacts(
                        model=self.best_lightning_module,
                        batch=_xai_batch,
                        modalities=list(_active_modalities),
                        execution_context=self._get_ctx(),
                    )
                    logger.info("  XAI artifacts generated for modalities: %s", _active_modalities)
                except Exception as xai_exc:
                    logger.warning("  XAI artifact generation skipped: %s", xai_exc)
                    xai_artifacts = {"error": str(xai_exc)}

            # Count pruned trials for frontend transparency
            _n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
            _n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

            # ----------------------------------------------------------------
            # 5b  Probability calibration on held-out validation split
            # ----------------------------------------------------------------
            calibration: Dict[str, Any] = {"enabled": False, "mode": "identity"}
            supports_calibration = (
                problem_type.startswith("classification")
                or problem_type == "multilabel_classification"
            )
            if supports_calibration and self.best_lightning_module is not None:
                try:
                    logits_np, targets_np = self._collect_validation_logits(
                        self.best_lightning_module,
                        val_loader,
                    )
                    calibrator = ProbabilityCalibrator()
                    calibration = calibrator.fit(
                        logits_np,
                        targets_np,
                        problem_type=problem_type,
                        execution_context=self._get_ctx(),
                    )
                    if calibration.get("enabled"):
                        self.probability_calibrator = calibrator
                        logger.info(
                            "  Calibration fitted: mode=%s",
                            calibration.get("mode", "unknown"),
                        )
                    else:
                        self.probability_calibrator = None
                        logger.info(
                            "  Calibration skipped: %s",
                            calibration.get("reason", "not enabled"),
                        )
                except Exception as cal_exc:
                    logger.warning("  Calibration fitting skipped: %s", cal_exc)
                    self.probability_calibrator = None
                    calibration = {
                        "enabled": False,
                        "mode": "identity",
                        "error": str(cal_exc),
                    }
            else:
                self.probability_calibrator = None

            # ----------------------------------------------------------------
            # 6  Build phase results compatible with /train-pipeline contract
            # ----------------------------------------------------------------
            trial_feedback_summary = dict(trial_feedback_summary or {})
            trial_feedback_summary["best_fit_type"] = best_fit_type
            trial_feedback_summary["posthoc_calibration"] = dict(calibration or {})

            elapsed = time.time() - phase_start

            results: Dict[str, Any] = {
                "best_trial":        best.number,
                "best_params":       best.params,
                "best_val_loss":     best_val_loss,
                "best_val_acc":      _best_metrics["val_acc"],
                "best_val_f1":       _best_metrics["val_f1"],
                "best_train_acc":    _best_metrics["train_acc"],
                "n_trials":          N_TRIALS,
                "n_pruned":          _n_pruned,
                "n_complete":        _n_complete,
                "fit_type":          best_fit_type,
                "trial_diagnostics": trial_diagnostics,
                "lw_schedule_history": lw_schedule_history,
                "trial_feedback_events": trial_feedback_events,
                "trial_feedback_summary": trial_feedback_summary,
                "next_run_feedback": next_run_feedback,
                "adaptive_feedback_state": dict(adaptive_feedback_state),
                "batch_size":        batch_size,
                "data_split":        {"train": n_train, "val": n_val, "total": n_total},
                "problem_type":      problem_type,
                "num_classes":       num_classes,
                "input_dims":        input_dims,
                "fusion_strategy":   best_fusion_strategy,
                "best_lora_config":  best_lora_config,
                "hpo_space":         hpo_space,
                "duration_seconds":  elapsed,
                # Scalar defaults from Phase 4 (kept for Phase 6 / registry)
                "epochs":            model_sel.get("epochs", 10),
                "learning_rate":     model_sel.get("learning_rate", 1e-3),
                # JIT encoder selection metadata
                "encoder_selection": {
                    "method": _jit_result.selection_method,
                    "image_encoder": _jit_result.image_encoder_name,
                    "text_encoder": _jit_result.text_encoder_name,
                    "tabular_encoder": _jit_result.tabular_encoder_name,
                    "total_capacity": _jit_result.total_capacity,
                    "peak_memory_mb": round(_jit_result.total_peak_memory_bytes / 1e6, 2),
                    "vram_budget_mb": round(_jit_result.vram_budget_bytes / 1e6, 2),
                    "rationale": _jit_result.rationale,
                },
                "meta_learning_saved": meta_saved,
                "embedding_cache": self.embedding_cache.stats(),
                "representation_summary": representation_summary,
                "alignment_summary": alignment_summary,
                "alignment_loss_history":   getattr(self.best_lightning_module, "_alignment_loss_history", []),
                "contrastive_loss_history": getattr(self.best_lightning_module, "_contrastive_loss_history", []),
                "fusion_summary": fusion_summary,
                "fusion_aux_weights": fusion_aux_weights,
                "calibration": calibration,
                "xai": xai_artifacts,
            }

            results["evaluation"] = self.evaluation_adapter.evaluate_training(
                results,
                problem_type=problem_type,
            )

            self._sync_training_results_to_context(
                results=results,
                active_modalities=list(schema_info.get("global_modalities", self.config.modalities)),
            )

            ctx_for_fit_feedback = self._get_ctx()
            if ctx_for_fit_feedback is not None:
                try:
                    fit_payload = dict(trial_feedback_summary)
                    fit_payload["trial_diagnostics"] = list(trial_diagnostics)
                    fit_payload["trial_feedback_events"] = list(trial_feedback_events)
                    fit_payload["next_run_feedback"] = dict(next_run_feedback or {})
                    fit_payload["adaptive_feedback_state"] = dict(adaptive_feedback_state)

                    if hasattr(ctx_for_fit_feedback, "apply_training_feedback"):
                        ctx_for_fit_feedback.apply_training_feedback(
                            fit_payload,
                            predictability_factors=dict(
                                (next_run_feedback or {}).get("predictability_factors", {}) or {}
                            ),
                        )
                    else:
                        ctx_for_fit_feedback.update_fit_analysis(fit_payload)
                except Exception as fit_ctx_exc:
                    logger.debug("  Context fit-analysis update skipped: %s", fit_ctx_exc)

            logger.info("\nPhase 5 Summary:")
            logger.info("  Trials completed  : %d (%d pruned)", _n_complete, _n_pruned)
            logger.info("  Best trial        : #%d", best.number)
            logger.info("  Best val_loss     : %.4f", best_val_loss)
            logger.info("  Best params       : %s", best.params)
            logger.info("  Duration          : %.2fs", elapsed)

            self.phase_results[Phase.TRAINING] = results
            self.state.set_slot("phase5_training", results)
            self.state.set_slot("training_evaluation", results.get("evaluation", {}))
            self.state.set_slot("training_calibration", results.get("calibration", {}))
            self.state.set_slot("xai_artifacts", xai_artifacts)
            self.state.set_phase_timing("TRAINING", elapsed)
            self._record_phase_timing_in_context("TRAINING", elapsed)
            self.current_phase = Phase.DRIFT_DETECTION

        except Exception as exc:
            logger.error("Phase 5 failed: %s", str(exc))
            raise
    
    def _execute_phase_6_drift_detection(self) -> None:
        """
        Phase 6: Drift Detection – compute KS, PSI, and FDD (MMD) statistics.

        Steps
        -----
        1. For each registered dataset create a chronological 70/30
           temporal split via ``DatasetManager.create_temporal_split``.
        2. Materialise both halves to numeric numpy arrays (≤ 25 000 rows
           per split to keep O(n²) MMD tractable).
        3. Concatenate reference and production arrays across all datasets.
        4. Run ``DriftDetector.detect()`` → ``DriftReport`` (KS, PSI, MMD).
        5. Store the report in ``phase_results[Phase.DRIFT_DETECTION]``
           in the same dict shape used by the /monitor-drift API contract.

        Falls back gracefully when no datasets are available or the registry
        contains no materialisable lazy refs.
        """
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 6: DRIFT DETECTION")
        logger.info("=" * 80)

        self._enforce_session_context("Phase 6")

        phase_start = time.time()
        MAX_ROWS_PER_SPLIT = 25_000
        self._phase6_reference_sample = None

        try:
            from monitoring.drift_detector import DriftDetector

            # ----------------------------------------------------------------
            # 1  Build temporal splits for every registered dataset
            # ----------------------------------------------------------------
            ref_frames: list = []
            prod_frames: list = []

            for name in self.dataset_registry.list_datasets():
                splits = self.dataset_registry.create_temporal_split(name)
                if splits is None:
                    logger.warning("  Temporal split returned None for dataset '%s' – skipping", name)
                    continue

                for split_key, split_ref in splits.items():
                    frames = ref_frames if split_key == "reference" else prod_frames
                    try:
                        import polars as pl
                        if isinstance(split_ref, pl.LazyFrame):
                            frames.append(split_ref.head(MAX_ROWS_PER_SPLIT).collect().to_pandas())
                            continue
                    except ImportError:
                        pass
                    try:
                        import dask.dataframe as dd
                        if isinstance(split_ref, dd.DataFrame):
                            frames.append(split_ref.head(MAX_ROWS_PER_SPLIT, compute=True))
                            continue
                    except ImportError:
                        pass
                    if isinstance(split_ref, pd.DataFrame):
                        frames.append(split_ref.head(MAX_ROWS_PER_SPLIT))

            # ----------------------------------------------------------------
            # 2  Materialise to pandas + float64 numpy arrays (numeric cols only)
            # ----------------------------------------------------------------
            def _concat_frames(frames_list: list) -> pd.DataFrame:
                if not frames_list:
                    return pd.DataFrame()
                return pd.concat(frames_list, ignore_index=True, sort=False)

            ref_df = _concat_frames(ref_frames)
            prod_df = _concat_frames(prod_frames)

            def _to_numeric_array(df: pd.DataFrame) -> np.ndarray:
                if df.empty:
                    return np.zeros((0, 1), dtype=np.float64)
                numeric_df = df.select_dtypes(include=[np.number])
                if numeric_df.empty:
                    return np.zeros((len(df), 1), dtype=np.float64)
                return numeric_df.fillna(0.0).values.astype(np.float64)

            ref_array = _to_numeric_array(ref_df)
            prod_array = _to_numeric_array(prod_df)

            schema_info = self.phase_results.get(Phase.SCHEMA_DETECTION, {})
            modality_columns: Dict[str, List[str]] = {}
            if isinstance(schema_info, dict):
                per_dataset = schema_info.get("per_dataset", [])
                if isinstance(per_dataset, list):
                    for dataset_schema in per_dataset:
                        if not isinstance(dataset_schema, dict):
                            continue
                        detected = dataset_schema.get("detected_columns", {})
                        if not isinstance(detected, dict):
                            continue
                        for modality, cols in detected.items():
                            if not isinstance(cols, list):
                                continue
                            bucket = modality_columns.setdefault(str(modality), [])
                            for col in cols:
                                if isinstance(col, str) and col not in bucket:
                                    bucket.append(col)

            # ----------------------------------------------------------------
            # 3  Derive feature names from tabular preprocessor (best-effort)
            # ----------------------------------------------------------------
            feature_names = None
            tabular_prep = self.fitted_transformers.get("tabular")
            if tabular_prep is not None and hasattr(tabular_prep, "get_feature_names_out"):
                try:
                    feature_names = list(tabular_prep.get_feature_names_out())
                except Exception:
                    pass

            # ----------------------------------------------------------------
            # 4  Run DriftDetector
            # ----------------------------------------------------------------
            modality_drift: Dict[str, Any] = {}

            if ref_array.shape[0] == 0 or prod_array.shape[0] == 0:
                logger.warning(
                    "  Phase 6: insufficient data for drift detection "
                    "(ref=%d rows, prod=%d rows) – reporting zero drift.",
                    ref_array.shape[0], prod_array.shape[0],
                )
                from monitoring.drift_detector import DriftReport
                report = DriftReport(
                    psi=0.0, ks_statistic=0.0, fdd=0.0,
                    drift_detected=False,
                    status={"psi": False, "ks_statistic": False, "fdd": False},
                    per_feature_ks={}, per_feature_psi={},
                    n_features=0, n_reference=0, n_production=0,
                )
            else:
                from pipeline.retraining_orchestrator import RetrainingOrchestrator

                retrain_orchestrator = RetrainingOrchestrator(
                    production_sources=list(self.config.dataset_sources),
                    problem_type=self.config.problem_type,
                    modalities=list(self.config.modalities),
                    schema_info=self.phase_results.get(Phase.SCHEMA_DETECTION),
                    cooldown_seconds=3600,
                    session_id=getattr(self._get_ctx(), "session_id", None),
                    execution_context=self._get_ctx(),
                )
                detector = DriftDetector(
                    retraining_orchestrator=retrain_orchestrator,
                    cooldown_seconds=3600,
                )
                report = detector.detect(
                    ref_array,
                    prod_array,
                    feature_names,
                    dataset_id="phase6_default",
                )
                self._phase6_reference_sample = getattr(report, "reference_sample", None)
                if modality_columns:
                    modality_drift = detector.detect_modality_drift(
                        reference_df=ref_df,
                        production_df=prod_df,
                        modality_columns=modality_columns,
                    )

            logger.info("Drift Detection Results:")
            logger.info("  PSI (Population Stability Index)")
            logger.info("    -> Value    : %.4f", report.psi)
            logger.info("    -> Threshold: 0.2500")
            logger.info("    -> Status   : %s", "DRIFT" if report.status["psi"] else "OK")
            logger.info("  KS Statistic (Kolmogorov-Smirnov)")
            logger.info("    -> Value    : %.4f", report.ks_statistic)
            logger.info("    -> Threshold: 0.3000")
            logger.info("    -> Status   : %s", "DRIFT" if report.status["ks_statistic"] else "OK")
            logger.info("  FDD / MMD (Feature Distribution Drift)")
            logger.info("    -> Value    : %.4f", report.fdd)
            logger.info("    -> Threshold: 0.5000")
            logger.info("    -> Status   : %s", "DRIFT" if report.status["fdd"] else "OK")
            if modality_drift:
                logger.info("  Modality-Level Drift")
                for modality, details in modality_drift.items():
                    metrics = details.get("metrics", {}) if isinstance(details, dict) else {}
                    if metrics:
                        logger.info(
                            "    -> %s: drift=%s psi=%.4f ks=%.4f fdd=%.4f",
                            modality,
                            "YES" if details.get("drift_detected") else "NO",
                            float(metrics.get("psi", 0.0)),
                            float(metrics.get("ks_statistic", 0.0)),
                            float(metrics.get("fdd", 0.0)),
                        )
                    else:
                        logger.info(
                            "    -> %s: drift=%s (%s)",
                            modality,
                            "YES" if details.get("drift_detected") else "NO",
                            details.get("reason", "no_metrics"),
                        )

            # ----------------------------------------------------------------
            # 5  Store results
            # ----------------------------------------------------------------
            elapsed = time.time() - phase_start
            results: Dict[str, Any] = {
                "drift_detected": report.drift_detected,
                "metrics": {
                    "psi":           report.psi,
                    "ks_statistic":  report.ks_statistic,
                    "fdd":           report.fdd,
                },
                "thresholds": {
                    "psi":           0.25,
                    "ks_statistic":  0.30,
                    "fdd":           0.50,
                },
                "status": report.status,
                "per_feature_ks":  report.per_feature_ks,
                "per_feature_psi": report.per_feature_psi,
                "modality_drift": modality_drift,
                "n_reference":     report.n_reference,
                "n_production":    report.n_production,
                "n_features":      report.n_features,
                "composite_score": report.composite_score,
                "reference_sample": getattr(report, "reference_sample", None),
                "duration_seconds": elapsed,
                "retrain_triggered": report.retrain_triggered,
                "retrain_info": report.retrain_info,
            }
            results["monitor"] = self.drift_adapter.build_monitor_payload(results)

            self._sync_drift_results_to_context(results, modality_drift)

            logger.info("\nPhase 6 Summary:")
            logger.info("  Drift Detected : %s", "YES" if report.drift_detected else "NO")
            logger.info("  Reference rows : %d", report.n_reference)
            logger.info("  Production rows: %d", report.n_production)
            logger.info("  Duration       : %.2fs", elapsed)

            self.phase_results[Phase.DRIFT_DETECTION] = results
            self.state.set_slot("phase6_drift", results)
            self.state.set_slot("drift_monitor", results.get("monitor", {}))
            self.state.set_phase_timing("DRIFT_DETECTION", elapsed)
            self._record_phase_timing_in_context("DRIFT_DETECTION", elapsed)
            self.current_phase = Phase.MODEL_REGISTRY

        except Exception as exc:
            logger.error("Phase 6 failed: %s", str(exc))
            raise
    
    def _execute_phase_7_model_registry(self) -> None:
        """
        Phase 7: Model Registry – physically serialise training artifacts.

        Artifact tree
        -------------
        models/registry/{model_id}/
        ├── artifacts/
        │   ├── model_weights.pth       ← torch.save(state_dict)
        │   ├── tabular_scaler.joblib   ← joblib.dump(TabularPreprocessor)
        │   ├── text_tokenizer/         ← tokenizer.save_pretrained(...)
        │   └── schema.json             ← GlobalSchema (Phase 2 output)
        └── metadata.json               ← full provenance + artifact_paths

        All writes are best-effort: a failed artifact save is logged as a
        warning but does NOT abort Phase 7 (``deployment_ready`` is set to
        ``False`` when the primary weight file could not be written).
        """
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 7: MODEL REGISTRY")
        logger.info("=" * 80)

        phase_start = time.time()

        try:
            import joblib

            # ----------------------------------------------------------------
            # 1  Create directory tree
            # ----------------------------------------------------------------
            model_id = f"apex_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            registry_root = MODEL_REGISTRY_DIR / model_id
            artifacts_dir = registry_root / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            logger.info("  Registry root : %s", registry_root)

            artifact_paths: Dict[str, str] = {}
            deployment_ready = True
            phase5_training = self.phase_results.get(Phase.TRAINING, {})
            phase5_training = phase5_training if isinstance(phase5_training, dict) else {}
            phase5_fusion_payload = _phase7_fusion_payload(phase5_training)
            best_lora_config = dict(phase5_training.get("best_lora_config", {}) or {})

            # ----------------------------------------------------------------
            # 2  Model weights  (requires best_lightning_module from Phase 5)
            # ----------------------------------------------------------------
            weights_path = artifacts_dir / "model_weights.pth"
            if self.best_lightning_module is not None:
                try:
                    torch.save(
                        self.best_lightning_module.model.state_dict(),
                        weights_path,
                    )
                    artifact_paths["model_weights"] = str(weights_path)
                    logger.info("  Weights saved : %s", weights_path)
                except Exception as exc:
                    logger.warning("  Weight save FAILED: %s", exc)
                    deployment_ready = False

                # Bug 1 fix: persist input_dims so inference can reconstruct head correctly
                # without heuristic dim reconstruction that breaks for non-ResNet encoders.
                _idims_to_save = phase5_training.get("input_dims", {}) if isinstance(phase5_training, dict) else {}
                if _idims_to_save:
                    _idims_path = artifacts_dir / "input_dims.json"
                    try:
                        with open(_idims_path, "w", encoding="utf-8") as _fh:
                            json.dump(_idims_to_save, _fh, indent=2)
                        artifact_paths["input_dims"] = str(_idims_path)
                        logger.info("  input_dims saved: %s", _idims_path)
                    except Exception as _exc:
                        logger.warning("  input_dims save FAILED: %s", _exc)

                # Save drift reference sample for stateless /monitor/drift endpoint
                try:
                    ref_sample = self._phase6_reference_sample
                    if ref_sample is None:
                        drift_res = self.phase_results.get(Phase.DRIFT_DETECTION, {})
                        if isinstance(drift_res, dict):
                            ref_sample = drift_res.get("reference_sample")

                    if ref_sample is not None:
                        if hasattr(ref_sample, "values"):
                            ref_values = np.asarray(ref_sample.values)
                        else:
                            ref_values = np.asarray(ref_sample)

                        if ref_values.size > 0:
                            ref_path = artifacts_dir / "reference_sample.npy"
                            np.save(str(ref_path), ref_values.astype(np.float32))
                            artifact_paths["reference_sample"] = str(ref_path)
                            logger.info("  Drift reference sample saved: %s", ref_path)
                except Exception as ref_exc:
                    logger.warning("Could not save drift reference sample: %s", ref_exc)

                # Save frozen encoder state dicts for inference
                _img_enc = getattr(self.best_lightning_module, "_image_encoder", None)
                _txt_enc = getattr(self.best_lightning_module, "_text_encoder", None)

                if _img_enc is not None:
                    img_enc_path = artifacts_dir / "image_encoder_state.pth"
                    try:
                        torch.save(_img_enc.state_dict(), img_enc_path)
                        artifact_paths["image_encoder_state"] = str(img_enc_path)
                        logger.info("  ImageEncoder saved: %s", img_enc_path)
                    except Exception as exc:
                        logger.warning("  ImageEncoder save FAILED: %s", exc)

                if _txt_enc is not None:
                    txt_enc_path = artifacts_dir / "text_encoder_state.pth"
                    try:
                        torch.save(_txt_enc.state_dict(), txt_enc_path)
                        artifact_paths["text_encoder_state"] = str(txt_enc_path)
                        logger.info("  TextEncoder saved: %s", txt_enc_path)
                    except Exception as exc:
                        logger.warning("  TextEncoder save FAILED: %s", exc)

                # Save trained tabular encoder state dict
                _tab_enc = getattr(
                    self.best_lightning_module, "tabular_encoder", None
                )
                if _tab_enc is not None:
                    tab_enc_path = artifacts_dir / "tabular_encoder_state.pth"
                    try:
                        torch.save(_tab_enc.state_dict(), tab_enc_path)
                        artifact_paths["tabular_encoder_state"] = str(tab_enc_path)
                        logger.info("  TabularEncoder saved: %s", tab_enc_path)
                    except Exception as exc:
                        logger.warning("  TabularEncoder save FAILED: %s", exc)

                # Part A.2 — Save LoRA adapter weights when LoRA was used
                # lora_text.pth / lora_image.pth contain only the low-rank A/B deltas
                # so inference can reconstruct the adapted encoder without full fine-tune storage.
                try:
                    from modelss.adapters.lora import lora_state_dict as _lora_sd
                    if _txt_enc is not None:
                        _text_lora = _lora_sd(_txt_enc)
                        if _text_lora:
                            _lora_txt_path = artifacts_dir / "lora_text.pth"
                            torch.save(_text_lora, _lora_txt_path)
                            artifact_paths["lora_text"] = str(_lora_txt_path)
                            logger.info("  LoRA text adapter saved (%d tensors): %s", len(_text_lora), _lora_txt_path)
                    if _img_enc is not None:
                        _image_lora = _lora_sd(_img_enc)
                        if _image_lora:
                            _lora_img_path = artifacts_dir / "lora_image.pth"
                            torch.save(_image_lora, _lora_img_path)
                            artifact_paths["lora_image"] = str(_lora_img_path)
                            logger.info("  LoRA image adapter saved (%d tensors): %s", len(_image_lora), _lora_img_path)
                except Exception as _lora_save_exc:
                    logger.warning("LoRA artifact save FAILED: %s", _lora_save_exc)

                # Part A.2 — Save ULA config when UnifiedLatentFusion was used
                try:
                    from modelss.fusion import UnifiedLatentFusion as _ULAClass
                    _head_fusion = getattr(self.best_lightning_module.model, "fusion", None)
                    if isinstance(_head_fusion, _ULAClass):
                        _ula_cfg = {
                            "latent_dim": _head_fusion.latent_dim,
                            "n_layers": len(_head_fusion.transformer.layers),
                            "n_heads": _head_fusion.transformer.layers[0].self_attn.num_heads
                            if hasattr(_head_fusion.transformer.layers[0], "self_attn") else None,
                            "token_mode": _head_fusion.token_mode,
                            "attention_rollout_compatible": True,
                        }
                        _ula_cfg_path = artifacts_dir / "ula_config.json"
                        with open(_ula_cfg_path, "w", encoding="utf-8") as _fh:
                            json.dump(_ula_cfg, _fh, indent=2, default=str)
                        artifact_paths["ula_config"] = str(_ula_cfg_path)
                        logger.info("  ULA config saved: %s", _ula_cfg_path)
                except Exception as _ula_save_exc:
                    logger.warning("ULA config save FAILED: %s", _ula_save_exc)

                # Save encoder config so inference knows model names / settings
                encoder_config: Dict[str, Any] = {}
                encoder_selection = dict(phase5_training.get("encoder_selection", {}) or {})
                if _txt_enc is not None:
                    encoder_config["text_encoder"] = {
                        "model_name": getattr(_txt_enc, "model_name", "bert-base-uncased"),
                        "max_length": getattr(_txt_enc, "max_length", 128),
                        "freeze_backbone": True,
                    }
                    if best_lora_config:
                        encoder_config["text_encoder"].update({
                            "lora_enabled": True,
                            "lora_r": int(best_lora_config.get("r", 8)),
                            "lora_alpha": float(best_lora_config.get("alpha", 16.0)),
                            "lora_lr_mult": float(best_lora_config.get("lr_mult", 0.1)),
                        })
                if _img_enc is not None:
                    encoder_config["image_encoder"] = {
                        "model_name": (
                            getattr(_img_enc, "model_name", None)
                            or encoder_selection.get("image_encoder")
                            or type(_img_enc).__name__
                        ),
                        "type": type(_img_enc).__name__,
                        "pretrained": True,
                        "freeze_backbone": True,
                    }
                    if best_lora_config:
                        encoder_config["image_encoder"].update({
                            "lora_enabled": True,
                            "lora_r": int(best_lora_config.get("r", 8)),
                            "lora_alpha": float(best_lora_config.get("alpha", 16.0)),
                            "lora_lr_mult": float(best_lora_config.get("lr_mult", 0.1)),
                        })
                if _tab_enc is not None:
                    encoder_config["tabular_encoder"] = {
                        "type": type(_tab_enc).__name__,
                        "input_dim": getattr(_tab_enc, "input_dim", None),
                        "output_dim": _tab_enc.get_output_dim(),
                    }
                if encoder_config:
                    enc_config_path = artifacts_dir / "encoder_config.json"
                    try:
                        with open(enc_config_path, "w", encoding="utf-8") as fh:
                            json.dump(encoder_config, fh, indent=2)
                        artifact_paths["encoder_config"] = str(enc_config_path)
                        logger.info("  Encoder config saved: %s", enc_config_path)
                    except Exception as exc:
                        logger.warning("  Encoder config save FAILED: %s", exc)
            else:
                logger.warning(
                    "  Phase 7: best_lightning_module is None – "
                    "Phase 5 may not have been executed; skipping weight save."
                )
                deployment_ready = False

            # ----------------------------------------------------------------
            # 3  Tabular scaler / preprocessor (sklearn pipeline)
            # ----------------------------------------------------------------
            tabular_prep = self.fitted_transformers.get("tabular")
            if tabular_prep is not None:
                scaler_path = artifacts_dir / "tabular_scaler.joblib"
                try:
                    joblib.dump(tabular_prep, scaler_path)
                    artifact_paths["tabular_scaler"] = str(scaler_path)
                    logger.info("  Scaler saved  : %s", scaler_path)
                except Exception as exc:
                    logger.warning("  Scaler save FAILED: %s", exc)
                    deployment_ready = False

            # ----------------------------------------------------------------
            # 3b Target encoder (LabelEncoder or StandardScaler)
            # ----------------------------------------------------------------
            target_enc = self.fitted_transformers.get("target_encoder")
            if target_enc is not None:
                target_enc_path = artifacts_dir / "target_encoder.joblib"
                try:
                    joblib.dump(target_enc, target_enc_path)
                    artifact_paths["target_encoder"] = str(target_enc_path)
                    logger.info("  Target encoder saved: %s", target_enc_path)
                except Exception as exc:
                    logger.warning("  Target encoder save FAILED: %s", exc)
                    deployment_ready = False

            # ----------------------------------------------------------------
            # 3c Probability calibrator (classification only)
            # ----------------------------------------------------------------
            if self.probability_calibrator is not None and self.probability_calibrator.fitted:
                calibrator_path = artifacts_dir / "probability_calibrator.joblib"
                try:
                    joblib.dump(self.probability_calibrator, calibrator_path)
                    artifact_paths["probability_calibrator"] = str(calibrator_path)
                    logger.info("  Probability calibrator saved: %s", calibrator_path)
                except Exception as exc:
                    logger.warning("  Probability calibrator save FAILED: %s", exc)

            # ----------------------------------------------------------------
            # 4  Text tokenizer (HuggingFace save_pretrained)
            # ----------------------------------------------------------------
            text_prep = self.fitted_transformers.get("text")
            if text_prep is not None:
                tokenizer_dir = artifacts_dir / "text_tokenizer"
                tokenizer_dir.mkdir(exist_ok=True)
                try:
                    tokenizer = getattr(text_prep, "tokenizer", None)
                    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
                        tokenizer.save_pretrained(str(tokenizer_dir))
                        artifact_paths["text_tokenizer"] = str(tokenizer_dir)
                        logger.info("  Tokenizer saved: %s", tokenizer_dir)
                    else:
                        logger.info("  Text preprocessor has no save_pretrained – skipping")
                except Exception as exc:
                    logger.warning("  Tokenizer save FAILED: %s", exc)
                    deployment_ready = False

            # ----------------------------------------------------------------
            # 5  GlobalSchema JSON  (Phase 2 output)
            # ----------------------------------------------------------------
            schema_path = artifacts_dir / "schema.json"
            schema_data = self.phase_results.get(Phase.SCHEMA_DETECTION, {})
            try:
                with open(schema_path, "w", encoding="utf-8") as fh:
                    json.dump(schema_data, fh, indent=2, default=str)
                artifact_paths["schema"] = str(schema_path)
                logger.info("  Schema saved  : %s", schema_path)
            except Exception as exc:
                logger.warning("  Schema save FAILED: %s", exc)

            # ----------------------------------------------------------------
            # 6  Metadata JSON  (provenance + artifact paths)
            # ----------------------------------------------------------------
            created_at = datetime.now().isoformat()
            research_metrics = self.research_metrics.compute(self.phase_results)
            state_snapshot = self.state.snapshot()
            ctx = self._get_ctx()
            artifact_versions = dict(getattr(ctx, "artifact_versions", {}) or {}) if ctx is not None else {}
            training_signals = dict(getattr(ctx, "training_signals", {}) or {}) if ctx is not None else {}
            training_fit_analysis = dict(getattr(ctx, "training_fit_analysis", {}) or {}) if ctx is not None else {}
            xai_config = dict(getattr(ctx, "xai_config", {}) or {}) if ctx is not None else {}
            head_architecture: Dict[str, int] = {}
            if self.best_lightning_module is not None:
                runtime_head = getattr(self.best_lightning_module, "model", None)
                if runtime_head is not None and hasattr(runtime_head, "state_dict"):
                    try:
                        head_state = runtime_head.state_dict()
                        w0 = head_state.get("layers.0.weight")
                        b_last = None
                        for last_key in ("layers.3.bias", "layers.4.bias", "layers.5.bias"):
                            if last_key in head_state:
                                b_last = head_state[last_key]
                                break
                        if w0 is None or b_last is None:
                            weight_keys = sorted(
                                [
                                    key
                                    for key in head_state
                                    if key.endswith(".weight") and head_state[key].ndim == 2
                                ]
                            )
                            bias_keys = sorted(
                                [key for key in head_state if key.endswith(".bias")]
                            )
                            if weight_keys and w0 is None:
                                w0 = head_state[weight_keys[0]]
                            if bias_keys and b_last is None:
                                b_last = head_state[bias_keys[-1]]

                        if w0 is not None and b_last is not None:
                            head_architecture = {
                                "hidden_dim": int(w0.shape[0]),
                                "total_dim": int(w0.shape[1]),
                                "num_outputs": int(b_last.shape[0]),
                            }
                    except Exception as exc:
                        logger.warning("  Head architecture extraction FAILED: %s", exc)
            results: Dict[str, Any] = {
                "model_id":        model_id,
                "created_at":      created_at,
                "config":          asdict(self.config),
                "phases_summary":  self._summarize_all_phases(),
                "artifact_paths":  artifact_paths,
                "artifact_versions": artifact_versions,
                "head_architecture": head_architecture,
                "training_signals": training_signals,
                "training_fit_analysis": training_fit_analysis,
                "xai_config": xai_config,
                "status":          "active",
                "deployment_ready": deployment_ready,
                "research_metrics": research_metrics,
                "fusion_strategy": phase5_fusion_payload["strategy"],
                "best_lora_config": best_lora_config,
                "fusion": phase5_fusion_payload,
                "xai": self.state.get_slot("xai_artifacts", {}),
                "state_slots": state_snapshot.get("slots", {}),
            }

            metadata_path = registry_root / "metadata.json"
            try:
                with open(metadata_path, "w", encoding="utf-8") as fh:
                    json.dump(results, fh, indent=2, default=str)
                logger.info("  Metadata saved: %s", metadata_path)
            except Exception as exc:
                logger.warning("  Metadata save FAILED: %s", exc)

            logger.info("Model Registration:")
            logger.info("  Model ID         : %s", model_id)
            logger.info("  Created          : %s", created_at)
            logger.info("  Status           : active")
            logger.info("  Deployment Ready : %s", deployment_ready)
            logger.info("  Artifacts saved  : %d", len(artifact_paths))

            elapsed = time.time() - phase_start
            results["duration_seconds"] = elapsed

            logger.info("\nPhase 7 Summary:")
            logger.info("  Artifacts : %d", len(artifact_paths))
            logger.info("  Duration  : %.2fs", elapsed)

            self.phase_results[Phase.MODEL_REGISTRY] = results
            self.state.set_slot("phase7_registry", results)
            self.state.set_slot("research_metrics", research_metrics)
            self.state.set_phase_timing("MODEL_REGISTRY", elapsed)
            self._record_phase_timing_in_context("MODEL_REGISTRY", elapsed)
            self._sync_model_registry_to_context(model_id, deployment_ready)

        except Exception as exc:
            logger.error("Phase 7 failed: %s", str(exc))
            raise
    
    def _summarize_all_phases(self) -> Dict[str, Any]:
        """Create summary of all phases."""
        ctx = self._get_ctx()
        artifact_versions = dict(getattr(ctx, "artifact_versions", {}) or {}) if ctx is not None else {}
        training_signals = dict(getattr(ctx, "training_signals", {}) or {}) if ctx is not None else {}
        training_fit_analysis = dict(getattr(ctx, "training_fit_analysis", {}) or {}) if ctx is not None else {}
        xai_config = dict(getattr(ctx, "xai_config", {}) or {}) if ctx is not None else {}
        summary = {}
        for phase in Phase:
            if phase in self.phase_results:
                result = self.phase_results[phase]
                phase_summary: Dict[str, Any] = {
                    "duration_seconds": result.get("duration_seconds", 0),
                    "status": "completed"
                }
                if phase == Phase.PREPROCESSING:
                    phase_summary.update({
                        "total_samples": result.get("total_samples"),
                        "text_columns": len(result.get("text_columns", [])),
                        "image_columns": len(result.get("image_columns", [])),
                        "tabular_columns": len(result.get("tabular_columns", [])),
                    })
                elif phase == Phase.MODEL_SELECTION:
                    phase_summary.update({
                        "image_encoder": result.get("image_encoder_name") or result.get("image_encoder"),
                        "text_encoder": result.get("text_encoder_name") or result.get("text_encoder"),
                        "tabular_encoder": result.get("tabular_encoder_name") or result.get("tabular_encoder"),
                        "fusion_strategy": result.get("fusion_strategy"),
                        "batch_size": result.get("batch_size"),
                    })
                elif phase == Phase.TRAINING:
                    phase_summary.update({
                        "best_trial": result.get("best_trial"),
                        "best_val_loss": result.get("best_val_loss"),
                        "best_val_acc": result.get("best_val_acc"),
                        "best_val_f1": result.get("best_val_f1"),
                        "best_train_acc": result.get("best_train_acc"),
                        "n_trials": result.get("n_trials"),
                        "fusion_strategy": result.get("fusion_strategy"),
                        "best_lora_config": result.get("best_lora_config", {}),
                        "calibration": result.get("calibration", {}),
                        "evaluation": result.get("evaluation", {}),
                        "alignment_summary": result.get("alignment_summary", {}),
                        "fusion_summary": result.get("fusion_summary", {}),
                        "fusion_aux_weights": result.get("fusion_aux_weights", {}),
                        "training_signals": training_signals,
                        "training_fit_analysis": training_fit_analysis,
                        "artifact_versions": artifact_versions,
                        "xai_config": xai_config,
                    })
                elif phase == Phase.DRIFT_DETECTION:
                    monitor = result.get("monitor", {}) if isinstance(result, dict) else {}
                    phase_summary.update({
                        "drift_detected": result.get("drift_detected"),
                        "metrics": result.get("metrics", {}),
                        "composite_score": result.get("composite_score"),
                        "severity": monitor.get("severity"),
                        "retrain_triggered": result.get("retrain_triggered", False),
                    })
                summary[phase.name] = phase_summary
        return summary
    
    def _compile_results(self, total_elapsed: float) -> Dict[str, Any]:
        """Compile final pipeline results."""
        # Convert Phase enum keys to strings for JSON serialization
        serializable_phases = {
            phase.name: result
            for phase, result in self.phase_results.items()
        }
        # Safely extract model_id — may not exist if Phase 7 was skipped
        model_id = "unknown"
        if Phase.MODEL_REGISTRY in self.phase_results:
            model_id = self.phase_results[Phase.MODEL_REGISTRY].get("model_id", "unknown")
        return {
            "status": "success",
            "model_id": model_id,
            "total_duration_seconds": total_elapsed,
            "phases": serializable_phases,
            "metadata": {
                "config": asdict(self.config),
                "timestamp": datetime.now().isoformat(),
                "pytorch_version": torch.__version__,
                "device": str(self.device),
                "state": self.state.snapshot(),
                "research_metrics": self.state.get_slot("research_metrics", {}),
            }
        }


# Example usage
if __name__ == "__main__":
    # Create configuration
    config = TrainingConfig(
        dataset_sources=[
            "https://kaggle.com/datasets/example1",
            "https://kaggle.com/datasets/example2"
        ],
        problem_type="classification_multiclass",
        modalities=["image", "text", "tabular"],
        target_column="label"
    )
    
    # Create orchestrator and run pipeline
    orchestrator = TrainingOrchestrator(config)
    results = asyncio.run(orchestrator.run_pipeline())
    
    # Save results
    output_path = Path("pipeline_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✅ Results saved to {output_path}")


# Alias for backward compatibility
PipelineOrchestrator = TrainingOrchestrator
