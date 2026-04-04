"""
Comprehensive Training Orchestrator - Coordinates all 7 phases of ML pipeline.

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
from preprocessing.tabular_preprocessor import TabularPreprocessor
from core.types import Phase, TrainingConfig, ModelSelectionResult, TrainingMetrics


# Configure logging
logger = logging.getLogger(__name__)


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
        detected = per_ds[0].get("detected_columns", {}) if per_ds else {}
        self._text_cols = [c for c in detected.get("text", []) if c in df.columns]
        self._image_cols = [c for c in detected.get("image", []) if c in df.columns]
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
            text_val = str(row[self._text_cols[0]])
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
            try:
                from PIL import Image as PILImage
                img_path = str(row[self._image_cols[0]])
                pil_img = PILImage.open(img_path).convert("RGB")
                if self.apply_augmentation and hasattr(self.image_preprocessor, "augment"):
                    pil_img = self.image_preprocessor.augment(pil_img)
                sample["image"] = self.image_preprocessor(pil_img)
            except Exception as exc:
                logger.warning("Image load failed for idx=%d path=%s: %s", idx, img_path, exc)
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
) -> torch.Tensor:
    """Run frozen text encoder over all samples once.

    Returns a ``[N, output_dim]`` float32 CPU tensor aligned to the
    dataset's row indices so that ``Subset`` indexing works correctly.
    """
    from torch.utils.data import DataLoader

    text_encoder.eval()
    all_embeds: List[torch.Tensor] = []
    n = len(dataset)

    # Iterate through dataset manually to collect tokenized text
    for start in range(0, n, batch_size):
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

    return torch.cat(all_embeds, dim=0)


def _precompute_image_embeddings(
    dataset: MultimodalPyTorchDataset,
    image_encoder,
    device: torch.device,
    batch_size: int = 32,
) -> torch.Tensor:
    """Run frozen image encoder over all samples once.

    Returns a ``[N, output_dim]`` float32 CPU tensor aligned to the
    dataset's row indices.  Must only be used on datasets **without**
    augmentation (validation split) to ensure deterministic embeddings.
    """
    image_encoder.eval()
    all_embeds: List[torch.Tensor] = []
    n = len(dataset)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        img_list = []
        for i in range(start, end):
            sample = dataset[i]
            img_list.append(sample["image"])

        images = torch.stack(img_list).to(device)

        with torch.no_grad():
            embeds = image_encoder(images)
            all_embeds.append(embeds.cpu())

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
        detected = per_ds[0].get("detected_columns", {}) if per_ds else {}
        self._text_cols: List[str] = detected.get("text", [])
        self._image_cols: List[str] = detected.get("image", [])

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
                        text_val = str(feature_chunk.iloc[i][text_cols[0]])
                        enc = self._text_preprocessor(text_val)
                        sample["input_ids"] = enc["input_ids"]
                        sample["attention_mask"] = enc["attention_mask"]

                    # Image (lazy PIL load + resize + normalize)
                    if self._image_preprocessor is not None and image_cols:
                        try:
                            from PIL import Image as PILImage
                            img_path = str(feature_chunk.iloc[i][image_cols[0]])
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

    def __init__(self, config: TrainingConfig):
        """Initialize orchestrator."""
        self.config = config
        self.current_phase = Phase.DATA_INGESTION
        self.phase_results = {}
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

        # Setup device
        self.device = torch.device(config.device)
        logger.info(f"Using device: {self.device}")

        if self.device.type == "cuda":
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
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

            for source_hash, lazy_ref in lazy_datasets.items():
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

            full_df: pd.DataFrame = (
                pd.concat(frames, ignore_index=True) if frames
                else pd.DataFrame()
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

            # ----------------------------------------------------------------
            # 3  Target separation and encoding
            # ----------------------------------------------------------------
            from sklearn.preprocessing import LabelEncoder, StandardScaler as SS

            if target_col != "Unknown" and target_col in full_df.columns:
                y_raw = full_df[target_col]
                feature_df = full_df.drop(columns=[target_col])
            else:
                logger.warning(
                    "  Target column '%s' not found – using last column as target", target_col
                )
                feature_df = full_df.iloc[:, :-1]
                y_raw = full_df.iloc[:, -1]
                target_col = full_df.columns[-1]

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

            # ----------------------------------------------------------------
            # 5  Fit modality preprocessors
            # ----------------------------------------------------------------
            text_prep = None
            image_prep = None
            tabular_prep = None
            output_shapes: Dict[str, Any] = {}
            preprocessing_stages = []

            if tabular_cols and "tabular" in schema_info.get("global_modalities", []):
                tabular_prep = TabularPreprocessor()
                tabular_prep.fit(feature_df[tabular_cols])
                self.fitted_transformers["tabular"] = tabular_prep
                output_dim = tabular_prep.get_output_dim()
                output_shapes["tabular"] = f"(N, {output_dim})"
                preprocessing_stages.append({
                    "stage": "tabular_preprocessing",
                    "status": "success",
                    "output_shape": output_shapes["tabular"],
                })
                logger.info("  Tabular preprocessor fitted: output_dim=%d", output_dim)

            if text_cols and "text" in schema_info.get("global_modalities", []):
                text_prep = TextPreprocessor()
                self.fitted_transformers["text"] = text_prep
                output_shapes["text"] = "(N, 128) per key"
                preprocessing_stages.append({
                    "stage": "text_preprocessing",
                    "status": "success",
                    "output_shape": output_shapes["text"],
                })
                logger.info("  Text preprocessor initialised (lazy tokeniser)")

            if image_cols and "image" in schema_info.get("global_modalities", []):
                image_prep = ImagePreprocessor()
                self.fitted_transformers["image"] = image_prep
                output_shapes["image"] = "(N, 3, 224, 224)"
                preprocessing_stages.append({
                    "stage": "image_preprocessing",
                    "status": "success",
                    "output_shape": output_shapes["image"],
                })
                logger.info("  Image preprocessor initialised")

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
                "text_columns": text_cols,
                "image_columns": image_cols,
                "tabular_columns": tabular_cols,
                "duration_seconds": elapsed,
            }

            logger.info("\nPhase 3 Summary:")
            logger.info("  Stages     : %d", len(preprocessing_stages))
            logger.info("  Samples    : %d", total_samples)
            logger.info("  Duration   : %.2fs", elapsed)

            self.phase_results[Phase.PREPROCESSING] = results
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
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 4: MODEL SELECTION")
        logger.info("=" * 80)

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

            modalities: List[str] = schema_info.get(
                "global_modalities", self.config.modalities
            )
            problem_type: str = schema_info.get(
                "global_problem_type", self.config.problem_type
            )
            dataset_size: int = prep_info.get("total_samples", 10_000)

            # Estimate avg_tokens from column count (no materialisation needed)
            text_cols: List[str] = prep_info.get("text_columns", [])
            avg_tokens: int = 128  # conservative default; Phase 5 can override

            logger.info(
                "  modalities=%s  problem=%s  dataset_size=%d",
                modalities, problem_type, dataset_size,
            )

            # ----------------------------------------------------------------
            # 2  Run AdvancedModelSelector
            # ----------------------------------------------------------------
            selector = AdvancedModelSelector()
            result = selector.select_models(
                problem_type=problem_type,
                modalities=modalities,
                dataset_size=dataset_size,
                avg_tokens=avg_tokens,
            )

            # ----------------------------------------------------------------
            # 3  Derive phase-5-compatible scalar defaults from HPO space
            #    (midpoint of epoch range; geometric mean of LR range)
            # ----------------------------------------------------------------
            epoch_space = result.hpo_space.get("epochs", {})
            epoch_default: int = (
                (epoch_space.get("low", 10) + epoch_space.get("high", 10)) // 2
                if epoch_space else 10
            )

            lr_space = result.hpo_space.get("learning_rate", {})
            import math
            lr_default: float = (
                math.sqrt(lr_space.get("low", 1e-4) * lr_space.get("high", 1e-3))
                if lr_space else 1e-3
            )

            # ----------------------------------------------------------------
            # 4  Resolve human-readable encoder names for logging
            # ----------------------------------------------------------------
            def _name(catalogue: Dict, tier: Optional[str]) -> Optional[str]:
                return catalogue[tier]["name"] if tier and tier in catalogue else None

            img_name = _name(IMAGE_ENCODERS,  result.image_encoder)
            txt_name = _name(TEXT_ENCODERS,   result.text_encoder)
            tab_name = _name(TABULAR_ENCODERS, result.tabular_encoder)

            logger.info("  Image encoder   : %s (%s)", img_name, result.image_encoder)
            logger.info("  Text encoder    : %s (%s)", txt_name, result.text_encoder)
            logger.info("  Tabular encoder : %s (%s)", tab_name, result.tabular_encoder)
            logger.info("  Fusion strategy : %s", result.fusion_strategy)
            logger.info("  Batch size      : %d  (PDF heuristic, not tuned)", result.batch_size)
            logger.info(
                "  Epoch range     : [%d, %d]  → default %d",
                epoch_space.get("low", "?"), epoch_space.get("high", "?"), epoch_default,
            )

            # ----------------------------------------------------------------
            # 5  Store results dict
            # ----------------------------------------------------------------
            elapsed = time.time() - phase_start
            phase_result: Dict[str, Any] = {
                "image_encoder":   result.image_encoder,
                "text_encoder":    result.text_encoder,
                "tabular_encoder": result.tabular_encoder,
                "image_encoder_name":   img_name,
                "text_encoder_name":    txt_name,
                "tabular_encoder_name": tab_name,
                "fusion_strategy": result.fusion_strategy,
                "batch_size":      result.batch_size,
                # Scalar defaults used by Phase 5 before HPO narrows them:
                "epochs":          epoch_default,
                "learning_rate":   lr_default,
                # Full Optuna search bounds:
                "hpo_space":       result.hpo_space,
                "rationale":       result.rationale,
                "hardware_info":   result.hardware_info,
                "duration_seconds": elapsed,
            }

            logger.info("\nPhase 4 Summary:")
            logger.info("  Batch size    : %d", result.batch_size)
            logger.info("  Epoch default : %d (HPO will tune in range)", epoch_default)
            logger.info("  LR default    : %.2e (HPO will tune in range)", lr_default)
            logger.info("  HPO params    : %d", len(result.hpo_space))
            logger.info("  Duration      : %.2fs", elapsed)

            self.phase_results[Phase.MODEL_SELECTION] = phase_result
            self.current_phase = Phase.TRAINING

        except Exception as exc:
            logger.error("Phase 4 failed: %s", str(exc))
            raise
    
    def _execute_phase_5_training(self, hp_overrides: Optional[Dict[str, Any]] = None,
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

        N_TRIALS: int = 3       # keep low for interactive use; raise for production
        VAL_SPLIT: float = 0.2

        # When hp_overrides are provided, skip HPO and do a single run
        if hp_overrides:
            N_TRIALS = 1
            logger.info("  HP overrides provided – skipping HPO, single run with: %s", hp_overrides)

        phase_start = time.time()

        try:
            import optuna
            import mlflow
            import pytorch_lightning as pl
            from torch.utils.data import DataLoader
            from automl.trainer import build_trainer

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
            if _use_stratify:
                try:
                    from sklearn.model_selection import train_test_split as _split
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

            # ── Compute class weights for imbalanced classification ──────
            _class_weights: Optional[torch.Tensor] = None
            if (problem_type.startswith("classification")
                    and problem_type != "multilabel_classification"):
                try:
                    from sklearn.utils.class_weight import compute_class_weight as _ccw
                    _train_targets = _aug_ds.targets[train_indices]
                    _train_labels = _train_targets.numpy().astype(int)
                    _unique_classes = np.sort(np.unique(_train_labels))
                    _raw_weights = _ccw("balanced", classes=_unique_classes, y=_train_labels)
                    _class_weights = torch.tensor(_raw_weights, dtype=torch.float32)
                    logger.info(
                        "  Class weights (balanced): %s",
                        {int(c): round(float(w), 3) for c, w in zip(_unique_classes, _raw_weights)},
                    )
                except Exception as cw_exc:
                    logger.warning("  Class weight computation failed: %s", cw_exc)

            model_sel: Dict[str, Any] = self.phase_results.get(Phase.MODEL_SELECTION, {})
            batch_size: int = model_sel.get("batch_size", 32)
            hpo_space: Dict[str, Any] = model_sel.get("hpo_space", {})

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

            _jit_selector = JITEncoderSelector(
                safety_margin=0.85,
                batch_size=batch_size,
            )
            _jit_result = _jit_selector.select(
                modalities=schema_info.get("global_modalities", self.config.modalities),
                device=self.device if self.device.type == "cuda" else None,
            )

            _image_encoder = _jit_result.image_encoder
            _text_encoder = _jit_result.text_encoder

            # Update input_dims from the selected encoder's actual output
            if _text_encoder is not None and hasattr(_text_encoder, "get_output_dim"):
                input_dims["text_pooled"] = _text_encoder.get_output_dim()

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
            if _text_encoder is not None and hasattr(_aug_ds, '_text_cols') and _aug_ds._text_cols:
                logger.info("  Pre-computing text embeddings (%d samples)...", n_total)
                _precomputed_text = _precompute_text_embeddings(
                    _clean_ds, _text_encoder, self.device, batch_size=batch_size,
                )
                _aug_ds._precomputed_text = _precomputed_text
                _clean_ds._precomputed_text = _precomputed_text
                logger.info("  Text embeddings cached: shape=%s", list(_precomputed_text.shape))
            else:
                _precomputed_text = None

            if _image_encoder is not None and hasattr(_clean_ds, '_image_cols') and _clean_ds._image_cols:
                logger.info("  Pre-computing image embeddings for val (%d samples)...", n_total)
                _precomputed_image_val = _precompute_image_embeddings(
                    _clean_ds, _image_encoder, self.device, batch_size=batch_size,
                )
                # Train images: NOT cached (random augmentation must be preserved)
                _aug_ds._precomputed_image = None
                # Val images: cached (deterministic preprocessing only)
                _clean_ds._precomputed_image = _precomputed_image_val
                logger.info("  Image embeddings cached (val only): shape=%s", list(_precomputed_image_val.shape))

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

            # Mutable containers capture the best trained LightningModule
            # inside the closure without needing 'nonlocal' on a scalar.
            _best_val: List[float] = [float("inf")]
            _best_module_ref: List[Any] = []
            _best_metrics: Dict[str, float] = {"val_acc": 0.0, "val_f1": 0.0, "train_acc": 0.0}

            def objective(trial: optuna.Trial) -> float:
                # Report trial number to progress callback
                if progress_callback is not None:
                    progress_callback.set_trial(trial.number + 1, N_TRIALS)

                if hp_overrides:
                    # Use manual overrides directly, no Optuna sampling
                    trial_lr      = hp_overrides.get("learning_rate", model_sel.get("learning_rate", 1e-3))
                    trial_wd      = hp_overrides.get("weight_decay", 1e-5)
                    trial_dropout = hp_overrides.get("dropout", 0.1)
                    trial_epochs  = hp_overrides.get("epochs", model_sel.get("epochs", 10))
                    trial_fusion  = hp_overrides.get("fusion_strategy", model_sel.get("fusion_strategy", "concatenation"))
                else:
                    trial_lr      = _sample(trial, "learning_rate", model_sel.get("learning_rate", 1e-3))
                    trial_wd      = _sample(trial, "weight_decay",  1e-5)
                    trial_dropout = _sample(trial, "dropout",       0.1)
                    trial_epochs  = _sample(trial, "epochs",        model_sel.get("epochs", 10))
                    trial_fusion  = _sample(trial, "fusion_strategy", model_sel.get("fusion_strategy", "concatenation"))

                # Create a FRESH tabular encoder for this trial (trainable,
                # random init).  Image/text encoders are shared (frozen).
                _trial_tabular_encoder = None
                if _tabular_encoder_class is not None and _tabular_input_dim is not None:
                    _trial_tabular_encoder = _tabular_encoder_class(
                        input_dim=_tabular_input_dim,
                    )

                lightning_module = build_trainer(
                    problem_type=problem_type,
                    num_classes=num_classes,
                    input_dims=input_dims,
                    learning_rate=trial_lr,
                    weight_decay=trial_wd,
                    dropout=trial_dropout,
                    max_epochs=trial_epochs,
                    image_encoder=_image_encoder,
                    text_encoder=_text_encoder,
                    tabular_encoder=_trial_tabular_encoder,
                    class_weights=_class_weights,
                    fusion_strategy=trial_fusion,
                )

                # Build a Lightning callback to push epoch metrics in real-time
                _pl_callbacks = []
                if progress_callback is not None:
                    class _EpochReporter(pl.Callback):
                        """Forwards per-epoch metrics to the API progress tracker."""
                        def on_validation_epoch_end(self, trainer, pl_module):
                            m = trainer.callback_metrics
                            progress_callback.log_epoch(
                                trial=trial.number,
                                epoch=trainer.current_epoch + 1,
                                max_epoch=trial_epochs,
                                train_loss=float(m.get("train_loss", 0)),
                                val_loss=float(m.get("val_loss", 0)),
                                train_acc=float(m.get("train_acc", 0)),
                                val_acc=float(m.get("val_acc", 0)),
                                train_f1=float(m.get("train_f1", 0)),
                                val_f1=float(m.get("val_f1", 0)),
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
                    patience=5,
                    verbose=False,
                ))

                # Optuna pruning: kill unpromising trials early (skip for manual HP runs)
                if not hp_overrides:
                    from optuna.integration import PyTorchLightningPruningCallback
                    _pl_callbacks.append(PyTorchLightningPruningCallback(
                        trial, monitor="val_loss",
                    ))

                pl_trainer = pl.Trainer(
                    max_epochs=trial_epochs,
                    accelerator=accelerator,
                    devices=1,
                    # AMP 16-bit mixed precision for GPU tensor core saturation;
                    # falls back to 32-bit on CPU where float16 is unsupported.
                    precision="16-mixed" if accelerator == "gpu" else "32-true",
                    enable_checkpointing=False,
                    enable_progress_bar=False,
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
                            "epochs":        trial_epochs,
                            "problem_type":  problem_type,
                            "num_classes":   num_classes,
                        })

                        try:
                            pl_trainer.fit(lightning_module, train_loader, val_loader)
                            # Retrieve best val_loss logged during training
                            cb_metrics = pl_trainer.callback_metrics
                            best_val_loss = float(cb_metrics.get("val_loss", float("inf")))
                            trial_val_acc = float(cb_metrics.get("val_acc", 0))
                            trial_val_f1  = float(cb_metrics.get("val_f1", 0))
                            trial_train_acc = float(cb_metrics.get("train_acc", 0))
                        except optuna.exceptions.TrialPruned:
                            logger.info("  Trial %d PRUNED by Optuna", trial.number)
                            if progress_callback is not None:
                                progress_callback.add_message(
                                    5, "detail",
                                    f"Trial {trial.number + 1} pruned (underperforming)",
                                )
                            raise  # re-raise so Optuna marks the trial as pruned
                        except Exception as trial_exc:
                            logger.warning("  Trial %d error: %s", trial.number, trial_exc)

                        mlflow.log_metric("val_loss", best_val_loss)

                    # Capture best module – mutable list avoids 'nonlocal'
                    if best_val_loss < _best_val[0]:
                        _best_val[0] = best_val_loss
                        _best_module_ref.clear()
                        _best_module_ref.append(lightning_module)
                        _best_metrics["val_acc"]   = trial_val_acc
                        _best_metrics["val_f1"]    = trial_val_f1
                        _best_metrics["train_acc"] = trial_train_acc

                    logger.info(
                        "  Trial %d: lr=%.2e  wd=%.2e  dropout=%.2f  "
                        "epochs=%d  val_loss=%.4f",
                        trial.number, trial_lr, trial_wd, trial_dropout,
                        trial_epochs, best_val_loss,
                    )

                    return best_val_loss

                finally:
                    # ── GPU cleanup: runs for success, error, AND TrialPruned ──
                    # Previously only in the normal return path, meaning pruned
                    # trials leaked GPU memory ("zombie trial" compute leak).
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
                    min_resource=1, max_resource=model_sel.get("epochs", 10),
                    reduction_factor=3,
                ) if N_TRIALS > 1 else optuna.pruners.NopPruner()
                study = optuna.create_study(
                    direction="minimize",
                    pruner=_pruner,
                )
                study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

            # Persist best model reference so Phase 7 can serialise weights
            if _best_module_ref:
                self.best_lightning_module = _best_module_ref[0]

            best = study.best_trial
            best_val_loss: float = best.value if best.value is not None else float("inf")

            # Count pruned trials for frontend transparency
            _n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
            _n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

            # ----------------------------------------------------------------
            # 6  Build phase results compatible with /train-pipeline contract
            # ----------------------------------------------------------------
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
                "batch_size":        batch_size,
                "data_split":        {"train": n_train, "val": n_val, "total": n_total},
                "problem_type":      problem_type,
                "num_classes":       num_classes,
                "input_dims":        input_dims,
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
            }

            logger.info("\nPhase 5 Summary:")
            logger.info("  Trials completed  : %d (%d pruned)", _n_complete, _n_pruned)
            logger.info("  Best trial        : #%d", best.number)
            logger.info("  Best val_loss     : %.4f", best_val_loss)
            logger.info("  Best params       : %s", best.params)
            logger.info("  Duration          : %.2fs", elapsed)

            self.phase_results[Phase.TRAINING] = results
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

        phase_start = time.time()
        MAX_ROWS_PER_SPLIT = 25_000

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
            # 2  Materialise to float64 numpy arrays (numeric cols only)
            # ----------------------------------------------------------------
            def _to_numeric_array(frames_list: list) -> np.ndarray:
                if not frames_list:
                    return np.zeros((0, 1), dtype=np.float64)
                df = pd.concat(frames_list, ignore_index=True)
                numeric_df = df.select_dtypes(include=[np.number])
                if numeric_df.empty:
                    return np.zeros((len(df), 1), dtype=np.float64)
                return numeric_df.fillna(0.0).values.astype(np.float64)

            ref_array = _to_numeric_array(ref_frames)
            prod_array = _to_numeric_array(prod_frames)

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
                detector = DriftDetector()
                report = detector.detect(ref_array, prod_array, feature_names)

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
                "n_reference":     report.n_reference,
                "n_production":    report.n_production,
                "n_features":      report.n_features,
                "duration_seconds": elapsed,
            }

            logger.info("\nPhase 6 Summary:")
            logger.info("  Drift Detected : %s", "YES" if report.drift_detected else "NO")
            logger.info("  Reference rows : %d", report.n_reference)
            logger.info("  Production rows: %d", report.n_production)
            logger.info("  Duration       : %.2fs", elapsed)

            self.phase_results[Phase.DRIFT_DETECTION] = results
            self.current_phase = Phase.MODEL_REGISTRY

            # ── Autonomous drift → retrain (CI/CD loop) ──────────────────
            # When drift is confirmed, kick off a fresh training run on the
            # same data sources in a background thread so Phase 6 returns
            # immediately.  Phase 6 is intentionally omitted from the retrain
            # run to prevent infinite recursion.
            if report.drift_detected:
                import threading

                def _retrain_background() -> None:
                    try:
                        from pipeline.retrain_executor import RetrainingPipeline
                        retrain_pipeline = RetrainingPipeline(model_id="drift_retrain")
                        retrain_result = retrain_pipeline.retrain(
                            production_sources=list(self.config.dataset_sources),
                            problem_type=self.config.problem_type,
                            modalities=list(self.config.modalities),
                            schema_info=self.phase_results.get(Phase.SCHEMA_DETECTION),
                        )
                        logger.info(
                            "  Autonomous retrain complete: new model_id=%s",
                            retrain_result.get("model_id"),
                        )
                    except Exception as re_exc:
                        logger.warning(
                            "  Autonomous retraining failed (non-fatal): %s", re_exc
                        )

                logger.info(
                    "  Drift confirmed – triggering autonomous retraining "
                    "(background thread, non-blocking)."
                )
                threading.Thread(
                    target=_retrain_background, daemon=True, name="drift-retrain"
                ).start()
                results["retrain_triggered"] = True
            else:
                results["retrain_triggered"] = False

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
            registry_root = Path("models") / "registry" / model_id
            artifacts_dir = registry_root / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            logger.info("  Registry root : %s", registry_root)

            artifact_paths: Dict[str, str] = {}
            deployment_ready = True

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

                # Save encoder config so inference knows model names / settings
                encoder_config: Dict[str, Any] = {}
                if _txt_enc is not None:
                    encoder_config["text_encoder"] = {
                        "model_name": getattr(_txt_enc, "model_name", "bert-base-uncased"),
                        "max_length": getattr(_txt_enc, "max_length", 128),
                        "freeze_backbone": True,
                    }
                if _img_enc is not None:
                    encoder_config["image_encoder"] = {
                        "pretrained": True,
                        "freeze_backbone": True,
                    }
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
            results: Dict[str, Any] = {
                "model_id":        model_id,
                "created_at":      created_at,
                "config":          asdict(self.config),
                "phases_summary":  self._summarize_all_phases(),
                "artifact_paths":  artifact_paths,
                "status":          "active",
                "deployment_ready": deployment_ready,
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

        except Exception as exc:
            logger.error("Phase 7 failed: %s", str(exc))
            raise
    
    def _summarize_all_phases(self) -> Dict[str, Any]:
        """Create summary of all phases."""
        summary = {}
        for phase in Phase:
            if phase in self.phase_results:
                result = self.phase_results[phase]
                summary[phase.name] = {
                    "duration_seconds": result.get("duration_seconds", 0),
                    "status": "completed"
                }
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
                "device": str(self.device)
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
