"""
Lazy-aware dataset registry for the ML pipeline.

Stores Polars LazyFrames, Dask DataFrames, or PyTorch LazyImageDatasets
without materialising any data into RAM.  Shape / metadata are captured at
registration time via lightweight schema introspection.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

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

from torch.utils.data import Dataset, Subset

# ---------------------------------------------------------------------------
# Type alias: all recognised lazy references
# ---------------------------------------------------------------------------
LazyRef = Union[
    "pl.LazyFrame",       # type: ignore[name-defined]
    "dd.DataFrame",       # type: ignore[name-defined]
    Dataset,
]


class DatasetManager:
    """
    Registry that stores lazy references to datasets.

    No data is materialised here.  Shape information is derived from schema
    introspection so downstream phases can inspect metadata cheaply without
    triggering a full table scan.

    Key public surface:
      - register_dataset(name, lazy_ref, metadata)  – add a lazy dataset
      - get(name)                                   – retrieve lazy ref
      - get_metadata(name)                          – retrieve metadata dict
      - get_shape_estimate(name)                    – (n_rows_or_None, n_cols)
      - list_datasets()                             – all registered names
      - split_dataset(name, train_ratio)            – lazy train/test split
    """

    def __init__(self) -> None:
        self._datasets: Dict[str, LazyRef] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #

    def register_dataset(
        self,
        name: str,
        lazy_ref: LazyRef,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a lazy dataset reference.

        Args:
            name:     Unique identifier (typically the SHA-256 hash).
            lazy_ref: Polars LazyFrame, Dask DataFrame, or PyTorch Dataset.
            metadata: Optional dict (source_url, hash, timestamps, etc.).
        """
        self._datasets[name] = lazy_ref
        base_meta: Dict[str, Any] = metadata or {}

        # Cheap schema probe – never triggers a full collect
        shape_estimate = self._probe_shape(lazy_ref)
        columns = self._probe_columns(lazy_ref)

        self._metadata[name] = {
            **base_meta,
            "shape_estimate": shape_estimate,
            "columns": columns,
        }

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #

    def get(self, name: str) -> Optional[LazyRef]:
        """Return the lazy reference for *name* without materialising it."""
        return self._datasets.get(name)

    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Return the full stored metadata dict for *name*."""
        return self._metadata.get(name, {})

    def get_shape_estimate(
        self,
        name: str,
    ) -> Optional[Tuple[Optional[int], int]]:
        """
        Return ``(n_rows_estimate_or_None, n_cols)`` without a full table scan.

        Row count is ``None`` for truly lazy sources (Polars LazyFrame, Dask)
        to avoid triggering an expensive aggregation.
        """
        meta = self._metadata.get(name, {})
        return meta.get("shape_estimate")

    def list_datasets(self) -> List[str]:
        """Return names of all registered datasets."""
        return list(self._datasets.keys())

    # ------------------------------------------------------------------ #
    # Lazy train / test split
    # ------------------------------------------------------------------ #

    def split_dataset(
        self,
        name: str,
        train_ratio: float = 0.8,
    ) -> Optional[Dict[str, LazyRef]]:
        """
        Produce a lazy train / test split without materialising the dataset.

        - Polars LazyFrame: row-index predicate pushdown split
        - PyTorch Dataset:  ``torch.utils.data.Subset`` slices
        - Dask DataFrame:   returned as-is (Dask fuses splits lazily)

        Returns ``None`` if *name* is not registered.
        """
        ref = self._datasets.get(name)
        if ref is None:
            return None

        if _POLARS_AVAILABLE and isinstance(ref, pl.LazyFrame):
            # Attach a temporary row index, split by predicate, then drop it
            # This keeps the plan lazy – no collect until downstream code asks
            df_with_idx = ref.with_row_index("__row_idx__")
            try:
                total: int = ref.select(pl.len()).collect().item()
            except Exception:
                # Re-raise instead of ref.collect().height which would
                # materialise the entire dataset into RAM (OOM on 50 GB+).
                raise RuntimeError(
                    "Cannot determine row count for lazy split. "
                    "Ensure the LazyFrame source is valid."
                )
            split_idx = int(total * train_ratio)
            train_lf = df_with_idx.filter(
                pl.col("__row_idx__") < split_idx
            ).drop("__row_idx__")
            test_lf = df_with_idx.filter(
                pl.col("__row_idx__") >= split_idx
            ).drop("__row_idx__")
            return {"train": train_lf, "test": test_lf}

        if isinstance(ref, Dataset):
            n = len(ref)
            split_idx = int(n * train_ratio)
            return {
                "train": Subset(ref, range(split_idx)),
                "test": Subset(ref, range(split_idx, n)),
            }

        # Dask fallback: return same ref for both splits; caller can
        # apply dask.dataframe iloc-equivalent if needed
        return {"train": ref, "test": ref}

    # ------------------------------------------------------------------ #
    # Temporal (chronological) reference / production split
    # ------------------------------------------------------------------ #

    def create_temporal_split(
        self,
        name: str,
        time_column: Optional[str] = None,
        ref_ratio: float = 0.70,
    ) -> Optional[Dict[str, LazyRef]]:
        """
        Split a dataset chronologically into **reference** (older 70 %) and
        **production** (recent 30 %) subsets for drift detection.

        Strategy
        --------
        - If ``time_column`` is provided and exists in the schema, rows are
          sorted ascending by that column before splitting so the oldest rows
          become the reference set.
        - When no time column is available (or the column is not found), a
          stable index-based split is used: the first ``ref_ratio`` fraction
          of rows is the reference set and the remainder is the production
          set.

        Supported backends
        ------------------
        - **Polars LazyFrame** – predicate-pushdown row-index split; sort by
          ``time_column`` when provided.  Row count is obtained via a single
          ``pl.len()`` collect (cheap, plan does not scan all data).
        - **pandas DataFrame** – ``sort_values`` / ``iloc`` split.
        - **PyTorch Dataset** – ``torch.utils.data.Subset`` index slices.
        - **Dask / other** – falls back to returning the same ref for both
          halves (caller must handle).

        Parameters
        ----------
        name        : Registered dataset name (typically its SHA-256 hash).
        time_column : Optional column name used for chronological ordering.
        ref_ratio   : Fraction of rows assigned to the reference set
                      (default 0.70).

        Returns
        -------
        ``{"reference": <lazy_ref>, "production": <lazy_ref>}`` or ``None``
        if *name* is not registered.
        """
        ref = self._datasets.get(name)
        if ref is None:
            return None

        # ── Polars LazyFrame ─────────────────────────────────────────────
        if _POLARS_AVAILABLE and isinstance(ref, pl.LazyFrame):
            schema_names = ref.collect_schema().names()

            if time_column and time_column in schema_names:
                # Sort ascending so oldest rows come first
                ordered = (
                    ref.sort(time_column, descending=False)
                    .with_row_index("__apex_split_idx__")
                )
            else:
                ordered = ref.with_row_index("__apex_split_idx__")

            # Single cheap aggregation to determine total row count
            try:
                total: int = ref.select(pl.len()).collect().item()
            except Exception:
                raise RuntimeError(
                    "Cannot determine row count for temporal split. "
                    "Ensure the LazyFrame source is valid."
                )

            split_idx: int = int(total * ref_ratio)

            reference_lf = (
                ordered
                .filter(pl.col("__apex_split_idx__") < split_idx)
                .drop("__apex_split_idx__")
            )
            production_lf = (
                ordered
                .filter(pl.col("__apex_split_idx__") >= split_idx)
                .drop("__apex_split_idx__")
            )
            return {"reference": reference_lf, "production": production_lf}

        # ── pandas DataFrame ─────────────────────────────────────────────
        try:
            import pandas as pd
            if isinstance(ref, pd.DataFrame):
                if time_column and time_column in ref.columns:
                    ordered_df = ref.sort_values(time_column, ascending=True)
                else:
                    ordered_df = ref
                split_idx = int(len(ordered_df) * ref_ratio)
                return {
                    "reference":  ordered_df.iloc[:split_idx],
                    "production": ordered_df.iloc[split_idx:],
                }
        except ImportError:
            pass

        # ── PyTorch Dataset ──────────────────────────────────────────────
        if isinstance(ref, Dataset):
            n: int = len(ref)
            split_idx = int(n * ref_ratio)
            return {
                "reference":  Subset(ref, range(split_idx)),
                "production": Subset(ref, range(split_idx, n)),
            }

        # ── Fallback (Dask / unsupported) ────────────────────────────────
        return {"reference": ref, "production": ref}

    # ------------------------------------------------------------------ #
    # Internal schema probers (never trigger a full scan)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _probe_shape(ref: LazyRef) -> Optional[Tuple[Optional[int], int]]:
        """Return ``(n_rows_or_None, n_cols)`` via schema introspection only."""
        if _POLARS_AVAILABLE and isinstance(ref, pl.LazyFrame):
            schema = ref.collect_schema()
            # Row count deliberately omitted – would require a full scan
            return (None, len(schema))

        if isinstance(ref, Dataset):
            return (len(ref), 1)  # n_cols not meaningful for image datasets

        # Dask
        try:
            return (None, len(ref.columns))  # type: ignore[union-attr]
        except Exception:
            return None

    @staticmethod
    def _probe_columns(ref: LazyRef) -> List[str]:
        """Return column names via schema introspection without materialising."""
        if _POLARS_AVAILABLE and isinstance(ref, pl.LazyFrame):
            return ref.collect_schema().names()

        if isinstance(ref, Dataset):
            return []  # image datasets have no column concept

        try:
            return list(ref.columns)  # type: ignore[union-attr]
        except Exception:
            return []
