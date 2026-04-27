"""ECG / time-series data adapter.

Converts ECG files (WFDB .hea/.dat or plain CSV with numeric columns) into
a standardised dict that the ingestion pipeline can consume. The modality
tag is ``"timeseries"`` so the TimeSeriesEncoder is selected automatically.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["ECGAdapter", "load_ecg_record"]


def load_ecg_record(path: str) -> Dict[str, Any]:
    """
    Load a single ECG record from a file.

    Supported formats:
    - CSV with numeric columns → treated as multi-channel time series
    - NumPy .npy files → shape (seq_len,) or (seq_len, channels)
    - WFDB format (.hea header) → uses ``wfdb`` library if available

    Returns
    -------
    dict with keys:
        ``signal``   : np.ndarray  shape (seq_len, channels)
        ``n_channels``: int
        ``seq_len``  : int
        ``source``   : str
        ``modality`` : "timeseries"
    """
    path = str(path)
    suffix = Path(path).suffix.lower()

    signal: Optional[np.ndarray] = None

    if suffix == ".npy":
        raw = np.load(path)
        signal = raw if raw.ndim == 2 else raw[:, None]

    elif suffix == ".hea":
        try:
            import wfdb  # type: ignore
            record = wfdb.rdrecord(path[:-4])  # strip .hea
            signal = np.array(record.p_signal, dtype=np.float32)
        except ImportError:
            logger.warning("wfdb not installed — falling back to CSV companion")
            csv_path = path.replace(".hea", ".csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                signal = df.select_dtypes(include=[np.number]).values.astype(np.float32)

    elif suffix in (".csv", ".tsv"):
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        signal = df.select_dtypes(include=[np.number]).values.astype(np.float32)

    if signal is None:
        raise ValueError(f"ECGAdapter: unsupported or unreadable format: {path}")

    if signal.ndim == 1:
        signal = signal[:, None]

    return {
        "signal": signal.astype(np.float32),
        "n_channels": signal.shape[1],
        "seq_len": signal.shape[0],
        "source": path,
        "modality": "timeseries",
    }


class ECGAdapter:
    """
    Batch adapter for ECG / time-series datasets.

    Converts a list of ECG file paths (or a directory) into a DataFrame
    where each row is one record and the ``signal`` column contains the
    numpy array.  Adds a ``modality`` column tagged as ``"timeseries"``
    so the downstream schema detector and modality router pick it up.
    """

    def __init__(self, max_seq_len: int = 5000) -> None:
        self.max_seq_len = int(max_seq_len)

    def load_directory(self, directory: str) -> pd.DataFrame:
        """Load all ECG files from a directory."""
        extensions = {".hea", ".npy", ".csv", ".tsv"}
        paths = [
            str(p)
            for p in Path(directory).iterdir()
            if p.suffix.lower() in extensions
        ]
        return self.load_files(paths)

    def load_files(self, paths: List[str]) -> pd.DataFrame:
        """Load a list of ECG file paths."""
        rows = []
        for path in paths:
            try:
                record = load_ecg_record(path)
                # Truncate or pad to max_seq_len
                sig = record["signal"]
                if sig.shape[0] > self.max_seq_len:
                    sig = sig[: self.max_seq_len]
                record["signal"] = sig
                rows.append(record)
            except Exception as exc:
                logger.warning("ECGAdapter: skipping %s: %s", path, exc)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["modality"] = "timeseries"
        return df
