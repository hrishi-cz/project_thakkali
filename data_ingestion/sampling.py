from __future__ import annotations

from data_ingestion.dataset_object import DatasetObject


def validate_dataset(dataset: DatasetObject) -> None:
    """Minimal dataset validation used by ingestion manager."""
    if not dataset.dataset_id:
        raise ValueError("DatasetObject.dataset_id is required")
    if dataset.lazy_data is None:
        raise ValueError("DatasetObject.lazy_data is None")
