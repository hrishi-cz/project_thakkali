from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class DatasetObject:
    """Container for ingested dataset reference and ingestion metadata."""

    dataset_id: str
    lazy_data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
