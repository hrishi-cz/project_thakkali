"""Data ingestion package for loading and managing datasets."""

from .loader import DataLoader
from .schema import DataSchema

__all__ = ["DataLoader", "DataSchema"]
# Alias for compatibility
DatasetLoader = DataLoader