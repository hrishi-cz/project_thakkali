"""Regression tests for modality-aware embedding cache skip policy."""

from __future__ import annotations

import torch

from pipeline.embedding_cache import EmbeddingCache


def test_low_priority_modality_is_skipped_with_hashed_key(tmp_path) -> None:
    cache = EmbeddingCache(
        cache_dir=tmp_path,
        modality_priorities={"text": 0.10, "image": 0.90},
    )

    hashed_key = EmbeddingCache.build_key({"encoder": "bert", "dataset": "abc", "idx": 1})
    tensor = torch.randn(4, 8)

    cache.set(hashed_key, tensor, meta={"modality": "text"})
    assert cache.get(hashed_key) is None

    stats = cache.stats()
    assert stats["skipped_writes"] >= 1


def test_high_priority_modality_is_cached(tmp_path) -> None:
    cache = EmbeddingCache(
        cache_dir=tmp_path,
        modality_priorities={"text": 0.10, "image": 0.90},
    )

    hashed_key = EmbeddingCache.build_key({"encoder": "resnet", "dataset": "abc", "idx": 2})
    tensor = torch.randn(2, 16)

    cache.set(hashed_key, tensor, meta={"modality": "image"})
    loaded = cache.get(hashed_key)

    assert loaded is not None
    assert tuple(loaded.shape) == tuple(tensor.shape)
    stats = cache.stats()
    assert stats["writes"] >= 1
