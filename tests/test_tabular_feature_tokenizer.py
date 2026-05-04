"""Tests for TabularFeatureTokenizer in preprocessing/tabular_preprocessor.py."""

import pytest
import torch

from preprocessing.tabular_preprocessor import TabularFeatureTokenizer


class TestTabularFeatureTokenizer:
    def test_output_shape(self):
        tok = TabularFeatureTokenizer(n_features=10, token_dim=64)
        x = torch.randn(8, 10)
        out = tok(x)
        assert out.shape == (8, 10, 64), f"Expected (8,10,64) got {out.shape}"

    def test_output_dim_getter(self):
        tok = TabularFeatureTokenizer(n_features=20, token_dim=256)
        assert tok.get_output_dim() == 256

    def test_different_batch_sizes(self):
        tok = TabularFeatureTokenizer(n_features=5, token_dim=32)
        for batch in (1, 4, 16, 32):
            out = tok(torch.randn(batch, 5))
            assert out.shape == (batch, 5, 32)

    def test_different_token_dims(self):
        for token_dim in (32, 64, 128, 256, 512):
            tok = TabularFeatureTokenizer(n_features=3, token_dim=token_dim)
            out = tok(torch.randn(2, 3))
            assert out.shape == (2, 3, token_dim)

    def test_n_projections_equals_n_features(self):
        n = 15
        tok = TabularFeatureTokenizer(n_features=n, token_dim=64)
        assert len(tok.projections) == n

    def test_type_embedding_applied(self):
        """Two identical feature values but different positions should produce different tokens."""
        torch.manual_seed(42)
        tok = TabularFeatureTokenizer(n_features=4, token_dim=32)
        x = torch.ones(1, 4)  # all features have same value
        out = tok(x)  # (1, 4, 32)
        # Different positions → different embeddings → tokens should differ
        assert not torch.allclose(out[0, 0], out[0, 1]), (
            "Tokens at different positions should differ due to type embeddings"
        )

    def test_gradients_flow_through(self):
        tok = TabularFeatureTokenizer(n_features=6, token_dim=32)
        x = torch.randn(3, 6, requires_grad=True)
        out = tok(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_truncates_to_n_features_when_input_larger(self):
        """If input has more features than declared, we truncate to n_features."""
        tok = TabularFeatureTokenizer(n_features=5, token_dim=32)
        x = torch.randn(2, 8)  # 8 features but tok only handles 5
        out = tok(x)
        assert out.shape == (2, 5, 32)
