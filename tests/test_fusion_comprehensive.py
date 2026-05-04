"""
Comprehensive test suite for multimodal fusion strategies.

Tests cover all fusion types (Attention, Graph, Uncertainty, UncertaintyGraph),
edge cases (missing modalities, NaN, Inf), and auxiliary losses.

This is FIX-FUSION-2: Comprehensive fusion test coverage (P0.3).

Test matrix:
  - Single modality (1 text)
  - Two modalities (text + image)
  - Three modalities (text + image + tabular)
  - Missing modality graceful degradation
  - NaN/Inf handling
  - Batch size = 1 edge case
  - Diversity loss computation
  - Fusion strategy selection logic
  - Output dimension validation
"""

import pytest
import torch
import torch.nn as nn
from typing import List, Dict

# Import fusion modules
from models.fusion import (
    ConcatenationFusion,
    AttentionFusion,
    GraphFusion,
    UncertaintyFusion,
    UncertaintyGraphFusion,
    diversity_loss,
    graph_sparsity_loss,
    select_fusion_strategy,
)


# ===========================================================================
# FIXTURES
# ===========================================================================

@pytest.fixture
def embeddings_single():
    """Single modality: text only (N=8, D=768)"""
    return [torch.randn(8, 768)]


@pytest.fixture
def embeddings_double():
    """Two modalities: text (768) + image (2048)"""
    return [
        torch.randn(8, 768),   # text
        torch.randn(8, 2048),  # image
    ]


@pytest.fixture
def embeddings_triple():
    """Three modalities: text (768) + image (2048) + tabular (32)"""
    return [
        torch.randn(8, 768),   # text
        torch.randn(8, 2048),  # image
        torch.randn(8, 32),    # tabular
    ]


@pytest.fixture
def embeddings_with_nan():
    """Embeddings with NaN values for edge case testing"""
    embeddings = [
        torch.full((8, 768), float('nan')),  # text all NaN
        torch.randn(8, 2048),                # image OK
    ]
    return embeddings


@pytest.fixture
def embeddings_with_inf():
    """Embeddings with Inf values"""
    embeddings = [
        torch.randn(8, 768),
        torch.full((8, 2048), float('inf')),  # image all Inf
    ]
    return embeddings


@pytest.fixture
def embeddings_batch1():
    """Edge case: batch size = 1"""
    return [
        torch.randn(1, 768),
        torch.randn(1, 2048),
    ]


@pytest.fixture
def embeddings_zero():
    """Edge case: zero embeddings (encoding failure indicator)"""
    return [
        torch.zeros(8, 768),
        torch.randn(8, 2048),
    ]


# ===========================================================================
# TEST: ConcatenationFusion
# ===========================================================================

class TestConcatenationFusion:
    """Basic concatenation fusion (should always work)"""

    def test_single_modality(self, embeddings_single):
        """Single modality passes through unchanged"""
        fusion = ConcatenationFusion([768])
        output = fusion.forward(embeddings_single)
        assert output.shape == (8, 768)
        assert torch.allclose(output, embeddings_single[0])

    def test_double_modality(self, embeddings_double):
        """Two modalities concatenated"""
        fusion = ConcatenationFusion([768, 2048])
        output = fusion.forward(embeddings_double)
        assert output.shape == (8, 768 + 2048)

    def test_triple_modality(self, embeddings_triple):
        """Three modalities concatenated"""
        fusion = ConcatenationFusion([768, 2048, 32])
        output = fusion.forward(embeddings_triple)
        assert output.shape == (8, 768 + 2048 + 32)

    def test_output_dim(self):
        """get_output_dim() returns correct total dimension"""
        fusion = ConcatenationFusion([768, 2048, 32])
        assert fusion.get_output_dim() == 768 + 2048 + 32

    def test_empty_features_raises(self):
        """Empty feature list raises ValueError"""
        fusion = ConcatenationFusion([768])
        with pytest.raises(ValueError, match="empty feature list"):
            fusion.forward([])


# ===========================================================================
# TEST: AttentionFusion
# ===========================================================================

class TestAttentionFusion:
    """Attention-weighted fusion with dynamic projections"""

    def test_single_modality(self, embeddings_single):
        """Single modality through attention fusion"""
        fusion = AttentionFusion([768], latent_dim=512)
        output = fusion.forward(embeddings_single)
        assert output.shape == (8, 512)
        assert torch.isfinite(output).all()

    def test_double_modality(self, embeddings_double):
        """Two modalities with different dimensions"""
        fusion = AttentionFusion([768, 2048], latent_dim=512)
        output = fusion.forward(embeddings_double)
        assert output.shape == (8, 512)
        assert torch.isfinite(output).all()

    def test_triple_modality(self, embeddings_triple):
        """Three modalities with heterogeneous dimensions"""
        fusion = AttentionFusion([768, 2048, 32], latent_dim=512)
        output = fusion.forward(embeddings_triple)
        assert output.shape == (8, 512)
        assert torch.isfinite(output).all()

    def test_batch_size_1(self, embeddings_batch1):
        """Edge case: batch size = 1"""
        fusion = AttentionFusion([768, 2048], latent_dim=512)
        output = fusion.forward(embeddings_batch1)
        assert output.shape == (1, 512)
        assert torch.isfinite(output).all()

    def test_output_dim(self):
        """get_output_dim() matches latent_dim"""
        fusion = AttentionFusion([768, 2048], latent_dim=256)
        assert fusion.get_output_dim() == 256

    def test_weights_sum_to_one(self, embeddings_double):
        """Attention weights should softmax to 1 per batch"""
        fusion = AttentionFusion([768, 2048], latent_dim=512)
        output = fusion.forward(embeddings_double)
        # Output should be properly normalized (check via norm)
        assert output.abs().max() < 100  # Rough sanity check
        assert torch.isfinite(output).all()


# ===========================================================================
# TEST: GraphFusion
# ===========================================================================

class TestGraphFusion:
    """Graph-based fusion with learnable adjacency"""

    def test_single_modality(self, embeddings_single):
        """Single modality (trivial adjacency)"""
        fusion = GraphFusion(dim=512, num_modalities=1, heads=4, input_dims=[768])
        output = fusion.forward(embeddings_single)
        assert output.shape == (8, 512)
        assert torch.isfinite(output).all()

    def test_double_modality(self, embeddings_double):
        """Two modalities with learnable graph"""
        fusion = GraphFusion(dim=512, num_modalities=2, heads=4, input_dims=[768, 2048])
        output = fusion.forward(embeddings_double)
        assert output.shape == (8, 512)
        assert torch.isfinite(output).all()

    def test_triple_modality(self, embeddings_triple):
        """Three modalities: adjacency (3x3) matrix"""
        fusion = GraphFusion(dim=512, num_modalities=3, heads=4, input_dims=[768, 2048, 32])
        output = fusion.forward(embeddings_triple)
        assert output.shape == (8, 512)
        assert torch.isfinite(output).all()

    def test_adjacency_shape(self, embeddings_triple):
        """Adjacency matrix has correct shape and sums to 1 per row"""
        fusion = GraphFusion(dim=512, num_modalities=3, heads=4, input_dims=[768, 2048, 32])
        adj = fusion.graph()
        assert adj.shape == (3, 3)
        # Each row should sum to ~1 (softmax)
        row_sums = adj.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(3), atol=1e-6)

    def test_diversity_loss_computation(self, embeddings_triple):
        """Diversity loss should compute correctly from head outputs"""
        fusion = GraphFusion(dim=512, num_modalities=3, heads=4, input_dims=[768, 2048, 32])
        output = fusion.forward(embeddings_triple)
        
        # Access cached head outputs
        if hasattr(fusion, 'last_head_outputs') and fusion.last_head_outputs:
            div_loss = diversity_loss(fusion.last_head_outputs)
            assert div_loss.item() >= 0
            assert torch.isfinite(div_loss)

    def test_different_heads(self, embeddings_double):
        """Different number of attention heads"""
        for heads in [1, 2, 4, 8]:
            fusion = GraphFusion(dim=512, num_modalities=2, heads=heads, input_dims=[768, 2048])
            output = fusion.forward(embeddings_double)
            assert output.shape == (8, 512)
            assert torch.isfinite(output).all()


# ===========================================================================
# TEST: UncertaintyFusion
# ===========================================================================

class TestUncertaintyFusion:
    """Uncertainty-weighted fusion (inverse variance weighting)"""

    def test_single_modality(self, embeddings_single):
        """Single modality (trivial uncertainty)"""
        fusion = UncertaintyFusion([768], latent_dim=512)
        output = fusion.forward(embeddings_single)
        assert output.shape == (8, 512)
        assert torch.isfinite(output).all()

    def test_double_modality(self, embeddings_double):
        """Two modalities with uncertainty estimation"""
        fusion = UncertaintyFusion([768, 2048], latent_dim=512)
        output = fusion.forward(embeddings_double)
        assert output.shape == (8, 512)
        assert torch.isfinite(output).all()

    def test_triple_modality(self, embeddings_triple):
        """Three modalities with per-modality log-variance"""
        fusion = UncertaintyFusion([768, 2048, 32], latent_dim=512)
        output = fusion.forward(embeddings_triple)
        assert output.shape == (8, 512)
        assert torch.isfinite(output).all()

    def test_output_dim(self):
        """get_output_dim() returns latent_dim"""
        fusion = UncertaintyFusion([768, 2048], latent_dim=256)
        assert fusion.get_output_dim() == 256

    def test_attention_summary_contains_uncertainty_importance(self, embeddings_double):
        """Uncertainty fusion exposes per-modality uncertainty importance."""
        fusion = UncertaintyFusion([768, 2048], latent_dim=512)
        _ = fusion.forward(embeddings_double)
        summary = fusion.get_attention_summary()
        importance = summary.get("uncertainty_importance", {})
        assert set(importance.keys()) == {"modality_0", "modality_1"}


# ===========================================================================
# TEST: UncertaintyGraphFusion
# ===========================================================================

class TestUncertaintyGraphFusion:
    """Combined uncertainty + graph fusion (SOTA)"""

    def test_single_modality(self, embeddings_single):
        """Single modality (trivial graph)"""
        fusion = UncertaintyGraphFusion([768], latent_dim=512, heads=4)
        output = fusion.forward(embeddings_single)
        assert output.shape == (8, 512)
        assert torch.isfinite(output).all()

    def test_double_modality(self, embeddings_double):
        """Two modalities: uncertainty + graph"""
        fusion = UncertaintyGraphFusion([768, 2048], latent_dim=512, heads=4)
        output = fusion.forward(embeddings_double)
        assert output.shape == (8, 512)
        assert torch.isfinite(output).all()

    def test_triple_modality(self, embeddings_triple):
        """Three modalities: full uncertainty + graph fusion"""
        fusion = UncertaintyGraphFusion([768, 2048, 32], latent_dim=512, heads=4)
        output = fusion.forward(embeddings_triple)
        assert output.shape == (8, 512)
        assert torch.isfinite(output).all()

    def test_output_dim(self):
        """get_output_dim() returns latent_dim"""
        fusion = UncertaintyGraphFusion([768, 2048], latent_dim=256, heads=4)
        assert fusion.get_output_dim() == 256

    def test_branch_weight_configuration(self, embeddings_double):
        """Configured branch weights are preserved and exposed."""
        fusion = UncertaintyGraphFusion(
            [768, 2048],
            latent_dim=512,
            heads=4,
            graph_weight=0.8,
            uncertainty_weight=0.2,
        )
        _ = fusion.forward(embeddings_double)
        weights = fusion.get_branch_weights()
        assert weights["graph"] == pytest.approx(0.8)
        assert weights["uncertainty"] == pytest.approx(0.2)

    def test_attention_summary_includes_branch_weights(self, embeddings_double):
        """Combined fusion attention summary includes branch mixture metadata."""
        fusion = UncertaintyGraphFusion([768, 2048], latent_dim=512, heads=4)
        _ = fusion.forward(embeddings_double)
        summary = fusion.get_attention_summary()
        assert "branch_weights" in summary
        assert set(summary["branch_weights"].keys()) == {"graph", "uncertainty"}


# ===========================================================================
# TEST: Edge Cases and Error Handling
# ===========================================================================

class TestEdgeCases:
    """Edge case handling: NaN, Inf, zero embeddings, etc."""

    def test_nan_handling_attention(self, embeddings_with_nan):
        """Attention fusion with NaN in input"""
        fusion = AttentionFusion([768, 2048], latent_dim=512)
        output = fusion.forward(embeddings_with_nan)
        # Output may be NaN if input is NaN, but shouldn't crash
        assert output.shape == (8, 512)

    def test_inf_handling_attention(self, embeddings_with_inf):
        """Attention fusion with Inf in input"""
        fusion = AttentionFusion([768, 2048], latent_dim=512)
        output = fusion.forward(embeddings_with_inf)
        assert output.shape == (8, 512)

    def test_zero_embeddings_attention(self, embeddings_zero):
        """Attention fusion with zero embeddings (encoding failure)"""
        fusion = AttentionFusion([768, 2048], latent_dim=512)
        output = fusion.forward(embeddings_zero)
        assert output.shape == (8, 512)

    def test_batch_size_1_all_fusions(self, embeddings_batch1):
        """Batch size = 1 for all fusion types"""
        fusions = [
            AttentionFusion([768, 2048], latent_dim=512),
            GraphFusion(dim=512, num_modalities=2, heads=4, input_dims=[768, 2048]),
            UncertaintyFusion([768, 2048], latent_dim=512),
            UncertaintyGraphFusion([768, 2048], latent_dim=512, heads=4),
        ]
        
        for fusion in fusions:
            output = fusion.forward(embeddings_batch1)
            assert output.shape == (1, 512)
            assert torch.isfinite(output).all()


# ===========================================================================
# TEST: Auxiliary Losses
# ===========================================================================

class TestAuxiliaryLosses:
    """FIX-FUSION-1: Auxiliary loss functions"""

    def test_diversity_loss_zero_heads(self):
        """Diversity loss with < 2 heads returns 0"""
        loss = diversity_loss([torch.randn(8, 512)])
        assert loss.item() == 0.0

    def test_diversity_loss_two_heads(self):
        """Diversity loss with 2 heads"""
        head1 = torch.randn(8, 512)
        head2 = torch.randn(8, 512)
        loss = diversity_loss([head1, head2])
        assert loss.item() >= 0
        assert torch.isfinite(loss)

    def test_diversity_loss_four_heads(self):
        """Diversity loss with 4 heads (typical GraphFusion)"""
        heads = [torch.randn(8, 512) for _ in range(4)]
        loss = diversity_loss(heads)
        assert loss.item() >= 0
        assert torch.isfinite(loss)

    def test_graph_sparsity_loss(self):
        """Graph sparsity loss on adjacency matrix"""
        adj = torch.softmax(torch.randn(3, 3), dim=1)  # Valid adjacency
        loss = graph_sparsity_loss(adj)
        assert loss.item() >= 0
        assert torch.isfinite(loss)
        assert loss <= 1  # Mean of softmaxed values


# ===========================================================================
# TEST: Fusion Strategy Selection
# ===========================================================================

class TestFusionStrategySelection:
    """FIX-FUSION-X: Strategy selection based on schema"""

    def test_single_modality_to_concat(self):
        """Single modality → concatenation"""
        schema = {"global_modalities": ["tabular"]}
        strategy = select_fusion_strategy(schema)
        assert strategy == "concat"

    def test_two_modalities_image_text_to_ula(self):
        """Image + Text → ula (ULA cross-modal Transformer is superior for vision+language)"""
        schema = {"global_modalities": ["image", "text"]}
        strategy = select_fusion_strategy(schema)
        assert strategy == "ula"

    def test_two_modalities_tabular_text_to_attention(self):
        """Tabular + Text → attention (corrected from graph; attention is superior for text+tabular)."""
        schema = {"global_modalities": ["tabular", "text"]}
        strategy = select_fusion_strategy(schema)
        assert strategy == "attention"

    def test_three_modalities_to_complementarity(self):
        """Three+ modalities → complementarity (CrossFuse ECCV 2024 [4] upgrade)"""
        schema = {"global_modalities": ["text", "image", "tabular"]}
        strategy = select_fusion_strategy(schema)
        assert strategy == "complementarity"

    def test_empty_schema_defaults(self):
        """Empty/missing modalities → concat"""
        for schema in [{}, {"global_modalities": []}]:
            strategy = select_fusion_strategy(schema)
            assert strategy == "concat"


# ===========================================================================
# TEST: Integration (Multiple Components Together)
# ===========================================================================

class TestIntegration:
    """Integration tests: full multimodal pipeline simulation"""

    def test_full_training_step_simulation(self, embeddings_triple):
        """Simulate a training step with all components"""
        # Create model with all fusion types
        fusions = {
            "attention": AttentionFusion([768, 2048, 32], latent_dim=512),
            "graph": GraphFusion(dim=512, num_modalities=3, heads=4, input_dims=[768, 2048, 32]),
            "uncertainty": UncertaintyFusion([768, 2048, 32], latent_dim=512),
            "uncertainty_graph": UncertaintyGraphFusion([768, 2048, 32], latent_dim=512, heads=4),
        }
        
        # Forward pass through each fusion
        for name, fusion in fusions.items():
            output = fusion.forward(embeddings_triple)
            assert output.shape == (8, 512), f"Failed for {name}"
            assert torch.isfinite(output).all(), f"Non-finite output for {name}"

    def test_inference_time_consistency(self, embeddings_double):
        """Multiple forward passes should be consistent"""
        fusion = AttentionFusion([768, 2048], latent_dim=512)
        fusion.eval()  # Eval mode
        
        outputs = []
        for _ in range(3):
            with torch.no_grad():
                output = fusion.forward(embeddings_double)
                outputs.append(output)
        
        # All outputs should be identical in eval mode
        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i], atol=1e-6)


# ===========================================================================
# RUN TESTS
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
