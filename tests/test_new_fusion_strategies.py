"""Tests for GatedFusion, UnifiedLatentFusion (ULA token mode), and FuseMoE."""

import pytest
import torch

from modelss.fusion import GatedFusion, UnifiedLatentFusion, FuseMoE


@pytest.fixture
def two_modality_features():
    return [torch.randn(4, 64), torch.randn(4, 128)]


@pytest.fixture
def three_modality_features():
    return [torch.randn(4, 16), torch.randn(4, 768), torch.randn(4, 512)]


# ─────────────────────────────────────────────────────────────────────────────
# GatedFusion
# ─────────────────────────────────────────────────────────────────────────────

class TestGatedFusion:
    def test_output_shape(self, three_modality_features):
        gf = GatedFusion([16, 768, 512], output_dim=256)
        out = gf(three_modality_features)
        assert out.shape == (4, 256)

    def test_output_dim_matches_get_output_dim(self, two_modality_features):
        gf = GatedFusion([64, 128], output_dim=512)
        assert gf.get_output_dim() == 512
        out = gf(two_modality_features)
        assert out.shape[1] == gf.get_output_dim()

    def test_gates_are_in_zero_one(self, three_modality_features):
        gf = GatedFusion([16, 768, 512], output_dim=128)
        _ = gf(three_modality_features)
        gates = gf._last_gates
        assert gates is not None
        assert all(0.0 <= g <= 1.0 for g in gates), f"Gates out of [0,1]: {gates}"

    def test_modality_mask_zeroes_contribution(self):
        gf = GatedFusion([32, 32], output_dim=32)
        features = [torch.ones(2, 32), torch.zeros(2, 32)]
        out_nomask = gf(features)
        # When first modality is masked, gate for it should be ~0
        out_masked = gf(
            features,
            modality_mask={"mod0": False, "mod1": True},
            modality_names=["mod0", "mod1"],
        )
        # Outputs differ (one modality suppressed)
        assert not torch.allclose(out_nomask, out_masked)

    def test_two_identical_inputs_deterministic(self):
        torch.manual_seed(7)
        gf = GatedFusion([16, 16], output_dim=32, dropout=0.0)
        gf.eval()  # disable dropout
        f = [torch.randn(2, 16), torch.randn(2, 16)]
        with torch.no_grad():
            out1 = gf(f)
            out2 = gf(f)
        assert torch.allclose(out1, out2)

    def test_attention_summary(self, two_modality_features):
        gf = GatedFusion([64, 128], output_dim=64)
        gf(two_modality_features)
        summary = gf.get_attention_summary()
        assert "gate_weights" in summary


# ─────────────────────────────────────────────────────────────────────────────
# UnifiedLatentFusion — pooled vector mode (backward-compatible)
# ─────────────────────────────────────────────────────────────────────────────

class TestUnifiedLatentFusionPooled:
    def test_output_shape_pooled(self, three_modality_features):
        ula = UnifiedLatentFusion([16, 768, 512], latent_dim=256, n_heads=4, n_layers=2)
        out = ula(three_modality_features)
        assert out.shape == (4, 256)

    def test_get_output_dim(self):
        ula = UnifiedLatentFusion([64, 128], latent_dim=128)
        assert ula.get_output_dim() == 128

    def test_alignment_loss_scalar(self, three_modality_features):
        ula = UnifiedLatentFusion([16, 768, 512], latent_dim=128)
        loss = ula.alignment_loss(three_modality_features)
        assert loss.shape == ()
        assert not torch.isnan(loss)
        assert loss.item() >= 0.0

    def test_all_modalities_masked_returns_cls(self):
        ula = UnifiedLatentFusion([32, 64], latent_dim=64, n_heads=2, n_layers=1)
        features = [torch.randn(2, 32), torch.randn(2, 64)]
        out = ula(
            features,
            modality_mask={"a": False, "b": False},
            modality_names=["a", "b"],
        )
        assert out.shape == (2, 64)

    def test_single_modality_present(self):
        ula = UnifiedLatentFusion([32, 64], latent_dim=64, n_heads=2, n_layers=1)
        features = [torch.randn(2, 32), torch.randn(2, 64)]
        out = ula(
            features,
            modality_mask={"a": True, "b": False},
            modality_names=["a", "b"],
        )
        assert out.shape == (2, 64)


# ─────────────────────────────────────────────────────────────────────────────
# UnifiedLatentFusion — token-sequence mode (cross-modal attention)
# ─────────────────────────────────────────────────────────────────────────────

class TestUnifiedLatentFusionTokenMode:
    def test_token_sequence_output_shape(self):
        ula = UnifiedLatentFusion([128, 768, 512], latent_dim=256, n_heads=4, n_layers=2, token_mode=True)
        tab   = torch.randn(4, 10, 128)   # 10 tabular feature tokens
        text  = torch.randn(4, 64, 768)   # 64 text tokens
        image = torch.randn(4, 196, 512)  # 196 image patch tokens
        out = ula([tab, text, image])
        assert out.shape == (4, 256)

    def test_mixed_pooled_and_sequence(self):
        """Pooled (N,D) and sequence (N,T,D) can be mixed in same forward call."""
        ula = UnifiedLatentFusion([16, 768], latent_dim=128, n_heads=2, n_layers=1)
        tab_pooled = torch.randn(4, 16)      # (N, D) — pooled
        text_seq   = torch.randn(4, 32, 768) # (N, T, D) — sequence
        out = ula([tab_pooled, text_seq])
        assert out.shape == (4, 128)

    def test_token_count_logged(self):
        ula = UnifiedLatentFusion([64, 128], latent_dim=64, n_heads=2, n_layers=1, token_mode=True)
        tab_seq  = torch.randn(2, 5, 64)
        text_seq = torch.randn(2, 10, 128)
        ula([tab_seq, text_seq])
        # CLS not counted in _last_token_count (it's the aggregation token)
        assert ula._last_token_count == 15  # 5 + 10

    def test_longer_sequences_do_not_crash(self):
        """Simulate real ViT+BERT token counts: 196 patches + 128 tokens."""
        ula = UnifiedLatentFusion([512, 768], latent_dim=256, n_heads=8, n_layers=2, token_mode=True)
        image_tokens = torch.randn(2, 196, 512)
        text_tokens  = torch.randn(2, 128, 768)
        out = ula([image_tokens, text_tokens])
        assert out.shape == (2, 256)

    def test_alignment_loss_on_token_sequences(self):
        ula = UnifiedLatentFusion([64, 128], latent_dim=64, n_heads=2, n_layers=1, token_mode=True)
        features = [torch.randn(4, 8, 64), torch.randn(4, 16, 128)]
        loss = ula.alignment_loss(features)
        assert loss.shape == ()
        assert not torch.isnan(loss)


# ─────────────────────────────────────────────────────────────────────────────
# FuseMoE
# ─────────────────────────────────────────────────────────────────────────────

class TestFuseMoE:
    def test_output_shape(self, three_modality_features):
        moe = FuseMoE([16, 768, 512], output_dim=256, n_experts=4, top_k=2)
        out = moe(three_modality_features)
        assert out.shape == (4, 256)

    def test_get_output_dim(self):
        moe = FuseMoE([32, 64], output_dim=128, n_experts=3, top_k=1)
        assert moe.get_output_dim() == 128

    def test_missing_modality_via_mask(self):
        moe = FuseMoE([32, 64, 128], output_dim=64, n_experts=4, top_k=2)
        features = [torch.randn(2, 32), torch.randn(2, 64), torch.randn(2, 128)]
        out_full = moe(features)
        out_partial = moe(
            features,
            modality_mask={"tab": True, "text": False, "image": True},
            modality_names=["tab", "text", "image"],
        )
        assert out_full.shape == out_partial.shape == (2, 64)
        # Different routing → different outputs
        assert not torch.allclose(out_full, out_partial)

    def test_routing_probabilities_sum_to_one(self, two_modality_features):
        moe = FuseMoE([64, 128], output_dim=64, n_experts=3, top_k=2)
        moe(two_modality_features)
        routing = moe._last_routing
        assert routing is not None
        assert abs(sum(routing) - 1.0) < 0.01

    def test_top_k_clamped(self):
        moe = FuseMoE([16], output_dim=32, n_experts=2, top_k=10)  # top_k clamped to n_experts
        assert moe.top_k <= moe.n_experts

    def test_attention_summary(self, two_modality_features):
        moe = FuseMoE([64, 128], output_dim=64, n_experts=3, top_k=2)
        moe(two_modality_features)
        summary = moe.get_attention_summary()
        assert "expert_routing" in summary
        assert "n_experts" in summary
