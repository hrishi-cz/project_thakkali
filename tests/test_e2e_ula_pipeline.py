"""End-to-end integration tests for ULA token-mode pipeline wiring."""

import pytest
import torch
import torch.nn as nn
import numpy as np


class TestULATokenModeFlag:
    """ApexLightningModule correctly sets _use_token_sequences for ULA fusion."""

    def test_non_ula_fusion_no_token_mode(self):
        from automl.trainer import build_trainer

        module = build_trainer(
            problem_type="classification_binary",
            num_classes=2,
            input_dims={"tabular": 16},
            fusion_strategy="concatenation",
        )
        assert module._use_token_sequences is False

    def test_ula_without_token_mode_false(self):
        from automl.trainer import build_trainer

        module = build_trainer(
            problem_type="classification_binary",
            num_classes=2,
            input_dims={"tabular": 16, "text_pooled": 768},
            fusion_strategy="ula",
            fusion_config={"latent_dim": 64, "n_layers": 1, "n_heads": 2, "token_mode": False},
        )
        # token_mode=False → _use_token_sequences must be False
        assert module._use_token_sequences is False

    def test_ula_with_token_mode_true(self):
        from automl.trainer import build_trainer

        module = build_trainer(
            problem_type="classification_binary",
            num_classes=2,
            input_dims={"tabular": 16, "text_pooled": 768},
            fusion_strategy="ula",
            fusion_config={"latent_dim": 64, "n_layers": 1, "n_heads": 2, "token_mode": True},
        )
        assert module._use_token_sequences is True

    def test_gated_fusion_no_token_mode(self):
        from automl.trainer import build_trainer

        module = build_trainer(
            problem_type="classification_binary",
            num_classes=2,
            input_dims={"tabular": 16, "image_pooled": 512},
            fusion_strategy="gated",
        )
        assert module._use_token_sequences is False


class TestTabularTokenizerPassthrough:
    """TabularFeatureTokenizer is stored and accessible via _tabular_tokenizer."""

    def test_tokenizer_stored_when_provided(self):
        from automl.trainer import build_trainer
        from preprocessing.tabular_preprocessor import TabularFeatureTokenizer

        tok = TabularFeatureTokenizer(n_features=8, token_dim=64)
        module = build_trainer(
            problem_type="classification_binary",
            num_classes=2,
            input_dims={"tabular": 8},
            fusion_strategy="ula",
            fusion_config={"latent_dim": 64, "n_layers": 1, "n_heads": 2, "token_mode": True},
            tabular_tokenizer=tok,
        )
        assert module._tabular_tokenizer is tok

    def test_tokenizer_none_by_default(self):
        from automl.trainer import build_trainer

        module = build_trainer(
            problem_type="classification_binary",
            num_classes=2,
            input_dims={"tabular": 8},
        )
        assert module._tabular_tokenizer is None


class TestLoRaWarmStartPresence:
    """LoRA warm-start state is accessible through ApexLightningModule."""

    def test_lora_applied_to_encoder(self):
        from automl.trainer import build_trainer
        from modelss.adapters.lora import LoRALinear

        text_enc = _make_fake_text_encoder()
        module = build_trainer(
            problem_type="classification_binary",
            num_classes=2,
            input_dims={"text_pooled": 768},
            text_encoder=text_enc,
            lora_config={"r": 4, "alpha": 8},
        )
        lora_layers = [m for m in text_enc.modules() if isinstance(m, LoRALinear)]
        assert len(lora_layers) > 0, "LoRA should be applied to text encoder"

    def test_lora_params_require_grad(self):
        from automl.trainer import build_trainer
        from modelss.adapters.lora import lora_parameters

        text_enc = _make_fake_text_encoder()
        module = build_trainer(
            problem_type="classification_binary",
            num_classes=2,
            input_dims={"text_pooled": 768},
            text_encoder=text_enc,
            lora_config={"r": 4, "alpha": 8},
        )
        params = list(lora_parameters(text_enc))
        assert len(params) > 0
        assert all(p.requires_grad for p in params)


class TestULAConfigSaveLoad:
    """ULA config JSON roundtrip (unit-level, no full pipeline)."""

    def test_ula_config_dict_structure(self):
        from modelss.fusion import UnifiedLatentFusion

        fusion = UnifiedLatentFusion(
            feature_dims=[16, 32],
            latent_dim=64,
            n_heads=2,
            n_layers=2,
            token_mode=False,
        )
        assert fusion.latent_dim == 64
        assert fusion.token_mode is False
        assert len(fusion.transformer.layers) == 2

    def test_ula_token_mode_flag_stored(self):
        from modelss.fusion import UnifiedLatentFusion

        fusion = UnifiedLatentFusion(
            feature_dims=[64],
            latent_dim=128,
            n_heads=4,
            n_layers=1,
            token_mode=True,
        )
        assert fusion.token_mode is True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_text_encoder():
    """Return a minimal TextEncoder-like nn.Module with attention-named Linears for LoRA."""
    class _FakeTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(32, 32)   # matches _DEFAULT_TARGETS
            self.value = nn.Linear(32, 32)   # matches _DEFAULT_TARGETS

        def forward(self, x):
            return x

    class _FakeTextEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = _FakeTransformer()
            self._projection = None

        def forward(self, x):
            return x

    return _FakeTextEncoder()
