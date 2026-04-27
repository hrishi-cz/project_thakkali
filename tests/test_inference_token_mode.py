"""Tests for inference engine ULA token-mode detection and attention rollout."""

import pytest
import torch
import torch.nn as nn
import numpy as np


class TestULATokenModeDetection:
    """_use_token_sequences is derived from the head's fusion class."""

    def _build_head_with_fusion(self, fusion_strategy, token_mode=False):
        from automl.trainer import _MultimodalHead
        head = _MultimodalHead(
            input_dims={"tabular": 16, "text_pooled": 768},
            hidden_dim=64,
            num_outputs=2,
            fusion_strategy=fusion_strategy,
            fusion_config={"latent_dim": 64, "n_layers": 1, "n_heads": 2, "token_mode": token_mode},
        )
        return head

    def test_concat_head_no_token_mode(self):
        from automl.trainer import ApexLightningModule
        head = self._build_head_with_fusion("concatenation")
        module = ApexLightningModule(
            model=head,
            problem_type="classification_binary",
            num_classes=2,
        )
        assert module._use_token_sequences is False

    def test_ula_head_token_mode_false(self):
        from automl.trainer import ApexLightningModule
        head = self._build_head_with_fusion("ula", token_mode=False)
        module = ApexLightningModule(
            model=head,
            problem_type="classification_binary",
            num_classes=2,
        )
        assert module._use_token_sequences is False

    def test_ula_head_token_mode_true(self):
        from automl.trainer import ApexLightningModule
        head = self._build_head_with_fusion("ula", token_mode=True)
        module = ApexLightningModule(
            model=head,
            problem_type="classification_binary",
            num_classes=2,
        )
        assert module._use_token_sequences is True


class TestAttentionRolloutShape:
    """_attention_rollout returns correct shape from ViT-like encoder."""

    def _make_vit_like_encoder(self, embed_dim=64, n_heads=2, n_layers=2, patch_grid=4):
        """Minimal TransformerEncoder that processes (N, P+1, D) where P = patch_grid^2."""
        class _ViTLike(nn.Module):
            def __init__(self):
                super().__init__()
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim, nhead=n_heads, batch_first=True
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
                self.embed_dim = embed_dim
                self.patch_grid = patch_grid

            def forward(self, x, return_all_tokens=False):
                # x: (N, 3, H, W) → flatten to (N, P+1, embed_dim) via rand proj
                N = x.shape[0]
                P = self.patch_grid ** 2
                tokens = torch.randn(N, P + 1, self.embed_dim)
                out = self.encoder(tokens)
                if return_all_tokens:
                    return out[:, 1:, :]   # (N, P, D)
                return out[:, 0, :]        # (N, D) CLS token

        return _ViTLike()

    def test_attention_rollout_fallback_no_transformers(self):
        """When encoder has no TransformerEncoderLayer, rollout returns error dict."""
        # Build a minimal engine-like object manually
        class _FakeEngine:
            device = torch.device("cpu")
            _image_encoder = nn.Sequential(nn.Conv2d(3, 16, 3))  # has Conv2d, not ViT

        from pipeline.inference_engine import MultimodalInferenceEngine
        # Monkey-patch just _attention_rollout onto a plain object
        eng = object.__new__(MultimodalInferenceEngine)
        object.__setattr__(eng, "device", torch.device("cpu"))
        object.__setattr__(eng, "_image_encoder", nn.Linear(8, 8))  # no TransformerEncoderLayer

        img = torch.randn(1, 3, 32, 32)
        result = eng._attention_rollout(img)
        assert "gradcam_available" in result
        assert result["gradcam_available"] is False


class TestLoRAInferenceLoad:
    """LoRA state dicts round-trip through save/load correctly."""

    def test_lora_state_dict_empty_for_plain_linear(self):
        from modelss.adapters.lora import lora_state_dict
        model = nn.Linear(16, 32)
        state = lora_state_dict(model)
        assert state == {}

    def test_lora_state_dict_nonempty_after_apply(self):
        from modelss.adapters.lora import apply_lora, lora_state_dict
        model = nn.Sequential(nn.Linear(16, 32))
        apply_lora(model, r=4, alpha=8, target_modules=("0",))
        state = lora_state_dict(model)
        assert len(state) > 0

    def test_load_lora_state_dict_copies_weights(self):
        from modelss.adapters.lora import apply_lora, lora_state_dict, load_lora_state_dict

        model_a = nn.Sequential(nn.Linear(16, 32))
        apply_lora(model_a, r=4, alpha=8, target_modules=("0",))
        for p in model_a.parameters():
            if p.requires_grad:
                nn.init.normal_(p)

        state = lora_state_dict(model_a)

        model_b = nn.Sequential(nn.Linear(16, 32))
        apply_lora(model_b, r=4, alpha=8, target_modules=("0",))
        load_lora_state_dict(model_b, state)

        state_b = lora_state_dict(model_b)
        for key in state:
            assert torch.allclose(state[key], state_b[key], atol=1e-6)
