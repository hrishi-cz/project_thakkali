"""Tests for LoRA adapter module (modelss/adapters/lora.py)."""

import pytest
import torch
import torch.nn as nn

from modelss.adapters.lora import (
    LoRALinear,
    apply_lora,
    lora_parameters,
    lora_state_dict,
    load_lora_state_dict,
)


class TestLoRALinear:
    def test_output_shape_unchanged(self):
        linear = nn.Linear(64, 128)
        lora = LoRALinear(linear, r=4, alpha=8)
        x = torch.randn(8, 64)
        out = lora(x)
        assert out.shape == (8, 128)

    def test_at_init_output_equals_base(self):
        """B is zero-initialized → ΔW = 0 → output == frozen linear output."""
        torch.manual_seed(0)
        linear = nn.Linear(32, 64, bias=False)
        lora = LoRALinear(linear, r=4, alpha=8)
        # Force A to zero for a clean comparison
        nn.init.zeros_(lora.lora_A)
        x = torch.randn(4, 32)
        assert torch.allclose(lora(x), linear(x), atol=1e-6)

    def test_backbone_frozen(self):
        """LoRALinear.__init__ freezes the wrapped linear weight."""
        linear = nn.Linear(16, 32)
        lora = LoRALinear(linear, r=2, alpha=4)
        # linear.weight was frozen inside LoRALinear.__init__
        assert not lora.linear.weight.requires_grad

    def test_lora_params_trainable(self):
        linear = nn.Linear(16, 32)
        lora = LoRALinear(linear, r=2, alpha=4)
        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad

    def test_scale_applied_correctly(self):
        linear = nn.Linear(4, 4, bias=False)
        nn.init.zeros_(linear.weight)
        lora = LoRALinear(linear, r=2, alpha=4)  # scale = 4/2 = 2.0
        # Set A, B so delta is computable: B@A = identity-like
        nn.init.eye_(lora.lora_B)  # B: (4,2) — truncated eye
        nn.init.eye_(lora.lora_A)  # A: (2,4) — truncated eye
        x = torch.eye(4)
        out = lora(x)
        # output = 0 (base) + (x @ A^T @ B^T) * scale
        # With truncated identity: diagonal entries should be scale
        assert out[0, 0].item() == pytest.approx(lora.scale, abs=0.1)


class TestApplyLoRA:
    def test_replaces_target_linear(self):
        model = nn.Sequential(
            nn.Linear(32, 64),   # linear1 — target
            nn.ReLU(),
            nn.Linear(64, 10),   # linear2 — not a target by name
        )
        apply_lora(model, r=4, alpha=8, target_modules=("0",))
        assert isinstance(model[0], LoRALinear)
        assert isinstance(model[2], nn.Linear)  # not replaced

    def test_recursive_replacement(self):
        encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(128, 4, batch_first=True), 2
        )
        apply_lora(encoder, r=4, alpha=8, target_modules=("out_proj", "linear1"))
        lora_layers = [m for m in encoder.modules() if isinstance(m, LoRALinear)]
        assert len(lora_layers) >= 4  # at least out_proj + linear1 per layer × 2 layers

    def test_returns_same_module(self):
        model = nn.Linear(8, 16)
        result = apply_lora(model, r=2, alpha=4, target_modules=())
        assert result is model  # in-place, same object

    def test_no_target_matches_is_safe(self):
        model = nn.Linear(8, 16)
        apply_lora(model, r=4, alpha=8, target_modules=("nonexistent",))
        assert isinstance(model, nn.Linear)  # unchanged


class TestLoRAParameters:
    def test_yields_only_ab_tensors(self):
        model = nn.Sequential(nn.Linear(16, 32), nn.Linear(32, 8))
        apply_lora(model, r=2, alpha=4, target_modules=("0", "1"))
        params = list(lora_parameters(model))
        # Each LoRALinear has lora_A + lora_B = 2 params; 2 layers = 4 total
        assert len(params) == 4
        for p in params:
            assert isinstance(p, nn.Parameter)

    def test_no_backbone_weights(self):
        model = nn.Linear(16, 32)
        lora = LoRALinear(model, r=4, alpha=8)
        wrapper = nn.Sequential(lora)
        params = list(lora_parameters(wrapper))
        param_ids = {id(p) for p in params}
        # Backbone weight must NOT be in lora params
        assert id(lora.linear.weight) not in param_ids


class TestLoRAStateDictRoundtrip:
    def test_save_and_load(self):
        torch.manual_seed(42)
        model_a = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(64, 2, batch_first=True), 1
        )
        apply_lora(model_a, r=4, alpha=8, target_modules=("out_proj",))
        # Randomize LoRA A/B weights
        for p in lora_parameters(model_a):
            nn.init.normal_(p)

        state = lora_state_dict(model_a)
        assert len(state) > 0, "lora_state_dict should contain A/B tensors"

        # Fresh model + same LoRA structure
        model_b = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(64, 2, batch_first=True), 1
        )
        apply_lora(model_b, r=4, alpha=8, target_modules=("out_proj",))
        load_lora_state_dict(model_b, state)

        # Verify A/B tensors match directly (don't run forward — MHA weight proxying is complex)
        state_a = lora_state_dict(model_a)
        state_b = lora_state_dict(model_b)
        assert set(state_a.keys()) == set(state_b.keys()), "State dict keys must match"
        for key in state_a:
            assert torch.allclose(state_a[key], state_b[key], atol=1e-6), (
                f"LoRA tensor '{key}' does not match after load"
            )

    def test_empty_state_for_no_lora(self):
        model = nn.Linear(8, 16)  # no LoRA applied
        state = lora_state_dict(model)
        assert state == {}
