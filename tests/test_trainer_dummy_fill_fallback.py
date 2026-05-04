"""Regression tests for trainer dummy-fill dimension fallback behavior."""

import pytest
import torch
import torch.nn as nn

from automl.trainer import ApexLightningModule, _MultimodalHead


class _DummyTabularEncoder(nn.Module):
    """Encoder stub without get_output_dim()."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_encode_batch_falls_back_to_head_dim_without_get_output_dim() -> None:
    head = _MultimodalHead(input_dims={"tabular": 4}, num_outputs=2)
    module = ApexLightningModule(
        model=head,
        problem_type="classification_binary",
        num_classes=2,
        tabular_encoder=_DummyTabularEncoder(),
    )

    batch = {
        "target": torch.tensor([0, 1]),
        "text_pooled": torch.randn(2, 768),
    }

    encoded = module._encode_batch(batch)

    assert "tabular" in encoded
    assert encoded["tabular"].shape == (2, 4)


def test_fusion_summary_exposes_auxiliary_weights_and_backend_module() -> None:
    head = _MultimodalHead(
        input_dims={"tabular": 4, "text_pooled": 8},
        num_outputs=2,
        fusion_strategy="uncertainty_graph",
        fusion_config={
            "uncertainty_graph_weight": 0.7,
            "uncertainty_branch_weight": 0.3,
        },
    )
    module = ApexLightningModule(
        model=head,
        problem_type="classification_binary",
        num_classes=2,
        fusion_aux_weights={
            "graph_sparsity_weight": 0.02,
            "diversity_loss_weight": 0.03,
            "uncertainty_aux_weight": 0.01,
        },
    )

    summary = module.get_fusion_summary()
    assert summary.get("fusion_type") == "UncertaintyGraphFusion"
    assert "modelss.fusion" in str(summary.get("backend_module", ""))
    assert summary.get("auxiliary_loss_weights", {}).get("graph_sparsity_weight") == pytest.approx(0.02)
    assert summary.get("auxiliary_loss_weights", {}).get("diversity_loss_weight") == pytest.approx(0.03)
    assert summary.get("auxiliary_loss_weights", {}).get("uncertainty_aux_weight") == pytest.approx(0.01)
    assert summary.get("branch_weights", {}).get("graph") == pytest.approx(0.7)
    assert summary.get("branch_weights", {}).get("uncertainty") == pytest.approx(0.3)
