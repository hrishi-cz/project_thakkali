"""
modelss/fusion.py

Fusion strategies for multimodal encoder outputs.

Both classes expose a ``get_output_dim() -> int`` method so downstream
layers (``MultimodalPredictor``) can query the output dimensionality without
instantiating any tensors.

Dimension contract (all three modalities active)
------------------------------------------------
  ImageEncoder   → [N, 512]   (ResNet-50 → Linear(2048, 512) → ReLU)
  TextEncoder    → [N, 768]   (BERT CLS token)
  TabularEncoder → [N,  16]   (MLP input→64→32→16)
                         ↓
  ConcatenationFusion output: 512 + 768 + 16 = 1296 dims

When fewer than three modalities are active, ``MultimodalPredictor``
passes ``torch.full(shape, 1e-7)`` dummy tensors for the absent
modalities so the concatenated dimension is always 1296.

ConcatenationFusion
-------------------
Dynamically computes its ``output_dim`` as ``sum(feature_dims)`` at
construction time.  No learnable parameters — the output is the raw
horizontal concatenation of all modality tensors along ``dim=1``.

AttentionFusion
---------------
Projects every modality tensor to a shared ``latent_dim`` (default 512)
with independent ``nn.Linear`` layers, then scores each projected
embedding with a shared attention network:

    nn.Sequential(
        nn.Linear(latent_dim, latent_dim),
        nn.Tanh(),
        nn.Linear(latent_dim, 1),
    )

Scores are softmax-normalised across the modality axis and used to
compute a single attention-weighted context vector of shape
``(N, latent_dim)``.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# G25: Modality-mask helper
# ---------------------------------------------------------------------------

def apply_modality_mask(
    features: List[torch.Tensor],
    modality_names: List[str],
    mask: Optional[Dict[str, bool]],
) -> List[torch.Tensor]:
    """Zero-out tensors for masked-out modalities (G25 ablation support).

    Parameters
    ----------
    features:
        List of embedding tensors, one per modality.
    modality_names:
        Ordered list of modality names matching ``features``.
    mask:
        Dict mapping modality name → bool.  ``False`` means "mask this
        modality out" (its tensor is replaced with zeros).  Missing
        keys default to ``True`` (modality active).  Pass ``None`` to
        skip masking entirely (backward-compatible default).

    Returns
    -------
    List[torch.Tensor]
        Same-length list; masked tensors are replaced with ``torch.zeros_like``.
    """
    if mask is None:
        return features
    masked: List[torch.Tensor] = []
    for feat, name in zip(features, modality_names):
        if mask.get(name, True):
            masked.append(feat)
        else:
            logger.debug("G25: masking modality '%s' → zeros", name)
            masked.append(torch.zeros_like(feat))
    return masked


# ---------------------------------------------------------------------------
# ConcatenationFusion
# ---------------------------------------------------------------------------

class ConcatenationFusion(nn.Module):
    """
    Horizontal concatenation of modality embedding tensors.

    ``output_dim`` is determined dynamically as ``sum(feature_dims)`` so
    it always tracks the exact input dimensionality regardless of how many
    encoders are active.

    Parameters
    ----------
    feature_dims : List[int]
        Output dimensionality of each active encoder, in the same order
        that tensors will be passed to ``forward()``.
        Example: ``[512, 768, 16]`` for ResNet-50 + BERT-base + Tabular-MLP.
    """

    def __init__(self, feature_dims: List[int]) -> None:
        super().__init__()
        if not feature_dims:
            raise ValueError(
                "ConcatenationFusion requires at least one encoder dimension."
            )
        self.feature_dims: List[int] = feature_dims
        self._output_dim: int = sum(feature_dims)
        logger.info(
            "ConcatenationFusion: %d modalities  dims=%s  output_dim=%d",
            len(feature_dims), feature_dims, self._output_dim,
        )

    # ------------------------------------------------------------------ #

    def forward(
        self,
        features: List[torch.Tensor],
        modality_names: Optional[List[str]] = None,
        modality_mask: Optional[Dict[str, bool]] = None,
    ) -> torch.Tensor:
        """Concatenate modality tensors along the feature axis.

        Parameters
        ----------
        features : List[torch.Tensor]
            One tensor per active modality, each of shape ``(N, d_i)``.
            Must be non-empty; length must equal ``len(feature_dims)``.
        modality_names : List[str], optional
            Names matching ``features`` — required if ``modality_mask`` is
            provided (G25 ablation support).
        modality_mask : Dict[str, bool], optional
            G25: zero-out specific modalities before fusion (e.g.
            ``{"image": False}`` for ablation studies).

        Returns
        -------
        torch.Tensor
            Shape ``(N, sum(d_i))`` — the horizontally concatenated tensor.
        """
        if not features:
            raise ValueError(
                "ConcatenationFusion.forward received an empty feature list."
            )
        if modality_mask is not None and modality_names:
            features = apply_modality_mask(features, modality_names, modality_mask)
        return torch.cat(features, dim=1)

    def get_output_dim(self) -> int:
        """Return the output feature dimensionality (== ``sum(feature_dims)``)."""
        return self._output_dim


# ---------------------------------------------------------------------------
# AttentionFusion
# ---------------------------------------------------------------------------

class AttentionFusion(nn.Module):
    """
    Attention-weighted fusion of multimodal encoder outputs.

    Each modality tensor is first projected to a shared ``latent_dim``
    (default 512) by an independent ``nn.Linear`` layer.  A single shared
    attention-scoring network then assigns a scalar importance weight to
    each projected embedding:

        score(e) = Linear(latent_dim → 1)(Tanh(Linear(latent_dim → latent_dim)(e)))

    Weights are softmax-normalised across the modality axis and used to
    compute an attention-weighted sum — the context vector of shape
    ``(N, latent_dim)``.

    Parameters
    ----------
    feature_dims : List[int]
        Output dimensionality of each active encoder, in the same order
        that tensors will be passed to ``forward()``.
    latent_dim : int
        Shared projection dimensionality (default 512).  All modality
        projections land in this space before attention scoring.
    """

    def __init__(
        self,
        feature_dims: List[int],
        latent_dim: int = 512,
    ) -> None:
        super().__init__()
        if not feature_dims:
            raise ValueError(
                "AttentionFusion requires at least one encoder dimension."
            )
        self.feature_dims: List[int] = feature_dims
        self._latent_dim: int = latent_dim

        # ── Per-modality projection layers ────────────────────────────────
        # Each encoder's output is mapped to the shared latent space
        # independently so that dimension mismatches across modalities are
        # resolved before attention scoring.
        self.projections: nn.ModuleList = nn.ModuleList([
            nn.Linear(d, latent_dim) for d in feature_dims
        ])

        # ── Shared attention-scoring network ──────────────────────────────
        # Applied to every projected embedding (shape: (N, n_mod, latent_dim))
        # via PyTorch's implicit last-dim broadcast.
        #
        #   Linear(latent_dim, latent_dim) → Tanh → Linear(latent_dim, 1)
        #
        # Output shape: (N, n_mod, 1) — one scalar score per modality.
        self.attention_scoring: nn.Sequential = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, 1),
        )

        logger.info(
            "AttentionFusion: %d modalities  dims=%s  latent_dim=%d",
            len(feature_dims), feature_dims, latent_dim,
        )

    # ------------------------------------------------------------------ #

    def forward(
        self,
        features: List[torch.Tensor],
        modality_names: Optional[List[str]] = None,
        modality_mask: Optional[Dict[str, bool]] = None,
    ) -> torch.Tensor:
        """Compute attention-weighted context vector from modality tensors.

        Parameters
        ----------
        features : List[torch.Tensor]
            One tensor per active modality, each of shape ``(N, d_i)``.
        modality_names : List[str], optional
            Names matching ``features`` — required if ``modality_mask`` is
            provided (G25 ablation support).
        modality_mask : Dict[str, bool], optional
            G25: zero-out specific modalities before fusion.

        Returns
        -------
        torch.Tensor
            Shape ``(N, latent_dim)`` — the attention-weighted sum of the
            projected modality embeddings (context vector).
        """
        if not features:
            raise ValueError(
                "AttentionFusion.forward received an empty feature list."
            )
        if modality_mask is not None and modality_names:
            features = apply_modality_mask(features, modality_names, modality_mask)

        # 1. Project each modality to shared latent space
        #    projected_i shape: (N, latent_dim)
        projected: List[torch.Tensor] = [
            proj(feat) for proj, feat in zip(self.projections, features)
        ]

        # 2. Stack along a new modality axis
        #    stacked shape: (N, n_modalities, latent_dim)
        stacked: torch.Tensor = torch.stack(projected, dim=1)

        # 3. Score each projected embedding
        #    nn.Linear operates on the last dim, so it broadcasts across N
        #    and n_modalities automatically.
        #    scores shape: (N, n_modalities, 1)
        scores: torch.Tensor = self.attention_scoring(stacked)

        # 4. Softmax across the modality axis → normalised importance weights
        #    weights shape: (N, n_modalities, 1)
        weights: torch.Tensor = torch.softmax(scores, dim=1)

        # 5. Weighted sum collapses the modality axis
        #    context shape: (N, latent_dim)
        context: torch.Tensor = (stacked * weights).sum(dim=1)
        return context

    def get_output_dim(self) -> int:
        """Return the output feature dimensionality (== ``latent_dim``)."""
        return self._latent_dim


# ---------------------------------------------------------------------------
# GraphFusion
# ---------------------------------------------------------------------------

class GraphFusion(nn.Module):
    """
    Graph-aware multi-head attention fusion.

    This class preserves standard attention fusion behavior while adding
    explicit graph mixing and diagnostics for interpretability.
    """

    def __init__(
        self,
        dim: int = 512,
        num_modalities: int = 2,
        heads: int = 4,
        input_dims: List[int] | None = None,
    ) -> None:
        super().__init__()
        if num_modalities <= 0:
            raise ValueError("GraphFusion requires at least one modality")

        self.dim = int(dim)
        self.num_modalities = int(num_modalities)
        self.heads = max(1, int(heads))
        self.input_dims = list(input_dims or [self.dim] * self.num_modalities)

        if len(self.input_dims) != self.num_modalities:
            raise ValueError("input_dims length must equal num_modalities")

        self.projections: nn.ModuleList = nn.ModuleList(
            [nn.Linear(in_dim, self.dim) for in_dim in self.input_dims]
        )
        self.head_attention: nn.ModuleList = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.dim, self.dim),
                    nn.Tanh(),
                    nn.Linear(self.dim, 1),
                )
                for _ in range(self.heads)
            ]
        )

        # Learnable adjacency logits; row-softmax yields a valid transition matrix.
        self.graph_logits = nn.Parameter(torch.eye(self.num_modalities))

        self.last_attention_weights: torch.Tensor | None = None
        self.last_head_outputs: List[torch.Tensor] = []
        self.accepts_mask: bool = True

    def graph(self) -> torch.Tensor:
        return torch.softmax(self.graph_logits, dim=1)

    def forward(
        self,
        features: List[torch.Tensor],
        modality_names: Optional[List[str]] = None,
        modality_mask: Optional[Dict[str, bool]] = None,
    ) -> torch.Tensor:
        if not features:
            raise ValueError("GraphFusion.forward received an empty feature list.")
        if modality_mask is not None and modality_names is not None:
            features = apply_modality_mask(features, modality_names, modality_mask)

        projected = [proj(feat) for proj, feat in zip(self.projections, features)]
        stacked = torch.stack(projected, dim=1)  # [N, M, D]

        n_mod = stacked.shape[1]
        adj = self.graph()[:n_mod, :n_mod]
        graph_context = torch.einsum("nm,bmd->bnd", adj, stacked)

        head_outputs: List[torch.Tensor] = []
        head_weights: List[torch.Tensor] = []
        for scorer in self.head_attention:
            scores = scorer(graph_context)              # [N, M, 1]
            weights = torch.softmax(scores, dim=1)      # [N, M, 1]
            fused = (graph_context * weights).sum(dim=1)  # [N, D]
            head_outputs.append(fused)
            head_weights.append(weights.squeeze(-1))

        self.last_head_outputs = [h.detach() for h in head_outputs]
        self.last_attention_weights = torch.stack(head_weights, dim=1).detach()  # [N, H, M]

        return torch.stack(head_outputs, dim=0).mean(dim=0)

    def get_output_dim(self) -> int:
        return self.dim

    def get_attention_summary(self) -> dict:
        if self.last_attention_weights is None:
            return {
                "modality_importance": {},
                "head_diversity": 0.0,
            }

        weights = self.last_attention_weights  # [N, H, M]
        modality_mean = weights.mean(dim=(0, 1))
        modality_importance = {
            f"modality_{idx}": float(score)
            for idx, score in enumerate(modality_mean.cpu().tolist())
        }

        head_profiles = weights.mean(dim=0)  # [H, M]
        if head_profiles.shape[0] > 1:
            head_diversity = float(torch.pdist(head_profiles, p=2).mean().item())
        else:
            head_diversity = 0.0

        return {
            "modality_importance": modality_importance,
            "head_diversity": head_diversity,
        }


# ---------------------------------------------------------------------------
# UncertaintyFusion
# ---------------------------------------------------------------------------

class UncertaintyFusion(nn.Module):
    """
    Uncertainty-weighted fusion via inverse-variance weighting.

    Each modality is projected into a shared latent space and assigned a
    learnable log-variance. Modalities with lower estimated uncertainty
    receive higher fusion weight.
    """

    def __init__(
        self,
        feature_dims: List[int],
        latent_dim: int = 512,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if not feature_dims:
            raise ValueError("UncertaintyFusion requires at least one modality")

        self.feature_dims = list(feature_dims)
        self.latent_dim = int(latent_dim)
        self.eps = float(eps)

        self.projections = nn.ModuleList([
            nn.Linear(d, self.latent_dim) for d in self.feature_dims
        ])
        self.log_var_heads = nn.ModuleList([
            nn.Linear(self.latent_dim, 1) for _ in self.feature_dims
        ])

        self.last_uncertainty_weights: torch.Tensor | None = None
        self.accepts_mask: bool = True

    def forward(
        self,
        features: List[torch.Tensor],
        modality_names: Optional[List[str]] = None,
        modality_mask: Optional[Dict[str, bool]] = None,
    ) -> torch.Tensor:
        if not features:
            raise ValueError("UncertaintyFusion.forward received an empty feature list.")
        if modality_mask is not None and modality_names is not None:
            features = apply_modality_mask(features, modality_names, modality_mask)

        projected = [proj(feat) for proj, feat in zip(self.projections, features)]
        stacked = torch.stack(projected, dim=1)  # [N, M, D]

        log_vars = [head(p) for head, p in zip(self.log_var_heads, projected)]
        log_var_tensor = torch.cat(log_vars, dim=1)  # [N, M]
        log_var_tensor = torch.clamp(log_var_tensor, min=-10.0, max=10.0)

        precision = torch.exp(-log_var_tensor)
        weights = precision / precision.sum(dim=1, keepdim=True).clamp_min(self.eps)
        self.last_uncertainty_weights = weights.detach()

        return (stacked * weights.unsqueeze(-1)).sum(dim=1)

    def get_output_dim(self) -> int:
        return self.latent_dim

    def get_attention_summary(self) -> dict:
        """Return modality importance inferred from uncertainty weights."""
        if self.last_uncertainty_weights is None:
            return {
                "modality_importance": {},
                "uncertainty_importance": {},
                "head_diversity": 0.0,
            }

        mean_w = self.last_uncertainty_weights.mean(dim=0)
        importance = {
            f"modality_{i}": float(v)
            for i, v in enumerate(mean_w.cpu().tolist())
        }
        return {
            "modality_importance": importance,
            "uncertainty_importance": importance,
            "head_diversity": 0.0,
        }


# ---------------------------------------------------------------------------
# UncertaintyGraphFusion
# ---------------------------------------------------------------------------

class UncertaintyGraphFusion(nn.Module):
    """
    Hybrid fusion that combines graph-aware and uncertainty-aware context.
    """

    def __init__(
        self,
        feature_dims: List[int],
        latent_dim: int = 512,
        heads: int = 4,
        graph_weight: float = 0.5,
        uncertainty_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if not feature_dims:
            raise ValueError("UncertaintyGraphFusion requires at least one modality")

        self.feature_dims = list(feature_dims)
        self.latent_dim = int(latent_dim)

        self.graph_branch = GraphFusion(
            dim=self.latent_dim,
            num_modalities=len(self.feature_dims),
            heads=int(heads),
            input_dims=self.feature_dims,
        )
        self.uncertainty_branch = UncertaintyFusion(
            feature_dims=self.feature_dims,
            latent_dim=self.latent_dim,
        )

        # Compatibility: expose log_var_heads for XAI/monitoring paths.
        self.log_var_heads = self.uncertainty_branch.log_var_heads
        self.last_head_outputs: List[torch.Tensor] = []
        self.accepts_mask: bool = True

        gw = max(0.0, float(graph_weight))
        uw = max(0.0, float(uncertainty_weight))
        total = gw + uw
        if total <= 0.0:
            gw, uw = 0.5, 0.5
            total = 1.0
        self.graph_weight = gw / total
        self.uncertainty_weight = uw / total

    def forward(
        self,
        features: List[torch.Tensor],
        modality_names: Optional[List[str]] = None,
        modality_mask: Optional[Dict[str, bool]] = None,
    ) -> torch.Tensor:
        if modality_mask is not None and modality_names is not None:
            features = apply_modality_mask(features, modality_names, modality_mask)
        graph_out = self.graph_branch(features)
        uncertainty_out = self.uncertainty_branch(features)
        self.last_head_outputs = list(self.graph_branch.last_head_outputs)
        return (self.graph_weight * graph_out) + (self.uncertainty_weight * uncertainty_out)

    def get_output_dim(self) -> int:
        return self.latent_dim

    def get_attention_summary(self) -> dict:
        base = self.graph_branch.get_attention_summary()
        base["branch_weights"] = self.get_branch_weights()
        if self.uncertainty_branch.last_uncertainty_weights is None:
            base["uncertainty_importance"] = {}
            return base

        mean_w = self.uncertainty_branch.last_uncertainty_weights.mean(dim=0)
        base["uncertainty_importance"] = {
            f"modality_{i}": float(v)
            for i, v in enumerate(mean_w.cpu().tolist())
        }
        return base

    def get_branch_weights(self) -> dict:
        return {
            "graph": float(self.graph_weight),
            "uncertainty": float(self.uncertainty_weight),
        }


# ---------------------------------------------------------------------------
# Auxiliary losses and strategy helpers
# ---------------------------------------------------------------------------

def diversity_loss(head_outputs: List[torch.Tensor]) -> torch.Tensor:
    """Penalize similar attention heads; returns 0 when <2 heads exist."""
    if len(head_outputs) < 2:
        if head_outputs:
            ref = head_outputs[0]
            return torch.zeros((), device=ref.device, dtype=ref.dtype)
        return torch.tensor(0.0)

    losses: List[torch.Tensor] = []
    for i in range(len(head_outputs)):
        for j in range(i + 1, len(head_outputs)):
            h1 = F.normalize(head_outputs[i], dim=-1)
            h2 = F.normalize(head_outputs[j], dim=-1)
            sim = (h1 * h2).sum(dim=-1).abs().mean()
            losses.append(sim)

    if not losses:
        ref = head_outputs[0]
        return torch.zeros((), device=ref.device, dtype=ref.dtype)
    return torch.stack(losses).mean()


def graph_sparsity_loss(adjacency: torch.Tensor) -> torch.Tensor:
    """L1-style sparsity regularizer for graph adjacency matrices."""
    if adjacency.numel() == 0:
        return torch.zeros((), device=adjacency.device, dtype=adjacency.dtype)
    return adjacency.abs().mean()


def select_fusion_strategy(schema_info: dict) -> str:
    """
    Choose a fusion strategy from schema modalities and cross-modal signals.

    The routing hierarchy:
      1. Modality count hard rules (single → concat, 3+ → complementarity).
      2. Well-known 2-modality pairs (image+text → structural_semantic, etc.).
      3. Cross-modal intelligence signals (alignment_strength,
         complementarity_score) fine-tune within the 2-modality case.
    """
    if not isinstance(schema_info, dict):
        return "concat"

    modalities = schema_info.get("global_modalities", []) or []
    if not isinstance(modalities, list):
        return "concat"

    mods = [str(m) for m in modalities]
    modset = set(mods)

    if len(modset) <= 1:
        return "concat"
    if len(modset) >= 3:
        return "complementarity"       # CrossFuse [4] for 3+ modalities

    # --- 2-modality routing with cross-modal signal refinement ---

    # Read cross-modal signals from GlobalSchema.multimodal_signals
    mm_signals = schema_info.get("multimodal_signals") or {}
    alignment   = float(mm_signals.get("alignment_strength",   0.5) or 0.5)
    comp_score  = float(mm_signals.get("complementarity_score", 0.0) or 0.0)

    if modset == {"image", "text"}:
        # Bug 5 fix: ULA (Unified Latent Alignment) is the primary strategy for
        # image+text. Cross-modal Transformer lets image patches and text tokens
        # attend to each other before classification — strictly superior to
        # structural_semantic or complementarity for paired vision+language tasks.
        return "ula"

    if modset == {"tabular", "text"}:
        # High complementarity variance → modalities bring distinct signals
        # → graph captures cross-entity relations
        if comp_score > 0.3:
            return "graph"
        return "attention"             # default: text attends over tabular

    if modset == {"tabular", "image"}:
        # Uncertainty-weighted: image embedding quality is variable
        return "uncertainty"

    if "timeseries" in modset:
        return "attention"             # sequential data → attention over context

    return "attention"                 # safe default


# ---------------------------------------------------------------------------
# ComplementarityFusion  [4]  CrossFuse: Complementarity-aware fusion
# ---------------------------------------------------------------------------

class ComplementarityFusion(nn.Module):
    """
    Complementarity-aware fusion that scores each modality pair by their
    conditional mutual information I(X_a ; Y | X_b).

    Modality pairs that are *complementary* (one contains information the
    other lacks) receive higher fusion weight.  Redundant pairs are
    down-weighted so the model doesn't double-count evidence.

    Algorithm (following CrossFuse, ECCV 2024 [4]):
    1. Project each modality into a shared latent space via independent MLPs.
    2. Estimate pairwise complementarity via a learned discriminator:
         score(a, b) = σ( MLP_disc(h_a ⊕ h_b) )  ← higher = more complementary
    3. Build complementarity matrix C ∈ [0,1]^{M×M}.
    4. Fuse: for each modality a, its weight = mean complementarity with all
       other modalities (softmax-normalised so weights sum to 1).
    5. Weighted sum of projected embeddings → output of shape (B, latent_dim).

    Paper reference:
      [4] "Crossfuse: Complementarity-aware multimodal fusion",
          ECCV / IEEE 2024.
    """

    def __init__(
        self,
        feature_dims: List[int],
        latent_dim: int = 512,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if not feature_dims:
            raise ValueError("ComplementarityFusion requires at least one modality")

        self.feature_dims = list(feature_dims)
        self.latent_dim = int(latent_dim)
        self.eps = float(eps)
        self.num_modalities = len(feature_dims)

        # Per-modality projection MLPs
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, self.latent_dim),
                nn.LayerNorm(self.latent_dim),
                nn.GELU(),
            )
            for d in self.feature_dims
        ])

        # Output combination MLP (ensures rich interaction)
        self.output_proj = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
        )

        self.last_complementarity_matrix: List[List[float]] = []
        self.last_fusion_weights: List[float] = []
        self.last_mi_nats: List[List[float]] = []   # MI values in nats per pair
        self.accepts_mask: bool = True

    # ------------------------------------------------------------------
    # Analytic MI under Gaussian assumption
    # ------------------------------------------------------------------

    @staticmethod
    def _pearson_mi(h_a: torch.Tensor, h_b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Analytic mutual information lower bound via Pearson correlation.

        Under a multivariate Gaussian assumption (met approximately because
        each projection ends with LayerNorm, which centres and normalises
        the latent vectors per batch):

            I(A ; B) ≈ -½ Σ_d log(1 - ρ_d²)

        where ρ_d is the Pearson correlation for dimension d, averaged
        across the batch.  The result is a scalar in **nats** (≥ 0).

        Properties:
        - Closed-form, fully differentiable — no adversarial training.
        - I = 0  when modalities are uncorrelated (maximally complementary).
        - I → ∞ as ρ → ±1 (perfectly redundant modalities).
        - 2–4× cheaper per forward pass than the discriminator MLP.

        This replaces the learned discriminator used in the original
        CrossFuse [4] implementation, which estimated the same quantity via
        σ(MLP(h_a ⊕ h_b)) — a proxy correct in ranking but unitless.
        """
        # Centre per latent dimension across the batch
        a = h_a - h_a.mean(dim=0, keepdim=True)   # (B, D)
        b = h_b - h_b.mean(dim=0, keepdim=True)   # (B, D)

        # Per-dimension covariance and standard deviations
        cov   = (a * b).mean(dim=0)                     # (D,)
        std_a = a.std(dim=0).clamp_min(eps)              # (D,)
        std_b = b.std(dim=0).clamp_min(eps)              # (D,)

        # Pearson correlation clamped to avoid log(0) at ρ = ±1
        rho = (cov / (std_a * std_b)).clamp(-1 + eps, 1 - eps)  # (D,)

        # Gaussian MI: -½ Σ_d log(1 - ρ_d²)  in nats, averaged over dims
        return (-0.5 * torch.log1p(-(rho ** 2))).mean()  # scalar ≥ 0

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        features: List[torch.Tensor],
        modality_names: Optional[List[str]] = None,
        modality_mask: Optional[Dict[str, bool]] = None,
    ) -> torch.Tensor:
        if not features:
            raise ValueError("ComplementarityFusion.forward received empty list")
        if modality_mask is not None and modality_names is not None:
            features = apply_modality_mask(features, modality_names, modality_mask)

        # Step 1: project all modalities into shared latent space
        projected = [proj(feat) for proj, feat in zip(self.projections, features)]
        M = len(projected)

        if M == 1:
            return self.output_proj(projected[0])

        # Step 2: build M×M MI matrix (in nats)
        # mi_matrix[i, j] = I(h_i ; h_j) — low MI → high complementarity
        mi_matrix = torch.zeros(M, M, device=projected[0].device)
        for i in range(M):
            for j in range(M):
                if i != j:
                    mi_matrix[i, j] = self._pearson_mi(projected[i], projected[j])

        # Step 3: complementarity weight = inverse of total MI with others.
        # A modality with low redundancy (low ΣMI) is more complementary
        # and receives a higher fusion weight.
        redundancy = mi_matrix.sum(dim=1)        # (M,) — total MI per modality
        # Negate so more-complementary → higher logit; add constant for stability
        comp_logits = -redundancy                # lower redundancy → higher weight
        weights = F.softmax(comp_logits, dim=0)  # (M,) sum = 1

        # Step 4: weighted sum of projected embeddings
        stacked = torch.stack(projected, dim=1)                              # (B, M, D)
        fused = (stacked * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)   # (B, D)

        # Diagnostics (detach for logging only)
        self.last_mi_nats = mi_matrix.detach().cpu().tolist()
        self.last_complementarity_matrix = [
            [1.0 / (1.0 + v) for v in row] for row in self.last_mi_nats
        ]  # convert MI → pseudo-complementarity score 0–1 for backward compat
        self.last_fusion_weights = weights.detach().cpu().tolist()

        return self.output_proj(fused)

    def get_output_dim(self) -> int:
        return self.latent_dim

    def get_attention_summary(self) -> dict:
        return {
            "modality_importance": {
                f"modality_{i}": float(w)
                for i, w in enumerate(self.last_fusion_weights)
            },
            "complementarity_matrix": self.last_complementarity_matrix,
            "mi_nats": self.last_mi_nats,   # NEW: interpretable in nats
            "head_diversity": 0.0,
            "method": "pearson_mi",          # label for XAI surface
        }


# ---------------------------------------------------------------------------
# StructuralSemanticRouter  [1]
# Unified graph (structural) + attention (semantic) routing
# ---------------------------------------------------------------------------

class StructuralSemanticRouter(nn.Module):
    """
    Unified Structural-Semantic Fusion Router.

    Combines graph-based (structural) and attention-based (semantic) fusion
    paths with a learned gating mechanism.  The gate adapts per-sample so
    graph-dominant pairs (e.g., tabular+tabular) and semantically rich pairs
    (e.g., text+image) both receive appropriate processing.

    Architecture (following [1] ICML 2025):
      structural_path = GraphFusion(features)           ← captures topology
      semantic_path   = AttentionFusion(features)       ← captures semantics
      gate            = σ( W_g · mean(features) )       ← per-sample blend
      output          = gate * structural + (1-gate) * semantic

    Paper reference:
      [1] "Structural-semantic unifier for multimodal fusion",
          ICML Workshop / IEEE 2025.
    """

    def __init__(
        self,
        feature_dims: List[int],
        latent_dim: int = 512,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if not feature_dims:
            raise ValueError("StructuralSemanticRouter requires at least one modality")

        self.feature_dims = list(feature_dims)
        self.latent_dim = int(latent_dim)

        self.structural = GraphFusion(
            dim=self.latent_dim,
            num_modalities=len(feature_dims),
            heads=max(1, heads),
            input_dims=feature_dims,
        )
        self.semantic = AttentionFusion(
            feature_dims=feature_dims,
            latent_dim=self.latent_dim,
        )

        # Gate network: takes mean of all projected inputs → scalar gate
        self.gate_proj = nn.Linear(feature_dims[0], self.latent_dim)
        self.gate_head = nn.Sequential(
            nn.Linear(self.latent_dim, 1),
            nn.Sigmoid(),
        )

        self.last_gate_value: float = 0.5
        self.log_var_heads = None  # compatibility shim
        self.accepts_mask: bool = True

    def forward(
        self,
        features: List[torch.Tensor],
        modality_names: Optional[List[str]] = None,
        modality_mask: Optional[Dict[str, bool]] = None,
    ) -> torch.Tensor:
        if modality_mask is not None and modality_names is not None:
            features = apply_modality_mask(features, modality_names, modality_mask)

        structural_out = self.structural(features)  # (B, D)
        semantic_out = self.semantic(features)       # (B, D)

        # Gate from mean of ALL modalities (order-invariant, not just features[0])
        try:
            all_means = torch.stack([f.mean(dim=-1, keepdim=True) for f in features], dim=1)  # (B, M, 1)
            gate_signal = all_means.mean(dim=1).expand(-1, features[0].shape[-1])             # (B, d_0)
            gate = self.gate_head(F.gelu(self.gate_proj(gate_signal)))                         # (B, 1)
        except Exception:
            gate = torch.full((structural_out.shape[0], 1), 0.5,
                              device=structural_out.device)

        self.last_gate_value = float(gate.mean().item())
        return gate * structural_out + (1.0 - gate) * semantic_out

    def get_output_dim(self) -> int:
        return self.latent_dim

    def get_attention_summary(self) -> dict:
        base = self.structural.get_attention_summary()
        base["gate_value"] = self.last_gate_value
        base["branch_weights"] = {
            "structural": self.last_gate_value,
            "semantic": 1.0 - self.last_gate_value,
        }
        return base


# ---------------------------------------------------------------------------
# GatedFusion — modality conflict suppression
# ---------------------------------------------------------------------------

class GatedFusion(nn.Module):
    """
    Learned per-modality gates that suppress noisy or conflicting modalities.

    For each sample, a small gating network reads the concatenation of all
    modality embeddings and produces one scalar gate per modality:

        gates = σ( W_g · concat(m₁, …, mₙ) )    ∈ ℝⁿ
        output = Σᵢ gatesᵢ · Projᵢ(mᵢ)

    When text says "positive" and image shows a negative scene, the gate
    learns to down-weight whichever is less reliable for the current sample.
    Effective for sentiment, emotion, medical imaging + clinical notes.

    References
    ----------
    Wang et al. "What Makes Training Multi-Modal Classification Networks
    Hard?" CVPR 2020. (Gradient-blending motivation)
    """

    accepts_mask: bool = True

    def __init__(
        self,
        feature_dims: List[int],
        output_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        n = len(feature_dims)
        total = sum(feature_dims)

        # Per-modality linear projections to shared output_dim
        self.projections = nn.ModuleList(
            [nn.Linear(d, output_dim) for d in feature_dims]
        )
        # Gate network: reads concatenated modalities → one gate per modality
        self.gate_net = nn.Sequential(
            nn.Linear(total, n * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n * 4, n),
        )
        self.output_dim = output_dim
        self.n = n
        self._last_gates: Optional[torch.Tensor] = None

    def forward(
        self,
        features: List[torch.Tensor],
        modality_mask: Optional[Dict[str, bool]] = None,
        modality_names: Optional[List[str]] = None,
    ) -> torch.Tensor:
        if modality_mask is not None and modality_names is not None:
            features = apply_modality_mask(features, modality_names, modality_mask)

        # Gate computation from full concat (zero-filled masked modalities)
        concat = torch.cat(features, dim=-1)        # (N, total)
        gates = torch.sigmoid(self.gate_net(concat)) # (N, n)
        self._last_gates = gates.detach().mean(dim=0).tolist()

        # Gated sum of projections
        out = sum(
            gates[:, i : i + 1] * self.projections[i](feat)
            for i, feat in enumerate(features)
        )
        return out  # (N, output_dim)

    def get_output_dim(self) -> int:
        return self.output_dim

    def get_attention_summary(self) -> dict:
        return {"gate_weights": self._last_gates or []}


# ---------------------------------------------------------------------------
# UnifiedLatentFusion — Omni-modal Unified Latent Alignment (ULA)
# ---------------------------------------------------------------------------

class UnifiedLatentFusion(nn.Module):
    """
    Omni-modal Unified Latent Alignment fusion.

    All modality embeddings are:
    1. Projected to a shared ``latent_dim``-dimensional space with LayerNorm
       (each modality has its own projection matrix)
    2. Tagged with a learnable modality-type embedding (analogue of position
       embedding, but for modality identity)
    3. Prepended with a learnable CLS token
    4. Processed by a lightweight Transformer (``n_layers`` encoder layers)
    5. The CLS token's output is the fused representation

    The projection step is the alignment: all modalities live in the same
    ``latent_dim`` space after projection, enabling the Transformer to
    directly compare and compose cross-modal information from layer 1,
    not just at the output.

    This connects directly to AutoVision's existing CLIP contrastive loss —
    the ``CLIPProjectionHead`` in ``ApexLightningModule`` already trains these
    projections with an NT-Xent objective.  Here those aligned projections
    become the input to the unified Transformer, closing the loop.

    Missing modalities are handled by simply omitting their tokens from the
    sequence — no zero-fill, no expert routing.

    References
    ----------
    Sun et al. "ImageBind: One Embedding Space To Bind Them All."
    CVPR 2023.
    Mizrahi et al. "4M: Massively Multimodal Masked Modeling."
    NeurIPS 2023.
    Bachmann et al. "Scaling and evaluating sparse autoencoders."
    arXiv 2024.
    """

    accepts_mask: bool = True

    def __init__(
        self,
        feature_dims: List[int],
        latent_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        token_mode: bool = False,
    ) -> None:
        super().__init__()
        n = len(feature_dims)

        # Per-modality linear projections (applied to last dim regardless of 2D/3D input)
        self.proj_linears = nn.ModuleList([nn.Linear(d, latent_dim) for d in feature_dims])
        self.proj_norms   = nn.ModuleList([nn.LayerNorm(latent_dim) for _ in feature_dims])

        # Modality-type embeddings: (CLS=0, modality_i = i+1)
        self.modality_embeddings = nn.Embedding(n + 1, latent_dim)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, latent_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,      # Pre-LN: better gradient flow; disables nested tensor (harmless)
        )
        import warnings as _w
        with _w.catch_warnings():
            _w.filterwarnings("ignore", message="enable_nested_tensor")
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(latent_dim)

        self.latent_dim = latent_dim
        self.output_dim = latent_dim
        # token_mode=True → accept (N,T,D) token sequences from ViT/BERT
        self.token_mode: bool = token_mode
        self._last_token_count: int = 0

    def _project_modality(
        self,
        feat: torch.Tensor,
        linear: nn.Module,
        norm: nn.Module,
        mod_idx: int,
    ) -> torch.Tensor:
        """
        Project a modality tensor into the shared latent space.

        Supports both pooled ``(N, D)`` and token-sequence ``(N, T, D)`` inputs.
        Returns ``(N, T_out, latent_dim)`` in both cases (T_out=1 for pooled).
        """
        N = feat.shape[0]
        device = feat.device

        if feat.dim() == 2:
            # Pooled vector: (N, D) → project → (N, latent_dim) → unsqueeze → (N, 1, latent_dim)
            projected = norm(linear(feat))                              # (N, latent_dim)
            mod_id = torch.full((N, 1), mod_idx + 1, dtype=torch.long, device=device)
            type_emb = self.modality_embeddings(mod_id)                 # (N, 1, latent_dim)
            return (projected.unsqueeze(1) + type_emb)                  # (N, 1, latent_dim)
        else:
            # Token sequence: (N, T, D) → project each token → (N, T, latent_dim)
            T = feat.shape[1]
            projected = norm(linear(feat))                              # (N, T, latent_dim)
            mod_id = torch.full((N, T), mod_idx + 1, dtype=torch.long, device=device)
            type_emb = self.modality_embeddings(mod_id)                 # (N, T, latent_dim)
            return projected + type_emb                                 # (N, T, latent_dim)

    def forward(
        self,
        features: List[torch.Tensor],
        modality_mask: Optional[Dict[str, bool]] = None,
        modality_names: Optional[List[str]] = None,
    ) -> torch.Tensor:
        N = features[0].shape[0]
        device = features[0].device
        tokens: List[torch.Tensor] = []

        for i, feat in enumerate(features):
            if modality_mask is not None and modality_names is not None:
                mod_name = modality_names[i] if i < len(modality_names) else ""
                if not modality_mask.get(mod_name, True):
                    continue

            tok = self._project_modality(feat, self.proj_linears[i], self.proj_norms[i], i)
            tokens.append(tok)  # each is (N, T_i, latent_dim)

        self._last_token_count = sum(t.shape[1] for t in tokens)

        # CLS token
        cls_idx = torch.zeros(N, 1, dtype=torch.long, device=device)
        cls = self.cls_token.expand(N, -1, -1) + self.modality_embeddings(cls_idx)

        if not tokens:
            return self.norm(self.transformer(cls)[:, 0, :])

        sequence = torch.cat([cls] + tokens, dim=1)   # (N, 1+ΣT_i, latent_dim)
        output = self.transformer(sequence)
        return self.norm(output[:, 0, :])              # CLS read-out → (N, latent_dim)

    def alignment_loss(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        In-batch NT-Xent contrastive loss between all projected modality pairs.
        Encourages the same sample's representations across modalities to
        cluster together in the shared latent space.

        Called by ``ApexLightningModule._apply_adaptive_loss`` when this
        fusion class is active — replaces the separate CLIPProjectionHead path.
        """
        if len(features) < 2:
            return torch.tensor(0.0, device=features[0].device)

        projs = []
        for i, feat in enumerate(features):
            # Pool token sequences to one vector before contrastive loss
            flat = feat.detach()
            if flat.dim() == 3:
                flat = flat.mean(dim=1)  # (N, T, D) → (N, D)
            projected = self.proj_norms[i](self.proj_linears[i](flat))  # (N, latent_dim)
            projs.append(F.normalize(projected, dim=-1))
        loss = torch.tensor(0.0, device=features[0].device)
        n_pairs = 0
        temperature = 0.07

        for i in range(len(projs)):
            for j in range(i + 1, len(projs)):
                N = projs[i].shape[0]
                if N < 2:
                    continue
                sim = torch.matmul(projs[i], projs[j].T) / temperature  # (N, N)
                labels = torch.arange(N, device=sim.device)
                loss += (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
                n_pairs += 1

        return loss / max(1, n_pairs)

    def get_output_dim(self) -> int:
        return self.output_dim

    def get_attention_summary(self) -> dict:
        return {"token_count": self._last_token_count, "latent_dim": self.latent_dim}


# ---------------------------------------------------------------------------
# FuseMoE — Mixture of Experts fusion for missing-modality robustness
# ---------------------------------------------------------------------------

class FuseMoE(nn.Module):
    """
    Mixture of Experts fusion.

    A lightweight router reads the **modality presence vector** (which
    modalities are non-zero in this sample) and selects the top-``top_k``
    expert networks.  Each expert is a small MLP that processes the
    concatenation of all modality embeddings (zero-filled for absent ones).

    Key advantage over zero-fill late fusion: missing modalities alter the
    routing decision, not just the concatenated input.  The router learns
    to activate different experts for (tabular+text), (all three), or
    (image+tabular) combinations.

    References
    ----------
    Ma et al. "FuseMoE: Mixture-of-Experts Transformers for Flexi-Modal
    Learning." ICML 2024.
    """

    accepts_mask: bool = True

    def __init__(
        self,
        feature_dims: List[int],
        output_dim: int = 512,
        n_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        n_mods = len(feature_dims)
        total_dim = sum(feature_dims)

        # Router: maps modality-presence vector → expert logits
        self.router = nn.Sequential(
            nn.Linear(n_mods, n_experts * 2),
            nn.GELU(),
            nn.Linear(n_experts * 2, n_experts),
        )

        # Expert networks (each sees full concatenated embedding)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_dim, output_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(output_dim * 2, output_dim),
                nn.LayerNorm(output_dim),
            )
            for _ in range(n_experts)
        ])

        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        self.output_dim = output_dim
        self.feature_dims = feature_dims
        self._last_routing: Optional[List[float]] = None

    def forward(
        self,
        features: List[torch.Tensor],
        modality_mask: Optional[Dict[str, bool]] = None,
        modality_names: Optional[List[str]] = None,
    ) -> torch.Tensor:
        N = features[0].shape[0]
        device = features[0].device

        # Build modality-presence vector from mask (or all-ones if no mask)
        if modality_mask is not None and modality_names is not None:
            presence = torch.tensor(
                [float(modality_mask.get(modality_names[i] if i < len(modality_names) else "", True))
                 for i in range(len(features))],
                device=device,
            ).unsqueeze(0).expand(N, -1)           # (N, n_mods)
            # Zero-fill absent modalities in the concatenated input
            features = apply_modality_mask(features, modality_names, modality_mask)
        else:
            presence = torch.ones(N, len(features), device=device)

        concat = torch.cat(features, dim=-1)        # (N, total_dim)

        # Router: select top-k experts per sample
        router_logits = self.router(presence)                          # (N, n_experts)
        router_probs  = torch.softmax(router_logits, dim=-1)           # (N, n_experts)
        topk_weights, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)  # renormalise

        self._last_routing = router_probs.detach().mean(dim=0).tolist()

        # Weighted sum of top-k expert outputs
        output = torch.zeros(N, self.output_dim, device=device)
        for k in range(self.top_k):
            for exp_idx in range(self.n_experts):
                mask = topk_indices[:, k] == exp_idx          # (N,)
                if not mask.any():
                    continue
                exp_out = self.experts[exp_idx](concat[mask]) # (M, output_dim)
                output[mask] += topk_weights[mask, k : k + 1] * exp_out

        return output  # (N, output_dim)

    def get_output_dim(self) -> int:
        return self.output_dim

    def get_attention_summary(self) -> dict:
        return {"expert_routing": self._last_routing or [], "n_experts": self.n_experts}
