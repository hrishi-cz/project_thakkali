"""
Advanced Model Selector – Optuna HPO search spaces + PDF heuristic tables.

Epoch bounds (PDF matrix)
-------------------------
Dataset size   | Image    | Text     | Tabular
<5K            | 45-50    | 40-45    | 30-35
5K-50K         | 18-25    | 15-20    | 12-15
50K-500K       | 12-18    | 10-15    | 8-12
>500K          | 10-15    | 8-12     | 6-10

Batch size rules (PDF)
----------------------
Image   : GPU <4 GB → 4 | <8 GB → 8 | <12 GB → 16 | ≥12 GB → 32
Text    : avg_tokens > 512 → 8 | ≤512 → GPU-scaled (same tiers as image, ×2)
Tabular : min(256, dataset_size // 100)

Returns Optuna-compatible search space dicts for every tuneable parameter.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optuna space type aliases
# ---------------------------------------------------------------------------

# int  : {"type": "int",         "low": lo,  "high": hi}
# float: {"type": "float",       "low": lo,  "high": hi, "log": bool}
# cat  : {"type": "categorical", "choices": [...]}
OptunaDist = Dict[str, Any]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SelectionResult:
    """
    Model selection outcome combining static encoder choices and Optuna HPO
    search spaces that Phase 5 will consume via ``optuna.trial.suggest_*``.

    Attributes
    ----------
    image_encoder   : tier key into IMAGE_ENCODERS (``None`` when not needed).
    text_encoder    : tier key into TEXT_ENCODERS.
    tabular_encoder : tier key into TABULAR_ENCODERS.
    fusion_strategy : ``"attention"`` | ``"concatenation"`` (static choice).
    batch_size      : Fixed value derived from PDF heuristics (not tuned).
    hpo_space       : Per-parameter Optuna search space specs.  Keys:
                        ``epochs``, ``learning_rate``, ``dropout``,
                        ``weight_decay``, and optionally ``fusion_strategy``
                        when more than one modality is active.
    rationale       : Human-readable selection rationale per component.
    hardware_info   : GPU/CPU environment snapshot.
    """

    image_encoder: Optional[str]
    text_encoder: Optional[str]
    tabular_encoder: Optional[str]
    fusion_strategy: str
    batch_size: int
    hpo_space: Dict[str, OptunaDist] = field(default_factory=dict)
    rationale: Dict[str, str] = field(default_factory=dict)
    hardware_info: Dict[str, Any] = field(default_factory=dict)
    meta_context: List[Dict[str, Any]] = field(default_factory=list)
    eligible_modalities: List[str] = field(default_factory=list)
    excluded_modalities: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Encoder catalogue (static metadata only)
# ---------------------------------------------------------------------------

IMAGE_ENCODERS: Dict[str, Dict[str, Any]] = {
    "lightweight": {"name": "MobileNetV3",    "output_dim": 512,  "params": "2.5M"},
    "balanced":    {"name": "ResNet50",        "output_dim": 512,  "params": "25M"},
    "sota":        {"name": "ConvNeXt-Tiny",   "output_dim": 512,  "params": "28.6M"},
}

TEXT_ENCODERS: Dict[str, Dict[str, Any]] = {
    "fast":     {"name": "MiniLM-L6-v2", "output_dim": 768,  "params": "22.7M"},
    "balanced": {"name": "BERT-base",    "output_dim": 768,  "params": "110M"},
    "sota":     {"name": "DeBERTa-v3",   "output_dim": 768,  "params": "183.8M"},
}

TABULAR_ENCODERS: Dict[str, Dict[str, Any]] = {
    "simple":       {"name": "MLP",  "output_dim": 16},
    "interpretable":{"name": "GRN",  "output_dim": 16},
    # NOTE: FT-Transformer is not implemented in this codebase yet.
    # Keep "sota" key mapped to the best available implemented encoder.
    "sota":         {"name": "GRN",  "output_dim": 16},
}


_ENCODER_LATENCY_MS: Dict[str, float] = {
    "ConvNeXt-Tiny": 18.0,
    "ResNet50": 9.0,
    "MobileNetV3": 4.5,
    "DeBERTa-v3": 85.0,
    "BERT-base": 42.0,
    "MiniLM-L6-v2": 12.0,
    "GRN": 0.8,
    "MLP": 0.4,
}

_ENCODER_VRAM_MB: Dict[str, int] = {
    "ConvNeXt-Tiny": 880,
    "ResNet50": 420,
    "MobileNetV3": 55,
    "DeBERTa-v3": 1600,
    "BERT-base": 440,
    "MiniLM-L6-v2": 90,
    "GRN": 12,
    "MLP": 6,
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AdvancedModelSelector:
    """
    Stateless model selector that maps hardware + data context to encoder
    choices and Optuna HPO search bounds.

    Public API
    ----------
    select_models(problem_type, modalities, dataset_size, avg_tokens, gpu_memory_gb)
        → SelectionResult  (primary method, consumed by orchestrator Phase 4)

    recommend_models(problem_type, modalities, dataset_size, avg_tokens)
        → List[Dict]       (legacy/API shim, consumed by /select-model endpoint)
    """

    # ------------------------------------------------------------------
    # Primary selection method
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        from automl.meta_learning import MetaLearningStore
        self._meta_store = MetaLearningStore()

    def select_models(
        self,
        problem_type: str,
        modalities: List[str],
        dataset_size: int,
        avg_tokens: int = 128,
        gpu_memory_gb: Optional[float] = None,
        dataset_meta: Optional[Dict[str, Any]] = None,
        latency_budget_ms: Optional[float] = None,
        memory_budget_mb: Optional[float] = None,
        predictability_scores: Optional[Dict[str, float]] = None,
    ) -> SelectionResult:
        """
        Select encoder tier, fixed batch size, and Optuna HPO search bounds.

        Parameters
        ----------
        problem_type  : e.g. ``"classification_binary"``.
        modalities    : Subset of ``["image", "text", "tabular"]``.
        dataset_size  : Total number of training samples.
        avg_tokens    : Mean number of tokens per text sample (default 128).
                        Used exclusively for the text batch-size rule.
        gpu_memory_gb : GPU RAM in GB.  Auto-detected when ``None``.

        Returns
        -------
        SelectionResult
        """
        input_modalities = [str(m) for m in (modalities or [])]
        predictability_map = dict(predictability_scores or {})
        excluded_modalities: Dict[str, str] = {}
        eligible_modalities: List[str] = []

        for modality in input_modalities:
            score = self._resolve_modality_predictability(modality, predictability_map)
            if score is not None and score < 0.25:
                excluded_modalities[modality] = f"predictability {score:.3f} < 0.250"
                continue
            eligible_modalities.append(modality)

        if not eligible_modalities and input_modalities:
            ranked = sorted(
                input_modalities,
                key=lambda m: self._resolve_modality_predictability(m, predictability_map)
                if self._resolve_modality_predictability(m, predictability_map) is not None
                else -1.0,
                reverse=True,
            )
            keep_modality = ranked[0]
            eligible_modalities = [keep_modality]
            excluded_modalities.pop(keep_modality, None)

        effective_modalities = eligible_modalities or input_modalities

        gpu_mem: float = (
            gpu_memory_gb if gpu_memory_gb is not None
            else self._probe_gpu_memory()
        )
        hardware_info = self._build_hardware_info(gpu_mem)

        # ── encoder tier selection ──────────────────────────────────────
        image_tier, img_rationale = (
            self._select_image_tier(dataset_size, gpu_mem)
            if "image" in effective_modalities
            else (None, "")
        )
        text_tier, txt_rationale = (
            self._select_text_tier(problem_type, dataset_size, gpu_mem, avg_tokens)
            if "text" in effective_modalities
            else (None, "")
        )
        tabular_tier, tab_rationale = (
            self._select_tabular_tier(dataset_size, gpu_mem)
            if "tabular" in effective_modalities
            else (None, "")
        )

        if latency_budget_ms is not None or memory_budget_mb is not None:
            image_tier = self._apply_resource_constraints(
                selected_tier=image_tier,
                catalogue=IMAGE_ENCODERS,
                fallback_chain=["sota", "balanced", "lightweight"],
                modality="image",
                latency_budget_ms=latency_budget_ms,
                memory_budget_mb=memory_budget_mb,
            )
            text_tier = self._apply_resource_constraints(
                selected_tier=text_tier,
                catalogue=TEXT_ENCODERS,
                fallback_chain=["sota", "balanced", "fast"],
                modality="text",
                latency_budget_ms=latency_budget_ms,
                memory_budget_mb=memory_budget_mb,
            )
            tabular_tier = self._apply_resource_constraints(
                selected_tier=tabular_tier,
                catalogue=TABULAR_ENCODERS,
                fallback_chain=["sota", "interpretable", "simple"],
                modality="tabular",
                latency_budget_ms=latency_budget_ms,
                memory_budget_mb=memory_budget_mb,
            )

        # ── batch size (PDF – not tuned) ────────────────────────────────
        batch_size = self._pdf_batch_size(
            effective_modalities,
            gpu_mem,
            dataset_size,
            avg_tokens,
        )

        # ── meta-learning context (optional bias; never overrides contracts) ─
        dataset_meta_norm = self._build_dataset_meta(
            problem_type=problem_type,
            modalities=effective_modalities,
            dataset_size=dataset_size,
            dataset_meta=dataset_meta,
        )
        similar_context = self._meta_store.get_similar_context(dataset_meta_norm)
        _current_modality_set = {str(m) for m in effective_modalities}
        fusion_relevant_context = [
            rec for rec in (similar_context or [])
            if {
                str(m)
                for m in (
                    (rec.get("dataset_meta", {}) or {}).get("modalities", [])
                    if isinstance(rec, dict)
                    else []
                )
            } == _current_modality_set
        ]
        non_fusion_meta_bias = self._derive_meta_bias(similar_context)
        meta_bias = self._derive_meta_bias(fusion_relevant_context)

        # ── fusion strategy (static – also offered as HPO when multimodal) ─
        from models.fusion import select_fusion_strategy

        fusion = self._normalize_fusion_key(
            select_fusion_strategy({"global_modalities": list(effective_modalities)})
        )
        if fusion in {"graph", "uncertainty", "uncertainty_graph"} and gpu_mem < 4:
            fusion = "concatenation"
            logger.info(
                "AdvancedModelSelector: GPU %.1fGB too small for %s, downgraded to concatenation",
                gpu_mem,
                self._normalize_fusion_key(
                    select_fusion_strategy({"global_modalities": list(effective_modalities)})
                ),
            )

        # ULA is the primary fusion for text+image: cross-modal Transformer attention
        # lets image patches and text tokens attend to each other directly.
        # Requires ≥4 GB GPU (Transformer fusion is compute-heavy).
        if (
            "text" in effective_modalities
            and "image" in effective_modalities
            and gpu_mem >= 4
        ):
            fusion = "ula"
            logger.info(
                "AdvancedModelSelector: text+image detected — defaulting to ULA "
                "(cross-modal Transformer patch/token attention)"
            )

        # Only apply meta_bias fusion hints when the stored context actually has
        # the SAME modalities. Tabular-only historical runs should never override
        # ULA for text+image — their fusion preferences are semantically irrelevant.
        _meta_is_relevant = bool(fusion_relevant_context)

        preferred_fusion = meta_bias.get("preferred_fusion") if _meta_is_relevant else None
        preferred_fusion = self._normalize_fusion_key(preferred_fusion)
        _ula_active = (
            "text" in effective_modalities
            and "image" in effective_modalities
            and gpu_mem >= 4
        )
        # Bug 4 fix: meta_bias must not downgrade ULA for text+image.
        # Early runs that used concatenation would permanently bias preferred_fusion
        # to concatenation, overwriting the architecturally-correct ULA choice.
        # Meta-bias may only UPGRADE (e.g. None → ula) or maintain ULA, never replace it.
        if _ula_active and preferred_fusion not in (None, "ula"):
            logger.info(
                "AdvancedModelSelector: meta_bias preferred_fusion=%s suppressed "
                "(text+image always uses ULA — meta-bias cannot downgrade)",
                preferred_fusion,
            )
            preferred_fusion = None
        if preferred_fusion in {
            "ula",
            "attention",
            "concatenation",
            "graph",
            "uncertainty",
            "uncertainty_graph",
        }:
            fusion = preferred_fusion
            logger.info(
                "AdvancedModelSelector: meta_bias preferred_fusion=%s applied (exact modality context)",
                preferred_fusion,
            )

        # ── Optuna HPO search spaces ────────────────────────────────────
        hpo_space = self._build_hpo_space(
            modalities=effective_modalities,
            dataset_size=dataset_size,
            problem_type=problem_type,
            gpu_mem=gpu_mem,
            fusion_static=fusion,
            lr_bias=non_fusion_meta_bias.get("lr_range"),
            # Only pass fusion_priority from meta when context is relevant
            fusion_priority=meta_bias.get("fusion_priority") if _meta_is_relevant else None,
        )

        rationale: Dict[str, str] = {}
        if img_rationale:
            rationale["image_encoder"] = img_rationale
        if txt_rationale:
            rationale["text_encoder"] = txt_rationale
        if tab_rationale:
            rationale["tabular_encoder"] = tab_rationale
        rationale["batch_size"] = (
            f"dataset_size={dataset_size}, gpu_mem={gpu_mem:.1f}GB, "
            f"avg_tokens={avg_tokens}"
        )
        if excluded_modalities:
            rationale["modality_gating"] = (
                "Excluded weak modalities: "
                + ", ".join(
                    f"{name} ({reason})" for name, reason in excluded_modalities.items()
                )
            )
        if similar_context:
            rationale["meta_learning"] = (
                f"Applied non-fusion priors from {len(similar_context)} similar experiments; "
                f"fusion priors used from {len(fusion_relevant_context)} exact-modality matches"
            )

        logger.info(
            "AdvancedModelSelector: image=%s  text=%s  tabular=%s  "
            "batch=%d  gpu=%.1f GB",
            image_tier, text_tier, tabular_tier, batch_size, gpu_mem,
        )
        return SelectionResult(
            image_encoder=image_tier,
            text_encoder=text_tier,
            tabular_encoder=tabular_tier,
            fusion_strategy=fusion,
            batch_size=batch_size,
            hpo_space=hpo_space,
            rationale=rationale,
            hardware_info=hardware_info,
            meta_context=similar_context,
            eligible_modalities=list(effective_modalities),
            excluded_modalities=excluded_modalities,
        )

    def record_experiment(
        self,
        dataset_meta: Dict[str, Any],
        best_params: Dict[str, Any],
        fusion_strategy: str,
        loss_weights: Optional[Dict[str, float]],
        performance: float,
    ) -> None:
        """Persist a completed training outcome for future search priors."""
        record = {
            "dataset_meta": {
                "num_rows": int(dataset_meta.get("num_rows", 0)),
                "num_cols": int(dataset_meta.get("num_cols", 0)),
                "modalities": list(dataset_meta.get("modalities", [])),
                "target_type": str(dataset_meta.get("target_type", "classification")),
            },
            "best_params": dict(best_params or {}),
            "fusion_strategy": str(fusion_strategy),
            "loss_weights": dict(loss_weights or {}),
            "performance": float(performance),
        }
        self._meta_store.add_experiment(record)

    @staticmethod
    def _target_type(problem_type: str) -> str:
        return "regression" if "regression" in str(problem_type).lower() else "classification"

    def _build_dataset_meta(
        self,
        problem_type: str,
        modalities: List[str],
        dataset_size: int,
        dataset_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        meta = dict(dataset_meta or {})
        meta.setdefault("num_rows", int(dataset_size))
        meta.setdefault("num_cols", int(meta.get("num_cols", len(modalities) * 8 or 1)))
        meta.setdefault("modalities", list(modalities))
        meta.setdefault("target_type", self._target_type(problem_type))
        return meta

    @staticmethod
    def _derive_meta_bias(similar_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not similar_context:
            return {}

        fusion_votes = Counter()
        lr_values: List[float] = []
        for rec in similar_context:
            fusion = rec.get("fusion_strategy")
            if isinstance(fusion, str):
                fusion_votes[fusion] += 1
            lr = rec.get("best_params", {}).get("learning_rate")
            try:
                if lr is not None:
                    lr_values.append(float(lr))
            except Exception:
                continue

        bias: Dict[str, Any] = {}
        if fusion_votes:
            ordered_fusions = [k for k, _ in fusion_votes.most_common()]
            bias["fusion_priority"] = ordered_fusions
            bias["preferred_fusion"] = ordered_fusions[0]

        if lr_values:
            lr_mean = sum(lr_values) / max(1, len(lr_values))
            bias["lr_range"] = (max(1e-6, lr_mean * 0.5), min(1e-1, lr_mean * 2.0))

        return bias

    # ------------------------------------------------------------------
    # API / frontend shim
    # ------------------------------------------------------------------

    def recommend_models(
        self,
        problem_type: str,
        modalities: List[str],
        dataset_size: int = 10_000,
        avg_tokens: int = 128,
        tabular_X: Optional[Any] = None,
        tabular_y: Optional[Any] = None,
        latency_budget_ms: Optional[float] = None,
        memory_budget_mb: Optional[float] = None,
        predictability_scores: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return a ranked list of model recommendation dicts suitable for the
        Streamlit frontend JSON contract.

        The first entry is always the primary (highest-quality) recommendation
        derived from ``select_models()``.  The remaining entries enumerate
        cheaper alternatives for each tier.
        """
        primary: SelectionResult = self.select_models(
            problem_type=problem_type,
            modalities=modalities,
            dataset_size=dataset_size,
            avg_tokens=avg_tokens,
            latency_budget_ms=latency_budget_ms,
            memory_budget_mb=memory_budget_mb,
            predictability_scores=predictability_scores,
        )

        tab_probe_scores: Dict[str, Dict[str, Any]] = {}
        probe_scores: Dict[str, Dict[str, Any]] = {}
        ranked_candidates: Dict[str, List[Dict[str, Any]]] = {}
        selection_metadata: Dict[str, Any] = {}

        def _encoder_name(catalogue: Dict, tier: Optional[str]) -> Optional[str]:
            return catalogue[tier]["name"] if tier and tier in catalogue else None

        primary_rec: Dict[str, Any] = {
            "name": self._build_model_name(
                primary.image_encoder,
                primary.text_encoder,
                primary.tabular_encoder,
                fusion=primary.fusion_strategy,
            ),
            "image_encoder":   _encoder_name(IMAGE_ENCODERS, primary.image_encoder),
            "text_encoder":    _encoder_name(TEXT_ENCODERS,  primary.text_encoder),
            "tabular_encoder": _encoder_name(TABULAR_ENCODERS, primary.tabular_encoder),
            "fusion_strategy": primary.fusion_strategy,
            "batch_size":      primary.batch_size,
            "hpo_space":       primary.hpo_space,
            "rationale":       primary.rationale,
            "hardware_info":   primary.hardware_info,
            "meta_context":    primary.meta_context,
            "eligible_modalities": list(primary.eligible_modalities),
            "excluded_modalities": dict(primary.excluded_modalities),
            "warm_start_params": (
                dict(primary.meta_context[0].get("best_params", {}))
                if primary.meta_context and isinstance(primary.meta_context[0], dict)
                else {}
            ),
            "selection_contract_version": "model_selection.v2",
            "probe_scores": {},
            "selection_metadata": {},
            "ranked_candidates": {},
            "tier":            "primary",
        }

        # Optional data-driven probe for tabular encoder preference.
        if (
            tabular_X is not None
            and tabular_y is not None
            and "tabular" in list(primary.eligible_modalities or modalities)
        ):
            try:
                from automl.candidate_selector import (
                    CandidateSelector,
                    TABULAR_CANDIDATE_POOL,
                )

                probe_selector = CandidateSelector()
                probe_X = tabular_X.toarray() if hasattr(tabular_X, "toarray") else np.asarray(tabular_X)
                probe_y = np.asarray(tabular_y)

                if probe_y.ndim > 1 and probe_y.shape[1] > 1:
                    probe_y = np.argmax(probe_y, axis=1)
                else:
                    probe_y = probe_y.ravel()
                    if probe_y.dtype.kind in ("U", "S", "O"):
                        probe_y = pd.factorize(probe_y)[0]
                    else:
                        try:
                            probe_y = probe_y.astype(int)
                        except Exception:
                            probe_y = pd.factorize(probe_y)[0]

                tab_probe_scores = probe_selector.quick_probe_tabular(
                    list(TABULAR_CANDIDATE_POOL),
                    probe_X,
                    probe_y,
                    problem_type,
                )

                if tab_probe_scores:
                    probe_scores["tabular"] = dict(tab_probe_scores)

                score_map = {
                    name: float(info.get("val_score", 0.0) or 0.0)
                    for name, info in tab_probe_scores.items()
                    if isinstance(info, dict)
                }
                top_model = max(score_map, key=score_map.get) if score_map else None

                if tab_probe_scores:
                    ranked_candidates["tabular"] = sorted(
                        [
                            {
                                "name": model_name,
                                "val_score": float(details.get("val_score", 0.0) or 0.0),
                                "latency_ms": float(details.get("latency_ms", 0.0) or 0.0),
                                "uncertainty": float(details.get("uncertainty", 0.0) or 0.0),
                                "confidence": details.get("confidence"),
                            }
                            for model_name, details in tab_probe_scores.items()
                            if isinstance(details, dict)
                        ],
                        key=lambda row: row.get("val_score", 0.0),
                        reverse=True,
                    )

                top_score = float(score_map[top_model]) if top_model else None
                selection_metadata = {
                    "probe_method": "tabular_3fold_cv",
                    "top_probe_model": top_model,
                    "top_probe_score": top_score,
                    "probe_scores": dict(tab_probe_scores),
                }

                if top_model in {"mlp", "grn"}:
                    preferred_tier = "simple" if top_model == "mlp" else "interpretable"
                    primary_rec["tabular_encoder"] = TABULAR_ENCODERS[preferred_tier]["name"]
                    primary_rec["name"] = self._build_model_name(
                        primary.image_encoder,
                        primary.text_encoder,
                        preferred_tier,
                        fusion=primary.fusion_strategy,
                    )

                if top_model is not None:
                    primary_rec["tabular_probe_top_model"] = str(top_model)
                    primary_rec["quick_probe_score"] = float(score_map[top_model])
                    primary_rec["probe_score"] = float(score_map[top_model])

                primary_rec["tabular_probe_scores"] = dict(tab_probe_scores)
                primary_rec.setdefault("rationale", {})["tabular_probe"] = (
                    "Tabular quick probe executed on cached sample"
                )
            except Exception as probe_exc:
                logger.warning("AdvancedModelSelector.recommend_models: tabular probe failed: %s", probe_exc)

        primary_rec["probe_scores"] = dict(probe_scores)
        primary_rec["selection_metadata"] = dict(selection_metadata)
        primary_rec["ranked_candidates"] = dict(ranked_candidates)

        # Lightweight fallback alternative
        alt_hpo = dict(primary.hpo_space)  # same search bounds
        fallback_modalities = set(primary.eligible_modalities or modalities)
        # Bug 7 fix: fallback fusion must be modality-aware, not hardcoded concatenation.
        # For text+image, ULA is always superior. Concatenation only valid as last resort
        # for single-modality or GPU-constrained scenarios.
        _fb_has_text_image = "text" in fallback_modalities and "image" in fallback_modalities
        _fallback_fusion = "ula" if _fb_has_text_image else "concatenation"
        alt_rec: Dict[str, Any] = {
            "name": "Lightweight Fallback",
            "image_encoder":   IMAGE_ENCODERS["lightweight"]["name"] if "image" in fallback_modalities else None,
            "text_encoder":    TEXT_ENCODERS["fast"]["name"]         if "text"  in fallback_modalities else None,
            "tabular_encoder": TABULAR_ENCODERS["simple"]["name"]    if "tabular" in fallback_modalities else None,
            "fusion_strategy": _fallback_fusion,
            "batch_size":      min(primary.batch_size, 8),
            "hpo_space":       alt_hpo,
            "rationale":       {"general": "Lightweight fallback for memory-constrained environments"},
            "hardware_info":   primary.hardware_info,
            "eligible_modalities": list(primary.eligible_modalities),
            "excluded_modalities": dict(primary.excluded_modalities),
            "selection_contract_version": "model_selection.v2",
            "probe_scores": {},
            "selection_metadata": {},
            "ranked_candidates": {},
            "tier":            "fallback",
        }

        return [primary_rec, alt_rec]

    # ------------------------------------------------------------------
    # PDF epoch bounds
    # ------------------------------------------------------------------

    @staticmethod
    def _pdf_epoch_bounds(
        dataset_size: int,
        modalities: List[str],
    ) -> Tuple[int, int]:
        """
        Return ``(low, high)`` epoch bounds from the PDF matrix.

        Priority: image > text > tabular when multiple modalities are active.
        """
        has_image   = "image"   in modalities
        has_text    = "text"    in modalities
        # has_tabular = "tabular" in modalities  # lowest priority

        if dataset_size < 5_000:
            if has_image:   return 45, 50
            if has_text:    return 40, 45
            return 30, 35
        elif dataset_size < 50_000:
            if has_image:   return 18, 25
            if has_text:    return 15, 20
            return 12, 15
        elif dataset_size < 500_000:
            if has_image:   return 12, 18
            if has_text:    return 10, 15
            return 8, 12
        else:
            if has_image:   return 10, 15
            if has_text:    return 8,  12
            return 6, 10

    # ------------------------------------------------------------------
    # PDF batch-size rules
    # ------------------------------------------------------------------

    @staticmethod
    def _pdf_batch_size(
        modalities: List[str],
        gpu_mem: float,
        dataset_size: int,
        avg_tokens: int,
    ) -> int:
        """
        Compute the fixed batch size from PDF heuristic rules.

        Image  : <4 GB→4 | <8 GB→8 | <12 GB→16 | ≥12 GB→32
        Text   : avg_tokens>512→8 | ≤512 scaled by GPU tier (×2 vs image)
        Tabular: min(256, dataset_size // 100)
        """
        if "image" in modalities:
            if gpu_mem < 4:   return 4
            if gpu_mem < 8:   return 8
            if gpu_mem < 12:  return 16
            return 32

        if "text" in modalities:
            if avg_tokens > 512:
                return 8
            # ≤512: same GPU tiers but doubled (as per PDF)
            if gpu_mem < 4:   return 8
            if gpu_mem < 8:   return 16
            if gpu_mem < 12:  return 32
            return 64

        # Pure tabular
        return min(256, max(1, dataset_size // 100))

    # ------------------------------------------------------------------
    # Encoder tier selection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _select_image_tier(dataset_size: int, gpu_mem: float) -> Tuple[str, str]:
        if dataset_size < 1_000:
            return "lightweight", "Dataset <1k: MobileNetV3 for efficiency"
        if dataset_size < 10_000:
            if gpu_mem >= 8:
                return "balanced", f"Dataset 1-10k + GPU {gpu_mem:.1f}GB: ResNet50"
            return "lightweight", f"Dataset 1-10k but GPU {gpu_mem:.1f}GB: MobileNetV3"
        if gpu_mem >= 12:
            return "sota",    f"Dataset >10k + GPU {gpu_mem:.1f}GB: ViT-Base"
        if gpu_mem >= 8:
            return "balanced", f"Dataset >10k + GPU {gpu_mem:.1f}GB: ResNet50"
        return "lightweight", f"Dataset >10k but GPU {gpu_mem:.1f}GB: MobileNetV3"

    @staticmethod
    def _select_text_tier(
        problem_type: str,
        dataset_size: int,
        gpu_mem: float,
        avg_tokens: int = 128,
    ) -> Tuple[str, str]:
        if avg_tokens > 1024:
            return "fast", f"Avg tokens {avg_tokens}: lightweight text tier to respect sequence budget"
        if avg_tokens > 512:
            return "balanced", f"Avg tokens {avg_tokens}: avoid SOTA text tier with short token limit"
        if "binary" in problem_type or dataset_size < 5_000:
            return "fast", "Binary/small dataset: DistilBERT for speed"
        if "multiclass" in problem_type and gpu_mem >= 8 and dataset_size > 10_000:
            return "sota", f"Multiclass + GPU {gpu_mem:.1f}GB + large dataset: RoBERTa-large"
        return "balanced", "Default multiclass/regression: BERT-base"

    @staticmethod
    def _select_tabular_tier(dataset_size: int, gpu_mem: float) -> Tuple[str, str]:
        if dataset_size < 5_000 and gpu_mem < 8:
            return "simple", "Small dataset + limited GPU: MLP"
        if gpu_mem >= 12:
            return "sota",    f"GPU {gpu_mem:.1f}GB: GRN"
        if gpu_mem >= 8:
            return "interpretable", f"GPU {gpu_mem:.1f}GB: GRN"
        return "simple", f"GPU {gpu_mem:.1f}GB: MLP"

    # ------------------------------------------------------------------
    # HPO space builder
    # ------------------------------------------------------------------

    def _build_hpo_space(
        self,
        modalities: List[str],
        dataset_size: int,
        problem_type: str,
        gpu_mem: float,
        fusion_static: str,
        lr_bias: Optional[Tuple[float, float]] = None,
        fusion_priority: Optional[List[str]] = None,
    ) -> Dict[str, OptunaDist]:
        """
        Build a fully-specified Optuna search space based on data context.

        Each entry is serialisable to JSON and can be consumed by Phase 5 as::

            value = trial.suggest_int(name, low, high)        # type == "int"
            value = trial.suggest_float(name, low, high, log=log)  # type == "float"
            value = trial.suggest_categorical(name, choices)  # type == "categorical"
        """
        epoch_lo, epoch_hi = self._pdf_epoch_bounds(dataset_size, modalities)

        # Learning rate: wider range for large datasets; log-uniform always
        lr_low  = 1e-5 if dataset_size > 50_000 else 5e-5
        lr_high = 1e-3 if dataset_size > 50_000 else 1e-2
        if lr_bias:
            bias_low, bias_high = lr_bias
            lr_low = max(lr_low, min(bias_low, bias_high))
            lr_high = min(lr_high, max(bias_low, bias_high))
            if lr_low >= lr_high:
                lr_low, lr_high = (1e-5, 1e-3) if dataset_size > 50_000 else (5e-5, 1e-2)

        space: Dict[str, OptunaDist] = {
            "epochs": {
                "type": "int",
                "low":  epoch_lo,
                "high": epoch_hi,
            },
            "learning_rate": {
                "type": "float",
                "low":  lr_low,
                "high": lr_high,
                "log":  True,
            },
            "dropout": {
                "type": "float",
                "low":  0.0,
                "high": 0.5,
            },
            "weight_decay": {
                "type": "float",
                "low":  1e-6,
                "high": 1e-2,
                "log":  True,
            },
        }

        # Alignment regularization is only meaningful when 2+ modalities exist.
        if len(modalities) >= 2:
            space["alignment_weight"] = {
                "type": "float",
                "low": 0.0,
                "high": 0.05,
                "log": False,
            }

        # Fusion strategy HPO — choices filtered by which modalities are active.
        # Only offer strategies that are architecturally valid for the modality set:
        #   text+image       → ULA (cross-modal Transformer), attention, gated, concatenation
        #   tabular+text     → attention, graph, concatenation  (graph = entity relations)
        #   tabular+image    → uncertainty, attention, concatenation (uncertainty = quality weights)
        #   3+ modalities    → ULA, uncertainty_graph, complementarity, concatenation
        # "graph" on text+image wastes a trial: no tabular entity relations to exploit.
        # "uncertainty" on text+image is also weak: both modalities have similar quality variance.
        if len(modalities) > 1:
            _has_text  = "text"    in modalities
            _has_image = "image"   in modalities
            _has_tab   = "tabular" in modalities
            _n_mods    = len(modalities)

            if gpu_mem >= 4:
                if _n_mods >= 3:
                    # Full multimodal — all strategies valid
                    choices = ["ula", "uncertainty_graph", "complementarity", "concatenation", "attention"]
                elif _has_text and _has_image:
                    # Text+image only: ULA is best; graph/uncertainty waste trials
                    choices = ["ula", "attention", "gated", "concatenation"]
                elif _has_tab and _has_text:
                    # Tabular+text: graph captures entity relations; attention is default
                    choices = ["attention", "graph", "concatenation"]
                elif _has_tab and _has_image:
                    # Tabular+image: uncertainty weights image quality; graph less useful
                    choices = ["uncertainty", "attention", "concatenation"]
                else:
                    choices = ["concatenation", "attention"]
            else:
                choices = ["concatenation", "attention"]

            if fusion_priority:
                normalized_priority = [self._normalize_fusion_key(f) for f in fusion_priority]
                ordered = [f for f in normalized_priority if f in choices]
                if ordered:
                    choices = ordered + [f for f in choices if f not in ordered]
            space["fusion_strategy"] = {
                "type":    "categorical",
                "choices": choices,
            }

        return space

    # ------------------------------------------------------------------
    # Hardware probing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _probe_gpu_memory() -> float:
        """Return total GPU memory in GB; 0.0 on CPU-only systems."""
        if not torch.cuda.is_available():
            return 0.0
        try:
            return round(
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2
            )
        except Exception:
            return 0.0

    @staticmethod
    def _build_hardware_info(gpu_mem: float) -> Dict[str, Any]:
        return {
            "gpu_available":  torch.cuda.is_available(),
            "gpu_memory_gb":  gpu_mem,
            "device":         "GPU" if torch.cuda.is_available() else "CPU",
            "cuda_device":    (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else "None"
            ),
        }

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_model_name(
        image_tier: Optional[str],
        text_tier:  Optional[str],
        tabular_tier: Optional[str],
        fusion: Optional[str] = None,
    ) -> str:
        parts: List[str] = []
        if image_tier   and image_tier   in IMAGE_ENCODERS:
            parts.append(IMAGE_ENCODERS[image_tier]["name"])
        if text_tier    and text_tier    in TEXT_ENCODERS:
            parts.append(TEXT_ENCODERS[text_tier]["name"])
        if tabular_tier and tabular_tier in TABULAR_ENCODERS:
            parts.append(TABULAR_ENCODERS[tabular_tier]["name"])
        base = " + ".join(parts) if parts else "Unsupervised"
        # Include fusion when non-default so the name reflects the architecture
        if fusion and fusion not in ("concatenation", "auto", ""):
            _label = {"ula": "ULA", "attention": "Attention", "graph": "RGAT",
                      "uncertainty": "Uncertainty", "uncertainty_graph": "UncertaintyGraph"}.get(fusion, fusion)
            return f"{base} [{_label}]"
        return base

    @staticmethod
    def _normalize_fusion_key(fusion: Optional[str]) -> str:
        value = str(fusion or "concatenation").strip().lower().replace("-", "_").replace(" ", "_")
        # Keep in sync with _canonical_fusion_strategy in training_orchestrator.py
        _ALIASES = {
            "concat":                  "concatenation",
            "concatenate":             "concatenation",
            "concatenationfusion":     "concatenation",
            "unified_latent":          "ula",
            "unified_latent_alignment":"ula",
            "omnimodal":               "ula",
            "unifiedlatentfusion":     "ula",
            "gated_fusion":            "gated",
            "gatedfusion":             "gated",
            "uncertaintygraph":        "uncertainty_graph",
            "uncertainty+graph":       "uncertainty_graph",
            "crossfuse":               "complementarity",
            "ssunifier":               "structural_semantic",
            "moe":                     "fusemoe",
            "mixture_of_experts":      "fusemoe",
        }
        return _ALIASES.get(value, value or "concatenation")

    @staticmethod
    def _resolve_modality_predictability(
        modality: str,
        predictability_scores: Dict[str, float],
    ) -> Optional[float]:
        if not predictability_scores:
            return None

        modality_key = str(modality).lower()
        direct = predictability_scores.get(modality)
        if isinstance(direct, (int, float)):
            return float(direct)

        for key, value in predictability_scores.items():
            if not isinstance(value, (int, float)):
                continue
            if modality_key in str(key).lower():
                return float(value)
        return None

    def _apply_resource_constraints(
        self,
        selected_tier: Optional[str],
        catalogue: Dict[str, Dict[str, Any]],
        fallback_chain: List[str],
        modality: str,
        latency_budget_ms: Optional[float],
        memory_budget_mb: Optional[float],
    ) -> Optional[str]:
        """Downgrade encoder tier when latency or memory budgets are exceeded."""
        if not selected_tier or selected_tier not in catalogue:
            return selected_tier

        selected_idx = fallback_chain.index(selected_tier) if selected_tier in fallback_chain else len(fallback_chain) - 1
        for tier in fallback_chain[selected_idx:]:
            if tier not in catalogue:
                continue
            name = str(catalogue[tier].get("name", ""))
            lat = float(_ENCODER_LATENCY_MS.get(name, 0.0))
            mem = float(_ENCODER_VRAM_MB.get(name, 0.0))
            lat_ok = latency_budget_ms is None or lat <= float(latency_budget_ms)
            mem_ok = memory_budget_mb is None or mem <= float(memory_budget_mb)
            if lat_ok and mem_ok:
                if tier != selected_tier:
                    logger.info(
                        "AdvancedModelSelector: %s tier downgraded %s -> %s due to resource budget",
                        modality,
                        selected_tier,
                        tier,
                    )
                return tier

        # If nothing satisfies constraints, keep the lightest available tier.
        lightest = fallback_chain[-1] if fallback_chain else selected_tier
        return lightest if lightest in catalogue else selected_tier
