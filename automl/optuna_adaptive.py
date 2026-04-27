"""Adaptive Optuna search-space and trial-budget policies."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AdaptiveOptunaController:
    """Adjust HPO bounds and trial count from dataset scale and modality mix."""

    def adapt_search_space(
        self,
        base_space: Dict[str, Any],
        dataset_size: int,
        modalities: List[str],
        problem_type: str,
    ) -> Dict[str, Any]:
        space: Dict[str, Any] = deepcopy(base_space or {})
        dataset_size = int(max(0, dataset_size))
        n_modalities = len(modalities or [])

        # Epoch and LR policy by data volume.
        if dataset_size <= 5_000:
            space["epochs"] = {"type": "int", "low": 12, "high": 40}
            space["learning_rate"] = {"type": "float", "low": 1e-4, "high": 3e-3, "log": True}
        elif dataset_size <= 50_000:
            space["epochs"] = {"type": "int", "low": 8, "high": 24}
            space["learning_rate"] = {"type": "float", "low": 5e-5, "high": 2e-3, "log": True}
        else:
            space["epochs"] = {"type": "int", "low": 4, "high": 16}
            space["learning_rate"] = {"type": "float", "low": 3e-5, "high": 1e-3, "log": True}

        # Keep dropout tighter for regression to avoid over-regularization.
        if "regression" in str(problem_type):
            space["dropout"] = {"type": "float", "low": 0.0, "high": 0.3}
        elif "dropout" not in space:
            space["dropout"] = {"type": "float", "low": 0.05, "high": 0.5}

        # Add alignment tuning when using multiple modalities.
        if n_modalities >= 2 and "alignment_weight" not in space:
            space["alignment_weight"] = {
                "type": "float",
                "low": 0.0,
                "high": 0.2,
            }

        if n_modalities >= 2 and "modality_dropout_prob" not in space:
            space["modality_dropout_prob"] = {
                "type": "float",
                "low": 0.05,
                "high": 0.30,
            }

        return space

    def suggest_trial_count(self, dataset_size: int, gpu_available: bool) -> int:
        dataset_size = int(max(0, dataset_size))
        if dataset_size <= 5_000:
            trials = 4
        elif dataset_size <= 50_000:
            trials = 3
        else:
            trials = 2

        if not gpu_available:
            trials = max(1, trials - 1)
        return int(trials)

    def seed_from_warm_start(
        self,
        study: Any,
        warm_params: Dict[str, Any],
        hpo_space: Dict[str, Any],
    ) -> None:
        """
        Enqueue a warm-start trial from historical meta-learning params.

        Only keys present in the active HPO space are forwarded.
        """
        if not warm_params or not hpo_space or study is None:
            return

        filtered = {
            key: value
            for key, value in dict(warm_params).items()
            if key in hpo_space and value is not None
        }
        if not filtered:
            return

        try:
            study.enqueue_trial(filtered)
            logger.info(
                "AdaptiveOptunaController: warm-start trial enqueued with %d param(s): %s",
                len(filtered),
                sorted(filtered.keys()),
            )
        except Exception as exc:
            logger.warning("AdaptiveOptunaController warm-start enqueue failed: %s", exc)

    def update_from_trial_diagnostics(
        self,
        feedback_state: Dict[str, Any],
        diagnostics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update adaptive feedback state from latest trial diagnostics."""
        state = dict(feedback_state or {})
        fit_type = str((diagnostics or {}).get("fit_type", "unknown")).lower()
        if fit_type not in {"overfitting", "underfitting", "good"}:
            return state

        history = list(state.get("history", []))
        history.append(fit_type)
        history = history[-8:]
        state["history"] = history
        state["last_fit_type"] = fit_type
        state["last_generalization_gap"] = (diagnostics or {}).get("generalization_gap")
        state["last_diagnostics"] = dict(diagnostics or {})
        return state

    def next_trial_overrides(
        self,
        feedback_state: Dict[str, Any],
        hpo_space: Dict[str, Any],
        recent_trials=None,
    ) -> Dict[str, Any]:
        """Suggest deterministic next-trial overrides based on fit trend (G20 enhanced)."""
        fit_type = str((feedback_state or {}).get("last_fit_type", "unknown")).lower()
        if fit_type not in {"overfitting", "underfitting", "good"}:
            return {}

        overrides: Dict[str, Any] = {}

        def _bounds(name: str) -> Optional[Tuple[float, float]]:
            spec = (hpo_space or {}).get(name)
            if not isinstance(spec, dict):
                return None
            if spec.get("type") not in {"int", "float"}:
                return None
            low = float(spec.get("low", 0.0))
            high = float(spec.get("high", low))
            if low > high:
                low, high = high, low
            return low, high

        lr_bounds = _bounds("learning_rate")
        wd_bounds = _bounds("weight_decay")
        dropout_bounds = _bounds("dropout")

        if fit_type == "overfitting":
            if lr_bounds is not None:
                low, high = lr_bounds
                overrides["learning_rate"] = low + (high - low) * 0.25
            if wd_bounds is not None:
                low, high = wd_bounds
                overrides["weight_decay"] = low + (high - low) * 0.80
            if dropout_bounds is not None:
                low, high = dropout_bounds
                overrides["dropout"] = low + (high - low) * 0.80
        elif fit_type == "underfitting":
            if lr_bounds is not None:
                low, high = lr_bounds
                overrides["learning_rate"] = low + (high - low) * 0.75
            if wd_bounds is not None:
                low, high = wd_bounds
                overrides["weight_decay"] = low + (high - low) * 0.25
            if dropout_bounds is not None:
                low, high = dropout_bounds
                overrides["dropout"] = low + (high - low) * 0.20

        # G20: hard cap from recent pruned steps
        _pruned_steps = [
            int(t.user_attrs.get("pruned_at_step"))
            for t in (recent_trials or [])
            if getattr(t, "user_attrs", {}).get("pruned_at_step") is not None
        ]
        if _pruned_steps:
            import numpy as _np_g20
            _median_prune = int(_np_g20.median(_pruned_steps))
            _upper_bound = max(int(_median_prune * 1.5), 8)
            _epoch_spec = (hpo_space or {}).get("epochs", {})
            _ep_min = int((_epoch_spec.get("low") or 3))
            _ep_cap = min(int((_epoch_spec.get("high") or 40)), _upper_bound)
            try:
                from automl.trial_intelligence import TrialIntelligence as _TI_G20
                _ti_g20 = _TI_G20()
                _last_d = (feedback_state or {}).get("last_diagnostics") or {}
                if _last_d:
                    _ti_g20.update_memory(_last_d)
                _suggestion = _ti_g20.estimate_epochs(
                    base=_ep_cap,
                    fit_type=_last_d.get("fit_type", "good"),
                    flat_epoch=_median_prune,
                )
                overrides["epochs"] = int(max(_ep_min, min(_ep_cap, _suggestion)))
                logger.info(
                    "G20: prune-capped epochs=%d (median_prune=%d, cap=%d)",
                    overrides["epochs"], _median_prune, _ep_cap,
                )
            except Exception as _g20_exc:
                logger.debug("G20: epoch cap failed: %s", _g20_exc)

        # Merge epoch suggestion from TrialIntelligence based on last diagnostics
        last_diag = (feedback_state or {}).get("last_diagnostics") or {}
        if last_diag.get("fit_type") in {"overfitting", "underfitting", "good"}:
            try:
                from automl.trial_intelligence import TrialIntelligence
                _ti = TrialIntelligence()
                _ti.update_memory(last_diag)
                _ti_adj = _ti.adjust_hyperparams(overrides or {})
                if "epochs" in _ti_adj:
                    epoch_spec = (hpo_space or {}).get("epochs", {})
                    ep_min = int(epoch_spec.get("low", 3))
                    ep_max = int(epoch_spec.get("high", 40))
                    overrides["epochs"] = int(max(ep_min, min(ep_max, _ti_adj["epochs"])))
                    logger.info(
                        "AdaptiveOptunaController: epoch override -> %d (fit_type=%s)",
                        overrides["epochs"],
                        last_diag.get("fit_type"),
                    )
            except Exception as _e:
                logger.debug("AdaptiveOptunaController: epoch adaptation failed: %s", _e)

        return overrides

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(float(low), min(float(high), float(value)))

    def derive_predictability_factors(
        self,
        trial_summary: Dict[str, Any],
        modalities: Optional[List[str]],
        modality_importance: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Convert trial feedback into modality-level predictability multipliers.

        Factors are centered around 1.0 and intentionally conservative so
        downstream Phase 3/4 routing changes smoothly between runs.
        """
        summary = dict(trial_summary or {})
        fit_type = str(summary.get("fit_type", "unknown") or "unknown").lower()
        penalty = self._clamp(float(summary.get("adaptive_penalty", 0.0) or 0.0), 0.0, 0.8)

        if fit_type == "overfitting":
            base = 1.0 - self._clamp(0.10 + penalty * 0.40, 0.0, 0.30)
        elif fit_type == "underfitting":
            base = 1.0 + self._clamp(0.04 + penalty * 0.20, 0.0, 0.16)
        else:
            base = 1.0 - self._clamp(penalty * 0.08, 0.0, 0.05)

        importance_map = {
            str(key): float(value)
            for key, value in dict(modality_importance or {}).items()
            if value is not None
        }

        modality_list = [str(mod) for mod in list(modalities or []) if mod]
        if not modality_list and importance_map:
            modality_list = list(importance_map.keys())

        factors: Dict[str, float] = {}
        for modality in modality_list:
            factor = float(base)
            importance = self._clamp(float(importance_map.get(modality, 0.0) or 0.0), 0.0, 1.0)

            if fit_type == "overfitting":
                factor -= min(0.10, importance * 0.08)
            elif fit_type == "underfitting":
                factor += min(0.08, importance * 0.06)

            factors[modality] = round(self._clamp(factor, 0.55, 1.25), 4)

        return factors

    def build_next_run_feedback(
        self,
        trial_summary: Dict[str, Any],
        feedback_state: Dict[str, Any],
        hpo_space: Dict[str, Any],
        modalities: Optional[List[str]] = None,
        modality_importance: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Build persisted feedback that can seed subsequent planning/selection.

        The bundle includes:
        - ``next_trial_overrides`` for deterministic warm guidance
        - ``predictability_factors`` consumed by ExecutionContext
        """
        summary = dict(trial_summary or {})
        state = dict(feedback_state or {})

        return {
            "fit_type": str(summary.get("fit_type", "unknown") or "unknown"),
            "adaptive_penalty": float(summary.get("adaptive_penalty", 0.0) or 0.0),
            "next_trial_overrides": self.next_trial_overrides(state, hpo_space),
            "predictability_factors": self.derive_predictability_factors(
                trial_summary=summary,
                modalities=modalities,
                modality_importance=modality_importance,
            ),
            "feedback_history": list(state.get("history", []))[-8:],
        }
