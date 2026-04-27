"""Normalize Phase-6 drift outputs into monitor-friendly payloads."""

from __future__ import annotations

from typing import Any, Dict, List


class DriftAdapter:
    """Convert raw drift metrics into composite monitoring summaries."""

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    def _composite_from_metrics(self, metrics: Dict[str, Any], thresholds: Dict[str, Any]) -> float:
        psi = self._safe_float(metrics.get("psi"), 0.0)
        ks = self._safe_float(metrics.get("ks_statistic"), 0.0)
        fdd = self._safe_float(metrics.get("fdd"), 0.0)

        psi_t = max(1e-8, self._safe_float(thresholds.get("psi"), 0.25))
        ks_t = max(1e-8, self._safe_float(thresholds.get("ks_statistic"), 0.30))
        fdd_t = max(1e-8, self._safe_float(thresholds.get("fdd"), 0.50))

        return float((psi / psi_t + ks / ks_t + fdd / fdd_t) / 3.0)

    @staticmethod
    def _severity(composite: float, breached_count: int) -> str:
        if breached_count >= 3 or composite >= 1.5:
            return "critical"
        if breached_count >= 2 or composite >= 1.1:
            return "high"
        if breached_count >= 1 or composite >= 0.7:
            return "moderate"
        return "low"

    def build_monitor_payload(self, drift_result: Dict[str, Any]) -> Dict[str, Any]:
        metrics = dict(drift_result.get("metrics", {}))
        thresholds = dict(drift_result.get("thresholds", {}))
        status = dict(drift_result.get("status") or drift_result.get("status_per_metric") or {})

        composite = self._safe_float(drift_result.get("composite_score"), -1.0)
        if composite < 0:
            composite = self._composite_from_metrics(metrics, thresholds)

        breached: List[str] = [k for k, v in status.items() if bool(v)]
        severity = self._severity(composite, len(breached))

        retrain_recommended = bool(drift_result.get("drift_detected", False)) or composite >= 1.0

        return {
            "drift_detected": bool(drift_result.get("drift_detected", False)),
            "composite_score": round(float(composite), 6),
            "severity": severity,
            "breached_metrics": breached,
            "metrics": metrics,
            "thresholds": thresholds,
            "status": status,
            "modality_drift": drift_result.get("modality_drift", {}),
            "retrain_recommended": retrain_recommended,
            "retrain_triggered": bool(drift_result.get("retrain_triggered", False)),
            "retrain_info": drift_result.get("retrain_info", {}),
        }
