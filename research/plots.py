"""Plot utilities for research and monitoring reports."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Best-effort numeric coercion for plotting inputs."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_metric(exp: Dict[str, Any], key: str) -> Optional[float]:
    """Read metric from either top-level keys or nested metrics dict."""
    if key in exp:
        return _safe_float(exp.get(key))
    metrics = exp.get("metrics", {})
    if isinstance(metrics, dict):
        return _safe_float(metrics.get(key))
    return None


def _extract_latency_ms(exp: Dict[str, Any]) -> Optional[float]:
    """Normalize latency from scalar or {mean, p95,...} mapping."""
    raw = exp.get("latency_ms")
    if isinstance(raw, dict):
        return _safe_float(raw.get("mean"))
    return _safe_float(raw)


def _ensure_parent(path: Path) -> None:
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def generate_accuracy_latency_plot(
    experiments: List[Dict[str, Any]],
    output_path: str = "reports/accuracy_latency.png",
) -> Optional[str]:
    """Generate accuracy-vs-latency scatter plot and return saved path."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping accuracy/latency plot")
        return None

    points = []
    for exp in experiments:
        acc = _extract_metric(exp, "accuracy")
        lat = _extract_latency_ms(exp)
        if acc is None or lat is None:
            continue
        points.append((exp.get("model_id", "unknown"), lat, acc))

    if not points:
        logger.warning("No valid accuracy/latency points available for plotting")
        return None

    out = Path(output_path)
    _ensure_parent(out)

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    ax.scatter(xs, ys, alpha=0.8)

    for model_id, lat, acc in points:
        ax.annotate(str(model_id)[:12], (lat, acc), fontsize=7)

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Latency")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return str(out)


def generate_calibration_plot(
    experiments: List[Dict[str, Any]],
    output_path: str = "reports/calibration.png",
) -> Optional[str]:
    """Generate ECE-vs-Brier scatter plot and return saved path."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping calibration plot")
        return None

    points = []
    for exp in experiments:
        ece = _extract_metric(exp, "ece")
        brier = _extract_metric(exp, "brier")
        if ece is None or brier is None:
            continue
        points.append((exp.get("model_id", "unknown"), ece, brier))

    if not points:
        logger.warning("No valid ECE/Brier points available for plotting")
        return None

    out = Path(output_path)
    _ensure_parent(out)

    fig, ax = plt.subplots(figsize=(7, 5))
    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    ax.scatter(xs, ys, alpha=0.8)

    for model_id, ece, brier in points:
        ax.annotate(str(model_id)[:12], (ece, brier), fontsize=7)

    ax.set_xlabel("ECE")
    ax.set_ylabel("Brier Score")
    ax.set_title("Calibration Overview")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return str(out)


def generate_fusion_comparison_plot(
    experiments: List[Dict[str, Any]],
    output_path: str = "reports/fusion_comparison.png",
) -> Optional[str]:
    """Generate bar chart of mean accuracy grouped by fusion strategy."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping fusion comparison plot")
        return None

    grouped: Dict[str, List[float]] = {}
    for exp in experiments:
        fusion = (
            exp.get("fusion_strategy")
            or exp.get("fusion_type")
            or "unknown"
        )
        acc = _extract_metric(exp, "accuracy")
        if acc is None:
            continue
        grouped.setdefault(str(fusion), []).append(acc)

    if not grouped:
        logger.warning("No fusion/accuracy data available for plotting")
        return None

    labels = sorted(grouped.keys())
    means = [sum(grouped[k]) / max(len(grouped[k]), 1) for k in labels]

    out = Path(output_path)
    _ensure_parent(out)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, means)
    ax.set_ylabel("Mean Accuracy")
    ax.set_xlabel("Fusion Strategy")
    ax.set_title("Fusion Strategy Comparison")
    ax.set_ylim(bottom=0.0, top=min(1.0, max(means) + 0.1))
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return str(out)
