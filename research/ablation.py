"""
research/ablation.py

Ablation study builder: compares model performance with/without
key components (fusion, XAI, etc.) to quantify their contribution.
"""

import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


def build_ablation(experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build ablation study from experiments.
    
    Compares:
    1. With fusion vs. without fusion → contribution of fusion strategy
    2. With XAI vs. without XAI → contribution of explainability
    3. Different modality combinations → modality contribution
    
    Parameters
    ----------
    experiments : List[Dict]
        Output from ExperimentCollector.collect().
    
    Returns
    -------
    Dict with ablation results:
        {
            "fusion_ablation": {
                "with_fusion": {...metrics...},
                "without_fusion": {...metrics...},
                "delta_accuracy": 0.05,
            },
            "xai_ablation": {...},
            "modality_ablation": {...},
        }
    """
    ablation = {}

    # -----------------------------------------------------------------------
    # Fusion Ablation: with vs. without fusion
    # -----------------------------------------------------------------------
    with_fusion = [
        e for e in experiments
        if e.get("fusion_type") not in (None, "concatenation", "linear")
    ]
    without_fusion = [
        e for e in experiments
        if e.get("fusion_type") in (None, "concatenation", "linear")
    ]

    ablation["fusion"] = _compare_groups(
        with_fusion,
        without_fusion,
        label_with="Advanced Fusion",
        label_without="Simple Concat",
    )

    # -----------------------------------------------------------------------
    # XAI Ablation: with vs. without XAI in metadata
    # -----------------------------------------------------------------------
    with_xai = [e for e in experiments if e.get("xai") and len(e.get("xai", {})) > 0]
    without_xai = [e for e in experiments if not e.get("xai") or len(e.get("xai", {})) == 0]

    ablation["xai"] = _compare_groups(
        with_xai,
        without_xai,
        label_with="With XAI Artifacts",
        label_without="Without XAI",
    )

    # -----------------------------------------------------------------------
    # Modality Ablation: multimodal vs. single modality
    # -----------------------------------------------------------------------
    multimodal = [e for e in experiments if len(e.get("modalities", [])) >= 2]
    unimodal = [e for e in experiments if len(e.get("modalities", [])) == 1]

    ablation["modality"] = _compare_groups(
        multimodal,
        unimodal,
        label_with="Multimodal",
        label_without="Single Modality",
    )

    # Backward-compatible aliases used by older dashboards/tests.
    ablation["fusion_impact"] = ablation["fusion"]
    ablation["xai_impact"] = ablation["xai"]
    ablation["modality_impact"] = ablation["modality"]

    return ablation


def _compare_groups(
    group_with: List[Dict],
    group_without: List[Dict],
    label_with: str = "With Feature",
    label_without: str = "Without Feature",
) -> Dict[str, Any]:
    """
    Compare two groups of experiments (with vs. without a feature).
    Compute mean metrics and differences.
    """
    result = {
        f"{label_with}_count": len(group_with),
        f"{label_without}_count": len(group_without),
    }

    # Compute mean metrics for each group
    metrics_with = _average_metrics(group_with)
    metrics_without = _average_metrics(group_without)

    result[f"{label_with}_metrics"] = metrics_with
    result[f"{label_without}_metrics"] = metrics_without

    # Delta (improvement from feature)
    if "accuracy" in metrics_with and "accuracy" in metrics_without:
        delta_acc = metrics_with["accuracy"] - metrics_without["accuracy"]
        result["delta_accuracy"] = round(delta_acc, 4)
        result["accuracy_delta"] = round(delta_acc, 4)

    if "f1" in metrics_with and "f1" in metrics_without:
        delta_f1 = metrics_with["f1"] - metrics_without["f1"]
        result["delta_f1"] = round(delta_f1, 4)

    return result


def _average_metrics(experiments: List[Dict]) -> Dict[str, float]:
    """
    Compute mean metrics across a group of experiments.
    """
    if not experiments:
        return {}

    metric_sums = {}
    count = 0

    for exp in experiments:
        metrics = exp.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}

        # Backfill from flat legacy fields.
        for key in ("accuracy", "f1", "auc_roc", "ece", "brier", "loss"):
            if key in exp and key not in metrics:
                metrics[key] = exp[key]

        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                metric_sums[key] = metric_sums.get(key, 0) + val
        count += 1

    if count == 0:
        return {}

    return {k: round(v / count, 4) for k, v in metric_sums.items()}


def format_ablation_table(ablation: Dict[str, Any]) -> str:
    """
    Format ablation results as a readable string/table.
    Useful for paper generation.
    """
    lines = ["# Ablation Study\n"]

    for component, results in ablation.items():
        lines.append(f"## {component.title()} Ablation\n")

        for key, val in results.items():
            if isinstance(val, dict):
                lines.append(f"**{key}:**")
                for subkey, subval in val.items():
                    lines.append(f"  - {subkey}: {subval}")
                lines.append("")
            elif isinstance(val, (int, float)):
                lines.append(f"- {key}: {val}")

        lines.append("")

    return "\n".join(lines)
