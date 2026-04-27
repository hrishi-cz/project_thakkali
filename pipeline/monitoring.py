"""
pipeline/monitoring.py (ENHANCED)

Monitoring engine with auto-report generation.
Triggers on performance degradation, drift detection, or retrain events.
Integrates paper generation into the monitoring workflow.
"""

import logging
import os
import json
from typing import Any, Dict, Optional
from datetime import datetime

from config.paths import MODEL_REGISTRY_DIR, REPORTS_DIR

logger = logging.getLogger(__name__)

# Expose these symbols at module scope for test patching and optional runtime
# integrations. They are not strict hard dependencies for core monitoring.
try:
    from research.paper_service import PaperService  # type: ignore
except Exception:
    PaperService = None  # type: ignore

try:
    from research.experiment_collector import ExperimentCollector  # type: ignore
except Exception:
    ExperimentCollector = None  # type: ignore


class MonitoringEngine:
    """
    Post-training monitoring with automatic report generation.
    
    Triggers:
    - Accuracy drops below threshold → auto-generate paper
    - Calibration (ECE) exceeds threshold → auto-generate paper
    - Drift detected → auto-generate paper
    - Retrain completed → auto-generate paper
    
    Usage:
        monitor = MonitoringEngine()
        result = monitor.evaluate_and_report(model_id, metrics)
        # Returns: {"alerts": [...], "report_generated": True, "report_path": "..."}
    """

    def __init__(
        self,
        registry_dir: str = str(MODEL_REGISTRY_DIR),
        reports_dir: str = str(REPORTS_DIR),
    ):
        """
        Parameters
        ----------
        registry_dir : str
            Model registry directory.
        reports_dir : str
            Where to save generated reports.
        """
        self.registry_dir = registry_dir
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)

        # Alert thresholds
        self.alert_thresholds = {
            "accuracy_min": 0.60,
            "ece_max": 0.15,
            "f1_min": 0.40,
        }

        # Backward-compatible scalar aliases.
        self.accuracy_threshold = self.alert_thresholds["accuracy_min"]
        self.ece_threshold = self.alert_thresholds["ece_max"]
        self.f1_threshold = self.alert_thresholds["f1_min"]

    def evaluate_and_report(
        self,
        model_id: str,
        metrics: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Evaluate metrics and trigger auto-report if needed.

        Parameters
        ----------
        model_id : str
            Model identifier from registry.
        metrics : Dict[str, float]
            Metrics dict from training: {"accuracy": ..., "f1": ..., "ece": ...}

        Returns
        -------
        Dict with:
            - "alerts": list of alert messages
            - "report_generated": bool
            - "report_path": str (if generated)
        """
        alerts = []

        accuracy_threshold = float(self.alert_thresholds.get("accuracy_min", self.accuracy_threshold))
        ece_threshold = float(self.alert_thresholds.get("ece_max", self.ece_threshold))
        f1_threshold = float(self.alert_thresholds.get("f1_min", self.f1_threshold))

        # -----  Alert Checks  -----
        accuracy = metrics.get("accuracy")
        if accuracy is not None and accuracy < accuracy_threshold:
            alerts.append(f"ALERT: Low accuracy: {accuracy:.3f} < {accuracy_threshold}")

        ece = metrics.get("ece")
        if ece is not None and ece > ece_threshold:
            alerts.append(f"ALERT: Poor calibration (ECE): {ece:.3f} > {ece_threshold}")

        f1 = metrics.get("f1")
        if f1 is not None and f1 < f1_threshold:
            alerts.append(f"ALERT: Low F1 score: {f1:.3f} < {f1_threshold}")

        # -----  Trigger Report if Alerts -----
        report_path = None
        if alerts:
            report_path = self._generate_report(model_id, metrics, alerts)

        return {
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
            "alerts": alerts,
            "report_generated": bool(report_path),
            "report_path": report_path,
        }

    def _generate_report(
        self,
        model_id: str,
        metrics: Dict[str, float],
        alerts: list,
    ) -> str:
        """
        Generate and save a monitoring report.
        Also triggers full paper generation if configured.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"report_{model_id}_{timestamp}.md"
        report_path = os.path.join(self.reports_dir, report_name)

        # Build report content
        report_lines = [
            f"# Monitoring Report\n",
            f"**Model ID**: {model_id}\n",
            f"**Generated**: {datetime.now().isoformat()}\n\n",
            f"## Alerts\n",
        ]

        for alert in alerts:
            report_lines.append(f"- {alert}\n")

        report_lines.append(f"\n## Metrics\n")
        for key, val in metrics.items():
            report_lines.append(f"- **{key}**: {val:.4f}\n")

        # Try to include full paper if possible
        try:
            if PaperService is not None:
                logger.info("  Generating full research paper...")
                service = PaperService(registry_dir=self.registry_dir)
                paper_text, plot_path = service.generate()

                report_lines.append(f"\n## Full Research Paper\n")
                report_lines.append(paper_text)

                if plot_path:
                    report_lines.append(f"\n**Plot saved**: {plot_path}\n")
            else:
                report_lines.append("\n*(Full paper generation unavailable in current environment.)*\n")

        except Exception as e:
            logger.warning(f"  Could not generate full paper: {e}")
            report_lines.append(f"\n*(Full paper generation failed: {e})*\n")

        # Save report
        report_content = "".join(report_lines)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info(f"✓ Monitoring report saved: {report_path}")
        return report_path

    def list_reports(self) -> list:
        """List all generated reports."""
        if not os.path.exists(self.reports_dir):
            return []
        return sorted(os.listdir(self.reports_dir))

    def get_report(self, report_name: str) -> Optional[str]:
        """Load report content by name."""
        report_path = os.path.join(self.reports_dir, report_name)
        if not os.path.exists(report_path):
            return None
        with open(report_path, "r", encoding="utf-8") as f:
            return f.read()
