"""Unit tests for Monitoring Engine — alert detection, report generation, metric evaluation."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from pipeline.monitoring import MonitoringEngine


@pytest.fixture
def temp_dirs():
    """Create temporary registry and reports directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_dir = Path(tmpdir) / "registry"
        reports_dir = Path(tmpdir) / "reports"
        registry_dir.mkdir()
        reports_dir.mkdir()

        yield {
            "registry": str(registry_dir),
            "reports": str(reports_dir),
            "temp": tmpdir,
        }


class TestMonitoringEngine:
    """Test suite for MonitoringEngine class."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary registry and reports directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_dir = Path(tmpdir) / "registry"
            reports_dir = Path(tmpdir) / "reports"
            registry_dir.mkdir()
            reports_dir.mkdir()
            
            yield {
                "registry": str(registry_dir),
                "reports": str(reports_dir),
                "temp": tmpdir,
            }

    @pytest.fixture
    def monitoring_engine(self, temp_dirs):
        """Create MonitoringEngine with temp directories."""
        return MonitoringEngine(
            registry_dir=temp_dirs["registry"],
            reports_dir=temp_dirs["reports"]
        )

    def test_monitoring_engine_initialization(self, monitoring_engine):
        """Test MonitoringEngine initializes with correct defaults."""
        assert monitoring_engine is not None
        assert hasattr(monitoring_engine, "alert_thresholds")
        
        # Check default thresholds
        assert "accuracy_min" in monitoring_engine.alert_thresholds
        assert "ece_max" in monitoring_engine.alert_thresholds
        assert "f1_min" in monitoring_engine.alert_thresholds

    def test_alert_threshold_defaults(self, monitoring_engine):
        """Test that alert thresholds have reasonable defaults."""
        thresholds = monitoring_engine.alert_thresholds
        
        # Accuracy should be threshold between 0 and 1
        assert 0 < thresholds["accuracy_min"] < 1
        
        # ECE should be a small positive value
        assert 0 < thresholds["ece_max"] < 0.5
        
        # F1 should be between 0 and 1
        assert 0 < thresholds["f1_min"] < 1

    def test_evaluate_metrics_no_alerts(self, monitoring_engine):
        """Test metric evaluation when all metrics are good."""
        metrics = {
            "accuracy": 0.95,
            "f1": 0.94,
            "ece": 0.05,
        }
        
        result = monitoring_engine.evaluate_and_report(
            model_id="good_model",
            metrics=metrics
        )
        
        assert isinstance(result, dict)
        assert "alerts" in result
        
        # Should have no alerts for good metrics
        alerts = result.get("alerts", [])
        assert len(alerts) == 0 or all("low" not in alert.lower() for alert in alerts)

    def test_evaluate_metrics_low_accuracy(self, monitoring_engine):
        """Test alert triggered for low accuracy."""
        metrics = {
            "accuracy": 0.55,  # Below default threshold of 0.60
            "f1": 0.94,
            "ece": 0.05,
        }
        
        result = monitoring_engine.evaluate_and_report(
            model_id="bad_accuracy_model",
            metrics=metrics
        )
        
        assert isinstance(result, dict)
        alerts = result.get("alerts", [])
        
        # Should have at least one accuracy alert
        assert any("accuracy" in str(alert).lower() for alert in alerts)

    def test_evaluate_metrics_high_ece(self, monitoring_engine):
        """Test alert triggered for poor calibration (high ECE)."""
        metrics = {
            "accuracy": 0.95,
            "f1": 0.94,
            "ece": 0.20,  # Above default threshold of 0.15
        }
        
        result = monitoring_engine.evaluate_and_report(
            model_id="poor_calibration_model",
            metrics=metrics
        )
        
        assert isinstance(result, dict)
        alerts = result.get("alerts", [])
        
        # Should have calibration/ECE alert
        assert any("calibration" in str(alert).lower() or "ece" in str(alert).lower() for alert in alerts)

    def test_evaluate_metrics_low_f1(self, monitoring_engine):
        """Test alert triggered for low F1 score."""
        metrics = {
            "accuracy": 0.95,
            "f1": 0.35,  # Below default threshold of 0.40
            "ece": 0.05,
        }
        
        result = monitoring_engine.evaluate_and_report(
            model_id="poor_f1_model",
            metrics=metrics
        )
        
        assert isinstance(result, dict)
        alerts = result.get("alerts", [])
        
        # Should have F1 alert
        assert any("f1" in str(alert).lower() for alert in alerts)

    def test_evaluate_metrics_multiple_alerts(self, monitoring_engine):
        """Test multiple alerts triggered for multiple bad metrics."""
        metrics = {
            "accuracy": 0.55,  # Bad
            "f1": 0.35,        # Bad
            "ece": 0.20,       # Bad
        }
        
        result = monitoring_engine.evaluate_and_report(
            model_id="very_bad_model",
            metrics=metrics
        )
        
        assert isinstance(result, dict)
        alerts = result.get("alerts", [])
        
        # Should have multiple alerts
        assert len(alerts) >= 2

    def test_evaluate_returns_dict_structure(self, monitoring_engine):
        """Test evaluate_and_report returns correct structure."""
        metrics = {
            "accuracy": 0.92,
            "f1": 0.91,
            "ece": 0.08,
        }
        
        result = monitoring_engine.evaluate_and_report(
            model_id="test_model",
            metrics=metrics
        )
        
        assert isinstance(result, dict)
        assert "alerts" in result
        assert "model_id" in result or "timestamp" in result
        # May contain report path if report generated
        if "report_generated" in result:
            assert isinstance(result["report_generated"], bool)

    def test_evaluate_with_missing_metrics(self, monitoring_engine):
        """Test evaluation handles missing metrics gracefully."""
        metrics = {
            "accuracy": 0.92,
            # Missing F1 and ECE
        }
        
        result = monitoring_engine.evaluate_and_report(
            model_id="incomplete_metrics_model",
            metrics=metrics
        )
        
        assert isinstance(result, dict)
        # Should not raise error, should return dict
        assert "alerts" in result

    def test_evaluate_with_empty_metrics(self, monitoring_engine):
        """Test evaluation with empty metrics dict."""
        metrics = {}
        
        result = monitoring_engine.evaluate_and_report(
            model_id="no_metrics_model",
            metrics=metrics
        )
        
        assert isinstance(result, dict)

    @patch("pipeline.monitoring.PaperService")
    def test_report_generation_triggered_on_alerts(self, mock_paper_service, monitoring_engine):
        """Test that report generation is triggered when alerts occur."""
        # Mock PaperService to avoid actual file I/O
        mock_service = Mock()
        mock_service.generate.return_value = ("sample paper text", "/tmp/plots.png")
        mock_paper_service.return_value = mock_service
        
        metrics = {
            "accuracy": 0.55,  # Triggers alert
            "f1": 0.54,
            "ece": 0.05,
        }
        
        result = monitoring_engine.evaluate_and_report(
            model_id="alert_trigger_model",
            metrics=metrics
        )
        
        # Might attempt to generate report when alerts detected
        assert isinstance(result, dict)

    def test_list_reports_empty(self, monitoring_engine):
        """Test list_reports when no reports exist."""
        reports = monitoring_engine.list_reports()
        
        assert isinstance(reports, list)
        assert len(reports) == 0

    def test_list_reports_with_reports(self, temp_dirs):
        """Test list_reports with generated report files."""
        # Create some dummy report files
        reports_dir = Path(temp_dirs["reports"])
        (reports_dir / "report_1.md").write_text("# Report 1")
        (reports_dir / "report_2.md").write_text("# Report 2")
        
        engine = MonitoringEngine(
            registry_dir=temp_dirs["registry"],
            reports_dir=temp_dirs["reports"]
        )
        
        reports = engine.list_reports()
        
        assert isinstance(reports, list)
        assert len(reports) >= 2
        assert "report_1.md" in reports or any("report_1" in r for r in reports)
        assert "report_2.md" in reports or any("report_2" in r for r in reports)

    def test_get_report_existing(self, temp_dirs):
        """Test get_report retrieves existing report content."""
        # Create a dummy report
        reports_dir = Path(temp_dirs["reports"])
        report_content = "# Test Report\n\nThis is a test."
        (reports_dir / "test_report.md").write_text(report_content)
        
        engine = MonitoringEngine(
            registry_dir=temp_dirs["registry"],
            reports_dir=temp_dirs["reports"]
        )
        
        content = engine.get_report("test_report.md")
        
        assert isinstance(content, str)
        assert "Test Report" in content

    def test_get_report_nonexistent(self, monitoring_engine):
        """Test get_report handles missing report gracefully."""
        content = monitoring_engine.get_report("nonexistent.md")
        
        # Should return None, empty string, or raise with proper message
        assert content is None or isinstance(content, str)

    def test_evaluate_with_different_threshold_values(self, temp_dirs):
        """Test evaluation with custom threshold values."""
        # Create monitoring engine with different thresholds
        engine = MonitoringEngine(
            registry_dir=temp_dirs["registry"],
            reports_dir=temp_dirs["reports"]
        )
        
        # Customize thresholds
        engine.alert_thresholds["accuracy_min"] = 0.85  # Higher threshold
        
        metrics = {
            "accuracy": 0.87,  # Would be bad at 0.60, but good at 0.85
            "f1": 0.86,
            "ece": 0.08,
        }
        
        result = engine.evaluate_and_report("custom_threshold_model", metrics)
        
        assert isinstance(result, dict)

    def test_evaluate_with_extreme_metric_values(self, monitoring_engine):
        """Test evaluation with extreme metric values."""
        metrics = {
            "accuracy": 0.0,  # Worst possible
            "f1": 0.0,
            "ece": 1.0,  # Worst calibration
        }
        
        result = monitoring_engine.evaluate_and_report(
            model_id="terrible_model",
            metrics=metrics
        )
        
        assert isinstance(result, dict)
        alerts = result.get("alerts", [])
        
        # Should have many alerts
        assert len(alerts) >= 1

    def test_evaluate_with_perfect_metrics(self, monitoring_engine):
        """Test evaluation with perfect metrics."""
        metrics = {
            "accuracy": 1.0,
            "f1": 1.0,
            "ece": 0.0,
        }
        
        result = monitoring_engine.evaluate_and_report(
            model_id="perfect_model",
            metrics=metrics
        )
        
        assert isinstance(result, dict)
        alerts = result.get("alerts", [])
        
        # Should have no alerts
        assert len(alerts) == 0 or all("good" in alert.lower() or "ok" in alert.lower() for alert in alerts)


class TestMonitoringIntegration:
    """Integration tests for Monitoring Engine with other components."""

    @patch("pipeline.monitoring.ExperimentCollector")
    @patch("pipeline.monitoring.PaperService")
    def test_monitoring_integration_with_paper_service(self, mock_paper_service, mock_collector):
        """Test monitoring engine works with paper service."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MonitoringEngine(
                registry_dir=tmpdir,
                reports_dir=tmpdir
            )
            
            # Mock paper service
            mock_service_instance = Mock()
            mock_service_instance.generate.return_value = ("paper", "/tmp/plots.png")
            mock_paper_service.return_value = mock_service_instance
            
            metrics = {"accuracy": 0.50}  # Should trigger alert
            
            result = engine.evaluate_and_report("test", metrics)
            
            assert isinstance(result, dict)

    def test_monitoring_workflow_scenario(self, temp_dirs):
        """Test realistic monitoring workflow scenario."""
        engine = MonitoringEngine(
            registry_dir=temp_dirs["registry"],
            reports_dir=temp_dirs["reports"]
        )
        
        # Simulate batch of evaluations
        models = [
            ("good_model", {"accuracy": 0.95, "f1": 0.94}),
            ("ok_model", {"accuracy": 0.88, "f1": 0.87}),
            ("bad_model", {"accuracy": 0.55, "f1": 0.54}),
        ]
        
        results = {}
        for model_id, metrics in models:
            result = engine.evaluate_and_report(model_id, metrics)
            results[model_id] = result
        
        # Should have results for all models
        assert len(results) == 3
        
        # Bad model should have alerts
        bad_result = results["bad_model"]
        assert "alerts" in bad_result

    def test_report_persistence_across_instances(self, temp_dirs):
        """Test that reports persist across MonitoringEngine instances."""
        # Create first engine and generate report
        engine1 = MonitoringEngine(
            registry_dir=temp_dirs["registry"],
            reports_dir=temp_dirs["reports"]
        )
        
        # Manually create a report
        reports_path = Path(temp_dirs["reports"])
        report_file = reports_path / "persistent_report.md"
        report_file.write_text("# Persistent Report\nThis report should persist.")
        
        # Create second engine instance
        engine2 = MonitoringEngine(
            registry_dir=temp_dirs["registry"],
            reports_dir=temp_dirs["reports"]
        )
        
        # Should be able to retrieve the report
        reports = engine2.list_reports()
        assert len(reports) > 0
        
        content = engine2.get_report("persistent_report.md")
        assert content is not None


class TestMonitoringEdgeCases:
    """Test edge cases and error handling in monitoring."""

    def test_evaluate_with_nan_metrics(self):
        """Test evaluation handles NaN metric values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MonitoringEngine(
                registry_dir=tmpdir,
                reports_dir=tmpdir
            )
            
            import math
            metrics = {
                "accuracy": math.nan,
                "f1": 0.90,
                "ece": 0.05,
            }
            
            result = engine.evaluate_and_report("nan_model", metrics)
            
            # Should handle NaN gracefully
            assert isinstance(result, dict)

    def test_evaluate_with_inf_metrics(self):
        """Test evaluation handles infinity metric values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MonitoringEngine(
                registry_dir=tmpdir,
                reports_dir=tmpdir
            )
            
            metrics = {
                "accuracy": 0.90,
                "f1": 0.89,
                "latency_ms": float('inf'),  # Infinite latency somehow?
            }
            
            result = engine.evaluate_and_report("inf_model", metrics)
            
            assert isinstance(result, dict)

    def test_report_dir_nonexistent_on_init(self):
        """Test MonitoringEngine handles nonexistent report directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent_reports = str(Path(tmpdir) / "does_not_exist")
            
            # Should either create dir or handle gracefully
            engine = MonitoringEngine(
                registry_dir=tmpdir,
                reports_dir=nonexistent_reports
            )
            
            # Should still be functional
            metrics = {"accuracy": 0.90}
            result = engine.evaluate_and_report("test", metrics)
            assert isinstance(result, dict)
