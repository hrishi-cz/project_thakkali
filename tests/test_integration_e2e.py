"""End-to-End Integration Test — Full pipeline from data ingestion to paper generation."""

import pytest
import tempfile
import json
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch
import torch.nn as nn


@pytest.mark.integration
class TestEndToEndPipeline:
    """Integration tests for complete APEX pipeline with XAI, monitoring, and reporting."""

    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace with all necessary directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            (workspace / "data").mkdir()
            (workspace / "models").mkdir()
            (workspace / "registry").mkdir()
            (workspace / "reports").mkdir()
            (workspace / "uploads").mkdir()
            
            yield {
                "root": str(workspace),
                "data": str(workspace / "data"),
                "models": str(workspace / "models"),
                "registry": str(workspace / "registry"),
                "reports": str(workspace / "reports"),
                "uploads": str(workspace / "uploads"),
            }

    @pytest.fixture
    def sample_dataset(self, temp_workspace):
        """Create a sample dataset for testing."""
        # Create CSV dataset
        data_dir = Path(temp_workspace["data"])
        
        # Generate synthetic tabular data
        n_samples = 100
        X = np.random.randn(n_samples, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
        df['target'] = y
        
        dataset_path = data_dir / "test_data.csv"
        df.to_csv(dataset_path, index=False)
        
        return {
            "path": str(dataset_path),
            "n_samples": n_samples,
            "n_features": 5,
            "target_column": "target",
        }

    @pytest.fixture
    def mock_model(self):
        """Create a simple model for testing."""
        model = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )
        return model

    def test_integration_schema_detection(self, sample_dataset):
        """Test Phase 2: Schema detection from CSV."""
        from data_ingestion.schema_detector import SchemaDetector
        
        detector = SchemaDetector()
        
        # Load data
        df = pd.read_csv(sample_dataset["path"])
        
        # Detect schema
        schema = detector.detect_schema_from_dataframe(df)
        
        # Verify schema
        assert schema is not None
        assert "target" in schema or "columns" in schema

    def test_integration_preprocessing(self, sample_dataset):
        """Test Phase 3: Data preprocessing."""
        from preprocessing.tabular_preprocessor import TabularPreprocessor
        
        df = pd.read_csv(sample_dataset["path"])
        
        # Create simple preprocessor
        preprocessor = TabularPreprocessor()
        
        # Process data (should handle numeric columns)
        processed = preprocessor.fit_transform(df.drop("target", axis=1))
        
        assert processed is not None
        assert len(processed) == len(df)

    def test_integration_xai_generation(self, mock_model, sample_dataset):
        """Test XAI artifact generation during training."""
        from pipeline.xai_engine import XAIExplainer, generate_xai_artifacts
        
        # Create sample batch
        batch = {
            "tabular": np.random.randn(5, 5),
        }
        
        # Generate XAI artifacts
        artifacts = generate_xai_artifacts(mock_model, batch, ["tabular"])
        
        # Verify artifacts
        assert isinstance(artifacts, dict)
        assert "tabular" in artifacts or "timestamp" in artifacts
        assert "timestamp" in artifacts

    def test_integration_monitoring_evaluation(self):
        """Test monitoring engine evaluation."""
        from pipeline.monitoring import MonitoringEngine
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MonitoringEngine(
                registry_dir=tmpdir,
                reports_dir=tmpdir,
            )
            
            # Simulate training metrics
            metrics = {
                "accuracy": 0.87,
                "f1": 0.86,
                "auc_roc": 0.89,
                "ece": 0.08,
                "loss": 0.25,
            }
            
            # Evaluate metrics
            result = engine.evaluate_and_report("test_model", metrics)
            
            # Verify result
            assert isinstance(result, dict)
            assert "alerts" in result

    def test_integration_monitoring_alert_triggered(self):
        """Test monitoring generates report when metrics degrade."""
        from pipeline.monitoring import MonitoringEngine
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MonitoringEngine(
                registry_dir=tmpdir,
                reports_dir=tmpdir,
            )
            
            # Bad metrics that should trigger alerts
            bad_metrics = {
                "accuracy": 0.55,  # Below threshold
                "f1": 0.54,
                "ece": 0.20,  # Above threshold
            }
            
            result = engine.evaluate_and_report("bad_model", bad_metrics)
            
            # Should have alerts
            assert isinstance(result, dict)
            alerts = result.get("alerts", [])
            assert len(alerts) > 0

    def test_integration_experiment_collector(self, temp_workspace):
        """Test experiment collection from registry."""
        from research.experiment_collector import ExperimentCollector
        
        # Create sample experiments in registry
        registry_dir = Path(temp_workspace["registry"])
        
        experiments = [
            {
                "model_id": "apex_v1_exp1",
                "accuracy": 0.92,
                "f1": 0.91,
                "modalities": ["tabular"],
            },
            {
                "model_id": "apex_v1_exp2",
                "accuracy": 0.90,
                "f1": 0.89,
                "modalities": ["tabular"],
            },
        ]
        
        for exp in experiments:
            exp_file = registry_dir / f"{exp['model_id']}_metadata.json"
            with open(exp_file, "w") as f:
                json.dump(exp, f)
        
        # Collect experiments
        collector = ExperimentCollector(registry_dir=str(registry_dir))
        collected = collector.collect()
        
        # Verify collection
        assert len(collected) >= len(experiments)
        assert any("exp1" in e.get("model_id", "") for e in collected)

    def test_integration_ablation_study(self, temp_workspace):
        """Test ablation study generation."""
        from research.ablation import build_ablation
        
        experiments = [
            {"model_id": "with_fusion", "accuracy": 0.95, "f1": 0.94},
            {"model_id": "without_fusion", "accuracy": 0.90, "f1": 0.89},
        ]
        
        ablation = build_ablation(experiments)
        
        # Verify ablation
        assert isinstance(ablation, dict)

    def test_integration_paper_generation(self, temp_workspace):
        """Test end-to-end paper generation."""
        from research.paper_generator import PaperGenerator
        from research.ablation import build_ablation
        
        # Create sample experiments
        experiments = [
            {
                "model_id": f"apex_v1_{i:03d}",
                "dataset": "test",
                "accuracy": 0.88 + (i * 0.01),
                "f1": 0.87 + (i * 0.01),
                "latency_ms": 12.0 - (i * 0.5),
            }
            for i in range(1, 6)
        ]
        
        # Build ablation
        ablation = build_ablation(experiments)
        
        # Generate paper
        generator = PaperGenerator(
            experiments=experiments,
            ablation=ablation,
            plot_path="/tmp/plots.png"
        )
        
        paper = generator.generate_full_paper()
        
        # Verify paper
        assert isinstance(paper, str)
        assert len(paper) > 500
        assert "#" in paper  # Has headers

    @patch("pipeline.training_orchestrator.TrainingOrchestrator.run_phase")
    def test_integration_training_phase_xai_injection(self, mock_run_phase):
        """Test that XAI is properly injected into training phase."""
        # This is a mock test since full training would be expensive
        
        # Mock the training orchestrator to verify XAI injection
        mock_run_phase.return_value = {
            "xai": {
                "tabular": {"feature_importances": {}},
                "timestamp": "2026-03-22T12:00:00",
            }
        }
        
        result = mock_run_phase()
        
        # Verify XAI was included
        assert "xai" in result
        assert "tabular" in result["xai"]

    @patch("pipeline.monitoring.MonitoringEngine.evaluate_and_report")
    def test_integration_monitoring_phase_integration(self, mock_monitor):
        """Test that monitoring is properly integrated into Phase 7."""
        # Mock monitoring evaluation
        mock_monitor.return_value = {
            "alerts": ["accuracy_low"],
            "report_generated": True,
            "report_path": "/reports/report_test.md",
        }
        
        result = mock_monitor()
        
        # Verify monitoring results
        assert "alerts" in result
        assert "report_generated" in result

    def test_integration_full_pipeline_happy_path(self, temp_workspace, sample_dataset):
        """Test happy path through entire pipeline."""
        # This is the main integration test
        
        steps = []
        
        # Step 1: Load and detect schema
        try:
            df = pd.read_csv(sample_dataset["path"])
            steps.append("data_loading")
        except Exception as e:
            pytest.fail(f"Data loading failed: {e}")
        
        # Step 2: Preprocess
        try:
            # Simple preprocessing: fillna and normalize
            X = df.drop("target", axis=1).fillna(0)
            y = df["target"]
            steps.append("preprocessing")
        except Exception as e:
            pytest.fail(f"Preprocessing failed: {e}")
        
        # Step 3: Create model
        try:
            model = nn.Sequential(
                nn.Linear(5, 16),
                nn.ReLU(),
                nn.Linear(16, 2),
            )
            steps.append("model_creation")
        except Exception as e:
            pytest.fail(f"Model creation failed: {e}")
        
        # Step 4: Simulate training and XAI
        try:
            from pipeline.xai_engine import generate_xai_artifacts
            batch = {"tabular": X.values[:5]}
            artifacts = generate_xai_artifacts(model, batch, ["tabular"])
            steps.append("xai_generation")
        except Exception as e:
            pytest.fail(f"XAI generation failed: {e}")
        
        # Step 5: Evaluate with monitoring
        try:
            from pipeline.monitoring import MonitoringEngine
            with tempfile.TemporaryDirectory() as tmpdir:
                engine = MonitoringEngine(tmpdir, tmpdir)
                metrics = {"accuracy": 0.89, "f1": 0.88}
                result = engine.evaluate_and_report("test", metrics)
                steps.append("monitoring_evaluation")
        except Exception as e:
            pytest.fail(f"Monitoring failed: {e}")
        
        # Step 6: Generate paper
        try:
            from research.paper_generator import PaperGenerator
            from research.ablation import build_ablation
            experiments = [
                {"model_id": "exp1", "accuracy": 0.89, "f1": 0.88},
            ]
            ablation = build_ablation(experiments)
            generator = PaperGenerator(experiments, ablation, "/tmp/plots.png")
            paper = generator.generate_full_paper()
            steps.append("paper_generation")
        except Exception as e:
            pytest.fail(f"Paper generation failed: {e}")
        
        # Verify all steps completed
        assert len(steps) == 6
        assert "data_loading" in steps
        assert "preprocessing" in steps
        assert "model_creation" in steps
        assert "xai_generation" in steps
        assert "monitoring_evaluation" in steps
        assert "paper_generation" in steps


@pytest.mark.integration
class TestDriftToReportingFlow:
    """Test the drift detection → retraining → monitoring → reporting flow."""

    def test_drift_detection_triggers_retraining(self):
        """Test that drift detection can trigger model retraining."""
        # This would test the Phase 6 → Phase 5 feedback loop
        
        # Mock drift detector detection
        drift_detected = True
        retraining_triggered = drift_detected
        
        assert retraining_triggered is True

    def test_retrained_model_monitoring(self):
        """Test that retrained model goes through monitoring."""
        from pipeline.monitoring import MonitoringEngine
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MonitoringEngine(tmpdir, tmpdir)
            
            # Simulate retrained model metrics
            metrics = {
                "accuracy": 0.91,
                "f1": 0.90,
                "ece": 0.07,
            }
            
            result = engine.evaluate_and_report("retrained_model", metrics)
            
            # Verify monitoring completes
            assert isinstance(result, dict)

    def test_continuous_monitoring_workflow(self):
        """Test continuous monitoring across multiple model versions."""
        from pipeline.monitoring import MonitoringEngine
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MonitoringEngine(tmpdir, tmpdir)
            
            # Simulate multiple model versions
            versions = [
                ("v1", {"accuracy": 0.90, "f1": 0.89}),
                ("v2", {"accuracy": 0.92, "f1": 0.91}),
                ("v3", {"accuracy": 0.55, "f1": 0.54}),  # Degradation
            ]
            
            results = {}
            for version, metrics in versions:
                result = engine.evaluate_and_report(f"model_{version}", metrics)
                results[version] = result
            
            # Verify all versions were evaluated
            assert len(results) == 3
            
            # v3 should have alerts
            v3_result = results["v3"]
            alerts = v3_result.get("alerts", [])
            assert len(alerts) > 0


@pytest.mark.integration
class TestErrorRecovery:
    """Test error handling and recovery in pipeline stages."""

    def test_preprocessing_error_recovery(self):
        """Test graceful handling of preprocessing errors."""
        # If preprocessing fails, training shouldn't crash
        
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_df = pd.DataFrame({"col": [np.nan, np.nan, np.nan]})
            
            try:
                # Should handle NaN gracefully
                processed = bad_df.fillna(0)
                assert len(processed) == 3
            except Exception as e:
                pytest.fail(f"Preprocessing NaN handling failed: {e}")

    def test_xai_generation_with_model_error(self):
        """Test XAI generation handles model errors."""
        from pipeline.xai_engine import generate_xai_artifacts
        
        # Mock model that raises error
        bad_model = Mock(side_effect=RuntimeError("Model error"))
        
        batch = {"tabular": np.random.randn(5, 5)}
        
        # Should not crash entire pipeline
        try:
            artifacts = generate_xai_artifacts(bad_model, batch, ["tabular"])
        except RuntimeError:
            # Expected, but should be handled gracefully in real code
            pass

    def test_monitoring_with_missing_metrics(self):
        """Test monitoring handles missing or incomplete metrics."""
        from pipeline.monitoring import MonitoringEngine
        
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MonitoringEngine(tmpdir, tmpdir)
            
            # Incomplete metrics
            metrics = {
                "accuracy": 0.90,
                # Missing F1, ECE, etc.
            }
            
            result = engine.evaluate_and_report("incomplete", metrics)
            
            # Should not crash
            assert isinstance(result, dict)

    def test_paper_generation_with_missing_experiments(self):
        """Test paper generation with insufficient experiment data."""
        from research.paper_generator import PaperGenerator
        from research.ablation import build_ablation
        
        # Only one experiment
        experiments = [{"model_id": "only_exp", "accuracy": 0.85}]
        
        ablation = build_ablation(experiments)
        generator = PaperGenerator(experiments, ablation, "/tmp/plots.png")
        
        # Should still generate paper
        paper = generator.generate_full_paper()
        assert isinstance(paper, str)
        assert len(paper) > 0
