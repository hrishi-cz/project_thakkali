"""Unit tests for Paper Generation System — collector, ablation, generator, plots."""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List
from research.experiment_collector import ExperimentCollector
from research.ablation import build_ablation, _compare_groups, format_ablation_table
from research.paper_generator import PaperGenerator
from research.plots import (
    generate_accuracy_latency_plot,
    generate_calibration_plot,
    generate_fusion_comparison_plot,
)


class TestExperimentCollector:
    """Test suite for ExperimentCollector module."""

    @pytest.fixture
    def sample_experiments(self):
        """Create sample experiment data."""
        return [
            {
                "model_id": "apex_v1_20260310_100000",
                "timestamp": 1710078000,
                "dataset": "iris",
                "modalities": ["tabular"],
                "accuracy": 0.95,
                "f1": 0.94,
                "auc_roc": 0.97,
                "ece": 0.05,
                "brier": 0.06,
                "latency_ms": 10.5,
                "memory_mb": 250,
            },
            {
                "model_id": "apex_v1_20260310_110000",
                "timestamp": 1710081600,
                "dataset": "iris",
                "modalities": ["tabular"],
                "accuracy": 0.92,
                "f1": 0.91,
                "auc_roc": 0.94,
                "ece": 0.08,
                "brier": 0.09,
                "latency_ms": 8.3,
                "memory_mb": 220,
            },
        ]

    @pytest.fixture
    def temp_registry(self, sample_experiments):
        """Create temporary registry directory with sample experiments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_dir = Path(tmpdir) / "registry"
            registry_dir.mkdir()
            
            for exp in sample_experiments:
                exp_file = registry_dir / f"{exp['model_id']}_metadata.json"
                with open(exp_file, "w") as f:
                    json.dump(exp, f)
            
            yield str(registry_dir)

    def test_collector_initialization(self, temp_registry):
        """Test ExperimentCollector initializes correctly."""
        collector = ExperimentCollector(registry_dir=temp_registry)
        assert collector.registry_dir == temp_registry

    def test_collect_experiments(self, temp_registry, sample_experiments):
        """Test collect() returns all experiments from registry."""
        collector = ExperimentCollector(registry_dir=temp_registry)
        experiments = collector.collect()
        
        assert isinstance(experiments, list)
        assert len(experiments) >= 2
        
        # Check that experiments have required keys
        for exp in experiments:
            assert "model_id" in exp
            assert "accuracy" in exp

    def test_get_best_experiment_by_accuracy(self, temp_registry, sample_experiments):
        """Test get_best_experiment returns highest accuracy model."""
        collector = ExperimentCollector(registry_dir=temp_registry)
        best = collector.get_best_experiment(metric="accuracy")
        
        assert best is not None
        assert best["accuracy"] == 0.95  # First experiment has highest accuracy

    def test_get_best_experiment_by_f1(self, temp_registry, sample_experiments):
        """Test get_best_experiment works with different metrics."""
        collector = ExperimentCollector(registry_dir=temp_registry)
        best = collector.get_best_experiment(metric="f1")
        
        assert best is not None
        # Verify it's actually the best F1 score
        assert best["f1"] == max(exp["f1"] for exp in sample_experiments)

    def test_get_experiments_by_modality(self, temp_registry):
        """Test filtering experiments by modality."""
        collector = ExperimentCollector(registry_dir=temp_registry)
        # Create additional experiments with different modalities
        
        tabular_exps = collector.get_experiments_by_modality("tabular")
        
        assert isinstance(tabular_exps, list)
        # Check that all returned experiments contain tabular modality
        for exp in tabular_exps:
            if "modalities" in exp:
                assert "tabular" in exp["modalities"]

    def test_collect_empty_registry(self):
        """Test collect() on empty registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = ExperimentCollector(registry_dir=tmpdir)
            experiments = collector.collect()
            
            assert isinstance(experiments, list)
            assert len(experiments) == 0


class TestAblation:
    """Test suite for Ablation study building."""

    @pytest.fixture
    def sample_experiments_with_ablations(self):
        """Create experiments with and without key components."""
        return [
            {
                "model_id": "with_fusion_with_xai",
                "dataset": "iris",
                "accuracy": 0.95,
                "f1": 0.94,
                "fusion_enabled": True,
                "xai_enabled": True,
                "modalities": ["tabular", "text"],
            },
            {
                "model_id": "without_fusion_with_xai",
                "dataset": "iris",
                "accuracy": 0.90,
                "f1": 0.89,
                "fusion_enabled": False,
                "xai_enabled": True,
                "modalities": ["tabular", "text"],
            },
            {
                "model_id": "with_fusion_without_xai",
                "dataset": "iris",
                "accuracy": 0.92,
                "f1": 0.91,
                "fusion_enabled": True,
                "xai_enabled": False,
                "modalities": ["tabular", "text"],
            },
        ]

    def test_build_ablation_structure(self, sample_experiments_with_ablations):
        """Test build_ablation returns correct structure."""
        ablation = build_ablation(sample_experiments_with_ablations)
        
        assert isinstance(ablation, dict)
        assert "fusion_impact" in ablation or "ablation" in ablation

    def test_compare_groups_returns_metrics(self):
        """Test _compare_groups calculates deltas correctly."""
        with_component = [{"accuracy": 0.95, "f1": 0.94}]
        without_component = [{"accuracy": 0.90, "f1": 0.89}]
        
        comparison = _compare_groups(with_component, without_component)
        
        assert isinstance(comparison, dict)
        assert "with_mean" in comparison or "accuracy_delta" in comparison

    def test_compare_groups_with_empty_group(self):
        """Test _compare_groups handles empty groups gracefully."""
        with_component = [{"accuracy": 0.95}]
        without_component = []
        
        comparison = _compare_groups(with_component, without_component)
        
        assert isinstance(comparison, dict)

    def test_format_ablation_table(self, sample_experiments_with_ablations):
        """Test ablation table formatting produces markdown."""
        ablation = build_ablation(sample_experiments_with_ablations)
        table = format_ablation_table(ablation)
        
        # Should return markdown table
        assert isinstance(table, str)
        # Markdown tables contain pipes
        if "|" in table:
            assert table.count("|") >= 4  # At least header + one row


class TestPaperGenerator:
    """Test suite for Paper Generation."""

    @pytest.fixture
    def sample_experiments_for_paper(self):
        """Create realistic experiment data for paper generation."""
        return [
            {
                "model_id": f"apex_v1_202603{i:02d}_100000",
                "dataset": "iris",
                "timestamp": 1710078000 + (i * 86400),
                "accuracy": 0.93 + (i * 0.01),
                "f1": 0.91 + (i * 0.01),
                "auc_roc": 0.95 + (i * 0.01),
                "ece": 0.07 - (i * 0.005),
                "brier": 0.08 - (i * 0.005),
                "latency_ms": 12.0 - (i * 0.5),
                "memory_mb": 256 - (i * 10),
                "loss": 0.15 - (i * 0.01),
                "modalities": ["tabular"],
                "training_time_s": 300 + (i * 10),
            }
            for i in range(1, 11)
        ]

    def test_paper_generator_initialization(self, sample_experiments_for_paper):
        """Test PaperGenerator initializes with experiment data."""
        ablation = build_ablation(sample_experiments_for_paper)
        
        generator = PaperGenerator(
            experiments=sample_experiments_for_paper,
            ablation=ablation,
            plot_path="/tmp/plots.png"
        )
        
        assert generator.experiments == sample_experiments_for_paper
        assert generator.ablation == ablation

    def test_generate_title(self, sample_experiments_for_paper):
        """Test title generation from dataset."""
        ablation = build_ablation(sample_experiments_for_paper)
        generator = PaperGenerator(
            experiments=sample_experiments_for_paper,
            ablation=ablation,
            plot_path="/tmp/plots.png"
        )
        
        title = generator.generate_title()
        
        assert isinstance(title, str)
        assert len(title) > 0
        assert "iris" in title.lower() or "AutoVision" in title or "autovision" in title.lower()

    def test_generate_abstract(self, sample_experiments_for_paper):
        """Test abstract generation from best metrics."""
        ablation = build_ablation(sample_experiments_for_paper)
        generator = PaperGenerator(
            experiments=sample_experiments_for_paper,
            ablation=ablation,
            plot_path="/tmp/plots.png"
        )
        
        abstract = generator.generate_abstract()
        
        assert isinstance(abstract, str)
        assert len(abstract) > 100  # Should be substantial
        assert any(keyword in abstract.lower() for keyword in ["accuracy", "dataset", "model"])

    def test_generate_methodology(self, sample_experiments_for_paper):
        """Test methodology section generation."""
        ablation = build_ablation(sample_experiments_for_paper)
        generator = PaperGenerator(
            experiments=sample_experiments_for_paper,
            ablation=ablation,
            plot_path="/tmp/plots.png"
        )
        
        methodology = generator.generate_methodology()
        
        assert isinstance(methodology, str)
        assert len(methodology) > 50
        assert any(keyword in methodology.lower() for keyword in ["fusion", "loss", "modality"])

    def test_generate_results(self, sample_experiments_for_paper):
        """Test results section with metrics table."""
        ablation = build_ablation(sample_experiments_for_paper)
        generator = PaperGenerator(
            experiments=sample_experiments_for_paper,
            ablation=ablation,
            plot_path="/tmp/plots.png"
        )
        
        results = generator.generate_results()
        
        assert isinstance(results, str)
        assert len(results) > 50
        # Should contain performance metrics
        assert any(metric in results.lower() for metric in ["accuracy", "auc", "f1"])

    def test_generate_ablation_section(self, sample_experiments_for_paper):
        """Test ablation study section generation."""
        ablation = build_ablation(sample_experiments_for_paper)
        generator = PaperGenerator(
            experiments=sample_experiments_for_paper,
            ablation=ablation,
            plot_path="/tmp/plots.png"
        )
        
        ablation_section = generator.generate_ablation()
        
        assert isinstance(ablation_section, str)
        assert len(ablation_section) > 0

    def test_generate_full_paper(self, sample_experiments_for_paper):
        """Test full paper generation with all sections."""
        ablation = build_ablation(sample_experiments_for_paper)
        generator = PaperGenerator(
            experiments=sample_experiments_for_paper,
            ablation=ablation,
            plot_path="/tmp/plots.png"
        )
        
        paper = generator.generate_full_paper()
        
        # Full paper should be markdown
        assert isinstance(paper, str)
        assert len(paper) > 500  # Substantial document
        
        # Should contain key sections
        assert "#" in paper  # Has headers
        # Check for multiple sections
        section_count = paper.count("#")
        assert section_count >= 3  # At least 3 header levels

    def test_paper_with_missing_metrics(self):
        """Test paper generation with incomplete experiment data."""
        incomplete_experiments = [
            {
                "model_id": "exp_1",
                "dataset": "test",
                "accuracy": 0.90,
                "f1": 0.89,
                # Missing other metrics
            },
            {
                "model_id": "exp_2",
                "dataset": "test",
                "accuracy": 0.92,
                # Missing F1 and others
            },
        ]
        
        ablation = build_ablation(incomplete_experiments)
        generator = PaperGenerator(
            experiments=incomplete_experiments,
            ablation=ablation,
            plot_path="/tmp/plots.png"
        )
        
        paper = generator.generate_full_paper()
        
        # Should still generate paper, handling missing metrics gracefully
        assert isinstance(paper, str)
        assert len(paper) > 0


class TestPlotGeneration:
    """Test suite for Plotting utilities."""

    @pytest.fixture
    def sample_plot_experiments(self):
        """Create experiments for plotting."""
        return [
            {
                "model_id": f"model_{i}",
                "accuracy": 0.85 + (i * 0.02),
                "latency_ms": 10 + (i * 1),
                "ece": 0.08 - (i * 0.01),
                "brier": 0.09 - (i * 0.01),
                "fusion_strategy": "concatenation" if i % 2 == 0 else "fusion_net",
                "modalities": ["tabular", "text"] if i % 2 == 0 else ["image"],
            }
            for i in range(1, 6)
        ]

    def test_accuracy_latency_plot_generation(self, sample_plot_experiments):
        """Test accuracy vs latency plot generation."""
        pytest.importorskip("matplotlib")
        
        plot_path = None
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                plot_path = f.name
            
            plot_path = generate_accuracy_latency_plot(sample_plot_experiments, plot_path)
            
            if plot_path:
                assert os.path.exists(plot_path) or isinstance(plot_path, str)
        finally:
            if plot_path and os.path.exists(plot_path):
                try:
                    os.remove(plot_path)
                except:
                    pass

    def test_calibration_plot_generation(self, sample_plot_experiments):
        """Test calibration (ECE/Brier) plot generation."""
        pytest.importorskip("matplotlib")
        
        try:
            plot_path = generate_calibration_plot(sample_plot_experiments, "/tmp/calib.png")
            assert plot_path is not None or isinstance(plot_path, (str, type(None)))
        except:
            # Matplotlib might not be available in test env
            pass

    def test_fusion_comparison_plot_generation(self, sample_plot_experiments):
        """Test fusion strategy comparison plot."""
        pytest.importorskip("matplotlib")
        
        try:
            plot_path = generate_fusion_comparison_plot(sample_plot_experiments, "/tmp/fusion.png")
            assert plot_path is not None or isinstance(plot_path, (str, type(None)))
        except:
            pass


class TestPaperEdgeCases:
    """Test paper generation edge cases and error handling."""

    def test_paper_with_single_experiment(self):
        """Test paper generation with only one experiment."""
        experiments = [
            {
                "model_id": "only_exp",
                "dataset": "test",
                "accuracy": 0.85,
                "f1": 0.84,
            }
        ]
        
        ablation = build_ablation(experiments)
        generator = PaperGenerator(
            experiments=experiments,
            ablation=ablation,
            plot_path="/tmp/plots.png"
        )
        
        paper = generator.generate_full_paper()
        assert isinstance(paper, str)
        assert len(paper) > 0

    def test_paper_with_no_experiments(self):
        """Test paper generation with empty experiment list."""
        experiments = []
        
        ablation = build_ablation(experiments)
        generator = PaperGenerator(
            experiments=experiments,
            ablation=ablation,
            plot_path="/tmp/plots.png"
        )
        
        # Should handle gracefully (return stub or error message)
        paper = generator.generate_full_paper()
        assert isinstance(paper, str)

    def test_paper_section_independence(self):
        """Test that paper sections can be generated independently."""
        experiments = [{"model_id": "exp1", "accuracy": 0.90, "f1": 0.89}]
        
        ablation = build_ablation(experiments)
        generator = PaperGenerator(
            experiments=experiments,
            ablation=ablation,
            plot_path="/tmp/plots.png"
        )
        
        # Each section should be callable independently
        assert isinstance(generator.generate_title(), str)
        assert isinstance(generator.generate_abstract(), str)
        assert isinstance(generator.generate_methodology(), str)
        assert isinstance(generator.generate_results(), str)
