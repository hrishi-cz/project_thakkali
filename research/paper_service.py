"""
research/paper_service.py

High-level orchestrator for end-to-end paper generation.
Combines experiment collection, ablation, plotting, and paper writing.
"""

import logging
from typing import Tuple, Optional

from config.paths import MODEL_REGISTRY_DIR
from research.experiment_collector import ExperimentCollector
from research.ablation import build_ablation
from research.paper_generator import PaperGenerator
from research.plots import generate_accuracy_latency_plot

logger = logging.getLogger(__name__)


class PaperService:
    """
    End-to-end paper generation service.
    
    Usage:
        service = PaperService(registry_dir="models/registry")
        paper_text, plot_path = service.generate()
        # Save paper_text to file, use plot_path for metadata
    """

    def __init__(self, registry_dir: str = str(MODEL_REGISTRY_DIR)):
        """
        Parameters
        ----------
        registry_dir : str
            Path to model registry.
        """
        self.registry_dir = registry_dir
        self.collector = ExperimentCollector(registry_dir=registry_dir)

    def generate(self) -> Tuple[str, Optional[str]]:
        """
        Generate complete research paper.
        
        Returns
        -------
        Tuple[str, Optional[str]]
            (paper_markdown, plot_path)
        """
        logger.info("[PaperService] Starting paper generation...")

        # Step 1: Collect experiments
        logger.info("[1/4] Collecting experiments from registry...")
        experiments = self.collector.collect()

        if not experiments:
            logger.warning("No experiments found in registry!")
            return "No experiments available for paper generation.", None

        logger.info(f"  ✓ Found {len(experiments)} experiments")

        # Step 2: Build ablation study
        logger.info("[2/4] Building ablation study...")
        ablation = build_ablation(experiments)
        logger.info("  ✓ Ablation complete")

        # Step 3: Generate plots
        logger.info("[3/4] Generating accuracy vs latency plot...")
        plot_path = generate_accuracy_latency_plot(experiments)
        logger.info(f"  ✓ Plot saved to {plot_path}")

        # Step 4: Generate paper
        logger.info("[4/4] Generating paper...")
        generator = PaperGenerator(experiments, ablation, plot_path=plot_path)
        paper = generator.generate_full_paper()
        logger.info("  ✓ Paper generated")

        logger.info("[PaperService] ✅ Paper generation complete!")
        return paper, plot_path
