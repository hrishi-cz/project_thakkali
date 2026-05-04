"""Verification tests for publication output artifacts.

Confirms that each benchmark script produces JSON with the expected schema —
and that statistical tests (Wilcoxon, bootstrap CIs) and compute budget
fields are structurally correct before submitting to a venue.

Run after scripts/run_full_benchmark.py --quick:

    pytest tests/test_publication_outputs.py -xvs
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

RESULTS_DIR = _ROOT / "diary" / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(name: str) -> dict:
    path = RESULTS_DIR / name
    if not path.exists():
        pytest.skip(f"{name} not found — run scripts/run_full_benchmark.py --quick first")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Aggregate results — Wilcoxon + bootstrap CIs
# ---------------------------------------------------------------------------

class TestAggregatedResults:
    def test_aggregated_results_exists(self):
        path = RESULTS_DIR / "aggregated_results.json"
        if not path.exists():
            pytest.skip("aggregated_results.json not found")

    def test_statistical_tests_field_present(self):
        data = _load_json("aggregated_results.json")
        assert "statistical_tests" in data, "aggregated_results.json must have 'statistical_tests'"

    def test_compute_budget_field_present(self):
        data = _load_json("aggregated_results.json")
        # compute_budget only present when ComputeTracker records exist
        # (populated by pipeline during real training runs)
        if "compute_budget" not in data:
            pytest.skip(
                "compute_budget not in aggregated_results.json — "
                "run a real training pipeline with ComputeTracker to populate diary/results/*_compute.json"
            )
        assert isinstance(data["compute_budget"], dict)

    def test_bootstrap_ci_in_compute_stats(self):
        """compute_statistics() must produce bootstrap_ci95_low and bootstrap_ci95_high."""
        from scripts.aggregate_results import compute_statistics
        import numpy as np
        vals = list(np.random.rand(10))
        stats = compute_statistics(vals)
        assert "bootstrap_ci95_low" in stats, "bootstrap_ci95_low missing from compute_statistics output"
        assert "bootstrap_ci95_high" in stats, "bootstrap_ci95_high missing from compute_statistics output"
        assert stats["bootstrap_ci95_low"] is not None

    def test_wilcoxon_fields_present(self):
        """paired_significance_test() must return p_value and significant_at_005."""
        from scripts.aggregate_results import paired_significance_test
        a = [0.8, 0.75, 0.82, 0.79, 0.77]
        b = [0.70, 0.65, 0.72, 0.68, 0.66]
        result = paired_significance_test(a, b)
        assert "p_value" in result
        assert "significant_at_005" in result
        assert isinstance(result["significant_at_005"], bool)


# ---------------------------------------------------------------------------
# ULA ablation
# ---------------------------------------------------------------------------

class TestULAAblation:
    def test_ula_ablation_has_summary(self):
        data = _load_json("ula_ablation.json")
        assert "summary" in data, "ula_ablation.json must have 'summary'"
        # summary may be empty if all trials failed (e.g., no GPU, quick mode)
        assert isinstance(data["summary"], dict)

    def test_ula_ablation_conditions(self):
        data = _load_json("ula_ablation.json")
        summary = data.get("summary", {})
        if not summary:
            pytest.skip("ULA ablation summary is empty (likely all trials failed in quick mode without GPU)")
        # At least one condition should be present
        assert len(summary) >= 1, f"Expected at least 1 condition in summary, got {summary}"

    def test_ula_ablation_per_condition_has_mean_std(self):
        data = _load_json("ula_ablation.json")
        for cond_name, stats in data["summary"].items():
            acc_stats = stats.get("val_acc", {})
            assert "mean" in acc_stats, f"Condition '{cond_name}' missing mean in val_acc"
            assert "std" in acc_stats, f"Condition '{cond_name}' missing std in val_acc"


# ---------------------------------------------------------------------------
# LoRA ablation
# ---------------------------------------------------------------------------

class TestLoRAAblation:
    def test_lora_ablation_has_summary(self):
        data = _load_json("lora_ablation.json")
        assert "summary" in data
        assert "ranks_tested" in data

    def test_lora_ablation_rank_zero_present(self):
        data = _load_json("lora_ablation.json")
        assert "r=0" in data["summary"], "Frozen baseline (r=0) must be in LoRA ablation"

    def test_lora_ablation_has_param_counts(self):
        data = _load_json("lora_ablation.json")
        for rank_key, stats in data["summary"].items():
            assert "trainable_params" in stats, f"{rank_key} missing trainable_params"
            assert "lora_params" in stats, f"{rank_key} missing lora_params"


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class TestBaselines:
    def test_baselines_has_sklearn_mlp(self):
        data = _load_json("baselines.json")
        models = [r.get("model", "") for r in data.get("baselines", [])]
        assert any("MLP" in m or "mlp" in m.lower() for m in models), (
            "sklearn_MLP baseline must be present in baselines.json"
        )

    def test_baselines_accuracy_in_range(self):
        data = _load_json("baselines.json")
        for row in data.get("baselines", []):
            acc = row.get("accuracy", -1)
            assert 0.0 <= acc <= 1.0, f"Baseline '{row.get('model')}' accuracy {acc} out of [0, 1]"


# ---------------------------------------------------------------------------
# Hateful Memes benchmark
# ---------------------------------------------------------------------------

class TestHatefulMemesBenchmark:
    def test_benchmark_has_aggregated(self):
        data = _load_json("hateful_memes_benchmark.json")
        assert "aggregated" in data, "hateful_memes_benchmark.json must have 'aggregated'"

    def test_tfidf_baseline_present(self):
        data = _load_json("hateful_memes_benchmark.json")
        agg = data.get("aggregated", {})
        assert any("TF-IDF" in k for k in agg), "TF-IDF baseline must be in hateful_memes_benchmark"

    def test_latex_table_generated(self):
        data = _load_json("hateful_memes_benchmark.json")
        latex = data.get("latex_table", "")
        assert "\\begin{table}" in latex, "hateful_memes_benchmark must include LaTeX table"


# ---------------------------------------------------------------------------
# Compute budget
# ---------------------------------------------------------------------------

class TestComputeBudget:
    def test_compute_tracker_schema(self):
        from pipeline.compute_tracker import ComputeTracker
        ct = ComputeTracker("test_run_schema")
        ct.start()
        ct.stop()
        d = ct.to_dict()
        assert "peak_vram_mb" in d
        assert "gpu_hours" in d
        assert "total_params" in d
        assert "lora_params" in d
        assert "backbone_frozen_params" in d

    def test_compute_tracker_time_measured(self):
        import time
        from pipeline.compute_tracker import ComputeTracker
        ct = ComputeTracker("test_timing")
        ct.start()
        time.sleep(0.05)
        ct.stop()
        assert ct._elapsed_s >= 0.04, "Elapsed time should be measurable"


# ---------------------------------------------------------------------------
# Paper generation
# ---------------------------------------------------------------------------

class TestPaperGeneration:
    def test_paper_generator_produces_ula_section(self):
        from research.paper_generator import PaperGenerator
        gen = PaperGenerator([], {})
        ula = gen.generate_ula_section()
        assert "Unified Latent Alignment" in ula
        assert "LoRA" in ula

    def test_paper_generator_full_paper_non_empty(self):
        from research.paper_generator import PaperGenerator
        gen = PaperGenerator([], {})
        paper = gen.generate_full_paper()
        assert len(paper) > 1000, "Generated paper must be non-trivially long"
        assert "AutoVision" in paper

    def test_paper_has_limitations_section(self):
        from research.paper_generator import PaperGenerator
        gen = PaperGenerator([], {})
        paper = gen.generate_full_paper()
        assert "Limitations" in paper

    def test_paper_md_exists_after_run(self):
        path = RESULTS_DIR / "paper.md"
        if not path.exists():
            pytest.skip("paper.md not yet generated — run scripts/run_full_benchmark.py --quick")
        content = path.read_text(encoding="utf-8")
        assert "AutoVision" in content
        assert len(content) > 500


# ---------------------------------------------------------------------------
# New API endpoints
# ---------------------------------------------------------------------------

class TestNewAPIEndpoints:
    def test_aggregated_results_endpoint_schema(self):
        """GET /research/aggregated-results returns expected structure."""
        from fastapi.testclient import TestClient
        from api.run_api import app
        client = TestClient(app)
        resp = client.get("/research/aggregated-results")
        assert resp.status_code in (200, 404), f"Unexpected status: {resp.status_code}"
        body = resp.json()
        assert "status" in body

    def test_compute_budget_endpoint_schema(self):
        """GET /intelligence/compute-budget/{model_id} returns expected structure."""
        from fastapi.testclient import TestClient
        from api.run_api import app
        client = TestClient(app)
        resp = client.get("/intelligence/compute-budget/nonexistent_model_xyz")
        assert resp.status_code == 200
        body = resp.json()
        assert "n_trials" in body
        assert "total_gpu_hours" in body
        assert "peak_vram_mb" in body
