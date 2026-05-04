.PHONY: test lint reproduce plots paper clean

SEEDS := 42 123 456 789 2026
PYTHON := python

# ── Testing ────────────────────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ \
		--ignore=tests/test_phase2_sessions.py \
		--ignore=tests/test_full_pipeline_real_e2e.py \
		--ignore=tests/test_e2e_real_datasets.py \
		-q --tb=short

test-slow:
	APEX_SEED=42 $(PYTHON) -m pytest -m slow tests/test_e2e_real_datasets.py -xvs

test-all: test test-slow

lint:
	ruff check . --select E,F,W --ignore E501,E402

# ── Reproducibility ───────────────────────────────────────────────────────
reproduce: reproduce-ablations reproduce-baselines reproduce-robustness aggregate paper
	@echo "✅ Full reproduction pipeline complete"

reproduce-ablations:
	@echo "Running ablations with seeds: $(SEEDS)"
	@for seed in $(SEEDS); do \
		echo "  Seed $$seed ..."; \
		APEX_SEED=$$seed $(PYTHON) scripts/run_ablations.py \
			--output diary/results/ablations_seed$${seed}.json; \
	done

reproduce-baselines:
	@echo "Running baselines with seeds: $(SEEDS)"
	@for seed in $(SEEDS); do \
		echo "  Seed $$seed ..."; \
		APEX_SEED=$$seed $(PYTHON) scripts/run_baselines.py \
			--output diary/results/baselines_seed$${seed}.json; \
	done

reproduce-robustness:
	@echo "Running modality robustness ablation"
	APEX_SEED=42 $(PYTHON) scripts/run_modality_robustness.py \
		--output diary/results/modality_robustness.json

aggregate:
	$(PYTHON) scripts/aggregate_results.py

# ── Paper generation ──────────────────────────────────────────────────────
plots:
	$(PYTHON) scripts/generate_plots.py

paper: plots
	$(PYTHON) -c "from research.paper_generator import PaperGenerator; PaperGenerator().generate_full_paper()"
	@echo "Paper generated at diary/results/paper.md"

paper-latex:
	$(PYTHON) -c "from research.paper_generator import PaperGenerator; PaperGenerator().generate_latex()"
	@echo "LaTeX generated at diary/results/paper.tex"

# ── Docker ────────────────────────────────────────────────────────────────
docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-gpu:
	docker compose --profile gpu up -d

# ── Cleanup ───────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage coverage.xml
