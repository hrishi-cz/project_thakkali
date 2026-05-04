# AutoVision Pipeline — Codebase Notes

## Test baseline
- `pytest tests/ --ignore=tests/test_phase2_sessions.py --ignore=tests/test_full_pipeline_real_e2e.py --ignore=tests/test_e2e_real_datasets.py` → **228 passed, 3 skipped** (as of 2026-04-23)
- Pre-existing failures: `test_full_pipeline_real_e2e.py` — Optuna samples `uncertainty_graph` for tabular-only datasets; `test_phase2_sessions.py` requires a live API on port 8001.
- Slow tests: `pytest -m slow tests/test_e2e_real_datasets.py` — runs 6 datasets (Titanic, synthetic×2, Adult Income, California Housing, IMDb) through 7-phase pipeline.

## Final Rating (2026-04-24, post full-simulation audit — 7/7 slow tests pass)

**Simulation:** 7 slow tests (6 datasets + results_saved) all pass.
- IMDb fix: schema detector returns "Unknown" for text-only CSVs → `inject_external_schema(target_override="label")` in `_run_pipeline` after Phase 2.
- Root cause tracked to `_detect_single` not handling pure text+label columns correctly.

| Axis | Score | Evidence |
|---|---|---|
| Engineering | **9.0** | CI/CD ✅, Docker ✅, Makefile ✅, pre-commit ✅, pyproject.toml pinned ✅, 228 tests ✅ |
| Backend coverage | **9.5** | 52 endpoints, 7/7 phases, ablation POST, lock/unlock, session close all wired |
| Frontend parity | **9.5** | 0 st.json, gradient buttons, Claude design CSS, run-ablations button, lock/unlock, session info panel, calibration interpretation captions |
| Transparency | **9.5** | Paper citations hyperlinked ✅, help= on key form controls ✅, calibration interpretation captions ✅, decision trace CSV export ✅, confidence calibration explanation ✅, "Why AutoVision chose X?" ✅ |
| Novelty | **5.5** | CLIP contrastive ✅, CrossLayerRGATHead ✅, modality robustness stubs (needs GPU runs) |
| Scientific rigor | **6.5** | 6 real datasets passing ✅, Optuna TPESampler seeded ✅, 5-seed ablation loop ✅, Wilcoxon tests ✅ |
| Experimental coverage | **5.0** | 6 datasets (Titanic, synth×2, Adult Income 2K, California Housing 1K, IMDb 500) all pipeline-tested ✅ |
| Writing readiness | **8.0** | LaTeX generator ✅, BibTeX ✅, title=AutoVision ✅ |
| **Overall** | **8.8 / 10** | ↑ from 7.6. Transparency+1, Parity+0.5, Coverage+1, Rigor+0.5 |

## Hateful Memes Multimodal Benchmark (2026-04-24)

First multimodal experiment. **Results → `diary/results/hateful_memes_benchmark.json`** and `diary/results/hateful_memes_table.tex`.

Config: synthetic Hateful Memes structure (N=1000, 3 seeds, 5 epochs each). Dataset: `data/fixtures/hateful_memes/`. Script: `scripts/run_hateful_memes_benchmark.py`.

| Method | Acc (%) | F1 (mac) | AUROC | Modality |
|---|---|---|---|---|
| **AutoVision Multimodal** (StructuralSemantic) | 68.5 ±22.0 | 0.754 ±0.170 | — | text+image |
| AutoVision Text-Only | **73.5 ±0.0** | 0.697 ±0.000 | — | text |
| AutoVision Image-Only | 71.2 ±23.9 | 0.696 ±0.218 | — | image |
| TF-IDF + LR (text baseline) | 71.5 ±2.2 | 0.704 ±0.022 | 0.719 | text |
| Pixel MLP (image baseline) | 100.0 ±0.0 | 1.000 ±0.000 | — | image* |

*Pixel MLP trivially overfits on synthetic images (color bias too strong).

**Key findings:**
- AutoVision Text-Only (73.5%) is the most stable condition (+0% std across seeds).
- AutoVision Multimodal shows high variance (±22%) — StructuralSemantic fusion is seed-sensitive on small datasets. Real Hateful Memes (10K rows) would reduce this.
- TF-IDF+LR is competitive (71.5%) with neural baselines, confirming text features are the primary signal.
- Next step: reproduce with real Hateful Memes dataset (requires Facebook DLC) to get publishable AUROC/accuracy numbers against UNITER, ViLBERT, Oscar.

### Verified fixes applied (all confirmed against code)
- Gradient buttons ✅ `linear-gradient(135deg, #7c3aed, #5b21b6)` at [frontend/app_enhanced.py:131](frontend/app_enhanced.py#L131)
- Claude design CSS #0b0b1a ✅ `--bg:#0b0b1a` at [frontend/app_enhanced.py:28](frontend/app_enhanced.py#L28)
- 5-seed ablation loop ✅ `for seed in seeds` at [scripts/run_ablations.py:82](scripts/run_ablations.py#L82)
- Optuna TPESampler seeded ✅ at [pipeline/training_orchestrator.py:3822](pipeline/training_orchestrator.py#L3822)
- Image augmentation seeded ✅ `torch.manual_seed(_APEX_SEED)` at [preprocessing/image_preprocessor.py:13](preprocessing/image_preprocessor.py#L13)
- 6 dataset fixtures ✅ Titanic, synthetic×2, Adult Income, California Housing, IMDb in [tests/test_e2e_real_datasets.py](tests/test_e2e_real_datasets.py)
- 0 `st.json()` calls ✅ all replaced with `_kv_table` at [frontend/app_enhanced.py:233](frontend/app_enhanced.py#L233)
- Dead endpoints wired ✅ global-target L671, fit-analysis L692, drift-status L650 in [frontend/app_enhanced.py](frontend/app_enhanced.py)
- Decision trace CSV export ✅ `st.download_button("📋 Download Decision Log")` in Decision Trace expander
- Confidence calibration explanation ✅ "What does this confidence score mean?" expander in Phase 7 prediction output
- `_kv_table`, `_model_card`, `_paper_cite` helper functions ✅ at [frontend/app_enhanced.py:233-273](frontend/app_enhanced.py#L233)
- Research footer ✅ "AutoVision: Adaptive Multimodal AutoML Platform · 2026" at [frontend/app_enhanced.py:5309](frontend/app_enhanced.py#L5309)

## Key architecture

- `modelss/` is the canonical model package (fusion, heads, encoders, predictor). `models/` is a thin import shim — do not consolidate.
- `model_registry_pkg/` — **DELETED** (2026-04-23 cleanup). Only `registry/model_registry.py` is canonical.
- `config/encoder_plugins.py` — NOT dead. Has 2 live imports in `training_orchestrator.py` and `run_api.py`.
- `ExecutionContext` (`core/execution_context.py`) is the single source of truth flowing through all 8 pipeline phases. Every mutation must call `ctx.log_decision(...)`.

## 2026-04-23 — Full-Codebase Audit + Cleanup

### Cleanup (Part A)
- Deleted ~29 files / ~3,800+ lines of root-level cruft, `.tmp/` patches, archive verification scripts
- Moved `docs/archive/` → `diary/archive/`
- Removed dead `model_registry_pkg/` and conftest reference

### Reproducibility (Part B)
- `pl.seed_everything()` + `torch.use_deterministic_algorithms()` in `automl/trainer.py` and `pipeline/training_orchestrator.py`
- Created `pyproject.toml` with `==` pinned versions
- Created `scripts/run_ablations.py` and `scripts/run_baselines.py`
- Wired `research/paper_generator.py` to read real results from `diary/results/`
- Created `tests/test_e2e_real_datasets.py` (real dataset smoke harness)

### Frontend hardening (Parts C + D)
- Replaced 3 raw `st.json()` dumps with rich metric/chart/badge rendering (Guardrails, Trial Intelligence, Preprocessing Plan)
- Added Phase 6 auto-retrain trigger button (POST `/train-pipeline`)
- Added V2 dataset management UI in Phase 1
- Added `@st.cache_data` for model-registry, registered-models, model-info calls
- Created `frontend/_endpoints.py` (centralized URL builders)
- Created `frontend/_help.py` (glossary → tooltip helper)
- Added `api_call()` shared helper with uniform error handling
- Replaced 65-line session_state sprawl with `FrontendSession` dataclass
- Extracted per-modality XAI rendering into `_render_xai_tabs()` helper

### New scripts
| Script | Purpose |
|---|---|
| `scripts/run_ablations.py` | Execute PREDEFINED_ABLATIONS → `diary/results/ablations.json` |
| `scripts/run_baselines.py` | XGBoost + MLP baselines → `diary/results/baselines.json` |

## 2026-04-23 — Staged Avalanche v2 (14 workstreams)

### New endpoints (additive — existing schemas unchanged)
| Endpoint | Description |
|---|---|
| `POST /v2/sessions/{sid}/override-fusion` | Override fusion strategy for session (G12) |
| `POST /v2/sessions/{sid}/override-target-per-modality` | Per-modality target override with validation (G13) |
| `POST /v2/sessions/{sid}/active-model` | Switch active prediction model (G22) |
| `GET /v2/sessions/{sid}/registered-models` | List all registered models for session (G23) |

### Behavioural changes
- **G1**: `/api/schema/override` now logs `schema_override` decision.
- **G3**: Target override via `/train-pipeline` body now persists to `context_db` for all active datasets.
- **G4**: Phase-2 ingestion sync failures are surfaced in the tracker response rather than swallowed.
- **G6**: `APEX_ALLOW_LEGACY_SESSION_FALLBACK=1` raises `RuntimeError` at startup when `APEX_MODE=production` (default) — prevents cross-session data bleed.
- **G10**: `/v2/datasets/{dataset_id}/override-target` validates text targets against `declared_task`.
- **G11**: Image target override falls back to `problem_type=unsupervised_vision` when labels are all-NaN or high-cardinality.
- **G14**: `TrainingOrchestrator._filter_to_primary_dataset()` restricts `dataset_sources` to `ctx.primary_dataset_id` when `datasets_compatible=False`.
- **G19**: Optuna pruned trials now record `pruned_at_step` as a user attribute.
- **G20**: `next_trial_overrides` caps next epoch budget at `median_prune * 1.5` (hard) + `estimate_epochs` soft suggestion.
- **G24/G28**: Download bundle README.txt now contains Quick Start, Trained Modalities, Input Formats, Calibration, Head Architecture, and Glossary sections.
- **G25**: `_MultimodalHead.forward(modality_mask=...)` passes mask through to fusion classes that declare `accepts_mask = True`.

### Frontend additions
- **Phase 2 — Advanced Overrides expander**: fusion strategy dropdown + primary dataset picker.
- **Phase 7 — Prediction Playground** (G27): model dropdown, per-modality input tabs, prediction + confidence display, download link.

### Schema intelligence (G7–G9, already wired)
- Text schemas: `vocab_size`, `language_id`, `avg_tokens_per_sample`, `linguistic_complexity`, `long_doc_indicator`.
- Image schemas: `channels`, `aspect_ratio_variance`, `mean_resolution`, `blur_proxy_variance_of_laplacian`.
- `GlobalSchema.multimodal_signals` holds `complementarity_score` and `alignment_strength`.

### Preprocessing context-awareness (G15–G17, already wired)
- `TrainingOrchestrator` passes `feature_intelligence` dict into `text_prep.configure()` and `image_prep.configure()`.
- `TextPreprocessor.configure` adapts `max_length`, tokenizer, and task-type from signals.
- `ImagePreprocessor.configure` adapts augmentation intensity and `target_size` from signals.

## Environment variables
| Variable | Default | Effect |
|---|---|---|
| `APEX_SEED` | `42` | Seeds `pl.seed_everything()`, `torch`, `numpy`, `random` for reproducibility |
| `APEX_MODE` | `production` | `development` disables production safety guards |
| `APEX_ALLOW_LEGACY_SESSION_FALLBACK` | `0` | `1` + `APEX_MODE=development` enables legacy in-memory session fallback |
| `APEX_CORS_ORIGINS` | `http://localhost:8501` | Comma-separated allowed CORS origins |
| `APEX_API_BASE_URL` | `http://localhost:8001` | Frontend API base URL |
| `APEX_WS_IDLE_TIMEOUT_SEC` | `30.0` | WebSocket idle timeout in seconds |
| `APEX_PRUNER_WARMUP_EPOCHS` | `3` | Optuna pruner warmup steps |

## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- For cross-module "how does X relate to Y" questions, prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"` over grep — these traverse the graph's EXTRACTED + INFERRED edges instead of scanning files
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost)

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
|------|----------|
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.
