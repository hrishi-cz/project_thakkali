# AutoVision+ Product Engineering Doctrine

This document serves as the living memory of the system. It records persistent engineering knowledge, absolute invariants, system design lessons, and rules discovered during the development lifecycle.

---

## I. Core System Principles

AutoVision+ is built on five foundational pillars:

1. **Deterministic Execution:** Seeded, reproducible multimodal ML pipelines.
2. **Schema-Safe Inference:** Zero tolerance for training-serving skew.
3. **Hardware-Aware Training:** Dynamic resource allocation and out-of-memory (OOM) fallbacks.
4. **GPU Resource Safety:** Aggressive, unconditional VRAM cleanup.
5. **Zero External Dependencies:** No Redis, no Kubernetes—pure Pythonic infrastructure.

---

## II. System Invariants (The Laws of AutoVision+)

1. **The Feature Source of Truth:** The Preprocessing layer defines the canonical, final feature schema.
2. **The API Contract:** The FastAPI backend exposes this exact schema dynamically via `/model-info/{model_id}`.
3. **Frontend Discipline:** The Streamlit UI must ingest `/model-info` and render **only** the surviving effective features. It must never request stripped columns.

---

## III. Observability & Telemetry Standards

Silent execution is a critical system failure. All long-running pipeline operations must emit structured logs to the SQLite WAL tracking:

- Phase Initialization & State Transitions
- Epoch Metrics & HPO Pruning Events
- Dataset Ingestion Failures
- Model Registry Writes & Serialization

---

## IV. Artifact Versioning Protocol

Serialized artifacts must maintain strict backwards compatibility or explicit version increments (`model_v1`, `model_v2`).
A major version increment is mandatory when:

- The schema format or data contract changes.
- The preprocessing pipeline logic is altered.
- The base encoder architecture is modified.
- Metadata storage structures change.

---

## V. System Design Learnings

_(AI Agent: Capture high-level architectural and design lessons here.)_

### [2026-03-13] — Streamlit Widget Key Mutation Rule

- **Trigger:** `StreamlitAPIException: st.session_state.phase_radio cannot be modified after the widget with key phase_radio is instantiated`. Navigation buttons set `st.session_state["phase_radio"]` AFTER the radio widget rendered, violating Streamlit's widget lifecycle.
- **Rule:** `st.session_state[key]` must be set BEFORE the widget with that key is instantiated, never after. Navigation buttons that change `workflow_stage` should only modify `workflow_stage` and call `st.rerun()` — the pre-widget sync block handles `phase_radio` on the next render.
- **Impact:** Eliminates the `StreamlitAPIException` and ensures navigation state is always consistent with the rendered widget.

### [2026-03-13] — Streamlit Rerun Button Handler Reachability

- **Trigger:** After training completes, `training_task_id` is set to `None`. On rerun, the `if task_id is None:` early-return renders the "Start Training" form, making the "Next: Monitoring" button handler at line 1353 unreachable dead code.
- **Rule:** When a workflow phase has a "completed" state that persists beyond a single render (e.g., training results), store the completion data in `session_state` and check for it in the `task_id is None` branch. Extract the completion view into a reusable function callable from both the live completion path and the cached-result path.
- **Impact:** Users can navigate Phase 5→6→7 after training completes, instead of being stuck on the start form.

### [2026-03-13] — Optuna HPO: Epochs Should Not Be a Hyperparameter

- **Trigger:** HPO sampled epochs as a tunable param (e.g., 30-50 range), wasting trial budget exploring epoch counts. Combined with EarlyStopping, the model often stopped at epoch 8 regardless of the sampled 45 — making 37 of 45 scheduled epochs meaningless.
- **Rule:** Use a fixed `max_epochs` ceiling from the PDF epoch matrix and let EarlyStopping + SmartTrainingCallback control actual duration. HPO should only tune parameters that directly affect model quality (LR, dropout, weight_decay, fusion strategy).
- **Impact:** Every trial now explores the same epoch space, allowing Optuna to focus budget on parameters that actually differentiate model quality.

### [2026-03-13] — Adaptive Pruner Selection

- **Trigger:** HyperbandPruner with 3 trials and `reduction_factor=3` could never prune (needs enough candidates per rung). MedianPruner with 50 trials is less aggressive than it could be.
- **Rule:** Select pruner based on trial count: `>=20` trials → HyperbandPruner (aggressive rung-based SHA), `<20` trials → MedianPruner (robust with fewer candidates), 1 trial → NopPruner. Always set `max_resource` to the fixed `max_epochs` value, not a sampled HP.
- **Impact:** Pruning strategy matches the available trial budget, maximizing GPU savings without losing statistical robustness.

### [2026-03-13] — Modality-Aware Fusion Strategy

- **Trigger:** Fusion strategy was selected purely based on GPU memory (`>=8GB → attention`), ignoring which modalities are active. Tabular+text concatenation works well, but image+text benefits significantly from attention-based cross-modal weighting.
- **Rule:** Default fusion should be modality-aware: image+text → attention (if GPU allows), all other combinations → concatenation. GPU memory remains a hard constraint (attention requires >=8GB).
- **Impact:** Image+text workloads get the more expressive attention fusion automatically, while simpler modality combinations use the lower-overhead concatenation.

---

## VI. Reflect and Record Log

_(AI Agent: Append new engineering insights, workflow bug fixes, and operational rules below.)_

### [2026-03-12]

- **Trigger:** UI requested input for features that were automatically removed during preprocessing.
- **Rule:** The frontend must dynamically query and render only the `effective_features` returned by the `/model-info` endpoint.
- **Impact:** Completely prevents schema/UI desynchronization and eliminates user confusion.

### [2026-03-12] — Catalogue-Rationale Consistency

- **Trigger:** `TABULAR_ENCODERS` in `advanced_selector.py` lacked a `"sota"` key, causing `None` encoder selection for GPUs ≥12 GB. Rationale strings referenced phantom encoders (ViT-Base, RoBERTa-large, TabNet) that have no implementation code.
- **Rule:** Every tier key referenced by selection logic must exist in its encoder catalogue. Rationale strings must reference only encoders present in the JIT registry — never aspirational or removed models.
- **Impact:** Eliminates silent `None` returns from encoder selection and prevents misleading selection explanations.

### [2026-03-12] — Phantom Code Audit

- **Trigger:** `config/hyperparameters.py` listed encoder names (ViT-Base, EfficientNet-B0, RoBERTa-large, DistilBERT, TabNet, FT-Transformer) and a fusion strategy ("weighted") that have no backing implementation anywhere in the codebase.
- **Rule:** Configuration option lists, Optuna search spaces, and preset dictionaries must only contain values that resolve to implemented code paths. Run a dead-reference audit whenever encoder registries change.
- **Impact:** Prevents HPO from exploring non-functional encoder combinations and eliminates user-facing options that would crash at runtime.

### [2026-03-12] — Frontend-Backend Parity

- **Trigger:** Frontend displayed disabled "Deploy to Production" and "Download Model" buttons with no backend support. Inference warnings (missing/extra columns) were silently swallowed and never shown to users.
- **Rule:** Every UI control must map to a functional API endpoint. Every API response field that affects user decisions must be surfaced in the frontend. Dead buttons and silent data transformations are UX defects.
- **Impact:** Users can now download and delete models from the UI, and see explicit warnings when prediction inputs don't match the training schema.

### [2026-03-12] — Smart Training Callback Architecture

- **Trigger:** EarlyStopping provides a single "patience stalled" signal but does not differentiate WHY training stopped. Users cannot distinguish convergence (good) from overfitting (bad) from flatline (meaningless).
- **Rule:** Training callbacks must classify stop reasons into distinct categories (flatline, overfitting, underfitting, convergence, completed) and actively trigger `trainer.should_stop = True` rather than just logging post-hoc labels. Underfitting detection should dynamically extend `trainer.fit_loop.max_epochs` (capped, once only) rather than failing fast.
- **Impact:** Users see actionable stop reason explanations. Underfitting models get extended training windows automatically, improving final accuracy without manual re-configuration.

### [2026-03-12] — Cross-Layer Key Alignment Audit

- **Trigger:** Frontend read encoder names from `best.get("encoders", {})` (nested dict) but backend returned flat keys `image_encoder`, `text_encoder`, `tabular_encoder`. Hardware info used `vram_mb` and `method` but backend returns `gpu_memory_gb` and `cuda_device`. Both mismatches caused silent "N/A" display.
- **Rule:** When adding new API response fields, trace the full data flow from source (orchestrator/selector) through API endpoint response to frontend rendering. Verify key names match at every layer boundary. Add this as a checklist item in the patch verification protocol.
- **Impact:** Eliminates silent "N/A" display for data that exists in the backend but is read under wrong keys in the frontend.

### [2026-03-12] — Metadata Richness for Model Registry Display

- **Trigger:** `_summarize_all_phases()` only captured `duration_seconds` and `status` per phase. The model registry UI could not display training metrics (val_loss, encoders, hyperparameters) because this data was discarded at serialization time.
- **Rule:** Phase summary serialization must carry forward key metrics that downstream consumers (registry UI, drift monitoring, model comparison) need. Define explicit "carry-forward key lists" per phase rather than making the summary opaque.
- **Impact:** Model registry now displays full training details (encoders, metrics, hyperparameters, stop reason) without requiring separate API calls.

---

## VII. Product Workflow Learnings

_(AI Agent: Capture product-level insights that improve usability.)_

### [2026-03-14]

- **Trigger:** Users struggled to understand which fields were required for prediction.
- **Rule:** Prediction UI must display schema guidance explaining each feature type.
- **Impact:** Improves user workflow clarity and reduces invalid prediction inputs.

---

## VIII. System Evolution Principle

When adding new capabilities to AutoVision+, prefer extending existing components rather than introducing new subsystems.
**Examples:**

- Extend preprocessing instead of adding new data pipelines.
- Extend the training orchestrator instead of creating parallel workflows.
- Extend API endpoints instead of creating redundant services.

This keeps the system architecture cohesive and prevents unnecessary complexity.

---

## IX. Telemetry Learnings

_(AI Agent: Capture observability improvements.)_

### [YYYY-MM-DD]

- **Trigger:** Debugging training failures required manual log inspection.
- **Rule:** All pipeline phase transitions must emit structured logs.
- **Impact:** Enables faster failure diagnosis and improves system observability.

---

## X. Documentation Evolution Principle

Whenever architecture or workflows change, documentation must evolve simultaneously.
The `README.md` must **always** reflect:

- The current pipeline architecture.
- System capabilities.
- Onboarding instructions.

## XI. Deterministic Execution Rule

All training runs must initialize deterministic seeds across the entire scientific stack to guarantee reproducibility:

- Python `random`
- `NumPy`
- PyTorch CPU
- PyTorch CUDA

The seed must be permanently stored in the model registry metadata.

---

## XII. Efficiency Audit Learnings

_(AI Agent: Capture design lessons from pipeline efficiency audits.)_

### [2026-03-12] — Canonical Encoder Naming Is a Cross-Cutting Contract

- **Trigger:** The same encoder had 3-4 different name strings across `advanced_selector.py`, `jit_encoder_selector.py`, and `hyperparameters.py` (e.g., `"MobileNetV3"` vs `"MobileNetV3-Small"` vs `"mobilenet_v3_small"`). Display names, registry keys, and HuggingFace model IDs were conflated.
- **Rule:** Establish a single canonical name set (matching the JIT registry names, the most specific). Use `CANONICAL_TO_HF` mapping for weight loading. HuggingFace IDs belong only inside factory functions — never in configuration or display contracts.
- **Impact:** Eliminates silent mismatches when Phase 4 tier names are compared to JIT profiler names or UI display strings.

### [2026-03-12] — JIT Selection Must Respect Heuristic Tier Signals

- **Trigger:** `JITEncoderSelector.select()` unconditionally picked `TABULAR_REGISTRY[0]` (GRN) regardless of Phase 4's heuristic recommendation. For small datasets on low-memory GPUs, the heuristic correctly chose MLP, but JIT silently overrode it.
- **Rule:** When two selection systems exist (heuristic + hardware-profiled), the hardware profiler should narrow the heuristic recommendation, not ignore it. Pass tier signals through the selection API.
- **Impact:** Small-dataset users now get the lighter MLP tabular encoder when Phase 4 recommends it.

### [2026-03-12] — GPU Cleanup Is Non-Negotiable at Every Inference Boundary

- **Trigger:** `predict_batch()` and `generate_explanations()` in the inference engine lacked `try/finally` GPU cleanup blocks, violating the CLAUDE.md V.1 mandate. Only the approximate token attribution fallback had proper cleanup.
- **Rule:** Every public method that touches GPU tensors must unconditionally run `gc.collect()` + `torch.cuda.empty_cache()` in a `finally` block. Audit all inference-layer entry points when adding new prediction modes.
- **Impact:** Prevents VRAM accumulation across repeated inference calls, especially in long-running API servers.

### [2026-03-12] — Monitoring Without Persistence Is Observability Theater

- **Trigger:** `PerformanceTracker` stored all prediction metrics in an unbounded Python list. History was lost on restart, leaked memory, and lacked F1/precision/recall for classification. Binary classification was silently treated as regression (MSE/MAE on 0/1 values).
- **Rule:** Any monitoring component must persist to durable storage (SQLite). Bound history size. Detect problem type from array shape and value distribution. Always compute F1/precision/recall for classification tasks.
- **Impact:** Performance metrics survive restarts, classification models get proper evaluation metrics, and memory usage is bounded.

---

## XIII. JIT Registry Expansion & System Benchmarks

1. **Registry Gap Analysis Rule:** When expanding an encoder registry, audit the capacity spectrum for gaps exceeding 5x between adjacent entries. Each new model must fill a measurable gap, not duplicate an existing capacity tier. Rejected candidates: RegNet-Y-400MF (too close to MobileNetV3-Small), Swin-Tiny (same capacity as ConvNeXt), RoBERTa-base (same as BERT-base).

2. **JIT Profiler Transparency Rule:** JIT profiler results (VRAM budget, peak memory, per-encoder profiles, selection rationale) must flow through the API to the frontend. If Phase 5 produces diagnostic data, it must be forwarded in `final_result` — otherwise the monitoring dashboard has no visibility.

3. **Per-Encoder Profiling Accumulation:** When the constrained optimizer iterates through encoder combinations, accumulate a profiling entry for every candidate (selected, rejected, or error). This data powers the System Benchmarks tab and enables post-hoc analysis of why specific encoders were rejected.

4. **Prediction Latency Capture:** Every inference request should capture end-to-end latency (submit → poll → result) and store it in prediction history. This enables the System Benchmarks tab to compute P50/P95/mean latency without any backend changes.

---

## XIV. Captum XAI Multi-Method Expansion

1. **Image Encoder Dispatch Rule:** `_load_image_encoder()` must read the actual encoder name from `metadata.json > phases_summary.TRAINING.encoder_selection.image_encoder` instead of hardcoding ResNet-50. A dispatch table maps canonical names to factory functions. This ensures GradCAM targets the correct convolutional layer and state_dict loading uses the right architecture.

2. **GradCAM Layer Detection Contract:** Each CNN-based vision encoder has a known "last conv layer" for GradCAM: ResNet-50 → `backbone.layer4[-1]`, MobileNetV3/EfficientNet/ConvNeXt → `backbone.features[-1]`. ViT-B-16 has no conv layers — GradCAM cannot be applied and the system must fall back to pixel-level IntegratedGradients automatically.

3. **Multi-Method Factory Pattern:** Captum attribution methods share a common interface (`attribute(inputs, ...)`) but differ in required arguments. A factory method (`_create_tabular_attribution`) maps string identifiers to Captum classes (IG, GradientShap, Saliency, Occlusion, FeatureAblation, NoiseTunnel). Method-specific argument construction (e.g., `sliding_window_shapes` for Occlusion, `nt_samples` for SmoothGrad) is handled in the caller.

4. **Attention Weight Extraction without Captum:** `AttentionFusion` attention weights can be extracted by replaying the forward pass through `fusion.projections` + `fusion.attention_scoring` + `softmax`. This does NOT require Captum — it uses the module's own projection/scoring layers directly. Only works when the model uses AttentionFusion (not ConcatenationFusion).
