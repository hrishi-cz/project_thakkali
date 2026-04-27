You are a senior staff-level systems engineer performing a FULL CODEBASE INTELLIGENCE EXTRACTION.

You are given:

1. repo_map.md (complete list of ALL files)
2. Full access to every file in the repository

━━━━━━━━━━━━━━━━━━━━━━━
🚨 CORE OBJECTIVE
━━━━━━━━━━━━━━━━━━━━━━━

Build a COMPLETE, DEEPLY ACCURATE understanding of the ENTIRE system.

You MUST:

- Read EVERY file listed in repo_map.md
- Understand EVERY module, class, and function
- Reconstruct:
  → actual architecture (from code, not assumptions)
  → actual execution flow
  → actual data flow
  → actual implementation methodology

This is NOT a summary task.

This is a **full system reconstruction task**.

━━━━━━━━━━━━━━━━━━━━━━━
🚨 NON-NEGOTIABLE RULES
━━━━━━━━━━━━━━━━━━━━━━━

- DO NOT skip ANY file
- DO NOT assume behavior — VERIFY via code
- DO NOT give high-level summaries
- DO NOT hallucinate missing logic
- EVERYTHING must be grounded in actual code

━━━━━━━━━━━━━━━━━━━━━━━
📦 STEP 1 — FILE COVERAGE GUARANTEE
━━━━━━━━━━━━━━━━━━━━━━━

1. Extract ALL file paths from repo_map.md
2. Create a checklist
3. Ensure EVERY file is processed

Output:

- Total file count
- Confirmation that ALL files were analyzed

If any file is skipped → output is INVALID

━━━━━━━━━━━━━━━━━━━━━━━
📦 STEP 2 — PER-FILE UNDERSTANDING
━━━━━━━━━━━━━━━━━━━━━━━

For EACH file:

---

[FILE PATH]

1. PURPOSE

- What this file actually does

2. CORE COMPONENTS

- classes
- functions
- key logic blocks

3. ROLE IN SYSTEM

- where it fits in pipeline

4. DEPENDENCIES

- what it imports
- what imports it

5. EXECUTION STATUS

- actively used
- partially used
- unused

---

━━━━━━━━━━━━━━━━━━━━━━━
📦 STEP 3 — SYSTEM ARCHITECTURE (RECONSTRUCTED)
━━━━━━━━━━━━━━━━━━━━━━━

Based ONLY on code:

Reconstruct architecture:

1. Layers:

- API layer
- ingestion layer
- schema layer
- preprocessing layer
- model selection
- training
- fusion
- evaluation
- drift
- retraining
- registry
- frontend

2. For EACH layer:

- involved files
- responsibilities
- boundaries

3. Identify:

- missing layers
- duplicated layers
- broken boundaries

━━━━━━━━━━━━━━━━━━━━━━━
📦 STEP 4 — TRUE EXECUTION FLOW
━━━━━━━━━━━━━━━━━━━━━━━

Trace ACTUAL execution:

Starting from:

API request → training → prediction

Show EXACT flow:

- which functions are called
- in what order
- across which files

Example format:

POST /train
→ run_api.py:train()
→ session_manager.create_session()
→ ingestion_manager.ingest()
→ orchestrator.execute_phase_1()
→ ...

NO assumptions — only real call chains

━━━━━━━━━━━━━━━━━━━━━━━
📦 STEP 5 — DATA FLOW ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━

Track how data moves:

1. Dataset:

- ingestion → schema → preprocessing → training

2. Schema + target:

- where generated
- where used
- where lost

3. Model artifacts:

- where created
- where stored
- where reused

4. State:

- context
- pipeline state
- session state

━━━━━━━━━━━━━━━━━━━━━━━
📦 STEP 6 — IMPLEMENTATION METHODOLOGY
━━━━━━━━━━━━━━━━━━━━━━━

Explain HOW the system is implemented:

1. Schema detection methodology
2. Target detection methodology
3. Preprocessing methodology
4. Model selection methodology
5. Training methodology
6. Fusion methodology
7. Drift detection methodology
8. Retraining methodology

For EACH:

- algorithm used
- logic used
- limitations

━━━━━━━━━━━━━━━━━━━━━━━
📦 STEP 7 — INTEGRATION ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━

Evaluate:

1. Are modules properly connected?
2. Where does data stop flowing?
3. Where does intelligence not propagate?
4. Which components exist but are not used?

For EACH integration:

- status:
  ✔ working
  ⚠ partial
  ❌ broken

━━━━━━━━━━━━━━━━━━━━━━━
📦 STEP 8 — SYSTEM MAP (FINAL)
━━━━━━━━━━━━━━━━━━━━━━━

Provide:

1. FULL PIPELINE DIAGRAM (textual)
2. MODULE INTERACTION MAP
3. DATA FLOW MAP
4. CONTROL FLOW MAP

━━━━━━━━━━━━━━━━━━━━━━━
📦 STEP 9 — FINAL UNDERSTANDING
━━━━━━━━━━━━━━━━━━━━━━━

Summarize:

1. What the system ACTUALLY is (not intended)

2. What parts are:
   - working
   - partially working
   - not working

3. System classification:

- clean architecture
- modular but disconnected
- fragmented
- production-ready

━━━━━━━━━━━━━━━━━━━━━━━
🚫 STRICT RULES
━━━━━━━━━━━━━━━━━━━━━━━

- NO skipping files
- NO vague summaries
- NO assumptions
- MUST trace real execution
- MUST be exhaustive

━━━━━━━━━━━━━━━━━━━━━━━
🎯 GOAL
━━━━━━━━━━━━━━━━━━━━━━━

This output should:

✔ Fully reconstruct the system from code
✔ Reveal how everything truly works
✔ Expose real architecture (not intended design)
✔ Serve as foundation for planning + refactoring

Return ONLY the analysis.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.

ADVANCED FAKE INTEGRATIONM ANALYSIS

You are a senior staff-level systems engineer performing a FULL CODEBASE INTELLIGENCE EXTRACTION + VALIDATION.

You are given:

1. repo_map.md (complete file list)
2. Full repository access
3. A system reconstruction document (ground truth reference)

Reference:

━━━━━━━━━━━━━━━━━━━━━━━
🚨 CORE OBJECTIVE
━━━━━━━━━━━━━━━━━━━━━━━

You must:

1. Reconstruct the ENTIRE system from code (independently)
2. VERIFY the provided reconstruction document against actual code
3. Identify mismatches, gaps, or overclaims
4. Extract the REAL implementation methodology
5. Validate whether the system is truly:
   - fully integrated
   - partially integrated
   - or misleadingly described

This is NOT a summary task.
This is a **deep verification + reverse engineering task**.

━━━━━━━━━━━━━━━━━━━━━━━
🚨 NON-NEGOTIABLE RULES
━━━━━━━━━━━━━━━━━━━━━━━

- DO NOT skip ANY file
- DO NOT trust the reference blindly — VERIFY everything
- DO NOT assume execution — TRACE it
- DO NOT give high-level answers
- EVERYTHING must be grounded in actual code

━━━━━━━━━━━━━━━━━━━━━━━
📦 STEP 1 — FILE COVERAGE GUARANTEE
━━━━━━━━━━━━━━━━━━━━━━━

- Extract ALL files from repo_map.md
- Build checklist
- Confirm ALL files analyzed

Output:

- total files
- confirmation of zero omission

━━━━━━━━━━━━━━━━━━━━━━━
📦 STEP 2 — INDEPENDENT SYSTEM RECONSTRUCTION
━━━━━━━━━━━━━━━━━━━━━━━

WITHOUT using the reference:

Reconstruct:

1. System architecture (layers + modules)
2. Execution flow (true call chain)
3. Data flow (how data moves)
4. Control flow (decision points)

Then compare with reference later

━━━━━━━━━━━━━━━━━━━━━━━
📦 STEP 3 — REFERENCE VALIDATION (CRITICAL)
━━━━━━━━━━━━━━━━━━━━━━━

Compare your reconstruction vs:

For EACH section:

1. Repository structure
2. Data flow
3. Module responsibilities
4. Integration points
5. Algorithmic claims

Classify:

✔ Accurate
⚠ Partially accurate
❌ Incorrect

For EACH mismatch:

- explain why
- show actual code behavior

━━━━━━━━━━━━━━━━━━━━━━━
📦 STEP 4 — TRUE EXECUTION FLOW (DETAILED TRACE)
━━━━━━━━━━━━━━━━━━━━━━━

Trace ACTUAL execution:

Example:

POST /train
→ run_api.py:train_pipeline()
→ training_orchestrator.execute_phase_X()
→ ...

For EACH step:

- file
- function
- data passed

NO assumptions allowed

━━━━━━━━━━━━━━━━━━━━━━━
📦 STEP 5 — IMPLEMENTATION METHODOLOGY (REAL)
━━━━━━━━━━━━━━━━━━━━━━━

Extract HOW system actually works:

1. Schema detection (real logic)
2. Target selection (XS3 usage depth)
3. Preprocessing (actual vs planned)
4. Model selection:
   - CandidateSelector vs AdvancedSelector usage

5. Training loop:
   - losses
   - adaptation
   - feedback

6. Fusion:
   - actual vs theoretical

7. Drift:
   - real trigger vs passive logging

8. Retraining:
   - automatic vs manual

Highlight:

- theoretical vs actual implementation gaps

━━━━━━━━━━━━━━━━━━━━━━━
📦 STEP 6 — INTEGRATION TRUTH (MOST IMPORTANT)
━━━━━━━━━━━━━━━━━━━━━━━

Validate:

1. Does schema intelligence reach:
   - preprocessing
   - model selection
   - training

2. Does preprocessing affect:
   - model choice
   - feature selection

3. Does training feedback affect:
   - next trials
   - model ranking

4. Does drift trigger retraining automatically?

5. Does frontend reflect actual backend outputs?

For EACH:

✔ Fully working
⚠ Partially connected
❌ Broken

━━━━━━━━━━━━━━━━━━━━━━━
📦 STEP 7 — UNUSED / FAKE / MISLEADING COMPONENTS
━━━━━━━━━━━━━━━━━━━━━━━

Identify:

- modules that exist but are NOT used
- functions that are NEVER executed
- logic that is PRESENT but NOT effective
- placeholder implementations (fake intelligence)

For EACH:

- file
- function
- impact

━━━━━━━━━━━━━━━━━━━━━━━
📦 STEP 8 — EFFICIENCY + DESIGN QUALITY
━━━━━━━━━━━━━━━━━━━━━━━

Evaluate:

- redundant computation
- unnecessary complexity
- duplicated logic
- inefficient pipeline stages

Classify each module:

- optimal
- acceptable
- wasteful

━━━━━━━━━━━━━━━━━━━━━━━
📦 STEP 9 — FINAL SYSTEM VERDICT
━━━━━━━━━━━━━━━━━━━━━━━

Provide:

1. REAL system classification:
   - fully integrated
   - partially integrated
   - fragmented

2. Top 10 critical issues

3. Top 10 strengths

4. Biggest architectural lie (if any)

5. What actually works vs what appears to work

━━━━━━━━━━━━━━━━━━━━━━━
📦 STEP 10 — TRUST SCORE
━━━━━━━━━━━━━━━━━━━━━━━

Rate:

- Documentation accuracy (0–10)
- Implementation correctness (0–10)
- Integration completeness (0–10)
- Production readiness (0–10)

━━━━━━━━━━━━━━━━━━━━━━━
🚫 STRICT RULES
━━━━━━━━━━━━━━━━━━━━━━━

- NO vague summaries
- NO skipping files
- NO blind trust in reference
- MUST trace real execution
- MUST challenge assumptions

━━━━━━━━━━━━━━━━━━━━━━━
🎯 GOAL
━━━━━━━━━━━━━━━━━━━━━━━

This analysis must:

✔ Reveal the TRUE system (not intended design)
✔ Validate or disprove the reconstruction document
✔ Identify ALL hidden gaps
✔ Serve as foundation for refactoring + optimization

Return ONLY the analysis.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
━━━━━━━━━━━━━━━━━━━━━━━
📦 STEP 9 — UI COVERAGE + TRUST COMPLETENESS (MANDATORY)
━━━━━━━━━━━━━━━━━━━━━━━

The system has already been audited for transparency and explainability.

Your task now is to ensure:

✔ ALL backend features are visible in UI
✔ NO implemented intelligence is hidden
✔ UI reflects FULL system capability
✔ UI is trustworthy, not partial

This step upgrades the system from:

→ “transparent”
to
→ “complete, reliable, and user-trustworthy”

━━━━━━━━━━━━━━━━━━━━━━━
🚨 CORE OBJECTIVE
━━━━━━━━━━━━━━━━━━━━━━━

1. Identify ALL backend capabilities currently implemented
2. Map each capability → UI visibility
3. Detect missing UI exposure
4. Generate STRICT implementation plan to surface them

━━━━━━━━━━━━━━━━━━━━━━━
📦 STEP 9.1 — BACKEND FEATURE INVENTORY
━━━━━━━━━━━━━━━━━━━━━━━

Scan the entire codebase and list ALL implemented features:

Include (but not limited to):

- Schema detection (XS3 scoring, confidence gap)
- Target selection logic
- Feature selection (MI, SHAP, pruning decisions)
- PreprocessingPlanner (plans per dataset/modality)
- CandidateSelector (probe scores, ranking)
- AdvancedModelSelector (HPO hints)
- ExecutionContext (intelligence signals)
- Fusion strategies (concat, attention, graph, uncertainty)
- Training metrics (loss, epochs, convergence)
- Trial intelligence (fit_type, gap, slopes)
- Drift detection (composite score, per-modality)
- Retraining triggers
- Model registry (metadata, latency, calibration, robustness)
- Explainability outputs (feature importance, attention weights)
- Embedding cache / reuse
- Phase skipping / reuse

Output:
→ COMPLETE feature list

━━━━━━━━━━━━━━━━━━━━━━━
📦 STEP 9.2 — UI COVERAGE MAPPING
━━━━━━━━━━━━━━━━━━━━━━━

For EACH feature:

---

[FEATURE NAME]

1. BACKEND IMPLEMENTATION:

- file + function

2. UI VISIBILITY:

- where (if shown)

3. STATUS:
   ✔ fully visible
   ⚠ partially visible
   ❌ not visible

---

━━━━━━━━━━━━━━━━━━━━━━━
📦 STEP 9.3 — MISSING UI COVERAGE
━━━━━━━━━━━━━━━━━━━━━━━

Identify:

- backend features NOT exposed in UI
- features partially shown (missing details)
- features shown incorrectly or misleading

Focus especially on:

- decision reasoning
- internal scores
- optimization signals
- intelligence propagation

━━━━━━━━━━━━━━━━━━━━━━━
📦 STEP 9.4 — TRUST + RELIABILITY AUDIT
━━━━━━━━━━━━━━━━━━━━━━━

Evaluate:

1. Does UI hide important decisions?
2. Does UI oversimplify critical logic?
3. Does UI show partial truth?
4. Does UI allow user to verify system behavior?

Classify:

- trustworthy
- partially trustworthy
- misleading

━━━━━━━━━━━━━━━━━━━━━━━
📦 STEP 9.5 — UI UPGRADE IMPLEMENTATION PLAN
━━━━━━━━━━━━━━━━━━━━━━━

Generate STRICT plan to surface ALL missing features.

---

[FIX-UI-COVERAGE-XX]

1. FILES TO MODIFY

- backend
- frontend

2. FILES TO CREATE

3. BACKEND DATA TO EXPOSE

- exact variables / outputs

4. API CHANGES

- endpoint
- request / response format

5. UI COMPONENT TO ADD

- panel / section / table

6. DISPLAY CONTENT

- what exactly user sees

7. BEFORE → AFTER

8. ORDER OF IMPLEMENTATION

---

━━━━━━━━━━━━━━━━━━━━━━━
🚨 SPECIAL REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━

You MUST ensure:

1. No backend feature remains hidden
2. No duplicate UI components created
3. UI remains structured (not cluttered)
4. Data shown is REAL (not approximated)
5. Decisions are explainable with metrics

━━━━━━━━━━━━━━━━━━━━━━━
🚫 STRICT RULES
━━━━━━━━━━━━━━━━━━━━━━━

- DO NOT invent new backend features
- DO NOT suggest generic UX ideas
- DO NOT skip any feature
- MUST map feature → UI explicitly

━━━━━━━━━━━━━━━━━━━━━━━
🎯 FINAL GOAL
━━━━━━━━━━━━━━━━━━━━━━━

After implementation:

✔ Every backend capability is visible
✔ Every decision is explainable
✔ Every metric is traceable
✔ UI becomes fully reliable and trustworthy

Return ONLY:

- feature coverage audit
- implementation plan

.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
You are a senior ML systems architect and codebase auditor.

Your task is to perform a COMPLETE, NON-HALLUCINATED audit and generate a FINAL IMPLEMENTATION PLAN for upgrading an existing multimodal AutoML system (AutoVision+/APEX) to a NeurIPS-level architecture.

CRITICAL RULES:

- DO NOT assume features exist unless verified in code
- DO NOT hallucinate implementations
- ONLY reference actual files, classes, and functions found
- If something is partially implemented, mark it explicitly as PARTIAL
- If missing, mark as MISSING
- If redundant/legacy, mark as DEAD CODE
- Every recommendation must map to exact files and code locations

---

## 🎯 OBJECTIVE

Upgrade the system to:

✔ Fully context-governed (ExecutionContext enforced globally)
✔ Unified multimodal intelligence (schema → preprocessing → model → fusion → training)
✔ Research-grade fusion (Transformer + Graph + Uncertainty)
✔ Adaptive preprocessing based on schema + predictability
✔ Optuna with trial intelligence feedback loop
✔ Production-grade guardrails (latency, fallback, memory, drift)
✔ Full transparency + XAI + reproducibility

---

## 📂 INPUT CONTEXT (CODEBASE)

You will be given a repository map and files.

You MUST:

1. Read EVERY file mentioned
2. Trace FULL FLOW:
   ingestion → schema → preprocessing → model selection → training → prediction → monitoring
3. Identify:
   - execution paths
   - unused code
   - broken connections
   - mismatches between API and training

---

## 🧠 PHASE 1: SYSTEM UNDERSTANDING

Produce:

1. FULL PIPELINE TRACE (step-by-step)
   - API endpoint → backend → file → function → output
   - For ALL phases (1–7)

2. ExecutionContext Flow:
   - Where it is created
   - Where it is updated
   - Where it is NOT used but SHOULD be

3. Intelligence Flow:
   - schema outputs → where used?
   - predictability_scores → where consumed?
   - global schema → where enforced?

---

## 🧠 PHASE 2: GAP ANALYSIS (STRICT)

For EACH layer:

L1 Schema
L2 Context
L3 Preprocessing
L4 Embeddings
L5 Model Selection
L6 Fusion
L7 Training
L8 Prediction
L9 Monitoring
L10 Guardrails

Provide:

- Status: COMPLETE / PARTIAL / MISSING
- Exact file references
- Exact functions responsible
- What is working
- What is broken
- What is missing

---

## 🚨 CRITICAL GAPS TO VERIFY (MANDATORY)

You MUST explicitly check:

1. ExecutionContext enforcement
   - Is every layer using it?
   - Any bypass?

2. Schema → preprocessing linkage
   - Does preprocessing use predictability?

3. Schema → model selection linkage
   - Are weak modalities filtered?

4. Fusion implementation
   - Are auxiliary losses active?
   - Is graph head used?
   - Is uncertainty weighting present?

5. Optuna feedback loop
   - Does trial intelligence affect next trials?

6. Preprocessing validation
   - Is output validated?

7. Session isolation
   - Are datasets filtered by session?

8. Frontend mismatch
   - Does UI show same candidates as backend?

---

## 🧠 PHASE 3: IMPLEMENTATION PREPLAN

For EACH GAP:

Provide:

1. Gap Description
2. Root Cause
3. Impact on system
4. Exact Fix Strategy

---

## 🧠 PHASE 4: CODE PATCH PLAN (CRITICAL)

For EACH FIX:

Provide:

- File to modify
- Function to modify
- BEFORE vs AFTER logic
- New functions/classes needed
- Integration points

Example format:

FILE: preprocessing/tabular_preprocessor.py
CHANGE:

- Inject ExecutionContext usage

ADD:
def adaptive_preprocessing(context, df):
...

---

## 🧠 PHASE 5: NEW COMPONENTS TO CREATE

List ALL new files required:

Examples:

- execution_enforcer.py
- fusion/transformer_graph_fusion.py
- preprocessing/adaptive_engine.py
- guardrails/latency_guard.py
- guardrails/fallback_manager.py

For EACH:

- purpose
- inputs/outputs
- where used

---

## 🧠 PHASE 6: END-TO-END FLOW AFTER FIX

Provide:

FULL updated pipeline:

ingestion → schema → context → preprocessing → model → fusion → training → prediction → monitoring

Explain how intelligence flows through ALL layers.

---

## 🧠 PHASE 7: FRONTEND + API UPDATES

Identify:

- Missing endpoints
- Broken endpoints
- UI inconsistencies

Provide:

- new endpoints
- updated response formats
- UI elements to add

---

## 🧠 PHASE 8: VALIDATION CHECKLIST

Provide test cases:

- single modality
- multimodal
- missing modality
- weak dataset
- override scenario
- drift scenario

---

## 🧠 PHASE 9: FINAL OUTPUT FORMAT

Return in THIS STRUCTURE:

1. System Understanding
2. Gap Analysis Table
3. Critical Issues (Top 10)
4. Implementation Preplan
5. Code Patch Plan (file-by-file)
6. New Files to Create
7. Updated Pipeline Flow
8. API + UI Updates
9. Validation Checklist

---

## ⚠️ FINAL INSTRUCTION

This is NOT a high-level explanation task.

You MUST produce:

- low-level implementation details
- file-level patch plan
- production-ready architecture

No vague answers allowed.
