# APEX Refactoring: Before & After Comparison

**Visual guide showing the architectural transformation**

---

## 1. ExecutionContext Flow

### BEFORE ❌

```
┌──────────────────────────────────────────────────────────┐
│                    FRONTEND                              │
│  Streamlit UI calls API endpoints                       │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│              API Layer (api/)                            │
│  ┌────────────────────────────────────────┐             │
│  │  ExecutionContext v1 (frontend)        │             │
│  │  - schema_result                       │             │
│  │  - target_candidates                   │             │
│  │  - preprocessing_plan                  │             │
│  └────────────────────────────────────────┘             │
│                                                          │
│  Schema Service → Target Service → Preprocessing        │
└────────────────────────┬─────────────────────────────────┘
                         │
                         │ [INTELLIGENCE LOST HERE] ❌
                         │
┌────────────────────────▼─────────────────────────────────┐
│            Backend Pipeline (pipeline/)                  │
│  ┌────────────────────────────────────────┐             │
│  │  ExecutionContext v2 (backend)         │             │
│  │  - schema (EMPTY!)                     │             │
│  │  - modality_presence (EMPTY!)          │             │
│  │  - model_candidates (EMPTY!)           │             │
│  └────────────────────────────────────────┘             │
│                                                          │
│  Model Selection → Training (with no schema info!)      │
└──────────────────────────────────────────────────────────┘

PROBLEM: Model selection has no schema, no target info, no preprocessing plan!
```

### AFTER ✅

```
┌──────────────────────────────────────────────────────────┐
│                    FRONTEND                              │
│  Streamlit UI calls API endpoints                       │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│              API Layer (api/)                            │
│  Thin orchestration only - no business logic            │
│                                                          │
│  FastAPI endpoints → core/orchestrator                  │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│         Orchestrator (core/orchestrator.py)              │
│  Coordinates all phases, manages single context         │
│                                                          │
│  load_or_create_context(session_id)                     │
│    ↓                                                     │
│  ┌──────────────────────────────────────────┐           │
│  │  ExecutionContext (UNIFIED)              │           │
│  │  Used by ALL 8 phases:                   │           │
│  │  - Phase 1: Ingestion                    │           │
│  │  - Phase 2: Schema Detection             │           │
│  │  - Phase 3: Target Detection             │           │
│  │  - Phase 4: Global Aggregation           │           │
│  │  - Phase 5: Preprocessing                │           │
│  │  - Phase 6: Model Selection ✅           │           │
│  │  - Phase 7: Training ✅                  │           │
│  │  - Phase 8: Monitoring ✅                │           │
│  └──────────────────────────────────────────┘           │
│                                                          │
│  save_context(ctx) → database/context_db.py             │
└──────────────────────────────────────────────────────────┘

SUCCESS: All phases use the same context with complete intelligence!
```

---

## 2. Service Layer Architecture

### BEFORE ❌

```
API Endpoints (run_api.py)
    ↓
┌─────────────────────────────────────────────┐
│  API Service Wrappers (DUPLICATE LOGIC)     │
├─────────────────────────────────────────────┤
│  schema_detection_service.py                │
│    - ContextAwareSchemaDetector             │
│    - Wraps COGMASchemaDetector              │
│    - Updates api/ExecutionContext           │
│                                             │
│  target_detection_service.py                │
│    - ContextAwareTargetDetector             │
│    - Reimplements ranking                   │
│    - Updates api/ExecutionContext           │
│                                             │
│  global_aggregation_service.py              │
│    - GlobalAggregationService               │
│    - Computes compatibility                 │
│    - Updates api/ExecutionContext           │
│                                             │
│  preprocessing_service.py                   │
│    - PreprocessingService                   │
│    - Plans preprocessing                    │
│    - Updates api/ExecutionContext           │
└─────────────────────────────────────────────┘
    ↓
Core Modules (data_ingestion/)
    - schema_detector.py (COGMA)
    - target_validator.py
    - integrator.py

PROBLEM:
- 4 wrapper files with no value
- Duplicate logic in wrappers + core
- Maintenance burden (change logic in 2 places)
```

### AFTER ✅

```
API Endpoints (run_api.py)
    ↓
┌─────────────────────────────────────────────┐
│  Orchestrator (core/orchestrator.py)        │
├─────────────────────────────────────────────┤
│  execute_phase_2_schema(ctx, data_map)      │
│    ↓ directly calls                         │
│    COGMASchemaDetector.detect_schema()      │
│    ↓ updates                                │
│    ctx.dataset_profiles[id].schema_result   │
│                                             │
│  execute_phase_3_target(ctx, data_map)      │
│    ↓ directly calls                         │
│    TargetValidator.validate()               │
│    ↓ updates                                │
│    ctx.dataset_profiles[id].chosen_target   │
│                                             │
│  execute_phase_4_aggregation(ctx, data_map) │
│    ↓ directly aggregates                    │
│    ctx.global_schema, ctx.global_target     │
│                                             │
│  execute_phase_5_preprocessing(ctx, data)   │
│    ↓ directly plans                         │
│    ctx.dataset_profiles[id].preprocessing   │
└─────────────────────────────────────────────┘
    ↓
Core Modules (data_ingestion/)
    - schema_detector.py (COGMA)
    - target_validator.py
    - integrator.py

SUCCESS:
- No wrappers
- Direct calls to core modules
- Logic in one place only
```

---

## 3. Database Layer

### BEFORE ❌

```
┌─────────────────────────────────────────────┐
│  Dataset Profile DB                         │
│  (database/dataset_profile_db.py)           │
├─────────────────────────────────────────────┤
│  Table: dataset_profiles                    │
│    - dataset_id (PK)                        │
│    - schema_result (JSON)                   │
│    - target_candidates (JSON)               │
│    - preprocessing_plan (JSON)              │
│                                             │
│  Methods:                                   │
│    - save_profile()                         │
│    - load_profile()                         │
│    - list_profiles()                        │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  Session DB                                 │
│  (database/session_db.py)                   │
├─────────────────────────────────────────────┤
│  Table: sessions                            │
│    - session_id (PK)                        │
│    - context_json (JSON)                    │
│    - Contains DUPLICATE of profiles!        │
│                                             │
│  Methods:                                   │
│    - create_session()                       │
│    - get_session()                          │
│    - update_session()                       │
└─────────────────────────────────────────────┘

PROBLEMS:
- Two separate DB files
- Duplicate profile storage (in both tables!)
- Unclear ownership
- Potential sync issues
```

### AFTER ✅

```
┌─────────────────────────────────────────────┐
│  Context DB (UNIFIED)                       │
│  (database/context_db.py)                   │
├─────────────────────────────────────────────┤
│  Table: sessions                            │
│    - session_id (PK)                        │
│    - created_at                             │
│    - updated_at                             │
│    - pipeline_stage                         │
│    - context_json (ExecutionContext)        │
│                                             │
│  Table: dataset_profiles                    │
│    - dataset_id (PK)                        │
│    - session_id (FK → sessions)             │
│    - schema_detected                        │
│    - schema_result (JSON)                   │
│    - target_candidates (JSON)               │
│    - chosen_target                          │
│    - preprocessing_plan (JSON)              │
│    - modality_breakdown (JSON)              │
│    - user_overrides (JSON)                  │
│                                             │
│  Methods:                                   │
│    - save_context() / load_context()        │
│    - save_profile() / load_profile()        │
│    - load_session_profiles()                │
│    - list_sessions()                        │
│    - close_session()                        │
│                                             │
│  Features:                                  │
│    ✓ Thread-safe (per-thread connections)  │
│    ✓ Transaction boundaries                │
│    ✓ Single source of truth                │
│    ✓ No duplication                        │
└─────────────────────────────────────────────┘

SUCCESS:
- One database file
- Clear ownership
- No duplication
- Thread-safe
```

---

## 4. SessionContext vs ExecutionContext

### BEFORE ❌

```
┌──────────────────────────────────────────────────────┐
│  SessionContext (api/session_manager.py)             │
│  165 lines of duplication                            │
├──────────────────────────────────────────────────────┤
│  Fields:                                             │
│    - session_id                                      │
│    - dataset_profiles (Dict) ← DUPLICATE             │
│    - global_schema (Dict) ← DUPLICATE                │
│    - global_target (str) ← DUPLICATE                 │
│    - execution_context (Dict) ← WHAT IS THIS?        │
│    - execution_history (List)                        │
│                                                      │
│  What is execution_context field?                    │
│    → Serialized ExecutionContext?                    │
│    → Duplicate of dataset_profiles?                  │
│    → No one knows! 🤷                                │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  ExecutionContext (api/execution_context.py)         │
│  Different class!                                    │
├──────────────────────────────────────────────────────┤
│  Fields:                                             │
│    - session_id                                      │
│    - dataset_profiles (Dict[str, DatasetProfile])    │
│    - global_schema (Dict)                            │
│    - global_target (str)                             │
└──────────────────────────────────────────────────────┘

PROBLEM: Duplicate fields, unclear relationship!
```

### AFTER ✅

```
┌──────────────────────────────────────────────────────┐
│  ExecutionContext (core/execution_context.py)        │
│  Single source of truth                              │
├──────────────────────────────────────────────────────┤
│  @dataclass                                          │
│  class ExecutionContext:                             │
│    # Session                                         │
│    session_id: str                                   │
│    created_at: datetime                              │
│    updated_at: datetime                              │
│                                                      │
│    # Datasets                                        │
│    active_dataset_ids: List[str]                     │
│    dataset_profiles: Dict[str, DatasetProfile]       │
│                                                      │
│    # Global Intelligence                             │
│    global_schema: Optional[Dict]                     │
│    global_target: Optional[str]                      │
│    global_target_confidence: float                   │
│                                                      │
│    # Phase 6-8 (merged from pipeline/)               │
│    model_candidates: List[Dict]                      │
│    selected_model: Optional[str]                     │
│    training_signals: Dict                            │
│    drift_detected: bool                              │
│                                                      │
│    # Explainability                                  │
│    execution_log: List[Dict]                         │
│    override_history: List[Dict]                      │
│                                                      │
│  Methods:                                            │
│    - to_dict() / from_dict()                         │
│    - add_dataset_profile()                           │
│    - get_dataset_profile()                           │
│    - log_decision()                                  │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  SessionManager (api/session_manager.py)             │
│  Thin wrapper - 150 lines                            │
├──────────────────────────────────────────────────────┤
│  Methods:                                            │
│    - create_session() → ExecutionContext             │
│    - get_session() → ExecutionContext                │
│    - update_session(ctx)                             │
│    - list_sessions()                                 │
│    - close_session()                                 │
│                                                      │
│  All methods delegate to context_db                  │
└──────────────────────────────────────────────────────┘

SUCCESS: No duplication, clear separation!
```

---

## 5. Code Metrics

### Files

| Category               | Before | After | Change |
| ---------------------- | ------ | ----- | ------ |
| ExecutionContext files | 3      | 1     | -67%   |
| Service wrapper files  | 4      | 0     | -100%  |
| Database files         | 2      | 1     | -50%   |
| Total refactored files | 10     | 4     | -60%   |

### Lines of Code

| File                              | Before    | After         | Change     |
| --------------------------------- | --------- | ------------- | ---------- |
| api/execution_context.py          | 284 lines | -             | Archived   |
| pipeline/execution_context.py     | 395 lines | -             | Archived   |
| **core/execution_context.py**     | -         | **600 lines** | **Merged** |
| api/session_manager.py            | 354 lines | 150 lines     | -58%       |
| api/schema_detection_service.py   | 228 lines | -             | Archived   |
| api/target_detection_service.py   | 310 lines | -             | Archived   |
| api/global_aggregation_service.py | 350 lines | -             | Archived   |
| api/preprocessing_service.py      | 400 lines | -             | Archived   |
| **core/orchestrator.py**          | -         | **250 lines** | **New**    |
| database/dataset_profile_db.py    | 348 lines | -             | Archived   |
| database/session_db.py            | 366 lines | -             | Archived   |
| **database/context_db.py**        | -         | **350 lines** | **Merged** |

**Total Before**: 3,035 lines (with duplication)  
**Total After**: 1,350 lines (clean, no duplication)  
**Reduction**: 55% less code, 100% more maintainable!

---

## 6. Data Flow Comparison

### BEFORE ❌

```
User Upload
    ↓
DataIngestionManager (isolated)
    ↓
api/schema_detection_service.py (wrapper)
    ↓
data_ingestion/schema_detector.py (COGMA)
    ↓
api/ExecutionContext (frontend state)
    ↓
api/target_detection_service.py (wrapper)
    ↓
api/ExecutionContext updated
    ↓
[Context saved to dataset_profile_db]
    ↓
    ↓ [USER CALLS NEXT PHASE]
    ↓
[New pipeline/ExecutionContext created - EMPTY!] ❌
    ↓
automl/model_selector (no schema, no target!)
    ↓
automl/trainer (training with no preprocessing plan!)
```

### AFTER ✅

```
User Upload
    ↓
DataIngestionManager
    ↓
orchestrator.register_ingested_datasets(ctx, hashes)
    ↓
orchestrator.execute_phase_2_schema(ctx, data_map)
    → COGMASchemaDetector.detect_schema()
    → ctx.dataset_profiles[id].schema_result = schema
    → context_db.save_profile()
    ↓
orchestrator.execute_phase_3_target(ctx, data_map)
    → ctx.dataset_profiles[id].chosen_target = target
    → context_db.save_profile()
    ↓
orchestrator.execute_phase_4_aggregation(ctx, data_map)
    → ctx.global_schema = merged
    → ctx.global_target = best_target
    → context_db.save_context()
    ↓
orchestrator.execute_phase_5_preprocessing(ctx, data_map)
    → ctx.dataset_profiles[id].preprocessing_plan = plan
    → context_db.save_profile()
    ↓
[SAME CONTEXT continues to phase 6-8] ✅
    ↓
automl/model_selector (has schema, target, plan!)
    → ctx.model_candidates = models
    ↓
automl/trainer (has all intelligence!)
    → ctx.training_history = results
```

---

## Summary

### Problems Fixed ✅

1. **Three ExecutionContext classes** → **One unified class**
2. **Frontend/backend split** → **Single context throughout**
3. **Intelligence lost between phases** → **Intelligence preserved**
4. **4 service wrappers with no value** → **Direct core calls**
5. **2 separate database layers** → **1 unified database**
6. **SessionContext duplication** → **Thin wrapper only**
7. **2,681 lines of duplicate code** → **0 lines of duplication**

### Result 🎯

**Clean, maintainable architecture with:**

- Single source of truth
- Clear layer separation
- No duplicate code
- Preserved working features
- 55% less code

**Status**: Ready for integration testing! 🚀
