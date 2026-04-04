# ✅ POST-REFACTOR VALIDATION CHECKLIST

**Repository**: APEX/AutoVision+ Multimodal AutoML  
**Audit Date**: 2026-04-04  
**Status**: 🟡 85% Complete — Action Required

---

## 🔴 URGENT ACTION ITEMS (Week 1)

### [ ] 1. Fix Circular Dependency (Priority 1)

**Issue**: `pipeline/training_orchestrator.py` ↔ `pipeline/retraining_pipeline.py`

**Steps**:

```bash
# 1. Rename file
git mv pipeline/retraining_pipeline.py pipeline/retrain_executor.py

# 2. Update import in training_orchestrator.py (line ~350)
# OLD: from pipeline.retraining_pipeline import RetrainingPipeline
# NEW: from pipeline.retrain_executor import RetrainingPipeline

# 3. Update import in retrain_executor.py (line ~70)
# KEEP: from pipeline.training_orchestrator import ...

# 4. Test
python -m pytest tests/
```

**Time**: 2 hours  
**Risk**: 🔴 HIGH (fragile dependency)

---

### [ ] 2. Consolidate State Management (Priority 2)

**Issue**: Local dataclasses in `training_orchestrator.py` fragment state

**Steps**:

```bash
# 1. Create core/types.py
touch core/types.py

# 2. Copy these classes from training_orchestrator.py (lines 43-92):
#    - Phase (Enum)
#    - TrainingConfig
#    - ModelSelectionResult
#    - TrainingMetrics

# 3. Add import to core/types.py:
cat > core/types.py << 'EOF'
"""Shared type definitions for APEX pipeline."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

class Phase(Enum):
    """Workflow phases."""
    DATA_INGESTION = 1
    SCHEMA_DETECTION = 2
    PREPROCESSING = 3
    MODEL_SELECTION = 4
    TRAINING = 5
    DRIFT_DETECTION = 6
    MODEL_REGISTRY = 7

@dataclass
class TrainingConfig:
    """Configuration for complete training workflow."""
    dataset_sources: List[str]
    problem_type: str
    modalities: List[str]
    target_column: Optional[str] = None
    test_split: float = 0.2
    val_split: float = 0.2
    seed: int = 42
    device: str = "cuda"

@dataclass
class ModelSelectionResult:
    """Result from Phase 4 model selection."""
    image_encoder: Optional[str]
    text_encoder: Optional[str]
    tabular_encoder: Optional[str]
    fusion_strategy: str
    batch_size: int
    epochs: int
    learning_rate: float
    dropout: float
    weight_decay: float
    selection_rationale: str

@dataclass
class TrainingMetrics:
    """Training metrics from Phase 5."""
    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    train_f1: Optional[float] = None
    val_f1: Optional[float] = None
EOF

# 4. Update training_orchestrator.py
# DELETE: Lines 43-92 (local dataclass definitions)
# ADD: from core.types import Phase, TrainingConfig, ModelSelectionResult, TrainingMetrics

# 5. Test
python -m pytest tests/test_integration_e2e.py
```

**Time**: 4 hours  
**Risk**: 🟡 MEDIUM (refactoring, but well-defined)

---

## 🟡 IMPORTANT ACTION ITEMS (Week 2)

### [ ] 3. Restructure API File Location

**Issue**: `run_api.py` in root directory, should be in `api/`

**Steps**:

```bash
# 1. Move file
git mv run_api.py api/run_api.py

# 2. Update api/run_server.py (line ~80)
# OLD: api_cmd = [sys.executable, "-m", "uvicorn", "run_api:app", ...]
# NEW: api_cmd = [sys.executable, "-m", "uvicorn", "api.run_api:app", ...]

# 3. Update documentation
sed -i 's/python run_api.py/python api\/run_api.py/g' README.md
sed -i 's/python run_api.py/python -m api.run_api/g' deployment_guide.md

# 4. Test
python api/run_server.py  # Should launch both API + UI
```

**Time**: 1 hour  
**Risk**: 🟡 MEDIUM (path changes)

---

### [ ] 4. Fix Folder Naming Typo

**Issue**: Folder named `modelss/` (extra 's'), should be `models/`

**Steps**:

```bash
# 1. Rename folder
git mv modelss models

# 2. Update all imports (automated)
find . -name "*.py" -type f -exec sed -i 's/from modelss\./from models./g' {} \;
find . -name "*.py" -type f -exec sed -i 's/import modelss\./import models./g' {} \;

# 3. Verify no remaining references
grep -r "modelss" --include="*.py" .

# 4. Test
python -m pytest tests/
```

**Time**: 30 minutes  
**Risk**: 🟢 LOW (automated find/replace)

---

### [ ] 5. Archive Stale Documentation

**Issue**: 11 redundant markdown files cluttering root directory

**Steps**:

```bash
# 1. Create archive directory
mkdir -p docs/archive

# 2. Move stale docs
mv AUDIT_SUMMARY.md docs/archive/
mv BEFORE_AFTER.md docs/archive/
mv CODEBASE_AUDIT_REPORT.md docs/archive/
mv COMPREHENSIVE_CODEBASE_AUDIT_2026.md docs/archive/
mv FIX4_FINAL_CHECKLIST.md docs/archive/
mv FIX4_research_paper.md docs/archive/
mv REFACTOR_COMPLETE.md docs/archive/
mv REFACTOR_VALIDATION_REPORT.md docs/archive/
mv TASK_REPORT.md docs/archive/
mv repo_map.md docs/archive/
mv skills.md docs/archive/

# 3. Keep essential docs
# ✅ README.md
# ✅ deployment_guide.md
# ✅ COMPREHENSIVE_AUDIT_2026-04-04.md (new)
# ✅ AUDIT_EXECUTIVE_SUMMARY.md (new)

# 4. Update .gitignore
echo "docs/archive/" >> .gitignore
```

**Time**: 15 minutes  
**Risk**: 🟢 LOW (cleanup only)

---

## 🟢 OPTIONAL CLEANUP (Anytime)

### [ ] 6. Remove Dead Import

**Issue**: `run_api.py` imports Flask in FastAPI app

**Steps**:

```python
# Edit api/run_api.py (after moving), delete line 17:
# from flask import session  # ← DELETE THIS LINE
```

**Time**: 5 minutes  
**Risk**: 🟢 LOW (unused import)

---

## 🧪 VALIDATION TESTS (Run After Each Fix)

### Syntax Validation

```bash
python -m py_compile api/run_api.py
python -m py_compile core/types.py
python -m py_compile pipeline/training_orchestrator.py
python -m py_compile pipeline/retrain_executor.py
```

### Import Chain Test

```python
# test_imports.py
import sys
print("Testing imports...")

try:
    from core.execution_context import ExecutionContext, DatasetProfile
    print("✅ core.execution_context")

    from core.orchestrator import orchestrator
    print("✅ core.orchestrator")

    from database.context_db import context_db
    print("✅ database.context_db")

    from api.session_manager import session_manager
    print("✅ api.session_manager")

    from core.types import Phase, TrainingConfig, ModelSelectionResult, TrainingMetrics
    print("✅ core.types")

    print("\n✅ ALL IMPORTS SUCCESSFUL")
except Exception as e:
    print(f"\n❌ IMPORT FAILED: {e}")
    sys.exit(1)
```

### Integration Test

```bash
# Launch API + UI
python api/run_server.py &
sleep 10

# Test health endpoint
curl http://localhost:8001/health

# Test ingestion
curl -X POST http://localhost:8001/ingest/datasets \
  -H "Content-Type: application/json" \
  -d '{"dataset_urls": ["https://example.com/test.csv"], "session_id": "test123"}'

# Clean up
pkill -f "uvicorn"
pkill -f "streamlit"
```

---

## 📊 PROGRESS TRACKING

| Task                    | Priority | Time  | Status  | Assignee | Due Date |
| ----------------------- | -------- | ----- | ------- | -------- | -------- |
| Fix Circular Dependency | 🔴 P1    | 2h    | ⬜️ TODO | \_\_\_   | Week 1   |
| Consolidate State       | 🔴 P2    | 4h    | ⬜️ TODO | \_\_\_   | Week 1   |
| Restructure API         | 🟡 M1    | 1h    | ⬜️ TODO | \_\_\_   | Week 2   |
| Fix Folder Typo         | 🟡 M2    | 30min | ⬜️ TODO | \_\_\_   | Week 2   |
| Archive Stale Docs      | 🟡 M3    | 15min | ⬜️ TODO | \_\_\_   | Week 2   |
| Remove Dead Import      | 🟢 L1    | 5min  | ⬜️ TODO | \_\_\_   | Anytime  |

**Total Effort**: ~8 hours (URGENT + IMPORTANT)

---

## 🎯 SUCCESS CRITERIA

After completing all tasks, verify:

- [ ] No circular import warnings when running `python -c "import pipeline.training_orchestrator"`
- [ ] All tests pass: `python -m pytest tests/ -v`
- [ ] API launches: `python api/run_server.py` works without errors
- [ ] Frontend loads: `http://localhost:8501` displays 7-phase UI
- [ ] Full pipeline works: Ingest → Schema → Preprocessing → Training → Prediction
- [ ] No "modelss" references: `grep -r "modelss" --include="*.py" .` returns nothing
- [ ] Clean root directory: Only essential files in root
- [ ] Documentation updated: README reflects new structure

---

## 📞 CONTACTS & RESOURCES

**Full Audit Report**: `COMPREHENSIVE_AUDIT_2026-04-04.md` (40KB)  
**Executive Summary**: `AUDIT_EXECUTIVE_SUMMARY.md` (6KB)  
**Architecture Docs**: `deployment_guide.md`

**Key Files to Review**:

- `core/execution_context.py` — Single source of truth (600 lines)
- `core/orchestrator.py` — Phase coordinator (250 lines)
- `database/context_db.py` — Unified persistence (360 lines)
- `api/run_api.py` — FastAPI endpoints (2600 lines)

**Help**: If stuck, review dependency audit in `COMPREHENSIVE_AUDIT_2026-04-04.md` Section 1-7

---

**Last Updated**: 2026-04-04  
**Next Review**: After Week 2 tasks complete  
**Estimated Completion**: 2 weeks (8 hours total effort)
