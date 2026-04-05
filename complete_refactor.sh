#!/bin/bash
# APEX Refactor - Complete Remaining Manual Tasks
# Execute this script to finish the refactoring work
# Estimated time: 5 minutes

set -e  # Exit on error

echo "=================================================="
echo "APEX/AutoVision+ - Completing Refactor Tasks"
echo "=================================================="
echo ""

# Store current directory
REPO_ROOT="$(pwd)"

echo "📍 Current location: $REPO_ROOT"
echo ""

# Task M1: Move run_api.py to api/
echo "⚙️  Task M1: Restructure API File Location"
echo "   Moving run_api.py → api/run_api.py"
if [ -f "run_api.py" ] && [ ! -f "api/run_api.py" ]; then
    git mv run_api.py api/run_api.py
    echo "   ✅ Moved successfully"
else
    echo "   ⚠️  Already moved or file not found"
fi
echo ""

# Task M2: Rename modelss/ to models/
echo "⚙️  Task M2: Fix Folder Naming Typo"
echo "   Renaming modelss/ → models/"
if [ -d "modelss" ] && [ ! -d "models" ]; then
    git mv modelss models
    echo "   ✅ Renamed successfully"
else
    echo "   ⚠️  Already renamed or folder not found"
fi
echo ""

# Clean up old retraining_pipeline.py
echo "⚙️  Cleanup: Remove old retraining_pipeline.py"
if [ -f "pipeline/retraining_pipeline.py" ]; then
    rm pipeline/retraining_pipeline.py
    git add pipeline/retraining_pipeline.py
    echo "   ✅ Deleted successfully"
else
    echo "   ⚠️  Already deleted"
fi
echo ""

# Task M3: Archive stale documentation
echo "⚙️  Task M3: Archive Stale Documentation"
echo "   Creating docs/archive/ directory"
mkdir -p docs/archive

DOCS_TO_ARCHIVE=(
    "AUDIT_SUMMARY.md"
    "BEFORE_AFTER.md"
    "CODEBASE_AUDIT_REPORT.md"
    "COMPREHENSIVE_CODEBASE_AUDIT_2026.md"
    "FIX4_FINAL_CHECKLIST.md"
    "FIX4_research_paper.md"
    "REFACTOR_COMPLETE.md"
    "REFACTOR_VALIDATION_REPORT.md"
    "TASK_REPORT.md"
    "repo_map.md"
    "skills.md"
)

MOVED_COUNT=0
for doc in "${DOCS_TO_ARCHIVE[@]}"; do
    if [ -f "$doc" ]; then
        git mv "$doc" "docs/archive/"
        ((MOVED_COUNT++))
    fi
done
echo "   ✅ Moved $MOVED_COUNT documentation files to docs/archive/"
echo ""

# Task L1: Remove dead Flask import
echo "⚙️  Task L1: Remove Dead Flask Import"
echo "   Editing api/run_api.py (removing line 17)"
if [ -f "api/run_api.py" ]; then
    # Remove the Flask import line
    sed -i '17d' api/run_api.py 2>/dev/null || sed -i '' '17d' api/run_api.py
    git add api/run_api.py
    echo "   ✅ Removed Flask import"
else
    echo "   ⚠️  File not found (may not have been moved yet)"
fi
echo ""

# Commit all changes
echo "📝 Committing all changes..."
git commit -m "Complete remaining refactor tasks

Tasks completed:
- M1: Moved run_api.py to api/run_api.py
- M2: Renamed modelss/ to models/
- M3: Archived 11 stale documentation files to docs/archive/
- L1: Removed unused Flask import from api/run_api.py
- Cleanup: Deleted old pipeline/retraining_pipeline.py

All refactoring tasks from POST_REFACTOR_CHECKLIST.md are now complete.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"

echo ""
echo "=================================================="
echo "✅ All Refactoring Tasks Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Run validation tests: python -m pytest tests/"
echo "2. Start the system: python api/run_server.py"
echo "3. Check health: curl http://localhost:8001/health"
echo ""
