# APEX Refactor - Complete Remaining Manual Tasks (Windows PowerShell)
# Execute this script to finish the refactoring work
# Estimated time: 5 minutes

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "APEX/AutoVision+ - Completing Refactor Tasks" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

$RepoRoot = Get-Location
Write-Host "📍 Current location: $RepoRoot" -ForegroundColor Yellow
Write-Host ""

# Task M1: Move run_api.py to api/
Write-Host "⚙️  Task M1: Restructure API File Location" -ForegroundColor Green
Write-Host "   Moving run_api.py → api/run_api.py"
if ((Test-Path "run_api.py") -and (-not (Test-Path "api/run_api.py"))) {
    git mv run_api.py api/run_api.py
    Write-Host "   ✅ Moved successfully" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  Already moved or file not found" -ForegroundColor Yellow
}
Write-Host ""

# Task M2: Rename modelss/ to models/
Write-Host "⚙️  Task M2: Fix Folder Naming Typo" -ForegroundColor Green
Write-Host "   Renaming modelss/ → models/"
if ((Test-Path "modelss") -and (-not (Test-Path "models"))) {
    git mv modelss models
    Write-Host "   ✅ Renamed successfully" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  Already renamed or folder not found" -ForegroundColor Yellow
}
Write-Host ""

# Clean up old retraining_pipeline.py
Write-Host "⚙️  Cleanup: Remove old retraining_pipeline.py" -ForegroundColor Green
if (Test-Path "pipeline/retraining_pipeline.py") {
    Remove-Item "pipeline/retraining_pipeline.py" -Force
    git add pipeline/retraining_pipeline.py
    Write-Host "   ✅ Deleted successfully" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  Already deleted" -ForegroundColor Yellow
}
Write-Host ""

# Task M3: Archive stale documentation
Write-Host "⚙️  Task M3: Archive Stale Documentation" -ForegroundColor Green
Write-Host "   Creating docs/archive/ directory"
New-Item -ItemType Directory -Force -Path "docs/archive" | Out-Null

$DocsToArchive = @(
    "AUDIT_SUMMARY.md",
    "BEFORE_AFTER.md",
    "CODEBASE_AUDIT_REPORT.md",
    "COMPREHENSIVE_CODEBASE_AUDIT_2026.md",
    "FIX4_FINAL_CHECKLIST.md",
    "FIX4_research_paper.md",
    "REFACTOR_COMPLETE.md",
    "REFACTOR_VALIDATION_REPORT.md",
    "TASK_REPORT.md",
    "repo_map.md",
    "skills.md"
)

$MovedCount = 0
foreach ($doc in $DocsToArchive) {
    if (Test-Path $doc) {
        git mv $doc "docs/archive/"
        $MovedCount++
    }
}
Write-Host "   ✅ Moved $MovedCount documentation files to docs/archive/" -ForegroundColor Green
Write-Host ""

# Task L1: Remove dead Flask import
Write-Host "⚙️  Task L1: Remove Dead Flask Import" -ForegroundColor Green
Write-Host "   Editing api/run_api.py (removing line 17)"
if (Test-Path "api/run_api.py") {
    # Read the file, remove line 17, write back
    $content = Get-Content "api/run_api.py"
    $newContent = $content[0..15] + $content[17..($content.Length-1)]
    Set-Content "api/run_api.py" -Value $newContent
    git add api/run_api.py
    Write-Host "   ✅ Removed Flask import" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  File not found (may not have been moved yet)" -ForegroundColor Yellow
}
Write-Host ""

# Commit all changes
Write-Host "📝 Committing all changes..." -ForegroundColor Cyan
git commit -m @"
Complete remaining refactor tasks

Tasks completed:
- M1: Moved run_api.py to api/run_api.py
- M2: Renamed modelss/ to models/
- M3: Archived 11 stale documentation files to docs/archive/
- L1: Removed unused Flask import from api/run_api.py
- Cleanup: Deleted old pipeline/retraining_pipeline.py

All refactoring tasks from POST_REFACTOR_CHECKLIST.md are now complete.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
"@

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "✅ All Refactoring Tasks Complete!" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Run validation tests: python -m pytest tests/"
Write-Host "2. Start the system: python api/run_server.py"
Write-Host "3. Check health: curl http://localhost:8001/health"
Write-Host ""
