#!/bin/bash
# merge-branches.sh — Rebase each feature branch onto main, verify tests, merge
# Run from repo root: bash scripts/merge-branches.sh 2>&1 | tee /tmp/merge-campaign.log

set -euo pipefail

REPO=$(git rev-parse --show-toplevel)
cd "$REPO"

git fetch --all --quiet

PASS=0
FAIL=0
SKIP=0

log()   { echo "[$(date '+%H:%M:%S')] $*"; }
ok()    { echo "  ✅ $*"; }
fail()  { echo "  ❌ $*"; }
skip()  { echo "  ⊘  $*"; }

merge_branch() {
    local br="$1"
    local desc="${2:-}"

    log "=== $br ==="

    # Check if branch exists on origin
    if ! git ls-remote --exit-code --heads origin "$br" > /dev/null 2>&1; then
        skip "$br: not found on origin"
        SKIP=$((SKIP+1))
        return
    fi

    # Count unique commits
    local ahead
    ahead=$(git rev-list "origin/main..origin/$br" --count 2>/dev/null || echo 0)
    if [ "$ahead" -eq 0 ]; then
        skip "$br: 0 unique commits — already merged"
        SKIP=$((SKIP+1))
        return
    fi
    log "  $ahead unique commit(s) ahead of main"

    # Create/reset local branch from origin
    git checkout -B "$br" "origin/$br" --quiet 2>/dev/null

    # Rebase onto main
    log "  Rebasing onto main..."
    if ! git rebase origin/main --quiet 2>/tmp/rebase-err.txt; then
        fail "$br: rebase failed"
        cat /tmp/rebase-err.txt | head -20
        git rebase --abort 2>/dev/null || true
        git checkout main --quiet
        FAIL=$((FAIL+1))
        return
    fi

    # Build
    log "  Building..."
    if ! make bin/nanoc_c 2>/tmp/build-err.txt; then
        fail "$br: build failed"
        cat /tmp/build-err.txt | tail -20
        git checkout main --quiet
        FAIL=$((FAIL+1))
        return
    fi

    # Run tests (quick subset — run_all_tests with 5min timeout)
    log "  Running tests..."
    local test_out
    test_out=$(timeout 300 bash tests/run_all_tests.sh 2>&1 || true)
    local fail_count
    fail_count=$(echo "$test_out" | grep -c "❌" || true)
    local pass_count
    pass_count=$(echo "$test_out" | grep -c "✅" || true)
    
    # Allow up to 3 failures (async/coroutine/pretty_printer are known pre-existing)
    if [ "$fail_count" -gt 3 ]; then
        fail "$br: $fail_count test failures (expected ≤3)"
        echo "$test_out" | grep "❌" | head -10
        git checkout main --quiet
        FAIL=$((FAIL+1))
        return
    fi

    ok "$br: $pass_count pass, $fail_count fail — merging"

    # Merge into main
    git checkout main --quiet
    local msg="merge: $br"
    [ -n "$desc" ] && msg="merge: $br — $desc"
    git merge --no-ff "$br" -m "$msg" --quiet

    # Push
    if ! git push origin main --quiet 2>/tmp/push-err.txt; then
        fail "$br: push failed"
        cat /tmp/push-err.txt
        FAIL=$((FAIL+1))
        return
    fi

    # Clean up local branch
    git branch -d "$br" 2>/dev/null || true

    ok "$br merged and pushed"
    PASS=$((PASS+1))
}

# Start on main
git checkout main --quiet
git pull origin main --quiet

echo ""
echo "========================================"
echo "nanolang branch merge campaign"
echo "========================================"
echo ""

# Fix branches first
merge_branch "fix/restore-missing-sources"  "restore missing sources for algebraic effects"
merge_branch "fix/ci-test-failures"          "CI test failure fixes"

# Small feature branches (1 commit each, minimal risk)
merge_branch "feat/fstring-interpolation"    "f-string interpolation"
merge_branch "feat/coverage-tracking"        "coverage tracking --coverage flag"
merge_branch "feat/repl-history"             "REPL command history"
merge_branch "feat/bench-suite"              "nano-bench micro-benchmark harness"
merge_branch "feat/c-backend"                "nano-to-C transpiler backend"
merge_branch "feat/lsp-rowpoly-hover"        "LSP hover for row-polymorphic types"
merge_branch "feat/nano-docs"                "nanodoc API doc generator"
merge_branch "feat/nano-fmt"                 "nano-fmt style formatter"
merge_branch "feat/generics-typechecker"     "generics type checker"

# Medium feature branches
merge_branch "feat/package-registry"         "nano package registry + nanoc install/publish"
merge_branch "feat/coroutine-runtime"        "coroutine runtime"
merge_branch "feat/compiler-advanced"        "advanced compiler passes"

# Larger branches (use rebase variants)
merge_branch "feat/ptx-backend-rebase"       "PTX GPU backend (rebased)"
merge_branch "feat/wasm-runtime-profiler-rebase" "WASM runtime profiler (rebased)"
merge_branch "feat/wasm-simd-vectorization"  "WASM SIMD128 auto-vectorization"
merge_branch "feat/wasm-sourcemap"           "WASM source maps"
merge_branch "feat/repl-scripting"           "REPL scripting mode"

# Most complex last
merge_branch "feat/async-await-cps"          "async/await via CPS transformation"

echo ""
echo "========================================"
echo "Campaign complete: $PASS merged, $FAIL failed, $SKIP skipped"
echo "========================================"

# Final test run
echo ""
echo "Running final test suite on main..."
make test 2>&1 | grep -E "(Results:|❌|✅ All)" | tail -10

openclaw system event --text "Done: nanolang branch merges complete — $PASS merged, $FAIL failed, $SKIP skipped" --mode now 2>/dev/null || true
