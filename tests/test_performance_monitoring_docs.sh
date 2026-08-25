#!/usr/bin/env bash
# Assert first-class performance-monitoring docs stay accurate vs the -pg wrapper.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fail() { echo "FAIL: $*" >&2; exit 1; }

need_file() {
    [ -f "$ROOT/$1" ] || fail "missing $1"
}

need_grep() {
    local file="$1"
    local pattern="$2"
    grep -qE -e "$pattern" "$ROOT/$file" || fail "$file must match /$pattern/"
}

forbid_grep() {
    local file="$1"
    local pattern="$2"
    if grep -qE -e "$pattern" "$ROOT/$file"; then
        fail "$file must not match /$pattern/"
    fi
}

need_file README.md
need_file docs/PERFORMANCE_MONITORING.md
need_file docs/README.md
need_file userguide/08_profiling.md
need_file userguide/guide/07_performance_profiling.md
need_file userguide/nav.txt

need_grep README.md '^## Performance Monitoring and LLM Optimization'
need_grep README.md 'docs/PERFORMANCE_MONITORING.md'
need_grep README.md 'userguide/guide/07_performance_profiling.md'

need_grep docs/README.md 'PERFORMANCE_MONITORING.md'
need_grep docs/DOCS_INDEX.md 'PERFORMANCE_MONITORING.md'

need_grep userguide/nav.txt 'guide/07_performance_profiling.md'
need_grep userguide/index.md 'guide/07_performance_profiling.md'
need_grep userguide/08_profiling.md 'guide/07_performance_profiling.md'

for f in docs/PERFORMANCE_MONITORING.md userguide/guide/07_performance_profiling.md; do
    need_grep "$f" 'xctrace'
    need_grep "$f" 'gprofng'
    need_grep "$f" 'sample'
    need_grep "$f" '--profile-output'
    need_grep "$f" '--pgo'
    need_grep "$f" '--profile-runtime'
    need_grep "$f" '_nl_run_with_profiling|_NL_PROFILING'
    need_grep "$f" '"profile_type": "sampling"'
    need_grep "$f" 'stdout'
    forbid_grep "$f" 'major innovation'
    forbid_grep "$f" '2-10x|2–10x|2–10×|2-10×'
    forbid_grep "$f" 'working on switching to xctrace'
    forbid_grep "$f" '"location":'
    forbid_grep "$f" '"per_call_us":'
done

need_grep docs/PERFORMANCE_MONITORING.md 'excl_pct \* 10|exclusive percent times ten'
need_grep docs/PERFORMANCE_MONITORING.md 'Command Line Tools'
need_grep docs/PERFORMANCE_MONITORING.md 'libgp-collector'
need_grep docs/PERFORMANCE_MONITORING.md 'not a PGO input'

echo "performance monitoring docs assertions passed"
