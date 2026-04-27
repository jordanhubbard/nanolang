#!/usr/bin/env bash
# scripts/run_bench.sh — nanolang benchmark suite runner
#
# Compiles and times each benchmark under multiple optimization levels.
# Outputs a markdown table + JSON results file with regression comparison.
#
# Usage:
#   bash scripts/run_bench.sh [--baseline bench/results.json] [--threshold 20]
#   make bench
#   make bench-compare BASELINE=bench/results.json

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BENCH_DIR="${BENCH_DIR:-bench}"
OUTPUT_FILE="${OUTPUT_FILE:-bench/results.json}"
BASELINE_FILE="${BASELINE_FILE:-}"
THRESHOLD="${THRESHOLD:-20}"
COMPILER="${COMPILER:-bin/nanoc_c}"
INTERPRETER="${INTERPRETER:-bin/nano}"
RUNS="${RUNS:-3}"

while [ $# -gt 0 ]; do
    case "$1" in
        --bench-dir)  BENCH_DIR="$2"; shift 2 ;;
        --output)     OUTPUT_FILE="$2"; shift 2 ;;
        --baseline)   BASELINE_FILE="$2"; shift 2 ;;
        --threshold)  THRESHOLD="$2"; shift 2 ;;
        --compiler)   COMPILER="$2"; shift 2 ;;
        --runs)       RUNS="$2"; shift 2 ;;
        --help)       sed -n 's/^# \{0,1\}//p' "$0" | head -30; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Timing helper ─────────────────────────────────────────────────────────
# time_cmd <cmd> → elapsed seconds to 3 dp
time_cmd() {
    local cmd="$1"
    local start end elapsed
    start=$(date +%s%3N)
    eval "$cmd" > /dev/null 2>&1
    end=$(date +%s%3N)
    elapsed=$(( end - start ))
    printf "%.3f" "$(echo "scale=3; $elapsed / 1000" | bc)"
}

# Median of N runs
time_median() {
    local cmd="$1" n="$2"
    local vals=()
    for i in $(seq 1 "$n"); do vals+=( "$(time_cmd "$cmd")" ); done
    IFS=$'\n' sorted=($(printf '%s\n' "${vals[@]}" | sort -n)); unset IFS
    local mid=$(( (n - 1) / 2 ))
    echo "${sorted[$mid]}"
}

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }

mkdir -p bench /tmp/nano_bench_bins

echo ""
echo "┌─────────────────────────────────────────────────────────────────────┐"
printf "│  nanolang benchmark suite  %-43s│\n" "$(timestamp)"
echo "├─────────────────────────────────────────────────────────────────────┤"
printf "│  interpreter: %-56s│\n" "$INTERPRETER"
printf "│  compiler:    %-56s│\n" "$COMPILER"
printf "│  bench dir:   %-56s│\n" "$BENCH_DIR"
printf "│  runs/bench:  %-56s│\n" "$RUNS"
echo "└─────────────────────────────────────────────────────────────────────┘"
echo ""
printf "%-22s %10s %10s %10s %10s\n" "Benchmark" "Interp(s)" "C -O0(s)" "C -O2(s)" "C -O3(s)"
printf "%-22s %10s %10s %10s %10s\n" "──────────────────────" "──────────" "──────────" "──────────" "──────────"

JSON_ENTRIES=""

for NANO_FILE in "$BENCH_DIR"/*.nano; do
    [ -f "$NANO_FILE" ] || continue
    BENCH_NAME=$(basename "$NANO_FILE" .nano)
    BIN_BASE="/tmp/nano_bench_bins/${BENCH_NAME}"

    # Interpreter
    INTERP_T=$(time_median "$INTERPRETER $NANO_FILE" "$RUNS")

    # Compile to C and run at each opt level
    O0_T="n/a"; O2_T="n/a"; O3_T="n/a"
    if [ -x "$COMPILER" ]; then
        # Emit C source once
        C_SRC="${BIN_BASE}.c"
        if "$COMPILER" "$NANO_FILE" --target c -o "$C_SRC" > /dev/null 2>&1; then
            for OLEVEL in 0 2 3; do
                BIN="${BIN_BASE}_O${OLEVEL}"
                if cc -std=c99 "-O${OLEVEL}" "$C_SRC" -o "$BIN" 2>/dev/null; then
                    T=$(time_median "$BIN" "$RUNS")
                    case $OLEVEL in
                        0) O0_T="$T" ;; 2) O2_T="$T" ;; 3) O3_T="$T" ;;
                    esac
                fi
            done
        fi
    fi

    printf "%-22s %10s %10s %10s %10s\n" \
        "$BENCH_NAME" "$INTERP_T" "$O0_T" "$O2_T" "$O3_T"

    ENTRY="{\"name\":\"$BENCH_NAME\",\"interp\":$INTERP_T,\"o0\":\"$O0_T\",\"o2\":\"$O2_T\",\"o3\":\"$O3_T\",\"ts\":\"$(timestamp)\"}"
    JSON_ENTRIES="${JSON_ENTRIES:+$JSON_ENTRIES,}$ENTRY"
done

echo ""

# ── Regression check ─────────────────────────────────────────────────────
REGRESSION_EXIT=0
if [ -n "$BASELINE_FILE" ] && [ -f "$BASELINE_FILE" ]; then
    echo "── Regression check vs: $BASELINE_FILE  (threshold: ${THRESHOLD}%) ──"
    python3 - "$BASELINE_FILE" "[${JSON_ENTRIES}]" "$THRESHOLD" << 'PYEOF' || REGRESSION_EXIT=$?
import sys, json
baseline = {e['name']: e for e in json.load(open(sys.argv[1]))}
current  = json.loads(sys.argv[2])
threshold = float(sys.argv[3])
regressions = 0
for e in current:
    name = e['name']
    if name not in baseline: continue
    for field in ('interp',):
        try:
            bv = float(baseline[name][field]); cv = float(e[field])
            pct = (cv - bv) / bv * 100
            if pct > threshold:
                print(f"  ⚠️  REGRESSION {name}.{field}: {bv:.3f}s → {cv:.3f}s (+{pct:.1f}%)")
                regressions += 1
            elif pct < -5:
                print(f"  ✅ improved   {name}.{field}: {bv:.3f}s → {cv:.3f}s ({pct:.1f}%)")
        except (ValueError, KeyError): pass
if regressions == 0:
    print("  ✅ No regressions detected")
sys.exit(1 if regressions > 0 else 0)
PYEOF
    echo ""
fi

# ── Write JSON results ────────────────────────────────────────────────────
mkdir -p "$(dirname "$OUTPUT_FILE")"
printf '[%s]\n' "$JSON_ENTRIES" > "$OUTPUT_FILE"
echo "Results → $OUTPUT_FILE"
echo ""
[ "$REGRESSION_EXIT" -eq 0 ] && echo "✅ Benchmark suite complete" || { echo "❌ Regressions detected"; exit 1; }
