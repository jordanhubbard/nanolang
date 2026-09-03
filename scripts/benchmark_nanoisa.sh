#!/usr/bin/env bash
#
# NanoISA benchmarks.
#
# This harness used to time one `nano_vm workload.nvm` per sample. That
# measured process startup and almost nothing else: the workloads here retire
# between 78 and 32,082 instructions, and every one of them took about the same
# 17 ms, because the spawn dominates by three orders of magnitude. A suite
# built that way cannot see an interpreter change of any size, which makes it
# useless for the decisions it exists to inform.
#
# So each workload is now measured twice per sample -- once with a single
# iteration and once with many, both behind one process startup -- and the
# per-iteration execution cost is the difference divided by the extra
# iterations. Pairing the two measurements within a sample also cancels
# machine drift between them.
#
# Cold startup is then a dimension in its own right rather than a confound:
# it is what a trivial module costs, which is the floor every other number
# used to be buried under.

set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-"$ROOT/.benchmark_results/nanoisa"}
RUNS=${NANOISA_BENCH_RUNS:-20}
# Enough iterations that the difference between the two timings is far above
# the noise of a single process spawn.
REPEAT=${NANOISA_BENCH_REPEAT:-1000}

mkdir -p "$OUT"
make -C "$ROOT" nano_virt nano_vm >/dev/null

workloads=(
  examples/language/nl_fibonacci.nano
  examples/language/nl_forth_interpreter.nano
  examples/language/nl_hashmap_word_count.nano
  examples/language/nl_string_operations.nano
  examples/language/nl_array_complete.nano
  examples/language/nl_function_variables.nano
  examples/language/nl_extern_math.nano
)

now_ns() { python3 -c 'import time; print(time.monotonic_ns())'; }

manifest="$OUT/manifest.tsv"
: > "$manifest"
printf 'workload\trun\telapsed_ns\texit_code\tprofile\n' >> "$manifest"

# ── Cold startup ────────────────────────────────────────────────────────────
# A module that does nothing, so the whole measurement is spawn, load, verify
# and teardown. Reported as its own workload because it is a real cost -- it
# is what a short-lived program pays -- and because naming it stops it hiding
# inside every other number.
startup_src="$OUT/_cold_startup.nano"
cat > "$startup_src" <<'NANO'
fn main() -> int {
    return 0
}
NANO
startup_nvm="$OUT/cold_startup.nvm"
"$ROOT/bin/nano_virt" "$startup_src" --emit-nvm -o "$startup_nvm" >/dev/null

for ((run = 1; run <= RUNS; run++)); do
    start=$(now_ns)
    set +e
    "$ROOT/bin/nano_vm" "$startup_nvm" >/dev/null
    exit_code=$?
    set -e
    stop=$(now_ns)
    printf 'cold_startup\t%d\t%d\t%d\t\n' "$run" "$((stop - start))" "$exit_code" >> "$manifest"
done

# ── Execution ───────────────────────────────────────────────────────────────
for source in "${workloads[@]}"; do
    name=$(basename "$source" .nano)
    nvm="$OUT/$name.nvm"
    "$ROOT/bin/nano_virt" "$ROOT/$source" --emit-nvm -o "$nvm"

    for ((run = 1; run <= RUNS; run++)); do
        # One iteration: startup plus a single execution.
        start=$(now_ns)
        set +e
        "$ROOT/bin/nano_vm" --repeat 1 "$nvm" >/dev/null
        exit_code=$?
        set -e
        one=$(( $(now_ns) - start ))

        # Many iterations behind the same startup.
        start=$(now_ns)
        set +e
        "$ROOT/bin/nano_vm" --repeat "$REPEAT" "$nvm" >/dev/null
        code_many=$?
        set -e
        many=$(( $(now_ns) - start ))
        [ "$code_many" -ne 0 ] && exit_code=$code_many

        # The startup terms cancel; what is left is execution.
        per_iteration=$(( (many - one) / (REPEAT - 1) ))
        [ "$per_iteration" -lt 0 ] && per_iteration=0

        # The profile comes from a single-iteration run so its counters
        # describe one execution rather than REPEAT of them.
        profile="$OUT/${name}.${run}.json"
        "$ROOT/bin/nano_vm" --profile-isa "$profile" "$nvm" >/dev/null 2>&1 || true

        printf '%s\t%d\t%d\t%d\t%s\n' \
            "$name" "$run" "$per_iteration" "$exit_code" "$profile" >> "$manifest"
    done
done

# ── Co-process boundary: crossings, crashes, restarts ───────────────────────
#
# The same FFI workload run in-process and through the co-process. The
# difference is what isolation costs per crossing, which is the number the
# batching path exists to reduce -- a batch pays one signal/ack pair for many
# calls instead of one pair each. Measured here rather than asserted, because
# "batching helps" is a claim about this difference and nothing else.
#
# Recovering from a crash means launching a replacement, so what is measured
# here is the launch: the same run with and without isolation, differing only
# by standing the child up. That bounds restart cost rather than reproducing a
# crash, and the distinction is worth keeping -- this number does not include
# detecting the death or reissuing the in-flight call. Recorded separately from
# throughput because a recovery cost averaged into steady state hides both.
ffi_nvm="$OUT/nl_extern_math.nvm"
if [ -f "$ffi_nvm" ]; then
    for ((run = 1; run <= RUNS; run++)); do
        start=$(now_ns)
        set +e
        "$ROOT/bin/nano_vm" --isolate-ffi --repeat 1 "$ffi_nvm" >/dev/null 2>&1
        code=$?
        set -e
        one=$(( $(now_ns) - start ))

        start=$(now_ns)
        set +e
        "$ROOT/bin/nano_vm" --isolate-ffi --repeat "$REPEAT" "$ffi_nvm" >/dev/null 2>&1
        code_many=$?
        set -e
        many=$(( $(now_ns) - start ))
        [ "$code_many" -ne 0 ] && code=$code_many

        per_iteration=$(( (many - one) / (REPEAT - 1) ))
        [ "$per_iteration" -lt 0 ] && per_iteration=0
        printf 'ffi_coprocess\t%d\t%d\t%d\t\n' "$run" "$per_iteration" "$code" >> "$manifest"
    done

    # Launch cost: one run with the co-process against the same run without it.
    # The difference is what standing the child up costs, which is the floor
    # for recovering from one that died.
    for ((run = 1; run <= RUNS; run++)); do
        start=$(now_ns)
        set +e
        "$ROOT/bin/nano_vm" --isolate-ffi --repeat 1 "$ffi_nvm" >/dev/null 2>&1
        code=$?
        set -e
        with_cop=$(( $(now_ns) - start ))

        start=$(now_ns)
        set +e
        "$ROOT/bin/nano_vm" --repeat 1 "$ffi_nvm" >/dev/null 2>&1
        set -e
        without_cop=$(( $(now_ns) - start ))

        delta=$(( with_cop - without_cop ))
        [ "$delta" -lt 0 ] && delta=0
        printf 'coprocess_launch\t%d\t%d\t%d\t\n' "$run" "$delta" "$code" >> "$manifest"
    done
fi

python3 "$ROOT/scripts/summarize_nanoisa_bench.py" "$OUT" >/dev/null
printf 'NanoISA benchmark results: %s\n' "$manifest"
printf 'NanoISA benchmark summary: %s\n' "$OUT/summary.json"
printf '\nPer-iteration execution time, startup excluded (%d samples, %d iterations each):\n' \
    "$RUNS" "$REPEAT"
python3 - "$OUT/summary.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
rows = summary["workloads"]
width = max(len(w["name"]) for w in rows)
print(f'  {"workload".ljust(width)}  {"median":>12}  {"IQR":>10}  {"IQR/median":>10}')
for w in rows:
    iqr = w["p75_ns"] - w["p25_ns"]
    share = (iqr / w["median_ns"] * 100) if w["median_ns"] else 0.0
    unit = f'{w["median_ns"] / 1000:.1f} us' if w["median_ns"] else "0"
    print(f'  {w["name"].ljust(width)}  {unit:>12}  {iqr / 1000:9.1f} us  {share:9.1f}%')
print("\n  IQR/median is the run-to-run noise. The optimization policy accepts a")
print("  change only when its median improvement exceeds this band.")
PY
