#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-"$ROOT/.benchmark_results/nanoisa"}
RUNS=${NANOISA_BENCH_RUNS:-20}

mkdir -p "$OUT"
make -C "$ROOT" nano_virt nano_vm >/dev/null

workloads=(
  examples/language/nl_fibonacci.nano
  examples/language/nl_forth_interpreter.nano
  examples/language/nl_hashmap_word_count.nano
  examples/language/nl_string_operations.nano
)

manifest="$OUT/manifest.tsv"
: > "$manifest"
printf 'workload\trun\telapsed_ns\tprofile\n' >> "$manifest"

for source in "${workloads[@]}"; do
    name=$(basename "$source" .nano)
    nvm="$OUT/$name.nvm"
    "$ROOT/bin/nano_virt" "$ROOT/$source" --emit-nvm -o "$nvm"

    for ((run = 1; run <= RUNS; run++)); do
        profile="$OUT/${name}.${run}.json"
        start=$(python3 -c 'import time; print(time.monotonic_ns())')
        "$ROOT/bin/nano_vm" --profile-isa "$profile" "$nvm" >/dev/null
        stop=$(python3 -c 'import time; print(time.monotonic_ns())')
        printf '%s\t%d\t%d\t%s\n' "$name" "$run" "$((stop - start))" "$profile" >> "$manifest"
    done
done

printf 'NanoISA benchmark results: %s\n' "$manifest"
