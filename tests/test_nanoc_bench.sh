#!/usr/bin/env bash
# nanoc --bench must run real functions and write non-zero ns/op JSON.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

NANOC="${NANOC:-./bin/nanoc}"
SAMPLE="examples/bench_sample.nano"

if [ ! -x "$NANOC" ]; then
    echo "ERROR: $NANOC is not built"
    exit 1
fi

if [ ! -f "$SAMPLE" ]; then
    echo "ERROR: missing $SAMPLE"
    exit 1
fi

out="$(mktemp "${TMPDIR:-/tmp}/bench-results.XXXXXX.json")"
trap 'rm -f "$out"' EXIT

if ! perl -e 'alarm 60; exec @ARGV' \
        "$NANOC" "$SAMPLE" --bench --bench-n 8 --bench-json "$out"; then
    echo "ERROR: nanoc --bench failed"
    exit 1
fi

python3 - "$out" <<'PY'
import json, sys
path = sys.argv[1]
text = open(path).read().strip()
if not text:
    print("ERROR: empty bench-results.json")
    sys.exit(1)
rows = []
for line in text.splitlines():
    line = line.strip()
    if not line:
        continue
    if line.startswith("["):
        rows = json.loads(text)
        break
    rows.append(json.loads(line))
if len(rows) < 2:
    print(f"ERROR: expected at least two benchmarks, got {len(rows)}")
    sys.exit(1)
for row in rows:
    name = row.get("name")
    mean = float(row.get("mean_ns") or 0)
    ops = float(row.get("ops_per_sec") or 0)
    if mean <= 0 or ops <= 0:
        print(f"ERROR: {name} has zero measurement: {row}")
        sys.exit(1)
    print(f"  {name}: {mean:.3f} ns/op  {ops:.1f} ops/s")
print("nanoc --bench produced non-zero measurements")
PY
