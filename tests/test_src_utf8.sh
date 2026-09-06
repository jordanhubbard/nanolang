#!/usr/bin/env bash
# nanoc rejects invalid UTF-8 source with stable id CSRC01.
set -euo pipefail

NANOC="${1:?usage: test_src_utf8.sh <nanoc>}"
run_nanoc() {
    perl -e 'alarm 30; exec @ARGV' -- "$NANOC" "$@"
}

WORK="$(mktemp -d "${TMPDIR:-/tmp}/nano-src-utf8.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

printf 'fn main() -> int { return 0 }\nshadow main { assert (== (main) 0) }\n' >"$WORK/ok.nano"
printf 'fn main() -> int { return 0 }\xff\n' >"$WORK/bad.nano"

if ! run_nanoc "$WORK/ok.nano" -o "$WORK/ok" >/dev/null 2>&1; then
    echo "valid UTF-8 source should compile" >&2
    exit 1
fi

diag="$WORK/diag.json"
set +e
run_nanoc "$WORK/bad.nano" --llm-diags-json "$diag" -o "$WORK/bad" >/dev/null 2>&1
status=$?
set -e
if [ "$status" -eq 0 ]; then
    echo "invalid UTF-8 source should fail closed" >&2
    exit 1
fi
python3 - "$diag" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
assert data.get("success") is False
codes = [d.get("code") for d in data.get("diagnostics", [])]
assert "CSRC01" in codes, codes
PY

printf 'fn caf\xc3\xa9() -> int { return 0 }\nfn main() -> int { return 0 }\nshadow main { assert (== (main) 0) }\n' >"$WORK/cafe.nano"
cafe_diag="$WORK/cafe.json"
set +e
run_nanoc "$WORK/cafe.nano" --llm-diags-json "$cafe_diag" -o "$WORK/cafe" >/dev/null 2>&1
cafe_status=$?
set -e
if [ "$cafe_status" -eq 0 ]; then
    echo "non-ASCII identifier should not compile" >&2
    exit 1
fi
python3 - "$cafe_diag" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
assert data.get("success") is False
codes = [d.get("code") for d in data.get("diagnostics", [])]
assert "CSRC01" not in codes, codes
assert "L0003" in codes, codes
PY

printf 'fn main() -> int { return x }\nshadow main { assert (== (main) 0) }\n' >"$WORK/undef.nano"
undef_json="$WORK/undef.json"
set +e
run_nanoc "$WORK/undef.nano" --json-errors -o "$WORK/undef" >"$undef_json" 2>/dev/null
undef_status=$?
set -e
if [ "$undef_status" -eq 0 ]; then
    echo "undefined variable should fail typecheck" >&2
    exit 1
fi
python3 - "$undef_json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
codes = [d.get("code") for d in data.get("diagnostics", [])]
assert "E024" in codes, codes
PY
echo "nanoc source UTF-8: PASS"
