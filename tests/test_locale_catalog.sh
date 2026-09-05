#!/usr/bin/env bash
# Localized human stderr from UTF-8 catalogs; JSON stays English.
set -euo pipefail

NANOC="${1:?usage: test_locale_catalog.sh <nanoc>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export NANO_CATALOG_DIR="$ROOT/catalogs/messages"

run_nanoc() {
    perl -e 'alarm 30; exec @ARGV' -- "$NANOC" "$@"
}

missing="$ROOT/catalogs/messages/no-such-input.nano"
diag="$(mktemp "${TMPDIR:-/tmp}/nano-locale-diag.XXXXXX")"
trap 'rm -f "$diag"' EXIT

expect_human() {
    local tag="$1" needle="$2"
    local err
    err="$(run_nanoc --locale "$tag" --llm-diags-json "$diag" "$missing" -o /tmp/nano-locale-out 2>&1 || true)"
    printf '%s' "$err" | grep -q "$needle" || {
        echo "--locale $tag stderr missing catalog text:" >&2
        printf '%s\n' "$err" >&2
        exit 1
    }
    python3 - "$diag" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
codes = [d.get("code") for d in data.get("diagnostics", [])]
assert "CIO01" in codes, codes
msgs = [d.get("message") or "" for d in data.get("diagnostics", [])]
assert any("Could not open input file" in m for m in msgs), msgs
PY
}

expect_human en "Could not open input file"
expect_human zh "无法打开输入文件"
expect_human hi "इनपुट फ़ाइल नहीं खोल सका"
expect_human es "No pude abrir el archivo de entrada"
expect_human ar "تعذر فتح ملف الإدخال"
expect_human fr "Impossible d'ouvrir le fichier d'entrée"

cafe="$(mktemp "${TMPDIR:-/tmp}/nano-locale-cafe.XXXXXX.nano")"
printf 'fn caf\xc3\xa9() -> int { return 0 }\nfn main() -> int { return 0 }\nshadow main { assert (== (main) 0) }\n' >"$cafe"
zh_lex="$(run_nanoc --locale zh --llm-diags-json "$diag" "$cafe" -o /tmp/nano-locale-cafe 2>&1 || true)"
printf '%s' "$zh_lex" | grep -q "未知字节；标识符为 ASCII" || {
    echo "--locale zh L0003 stderr missing catalog text:" >&2
    printf '%s\n' "$zh_lex" >&2
    rm -f "$cafe"
    exit 1
}
python3 - "$diag" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
codes = [d.get("code") for d in data.get("diagnostics", [])]
assert "L0003" in codes, codes
msgs = [d.get("message") or "" for d in data.get("diagnostics", [])]
assert any("Unknown byte; identifiers are ASCII" in m for m in msgs), msgs
PY
rm -f "$cafe"

echo "nanoc locale catalogs: PASS"
