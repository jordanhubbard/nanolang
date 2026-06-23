#!/usr/bin/env bash

set -euo pipefail

NANOC_BIN=${1:-"./bin/nanoc"}
NANOC_C_BIN=${2:-"./bin/nanoc_c"}
SMOKE_SOURCE=${3:-"examples/nl_hello.nano"}
OUT_BIN=${4:-"$(mktemp /tmp/nanolang_verify_no_nanoc_c.XXXXXX)"}

# Remove the mktemp placeholder so the compiler can create it again
rm -f "$OUT_BIN"

if [[ ! -x "$NANOC_BIN" ]]; then
  echo "❌ verify-no-nanoc_c: missing compiler at $NANOC_BIN" >&2
  exit 1
fi

if [[ ! -f "$SMOKE_SOURCE" ]]; then
  echo "❌ verify-no-nanoc_c: missing smoke test source $SMOKE_SOURCE" >&2
  exit 1
fi

TMP_BACKUP=""

restore() {
  local status=$?
  rm -f "$OUT_BIN"
  if [[ -n "$TMP_BACKUP" && -f "$TMP_BACKUP" ]]; then
    mv "$TMP_BACKUP" "$NANOC_C_BIN"
  fi
  exit $status
}

trap restore EXIT

if [[ -x "$NANOC_C_BIN" ]]; then
  TMP_BACKUP="${NANOC_C_BIN}.hidden.$$"
  mv "$NANOC_C_BIN" "$TMP_BACKUP"
fi

echo "🔍 Verifying self-hosted compiler works without bin/nanoc_c..."
"$NANOC_BIN" "$SMOKE_SOURCE" -o "$OUT_BIN" >/dev/null
"$OUT_BIN" >/dev/null
echo "✅ bin/nanoc succeeded with bin/nanoc_c removed"
