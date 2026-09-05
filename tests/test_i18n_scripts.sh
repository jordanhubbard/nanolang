#!/usr/bin/env bash
# Compile and run a program whose string literals use the six Phase 15 scripts.
set -euo pipefail

NANOC="${1:?usage: test_i18n_scripts.sh <nanoc>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/examples/language/i18n_six_scripts.nano"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/nano-i18n.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

run() {
    perl -e 'alarm 30; exec @ARGV' -- "$@"
}

run "$NANOC" "$SRC" -o "$WORK/six" >/dev/null
out="$("$WORK/six")"
printf '%s' "$out" | grep -q 'Hello' || { echo "C path missing Hello: $out" >&2; exit 1; }
printf '%s' "$out" | grep -q '你好' || { echo "C path missing Mandarin: $out" >&2; exit 1; }
printf '%s' "$out" | grep -q 'नमस्ते' || { echo "C path missing Hindi: $out" >&2; exit 1; }
printf '%s' "$out" | grep -q 'Hola' || { echo "C path missing Spanish: $out" >&2; exit 1; }
printf '%s' "$out" | grep -q 'مرحبا' || { echo "C path missing Arabic: $out" >&2; exit 1; }
printf '%s' "$out" | grep -q 'Bonjour' || { echo "C path missing French: $out" >&2; exit 1; }

if [ -x "$ROOT/bin/nano_virt" ] && [ -x "$ROOT/bin/nano_vm" ]; then
    run "$ROOT/bin/nano_virt" "$SRC" --emit-nvm -o "$WORK/six.nvm" >/dev/null
    vmout="$(run "$ROOT/bin/nano_vm" "$WORK/six.nvm")"
    printf '%s' "$vmout" | grep -q '你好' || { echo "NanoVM path missing Mandarin: $vmout" >&2; exit 1; }
    printf '%s' "$vmout" | grep -q 'مرحبا' || { echo "NanoVM path missing Arabic: $vmout" >&2; exit 1; }
else
    echo "nano_virt/nano_vm not built; skipped NanoVM path" >&2
    exit 1
fi

echo "six-script C and NanoVM: PASS"
