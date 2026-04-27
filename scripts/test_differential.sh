#!/bin/bash
# Differential Testing: Coq-extracted reference interpreter vs NanoVM
#
# Runs the same NanoCore programs through both:
#   1. nanocore-ref (OCaml, extracted from Coq proofs — provably correct)
#   2. nano_virt --run (C VM — unverified implementation)
#
# Any divergence indicates a bug in the VM (since the reference is proven correct).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REF="$ROOT_DIR/tools/nanocore_ref/nanocore-ref"
VIRT="$ROOT_DIR/bin/nano_virt"
TEST_DIR="$ROOT_DIR/tests/differential"

passed=0
failed=0
skipped=0
errors=""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

# Check prerequisites
if [ ! -x "$REF" ]; then
    echo "Reference interpreter not found at $REF"
    echo "Build it with: cd formal && make extract && make nanocore-ref"
    exit 1
fi

if [ ! -x "$VIRT" ]; then
    echo "nano_virt not found at $VIRT"
    echo "Build it with: make vm"
    exit 1
fi

echo "Differential Testing: Coq Reference vs NanoVM"
echo "=============================================="
echo ""

# Each test case is a pair of files:
#   foo.sexp  — NanoCore S-expression for nanocore-ref
#   foo.nano  — Equivalent NanoLang source for nano_virt
#   foo.expected — Expected output (optional, for documentation)
#
# We compare stdout from both and flag divergence.

for sexp_file in "$TEST_DIR"/*.sexp; do
    [ -f "$sexp_file" ] || continue
    base=$(basename "$sexp_file" .sexp)
    nano_file="$TEST_DIR/$base.nano"

    if [ ! -f "$nano_file" ]; then
        printf "  ${YELLOW}SKIP${NC} %-40s (no .nano file)\n" "$base"
        skipped=$((skipped + 1))
        continue
    fi

    # Run reference interpreter (outputs Coq values like "(VInt 42)")
    ref_raw=$("$REF" < "$sexp_file" 2>/dev/null || echo "REF_ERROR")

    # Normalize reference output: extract printable value
    # (VInt N) -> N, (VBool true/false) -> true/false, (VString "s") -> s
    ref_output=$(echo "$ref_raw" | sed \
        -e 's/^(VInt \(.*\))$/\1/' \
        -e 's/^(VBool \(.*\))$/\1/' \
        -e 's/^(VString "\(.*\)")$/\1/' \
        -e 's/^(VUnit)$//' | tr -d '[:space:]')

    # Run VM (outputs program stdout — the .nano file should println the result)
    vm_raw=$("$VIRT" "$nano_file" --run 2>/dev/null || echo "VM_ERROR")
    vm_output=$(echo "$vm_raw" | tr -d '[:space:]')

    # Compare
    if [ "$ref_output" = "$vm_output" ]; then
        printf "  ${GREEN}PASS${NC} %-40s\n" "$base"
        passed=$((passed + 1))
    elif [ -f "$TEST_DIR/$base.expected" ] && grep -qi "KNOWN DIVERGENCE" "$TEST_DIR/$base.expected"; then
        printf "  ${YELLOW}XDIV${NC} %-40s (known divergence)\n" "$base"
        skipped=$((skipped + 1))
    elif echo "$ref_raw" | grep -q "REF_ERROR"; then
        printf "  ${YELLOW}SKIP${NC} %-40s (reference interpreter error)\n" "$base"
        skipped=$((skipped + 1))
    elif echo "$vm_raw" | grep -q "VM_ERROR"; then
        printf "  ${RED}FAIL${NC} %-40s (VM error)\n" "$base"
        printf "         ref: %s\n" "$ref_raw"
        failed=$((failed + 1))
        errors="${errors}\n  $base: VM execution error (ref produced: $ref_raw)"
    else
        printf "  ${RED}FAIL${NC} %-40s (output mismatch)\n" "$base"
        printf "         ref: %s (raw: %s)\n" "$ref_output" "$ref_raw"
        printf "          vm: %s\n" "$vm_output"
        failed=$((failed + 1))
        errors="${errors}\n  $base: ref='$ref_output' vm='$vm_output'"
    fi
done

echo ""
echo "=============================================="
printf "Results: ${GREEN}%d passed${NC}" "$passed"
if [ "$failed" -gt 0 ]; then
    printf ", ${RED}%d failed${NC}" "$failed"
fi
if [ "$skipped" -gt 0 ]; then
    printf ", ${YELLOW}%d skipped${NC}" "$skipped"
fi
echo ""

if [ "$failed" -gt 0 ]; then
    printf "\n${RED}Failures:${NC}%b\n" "$errors"
    exit 1
fi
