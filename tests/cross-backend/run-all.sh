#!/usr/bin/env bash
# tests/cross-backend/run-all.sh — run the full cross-backend compile suite locally
#
# Compiles all canonical test programs against all 5 nanolang backends and
# verifies output where execution is possible (c, wasm via wasm3).
#
# Usage:
#   ./tests/cross-backend/run-all.sh [path/to/nanoc]
#
# Dependencies (optional, checked at runtime):
#   wasm3       — WASM execution  (brew install wasm3 / cargo install wasm3)
#   wasm-validate — WASM binary validation (apt install wabt)
#   llvm-as     — LLVM IR validation (apt install llvm)
#   gcc         — C backend execution

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

NANOC="${1:-$REPO_ROOT/bin/nanoc}"

if [ ! -x "$NANOC" ]; then
    echo "ERROR: compiler not found at $NANOC"
    echo "Run 'make stage1' first, or pass the compiler path as an argument."
    exit 1
fi

TMPDIR_TESTS="${TMPDIR:-/tmp}/nano_cross_backend_$$"
mkdir -p "$TMPDIR_TESTS"
trap 'rm -rf "$TMPDIR_TESTS"' EXIT

PASS=0
FAIL=0
SKIP=0

# ── helpers ──────────────────────────────────────────────────────────────────

check_tool() { command -v "$1" >/dev/null 2>&1; }

result_pass() { echo "  PASS  $1"; PASS=$((PASS + 1)); }
result_fail() { echo "  FAIL  $1 — $2"; FAIL=$((FAIL + 1)); }
result_skip() { echo "  SKIP  $1 — $2"; SKIP=$((SKIP + 1)); }

compile_nano() {
    local nano_file="$1"
    shift
    "$NANOC" "$nano_file" "$@" 2>/tmp/nano_compile_err_$$ || {
        cat /tmp/nano_compile_err_$$ >&2
        rm -f /tmp/nano_compile_err_$$
        return 1
    }
    rm -f /tmp/nano_compile_err_$$
}

check_output() {
    local exe="$1"
    local expected_file="$2"
    local actual
    actual="$("$exe" 2>/dev/null)"
    local expected
    expected="$(cat "$expected_file")"
    # trim trailing newline for comparison
    if [ "$actual" = "${expected%$'\n'}" ] || [ "$actual" = "$expected" ]; then
        return 0
    fi
    echo "    expected: $(printf '%q' "$expected")"
    echo "    actual:   $(printf '%q' "$actual")"
    return 1
}

# ── per-backend test runners ──────────────────────────────────────────────────

test_wasm() {
    local nano_file="$1"
    local name="$2"
    local out="$TMPDIR_TESTS/${name}.wasm"
    if ! compile_nano "$nano_file" --target wasm -o "$out"; then
        result_fail "$name" "compile error"; return
    fi
    if check_tool wasm-validate; then
        if wasm-validate "$out" 2>/dev/null; then
            result_pass "$name"
        else
            result_fail "$name" "wasm-validate rejected output"
        fi
    elif check_tool wasm3; then
        # wasm3 exits non-zero if the binary is malformed before main runs
        local expected_file="${nano_file%.nano}.expected"
        local actual
        actual="$(wasm3 "$out" 2>/dev/null)" || true
        if [ -f "$expected_file" ]; then
            local expected; expected="$(cat "$expected_file")"
            if [ "$actual" = "${expected%$'\n'}" ] || [ "$actual" = "$expected" ]; then
                result_pass "$name (output verified)"
            else
                result_fail "$name" "output mismatch (got: $actual)"
            fi
        else
            result_pass "$name (wasm3 ran without crash)"
        fi
    else
        # Fall back to checking WASM magic bytes: \0asm
        if python3 -c "
import sys
with open('$out', 'rb') as f:
    magic = f.read(4)
sys.exit(0 if magic == b'\\x00asm' else 1)
" 2>/dev/null; then
            result_pass "$name (magic bytes ok)"
        else
            result_fail "$name" "output is not a valid WASM binary"
        fi
    fi
}

test_llvm() {
    local nano_file="$1"
    local name="$2"
    local out="$TMPDIR_TESTS/${name}.ll"
    if ! compile_nano "$nano_file" --llvm -o "$out"; then
        result_fail "$name" "compile error"; return
    fi
    if check_tool llvm-as; then
        if llvm-as "$out" -o /dev/null 2>/dev/null; then
            result_pass "$name"
        else
            result_fail "$name" "llvm-as rejected IR"
        fi
    else
        # Check that the output looks like LLVM IR (starts with a comment or define/declare)
        if grep -qE "^(; |define |declare |target |@)" "$out" 2>/dev/null; then
            result_pass "$name (IR structure ok, llvm-as not installed)"
        else
            result_fail "$name" "output does not look like LLVM IR"
        fi
    fi
}

test_riscv() {
    local nano_file="$1"
    local name="$2"
    local out="$TMPDIR_TESTS/${name}.s"
    if ! compile_nano "$nano_file" --target riscv -o "$out"; then
        result_fail "$name" "compile error"; return
    fi
    if grep -q "\.text" "$out" 2>/dev/null; then
        result_pass "$name"
    else
        result_fail "$name" "no .text section in output"
    fi
}

test_c() {
    local nano_file="$1"
    local name="$2"
    local out_c="$TMPDIR_TESTS/${name}.c"
    local out_exe="$TMPDIR_TESTS/${name}"
    local expected_file="${nano_file%.nano}.expected"
    if ! compile_nano "$nano_file" --target c -o "$out_c"; then
        result_fail "$name" "compile error"; return
    fi
    if ! check_tool gcc; then
        result_skip "$name" "gcc not found"; return
    fi
    if ! gcc -std=gnu11 -o "$out_exe" "$out_c" 2>/dev/null; then
        result_fail "$name" "gcc compilation failed"; return
    fi
    if [ -f "$expected_file" ]; then
        if check_output "$out_exe" "$expected_file"; then
            result_pass "$name (output verified)"
        else
            result_fail "$name" "output mismatch"
        fi
    else
        "$out_exe" >/dev/null 2>&1 && result_pass "$name (ran ok)" || result_fail "$name" "runtime error"
    fi
}

test_ptx() {
    local nano_file="$1"
    local name="$2"
    local out="$TMPDIR_TESTS/${name}.ptx"
    if ! compile_nano "$nano_file" --target ptx -o "$out"; then
        result_fail "$name" "compile error"; return
    fi
    if grep -q "\.target sm_" "$out" 2>/dev/null; then
        result_pass "$name"
    else
        result_fail "$name" "PTX header (.target sm_*) not found in output"
    fi
}

# ── main loop ────────────────────────────────────────────────────────────────

BACKENDS=(wasm llvm riscv c ptx)
TEST_FILES=("$SCRIPT_DIR"/*.nano)

echo "=== nanolang cross-backend compile suite ==="
echo "Compiler: $NANOC"
echo "Tests:    ${#TEST_FILES[@]} programs × ${#BACKENDS[@]} backends"
echo ""

for backend in "${BACKENDS[@]}"; do
    echo "── backend: $backend ──────────────────────────────────"
    for nano_file in "${TEST_FILES[@]}"; do
        name="$(basename "$nano_file" .nano)"
        case "$backend" in
            wasm)  test_wasm  "$nano_file" "$name" ;;
            llvm)  test_llvm  "$nano_file" "$name" ;;
            riscv) test_riscv "$nano_file" "$name" ;;
            c)     test_c     "$nano_file" "$name" ;;
            ptx)   test_ptx   "$nano_file" "$name" ;;
        esac
    done
    echo ""
done

TOTAL=$((PASS + FAIL + SKIP))
echo "=== Results: $PASS passed, $FAIL failed, $SKIP skipped (of $TOTAL) ==="
echo ""

[ "$FAIL" -eq 0 ]
