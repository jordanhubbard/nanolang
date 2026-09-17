#!/usr/bin/env bash
# tests/cross-backend/run-all.sh — run the full cross-backend compile suite locally
#
# I execute the direct C artifact and separately report RISC-V/PTX structural
# checks. My NanoISA LLVM/Wasm translators have their own acceptance gates.
#
# Usage:
#   ./tests/cross-backend/run-all.sh [path/to/nanoc]
#
# Per-test xfail support:
#   Create <test>.xfail next to <test>.nano. List backend names (one per
#   line, # for comments) that are known to fail. Listed backends report
#   XFAIL instead of FAIL. If the test actually passes, XPASS is reported
#   and the xfail entry should be removed.
#
# Dependency: gcc / cc for direct C execution.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

NANOC="${1:-$REPO_ROOT/bin/nanoc}"

if [ ! -x "$NANOC" ]; then
    echo "ERROR: compiler not found at $NANOC"
    echo "Run 'make stage1' first, or pass the compiler path as an argument."
    exit 1
fi

TMPDIR_TESTS="$(mktemp -d "${TMPDIR:-/tmp}/nano_cross_backend.XXXXXX")"
trap 'rm -rf "$TMPDIR_TESTS"' EXIT

PASS=0
FAIL=0
SKIP=0
XFAIL=0
XPASS=0
VALIDATE_ONLY=0

# ── helpers ──────────────────────────────────────────────────────────────────

check_tool() { command -v "$1" >/dev/null 2>&1; }

# is_xfail <backend> <nano_file>: returns 0 if the backend is listed in <test>.xfail
is_xfail() {
    local backend="$1"
    local nano_file="$2"
    local xfail_file="${nano_file%.nano}.xfail"
    [ -f "$xfail_file" ] || return 1
    grep -vE '^\s*(#|$)' "$xfail_file" | grep -qxF "$backend"
}

# emit_result <kind> <backend> <name> [<detail>]
emit_result() {
    local kind="$1"
    local backend="$2"
    local name="$3"
    local detail="${4:-}"
    local tag
    case "$kind" in
        PASS)         tag="PASS         "; PASS=$((PASS + 1)) ;;
        FAIL)         tag="FAIL         "; FAIL=$((FAIL + 1)) ;;
        SKIP)         tag="SKIP         "; SKIP=$((SKIP + 1)) ;;
        XFAIL)        tag="XFAIL        "; XFAIL=$((XFAIL + 1)) ;;
        XPASS)        tag="XPASS        "; XPASS=$((XPASS + 1)) ;;
        VALIDATE)     tag="VALIDATE-ONLY"; VALIDATE_ONLY=$((VALIDATE_ONLY + 1)) ;;
        *)            tag="$kind        " ;;
    esac
    if [ -n "$detail" ]; then
        echo "  $tag  $name — $detail"
    else
        echo "  $tag  $name"
    fi
}

# report_outcome <pass|fail> <backend> <nano_file> <name> [<detail>]
# Applies xfail logic and emits the right kind.
report_outcome() {
    local outcome="$1"
    local backend="$2"
    local nano_file="$3"
    local name="$4"
    local detail="${5:-}"
    if is_xfail "$backend" "$nano_file"; then
        if [ "$outcome" = "pass" ]; then
            emit_result XPASS "$backend" "$name" "xfail entry should be removed"
        else
            emit_result XFAIL "$backend" "$name" "$detail"
        fi
    else
        if [ "$outcome" = "pass" ]; then
            emit_result PASS "$backend" "$name"
        else
            emit_result FAIL "$backend" "$name" "$detail"
        fi
    fi
}

compile_nano() {
    local nano_file="$1"
    shift
    "$NANOC" "$nano_file" "$@" 2>"$TMPDIR_TESTS/compile.err" || {
        cat "$TMPDIR_TESTS/compile.err" >&2
        return 1
    }
}

# diff_output <actual_text> <expected_file>: returns 0 on match
diff_output() {
    local actual="$1"
    local expected_file="$2"
    local expected
    expected="$(cat "$expected_file")"
    # tolerate a single trailing newline either way
    if [ "$actual" = "$expected" ] || [ "$actual" = "${expected%$'\n'}" ] || [ "${actual}"$'\n' = "$expected" ]; then
        return 0
    fi
    return 1
}

# show_mismatch <actual_text> <expected_file>
show_mismatch() {
    echo "    expected: $(printf '%q' "$(cat "$2")")"
    echo "    actual:   $(printf '%q' "$1")"
}

# ── per-backend test runners ──────────────────────────────────────────────────

test_riscv() {
    local nano_file="$1"
    local name="$2"
    local out="$TMPDIR_TESTS/${name}.s"
    if ! compile_nano "$nano_file" --target riscv -o "$out"; then
        report_outcome fail riscv "$nano_file" "$name" "compile error"
        return
    fi
    if grep -q "\.text" "$out" 2>/dev/null; then
        emit_result VALIDATE riscv "$name" "assembly looks well-formed; no qemu-user wired up"
    else
        report_outcome fail riscv "$nano_file" "$name" "no .text section in output"
    fi
}

test_c() {
    local nano_file="$1"
    local name="$2"
    local out_c="$TMPDIR_TESTS/${name}.c"
    local out_exe="$TMPDIR_TESTS/${name}"
    local expected_file="${nano_file%.nano}.expected"
    if ! compile_nano "$nano_file" --target c -o "$out_c"; then
        report_outcome fail c "$nano_file" "$name" "compile error"
        return
    fi
    local cc=""
    if check_tool gcc; then cc=gcc
    elif check_tool cc; then cc=cc
    else
        emit_result SKIP c "$name" "no C compiler found"
        return
    fi
    # I supply repository runtime headers for self-hosted C output. These
    # scalar corpus programs need libc/libm, not external module libraries.
    if ! "$cc" -O2 -std=gnu11 -I"$REPO_ROOT/src" -I"$REPO_ROOT/modules/std" \
            -o "$out_exe" "$out_c" -lm 2>"$TMPDIR_TESTS/c-compile.err"; then
        report_outcome fail c "$nano_file" "$name" "$cc compilation failed"
        cat "$TMPDIR_TESTS/c-compile.err" >&2
        return
    fi
    if [ ! -f "$expected_file" ]; then
        emit_result SKIP c "$name" "no .expected file"
        return
    fi
    local actual
    if ! actual="$("$out_exe" 2>"$TMPDIR_TESTS/runtime.err")"; then
        report_outcome fail c "$nano_file" "$name" "execution failed"
        cat "$TMPDIR_TESTS/runtime.err" >&2
        return
    fi
    if diff_output "$actual" "$expected_file"; then
        report_outcome pass c "$nano_file" "$name"
    else
        local detail; detail="$(printf 'output mismatch (got: %q)' "$actual")"
        report_outcome fail c "$nano_file" "$name" "$detail"
    fi
}

test_ptx() {
    local nano_file="$1"
    local name="$2"
    local out="$TMPDIR_TESTS/${name}.ptx"
    if ! compile_nano "$nano_file" --target ptx -o "$out"; then
        report_outcome fail ptx "$nano_file" "$name" "compile error"
        return
    fi
    if grep -q "\.target sm_" "$out" 2>/dev/null; then
        emit_result VALIDATE ptx "$name" "PTX header ok; no CUDA hardware wired up"
    else
        report_outcome fail ptx "$nano_file" "$name" "PTX header (.target sm_*) not found in output"
    fi
}

# ── main loop ────────────────────────────────────────────────────────────────

read -r -a BACKENDS <<<"${NANOLANG_TEST_BACKENDS-riscv c ptx}"
if [ "${#BACKENDS[@]}" -eq 0 ]; then
    echo "I need at least one backend to test." >&2
    exit 1
fi
for backend in "${BACKENDS[@]}"; do
    case "$backend" in
        riscv|c|ptx) ;;
        wasm|llvm) echo "I retired direct AST $backend compilation; use the separate test-nvm2$backend gate." >&2; exit 1 ;;
        *) echo "I do not recognize backend: $backend" >&2; exit 1 ;;
    esac
done
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
            riscv) test_riscv "$nano_file" "$name" ;;
            c)     test_c     "$nano_file" "$name" ;;
            ptx)   test_ptx   "$nano_file" "$name" ;;
        esac
    done
    echo ""
done

TOTAL=$((PASS + FAIL + SKIP + XFAIL + XPASS + VALIDATE_ONLY))
echo "=== Results ==="
echo "  PASS:          $PASS"
echo "  FAIL:          $FAIL"
echo "  XFAIL:         $XFAIL  (expected failures, listed in .xfail)"
echo "  XPASS:         $XPASS  (unexpected passes — clean up .xfail entries)"
echo "  VALIDATE-ONLY: $VALIDATE_ONLY  (structural check, no execution)"
echo "  SKIP:          $SKIP  (missing tools or .expected)"
echo "  TOTAL:         $TOTAL"
echo ""

# Non-zero exit only on real failures or unexpected passes. xfail and
# validate-only are visible-but-not-blocking; skip is host-environment.
if [ "$FAIL" -gt 0 ] || [ "$XPASS" -gt 0 ]; then
    exit 1
fi
exit 0
