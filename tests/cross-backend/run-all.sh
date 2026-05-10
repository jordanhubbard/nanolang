#!/usr/bin/env bash
# tests/cross-backend/run-all.sh — run the full cross-backend compile suite locally
#
# Compiles all canonical test programs against all 5 nanolang backends and
# verifies output by executing each backend's artifact against the .expected
# file. Structural-only validation (wasm-validate / llvm-as) is reported
# separately from real execution so the two cannot be confused.
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
# Dependencies (optional, checked at runtime — missing tools → SKIP, not PASS):
#   wasmtime / wasm3       — WASM execution
#   wasm-validate          — WASM structural validation (apt install wabt)
#   lli                    — LLVM IR interpreter (apt install llvm; macOS: brew install llvm)
#   llvm-as                — LLVM IR validation
#   gcc / cc               — C backend execution

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

NANOC="${1:-$REPO_ROOT/bin/nanoc}"

if [ ! -x "$NANOC" ]; then
    echo "ERROR: compiler not found at $NANOC"
    echo "Run 'make stage1' first, or pass the compiler path as an argument."
    exit 1
fi

# Add Homebrew LLVM to PATH if available (macOS), so lli/llvm-as are findable.
if [ -d /opt/homebrew/opt/llvm/bin ] && ! command -v lli >/dev/null 2>&1; then
    export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
fi

TMPDIR_TESTS="${TMPDIR:-/tmp}/nano_cross_backend_$$"
mkdir -p "$TMPDIR_TESTS"
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
    "$NANOC" "$nano_file" "$@" 2>/tmp/nano_compile_err_$$ || {
        cat /tmp/nano_compile_err_$$ >&2
        rm -f /tmp/nano_compile_err_$$
        return 1
    }
    rm -f /tmp/nano_compile_err_$$
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

test_wasm() {
    local nano_file="$1"
    local name="$2"
    local out="$TMPDIR_TESTS/${name}.wasm"
    local expected_file="${nano_file%.nano}.expected"

    if ! compile_nano "$nano_file" --target wasm -o "$out"; then
        report_outcome fail wasm "$nano_file" "$name" "compile error"
        return
    fi

    # Prefer execution over validation.
    local actual=""
    local executor=""
    if check_tool wasmtime; then
        executor="wasmtime"
        actual="$(wasmtime run --invoke main "$out" 2>/dev/null)" || true
    elif check_tool wasm3; then
        executor="wasm3"
        actual="$(wasm3 "$out" 2>/dev/null)" || true
    fi

    if [ -n "$executor" ]; then
        if [ ! -f "$expected_file" ]; then
            emit_result SKIP wasm "$name" "no .expected file"
            return
        fi
        if diff_output "$actual" "$expected_file"; then
            report_outcome pass wasm "$nano_file" "$name"
        else
            local detail; detail="$(printf 'output mismatch via %s (got: %q)' "$executor" "$actual")"
            report_outcome fail wasm "$nano_file" "$name" "$detail"
        fi
        return
    fi

    # No executor — fall back to structural validation only, reported loudly.
    if check_tool wasm-validate; then
        if wasm-validate "$out" 2>/dev/null; then
            emit_result VALIDATE wasm "$name" "wasm-validate ok; no executor installed"
        else
            report_outcome fail wasm "$nano_file" "$name" "wasm-validate rejected output"
        fi
    else
        emit_result SKIP wasm "$name" "no wasm executor or validator installed"
    fi
}

test_llvm() {
    local nano_file="$1"
    local name="$2"
    local out="$TMPDIR_TESTS/${name}.ll"
    local expected_file="${nano_file%.nano}.expected"

    if ! compile_nano "$nano_file" --target llvm -o "$out"; then
        report_outcome fail llvm "$nano_file" "$name" "compile error"
        return
    fi

    # Prefer execution via lli. Falls back to structural validation if absent.
    if check_tool lli; then
        if [ ! -f "$expected_file" ]; then
            emit_result SKIP llvm "$name" "no .expected file"
            return
        fi
        local actual
        actual="$(lli "$out" 2>/dev/null)" || true
        if diff_output "$actual" "$expected_file"; then
            report_outcome pass llvm "$nano_file" "$name"
        else
            local detail; detail="$(printf 'output mismatch via lli (got: %q)' "$actual")"
            report_outcome fail llvm "$nano_file" "$name" "$detail"
        fi
        return
    fi

    # No lli — fall back to structural-only.
    if check_tool llvm-as; then
        if llvm-as "$out" -o /dev/null 2>/dev/null; then
            emit_result VALIDATE llvm "$name" "llvm-as ok; lli not installed"
        else
            report_outcome fail llvm "$nano_file" "$name" "llvm-as rejected IR"
        fi
    else
        emit_result SKIP llvm "$name" "no lli or llvm-as installed"
    fi
}

# RISC-V/PTX still structural-only — cross-execution requires qemu-user or
# CUDA hardware. Reported as VALIDATE so they don't masquerade as PASS.

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
    if ! "$cc" -std=gnu11 -o "$out_exe" "$out_c" 2>/dev/null; then
        report_outcome fail c "$nano_file" "$name" "$cc compilation failed"
        return
    fi
    if [ ! -f "$expected_file" ]; then
        emit_result SKIP c "$name" "no .expected file"
        return
    fi
    local actual; actual="$("$out_exe" 2>/dev/null)" || true
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
