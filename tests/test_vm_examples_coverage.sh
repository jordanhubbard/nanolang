#!/bin/bash
# =============================================================================
# Regression Test: every eligible example lowers to NanoVM bytecode
# =============================================================================
# `make vm-examples` used to report "skipped: 0" while quietly building only
# 154 of the 236 eligible examples. Two separate holes let that happen:
#
#   1. The VM source list was assembled from the native build's variables,
#      which filter on host library availability (SDL2, ncurses, OpenGL,
#      MuJoCo, Bullet, libuv, libreadline). nano_virt links nothing -- FFI is
#      resolved at run time by nano_vm's co-process loader -- so those filters
#      only hid examples. An example that is never attempted cannot be skipped,
#      so the counter stayed at zero.
#   2. Nine examples crashed nano_virt outright with SIGSEGV rather than
#      reporting an error.
#
# This test locks both holes shut, and guards the exclusion list against being
# used to bury the next failure.
#
# Usage: bash tests/test_vm_examples_coverage.sh
# Requires: bin/nano_virt (make nano_virt)
# =============================================================================

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VM_COMPILER="bin/nano_virt"
EXAMPLES_DIR="examples"
NATIVE_COMPILER="bin/nanoc_c"

export NANO_MODULE_PATH="${NANO_MODULE_PATH:-$REPO_ROOT/modules}"

failures=0
work_dir="$(mktemp -d "${TMPDIR:-/tmp}/vm_examples_coverage.XXXXXX")"
trap 'rm -rf "$work_dir"' EXIT

fail() {
    echo "  ❌ $*"
    failures=$((failures + 1))
}

echo "=========================================="
echo "NanoVM example coverage"
echo "=========================================="
echo ""

if [ ! -x "$VM_COMPILER" ]; then
    echo "❌ $VM_COMPILER not found. Run 'make nano_virt' first."
    exit 1
fi

# Ask the Makefile for the lists rather than duplicating them here, so the
# test tracks the build instead of drifting from it.
print_mk="$work_dir/print.mk"
printf 'include Makefile\nprint-%%: ; @echo $($*)\n' > "$print_mk"
# MAKEFLAGS/MAKELEVEL are unset so that running under `make test-vm-examples`
# does not mix "Entering directory" chatter into the variable's value.
mk_var() {
    (cd "$EXAMPLES_DIR" && unset MAKEFLAGS MAKELEVEL MFLAGS \
        && make -s --no-print-directory -f "$print_mk" "print-$1" 2>/dev/null) \
        | tr ' ' '\n' | sed '/^$/d' | sort -u
}

mk_var ALL_VM_SOURCES   > "$work_dir/eligible.txt"
mk_var VM_EXCLUDED_SOURCES > "$work_dir/excluded.txt"
find "$EXAMPLES_DIR" -name '*.nano' -not -path '*/.*' \
    | sed "s|^$EXAMPLES_DIR/||" | sort -u > "$work_dir/all.txt"

eligible_count=$(wc -l < "$work_dir/eligible.txt" | tr -d ' ')
excluded_count=$(wc -l < "$work_dir/excluded.txt" | tr -d ' ')
all_count=$(wc -l < "$work_dir/all.txt" | tr -d ' ')

echo "Examples on disk: $all_count   eligible: $eligible_count   excluded: $excluded_count"
echo ""

# ---------------------------------------------------------------------------
# Test 1: the eligible list plus the exclusion list must account for every
# example on disk. This is what catches a host-library filter creeping back in
# and silently shrinking the VM build.
# ---------------------------------------------------------------------------
echo "Test 1: VM build covers the whole example tree"

sort -u "$work_dir/eligible.txt" "$work_dir/excluded.txt" > "$work_dir/accounted.txt"
if ! missing=$(comm -13 "$work_dir/accounted.txt" "$work_dir/all.txt") || [ -n "$missing" ]; then
    fail "these examples are neither built for the VM nor declared excluded:"
    printf '       %s\n' $missing
    echo "       Add them to the VM build, or declare them in examples/Makefile."
fi

if [ "$eligible_count" -eq 0 ]; then
    fail "ALL_VM_SOURCES is empty"
fi

stale=$(comm -13 "$work_dir/all.txt" "$work_dir/excluded.txt")
if [ -n "$stale" ]; then
    fail "VM_EXCLUDED_SOURCES names sources that no longer exist:"
    printf '       %s\n' $stale
fi

[ "$failures" -eq 0 ] && echo "  ✓ every example is either built or explicitly declared"
echo ""

# ---------------------------------------------------------------------------
# Test 2: every eligible example must actually lower to bytecode.
# ---------------------------------------------------------------------------
echo "Test 2: every eligible example compiles to NanoISA bytecode"

skipped_list=""
skipped=0
while read -r src; do
    [ -n "$src" ] || continue
    log="$work_dir/build.log"
    if (cd "$EXAMPLES_DIR" && "$REPO_ROOT/$VM_COMPILER" "$src" --emit-nvm \
            -o "$work_dir/out.nvm") > "$log" 2>&1; then
        continue
    fi
    rc=$?
    reason=$(grep -m1 '^error:' "$log" | sed 's/^error: //')
    if [ -z "$reason" ]; then
        if [ "$rc" -gt 128 ]; then
            reason="nano_virt died on signal $((rc - 128)) with no diagnostic"
        else
            reason="nano_virt exited $rc with no diagnostic"
        fi
    fi
    skipped_list="$skipped_list
       $src: $reason"
    skipped=$((skipped + 1))
done < "$work_dir/eligible.txt"

if [ "$skipped" -ne 0 ]; then
    fail "$skipped of $eligible_count eligible examples did not compile:$skipped_list"
    echo "       Fix the NanoVirt/compiler gap rather than excluding the example."
else
    echo "  ✓ all $eligible_count eligible examples lowered to bytecode"
fi
echo ""

# ---------------------------------------------------------------------------
# Test 3: the exclusion list has to stay honest.
#
# VM_EXCLUDED_SOURCES exists for sources that are not standalone programs --
# library modules with no main, a deliberately-invalid diagnostics fixture,
# unmaintained sketches the native compiler rejects too. If an excluded source
# compiles for the VM, it is no longer a valid exclusion and belongs in the
# build. Without this check, the cheapest way to make Test 2 pass would be to
# append the failing example here.
# ---------------------------------------------------------------------------
echo "Test 3: every exclusion is still genuinely ineligible"

now_compiles=""
while read -r src; do
    [ -n "$src" ] || continue
    [ -f "$EXAMPLES_DIR/$src" ] || continue
    if (cd "$EXAMPLES_DIR" && "$REPO_ROOT/$VM_COMPILER" "$src" --emit-nvm \
            -o "$work_dir/out.nvm") > /dev/null 2>&1; then
        now_compiles="$now_compiles
       $src"
    fi
done < "$work_dir/excluded.txt"

if [ -n "$now_compiles" ]; then
    fail "these are excluded from the VM build but compile fine:$now_compiles"
    echo "       Remove them from VM_EXCLUDED_SOURCES in examples/Makefile."
else
    echo "  ✓ all $excluded_count exclusions still fail to compile"
fi
echo ""

# ---------------------------------------------------------------------------
# Test 4: an excluded source must be one the native compiler rejects too.
#
# This is the rule the exclusion list encodes: exclusions are for broken or
# non-program sources, never for "nano_virt cannot do this yet". Library
# modules are exempt -- they legitimately fail both, for the same reason.
# Skipped when nanoc_c has not been built.
# ---------------------------------------------------------------------------
echo "Test 4: exclusions are broken sources, not NanoVirt gaps"

if [ ! -x "$NATIVE_COMPILER" ]; then
    echo "  – skipped ($NATIVE_COMPILER not built)"
else
    native_ok=""
    while read -r src; do
        [ -n "$src" ] || continue
        [ -f "$EXAMPLES_DIR/$src" ] || continue
        if (cd "$EXAMPLES_DIR" && "$REPO_ROOT/$NATIVE_COMPILER" "$src" \
                -o "$work_dir/out_native") > /dev/null 2>&1; then
            native_ok="$native_ok
       $src"
        fi
    done < "$work_dir/excluded.txt"

    if [ -n "$native_ok" ]; then
        fail "these are excluded from the VM build but compile natively:$native_ok"
        echo "       An example the native build accepts is a NanoVirt gap to fix,"
        echo "       not an exclusion to declare."
    else
        echo "  ✓ no exclusion compiles natively"
    fi
fi
echo ""

# ---------------------------------------------------------------------------
# Test 5: deep statement nesting must not exhaust the stack.
#
# compile_stmt used to reserve ~135 KB per frame, because the AST_FUNCTION arm
# of its switch declared a full CG snapshot plus the locals/loops/upvalue
# tables and a compiler reserves the union of every arm in one frame. Statement
# compilation recurses (block -> if -> block -> ...), so about sixty levels of
# ordinary nesting overflowed an 8 MB stack: nano_virt took SIGSEGV and
# `make vm-examples` counted it as an ordinary skip. Nine real examples,
# including every REPL sample and the SDL launcher, died this way.
#
# 400 levels is far past the old ~60 limit and far below anything a real
# program needs.
# ---------------------------------------------------------------------------
echo "Test 5: deeply nested statements compile without crashing"

deep_src="$work_dir/deep_nesting.nano"
python3 - "$deep_src" <<'PY'
import sys

DEPTH = 400
out = ["fn main() -> int {", "    let mut acc: int = 0"]
for i in range(DEPTH):
    out.append("    " + "    " * i + "if (> acc %d) {" % (-i - 1))
    out.append("    " + "    " * (i + 1) + "set acc (+ acc 1)")
for i in reversed(range(DEPTH)):
    out.append("    " + "    " * i + "}")
out.append("    (println acc)")
out.append("    return 0")
out.append("}")
open(sys.argv[1], "w").write("\n".join(out) + "\n")
PY

if "$VM_COMPILER" "$deep_src" --emit-nvm -o "$work_dir/deep.nvm" > "$work_dir/deep.log" 2>&1; then
    echo "  ✓ 400 levels of nesting compiled"
    if [ -x bin/nano_vm ]; then
        got=$(bin/nano_vm "$work_dir/deep.nvm" 2>/dev/null | tr -d '[:space:]')
        if [ "$got" = "400" ]; then
            echo "  ✓ and executes correctly (acc = 400)"
        else
            fail "deeply nested program printed '$got', expected '400'"
        fi
    fi
else
    rc=$?
    if [ "$rc" -gt 128 ]; then
        fail "nano_virt died on signal $((rc - 128)) compiling 400 nested statements (stack overflow regression)"
    else
        fail "nano_virt failed on 400 nested statements: $(head -3 "$work_dir/deep.log")"
    fi
fi
echo ""

# ---------------------------------------------------------------------------
# Test 6: builtins the VM backend was missing.
#
# run_examples.nano needed tmp_dir, and the gpu/ examples needed the launch
# geometry intrinsics. Both were absent from nano_virt's builtin tables, so
# those examples failed codegen with "undefined function".
# ---------------------------------------------------------------------------
echo "Test 6: VM builtins that examples depend on"

if [ "$(uname -s)" = "Linux" ]; then
    if nm -D bin/nano_vm | grep -q ' vm_tmp_dir$'; then
        echo "  ✓ nano_vm exports builtins even when LDFLAGS is overridden"
    else
        fail "nano_vm does not export its builtin symbols for FFI resolution"
    fi
fi

builtin_src="$work_dir/builtins.nano"
cat > "$builtin_src" <<'EOF'
fn main() -> int {
    (println (tmp_dir))
    (println (mktemp "vm_coverage_"))
    (println (global_id_x))
    (println (global_id_y))
    (println (thread_id_x))
    (println (block_dim_x))
    (println (grid_dim_z))
    return 0
}
EOF

if "$VM_COMPILER" "$builtin_src" --emit-nvm -o "$work_dir/builtins.nvm" \
        > "$work_dir/builtins.log" 2>&1; then
    echo "  ✓ tmp_dir, mktemp and the GPU geometry intrinsics compile"
    if [ -x bin/nano_vm ]; then
        out=$(bin/nano_vm "$work_dir/builtins.nvm" 2>/dev/null)
        tmp_line=$(echo "$out" | sed -n 1p)
        mktemp_line=$(echo "$out" | sed -n 2p)
        # The launch geometry collapses to "thread 0 of one block" on a host
        # target, matching eval.c and the modules/gpu host stubs.
        geometry=$(echo "$out" | sed -n '3,7p' | tr '\n' ' ' | sed 's/ *$//')

        [ -d "$tmp_line" ] || fail "tmp_dir returned '$tmp_line', which is not a directory"
        [ -f "$mktemp_line" ] || fail "mktemp returned '$mktemp_line', which is not a file"
        rm -f "$mktemp_line"

        if [ "$geometry" = "0 0 0 256 1" ]; then
            echo "  ✓ and return the documented host-target values"
        else
            fail "GPU geometry returned '$geometry', expected '0 0 0 256 1'"
        fi
    fi
else
    fail "missing VM builtin: $(grep -m1 '^error:' "$work_dir/builtins.log")"
fi
echo ""

echo "=========================================="
if [ "$failures" -eq 0 ]; then
    echo "✅ NanoVM example coverage: all checks passed"
    echo "   $eligible_count examples build for the VM."
    exit 0
fi
echo "❌ NanoVM example coverage: $failures check(s) failed"
exit 1
