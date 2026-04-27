#!/usr/bin/env bash
# test_nano_fmt.sh — tests for the nano-fmt code formatter
NANO_FMT="${1:-./bin/nano-fmt}"
PASS=0; FAIL=0

check_output() {
    local desc="$1"; local got_file="$2"; local expected_file="$3"
    if diff "$expected_file" "$got_file" > /dev/null 2>&1; then
        echo "  ✅ $desc"; PASS=$((PASS+1))
    else
        echo "  ❌ $desc"; diff "$expected_file" "$got_file" | head -5; FAIL=$((FAIL+1))
    fi
}

echo "nano-fmt tests:"

# ── Test 1: basic function formatting ─────────────────────────────────
cat > /tmp/nf_input1.nano << 'NANO'
fn add(a:int,b:int)->int{return (+ a b)}
shadow add{assert (==(add 1 2) 3)}
NANO

cat > /tmp/nf_expected1.nano << 'NANO'
fn add(a: int, b: int) -> int {
    return (+ a b)
}

shadow add {
    assert (== (add 1 2) 3)
}
NANO

"$NANO_FMT" /tmp/nf_input1.nano > /tmp/nf_got1.nano
check_output "basic fn formatting" /tmp/nf_got1.nano /tmp/nf_expected1.nano

# ── Test 2: let/return formatting ──────────────────────────────────────
cat > /tmp/nf_input2.nano << 'NANO'
fn main()->int{let x:int=42
return x}
shadow main{assert (==(main) 42)}
NANO

cat > /tmp/nf_expected2.nano << 'NANO'
fn main() -> int {
    let x: int = 42
    return x
}

shadow main {
    assert (== (main) 42)
}
NANO

"$NANO_FMT" /tmp/nf_input2.nano > /tmp/nf_got2.nano
check_output "let/return formatting" /tmp/nf_got2.nano /tmp/nf_expected2.nano

# ── Test 3: idempotency ────────────────────────────────────────────────
"$NANO_FMT" /tmp/nf_got1.nano > /tmp/nf_idem2.nano
diff /tmp/nf_got1.nano /tmp/nf_idem2.nano > /dev/null 2>&1 && {
    echo "  ✅ idempotency"; PASS=$((PASS+1))
} || {
    echo "  ❌ idempotency"; diff /tmp/nf_got1.nano /tmp/nf_idem2.nano | head -5; FAIL=$((FAIL+1))
}

# ── Test 4: --check (already formatted) ────────────────────────────────
"$NANO_FMT" --check /tmp/nf_expected1.nano > /dev/null 2>&1 && {
    echo "  ✅ --check exits 0 for formatted file"; PASS=$((PASS+1))
} || {
    echo "  ❌ --check exits non-0 for formatted file"; FAIL=$((FAIL+1))
}

# ── Test 5: --check (needs reformatting) ───────────────────────────────
"$NANO_FMT" --check /tmp/nf_input1.nano > /dev/null 2>&1; RC=$?
[ "$RC" = "2" ] && {
    echo "  ✅ --check exits 2 for unformatted file"; PASS=$((PASS+1))
} || {
    echo "  ❌ --check should exit 2 (got $RC)"; FAIL=$((FAIL+1))
}

# ── Test 6: --write mode ────────────────────────────────────────────────
cp /tmp/nf_input1.nano /tmp/nf_write.nano
"$NANO_FMT" --write /tmp/nf_write.nano
diff /tmp/nf_expected1.nano /tmp/nf_write.nano > /dev/null 2>&1 && {
    echo "  ✅ --write reformats in place"; PASS=$((PASS+1))
} || {
    echo "  ❌ --write result wrong"; FAIL=$((FAIL+1))
}

# ── Test 7: string literals preserved ──────────────────────────────────
cat > /tmp/nf_str.nano << 'NANO'
fn greet(name: string) -> string {
    return (str_concat "Hello, " name)
}
shadow greet {
    assert (str_equals (greet "World") "Hello, World")
}
NANO
"$NANO_FMT" /tmp/nf_str.nano | grep -q '"Hello, "' && {
    echo "  ✅ string literals preserved"; PASS=$((PASS+1))
} || {
    echo "  ❌ string literals broken"; FAIL=$((FAIL+1))
}

# ── Test 8: binary operators spaced ────────────────────────────────────
cat > /tmp/nf_ops.nano << 'NANO'
fn calc(a: int) -> int { return (+ (* a 2) (- a 1)) }
shadow calc { assert (== (calc 5) 14) }
NANO
"$NANO_FMT" /tmp/nf_ops.nano | fgrep -q '(+ (* a 2) (- a 1))' && {
    echo "  ✅ binary operators spaced correctly"; PASS=$((PASS+1))
} || {
    echo "  ❌ binary operator spacing wrong"
    "$NANO_FMT" /tmp/nf_ops.nano | grep return
    FAIL=$((FAIL+1))
}

# cleanup
rm -f /tmp/nf_*.nano /tmp/nf_*.nano

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ $FAIL -eq 0 ]
