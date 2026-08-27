#!/usr/bin/env bash
# ============================================================================
# Module validator — keg-only detection and package-manager hints
# ============================================================================
#
# scripts/validate-modules.sh is what `make modules` runs. Two of its checks
# used to lie on macOS:
#
#   1. `pkg-config --exists readline` without Homebrew's keg-only prefix, so
#      an installed readline looked missing
#   2. `subprocess.run(['command', '-v', 'apt-get'], shell=True)`, which is
#      not a presence check (the -c string is just `command`) and exits 0 on
#      bash, so the remediation hint was always `sudo apt-get install …`
#
# This file pins both. It needs no extra packages: it greps the script for
# the traps, drives a stubbed pkg-config to prove the keg path is consulted,
# and on Darwin runs the real validator against the checkout.
#
# Usage:
#   bash tests/test_validate_modules.sh
# ============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

FAILURES=0
pass() { echo -e "${GREEN}✅${NC} $1"; }
fail() { echo -e "${RED}❌${NC} $1"; FAILURES=$((FAILURES + 1)); }

VALIDATOR="$PROJECT_ROOT/scripts/validate-modules.sh"

echo "=========================================="
echo "Module validator tests"
echo "=========================================="

# --- 1. the shell=True apt-get trap must not return -------------------------
# A list passed to subprocess.run(..., shell=True) makes the first item the
# -c string. `command` with no operands exits 0, so that pattern always
# reports apt as present. The validator documents that trap in comments;
# only a non-comment call site is a regression.

if grep "shell=True" "$VALIDATOR" | grep -v '#' >/dev/null; then
    fail "validator still uses subprocess.run(..., shell=True) as a presence check"
    grep -n "shell=True" "$VALIDATOR"
else
    pass "validator does not use shell=True to detect apt-get"
fi

if grep -q "shutil.which('apt-get')" "$VALIDATOR" && grep -q "sys.platform == 'darwin'" "$VALIDATOR"; then
    pass "validator prefers brew on Darwin and shutil.which elsewhere"
else
    fail "validator lost Darwin-first package-manager detection"
fi

# --- 2. keg-only prefixes must be on PKG_CONFIG_PATH ------------------------

if grep -q "/opt/homebrew/opt" "$VALIDATOR" && grep -q "/usr/local/opt" "$VALIDATOR"; then
    pass "validator searches Homebrew keg-only pkg-config prefixes"
else
    fail "validator no longer searches Homebrew opt prefixes"
fi

# --- 3. a stub pkg-config only succeeds when the keg path is exported ------

STUB_DIR="$(mktemp -d)"
trap 'rm -rf "$STUB_DIR"' EXIT

cat >"$STUB_DIR/pkg-config" <<'STUB'
#!/usr/bin/env bash
# Succeed for readline only when the caller put the keg-only prefix on
# PKG_CONFIG_PATH — the same condition module_builder.c already uses.
if [ "${1:-}" != "--exists" ] || [ "${2:-}" != "readline" ]; then
  exit 1
fi
case ":${PKG_CONFIG_PATH:-}:" in
  *:/opt/homebrew/opt/readline/lib/pkgconfig:*|*:/usr/local/opt/readline/lib/pkgconfig:*)
    exit 0
    ;;
esac
exit 1
STUB
chmod +x "$STUB_DIR/pkg-config"

PATH="$STUB_DIR:$PATH" python3 - <<'PY'
import os
import subprocess
import sys

def pkg_config_exists(pkg):
    env = os.environ.copy()
    extra = [
        os.path.join('/opt/homebrew/opt', pkg, 'lib', 'pkgconfig'),
        os.path.join('/usr/local/opt', pkg, 'lib', 'pkgconfig'),
    ]
    previous = env.get('PKG_CONFIG_PATH', '')
    env['PKG_CONFIG_PATH'] = os.pathsep.join(
        extra + ([previous] if previous else [])
    )
    result = subprocess.run(
        ['pkg-config', '--exists', pkg],
        env=env,
        capture_output=True,
        timeout=2,
    )
    return result.returncode == 0

bare = subprocess.run(
    ['pkg-config', '--exists', 'readline'],
    capture_output=True,
).returncode == 0
with_keg = pkg_config_exists('readline')
if bare:
    sys.exit(2)
if not with_keg:
    sys.exit(3)
sys.exit(0)
PY
probe_rc=$?
if [ "$probe_rc" -eq 0 ]; then
    pass "keg-only PKG_CONFIG_PATH makes pkg-config --exists readline succeed"
elif [ "$probe_rc" -eq 2 ]; then
    fail "stub pkg-config succeeded without a keg-only prefix"
elif [ "$probe_rc" -eq 3 ]; then
    fail "keg-only PKG_CONFIG_PATH did not reach pkg-config"
else
    fail "keg-only pkg-config probe crashed (exit $probe_rc)"
fi

# --- 4. Darwin: the live validator must not invent an apt hint -------------

if [ "$(uname -s)" = Darwin ]; then
    out="$(mktemp)"
    set +e
    "$VALIDATOR" >"$out" 2>&1
    set -e
    if grep -q "sudo apt-get install" "$out"; then
        fail "validator printed an apt-get hint on Darwin"
        grep "apt-get" "$out"
    else
        pass "validator does not print apt-get hints on Darwin"
    fi
    if grep -Eq "MISSING:readline:|missing: readline|✗.*readline" "$out"; then
        fail "validator still reports readline missing on Darwin"
        grep -E "readline" "$out"
    else
        pass "validator finds keg-only readline on Darwin"
    fi
    # CI Darwin images omit optional formulae (bullet, libevent, …). Exit 1
    # is then expected from `make modules`; it is not a readline regression.
    rm -f "$out"
else
    pass "skip Darwin live-validator checks on $(uname -s)"
fi

echo "=========================================="
if [ "$FAILURES" -eq 0 ]; then
    echo -e "${GREEN}All module validator tests passed${NC}"
    exit 0
fi
echo -e "${RED}${FAILURES} module validator test(s) failed${NC}"
exit 1
