#!/usr/bin/env bash
# ============================================================================
# Public validator contract: keg-only readline, host-correct install hints
# ============================================================================
#
# Public interface: `scripts/validate-modules.sh` (what `make modules`
# runs). Two user-visible lies this pins:
#
#   1. An installed Homebrew keg-only readline is reported present.
#   2. On Darwin, missing-package remediation is not `sudo apt-get install`.
#
# The keg-only check drives the real script with a stub pkg-config on PATH
# that only accepts `readline` when the keg prefix is on PKG_CONFIG_PATH.
# Darwin hint check runs the script against this checkout as a user would.
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

STUB_DIR="$(mktemp -d)"
trap 'rm -rf "$STUB_DIR"' EXIT

cat >"$STUB_DIR/pkg-config" <<'STUB'
#!/usr/bin/env bash
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

out="$(mktemp)"
set +e
PATH="$STUB_DIR:$PATH" "$VALIDATOR" >"$out" 2>&1
set -e

if grep -aE "✓.*readline" "$out" >/dev/null; then
    pass "validator reports readline present when only a keg-only pkg-config path would find it"
else
    fail "validator did not report readline present under a keg-only-only pkg-config"
    grep -E "readline" "$out" || true
fi

if grep -Eq "missing:.*readline|MISSING:readline" "$out"; then
    fail "validator still reports readline missing when the keg-only path is the only hit"
    grep -E "readline" "$out" || true
fi

if [ "$(uname -s)" = Darwin ]; then
    live="$(mktemp)"
    set +e
    "$VALIDATOR" >"$live" 2>&1
    set -e
    if grep -q "sudo apt-get install" "$live"; then
        fail "validator printed an apt-get hint on Darwin"
        grep "apt-get" "$live"
    else
        pass "validator does not print apt-get hints on Darwin"
    fi
    rm -f "$live"
fi

rm -f "$out"

echo "=========================================="
if [ "$FAILURES" -eq 0 ]; then
    echo -e "${GREEN}All module validator tests passed${NC}"
    exit 0
fi
echo -e "${RED}${FAILURES} module validator test(s) failed${NC}"
exit 1
