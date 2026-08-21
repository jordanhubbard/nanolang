#!/usr/bin/env bash
# ============================================================================
# CI dependency-install helpers — retry and call-site coverage
# ============================================================================
#
# Every observed CI failure on this repository's recent main history came from
# a dependency-install step, not from the compiler: an apt or Homebrew mirror
# stalls, the step exits non-zero, and the whole matrix leg goes red even
# though nothing in the tree changed. scripts/ci-apt-install.sh and
# scripts/ci-brew-install.sh are the shared boundary that turns those stalls
# into retried, time-bounded attempts. This test covers that boundary:
#
#   1. retry behaviour — a package manager that fails and then succeeds within
#      the attempt budget yields a successful install
#   2. budget exhaustion — a package manager that always fails exits non-zero
#      instead of hanging or reporting success
#   3. brew skip path — formulae already present on the runner image are not
#      reinstalled, because `brew install` on a preinstalled formula fails on
#      link conflicts
#   4. failure diagnostics — the retry log reports the package manager's real
#      exit code, which is the only breadcrumb left once the run's logs age
#      out of GitHub's retention window
#   5. usage errors — no arguments is a usage error, not a silent no-op
#   6. call-site guard — no workflow installs dependencies with a bare
#      `apt-get install` or `brew install`; they all go through the helpers
#   7. no-timeout hosts — `make test` runs this file on every platform, and
#      macOS has no `timeout` (GNU coreutils installs it as `gtimeout`), so
#      neither helper may treat a missing timeout binary as fatal
#
# Both package managers are stubbed on PATH, so this runs anywhere bash does:
# no apt, no Homebrew, no network, no sudo.
#
# Usage:
#   bash tests/test_ci_dependency_install.sh
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

APT_HELPER="$PROJECT_ROOT/scripts/ci-apt-install.sh"
BREW_HELPER="$PROJECT_ROOT/scripts/ci-brew-install.sh"

STUB_DIR="$(mktemp -d)"
trap 'rm -rf "$STUB_DIR"' EXIT

# Attempt counters live on disk so the stubs stay stateless across the
# separate processes the helpers spawn per attempt.
ATTEMPTS_FILE="$STUB_DIR/attempts"

# --- stubs ------------------------------------------------------------------
# STUB_FAIL_UNTIL: number of leading attempts that fail before one succeeds.
# Set it above the attempt budget to model a package manager that never works.

cat >"$STUB_DIR/apt-get" <<'STUB'
#!/usr/bin/env bash
# `update` is the quiet half of the helper's attempt; only `install` counts.
[ "${1:-}" = "update" ] && exit 0
for arg in "$@"; do
  if [ "$arg" = "install" ]; then
    n=$(( $(cat "$ATTEMPTS_FILE") + 1 ))
    echo "$n" >"$ATTEMPTS_FILE"
    if [ "$n" -le "${STUB_FAIL_UNTIL:-0}" ]; then
      echo "stub apt-get: simulated mirror failure (attempt $n)" >&2
      exit 100
    fi
    exit 0
  fi
done
exit 0
STUB

cat >"$STUB_DIR/sudo" <<'STUB'
#!/usr/bin/env bash
# Drop sudo's own flags and run the command directly; CI stubs need no root.
while [ "$#" -gt 0 ]; do
  case "$1" in
    -E|-n|-H) shift ;;
    *) break ;;
  esac
done
exec "$@"
STUB

cat >"$STUB_DIR/killall" <<'STUB'
#!/usr/bin/env bash
exit 0
STUB

cat >"$STUB_DIR/brew" <<'STUB'
#!/usr/bin/env bash
case "${1:-}" in
  list)
    # Last argument is the formula being probed.
    formula="${@: -1}"
    case " ${STUB_PREINSTALLED:-} " in
      *" $formula "*) echo "$formula 1.0.0"; exit 0 ;;
      *) exit 1 ;;
    esac
    ;;
  install)
    n=$(( $(cat "$ATTEMPTS_FILE") + 1 ))
    echo "$n" >"$ATTEMPTS_FILE"
    if [ "$n" -le "${STUB_FAIL_UNTIL:-0}" ]; then
      echo "stub brew: simulated download failure (attempt $n)" >&2
      exit 1
    fi
    exit 0
    ;;
esac
exit 0
STUB

chmod +x "$STUB_DIR"/apt-get "$STUB_DIR"/sudo "$STUB_DIR"/killall "$STUB_DIR"/brew

# A PATH that models a macOS runner: the stubs plus only the utilities the
# helpers actually reach for, and deliberately no `timeout`/`gtimeout`. It has
# to be an allow-list rather than a filtered copy of the real PATH, because the
# point is to guarantee the timeout binaries are absent.
NO_TIMEOUT_DIR="$STUB_DIR/no-timeout"
mkdir -p "$NO_TIMEOUT_DIR"
for stub in apt-get sudo killall brew; do
    ln -s "$STUB_DIR/$stub" "$NO_TIMEOUT_DIR/$stub"
done
for tool in bash sleep cat; do
    ln -s "$(command -v "$tool")" "$NO_TIMEOUT_DIR/$tool"
done

# Run a helper against the stubs. Backoff is collapsed to zero so the retry
# paths cost no wall-clock; the helpers read it from the environment.
run_helper() {
    local helper="$1"; shift
    echo 0 >"$ATTEMPTS_FILE"
    PATH="${HELPER_PATH:-$STUB_DIR:$PATH}" \
    ATTEMPTS_FILE="$ATTEMPTS_FILE" \
    CI_APT_ATTEMPTS=3 CI_APT_BACKOFF_SECS=0 CI_APT_TIMEOUT_SECS=30 \
    CI_BREW_ATTEMPTS=3 CI_BREW_BACKOFF_SECS=0 CI_BREW_TIMEOUT_SECS=30 \
    STUB_FAIL_UNTIL="${STUB_FAIL_UNTIL:-0}" \
    STUB_PREINSTALLED="${STUB_PREINSTALLED:-}" \
        bash "$helper" "$@" >"$STUB_DIR/out" 2>&1
}

attempts_made() { cat "$ATTEMPTS_FILE"; }

echo "=========================================="
echo "CI dependency-install helper tests"
echo "=========================================="

# --- 1. retry behaviour -----------------------------------------------------

STUB_FAIL_UNTIL=2 run_helper "$APT_HELPER" build-essential
if [ "$?" -eq 0 ] && [ "$(attempts_made)" -eq 3 ]; then
    pass "apt helper retries a failing mirror and succeeds within budget"
else
    fail "apt helper did not recover from a transient failure (attempts=$(attempts_made))"
    cat "$STUB_DIR/out"
fi

STUB_FAIL_UNTIL=2 run_helper "$BREW_HELPER" sdl2
if [ "$?" -eq 0 ] && [ "$(attempts_made)" -eq 3 ]; then
    pass "brew helper retries a failing download and succeeds within budget"
else
    fail "brew helper did not recover from a transient failure (attempts=$(attempts_made))"
    cat "$STUB_DIR/out"
fi

# --- 2. budget exhaustion ---------------------------------------------------

STUB_FAIL_UNTIL=99 run_helper "$APT_HELPER" build-essential
if [ "$?" -ne 0 ] && [ "$(attempts_made)" -eq 3 ]; then
    pass "apt helper fails after exhausting its attempt budget"
else
    fail "apt helper masked a permanent failure (attempts=$(attempts_made))"
    cat "$STUB_DIR/out"
fi

STUB_FAIL_UNTIL=99 run_helper "$BREW_HELPER" sdl2
if [ "$?" -ne 0 ] && [ "$(attempts_made)" -eq 3 ]; then
    pass "brew helper fails after exhausting its attempt budget"
else
    fail "brew helper masked a permanent failure (attempts=$(attempts_made))"
    cat "$STUB_DIR/out"
fi

# --- 3. brew skip path ------------------------------------------------------

STUB_PREINSTALLED="openssl" STUB_FAIL_UNTIL=99 run_helper "$BREW_HELPER" openssl
if [ "$?" -eq 0 ] && [ "$(attempts_made)" -eq 0 ]; then
    pass "brew helper skips a formula the runner image already provides"
else
    fail "brew helper reinstalled a preinstalled formula (attempts=$(attempts_made))"
    cat "$STUB_DIR/out"
fi

# A preinstalled formula must not stop the rest of the list from installing.
STUB_PREINSTALLED="openssl" STUB_FAIL_UNTIL=0 run_helper "$BREW_HELPER" openssl sdl2
if [ "$?" -eq 0 ] && [ "$(attempts_made)" -eq 1 ]; then
    pass "brew helper installs the remaining formulae after a skip"
else
    fail "brew helper mishandled a mixed skip/install list (attempts=$(attempts_made))"
    cat "$STUB_DIR/out"
fi

# --- 4. failure diagnostics -------------------------------------------------
# The stubs exit with a distinctive status; the helper must surface that status
# rather than the 0 that `rc=$?` after `fi` would report.

STUB_FAIL_UNTIL=99 run_helper "$APT_HELPER" build-essential
if grep -q "apt attempt 1 failed (exit 100)" "$STUB_DIR/out"; then
    pass "apt helper reports the package manager's real exit code"
else
    fail "apt helper lost the failing exit code"
    cat "$STUB_DIR/out"
fi

STUB_FAIL_UNTIL=99 run_helper "$BREW_HELPER" sdl2
if grep -q "brew attempt 1 failed for sdl2 (exit 1)" "$STUB_DIR/out"; then
    pass "brew helper reports the package manager's real exit code"
else
    fail "brew helper lost the failing exit code"
    cat "$STUB_DIR/out"
fi

# --- 5. usage errors --------------------------------------------------------

for helper in "$APT_HELPER" "$BREW_HELPER"; do
    run_helper "$helper"
    if [ "$?" -eq 2 ]; then
        pass "$(basename "$helper") reports a usage error when given no packages"
    else
        fail "$(basename "$helper") accepted an empty package list"
    fi
done

# --- 6. call-site guard -----------------------------------------------------
# Workflow steps must route installs through the helpers. Helper invocations
# themselves mention neither `apt-get install` nor `brew install`, so any hit
# here is a raw call site that skipped the retry boundary.

BARE_CALLS="$(grep -rn -E '(sudo +)?apt-get +(-[^ ]+ +)*install|(^|[^-])\bbrew +install' \
    .github/workflows/*.yml 2>/dev/null || true)"
if [ -z "$BARE_CALLS" ]; then
    pass "no workflow installs dependencies outside the retry helpers"
else
    fail "workflow steps bypass the retry helpers:"
    echo "$BARE_CALLS"
fi

# --- 7. no-timeout hosts ----------------------------------------------------
# `make test` drives this file on macOS too, where `timeout` does not exist
# under that name. A helper that refuses to run without it reports a red test
# leg for a platform difference rather than for a packaging problem, so both
# helpers must degrade to an unbounded attempt and keep their retry contract.

HELPER_PATH="$NO_TIMEOUT_DIR" STUB_FAIL_UNTIL=2 run_helper "$APT_HELPER" build-essential
if [ "$?" -eq 0 ] && [ "$(attempts_made)" -eq 3 ]; then
    pass "apt helper still retries and installs on a host without GNU timeout"
else
    fail "apt helper is unusable without GNU timeout (attempts=$(attempts_made))"
    cat "$STUB_DIR/out"
fi

HELPER_PATH="$NO_TIMEOUT_DIR" STUB_FAIL_UNTIL=99 run_helper "$APT_HELPER" build-essential
if [ "$?" -ne 0 ] && [ "$(attempts_made)" -eq 3 ]; then
    pass "apt helper still exhausts its budget on a host without GNU timeout"
else
    fail "apt helper lost its retry budget without GNU timeout (attempts=$(attempts_made))"
    cat "$STUB_DIR/out"
fi

HELPER_PATH="$NO_TIMEOUT_DIR" STUB_FAIL_UNTIL=2 run_helper "$BREW_HELPER" sdl2
if [ "$?" -eq 0 ] && [ "$(attempts_made)" -eq 3 ]; then
    pass "brew helper still retries and installs on a host without GNU timeout"
else
    fail "brew helper is unusable without GNU timeout (attempts=$(attempts_made))"
    cat "$STUB_DIR/out"
fi

echo "=========================================="
if [ "$FAILURES" -eq 0 ]; then
    echo -e "${GREEN}All CI dependency-install helper tests passed${NC}"
    exit 0
fi
echo -e "${RED}${FAILURES} CI dependency-install helper test(s) failed${NC}"
exit 1
