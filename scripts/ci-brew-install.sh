#!/usr/bin/env bash
# Resilient brew install for GitHub Actions.
#
# The macOS leg of the build matrix installed its dependencies with a bare
# `brew install`, while every Linux leg already went through
# scripts/ci-apt-install.sh. That left the macOS runners as the only
# dependency-install step with no per-attempt timeout and no retry, so a
# Homebrew mirror stall or a transient download failure took the whole job
# down instead of being retried. This helper closes that gap with the same
# contract as the apt helper.
#
# Formulae that the runner image already ships are skipped rather than
# reinstalled: `brew install` on a preinstalled-but-unlinked formula (gcc,
# openssl, ...) fails on link conflicts, which is noise rather than signal.
#
# Usage:
#   ./scripts/ci-brew-install.sh <formula> [formula...]
#
# Environment:
#   CI_BREW_ATTEMPTS      Max attempts per formula (default: 3)
#   CI_BREW_TIMEOUT_SECS  Per-attempt timeout in seconds (default: 600)
#   CI_BREW_BACKOFF_SECS  Backoff base in seconds, scaled by attempt (default: 5)

set -euo pipefail

MAX_ATTEMPTS="${CI_BREW_ATTEMPTS:-3}"
ATTEMPT_TIMEOUT="${CI_BREW_TIMEOUT_SECS:-600}"
BACKOFF_SECS="${CI_BREW_BACKOFF_SECS:-5}"

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <formula> [formula...]" >&2
  exit 2
fi

if ! command -v brew >/dev/null 2>&1; then
  echo "error: brew is required" >&2
  exit 2
fi

# Auto-update and post-install cleanup are the two long poles that turn a
# short install into a job-length stall on a slow mirror; neither buys CI
# anything, because the runner image pins its own Homebrew snapshot.
export HOMEBREW_NO_AUTO_UPDATE=1
export HOMEBREW_NO_INSTALL_CLEANUP=1
export HOMEBREW_NO_ENV_HINTS=1

# macOS ships no `timeout`; it arrives as `gtimeout` with coreutils. Bound each
# attempt when either is present and fall back to the job-level timeout when
# neither is, rather than refusing to install at all.
TIMEOUT_BIN=""
for candidate in timeout gtimeout; do
  if command -v "$candidate" >/dev/null 2>&1; then
    TIMEOUT_BIN="$candidate"
    break
  fi
done
if [ -z "$TIMEOUT_BIN" ]; then
  echo "note: no timeout binary found; relying on the job timeout" >&2
fi

install_once() {
  if [ -n "$TIMEOUT_BIN" ]; then
    "$TIMEOUT_BIN" --foreground --signal=KILL "$ATTEMPT_TIMEOUT" \
      brew install --quiet "$1"
  else
    brew install --quiet "$1"
  fi
}

for formula in "$@"; do
  if brew list --formula --versions "$formula" >/dev/null 2>&1; then
    echo "brew: ${formula} already installed; skipping"
    continue
  fi

  attempt=1
  installed=false
  while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
    echo "brew install attempt ${attempt}/${MAX_ATTEMPTS} (timeout ${ATTEMPT_TIMEOUT}s): ${formula}"
    # Capture the status inside the condition: `rc=$?` after `fi` reads the
    # status of the `if` itself, which is 0 on a false condition, so the
    # failure diagnostic would report "exit 0" for every failed attempt.
    rc=0
    install_once "$formula" || rc=$?
    if [ "$rc" -eq 0 ]; then
      echo "brew install succeeded on attempt ${attempt}: ${formula}"
      installed=true
      break
    fi
    echo "brew attempt ${attempt} failed for ${formula} (exit ${rc})"

    if [ "$attempt" -eq "$MAX_ATTEMPTS" ]; then
      break
    fi

    sleep $((attempt * BACKOFF_SECS))
    attempt=$((attempt + 1))
  done

  if [ "$installed" != true ]; then
    echo "brew install failed after ${MAX_ATTEMPTS} attempts: ${formula}" >&2
    exit 1
  fi
done
