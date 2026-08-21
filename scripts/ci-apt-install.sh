#!/usr/bin/env bash
# Resilient apt-get install for GitHub Actions.
#
# Some Ubuntu runners occasionally hang indefinitely inside apt-get update/install
# (mirror stalls, dpkg locks). Without a per-attempt timeout those hangs consume the
# entire job budget and surface as a cancelled workflow rather than a retryable
# install failure. This helper bounds each attempt and retries a few times.
#
# Usage:
#   ./scripts/ci-apt-install.sh <package> [package...]
#
# Environment:
#   CI_APT_ATTEMPTS      Max attempts (default: 3)
#   CI_APT_TIMEOUT_SECS  Per-attempt timeout in seconds (default: 180)
#   CI_APT_BACKOFF_SECS  Backoff base in seconds, scaled by attempt (default: 5)

set -euo pipefail

MAX_ATTEMPTS="${CI_APT_ATTEMPTS:-3}"
ATTEMPT_TIMEOUT="${CI_APT_TIMEOUT_SECS:-180}"
BACKOFF_SECS="${CI_APT_BACKOFF_SECS:-5}"

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <package> [package...]" >&2
  exit 2
fi

# `timeout` is GNU coreutils, so it is absent on macOS; coreutils installs it
# there as `gtimeout`, never under the plain name. The per-attempt bound is
# hardening, not a precondition, so a missing timeout binary falls back to the
# job-level timeout the same way scripts/ci-brew-install.sh does. Treating it
# as fatal made this helper unrunnable on macOS, which turned the macOS test
# leg red once `make test` started exercising both helpers on every platform.
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

export DEBIAN_FRONTEND=noninteractive

APT_ATTEMPT='
  set -euo pipefail
  export DEBIAN_FRONTEND=noninteractive
  sudo -E apt-get -o Acquire::Retries=3 -o Dpkg::Use-Pty=0 update -qq
  sudo -E apt-get -o Acquire::Retries=3 -o Dpkg::Use-Pty=0 install -y "$@"
'

install_once() {
  if [ -n "$TIMEOUT_BIN" ]; then
    "$TIMEOUT_BIN" --foreground --signal=KILL "$ATTEMPT_TIMEOUT" \
      bash -c "$APT_ATTEMPT" ci-apt "$@"
  else
    bash -c "$APT_ATTEMPT" ci-apt "$@"
  fi
}

attempt=1
while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
  echo "apt install attempt ${attempt}/${MAX_ATTEMPTS} (timeout ${ATTEMPT_TIMEOUT}s): $*"
  # Capture the status inside the condition: `rc=$?` after `fi` reads the
  # status of the `if` itself, which is 0 on a false condition, so the failure
  # diagnostic would report "exit 0" for every failed attempt.
  rc=0
  install_once "$@" || rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "apt install succeeded on attempt ${attempt}"
    exit 0
  fi
  echo "apt attempt ${attempt} failed (exit ${rc})"

  if [ "$attempt" -eq "$MAX_ATTEMPTS" ]; then
    break
  fi

  # Best-effort cleanup before retrying a hung/partial apt state.
  sudo killall -q apt-get apt dpkg 2>/dev/null || true
  sleep $((attempt * BACKOFF_SECS))
  attempt=$((attempt + 1))
done

echo "apt install failed after ${MAX_ATTEMPTS} attempts" >&2
exit 1
