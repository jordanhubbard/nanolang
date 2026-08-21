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

if ! command -v timeout >/dev/null 2>&1; then
  echo "error: GNU timeout is required" >&2
  exit 2
fi

export DEBIAN_FRONTEND=noninteractive

attempt=1
while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
  echo "apt install attempt ${attempt}/${MAX_ATTEMPTS} (timeout ${ATTEMPT_TIMEOUT}s): $*"
  # Capture the status inside the condition: `rc=$?` after `fi` reads the
  # status of the `if` itself, which is 0 on a false condition, so the failure
  # diagnostic would report "exit 0" for every failed attempt.
  rc=0
  timeout --foreground --signal=KILL "${ATTEMPT_TIMEOUT}" \
      bash -c '
        set -euo pipefail
        export DEBIAN_FRONTEND=noninteractive
        sudo -E apt-get -o Acquire::Retries=3 -o Dpkg::Use-Pty=0 update -qq
        sudo -E apt-get -o Acquire::Retries=3 -o Dpkg::Use-Pty=0 install -y "$@"
      ' ci-apt "$@" || rc=$?
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
