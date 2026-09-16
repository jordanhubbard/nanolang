#!/usr/bin/env bash
# I run the daemon gate with private endpoints and owned processes.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$SCRIPT_DIR/../tests/daemon_integration.py" "$@"
