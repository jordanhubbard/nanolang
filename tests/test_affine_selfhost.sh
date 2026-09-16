#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
exec python3 -m unittest tests.test_affine_frontend_parity tests.test_affine_contract_boundaries
