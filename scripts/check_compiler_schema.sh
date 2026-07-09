#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "[schema] Regenerating compiler schema artifacts..."
python3 scripts/gen_compiler_schema.py

if ! git diff --quiet -- src/generated/compiler_schema.h src_nano/generated/compiler_schema.nano src_nano/generated/compiler_ast.nano; then
  echo "[schema] ERROR: Generated compiler schema artifacts are out of date." >&2
  echo "[schema] Please run scripts/gen_compiler_schema.py and commit the changes." >&2
  git status --short src/generated/compiler_schema.h src_nano/generated/compiler_schema.nano src_nano/generated/compiler_ast.nano >&2
  exit 1
fi

echo "[schema] Compiler schema artifacts are up-to-date."
