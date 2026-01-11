#!/bin/bash
# Conflict resolution script for nanolang git reconciliation

set -e

echo "=== Resolving remaining conflicts ==="

# For generated file, use remote version (will regenerate later)
echo "Resolving src_nano/generated/compiler_ast.nano (generated file)..."
git checkout --theirs src_nano/generated/compiler_ast.nano
git add src_nano/generated/compiler_ast.nano

# Resolve examples/Makefile by manually editing out remaining markers
echo "Checking examples/Makefile for remaining conflicts..."
if grep -q "<<<<<<< Updated upstream" examples/Makefile; then
    echo "ERROR: examples/Makefile still has conflicts. Please resolve manually."
    exit 1
fi

echo "All automatic resolutions complete."
echo "Remaining manual conflicts:"
git status --short | grep "^UU" || echo "None!"
