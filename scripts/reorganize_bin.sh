#!/bin/bash
# ============================================================================
# Reorganize bin/ directory for GCC-style bootstrap
# ============================================================================
#
# This script moves artifacts from the old bin/ structure to the new one:
#   - Bootstrap/stage artifacts → build/
#   - Test binaries → build/tests/
#   - Example binaries → examples/bin/
#   - Keep only final tools in bin/
#
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║          Reorganizing bin/ for Bootstrap Structure             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Create directory structure
echo "Creating directory structure..."
mkdir -p build/bootstrap
mkdir -p build/stage1
mkdir -p build/stage2
mkdir -p build/tests
mkdir -p examples/bin

# ============================================================================
# Move Bootstrap/Stage Artifacts
# ============================================================================

echo ""
echo "Moving bootstrap/stage artifacts to build/..."

# nanoc_sh is the self-hosted compiler - this goes to stage1
if [ -f bin/nanoc_sh ]; then
    echo "  • nanoc_sh → build/stage1/nanoc"
    mv bin/nanoc_sh build/stage1/nanoc
fi

# Old stage artifacts
if [ -f bin/nanoc_stage0 ]; then
    echo "  • nanoc_stage0 → build/bootstrap/nanoc"
    mv bin/nanoc_stage0 build/bootstrap/nanoc
fi

if [ -f bin/nanoc_stage1 ]; then
    echo "  • nanoc_stage1 → build/stage1/nanoc.old"
    mv bin/nanoc_stage1 build/stage1/nanoc.old
fi

if [ -f bin/nano_stage1 ]; then
    echo "  • nano_stage1 → build/stage1/nano.old"
    mv bin/nano_stage1 build/stage1/nano.old
fi

# ============================================================================
# Move Test Binaries
# ============================================================================

echo ""
echo "Moving test binaries to build/tests/..."

for test_bin in bin/selfhost_test_* bin/test_* bin/demo_*; do
    if [ -f "$test_bin" ]; then
        basename_file=$(basename "$test_bin")
        echo "  • $basename_file"
        mv "$test_bin" "build/tests/"
    fi
done

# ============================================================================
# Move Example Binaries
# ============================================================================

echo ""
echo "Moving example binaries to examples/bin/..."

# List of example binaries (SDL, OpenGL, etc.)
EXAMPLES=(
    "boids_sdl"
    "checkers_sdl"
    "falling_sand_sdl"
    "particles_sdl"
    "terrain_explorer_sdl"
    "terrain_explorer_test"
    "opengl_cube"
    "opengl_teapot"
    "opengl_teapot_glut"
    "raytracer_simple"
)

for example in "${EXAMPLES[@]}"; do
    if [ -f "bin/$example" ]; then
        echo "  • $example"
        mv "bin/$example" "examples/bin/"
    fi
done

# ============================================================================
# Move Component Binaries
# ============================================================================

echo ""
echo "Moving component binaries to build/..."

COMPONENTS=(
    "parser"
    "typecheck"
    "transpiler"
    "type_adapters"
)

for comp in "${COMPONENTS[@]}"; do
    if [ -f "bin/$comp" ]; then
        echo "  • $comp → build/"
        mv "bin/$comp" "build/"
    fi
done

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    Reorganization Complete                      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Final bin/ directory should contain:"
echo "  • nano        - C-compiled interpreter ✓"
echo "  • nanoc       - C-compiled compiler (will be replaced by bootstrap)"
echo "  • nanoc-ffi   - FFI binding generator ✓"
echo ""
echo "Run 'make clean && make' to perform full bootstrap."
echo "After bootstrap, bin/nanoc will be the SELF-HOSTED version!"
echo ""

# Show what's left in bin/
echo "Current bin/ contents:"
ls -lh bin/ | grep -v ^d | awk '{printf "  %-20s %8s\n", $9, $5}' | grep -v "^  $"

echo ""
echo "Build artifacts moved to:"
echo "  build/bootstrap/   - Bootstrap compiler"
echo "  build/stage1/      - Stage 1 artifacts"
echo "  build/tests/       - Test binaries"
echo "  examples/bin/      - Example binaries"
echo ""
echo "✅ Ready for bootstrap!"
