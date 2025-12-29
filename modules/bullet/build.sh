#!/bin/bash
# Build Bullet Physics module for NanoLang

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== Building Bullet Physics Module ==="

# Check for bullet installation
if ! pkg-config --exists bullet; then
    echo "Error: Bullet Physics not found"
    echo "Install with: brew install bullet (macOS) or sudo apt-get install libbullet-dev (Linux)"
    exit 1
fi

echo "✓ Bullet Physics found"

# Get compilation flags
CFLAGS=$(pkg-config --cflags bullet)
LIBS=$(pkg-config --libs bullet)

echo "Compiling bullet_ffi.c..."
clang++ -fPIC -shared -O2 $CFLAGS \
    -I"$PROJECT_ROOT/src" \
    -o "$SCRIPT_DIR/bullet_ffi.so" \
    "$SCRIPT_DIR/bullet_ffi.c" \
    $LIBS

echo "✓ Compilation successful: bullet_ffi.so"
echo ""
echo "Module ready to use!"
echo "Import with: import \"modules/bullet/bullet.nano\" as Bullet"

