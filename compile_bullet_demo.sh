#!/bin/bash
# Compile Bullet demo with FFI linkage

set -e

echo "=== Compiling Bullet Soft Body Demo ==="

# Get Bullet flags
BULLET_FLAGS=$(pkg-config --cflags --libs bullet)

echo "✓ Bullet flags: $BULLET_FLAGS"

# Compile NanoLang to C (it will fail at link stage, but that's OK)
echo "Transpiling NanoLang to C..."
perl -e 'alarm 60; exec @ARGV' ./bin/nanoc examples/bullet_beads_simple.nano -o bin/bullet_test 2>&1 | grep -v "^Warning" > /tmp/nanoc.log || true

# Find the generated C file
C_FILE=$(ls -t /var/folders/*/T/tmp.*/bullet_test.c 2>/dev/null | head -1)

if [ -z "$C_FILE" ]; then
    echo "Error: Could not find generated C file"
    cat /tmp/nanoc.log
    exit 1
fi

echo "✓ Found C file: $C_FILE"

# Compile with Bullet FFI
echo "Compiling C code with Bullet FFI..."
clang++ -O2 \
    -I./src \
    -I./src/runtime \
    "$C_FILE" \
    modules/bullet/bullet_ffi.c \
    src/runtime/*.c \
    $BULLET_FLAGS \
    -o bin/bullet_test

if [ $? -eq 0 ]; then
    echo "✓ Compilation successful!"
    echo ""
    echo "Run with: ./bin/bullet_test"
else
    echo "Error: C compilation failed"
    exit 1
fi

