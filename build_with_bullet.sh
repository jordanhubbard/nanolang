#!/bin/bash
# Build NanoLang program with Bullet Physics FFI

set -e

NANO_FILE="$1"
OUTPUT="$2"

if [ -z "$NANO_FILE" ] || [ -z "$OUTPUT" ]; then
    echo "Usage: $0 <input.nano> <output-binary>"
    exit 1
fi

echo "=== Building $NANO_FILE with Bullet Physics ==="

# Transpile to C (will fail at link, but generates .genC file)
echo "Step 1: Transpiling to C..."
perl -e 'alarm 60; exec @ARGV' ./bin/nanoc "$NANO_FILE" -S -o /tmp/dummy 2>&1 | grep -v "^Warning" > /tmp/nanoc.log || true

# Find generated C file
GEN_C="${NANO_FILE}.genC"
if [ ! -f "$GEN_C" ]; then
    echo "Error: Generated C file not found: $GEN_C"
    cat /tmp/nanoc.log
    exit 1
fi

echo "✓ Generated C code: $GEN_C"

# Compile runtime C files
echo "Step 2: Compiling runtime..."
SDL_FLAGS=$(pkg-config --cflags sdl2 2>/dev/null || echo "")
clang -c -O2 -I./src -I./src/runtime $SDL_FLAGS \
    src/runtime/dyn_array.c \
    src/runtime/gc.c \
    src/runtime/gc_struct.c \
    src/runtime/list_int.c \
    src/runtime/list_string.c \
    src/runtime/nl_string.c \
    2>/dev/null

echo "✓ Runtime compiled"

# Compile generated C file  
echo "Step 3: Compiling generated code..."
SDL_FLAGS=$(pkg-config --cflags sdl2 2>/dev/null || echo "")
clang -x c -c -O2 -I./src -I./src/runtime $SDL_FLAGS "$GEN_C" -o /tmp/nano_main.o

if [ ! -f /tmp/nano_main.o ]; then
    echo "Error: Failed to compile main program"
    exit 1
fi

echo "✓ Main program compiled"

# Link everything
echo "Step 4: Linking with Bullet FFI..."
SDL_LIBS=$(pkg-config --libs sdl2 2>/dev/null || echo "-lSDL2")
clang++ -O2 \
    /tmp/nano_main.o \
    modules/bullet/bullet_ffi.o \
    dyn_array.o gc.o gc_struct.o list_int.o list_string.o nl_string.o \
    -L/opt/homebrew/Cellar/bullet/3.25/lib \
    -lBulletSoftBody -lBulletDynamics -lBulletCollision -lLinearMath \
    $SDL_LIBS \
    -o "$OUTPUT"

echo "✓ Build successful: $OUTPUT"
echo ""
rm -f *.o /tmp/nano_main.o

