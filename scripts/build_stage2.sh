#!/bin/bash
# Build Stage 2: Use Stage 1 to compile itself

set -e

# Configuration
PROJECT_ROOT="/Users/jordanh/Src/nanolang"
BIN_DIR="$PROJECT_ROOT/bin"
SRC_NANO="$PROJECT_ROOT/src_nano"
BUILD_DIR="$PROJECT_ROOT/build_bootstrap"

echo "==================================================================="
echo "  Building Stage 2 Self-Hosted Compiler"
echo "==================================================================="
echo ""
echo "Stage 1:      $BIN_DIR/nanoc_stage1 (nanolang compiler)"
echo "Stage 2:      $BIN_DIR/nanoc_stage2 (Stage 1 compiling itself)"
echo ""

# Check Stage 1 exists
if [ ! -f "$BIN_DIR/nanoc_stage1" ]; then
    echo "✗ Stage 1 compiler not found. Run build_stage1.sh first."
    exit 1
fi

echo "✓ Stage 1 compiler found"
echo ""

# Step 1: Use Stage 1 to generate C code for itself
echo "Step 1: Stage 1 compiling itself to C..."
echo "-------------------------------------------------------------------"

cd "$PROJECT_ROOT"

# Use Stage 1 to transpile itself (will fail at linking, but generates .genC)
echo "Transpiling stage1_compiler.nano with Stage 1..."
NANO_MODULE_PATH=modules "$BIN_DIR/nanoc_stage1" \
    "$SRC_NANO/stage1_compiler.nano" \
    --keep-c \
    -o "$BUILD_DIR/nanoc_stage2_temp" 2>&1 | grep -v "Undefined symbols\|ld:\|clang:" || true

# Check if .genC file was created
if [ -f "$SRC_NANO/stage1_compiler.nano.genC" ]; then
    echo "✓ Stage 1 successfully generated C code for itself"
else
    echo "✗ Failed to generate C code"
    exit 1
fi

echo ""

# Step 2: Extract and patch generated C code
echo "Step 2: Extracting and patching C code..."
echo "-------------------------------------------------------------------"

cp "$SRC_NANO/stage1_compiler.nano.genC" "$BUILD_DIR/stage2_compiler.c"

# Fix main() to accept argc/argv and initialize CLI args
sed -i.bak 's/int main() {/extern void nl_cli_args_init(int argc, char **argv);\n\nint main(int argc, char **argv) {\n    nl_cli_args_init(argc, argv);/' "$BUILD_DIR/stage2_compiler.c"
rm "$BUILD_DIR/stage2_compiler.c.bak"

echo "✓ C code extracted and patched"
echo ""

# Step 3: Compile Stage 2 with all dependencies
echo "Step 3: Compiling Stage 2 with C implementation..."
echo "-------------------------------------------------------------------"

gcc -std=c99 -g \
    "$BUILD_DIR/stage2_compiler.c" \
    "$SRC_NANO/file_io.c" \
    "$SRC_NANO/cli_args.c" \
    "$SRC_NANO/compiler_extern.c" \
    src/lexer.c \
    src/parser.c \
    src/typechecker.c \
    src/transpiler.c \
    src/eval.c \
    src/env.c \
    src/module.c \
    src/module_metadata.c \
    src/module_builder.c \
    src/cJSON.c \
    src/runtime/list_int.c \
    src/runtime/list_string.c \
    src/runtime/list_token.c \
    src/runtime/gc.c \
    src/runtime/dyn_array.c \
    src/runtime/gc_struct.c \
    src/runtime/nl_string.c \
    -Isrc \
    -o "$BIN_DIR/nanoc_stage2" \
    -lm

if [ $? -eq 0 ]; then
    echo "✓ Stage 2 compilation successful"
else
    echo "✗ Stage 2 compilation failed"
    exit 1
fi

echo ""

# Step 4: Test Stage 2
echo "Step 4: Testing Stage 2 compiler..."
echo "-------------------------------------------------------------------"

# Test with hello program
echo "Testing with test_hello.nano..."
"$BIN_DIR/nanoc_stage2" "$BUILD_DIR/test_hello.nano" -o "$BUILD_DIR/test_stage2"

if [ $? -eq 0 ]; then
    echo "✓ Stage 2 compiler can compile test program"
else
    echo "✗ Stage 2 compilation test failed"
    exit 1
fi

# Run the test program
echo "Running test program..."
"$BUILD_DIR/test_stage2"

if [ $? -eq 0 ]; then
    echo "✓ Test program runs successfully"
else
    echo "✗ Test program failed"
    exit 1
fi

echo ""
echo "==================================================================="
echo "  Stage 2 Compiler Build Complete!"
echo "==================================================================="
echo ""
echo "Executable: $BIN_DIR/nanoc_stage2"
echo ""
echo "Stage 2 was built by Stage 1 compiling itself!"
echo ""
echo "Next step:"
echo "  Verify Stage 1 and Stage 2 produce identical output"
echo ""
