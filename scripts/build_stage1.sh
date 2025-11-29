#!/bin/bash
# =============================================================================
# Build Stage 1 Self-Hosted Compiler
# =============================================================================
# Builds the nanolang Stage 1 compiler using the C compiler (Stage 0)
#
# Stage 0: C compiler (bin/nanoc)
# Stage 1: Self-hosted nanolang compiler (bin/nanoc_stage1)

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_NANO="$PROJECT_ROOT/src_nano"
BIN_DIR="$PROJECT_ROOT/bin"
BUILD_DIR="$PROJECT_ROOT/build_bootstrap"

echo "==================================================================="
echo "  Building Stage 1 Self-Hosted Compiler"
echo "==================================================================="
echo ""
echo "Project root: $PROJECT_ROOT"
echo "Stage 0:      $BIN_DIR/nanoc (C compiler)"
echo "Stage 1:      $BIN_DIR/nanoc_stage1 (nanolang compiler)"
echo ""

# Create directories
mkdir -p "$BUILD_DIR"
mkdir -p "$BIN_DIR"

# Step 1: Ensure C compiler exists
if [ ! -f "$BIN_DIR/nanoc" ]; then
    echo "Error: C compiler not found at $BIN_DIR/nanoc"
    echo "Please run 'make' first to build the C compiler"
    exit 1
fi

echo "✓ Stage 0 (C compiler) found"
echo ""

# Step 2: Compile Stage 1 compiler with Stage 0
echo "Step 1: Compiling Stage 1 compiler..."
echo "-------------------------------------------------------------------"

cd "$PROJECT_ROOT"

# Compile stage1_compiler.nano to C (generate C code only, don't compile)
echo "Transpiling stage1_compiler.nano to C..."
NANO_MODULE_PATH=modules "$BIN_DIR/nanoc" \
    "$SRC_NANO/stage1_compiler.nano" \
    -S 2>&1 | grep -v "Undefined symbols\|ld:\|clang:" || true

# Check if .genC file was created (nanoc -S creates this even if linking fails)
if [ -f "$SRC_NANO/stage1_compiler.nano.genC" ]; then
    echo "✓ Transpilation successful"
else
    echo "✗ Failed to generate C code"
    exit 1
fi
echo ""

# Step 3: Extract generated C code
echo "Step 2: Extracting generated C code..."
echo "-------------------------------------------------------------------"

if [ -f "$SRC_NANO/stage1_compiler.nano.genC" ]; then
    cp "$SRC_NANO/stage1_compiler.nano.genC" "$BUILD_DIR/stage1_compiler.c"
    
    # Fix main() to accept argc/argv and initialize CLI args
    sed -i.bak 's/int main() {/extern void nl_cli_args_init(int argc, char **argv);\n\nint main(int argc, char **argv) {\n    nl_cli_args_init(argc, argv);/' "$BUILD_DIR/stage1_compiler.c"
    rm "$BUILD_DIR/stage1_compiler.c.bak"
    
    echo "✓ C code saved and patched to build_bootstrap/stage1_compiler.c"
else
    echo "✗ Generated C code not found"
    exit 1
fi

echo ""

# Step 4: Compile C code with compiler_extern.c
echo "Step 3: Compiling with C implementation..."
echo "-------------------------------------------------------------------"

# Compile the generated C code along with the extern implementations
gcc -std=c99 -g \
    "$BUILD_DIR/stage1_compiler.c" \
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
    -o "$BIN_DIR/nanoc_stage1" \
    -lm

if [ $? -ne 0 ]; then
    echo "✗ C compilation failed"
    exit 1
fi

echo "✓ Compilation successful"
echo ""

# Step 5: Test Stage 1 compiler
echo "Step 4: Testing Stage 1 compiler..."
echo "-------------------------------------------------------------------"

# Test with a simple hello world program
cat > "$BUILD_DIR/test_hello.nano" << 'EOF'
fn main() -> int {
    (println "Hello from Stage 1!")
    return 0
}

shadow main {
    assert (== (main) 0)
}
EOF

echo "Testing with test_hello.nano..."
"$BIN_DIR/nanoc_stage1" "$BUILD_DIR/test_hello.nano" -o "$BUILD_DIR/test_hello"

if [ $? -eq 0 ]; then
    echo "✓ Stage 1 compiler can compile test program"
    
    # Run the test program
    if [ -f "$BUILD_DIR/test_hello" ]; then
        echo "Running test program..."
        "$BUILD_DIR/test_hello"
        echo "✓ Test program runs successfully"
    fi
else
    echo "✗ Stage 1 compiler failed to compile test program"
    exit 1
fi

echo ""

# Clean up test files
rm -f "$BUILD_DIR/test_hello.nano" "$BUILD_DIR/test_hello" "$BUILD_DIR/test_hello.nano.genC"
rm -f "$BUILD_DIR/stage1_compiler_temp"

echo "==================================================================="
echo "  Stage 1 Compiler Build Complete!"
echo "==================================================================="
echo ""
echo "Executable: $BIN_DIR/nanoc_stage1"
echo ""
echo "Usage:"
echo "  $BIN_DIR/nanoc_stage1 <input.nano> -o <output>"
echo "  $BIN_DIR/nanoc_stage1 --help"
echo ""
echo "Next steps:"
echo "  1. Test Stage 1: $BIN_DIR/nanoc_stage1 examples/hello.nano -o hello"
echo "  2. Build Stage 2: $BIN_DIR/nanoc_stage1 $SRC_NANO/stage1_compiler.nano -o $BIN_DIR/nanoc_stage2"
echo "  3. Verify: Compare Stage 1 and Stage 2 outputs"
echo ""
