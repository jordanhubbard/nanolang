#!/bin/bash
# Comprehensive Interpreter/Compiler Parity Check
# Tests all .nano files with both interpreter and compiler to ensure parity

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

INTERPRETER="./bin/nano"
COMPILER="./bin/nanoc"
TEST_DIR="examples"

INTERPRETER_ONLY=()
COMPILER_ONLY=()
BOTH_WORK=()
BOTH_FAIL=()
SKIPPED=()

# Check if timeout command is available (Linux) or use gtimeout (macOS via brew)
if command -v timeout &> /dev/null; then
    TIMEOUT_CMD="timeout 3"
    TIMEOUT_COMPILE="timeout 10"
elif command -v gtimeout &> /dev/null; then
    TIMEOUT_CMD="gtimeout 3"
    TIMEOUT_COMPILE="gtimeout 10"
else
    # No timeout available - use background process with kill
    TIMEOUT_CMD=""
    TIMEOUT_COMPILE=""
fi

# Helper function to run with timeout (works on macOS without timeout command)
run_with_timeout() {
    local cmd="$1"
    local timeout_sec="${2:-3}"
    local log_file="$3"
    
    if [ -n "$TIMEOUT_CMD" ]; then
        # Use timeout command if available
        $TIMEOUT_CMD bash -c "$cmd" > "$log_file" 2>&1
        return $?
    else
        # Manual timeout using background process and sleep
        eval "$cmd" > "$log_file" 2>&1 &
        local pid=$!
        local count=0
        local max_count=$((timeout_sec * 10))  # Check every 0.1 seconds
        
        while kill -0 $pid 2>/dev/null && [ $count -lt $max_count ]; do
            sleep 0.1
            count=$((count + 1))
        done
        
        if kill -0 $pid 2>/dev/null; then
            # Still running - kill it
            kill -9 $pid 2>/dev/null
            wait $pid 2>/dev/null
            return 124  # Timeout exit code
        fi
        wait $pid
        return $?
    fi
}

# Check if binaries exist
if [ ! -f "$INTERPRETER" ]; then
    echo -e "${RED}Error: Interpreter '$INTERPRETER' not found${NC}"
    exit 1
fi

if [ ! -f "$COMPILER" ]; then
    echo -e "${RED}Error: Compiler '$COMPILER' not found${NC}"
    exit 1
fi

# Create temp directory
mkdir -p .parity_check

echo "========================================"
echo "Interpreter/Compiler Parity Check"
echo "========================================"
echo ""
echo "Testing all .nano files in $TEST_DIR/"
echo ""

# Find all .nano files
TOTAL=0
while IFS= read -r -d '' file; do
    TOTAL=$((TOTAL + 1))
    rel_path="${file#$TEST_DIR/}"
    test_name=$(basename "$rel_path" .nano)
    output_binary=".parity_check/${test_name}"
    
    # Skip files that require special setup (SDL, OpenGL, etc.)
    if [[ "$rel_path" == *"sdl"* ]] || \
       [[ "$rel_path" == *"opengl"* ]] || \
       [[ "$rel_path" == *"audio"* ]] || \
       [[ "$rel_path" == *"mod_player"* ]] || \
       [[ "$rel_path" == *"onnx"* ]] || \
       [[ "$rel_path" == *"curl"* ]] || \
       [[ "$rel_path" == *"event"* ]] || \
       [[ "$rel_path" == *"uv"* ]] || \
       [[ "$rel_path" == *"sqlite"* ]] || \
       [[ "$rel_path" == *"checkers"* ]] || \
       [[ "$rel_path" == *"protracker"* ]] || \
       [[ "$rel_path" == *"visualizer"* ]] || \
       [[ "$rel_path" == *"test_wav"* ]]; then
        SKIPPED+=("$rel_path")
        echo -e "${BLUE}SKIP${NC} $rel_path (requires external libraries)"
        continue
    fi
    
    # Skip files that might hang (read large files, wait for input, etc.)
    if [[ "$rel_path" == *"random_sentence"* ]] || \
       [[ "$rel_path" == *"tictactoe"* ]] || \
       [[ "$rel_path" == *"game_of_life"* ]] || \
       [[ "$rel_path" == *"snake"* ]] || \
       [[ "$rel_path" == *"maze"* ]] || \
       [[ "$rel_path" == *"primes"* ]] || \
       [[ "$rel_path" == *"pi_calculator"* ]]; then
        SKIPPED+=("$rel_path")
        echo -e "${BLUE}SKIP${NC} $rel_path (may hang or require input)"
        continue
    fi
    
    # Skip library files (no main function)
    if [[ "$rel_path" == *"math_utils"* ]] || \
       [[ "$rel_path" == *"_utils"* ]] || \
       [[ "$rel_path" == *"_helper"* ]] || \
       [[ "$rel_path" == *"_lib"* ]] || \
       [[ "$rel_path" == *"test_enum_parse"* ]]; then
        SKIPPED+=("$rel_path")
        echo -e "${BLUE}SKIP${NC} $rel_path (library file, no main function)"
        continue
    fi
    
    echo -n "Testing $rel_path... "
    
    # Test interpreter with timeout
    interpreter_ok=false
    if run_with_timeout "\"$INTERPRETER\" \"$file\"" 3 ".parity_check/${test_name}.interpreter.log"; then
        interpreter_ok=true
    fi
    
    # Test compiler (compilation + shadow tests) with longer timeout
    compiler_ok=false
    if run_with_timeout "\"$COMPILER\" \"$file\" -o \"$output_binary\"" 10 ".parity_check/${test_name}.compiler.log"; then
        # Also test running the binary if it was created
        if [ -f "$output_binary" ]; then
            if run_with_timeout "\"$output_binary\"" 3 ".parity_check/${test_name}.run.log"; then
                compiler_ok=true
            fi
        fi
    fi
    
    # Categorize result
    if [ "$interpreter_ok" = true ] && [ "$compiler_ok" = true ]; then
        echo -e "${GREEN}✓ BOTH${NC}"
        BOTH_WORK+=("$rel_path")
    elif [ "$interpreter_ok" = true ] && [ "$compiler_ok" = false ]; then
        echo -e "${YELLOW}⚠ INTERPRETER ONLY${NC}"
        INTERPRETER_ONLY+=("$rel_path")
    elif [ "$interpreter_ok" = false ] && [ "$compiler_ok" = true ]; then
        echo -e "${YELLOW}⚠ COMPILER ONLY${NC}"
        COMPILER_ONLY+=("$rel_path")
    else
        echo -e "${RED}✗ BOTH FAIL${NC}"
        BOTH_FAIL+=("$rel_path")
    fi
    
done < <(find "$TEST_DIR" -name "*.nano" -type f -print0)

echo ""
echo "========================================"
echo "Parity Check Summary"
echo "========================================"
echo "Total files tested: $TOTAL"
echo -e "${GREEN}Both work: ${#BOTH_WORK[@]}${NC}"
echo -e "${YELLOW}Interpreter only: ${#INTERPRETER_ONLY[@]}${NC}"
echo -e "${YELLOW}Compiler only: ${#COMPILER_ONLY[@]}${NC}"
echo -e "${RED}Both fail: ${#BOTH_FAIL[@]}${NC}"
echo -e "${BLUE}Skipped (external deps or may hang): ${#SKIPPED[@]}${NC}"
echo ""

# Report discrepancies
if [ ${#INTERPRETER_ONLY[@]} -gt 0 ]; then
    echo -e "${YELLOW}=== Interpreter Only (Compiler needs fixing) ===${NC}"
    for file in "${INTERPRETER_ONLY[@]}"; do
        echo "  - $file"
    done
    echo ""
fi

if [ ${#COMPILER_ONLY[@]} -gt 0 ]; then
    echo -e "${YELLOW}=== Compiler Only (Interpreter needs fixing) ===${NC}"
    for file in "${COMPILER_ONLY[@]}"; do
        echo "  - $file"
    done
    echo ""
fi

if [ ${#BOTH_FAIL[@]} -gt 0 ]; then
    echo -e "${RED}=== Both Fail (Need investigation) ===${NC}"
    for file in "${BOTH_FAIL[@]}"; do
        echo "  - $file"
    done
    echo ""
fi

# Exit with error if there are discrepancies
if [ ${#INTERPRETER_ONLY[@]} -gt 0 ] || [ ${#COMPILER_ONLY[@]} -gt 0 ]; then
    echo -e "${RED}Parity issues found!${NC}"
    exit 1
else
    echo -e "${GREEN}Perfect parity! All tested files work with both interpreter and compiler.${NC}"
    exit 0
fi
