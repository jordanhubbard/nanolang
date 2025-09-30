#!/bin/bash
# Simple test script for NanoLang interpreter

echo "=== NanoLang Test Suite ==="
echo

# Build the interpreter
echo "Building interpreter..."
make clean > /dev/null 2>&1
make > /dev/null 2>&1

if [ ! -f ./nano ]; then
    echo "✗ Build failed"
    exit 1
fi
echo "✓ Build successful"
echo

# Test each example
examples=(
    "variables.nano"
    "conditionals.nano"
    "fibonacci.nano"
    "factorial.nano"
    "prime.nano"
    "comprehensive.nano"
)

echo "Running example programs..."
for example in "${examples[@]}"; do
    if ./nano "examples/$example" > /dev/null 2>&1; then
        echo "✓ $example"
    else
        echo "✗ $example"
    fi
done

echo
echo "=== All tests completed ==="
