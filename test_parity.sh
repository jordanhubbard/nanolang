#!/bin/bash
# Test interpreter vs compiler feature parity

EXAMPLES=(
    "examples/hello.nano"
    "examples/factorial.nano"
    "examples/fibonacci.nano"
    "examples/calculator.nano"
    "examples/primes.nano"
    "examples/01_operators.nano"
    "examples/02_strings.nano"
    "examples/03_floats.nano"
    "examples/04_loops_working.nano"
    "examples/05_mutable.nano"
    "examples/06_logical.nano"
    "examples/07_comparisons.nano"
    "examples/08_types.nano"
    "examples/09_math.nano"
    "examples/10_os_basic.nano"
    "examples/11_stdlib_test.nano"
    "examples/12_advanced_math.nano"
    "examples/13_string_ops.nano"
    "examples/14_arrays_simple.nano"
    "examples/17_struct_test.nano"
    "examples/18_enum_test.nano"
    "examples/20_string_operations.nano"
    "examples/28_union_types.nano"
    "examples/31_first_class_functions.nano"
    "examples/32_filter_map_fold.nano"
)

PASS=0
FAIL=0
DIFF=0

for example in "${EXAMPLES[@]}"; do
    echo "=================================================="
    echo "Testing: $example"
    echo "=================================================="
    
    # Run interpreter - capture stdout and stderr separately
    echo "--- Interpreter Output ---"
    NANO_MODULE_PATH=modules ./bin/nano "$example" > /tmp/interp_stdout.txt 2> /tmp/interp_stderr.txt
    INTERP_EXIT=$?
    # Filter warnings from stderr (warnings are multi-line, so use sed to remove warning blocks)
    cat /tmp/interp_stderr.txt | sed '/^Warning:/,/^$/d' | grep -v "^Running shadow tests" | grep -v "^Testing" | grep -v "PASSED$" | grep -v "SKIPPED" | grep -v "All shadow tests passed" > /tmp/interp_stderr_filtered.txt
    # Combine stdout and filtered stderr
    (cat /tmp/interp_stdout.txt; cat /tmp/interp_stderr_filtered.txt) > /tmp/interp_out.txt
    cat /tmp/interp_out.txt
    
    # Compile and run - capture compilation output separately
    echo ""
    echo "--- Compiler Output ---"
    NANO_MODULE_PATH=modules ./bin/nanoc "$example" -o /tmp/test_bin > /tmp/compile_stdout.txt 2> /tmp/compile_stderr.txt
    COMPILE_EXIT=$?
    # Filter shadow test output from stderr
    cat /tmp/compile_stderr.txt | grep -v "^Running shadow tests" | grep -v "^Testing" | grep -v "PASSED$" | grep -v "SKIPPED" | grep -v "All shadow tests passed" > /tmp/compile_filtered.txt || true
    
    if [ -f /tmp/test_bin ]; then
        /tmp/test_bin > /tmp/compiled_out.txt 2>&1
        COMP_EXIT=$?
        cat /tmp/compiled_out.txt
        rm /tmp/test_bin
    else
        echo "COMPILATION FAILED:"
        cat /tmp/compile_stdout.txt
        cat /tmp/compile_filtered.txt
        COMP_EXIT=1
    fi
    
    echo ""
    echo "--- Comparison ---"
    if [ $INTERP_EXIT -ne 0 ] && [ $COMP_EXIT -ne 0 ]; then
        echo "✓ Both failed (expected for some examples)"
        PASS=$((PASS + 1))
    elif [ $INTERP_EXIT -eq 0 ] && [ $COMP_EXIT -eq 0 ]; then
        if diff -q /tmp/interp_out.txt /tmp/compiled_out.txt > /dev/null; then
            echo "✓ PASS - Outputs match"
            PASS=$((PASS + 1))
        else
            echo "⚠ DIFF - Outputs differ:"
            diff /tmp/interp_out.txt /tmp/compiled_out.txt || true
            DIFF=$((DIFF + 1))
        fi
    else
        echo "✗ FAIL - Different exit codes (interp: $INTERP_EXIT, comp: $COMP_EXIT)"
        FAIL=$((FAIL + 1))
    fi
    
    echo ""
done

echo "=================================================="
echo "SUMMARY"
echo "=================================================="
echo "PASS: $PASS"
echo "DIFF: $DIFF (outputs differ but both work)"
echo "FAIL: $FAIL (different exit codes)"

