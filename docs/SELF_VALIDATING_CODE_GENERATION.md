# Self-Validating Code Generation with NanoLang

> **For LLM Agents:** A complete guide to generating, testing, and automatically correcting code without human intervention.

---

## Overview

**Self-validating code generation** is a workflow where LLM agents:
1. Generate code with embedded tests
2. Compile and run tests automatically
3. Parse failures and fix errors
4. Iterate until all tests pass

**Result:** Correct code generated with minimal human oversight.

---

## The 4-Step Workflow

```
┌──────────────────────────────────────────┐
│ STEP 1: Generate Code + Tests           │
│ - Write function implementation          │
│ - Include shadow tests (mandatory)       │
│ - Add property tests (optional)          │
│ - Include structured logging             │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│ STEP 2: Compile                          │
│ $ ./bin/nanoc generated.nano -o output   │
└──────────────┬───────────────────────────┘
               │
               ▼
         ┌─────┴─────┐
         │  Success? │
         └─────┬─────┘
         ┌─────┴─────┐
    YES  │           │  NO
         ▼           ▼
   ┌─────────┐  ┌──────────────────────┐
   │  DONE!  │  │ STEP 3: Parse Error  │
   └─────────┘  │ - Extract line number │
                │ - Read error message  │
                │ - Identify problem    │
                └──────────┬────────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ STEP 4: Fix & Retry  │
                │ - Correct the issue  │
                │ - Regenerate code    │
                │ - Back to STEP 2     │
                └──────────────────────┘
```

---

## STEP 1: Generate Code with Tests

### Minimum Viable Generation

Every function MUST have:
1. **Implementation** - The actual code
2. **Shadow test** - At least one assertion

```nano
fn factorial(n: int) -> int {
    if (<= n 1) {
        return 1
    } else {
        return (* n (factorial (- n 1)))
    }
}

shadow factorial {
    assert (== (factorial 0) 1)
    assert (== (factorial 5) 120)
    assert (== (factorial 10) 3628800)
}
```

### Enhanced Generation (Recommended)

Add these for robustness:

```nano
from "stdlib/log.nano" import log_debug, log_info
from "modules/proptest/proptest.nano" import proptest_int

fn factorial(n: int) -> int {
    (log_debug "factorial" (+ "Computing factorial(" (+ (int_to_string n) ")")))
    
    if (<= n 1) {
        (log_debug "factorial" "Base case reached")
        return 1
    } else {
        let result: int = (* n (factorial (- n 1)))
        (log_info "factorial" (+ "factorial(" (+ (int_to_string n) (+ ") = " (int_to_string result)))))
        return result
    }
}

shadow factorial {
    # Example-based tests
    assert (== (factorial 0) 1)
    assert (== (factorial 5) 120)
    assert (== (factorial 10) 3628800)
    
    # Property-based test: factorial(n) * n+1 = factorial(n+1)
    (proptest_int "recursive_property" 50
        (fn (n: int) -> bool {
            if (or (< n 0) (> n 12)) { return true }  # Skip out of range
            let fn: int = (factorial n)
            let fn_plus_1: int = (factorial (+ n 1))
            return (== (* fn (+ n 1)) fn_plus_1)
        }))
}
```

**Benefits:**
- Logging reveals runtime behavior during debugging
- Property tests catch edge cases
- Multiple assertion types increase confidence

---

## STEP 2: Compile

### Command

```bash
./bin/nanoc generated_code.nano -o output
```

### Expected Outputs

#### Success
```
✓ Type checking complete
✓ Compilation successful
Running shadow tests...
Testing factorial... PASSED
All shadow tests passed!
```

→ **Proceed to run the program**

#### Compile Error (Syntax/Type)
```
Error at line 15, column 20: Type mismatch in let statement
  Expected: int
  Got: string
  Hint: Check the type annotation matches the assigned value
Type checking failed
```

→ **Proceed to STEP 3 (Parse Error)**

#### Shadow Test Failure
```
Assertion failed at line 23, column 5
Shadow test 'factorial' failed
```

→ **Proceed to STEP 3 (Logic Error)**

---

## STEP 3: Parse Errors

### Error Categories

| Error Type | Example | Root Cause |
|------------|---------|------------|
| **Syntax** | `Expected '}' after block` | Malformed code structure |
| **Type** | `Expected int, found string` | Type annotation mismatch |
| **Undefined** | `Undefined function 'foo'` | Missing import or typo |
| **Logic** | `Assertion failed at line X` | Algorithm incorrect |

### Parsing Strategy

#### 1. Extract Location
```
Error at line 15, column 20: ...
           ^^^^^  ^^^^^^^^^
           Line    Column
```

**Action:** Identify which line in your generated code has the error.

#### 2. Extract Error Code
```
Error at line 15, column 20: Type mismatch [E0001]
                                             ^^^^^^
                                           Error code
```

**Use:** Reference documentation for common fixes.

#### 3. Extract Context
```
Error at line 15, column 20: Type mismatch in let statement
  Expected: int
  Got: string
  Hint: Check the type annotation matches the assigned value
```

**Action:** 
- Read "Expected" type
- Read "Got" type
- Apply hint

---

## STEP 4: Fix and Retry

### Example: Type Mismatch

**Generated code (WRONG):**
```nano
fn process_age(age_str: string) -> int {
    let age: int = age_str  # ERROR: Type mismatch
    return age
}
```

**Error:**
```
Error at line 2, column 5: Type mismatch in let statement
  Expected: int
  Got: string
```

**Fix:**
```nano
fn process_age(age_str: string) -> int {
    let age: int = (parse_int age_str)  # ✓ FIXED: Convert string to int
    return age
}
```

**Retry:** Recompile → Success!

### Example: Undefined Function

**Generated code (WRONG):**
```nano
fn calculate_sum(arr: array<int>) -> int {
    return (sum arr)  # ERROR: Undefined function 'sum'
}
```

**Error:**
```
Error at line 2, column 12: Undefined function 'sum'
```

**Fix (Option A - Implement):**
```nano
fn calculate_sum(arr: array<int>) -> int {
    let mut total: int = 0
    let mut i: int = 0
    while (< i (array_length arr)) {
        set total (+ total (at arr i))
        set i (+ i 1)
    }
    return total
}
```

**Fix (Option B - Use stdlib):**
```nano
from "stdlib/array.nano" import array_reduce  # If it exists

fn calculate_sum(arr: array<int>) -> int {
    return (array_reduce arr (fn (acc: int, x: int) -> int { (+ acc x) }) 0)
}
```

**Retry:** Recompile → Success!

### Example: Assertion Failure

**Generated code (WRONG):**
```nano
fn reverse_string(s: string) -> string {
    # Implementation missing!
    return s  # Just returns original
}

shadow reverse_string {
    assert (str_equals (reverse_string "hello") "olleh")  # FAILS
}
```

**Error:**
```
Assertion failed at line 7, column 5
Shadow test 'reverse_string' failed
```

**Diagnosis:**
- Implementation doesn't actually reverse
- Need to write the algorithm

**Fix:**
```nano
fn reverse_string(s: string) -> string {
    let len: int = (str_length s)
    let mut result: string = ""
    let mut i: int = (- len 1)
    
    while (>= i 0) {
        let ch: string = (str_substring s i 1)
        set result (+ result ch)
        set i (- i 1)
    }
    
    return result
}

shadow reverse_string {
    assert (str_equals (reverse_string "hello") "olleh")  # ✓ PASSES
    assert (str_equals (reverse_string "") "")
    assert (str_equals (reverse_string "a") "a")
}
```

**Retry:** Recompile → Success!

---

## Complete Example: LLM Agent Iteration

### Iteration 1: Initial Generation

```nano
# Goal: Sort an array of integers

fn sort_array(arr: array<int>) -> array<int> {
    return arr  # TODO: Implement
}

shadow sort_array {
    assert (arrays_equal (sort_array [3, 1, 2]) [1, 2, 3])
}
```

**Compile:**
```
Error at line 7, column 19: Undefined function 'arrays_equal'
```

### Iteration 2: Fix Undefined Function

```nano
fn arrays_equal(a: array<int>, b: array<int>) -> bool {
    if (!= (array_length a) (array_length b)) {
        return false
    }
    let mut i: int = 0
    while (< i (array_length a)) {
        if (!= (at a i) (at b i)) {
            return false
        }
        set i (+ i 1)
    }
    return true
}

fn sort_array(arr: array<int>) -> array<int> {
    return arr  # TODO: Implement
}

shadow sort_array {
    assert (arrays_equal (sort_array [3, 1, 2]) [1, 2, 3])
}
```

**Compile:**
```
✓ Compilation successful
Assertion failed at line 19, column 5
Shadow test 'sort_array' failed
```

### Iteration 3: Implement Algorithm

```nano
fn arrays_equal(a: array<int>, b: array<int>) -> bool {
    # ... (same as before)
}

fn sort_array(arr: array<int>) -> array<int> {
    let len: int = (array_length arr)
    if (<= len 1) { return arr }
    
    # Bubble sort implementation
    let mut result: array<int> = []
    let mut i: int = 0
    while (< i len) {
        set result (array_push result (at arr i))
        set i (+ i 1)
    }
    
    let mut swapped: bool = true
    while swapped {
        set swapped false
        set i 0
        while (< i (- (array_length result) 1)) {
            if (> (at result i) (at result (+ i 1))) {
                # Swap
                let temp: int = (at result i)
                set result (array_set result i (at result (+ i 1)))
                set result (array_set result (+ i 1) temp)
                set swapped true
            }
            set i (+ i 1)
        }
    }
    
    return result
}

shadow sort_array {
    assert (arrays_equal (sort_array [3, 1, 2]) [1, 2, 3])
    assert (arrays_equal (sort_array []) [])
    assert (arrays_equal (sort_array [1]) [1])
    assert (arrays_equal (sort_array [5, 4, 3, 2, 1]) [1, 2, 3, 4, 5])
}
```

**Compile:**
```
✓ Compilation successful
Running shadow tests...
Testing sort_array... PASSED
All shadow tests passed!
```

**SUCCESS!** ✓

---

## Advanced Patterns

### Pattern 1: Logging for Debugging

When tests fail and you can't figure out why, add logging:

```nano
from "stdlib/log.nano" import log_debug

fn mysterious_function(x: int) -> int {
    (log_debug "mystery" (+ "Input: " (int_to_string x)))
    
    let step1: int = (* x 2)
    (log_debug "mystery" (+ "After doubling: " (int_to_string step1)))
    
    let step2: int = (+ step1 10)
    (log_debug "mystery" (+ "After adding 10: " (int_to_string step2)))
    
    return step2
}
```

**Run to see intermediate values:**
```
[DEBUG] mystery: Input: 5
[DEBUG] mystery: After doubling: 10
[DEBUG] mystery: After adding 10: 20
```

**Remove logging once working!**

### Pattern 2: Incremental Complexity

Don't generate everything at once. Build incrementally:

```nano
# STEP 1: Basic case
fn fibonacci(n: int) -> int {
    if (<= n 1) { return n }
    return 0  # Placeholder
}

shadow fibonacci {
    assert (== (fibonacci 0) 0)
    assert (== (fibonacci 1) 1)
}
```

**Compile → Tests pass for base case**

```nano
# STEP 2: Add recursion
fn fibonacci(n: int) -> int {
    if (<= n 1) { return n }
    return (+ (fibonacci (- n 1)) (fibonacci (- n 2)))
}

shadow fibonacci {
    assert (== (fibonacci 0) 0)
    assert (== (fibonacci 1) 1)
    assert (== (fibonacci 5) 5)  # Add recursive test
    assert (== (fibonacci 10) 55)
}
```

**Compile → All tests pass!**

### Pattern 3: Property Tests for Validation

Once basic tests pass, add property tests for confidence:

```nano
from "modules/proptest/proptest.nano" import proptest_int_array

shadow sort_array {
    # Example tests
    assert (arrays_equal (sort_array [3, 1, 2]) [1, 2, 3])
    
    # Property tests (catch edge cases)
    (proptest_int_array "length_preserved" 100
        (fn (arr: array<int>) -> bool {
            return (== (array_length arr) (array_length (sort_array arr)))
        }))
    
    (proptest_int_array "is_sorted" 100
        (fn (arr: array<int>) -> bool {
            let sorted: array<int> = (sort_array arr)
            return (is_ascending_order sorted)
        }))
}
```

**If property test fails:**
```
✗ Property 'is_sorted' failed after 42 tests
  Counterexample: [-2147483648, 0, 1]
```

→ Found an edge case with minimum integer!

---

## Error Patterns Cheatsheet

### Common Errors and Fixes

| Error Message | Cause | Fix |
|---------------|-------|-----|
| `Expected int, found string` | Wrong type | Convert: `(parse_int s)` or `(int_to_string n)` |
| `Undefined function 'X'` | Missing import | Add `from "module" import X` or implement |
| `Undefined variable 'X'` | Typo or not declared | Check spelling, add `let X: Type = ...` |
| `Cannot assign to immutable` | Missing `mut` | Change to `let mut X: Type = ...` |
| `Assertion failed` | Logic error | Debug algorithm, add logging |
| `Unexpected token` | Syntax error | Check parentheses, braces matching |
| `Module not found` | Wrong path | Verify module path exists |

---

## Best Practices for LLM Agents

### ✅ DO

1. **Always include shadow tests** - Mandatory for validation
2. **Start simple** - Implement minimum viable version first
3. **Add logging during debugging** - Remove once working
4. **Use property tests for algorithms** - Catch edge cases
5. **Parse error messages carefully** - Extract line, type, hint
6. **Iterate methodically** - Fix one error at a time
7. **Test edge cases explicitly** - Empty input, zero, negative, max values

### ❌ DON'T

1. **Don't generate without tests** - You'll have no validation
2. **Don't ignore compiler warnings** - They indicate potential issues
3. **Don't fix multiple errors at once** - Change one thing per iteration
4. **Don't assume correctness** - Always verify with tests
5. **Don't leave debug logging** - Remove before final version
6. **Don't skip shadow tests** - They're mandatory, not optional

---

## Metrics for Success

Track these to measure agent effectiveness:

| Metric | Good | Needs Improvement |
|--------|------|-------------------|
| **Iterations to Success** | 1-3 | >5 |
| **Test Coverage** | Shadow + property tests | Shadow only |
| **Error Recovery Rate** | >90% auto-fixed | <70% auto-fixed |
| **Time to Fix** | <30 seconds/iteration | >2 minutes/iteration |

---

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: LLM-Generated Code Validation
on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Build NanoLang Compiler
        run: make
      
      - name: Compile Generated Code
        run: |
          ./bin/nanoc generated/algorithm.nano -o output
          # Shadow tests run automatically during compilation
      
      - name: Run Program
        run: ./output
      
      - name: Verify Output
        run: |
          if [ $? -eq 0 ]; then
            echo "✓ All tests passed"
          else
            echo "✗ Tests failed"
            exit 1
          fi
```

---

## Summary

**Self-validating code generation workflow:**

```
Generate → Compile → Parse Errors → Fix → Repeat until Success
```

**Key components:**
- ✅ Shadow tests (mandatory validation)
- ✅ Property tests (edge case coverage)
- ✅ Structured logging (debugging)
- ✅ Error parsing (automated fixes)
- ✅ Iterative refinement (continuous improvement)

**Result:** LLM agents that autonomously generate correct, tested code with minimal human intervention.

---

## Further Reading

- **Debugging Guide:** `docs/DEBUGGING_GUIDE.md`
- **Property Testing Guide:** `docs/PROPERTY_TESTING_GUIDE.md`
- **Canonical Style:** `docs/CANONICAL_STYLE.md`
- **LLM Core Subset:** `docs/LLM_CORE_SUBSET.md`

**Start generating self-validating code today!** 🚀
