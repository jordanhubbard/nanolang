# My Debugging and Tracing Guide

> **For LLM Agents:** This is your canonical reference for debugging programs I run and implementing self-validating code generation.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [My Structured Logging API](#structured-logging-api)
3. [Shadow Tests for Compile-Time Validation](#shadow-tests)
4. [Property-Based Testing](#property-based-testing)
5. [My Compiler Diagnostics](#compiler-diagnostics)
6. [Common Debugging Patterns](#common-debugging-patterns)
7. [LLM Agent Feedback Loops](#llm-agent-feedback-loops)

---

## Quick Start

### My 3-Layer Debugging Strategy

I provide three complementary debugging mechanisms:

```
┌─────────────────────────────────────────┐
│ 1. Structured Logging (Runtime)        │ ← Use during development
│    - Log levels (TRACE → FATAL)        │
│    - Category-based organization        │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│ 2. Shadow Tests (Compile-Time)         │ ← I require these for functions
│    - Inline test assertions             │
│    - Run before program execution       │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│ 3. Property Tests (Systematic)          │ ← For algorithmic validation
│    - Randomized testing                 │
│    - Automatic shrinking                │
└─────────────────────────────────────────┘
```

**My rule of thumb:**
- **Shadow tests** for correctness. I require these.
- **Logging** for runtime behavior during your debugging process.
- **Property tests** for algorithmic properties. These are optional but I find them powerful.

---

## My Structured Logging API

### Basic Usage

```nano
from "stdlib/log.nano" import log_info, log_debug, log_error

fn process_user_input(input: string) -> bool {
    (log_info "validation" "Processing user input")
    
    if (str_equals input "") {
        (log_error "validation" "Empty input received")
        return false
    }
    
    (log_debug "validation" (+ "Input length: " (int_to_string (str_length input))))
    (log_info "validation" "Input validated successfully")
    return true
}
```

**Output:**
```
[INFO] validation: Processing user input
[DEBUG] validation: Input length: 42
[INFO] validation: Input validated successfully
```

### Log Levels

| Level | Severity | Use Case | Example |
|-------|----------|----------|---------|
| TRACE | Lowest | Detailed execution flow | Function entry and exit points |
| DEBUG | Low | Development debugging | Variable values, intermediate results |
| INFO | Normal | Operational milestones | "Server started", "File loaded" |
| WARN | Medium | Potentially problematic | Deprecated API usage, recoverable errors |
| ERROR | High | Operation failures | Failed validation, I/O errors |
| FATAL | Highest | Critical failures | Cannot continue execution |

**My default threshold:** INFO. I suppress DEBUG and TRACE unless you tell me otherwise.

### API Reference

#### Category-Based Logging
```nano
from "stdlib/log.nano" import log_trace, log_debug, log_info, 
                                log_warn, log_error, log_fatal

(log_trace "module" "message")   # [TRACE] module: message
(log_debug "module" "message")   # [DEBUG] module: message
(log_info "module" "message")    # [INFO] module: message
(log_warn "module" "message")    # [WARN] module: message
(log_error "module" "message")   # [ERROR] module: message
(log_fatal "module" "message")   # [FATAL] module: message
```

#### Convenience Functions (No Category)
```nano
from "stdlib/log.nano" import trace, debug, info, warn, error, fatal

(info "Simple message")          # [INFO] Simple message
(error "Error occurred")         # [ERROR] Error occurred
```

### Best Practices

#### DO: Use Categories for Organization
```nano
fn load_config(path: string) -> bool {
    (log_info "config" "Loading configuration")
    (log_debug "config" (+ "Path: " path))
    # ...
}

fn validate_data(data: array<int>) -> bool {
    (log_info "validation" "Starting validation")
    # ...
}
```

**Benefit:** You can easily filter my logs by subsystem.

#### DO: Log State Transitions
```nano
fn connect_to_server(host: string) -> bool {
    (log_info "network" "Attempting connection")
    
    if (not (is_reachable host)) {
        (log_error "network" (+ "Host unreachable: " host))
        return false
    }
    
    (log_info "network" "Connection established")
    return true
}
```

#### DO NOT: Log in Tight Loops
```nano
# BAD - floods output
for (let i: int = 0) (< i 10000) (set i (+ i 1)) {
    (log_debug "loop" (+ "Iteration " (int_to_string i)))  # Too verbose!
}

# GOOD - log summary
(log_debug "loop" (+ "Processing " (int_to_string 10000) " items"))
```

#### DO NOT: Log Sensitive Data
```nano
# BAD - exposes password
(log_info "auth" (+ "Password: " user_password))

# GOOD - log without sensitive info
(log_info "auth" (+ "Authenticating user: " username))
```

---

## Shadow Tests

### Purpose

Shadow tests are my **mandatory inline tests** that run at compile time. I will not compile a function unless it has a shadow test, unless it is an `extern` FFI function.

### Syntax

```nano
fn add(a: int, b: int) -> int {
    return (+ a b)
}

shadow add {
    assert (== (add 2 3) 5)
    assert (== (add 0 0) 0)
    assert (== (add -1 1) 0)
}
```

### When I Run Shadow Tests

```
1. Code written → 2. Compiled → 3. I execute shadow tests → 4. Program runs
                                       ↓
                               [FAIL: I abort compilation]
                               [PASS: I continue to program]
```

### Testing Strategy

#### Test Edge Cases
```nano
fn divide(a: int, b: int) -> int {
    return (/ a b)
}

shadow divide {
    assert (== (divide 10 2) 5)      # Normal case
    assert (== (divide 0 5) 0)       # Zero numerator
    assert (== (divide 7 3) 2)       # Integer division (truncates)
    # Note: divide-by-zero would crash - handle in production code
}
```

#### Test Boundary Conditions
```nano
fn clamp(value: int, min: int, max: int) -> int {
    if (< value min) { return min } else {
    if (> value max) { return max } else {
        return value
    }}
}

shadow clamp {
    assert (== (clamp 5 0 10) 5)     # Within range
    assert (== (clamp -5 0 10) 0)    # Below minimum
    assert (== (clamp 15 0 10) 10)   # Above maximum
    assert (== (clamp 0 0 10) 0)     # At minimum boundary
    assert (== (clamp 10 0 10) 10)   # At maximum boundary
}
```

#### Test Type-Specific Behaviors
```nano
fn concat_with_separator(a: string, b: string, sep: string) -> string {
    return (+ a (+ sep b))
}

shadow concat_with_separator {
    assert (str_equals (concat_with_separator "hello" "world" " ") "hello world")
    assert (str_equals (concat_with_separator "a" "b" "") "ab")
    assert (str_equals (concat_with_separator "" "" ",") ",")
}
```

### My Shadow Test Limitations

**I cannot test:**
- Side effects (I/O, global state)
- Non-deterministic behavior (random numbers, time)
- External dependencies (network, filesystem)

**You should test:**
- Pure functions (same input produces same output)
- Mathematical properties
- Data transformations
- Algorithm correctness

---

## Property-Based Testing

### Overview

My property-based testing module **generates random inputs** to test algorithmic properties. I find this much more powerful than example-based testing.

**Module:** `modules/proptest/proptest.nano`

### Example: Testing List Reversal

```nano
from "modules/proptest/proptest.nano" import proptest_int_array

fn reverse_list(lst: array<int>) -> array<int> {
    let mut result: array<int> = []
    let len: int = (array_length lst)
    let mut i: int = (- len 1)
    
    while (>= i 0) {
        set result (array_push result (at lst i))
        set i (- i 1)
    }
    
    return result
}

shadow reverse_list {
    # Property: reverse(reverse(x)) == x
    (proptest_int_array "reverse_twice_is_identity" 100 
        (fn (lst: array<int>) -> bool {
            let reversed_once: array<int> = (reverse_list lst)
            let reversed_twice: array<int> = (reverse_list reversed_once)
            return (arrays_equal lst reversed_twice)
        }))
}
```

**What I do here:**
1. I generate 100 random integer arrays.
2. I test the property for each array.
3. If I find a failure, I **shrink** it to the minimal failing case.
4. I report: "Failed on input: [1, 2]".

### Common Properties to Test

#### Idempotence
```nano
# Property: f(f(x)) == f(x)
fn normalize(s: string) -> string {
    # Remove leading and trailing whitespace
}

shadow normalize {
    (proptest_string "normalize_is_idempotent" 50
        (fn (s: string) -> bool {
            let once: string = (normalize s)
            let twice: string = (normalize once)
            return (str_equals once twice)
        }))
}
```

#### Symmetry
```nano
# Property: distance(a, b) == distance(b, a)
fn euclidean_distance(x1: int, y1: int, x2: int, y2: int) -> float {
    # Calculate distance
}

shadow euclidean_distance {
    (proptest_int_array "distance_is_symmetric" 50
        (fn (coords: array<int>) -> bool {
            if (!= (array_length coords) 4) { return true }
            let d1: float = (euclidean_distance (at coords 0) (at coords 1) (at coords 2) (at coords 3))
            let d2: float = (euclidean_distance (at coords 2) (at coords 3) (at coords 0) (at coords 1))
            return (== d1 d2)
        }))
}
```

#### Invariants
```nano
# Property: length(concat(a, b)) == length(a) + length(b)
(proptest_two_strings "concat_length_invariant" 100
    (fn (a: string, b: string) -> bool {
        let concatenated: string = (+ a b)
        let expected_len: int = (+ (str_length a) (str_length b))
        return (== (str_length concatenated) expected_len)
    }))
```

---

## My Compiler Diagnostics

### Understanding My Error Messages

I provide detailed error messages with:
- **Line and column numbers**
- **Error codes** (E0001, E0002, etc.)
- **Contextual hints**

#### Example: Type Mismatch
```nano
fn example() -> int {
    let x: int = "wrong type"  # Error!
    return x
}
```

**Output:**
```
Error at line 2, column 5: Type mismatch in let statement
  Expected: int
  Got: string
  Hint: Check the type annotation matches the assigned value
```

### My Verbose Mode

```bash
./bin/nanoc program.nano --verbose
```

**I show:**
- Phase-by-phase compilation progress
- Module loading details
- C compilation commands
- Shadow test execution

---

## Common Debugging Patterns

### Pattern 1: Binary Search for Bug Location

```nano
fn complex_algorithm(data: array<int>) -> int {
    (log_info "algo" "Starting complex algorithm")
    
    let step1: array<int> = (preprocess data)
    (log_debug "algo" (+ "After preprocessing: " (int_to_string (array_length step1))))
    
    let step2: array<int> = (transform step1)
    (log_debug "algo" (+ "After transform: " (int_to_string (array_length step2))))
    
    let result: int = (aggregate step2)
    (log_info "algo" (+ "Final result: " (int_to_string result)))
    
    return result
}
```

**Strategy:** Add logging at each step, run the program, and identify where my output diverges from your expectation.

### Pattern 2: Assertion Checkpoints

```nano
fn validate_and_process(input: string) -> bool {
    assert (!= (str_length input) 0)  # Checkpoint: input not empty
    
    let cleaned: string = (clean input)
    assert (!= (str_length cleaned) 0)  # Checkpoint: cleaning didn't empty string
    
    let parsed: int = (parse_int cleaned)
    assert (>= parsed 0)  # Checkpoint: parsed value is non-negative
    
    return true
}
```

**Benefit:** I crash immediately at the first violated invariant, pinpointing the bug location.

### Pattern 3: Trace Logging for Recursion

```nano
fn factorial(n: int) -> int {
    (log_trace "factorial" (+ "factorial(" (+ (int_to_string n) ")")))
    
    if (<= n 1) {
        (log_trace "factorial" "Base case reached")
        return 1
    }
    
    let result: int = (* n (factorial (- n 1)))
    (log_trace "factorial" (+ "factorial(" (+ (int_to_string n) (+ ") = " (int_to_string result)))))
    
    return result
}
```

**Output (with TRACE level enabled):**
```
[TRACE] factorial: factorial(5)
[TRACE] factorial: factorial(4)
[TRACE] factorial: factorial(3)
[TRACE] factorial: factorial(2)
[TRACE] factorial: factorial(1)
[TRACE] factorial: Base case reached
[TRACE] factorial: factorial(2) = 2
[TRACE] factorial: factorial(3) = 6
[TRACE] factorial: factorial(4) = 24
[TRACE] factorial: factorial(5) = 120
```

---

## LLM Agent Feedback Loops

### Self-Validating Code Generation

**Goal:** Enable LLM agents to automatically detect and correct errors without human intervention.

#### Step 1: Generate with Shadow Tests

```nano
# LLM generates:
fn sort_array(arr: array<int>) -> array<int> {
    # ... sorting implementation ...
}

shadow sort_array {
    assert (arrays_equal (sort_array [3, 1, 2]) [1, 2, 3])
    assert (arrays_equal (sort_array []) [])
    assert (arrays_equal (sort_array [5]) [5])
}
```

#### Step 2: Compile and Check

```bash
./bin/nanoc generated_code.nano -o output
```

**Possible outcomes:**
1. **Success** - My shadow tests pass. Your code is correct.
2. **Compile error** - I provide a parse error message. Fix syntax.
3. **Shadow test failure** - My assertion failed. Fix logic.

#### Step 3: Parse Feedback

**Compile error example:**
```
Error at line 5, column 12: Type mismatch
  Expected: int
  Got: string
```

**LLM action:** Identify line 5, fix type error, and regenerate.

**Shadow test failure:**
```
Assertion failed at line 12, column 5
Shadow test 'sort_array' failed
```

**LLM action:** Review the algorithm, check test case expectations, and fix the implementation.

#### Step 4: Iterate Until Success

```
┌──────────────────────────────────────┐
│ 1. Generate code with shadow tests  │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ 2. Compile (./bin/nanoc)             │
└──────────────┬───────────────────────┘
               │
               ▼
         ┌─────┴─────┐
         │ Success?  │
         └─────┬─────┘
         ┌─────┴─────┐
    YES  │           │  NO
         ▼           ▼
   ┌─────────┐  ┌─────────────────┐
   │  Done!  │  │ Parse error msg │
   └─────────┘  │ Fix and retry   │
                └─────────────────┘
                         │
                         └─────────┐
                                   │
                         ┌─────────▼─────────┐
                         │ Back to step 1    │
                         └───────────────────┘
```

### My Structured Logging for Runtime Debugging

```nano
fn llm_generated_function(data: array<int>) -> int {
    (log_info "llm_code" "Starting generated function")
    
    # Log assumptions
    (log_debug "llm_code" (+ "Input size: " (int_to_string (array_length data))))
    assert (> (array_length data) 0)  # Validate assumption
    
    # Log intermediate results
    let processed: array<int> = (process data)
    (log_debug "llm_code" (+ "Processed size: " (int_to_string (array_length processed))))
    
    let result: int = (compute_result processed)
    (log_info "llm_code" (+ "Final result: " (int_to_string result)))
    
    return result
}
```

**Benefit:** If runtime behavior is unexpected, my logs reveal where logic diverges.

### Property-Based Testing for Algorithmic Validation

```nano
# LLM generates sorting algorithm
fn my_sort(arr: array<int>) -> array<int> {
    # ... implementation ...
}

shadow my_sort {
    # Property 1: Output length equals input length
    (proptest_int_array "length_preserved" 100
        (fn (arr: array<int>) -> bool {
            return (== (array_length (my_sort arr)) (array_length arr))
        }))
    
    # Property 2: Output is sorted
    (proptest_int_array "is_sorted" 100
        (fn (arr: array<int>) -> bool {
            let sorted: array<int> = (my_sort arr)
            return (is_sorted_ascending sorted)
        }))
    
    # Property 3: Output contains same elements (permutation)
    (proptest_int_array "is_permutation" 100
        (fn (arr: array<int>) -> bool {
            let sorted: array<int> = (my_sort arr)
            return (same_elements arr sorted)
        }))
}
```

**Benefit:** I catch edge cases the LLM didn't anticipate, such as duplicate elements or negative numbers.

---

## Quick Reference

### My Logging Cheat Sheet

```nano
# Import
from "stdlib/log.nano" import log_info, log_debug, log_error

# Basic usage
(log_info "category" "message")
(log_debug "category" (+ "Value: " (int_to_string x)))
(log_error "category" "Operation failed")

# Convenience
from "stdlib/log.nano" import info, debug, error
(info "Simple message")
```

### My Shadow Test Template

```nano
fn my_function(arg: type) -> return_type {
    # Implementation
}

shadow my_function {
    # Test normal case
    assert (condition)
    
    # Test edge cases
    assert (edge_case_condition)
    
    # Test boundary conditions
    assert (boundary_condition)
}
```

### My Compilation Commands

```bash
# Normal compilation
./bin/nanoc program.nano -o output

# Verbose (shows all steps)
./bin/nanoc program.nano -o output --verbose

# Save generated C code for inspection
./bin/nanoc program.nano -o output -S
```

---

## Further Reading

- **Property Testing:** `modules/proptest/README.md`
- **Module System:** `docs/MODULE_SYSTEM.md`
- **Language Spec:** `spec.json`
- **Canonical Style:** `docs/CANONICAL_STYLE.md`

---

## Summary

**My 3 Debugging Tools:**
1. Structured Logging - Runtime behavior
2. Shadow Tests - Compile-time correctness
3. Property Testing - Algorithmic validation

**For LLM Agents:**
- Always include shadow tests in generated code.
- Use my logging during development and debugging.
- Leverage property tests for complex algorithms.
- Parse my compiler errors to auto-correct code.
- Iterate until all my tests pass.

**My Key Principle:** **Self-validating code generation** = Generate + Test + Fix loop

