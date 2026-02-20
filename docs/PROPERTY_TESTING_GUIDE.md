# My Property-Based Testing Guide

> **For LLM Agents:** My property-based testing generates random inputs to validate algorithmic properties. It catches edge cases that example-based tests miss.

---

## Table of Contents

1. [Why I Use Property-Based Testing](#why-i-use-property-based-testing)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [API Reference](#api-reference)
5. [Common Properties](#common-properties)
6. [Advanced Patterns](#advanced-patterns)
7. [CI Integration](#ci-integration)

---

## Why I Use Property-Based Testing

### The Problem with Example-Based Tests

```nano
fn reverse_list(lst: array<int>) -> array<int> {
    # ... implementation ...
}

shadow reverse_list {
    # Example-based: Only tests 3 specific cases
    assert (arrays_equal (reverse_list [1, 2, 3]) [3, 2, 1])
    assert (arrays_equal (reverse_list []) [])
    assert (arrays_equal (reverse_list [5]) [5])
}
```

**What is missing?**
- Large arrays (100+ elements)
- Negative numbers
- Duplicates
- Boundary values (INT_MIN, INT_MAX)
- Random patterns

### My Property-Based Solution

```nano
from "modules/proptest/proptest.nano" import proptest_int_array

shadow reverse_list {
    # Property-based: Tests 100 random arrays
    (proptest_int_array "reverse_twice_is_identity" 100
        (fn (lst: array<int>) -> bool {
            let once: array<int> = (reverse_list lst)
            let twice: array<int> = (reverse_list once)
            return (arrays_equal lst twice)
        }))
}
```

**What I test:**
- I automatically generate 100 random arrays.
- I test the property: `reverse(reverse(x)) == x`.
- If I find a failure, I shrink it to the minimal failing case.
- Example output: "Failed on: [0, -1, 0]"

---

## Quick Start

### Installation

```bash
# My property testing module is included with my source
# Located at: modules/proptest/proptest.nano
```

### Your First Property Test

```nano
from "modules/proptest/proptest.nano" import proptest_int

fn absolute_value(x: int) -> int {
    if (< x 0) {
        return (* x -1)
    } else {
        return x
    }
}

shadow absolute_value {
    # Property: abs(x) is always non-negative
    (proptest_int "always_non_negative" 100
        (fn (x: int) -> bool {
            return (>= (absolute_value x) 0)
        }))
    
    # Property: abs(-x) == abs(x)
    (proptest_int "symmetric" 100
        (fn (x: int) -> bool {
            return (== (absolute_value (* x -1)) (absolute_value x))
        }))
}
```

### Running Tests

```bash
# Compile and run (shadow tests execute automatically)
./bin/nanoc your_program.nano -o output
./output
```

**Success output:**
```
✓ Property 'always_non_negative' passed (100 cases)
✓ Property 'symmetric' passed (100 cases)
```

**Failure output:**
```
✗ Property 'always_non_negative' failed
  Counterexample found after 42 tests
  Minimal failing case (after shrinking): x = -2147483648
```

---

## Core Concepts

### 1. Properties vs. Examples

Example-based testing checks specific inputs:
```nano
assert (== (add 2 3) 5)  # True for 2 and 3, but what about other inputs?
```

Property-based testing checks universal properties:
```nano
(proptest_int_pair "commutative" 100
    (fn (a: int, b: int) -> bool {
        return (== (add a b) (add b a))  # True for ALL integers
    }))
```

### 2. Generators

My generators produce random test inputs:

| Generator | Input Type | Use Case |
|-----------|------------|----------|
| `proptest_int` | `int` | Single integers |
| `proptest_int_pair` | `(int, int)` | Pairs of integers |
| `proptest_int_array` | `array<int>` | Integer arrays |
| `proptest_string` | `string` | Strings (future) |

### 3. Properties

A property is a function that returns `bool`:
- `true` means the test passes for this input.
- `false` means the test fails and a counterexample was found.

```nano
# Property function signature
fn my_property(input: InputType) -> bool {
    # Test the property, return true/false
}
```

### 4. Shrinking

When a test fails, I use shrinking to find the minimal failing case:

```
Original failing input: [17, -99, 42, 5, 23, -8, 0, 1, 34]
After shrinking:        [0, -8]
```

This makes it easier for you to debug with a minimal reproduction case.

---

## API Reference

### proptest_int

I test properties with random integers.

```nano
from "modules/proptest/proptest.nano" import proptest_int

(proptest_int "property_name" num_tests property_function)
```

**Parameters:**
- `property_name`: String. A descriptive name for the test.
- `num_tests`: Int. The number of random inputs I should generate. I typically use 50 to 100.
- `property_function`: `fn(int) -> bool`. The function that tests the property.

**Example:**
```nano
shadow is_even {
    (proptest_int "double_is_even" 100
        (fn (x: int) -> bool {
            let doubled: int = (* x 2)
            return (== (% doubled 2) 0)  # Always true
        }))
}
```

### proptest_int_pair

I test properties with pairs of random integers.

```nano
from "modules/proptest/proptest.nano" import proptest_int_pair

(proptest_int_pair "property_name" num_tests property_function)
```

**Parameters:**
- `property_function`: `fn(int, int) -> bool`.

**Example:**
```nano
shadow add {
    # Property: Addition is commutative
    (proptest_int_pair "commutative" 100
        (fn (a: int, b: int) -> bool {
            return (== (add a b) (add b a))
        }))
    
    # Property: Addition is associative
    # Note: Requires proptest_int_triple (not yet implemented)
}
```

### proptest_int_array

I test properties with random integer arrays.

```nano
from "modules/proptest/proptest.nano" import proptest_int_array

(proptest_int_array "property_name" num_tests property_function)
```

**Parameters:**
- `property_function`: `fn(array<int>) -> bool`.

**Example:**
```nano
shadow sort_array {
    # Property: Sorted arrays have same length as input
    (proptest_int_array "length_preserved" 100
        (fn (arr: array<int>) -> bool {
            let sorted: array<int> = (sort_array arr)
            return (== (array_length arr) (array_length sorted))
        }))
    
    # Property: Sorted arrays are actually sorted
    (proptest_int_array "is_sorted" 100
        (fn (arr: array<int>) -> bool {
            let sorted: array<int> = (sort_array arr)
            return (is_ascending_order sorted)
        }))
}
```

### Custom Properties with prop_pass/prop_fail

For more control, you can use explicit pass or fail results:

```nano
from "modules/proptest/proptest.nano" import prop_pass, prop_fail

fn my_property(x: int) -> string {
    if (>= x 0) {
        return (prop_pass)
    } else {
        return (prop_fail "Expected non-negative, got negative")
    }
}
```

---

## Common Properties

### Mathematical Properties

#### Commutativity
```nano
# Property: f(a, b) == f(b, a)
(proptest_int_pair "add_commutative" 100
    (fn (a: int, b: int) -> bool {
        return (== (add a b) (add b a))
    }))
```

#### Associativity
```nano
# Property: f(f(a, b), c) == f(a, f(b, c))
# Requires custom generator for triples
```

#### Identity Element
```nano
# Property: f(x, identity) == x
(proptest_int "add_zero_identity" 100
    (fn (x: int) -> bool {
        return (== (add x 0) x)
    }))
```

#### Inverse Element
```nano
# Property: f(x, inverse(x)) == identity
(proptest_int "add_inverse" 100
    (fn (x: int) -> bool {
        return (== (add x (* x -1)) 0)
    }))
```

### Algorithmic Properties

#### Idempotence
```nano
# Property: f(f(x)) == f(x)
(proptest_string "normalize_idempotent" 100
    (fn (s: string) -> bool {
        let once: string = (normalize s)
        let twice: string = (normalize once)
        return (str_equals once twice)
    }))
```

#### Inversion
```nano
# Property: inverse(f(x)) == x
(proptest_int_array "sort_reverse" 100
    (fn (arr: array<int>) -> bool {
        let sorted: array<int> = (sort_ascending arr)
        let reversed: array<int> = (reverse sorted)
        let desc_sorted: array<int> = (sort_descending arr)
        return (arrays_equal reversed desc_sorted)
    }))
```

#### Equivalence
```nano
# Property: Two implementations produce same result
(proptest_int_array "bubble_vs_quick_sort" 100
    (fn (arr: array<int>) -> bool {
        let bubble: array<int> = (bubble_sort arr)
        let quick: array<int> = (quick_sort arr)
        return (arrays_equal bubble quick)
    }))
```

### Data Structure Properties

#### Length Preservation
```nano
(proptest_int_array "map_preserves_length" 100
    (fn (arr: array<int>) -> bool {
        let mapped: array<int> = (map double arr)
        return (== (array_length arr) (array_length mapped))
    }))
```

#### Element Preservation
```nano
(proptest_int_array "filter_removes_only" 100
    (fn (arr: array<int>) -> bool {
        let filtered: array<int> = (filter is_even arr)
        # Every element in filtered must be in original
        return (all_elements_in filtered arr)
    }))
```

#### Ordering Properties
```nano
(proptest_int_array "sort_is_ascending" 100
    (fn (arr: array<int>) -> bool {
        let sorted: array<int> = (sort arr)
        let mut i: int = 0
        while (< i (- (array_length sorted) 1)) {
            if (> (at sorted i) (at sorted (+ i 1))) {
                return false
            }
            set i (+ i 1)
        }
        return true
    }))
```

---

## Advanced Patterns

### Conditional Properties

Sometimes properties only hold under certain conditions:

```nano
from "modules/proptest/proptest.nano" import prop_discard

(proptest_int_pair "divide_inverse" 100
    (fn (a: int, b: int) -> bool {
        # I discard cases where b == 0 because division is undefined
        if (== b 0) {
            return (prop_discard "Division by zero")
        }
        
        let quotient: int = (/ a b)
        let reconstructed: int = (* quotient b)
        # Property: (a / b) * b ≈ a (within integer division rounding)
        return (<= (absolute_value (- a reconstructed)) (absolute_value b))
    }))
```

**Output:**
```
✓ Property 'divide_inverse' passed (87 cases, 13 discarded)
```

### Stateful Properties

I test sequences of operations to ensure they maintain invariants:

```nano
struct Stack {
    items: array<int>,
    size: int
}

fn push(stack: Stack, value: int) -> Stack {
    # ... implementation ...
}

fn pop(stack: Stack) -> Stack {
    # ... implementation ...
}

shadow stack_operations {
    (proptest_int "push_pop_identity" 100
        (fn (value: int) -> bool {
            let empty: Stack = Stack { items: [], size: 0 }
            let after_push: Stack = (push empty value)
            let after_pop: Stack = (pop after_push)
            
            # Property: push then pop returns to original state
            return (== after_pop.size empty.size)
        }))
}
```

### Oracle Testing

I compare my results against a known-correct implementation:

```nano
fn fast_fibonacci(n: int) -> int {
    # Optimized O(log n) implementation
}

fn slow_fibonacci(n: int) -> int {
    # Simple O(2^n) recursive implementation (oracle)
}

shadow fast_fibonacci {
    (proptest_int "matches_oracle" 50
        (fn (n: int) -> bool {
            # I only test small n for the slow oracle
            if (> n 30) { return true }  # Skip large inputs
            
            return (== (fast_fibonacci n) (slow_fibonacci n))
        }))
}
```

---

## CI Integration

### Basic CI Pipeline

```yaml
# .github/workflows/test.yml
name: Property Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Build my compiler
        run: make
      
      - name: Run property tests
        run: |
          ./bin/nanoc tests/test_properties.nano -o test_runner
          ./test_runner
```

### Test Organization

```
tests/
├── test_sorting_properties.nano      # Sorting algorithm properties
├── test_math_properties.nano         # Mathematical function properties
├── test_string_properties.nano       # String manipulation properties
└── test_data_structure_properties.nano  # DS invariant properties
```

### Test Configuration

```nano
# tests/test_config.nano

# I adjust my test count based on the environment
let is_ci: bool = (not (str_equals (getenv "CI") ""))
let test_count: int = (cond
    (is_ci 1000)  # I run more tests in CI
    (else 100)    # I run fewer tests locally for speed
)

shadow my_function {
    (proptest_int "property" test_count property_function)
}
```

### Handling Flaky Tests

If property tests occasionally fail due to rare edge cases:

```nano
# I document my known limitations
shadow sort_with_overflow {
    # Property: Sorting preserves all elements
    # Note: I may fail on extremely large arrays (>10000 elements)
    # due to stack overflow in recursive quicksort
    (proptest_int_array "elements_preserved" 100
        (fn (arr: array<int>) -> bool {
            if (> (array_length arr) 10000) { return true }  # Skip
            
            let sorted: array<int> = (sort arr)
            return (same_elements arr sorted)
        }))
}
```

### Reproducible Test Failures

When a property test fails in CI:

1. **Capture the seed.** If my module supports it, I will report the seed.
   ```
   ✗ Property 'my_test' failed (seed: 0x12345678)
   ```

2. **Reproduce locally:**
   ```bash
   PROPTEST_SEED=0x12345678 ./test_runner
   ```

3. **Fix the bug.** I use the counterexample to reveal the error.

---

## Best Practices

### Start Simple

```nano
# I begin with basic properties
shadow add {
    (proptest_int_pair "commutative" 100
        (fn (a: int, b: int) -> bool {
            return (== (add a b) (add b a))
        }))
}

# Then I add more complex properties
shadow add {
    (proptest_int "identity" 100
        (fn (x: int) -> bool {
            return (== (add x 0) x)
        }))
}
```

### Name Properties Clearly

```nano
# GOOD: This describes exactly what I am testing
(proptest_int_array "sort_preserves_length" 100 ...)
(proptest_int "abs_always_non_negative" 100 ...)

# BAD: These names are vague
(proptest_int_array "test1" 100 ...)
(proptest_int "property" 100 ...)
```

### Test Properties, Not Implementation

```nano
# GOOD: I test observable behavior
(proptest_int_array "sorted_output_is_ascending" 100 ...)

# BAD: This tests implementation details
(proptest_int_array "uses_quicksort_partition" 100 ...)
```

### Do Not Test Non-Deterministic Functions

```nano
# BAD: Random functions do not have consistent properties
fn get_random() -> int {
    return (random_int)
}

shadow get_random {
    # This will fail
    (proptest_int "always_returns_same" 100
        (fn (_: int) -> bool {
            return (== (get_random) (get_random))
        }))
}
```

### Do Not Duplicate the Implementation

```nano
# BAD: The property is just reimplementing my function
shadow add {
    (proptest_int_pair "works" 100
        (fn (a: int, b: int) -> bool {
            return (== (add a b) (+ a b))  # This is a tautology
        }))
}

# GOOD: I test mathematical properties
shadow add {
    (proptest_int_pair "commutative" 100
        (fn (a: int, b: int) -> bool {
            return (== (add a b) (add b a))
        }))
}
```

---

## Troubleshooting

### "Property failed but I cannot reproduce it"

**Cause:** Shrinking might have found a very specific edge case.

**Solution:** Pay attention to the counterexample I provide:
```
✗ Counterexample: arr = [0, -2147483648, 1]
```

Test this specific input manually:
```nano
shadow my_function {
    # Add an explicit test for the discovered edge case
    let edge_case: array<int> = [0, -2147483648, 1]
    assert (my_property edge_case)
}
```

### "Tests pass locally but fail in CI"

**Cause:** Different random seeds or timing.

**Solution:** Use a deterministic test count:
```nano
let CI_TEST_COUNT: int = 1000
let LOCAL_TEST_COUNT: int = 100

shadow my_function {
    (proptest_int "property" CI_TEST_COUNT property_fn)
}
```

### "Property test is too slow"

**Cause:** You are using too many test cases or an expensive property function.

**Solution:** Reduce the test count or optimize the property:
```nano
# Before: 1000 tests, which is slow
(proptest_int_array "expensive_check" 1000 slow_property)

# After: 100 tests, which is optimized
(proptest_int_array "fast_check" 100 optimized_property)
```

---

## Quick Reference

### Import

```nano
from "modules/proptest/proptest.nano" import 
    proptest_int, proptest_int_pair, proptest_int_array,
    prop_pass, prop_fail, prop_discard
```

### Basic Pattern

```nano
shadow my_function {
    (proptest_TYPE "property_name" num_tests
        (fn (input: TYPE) -> bool {
            # Test the property and return true or false
        }))
}
```

### Property Types Checklist

- [ ] Commutativity: `f(a,b) == f(b,a)`
- [ ] Associativity: `f(f(a,b),c) == f(a,f(b,c))`
- [ ] Identity: `f(x, id) == x`
- [ ] Inverse: `f(x, inv(x)) == id`
- [ ] Idempotence: `f(f(x)) == f(x)`
- [ ] Length preservation: `len(f(x)) == len(x)`
- [ ] Ordering: `is_sorted(sort(x))`

---

## Summary

**My property-based testing:**
- I generate random inputs automatically.
- I find edge cases you did not think of.
- I shrink to minimal failing examples.
- I validate algorithmic properties universally.

**Use me when:**
- You are testing algorithms like sorting, searching, or math.
- You are validating invariants in data structures or protocols.
- You are comparing implementations.

**Avoid me when:**
- You are testing I/O or side effects.
- You have non-deterministic functions.
- You are testing UI or visual output.

**For LLM agents:** Generate property tests alongside shadow tests. This catches edge cases and validates algorithmic correctness automatically.
