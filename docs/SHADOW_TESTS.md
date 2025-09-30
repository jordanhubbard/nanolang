# Shadow-Tests: The Heart of nanolang

Shadow-tests are nanolang's unique approach to ensuring code correctness. This document explains what they are, why they exist, and how to write effective shadow-tests.

## What are Shadow-Tests?

Shadow-tests are mandatory test blocks that "shadow" every function in nanolang. They:

1. **Run at compile time** - Execute during compilation, not at runtime
2. **Are mandatory** - Code without shadow-tests won't compile
3. **Prove correctness** - Failed tests = failed compilation
4. **Document behavior** - Show how functions should be used
5. **Are stripped in production** - Zero runtime overhead

## Basic Syntax

```nano
fn function_name(params) -> return_type {
    # implementation
}

shadow function_name {
    # tests using assert
}
```

## Why Shadow-Tests?

### 1. Immediate Feedback

Traditional testing workflow:
```
Write code → Compile → Write tests → Run tests → Find bugs → Fix
```

Shadow-test workflow:
```
Write code + tests → Compile (tests run automatically) → Done or fix
```

### 2. Guaranteed Correctness

Code without tests doesn't compile. This means:
- No untested code in production
- Tests can't fall out of sync with code
- Refactoring is safer

### 3. Living Documentation

Shadow-tests show how to use functions:

```nano
fn divide(a: int, b: int) -> int {
    return (/ a b)
}

shadow divide {
    # Documentation through examples
    assert (== (divide 10 2) 5)
    assert (== (divide 7 2) 3)    # Integer division
    assert (== (divide 0 5) 0)    # Zero numerator OK
    # Note: divide by zero is undefined behavior
}
```

### 4. LLM-Friendly

When LLMs generate code, they must also generate tests, ensuring they think through edge cases and expected behavior.

## Writing Effective Shadow-Tests

### Cover Normal Cases

Test typical usage:

```nano
fn add(a: int, b: int) -> int {
    return (+ a b)
}

shadow add {
    assert (== (add 2 3) 5)
    assert (== (add 10 20) 30)
}
```

### Cover Edge Cases

Test special values:

```nano
fn factorial(n: int) -> int {
    if (<= n 1) {
        return 1
    } else {
        return (* n (factorial (- n 1)))
    }
}

shadow factorial {
    # Edge cases
    assert (== (factorial 0) 1)   # Zero case
    assert (== (factorial 1) 1)   # Base case
    
    # Normal cases
    assert (== (factorial 5) 120)
    assert (== (factorial 10) 3628800)
}
```

### Cover Boundary Conditions

Test limits and transitions:

```nano
fn is_positive(n: int) -> bool {
    return (> n 0)
}

shadow is_positive {
    # Boundaries
    assert (== (is_positive 0) false)     # Zero boundary
    assert (== (is_positive 1) true)      # Just above zero
    assert (== (is_positive -1) false)    # Just below zero
    
    # Normal cases
    assert (== (is_positive 100) true)
    assert (== (is_positive -100) false)
}
```

### Cover Error Conditions

Test how functions handle invalid input:

```nano
fn safe_divide(a: int, b: int) -> int {
    if (== b 0) {
        return 0  # Safe default
    } else {
        return (/ a b)
    }
}

shadow safe_divide {
    # Normal cases
    assert (== (safe_divide 10 2) 5)
    
    # Error condition
    assert (== (safe_divide 10 0) 0)  # Handles divide by zero
}
```

## Shadow-Test Patterns

### Pattern 1: Simple Functions

For simple, pure functions:

```nano
fn double(x: int) -> int {
    return (* x 2)
}

shadow double {
    assert (== (double 0) 0)
    assert (== (double 5) 10)
    assert (== (double -3) -6)
}
```

### Pattern 2: Recursive Functions

Test base cases and recursion:

```nano
fn sum_to_n(n: int) -> int {
    if (<= n 0) {
        return 0
    } else {
        return (+ n (sum_to_n (- n 1)))
    }
}

shadow sum_to_n {
    # Base cases
    assert (== (sum_to_n 0) 0)
    assert (== (sum_to_n -5) 0)
    
    # Recursive cases
    assert (== (sum_to_n 1) 1)
    assert (== (sum_to_n 5) 15)
    assert (== (sum_to_n 10) 55)
}
```

### Pattern 3: Boolean Functions

Test true and false cases:

```nano
fn is_even(n: int) -> bool {
    return (== (% n 2) 0)
}

shadow is_even {
    # True cases
    assert (== (is_even 0) true)
    assert (== (is_even 2) true)
    assert (== (is_even -4) true)
    
    # False cases
    assert (== (is_even 1) false)
    assert (== (is_even 3) false)
    assert (== (is_even -5) false)
}
```

### Pattern 4: Void Functions

For functions that don't return values:

```nano
fn print_greeting(name: string) -> void {
    print "Hello, "
    print name
}

shadow print_greeting {
    # Just verify it doesn't crash
    print_greeting "World"
    print_greeting "Alice"
    print_greeting ""
}
```

### Pattern 5: Multiple Related Functions

When functions work together:

```nano
fn add(a: int, b: int) -> int {
    return (+ a b)
}

shadow add {
    assert (== (add 2 3) 5)
}

fn add_three(a: int, b: int, c: int) -> int {
    return (add (add a b) c)
}

shadow add_three {
    assert (== (add_three 1 2 3) 6)
    # Implicitly tests add() as well
}
```

## How Shadow-Tests Execute

### Compilation Flow

```
1. Parse source code
   ↓
2. Extract functions and their shadow-tests
   ↓
3. Type check everything
   ↓
4. Execute shadow-tests in definition order
   ↓
5. If all pass: generate output
   If any fail: stop with error
```

### Execution Order

Shadow-tests run after their function is defined:

```nano
fn helper(x: int) -> int {
    return (* x 2)
}

shadow helper {
    # Runs immediately after helper is defined
    assert (== (helper 5) 10)
}

fn main() -> int {
    # Can safely use helper here
    return (helper 21)
}

shadow main {
    # Runs after main is defined
    # Can use helper in tests
    assert (== (main) 42)
}
```

## Shadow-Tests vs Traditional Tests

### Traditional Unit Tests

```python
def add(a, b):
    return a + b

# In a separate test file
def test_add():
    assert add(2, 3) == 5
    assert add(0, 0) == 0
```

**Problems:**
- Tests in separate files
- Tests might not run
- Tests can fall out of sync
- Optional testing

### Shadow-Tests

```nano
fn add(a: int, b: int) -> int {
    return (+ a b)
}

shadow add {
    assert (== (add 2 3) 5)
    assert (== (add 0 0) 0)
}
```

**Benefits:**
- Tests with code
- Tests always run
- Tests always in sync
- Mandatory testing

## Best Practices

### DO: Test Edge Cases

```nano
shadow my_function {
    assert (== (my_function 0) 0)      # Zero
    assert (== (my_function -1) -1)    # Negative
    assert (== (my_function 1) 1)      # Positive
}
```

### DO: Keep Tests Clear

```nano
shadow calculate {
    # Clear test cases
    assert (== (calculate 2 3) 6)
    assert (== (calculate 0 5) 0)
}
```

### DON'T: Over-Test

```nano
shadow add {
    # Too many similar tests
    assert (== (add 1 1) 2)
    assert (== (add 2 2) 4)
    assert (== (add 3 3) 6)
    # ... 100 more lines
}
```

### DO: Test Boundaries

```nano
shadow clamp {
    assert (== (clamp -5 0 10) 0)   # Below minimum
    assert (== (clamp 5 0 10) 5)    # In range
    assert (== (clamp 15 0 10) 10)  # Above maximum
}
```

### DON'T: Test Implementation

Test behavior, not implementation:

```nano
# BAD: Testing internal state
shadow process {
    let result: int = (process 5)
    # Don't test "how" it calculated
}

# GOOD: Testing observable behavior
shadow process {
    assert (== (process 5) 10)
    # Test "what" it returns
}
```

## Shadow-Tests for Complex Functions

### State-Changing Functions

```nano
fn increment_counter(counter: int) -> int {
    return (+ counter 1)
}

shadow increment_counter {
    # Test the state transition
    let initial: int = 0
    let after_one: int = (increment_counter initial)
    let after_two: int = (increment_counter after_one)
    
    assert (== after_one 1)
    assert (== after_two 2)
}
```

### Functions with Complex Logic

```nano
fn fizzbuzz(n: int) -> int {
    if (== (% n 15) 0) {
        return 15
    } else {
        if (== (% n 3) 0) {
            return 3
        } else {
            if (== (% n 5) 0) {
                return 5
            } else {
                return n
            }
        }
    }
}

shadow fizzbuzz {
    # Test each branch
    assert (== (fizzbuzz 15) 15)  # Multiple of 15
    assert (== (fizzbuzz 9) 3)    # Multiple of 3 only
    assert (== (fizzbuzz 10) 5)   # Multiple of 5 only
    assert (== (fizzbuzz 7) 7)    # Neither
}
```

## Philosophy

Shadow-tests embody nanolang's philosophy:

1. **Correctness First** - Code must be proven correct to compile
2. **Simplicity** - Tests are part of the code, not separate
3. **Clarity** - Tests document expected behavior
4. **LLM-Friendly** - Forces consideration of test cases
5. **Zero Overhead** - Tests removed from production builds

## FAQ

**Q: Can I skip shadow-tests for simple functions?**  
A: No. All functions must have shadow-tests. This ensures consistency and catches unexpected bugs.

**Q: How many tests should I write?**  
A: Enough to cover normal operation, edge cases, and boundaries. Usually 3-7 assertions.

**Q: Can shadow-tests call other functions?**  
A: Yes, but only functions defined before the test runs.

**Q: What if my function has no meaningful tests?**  
A: Every function has tests. Even `main` should verify it returns the correct exit code.

**Q: Do shadow-tests slow down compilation?**  
A: Slightly, but they catch bugs before runtime, saving overall development time.

**Q: Can I test for expected failures?**  
A: In v0.1, no. Future versions may add expect-error syntax.

## Conclusion

Shadow-tests are not just a feature of nanolang—they're its soul. They ensure that every line of code is tested, documented, and correct. By making tests mandatory and integrated, nanolang creates a culture of quality and confidence.

Write code. Write tests. Ship confidently. 🚀
