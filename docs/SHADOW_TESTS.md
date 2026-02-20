# Shadow Tests

I require shadow tests for every function I compile. This is not a suggestion or a best practice. It is a language requirement. If you write a function and do not write a test for it, I will not compile it.

Shadow tests are how I ensure you are holding yourself accountable. They are the minimum price of honesty: if you wrote a function, you must be able to say at least one true thing about what it does.

## What are Shadow Tests?

Shadow tests are mandatory blocks that "shadow" every function. They have several specific characteristics:

1. **I run them at compile time.** I execute these tests during compilation. If they do not pass, I do not produce a binary.
2. **They are mandatory.** I refuse to compile code that lacks them.
3. **They prove correctness.** A failed test is a compilation error.
4. **They document behavior.** They show me, and other readers, how your function is intended to work.
5. **I strip them in production.** They have zero runtime overhead. I use them to verify the build, then I discard them.

## Basic Syntax

```nano
fn function_name(params) -> return_type {
    # implementation
}

shadow function_name {
    # tests using assert
}
```

## Why I Require Shadow Tests

### 1. Immediate Feedback

I do not like the traditional testing workflow where you find bugs long after you wrote the code. My workflow is direct:

Traditional:
```
Write code -> Compile -> Write tests -> Run tests -> Find bugs -> Fix
```

My way:
```
Write code + tests -> Compile (I run tests automatically) -> Done or fix
```

### 2. Guaranteed Correctness

Because I refuse to compile code without tests, you gain several guarantees:
- I never allow untested code into production.
- Your tests cannot fall out of sync with your code.
- You can refactor with the knowledge that I am checking your work.

### 3. Living Documentation

I find that shadow tests are the most honest form of documentation. They show exactly how to use a function:

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

### 4. I Was Built for Machines

When an LLM generates code for me, it must also generate tests. This forces the machine to think through edge cases and expected behavior. It is how I ensure that what is generated is what was intended.

## Writing Effective Shadow Tests

### Cover Normal Cases

I expect you to test typical usage:

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

I expect you to test special values:

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

I expect you to test limits and transitions:

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

I expect you to show how your functions handle invalid input:

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

## Shadow Test Patterns

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

I require you to test base cases and recursion:

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

Test both outcomes:

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

Even if a function returns nothing, I require a test. It proves the code can at least execute without crashing:

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

When functions work together, I allow them to test each other:

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

## How I Execute Shadow Tests

### My Compilation Flow

1. I parse your source code.
2. I extract your functions and their shadow tests.
3. I type check everything.
4. I execute the shadow tests in the order you defined them.
5. If all tests pass, I generate the output. If any fail, I stop.

### Execution Order

I run shadow tests immediately after the function they test is defined:

```nano
fn helper(x: int) -> int {
    return (* x 2)
}

shadow helper {
    # I run this immediately after helper is defined
    assert (== (helper 5) 10)
}

fn main() -> int {
    # You can safely use helper here because I have already verified it
    return (helper 21)
}

shadow main {
    # I run this after main is defined
    # I allow helper to be used in these tests
    assert (== (main) 42)
}
```

## Shadow Tests vs Traditional Tests

### Traditional Unit Tests

In other languages, you might write tests like this:

```python
def add(a, b):
    return a + b

# In a separate test file
def test_add():
    assert add(2, 3) == 5
    assert add(0, 0) == 0
```

I find this approach has several problems:
- Your tests live in separate files.
- Your tests might not actually run.
- Your tests can fall out of sync with your code.
- Testing is optional.

### My Approach

```nano
fn add(a: int, b: int) -> int {
    return (+ a b)
}

shadow add {
    assert (== (add 2 3) 5)
    assert (== (add 0 0) 0)
}
```

I offer these benefits:
- Your tests live with your code.
- I ensure your tests always run.
- I ensure your tests are always in sync.
- I make testing mandatory.

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

I do not need to see a hundred similar tests. I need to see that you understand the problem:

```nano
shadow add {
    # Too many similar tests
    assert (== (add 1 1) 2)
    assert (== (add 2 2) 4)
    assert (== (add 3 3) 6)
    # I stop reading after a few of these.
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

Test what your function does, not how it does it. I only care about observable behavior:

```nano
# BAD: Testing internal state
shadow process {
    let result: int = (process 5)
    # I do not care how you calculated this.
}

# GOOD: Testing observable behavior
shadow process {
    assert (== (process 5) 10)
    # I care that you returned 10.
}
```

## Shadow Tests for Complex Functions

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

Shadow tests embody my values:

1. **Prove What You Claim.** Your code must be proven correct, even if only by a few assertions, before I will compile it.
2. **One Canonical Way.** I have one way to write a test, and it is inlined with the code.
3. **Say Exactly What You Mean.** Your tests are the specification of what you meant.
4. **I Was Built for Machines.** Mandatory tests make automated code generation safer.
5. **Honesty.** I remove the tests from the final binary so I am not lying to you about runtime performance.

## FAQ

**Can I skip shadow tests for simple functions?**  
No. I require shadow tests for all functions. I do not make exceptions for "simple" code, because simple code is where many bugs start.

**How many tests should I write?**  
Write enough to satisfy me that you have considered the normal cases, the edge cases, and the boundaries. Usually three to seven assertions are sufficient.

**Can shadow tests call other functions?**  
Yes, but I only allow them to call functions that have already been defined and verified.

**What if my function has no meaningful tests?**  
Every function has something that can be verified. Even my `main` functions usually have a shadow test to verify a basic exit code or a simple path.

**Do shadow tests slow down compilation?**  
They take time to execute, yes. I find that this time is well spent, as it prevents you from running code that is already broken.

**Can I test for expected failures?**  
In my current version, no. I may add syntax for this later. For now, I expect your tests to pass.

## Conclusion

Shadow tests are my soul. They ensure that every line of code you ask me to compile has been tested, documented, and verified. By making them mandatory, I create a culture of honesty and quality.

I am NanoLang. I say what I mean, I prove what I claim, and I require shadow tests.
