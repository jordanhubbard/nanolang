# nanolang Examples

This directory contains example programs written in nanolang to demonstrate the language's features and syntax.

## Examples

### 1. hello.nano - Hello World
The simplest possible nanolang program.

**Demonstrates:**
- Function definition with `fn`
- The `main` entry point
- String printing
- Shadow-tests

### 2. calculator.nano - Basic Arithmetic
A collection of arithmetic functions demonstrating prefix notation.

**Demonstrates:**
- Multiple function definitions
- Prefix notation for all operations
- Integer arithmetic (+, -, *, /, %)
- Conditional logic with `if/else`
- Comprehensive shadow-tests

### 3. factorial.nano - Factorial Calculator
Recursive calculation of factorials.

**Demonstrates:**
- Recursion
- Conditional returns
- Loops with `while`
- Mutable variables with `let mut`
- Variable assignment with `set`

### 4. fibonacci.nano - Fibonacci Sequence
Classic Fibonacci implementation.

**Demonstrates:**
- Recursive function calls
- Multiple function calls in expressions
- `for` loops with `range`
- Testing recursive functions

### 5. primes.nano - Prime Number Checker
Check if numbers are prime and count primes up to a limit.

**Demonstrates:**
- Boolean return types
- Complex control flow
- Loop optimizations
- Multiple related functions
- Edge case testing

## Running Examples

Once the nanolang compiler is implemented, you'll be able to run these examples:

```bash
# Compile an example
nanoc examples/hello.nano -o hello

# Run it
./hello
```

## Understanding Shadow-Tests

Every example includes shadow-tests that demonstrate how to test nanolang code. Notice how:

1. **Every function has a shadow block** - This is mandatory in nanolang
2. **Tests cover edge cases** - Like 0, negative numbers, boundaries
3. **Tests are readable** - They serve as documentation
4. **Tests run at compile time** - Ensuring correctness before execution

## Learning Path

We recommend exploring the examples in this order:

1. `hello.nano` - Start here to understand basic structure
2. `calculator.nano` - Learn prefix notation and basic operations
3. `factorial.nano` - Understand recursion and loops
4. `fibonacci.nano` - Practice with classic algorithms
5. `primes.nano` - See more complex logic and optimizations

## Writing Your Own Programs

When writing nanolang programs, remember:

1. **Prefix notation**: `(+ a b)` not `a + b`
2. **Explicit types**: Every variable needs a type annotation
3. **Shadow-tests**: Every function needs tests
4. **Immutable by default**: Use `mut` only when needed
5. **If requires else**: Both branches are mandatory

## Example Structure

Every nanolang program follows this pattern:

```nano
# 1. Define helper functions with their shadow-tests
fn helper(param: type) -> return_type {
    # implementation
}

shadow helper {
    # tests
}

# 2. Define main function
fn main() -> int {
    # program logic
    return 0
}

shadow main {
    # tests
}
```

## Contributing Examples

If you'd like to contribute more examples, please ensure they:

- Demonstrate a specific feature or pattern
- Include comprehensive shadow-tests
- Are well-commented
- Follow the language specification
- Start simple and build complexity gradually
