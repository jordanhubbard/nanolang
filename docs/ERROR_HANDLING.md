# Error Handling in NanoLang

**Philosophy, patterns, and best practices for robust error handling**

---

## Table of Contents

1. [Philosophy](#philosophy)
2. [Error Handling Mechanisms](#error-handling-mechanisms)
3. [Result Type](#result-type)
4. [Option Type](#option-type)
5. [Assertions](#assertions)
6. [Panic and Abort](#panic-and-abort)
7. [Error Propagation](#error-propagation)
8. [Best Practices](#best-practices)
9. [Common Patterns](#common-patterns)
10. [FFI Error Handling](#ffi-error-handling)
11. [Testing Error Paths](#testing-error-paths)

---

## Philosophy

**Core Principle:** **Explicit error handling with zero-cost abstractions**

NanoLang follows the principle that **errors are part of your program's normal flow**, not exceptional circumstances. This leads to:

1. **Explicit over implicit** - Errors are values, not exceptions
2. **Compile-time safety** - Errors must be handled or explicitly ignored
3. **Zero runtime cost** - Error handling compiles to efficient code
4. **Fail-fast** - Detect errors early, fail loudly
5. **Recoverable vs unrecoverable** - Clear distinction between errors you can handle and bugs

### Why No Exceptions?

NanoLang intentionally **does not** have exception handling (try/catch) because:

❌ **Problems with exceptions:**
- Hidden control flow (functions can exit via throw)
- Performance overhead (stack unwinding)
- Unclear error types (what can this function throw?)
- Easy to ignore (forgot to catch)

✅ **Benefits of Result types:**
- Explicit in function signature (`-> Result<T, E>`)
- Zero runtime overhead (compile-time optimization)
- Must be handled or explicitly ignored
- Compiler catches forgotten error handling

---

## Error Handling Mechanisms

NanoLang provides four mechanisms for different situations:

| Mechanism | Use Case | Example |
|-----------|----------|---------|
| **Result<T, E>** | Recoverable errors | File not found, parsing failed |
| **Option<T>** | Values that may not exist | Array index, hash map lookup |
| **assert** | Invariant checking | Preconditions, postconditions |
| **panic/abort** | Unrecoverable bugs | Out of memory, programmer error |

---

## Result Type

The `Result<T, E>` type represents either success (`Ok(T)`) or failure (`Err(E)`).

### Definition

```nano
union Result<T, E> {
    Ok(T)
    Err(E)
}
```

### Creating Results

```nano
fn divide(a: int, b: int) -> Result<int, string> {
    if (== b 0) {
        return Result::Err("Division by zero")
    } else {
        return Result::Ok((/ a b))
    }
}
```

### Checking Results

```nano
fn main() -> int {
    let result: Result<int, string> = (divide 10 2)

    if (result_is_ok result) {
        let value: int = (result_unwrap result)
        (println (int_to_string value))  # Prints 5
    } else {
        let error: string = (result_unwrap_err result)
        (println error)
    }

    return 0
}
```

### Result Operations

#### `result_is_ok(r: Result<T, E>) -> bool`
Check if Result is Ok variant.

#### `result_is_err(r: Result<T, E>) -> bool`
Check if Result is Err variant.

#### `result_unwrap(r: Result<T, E>) -> T`
Extract Ok value. **Panics if Err!**

```nano
# ⚠️ Unsafe: panics on Err
let value: int = (result_unwrap result)

# ✅ Safe: check first
if (result_is_ok result) {
    let value: int = (result_unwrap result)
}
```

#### `result_unwrap_or(r: Result<T, E>, default: T) -> T`
Extract Ok value or return default.

```nano
# Safe: always returns a value
let value: int = (result_unwrap_or result 0)
```

#### `result_unwrap_err(r: Result<T, E>) -> E`
Extract Err value. **Panics if Ok!**

#### `result_map(r: Result<T, E>, f: fn(T) -> U) -> Result<U, E>`
Transform Ok value, pass through Err.

```nano
fn double(x: int) -> int {
    return (* x 2)
}

let result1: Result<int, string> = Result::Ok(5)
let result2: Result<int, string> = (result_map result1 double)
# result2 is Ok(10)
```

#### `result_and_then(r: Result<T, E>, f: fn(T) -> Result<U, E>) -> Result<U, E>`
Chain operations that can fail.

```nano
fn parse_positive_int(s: string) -> Result<int, string> {
    let n: int = (string_to_int s)
    if (<= n 0) {
        return Result::Err("Not positive")
    } else {
        return Result::Ok(n)
    }
}

fn divide_by_two(n: int) -> Result<int, string> {
    if (== (% n 2) 0) {
        return Result::Ok((/ n 2))
    } else {
        return Result::Err("Not even")
    }
}

# Chain: parse → divide
let result: Result<int, string> =
    (result_and_then (parse_positive_int "10") divide_by_two)
# result is Ok(5)
```

---

## Option Type

The `Option<T>` type represents a value that may or may not exist.

### Definition

```nano
union Option<T> {
    Some(T)
    None
}
```

### Creating Options

```nano
fn safe_array_get(arr: array<int>, index: int) -> Option<int> {
    let len: int = (array_length arr)
    if (or (< index 0) (>= index len)) {
        return Option::None
    } else {
        return Option::Some((at arr index))
    }
}
```

### Using Options

```nano
let arr: array<int> = [1, 2, 3]
let maybe_value: Option<int> = (safe_array_get arr 1)

match maybe_value {
    Option::Some(v) => (println (int_to_string v))
    Option::None => (println "Index out of bounds")
}
```

### Option vs Result

**Use Option when:**
- Value may not exist (not an error)
- No context needed about why it's missing
- Example: `array_get`, `hash_map_get`

**Use Result when:**
- Operation can fail with specific reasons
- Error context is important
- Example: `file_read`, `parse_json`

```nano
# Option: value may not exist (not an error)
fn find_max(arr: array<int>) -> Option<int> {
    if (== (array_length arr) 0) {
        return Option::None  # Empty array has no max
    }
    # ... find and return max
}

# Result: operation can fail with reason
fn parse_config(path: string) -> Result<Config, string> {
    if (not (file_exists path)) {
        return Result::Err("Config file not found")
    }
    # ... parse and return config
}
```

---

## Assertions

Assertions check invariants and preconditions. They should **never fail** in correct code.

### Use Assertions For

1. **Preconditions** - Requirements that must be true before function executes
2. **Postconditions** - Guarantees after function executes
3. **Invariants** - Conditions that always hold
4. **Shadow tests** - Verify function behavior during compilation

### Assertion Examples

```nano
fn sqrt_int(n: int) -> int {
    # Precondition: n must be non-negative
    assert (>= n 0)

    let result: int = (isqrt n)

    # Postcondition: result * result <= n < (result+1) * (result+1)
    assert (<= (* result result) n)
    assert (< n (* (+ result 1) (+ result 1)))

    return result
}

shadow sqrt_int {
    assert (== (sqrt_int 0) 0)
    assert (== (sqrt_int 1) 1)
    assert (== (sqrt_int 4) 2)
    assert (== (sqrt_int 9) 3)
    assert (== (sqrt_int 15) 3)
    assert (== (sqrt_int 16) 4)
}
```

### When NOT to Use Assertions

❌ **Don't use assertions for:**
- Validating user input (use Result instead)
- Handling expected errors (use Result/Option)
- Recoverable errors (use error handling)
- Performance-critical checks in production

✅ **Do use assertions for:**
- Debugging during development
- Shadow tests
- Checking programmer errors
- Documenting assumptions

---

## Panic and Abort

Panics terminate the program immediately. Use for **unrecoverable errors** only.

### When to Panic

✅ **Appropriate panic situations:**
- Out of memory (allocation failed)
- Programmer error (invariant violated)
- Corrupt data structures
- Impossible state reached

❌ **Don't panic for:**
- User input errors
- File not found
- Network timeout
- Any recoverable error

### Example: Appropriate Panic

```nano
fn internal_assert(condition: bool, message: string) -> void {
    if (not condition) {
        (println "FATAL: Internal assertion failed")
        (println message)
        (exit 1)  # Panic: unrecoverable
    }
}

fn buffer_allocate(size: int) -> string {
    if (<= size 0) {
        (internal_assert false "Invalid buffer size")
    }

    let buffer: string = (malloc size)
    if (== buffer "") {
        (internal_assert false "Out of memory")  # Can't recover
    }

    return buffer
}
```

---

## Error Propagation

Propagate errors up the call stack for handling at the appropriate level.

### Pattern 1: Early Return

```nano
fn process_file(path: string) -> Result<void, string> {
    # Check if file exists
    if (not (file_exists path)) {
        return Result::Err("File not found")
    }

    # Read file
    let content: string = (file_read path)
    if (== (str_length content) 0) {
        return Result::Err("File is empty")
    }

    # Process content
    let result: Result<Data, string> = (parse_content content)
    if (result_is_err result) {
        return Result::Err((result_unwrap_err result))
    }

    return Result::Ok(void)
}
```

### Pattern 2: Error Context

Add context as errors propagate up.

```nano
fn load_config() -> Result<Config, string> {
    let result: Result<string, string> = (read_config_file)
    if (result_is_err result) {
        let err: string = (result_unwrap_err result)
        return Result::Err((+ "Failed to load config: " err))
    }

    let content: string = (result_unwrap result)
    let parse_result: Result<Config, string> = (parse_config content)
    if (result_is_err parse_result) {
        let err: string = (result_unwrap_err parse_result)
        return Result::Err((+ "Failed to parse config: " err))
    }

    return parse_result
}
```

### Pattern 3: Error Accumulation

Collect multiple errors before failing.

```nano
fn validate_user(user: User) -> Result<void, array<string>> {
    let mut errors: array<string> = []

    # Check name
    if (== (str_length user.name) 0) {
        (array_push errors "Name is required")
    }

    # Check age
    if (< user.age 0) {
        (array_push errors "Age must be positive")
    }

    # Check email
    if (not (str_contains user.email "@")) {
        (array_push errors "Invalid email")
    }

    # Return all errors or success
    if (> (array_length errors) 0) {
        return Result::Err(errors)
    } else {
        return Result::Ok(void)
    }
}
```

---

## Best Practices

### 1. Return Result for Fallible Operations

✅ **Good:**
```nano
fn parse_int(s: string) -> Result<int, string> {
    let n: int = (string_to_int s)
    if (== n 0) {
        if (!= s "0") {
            return Result::Err("Invalid integer")
        }
    }
    return Result::Ok(n)
}
```

❌ **Bad:**
```nano
fn parse_int(s: string) -> int {
    return (string_to_int s)  # Returns 0 on error, ambiguous!
}
```

### 2. Don't Ignore Errors

✅ **Good:**
```nano
fn save_data(data: string) -> Result<void, string> {
    let result: Result<void, string> = (file_write "data.txt" data)
    if (result_is_err result) {
        (println "Failed to save data")
        return result
    }
    return Result::Ok(void)
}
```

❌ **Bad:**
```nano
fn save_data(data: string) -> void {
    (file_write "data.txt" data)  # Ignores errors!
}
```

### 3. Use Descriptive Error Messages

✅ **Good:**
```nano
return Result::Err("Failed to open file 'config.json': file not found")
```

❌ **Bad:**
```nano
return Result::Err("error")
```

### 4. Handle Errors at the Right Level

```nano
# Low level: return errors
fn read_file(path: string) -> Result<string, string> {
    if (not (file_exists path)) {
        return Result::Err("File not found")
    }
    return Result::Ok((file_read path))
}

# Mid level: add context
fn load_config() -> Result<Config, string> {
    let result: Result<string, string> = (read_file "config.json")
    if (result_is_err result) {
        return Result::Err("Failed to load config")
    }
    # ... parse config
}

# High level: handle and report to user
fn main() -> int {
    let config_result: Result<Config, string> = (load_config)
    if (result_is_err config_result) {
        let err: string = (result_unwrap_err config_result)
        (println (+ "Error: " err))
        return 1
    }
    return 0
}
```

### 5. Document Error Conditions

```nano
# Parses a positive integer from string.
#
# Returns:
# - Ok(n) if string is a valid positive integer
# - Err("Invalid format") if string is not a number
# - Err("Not positive") if number is <= 0
fn parse_positive_int(s: string) -> Result<int, string> {
    let n: int = (string_to_int s)
    if (== n 0) {
        if (!= s "0") {
            return Result::Err("Invalid format")
        }
        return Result::Err("Not positive")
    }
    if (<= n 0) {
        return Result::Err("Not positive")
    }
    return Result::Ok(n)
}
```

---

## Common Patterns

### Pattern: Retry with Exponential Backoff

```nano
fn retry_operation<T, E>(
    operation: fn() -> Result<T, E>,
    max_retries: int
) -> Result<T, E> {
    let mut attempt: int = 0
    let mut delay: int = 100  # ms

    while (< attempt max_retries) {
        let result: Result<T, E> = (operation)
        if (result_is_ok result) {
            return result
        }

        # Exponential backoff
        (sleep delay)
        set delay (* delay 2)
        set attempt (+ attempt 1)
    }

    return Result::Err("Max retries exceeded")
}
```

### Pattern: Graceful Degradation

```nano
fn get_user_data(user_id: int) -> User {
    # Try cache first
    let cache_result: Option<User> = (cache_get user_id)
    match cache_result {
        Option::Some(user) => return user
        Option::None => {}
    }

    # Try database
    let db_result: Result<User, string> = (db_query user_id)
    if (result_is_ok db_result) {
        let user: User = (result_unwrap db_result)
        (cache_set user_id user)  # Update cache
        return user
    }

    # Fallback: return default
    return User { id: user_id, name: "Unknown", email: "" }
}
```

### Pattern: Transaction (All or Nothing)

```nano
fn transfer_money(from: int, to: int, amount: int) -> Result<void, string> {
    # Begin transaction
    let tx: Transaction = (transaction_begin)

    # Debit from account
    let debit_result: Result<void, string> = (account_debit from amount)
    if (result_is_err debit_result) {
        (transaction_rollback tx)
        return debit_result
    }

    # Credit to account
    let credit_result: Result<void, string> = (account_credit to amount)
    if (result_is_err credit_result) {
        (transaction_rollback tx)
        return credit_result
    }

    # Commit transaction
    (transaction_commit tx)
    return Result::Ok(void)
}
```

---

## FFI Error Handling

C functions use different error conventions. Translate to Result/Option.

### Pattern: errno Translation

```nano
extern fn c_open(path: string, flags: int) -> int
extern fn c_errno() -> int
extern fn c_strerror(errno: int) -> string

fn safe_open(path: string, flags: int) -> Result<int, string> {
    let fd: int = (c_open path flags)
    if (< fd 0) {
        let err: int = (c_errno)
        let msg: string = (c_strerror err)
        return Result::Err(msg)
    }
    return Result::Ok(fd)
}
```

### Pattern: NULL Pointer Translation

```nano
extern fn c_fopen(path: string, mode: string) -> int

fn safe_fopen(path: string, mode: string) -> Result<int, string> {
    let handle: int = (c_fopen path mode)
    if (== handle 0) {  # NULL
        return Result::Err("Failed to open file")
    }
    return Result::Ok(handle)
}
```

### Pattern: Status Code Translation

```nano
extern fn c_operation(arg: string) -> int

fn safe_operation(arg: string) -> Result<void, string> {
    let status: int = (c_operation arg)
    if (!= status 0) {
        return Result::Err((status_to_string status))
    }
    return Result::Ok(void)
}
```

---

## Testing Error Paths

Always test both success and failure cases.

### Shadow Test Example

```nano
fn divide(a: int, b: int) -> Result<int, string> {
    if (== b 0) {
        return Result::Err("Division by zero")
    }
    return Result::Ok((/ a b))
}

shadow divide {
    # Test success case
    let ok_result: Result<int, string> = (divide 10 2)
    assert (result_is_ok ok_result)
    assert (== (result_unwrap ok_result) 5)

    # Test error case
    let err_result: Result<int, string> = (divide 10 0)
    assert (result_is_err err_result)
    let err_msg: string = (result_unwrap_err err_result)
    assert (str_contains err_msg "zero")
}
```

### Edge Case Testing

```nano
fn parse_positive_int(s: string) -> Result<int, string> {
    # ... implementation
}

shadow parse_positive_int {
    # Valid cases
    assert (result_is_ok (parse_positive_int "123"))
    assert (result_is_ok (parse_positive_int "1"))

    # Invalid cases
    assert (result_is_err (parse_positive_int ""))
    assert (result_is_err (parse_positive_int "0"))
    assert (result_is_err (parse_positive_int "-5"))
    assert (result_is_err (parse_positive_int "abc"))
    assert (result_is_err (parse_positive_int "12.5"))

    # Edge cases
    assert (result_is_err (parse_positive_int "2147483648"))  # Overflow
}
```

---

## Error Type Design

Design error types that are useful and composable.

### Simple String Errors

✅ **Good for:** Prototypes, simple tools, error messages for humans

```nano
fn parse_config(s: string) -> Result<Config, string> {
    if (== (str_length s) 0) {
        return Result::Err("Config string is empty")
    }
    # ...
}
```

### Enum Error Types

✅ **Good for:** Libraries, APIs, programmatic error handling

```nano
enum ParseError {
    EmptyInput
    InvalidFormat
    UnknownField
    MissingRequired
}

fn parse_config(s: string) -> Result<Config, ParseError> {
    if (== (str_length s) 0) {
        return Result::Err(ParseError::EmptyInput)
    }
    # ...
}
```

### Struct Error Types (with context)

✅ **Good for:** Complex errors, debugging, logging

```nano
struct ParseError {
    kind: ParseErrorKind
    line: int
    column: int
    context: string
}

enum ParseErrorKind {
    UnexpectedToken
    MissingBracket
    InvalidNumber
}

fn parse_expression(s: string) -> Result<Expr, ParseError> {
    # ... parsing logic
    if (unexpected_token) {
        return Result::Err(ParseError {
            kind: ParseErrorKind::UnexpectedToken,
            line: current_line,
            column: current_column,
            context: (str_substring s start end)
        })
    }
}
```

---

## Summary

**Error Handling Checklist:**

- [ ] Use Result<T, E> for fallible operations
- [ ] Use Option<T> for values that may not exist
- [ ] Use assertions for invariants and shadow tests
- [ ] Only panic for unrecoverable errors
- [ ] Propagate errors with context
- [ ] Handle errors at the appropriate level
- [ ] Write descriptive error messages
- [ ] Test both success and error paths
- [ ] Document error conditions
- [ ] Translate FFI errors to Result/Option

**Remember:**
- Errors are values, not exceptions
- Explicit is better than implicit
- Fail-fast, fail-loudly
- Make errors impossible to ignore
- Test error paths as thoroughly as success paths

---

## Related Documentation

- [SPECIFICATION.md](SPECIFICATION.md) - Result and Option types
- [EXTERN_FFI.md](EXTERN_FFI.md) - FFI error handling
- [SHADOW_TESTS.md](SHADOW_TESTS.md) - Testing philosophy
- [STDLIB.md](STDLIB.md) - Standard library error functions

---

**Last Updated:** January 25, 2026
**Philosophy:** Explicit errors, zero-cost abstractions
**Version:** 0.3.0+
