# Result<T, E> Type Design Specification
## Structured Error Handling for NanoLang

**Status**: Draft Design  
**Version**: 1.0  
**Date**: 2025-12-16  
**Author**: Language Design Team  
**Priority**: P1 - Critical for Production Readiness

---

## Executive Summary

This document specifies the design and implementation of `Result<T, E>`, a
generic enum type for structured error handling in NanoLang. This feature
addresses the current error handling gap (rated D in language review) and
enables production-ready software development.

**Key Decisions**:
- Rust-style `Result<T, E>` enum (not exception-based)
- Zero-cost abstraction (compile-time only)
- Pattern matching integration for ergonomic error handling
- Gradual stdlib migration path
- Backward compatible with existing code

---

## 1. Motivation & Problem Statement

### 1.1 Current Error Handling Landscape

**Status Quo**: NanoLang currently lacks structured error handling:

```nano
# Current approach: sentinel values and assumptions
fn read_file(path: string) -> string {
    # What if file doesn't exist?
    # What if permission denied?
    # What if disk error?
    return "content"  # Assumes success
}

# Current approach: return codes
fn open_file(path: string) -> int {
    # Returns file descriptor or -1 on error
    # Loses error information (why did it fail?)
    return -1
}
```

**Problems**:
1. ❌ **Silent failures**: Functions assume success
2. ❌ **Lost error information**: Sentinel values don't convey why
3. ❌ **Unclear contracts**: No type-level indication of fallibility
4. ❌ **Easy to ignore**: Developers forget to check return codes
5. ❌ **No composition**: Can't chain fallible operations elegantly

### 1.2 Why Not Exceptions?

**Exceptions considered but rejected:**

```nano
# Exception-based approach (NOT chosen)
try {
    let content = (read_file "config.json")
    let parsed = (parse_json content)
} catch (e: IOException) {
    (println "I/O error")
} catch (e: ParseError) {
    (println "Parse error")
}
```

**Reasons against exceptions**:
1. ❌ Hidden control flow (functions can throw without declaration)
2. ❌ Runtime overhead (stack unwinding, exception tables)
3. ❌ Unclear from types which functions throw
4. ❌ Difficult to enforce exhaustive error handling
5. ❌ Poor fit for systems programming (unpredictable performance)

### 1.3 Why Result<T, E>?

**Advantages**:
1. ✅ Errors are values (explicit in function signatures)
2. ✅ Zero runtime cost (compile-time enum dispatch)
3. ✅ Compiler-enforced handling (can't ignore errors)
4. ✅ Composable (map, and_then, or_else)
5. ✅ Clear semantics (either Ok or Err, never both)

**Proven approach**: Rust, Haskell (Either), Swift, OCaml (result)

---

## 2. Core Design

### 2.1 Type Definition

```nano
# Built-in generic enum type
enum Result<T, E> {
    Ok(T)    # Success variant containing value
    Err(E)   # Error variant containing error
}
```

**Key properties**:
- Generic over success type `T` and error type `E`
- Variants are constructors (not values)
- Must be pattern matched to extract value
- Cannot access value without handling error case

### 2.2 Basic Usage

```nano
# Example: Fallible file reading
enum IOError {
    FileNotFound
    PermissionDenied
    DiskFull
    Unknown(string)
}

fn read_file(path: string) -> Result<string, IOError> {
    if (not (file_exists path)) {
        return Err(IOError::FileNotFound)
    }
    
    if (not (has_read_permission path)) {
        return Err(IOError::PermissionDenied)
    }
    
    # Actual file reading...
    let content: string = (do_read path)
    return Ok(content)
}

# Usage with pattern matching
fn main() -> int {
    match (read_file "/etc/config.txt") {
        Ok(content) => {
            (println "File contents:")
            (println content)
        }
        Err(IOError::FileNotFound) => {
            (println "Error: File not found")
        }
        Err(IOError::PermissionDenied) => {
            (println "Error: Permission denied")
        }
        Err(e) => {
            (println "Error: Unknown")
        }
    }
    return 0
}
```

### 2.3 Type Signatures

```nano
# Function returning Result
fn divide(a: int, b: int) -> Result<int, string> {
    if (== b 0) {
        return Err("division by zero")
    }
    return Ok((/ a b))
}

# Function that can't fail
fn add(a: int, b: int) -> int {
    return (+ a b)
}

# Function with multiple error types
enum MathError {
    DivisionByZero
    Overflow
    NegativeSquareRoot
}

fn safe_divide(a: int, b: int) -> Result<int, MathError> {
    if (== b 0) {
        return Err(MathError::DivisionByZero)
    }
    
    let result: int = (/ a b)
    if (> result 2147483647) {  # INT_MAX
        return Err(MathError::Overflow)
    }
    
    return Ok(result)
}
```

---

## 3. Pattern Matching Integration

### 3.1 Match Expression (Primary Method)

```nano
# Exhaustive pattern matching
let result: Result<int, string> = (divide 10 2)

match result {
    Ok(value) => (println value)
    Err(msg) => (println msg)
}

# Nested matching
match (read_file "data.json") {
    Ok(json_str) => {
        match (parse_json json_str) {
            Ok(data) => (process data)
            Err(parse_err) => (println "Parse error")
        }
    }
    Err(io_err) => (println "I/O error")
}
```

### 3.2 Helper Methods

```nano
# unwrap() - panics on error (use in tests only)
let value: int = (result_unwrap (divide 10 2))

# unwrap_or() - provides default value
let value: int = (result_unwrap_or (divide 10 0) 0)

# is_ok() and is_err() - boolean checks
if (result_is_ok result) {
    (println "Success!")
}

# map() - transform Ok value
let doubled: Result<int, string> = (result_map (divide 10 2) 
    (lambda (x: int) -> int { return (* x 2) }))

# and_then() - chain operations
let result: Result<int, string> = 
    (result_and_then (divide 10 2)
        (lambda (x: int) -> Result<int, string> { 
            return (divide x 2)
        }))

# or_else() - provide alternative on error
let result: Result<int, string> =
    (result_or_else (divide 10 0)
        (lambda (err: string) -> Result<int, string> {
            return Ok(0)
        }))
```

---

## 4. Standard Library Migration

### 4.1 Phased Migration Strategy

**Phase 1: New Functions (Immediate)**
- All new stdlib functions return Result where fallible
- Examples: `fs::read_file`, `net::connect`, `json::parse`

**Phase 2: Parallel Versions (6 months)**
- Keep existing unsafe functions (deprecated)
- Add safe versions with `_safe` suffix
- Example: `nl_file_read()` → `nl_file_read_safe()`

**Phase 3: Breaking Changes (12 months)**
- Remove deprecated unsafe functions
- Rename `_safe` versions to original names
- Major version bump (v2.0.0)

### 4.2 Example Migrations

**Before**:
```nano
# Unsafe - assumes success
fn nl_file_read(path: string) -> string {
    # Returns empty string on error (no way to know why)
    return ""
}
```

**After**:
```nano
# Safe - explicit error handling
fn nl_file_read(path: string) -> Result<string, IOError> {
    if (not (file_exists path)) {
        return Err(IOError::FileNotFound)
    }
    
    # ... actual reading ...
    return Ok(content)
}

# Deprecated version for compatibility
fn nl_file_read_unsafe(path: string) -> string {
    match (nl_file_read path) {
        Ok(content) => return content
        Err(_) => return ""
    }
}
```

### 4.3 Affected Modules

**High Priority** (Return Result):
1. **File I/O**: `nl_file_read`, `nl_file_write`, `nl_file_open`
2. **Network**: `nl_socket_connect`, `nl_http_get`
3. **Parsing**: `nl_parse_int`, `nl_parse_json`
4. **System**: `nl_env_get`, `nl_spawn_process`

**Medium Priority** (Consider Result):
1. **Collections**: `array_get`, `map_get`
2. **Strings**: `string_to_int`, `string_split`
3. **Math**: `sqrt` (negative input), `log` (zero/negative)

**Low Priority** (Keep as-is):
1. Pure math functions (no failure cases)
2. Constructors (use Option if needed)
3. Utility functions (trivial operations)

---

## 5. Compiler & Transpiler Implementation

### 5.1 Compiler Changes

**Parser Changes**:
```c
// src/parser.c
typedef enum {
    // ... existing node types ...
    PARSE_NODE_RESULT_OK,     // Ok(value)
    PARSE_NODE_RESULT_ERR,    // Err(error)
    PARSE_NODE_RESULT_TYPE    // Result<T, E>
} ParseNodeType;

// Parse Result type syntax
ParseNode* parse_result_type(Parser* parser) {
    // Result<T, E>
    expect(parser, TOKEN_RESULT);
    expect(parser, TOKEN_LESS);
    ParseNode* ok_type = parse_type(parser);
    expect(parser, TOKEN_COMMA);
    ParseNode* err_type = parse_type(parser);
    expect(parser, TOKEN_GREATER);
    return create_result_type_node(ok_type, err_type);
}
```

**Type Checker Changes**:
```c
// src/type_checker.c
TypeInfo* check_result_match(MatchExpr* match_expr, Scope* scope) {
    TypeInfo* scrutinee_type = infer_type(match_expr->scrutinee, scope);
    
    if (!is_result_type(scrutinee_type)) {
        error("Match on non-Result type");
    }
    
    // Ensure Ok and Err patterns are present
    bool has_ok = false;
    bool has_err = false;
    
    for (int i = 0; i < match_expr->num_arms; i++) {
        Pattern* pattern = match_expr->arms[i].pattern;
        if (is_ok_pattern(pattern)) has_ok = true;
        if (is_err_pattern(pattern)) has_err = true;
    }
    
    if (!has_ok || !has_err) {
        error("Non-exhaustive Result match");
    }
    
    return check_match_arms(match_expr, scope);
}
```

### 5.2 Transpiler Implementation

**C Code Generation**:
```c
// Result<T, E> transpiles to tagged union
typedef struct {
    enum { OK_VARIANT, ERR_VARIANT } tag;
    union {
        int ok_value;      // T
        char* err_value;   // E
    } data;
} Result_int_string;

// Ok constructor
Result_int_string Ok_int(int value) {
    Result_int_string result;
    result.tag = OK_VARIANT;
    result.data.ok_value = value;
    return result;
}

// Err constructor
Result_int_string Err_string(char* error) {
    Result_int_string result;
    result.tag = ERR_VARIANT;
    result.data.err_value = error;
    return result;
}

// Match expression transpiles to switch
Result_int_string r = divide(10, 2);
switch (r.tag) {
    case OK_VARIANT:
        printf("%d\n", r.data.ok_value);
        break;
    case ERR_VARIANT:
        printf("Error: %s\n", r.data.err_value);
        break;
}
```

**Optimization**: For small T and E types, use stack allocation. For large
types, use heap allocation and reference counting.

---

## 6. Error Type Design Guidelines

### 6.1 Error Type Hierarchy

```nano
# Generic error trait (future feature)
trait Error {
    fn message(self) -> string
    fn code(self) -> int
}

# Domain-specific error enums
enum IOError {
    FileNotFound
    PermissionDenied
    DiskFull
    NetworkTimeout
    ConnectionRefused
    Unknown(string)
}

enum ParseError {
    InvalidSyntax(int, int)  # line, column
    UnexpectedEOF
    InvalidCharacter(char)
}

enum ValidationError {
    TooShort(int)        # minimum length
    TooLong(int)         # maximum length
    InvalidFormat(string)
}
```

### 6.2 Error Composition

```nano
# Option 1: Enum composition
enum AppError {
    IO(IOError)
    Parse(ParseError)
    Validation(ValidationError)
}

fn process_file(path: string) -> Result<Data, AppError> {
    let content: string = match (read_file path) {
        Ok(c) => c
        Err(e) => return Err(AppError::IO(e))
    }
    
    let parsed: JsonValue = match (parse_json content) {
        Ok(p) => p
        Err(e) => return Err(AppError::Parse(e))
    }
    
    return Ok(parsed)
}

# Option 2: Question mark operator (future syntax sugar)
fn process_file(path: string) -> Result<Data, AppError> {
    let content: string = (read_file path)?  # Auto-convert and early return
    let parsed: JsonValue = (parse_json content)?
    return Ok(parsed)
}
```

### 6.3 Error Conversion

```nano
# Explicit conversion between error types
fn convert_io_error(e: IOError) -> AppError {
    return AppError::IO(e)
}

# Implicit conversion via From trait (future feature)
impl From<IOError> for AppError {
    fn from(e: IOError) -> AppError {
        return AppError::IO(e)
    }
}
```

---

## 7. Examples & Common Patterns

### 7.1 Parsing with Error Recovery

```nano
fn parse_config(text: string) -> Result<Config, ParseError> {
    let lines: array<string> = (string_split text "\n")
    let mut config: Config = Config::new()
    
    let mut i: int = 0
    while (< i (array_length lines)) {
        let line: string = (at lines i)
        
        match (parse_config_line line) {
            Ok(entry) => {
                (config_add_entry config entry)
            }
            Err(e) => {
                return Err(ParseError::InvalidSyntax(i, 0))
            }
        }
        
        (set i (+ i 1))
    }
    
    return Ok(config)
}
```

### 7.2 Retry Logic

```nano
fn fetch_with_retry(url: string, max_attempts: int) 
    -> Result<string, NetworkError> {
    let mut attempts: int = 0
    
    while (< attempts max_attempts) {
        match (http_get url) {
            Ok(response) => return Ok(response)
            Err(NetworkError::Timeout) => {
                (println "Timeout, retrying...")
                (set attempts (+ attempts 1))
            }
            Err(e) => return Err(e)  # Don't retry other errors
        }
    }
    
    return Err(NetworkError::MaxRetriesExceeded)
}
```

### 7.3 Validation Pipeline

```nano
fn validate_email(email: string) -> Result<string, ValidationError> {
    if (< (string_length email) 5) {
        return Err(ValidationError::TooShort(5))
    }
    
    if (not (string_contains email "@")) {
        return Err(ValidationError::InvalidFormat("Missing @"))
    }
    
    return Ok(email)
}

fn create_user(email: string, password: string) 
    -> Result<User, ValidationError> {
    let valid_email: string = match (validate_email email) {
        Ok(e) => e
        Err(err) => return Err(err)
    }
    
    let valid_password: string = match (validate_password password) {
        Ok(p) => p
        Err(err) => return Err(err)
    }
    
    return Ok(User::new(valid_email, valid_password))
}
```

---

## 8. Implementation Roadmap

### Phase 1: Core Implementation (2-4 weeks)

**Week 1-2: Compiler & Parser**
- [ ] Add Result<T, E> type to type system
- [ ] Implement Ok/Err constructors
- [ ] Add pattern matching for Result
- [ ] Exhaustiveness checking for Result matches
- [ ] Unit tests for type checker

**Week 3-4: Transpiler & Codegen**
- [ ] C code generation for Result types
- [ ] Generate efficient tagged unions
- [ ] Optimize for small types (stack allocation)
- [ ] Integration tests with examples

### Phase 2: Standard Library (4-6 weeks)

**Week 1-2: Core Modules**
- [ ] Migrate file I/O to Result
- [ ] Add IOError enum
- [ ] Backward compatibility shims
- [ ] Update documentation

**Week 3-4: Extended Modules**
- [ ] Network operations
- [ ] Parsing functions
- [ ] System operations
- [ ] Math functions (where applicable)

**Week 5-6: Helper Functions**
- [ ] result_map, result_and_then, result_or_else
- [ ] result_unwrap, result_unwrap_or
- [ ] result_is_ok, result_is_err
- [ ] Comprehensive stdlib tests

### Phase 3: Developer Experience (2-3 weeks)

**Week 1: Documentation**
- [ ] Language guide chapter on Result
- [ ] Migration guide for existing code
- [ ] Error handling best practices
- [ ] Real-world examples

**Week 2: Tooling**
- [ ] Syntax highlighting for Result
- [ ] LSP autocomplete for Result methods
- [ ] Error message improvements

**Week 3: Examples & Testing**
- [ ] Update all examples to use Result
- [ ] Add Result usage examples
- [ ] Integration test suite
- [ ] Performance benchmarks

### Phase 4: Advanced Features (4-6 weeks, optional)

**Future Enhancements**:
- [ ] Question mark operator (?)
- [ ] Error trait system
- [ ] Automatic error conversion (From trait)
- [ ] Result combinators (map_err, flatten, etc.)
- [ ] async Result<T, E> integration

---

## 9. Testing Strategy

### 9.1 Unit Tests

```nano
# tests/result_basic.nano
shadow test_result_ok {
    let result: Result<int, string> = Ok(42)
    assert (result_is_ok result)
    assert (not (result_is_err result))
    assert (== (result_unwrap result) 42)
}

shadow test_result_err {
    let result: Result<int, string> = Err("failed")
    assert (result_is_err result)
    assert (not (result_is_ok result))
}

shadow test_result_map {
    let result: Result<int, string> = Ok(5)
    let doubled: Result<int, string> = (result_map result
        (lambda (x: int) -> int { return (* x 2) }))
    assert (== (result_unwrap doubled) 10)
}
```

### 9.2 Integration Tests

```nano
# tests/result_file_io.nano
shadow test_file_read_success {
    # Create test file
    (nl_file_write "test.txt" "hello")
    
    match (nl_file_read "test.txt") {
        Ok(content) => assert (== content "hello")
        Err(_) => assert false  # Should not error
    }
    
    (nl_file_delete "test.txt")
}

shadow test_file_read_not_found {
    match (nl_file_read "nonexistent.txt") {
        Ok(_) => assert false  # Should error
        Err(IOError::FileNotFound) => assert true
        Err(_) => assert false  # Wrong error type
    }
}
```

### 9.3 Performance Tests

```c
// Benchmark: Result vs raw return codes
BENCHMARK(result_vs_raw) {
    // Test 1: Result<int, string>
    start_timer();
    for (int i = 0; i < 1000000; i++) {
        Result_int_string r = divide(100, 10);
        if (r.tag == OK_VARIANT) {
            volatile int x = r.data.ok_value;
        }
    }
    double result_time = stop_timer();
    
    // Test 2: Raw return code
    start_timer();
    for (int i = 0; i < 1000000; i++) {
        int value;
        int status = divide_raw(100, 10, &value);
        if (status == 0) {
            volatile int x = value;
        }
    }
    double raw_time = stop_timer();
    
    // Result should be within 5% of raw performance
    assert(result_time < raw_time * 1.05);
}
```

---

## 10. Migration Guide

### 10.1 For Library Authors

**Step 1**: Identify fallible functions
```nano
# Before
fn parse_int(s: string) -> int {
    return 0  # Returns 0 on error - ambiguous!
}

# After
fn parse_int(s: string) -> Result<int, ParseError> {
    if (string_is_empty s) {
        return Err(ParseError::EmptyString)
    }
    
    # ... parsing logic ...
    return Ok(value)
}
```

**Step 2**: Provide transition functions
```nano
# Deprecated: Keep for 6 months
fn parse_int_unsafe(s: string) -> int {
    match (parse_int s) {
        Ok(v) => return v
        Err(_) => return 0
    }
}
```

**Step 3**: Update documentation
```markdown
### parse_int (v2.0+)
Returns `Result<int, ParseError>` to handle parsing failures explicitly.

**Migration**: Replace `parse_int(s)` with match statement or use
`parse_int_unsafe(s)` (deprecated) for drop-in replacement.
```

### 10.2 For Application Developers

**Pattern 1**: Replace sentinel values
```nano
# Before
let port: int = (parse_int port_str)
if (== port 0) {
    (println "Invalid port")
}

# After
match (parse_int port_str) {
    Ok(port) => {
        (connect_to_server host port)
    }
    Err(e) => {
        (println "Invalid port:")
        (println (error_message e))
    }
}
```

**Pattern 2**: Chain operations
```nano
# Before (nested ifs)
let json_str: string = (read_file "config.json")
if (!= json_str "") {
    let config: Config = (parse_json json_str)
    if (config_is_valid config) {
        (use_config config)
    }
}

# After (match expressions)
match (read_file "config.json") {
    Ok(json_str) => {
        match (parse_json json_str) {
            Ok(config) => (use_config config)
            Err(e) => (println "Parse error")
        }
    }
    Err(e) => (println "File error")
}
```

---

## 11. Comparison with Other Languages

### 11.1 Rust

**Similarities**:
- `Result<T, E>` enum with Ok/Err variants
- Pattern matching for error handling
- Zero-cost abstraction
- Composable with map/and_then

**Differences**:
- NanoLang: Simpler syntax (prefix notation)
- Rust: `?` operator for error propagation
- Rust: More sophisticated trait system (Error, From, Into)

### 11.2 Haskell (Either)

**Similarities**:
- Either Left Right ≈ Result Err Ok
- Monadic composition
- Type-driven error handling

**Differences**:
- NanoLang: More explicit (no do-notation)
- Haskell: More powerful type system
- NanoLang: Systems programming focus

### 11.3 Go

**Contrasts**:
- Go: Multiple return values `(value, error)`
- Go: Errors are values but not type-enforced
- NanoLang Result: Compile-time enforcement
- NanoLang Result: Single return value

**Advantage of Result**: Impossible to forget error handling (compile error)

---

## 12. Open Questions & Future Work

### 12.1 Resolved Questions

✅ **Q**: Result vs Exceptions?  
**A**: Result type chosen (better for systems programming, zero-cost)

✅ **Q**: Generic over error type E?  
**A**: Yes, allows domain-specific error types

✅ **Q**: Pattern matching required?  
**A**: Yes, primary method for handling Result

### 12.2 Future Enhancements

🔮 **Question Mark Operator**:
```nano
fn process() -> Result<Data, Error> {
    let x = (fallible_op1())?
    let y = (fallible_op2(x))?
    return Ok(y)
}
```

🔮 **Error Trait System**:
```nano
trait Error {
    fn message(self) -> string
    fn source(self) -> Option<Error>
}

impl Error for IOError { /* ... */ }
```

🔮 **Async Result**:
```nano
async fn fetch(url: string) -> Result<Response, NetworkError> {
    # ...
}
```

---

## 13. Conclusion

The Result<T, E> type is a **critical addition** to NanoLang that:

1. ✅ **Addresses the biggest language gap** (error handling rated D)
2. ✅ **Enables production-ready software** (explicit error handling)
3. ✅ **Zero runtime cost** (compiles to efficient C code)
4. ✅ **Backward compatible** (gradual migration path)
5. ✅ **Industry-proven approach** (Rust, Haskell, Swift)

**Recommendation**: **Approve and implement** as P1 feature.

**Estimated Effort**: 8-12 weeks full implementation including stdlib
migration, documentation, and testing.

**Next Steps**:
1. Review and approve this design document
2. Create implementation issues for each phase
3. Begin Phase 1: Compiler & Parser implementation
4. Parallel work on documentation and examples

---

**Reviewers**: Please provide feedback on:
- Syntax choices (Ok/Err naming)
- Error type design guidelines
- Migration timeline (too aggressive?)
- Missing use cases or concerns

**Document Status**: Draft - Awaiting Review

