# Error Propagation Operator (?) - Implementation Plan

## Goal

Add Rust-style `?` operator to unwrap `Result<T,E>` or early-return errors.

## Problem Statement

Error handling is verbose:

```nano
fn parse_and_double(s: string) -> Result<int, string> {
    let r1: Result<int, string> = (str_to_int s)
    match r1 {
        Ok(v) => {
            let doubled: int = (* v.value 2)
            return Result.Ok { value: doubled }
        }
        Error(e) => {
            return Result.Error { error: e.error }
        }
    }
}
```

With `?` operator:

```nano
fn parse_and_double(s: string) -> Result<int, string> {
    let num: int = (str_to_int s)?  // Unwrap or early return
    return Result.Ok { value: (* num 2) }
}
```

## Syntax Design

```nano
// Postfix ? operator
let value: T = (result_expr)?

// Desugars to:
let temp_result: Result<T, E> = (result_expr)
match temp_result {
    Ok(v) => v.value,
    Error(e) => return Result.Error { error: e.error }
}
```

## Examples

### Simple Error Propagation

```nano
fn read_and_parse(path: string) -> Result<int, string> {
    let contents: string = (read_file path)?
    let num: int = (str_to_int contents)?
    return Result.Ok { value: num }
}
```

### Chaining Multiple Operations

```nano
fn complex_operation(x: int) -> Result<int, string> {
    let a: int = (checked_add x 10)?
    let b: int = (checked_mul a 2)?
    let c: int = (checked_div b 3)?
    return Result.Ok { value: c }
}
```

### Error Context

```nano
fn parse_config(path: string) -> Result<Config, string> {
    let json: string = (read_file path)?
        .map_err("Failed to read config")?
    let config: Config = (parse_json json)?
        .map_err("Failed to parse config")?
    return Result.Ok { value: config }
}
```

## Implementation Strategy

### Phase 1: Lexer (1 hour)

```nano
// Add QUESTION token
enum TokenType {
    // ... existing ...
    TOKEN_QUESTION = 68  // After TOKEN_UNSAFE
}

// Lexer rule
if (c == '?') {
    return Token { token_type: TOKEN_QUESTION, ... }
}
```

### Phase 2: Parser (3 hours)

```nano
// Parse postfix ? operator
fn parse_postfix_expr(base: ASTNode) -> ASTNode {
    if (current_token == TOKEN_QUESTION) {
        consume(TOKEN_QUESTION)
        return AST_TRY_UNWRAP { expr: base }
    }
    return base
}

// AST node
enum ASTNodeType {
    // ... existing ...
    AST_TRY_UNWRAP  // expr?
}

struct ASTTryUnwrap {
    node_type: int,
    line: int,
    column: int,
    expr: int  // Expression that returns Result<T,E>
}
```

### Phase 3: Type Checker (5 hours)

```nano
fn check_try_unwrap(tc: TypeChecker, node: ASTTryUnwrap) -> Type {
    // 1. Check expr type
    let expr_type: Type = (check_expr tc node.expr)
    
    // 2. Verify it's Result<T, E>
    if (!= expr_type.kind TypeKind.TYPE_GENERIC) {
        error("? operator requires Result<T,E> type")
    }
    if (!= expr_type.name "Result") {
        error("? operator requires Result<T,E> type")
    }
    
    // 3. Verify function returns Result<_, E> (same error type)
    let fn_ret_type: Type = (current_function_return_type tc)
    if (!= fn_ret_type.kind TypeKind.TYPE_GENERIC) {
        error("? operator requires function to return Result<_,E>")
    }
    
    // 4. Verify error types match
    let expr_err_type: Type = (type_param_at expr_type 1)
    let fn_err_type: Type = (type_param_at fn_ret_type 1)
    if (not (types_compatible expr_err_type fn_err_type)) {
        error("Error type mismatch")
    }
    
    // 5. Extract success type T
    return (type_param_at expr_type 0)
}
```

### Phase 4: Transpiler (6 hours)

```nano
// Desugar ? operator to match expression
fn transpile_try_unwrap(t: Transpiler, node: ASTTryUnwrap) -> string {
    // Generate unique temp variable
    let temp_var: string = (gen_temp_var t)
    
    // Build match expression
    let result: StringBuilder = (sb_new)
    (sb_append result "({ ")
    
    // Evaluate expression into temp
    (sb_append result (type_to_c_type (get_expr_type node.expr)))
    (sb_append result " ")
    (sb_append result temp_var)
    (sb_append result " = ")
    (sb_append result (transpile_expr t node.expr))
    (sb_append result "; ")
    
    // Match on result
    (sb_append result "if (")
    (sb_append result temp_var)
    (sb_append result ".tag == RESULT_OK) { ")
    (sb_append result temp_var)
    (sb_append result ".as.ok.value")
    (sb_append result " } else { return ")
    (sb_append result temp_var)  // Return error
    (sb_append result "; } })")
    
    return (sb_to_string result)
}
```

### Phase 5: Testing (2 hours)

```nano
// Test basic unwrap
shadow test_try_basic {
    fn ok_case() -> Result<int, string> {
        let r: Result<int, string> = Result.Ok { value: 42 }
        let v: int = r?
        return Result.Ok { value: v }
    }
    
    match (ok_case) {
        Ok(v) => assert (== v.value 42),
        Error(e) => assert false
    }
}

// Test error propagation
shadow test_try_error {
    fn error_case() -> Result<int, string> {
        let r: Result<int, string> = Result.Error { error: "fail" }
        let v: int = r?  // Should return early
        return Result.Ok { value: v }
    }
    
    match (error_case) {
        Ok(v) => assert false,
        Error(e) => assert (str_equals e.error "fail")
    }
}

// Test chaining
shadow test_try_chain {
    fn chain() -> Result<int, string> {
        let a: int = (checked_add 5 10)?
        let b: int = (checked_mul a 2)?
        return Result.Ok { value: b }
    }
    
    match (chain) {
        Ok(v) => assert (== v.value 30),
        Error(e) => assert false
    }
}
```

### Phase 6: Documentation (2 hours)

- Update spec.json with `?` operator
- Add MEMORY.md section on error propagation
- Create examples/nl_try_operator_demo.nano
- Update CONTRIBUTING.md with pattern

## Breaking Changes

**None** - This is a pure addition, opt-in feature.

## Integration

### With Checked Arithmetic

```nano
fn safe_compute(a: int, b: int, c: int) -> Result<int, string> {
    let sum: int = (checked_add a b)?
    let product: int = (checked_mul sum c)?
    return Result.Ok { value: product }
}
```

### With File I/O

```nano
fn process_file(path: string) -> Result<int, string> {
    let contents: string = (read_file path)?
    let lines: array<string> = (str_split contents "\n")?
    return Result.Ok { value: (array_length lines) }
}
```

## Estimated Effort

- **Lexer:** 1 hour
- **Parser:** 3 hours (C + Nano)
- **Type Checker:** 5 hours (C + Nano)
- **Transpiler:** 6 hours (C + Nano)
- **Testing:** 2 hours
- **Documentation:** 2 hours
- **Examples:** 1 hour
- **TOTAL:** ~20 hours

**Dual-Impl Overhead:** 1.67x (most work is algorithmic, not typing)

## Comparison with Other Languages

| Language | Syntax | Notes |
|----------|--------|-------|
| Rust | `let x = expr?;` | Unwraps Result/Option |
| Swift | `let x = try expr` | Throws on error |
| Kotlin | `val x = expr!!` | Panics on null |
| Zig | `const x = try expr;` | Error union |
| **NanoLang** | `let x: int = expr?` | Unwraps Result<T,E> |

## Status

🟢 **READY FOR IMPLEMENTATION** - Well-defined, small scope

**Recommendation:**
- High value (dramatically improves ergonomics)
- Low cost (20 hours)
- No breaking changes
- Fits existing patterns

**Priority:** HIGH - Should be implemented soon

Related: nanolang-nt2j

