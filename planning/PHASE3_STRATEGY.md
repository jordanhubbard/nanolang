# Phase 3 Implementation Strategy

**Date:** November 29, 2025  
**Goal:** Implement full expression/statement handling for end-to-end compilation

## Overview

Phase 3 completes the self-hosted compiler by implementing:
1. Full expression type checking and code generation
2. Complete statement code generation
3. Function parameter handling
4. End-to-end compilation of real programs

## Implementation Plan

### Step 1: Parameter Handling (1-2 days)

**Parser Enhancement:**
- Add parameter storage to Parser (already has structures)
- Add accessor functions for parameters
- Extract parameter names and types

**Functions to Add:**
```nano
fn parser_get_function_param_count(p: Parser, func_idx: int) -> int
fn parser_get_function_param_name(p: Parser, func_idx: int, param_idx: int) -> string
fn parser_get_function_param_type(p: Parser, func_idx: int, param_idx: int) -> string
```

### Step 2: Expression Type Checking (2-3 days)

**Type Checker Enhancement:**
- Implement recursive expression type checking
- Walk expression trees (binary ops, calls, literals)
- Validate operator usage
- Check function call types

**Key Functions:**
```nano
fn typecheck_expression(parser: Parser, node_id: int, node_type: int, symbols: array<Symbol>) -> Type
fn typecheck_binary_op(parser: Parser, binop: ASTBinaryOp, symbols: array<Symbol>) -> Type
fn typecheck_call(parser: Parser, call: ASTCall, symbols: array<Symbol>) -> Type
```

### Step 3: Expression Code Generation (2-3 days)

**Transpiler Enhancement:**
- Generate C code for all expression types
- Handle literals (numbers, strings, bools)
- Handle identifiers
- Handle binary operations
- Handle function calls

**Key Functions:**
```nano
fn generate_expression(parser: Parser, node_id: int, node_type: int) -> string
fn generate_binary_op(parser: Parser, binop: ASTBinaryOp) -> string
fn generate_call(parser: Parser, call: ASTCall) -> string
```

### Step 4: Statement Code Generation (2-3 days)

**Transpiler Enhancement:**
- Generate let statements
- Generate return statements  
- Generate if/else statements
- Generate while loops
- Generate blocks

**Key Functions:**
```nano
fn generate_statement(parser: Parser, stmt_id: int, stmt_type: int) -> string
fn generate_let_statement(parser: Parser, let_stmt: ASTLet) -> string
fn generate_return_statement(parser: Parser, ret_stmt: ASTReturn) -> string
fn generate_if_statement(parser: Parser, if_stmt: ASTIf) -> string
```

### Step 5: Function Body Generation (1-2 days)

**Transpiler Enhancement:**
- Walk through function body blocks
- Generate all statements in order
- Handle parameters
- Combine into complete function

**Key Changes:**
```nano
fn transpile_parser(parser: Parser) -> string {
    /* For each function */
    /* Get function body block */
    /* Generate all statements */
    /* Combine with function signature */
}
```

### Step 6: End-to-End Testing (2-3 days)

**Create Test Programs:**
1. Simple arithmetic: `fn add(a: int, b: int) -> int { return (+ a b) }`
2. Hello world: `fn main() -> int { (println "Hello") return 0 }`
3. Control flow: if statements, while loops
4. Multiple functions: function calls

**Test Process:**
1. Compile test program with self-hosted compiler
2. Verify generated C code
3. Compile C code with gcc
4. Run executable
5. Verify output

## Success Criteria

- [ ] Type checker validates all expression types
- [ ] Transpiler generates C for all expression types
- [ ] Functions with parameters compile correctly
- [ ] All statement types generate proper C
- [ ] Simple programs compile end-to-end
- [ ] Generated executables run and produce correct output

## Timeline

| Task | Estimated | Priority |
|------|-----------|----------|
| Parameter Handling | 1-2 days | High |
| Expression Type Checking | 2-3 days | High |
| Expression Code Gen | 2-3 days | High |
| Statement Code Gen | 2-3 days | High |
| Function Body Gen | 1-2 days | High |
| End-to-End Testing | 2-3 days | High |

**Total:** 10-16 days

## Risk Mitigation

**Risk 1:** Expression tree complexity
- Mitigation: Start with simple literals, add complexity incrementally

**Risk 2:** Statement nesting
- Mitigation: Test each statement type independently first

**Risk 3:** Type checking edge cases
- Mitigation: Use simple types first (int, bool), add more later
