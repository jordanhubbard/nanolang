# Phase 2 Implementation Strategy

**Date:** November 29, 2025  
**Goal:** Implement full AST walking in type checker and transpiler

## Core Challenge

The main challenge is accessing Parser's AST data from type checker and transpiler:
- Parser has `List<ASTNode>` types instantiated at runtime
- Type checker and transpiler compile separately
- Need to pass AST information between modules

## Strategy: Accessor Functions

Instead of directly accessing Parser fields, create accessor functions that:
1. Return counts (already have these via struct fields)
2. Return individual nodes by index
3. Extract specific node information

### Approach

```nano
/* In parser_mvp.nano - add accessor functions */
fn parser_get_function_count(p: Parser) -> int {
    return (List_ASTFunction_length p.functions)
}

fn parser_get_function(p: Parser, idx: int) -> ASTFunction {
    return (List_ASTFunction_get p.functions idx)
}
```

Then in type checker/transpiler:
```nano
/* Declare as extern */
extern fn parser_get_function_count(p: Parser) -> int
extern fn parser_get_function(p: Parser, idx: int) -> ASTFunction

/* Use in type checking */
fn typecheck_parser(p: Parser) -> int {
    let count: int = (parser_get_function_count p)
    let mut i: int = 0
    while (< i count) {
        let func: ASTFunction = (parser_get_function p i)
        /* Type check function */
        set i (+ i 1)
    }
    return 0
}
```

## Implementation Plan

### Phase 2.1: Parser Accessor Functions (1-2 days)
- Add getter functions for all AST node types
- Test each function individually
- Document accessor API

### Phase 2.2: Type Checker Enhancement (2-3 days)
- Update to use accessor functions
- Implement full expression type checking
- Implement full statement type checking
- Walk through function bodies
- Add comprehensive error reporting

### Phase 2.3: Transpiler Enhancement (2-3 days)
- Update to use accessor functions
- Implement full expression code generation
- Implement full statement code generation
- Generate functions from AST
- Handle all node types

### Phase 2.4: Integration Update (1 day)
- Pass Parser to type checker
- Pass Parser to transpiler
- Update compile_program() to use new signatures

### Phase 2.5: Testing (2-3 days)
- Create test programs
- Validate type checking
- Verify C code generation
- Compile and run generated programs

## Success Criteria

- [ ] Type checker validates actual AST from parser
- [ ] Transpiler generates C code from actual AST
- [ ] Simple programs compile end-to-end
- [ ] Generated executables run correctly
- [ ] All tests pass

## Timeline

Total estimated: 8-12 days
