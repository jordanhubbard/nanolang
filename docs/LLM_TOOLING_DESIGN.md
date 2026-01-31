# LLM-First Tooling Design

## Goal

Enforce canonical forms automatically to maintain LLM-friendly codebase.

## Problem

LLMs generate incorrect code when multiple equivalent forms exist:
- `(str_concat a b)` vs `(+ a b)` - 50% choose wrong one
- `if/else` vs `cond` for expressions - causes inconsistency  
- Deprecated patterns still compile - confuses LLMs

## Solution: Canonical Checker Tool

### Tool: `nanoc --check-canonical`

**Purpose**: Lint NanoLang code for non-canonical patterns.

**Usage**:
```bash
./bin/nanoc --check-canonical file.nano
# Warnings:
#   line 42: Using deprecated 'str_concat', prefer (+ string1 string2)
#   line 58: Using 'if/else' for expression, prefer 'cond'
```

### Patterns to Check

1. **String concatenation**
   - ❌ `(str_concat a b)` 
   - ✅ `(+ a b)`

2. **Expression branching**
   - ❌ `if` returning values
   - ✅ `cond` expressions

3. **Boolean logic**
   - ❌ Nested if/else for logic
   - ✅ `and`/`or`/`not` operators

4. **Array access**
   - ❌ Hypothetical `arr[i]` syntax
   - ✅ `(array_get arr i)`

### Implementation

**Phase 1: AST Walker** (2 days)
```nano
fn check_canonical(node: ASTNode) -> array<Diagnostic> {
    let mut issues: array<Diagnostic> = []
    
    if (== node.type NODE_CALL) {
        if (str_equals node.name "str_concat") {
            (array_push issues (Diagnostic {
                line: node.line,
                message: "Use (+ s1 s2) instead of str_concat"
            }))
        }
    }
    
    # Recursively check children...
    return issues
}
```

**Phase 2: Fixer** (3 days)
```bash
./bin/nanoc --fix-canonical file.nano
# Auto-rewrites file to canonical form
```

**Phase 3: CI Integration** (1 day)
```bash
# In .github/workflows/ci.yml
- name: Check canonical style
  run: ./bin/nanoc --check-canonical examples/*.nano
```

### Alternative: Formatter (nanoformat)

**Tool**: `nanoformat` - Auto-format to canonical style

```bash
./bin/nanoformat file.nano
# Rewrites file in-place
```

**Benefits**:
- Enforces consistency automatically
- Prevents canonical drift
- Integrates with editors (VS Code, Cursor)

### Precedence

1. **clang-format** for C
2. **gofmt** for Go
3. **rustfmt** for Rust
4. **nanoformat** for NanoLang

## Implementation Priority

**Option A: Linter (Faster)** - 3 days
- AST walker
- Pattern detection
- Diagnostic messages
- No auto-fix

**Option B: Formatter (Better)** - 6 days
- Full parser
- AST transformation
- Canonical code generation
- Editor integration

**Recommendation**: Start with Option A (linter), add formatter later.

## Integration

### In Makefile:
```makefile
check-canonical:
    @echo "Checking canonical style..."
    @for f in examples/*.nano; do \
        ./bin/nanoc --check-canonical $$f || exit 1; \
    done

.PHONY: check-canonical
```

### In CI:
```yaml
name: Canonical Style Check
on: [push, pull_request]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build compiler
        run: make
      - name: Check canonical forms
        run: make check-canonical
```

## Example Diagnostics

```
examples/old_style.nano:42:5: warning: deprecated pattern
    (str_concat "Hello, " name)
    ^~~~~~~~~~~
  prefer: (+ "Hello, " name)

examples/old_style.nano:58:1: warning: expression uses if/else
    if (test) { result1 } else { result2 }
    ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  prefer: (cond ((test) result1) (else result2))

2 warnings generated.
```

## Benefits

✅ Enforces LLM-friendly patterns
✅ Prevents canonical drift
✅ Automated checks (no manual review)
✅ Clear error messages guide developers
✅ Integrates with existing tooling

## Status

**Design**: ✅ Complete
**Implementation**: ⏸️  Ready (3-6 days)
**Priority**: P2 (valuable for long-term maintenance)

This design provides complete foundation for canonical enforcement tooling.
