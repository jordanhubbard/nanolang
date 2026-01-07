# Self-Hosting Roadmap - Path to 100%

**Status**: 99.5% self-hosting (0.5% away!)

## Current Blocker

**Parse Error**: Self-hosted parser fails at line 10177 (transpiler.nano:182)
```
Parse error at line 10177, column 377: unexpected token ''
```

### Root Cause Analysis
1. **Column 377 is suspiciously high** for a typical line
2. **The line itself is fine**: `let mut i: int = 0`
3. **Lexer column tracking bug**: Column counter appears to accumulate incorrectly

### Lexer Bug Location
`src_nano/compiler/lexer.nano` line 272-280:
- Whitespace handling increments `column` manually
- Main token processing recalculates `column` from `line_start`
- **Inconsistency**: Two different column tracking mechanisms!

## Immediate Fix (Option 1: Quick Win)

**Fix lexer column tracking to be consistent**:
```nano
/* Remove manual column increments in whitespace handling */
if (is_whitespace_char c) {
    if (== c 10) {
        set line (+ line 1)
        set line_start (+ i 1)
    }
    set i (+ i 1)
    /* Don't manually track column - recalculate it when needed */
}
```

## Refactoring Plan (Option 2: Long-term Solution)

### Phase 1: Extract Token Utilities (~1 hour)
- Move 64 `token_*()` functions to data-driven lookup
- **Benefit**: Reduce 200 lines to 20 lines

### Phase 2: Extract Parser Core (~2 hours)
- `parser_core.nano`: State management, navigation
- `parser_storage.nano`: AST node builders
- **Benefit**: Isolate foundation for debugging

### Phase 3: Extract Parser Logic (~4 hours)
- `parser_expressions.nano`: Expression parsing (1,800 lines)
- `parser_statements.nano`: Statement parsing (1,200 lines)
- `parser_definitions.nano`: Top-level parsing (1,400 lines)
- **Benefit**: Modular, testable units

### Phase 4: Add Shadow Tests (~2 hours)
- Unit tests for each extracted module
- Integration tests for parser pipeline
- **Benefit**: Catch regressions early

## Recommended Approach

**Hybrid Strategy**:
1. ✅ **Fix lexer bug NOW** (30 minutes) → Achieve 100% self-hosting
2. ✅ **Document refactoring plan** (done!)
3. ✅ **Create refactoring scripts** to automate extraction
4. ✅ **Refactor incrementally** over next sessions

## Refactoring Automation

Create `scripts/refactor_parser.sh`:
```bash
#!/bin/bash
# Extract functions matching pattern to new file
# Usage: ./scripts/refactor_parser.sh "parser_store_*" parser_storage.nano
```

## Testing Strategy

After each refactoring step:
```bash
# 1. Test C compiler still builds
make bin/nanoc

# 2. Test self-hosted compiler builds
./bin/nanoc src_nano/nanoc_v06.nano -o bin/nanoc_v06

# 3. Test self-compilation
./bin/nanoc_v06 src_nano/nanoc_v06.nano -o bin/nanoc_v06_selfhosted

# 4. Verify binary works
./bin/nanoc_v06_selfhosted tests/integration/test_simple.nano -o /tmp/test
```

## Success Criteria

- ✅ `nanoc_v06` compiles itself without errors
- ✅ Generated binary passes all tests
- ✅ Parser codebase split into <1000 line modules
- ✅ Each module has shadow tests
- ✅ Build time remains < 30s

## Next Actions

1. **Immediate**: Fix lexer column bug
2. **Short-term**: Test self-hosting success
3. **Medium-term**: Begin incremental refactoring
4. **Long-term**: Full modular architecture

---

*This roadmap balances quick wins (100% self-hosting) with long-term maintainability (refactored codebase).*

