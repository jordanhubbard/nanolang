---
description: "Code quality standards, parser architecture patterns, and development conventions"
alwaysApply: true
---

# Code Quality Standards

## When Adding Language Features

If you add a new language feature to nanolang:

1. **Stage 0 (C Compiler):**  
   - Add feature to `src/parser.c` first
   - Ensure C compiler handles it (100% support required)

2. **Stage 2 (Self-Hosted Parser):**  
   - Add corresponding AST struct in `parser_mvp.nano`
   - Add `parser_store_*` function
   - **Integrate into parsing logic** (critical!)
   - Add token helpers if needed

3. **Testing:**  
   - Run `./bin/nanoc src_nano/parser_mvp.nano` - must pass ✅
   - Run `./tests/run_all_tests.sh` - must pass ✅
   - Run `make examples` - must build ✅

4. **Documentation:**  
   - Update feature lists in markdown docs
   - Update completion percentages

## Parser Architecture

### Current Status

The self-hosted parser has:
- **31 node types** (complete)
- **29 AST structs** (complete)
- **67 Parser fields** (complete)
- **22+ parsing functions** (production-ready)

### Adding New Node Types

1. Add to `ParseNodeType` enum
2. Create AST struct definition
3. Add list field to Parser struct
4. Add count field to Parser struct
5. Initialize in `parser_init_ast_lists`
6. Initialize in `parser_new`
7. Create `parser_store_*` function (copy pattern from existing)
8. Add parsing logic to appropriate `parse_*` function
9. **Test immediately**

### Code Patterns

Follow existing patterns:
- Use functional style (return new Parser)
- No mutations except with `mut` keyword
- Consistent error handling
- Shadow tests for validation

## Parser Changes

1. **Never break existing functionality**
2. **Always add shadow tests for new functions**
3. **Keep architecture consistent**
4. **Document complex logic**

## Error Handling

### Parser Errors

- Use `parser_with_error` for syntax errors
- Include line/column information
- Provide clear error messages
- Never silently fail

## Performance Standards

### Parser Performance

- Must compile itself in < 10 seconds
- No memory leaks
- Reasonable file size (< 5000 lines preferred)
- Clean compilation (no warnings)

## Documentation Standards

### Code Comments

- Document WHY, not WHAT
- Complex algorithms need explanation
- Edge cases should be noted
- TODOs must have tracking

## Git Workflow

### Branch Names
- `feat/*` - New features
- `fix/*` - Bug fixes
- `docs/*` - Documentation only
- `test/*` - Test additions

### Commit Messages

Format:
```
type: Brief description

Detailed explanation if needed

- Changes made
- Tests: status
- Self-hosting: status
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`

### Pre-Commit Checklist

```bash
# 1. Compile check
./bin/nanoc src_nano/parser_mvp.nano || exit 1

# 2. Test check  
./tests/run_all_tests.sh || exit 1

# 3. Self-hosting check
./bin/nanoc src_nano/lexer_complete.nano || exit 1
```
