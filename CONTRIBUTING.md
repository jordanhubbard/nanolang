# Contributing to NanoLang

## Ground Rules for Feature Development

These rules ensure the long-term maintainability and self-hosting capability of NanoLang.

### 1. **Interpreter/Compiler Feature Parity**

**RULE**: The interpreter and compiler must ALWAYS support the same features.

**Why**: Shadow tests run in the interpreter to validate code correctness. If the interpreter doesn't support a feature that the compiler does, shadow tests will fail and we lose our testing infrastructure.

**Implementation**:
- When adding a new language feature (operators, types, built-ins), implement it in BOTH:
  - Compiler (`src/transpiler.c`, `src/parser.c`, `src/typechecker.c`)
  - Interpreter (`src/eval.c`)
- Before marking a feature complete, verify shadow tests work for that feature
- Shadow tests are NOT optional - they're our quality guarantee

**Example**: Generic `List<T>` support
- ✅ Compiler: Generates C code with `list_TypeName_*` functions
- ✅ Interpreter: Generic handler in `eval.c` delegates to `list_int_*` as backing store
- ✅ Result: Shadow tests work, feature parity achieved

### 2. **Warning-Free, Clean Sources**

**RULE**: All code must compile without warnings. The codebase should be clean and maintainable.

**Why**: Warnings hide real issues. A warning-free codebase makes it easy to spot new problems immediately.

**Requirements**:
- C code: `-Wall -Wextra -Werror` compliance (no warnings = no errors)
- NanoLang code: No type warnings, no unused variable warnings
- CI/CD: Build should fail on warnings
- Exception: Acceptable warnings (like C11 typedef redefinitions) must be documented

**How to maintain**:
```bash
make clean && make  # Should show no warnings
./bin/nanoc file.nano  # Should show no warnings
```

### 3. **Dual Implementation: C Reference + NanoLang Self-Hosted**

**RULE**: All features must be implemented twice:
1. **C Reference Implementation** (in `src/`) - Bootstrap compiler
2. **NanoLang Self-Hosted** (in `src_nano/`) - Self-hosting components

**Why**: 
- C reference compiles NanoLang code initially (bootstrapping)
- NanoLang self-hosted version proves the language is powerful enough to implement itself
- Self-hosting is the ultimate test of language completeness

**Workflow**:
1. **First**: Implement feature in C reference implementation
2. **Test**: Verify it works with existing test suite
3. **Second**: Implement same feature in NanoLang (in `src_nano/`)
4. **Verify**: Self-hosted component uses the new feature correctly
5. **Dogfood**: Use new feature throughout codebase

**File Structure**:
```
src/          - C reference implementation (bootstrap compiler)
  parser.c    - C parser
  typechecker.c - C typechecker
  transpiler.c  - C transpiler
  eval.c      - C interpreter

src_nano/     - NanoLang self-hosted implementation
  parser_mvp.nano - Parser in NanoLang
  typechecker_minimal.nano - Typechecker in NanoLang
  transpiler_minimal.nano - Transpiler in NanoLang
```

### 4. **Test-First Development**

**RULE**: Every feature must have tests BEFORE being marked complete.

**Test Hierarchy**:
1. **Shadow Tests** (inline tests in NanoLang functions)
   - Unit tests for individual functions
   - Run in interpreter during compilation
   - Mandatory for all exported functions

2. **Integration Tests** (`tests/*.nano`)
   - Test feature combinations
   - Run both interpreter and compiler paths
   - Must pass 100%

3. **Self-Hosting Tests**
   - Bootstrap compilation (C compiles NanoLang compiler)
   - Self-compilation (NanoLang compiles itself)
   - Fixed-point verification (C1 ≡ C2)

**Example**:
```nano
fn list_length(list: List<int>) -> int {
    return (list_int_length list)
}

shadow list_length {
    let xs: List<int> = (list_int_new)
    (list_int_push xs 5)
    (list_int_push xs 10)
    assert (== (list_length xs) 2)
}
```

### 5. **Documentation Standards**

**RULE**: Code should be self-documenting, with comments only where necessary.

**What to Document**:
- ✅ Complex algorithms (why, not what)
- ✅ Non-obvious design decisions
- ✅ Workarounds for language/compiler limitations
- ✅ Public API functions (brief description)

**What NOT to Document**:
- ❌ Obvious code (`i++  // increment i`)
- ❌ Redundant README files that duplicate code docs
- ❌ Over-commented code that restates what it does

**README Policy**:
- Update only when requested by contributor/maintainer
- Keep focused: What is NanoLang, how to build, how to run
- Avoid documenting every function (use `--help` flags instead)

### 6. **Error Messages Must Be Excellent**

**RULE**: Error messages should immediately tell the user what's wrong and how to fix it.

**Format**:
```
Error at line X, column Y: <What went wrong>
  Note: <Additional context>
  Hint: <How to fix it>
```

**Example**:
```
Error at line 42, column 18: Undefined function 'list_Point_new'
  Note: struct or enum 'Point' is not defined
  Hint: Define 'struct Point { ... }' before using List<Point>
```

**Line numbers are mandatory**: Every error during parsing, type-checking, or transpiling must include line and column numbers.

### 7. **Backward Compatibility**

**RULE**: Once a feature is released, it must continue to work in future versions.

**Breaking Changes Require**:
- Major version bump
- Migration guide
- Deprecation warnings in previous version
- Community discussion

**Safe Changes**:
- Adding new features (non-breaking)
- Fixing bugs
- Improving error messages
- Performance optimizations

### 8. **Performance Considerations**

**RULE**: Don't sacrifice correctness for performance, but be mindful of efficiency.

**Guidelines**:
- Use appropriate data structures (`List<T>` for dynamic, arrays for fixed)
- Avoid N² algorithms where N could be large
- Profile before optimizing ("premature optimization is evil")
- Memory leaks are bugs - always free allocated memory

**Self-Hosting Performance**:
- Compilation should be fast enough for development (< 5 seconds for small files)
- Shadow tests should run quickly (< 1 second for typical test suite)
- Large files (> 2000 lines) may need optimization or splitting

## Contribution Workflow

### 1. Before Starting

- Read this CONTRIBUTING.md
- Check existing issues/PRs to avoid duplicates
- Discuss large changes in an issue first

### 2. During Development

- Follow the 8 ground rules above
- Write tests as you code (not after)
- Run `make test` frequently
- Keep commits atomic and well-described

### 3. Before Submitting PR

**Checklist**:
```bash
# Clean build
make clean && make

# All tests pass
make test  # Must show 100% pass rate

# No warnings
./bin/nanoc your_file.nano  # Should be warning-free

# Self-hosted components still build
make  # Should rebuild parser_mvp, typechecker_minimal, etc.

# Your feature has tests
# Shadow tests in your .nano file
# Integration test in tests/ if appropriate
```

### 4. PR Description Template

```markdown
## What Changed
Brief description of the feature/fix

## Implementation
- C Reference: <files modified>
- NanoLang Self-Hosted: <files modified>
- Interpreter Parity: ✅ YES / ❌ NO / ⚠️ N/A

## Testing
- Shadow tests: ✅ Added / ⚠️ Updated / ❌ None
- Integration tests: ✅ Added / ⚠️ Updated / ❌ None
- All tests passing: ✅ YES / ❌ NO

## Breaking Changes
YES / NO - if yes, describe migration path

## Checklist
- [ ] Builds warning-free
- [ ] All tests pass (100%)
- [ ] Interpreter/compiler parity maintained
- [ ] Documentation updated (if needed)
- [ ] Self-hosted components updated (if needed)
```

## Questions?

Open an issue or discussion. We're here to help!

## Code of Conduct

- Be respectful and constructive
- Focus on the code, not the person
- Assume good intentions
- Help newcomers learn

**Welcome to NanoLang! Let's build something amazing together.** 🚀
