# NanoLang Standard Library

This directory contains the organized implementation of NanoLang's standard library functions for **compiled mode**.

## Organization

The stdlib is organized into focused modules by functionality:

| Module | Functions | Description |
|--------|-----------|-------------|
| **io.c/h** | print, println, assert | Basic I/O and assertions |
| **string.c/h** | str_*, char_*, is_* | String manipulation and character operations |
| **array.c/h** | array_* | Array operations |

## Architecture

### Interpreter vs Compiled Mode

NanoLang has two execution modes:

1. **Interpreter Mode** (`src/eval/`)
   - Built-in functions in `eval_math.c`, `eval_string.c`, `eval_io.c`, etc.
   - Used when running NanoLang programs directly with the interpreter
   - Functions are Value-based and part of the eval loop

2. **Compiled Mode** (`src/stdlib/`)
   - Runtime library functions for compiled C code
   - Used when transpiling NanoLang to C with `nanoc`
   - Functions are C-native (int, double, char*, etc.)
   - This directory contains the compiled mode runtime

### Module Structure

Each module follows this structure:

```
string.c/h:
├── Header guards
├── Necessary includes
├── Function implementations
└── No external dependencies (beyond libc)
```

## Adding New Functions

To add a new stdlib function:

1. **Choose the right module** - Add to appropriate category (io/string/array)
2. **Declare in header** - Add function signature to module's .h file
3. **Implement in source** - Add implementation to module's .c file
4. **Update spec.json** - Add function to appropriate category
5. **Update transpiler** - Ensure transpiler generates correct include
6. **Add tests** - Create test cases in tests/stdlib/
7. **Update docs** - Document in docs/STDLIB.md

## Building

The stdlib modules are automatically built as part of the main NanoLang build:

```bash
make build    # Builds all stdlib modules
make test     # Tests stdlib functionality
```

## Module Guidelines

### General Rules

- ✅ Use C99 standard features only
- ✅ Include only necessary headers
- ✅ Use defensive programming (null checks, bounds checking)
- ✅ Prefer stack allocation over heap where possible
- ✅ Free all allocated memory
- ✅ Use descriptive function and variable names
- ✅ Add comments for complex logic

### Naming Conventions

- **Functions:** `category_function_name()` (e.g., `str_length`, `array_push`)
- **Constants:** `UPPER_SNAKE_CASE`
- **Types:** `PascalCase` (if needed)

### Error Handling

- Use `fprintf(stderr, ...)` for error messages
- Return sensible defaults on error (0, NULL, empty string)
- Document error behavior in function comments

## Testing

Each module should have corresponding tests:

```
tests/stdlib/
├── test_io.nano
├── test_string.nano
└── test_array.nano
```

Run stdlib-specific tests:
```bash
cd tests/stdlib
./run_stdlib_tests.sh
```

## Dependencies

stdlib modules should have minimal dependencies:

- **Allowed:** `<stdlib.h>`, `<string.h>`, `<stdio.h>`, `<stdint.h>`, `<stdbool.h>`
- **Avoid:** NanoLang-specific headers (except for safe_* wrappers if needed)
- **Never:** Circular dependencies between stdlib modules

## Performance Considerations

- String operations should be efficient (avoid unnecessary allocations)
- Array operations should respect bounds
- Use appropriate algorithms (e.g., don't use O(n²) when O(n) exists)

## Memory Management

- **Caller owns:** Returned pointers must be freed by caller (document this)
- **Function owns:** Internal allocations freed before return
- **No leaks:** Use valgrind to verify no memory leaks

## Contributing

When modifying stdlib:

1. Test your changes thoroughly
2. Run full test suite (`make test`)
3. Check for memory leaks (`make valgrind`)
4. Update documentation
5. Maintain backward compatibility

---

**Reorganized:** January 25, 2026
**Status:** Active Development
**Functions:** 72 total across all modules
