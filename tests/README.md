# Nanolang Test Suite

This directory contains comprehensive tests for the nanolang compiler and language features.

## Test Organization

Tests are organized by category using filename prefixes:

### Core Language Tests (`nl_*`)

These tests verify fundamental nanolang language features:

| Prefix | Category | Description |
|--------|----------|-------------|
| `nl_syntax_*` | Syntax | Literals, operators, expressions |
| `nl_types_*` | Types | Structs, enums, unions, tuples, generics |
| `nl_control_*` | Control Flow | if/else, while, for, match |
| `nl_functions_*` | Functions | Definitions, calls, recursion, higher-order |
| `nl_modules_*` | Modules | Import, namespacing |

### Application Tests

Higher-level tests that exercise combinations of features:

| Pattern | Description |
|---------|-------------|
| `tuple_*.nano` | Tuple usage patterns |
| `test_nested_*.nano` | Nested structures |
| `test_*.nano` | Various feature compositions |

### Unit Tests (`unit/`)

Comprehensive feature tests in the `unit/` subdirectory:

- `test_control_flow.nano` - Complete control flow coverage
- `test_enums_comprehensive.nano` - Full enum testing
- `test_generics_comprehensive.nano` - Generic type testing
- `test_operators_comprehensive.nano` - All operators
- `test_unions_match_comprehensive.nano` - Union and match patterns

### Other Directories

| Directory | Purpose |
|-----------|---------|
| `integration/` | Module integration tests |
| `negative/` | Expected failure tests |
| `performance/` | Performance benchmarks |
| `regression/` | Regression tests |
| `selfhost/` | Self-hosted compiler tests |

## Running Tests

### All Tests
```bash
make test              # Run all tests
./tests/run_all_tests.sh  # Direct execution
```

### By Category
```bash
make test-lang         # Core language tests (nl_*)
make test-app          # Application tests
make test-unit         # Unit tests
make test-quick        # Quick check (lang only)
make test-full         # All tests + examples
```

### Direct Script Options
```bash
./tests/run_all_tests.sh --lang   # Language tests only
./tests/run_all_tests.sh --app    # Application tests only
./tests/run_all_tests.sh --unit   # Unit tests only
```

## Test File Format

Each test file should:

1. Include a header comment describing the test category
2. Use `shadow` tests for validation
3. Include a `main` function with success indicator

Example:
```nanolang
/* nl_types_struct.nano - Struct type tests
 * Category: Core Language - Types
 */

struct Point {
    x: int,
    y: int
}

fn test_point() -> int {
    let p: Point = Point { x: 10, y: 20 }
    return (+ p.x p.y)
}

shadow test_point {
    assert (== (test_point) 30)
}

fn main() -> int {
    (println "All tests passed!")
    return 0
}
```

## Adding New Tests

### For Language Features

1. Create file: `tests/nl_{category}_{feature}.nano`
2. Follow naming convention:
   - `nl_syntax_*` for syntax features
   - `nl_types_*` for type system
   - `nl_control_*` for control flow
   - `nl_functions_*` for function features
3. Include shadow tests for each function
4. Add main function for integration test

### For Applications

1. Create file: `tests/test_{feature}.nano` or `tests/{feature}_test.nano`
2. Include comprehensive shadow tests
3. Document what feature combinations are tested

## Expected Failures

Some tests are marked as expected failures due to incomplete transpiler support:

- `test_firstclass_functions.nano` - First-class functions not implemented
- `test_unions_match_comprehensive.nano` - Union transpilation ordering

These are skipped automatically and shown in blue in test output.

## Test Coverage

### Core Language Features (nl_*)

| Feature | Test File | Status |
|---------|-----------|--------|
| Literals | `nl_syntax_literals.nano` | ✅ |
| Operators | `nl_syntax_operators.nano` | ✅ |
| Structs | `nl_types_struct.nano` | ✅ |
| Enums | `nl_types_enum.nano` | ✅ |
| Tuples | `nl_types_tuple.nano` | ✅ |
| Union Construction | `nl_types_union_construct.nano` | ✅ |
| If/While | `nl_control_if_while.nano` | ✅ |
| For Loops | `nl_control_for.nano` | ✅ |
| Match | `nl_control_match.nano` | ✅ |
| Functions | `nl_functions_basic.nano` | ✅ |

### Edge Cases Tested

The following advanced features have dedicated test coverage:

1. **Match Expressions** (`nl_control_match.nano`)
   - Basic match on union variants
   - Match with field extraction
   - Match in control flow
   - Chained match operations
   - Error handling patterns

2. **Tuple Literals** (`nl_types_tuple.nano`)
   - Tuple disambiguation vs parenthesized expressions
   - Mixed type tuples
   - Tuple return values
   - Tuple field access
   - Complex tuple expressions

3. **Union Construction** (`nl_types_union_construct.nano`)
   - Variant construction with fields
   - Empty variant construction
   - Computed field values
   - Union in loops
   - Either pattern (Left/Right)

## CI/CD Integration

The test suite returns:
- Exit code 0: All tests passed
- Exit code 1: Some tests failed

Use in CI:
```bash
make test && echo "Tests passed" || echo "Tests failed"
```
