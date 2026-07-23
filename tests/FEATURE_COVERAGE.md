# Nanolang Feature Test Coverage Matrix

## Test Coverage Status

### ✅ Core Language Features with Tests

| Feature | Test File(s) | Status |
|---------|-------------|--------|
| **Basic Types** | | |
| - int | Multiple tests | ✅ |
| - float | test_vector2d.nano | ✅ |
| - bool | test_enums_comprehensive.nano | ✅ |
| - string | Multiple tests | ✅ |
| - void | All function returns | ✅ |
| **Composite Types** | | |
| - Arrays | test_dynamic_arrays.nano | ✅ |
| - Structs | test_vector2d.nano, lexer_struct_test.nano | ✅ |
| - Enums | unit/test_enums_comprehensive.nano | ✅ |
| - Unions | unit/test_unions_match_comprehensive.nano | ✅ |
| - Tuples | tuple_*.nano (5 files) | ✅ |
| - Generics | unit/test_generics_comprehensive.nano | ✅ |
| - First-class functions | unit/test_firstclass_functions.nano | ✅ |
| **Statements** | | |
| - let (immutable) | Multiple tests | ✅ |
| - let mut (mutable) | negative/mutability_errors/ | ✅ |
| - set | negative/mutability_errors/ | ✅ |
| - if/else | Multiple tests | ✅ |
| - while | nl_control_while.nano, nl_control_flow.nano | ✅ |
| - for | regression/bug_2025_09_30_for_loop_segfault.nano | ✅ |
| - return | negative/return_errors/ | ✅ |
| - match | unit/test_unions_match_comprehensive.nano | ✅ |
| **Operations** | | |
| - Arithmetic (+, -, *, /, %) | nl_syntax_operators.nano | ✅ |
| - Comparison (==, !=, <, <=, >, >=) | nl_syntax_operators.nano | ✅ |
| - Logical (and, or, not) | nl_syntax_operators.nano | ✅ |
| **Special Features** | | |
| - Shadow tests | All tests | ✅ |
| - Type checking | negative/type_errors/ | ✅ |
| - Mutability | negative/mutability_errors/ | ✅ |
| - Namespacing | integration/test_namespacing.nano | ✅ |
| - Modules/Imports | integration/test_modules.nano | ✅ |

### ✅ Standard Library Coverage

Comprehensive stdlib coverage now lives in `tests/unit/test_stdlib_comprehensive.nano`.

### 📝 Test Categories

- **Unit Tests** (`tests/unit/`): 22+ tests ✅
- **Integration Tests** (`tests/integration/`): 5 tests ✅
- **Negative Tests** (`tests/negative/`): 39 error condition tests (incl. subdirs) ✅
- **Regression Tests** (`tests/regression/`): 2 tests ✅
- **Performance Tests** (`tests/performance/`): optional
- **Warning Tests** (`tests/warnings/`): 2 tests ✅

**Total**: 340+ `.nano` test files across all categories (run `find tests -name '*.nano' | wc -l` to confirm).

## Action Items

The previously listed comprehensive tests now exist under `tests/unit/`:

- ✅ `unit/test_operators_comprehensive.nano` - All operators
- ✅ `unit/test_control_flow.nano` - while, for, if/else
- ✅ `unit/test_stdlib_comprehensive.nano` - Standard library coverage

No open action items remain in this matrix. Run the suite with `make test-quick`
(or the full `make test`) to exercise these under both the interpreter and compiler.

