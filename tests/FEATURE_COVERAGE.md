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
| - while | ❓ Need to verify |
| - for | regression/bug_2025_09_30_for_loop_segfault.nano | ✅ |
| - return | negative/return_errors/ | ✅ |
| - match | unit/test_unions_match_comprehensive.nano | ✅ |
| **Operations** | | |
| - Arithmetic (+, -, *, /, %) | ❓ Need comprehensive test |
| - Comparison (==, !=, <, <=, >, >=) | ❓ Need comprehensive test |
| - Logical (and, or, not) | ❓ Need comprehensive test |
| **Special Features** | | |
| - Shadow tests | All tests | ✅ |
| - Type checking | negative/type_errors/ | ✅ |
| - Mutability | negative/mutability_errors/ | ✅ |
| - Namespacing | integration/test_namespacing.nano | ✅ |
| - Modules/Imports | integration/test_modules.nano | ✅ |

### ❌ Missing Critical Tests

1. **Comprehensive Operations Test** - Need test covering all arithmetic, comparison, and logical operations
2. **While Loop Test** - Need explicit while loop test
3. **Standard Library Test** - Need comprehensive stdlib coverage test

### 📝 Test Categories

- **Unit Tests** (`tests/unit/`): 6 comprehensive tests ✅
- **Integration Tests** (`tests/integration/`): 4 tests ✅
- **Negative Tests** (`tests/negative/`): 13+ error condition tests ✅
- **Regression Tests** (`tests/regression/`): 1 test ✅
- **Performance Tests** (`tests/performance/`): 0 tests (optional)
- **Warning Tests** (`tests/warnings/`): 2 tests ✅

**Total**: 48 test files

## Action Items

1. Create `unit/test_operators_comprehensive.nano` - All operators
2. Create `unit/test_control_flow.nano` - while, for, if/else
3. Create `unit/test_stdlib_comprehensive.nano` - All 37 stdlib functions
4. Run all tests with interpreter
5. Run all tests with compiler

