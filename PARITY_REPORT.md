# Interpreter/Compiler Parity Report

**Date:** 2025-01-15  
**Status:** ⚠️ **PARITY ISSUES FOUND**

## Summary

Ran comprehensive parity check on all `.nano` files in `examples/` directory.

### Results
- **Total files tested:** 104
- **Both work:** 44 ✅
- **Interpreter only:** 17 ⚠️ (Compiler needs fixing)
- **Compiler only:** 0 ✅
- **Both fail:** 18 ❌ (Need investigation)
- **Skipped (external deps):** 25 (SDL, OpenGL, etc.)

## Critical Issues Found

### 1. Print Statement Parsing Issue

**Problem:** `print "hello"` statements are parsed as expression statements (just identifier `print`), not as `AST_PRINT` nodes.

**Affected Files:**
- `17_struct_test.nano`
- `18_enum_test.nano`
- Many others using `print` statements

**Error:** Compiler generates `print;` instead of `printf(...)` calls.

**Root Cause:** Parser treats `print "hello"` as an expression statement (identifier), not recognizing it as a print statement.

**Fix Needed:** Parser should detect `print` identifier followed by expression and create `AST_PRINT` node, OR examples should use `(print "hello")` function call syntax.

### 2. Extern Functions Not Implemented in Interpreter

**Problem:** Extern functions like `strlen`, `strcmp`, `strncmp` work in compiler but not in interpreter.

**Affected Files:**
- `23_extern_string.nano`
- `21_extern_math.nano`
- `22_extern_char.nano`

**Error:** "Built-in function 'strlen' not implemented in interpreter"

**Root Cause:** Extern functions are FFI calls that need to be linked. Interpreter doesn't have implementations.

**Fix Needed:** Either:
1. Implement extern functions in interpreter runtime
2. Or mark these as compiler-only examples

### 3. OS Functions Not Available in Compiler

**Problem:** OS functions like `file_read`, `file_write` work in interpreter but may not be available in compiled binaries.

**Affected Files:**
- `10_os_basic.nano`

**Fix Needed:** Ensure OS stdlib functions are available in compiled binaries.

## Files Requiring Investigation

### Interpreter Only (17 files)
1. `boids_complete.nano`
2. `17_struct_test.nano` - Print statement issue
3. `23_extern_string.nano` - Extern functions
4. `20_string_operations.nano`
5. `asteroids_simple_game.nano`
6. `29_generic_lists.nano`
7. `21_extern_math.nano` - Extern functions
8. `26_tictactoe.nano`
9. `language_features_demo.nano`
10. `24_random_sentence.nano`
11. `19_list_int_test.nano`
12. `22_extern_char.nano` - Extern functions
13. `25_pi_calculator.nano`
14. `primes.nano`
15. `10_os_basic.nano` - OS functions
16. `18_enum_test.nano` - Print statement issue
17. `new_features.nano`

### Both Fail (18 files)
1. `checkers_simple.nano`
2. `maze.nano`
3. `33_function_factories.nano`
4. `30_generic_list_basics.nano`
5. `protracker-clone/*.nano` (multiple files)
6. `test_wav_playback.nano`
7. `33_function_factories_v2.nano`
8. `test_enum_parse.nano`
9. `30_generic_list_point.nano`
10. `math_utils.nano`
11. `test_mod_loading.nano`
12. `checkers.nano`
13. `visualizer/mod_visualizer.nano`

## Recommendations

### High Priority
1. **Fix print statement parsing** - Most critical issue affecting many files
2. **Implement extern functions in interpreter** - Or document as compiler-only
3. **Fix OS stdlib in compiler** - Ensure file operations work

### Medium Priority
4. Investigate "both fail" files - May have syntax errors or missing features
5. Standardize print syntax - Either `print "hello"` or `(print "hello")`

### Low Priority
6. Add more comprehensive tests
7. Document compiler-only vs interpreter-only features

## Next Steps

1. Fix print statement parsing in parser
2. Test all affected files after fix
3. Implement extern function stubs in interpreter
4. Verify OS stdlib works in compiler
5. Re-run parity check
