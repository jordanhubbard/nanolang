# Stdlib Reorganization Plan

**Status:** Planned (Not Yet Implemented)  
**Priority:** High  
**Complexity:** High  
**Estimated Effort:** 12-18 hours  

## Problem

Standard library functions are currently scattered across multiple files:
- Core stdlib in `src/stdlib_runtime.c` (~800 lines)
- Math/string functions in `src/eval.c` (~1,500 lines)
- No clear organization by functionality
- Difficult to find and maintain specific functions

## Goal

Create a well-organized `src/stdlib/` directory with functions grouped by category, making stdlib easier to maintain and extend.

## Current Stdlib Organization

**In spec.json (72 functions across 9 categories):**
1. IO: 3 functions (print, println, assert)
2. Math: 11 functions (abs, min, max, sqrt, pow, floor, ceil, round, sin, cos, tan)
3. String: 18 functions (length, concat, substring, contains, equals, char_at, etc.)
4. BString: 12 functions (new, length, concat, substring, equals, byte_at, etc.)
5. Array: 10 functions (length, push, pop, at, remove_at, filter, map, reduce, etc.)
6. OS: 3 functions (getcwd, getenv, range)
7. Generics: 4 functions (List_new, List_push, List_get, List_length)
8. Checked Math: 5 functions (checked_add, checked_sub, checked_mul, checked_div, checked_mod)
9. HashMap: 6 functions (map_new, map_put, map_get, map_has, map_size, map_free)

**Current Implementation:**
- `src/stdlib_runtime.c`: Core functions (print, assert, array ops, some string ops)
- `src/eval.c`: Math, trig, additional string ops, file IO
- `runtime/`: GC, List<T> implementations
- No unified organization

## Proposed Structure

```
src/
├── stdlib/
│   ├── stdlib.h              # Main header (includes all)
│   ├── io.c                  # print, println, assert
│   ├── io.h
│   ├── math.c                # abs, min, max, sqrt, pow, floor, ceil, round
│   ├── math.h
│   ├── trig.c                # sin, cos, tan
│   ├── trig.h
│   ├── string.c              # str_* functions
│   ├── string.h
│   ├── bstring.c             # bstr_* functions  
│   ├── bstring.h
│   ├── array.c               # array_* functions
│   ├── array.h
│   ├── os.c                  # getcwd, getenv, range
│   ├── os.h
│   └── README.md             # Stdlib organization guide
├── stdlib_runtime.c          # Deprecated/compatibility wrapper
└── stdlib_runtime.h          # Deprecated/compatibility header
```

## Implementation Steps

### Phase 1: Infrastructure Setup (2 hours)
1. Create `src/stdlib/` directory
2. Create `src/stdlib/stdlib.h` master header
3. Create `src/stdlib/README.md` documentation
4. Update Makefile.gnu to build stdlib modules
5. **Test:** Build succeeds

### Phase 2: Extract IO Functions (1 hour)
**Functions:** print, println, assert

1. Move implementations to `src/stdlib/io.c`
2. Create `src/stdlib/io.h` with declarations
3. Update includes in other files
4. **Test:** Basic examples work

### Phase 3: Extract Math Functions (2 hours)
**Functions:** abs, min, max, sqrt, pow, floor, ceil, round

1. Move from stdlib_runtime.c and eval.c to `src/stdlib/math.c`
2. Create `src/stdlib/math.h`
3. Consolidate duplicate implementations
4. **Test:** Math examples work

### Phase 4: Extract Trig Functions (1 hour)
**Functions:** sin, cos, tan

1. Move from eval.c to `src/stdlib/trig.c`
2. Create `src/stdlib/trig.h`
3. **Test:** Trig examples work

### Phase 5: Extract String Functions (2 hours)
**Functions:** str_length, str_concat, str_substring, str_contains, str_equals, char_at, string_from_char, is_digit, is_alpha, is_alnum, is_whitespace, is_upper, is_lower, int_to_string, string_to_int, digit_value, char_to_lower, char_to_upper

1. Move from stdlib_runtime.c to `src/stdlib/string.c`
2. Create `src/stdlib/string.h`
3. **Test:** String examples work

### Phase 6: Extract BString Functions (2 hours)
**Functions:** bstr_* (12 functions)

1. Move from stdlib_runtime.c to `src/stdlib/bstring.c`
2. Create `src/stdlib/bstring.h`
3. **Test:** BString examples work

### Phase 7: Extract Array Functions (2 hours)
**Functions:** array_length, array_push, array_pop, at, array_remove_at, filter, map, reduce

1. Move from stdlib_runtime.c to `src/stdlib/array.c`
2. Create `src/stdlib/array.h`
3. **Test:** Array examples work

### Phase 8: Extract OS Functions (1 hour)
**Functions:** getcwd, getenv, range

1. Move from stdlib_runtime.c to `src/stdlib/os.c`
2. Create `src/stdlib/os.h`
3. **Test:** OS examples work

### Phase 9: Update Build System (2 hours)
1. Update Makefile.gnu with all new stdlib object files
2. Update dependency tracking
3. Remove/deprecate old stdlib_runtime.c
4. Update transpiler includes
5. **Test:** Clean build

### Phase 10: Integration Testing (3 hours)
1. Run full test suite
2. Test all 150+ examples
3. Verify stdlib function count matches spec.json
4. Memory leak testing
5. Performance benchmarking

## File Size Estimates

| File | Estimated Lines | Functions |
|------|-----------------|-----------|
| io.c | 80 | 3 |
| math.c | 250 | 11 |
| trig.c | 120 | 3 |
| string.c | 450 | 18 |
| bstring.c | 350 | 12 |
| array.c | 400 | 10 |
| os.c | 100 | 3 |
| **Total** | **~1,750** | **60** |

## Benefits

✅ **Organization:** Clear module structure by functionality  
✅ **Maintainability:** Easy to find and modify specific functions  
✅ **Extensibility:** Simple to add new functions to appropriate module  
✅ **Testing:** Can test modules independently  
✅ **Documentation:** Clearer stdlib documentation  
✅ **Code Review:** Easier to review changes per module  

## Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Breaking builds** | High | Test after each phase, incremental changes |
| **Missing functions** | High | Audit against spec.json, comprehensive tests |
| **Include path issues** | Medium | Use proper relative includes, test thoroughly |
| **Transpiler changes** | High | Update include generation carefully |
| **Circular dependencies** | Low | Design headers carefully |

## Testing Strategy

After each phase:
1. ✅ Build succeeds
2. ✅ Unit tests pass
3. ✅ Examples using affected functions work
4. ✅ No linker errors
5. ✅ No duplicate symbols

## Success Criteria

- [ ] All 72 stdlib functions organized in src/stdlib/
- [ ] Each module compiles independently
- [ ] Full test suite passes
- [ ] All 150+ examples work
- [ ] Clean build with no warnings
- [ ] Documentation updated
- [ ] spec.json audit passes

## Dependencies

Must complete first:
- ✅ Full test suite passing
- ✅ Clean git working directory
- ⏳ eval.c refactoring complete (or can do concurrently)

## Follow-up Work

After reorganization:
- Add per-module tests in tests/stdlib/
- Create stdlib API documentation
- Add benchmarks for stdlib functions
- Consider auto-generating stdlib docs from source

## Coordination with BEAD-006

This task is related to BEAD-006 (split eval.c). Options:

**Option A: Sequential** (Recommended)
1. Complete eval.c refactoring first
2. Then reorganize stdlib
3. Clearer separation of concerns

**Option B: Concurrent**
1. Do both simultaneously
2. Faster but riskier
3. More merge conflicts

**Recommendation:** Do eval.c refactoring first, then stdlib reorganization.

---

**BEAD-005:** Reorganize stdlib implementation structure  
**Assigned To:** (Unassigned)  
**Status:** Planned  
**Created:** January 25, 2026  
