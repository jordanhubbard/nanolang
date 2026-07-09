# eval.c Refactoring Plan

**Status:** Planned (Not Yet Implemented)  
**Priority:** High  
**Complexity:** Very High  
**Estimated Effort:** 16-24 hours  

## Problem

`src/eval.c` is currently **5,943 lines** - too large for maintainability. It contains:
- Interpreter core logic
- HashMap implementation
- All stdlib function implementations (math, string, IO, file, process operations)
- Shadow test execution
- Expression/statement evaluation

## Goal

Split `eval.c` into focused, maintainable modules while preserving all functionality.

## Proposed Structure

```
src/
├── eval/
│   ├── eval_core.c          # Main interpreter loop, expression/statement eval
│   ├── eval_core.h
│   ├── eval_hashmap.c       # HashMap interpreter implementation
│   ├── eval_hashmap.h
│   ├── eval_stdlib_math.c   # Math functions (sqrt, pow, sin, cos, etc.)
│   ├── eval_stdlib_math.h
│   ├── eval_stdlib_string.c # String operations
│   ├── eval_stdlib_string.h
│   ├── eval_stdlib_io.c     # File/directory/path/process operations
│   ├── eval_stdlib_io.h
│   └── eval_shadow.c        # Shadow test execution
│       eval_shadow.h
└── eval.c                   # Thin wrapper that includes all modules (for compatibility)
```

## Line Count Breakdown

Current `eval.c` sections:
- HashMap implementation: ~300 lines (lines 29-333)
- Forward declarations: ~200 lines (lines 333-529)
- File operations: ~180 lines (lines 529-709)
- Directory operations: ~54 lines (lines 709-763)
- Path operations: ~149 lines (lines 763-912)
- Process operations: ~205 lines (lines 912-1117)
- Print/value functions: ~165 lines (lines 1117-1282)
- Math functions: ~64 lines (lines 1282-1346)
- Trig functions: ~788 lines (lines 1346-2134)
- Dynamic array helpers: ~668 lines (lines 2134-2802)
- Boolean helpers: ~16 lines (lines 2802-2818)
- Prefix operations: ~536 lines (lines 2818-3354)
- Function calls: ~1387 lines (lines 3354-4741)
- Expression evaluation: ~485 lines (lines 4741-5226)
- Statement evaluation: ~362 lines (lines 5226-5588)
- Extern checking: ~59 lines (lines 5588-5647)
- Shadow tests: ~121 lines (lines 5647-5768)
- Main interpreter: ~43 lines (lines 5768-5811)
- Function by name: ~132 lines (lines 5811-5943)

**Target file sizes:**
- eval_core.c: ~1,500 lines (expression/statement eval, main loop)
- eval_hashmap.c: ~350 lines (HashMap runtime)
- eval_stdlib_math.c: ~900 lines (math + trig)
- eval_stdlib_string.c: ~400 lines (string operations)
- eval_stdlib_io.c: ~600 lines (file/dir/path/process)
- eval_shadow.c: ~200 lines (shadow test execution)

## Implementation Steps

### Phase 1: Preparation (2 hours)
1. Create `src/eval/` directory
2. Read eval.c fully to understand dependencies
3. Map all function dependencies
4. Identify shared types and data structures
5. Create header files with function declarations

### Phase 2: Extract HashMap (2 hours)
1. Move HashMap implementation to `eval_hashmap.c`
2. Create `eval_hashmap.h` with public interface
3. Update includes in eval.c
4. **Test:** Run interpreter tests
5. **Verify:** All HashMap examples work

### Phase 3: Extract Math Functions (2 hours)
1. Move math/trig functions to `eval_stdlib_math.c`
2. Create `eval_stdlib_math.h`
3. Update includes
4. **Test:** Run math examples
5. **Verify:** Shadow tests pass

### Phase 4: Extract String Operations (2 hours)
1. Move string functions to `eval_stdlib_string.c`
2. Create `eval_stdlib_string.h`
3. Update includes
4. **Test:** Run string examples
5. **Verify:** Shadow tests pass

### Phase 5: Extract IO Operations (3 hours)
1. Move file/directory/path/process ops to `eval_stdlib_io.c`
2. Create `eval_stdlib_io.h`
3. Update includes
4. **Test:** Run IO examples
5. **Verify:** File operations work

### Phase 6: Extract Shadow Tests (2 hours)
1. Move shadow test execution to `eval_shadow.c`
2. Create `eval_shadow.h`
3. Update includes
4. **Test:** Run full test suite
5. **Verify:** All shadow tests execute

### Phase 7: Core Cleanup (2 hours)
1. Keep expression/statement eval in eval_core.c
2. Move helper functions appropriately
3. Ensure clean module boundaries
4. **Test:** Run all tests
5. **Verify:** No regressions

### Phase 8: Build System Updates (2 hours)
1. Update Makefile.gnu to compile new files
2. Add new object files to linking
3. Update dependency tracking
4. **Test:** Clean build from scratch
5. **Verify:** All targets build

### Phase 9: Integration Testing (3 hours)
1. Run full test suite
2. Test all examples
3. Run coverage tests
4. Memory leak testing with valgrind
5. Performance benchmarking

## Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Breaking interpreter** | High | Test after each phase, keep backup |
| **Circular dependencies** | High | Design headers carefully, use forward declarations |
| **Build system complexity** | Medium | Update Makefile incrementally |
| **Performance regression** | Low | Benchmark before/after |
| **Missing dependencies** | Medium | Test thoroughly after each change |

## Testing Strategy

After each phase:
1. ✅ Build succeeds without warnings
2. ✅ All unit tests pass
3. ✅ Shadow tests pass
4. ✅ Example programs run correctly
5. ✅ No memory leaks (valgrind clean)

## Rollback Plan

If refactoring causes issues:
1. Git revert to last working commit
2. Identify specific breaking change
3. Fix incrementally
4. Resume from failed phase

## Success Criteria

- [ ] eval.c reduced to <500 lines (thin wrapper)
- [ ] All new files compile without warnings
- [ ] Full test suite passes
- [ ] All examples run correctly
- [ ] No performance regression (±5%)
- [ ] Clean valgrind run
- [ ] Documentation updated

## Dependencies

Must complete before starting:
- ✅ Full test suite passing
- ✅ No outstanding build issues
- ✅ Clean git working directory

## Follow-up Work

After refactoring:
- Update CONTRIBUTING.md with new structure
- Document module organization
- Add per-module tests
- Consider further splitting stdlib into src/stdlib/

---

**BEAD-006:** Split eval.c into focused modules  
**Assigned To:** (Unassigned)  
**Status:** Planned  
**Created:** January 25, 2026  
