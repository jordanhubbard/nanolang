# Transpiler Audit → Beads Issues

**Date:** 2025-12-15  
**Audit Document:** [TRANSPILER_CODE_AUDIT_2025-12-15.md](TRANSPILER_CODE_AUDIT_2025-12-15.md)

---

## Epic

**nanolang-n2z**: Transpiler Memory Safety & Code Quality Improvements (P0, epic)

Comprehensive improvements based on audit that found 23 issues (8 CRITICAL, 6 HIGH, 5 MEDIUM, 4 LOW).

---

## Critical Issues (P0)

### nanolang-kg3: Add NULL checks after all malloc/calloc/realloc calls
- **Severity:** CRITICAL
- **Effort:** 4-6 hours
- **Problem:** 36 allocations, only 3 NULL checks (8% coverage)
- **Impact:** Segfault on out-of-memory instead of graceful failure
- **Files:** sb_create(), get_tuple_typedef_name(), get_function_typedef_name(), all registries

### nanolang-5qx: Fix unsafe strcpy/strcat in generated C code 🔥
- **Severity:** CRITICAL (HIGHEST PRIORITY)
- **Effort:** 2-3 hours
- **Problem:** Generated code uses strcpy/strcat → buffer overflows
- **Impact:** **Affects ALL compiled user programs**
- **Files:** transpiler.c:872-873 (dir listing), 1257-1258 (string concat)
- **Fix:** Replace with memcpy

### nanolang-5th: Fix realloc() error handling to prevent memory leaks
- **Severity:** CRITICAL
- **Effort:** 2 hours
- **Problem:** 6 realloc calls don't check return value
- **Impact:** Memory leak if realloc fails, crash on next use
- **Files:** transpiler.c:27, 69, 144, 297-298, 351-354, 1500

### nanolang-cyg: Add error propagation to transpiler functions
- **Severity:** CRITICAL
- **Effort:** 6-8 hours
- **Problem:** Many functions return void, can't signal errors
- **Impact:** Errors silently propagate until crash
- **Dependencies:** BLOCKS on nanolang-kg3 (NULL checks) and nanolang-5th (realloc fixes)
- **Fix:** Make key functions return bool/error codes

### nanolang-5uc: Fix integer overflow in StringBuilder buffer growth
- **Severity:** CRITICAL
- **Effort:** 1 hour
- **Problem:** `capacity *= 2` can overflow if capacity > INT_MAX/2
- **Impact:** Buffer overflow or allocation failure
- **Files:** transpiler.c:25-27

---

## High Priority Issues (P1)

### nanolang-1fz: Convert static buffers to dynamic allocation
- **Severity:** HIGH
- **Effort:** 3-4 hours
- **Problem:** Static buffers cause race conditions in concurrent use
- **Impact:** Thread-safety issues, incorrect behavior with multiple calls
- **Files:** transpiler.c:72, 86, 93, 535; iterative:176

### nanolang-l2j: Implement struct/union return type handling
- **Severity:** HIGH
- **Effort:** 8-12 hours
- **Problem:** TODO comment at transpiler.c:1874, currently skipped
- **Impact:** Link errors for programs with struct/union return types

---

## Medium Priority Issues (P2)

### nanolang-6rs: Refactor transpile_to_c() into smaller functions
- **Severity:** MEDIUM
- **Effort:** 8-12 hours
- **Problem:** transpile_to_c() is 1,458 lines (23% of codebase)
- **Impact:** Maintainability, testability
- **Related:** nanolang-cyg (error handling)

### nanolang-4u8: Add unit tests for transpiler components
- **Severity:** MEDIUM
- **Effort:** 12-16 hours
- **Problem:** No isolated tests for StringBuilder, registries, error paths
- **Dependencies:** BLOCKS on nanolang-cyg (error propagation)

---

## Dependency Graph

```
nanolang-n2z (Epic)
├── nanolang-kg3 (NULL checks) → [READY]
├── nanolang-5qx (unsafe strings) → [READY] 🔥 DO THIS FIRST
├── nanolang-5th (realloc) → [READY]
├── nanolang-5uc (overflow) → [READY]
├── nanolang-cyg (error propagation) → BLOCKED by kg3, 5th
├── nanolang-1fz (static buffers) → [READY]
├── nanolang-l2j (struct returns) → [READY]
├── nanolang-6rs (refactor) → [READY] (related to cyg)
└── nanolang-4u8 (tests) → BLOCKED by cyg
```

---

## Recommended Work Order

**Phase 1: Critical Fixes (8-13 hours)**
1. ✅ **nanolang-5qx** - Fix unsafe generated strings (2-3h) 🔥 HIGHEST IMPACT
2. **nanolang-kg3** - Add NULL checks (4-6h)
3. **nanolang-5th** - Fix realloc errors (2h)
4. **nanolang-5uc** - Fix integer overflow (1h)

**Phase 2: Error Handling (6-8 hours)**
5. **nanolang-cyg** - Add error propagation (6-8h)

**Phase 3: Thread Safety & Features (11-16 hours)**
6. **nanolang-1fz** - Static buffers (3-4h)
7. **nanolang-l2j** - Struct/union returns (8-12h)

**Phase 4: Code Quality (20-28 hours)**
8. **nanolang-6rs** - Refactor transpile_to_c() (8-12h)
9. **nanolang-4u8** - Add unit tests (12-16h)

---

## Quick Commands

```bash
# View ready work
bd ready

# View epic
bd show nanolang-n2z

# View specific issue
bd show nanolang-5qx

# Start work
bd update nanolang-5qx --status in_progress

# Complete work
bd close nanolang-5qx --reason "Fixed strcpy/strcat in generated code"

# View dependency tree
bd dep tree nanolang-n2z
```

---

## Total Effort Estimate

- **Critical (Phase 1):** 8-13 hours
- **Error Handling (Phase 2):** 6-8 hours
- **High Priority (Phase 3):** 11-16 hours
- **Medium Priority (Phase 4):** 20-28 hours
- **Total:** 45-65 hours

---

## Related Documents

- [TRANSPILER_CODE_AUDIT_2025-12-15.md](TRANSPILER_CODE_AUDIT_2025-12-15.md) - Full audit report
- [TRANSPILER_AUDIT_2025-12-15.md](TRANSPILER_AUDIT_2025-12-15.md) - Crash fix session
- Session that found and fixed:
  - ✅ Memory leaks in free_fn_type_registry()
  - ✅ Memory leaks in free_tuple_type_registry()
  - ✅ Double-free bug with outer_sig
  - ✅ NULL pointer dereference in function call handling

---

**Status:** Issues created 2025-12-15  
**Next Step:** Start with nanolang-5qx (unsafe generated strings) - highest impact, affects all users
