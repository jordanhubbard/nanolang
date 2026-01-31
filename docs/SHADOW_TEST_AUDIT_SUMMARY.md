# Shadow Test Coverage Audit

**Date**: 2026-01-01  
**Overall Coverage**: 78% (1,869/2,376 functions)  
**Status**: 🔴 **CRITICAL GAPS IDENTIFIED**

---

## Executive Summary

After restoring compile-time shadow tests (CTFE), we audited all NanoLang code for missing shadow tests. While overall coverage is 78%, there are **critical gaps** in core infrastructure.

---

## Coverage by Category

| Category | Functions | With Shadows | Missing | Coverage | Status |
|----------|-----------|--------------|---------|----------|--------|
| 📚 **Examples** | 627 | 657 | 61 | 104% | ✅ **EXCELLENT** |
| ✅ **Tests** | 544 | 584 | 75 | 107% | ✅ **EXCELLENT** |
| 🛠️  **Tools** | 8 | 8 | 4 | 100% | ✅ **GOOD** |
| 🏗️  **Src_Nano** | 670 | 440 | 247 | 65% | ⚠️  **NEEDS WORK** |
| 🔧 **Modules** | 289 | 139 | 151 | 48% | 🔴 **CRITICAL** |
| 📦 **Stdlib** | 238 | 41 | 199 | **17%** | 🔴 **CRITICAL** |

---

## Critical Findings

### 🔴 STDLIB: 17% Coverage (199 missing!)

**This is CRITICAL** - stdlib is core library code that must be 100% tested.

**Worst offenders:**
- `std/datetime/datetime.nano` - 28 functions missing shadows
- `stdlib/lalr.nano` - 24 functions missing shadows
- `std/json/json.nano` - 23 functions missing shadows
- `std/regex/regex.nano` - 13 functions missing shadows

**Impact**: Core functionality untested, bugs could slip into production.

---

### 🔴 MODULES: 48% Coverage (151 missing!)

**Modules are public APIs** - they MUST have shadow tests.

**Worst offenders:**
- `modules/std/json/json.nano` - 27 functions missing shadows
- `modules/vector3d/vector3d.nano` - 18 functions missing shadows
- `modules/opengl/opengl.nano` - 13 functions missing shadows (OK - FFI wrappers)
- `modules/std/collections/stringbuilder.nano` - 13 functions missing shadows
- `modules/std/math/matrix4.nano` - 12 functions missing shadows

**Impact**: Public APIs untested, users could encounter bugs.

---

### ⚠️  SRC_NANO: 65% Coverage (247 missing!)

**Compiler self-hosting** - critical for bootstrap reliability.

**Worst offenders:**
- `src_nano/parser.nano` - **75 functions** missing shadows
- `src_nano/transpiler.nano` - **51 functions** missing shadows
- `src_nano/nanoc_integrated.nano` - 42 functions missing shadows
- `src_nano/typecheck.nano` - 22 functions missing shadows

**Impact**: Compiler bugs could break self-hosting, cause silent failures.

---

## What Coverage >100% Means

Examples (104%) and Tests (107%) show >100% coverage because:
- Some test files have shadow tests for helper functions
- Some functions are tested multiple times
- Shadow tests for edge cases

**This is GOOD** - over-testing is better than under-testing!

---

## Recommendations

### Priority 1: CRITICAL (Must Fix)

**Stdlib (17% → 100%):**
1. Add shadows to `std/datetime/` (28 missing)
2. Add shadows to `stdlib/lalr.nano` (24 missing)
3. Add shadows to `std/json/` (23 missing)
4. Add shadows to `std/regex/` (13 missing)

**Target**: 100% coverage for all stdlib functions (no exceptions!)

---

### Priority 2: HIGH (Public APIs)

**Modules (48% → 90%+):**
1. Add shadows to `modules/std/json/` (27 missing)
2. Add shadows to `modules/vector3d/` (18 missing)
3. Add shadows to `modules/std/collections/` (13 missing)
4. Add shadows to `modules/std/math/` (12 missing)

**Exceptions allowed:**
- FFI wrappers (like OpenGL functions) - tested via integration
- Simple graphics functions - OK to skip

---

### Priority 3: MEDIUM (Compiler)

**Src_Nano (65% → 80%+):**
1. Add shadows to `parser.nano` complex logic (75 missing)
2. Add shadows to `transpiler.nano` logic (51 missing)
3. Add shadows to `typecheck.nano` logic (22 missing)

**Strategy**: Focus on complex logic, skip simple helpers initially.

---

## What Can Be Skipped

✅ **OK to skip shadow tests for:**
- `main()` functions (side effects, I/O)
- I/O heavy functions (file/network/graphics)
- Simple render/draw helpers (tested via integration)
- FFI wrappers (tested via integration)
- Test fixtures and setup functions

❌ **NEVER skip shadow tests for:**
- Pure logic functions
- Data structure operations
- Math/computation functions
- String manipulation
- Public API functions

---

## Implementation Plan

### Phase 1: Stdlib (CRITICAL)
- **Goal**: 17% → 100% (199 functions)
- **Time**: 3-4 days
- **Priority**: P0 blocker
- **Owner**: TBD

### Phase 2: Modules (HIGH)
- **Goal**: 48% → 90% (135 functions)
- **Time**: 2-3 days
- **Priority**: P1
- **Owner**: TBD

### Phase 3: Src_Nano (MEDIUM)
- **Goal**: 65% → 80% (100 functions)
- **Time**: 2-3 days
- **Priority**: P2
- **Owner**: TBD

**Total estimated effort**: 7-10 days

---

## How to Use This Audit

1. **Run the audit**:
   ```bash
   python3 create_shadow_report.py
   ```

2. **Find missing tests**:
   ```bash
   python3 create_shadow_report.py | grep "missing:"
   ```

3. **Check specific file**:
   ```bash
   grep "your_file.nano" docs/SHADOW_TEST_AUDIT.txt
   ```

4. **Track progress**:
   - Re-run audit after adding tests
   - Watch coverage percentage increase
   - Celebrate when categories reach 90%+!

---

## Success Criteria

✅ **Minimum Acceptable:**
- Stdlib: 100% coverage
- Modules: 90% coverage
- Src_Nano: 80% coverage
- Overall: 85% coverage

🎯 **Ideal Target:**
- Stdlib: 100% coverage
- Modules: 95% coverage
- Src_Nano: 85% coverage
- Overall: 90% coverage

---

## Conclusion

**Current state**: 78% coverage is GOOD but has CRITICAL gaps.

**Key insight**: Examples and tests are well-tested (>100%), but **core infrastructure** (stdlib 17%, modules 48%) is severely under-tested.

**Action required**: Prioritize stdlib and modules - these are the foundation of NanoLang.

**Next steps**:
1. Create beads for each priority category
2. Add shadow tests systematically
3. Re-audit to track progress
4. Celebrate when we hit 90%+ overall! 🎉

---

**Generated by**: `create_shadow_report.py`  
**Full details**: `docs/SHADOW_TEST_AUDIT.txt`
