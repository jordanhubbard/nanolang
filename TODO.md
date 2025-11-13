# Nanolang TODO List

**Last Updated:** November 13, 2025  
**Current Focus:** Union Types (70% complete)  
**Progress:** 6/20 tasks completed (30%)

---

## 🎯 Active Session: Union Types Implementation

**Status:** In Progress (70% complete)  
**Branch:** `feature/union-types`  
**Time Invested:** 6 hours  
**Time Remaining:** 5-8 hours

### ✅ Completed (6/20)

#### 1. ✅ Released v1.0.0 with production-ready compiler
- **Status:** Complete
- **Tag:** `v1.0.0`
- **Features:** 20/20 tests passing, Stage 0-1.5 working, production ready

#### 2. ✅ Union Types - Lexer
- **Status:** Complete
- **Time:** 0.5 hours
- **Deliverables:**
  - TOKEN_UNION keyword
  - TOKEN_MATCH keyword
  - TOKEN_ARROW reused for `=>`

#### 3. ✅ Union Types - Parser
- **Status:** Complete
- **Time:** 3 hours
- **Lines Added:** 450
- **Deliverables:**
  - `parse_union_def()` - Union definitions
  - `parse_union_construct()` - Union construction
  - `parse_match_expr()` - Pattern matching
  - AST structures with memory management
  - Free handlers for all union nodes

#### 4. ✅ Union Types - Type Checker
- **Status:** Complete
- **Time:** 2 hours
- **Lines Added:** 200
- **Deliverables:**
  - Union definition registration
  - Union construction validation
  - Match expression validation
  - Environment storage functions
  - TYPE_UNION integration

#### 5. ✅ Union Types - Test Infrastructure
- **Status:** Complete
- **Time:** 0.5 hours
- **Deliverables:**
  - 4 positive tests (all passing)
  - 2 negative tests (error detection)
  - Automated test runner (`tests/unit/unions/test_runner.sh`)
  - Test-driven development approach

#### 6. ✅ Repository Structure Cleanup
- **Status:** Complete
- **Time:** 0.5 hours
- **Deliverables:**
  - Root directory clean (only README.md)
  - RELEASE_NOTES moved to docs/
  - planning/README.md created
  - CLEANUP_COMPLETE.md documented

---

### 🚧 In Progress (3/20)

#### 7. 🚧 Union Types - Type Annotations
- **Status:** In Progress (blocked - needs fix)
- **Priority:** CRITICAL
- **Time Estimate:** 1-2 hours
- **Blocker:** Parser can't recognize union types in function signatures
- **Next Steps:**
  1. Update type resolution in type checker
  2. Look up type names in both struct and union registries
  3. Test with function return types and parameters
  4. Update error messages

**Problem:**
```nano
fn get_status() -> Status {  # ERROR: Parser treats 'Status' as struct
    return Status.Ok {}
}
```

**Solution:** Type checker needs to check both struct and union registries when resolving type names.

#### 8. 🚧 Union Types - Transpiler
- **Status:** Not Started (blocked by #7)
- **Priority:** HIGH
- **Time Estimate:** 3-4 hours
- **Dependencies:** Type annotations must be fixed first
- **Next Steps:**
  1. Generate C tag enum for each union (1h)
  2. Generate C tagged union struct (1h)
  3. Generate union construction code (1h)
  4. Generate match as switch statement (1-2h)

**Target C Code:**
```c
typedef enum {
    COLOR_TAG_RED,
    COLOR_TAG_BLUE
} Color_Tag;

typedef struct Color {
    Color_Tag tag;
    union {
        struct {} red;
        struct { int64_t intensity; } blue;
    } data;
} Color;
```

#### 9. 🚧 Union Types - Testing Completion
- **Status:** Partial (6/10+ tests)
- **Priority:** MEDIUM
- **Time Estimate:** 1-2 hours
- **Dependencies:** Transpiler must be complete
- **Next Steps:**
  1. Union construction tests
  2. Match expression tests
  3. Integration tests
  4. Recursive union tests (AST-like)

---

### 📋 Pending (11/20)

#### Language Extensions (v2.0 Features)

##### 10. ⏸️ Generic Types - Parser
- **Status:** Not Started
- **Priority:** LOW (after unions)
- **Time Estimate:** 10-15 hours
- **Description:** Parse `<T>` syntax for generic functions and types

##### 11. ⏸️ Generic Types - Type Checker
- **Status:** Not Started
- **Priority:** LOW (after unions)
- **Time Estimate:** 10-15 hours
- **Description:** Type substitution and monomorphization

##### 12. ⏸️ Generic Types - Transpiler
- **Status:** Not Started
- **Priority:** LOW (after unions)
- **Time Estimate:** 5-8 hours
- **Description:** Generate specialized versions for each type

##### 13. ⏸️ Generic Types - Testing
- **Status:** Not Started
- **Priority:** LOW (after unions)
- **Time Estimate:** 2-3 hours
- **Description:** Test generic lists and functions

#### File I/O Support

##### 14. ⏸️ File I/O - Runtime Functions
- **Status:** Not Started
- **Priority:** MEDIUM
- **Time Estimate:** 5-10 hours
- **Description:** Implement open, read, write, close functions

##### 15. ⏸️ File I/O - Compiler Integration
- **Status:** Not Started
- **Priority:** MEDIUM
- **Time Estimate:** 2-3 hours
- **Description:** Add builtins and type signatures

##### 16. ⏸️ File I/O - Testing
- **Status:** Not Started
- **Priority:** MEDIUM
- **Time Estimate:** 2-3 hours
- **Description:** Test file operations

#### Project Milestones

##### 17. ⏸️ v2.0 Integration Testing
- **Status:** Not Started
- **Priority:** HIGH (before release)
- **Time Estimate:** 3-5 hours
- **Description:** Test all language extensions together

##### 18. ⏸️ v2.0 Release
- **Status:** Not Started
- **Priority:** HIGH (final step)
- **Time Estimate:** 2-3 hours
- **Dependencies:** All v2.0 features complete
- **Description:** Tag, document, and release v2.0.0

#### Documentation (Future)

##### 19. ⏸️ Union Types Documentation
- **Status:** Planning docs complete, user docs needed
- **Priority:** MEDIUM
- **Time Estimate:** 2-3 hours
- **Description:** Add to SPECIFICATION.md, GETTING_STARTED.md, examples

##### 20. ⏸️ Generic Types Documentation
- **Status:** Not Started
- **Priority:** LOW
- **Time Estimate:** 2-3 hours
- **Description:** Add to language docs after implementation

---

## 📊 Progress Summary

### Overall Progress
```
Completed:   ████████░░░░░░░░░░░░ 6/20 (30%)
In Progress: ██░░░░░░░░░░░░░░░░░░ 3/20 (15%)
Pending:     ███████████░░░░░░░░░ 11/20 (55%)
```

### Union Types Progress (Current Focus)
```
✅ Lexer:           ████████████████████ 100%
✅ Parser:          ████████████████████ 100%
✅ Type Checker:    ████████████████████ 100%
✅ Tests:           ████████████████████ 100% (for implemented features)
🚧 Type Annotations: ░░░░░░░░░░░░░░░░░░░░ 0% (BLOCKER)
⏸️ Transpiler:       ░░░░░░░░░░░░░░░░░░░░ 0%
⏸️ Full Testing:     ░░░░░░░░░░░░░░░░░░░░ 0%

Overall: ██████████████░░░░░░ 70%
```

---

## 🎯 Next Session Priorities

### Immediate (Must Do)
1. **Fix Type Annotations** (1-2h) - CRITICAL BLOCKER
   - Update type resolution in type checker
   - Test with union types in function signatures
   - Verify let statements work with unions

### High Priority (Should Do)
2. **Implement Transpiler** (3-4h)
   - Generate C tag enum
   - Generate C tagged union struct
   - Generate union construction code
   - Generate match as switch statement

3. **Complete Testing** (1-2h)
   - Union construction tests
   - Match expression tests
   - Integration tests

### Medium Priority (Nice to Have)
4. **Merge to Main** (30min)
   - Final test run
   - Update documentation
   - Merge feature/union-types → main
   - Tag v1.1.0 or v2.0-beta

---

## 📈 Time Tracking

### Completed Work
| Task | Time Spent | Status |
|------|------------|--------|
| v1.0 Release | 60+ hours | ✅ Complete |
| Union Lexer | 0.5 hours | ✅ Complete |
| Union Parser | 3 hours | ✅ Complete |
| Union Type Checker | 2 hours | ✅ Complete |
| Union Tests | 0.5 hours | ✅ Complete |
| Repository Cleanup | 0.5 hours | ✅ Complete |
| **Total** | **66.5 hours** | **6/20 tasks** |

### Remaining Work
| Task | Time Estimate | Priority |
|------|---------------|----------|
| Union Type Annotations | 1-2 hours | CRITICAL |
| Union Transpiler | 3-4 hours | HIGH |
| Union Testing Complete | 1-2 hours | HIGH |
| Union Documentation | 2-3 hours | MEDIUM |
| Generic Types (all) | 30-40 hours | LOW |
| File I/O (all) | 10-15 hours | MEDIUM |
| v2.0 Release | 5-8 hours | HIGH |
| **Total** | **52-76 hours** | **14/20 tasks** |

---

## 🚀 Path to Completion

### Union Types (5-8 hours remaining)
```
Current → Type Annotations → Transpiler → Testing → Merge
  70%         1-2h              3-4h        1-2h     30m
                                                    ↓
                                                  DONE!
```

### v2.0 Release (Total: ~60-85 hours)
```
Union Types → Generic Types → File I/O → Integration → Release
   5-8h         30-40h         10-15h       3-5h       2-3h
```

---

## 📝 Notes

### Current Session
- **Branch:** `feature/union-types`
- **Tag:** `union-types-70pct`
- **Commits:** 16 on branch
- **Tests:** 6/6 passing (100% of implemented features)
- **Status:** Excellent progress, clear path forward

### Blockers
1. **Type Annotations** - Parser treats union names as struct names
   - **Impact:** Cannot use unions in function signatures or let statements
   - **Solution:** Update type checker to look up in both struct and union registries
   - **Estimated Fix:** 1-2 hours

### Quality Metrics
- ✅ Test-driven development
- ✅ Comprehensive documentation
- ✅ Clean repository structure
- ✅ All implemented features tested
- ✅ No compiler warnings

### Architecture Notes
- Parser/Type Checker separation working well
- Test-first approach caught blocker early
- Environment storage pattern scales well
- Transpiler will follow established patterns

---

## 🔗 References

### Documentation
- **Planning:** `/planning/` directory (35 files)
  - `SESSION_COMPLETE.md` - Latest session summary
  - `UNION_IMPLEMENTATION_SUMMARY.md` - Complete overview
  - `NEXT_STEPS.md` - Detailed next steps
- **User Docs:** `/docs/` directory (32 files)
  - `SPECIFICATION.md` - Language specification
  - `GETTING_STARTED.md` - Getting started guide

### Tests
- **Union Tests:** `tests/unit/unions/` (6 tests, all passing)
- **Test Runner:** `tests/unit/unions/test_runner.sh`

### Code
- **Lexer:** `src/lexer.c` (+10 lines)
- **Parser:** `src/parser.c` (+450 lines)
- **Type Checker:** `src/typechecker.c` (+200 lines)
- **Environment:** `src/env.c` (+50 lines)
- **Headers:** `src/nanolang.h` (+100 lines)

---

**Last Updated:** November 13, 2025  
**Next Update:** After union types completion  
**Status:** Ready for next session 🚀

