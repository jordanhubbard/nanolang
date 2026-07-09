# Transpiler Refactoring Plan

**Epic**: nanolang-6rs  
**Goal**: Reduce transpile_to_c() from 2,324 lines to <1,000 lines  
**Status**: 5/12+ tasks complete (42% done)  
**Last Updated**: 2025-12-16

## Overview

The transpile_to_c() function is 2,324 lines (36% of transpiler.c, 23% of entire codebase). This makes it:
- Hard to understand and modify
- Difficult to test in isolation
- Prone to bugs and maintenance issues
- A barrier to new contributors

## Completed Work (Session 2025-12-16)

### ✅ nanolang-sjk: Header Generation (DONE)
- **Commit**: bf73e48
- **Lines**: Extracted ~35 lines
- **Function**: `generate_c_headers(sb)`
- **Impact**: Clean separation of header includes

### ✅ nanolang-0hq: List Specialization (DONE)
- **Commit**: decf558
- **Lines**: Extracted ~100 lines into 2 helpers
- **Functions**: `generate_list_specializations(env, sb)`, `generate_list_implementations(env, sb)`
- **Impact**: List generation fully isolated

### ✅ nanolang-y74: Type Definitions (DONE - 3 commits)
- **Commits**: 72e03f4 (enums), e665814 (structs), b8fd82a (unions)
- **Lines**: Extracted ~150 lines into 3 helpers
- **Functions**: `generate_enum_definitions()`, `generate_struct_definitions()`, `generate_union_definitions()`
- **Impact**: All type generation fully isolated and testable

**TOTAL EXTRACTED**: ~360 lines into 6 focused helper functions

## Remaining Work

### High Priority Extractions

#### 1. nanolang-0hq: List Specialization (~100 lines)
- **Lines**: 1516-1520, 1571-1608, 1613-1662
- **Complexity**: Medium
- **Function**: `generate_list_specializations(program, env, sb)`
- **Includes**: 
  - List type detection
  - Forward declarations
  - List_T struct definitions
  - List_T_new/push/get/length functions
- **Estimated**: 1-2 hours
- **Dependencies**: None

#### 2. nanolang-y74: Type Definitions (~150 lines)
- **Lines**: 1465-1490 (enums), 1524-1565 (structs), 1671-1725 (unions)
- **Complexity**: Medium (scattered across 3 sections)
- **Function**: `generate_type_definitions(env, sb)`
- **Includes**:
  - Enum typedefs
  - Struct typedefs with field handling
  - Union typedefs with variant handling
- **Estimated**: 2-3 hours
- **Dependencies**: None

#### 3. nanolang-1wz: Function Declarations (~250 lines)
- **Lines**: 1850-2100
- **Complexity**: Medium-High
- **Function**: `generate_function_declarations(program, env, sb)`
- **Includes**:
  - Extern function declarations
  - Module function declarations
  - Complex return type handling (struct/union/List)
- **Estimated**: 2-3 hours
- **Dependencies**: Type definitions must be done first

#### 4. nanolang-9s7: Function Implementations (~200 lines)
- **Lines**: 2100-2300
- **Complexity**: High (core transpilation logic)
- **Function**: `generate_function_implementations(program, env, sb)`
- **Includes**: Main transpilation loop
- **Estimated**: 2-3 hours
- **Dependencies**: Should be LAST extraction

### Stdlib Runtime Extraction (560 lines total)

**Two Approaches:**

#### Approach A: In-Function Helpers (Incremental)
Break down into 5 sub-tasks:

1. **nanolang-vx3**: File operations (~45 lines)
   - `generate_file_operations(sb)`
   - file_read, file_write, file_exists, file_delete
   
2. **nanolang-9v5**: Directory operations (~50 lines)
   - `generate_dir_operations(sb)`
   - dir_list, dir_create, dir_exists, dir_delete
   
3. **nanolang-jrn**: Path operations (~50 lines)
   - `generate_path_operations(sb)`
   - path_join, path_dirname, path_basename, path_absolute
   
4. **nanolang-41h**: String operations (~80 lines)
   - `generate_string_operations(sb)`
   - string_split, join, replace, trim, etc.
   
5. **nanolang-sxp**: Math/utility builtins (~290 lines)
   - `generate_math_utility_builtins(sb)`
   - Math functions, random, time, array ops

**Total Estimated**: 6-8 hours

#### Approach B: Separate File (Architectural)

**nanolang-dm7**: Create `stdlib_runtime.c`
- Move ALL stdlib runtime generation (lines 903-1462) to separate file
- Benefits:
  - Cleaner separation of concerns
  - Easier to test stdlib generation in isolation
  - Reduces transpiler.c by 560 lines immediately
  - Better code organization
- **Estimated**: 3-4 hours
- **Recommended**: This is the better long-term approach

### Deferred: Large Stdlib Extraction

#### nanolang-86x: Full Stdlib Runtime (560 lines)
- **Status**: Deferred pending approach decision
- **Blocker**: Too large for single manual extraction
- **Resolution**: Use Approach A (incremental) or B (separate file)

## Extraction Order

### Recommended Sequence

1. ✅ **nanolang-sjk**: Headers (DONE - bf73e48)
2. ✅ **nanolang-0hq**: List specializations (DONE - decf558)
3. ✅ **nanolang-y74**: Type definitions (DONE - 72e03f4, e665814, b8fd82a)
4. **nanolang-dm7**: Move stdlib to separate file (560 lines) **← RECOMMENDED NEXT**
   - **Alt**: Do nanolang-vx3, 9v5, jrn, 41h, sxp incrementally
5. **nanolang-ike** + **nanolang-1wz**: Function declarations (390 lines, needs 3 sub-tasks)
6. **nanolang-9s7**: Function implementations (200 lines, LAST)

**Total Lines to Extract**: ~1,260 lines  
**Expected Final Size**: ~1,000-1,100 lines  
**Reduction**: 45-55%

## Success Metrics

- [ ] transpile_to_c() under 1,000 lines
- [ ] All extractions have tests passing (62/62)
- [ ] No performance regression
- [ ] Code maintainability improved (subjective but measurable via reviews)
- [ ] New helper functions properly documented

## Testing Strategy

After each extraction:
1. `make clean && make` - verify compilation
2. `make test` - verify all 62 tests pass
3. Git commit with descriptive message
4. Close bead with summary

## Notes

- **Current file size**: 2,324 lines (transpiler.c)
- **Target reduction**: 1,260 lines extracted
- **Pattern established**: Extract → Test → Commit works well
- **Key learning**: Large extractions (500+ lines) need better tooling or separate file approach

## Related Issues

- **nanolang-n2z**: Memory safety epic (may inform refactoring)
- **nanolang-4u8**: Unit tests (can test extracted helpers)
- **nanolang-26q**: Module metadata (COMPLETE - provides foundation)

---

**Last Updated**: 2025-12-16  
**Document Owner**: Droid  
**Status**: Active planning
