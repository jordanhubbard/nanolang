# nanolang TODO List

**Last Updated:** November 15, 2025  
**Current Focus:** Full Generics Implementation Complete!  
**Progress:** Phase 1 Complete (100%) + Extended Generics (100%)

---

## 🎉 Phase 1: Essential Features - COMPLETE! (7/7)

All essential language features for self-hosting are now implemented:

### ✅ 1. Released v1.0.0 with production-ready compiler
- **Status:** Complete
- **Tag:** `v1.0.0`
- **Features:** 20/20 tests passing, production ready

### ✅ 2. Structs
- **Status:** Complete
- **Time Invested:** Implemented November 2025
- **Features:**
  - Struct definitions with typed fields
  - Struct literals with field initialization
  - Field access with dot notation
  - Type checking for struct fields
  - C code generation

### ✅ 3. Enums
- **Status:** Complete
- **Time Invested:** Implemented November 2025
- **Features:**
  - Enum definitions with named constants
  - Integer-based enum values
  - Enum variant access
  - Type checking for enums (treated as ints)
  - C code generation

### ✅ 4. Union Types (Tagged Unions)
- **Status:** Complete ✨
- **Time Invested:** ~12 hours total
- **Features:**
  - Union definitions with multiple variants
  - Each variant can have typed fields
  - Union construction with type safety
  - Pattern matching (basic implementation)
  - Full type checking and validation
  - C code generation with tagged unions
- **Deliverables:**
  - Lexer: TOKEN_UNION, TOKEN_MATCH, multi-line comments
  - Parser: parse_union_def(), parse_union_construct(), parse_match_expr()
  - Type Checker: Union type resolution and validation
  - Transpiler: C tag enums and tagged union structs
  - Tests: 5 unit tests (all passing)
  - Documentation: Updated SPECIFICATION.md, QUICK_REFERENCE.md
  - Example: examples/28_union_types.nano

### ✅ 5. Dynamic Lists
- **Status:** Complete
- **Features:**
  - `list_int` for integer lists
  - `list_string` for string lists
  - `list_token` for token lists
  - Operations: create, push, get, length, free

### ✅ 6. File I/O
- **Status:** Complete
- **Features:**
  - `file_read`, `file_write`, `file_append`
  - `file_exists`, `file_size`, `file_remove`
  - Directory operations
  - Complete OS standard library

### ✅ 7. String Operations
- **Status:** Complete
- **Features:**
  - 13+ string functions
  - Length, concat, substring, charAt
  - Search, replace, case conversion
  - All operations memory-safe

---

## 🎯 Phase 2: Self-Hosting (Next Major Milestone)

**Goal:** Rewrite the entire compiler in nanolang itself

**Total Estimate:** 13-18 weeks (260-360 hours)

### Step 1: Lexer Rewrite
- **Status:** ⏳ Not Started (READY TO BEGIN!)
- **Priority:** HIGH
- **Time Estimate:** 2-3 weeks (40-60 hours)
- **Description:** Rewrite `src/lexer.c` in nanolang
- **Input:** Source code string
- **Output:** `list_token` of parsed tokens
- **Dependencies:** None - all features available!
- **Key Tasks:**
  - Define `struct Token { type: int, value: string, line: int, column: int }`
  - Define `enum TokenType { ... }`
  - Implement `fn tokenize(source: string) -> list_token`
  - Character classification helpers
  - Keyword recognition
  - Number/string literal parsing
  - Comment handling
  - Comprehensive shadow tests

### Step 2: Parser Rewrite
- **Status:** ⏳ Not Started
- **Priority:** HIGH
- **Time Estimate:** 3-4 weeks (60-80 hours)
- **Description:** Rewrite `src/parser.c` in nanolang
- **Input:** `list_token`
- **Output:** AST (tree structure)
- **Dependencies:** Lexer complete
- **Key Tasks:**
  - Define AST node structs
  - Use unions for different node types
  - Recursive descent parser
  - Expression parsing
  - Statement parsing
  - Shadow tests for each production

### Step 3: Type Checker Rewrite
- **Status:** ⏳ Not Started
- **Priority:** HIGH
- **Time Estimate:** 4-5 weeks (80-100 hours)
- **Description:** Rewrite `src/typechecker.c` in nanolang
- **Input:** AST
- **Output:** Validated AST with type information
- **Dependencies:** Parser complete
- **Key Tasks:**
  - Symbol table management
  - Type inference and checking
  - Function signature validation
  - Scope management
  - Error reporting

### Step 4: Transpiler Rewrite
- **Status:** ⏳ Not Started
- **Priority:** HIGH
- **Time Estimate:** 3-4 weeks (60-80 hours)
- **Description:** Rewrite `src/transpiler.c` in nanolang
- **Input:** Typed AST
- **Output:** C source code (string)
- **Dependencies:** Type checker complete
- **Key Tasks:**
  - C code generation for all AST nodes
  - String building utilities
  - Indentation management
  - Type-to-C mapping
  - Runtime library integration

### Step 5: Main Driver
- **Status:** ⏳ Not Started
- **Priority:** HIGH
- **Time Estimate:** 1-2 weeks (20-40 hours)
- **Description:** Rewrite `src/main.c` in nanolang
- **Input:** Command line arguments
- **Output:** Compiled executable
- **Dependencies:** All components complete
- **Key Tasks:**
  - Orchestrate compilation pipeline
  - File I/O for source and output
  - Invoke system C compiler
  - Error handling and reporting
  - Command-line argument parsing

---

## 🔧 Phase 1.5: Quality of Life Improvements (Before Phase 2)

These improvements will make Phase 2 easier and more pleasant:

### ✅ A. Generics Support - COMPLETE!
- **Status:** ✅ Complete (November 15, 2025)
- **Priority:** MEDIUM-HIGH
- **Time Invested:** ~6 hours (much faster than estimate!)
- **Benefit:** Clean generic lists for any user-defined type!
- **Description:**
  - Full monomorphization: `List<Point>`, `List<Player>`, etc.
  - Automatic code generation for each instantiation
  - Type-safe specialized functions
  - Supports arbitrary user-defined struct types
  - Compile-time specialization (zero runtime overhead)
- **Implementation Completed:**
  1. ✅ Parser: Extended to handle `List<UserType>` syntax
  2. ✅ Type System: Added `TYPE_LIST_GENERIC` with type parameter tracking
  3. ✅ Type Checker: Instantiation registration and validation
  4. ✅ Transpiler: Generates specialized C code for each type
  5. ✅ Environment: Auto-registers specialized functions
  6. ✅ Testing: Verified with multiple instantiations
  7. ✅ Example: `examples/30_generic_list_basics.nano`
- **Documentation:** `planning/PHASE3_EXTENDED_GENERICS_COMPLETE.md`
- **Impact:** Self-hosted compiler can now use clean `List<Token>`, `List<ASTNode>` syntax!

### B. First-Class Functions (Without User-Visible Pointers)
- **Status:** ⏳ Not Started
- **Priority:** HIGH (Required before self-hosting)
- **Time Estimate:** 20-30 hours total
- **Benefit:** Enables functional programming patterns without pointers in user code!
- **Philosophy:** Functions as values WITHOUT exposing pointers - clean syntax, C implementation
- **Description:**
  - Pass functions to functions: `fn filter(items: array<int>, test: fn(int) -> bool) -> array<int>`
  - Return functions from functions: `fn get_op(choice: int) -> fn(int, int) -> int`
  - Store functions in variables: `let my_func: fn(int) -> int = double`
  - Call without dereferencing (transpiler handles pointer mechanics)
  - No `fn*` syntax - user writes `fn(int) -> bool`, transpiler generates function pointers
  - All types known at compile time (no dynamic dispatch)
  
- **Key Innovation:** User never sees pointers, but gets full higher-order function capabilities!

**Implementation Plan (3 Phases):**

#### B1. Functions as Parameters (10-12 hours)
- **Priority:** CRITICAL
- **Benefit:** Enables map, filter, fold patterns
- **Tasks:**
  1. Lexer/Parser: Recognize `fn(type1, type2) -> return_type` syntax (2h)
  2. Type System: Add `TYPE_FUNCTION` with signature info (3h)
  3. Type Checker: Validate function signature compatibility (2h)
  4. Transpiler: Generate C function pointer typedefs (2h)
  5. Transpiler: Convert function names to pointers at call sites (1h)
  6. Testing: map, filter, fold examples (2h)
  
**Example:**
```nano
fn filter(numbers: array<int>, predicate: fn(int) -> bool) -> array<int> {
    /* predicate is just called like any function */
}

fn is_positive(x: int) -> bool { return (> x 0) }
let positives: array<int> = (filter numbers is_positive)
```

#### B2. Functions as Return Values (5-8 hours)
- **Priority:** HIGH
- **Benefit:** Function factories, strategy pattern
- **Tasks:**
  1. Parser: Handle function types in return position (1h)
  2. Type Checker: Validate returned function signatures (2h)
  3. Transpiler: Generate correct return type (function pointer) (2h)
  4. Testing: Function factory examples (2h)
  
**Example:**
```nano
fn get_operation(choice: int) -> fn(int, int) -> int {
    if (== choice 0) { return add } else { return multiply }
}
let op: fn(int, int) -> int = (get_operation 1)
let result: int = (op 5 3)  /* Calls multiply */
```

#### B3. Function Variables (5-10 hours)
- **Priority:** MEDIUM-HIGH
- **Benefit:** Function dispatch tables, cleaner code organization
- **Tasks:**
  1. Type Checker: Allow function type in let statements (2h)
  2. Transpiler: Generate function pointer variables (2h)
  3. Integration: Works with generics (future: `fn map<T,U>(f: fn(T)->U)`) (3h)
  4. Testing: Comprehensive examples (3h)
  
**Example:**
```nano
let my_filter: fn(int) -> bool = is_positive
let result: array<int> = (filter numbers my_filter)
```

**Post-Implementation:**

#### B4. Documentation (3-5 hours)
- **Tasks:**
  1. `docs/FIRST_CLASS_FUNCTIONS.md` - User guide (2h)
  2. Update `docs/SPECIFICATION.md` - Syntax reference (1h)
  3. `examples/31_higher_order_functions.nano` - Comprehensive examples (2h)
  
#### B5. Code Audit for Optimization (5-8 hours)
- **Tasks:**
  1. Audit `src_nano/lexer_complete.nano` line by line (1h)
  2. Audit `src_nano/parser_*.nano` line by line (2h)
  3. Identify opportunities for map/filter/fold patterns (1h)
  4. Refactor using first-class functions (3h)
  5. Verify all shadow tests still pass (1h)

**Total Time:** 30-40 hours (including audit and documentation)

### C. Pattern Matching Improvements
- **Status:** ⏳ Not Started (Basic implementation exists)
- **Priority:** MEDIUM
- **Time Estimate:** 10-15 hours
- **Benefit:** Essential for working with union types in compiler
- **Description:**
  - Complete match expression implementation
  - Pattern binding (extract variant fields)
  - Exhaustiveness checking
  - Better error messages
- **Current Limitation:** Match exists but binding not fully implemented
- **Implementation:**
  1. Parser: Fix pattern binding syntax (3h)
  2. Type Checker: Validate patterns and bindings (4h)
  3. Transpiler: Generate correct C code for bindings (5h)
  4. Testing and examples (3h)

### D. Better Error Messages
- **Status:** ⏳ Not Started
- **Priority:** MEDIUM
- **Time Estimate:** 8-12 hours
- **Benefit:** Easier debugging during self-hosting
- **Description:**
  - Colored output (errors in red, warnings in yellow)
  - Show code snippet at error location
  - "Did you mean...?" suggestions
  - Better type mismatch messages
  - Stack traces for runtime errors

### D. Language Server Protocol (LSP)
- **Status:** ⏳ Not Started
- **Priority:** LOW
- **Time Estimate:** 40-60 hours
- **Benefit:** Editor integration (VSCode, etc.)
- **Description:**
  - Syntax highlighting
  - Auto-completion
  - Go-to-definition
  - Error checking as you type
  - Hover documentation
- **Note:** Nice to have but not essential for self-hosting

---

## 📊 Overall Progress Summary

```
Phase 1 (Essential Features):  ████████████████████ 100% (7/7)
Phase 1.5 (QoL Improvements):  ░░░░░░░░░░░░░░░░░░░░   0% (0/4)
Phase 2 (Self-Hosting):        ░░░░░░░░░░░░░░░░░░░░   0% (0/5)
Phase 3 (Bootstrap):           ░░░░░░░░░░░░░░░░░░░░   0% (0/3)

Overall Project: ████████░░░░░░░░░░░░ 37% (7/19 major tasks)
```

---

## 🎯 Recommended Roadmap

### Option A: Direct to Self-Hosting (Fastest)
```
Week 1-3:   Lexer Rewrite          →  Can tokenize nanolang source
Week 4-7:   Parser Rewrite         →  Can parse into AST  
Week 8-12:  Type Checker Rewrite   →  Can validate programs
Week 13-16: Transpiler Rewrite     →  Can generate C code
Week 17-18: Integration            →  Full compiler working
Week 19-24: Bootstrap              →  Self-hosting achieved!
```
**Total:** ~6 months to self-hosting

### Option B: QoL First, Then Self-Hosting (Cleaner)
```
Week 1-2:   Generics               →  Better abstractions
Week 3:     Pattern Matching       →  Cleaner union handling  
Week 4:     Better Errors          →  Easier debugging
----- Quality of Life Complete -----
Week 5-7:   Lexer Rewrite
Week 8-11:  Parser Rewrite
Week 12-16: Type Checker Rewrite
Week 17-20: Transpiler Rewrite
Week 21-22: Integration
Week 23-28: Bootstrap
```
**Total:** ~7 months to self-hosting (but cleaner codebase)

---

## 🚀 Immediate Next Actions

Choose your path:

### Path A: Start Self-Hosting Now
1. Create `src_nano/lexer.nano`
2. Define Token struct and TokenType enum
3. Implement tokenize() function
4. Write comprehensive shadow tests
5. Compile and test lexer in isolation

### Path B: QoL Improvements First
1. ✅ Update TODO.md (done!)
2. Implement Generics (30-40h)
   - Makes list handling much cleaner
   - Reduces code duplication
3. Complete Pattern Matching (10-15h)
   - Essential for compiler work
   - Makes union handling ergonomic
4. Then proceed to lexer rewrite

### Path C: Hybrid Approach
1. ✅ Update TODO.md (done!)
2. Implement Pattern Matching (10-15h)
   - Quick win, immediately useful
3. Start Lexer Rewrite
4. Add Generics later if needed

---

## 📈 Time Investment Summary

### Completed (Phase 1)
- v1.0.0 Release: 60+ hours
- Structs: ~40 hours  
- Enums: ~30 hours
- Union Types: ~12 hours
- Lists: ~30 hours
- File I/O: ~20 hours
- String Operations: ~15 hours
- **Total Phase 1:** ~207 hours ✅

### Remaining Work

**Phase 1.5 (QoL):**
- Generics: 30-40 hours
- Pattern Matching: 10-15 hours
- Better Errors: 8-12 hours
- LSP: 40-60 hours (optional)
- **Total Phase 1.5:** ~90-130 hours

**Phase 2 (Self-Hosting):**
- Lexer: 40-60 hours
- Parser: 60-80 hours
- Type Checker: 80-100 hours
- Transpiler: 60-80 hours
- Driver: 20-40 hours
- **Total Phase 2:** ~260-360 hours

**Phase 3 (Bootstrap):**
- ~80-120 hours

**Total Remaining:** ~430-610 hours

---

## 🔗 Key Resources

### Documentation
- [SPECIFICATION.md](docs/SPECIFICATION.md) - Language reference
- [SELF_HOSTING_REQUIREMENTS.md](docs/SELF_HOSTING_REQUIREMENTS.md) - Feature requirements
- [SELF_HOSTING_CHECKLIST.md](docs/SELF_HOSTING_CHECKLIST.md) - Implementation tracking

### Examples
- [examples/28_union_types.nano](examples/28_union_types.nano) - Union types demo
- [examples/17_struct_test.nano](examples/17_struct_test.nano) - Struct examples
- [examples/18_enum_test.nano](examples/18_enum_test.nano) - Enum examples

### Test Infrastructure
- `tests/unit/unions/` - 5 union tests
- `tests/unit/` - Unit tests for all features
- `make test` - Run full test suite (20/20 passing)

---

## 💡 Notes

### Why Self-Hosting?
1. **Proof of Language Completeness** - Can nanolang compile itself?
2. **Dogfooding** - Best way to find missing features
3. **Performance** - Self-hosted compiler can be optimized
4. **Independence** - No C dependency after bootstrap
5. **Credibility** - Self-hosting is a major milestone

### Why QoL First?
1. **Generics make everything cleaner** - One `list<T>` vs many specialized lists
2. **Pattern matching is essential** - Compiler works heavily with unions
3. **Better errors save time** - Debugging self-hosted code is hard
4. **Investment pays off** - Cleaner code = faster development

### Why Direct to Self-Hosting?
1. **Fastest path to milestone** - 6 months vs 7 months
2. **Can add QoL later** - Not blocking
3. **Momentum matters** - Strike while iron is hot
4. **Learn by doing** - Find missing features naturally

---

**Current Status:** 🎉 Phase 1 Complete!  
**Next Milestone:** Phase 2 - Self-Hosting  
**Estimated Completion:** 6-7 months (depending on path)

**Last Updated:** November 14, 2025  
**Ready for:** Self-hosting adventure! 🚀
