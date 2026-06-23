# Remaining TODOs and Next Steps

**Last Updated:** November 12, 2025  
**Status:** After tracing system implementation and documentation organization

---

## Code TODOs

### 1. Function Call Arguments Parsing (Low Priority)
**Location:** `src/interpreter_main.c`
- **TODO:** Parse and convert call_args to proper argument values
- **TODO:** Parse additional arguments after function name
- **Status:** Interpreter currently only supports calling functions with no arguments via `--call`
- **Impact:** Low - workaround exists (call functions directly in code)
- **Estimated Effort:** 2-4 hours

### 2. Array Element Type Storage (Medium Priority)
**Location:** `src/parser.c`
- **TODO:** Store element_type somewhere - for now just return TYPE_ARRAY
- **Status:** Array type propagation partially implemented in typechecker
- **Impact:** Medium - affects array type checking accuracy
- **Estimated Effort:** 4-6 hours

---

## Self-Hosting Roadmap

### Phase 1: Essential Features ✅ COMPLETE (6/6)

All essential features for self-hosting are **complete**:

1. ✅ **Structs** - Implemented (November 2025)
2. ✅ **Enums** - Implemented (November 2025)
3. ✅ **Dynamic Lists** - `list_int` and `list_string` implemented
4. ✅ **File I/O** - Complete via stdlib (`file_read`, `file_write`, etc.)
5. ✅ **Advanced String Operations** - 13+ functions implemented
6. ✅ **System Execution** - Complete via stdlib (`system` function)

### Phase 2: Rewrite Compiler Components (Not Started)

The next major milestone is rewriting the compiler in nanolang:

1. ⏳ **Lexer** - Rewrite tokenization in nanolang
   - **Status:** Not started
   - **Estimated Effort:** 2-3 weeks
   - **Dependencies:** None (all features available)

2. ⏳ **Parser** - Rewrite AST generation in nanolang
   - **Status:** Not started
   - **Estimated Effort:** 3-4 weeks
   - **Dependencies:** Lexer complete

3. ⏳ **Type Checker** - Rewrite type checking in nanolang
   - **Status:** Not started
   - **Estimated Effort:** 4-5 weeks
   - **Dependencies:** Parser complete

4. ⏳ **Transpiler** - Rewrite C code generation in nanolang
   - **Status:** Not started
   - **Estimated Effort:** 3-4 weeks
   - **Dependencies:** Type checker complete

5. ⏳ **Main Driver** - Rewrite compiler driver in nanolang
   - **Status:** Not started
   - **Estimated Effort:** 1-2 weeks
   - **Dependencies:** All components complete

**Total Estimated Effort:** 13-18 weeks (3-4.5 months)

### Phase 3: Bootstrap (Not Started)

1. ⏳ **Bootstrap Level 1** - Compile nanolang compiler with itself
2. ⏳ **Bootstrap Level 2+** - Verify fixed point
3. ⏳ **Testing & Optimization** - Performance tuning

**Estimated Effort:** 4-6 weeks

---

## Documentation TODOs

### 1. Update Outdated Status Documents
- **SELF_HOSTING_CHECKLIST.md** - Shows 0% progress but should show 100% for essential features
- **SELF_HOSTING_IMPLEMENTATION_PLAN.md** - Shows 4/6 features but should show 6/6
- **ROADMAP.md** - Has inconsistent checkmarks (says 6/6 complete but lists items as unchecked)

**Action:** Update these documents to reflect actual implementation status

### 2. Consolidate Self-Hosting Documentation
- Multiple documents cover similar ground:
  - `docs/SELF_HOSTING_REQUIREMENTS.md`
  - `docs/SELF_HOSTING_CHECKLIST.md`
  - `docs/SELF_HOSTING_FEATURE_GAP.md`
  - `planning/SELF_HOSTING_IMPLEMENTATION_PLAN.md`
- **Action:** Review for duplication and consolidate

---

## Feature Enhancements (Future)

### Language Features
- [ ] More array operations (map, filter, reduce, slice)
- [ ] Generics/templates (for true generic lists)
- [ ] Pattern matching
- [ ] Modules/imports
- [ ] Error handling (Result type)
- [ ] Tuples

### Tooling
- [ ] REPL (Read-Eval-Print Loop)
- [ ] Language server (LSP)
- [ ] Debugger integration
- [ ] Package manager
- [ ] Build system improvements
- [ ] Documentation generator

### Optimizations
- [ ] Tail call optimization
- [ ] Constant folding
- [ ] Dead code elimination
- [ ] Inlining
- [ ] LLVM backend (alternative to C)

---

## Immediate Next Steps (Priority Order)

1. **Fix Documentation Inconsistencies** (1-2 hours)
   - Update self-hosting status documents
   - Ensure all docs reflect actual implementation state

2. **Complete Array Element Type Storage** (4-6 hours)
   - Finish array type propagation implementation
   - Improve type checking accuracy

3. **Start Lexer Rewrite** (2-3 weeks)
   - Begin Phase 2 of self-hosting
   - First compiler component to rewrite in nanolang

4. **Function Call Arguments** (2-4 hours)
   - Low priority but would improve interpreter usability
   - Nice-to-have feature

---

## Notes

- **Tracing System:** ✅ Complete and ready to use
- **Documentation Organization:** ✅ Complete with `.cursorrules` policy
- **Essential Features:** ✅ All 6 features complete (100%)
- **Self-Hosting:** Ready to begin Phase 2 (rewrite compiler components)

The project is in excellent shape! All essential features are complete, and we're ready to begin the exciting work of rewriting the compiler in nanolang itself.

