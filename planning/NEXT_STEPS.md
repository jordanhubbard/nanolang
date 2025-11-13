# Union Types - Next Steps

**Current Status:** 70% Complete, Type Annotation Blocker Identified  
**Time Remaining:** 5-8 hours  
**Branch:** `feature/union-types`

---

## Immediate Next Steps

### 1. Fix Type Annotations (CRITICAL) - 1-2 hours

**Problem:** Parser can't recognize union types in function signatures and let statements.

**Solution:** Update parser to track union type names (similar to struct handling).

**Files to Modify:**
- `src/parser.c` - `parse_type()` function
- `src/nanolang.h` - Add union_name fields to AST nodes
- `src/typechecker.c` - Handle union type resolution

**Implementation:**
```c
// In parse_type(), treat identifiers as potential unions too
case TOKEN_IDENTIFIER:
    // Could be struct OR union type
    type = TYPE_STRUCT;  // Will be resolved in type checker
    advance(p);
    return type;
```

Then in type checker, check both struct and union registries when resolving types.

---

### 2. Implement Transpiler - 3-4 hours

Once type annotations work, implement C code generation:

**A. Union Definition Generation (1 hour)**
```c
// Generate tag enum
typedef enum {
    UNIONNAME_TAG_VARIANT1,
    UNIONNAME_TAG_VARIANT2
} UnionName_Tag;

// Generate tagged union struct
typedef struct UnionName {
    UnionName_Tag tag;
    union {
        struct {...} variant1;
        struct {...} variant2;
    } data;
} UnionName;
```

**B. Union Construction Generation (1 hour)**
```c
// Generate construction code
UnionName value = {
    .tag = UNIONNAME_TAG_VARIANT,
    .data.variant = { fields... }
};
```

**C. Match Expression Generation (1-2 hours)**
```c
// Generate switch statement
switch (expr.tag) {
    case UNIONNAME_TAG_VARIANT1: {
        // binding = expr.data.variant1
        // arm body
    } break;
    // ...
}
```

---

### 3. Complete Testing - 1-2 hours

**A. Union Construction Tests**
- Empty variant construction
- Variant with single field
- Variant with multiple fields
- All tests should compile and run

**B. Match Expression Tests**
- Simple match (2 variants)
- Complex match (many variants)
- Match with field access
- Match return values

**C. Integration Tests**
- Recursive unions (AST-like structures)
- Mixed struct + union programs
- Complex pattern matching scenarios

---

## Success Criteria

Before merging to main:
- [ ] All 10+ tests passing
- [ ] No compiler warnings
- [ ] Documentation updated
- [ ] Examples added
- [ ] Self-hosting test (compile lexer with unions)

---

## Timeline

**Session 1 (Done):** 6 hours
- Lexer ✅
- Parser ✅  
- Type Checker ✅
- Initial Tests ✅

**Session 2 (Next):** 5-8 hours
- Fix type annotations (1-2h)
- Implement transpiler (3-4h)
- Complete testing (1-2h)

**Total:** 11-14 hours (under original 15-20h estimate!)

---

## Commands to Run

```bash
# After fixing type annotations
cd /Users/jordanh/Src/nanolang
make clean && make
tests/unit/unions/test_runner.sh

# After transpiler
./bin/nanoc tests/unit/unions/05_union_construction_empty.nano -o /tmp/test
/tmp/test  # Should return 1

# Full test suite
make test
```

---

**Status:** Ready to continue implementation 🚀

