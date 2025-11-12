# Self-Hosting Implementation Plan

**Date:** November 12, 2025  
**Goal:** Make nanolang compiler self-hosting  
**Current Progress:** 4/6 essential features complete (67%)

---

## Executive Summary

We're 67% of the way to self-hosting! With structs and enums complete, we now need to implement the final two essential features:

1. **Dynamic Lists** - For storing collections of tokens, AST nodes, symbols
2. **Advanced String Operations** - For parsing and text manipulation

**Estimated Timeline:** 3-4 weeks to feature complete, then 8-12 weeks to rewrite compiler

---

## Current Status

### ✅ Completed Features (4/6)

| Feature | Status | Completion Date | Notes |
|---------|--------|----------------|-------|
| **Structs** | ✅ COMPLETE | Nov 2025 | Token, ASTNode, Symbol representation |
| **Enums** | ✅ COMPLETE | Nov 12, 2025 | TokenType, NodeType, etc. |
| **File I/O** | ✅ COMPLETE | Oct 2025 | Read source, write C output |
| **System Execution** | ✅ COMPLETE | Oct 2025 | Execute gcc via system() |

### ⏳ Remaining Features (2/6)

| Feature | Status | Priority | Estimated Time |
|---------|--------|----------|----------------|
| **Dynamic Lists** | 🚧 IN PROGRESS | P0 | 2-3 weeks |
| **Advanced Strings** | ⏸️ PLANNED | P1 | 1-2 weeks |

---

## Phase 1: Dynamic Lists (Weeks 1-3)

### Why Lists First?

Lists are the **foundation** for everything:
- Lexer returns `list_token`
- Parser returns `list_astnode`
- Environment stores `list_symbol`, `list_function`
- Without lists, we can't build collections

### Implementation Strategy: Specialized Lists

**Approach:** Create 4 specific list types (no generics)

1. `list_int` - For testing and basic collections
2. `list_string` - For string collections
3. `list_token` - For lexer output
4. `list_astnode` - For parser output

**Why no generics?** Simpler to implement, easier to debug, sufficient for self-hosting.

### Week 1: Basic List Infrastructure

#### Day 1-2: list_int Implementation
- [ ] Create `src/runtime/` directory
- [ ] Implement `src/runtime/list_int.h` (type definition + 12 functions)
- [ ] Implement `src/runtime/list_int.c` (dynamic array with growth)
- [ ] Add to type system (`TYPE_LIST_INT`)
- [ ] Update parser to recognize `list_int` type

**Functions to implement:**
```c
List_int* list_int_new(void);                      // Create empty list
List_int* list_int_with_capacity(int capacity);    // Pre-allocate
void list_int_push(List_int *list, int64_t value); // Append
int64_t list_int_pop(List_int *list);              // Remove last
int64_t list_int_get(List_int *list, int index);   // Access by index
void list_int_set(List_int *list, int index, int64_t value);  // Update
void list_int_insert(List_int *list, int index, int64_t value); // Insert
int64_t list_int_remove(List_int *list, int index); // Remove at index
int list_int_length(List_int *list);               // Get length
int list_int_capacity(List_int *list);             // Get capacity
bool list_int_is_empty(List_int *list);            // Check empty
void list_int_clear(List_int *list);               // Clear all
void list_int_free(List_int *list);                // Deallocate
```

#### Day 3: Type System Integration
- [ ] Add list types to `TYPE` enum
- [ ] Update `type_to_string()`
- [ ] Add list functions to builtin registry
- [ ] Type checker recognizes list operations

#### Day 4-5: Transpiler & Testing
- [ ] Transpiler includes runtime headers
- [ ] Transpiler handles list types in declarations
- [ ] Write C unit tests for list_int
- [ ] Write nanolang test: `examples/19_list_int_test.nano`
- [ ] All tests passing

### Week 2: String Lists

#### Day 1-2: list_string Implementation
- [ ] Implement `src/runtime/list_string.h`
- [ ] Implement `src/runtime/list_string.c`
  - Handle string copying (strdup)
  - Handle string cleanup (free in list_string_free)
- [ ] Add `TYPE_LIST_STRING` to type system

#### Day 3-4: Integration & Testing
- [ ] Register list_string builtins
- [ ] Update transpiler
- [ ] Write C unit tests
- [ ] Write nanolang test: `examples/20_list_string_test.nano`
- [ ] Test string ownership and memory safety

#### Day 5: Bug Fixes & Polish
- [ ] Memory leak testing (valgrind)
- [ ] Edge case testing (empty lists, bounds checks)
- [ ] Performance testing (large lists)

### Week 3: Struct Lists (Token & ASTNode)

#### Day 1-2: list_token Implementation
- [ ] Implement `src/runtime/list_token.h`
- [ ] Implement `src/runtime/list_token.c`
  - Handle Token struct copying
  - Consider shallow vs deep copy
- [ ] Add `TYPE_LIST_TOKEN` to type system

#### Day 3-4: list_astnode Implementation
- [ ] Implement `src/runtime/list_astnode.h`
- [ ] Implement `src/runtime/list_astnode.c`
  - Handle ASTNode struct copying
  - Handle recursive structures
- [ ] Add `TYPE_LIST_ASTNODE` to type system

#### Day 5: Integration Testing
- [ ] Test list of tokens in lexer simulation
- [ ] Test list of AST nodes in parser simulation
- [ ] Memory testing for struct lists
- [ ] Write `examples/21_list_advanced_test.nano`

---

## Phase 2: Advanced String Operations (Weeks 4-5)

### Why Strings Next?

The compiler needs extensive string manipulation:
- Character-by-character parsing
- String building for C output
- Substring extraction
- Character classification (isdigit, isalpha)

### Required String Operations

#### Character Access
```nano
fn char_at(s: string, index: int) -> int  # Return ASCII value
fn set_char(s: string, index: int, c: int) -> string  # Return new string
```

#### String Building
```nano
fn string_new(capacity: int) -> string
fn string_append(s1: string, s2: string) -> string
fn string_append_char(s: string, c: int) -> string
```

#### Character Classification
```nano
fn is_digit(c: int) -> bool
fn is_alpha(c: int) -> bool
fn is_alphanumeric(c: int) -> bool
fn is_whitespace(c: int) -> bool
```

#### String Parsing
```nano
fn string_to_int(s: string) -> int
fn int_to_string(n: int) -> string
fn string_split(s: string, delimiter: string) -> list_string
```

### Implementation Plan

#### Week 4: Core String Functions
- [ ] Implement character access functions
- [ ] Implement string builder pattern
- [ ] Add to type checker as builtins
- [ ] Test with example: `examples/22_string_advanced_test.nano`

#### Week 5: String Parsing & Utilities
- [ ] Character classification functions
- [ ] String conversion functions (to/from int)
- [ ] String splitting
- [ ] Integration testing

---

## Phase 3: Self-Hosting Compiler Rewrite (Weeks 6-18)

Once lists and strings are complete, we can rewrite the compiler in nanolang.

### Week 6-7: Lexer in nanolang
```nano
# src_nano/lexer.nano

fn tokenize(source: string) -> list_token {
    let mut tokens: list_token = (list_token_new)
    let mut i: int = 0
    let len: int = (str_length source)
    
    while (< i len) {
        let c: int = (char_at source i)
        
        if (is_digit c) {
            # Parse number token
            let tok: Token = (parse_number source i)
            (list_token_push tokens tok)
        }
        
        # ... more token types ...
        
        set i (+ i 1)
    }
    
    return tokens
}
```

### Week 8-10: Parser in nanolang
```nano
# src_nano/parser.nano

fn parse_expression(tokens: list_token, pos: int) -> ASTNode {
    let tok: Token = (list_token_get tokens pos)
    
    if (== tok.type TOKEN_NUMBER) {
        return (make_number_node tok.value)
    }
    
    # ... more parsing logic ...
}
```

### Week 11-13: Type Checker in nanolang
```nano
# src_nano/typechecker.nano

fn check_expression(node: ASTNode, env: Environment) -> Type {
    # Type checking logic
}
```

### Week 14-16: Transpiler in nanolang
```nano
# src_nano/transpiler.nano

fn transpile_to_c(program: ASTNode) -> string {
    # C code generation
}
```

### Week 17-18: Bootstrap & Integration
- [ ] Compile nanolang compiler with C compiler
- [ ] Use nanolang compiler to compile itself
- [ ] Verify output is identical (bootstrap successful)
- [ ] Performance benchmarking
- [ ] Final testing and bug fixes

---

## Success Criteria

### Phase 1 Complete When:
- ✅ All 4 list types implemented and tested
- ✅ Can create and manipulate lists in nanolang
- ✅ Memory safe (no leaks, bounds checked)
- ✅ Examples demonstrate all list operations
- ✅ Documentation complete

### Phase 2 Complete When:
- ✅ All string operations implemented
- ✅ Can parse character-by-character
- ✅ String building works efficiently
- ✅ Examples demonstrate string manipulation
- ✅ Documentation complete

### Phase 3 Complete When:
- ✅ Entire compiler rewritten in nanolang
- ✅ nanolang compiler can compile itself
- ✅ Bootstrap process works reliably
- ✅ Output binaries functionally equivalent to C version
- ✅ Performance within 2-3x of C compiler
- ✅ All tests pass (200+ shadow tests)

---

## Risk Assessment

### High Risk Items
1. **List Memory Management** - Potential for leaks
   - Mitigation: Extensive valgrind testing
   
2. **Bootstrap Complexity** - Self-compilation edge cases
   - Mitigation: Incremental testing, small compiler first

3. **Performance** - Self-hosted compiler might be slow
   - Mitigation: Profile and optimize, acceptable if < 3x slower

### Medium Risk Items
1. **String Operations** - Character encoding issues
   - Mitigation: ASCII only for now
   
2. **Test Coverage** - Hard to test compiler internals
   - Mitigation: Comprehensive shadow tests

---

## Resource Requirements

### Development Time
- **Lists:** 2-3 weeks (120-160 hours)
- **Strings:** 1-2 weeks (60-80 hours)
- **Compiler Rewrite:** 8-12 weeks (320-480 hours)
- **Total:** 11-17 weeks (500-720 hours)

### Code Size
- **Lists:** ~800 lines (runtime + integration)
- **Strings:** ~400 lines (builtins + integration)
- **Compiler in nano:** ~4,000 lines (from 3,200 C lines)
- **Total new code:** ~5,200 lines

---

## Next Immediate Steps

### This Week:
1. ✅ Create `src/runtime/` directory
2. ✅ Implement `list_int.h` and `list_int.c`
3. ✅ Add to type system and parser
4. ✅ Write first test: `examples/19_list_int_test.nano`
5. ✅ Get one list type working end-to-end

### This Month:
1. Complete all 4 list types
2. Begin string operations
3. Write comprehensive tests
4. Document list API

### This Quarter:
1. Complete all prerequisites (lists + strings)
2. Begin lexer rewrite
3. Achieve first self-hosted milestone

---

## Tracking & Metrics

### Key Performance Indicators (KPIs)
- **Feature Completion:** 4/6 → 6/6 (Target: 3 weeks)
- **Test Coverage:** Maintain 95%+ pass rate
- **Memory Safety:** 0 leaks in valgrind
- **Performance:** Lists operations O(1) amortized
- **Documentation:** 100% API coverage

### Weekly Updates
- Update SELF_HOSTING_PROGRESS.md
- Track blockers and risks
- Adjust timeline as needed

---

## Conclusion

We're well-positioned for self-hosting success:
- ✅ Core language features complete
- ✅ Data structures (structs, enums, arrays) working
- ✅ I/O and system interaction ready
- 🚧 Just need lists and strings
- 🎯 Then ready to rewrite compiler

**Next Action:** Begin list_int implementation immediately.

---

**Status:** 🟢 READY TO PROCEED  
**Confidence:** 🟢 HIGH (Clear path, proven patterns)  
**Timeline:** 🟡 OPTIMISTIC (3-4 months total)

