# Full Self-Hosting Implementation Plan

**Date:** November 29, 2025  
**Goal:** Nanolang compiler in nanolang that compiles itself and passes all tests

## Current State Assessment

### What We Have ✅
- Parser: Can parse functions, structs, enums, expressions, statements
- Type Checker: Basic type system, symbol tables, expression type checking
- Transpiler: Basic C code generation for functions and expressions
- Integration: Full pipeline connecting all components
- **Total:** ~4,200 lines of working compiler code

### What's Missing ❌

#### 1. Expression Types Not Implemented
- [ ] Function calls with arguments
- [ ] Array/list access and operations
- [ ] Struct field access
- [ ] Method calls
- [ ] String operations
- [ ] Type casting
- [ ] Parenthesized expressions

#### 2. Statement Types Not Implemented
- [ ] Let with initialization
- [ ] If/else statements
- [ ] While loops
- [ ] For loops
- [ ] Block statements
- [ ] Set (assignment) statements
- [ ] Assert statements
- [ ] Expression statements

#### 3. Type System Features
- [ ] Generic types (List<T>)
- [ ] Struct types
- [ ] Enum types
- [ ] Array types
- [ ] Function types
- [ ] Type inference

#### 4. Code Generation Features
- [ ] Function parameters
- [ ] Local variables
- [ ] Control flow (if, while, for)
- [ ] Struct declarations
- [ ] Enum declarations
- [ ] Array/list operations
- [ ] String operations

#### 5. Advanced Features
- [ ] Extern functions
- [ ] Shadow tests
- [ ] Module system
- [ ] Include/import
- [ ] Nested expressions
- [ ] Complex operator precedence

## Implementation Strategy

### Phase A: Complete Expression Handling (3-5 days)

**Priority: CRITICAL**

1. **Function Calls**
   - Parse call arguments
   - Store in Parser
   - Type check arguments vs parameters
   - Generate C function call code

2. **Binary Operations (Complete)**
   - Recursive left/right generation
   - Operator mapping (+, -, *, /, ==, <, >, etc.)
   - Type checking for operator compatibility

3. **Parenthesized Expressions**
   - Parse and store
   - Generate with proper C syntax

4. **Struct Field Access**
   - Parse field.name syntax
   - Type check field exists
   - Generate C struct access

### Phase B: Complete Statement Handling (3-5 days)

**Priority: CRITICAL**

1. **Let Statements**
   - Parse with type and initialization
   - Generate C variable declaration
   - Add to symbol table

2. **If/Else Statements**
   - Parse condition and branches
   - Generate C if/else
   - Handle nested blocks

3. **While Loops**
   - Parse condition and body
   - Generate C while
   - Handle break/continue (if supported)

4. **Block Statements**
   - Walk through statement list
   - Generate each statement
   - Handle scoping

5. **Set Statements**
   - Parse assignment
   - Type check LHS = RHS
   - Generate C assignment

### Phase C: Type System Completion (2-3 days)

**Priority: HIGH**

1. **Struct Support**
   - Store struct definitions
   - Type check struct usage
   - Generate C struct declarations

2. **Enum Support**
   - Store enum definitions
   - Type check enum values
   - Generate C enum declarations

3. **Generic Types**
   - Handle List<T> types
   - Type checking for generics
   - Generate appropriate C code

### Phase D: Code Generation Completion (3-4 days)

**Priority: HIGH**

1. **Function Parameters**
   - Store parameter names and types
   - Generate C parameter lists
   - Use in function body

2. **Complete Function Bodies**
   - Walk all statements in body
   - Generate each statement type
   - Handle local variables

3. **Runtime Functions**
   - Generate calls to nanolang runtime
   - Handle print, println, etc.
   - String operations

### Phase E: Bootstrap Preparation (2-3 days)

**Priority: HIGH**

1. **Self-Compilation Test**
   - Try to compile lexer with self-hosted compiler
   - Fix any issues found
   - Iterate until successful

2. **Module Handling**
   - Handle multiple source files
   - Link together
   - Generate single executable

3. **Complete Test Suite**
   - Run simple programs
   - Verify outputs match
   - Fix any mismatches

### Phase F: Full Bootstrap (3-5 days)

**Priority: CRITICAL**

1. **Compile Components**
   - Compile lexer_main.nano
   - Compile parser_mvp.nano
   - Compile typechecker_minimal.nano
   - Compile transpiler_minimal.nano

2. **Link Together**
   - Combine all compiled C files
   - Add integration code
   - Build complete compiler

3. **Verification**
   - Self-compiled compiler compiles itself again
   - Output matches reference compiler
   - All tests pass

## Critical Features for Bootstrap

### Minimum Viable Features

These MUST work for self-hosting:

1. **Function calls with arguments** - Most common operation
2. **Let statements** - Variable declarations everywhere
3. **If/else statements** - Control flow
4. **While loops** - Iteration
5. **Binary operations** - Arithmetic and logic
6. **Return statements** - Already done ✅
7. **Struct field access** - AST node access
8. **List operations** - List_* function calls
9. **String operations** - str_concat, etc.
10. **Set statements** - Variable assignments

### Can Defer (Nice to Have)

1. For loops (can use while)
2. Advanced generics (can hardcode types)
3. Shadow tests (can skip for bootstrap)
4. Module imports (can concatenate files)
5. Extern declarations (can inline)

## Implementation Priority

### Week 1: Expressions
- Days 1-2: Function calls
- Days 3-4: Binary operations (complete)
- Day 5: Struct field access

### Week 2: Statements
- Days 1-2: Let and Set statements
- Days 3-4: If/else and While
- Day 5: Block walking

### Week 3: Code Generation
- Days 1-2: Function parameters
- Days 3-4: Complete function bodies
- Day 5: Runtime integration

### Week 4: Bootstrap
- Days 1-2: Self-compilation tests
- Days 3-4: Fix issues
- Day 5: Full bootstrap

## Success Criteria

### Level 1: Basic Compilation ✅
- [x] Can parse simple functions
- [x] Can type check basic expressions
- [x] Can generate C code for functions
- [x] All components compile

### Level 2: Feature Complete
- [ ] All expression types work
- [ ] All statement types work
- [ ] Type system complete
- [ ] Can compile real programs

### Level 3: Self-Hosting
- [ ] Can compile lexer_main.nano
- [ ] Can compile parser_mvp.nano
- [ ] Can compile typechecker_minimal.nano
- [ ] Can compile transpiler_minimal.nano

### Level 4: Bootstrap Complete
- [ ] Self-compiled compiler works
- [ ] Can compile itself again
- [ ] Output is identical (fixpoint)
- [ ] All tests pass

## Risk Assessment

### High Risk Items
1. **Generic type handling** - Complex to implement correctly
2. **Recursive expressions** - Easy to get wrong
3. **Symbol table scoping** - Must track correctly
4. **C code generation edge cases** - Many corner cases

### Mitigation Strategies
1. **Incremental testing** - Test each feature immediately
2. **Reference comparison** - Compare output to C compiler
3. **Simple test cases** - Start with minimal examples
4. **Fallback options** - Can simplify features if needed

## Timeline Estimate

**Optimistic:** 3-4 weeks  
**Realistic:** 4-6 weeks  
**Pessimistic:** 6-8 weeks

**Current velocity:** 3-5x faster than estimates  
**Adjusted realistic:** 2-3 weeks with current velocity

## Next Immediate Steps

1. **Implement function calls** (Day 1-2)
   - Most critical missing feature
   - Needed for nearly every operation
   - Foundation for everything else

2. **Implement let statements** (Day 2-3)
   - Second most critical
   - Every function has variables
   - Needed for symbol table

3. **Implement if/else** (Day 3-4)
   - Critical for control flow
   - Used extensively in compiler

4. **Complete binary operations** (Day 4-5)
   - Already started
   - Finish recursive generation
   - Test thoroughly

5. **Test end-to-end** (Day 5-6)
   - Compile simple program
   - Verify output
   - Build momentum

## Resource Requirements

### Code to Write
- Estimated: 2,000-3,000 additional lines
- Current: 4,200 lines
- Total: 6,000-7,000 lines

### Testing
- Unit tests for each feature
- Integration tests for combinations
- Self-compilation tests
- Regression tests

### Documentation
- Feature completion docs
- Bootstrap process
- Known limitations
- Future work

## Conclusion

Full self-hosting is achievable with focused effort on the critical features. The infrastructure from Phases 1-3 provides a solid foundation. By prioritizing function calls, statements, and code generation, we can reach self-hosting in 2-3 weeks.

**Next Action:** Begin Phase A - Implement function calls with arguments
