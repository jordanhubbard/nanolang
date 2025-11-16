# Nanolang Self-Hosting Roadmap

## Executive Summary

This document outlines the roadmap for achieving full self-hosting of the nanolang compiler. As of November 2025, we have completed the lexer and parser components in nanolang, and the C-based compiler supports all modern language features (unions, generics, first-class functions, enums, namespacing).

## Current State

### ✅ Completed Components

#### 1. Lexer (Self-Hosted)
- **File**: `src_nano/lexer_main.nano` (616 lines)
- **Status**: ✅ Compiles successfully
- **Tests**: All shadow tests pass
- **Features**:
  - Full tokenization of nanolang syntax
  - Keyword recognition (6 groups)
  - String and number literal parsing
  - Comment handling (single-line and multi-line)
  - Identifier and operator parsing

#### 2. Parser (Self-Hosted)
- **File**: `src_nano/parser_mvp.nano` (2336 lines)
- **Status**: ✅ Compiles successfully
- **Tests**: All shadow tests pass
- **Features**:
  - Complete AST structure definitions
  - Expression parsing (binary ops, function calls, literals)
  - Statement parsing (let, if/else, while, return, blocks)
  - Definition parsing (functions, structs, enums, unions)
  - Functional programming style (immutable parser state)
  - Token management and error handling

#### 3. C-Based Compiler Features
- **Unions**: ✅ Fully working (6 comprehensive tests pass)
- **First-Class Functions**: ✅ Fully working (all examples pass)
- **Enums**: ✅ Fully working (all tests pass)
- **Namespacing (nl_ prefix)**: ✅ Working (integration tests pass)
- **Generics (List<T>)**: ⚠️  Partially working (transpiler bug with complex generics)

### 🚧 In Progress

#### Generic Types Issue
- **Problem**: Transpiler generates incorrect extern declarations for generic list functions
- **Impact**: Complex generic examples fail to compile
- **Example**: `List<Token>` generates wrong C prototypes
- **Workaround**: Basic `List<int>`, `List<string>` work in simple cases
- **Fix Required**: Update transpiler's generic handling

## Missing Components for Full Self-Hosting

### 1. Type Checker (Estimated: 2500-3000 lines)

**Required Features**:
- Type resolution and inference
- Environment/symbol table management
- Function signature validation
- Struct/enum/union type checking
- Generic type instantiation
- Union type matching validation
- First-class function type checking
- Error reporting and diagnostics

**Complexity**: High
- Must handle all type system features
- Generic monomorphization
- Union type validation
- Function signature compatibility

**Estimated Effort**: 3-5 days of focused development

### 2. Transpiler (Estimated: 2500-3000 lines)

**Required Features**:
- C code generation for all AST nodes
- Memory management code generation
- Generic instantiation in C
- Union type transpilation
- Match expression code generation
- First-class function pointer handling
- Struct/enum/union definition generation
- Namespacing (nl_ prefix application)

**Complexity**: Very High
- Must generate correct C code for all constructs
- Handle memory management (malloc/free)
- Generate runtime support code
- Coordinate with C runtime library

**Estimated Effort**: 4-6 days of focused development

### 3. Integration & Testing (Estimated: 500-1000 lines)

**Required Features**:
- Main compiler driver
- Lexer → Parser → Type Checker → Transpiler pipeline
- Error propagation and reporting
- Comprehensive test suite
- End-to-end compilation verification

**Estimated Effort**: 1-2 days

## Phased Implementation Plan

### Phase 1: Minimal Self-Hosting (Recommended First Step)
**Timeline**: 1-2 weeks  
**Goal**: Compile simple nanolang programs end-to-end

**Scope**:
1. Simplified type checker (~500 lines)
   - Basic types only (int, string, bool)
   - Function signatures
   - Variable declarations
   - No generics, unions, or complex types

2. Simplified transpiler (~500 lines)
   - Basic expressions and statements
   - Function definitions
   - Simple struct support
   - No generics, unions, or advanced features

3. Integration (~200 lines)
   - Basic pipeline
   - Simple error handling

**Success Criteria**:
- Compile hello world program
- Compile simple calculator
- Compile basic data structures

**Benefits**:
- Rapid feedback loop
- Validates architecture
- Demonstrates progress
- Foundation for iteration

### Phase 2: Feature Expansion
**Timeline**: 2-4 weeks  
**Goal**: Add modern language features incrementally

**Iteration 1: Structs and Enums**
- Add struct type checking
- Add struct transpilation
- Add enum support

**Iteration 2: Unions**
- Add union type checking
- Add union transpilation
- Add match expression support

**Iteration 3: First-Class Functions**
- Add function type checking
- Add function pointer generation
- Test callback patterns

**Iteration 4: Generics**
- Add generic instantiation
- Fix transpiler generic bugs
- Test List<T> support

### Phase 3: Full Featured Compiler
**Timeline**: 1-2 weeks  
**Goal**: Complete all language features

**Remaining Features**:
- Module system (import/export)
- Advanced type inference
- Optimization passes
- Better error messages
- Documentation generation

### Phase 4: Bootstrap
**Timeline**: 1 week  
**Goal**: True self-hosting (compile compiler with itself)

**Steps**:
1. Compile lexer with self-hosted compiler
2. Compile parser with self-hosted compiler
3. Compile type checker with self-hosted compiler
4. Compile transpiler with self-hosted compiler
5. Compile full compiler (Stage 2) with Stage 1
6. Compile Stage 2 with itself (Stage 3)
7. Verify Stage 2 and Stage 3 are identical

## Alternative Approaches

### Option A: Minimal Viable Self-Hosting
**Focus**: Get something working quickly
**Timeline**: 2-3 weeks
**Outcome**: Can compile simple programs in nanolang

**Pros**:
- Fast path to working compiler
- Demonstrates feasibility
- Good for demos/presentations

**Cons**:
- Limited feature support
- Not production-ready
- Requires significant future work

### Option B: Full Featured Self-Hosting
**Focus**: Complete implementation of all features
**Timeline**: 2-3 months
**Outcome**: Production-ready self-hosted compiler

**Pros**:
- Complete feature parity
- Production ready
- No technical debt

**Cons**:
- Long development timeline
- High complexity
- Risk of scope creep

### Option C: Hybrid Approach (Recommended)
**Focus**: Incremental development with working milestones
**Timeline**: 1-2 months
**Outcome**: Working compiler with iterative feature additions

**Pros**:
- Regular working milestones
- Manageable complexity
- Flexibility to adjust priorities

**Cons**:
- Requires discipline to avoid shortcuts
- May have temporary limitations

## Technical Challenges

### 1. Generic Instantiation
**Problem**: Monomorphization in nanolang code
**Complexity**: High
**Approach**: Generate specialized code for each type instantiation

### 2. Memory Management
**Problem**: Generating correct malloc/free in transpiler
**Complexity**: Medium
**Approach**: Follow patterns from C transpiler

### 3. Union Type Checking
**Problem**: Ensuring exhaustive pattern matching
**Complexity**: Medium
**Approach**: Track variant coverage in type checker

### 4. Error Reporting
**Problem**: Generating helpful error messages
**Complexity**: Medium
**Approach**: Track source locations through all phases

## Success Metrics

### Milestone 1: Basic Self-Compilation
- ✅ Lexer compiles
- ✅ Parser compiles
- ⬜ Type checker compiles
- ⬜ Transpiler compiles

### Milestone 2: Feature Parity
- ⬜ All language features supported
- ⬜ All tests pass when compiled with self-hosted compiler
- ⬜ Performance within 2x of C compiler

### Milestone 3: Bootstrap Complete
- ⬜ Compiler can compile itself
- ⬜ Stage 2 == Stage 3 (bit-identical)
- ⬜ All examples work

### Milestone 4: Production Ready
- ⬜ Comprehensive test suite
- ⬜ Documentation complete
- ⬜ Error messages helpful
- ⬜ Community adoption

## Current Recommendation

Given the scope and complexity, we recommend **Option C (Hybrid Approach)** with **Phase 1 (Minimal Self-Hosting)** as the immediate next step.

**Immediate Actions** (Next 2-4 weeks):
1. ✅ Verify all C compiler features work
2. Fix generic transpiler bug
3. Implement minimal type checker in nanolang
4. Implement minimal transpiler in nanolang
5. Create integration test suite
6. Compile first simple program end-to-end

**Success Criteria**:
- Compile hello world with self-hosted compiler
- Compile calculator example
- All basic tests pass

This provides a working foundation for iterative development while maintaining manageable scope and regular milestones.

## Long-Term Vision

The ultimate goal is a fully self-hosted nanolang compiler that:
1. Compiles itself (true bootstrap)
2. Supports all language features
3. Generates efficient C code
4. Provides excellent error messages
5. Serves as a reference implementation
6. Demonstrates language capabilities

This roadmap provides a realistic path to achieving that vision through incremental, testable milestones.

---

**Document Status**: Living document, updated November 2025
**Next Review**: After Phase 1 completion

