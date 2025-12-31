# Self-Hosting Strategy for NanoLang (2025)

**Status**: Active Priority  
**Created**: 2025-12-30  
**Target**: Full self-hosted compiler by Q2 2025  

---

## Executive Summary

NanoLang currently has a **3-stage bootstrap** that successfully compiles the self-hosted compiler components (parser, typechecker, transpiler) written in NanoLang using the C reference compiler. However, **we still depend on the C compiler** (`bin/nanoc_c`) for the final integration step.

**Goal**: Eliminate `bin/nanoc_c` dependency and run the entire compiler pipeline in pure NanoLang.

**Current State** (as of 2025-12-30):
- ✅ **Stage 1**: C reference compiler (`bin/nanoc`) works perfectly
- ✅ **Stage 2**: Self-hosted components compile successfully
- ✅ **Stage 3**: Bootstrap validation passes
- ❌ **Integration**: Still use C compiler for final orchestration
- ❌ **Driver**: No pure-NanoLang compiler driver yet

---

## Architecture Overview

### Current Architecture (C-Dependent)

```
User runs: nanoc myprogram.nano

┌─────────────────────────────────────────────────────────────┐
│  bin/nanoc (C binary)                                       │
│  ├─ Parse arguments                                         │
│  ├─ Invoke: src/lexer.c                                     │
│  ├─ Invoke: src/parser.c                                    │
│  ├─ Invoke: src/typechecker.c                               │
│  ├─ Invoke: src/transpiler.c                                │
│  ├─ Generate C code                                         │
│  └─ Invoke: cc (system C compiler)                          │
└─────────────────────────────────────────────────────────────┘
```

### Target Architecture (Self-Hosted)

```
User runs: nanoc_nano myprogram.nano

┌─────────────────────────────────────────────────────────────┐
│  bin/nanoc_nano (NanoLang binary, compiled by Stage 2)     │
│  ├─ Parse arguments (NanoLang: src_nano/driver.nano)       │
│  ├─ Invoke: src_nano/lexer.nano                            │
│  ├─ Invoke: src_nano/parser.nano                           │
│  ├─ Invoke: src_nano/typecheck.nano                        │
│  ├─ Invoke: src_nano/transpiler.nano                       │
│  ├─ Generate C code                                         │
│  └─ Invoke: cc (system C compiler) via std::process        │
└─────────────────────────────────────────────────────────────┘
```

**Key Insight**: We're not replacing the C compiler (cc) - we're replacing the NanoLang compiler frontend.

---

## Remaining Gaps

### 1. Missing: Compiler Driver (`src_nano/driver.nano`)

**What it needs to do**:
- Parse command-line arguments (`-o output`, `-I include_path`, etc.)
- Orchestrate compilation pipeline:
  1. Lex → tokens
  2. Parse → AST
  3. Typecheck → validated AST
  4. Transpile → C code
  5. Invoke `cc` to produce binary
- Handle errors at each stage
- Manage temporary files
- Return appropriate exit codes

**Dependencies**:
- ✅ `std::env` - Get command-line args (completed today!)
- ✅ `std::process` - Invoke `cc` (completed today!)
- ✅ `std::fs` - File I/O, temp files (completed today!)
- ❌ `std::io` - stdin/stdout/stderr handling
- ❌ String formatting/interpolation (for error messages)

**Complexity**: Medium (500-800 lines)  
**Effort**: 2-3 weeks  

---

### 2. Missing: Typechecker Feature Parity

**Current State**:
- ✅ Basic types (int, bool, string, void, float)
- ✅ Structs, enums, unions
- ✅ Functions, let, set, if, while, for
- ✅ Binary operations
- ✅ Arrays (basic)
- ❌ **Generics** (List<T> type checking)
- ❌ **Match expression binding** (can't do `match x { Some(v) => v.field }`)
- ❌ **Tuples** (tuple construction and indexing)
- ❌ **Imports** (module system)

**Priority**: **HIGH** - Needed to compile `src_nano/transpiler.nano`  

**Most Critical**:
1. **Import/module type checking** - Without this, can't compile multi-file programs
2. **Match bindings** - Extensively used in transpiler
3. **Generics (List<T>)** - Used everywhere in compiler

**Complexity**: High (1500-2000 lines)  
**Effort**: 6-8 weeks  

---

### 3. Missing: Transpiler Feature Parity

**Current State**:
- ✅ Basic expressions (arithmetic, logic, calls)
- ✅ Statements (let, set, if, while, for, return)
- ✅ Structs, enums, unions
- ✅ Match expressions (basic)
- ❌ **List<T> monomorphization** (generating `list_Point_new`, etc.)
- ❌ **Generic function calls** (type specialization)
- ❌ **Module imports** (cross-file references)
- ❌ **Tuple codegen**

**Priority**: **MEDIUM** - Mostly works, needs finishing touches  

**Most Critical**:
1. **List<T> specialization** - Generate `/tmp/list_Point.h` files
2. **Import resolution** - Generate correct `#include` directives
3. **Name mangling** - Handle namespace collisions

**Complexity**: High (2000-2500 lines)  
**Effort**: 8-10 weeks  

---

### 4. Missing: Build System Integration

**What needs to work**:
```bash
# Current (C compiler)
make nanoc

# Target (self-hosted)
make nanoc_nano

# Verification
make verify-selfhost  # Ensures no C compiler dependency
```

**Requirements**:
- Makefile targets for building `nanoc_nano`
- Dependency tracking (rebuild on source changes)
- Clean separation from C build
- CI/CD integration

**Complexity**: Low (200-300 lines)  
**Effort**: 1-2 weeks  

---

## Implementation Phases

### Phase 1: Foundation (4 weeks) ✅ COMPLETE

**Goal**: Core stdlib functionality for compiler driver

- [x] `std::env` - Command-line args
- [x] `std::process` - Invoke external commands
- [x] `std::fs` - File operations, path handling
- [x] Result types for error handling
- [x] Diagnostics infrastructure

**Status**: ✅ **COMPLETED** 2025-12-30

---

### Phase 2: Compiler Driver (3-4 weeks)

**Goal**: Pure NanoLang driver that orchestrates compilation

**Tasks**:
1. **Argument Parsing** (1 week)
   - Parse `-o`, `-I`, `-L`, `-l` flags
   - Handle `--help`, `--version`
   - Validate inputs

2. **Pipeline Orchestration** (1 week)
   - Lex → Parse → Typecheck → Transpile
   - Error handling at each stage
   - Pass data between stages

3. **C Compiler Invocation** (1 week)
   - Build `cc` command with flags
   - Capture stdout/stderr
   - Handle compilation errors

4. **Integration & Testing** (1 week)
   - Test on all examples
   - Verify binary compatibility
   - Performance benchmarking

**Deliverable**: `src_nano/driver.nano` that compiles simple programs

---

### Phase 3: Typechecker Parity (6-8 weeks)

**Goal**: Self-hosted typechecker supports all NanoLang features

**Priority Order**:

#### 3.1 Imports & Modules (2 weeks) - **CRITICAL**
```nano
import "std/io.nano"
import "modules/sdl/sdl.nano"

fn main() -> int {
    (println "Hello")  // Must resolve println from std/io
    return 0
}
```

**What's needed**:
- Track imported modules
- Resolve qualified names (`std::io::println`)
- Validate import paths
- Prevent circular imports

#### 3.2 Match Bindings (2 weeks) - **CRITICAL**
```nano
enum Option<T> { Some(T), None }

match opt {
    Some(x) => (println x),  // 'x' must be bound correctly
    None => (println "empty")
}
```

**What's needed**:
- Bind pattern variables in match arms
- Type check bound variables
- Handle nested patterns
- Support struct/tuple destructuring

#### 3.3 Generics (List<T>) (2-3 weeks) - **HIGH**
```nano
let numbers: List<int> = (list_int_new)
let points: List<Point> = (list_Point_new)
```

**What's needed**:
- Parse `List<T>` syntax
- Track generic instantiations
- Validate element types
- Support nested generics (`List<List<int>>`)

#### 3.4 Tuples (1 week) - **MEDIUM**
```nano
let pair: (int, string) = (42, "hello")
let x: int = pair.0
let y: string = pair.1
```

**What's needed**:
- Parse tuple types
- Type check tuple construction
- Type check tuple indexing
- Support nested tuples

---

### Phase 4: Transpiler Parity (8-10 weeks)

**Goal**: Generate production-quality C code for all features

#### 4.1 List<T> Monomorphization (3-4 weeks) - **CRITICAL**
```nano
// Input: List<Point>
// Output: 
//   #include "/tmp/list_Point.h"
//   typedef struct { Point *data; int count; int capacity; } List_Point;
```

**What's needed**:
- Detect `List<T>` usage in AST
- Generate specialized C structs
- Generate specialized functions (`list_Point_new`, etc.)
- Write to temp files or inline

#### 4.2 Import Resolution (2-3 weeks) - **HIGH**
```nano
// Input: import "std/io.nano"
// Output: #include "modules/std/io.h"
```

**What's needed**:
- Map NanoLang imports to C includes
- Handle relative vs absolute paths
- Generate correct include guards
- Link external modules

#### 4.3 Name Mangling (2-3 weeks) - **HIGH**
```nano
// Input: mod foo { fn bar() }
// Output: void nl_foo_bar()
```

**What's needed**:
- Prefix user types with `nl_`
- Handle namespace collisions
- Preserve C stdlib names
- Support generic specializations

---

### Phase 5: Integration & Validation (2-3 weeks)

**Goal**: Prove self-hosting works end-to-end

#### 5.1 Build System (1 week)
- Add `make nanoc_nano` target
- Add `make verify-selfhost` target
- Update CI/CD pipeline

#### 5.2 Compatibility Testing (1 week)
- Compile all examples with `nanoc_nano`
- Compare binaries (should be identical)
- Performance comparison (should be similar)

#### 5.3 Documentation (1 week)
- Update README with self-hosting status
- Document `nanoc_nano` usage
- Migration guide from `nanoc`

---

## Timeline & Milestones

### Q1 2025 (Jan-Mar)

**January**:
- ✅ Week 1-2: stdlib foundation (DONE 2025-12-30)
- 🔄 Week 3-4: Compiler driver skeleton

**February**:
- Week 1-2: Typechecker - imports & modules
- Week 3-4: Typechecker - match bindings

**March**:
- Week 1-2: Typechecker - generics
- Week 3-4: Typechecker - tuples

### Q2 2025 (Apr-Jun)

**April**:
- Week 1-2: Transpiler - List<T> monomorphization
- Week 3-4: Transpiler - import resolution

**May**:
- Week 1-2: Transpiler - name mangling
- Week 3-4: Transpiler - polish & optimization

**June**:
- Week 1-2: Integration & build system
- Week 3-4: Testing, documentation, release

**Target**: **Self-hosted NanoLang by June 30, 2025** 🎯

---

## Risk Assessment

### High Risk Areas

#### 1. Typechecker Complexity
**Risk**: Type inference for generics is notoriously difficult  
**Mitigation**: Start with explicit type annotations, defer inference  
**Contingency**: Keep C typechecker as fallback for complex cases  

#### 2. Performance Regression
**Risk**: Self-hosted compiler may be slower than C version  
**Mitigation**: Benchmark each phase, optimize hot paths  
**Contingency**: Use C compiler for production, NanoLang for development  

#### 3. Bootstrap Fragility
**Risk**: Breaking Stage 1 breaks entire pipeline  
**Mitigation**: Extensive testing, version pinning, backup binaries  
**Contingency**: Maintain C compiler indefinitely as backup  

### Medium Risk Areas

#### 4. Module System Complexity
**Risk**: Import resolution interacts with filesystem, namespaces, generics  
**Mitigation**: Prototype on simple cases first  
**Contingency**: Simplified module system (no nested modules)  

#### 5. C Interop Edge Cases
**Risk**: Some C constructs may not map cleanly to NanoLang  
**Mitigation**: Extensive testing with real-world C libraries  
**Contingency**: Allow escape hatches for advanced C features  

---

## Success Criteria

### Minimum Viable Self-Hosting (MVSH)

**Must work**:
- ✅ Compile simple programs (hello world, arithmetic)
- ✅ Compile programs with structs/enums
- ✅ Compile programs with imports
- ✅ Pass all existing test suite
- ✅ Bootstrap successfully (Stage 1 → Stage 2 → Stage 3)

**Can defer**:
- ❌ Full generic function support
- ❌ Advanced type inference
- ❌ Optimization passes
- ❌ Incremental compilation

### Full Self-Hosting

**Additional requirements**:
- Compile all examples (SDL, ncurses, Bullet Physics)
- Performance within 2x of C compiler
- Error messages as good or better than C compiler
- Support for external modules
- Stable over multiple bootstrap iterations

---

## Alternative Strategies

### Strategy A: Big Bang (Current Plan)
**Approach**: Complete all features, then switch  
**Pros**: Clean cut-over, no hybrid maintenance  
**Cons**: High risk, long time to value  
**Timeline**: 6 months  

### Strategy B: Incremental
**Approach**: Use self-hosted components progressively  
**Pros**: Lower risk, earlier validation  
**Cons**: Complex hybrid system, more testing  
**Timeline**: 9 months (more total work, but earlier milestones)  

**Example incremental path**:
1. Use self-hosted lexer only (2 weeks)
2. Add self-hosted parser (4 weeks)
3. Add self-hosted typechecker (8 weeks)
4. Add self-hosted transpiler (12 weeks)
5. Replace driver (16 weeks)

### Strategy C: Parallel Development
**Approach**: Develop self-hosted alongside C compiler  
**Pros**: No risk to existing system  
**Cons**: Maintenance burden, feature drift  
**Timeline**: 12+ months  

**Recommendation**: **Strategy A (Big Bang)** - We're close enough that a focused 6-month push is feasible.

---

## Recommendations

### Immediate Next Steps (January 2025)

1. **Start with Compiler Driver** (Week 1-2)
   - Create `src_nano/driver.nano` skeleton
   - Get basic pipeline working (lex → parse → typecheck → transpile)
   - Test on hello world

2. **Tackle Import/Module Support** (Week 3-4)
   - This unblocks everything else
   - Focus on typechecker first
   - Transpiler can be simplified initially

3. **Continuous Integration**
   - Add `make test-selfhost` that runs weekly
   - Track progress metrics (% of examples compiling)
   - Maintain compatibility with C compiler

### Strategic Principles

**Do**:
- ✅ Focus on **compatibility** over **optimization**
- ✅ Maintain **C compiler as backup** indefinitely
- ✅ Test **continuously** (don't wait for completion)
- ✅ Document **migration path** for users
- ✅ Keep **simple** (avoid over-engineering)

**Don't**:
- ❌ Rewrite everything from scratch
- ❌ Optimize prematurely
- ❌ Break backward compatibility
- ❌ Add new features (focus on parity)
- ❌ Rush (quality over speed)

---

## Conclusion

Self-hosting NanoLang is **achievable by Q2 2025** with focused effort on:
1. Compiler driver (4 weeks)
2. Typechecker parity (8 weeks)
3. Transpiler parity (10 weeks)
4. Integration (3 weeks)

**Total**: ~25 weeks = 6 months

The foundation work completed in December 2025 (`std::env`, `std::process`, `std::fs`, `Result`, diagnostics) accelerates this timeline significantly.

**Key risk**: Typechecker complexity, especially generics and imports  
**Key mitigation**: Start simple, iterate, maintain C fallback  

**Next Action**: Begin `src_nano/driver.nano` development (January 2025)

---

**Status**: Ready to Execute  
**Owner**: Core Team  
**Review Date**: Monthly  


