# Nanolang Bootstrap Strategy

**Date:** November 12, 2025  
**Goal:** Define the multi-stage bootstrap process for nanolang self-hosting  
**Status:** Planning

---

## Overview

Unlike GCC (which compiles the same source files with different compilers), nanolang transitions from **C source code** to **nanolang source code**. This requires a carefully planned bootstrap process to ensure correctness and reproducibility.

---

## Key Difference from GCC

### GCC Bootstrap (Simpler)
```
Stage 1: Host compiler (clang/gcc) → GCC source → GCC binary
Stage 2: GCC binary → GCC source → GCC binary (self-compiled)
Stage 3: Install self-compiled GCC
```
**Same source files** in all stages, just different compilers.

### Nanolang Bootstrap (More Complex)
```
Stage 1: C compiler → nanolang compiler (C source) → nanolang binary
Stage 2: C compiler → nanolang compiler (nanolang source) → nanolang binary
Stage 3: nanolang binary → nanolang compiler (nanolang source) → nanolang binary (self-compiled)
Stage 4: Install self-compiled nanolang
```
**Different source files** (C → nanolang), requiring careful validation.

---

## Bootstrap Stages

### Stage 0: C Compiler (Current State)

**Status:** ✅ Complete

**What it is:**
- Compiler written entirely in C (`src/*.c`)
- Compiles nanolang programs → C → binary
- Supports all language features needed for self-hosting

**Artifacts:**
- `bin/nanoc` - C-compiled nanolang compiler
- `src/lexer.c`, `src/parser.c`, `src/typechecker.c`, `src/transpiler.c` - C source

**Purpose:**
- Reference implementation
- Validates nanolang language features
- Used to bootstrap Stage 1

---

### Stage 1: Hybrid Compiler (C + Nanolang)

**Status:** 🚧 In Progress (lexer started)

**What it is:**
- Compiler partially rewritten in nanolang
- Some components still in C (e.g., runtime, main driver)
- Compiled by C compiler (`bin/nanoc`)

**Process:**
```bash
# 1. Write lexer in nanolang
src_nano/lexer.nano

# 2. Compile lexer with C compiler
./bin/nanoc src_nano/lexer.nano -o lexer.o

# 3. Link with C runtime and other C components
gcc lexer.o src/runtime/*.o src/main.o -o bin/nanolanc_stage1
```

**Success Criteria:**
- [ ] Lexer in nanolang compiles successfully
- [ ] Lexer output matches C lexer output
- [ ] Can compile simple nanolang programs
- [ ] All shadow tests pass

**Artifacts:**
- `src_nano/lexer.nano` - Lexer in nanolang
- `bin/nanolanc_stage1` - Hybrid compiler (nanolang lexer + C parser/typechecker/transpiler)

**Timeline:** 2-3 weeks per component

---

### Stage 2: Full Nanolang Compiler (Compiled by C)

**Status:** ❌ Not Started

**What it is:**
- Entire compiler rewritten in nanolang (`src_nano/*.nano`)
- All components: lexer, parser, type checker, transpiler, main driver
- Still compiled by C compiler (`bin/nanoc`)

**Process:**
```bash
# 1. Write all compiler components in nanolang
src_nano/lexer.nano
src_nano/parser.nano
src_nano/typechecker.nano
src_nano/transpiler.nano
src_nano/main.nano

# 2. Compile entire compiler with C compiler
./bin/nanoc src_nano/compiler.nano -o bin/nanolanc_stage2

# 3. Test: Use stage2 compiler to compile a simple program
./bin/nanolanc_stage2 examples/hello.nano -o hello_test
./hello_test  # Should work identically to C compiler
```

**Success Criteria:**
- [ ] All compiler components in nanolang
- [ ] Stage2 compiler compiles `examples/hello.nano` successfully
- [ ] Output identical to C compiler output
- [ ] All shadow tests pass
- [ ] Can compile all examples
- [ ] Performance within 2-3x of C compiler

**Artifacts:**
- `src_nano/compiler.nano` - Complete compiler in nanolang
- `bin/nanolanc_stage2` - Full nanolang compiler (compiled by C)

**Timeline:** 13-18 weeks (all components)

**Validation:**
```bash
# Compare outputs
./bin/nanoc examples/hello.nano -o hello_c.nano.c
./bin/nanolanc_stage2 examples/hello.nano -o hello_nano.nano.c
diff hello_c.nano.c hello_nano.nano.c  # Should be identical (or functionally equivalent)
```

---

### Stage 3: Self-Compiled Nanolang Compiler

**Status:** ❌ Not Started

**What it is:**
- Use Stage 2 compiler to compile itself
- This proves the compiler can compile itself
- Creates a "self-compiled" version

**Process:**
```bash
# 1. Use stage2 compiler to compile itself
./bin/nanolanc_stage2 src_nano/compiler.nano -o bin/nanolanc_stage3

# 2. Test: Use stage3 compiler to compile a simple program
./bin/nanolanc_stage3 examples/hello.nano -o hello_test
./hello_test  # Should work identically

# 3. Verify: Compare stage2 and stage3 outputs
./bin/nanolanc_stage2 examples/hello.nano -o hello_stage2.nano.c
./bin/nanolanc_stage3 examples/hello.nano -o hello_stage3.nano.c
diff hello_stage2.nano.c hello_stage3.nano.c  # Should be identical
```

**Success Criteria:**
- [ ] Stage3 compiler compiles `examples/hello.nano` successfully
- [ ] Output identical to Stage2 compiler output
- [ ] All shadow tests pass
- [ ] Can compile all examples
- [ ] Self-compilation successful

**Artifacts:**
- `bin/nanolanc_stage3` - Self-compiled nanolang compiler

**Timeline:** 1-2 weeks (validation and testing)

**Validation:**
```bash
# Test self-compilation
./bin/nanolanc_stage3 src_nano/compiler.nano -o bin/nanolanc_stage3_test
./bin/nanolanc_stage3_test examples/hello.nano -o hello_test
./hello_test  # Should work
```

---

### Stage 4: Fixed Point Verification

**Status:** ❌ Not Started

**What it is:**
- Continue bootstrapping: stage3 → stage4 → stage5, etc.
- Check for "fixed point" (stageN == stageN+1)
- Ensures compiler is stable and reproducible

**Process:**
```bash
# Bootstrap chain
./bin/nanolanc_stage3 src_nano/compiler.nano -o bin/nanolanc_stage4
./bin/nanolanc_stage4 src_nano/compiler.nano -o bin/nanolanc_stage5
./bin/nanolanc_stage5 src_nano/compiler.nano -o bin/nanolanc_stage6

# Check for fixed point
diff bin/nanolanc_stage4 bin/nanolanc_stage5  # Should be identical (or functionally equivalent)
diff bin/nanolanc_stage5 bin/nanolanc_stage6  # Should be identical
```

**Success Criteria:**
- [ ] Fixed point reached (stageN == stageN+1)
- [ ] All tests pass at every stage
- [ ] Bootstrapping is repeatable
- [ ] No regressions introduced

**Artifacts:**
- `bin/nanolanc_stage4`, `bin/nanolanc_stage5`, etc. - Bootstrap chain

**Timeline:** 1-2 weeks (verification)

---

### Stage 5: Installation & Distribution

**Status:** ❌ Not Started

**What it is:**
- Install the self-compiled nanolang compiler
- Make it available for users
- Create distribution packages

**Process:**
```bash
# 1. Install self-compiled compiler
cp bin/nanolanc_stage5 /usr/local/bin/nanolangc
chmod +x /usr/local/bin/nanolangc

# 2. Verify installation
nanolangc examples/hello.nano -o hello_test
./hello_test  # Should work

# 3. Create distribution
# - Source tarball with src_nano/*.nano
# - Binary for common platforms
# - Documentation
```

**Success Criteria:**
- [ ] Installed compiler works correctly
- [ ] Can compile all examples
- [ ] Distribution packages created
- [ ] Documentation complete

**Artifacts:**
- `/usr/local/bin/nanolangc` - Installed compiler
- `nanolang-X.Y.Z.tar.gz` - Source distribution
- `nanolang-X.Y.Z-linux-amd64.tar.gz` - Binary distribution

**Timeline:** 1-2 weeks (packaging and distribution)

---

## Bootstrap Trust Chain

### Trust Validation

Each stage validates the previous stage:

```
Stage 0 (C compiler)
    ↓ validates language features
Stage 1 (Hybrid)
    ↓ validates nanolang lexer
Stage 2 (Full nanolang, compiled by C)
    ↓ validates entire nanolang compiler
Stage 3 (Self-compiled)
    ↓ validates self-compilation
Stage 4 (Fixed point)
    ↓ validates stability
Stage 5 (Installation)
    ↓ ready for users
```

### Validation Strategy

**At each stage, verify:**
1. **Functional equivalence:** Output identical to previous stage
2. **Test coverage:** All shadow tests pass
3. **Example compilation:** All examples compile successfully
4. **Performance:** Within acceptable bounds (2-3x slower acceptable)

**Comparison tools:**
```bash
# Compare generated C code
diff output_stageN.c output_stageN+1.c

# Compare binary outputs
diff output_stageN output_stageN+1

# Compare execution results
./output_stageN > result_stageN.txt
./output_stageN+1 > result_stageN+1.txt
diff result_stageN.txt result_stageN+1.txt
```

---

## File Organization

### Source Files

```
nanolang/
├── src/                    # C compiler (Stage 0)
│   ├── lexer.c
│   ├── parser.c
│   ├── typechecker.c
│   ├── transpiler.c
│   └── main.c
│
├── src_nano/               # Nanolang compiler (Stage 2+)
│   ├── lexer.nano
│   ├── parser.nano
│   ├── typechecker.nano
│   ├── transpiler.nano
│   └── main.nano
│
└── bin/                    # Compiled binaries
    ├── nanoc               # Stage 0 (C compiler)
    ├── nanolanc_stage1     # Stage 1 (Hybrid)
    ├── nanolanc_stage2     # Stage 2 (Full nanolang, C-compiled)
    ├── nanolanc_stage3     # Stage 3 (Self-compiled)
    └── nanolanc_stage4+    # Stage 4+ (Fixed point)
```

### Build Process

**Stage 0 → Stage 1:**
```makefile
# Hybrid compiler
nanolanc_stage1: src_nano/lexer.nano src/parser.c src/typechecker.c src/transpiler.c src/main.c
	./bin/nanoc src_nano/lexer.nano -o obj/lexer_nano.o
	gcc obj/lexer_nano.o src/parser.o src/typechecker.o src/transpiler.o src/main.o -o bin/nanolanc_stage1
```

**Stage 1 → Stage 2:**
```makefile
# Full nanolang compiler (compiled by C)
nanolanc_stage2: src_nano/compiler.nano
	./bin/nanoc src_nano/compiler.nano -o bin/nanolanc_stage2
```

**Stage 2 → Stage 3:**
```makefile
# Self-compiled compiler
nanolanc_stage3: src_nano/compiler.nano bin/nanolanc_stage2
	./bin/nanolanc_stage2 src_nano/compiler.nano -o bin/nanolanc_stage3
```

**Stage 3 → Stage 4+:**
```makefile
# Fixed point verification
nanolanc_stage4: src_nano/compiler.nano bin/nanolanc_stage3
	./bin/nanolanc_stage3 src_nano/compiler.nano -o bin/nanolanc_stage4
	diff bin/nanolanc_stage3 bin/nanolanc_stage4 || echo "Fixed point not reached"
```

---

## Challenges & Solutions

### Challenge 1: Source Code Transition (C → Nanolang)

**Problem:** Different source files mean we can't directly compare like GCC does.

**Solution:**
- Compare **outputs** (generated C code, binaries, execution results)
- Maintain C compiler as reference implementation
- Extensive testing at each stage

### Challenge 2: Runtime Dependencies

**Problem:** Nanolang compiler needs runtime libraries (`list_int`, `list_string`, etc.)

**Solution:**
- Runtime libraries remain in C (compiled separately)
- Nanolang compiler links against C runtime
- Runtime is stable and doesn't need to be rewritten

### Challenge 3: Bootstrap Validation

**Problem:** How do we know the nanolang compiler is correct?

**Solution:**
- **Stage 2 validation:** Compare outputs with C compiler
- **Stage 3 validation:** Self-compilation must produce identical outputs
- **Stage 4 validation:** Fixed point proves stability
- **Extensive testing:** All examples, all shadow tests

### Challenge 4: Performance Regression

**Problem:** Nanolang compiler might be slower than C compiler.

**Solution:**
- Accept 2-3x slowdown initially
- Optimize critical paths later
- Profile and identify bottlenecks
- C compiler remains available as reference

---

## Timeline

### Phase 1: Stage 1 (Hybrid Compiler)
- **Duration:** 2-3 weeks per component
- **Components:** Lexer (in progress), Parser, Type Checker, Transpiler, Main Driver
- **Total:** 13-18 weeks

### Phase 2: Stage 2 (Full Nanolang Compiler)
- **Duration:** Already included in Phase 1 (when all components done)
- **Validation:** 1-2 weeks

### Phase 3: Stage 3 (Self-Compilation)
- **Duration:** 1-2 weeks
- **Validation:** Extensive testing

### Phase 4: Stage 4 (Fixed Point)
- **Duration:** 1-2 weeks
- **Validation:** Bootstrap chain verification

### Phase 5: Stage 5 (Installation)
- **Duration:** 1-2 weeks
- **Tasks:** Packaging, distribution, documentation

**Total Timeline:** 18-26 weeks (4.5-6.5 months)

---

## Success Metrics

### Technical Metrics
- ✅ All stages compile successfully
- ✅ Outputs identical across stages
- ✅ All shadow tests pass
- ✅ All examples compile
- ✅ Fixed point reached
- ✅ Performance within 2-3x of C compiler

### Code Metrics
- ✅ ~5,000 lines of nanolang compiler code
- ✅ 100% shadow test coverage
- ✅ Zero known bugs
- ✅ Documentation complete

### Process Metrics
- ✅ Bootstrap process repeatable
- ✅ Can bootstrap from Stage 0 to Stage 5
- ✅ No manual intervention required
- ✅ Automated validation at each stage

---

## Next Steps

1. **Complete Stage 1:** Finish lexer rewrite, then parser, type checker, transpiler, main driver
2. **Validate Stage 2:** Ensure full nanolang compiler works identically to C compiler
3. **Test Stage 3:** Self-compilation must produce identical outputs
4. **Verify Stage 4:** Fixed point ensures stability
5. **Package Stage 5:** Create distribution packages

---

## References

- [SELF_HOSTING_IMPLEMENTATION_PLAN.md](SELF_HOSTING_IMPLEMENTATION_PLAN.md) - Implementation details
- [SELF_HOSTING_CHECKLIST.md](SELF_HOSTING_CHECKLIST.md) - Feature checklist
- GCC Bootstrap Process: https://gcc.gnu.org/install/build.html

---

**Last Updated:** 2025-11-12  
**Status:** Planning  
**Next Review:** After lexer rewrite complete

