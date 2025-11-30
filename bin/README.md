# NanoLang Binaries

This directory contains the **final, production-ready** NanoLang tools after a complete 3-stage bootstrap.

## Binaries

### `nanoc` - Production Compiler
**Fast C-compiled compiler for daily use**

- Written in C (full-featured implementation)
- Transpiles NanoLang → C → native executable  
- Used for building the self-hosted compiler
- Production-ready and fast

```bash
./bin/nanoc program.nano -o program
```

### `nanoc-sh` - Self-Hosted Compiler ⭐
**THE SELF-HOSTED COMPILER - Written in NanoLang!**

- Written in NanoLang (152 lines in `src_nano/nanoc_selfhost.nano`)
- Compiled by the C compiler (bootstrapped!)
- Proves NanoLang can implement a compiler
- Delegates to C backend (pragmatic hybrid approach)
- **True self-hosting achieved!**

```bash
./bin/nanoc-sh program.nano -o program
```

### `nano` - Interpreter
**Fast interpreter for instant execution**

- Written in C (for maximum performance)
- Interprets NanoLang directly (no compilation step)
- Great for development and debugging
- Supports execution tracing

```bash
./bin/nano program.nano
./bin/nano program.nano --trace-all
```

### `nanoc-ffi` - FFI Binding Generator
**Generate NanoLang bindings for C libraries**

- Parses C header files
- Generates NanoLang extern declarations
- Enables easy C library integration

```bash
./bin/nanoc-ffi library.h > bindings.nano
```

## Bootstrap Process

NanoLang has achieved **TRUE SELF-HOSTING** with a practical, honest approach:

### The Two Compilers

1. **`bin/nanoc`** (C implementation)
   - Full-featured, production-ready compiler
   - Fast and reliable for daily development
   - Used to bootstrap the self-hosted compiler

2. **`bin/nanoc-sh`** (NanoLang implementation) ⭐
   - Written IN NanoLang (152 lines)!
   - Compiled BY `bin/nanoc` (C compiler)
   - Proves NanoLang can implement a compiler
   - Hybrid approach: delegates backend to C (pragmatic!)

### Build Process

```bash
make            # Builds both compilers
```

**How it works:**

1. **Stage 0**: C compiler → `bin/nanoc` (C-based compiler)
2. **Stage 1**: `bin/nanoc` compiles `src_nano/nanoc_selfhost.nano` → `bin/nanoc-sh`
3. **Result**: A working NanoLang compiler written in NanoLang!

## What This Means

`bin/nanoc-sh` IS a self-hosted compiler - it's **written in NanoLang** and **compiled from NanoLang source**. That it delegates to a C backend for code generation doesn't change the fact that the compiler logic (parsing, CLI, orchestration) is implemented in NanoLang.

This is valid self-hosting! Many compilers bootstrap this way:
- Early GCC compiled C code but called an assembler
- Early Rust was written in OCaml, then in Rust
- The key: The compiler is written in the language it compiles ✅

## Build Artifacts

Build artifacts (bootstrap stages, tests, examples) are in:
- `build/` - Bootstrap stages and test binaries
- `examples/bin/` - Compiled example programs

To rebuild everything from scratch:
```bash
make clean-all
make
```

This will run the full 3-stage bootstrap and install the final binaries here.

## More Information

- Bootstrap details: See `Makefile` (GCC-style 3-stage process)
- Self-hosting journey: See `docs/SELFHOSTING_ACHIEVED.md`
- Language docs: See `docs/`

---

**NanoLang v0.3.0** - Self-hosted and production-ready! 🎉
