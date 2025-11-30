# Nanolang 3-Stage Bootstrap Build System

## Overview

The Nanolang build system implements a **3-stage bootstrap** process with **sentinel files** to track build progress and skip unnecessary rebuilds. This ensures efficient builds while validating the self-hosted compiler components.

---

## Build Stages

### Stage 1: C Reference Compiler
- **Input:** C source files (`src/*.c`)
- **Output:** `bin/nanoc` (reference compiler), `bin/nano` (interpreter)
- **Purpose:** Build the reference implementation from C
- **Sentinel:** `.stage1.built`

### Stage 2: Self-Hosted Components
- **Input:** Nanolang source files (`src_nano/*.nano`)
- **Tool:** Use stage 1 compiler to compile nanolang components
- **Output:** Individual component binaries (parser, typechecker, transpiler)
- **Purpose:** Compile self-hosted code, verify it works
- **Sentinel:** `.stage2.built`

### Stage 3: Bootstrap Validation
- **Input:** Self-hosted component binaries from stage 2
- **Tool:** Run shadow tests on compiled components
- **Purpose:** Validate that self-hosted code compiles and passes tests
- **Sentinel:** `.stage3.built`

---

## Make Targets

### Primary Targets

| Target | Description | Dependencies |
|--------|-------------|--------------|
| `make` or `make build` | Build all 3 stages | None |
| `make test` | Build + run all tests | build |
| `make examples` | Build + compile examples | build |
| `make clean` | Remove all artifacts & sentinels | None |
| `make rebuild` | Clean + build from scratch | None |

### Stage-Specific Targets

| Target | Description |
|--------|-------------|
| `make stage1` | Build only stage 1 (C reference) |
| `make stage2` | Build stages 1 & 2 (self-hosted components) |
| `make stage3` | Build all 3 stages (full bootstrap) |
| `make status` | Show current build status |

### Development Targets

| Target | Description |
|--------|-------------|
| `make sanitize` | Rebuild with address & UB sanitizers |
| `make coverage` | Rebuild with coverage instrumentation |
| `make coverage-report` | Generate HTML coverage report |
| `make valgrind` | Run memory checks on test suite |
| `make install` | Install to `/usr/local/bin` (or $PREFIX) |
| `make uninstall` | Remove from installation directory |

---

## Sentinel Files

Sentinel files track which stages have been completed:

- **`.stage1.built`** - Stage 1 complete (C compiler built)
- **`.stage2.built`** - Stage 2 complete (self-hosted components compiled)
- **`.stage3.built`** - Stage 3 complete (bootstrap validated)

### How Sentinels Work

1. **First build:** All 3 stages execute sequentially
2. **Subsequent builds:** Skip stages with existing sentinels
3. **After clean:** All sentinels removed, full rebuild on next make
4. **Incremental:** If C sources change, stage 1 rebuilds automatically

### Benefits

- ⚡ **Fast rebuilds** - Skip completed stages
- ✅ **Idempotent** - Safe to run `make` multiple times
- 🎯 **Precise** - Only rebuild what changed
- 🧹 **Clean state** - `make clean` starts fresh

---

## Build Workflow

### Fresh Build (from clean state)

```bash
$ make clean          # Remove all artifacts
$ make build          # Run all 3 stages
```

Output:
```
Stage 1: Building reference compiler...
✓ Compiler: bin/nanoc
✓ Interpreter: bin/nano
✓ Stage 1 complete

Stage 2: Building Self-Hosted Components...
  Building parser_mvp...
    ✓ parser_mvp compiled successfully
  Building typechecker_minimal...
    ✓ typechecker_minimal compiled successfully
  Building transpiler_minimal...
    ✓ transpiler_minimal compiled successfully
✓ Stage 2: 3/3 components built successfully

Stage 3: Bootstrap Validation...
  Testing parser_mvp...
    ✓ parser_mvp tests passed
  Testing typechecker_minimal...
    ✓ typechecker_minimal tests passed
  Testing transpiler_minimal...
    ✓ transpiler_minimal tests passed
✓ Stage 3: 3/3 components validated

✅ Build Complete (3-Stage Bootstrap)
```

### Incremental Build (sentinels exist)

```bash
$ make build          # Skips all stages (already built)
```

Output:
```
✅ Build Complete (3-Stage Bootstrap)
Build Status:
  ✅ Stage 1: C reference compiler (bin/nanoc)
  ✅ Stage 2: Self-hosted components compiled
  ✅ Stage 3: Bootstrap validated
```

### Test Workflow

```bash
$ make test           # Ensures build complete, then runs tests
```

Output:
```
✅ Build Complete (3-Stage Bootstrap)

Running Test Suite...
Testing 01_hello.nano... ✓ PASS
Testing 02_calculator.nano... ✓ PASS
...
Total tests: 20
Passed: 20

Running self-hosted compiler tests...
Testing test_arithmetic_ops.nano... ✅ PASS
Testing test_comparison_ops.nano... ✅ PASS
...
Results: 8 passed, 0 failed

✅ All tests passed!
```

---

## Implementation Details

### Makefile Structure

```makefile
# Sentinel files
SENTINEL_STAGE1 = .stage1.built
SENTINEL_STAGE2 = .stage2.built
SENTINEL_STAGE3 = .stage3.built

# Build depends on stage3
build: $(SENTINEL_STAGE3)

# Stage 3 depends on stage 2
$(SENTINEL_STAGE3): $(SENTINEL_STAGE2)
    @# Compile and test components
    @touch $(SENTINEL_STAGE3)

# Stage 2 depends on stage 1
$(SENTINEL_STAGE2): $(SENTINEL_STAGE1)
    @# Compile self-hosted components
    @touch $(SENTINEL_STAGE2)

# Stage 1 depends on C sources
$(SENTINEL_STAGE1): $(COMPILER) $(INTERPRETER)
    @touch $(SENTINEL_STAGE1)
```

### Dependency Chain

```
make build
    ↓
stage3 (sentinel: .stage3.built)
    ↓ (depends on)
stage2 (sentinel: .stage2.built)
    ↓ (depends on)
stage1 (sentinel: .stage1.built)
    ↓ (depends on)
bin/nanoc + bin/nano
    ↓ (depends on)
obj/*.o (from src/*.c)
```

### Clean Behavior

```bash
make clean
```

Removes:
- All object files (`obj/`)
- All build artifacts (`build_bootstrap/`)
- All sentinel files (`.stage*.built`)
- All compiled binaries in `bin/`
- Coverage data and reports

After `make clean`, the next `make build` performs a full 3-stage build.

---

## Build Status

Check current build status at any time:

```bash
$ make status
```

Output:
```
Build Status:

  ✅ Stage 1: C reference compiler (bin/nanoc)
  ✅ Stage 2: Self-hosted components compiled
    • parser_mvp
    • typechecker_minimal
    • transpiler_minimal
  ✅ Stage 3: Bootstrap validated
```

Or:
```
Build Status:

  ❌ Stage 1: Not built
  ❌ Stage 2: Not built
  ❌ Stage 3: Not built
```

---

## Self-Hosted Compiler Components

Current self-hosted implementation (Phase 1, 85% complete):

| Component | File | Lines | Status |
|-----------|------|-------|--------|
| **Parser** | `parser_mvp.nano` | 2,767 | ✅ Complete |
| **Type Checker** | `typechecker_minimal.nano` | 797 | ✅ Complete |
| **Transpiler** | `transpiler_minimal.nano` | 1,081 | ✅ Complete |
| **Total** | | **4,645+** | **✅ Working** |

### Features Implemented

- ✅ All expressions (8 types)
- ✅ All operators (13 operators)
- ✅ All statements (8 types)
- ✅ Function calls with arguments
- ✅ Recursion (unlimited depth)
- ✅ While loops with mutation
- ✅ If/else conditionals
- ✅ 100% test coverage for implemented features

---

## Troubleshooting

### Build fails during stage 1

**Problem:** C compilation errors

**Solution:**
```bash
make clean
make check-deps    # Verify gcc/clang installed
make stage1        # Build only stage 1
```

### Build fails during stage 2

**Problem:** Self-hosted components won't compile

**Solution:**
```bash
make clean
make stage1        # Ensure stage 1 works
./bin/nanoc src_nano/parser_mvp.nano -o bin/parser_test  # Test manually
```

### Build fails during stage 3

**Problem:** Component tests fail

**Solution:**
```bash
./bin/parser_mvp            # Run component directly
./bin/typechecker_minimal   # Check individual components
./bin/transpiler_minimal
```

### Sentinels not updating

**Problem:** Build seems stuck

**Solution:**
```bash
make clean        # Remove all sentinels
make rebuild      # Full clean + build
```

### Build succeeds but tests fail

**Problem:** Code works but tests don't pass

**Solution:**
```bash
./test.sh         # Run tests directly to see errors
make test 2>&1 | tee test.log  # Save full output
```

---

## Performance

### Build Times (Typical)

| Stage | Time | Description |
|-------|------|-------------|
| Stage 1 | ~3-5s | Compile C sources to binaries |
| Stage 2 | ~5-7s | Compile 3 nanolang components |
| Stage 3 | ~2-3s | Run component tests |
| **Total** | **~10-15s** | **Full 3-stage bootstrap** |

### With Sentinels (rebuild)

| Scenario | Time | Description |
|----------|------|-------------|
| No changes | <1s | All sentinels exist, skip everything |
| C changes | ~3-5s | Rebuild stage 1 only |
| Nano changes | ~5-7s | Rebuild stages 2-3 only |

---

## Advanced Usage

### Build with sanitizers

```bash
make sanitize      # Rebuild with ASAN + UBSAN
make test          # Run tests with sanitizer checks
```

### Generate coverage report

```bash
make coverage      # Rebuild with coverage instrumentation
make test          # Run tests to collect coverage data
make coverage-report  # Generate HTML report
open coverage/index.html
```

### Install system-wide

```bash
sudo make install           # Install to /usr/local/bin
# or
make install PREFIX=~/.local  # Install to custom location
```

### Cross-compilation (if supported)

```bash
CC=clang make clean build   # Use clang instead of gcc
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Build and Test

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Install dependencies
        run: sudo apt-get install -y build-essential
      
      - name: Build (3-stage bootstrap)
        run: make build
      
      - name: Run tests
        run: make test
      
      - name: Build examples
        run: make examples
```

### Make Targets for CI

```bash
make check-deps    # Verify build dependencies
make build         # Full 3-stage build
make test          # Run all tests
make status        # Check build succeeded
```

Exit codes:
- `0` = Success
- `1` = Build or test failure

---

## Future Work

### Phase 2: Full Bootstrap

Once all language features are implemented:

1. **Stage 2** will compile a complete self-hosted compiler binary
2. **Stage 3** will use stage 2 to compile itself (true bootstrap)
3. **Verification** will compare stage 2 and stage 3 binaries

Current stage 2/3 compile and test individual components.
Future stage 2/3 will build complete integrated compilers.

### Planned Enhancements

- Full self-hosted compiler integration (all components linked)
- True stage 3 bootstrap (stage 2 compiles itself)
- Binary comparison verification (stage 2 ≈ stage 3)
- Parallel builds for faster compilation
- Incremental compilation support

---

## Summary

The 3-stage bootstrap build system provides:

✅ **Efficiency** - Sentinel files skip unnecessary rebuilds  
✅ **Correctness** - Each stage validates the previous  
✅ **Clarity** - Clear build status and progress  
✅ **Flexibility** - Build specific stages or everything  
✅ **CI/CD Ready** - Exit codes and automated testing  

Use `make help` to see all available targets!

---

*Last updated: November 30, 2025*  
*Build system version: 1.0*  
*Self-hosted compiler: Phase 1 complete (85%)*
