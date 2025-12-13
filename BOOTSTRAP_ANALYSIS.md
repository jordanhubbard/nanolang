# Bootstrap Architecture Analysis

## Expected vs Actual Behavior

### Expected Architecture (User's Understanding)
```
make clean bootstrap test examples
↓
1. Clean everything
2. Bootstrap: Stage 0 → Stage 1 → Stage 2 → Stage 3
3. Test: Use Stage 3 (nanoc_stage2) to run tests
4. Examples: Use Stage 3 (nanoc_stage2) to build examples
```

### Actual Implementation (What Really Happens)

#### Two Separate Bootstrap Processes

**Process A: TRUE Bootstrap (nanoc_v05.nano)**
- Target: `make bootstrap` → `$(SENTINEL_BOOTSTRAP3)`
- Stage 0: C sources → `bin/nanoc_c` (C reference compiler)
- Stage 1: `nanoc_c` compiles `nanoc_v05.nano` → `bin/nanoc_stage1`
- Stage 2: `nanoc_stage1` compiles `nanoc_v05.nano` → `bin/nanoc_stage2`
- Stage 3: Verify `nanoc_stage1` == `nanoc_stage2` (reproducibility check)
- Result: ✅ SUCCEEDS - Creates `nanoc_stage1` and `nanoc_stage2`

**Process B: Component Bootstrap (parser/typechecker/transpiler)**
- Target: `make test` → `make build` → `$(SENTINEL_STAGE3)`
- Stage 1: C sources → `bin/nanoc` + `bin/nano`
- Stage 2: `bin/nanoc` compiles `parser_mvp`, `typechecker_minimal`, `transpiler_minimal`
- Stage 3: Validate components
- Result: ❌ FAILS - Parser compilation has warnings, only 2/3 components work

#### The Critical Issue

**Line 224-225 of Makefile:**
```makefile
$(COMPILER): $(COMPILER_C) | $(BIN_DIR)
	@ln -sf nanoc_c $(COMPILER)
```

This means `bin/nanoc` is **always** a symlink to `bin/nanoc_c` (the C compiler), not to the self-hosted `nanoc_stage2`!

**Current State:**
```
bin/nanoc -> bin/nanoc_c (C reference compiler, 462KB)
bin/nanoc_stage1 (self-hosted, 74KB)  ✅ EXISTS BUT UNUSED
bin/nanoc_stage2 (self-hosted, 74KB)  ✅ EXISTS BUT UNUSED
```

## Flow Analysis

### What `make clean bootstrap test examples` Actually Does:

1. **Clean**: ✅ Works correctly
   - Removes all artifacts
   - Clears all sentinels

2. **Bootstrap** (Process A): ✅ Works correctly
   - Creates `nanoc_c`, `nanoc_stage1`, `nanoc_stage2`
   - Verifies self-hosting (stage1 vs stage2 comparison)
   - Creates `bin/nanoc` → `nanoc_c` symlink
   - Sets `.bootstrap0.built` through `.bootstrap3.built`

3. **Test** (depends on `build`, which triggers Process B):
   - Requires `$(SENTINEL_STAGE3)`
   - Builds Stage 1: Creates `bin/nanoc` → `nanoc_c` symlink (C compiler)
   - Builds Stage 2: Uses `bin/nanoc` (C compiler) to build components
   - **Problem**: Uses C compiler instead of self-hosted compiler!
   - Fails: Parser has warnings, only 2/3 components build
   - Test suite: 1 test fails (`test_generics_comprehensive.nano`)

4. **Examples** (depends on `build`):
   - Uses `bin/nanoc` (C compiler) to build examples
   - **Problem**: Not using the validated self-hosted compiler!

## The Architectural Disconnect

### What the User Expected:
```
bootstrap → nanoc_stage2 becomes the primary compiler
         → test uses nanoc_stage2
         → examples uses nanoc_stage2
```

### What Actually Happens:
```
bootstrap → creates nanoc_stage2 (unused artifact)
         → bin/nanoc still points to nanoc_c
         → test uses nanoc_c (C compiler)
         → examples uses nanoc_c (C compiler)
```

## Issues Identified

### 1. **Unused Bootstrap Artifacts**
The entire TRUE bootstrap creates `nanoc_stage1` and `nanoc_stage2`, but nothing uses them!
- They exist in `bin/` for verification only
- `make bootstrap-install` is required to actually USE the self-hosted compiler

### 2. **Dual Bootstrap Systems**
There are two incompatible bootstrap philosophies:
- **TRUE Bootstrap**: GCC-style compiler-compiling-compiler
- **Component Bootstrap**: Modular self-hosted components

These don't integrate - they run independently.

### 3. **Wrong Compiler Used**
After bootstrap, the system should use the validated self-hosted compiler (`nanoc_stage2`), but instead uses the C reference compiler (`nanoc_c`).

### 4. **Component Bootstrap Fails**
The component bootstrap (parser_mvp, etc.) has issues:
- Parser compilation has warnings
- Only 2/3 components build successfully
- This suggests the components are outdated or incompatible

### 5. **Misleading Makefile Comments**
The Makefile header says:
```makefile
# BUILD TARGETS (Component Testing):
# - Stage 1: C Reference Compiler (from C sources)
# - Stage 2: Self-Hosted Components (compiled with stage1, tested individually)  
# - Stage 3: Component Validation (test components)
```

But this describes the component bootstrap, not the TRUE bootstrap.

## Recommendations

### Option 1: Integrate TRUE Bootstrap with Build (Recommended)
Make the TRUE bootstrap the default path:
```makefile
# After bootstrap3, replace $(COMPILER) target
$(COMPILER): $(SENTINEL_BOOTSTRAP3)
	@ln -sf nanoc_stage2 $(COMPILER)
```

This ensures `make bootstrap test examples` uses the self-hosted compiler.

### Option 2: Remove Component Bootstrap
If the component bootstrap (parser_mvp, etc.) is obsolete:
- Remove Stage 2/Stage 3 from the build targets
- Keep only the TRUE bootstrap
- Simplify to: C compiler → TRUE bootstrap → done

### Option 3: Fix Component Bootstrap
If components are still needed:
- Fix parser_mvp compilation warnings
- Update components to work with current language version
- Integrate them into the bootstrap chain

### Option 4: Clarify Documentation
At minimum:
- Document that `make bootstrap` doesn't install the compiler
- Require `make bootstrap-install` to use self-hosted version
- Explain the dual bootstrap systems

## Test Failure Analysis

The test suite shows:
- **53 passed, 1 failed, 7 skipped**
- Failed test: `test_generics_comprehensive.nano`

This suggests a language feature issue with generics, not a bootstrap issue per se.

## Summary

**The 3-stage bootstrap architecture exists but wasn't properly integrated (NOW FIXED):**

1. ✅ TRUE Bootstrap works (creates self-hosted compiler)
2. ✅ Build system now automatically installs the bootstrapped compiler
3. ✅ Tests run with self-hosted compiler
4. ✅ Examples build with self-hosted compiler
5. ⚠️ Component bootstrap is broken/obsolete (separate issue)

**Fixed Implementation:**

The following changes were made to Makefile and nanoc_v05.nano:

1. **Makefile Bootstrap Stage 3** now automatically updates `bin/nanoc` symlink:
   - After verification, `bin/nanoc` → `nanoc_stage2` (was: → `nanoc_c`)
   - All subsequent builds use self-hosted compiler

2. **nanoc_v05.nano CLI Updated** to support both modes:
   - `nanoc input.nano output` - Compile to binary
   - `nanoc input.nano` - Run shadow tests only (test suite compatibility)

3. **$(COMPILER) Target** respects bootstrap:
   - Checks if bootstrap3 completed
   - If yes: symlink to `nanoc_stage2`
   - If no: symlink to `nanoc_c`

**Test Results with Self-Hosted Compiler:**
- ✅ 53 tests passed
- ❌ 1 test failed (test_generics_comprehensive.nano - pre-existing issue)
- ⊘ 7 tests skipped (expected failures)

**User's expectation was correct** - the architecture now uses the validated 3-stage bootstrap for everything downstream as intended.

## Status: RESOLVED ✅

The bootstrap architecture now works as expected:
```bash
make clean bootstrap test examples
```
This will:
1. Clean everything
2. Bootstrap through 3 stages (C → stage1 → stage2)
3. Install stage2 as `bin/nanoc`
4. Run tests with self-hosted compiler ✅
5. Build examples with self-hosted compiler ✅
