# Bootstrap Dependency Audit

> **Principle:** NanoLang should bootstrap from **C only**, then use **NanoLang** for all tooling. Shell scripts (`/bin/sh`) are acceptable for platform scripts. **Python should be eliminated.**

---

## Current State

### Python Scripts (7 total, 1,422 lines)

| File | LOC | Used in Build? | Status | Action |
|------|-----|----------------|--------|--------|
| `scripts/gen_compiler_schema.py` | 232 | ✅ **CRITICAL** | Production | **Rewrite in NanoLang** (P0) |
| `tools/generate_module_index.py` | 152 | ✅ Yes (`make modules-index`) | Production | **Already started** (nanolang-ofgl) |
| `tools/validate_schema_sync.py` | 336 | ❌ No | Utility | **Rewrite in NanoLang** or shell (P2) |
| `tools/estimate_feature_cost.py` | 290 | ❌ No | Utility | **Rewrite in NanoLang** (P3) |
| `tools/merge_imports.py` | 124 | ❌ No | Utility | **Rewrite in NanoLang** (P3) |
| `scripts/check_markdown_links.py` | 130 | ❌ No | Documentation | **Rewrite in shell** or remove (P4) |
| `examples/models/create_test_model.py` | 158 | ❌ No | ONNX test data | **Keep** (external tool) |

**Total to eliminate:** 1,264 lines (excluding ONNX test tool)

---

## Priority Ranking

### P0 - Critical Build Blocker (MUST DO)

#### 1. `scripts/gen_compiler_schema.py` (232 lines)
**What it does:**
- Reads `schema/compiler_schema.json`
- Generates C header: `src/generated/compiler_schema.h`
- Generates NanoLang enums: `src_nano/generated/compiler_schema.nano`
- Generates AST types: `src_nano/generated/compiler_ast.nano`
- **Used in build:** `make schema` (required by `make build`)

**Rewrite plan:**
- Use `modules/std/json/json.nano` for JSON parsing
- Use `modules/std/fs.nano` for file I/O
- Template generation in NanoLang
- Estimated: 300-400 lines of NanoLang

**Blocking:** This is the **ONLY** Python dependency in the critical build path (besides module index).

---

### P1 - Active Work (IN PROGRESS)

#### 2. `tools/generate_module_index.py` (152 lines)
**Status:** ✅ 80% complete in NanoLang (`tools/generate_module_index.nano`)  
**Blocker:** Module import path resolution  
**Bead:** nanolang-ofgl

---

### P2 - Validation Tools (SHOULD DO)

#### 3. `tools/validate_schema_sync.py` (336 lines)
**What it does:**
- Validates C and NanoLang AST/IR definitions match schema
- Checks generated files are in sync
- Reports drift between implementations

**Rewrite options:**
- **Option A:** NanoLang with JSON + regex
- **Option B:** Shell script with `grep`/`awk`/`diff`
- **Recommendation:** Shell script (validation is simple pattern matching)

**Estimated:** 100-150 lines of shell

---

### P3 - Developer Utilities (NICE TO HAVE)

#### 4. `tools/estimate_feature_cost.py` (290 lines)
**What it does:**
- Estimates implementation cost for new language features
- Quantifies dual-implementation overhead (C + NanoLang)
- Interactive cost calculator

**Rewrite plan:**
- NanoLang with JSON config for cost matrix
- CLI interface with `println`/`read_line`
- Estimated: 350-400 lines of NanoLang

---

#### 5. `tools/merge_imports.py` (124 lines)
**What it does:**
- Merges NanoLang source file with all imports into single file
- Handles circular dependencies
- Prevents duplicates

**Rewrite plan:**
- NanoLang with file I/O and string manipulation
- Dependency graph building
- Estimated: 200-250 lines of NanoLang

---

### P4 - Documentation Tools (LOW PRIORITY)

#### 6. `scripts/check_markdown_links.py` (130 lines)
**What it does:**
- Validates markdown links in documentation
- Checks for broken internal links
- Reports stale references

**Rewrite options:**
- **Option A:** NanoLang with regex (if regex module exists)
- **Option B:** Shell script with `find` + `grep`
- **Option C:** Remove (not critical)
- **Recommendation:** Simple shell script or remove

---

### KEEP (External Tool)

#### 7. `examples/models/create_test_model.py` (158 lines)
**What it does:**
- Creates ONNX test models using PyTorch
- Test data generation for ONNX module

**Status:** ✅ **Keep as-is**  
**Reason:** External tool dependency (PyTorch). Not part of NanoLang build. Only needed for ONNX module development.

---

## Shell Scripts (Acceptable - 21 files)

Shell scripts are **acceptable** per the bootstrap principle (`/bin/sh` is universal):

**Build & Bootstrap:**
- `scripts/bootstrap.sh` - 3-stage bootstrap
- `scripts/bootstrap-macos.sh` - macOS-specific setup
- `scripts/assemble_compiler.sh` - Compiler assembly

**Testing:**
- `tests/run_all_tests.sh` - Test runner
- `tests/run_negative_tests.sh` - Error case tests
- `tests/selfhost/run_selfhost_tests.sh` - Self-hosting tests

**Validation:**
- `scripts/check_compiler_schema.sh` - Schema validation
- `scripts/verify_no_nanoc_c.sh` - Build verification
- `tools/check_feature_parity.sh` - Feature parity checks
- `tools/differential_test.sh` - Differential testing

**Module System:**
- `modules/tools/install_module.sh` - Module installation
- `modules/tools/build_module.sh` - Module building
- `modules/tools/dep_locator.sh` - Dependency resolution
- `scripts/check-module-deps.sh` - Module dependency checks

**Utilities:**
- `scripts/generate_list.sh` - List generation
- `scripts/watch.sh` - File watching
- `scripts/benchmark.sh` - Benchmarking
- `scripts/reorganize_bin.sh` - Binary organization

**Status:** ✅ **All shell scripts are acceptable and should remain.**

---

## Implementation Plan

### Phase 1: Critical Path (P0)
**Goal:** Remove Python from build system

- [ ] **Task 1.1:** Rewrite `gen_compiler_schema.py` in NanoLang
  - Create `scripts/gen_compiler_schema.nano`
  - Implement JSON parsing
  - Implement template generation (C header, NanoLang enums, AST types)
  - Add shadow tests
  - Update Makefile to use NanoLang version
  - **Estimated:** 2-3 days

- [ ] **Task 1.2:** Complete `generate_module_index.nano`
  - Fix module import paths
  - Implement full reverse index building
  - Add shadow tests
  - Update Makefile
  - Remove Python version
  - **Status:** 80% complete (nanolang-ofgl)
  - **Estimated:** 1 day

**Milestone:** Build system is 100% Python-free

---

### Phase 2: Validation Tools (P2)
**Goal:** Rewrite validation tools in shell

- [ ] **Task 2.1:** Rewrite `validate_schema_sync.py` as shell script
  - Create `scripts/validate_schema_sync.sh`
  - Use `grep`/`awk`/`diff` for pattern matching
  - Add to CI/CD if needed
  - Remove Python version
  - **Estimated:** 1 day

**Milestone:** No Python in validation pipeline

---

### Phase 3: Developer Utilities (P3)
**Goal:** Rewrite developer tools in NanoLang

- [ ] **Task 3.1:** Rewrite `estimate_feature_cost.py` in NanoLang
  - Create `tools/estimate_feature_cost.nano`
  - JSON config for cost matrix
  - CLI interface
  - **Estimated:** 2 days

- [ ] **Task 3.2:** Rewrite `merge_imports.py` in NanoLang
  - Create `tools/merge_imports.nano`
  - Dependency graph building
  - Circular dependency detection
  - **Estimated:** 2 days

**Milestone:** All developer tools in NanoLang

---

### Phase 4: Documentation Tools (P4)
**Goal:** Simplify or remove doc tools

- [ ] **Task 4.1:** Replace or remove `check_markdown_links.py`
  - **Option A:** Simple shell script with `find` + `grep`
  - **Option B:** Remove (low value)
  - **Decision:** TBD

---

## Success Criteria

### Tier 1: Critical (MUST HAVE)
- ✅ Build system works without Python
- ✅ `make` completes successfully
- ✅ Bootstrap works end-to-end
- ✅ Schema generation works
- ✅ Module index generation works

### Tier 2: Validation (SHOULD HAVE)
- ✅ Schema sync validation works (shell or NanoLang)
- ✅ CI/CD passes without Python

### Tier 3: Developer Experience (NICE TO HAVE)
- ✅ All developer tools in NanoLang
- ✅ Feature cost estimation works
- ✅ Import merging works

---

## Benefits

### 1. **Clean Bootstrap**
- **Before:** C → Python → NanoLang
- **After:** C → NanoLang

### 2. **Reduced Dependencies**
- Eliminates Python runtime dependency
- Eliminates Python package ecosystem
- Simplifies CI/CD (no `pip install`)

### 3. **Dogfooding**
- Uses NanoLang to build NanoLang
- Validates language capabilities for real tooling
- Demonstrates "NanoLang is production-ready"

### 4. **Self-Sufficiency**
- Language can build itself with only C compiler
- No external runtimes required
- True systems language

---

## Metrics

### Current State
- **Python LOC:** 1,422 lines
- **Build dependency:** 2 Python scripts (critical)
- **Runtime dependency:** Python 3.x

### Target State (After Phase 1)
- **Python LOC in build:** 0 lines
- **Build dependency:** 0 Python scripts
- **Runtime dependency:** None

### Target State (After Phase 3)
- **Python LOC total:** 158 lines (ONNX test tool only)
- **Build dependency:** 0 Python scripts
- **Runtime dependency:** None (except ONNX dev)

---

## Timeline

| Phase | Tasks | Estimated | Priority |
|-------|-------|-----------|----------|
| Phase 1 | Schema generator + Module index | 3-4 days | **P0 - Critical** |
| Phase 2 | Validation tools (shell) | 1 day | **P2 - Important** |
| Phase 3 | Developer utilities | 4 days | **P3 - Nice-to-have** |
| Phase 4 | Doc tools | 1 day | **P4 - Optional** |
| **Total** | | **8-10 days** | |

---

## Decision: ONNX Test Tool

**Keep `examples/models/create_test_model.py`**

**Rationale:**
- External tool (PyTorch) for test data generation
- Only used during ONNX module development
- Not part of build system
- Not a bootstrap dependency
- Users can skip ONNX if they don't have PyTorch

**Alternative:** Could rewrite in C with ONNX C API, but low ROI.

---

## Next Steps

1. **Immediate:** Create bead for P0 schema generator rewrite
2. **Next:** Complete module index generator (nanolang-ofgl)
3. **Then:** Tackle validation tools (shell scripts)
4. **Later:** Developer utilities (as time allows)

---

## Appendix: Module Requirements

### NanoLang Modules Needed

All these modules **already exist**:

- ✅ `modules/std/json/json.nano` - JSON parsing/generation
- ✅ `modules/std/fs.nano` - File system operations
- ✅ `modules/std/io/stdio.nano` - File I/O
- ✅ Built-in string operations - String manipulation

### C Libraries Needed

All available on every platform:

- ✅ `<stdio.h>` - Standard I/O
- ✅ `<stdlib.h>` - Memory allocation
- ✅ `<string.h>` - String operations

**Conclusion:** NanoLang has everything needed to replace Python.

---

**Status:** Audit complete. Ready to proceed with Phase 1.

