# Python Elimination Blocker: Module Import Path Resolution

## Problem

**NanoLang tools cannot import NanoLang modules** due to C FFI header resolution failure.

### Error

```
/tmp/nanoc_76439_gen_compiler_schema.c:18:10: fatal error: 'fs.h' file not found
   18 | #include <fs.h>  /* priority: 0 */
```

### Affected Tools

1. ✗ `tools/generate_module_index.nano` (80% complete, blocked)
2. ✗ `scripts/gen_compiler_schema.nano` (30% complete, blocked)
3. ✗ ALL future NanoLang tools that need modules

### Root Cause

When compiling standalone `.nano` files (tools/scripts), the compiler:
1. Correctly parses `from "modules/std/fs.nano" import ...`
2. Generates C include: `#include <fs.h>`
3. **FAILS**: C compiler can't find `modules/std/fs.h`

The problem: nanoc doesn't add module directories to C include paths for standalone tool compilation.

## Impact

**BLOCKS Python elimination roadmap:**
- ✗ P0: gen_compiler_schema.py → .nano (BLOCKED)
- ✗ P1: generate_module_index.py → .nano (BLOCKED)
- ✗ P2: build_module.sh → .nano (BLOCKED)
- ✗ P3: estimate_feature_cost.py → .nano (BLOCKED)
- ✗ P3: merge_imports.py → .nano (BLOCKED)

## Workaround Options

### Option A: Fix nanoc Module Resolution (Best, but complex)

**Solution:** Teach nanoc to add `-I` flags for all imported module directories.

```bash
# Current (fails):
nanoc scripts/gen_compiler_schema.nano -o bin/gen_compiler_schema

# Should generate:
gcc ... -I modules/std -I modules/std/json -I ... /tmp/nanoc_X_gen_compiler_schema.c
```

**Complexity:** Medium-High (compiler changes)
**Time:** 2-4 days

### Option B: Python → Shell Scripts (Fastest)

**Solution:** Rewrite Python tools as shell scripts using standard UNIX tools.

```bash
# Example: validate_schema_sync.py → validate_schema_sync.sh
# Uses: grep, awk, diff, jq (already installed)
```

**Benefits:**
- ✅ No Python dependency
- ✅ No nanoc changes needed
- ✅ Fast to implement (1 day per script)
- ✅ Works TODAY

**Downsides:**
- Shell is less elegant than NanoLang
- Still need external tools (jq for JSON)

### Option C: C Tools (Most Control)

**Solution:** Rewrite Python tools in C using cJSON.

**Benefits:**
- ✅ No Python dependency
- ✅ Full control, no compiler issues
- ✅ Can be part of nanoc itself

**Downsides:**
- Verbose (C is 3-5x more code than NanoLang)
- Time-consuming

### Option D: Hybrid Approach

**Solution:** 
1. **Critical build tools (P0)** → C (gen_compiler_schema)
2. **Validation tools (P2)** → Shell (validate_schema_sync)
3. **Utilities (P3)** → Wait for nanoc fix, then NanoLang

## Recommendation

**SHORT TERM (This Week):**
- Option B: Rewrite `validate_schema_sync.py` → shell script
- Option C: Rewrite `gen_compiler_schema.py` → C (integrate into nanoc?)

**MEDIUM TERM (Next Week):**
- Option A: Fix nanoc module resolution for standalone tools

**LONG TERM (After nanoc fix):**
- Complete NanoLang versions of all tools
- Remove Python completely

## Current Status (2026-01-01)

### Completed:
- ✅ ONNX module removed (eliminated 158 lines Python test data generator)

### Blocked:
- ⏸️ nanolang-261g (P0): gen_compiler_schema.py → .nano
- ⏸️ nanolang-ofgl (P1): generate_module_index.nano

### Alternative Path Forward:
- 🔄 Create nanolang-XXXX: Fix nanoc module resolution for standalone tools
- 🔄 Create nanolang-YYYY: Rewrite gen_compiler_schema.py in C
- ✅ Proceed with nanolang-8853: validate_schema_sync.py → .sh (P2, no blocker)

---

**Decision Required:** Which option should we pursue?
