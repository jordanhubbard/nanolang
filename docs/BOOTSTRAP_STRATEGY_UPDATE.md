# Bootstrap Strategy Update: Write Tools in C, Not Python

## User Insight

"Why do tools require Python? Couldn't C also be used?"

**Answer**: YES! And it's actually MORE consistent with bootstrap philosophy.

## Bootstrap Principle

**Goal**: Require only C to bootstrap NanoLang
**Current**: C → NanoLang + Python dev tools (inconsistent!)
**Better**: C → NanoLang + C dev tools (consistent!)

## Why C Tools Make Sense

### 1. We Already Have the Infrastructure

```c
// modules/std/fs.c provides:
- walkdir_recursive() - Directory traversal
- DynArray infrastructure
- String handling

// We use cJSON for:
- JSON parsing
- JSON generation
```

### 2. Simple File Processing

**generate_module_index** (Python: 152 lines → C: ~200 lines)
```c
#include "modules/std/fs.h"
#include "cJSON.h"

int main() {
    // Walk modules/ directory
    DynArray* manifests = find_manifests("modules");
    
    // Parse each manifest with cJSON
    cJSON* index = cJSON_CreateObject();
    
    // Build index structure
    for (int i = 0; i < manifests->length; i++) {
        const char* path = dyn_array_get_string(manifests, i);
        cJSON* manifest = parse_manifest(path);
        // Add to index...
    }
    
    // Write output
    char* json_str = cJSON_Print(index);
    write_file("modules/index.json", json_str);
}
```

### 3. No Transpiler Blockers

**Problem with NanoLang tools**:
- NanoLang imports modules/std/fs.nano
- Transpiler generates incorrect C (dyn_array_get bugs)
- Can't compile

**With C tools**:
- C directly uses modules/std/fs.c
- No transpiler involved
- Just regular C compilation

### 4. Bootstrap Consistency

```
Current (Mixed):
C compiler → NanoLang examples
Python → Dev tools (inconsistent!)

Better (Pure):
C compiler → NanoLang examples
C tools → Dev tooling (consistent!)
```

## Implementation Plan

### Phase 1: generate_module_index (2 days)

**File**: `tools/generate_module_index.c`

```c
// Use existing infrastructure:
#include "../modules/std/fs.h"
#include "../deps/cJSON/cJSON.h"

// ~200 lines C
// Direct file system access
// Direct JSON manipulation
// No Python dependency
```

### Phase 2: estimate_feature_cost (1 day)

**File**: `tools/estimate_feature_cost.c`

```c
// Count lines in files
// Parse source with simple regex
// ~150 lines C
```

### Phase 3: merge_imports (1 day)

**File**: `tools/merge_imports.c`

```c
// Parse NanoLang source
// Deduplicate imports
// Rewrite file
// ~100 lines C
```

## Benefits

✅ **Eliminates Python entirely** (except test model generator)
✅ **Uses existing C infrastructure** (fs.c, cJSON)
✅ **No transpiler bugs** (direct C compilation)
✅ **Consistent with bootstrap principle** (C only)
✅ **Simpler build** (no Python dependency)
✅ **Better performance** (compiled C vs interpreted Python)

## Comparison

| Tool | Python | C | NanoLang (blocked) |
|------|--------|---|--------------------|
| generate_module_index | 152 lines | ~200 lines | 80% done, can't compile |
| estimate_feature_cost | 290 lines | ~150 lines | Not started |
| merge_imports | 124 lines | ~100 lines | Not started |
| **Total** | **566 lines** | **~450 lines** | **Blocked by transpiler** |

## Revised Recommendation

**OLD**: Keep Python tools (pragmatic but inconsistent)
**NEW**: Rewrite 3 tools in C (consistent with bootstrap philosophy!)

- generate_module_index.py → generate_module_index.c (2 days)
- estimate_feature_cost.py → estimate_feature_cost.c (1 day)
- merge_imports.py → merge_imports.c (1 day)

**Total**: 4 days to eliminate Python entirely (except test data)

## Updated Status

**Python Elimination**:
- Scripts removed: 2 (ONNX generator, schema validator)
- Scripts to rewrite in C: 3 (module index, cost estimator, import merger)
- Scripts to keep: 1 (examples/models/create_test_model.py - external test data)

**Result**: 86% Python elimination (6/7 scripts)

## Next Steps

1. Create `tools/generate_module_index.c` using fs.c + cJSON
2. Test against Python version output (should match exactly)
3. Replace in Makefile: `python3 tools/generate_module_index.py` → `./bin/generate_module_index`
4. Repeat for other 2 tools
5. Update BOOTSTRAP_AUDIT.md

This aligns perfectly with "C → NanoLang only" bootstrap principle!
