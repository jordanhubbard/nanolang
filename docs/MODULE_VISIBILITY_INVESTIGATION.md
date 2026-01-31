# Module Visibility Investigation - nanolang-zmwa

**Date:** 2026-01-21  
**Issue:** Self-hosted compiler cannot see transitively imported types  
**Impact:** 191 parser errors when compiling `nanoc_v06.nano`  

## Problem Statement

When the self-hosted compiler (`nanoc_v06.nano`) attempts to compile `transpiler.nano`:
- `transpiler.nano` imports `ast_shared.nano`
- `ast_shared.nano` imports `compiler_ast.nano`
- `compiler_ast.nano` defines types like `Parser`, `ASTReturn`, `GenEnv`
- `transpiler.nano` uses these types in function signatures
- **ERROR:** Self-hosted compiler reports "I cannot find a variable named Parser"

The C reference compiler (`bin/nanoc_c`) works fine with the same code.

## Root Cause Analysis

### C Compiler Behavior (WORKS) ✅

Located in `src/module.c`:

```c
static ASTNode *load_module_internal(const char *module_path, Environment *env, ...) {
    // 1. Parse module source
    ASTNode *module_ast = parse_program(tokens, token_count);
    
    // 2. RECURSIVELY process imports (line 511)
    if (!process_imports(module_ast, env, modules_to_track, module_path)) {
        // This loads compiler_ast.nano when processing ast_shared.nano
    }
    
    // 3. Type check with all transitive symbols available
    if (!type_check_module(module_ast, env)) { ... }
    
    // 4. Cache AST for reuse
    cache_module_with_ast(module_path, module_ast);
}
```

**Key insight:** `process_imports()` at line 510-517 recursively loads ALL transitive imports before type checking.

### Self-Hosted Compiler Behavior (BROKEN) ❌

Located in `src_nano/nanoc_v06.nano` line 927:

```nano
let parser: Parser = (parse_program tokens token_count source_to_compile)
```

**Problem:** 
- Only parses the main file
- Does NOT recursively process imports
- Types from `compiler_ast.nano` never get loaded into the symbol table
- When typechecker encounters `parser: Parser` in transpiler.nano, it fails

## Architectural Comparison

| Feature | C Compiler | Self-Hosted Compiler |
|---------|-----------|---------------------|
| Parse main file | ✅ | ✅ |
| Parse direct imports | ✅ | ❌ |
| Parse transitive imports | ✅ | ❌ |
| Build global symbol table | ✅ | ❌ |
| Cache loaded modules | ✅ | ❌ |
| Prevent circular imports | ✅ | ❌ |

## Solution Design

### Option 1: Add Recursive Import Loading (RECOMMENDED)

Implement equivalent of C's `process_imports()` in NanoLang:

```nano
fn process_imports(parser: Parser, env: array<Symbol>, module_cache: ModuleCache) -> array<Symbol> {
    let import_count: int = (parser_get_import_count parser)
    let mut i: int = 0
    let mut updated_env: array<Symbol> = env
    
    while (< i import_count) {
        let import_node: ASTImport = (parser_get_import parser i)
        let module_path: string = (resolve_module_path import_node.module_path)
        
        /* Check cache to prevent reloading */
        if (not (module_cache_contains module_cache module_path)) {
            /* Load and parse module */
            let module_source: string = (file_read module_path)
            let module_tokens: List<LexerToken> = (tokenize_file module_source)
            let module_parser: Parser = (parse_program module_tokens)
            
            /* RECURSIVELY process module's imports */
            set updated_env (process_imports module_parser updated_env module_cache)
            
            /* Extract type definitions from module */
            set updated_env (extract_type_symbols module_parser updated_env)
            
            /* Cache module */
            set module_cache (module_cache_add module_cache module_path module_parser)
        } else {
            /* Use cached symbols */
            let cached_parser: Parser = (module_cache_get module_cache module_path)
            set updated_env (extract_type_symbols cached_parser updated_env)
        }
        
        set i (+ i 1)
    }
    
    return updated_env
}
```

**New components needed:**
1. `ModuleCache` struct to track loaded modules
2. `resolve_module_path()` to find .nano files
3. `extract_type_symbols()` to get struct/enum/union definitions
4. Integration into `nanoc_v06.nano` compilation pipeline

**Estimated effort:** 2-3 days

### Option 2: Flatten Import Graph (WORKAROUND)

Manually inline all transitive imports in each file:

```nano
/* transpiler.nano */
import "src_nano/generated/compiler_schema.nano"  /* Add explicitly */
import "src_nano/generated/compiler_ast.nano"     /* Add explicitly */
import "src_nano/ast_shared.nano"
```

**Pros:** Quick fix, no architectural changes  
**Cons:** Brittle, violates DRY, doesn't scale

### Option 3: Use `extern struct` Declarations (WORKAROUND)

Add forward declarations in files that need transitive types:

```nano
/* transpiler.nano */
extern struct Parser { /* minimal forward decl */ }
extern struct ASTReturn { /* minimal forward decl */ }

import "src_nano/ast_shared.nano"
```

**Pros:** No import changes needed  
**Cons:** Duplicate type definitions, maintenance burden

## Recommendation

**Implement Option 1** (Recursive Import Loading)

This is the proper architectural fix that:
- Matches C compiler behavior
- Scales to complex module graphs
- Enables true self-hosting
- Required for production-ready self-hosted compiler

The workarounds (Options 2-3) are technical debt that will block future development.

## Implementation Plan

### Phase 1: Core Infrastructure (1 day)
1. Create `ModuleCache` struct and helper functions
2. Implement `resolve_module_path()` 
3. Add `extract_type_symbols()` to build symbol table from Parser

### Phase 2: Recursive Loading (1 day)
4. Implement `process_imports()` with recursion
5. Add circular dependency detection
6. Test with simple 2-level import case

### Phase 3: Integration (0.5 days)
7. Integrate into `nanoc_v06.nano` compilation pipeline
8. Test with `transpiler.nano` → `ast_shared.nano` → `compiler_ast.nano` chain

### Phase 4: Validation (0.5 days)
9. Compile `nanoc_v06.nano` with self-hosted compiler
10. Verify all 191 errors resolved
11. Bootstrap test: compile compiler with itself

**Total estimate: 2-3 days**

## Files to Modify

### New Files
- `src_nano/compiler/module_loader.nano` - Module loading/caching logic
- `src_nano/compiler/symbol_extractor.nano` - Extract types from Parser

### Modified Files
- `src_nano/nanoc_v06.nano` - Add import processing before type checking
- `src_nano/typecheck.nano` - Accept pre-built symbol table from imports

### Test Files
- `tests/module_transitive_imports.nano` - Test case for 2-level imports
- `tests/module_circular_imports.nano` - Test circular dependency detection

## Success Criteria

1. ✅ Self-hosted compiler can compile `transpiler.nano` without errors
2. ✅ All 191 "I cannot find a variable named X" errors resolved
3. ✅ `nanoc_v06.nano` can compile itself (bootstrap test)
4. ✅ No regressions in C compiler behavior
5. ✅ Module cache prevents infinite loops on circular imports

## Notes

- The C compiler's `process_imports()` (src/module.c:724) is the reference implementation
- Module caching is CRITICAL to prevent exponential load times
- Circular import detection prevents infinite recursion
- Symbol extraction must handle `extern struct` declarations specially

## References

- C implementation: `src/module.c` lines 459-559, 724-824
- Current self-hosted driver: `src_nano/nanoc_v06.nano` line 927
- Type resolution: `src_nano/typecheck.nano` line 775-851
- Issue: nanolang-zmwa
