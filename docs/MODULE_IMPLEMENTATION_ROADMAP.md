# Module System Redesign: Implementation Roadmap

**Date:** 2025-01-08  
**Status:** 🔴 Proposal - Awaiting Approval  
**Estimated Time:** 6-8 weeks total

---

## Overview

This roadmap breaks down the module system redesign into 5 phases, each deliverable independently. Each phase adds value without breaking existing code.

---

## Phase 1: Module-Level Safety Annotations

**Duration:** 1-2 weeks  
**Priority:** P4 - Critical  
**Goal:** Eliminate scattered `unsafe {}` blocks

### Changes Required

#### 1.1 Parser Changes

**File:** `src/parser_iterative.c`

**Add AST Node:**
```c
typedef struct ModuleDeclaration {
    char *name;
    bool is_unsafe;
    ASTNode *body;  // List of functions, structs, etc.
    int line;
    int column;
} ModuleDeclaration;
```

**Parse Function:**
```c
static ASTNode *parse_module_declaration(Parser *p) {
    // Parse: unsafe module sdl { ... }
    bool is_unsafe = false;
    
    if (check_keyword(p, "unsafe")) {
        is_unsafe = true;
        advance(p);
    }
    
    if (!check_keyword(p, "module")) {
        error(p, "Expected 'module'");
        return NULL;
    }
    advance(p);
    
    Token *name_token = expect(p, TOKEN_IDENTIFIER, "Expected module name");
    
    expect(p, TOKEN_LBRACE, "Expected '{' after module name");
    
    // Parse module body (functions, structs, etc.)
    ASTNode *body = parse_module_body(p);
    
    expect(p, TOKEN_RBRACE, "Expected '}' at end of module");
    
    ModuleDeclaration *mod = malloc(sizeof(ModuleDeclaration));
    mod->name = strdup(name_token->value);
    mod->is_unsafe = is_unsafe;
    mod->body = body;
    
    return create_ast_node(AST_MODULE_DECLARATION, mod);
}
```

**Integration:**
- Modify `parse_program()` to recognize `module` keyword
- Store module safety info in environment

---

#### 1.2 Environment Changes

**File:** `src/nanolang.h`

**Add to Environment:**
```c
typedef struct {
    char *name;
    bool is_unsafe;
    bool has_ffi;  // Track if module has extern functions
} ModuleInfo;

typedef struct Environment {
    /* Existing fields... */
    ModuleInfo *modules;
    int module_count;
    int module_capacity;
    char *current_module_name;  // Track which module we're in
    bool current_module_unsafe;  // Is current module unsafe?
} Environment;
```

**Helper Functions:**
```c
void env_register_module(Environment *env, const char *name, bool is_unsafe);
bool env_is_current_module_unsafe(Environment *env);
ModuleInfo *env_get_module(Environment *env, const char *name);
```

---

#### 1.3 Typechecker Changes

**File:** `src/typechecker.c`

**Modify `check_function_call()`:**
```c
static void check_function_call(ASTNode *call, Environment *env) {
    Function *func = env_get_function(env, call->as.call.function_name);
    
    if (func && func->is_extern) {
        // This is an FFI call
        
        // Check if we're in an unsafe module
        if (env_is_current_module_unsafe(env)) {
            // ✅ Allowed - we're in unsafe module
            return;
        }
        
        // Check if we're in an unsafe block
        if (env->in_unsafe_block) {
            // ✅ Allowed - we're in unsafe block
            return;
        }
        
        // ❌ Error - extern call requires unsafe
        error("Call to extern function '%s' requires unsafe block or unsafe module",
              call->as.call.function_name);
    }
}
```

**Add Module Tracking:**
```c
static void typecheck_module_declaration(ASTNode *node, Environment *env) {
    ModuleDeclaration *mod = node->as.module_declaration;
    
    // Register module
    env_register_module(env, mod->name, mod->is_unsafe);
    
    // Set current module context
    char *prev_module = env->current_module_name;
    bool prev_unsafe = env->current_module_unsafe;
    
    env->current_module_name = mod->name;
    env->current_module_unsafe = mod->is_unsafe;
    
    // Typecheck module body
    typecheck_program(mod->body, env);
    
    // Restore previous context
    env->current_module_name = prev_module;
    env->current_module_unsafe = prev_unsafe;
}
```

---

#### 1.4 Import Statement Changes

**File:** `src/parser_iterative.c`

**Parse Import with Safety:**
```c
static ASTNode *parse_import(Parser *p) {
    // Parse: import unsafe "modules/sdl/sdl.nano"
    bool expect_unsafe = false;
    
    if (check_keyword(p, "import")) {
        advance(p);
        
        // Check for safety annotation
        if (check_keyword(p, "unsafe")) {
            expect_unsafe = true;
            advance(p);
        } else if (check_keyword(p, "safe")) {
            expect_unsafe = false;
            advance(p);
        }
    }
    
    Token *path_token = expect(p, TOKEN_STRING, "Expected module path");
    
    ImportStmt *import = malloc(sizeof(ImportStmt));
    import->module_path = strdup(path_token->value);
    import->expect_unsafe = expect_unsafe;
    
    return create_ast_node(AST_IMPORT, import);
}
```

**Validation in Typechecker:**
```c
static void check_import_safety(ImportStmt *import, Environment *env) {
    ModuleInfo *mod = env_get_module(env, import->module_path);
    
    if (!mod) return;  // Module not loaded yet
    
    if (import->expect_unsafe && !mod->is_unsafe) {
        warning("Import marked 'unsafe' but module '%s' is safe", mod->name);
    }
    
    if (!import->expect_unsafe && mod->is_unsafe) {
        // Optionally warn (controlled by --warn-unsafe-imports)
        if (env->warn_unsafe_imports) {
            warning("Importing unsafe module '%s' without 'unsafe' annotation", mod->name);
        }
    }
}
```

---

#### 1.5 Backward Compatibility

**Strategy:** Gradual migration

1. **Phase 1a (Week 1):** Parser recognizes `unsafe module` but doesn't enforce
2. **Phase 1b (Week 2):** Typechecker uses module safety, but still allows old-style `unsafe {}` blocks
3. **Phase 1c (Future):** Deprecation warnings for scattered `unsafe {}`
4. **Phase 1d (Future):** Remove support for `unsafe {}` inside `unsafe module`

**Migration Tool:**
```bash
# Auto-convert modules with extern functions
$ nalang migrate --add-unsafe-module modules/sdl/sdl.nano

# Output:
unsafe module sdl {
    extern fn SDL_Init(flags: int) -> int
    /* ... rest of file ... */
}
```

---

### Testing

**Test Cases:**
1. `tests/test_unsafe_module_basic.nano` - Simple unsafe module
2. `tests/test_unsafe_module_ffi.nano` - FFI calls in unsafe module
3. `tests/test_safe_module_with_unsafe_block.nano` - Safe module using `unsafe {}`
4. `tests/test_import_safety_annotation.nano` - `import unsafe`
5. `tests/test_backward_compat.nano` - Existing code still works

---

### Documentation

**Files to Update:**
- `docs/MODULE_SYSTEM.md` - Add unsafe module section
- `MEMORY.md` - Update unsafe blocks section
- `docs/EXTERN_FFI.md` - Add module-level safety
- `examples/` - Add example unsafe modules

---

### Deliverable

**Success Criteria:**
- ✅ Parser accepts `unsafe module { ... }`
- ✅ Typechecker allows FFI in unsafe modules without `unsafe {}`
- ✅ `import unsafe` validates module safety
- ✅ All existing tests pass (backward compatible)
- ✅ New tests for module-level safety pass

---

## Phase 2: Module Introspection

**Duration:** 1-2 weeks  
**Priority:** P4 - Critical  
**Goal:** Auto-generate module metadata functions

### Changes Required

#### 2.1 Transpiler Changes

**File:** `src/transpiler_iterative_v3_twopass.c`

**Add Function:**
```c
static void generate_module_metadata(Environment *env, StringBuilder *sb) {
    for (int i = 0; i < env->module_count; i++) {
        ModuleInfo *mod = &env->modules[i];
        
        sb_appendf(sb, "\n/* Metadata for module %s */\n", mod->name);
        
        // __module_info_NAME
        sb_appendf(sb, "static inline struct ModuleInfo ___module_info_%s(void) {\n", mod->name);
        sb_appendf(sb, "    struct ModuleInfo info;\n");
        sb_appendf(sb, "    info.name = \"%s\";\n", mod->name);
        sb_appendf(sb, "    info.is_safe = %d;\n", !mod->is_unsafe);
        sb_appendf(sb, "    info.has_ffi = %d;\n", mod->has_ffi);
        sb_append(sb, "    return info;\n");
        sb_append(sb, "}\n\n");
        
        // __module_has_function_NAME
        sb_appendf(sb, "static inline int ___module_has_function_%s(const char* name) {\n", mod->name);
        
        // Iterate exported functions
        for (int j = 0; j < env->function_count; j++) {
            Function *func = &env->functions[j];
            if (func->module && strcmp(func->module, mod->name) == 0) {
                sb_appendf(sb, "    if (strcmp(name, \"%s\") == 0) return 1;\n", func->name);
            }
        }
        
        sb_append(sb, "    return 0;\n");
        sb_append(sb, "}\n\n");
        
        // __module_exported_functions_NAME
        // (Returns array of function names)
        sb_appendf(sb, "static inline const char** ___module_exported_functions_%s(int* count) {\n", mod->name);
        
        int func_count = 0;
        for (int j = 0; j < env->function_count; j++) {
            Function *func = &env->functions[j];
            if (func->module && strcmp(func->module, mod->name) == 0) {
                func_count++;
            }
        }
        
        sb_appendf(sb, "    static const char* funcs[] = {\n");
        for (int j = 0; j < env->function_count; j++) {
            Function *func = &env->functions[j];
            if (func->module && strcmp(func->module, mod->name) == 0) {
                sb_appendf(sb, "        \"%s\",\n", func->name);
            }
        }
        sb_append(sb, "    };\n");
        sb_appendf(sb, "    *count = %d;\n", func_count);
        sb_append(sb, "    return funcs;\n");
        sb_append(sb, "}\n\n");
    }
}
```

**Call in `generate_c_program()`:**
```c
void generate_c_program(Environment *env, ASTNode *program, StringBuilder *sb) {
    /* ... existing code ... */
    
    // Generate struct definitions
    generate_struct_and_union_definitions_ordered(env, sb);
    
    // Generate module metadata (NEW)
    generate_module_metadata(env, sb);
    
    /* ... rest of code ... */
}
```

---

#### 2.2 Runtime Structs

**File:** `src/runtime/module_info.h` (NEW)

```c
#ifndef NANOLANG_MODULE_INFO_H
#define NANOLANG_MODULE_INFO_H

typedef struct {
    const char *name;
    int is_safe;
    int has_ffi;
    int exported_function_count;
} ModuleInfo;

typedef struct {
    const char *name;
    int is_extern;
    int param_count;
} FunctionInfo;

#endif
```

**File:** `src/runtime/module_info.c` (NEW)

```c
#include "module_info.h"
#include <string.h>

// Helper functions for module introspection
```

---

#### 2.3 NanoLang Self-Hosted Usage

**File:** `src_nano/typecheck.nano`

**Add Module Introspection:**
```nano
/* Auto-generated by C transpiler */
extern fn ___module_info_Parser() -> ModuleInfo
extern fn ___module_has_function_Parser(name: string) -> bool

struct ModuleInfo {
    name: string,
    is_safe: bool,
    has_ffi: bool,
    exported_function_count: int
}

fn check_module_safety(module_name: string) -> bool {
    /* Use reflection to check if module is safe */
    if (== module_name "Parser") {
        let info: ModuleInfo = (___module_info_Parser)
        return info.is_safe
    } else {
        return true  /* Unknown modules assumed safe */
    }
}
```

---

### Testing

**Test Cases:**
1. `tests/test_module_metadata_simple.nano` - Query module info
2. `tests/test_module_has_function.nano` - Check function exists
3. `tests/test_module_list_functions.nano` - List all functions
4. `tests/test_module_safety_query.nano` - Runtime safety check

---

### Deliverable

**Success Criteria:**
- ✅ Transpiler generates `___module_info_*` functions
- ✅ `ModuleInfo` struct available in runtime
- ✅ NanoLang code can query module metadata
- ✅ Self-hosted compiler uses module introspection
- ✅ Tests pass

---

## Phase 3: Warning System

**Duration:** 1 week  
**Priority:** P3 - High  
**Goal:** Compiler flags for safety warnings

### Changes Required

#### 3.1 CLI Flag Parsing

**File:** `src/main.c`

**Add Flags:**
```c
typedef struct {
    /* Existing flags... */
    bool warn_unsafe_imports;
    bool warn_unsafe_calls;
    bool warn_ffi;
    bool forbid_unsafe;
} CompilerOptions;

static void parse_args(int argc, char **argv, CompilerOptions *opts) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--warn-unsafe-imports") == 0) {
            opts->warn_unsafe_imports = true;
        } else if (strcmp(argv[i], "--warn-unsafe-calls") == 0) {
            opts->warn_unsafe_calls = true;
        } else if (strcmp(argv[i], "--warn-ffi") == 0) {
            opts->warn_ffi = true;
        } else if (strcmp(argv[i], "--forbid-unsafe") == 0) {
            opts->forbid_unsafe = true;
        }
        /* ... other flags ... */
    }
}
```

---

#### 3.2 Warning Emission

**File:** `src/typechecker.c`

**Import Warnings:**
```c
static void check_import(ImportStmt *import, Environment *env, CompilerOptions *opts) {
    ModuleInfo *mod = env_get_module(env, import->module_path);
    
    if (!mod) return;
    
    if (opts->forbid_unsafe && mod->is_unsafe) {
        error("Unsafe module '%s' forbidden by --forbid-unsafe", mod->name);
        return;
    }
    
    if (opts->warn_unsafe_imports && mod->is_unsafe) {
        warning("Importing unsafe module '%s' at line %d", mod->name, import->line);
    }
}
```

**Call Warnings:**
```c
static void check_function_call(ASTNode *call, Environment *env, CompilerOptions *opts) {
    Function *func = env_get_function(env, call->as.call.function_name);
    
    if (!func) return;
    
    bool is_ffi = func->is_extern;
    bool is_unsafe_module = (func->module && is_module_unsafe(env, func->module));
    
    if (opts->forbid_unsafe && (is_ffi || is_unsafe_module)) {
        error("Call to unsafe function '%s' forbidden by --forbid-unsafe", func->name);
        return;
    }
    
    if (opts->warn_ffi && is_ffi) {
        warning("FFI call to '%s' at line %d", func->name, call->line);
    }
    
    if (opts->warn_unsafe_calls && is_unsafe_module) {
        warning("Call to function '%s' from unsafe module '%s' at line %d",
                func->name, func->module, call->line);
    }
}
```

---

#### 3.3 Warning Format

**Output:**
```
Warning: Importing unsafe module 'sdl' at line 3
  → modules/sdl/sdl.nano
  This module contains FFI calls to external C library

Warning: FFI call to 'SDL_Init' at line 10
  → game.nano:10:5
  Extern function call may have side effects
```

---

### Testing

**Test Cases:**
```bash
# Test each warning level
make test-warnings

# Specific tests
tests/test_warn_unsafe_imports.sh
tests/test_warn_ffi.sh
tests/test_forbid_unsafe.sh
```

---

### Deliverable

**Success Criteria:**
- ✅ `--warn-unsafe-imports` warns on unsafe imports
- ✅ `--warn-ffi` warns on FFI calls
- ✅ `--forbid-unsafe` errors on unsafe code
- ✅ Warning messages are clear and actionable
- ✅ Tests cover all warning modes

---

## Phase 4: Module-Qualified Calls

**Duration:** 1 week  
**Priority:** P2 - Medium  
**Goal:** Fix `Module.function()` parsed as field access

### Changes Required

#### 4.1 Parser Changes

**File:** `src/parser_iterative.c`

**New AST Node:**
```c
typedef struct {
    char *module_name;  // "Vec"
    char *function_name;  // "add"
    ASTNode **args;
    int arg_count;
    int line;
    int column;
} ModuleQualifiedCall;
```

**Parse Logic:**
```c
static ASTNode *parse_primary(Parser *p) {
    if (check_type(p, TOKEN_IDENTIFIER)) {
        Token *name = advance(p);
        
        // Check for module-qualified call: Module.function
        if (check_type(p, TOKEN_DOT)) {
            advance(p);  // consume dot
            
            Token *func_name = expect(p, TOKEN_IDENTIFIER, "Expected function name");
            
            // This is Module.function, not field access
            if (check_type(p, TOKEN_LPAREN)) {
                // Parse as module-qualified call
                return parse_module_call(p, name->value, func_name->value);
            }
        }
        
        // Regular identifier or field access
        /* ... existing logic ... */
    }
}
```

---

#### 4.2 Typechecker Changes

**File:** `src/typechecker.c`

**Resolve Module Functions:**
```c
static void check_module_qualified_call(ASTNode *call, Environment *env) {
    ModuleQualifiedCall *mqc = call->as.module_qualified_call;
    
    // Find module
    ModuleInfo *mod = env_get_module(env, mqc->module_name);
    if (!mod) {
        error("Unknown module '%s' at line %d", mqc->module_name, call->line);
        return;
    }
    
    // Find function in module
    Function *func = env_get_function_in_module(env, mqc->function_name, mod->name);
    if (!func) {
        error("Module '%s' has no function '%s' at line %d",
              mqc->module_name, mqc->function_name, call->line);
        return;
    }
    
    // Typecheck arguments
    check_function_args(call, func, env);
}
```

---

#### 4.3 Transpiler Changes

**File:** `src/transpiler_iterative_v3_twopass.c`

**Generate Call:**
```c
static void transpile_module_qualified_call(ASTNode *call, StringBuilder *sb, Environment *env) {
    ModuleQualifiedCall *mqc = call->as.module_qualified_call;
    
    // Emit: nl_ModuleName_function_name(args)
    sb_appendf(sb, "nl_%s_%s(", mqc->module_name, mqc->function_name);
    
    for (int i = 0; i < mqc->arg_count; i++) {
        if (i > 0) sb_append(sb, ", ");
        transpile_expr(mqc->args[i], sb, env);
    }
    
    sb_append(sb, ")");
}
```

---

### Testing

**Test Cases:**
1. `tests/test_module_qualified_call.nano` - `Vec.add(v1, v2)`
2. `tests/test_nested_module_call.nano` - Chained calls
3. `tests/test_field_vs_module_call.nano` - Disambiguate syntax

---

### Deliverable

**Success Criteria:**
- ✅ `Module.function()` parsed as module call, not field access
- ✅ Typechecker resolves function in module namespace
- ✅ Generated C code calls correct function
- ✅ Existing field access still works
- ✅ Tests pass

**Related Issue:** `nanolang-3oda` (already created)

---

## Phase 5: Module as First-Class Value (Future)

**Duration:** 2-3 weeks  
**Priority:** P1 - Low (Future Enhancement)  
**Goal:** Pass modules as function arguments

### Design Overview

**This phase is OPTIONAL and can be deferred.**

**Concept:**
```nano
type Module = opaque  /* Pointer to ModuleInfo */

fn apply_to_module(m: Module, operation: string) -> void {
    let info: ModuleInfo = (module_info m)
    (println (+ "Operating on module: " info.name))
}

fn main() -> int {
    let vec_module: Module = (get_module "vector2d")
    (apply_to_module vec_module "test")
    return 0
}
```

**Challenges:**
- Requires runtime module registry
- Adds memory management complexity
- May not fit NanoLang's compile-time philosophy

**Decision:** Defer until we see real use cases requiring this feature.

---

## Timeline Summary

| Phase | Duration | Priority | Dependencies |
|-------|----------|----------|--------------|
| Phase 1: Safety Annotations | 1-2 weeks | P4 Critical | None |
| Phase 2: Introspection | 1-2 weeks | P4 Critical | Phase 1 |
| Phase 3: Warning System | 1 week | P3 High | Phase 1, 2 |
| Phase 4: Qualified Calls | 1 week | P2 Medium | None (independent) |
| Phase 5: Module Values | 2-3 weeks | P1 Low | Phase 1, 2 (deferred) |

**Total (Phases 1-4):** 4-6 weeks  
**Total (All Phases):** 6-9 weeks

---

## Success Metrics

### Phase 1 Success

- 50%+ reduction in `unsafe {}` block count in examples/
- All SDL examples use `unsafe module` instead of scattered blocks
- Zero regressions in existing tests

### Phase 2 Success

- Self-hosted compiler uses module introspection for typecheck
- 10+ examples demonstrating module metadata queries
- Module introspection docs published

### Phase 3 Success

- 4+ compiler warning modes working
- Users can choose safety level appropriate to their project
- Warning messages are clear and actionable

### Phase 4 Success

- `Module.function()` syntax works in all examples
- Typechecker correctly resolves module-qualified calls
- No confusion between field access and module calls

---

## Migration Guide

### For Users

**Step 1: Assess Current Code**
```bash
# Count unsafe blocks
$ grep -r "unsafe {" examples/ | wc -l
45  # Before migration

$ grep -r "unsafe module" examples/ | wc -l
0  # None yet
```

**Step 2: Migrate One Module**
```bash
# Convert SDL module
$ nalang migrate --add-unsafe-module modules/sdl/sdl.nano
```

**Step 3: Update Imports**
```nano
/* Before */
import "modules/sdl/sdl.nano"

/* After */
import unsafe "modules/sdl/sdl.nano"
```

**Step 4: Remove Scattered Unsafe Blocks**
```nano
/* Before */
fn render() -> void {
    unsafe { (SDL_Init) }
    unsafe { (SDL_Present) }
    unsafe { (SDL_Quit) }
}

/* After */
fn render() -> void {
    (SDL_Init)
    (SDL_Present)
    (SDL_Quit)
}
```

**Step 5: Test**
```bash
$ make test
$ nanoc examples/sdl_window.nano -o bin/sdl_window
$ ./bin/sdl_window  # Verify it works
```

---

### For Module Authors

**Creating Safe Module:**
```nano
safe module vector2d {
    export struct Vec2 { x: float, y: float }
    export fn add(v1: Vec2, v2: Vec2) -> Vec2 { /* ... */ }
}
```

**Creating Unsafe Module:**
```nano
unsafe module sdl {
    extern fn SDL_Init(flags: int) -> int
    
    export fn init() -> bool {
        /* No unsafe block needed */
        let result: int = (SDL_Init SDL_INIT_VIDEO)
        return (== result 0)
    }
}
```

**Adding Metadata:**
```json
{
  "name": "sdl",
  "safety": {
    "level": "unsafe",
    "has_ffi": true,
    "audit_date": "2025-01-01"
  }
}
```

---

## Risk Mitigation

### Risk 1: Breaking Existing Code

**Mitigation:**
- Phase 1a: Parser only (no enforcement)
- Phase 1b: Backward compatibility mode
- Phase 1c: Deprecation warnings
- Phase 1d: Remove old syntax (far future)

**Timeline:** 2 months minimum before any breaking changes

---

### Risk 2: Self-Hosted Compiler Complexity

**Mitigation:**
- Implement in C compiler first
- Test thoroughly before porting to NanoLang
- Use metadata system to reduce manual tracking

**Timeline:** Self-hosted port happens AFTER C implementation stable

---

### Risk 3: Performance Impact

**Mitigation:**
- Module metadata is `static inline` - zero runtime cost
- No dynamic module loading (all compile-time)
- Reflection functions optimized out if unused

**Testing:** Benchmark before/after, ensure <1% overhead

---

## Decision Points

### Decision 1: Module Syntax

**Options:**
- A) `unsafe module name { ... }` (explicit wrapper)
- B) `@unsafe` attribute at top of file
- C) Infer from `extern` presence

**Recommendation:** A (clearest scope)

---

### Decision 2: Backward Compatibility

**Options:**
- A) Keep `unsafe {}` forever (parallel systems)
- B) Deprecate `unsafe {}` in unsafe modules
- C) Remove `unsafe {}` entirely

**Recommendation:** B (deprecate but allow transition)

---

### Decision 3: Module as Value

**Options:**
- A) Implement now (6-9 weeks total)
- B) Defer to later (4-6 weeks for Phases 1-4)
- C) Never implement (keep compile-time only)

**Recommendation:** B (defer until use cases emerge)

---

## Next Steps

1. **Review this roadmap** - Get approval from project owner
2. **Create GitHub issues** - One per phase
3. **Assign priorities** - P4 for Phase 1-2, P3 for Phase 3, P2 for Phase 4
4. **Begin Phase 1** - Start with parser changes
5. **Iterate** - Ship each phase independently

---

**Status:** 🔴 **Awaiting Approval**  
**Owner:** TBD  
**Estimated Start:** 2025-01-09  
**Estimated Completion:** 2025-02-20 (Phases 1-4)
