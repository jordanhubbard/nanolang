# Import Aliases Design

## Goal
Enable `import "module.nano" as Alias` to namespace all symbols from the imported module.

## Syntax
```nano
import "math.nano" as Math
import "src_nano/lexer_main.nano" as Lexer

let x: int = (Math.sqrt 16)
let tokens = (Lexer.tokenize source)
```

## Implementation Plan

### 1. Environment Changes (src/env.c, src/nanolang.h)

Add module namespace tracking to Environment:

```c
typedef struct {
    char *alias;              // Module alias name (e.g., "Math", "Lexer")
    char **function_names;    // Functions from this module
    int function_count;
    char **struct_names;      // Structs from this module  
    int struct_count;
    char **enum_names;        // Enums from this module
    int enum_count;
    char **union_names;       // Unions from this module
    int union_count;
} ModuleNamespace;

typedef struct {
    Symbol *symbols;
    // ... existing fields ...
    
    ModuleNamespace *namespaces;  // Module alias → symbols mapping
    int namespace_count;
    int namespace_capacity;
} Environment;
```

### 2. Symbol Lookup Changes (src/env.c)

Modify lookup functions to handle `Module.symbol` patterns:

```c
Function *env_get_function(Environment *env, const char *name) {
    // Check for dot notation: "Module.function"
    const char *dot = strchr(name, '.');
    if (dot) {
        char module[256];
        strncpy(module, name, dot - name);
        module[dot - name] = '\0';
        const char *func_name = dot + 1;
        
        // Find namespace
        for (int i = 0; i < env->namespace_count; i++) {
            if (strcmp(env->namespaces[i].alias, module) == 0) {
                // Check if function is in this namespace
                for (int j = 0; j < env->namespaces[i].function_count; j++) {
                    if (strcmp(env->namespaces[i].function_names[j], func_name) == 0) {
                        // Look up the actual function by original name
                        return lookup_function_direct(env, func_name);
                    }
                }
                return NULL;  // Function not in this module
            }
        }
        return NULL;  // Module not found
    }
    
    // Fall back to regular lookup
    return lookup_function_direct(env, name);
}
```

Similar for `env_get_struct`, `env_get_enum`, `env_get_union`.

### 3. Import Processing Changes (src/module.c)

Update `process_imports` to register namespaces:

```c
bool process_imports(ASTNode *program, Environment *env, ...) {
    for (each import in program) {
        char *module_path = import->as.import_stmt.module_path;
        char *alias = import->as.import_stmt.module_name;
        
        if (alias) {
            // Load module into temporary environment
            Environment *module_env = create_environment();
            load_module_into_env(module_path, module_env);
            
            // Register namespace with all module symbols
            ModuleNamespace ns;
            ns.alias = strdup(alias);
            ns.function_names = extract_function_names(module_env);
            ns.function_count = module_env->function_count;
            ns.struct_names = extract_struct_names(module_env);
            ns.struct_count = module_env->struct_count;
            // ... enums, unions ...
            
            add_namespace(env, ns);
            
            // Merge module symbols into main env (for actual lookup)
            merge_environment(env, module_env);
        } else {
            // No alias - merge directly (current behavior)
            load_and_merge_module(module_path, env);
        }
    }
}
```

### 4. Parser Changes (src/parser.c)

The parser already handles `import "module" as alias` ✅

Need to handle `Module.identifier` in expressions:

```c
// In parse_primary or parse_postfix:
if (match(p, TOKEN_IDENTIFIER)) {
    char *name = current_token(p)->value;
    advance(p);
    
    // Check for dot (Module.symbol)
    if (match(p, TOKEN_DOT)) {
        advance(p);
        if (!match(p, TOKEN_IDENTIFIER)) {
            error("Expected identifier after '.'");
        }
        char *member = current_token(p)->value;
        advance(p);
        
        // Create qualified name "Module.member"
        char qualified[512];
        snprintf(qualified, sizeof(qualified), "%s.%s", name, member);
        
        // Create identifier node with qualified name
        return create_identifier_node(qualified);
    }
    
    return create_identifier_node(name);
}
```

### 5. Type Checker Changes (src/typechecker.c)

Update type resolution to handle `Module.Type`:

```c
Type resolve_type_name(const char *type_name, Environment *env) {
    // Check for Module.Type pattern
    const char *dot = strchr(type_name, '.');
    if (dot) {
        char module[256];
        strncpy(module, type_name, dot - type_name);
        const char *type = dot + 1;
        
        // Verify module namespace exists
        ModuleNamespace *ns = find_namespace(env, module);
        if (!ns) {
            error("Unknown module '%s'", module);
        }
        
        // Look up type in namespace
        if (is_in_namespace(ns, type, NAMESPACE_STRUCT)) {
            return TYPE_STRUCT with struct_name = type;
        }
        // ... check enums, unions ...
    }
    
    // Fall back to regular type resolution
    return resolve_type_regular(type_name, env);
}
```

## Testing Plan

### Test 1: Simple Alias
```nano
// math.nano
fn square(x: int) -> int {
    return (* x x)
}

// main.nano
import "math.nano" as Math

fn main() -> int {
    return (Math.square 5)
}
```

### Test 2: Multiple Aliases
```nano
import "lexer.nano" as Lexer
import "parser.nano" as Parser

fn compile(source: string) -> Parser.AST {
    let tokens = (Lexer.tokenize source)
    return (Parser.parse tokens)
}
```

### Test 3: Aliased Types
```nano
import "types.nano" as Types

fn process(point: Types.Point) -> int {
    return point.x
}
```

## Implementation Estimate

- **Environment changes**: 2-3 hours
- **Symbol lookup**: 2-3 hours  
- **Import processing**: 2-3 hours
- **Parser dot notation**: 1-2 hours
- **Type checker**: 2-3 hours
- **Testing & debugging**: 2-4 hours

**Total**: 11-18 hours (1.5-2 days)

## Benefits

✅ Clean module integration without name conflicts  
✅ Keep test `main()` functions in each module  
✅ Self-documenting code (clear symbol provenance)  
✅ No need to manually deduplicate structs  
✅ Enables true self-hosting for nanolang compiler
