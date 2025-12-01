# Import Alias Implementation Design

## Goal

Enable modular self-hosting by allowing import aliases:

```nano
import "parser_mvp.nano" as Parser
import "typechecker_minimal.nano" as TC
import "transpiler_minimal.nano" as Trans

fn compile(source: string) -> int {
    let tokens = (tokenize source)
    let parser = (Parser.parse_program tokens)  // Parser.function_name
    let result = (TC.typecheck_parser parser)
    ...
}
```

## Syntax Design

### Current (no aliases)
```nano
import "module.nano"
```

### Proposed (with aliases)
```nano
import "module.nano" as ModuleName
```

### Grammar Extension
```
import_stmt = "import" string_literal ["as" identifier]
```

## Implementation Steps

### 1. Lexer (src/lexer.c)

Add `as` keyword:
```c
// In keyword detection (around line 45)
if (strcmp(str, "as") == 0) return TOKEN_AS;
```

Add token type:
```c
// In nanolang.h TokenType enum
TOKEN_AS = 58,  // "as" keyword
```

### 2. Parser (src/parser.c)

Update `ASTImport` structure:
```c
typedef struct {
    char *module_path;     // "parser_mvp.nano"
    char *module_name;     // "parser_mvp" (derived)
    char *alias;           // "Parser" (if "as Parser" specified)
    int line;
    int column;
} ASTImport;
```

Update `parse_import()`:
```c
static ASTNode *parse_import(Parser *p) {
    // ... existing code ...
    
    // After parsing module_path:
    char *alias = NULL;
    if (match(p, TOKEN_AS)) {
        advance(p);  // consume 'as'
        if (!expect(p, TOKEN_IDENTIFIER, "Expected identifier after 'as'")) {
            return NULL;
        }
        alias = strdup(current_token(p)->value);
        advance(p);
    }
    
    // Create import node with alias
    ASTImport *import = create_import(module_path, module_name, alias, line, column);
    ...
}
```

### 3. Module System (src/module.c)

Track module aliases in environment:
```c
typedef struct {
    char *file_path;
    char *module_name;
    char *alias;  // NEW: Optional alias
    ...
} ImportedModule;
```

When processing imports:
```c
void process_import(ASTImport *import, ...) {
    // Load module
    ImportedModule *mod = load_module(import->module_path);
    
    // Register with alias if provided
    if (import->alias) {
        mod->alias = strdup(import->alias);
        register_module_alias(env, import->alias, mod);
    } else {
        register_module_name(env, mod->module_name, mod);
    }
}
```

### 4. Name Resolution (src/typechecker.c, src/eval.c)

Resolve qualified names:
```c
// When encountering "Parser.parse_program":
// 1. Split on '.'
// 2. Look up "Parser" in module aliases
// 3. Find "parse_program" in that module's exports
```

Update function call resolution:
```c
Type *resolve_function_call(char *name) {
    // Check for qualified name (Module.function)
    char *dot = strchr(name, '.');
    if (dot) {
        char *module_part = strndup(name, dot - name);
        char *function_part = dot + 1;
        
        // Find module by alias
        ImportedModule *mod = find_module_by_alias(env, module_part);
        if (mod) {
            return find_function_in_module(mod, function_part);
        }
    }
    
    // Fall back to regular resolution
    return find_function(env, name);
}
```

### 5. Struct/Type Resolution

Same principle for types:
```c
Type *resolve_type(char *type_name) {
    char *dot = strchr(type_name, '.');
    if (dot) {
        char *module_part = strndup(type_name, dot - type_name);
        char *type_part = dot + 1;
        
        ImportedModule *mod = find_module_by_alias(env, module_part);
        if (mod) {
            return find_type_in_module(mod, type_part);
        }
    }
    
    return find_type(env, type_name);
}
```

## Usage Example

After implementation:

**parser_helper.nano:**
```nano
import "parser_mvp.nano" as P
import "ast_shared.nano" as AST

fn create_parser(source: string) -> P.Parser {
    let tokens: List<AST.Token> = (tokenize source)
    return (P.parser_new tokens (List_Token_length tokens))
}
```

**main_compiler.nano:**
```nano
import "lexer_main.nano" as Lexer
import "parser_mvp.nano" as Parser
import "typechecker_minimal.nano" as TC
import "transpiler_minimal.nano" as Trans

fn compile(input: string) -> int {
    let source: string = (read_file input)
    
    // Use aliased modules
    let tokens = (Lexer.tokenize source)
    let parser = (Parser.parse_program tokens (List_Token_length tokens))
    let result = (TC.typecheck_parser parser)
    let c_code = (Trans.transpile_parser parser)
    
    // ... rest of compilation
    return 0
}
```

## Benefits

1. **No struct conflicts** - Each module has its own namespace
2. **Clear provenance** - Easy to see where functions come from
3. **Modular** - Components stay separate
4. **Maintainable** - Changes isolated to modules
5. **Self-hosting ready** - Can import NanoLang compiler components

## Testing Plan

### Phase 1: Basic alias
```nano
import "test_module.nano" as T
(T.function_name args)
```

### Phase 2: Type resolution
```nano
import "types.nano" as Types
let x: Types.MyStruct = Types.MyStruct { field: 42 }
```

### Phase 3: Multiple imports
```nano
import "a.nano" as A
import "b.nano" as B
// A.Parser vs B.Parser - no conflict!
```

### Phase 4: Self-hosted compiler
```nano
import "parser_mvp.nano" as Parser
import "typechecker_minimal.nano" as TC
import "transpiler_minimal.nano" as Trans
// Full modular self-hosting!
```

## Estimated Effort

- Lexer: 30 minutes (add `as` keyword)
- Parser: 1-2 hours (update ASTImport, parse alias)
- Module system: 2-3 hours (track aliases, resolution)
- Type checker: 2-3 hours (qualified name resolution)
- Transpiler: 1 hour (qualified name handling)
- Testing: 1-2 hours (test cases)

**Total: ~8-12 hours of focused work**

## Success Criteria

✅ `import "X.nano" as Y` parses correctly
✅ `Y.function()` resolves to X's function  
✅ `Y.Type` resolves to X's type
✅ No namespace conflicts between modules
✅ Self-hosted compiler components can import each other
✅ TRUE SELF-HOSTING ACHIEVED!
