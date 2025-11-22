# Opaque Pointer Types - Proper Architecture

## Problem Statement

The current implementation violates separation of concerns by hardcoding library-specific knowledge in the transpiler:

```c
// src/transpiler.c:388 - BAD: Transpiler knows about SDL, GLFW, etc.
static const char *get_sdl_c_type(const char *func_name, int param_index, bool is_return) {
    if (strstr(func_name, "CreateWindow")) return "SDL_Window*";
    if (strstr(func_name, "CreateRenderer")) return "SDL_Renderer*";
    // ... hundreds of lines of library-specific code
}
```

**This doesn't scale.** Every new C library requires modifying the transpiler.

## Root Cause

The current system has:
1. **Extern functions** that declare C function signatures
2. **No way to declare opaque C types** 
3. **Transpiler that guesses types** based on function names

**What's missing:** A way for modules to declare "this identifier represents an opaque C pointer type."

---

## Proper Solution: Opaque Type Declarations

### Design Principle

**Modules declare opaque types. The transpiler handles them generically.**

The transpiler should NEVER have library-specific knowledge. All type information comes from module declarations.

### Proposed Syntax

```nano
# Declare that GLFWwindow is an opaque pointer type from C
opaque type GLFWwindow

# Now use it in extern function signatures
extern fn glfwCreateWindow(width: int, height: int, title: string, 
                           monitor: int, share: int) -> GLFWwindow

extern fn glfwMakeContextCurrent(window: GLFWwindow) -> void
extern fn glfwWindowShouldClose(window: GLFWwindow) -> int
extern fn glfwDestroyWindow(window: GLFWwindow) -> void
```

### How It Works

1. **Module declares opaque type:**
   ```nano
   opaque type GLFWwindow
   ```

2. **Parser adds to type table:**
   ```c
   typedef struct {
       char *name;              // "GLFWwindow"
       char *c_type_name;       // "GLFWwindow*" (pointer in C)
       bool is_opaque;          // true
   } OpaqueTypeDef;
   ```

3. **Type checker validates usage:**
   - Opaque types can be passed to/from extern functions
   - Opaque types cannot be dereferenced or modified in nanolang
   - Attempting to access fields generates a compile error

4. **Transpiler generates casts automatically:**
   
   **Nanolang:**
   ```nano
   let window: GLFWwindow = (glfwCreateWindow 800 600 "Title" 0 0)
   (glfwMakeContextCurrent window)
   ```
   
   **Generated C:**
   ```c
   int64_t window = (int64_t)glfwCreateWindow(800, 600, "Title", 0, 0);
   glfwMakeContextCurrent((GLFWwindow*)window);
   ```

5. **Generic casting logic:**
   ```c
   // Transpiler code - works for ANY opaque type
   OpaqueTypeDef *opaque = env_get_opaque_type(env, param_type_name);
   if (opaque) {
       sb_appendf(sb, "(%s)", opaque->c_type_name);  // Cast to C pointer type
   }
   ```

---

## Complete Example: GLFW Module

### modules/glfw/glfw.nano

```nano
# GLFW - OpenGL Window Library
#
# Installation:
#   macOS:  brew install glfw
#   Linux:  sudo apt install libglfw3-dev

# ============================================================================
# Opaque Type Declarations
# ============================================================================

opaque type GLFWwindow   # Represents GLFWwindow* in C
opaque type GLFWmonitor  # Represents GLFWmonitor* in C

# ============================================================================
# Initialization
# ============================================================================

extern fn glfwInit() -> int
extern fn glfwTerminate() -> void

# ============================================================================
# Window Management
# ============================================================================

extern fn glfwCreateWindow(width: int, height: int, title: string, 
                           monitor: GLFWmonitor, share: GLFWwindow) -> GLFWwindow

extern fn glfwDestroyWindow(window: GLFWwindow) -> void

extern fn glfwWindowShouldClose(window: GLFWwindow) -> int

extern fn glfwSetWindowShouldClose(window: GLFWwindow, value: int) -> void

# ============================================================================
# Context Management
# ============================================================================

extern fn glfwMakeContextCurrent(window: GLFWwindow) -> void

extern fn glfwSwapBuffers(window: GLFWwindow) -> void

extern fn glfwPollEvents() -> void

# ============================================================================
# Input
# ============================================================================

extern fn glfwGetKey(window: GLFWwindow, key: int) -> int

extern fn glfwGetMouseButton(window: GLFWwindow, button: int) -> int

# ============================================================================
# Constants
# ============================================================================

let GLFW_KEY_ESCAPE: int = 256
let GLFW_KEY_SPACE: int = 32
let GLFW_PRESS: int = 1
let GLFW_RELEASE: int = 0
```

### Usage

```nano
import "modules/glfw/glfw.nano"

fn main() -> int {
    if (== (glfwInit) 0) {
        (println "Failed to initialize GLFW")
        return 1
    } else {}
    
    # Type-safe: window has type GLFWwindow
    let window: GLFWwindow = (glfwCreateWindow 800 600 "My Window" 0 0)
    
    if (== window 0) {
        (println "Failed to create window")
        (glfwTerminate)
        return 1
    } else {}
    
    (glfwMakeContextCurrent window)  # Type-checked: expects GLFWwindow
    
    while (== (glfwWindowShouldClose window) 0) {
        # Render...
        (glfwSwapBuffers window)
        (glfwPollEvents)
    }
    
    (glfwDestroyWindow window)
    (glfwTerminate)
    return 0
}
```

---

## Type Safety Benefits

### Prevents Type Confusion

**With opaque types:**
```nano
opaque type GLFWwindow
opaque type SDL_Window

let glfw_win: GLFWwindow = (glfwCreateWindow ...)
let sdl_win: SDL_Window = (SDL_CreateWindow ...)

# ERROR: Type mismatch
(glfwMakeContextCurrent sdl_win)  # Compiler error!
```

**Without opaque types (current system):**
```nano
let glfw_win: int = (glfwCreateWindow ...)
let sdl_win: int = (SDL_CreateWindow ...)

# SILENTLY WRONG: Both are int, no type checking
(glfwMakeContextCurrent sdl_win)  # Compiles but crashes at runtime
```

### Prevents Invalid Operations

```nano
opaque type GLFWwindow

let window: GLFWwindow = (glfwCreateWindow ...)

# ERROR: Cannot dereference opaque type
let value: int = window.field  # Compile error

# ERROR: Cannot modify opaque pointer
set window (+ window 1)  # Compile error

# ERROR: Cannot perform pointer arithmetic
let next: GLFWwindow = (+ window 8)  # Compile error

# OK: Can pass to functions expecting GLFWwindow
(glfwMakeContextCurrent window)  # Type-safe
```

---

## Implementation Plan

### Phase 1: Add Opaque Type Declarations

**1. Parser (`src/parser.c`):**
```c
// Parse: opaque type TypeName
ASTNode *parse_opaque_type_declaration(Parser *p) {
    expect(p, TOKEN_IDENTIFIER, "opaque");
    expect(p, TOKEN_IDENTIFIER, "type");
    char *type_name = expect(p, TOKEN_IDENTIFIER, NULL);
    
    ASTNode *node = create_ast_node(AST_OPAQUE_TYPE);
    node->as.opaque_type.name = type_name;
    return node;
}
```

**2. Environment (`src/env.c`):**
```c
typedef struct {
    char *name;              // "GLFWwindow"
    char *c_type_name;       // "GLFWwindow*"
} OpaqueTypeDef;

// Add to Environment struct:
struct Environment {
    // ... existing fields ...
    OpaqueTypeDef *opaque_types;
    int opaque_type_count;
    int opaque_type_capacity;
};

void env_add_opaque_type(Environment *env, const char *name) {
    // Add opaque type to registry
    // c_type_name = name + "*" (add pointer)
}

OpaqueTypeDef *env_get_opaque_type(Environment *env, const char *name) {
    // Look up opaque type
}
```

**3. Type System (`src/nanolang.h`):**
```c
typedef enum {
    TYPE_INT,
    TYPE_FLOAT,
    TYPE_STRING,
    TYPE_BOOL,
    TYPE_VOID,
    // ... existing types ...
    TYPE_OPAQUE,        // NEW: Opaque C pointer type
} Type;

typedef struct {
    Type base_type;
    char *opaque_type_name;  // For TYPE_OPAQUE: "GLFWwindow"
    // ... other fields ...
} TypeInfo;
```

### Phase 2: Type Checking

**Type checker (`src/typechecker.c`):**
```c
// Opaque types can ONLY be:
// 1. Declared
// 2. Assigned from extern function returns
// 3. Passed to extern function parameters
// 4. Compared to 0 (null check)

void check_opaque_type_usage(ASTNode *expr, Environment *env) {
    if (expr->type_info && expr->type_info->base_type == TYPE_OPAQUE) {
        // Verify this is a valid usage
        switch (expr->node_type) {
            case AST_CALL:  // OK: calling function
            case AST_VARIABLE:  // OK: reading variable
            case AST_BINOP:  // Check: only == and != with 0
                if (!is_null_comparison(expr)) {
                    error("Cannot perform operations on opaque type");
                }
                break;
            case AST_FIELD_ACCESS:  // ERROR: cannot access fields
                error("Cannot access fields of opaque type '%s'", 
                      expr->type_info->opaque_type_name);
                break;
            default:
                error("Invalid operation on opaque type");
        }
    }
}
```

### Phase 3: Transpiler (Generic Casting)

**Remove library-specific code:**
```c
// DELETE THIS ENTIRE FUNCTION:
static const char *get_sdl_c_type(...) {
    // 400+ lines of hardcoded SDL knowledge
}
```

**Replace with generic casting:**
```c
// Generic opaque type casting
static void transpile_opaque_cast(StringBuilder *sb, 
                                  TypeInfo *type_info, 
                                  Environment *env) {
    if (type_info && type_info->base_type == TYPE_OPAQUE) {
        OpaqueTypeDef *opaque = env_get_opaque_type(env, 
                                                     type_info->opaque_type_name);
        if (opaque) {
            sb_appendf(sb, "(%s)", opaque->c_type_name);
        }
    }
}

// Use in function calls:
void transpile_call(ASTNode *call, StringBuilder *sb, Environment *env) {
    Function *func = env_get_function(env, call->as.call.name);
    
    // For each argument:
    for (int i = 0; i < call->as.call.arg_count; i++) {
        if (func->params[i].type_info->base_type == TYPE_OPAQUE) {
            // Cast int64_t to C pointer type
            transpile_opaque_cast(sb, func->params[i].type_info, env);
        }
        transpile_expression(call->as.call.args[i], sb, env);
    }
}
```

### Phase 4: Remove Hardcoded Library Knowledge

**Files to clean up:**
1. `src/transpiler.c` - Remove `get_sdl_c_type()` and all SDL-specific logic
2. `src/transpiler.c` - Remove hardcoded `is_sdl_func`, `is_mix_func` checks
3. Update all module files to use `opaque type` declarations

---

## Migration Path

### Step 1: Add Opaque Type Support (Week 1)

- [ ] Implement parser for `opaque type` declarations
- [ ] Add `OpaqueTypeDef` to environment
- [ ] Add `TYPE_OPAQUE` to type system
- [ ] Update type checker with opaque type validation

### Step 2: Update Transpiler (Week 1-2)

- [ ] Implement generic opaque type casting
- [ ] Remove `get_sdl_c_type()` function
- [ ] Remove all hardcoded library checks
- [ ] Test with existing SDL examples

### Step 3: Update Modules (Week 2)

- [ ] Update `modules/sdl/sdl.nano` with opaque type declarations
- [ ] Update `modules/glfw/glfw.nano` with opaque type declarations
- [ ] Update `modules/glew/glew.nano` with opaque type declarations
- [ ] Update all other modules

### Step 4: Fix OpenGL Examples (Week 2)

- [ ] Test `opengl_cube.nano` - should compile cleanly
- [ ] Test `opengl_teapot/teapot.nano` - should compile cleanly
- [ ] Remove manual constant definitions if they conflict with headers

---

## Benefits

### Scalability
- ✅ Add new C libraries WITHOUT modifying transpiler
- ✅ Modules are self-contained with type declarations
- ✅ No library-specific code in compiler

### Type Safety
- ✅ Prevent mixing incompatible pointer types
- ✅ Prevent invalid operations on pointers
- ✅ Catch errors at compile time

### Maintainability  
- ✅ Separation of concerns: transpiler doesn't know about libraries
- ✅ Simpler transpiler code
- ✅ Easier to add new modules

### Philosophy Alignment
- ✅ No pointers exposed to nanolang programmers
- ✅ Opaque types are read-only from nanolang
- ✅ Cannot dereference or modify pointers
- ✅ Type-safe FFI without exposing pointer mechanics

---

## Comparison: Before vs After

### Before (Current System)

**Transpiler has hardcoded knowledge:**
```c
// src/transpiler.c - BAD
if (strcmp(func_name, "glfwMakeContextCurrent") == 0 && i == 0) {
    return "GLFWwindow*";
}
if (strcmp(func_name, "SDL_CreateWindow") == 0) {
    return "SDL_Window*";
}
// ... hundreds more lines for every library
```

**Module just declares functions:**
```nano
# modules/glfw/glfw.nano - Incomplete
extern fn glfwMakeContextCurrent(window: int) -> void
```

**Result:** Compilation fails with type errors because transpiler doesn't know about GLFW.

### After (Proper System)

**Transpiler is generic:**
```c
// src/transpiler.c - GOOD
OpaqueTypeDef *opaque = env_get_opaque_type(env, param_type_name);
if (opaque) {
    sb_appendf(sb, "(%s)", opaque->c_type_name);
}
// Works for ANY library automatically
```

**Module declares types:**
```nano
# modules/glfw/glfw.nano - Complete
opaque type GLFWwindow
extern fn glfwMakeContextCurrent(window: GLFWwindow) -> void
```

**Result:** Compilation succeeds. Transpiler automatically generates correct casts.

---

## Conclusion

**Current architecture is a hack.** The transpiler should not know about SDL, GLFW, or any specific library.

**Proper architecture:** 
1. Modules declare opaque types
2. Transpiler handles them generically
3. Type system provides safety
4. No library-specific code in compiler

This scales to ANY C library and maintains separation of concerns.

**Implementation estimate:** 1-2 weeks for full implementation and migration of existing modules.

