# Enum Implementation Plan
**Date:** November 12, 2025  
**Status:** In Progress  
**Test File:** `examples/18_enum_test.nano`

---

## Current Status

### ✅ What's Working
- Parser recognizes enum definitions
- Parser stores variant names and values
- Type checker registers enums in environment
- Environment can look up enum variants

### ❌ What's Missing
1. Type checker doesn't recognize `Color.Red` as enum variant
2. Interpreter doesn't evaluate enum values
3. Transpiler doesn't generate C enum code

---

## Implementation Strategy

### Approach: Smart Type Checking (Recommended)

**Current behavior:**
```nano
let c: Color = Color.Red  # Parsed as: identifier Color, field access .Red
```

**Parser sees:**
- `AST_IDENTIFIER("Color")`  
- `AST_FIELD_ACCESS(object=Color, field="Red")`

**Solution:**
In type checker, detect when "object" in field access is an enum name:
1. Check if identifier is an enum name
2. If yes, look up variant value
3. Treat as integer constant
4. Check type compatibility

---

## Implementation Steps

### Step 1: Type Checker - Recognize Enum Variants

**File:** `src/typechecker.c`

**Location:** `check_expression()` function, `AST_FIELD_ACCESS` case

**Current code:**
```c
case AST_FIELD_ACCESS: {
    // Get type of object
    Type obj_type = check_expression(expr->as.field_access.object, env);
    
    if (obj_type != TYPE_STRUCT) {
        fprintf(stderr, "Field access requires a struct\n");
        return TYPE_UNKNOWN;
    }
    // ... struct field lookup
}
```

**New code:**
```c
case AST_FIELD_ACCESS: {
    // Special case: Check if this is an enum variant access
    if (expr->as.field_access.object->type == AST_IDENTIFIER) {
        const char *enum_name = expr->as.field_access.object->as.identifier;
        EnumDef *enum_def = env_get_enum(env, enum_name);
        
        if (enum_def) {
            // This is an enum variant access (e.g., Color.Red)
            const char *variant_name = expr->as.field_access.field_name;
            
            // Verify variant exists
            for (int i = 0; i < enum_def->variant_count; i++) {
                if (strcmp(enum_def->variant_names[i], variant_name) == 0) {
                    return TYPE_ENUM;  // Or TYPE_INT
                }
            }
            
            fprintf(stderr, "Error: Enum '%s' has no variant '%s'\n",
                    enum_name, variant_name);
            tc.has_error = true;
            return TYPE_UNKNOWN;
        }
    }
    
    // Regular struct field access
    Type obj_type = check_expression(expr->as.field_access.object, env);
    // ... rest of struct handling
}
```

### Step 2: Type Checker - Handle Enum Types

**Variables with enum types:**
```nano
let c: Color = Color.Red
```

Currently this fails because:
1. `Color` as a type isn't recognized
2. Type mismatch checking doesn't understand enums

**Solution:** 
- Enums are represented as `TYPE_ENUM` with a name
- Need to track which enum type (like structs with `struct_type_name`)

**OR simpler:**
- Treat enums as `TYPE_INT` everywhere
- Enums are just named integer constants
- No type safety for enum assignments

**Recommendation:** Simple approach (enums as integers)
- Easier to implement
- Matches C behavior  
- Can add type safety later

### Step 3: Interpreter - Evaluate Enum Variants

**File:** `src/eval.c`

**Location:** `eval_expression()` function

**Add to `AST_FIELD_ACCESS` case:**
```c
case AST_FIELD_ACCESS: {
    // Check if this is an enum variant
    if (expr->as.field_access.object->type == AST_IDENTIFIER) {
        const char *enum_name = expr->as.field_access.object->as.identifier;
        EnumDef *enum_def = env_get_enum(env, enum_name);
        
        if (enum_def) {
            // Lookup variant value
            const char *variant_name = expr->as.field_access.field_name;
            for (int i = 0; i < enum_def->variant_count; i++) {
                if (strcmp(enum_def->variant_names[i], variant_name) == 0) {
                    return create_int(enum_def->variant_values[i]);
                }
            }
            
            fprintf(stderr, "Error: Enum '%s' has no variant '%s'\n",
                    enum_name, variant_name);
            return create_void();
        }
    }
    
    // Regular struct field access
    // ... existing code
}
```

### Step 4: Transpiler - Generate C Enums

**File:** `src/transpiler.c`

**Location:** After struct definitions, before function declarations

**Add:**
```c
/* Generate enum typedefs */
sb_append(sb, "/* ========== Enum Definitions ========== */\n\n");
for (int i = 0; i < env->enum_count; i++) {
    EnumDef *edef = &env->enums[i];
    sb_appendf(sb, "typedef enum {\n");
    for (int j = 0; j < edef->variant_count; j++) {
        sb_appendf(sb, "    %s_%s = %d",
                  edef->name,
                  edef->variant_names[j],
                  edef->variant_values[j]);
        if (j < edef->variant_count - 1) sb_append(sb, ",\n");
        else sb_append(sb, "\n");
    }
    sb_appendf(sb, "} %s;\n\n", edef->name);
}
sb_append(sb, "/* ========== End Enum Definitions ========== */\n\n");
```

**Generates:**
```c
typedef enum {
    Color_Red = 0,
    Color_Green = 1,
    Color_Blue = 2
} Color;
```

**For enum variant access:**
In `transpile_expression()`, add to `AST_FIELD_ACCESS`:
```c
case AST_FIELD_ACCESS: {
    // Check if enum variant
    if (expr->as.field_access.object->type == AST_IDENTIFIER) {
        const char *enum_name = expr->as.field_access.object->as.identifier;
        if (env_get_enum(env, enum_name)) {
            // Transpile as: EnumName_VariantName
            sb_appendf(sb, "%s_%s",
                      enum_name,
                      expr->as.field_access.field_name);
            return;
        }
    }
    
    // Regular field access
    // ... existing code
}
```

---

## Testing Plan

### Test 1: Basic Enum
```nano
enum Color { Red, Green, Blue }

fn test() -> int {
    let c: Color = Color.Red
    return c
}

shadow test {
    assert (== (test) 0)
}
```

### Test 2: Explicit Values
```nano
enum Status { Pending = 0, Active = 1, Done = 2 }

fn test() -> int {
    let s: Status = Status.Done
    return s
}

shadow test {
    assert (== (test) 2)
}
```

### Test 3: Enum Comparison
```nano
enum HttpStatus { Ok = 200, NotFound = 404 }

fn is_error(status: HttpStatus) -> bool {
    return (>= status 400)
}

shadow is_error {
    assert (== (is_error HttpStatus.Ok) false)
    assert (== (is_error HttpStatus.NotFound) true)
}
```

### Test 4: Enum in Function
```nano
fn color_to_string(c: Color) -> string {
    if (== c Color.Red) {
        return "red"
    }
    if (== c Color.Green) {
        return "green"
    }
    return "blue"
}
```

---

## Implementation Order

1. ✅ Type checker - recognize enum variants (30 min)
2. ✅ Interpreter - evaluate enum variants (20 min)
3. ✅ Transpiler - generate enum typedefs (20 min)
4. ✅ Transpiler - transpile enum access (10 min)
5. ✅ Testing - run 18_enum_test.nano (10 min)

**Total Estimated Time:** 1.5 hours

---

## Alternative: Simpler Approach

If time is limited, enums can work with minimal changes:

**Ultra-simple:**
1. Treat enum types as `int` everywhere
2. Only add enum variant evaluation in interpreter
3. Only add enum typedef generation in transpiler
4. Skip full type tracking

This would make enums work in ~30 minutes but with less type safety.

---

## Next Session

Start with Step 1 in `src/typechecker.c`, test incrementally, then move to interpreter, then transpiler.


