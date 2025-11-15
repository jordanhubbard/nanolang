# Phase 3: Extended Generics - Progress Update

## Status: Parser Complete, Type Checker & Transpiler In Progress

---

## ✅ Completed: Parser Support

Successfully extended the parser to accept arbitrary type parameters for generic lists!

### Changes Made

**1. Added TYPE_LIST_GENERIC** (`src/nanolang.h`):
```c
TYPE_LIST_GENERIC, /* Generic list with user-defined type: List<Point>, List<Player>, etc. */
```

**2. Extended GenericInstantiation** (`src/nanolang.h`):
```c
typedef struct {
    char *generic_name;        /* e.g., "List" */
    Type *type_args;           /* e.g., [TYPE_INT] or [TYPE_LIST_GENERIC] */
    int type_arg_count;
    char **type_arg_names;     /* NEW: e.g., ["Point"] for user types */
    char *concrete_name;       /* e.g., "List_Point" */
} GenericInstantiation;
```

**3. Updated Parser** (`src/parser.c`):
- Added `type_param_name_out` parameter to `parse_type_with_element`
- Extended `List<T>` parsing to accept arbitrary identifiers
- Captures type parameter name (e.g., "Point") for `TYPE_LIST_GENERIC`
- Stores type parameter in AST node's `type_name` field

### Test Results

**Before**:
```nano
let points: List<Point> = ...
/* Error: Unsupported generic type parameter 'Point' for List */
```

**After**:
```nano
let points: List<Point> = ...
/* ✅ Parser accepts this! */
/* ❌ Type checker rejects (not implemented yet) */
```

---

## 🚧 In Progress: Type Checker & Transpiler

### Remaining Tasks

**1. Type Checker** (`src/typechecker.c`):
- [ ] Handle `TYPE_LIST_GENERIC` in type checking
- [ ] Register instantiations when `List<UserType>` is used
- [ ] Track which specializations are needed

**2. Environment** (`src/env.c`):
- [ ] Add helper function: `register_list_instantiation()`
- [ ] Initialize `generic_instances` array
- [ ] Free type_arg_names in cleanup

**3. Transpiler** (`src/transpiler.c`):
- [ ] Generate specialized list types
- [ ] Generate specialized functions (new, push, get, length)
- [ ] Handle `TYPE_LIST_GENERIC` in variable declarations
- [ ] Use specialized names in generated C code

---

## Implementation Plan

### Step 1: Type Checker Support (1 hour)

**Add to `src/typechecker.c`**:
```c
/* In check_statement for AST_LET */
if (stmt->as.let.var_type == TYPE_LIST_GENERIC) {
    const char *element_type = stmt->as.let.type_name;
    
    /* Verify element type exists (struct defined) */
    if (!env_get_struct(env, element_type)) {
        fprintf(stderr, "Error: Unknown type '%s' in List<%s>\n", 
                element_type, element_type);
        return false;
    }
    
    /* Register this instantiation for code generation */
    register_list_instantiation(env, element_type);
}
```

**Add to `src/env.c`**:
```c
void register_list_instantiation(Environment *env, const char *element_type) {
    /* Check if already registered */
    for (int i = 0; i < env->generic_instance_count; i++) {
        GenericInstantiation *inst = &env->generic_instances[i];
        if (strcmp(inst->generic_name, "List") == 0 &&
            inst->type_arg_names && 
            strcmp(inst->type_arg_names[0], element_type) == 0) {
            return;  /* Already registered */
        }
    }
    
    /* Add new instantiation */
    if (env->generic_instance_count >= env->generic_instance_capacity) {
        env->generic_instance_capacity *= 2;
        env->generic_instances = realloc(env->generic_instances,
            sizeof(GenericInstantiation) * env->generic_instance_capacity);
    }
    
    GenericInstantiation inst;
    inst.generic_name = strdup("List");
    inst.type_arg_count = 1;
    inst.type_args = malloc(sizeof(Type));
    inst.type_args[0] = TYPE_LIST_GENERIC;
    inst.type_arg_names = malloc(sizeof(char*));
    inst.type_arg_names[0] = strdup(element_type);
    
    /* Generate specialized name: List<Point> -> List_Point */
    char specialized[256];
    snprintf(specialized, sizeof(specialized), "List_%s", element_type);
    inst.concrete_name = strdup(specialized);
    
    env->generic_instances[env->generic_instance_count++] = inst;
}
```

### Step 2: Transpiler Code Generation (2 hours)

**Generate Specialized Types** in `src/transpiler.c`:
```c
/* Before main code, generate all list specializations */
for (int i = 0; i < env->generic_instance_count; i++) {
    GenericInstantiation *inst = &env->generic_instances[i];
    if (strcmp(inst->generic_name, "List") == 0) {
        generate_list_specialization(sb, inst->type_arg_names[0], inst->concrete_name);
    }
}

void generate_list_specialization(StringBuilder *sb, 
                                  const char *elem_type,
                                  const char *specialized_name) {
    /* Struct definition */
    sb_appendf(sb, "typedef struct {\n");
    sb_appendf(sb, "    struct %s *data;\n", elem_type);
    sb_appendf(sb, "    int count;\n");
    sb_appendf(sb, "    int capacity;\n");
    sb_appendf(sb, "} %s;\n\n", specialized_name);
    
    /* Constructor */
    sb_appendf(sb, "%s* %s_new() {\n", specialized_name, specialized_name);
    sb_appendf(sb, "    %s *list = malloc(sizeof(%s));\n", 
               specialized_name, specialized_name);
    sb_appendf(sb, "    list->data = malloc(sizeof(struct %s) * 4);\n", elem_type);
    sb_appendf(sb, "    list->count = 0;\n");
    sb_appendf(sb, "    list->capacity = 4;\n");
    sb_appendf(sb, "    return list;\n");
    sb_appendf(sb, "}\n\n");
    
    /* Push */
    sb_appendf(sb, "void %s_push(%s *list, struct %s value) {\n",
               specialized_name, specialized_name, elem_type);
    sb_appendf(sb, "    if (list->count >= list->capacity) {\n");
    sb_appendf(sb, "        list->capacity *= 2;\n");
    sb_appendf(sb, "        list->data = realloc(list->data, "
                   "sizeof(struct %s) * list->capacity);\n", elem_type);
    sb_appendf(sb, "    }\n");
    sb_appendf(sb, "    list->data[list->count++] = value;\n");
    sb_appendf(sb, "}\n\n");
    
    /* Get */
    sb_appendf(sb, "struct %s %s_get(%s *list, int index) {\n",
               elem_type, specialized_name, specialized_name);
    sb_appendf(sb, "    return list->data[index];\n");
    sb_appendf(sb, "}\n\n");
    
    /* Length */
    sb_appendf(sb, "int %s_length(%s *list) {\n", specialized_name, specialized_name);
    sb_appendf(sb, "    return list->count;\n");
    sb_appendf(sb, "}\n\n");
}
```

**Handle TYPE_LIST_GENERIC in variable transpilation**:
```c
case AST_LET:
    if (stmt->as.let.var_type == TYPE_LIST_GENERIC) {
        /* Generate specialized pointer type */
        char specialized[256];
        snprintf(specialized, sizeof(specialized), "List_%s", stmt->as.let.type_name);
        sb_appendf(sb, "%s* %s = ", specialized, stmt->as.let.name);
    } else {
        /* existing code */
    }
    break;
```

### Step 3: Testing (30 minutes)

**Test Case 1**: Simple struct
```nano
struct Point { x: int, y: int }

fn test() -> int {
    let points: List<Point> = (List_Point_new)
    (List_Point_push points (Point { x: 10, y: 20 }))
    let len: int = (List_Point_length points)
    return len
}
```

**Test Case 2**: Multiple types
```nano
struct Point { x: int, y: int }
struct Color { r: int, g: int, b: int }

fn test() -> int {
    let points: List<Point> = (List_Point_new)
    let colors: List<Color> = (List_Color_new)
    /* Compiler generates both List_Point and List_Color */
    return 0
}
```

---

## Timeline

**Completed**: Parser (1 hour) ✅  
**Remaining**:
- Type Checker: 1 hour  
- Transpiler: 2 hours  
- Testing: 30 minutes

**Total Remaining**: ~3.5 hours

---

## Next Session Recommendation

Pick up from here with type checker implementation. All foundation is in place:
- ✅ Type system extended
- ✅ Parser accepts syntax
- ✅ AST stores information
- 🚧 Need: Type checker + transpiler

---

*Paused at: Parser complete, ready for type checker*  
*Date: November 14, 2025*  
*Token usage: Efficient progress*

