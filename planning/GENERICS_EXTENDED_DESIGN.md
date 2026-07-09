# Extended Generics Design - Arbitrary Type Parameters

## Goal

Extend the MVP generics (`List<int>`, `List<string>`, `List<Token>`) to support **arbitrary user-defined types**: `List<Point>`, `List<Player>`, etc.

---

## Current MVP Limitations

**What Works**:
```nano
let numbers: List<int> = (list_int_new)
let words: List<string> = (list_string_new)
let tokens: List<Token> = (list_token_new)
```

**What Doesn't Work**:
```nano
struct Point { x: int, y: int }
let points: List<Point> = ???  /* ❌ Not supported */
```

**Why**: Parser hardcodes mapping:
```c
if (strcmp(type_param_tok->value, "Token") == 0) {
    type = TYPE_LIST_TOKEN;
} else {
    /* Error: unsupported */
}
```

---

## Extended Design: On-Demand Specialization

### Strategy

Instead of full monomorphization (which requires generic function definitions), use **on-demand specialization**:

1. **Parser**: Recognize `List<UserType>` and store type parameter
2. **Type Checker**: Track which `List<T>` instantiations are used
3. **Transpiler**: Generate specialized `List_UserType` struct and functions
4. **Runtime**: Create specialized list implementations as needed

### Example Flow

**Input**:
```nano
struct Point { x: int, y: int }

fn test() -> int {
    let points: List<Point> = (list_point_new)
    (list_point_push points (Point { x: 10, y: 20 }))
    return 0
}
```

**Step 1: Parser**
- Sees `List<Point>`
- Creates internal representation: `GenericType{name: "List", param: "Point"}`
- Maps to new type: `TYPE_LIST_GENERIC` with stored param name

**Step 2: Type Checker**
- Registers instantiation: `List<Point>` → needs `List_Point` specialization
- Tracks in `env->generic_instances`

**Step 3: Transpiler**
- Generates specialized struct:
  ```c
  typedef struct {
      struct Point *data;
      int count;
      int capacity;
  } List_Point;
  ```
- Generates specialized functions:
  ```c
  List_Point* list_point_new();
  void list_point_push(List_Point *list, struct Point value);
  ```
- Transpiles code to use specialized types

---

## Implementation Plan

### Phase 1: Type System Extensions

**Add to `nanolang.h`**:
```c
/* Extended type for generic list with user type */
#define TYPE_LIST_GENERIC 100  /* Or next available number */

/* Track generic list instantiations */
typedef struct {
    char *element_type_name;  /* e.g., "Point" */
    char *specialized_name;   /* e.g., "List_Point" */
} ListInstantiation;

/* Add to Environment */
typedef struct {
    /* ... existing fields ... */
    ListInstantiation *list_instances;
    int list_instance_count;
    int list_instance_capacity;
} Environment;
```

### Phase 2: Parser Changes

**Extend `parse_type_with_element`** in `src/parser.c`:

```c
/* Check for generic type syntax: List<T> */
if (strcmp(tok->value, "List") == 0) {
    advance(p);  /* consume 'List' */
    if (current_token(p)->type == TOKEN_LT) {
        advance(p);  /* consume '<' */
        
        /* Parse type parameter */
        Token *type_param_tok = current_token(p);
        if (type_param_tok->type == TOKEN_TYPE_INT) {
            type = TYPE_LIST_INT;
            advance(p);
        } else if (type_param_tok->type == TOKEN_TYPE_STRING) {
            type = TYPE_LIST_STRING;
            advance(p);
        } else if (type_param_tok->type == TOKEN_IDENTIFIER) {
            /* NEW: Support arbitrary types */
            if (strcmp(type_param_tok->value, "Token") == 0) {
                type = TYPE_LIST_TOKEN;
            } else {
                /* Generic user type - store for later */
                type = TYPE_LIST_GENERIC;
                /* TODO: Store element_type_name somewhere */
            }
            advance(p);
        }
        
        /* Expect '>' */
        if (current_token(p)->type != TOKEN_GT) {
            /* Error */
        }
        advance(p);
        return type;
    }
}
```

**Challenge**: Need to store the element type name (`"Point"`) for later use.

**Solution**: Extend AST nodes that have types to include type name:
```c
typedef struct {
    char *name;
    Type var_type;
    char *type_name;      /* NEW: For TYPE_LIST_GENERIC, stores "Point" */
    bool is_mut;
    struct ASTNode *value;
} ASTLet;
```

### Phase 3: Type Checker Changes

**Track Instantiations** in `src/typechecker.c`:

```c
/* When processing let statement with TYPE_LIST_GENERIC */
if (stmt->as.let.var_type == TYPE_LIST_GENERIC) {
    const char *element_type = stmt->as.let.type_name;
    
    /* Register this instantiation */
    register_list_instantiation(env, element_type);
}

/* Helper function */
void register_list_instantiation(Environment *env, const char *element_type) {
    /* Check if already registered */
    for (int i = 0; i < env->list_instance_count; i++) {
        if (strcmp(env->list_instances[i].element_type_name, element_type) == 0) {
            return;  /* Already have it */
        }
    }
    
    /* Add new instantiation */
    if (env->list_instance_count >= env->list_instance_capacity) {
        env->list_instance_capacity *= 2;
        env->list_instances = realloc(env->list_instances, 
            sizeof(ListInstantiation) * env->list_instance_capacity);
    }
    
    ListInstantiation inst;
    inst.element_type_name = strdup(element_type);
    
    /* Generate specialized name: List<Point> -> List_Point */
    char specialized[256];
    snprintf(specialized, sizeof(specialized), "List_%s", element_type);
    inst.specialized_name = strdup(specialized);
    
    env->list_instances[env->list_instance_count++] = inst;
}
```

### Phase 4: Transpiler Changes

**Generate Specialized Types** in `src/transpiler.c`:

```c
/* In generate_code, before main code generation */
for (int i = 0; i < env->list_instance_count; i++) {
    ListInstantiation *inst = &env->list_instances[i];
    generate_list_type(sb, inst);
}

/* Generate list type */
void generate_list_type(StringBuilder *sb, ListInstantiation *inst) {
    const char *elem_type = inst->element_type_name;
    const char *specialized = inst->specialized_name;
    
    /* Generate struct */
    sb_appendf(sb, "typedef struct {\n");
    sb_appendf(sb, "    struct %s *data;\n", elem_type);
    sb_appendf(sb, "    int count;\n");
    sb_appendf(sb, "    int capacity;\n");
    sb_appendf(sb, "} %s;\n\n", specialized);
    
    /* Generate constructor */
    sb_appendf(sb, "%s* %s_new() {\n", specialized, specialized);
    sb_appendf(sb, "    %s *list = malloc(sizeof(%s));\n", specialized, specialized);
    sb_appendf(sb, "    list->data = malloc(sizeof(struct %s) * 4);\n", elem_type);
    sb_appendf(sb, "    list->count = 0;\n");
    sb_appendf(sb, "    list->capacity = 4;\n");
    sb_appendf(sb, "    return list;\n");
    sb_appendf(sb, "}\n\n");
    
    /* Generate push */
    sb_appendf(sb, "void %s_push(%s *list, struct %s value) {\n", 
               specialized, specialized, elem_type);
    sb_appendf(sb, "    if (list->count >= list->capacity) {\n");
    sb_appendf(sb, "        list->capacity *= 2;\n");
    sb_appendf(sb, "        list->data = realloc(list->data, "
                   "sizeof(struct %s) * list->capacity);\n", elem_type);
    sb_appendf(sb, "    }\n");
    sb_appendf(sb, "    list->data[list->count++] = value;\n");
    sb_appendf(sb, "}\n\n");
    
    /* Generate get */
    sb_appendf(sb, "struct %s %s_get(%s *list, int index) {\n",
               elem_type, specialized, specialized);
    sb_appendf(sb, "    return list->data[index];\n");
    sb_appendf(sb, "}\n\n");
    
    /* Generate length */
    sb_appendf(sb, "int %s_length(%s *list) {\n", specialized, specialized);
    sb_appendf(sb, "    return list->count;\n");
    sb_appendf(sb, "}\n\n");
}
```

**Transpile Variables**:
```c
case AST_LET:
    if (stmt->as.let.var_type == TYPE_LIST_GENERIC) {
        const char *specialized = get_specialized_name(env, stmt->as.let.type_name);
        sb_appendf(sb, "%s* %s = ", specialized, stmt->as.let.name);
    } else {
        /* existing code */
    }
    break;
```

---

## Usage Example

**User Code**:
```nano
struct Point {
    x: int,
    y: int
}

fn test_points() -> int {
    let points: List<Point> = (List_Point_new)
    (List_Point_push points (Point { x: 10, y: 20 }))
    (List_Point_push points (Point { x: 30, y: 40 }))
    
    let len: int = (List_Point_length points)
    if (!= len 2) { return 1 } else {}
    
    let first: Point = (List_Point_get points 0)
    if (!= first.x 10) { return 2 } else {}
    
    return 0
}

shadow test_points {
    assert (== (test_points) 0)
}
```

**Generated C**:
```c
struct Point {
    int64_t x;
    int64_t y;
};

/* Auto-generated by compiler */
typedef struct {
    struct Point *data;
    int count;
    int capacity;
} List_Point;

List_Point* List_Point_new() {
    List_Point *list = malloc(sizeof(List_Point));
    list->data = malloc(sizeof(struct Point) * 4);
    list->count = 0;
    list->capacity = 4;
    return list;
}

void List_Point_push(List_Point *list, struct Point value) {
    if (list->count >= list->capacity) {
        list->capacity *= 2;
        list->data = realloc(list->data, sizeof(struct Point) * list->capacity);
    }
    list->data[list->count++] = value;
}

/* ... main code uses List_Point ... */
```

---

## Limitations & Future Work

### Current Scope (This Implementation)
- ✅ `List<UserType>` for any user-defined struct
- ✅ Automatic generation of specialized types
- ✅ Type-safe usage

### Out of Scope (Future)
- ❌ Multiple type parameters: `Map<K, V>`
- ❌ Generic functions: `fn first<T>(list: List<T>) -> T`
- ❌ Nested generics: `List<List<int>>`
- ❌ Generic structs: `struct Pair<T, U> { ... }`

### Rationale
Focus on the 80% use case (generic lists) that provides immediate value for self-hosting.

---

## Testing Strategy

### Test 1: Simple Struct
```nano
struct Point { x: int, y: int }
let points: List<Point> = (List_Point_new)
```

### Test 2: Complex Struct
```nano
struct Player {
    name: string,
    score: int,
    active: bool
}
let players: List<Player> = (List_Player_new)
```

### Test 3: Multiple Lists
```nano
struct Point { x: int, y: int }
struct Color { r: int, g: int, b: int }

let points: List<Point> = (List_Point_new)
let colors: List<Color> = (List_Color_new)
/* Compiler generates both List_Point and List_Color */
```

---

## Implementation Checklist

- [ ] Add `TYPE_LIST_GENERIC` to type enum
- [ ] Add `ListInstantiation` tracking to Environment
- [ ] Extend AST nodes to store type names
- [ ] Update parser to recognize arbitrary types
- [ ] Implement instantiation tracking in type checker
- [ ] Generate specialized types in transpiler
- [ ] Update variable transpilation for generic lists
- [ ] Write comprehensive tests
- [ ] Update documentation

---

## Timeline

- **Phase 1** (Type System): 1 hour
- **Phase 2** (Parser): 1 hour
- **Phase 3** (Type Checker): 1 hour
- **Phase 4** (Transpiler): 2 hours
- **Testing**: 1 hour

**Total**: ~6 hours (as estimated)

---

*Ready to implement!*

