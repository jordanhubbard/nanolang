# Generics Implementation Design

## Overview
Implement generic types for nanolang using **monomorphization** (compile-time code generation). This approach generates specialized versions of generic types for each concrete type used.

## Goals
1. Enable `List<T>` for any type T
2. Support generic structs and functions
3. Zero runtime overhead (all resolved at compile time)
4. Clean, readable syntax
5. Foundation for future generic features

## Syntax Design

### Generic Type Parameters
```nano
/* Generic struct */
struct List<T> {
    data: array<T>,
    length: int,
    capacity: int
}

/* Generic function */
fn first<T>(list: List<T>) -> T {
    return (at list.data 0)
}

/* Usage - type parameters inferred or explicit */
let numbers: List<int> = (list_new<int>)
let tokens: List<Token> = (list_new<Token>)
```

### Multiple Type Parameters
```nano
struct Pair<A, B> {
    first: A,
    second: B
}

struct Map<K, V> {
    keys: List<K>,
    values: List<V>
}
```

## Implementation Strategy: Monomorphization

### Concept
Generate specialized code for each concrete type usage.

**User writes:**
```nano
let nums: List<int> = (list_new<int>)
let toks: List<Token> = (list_new<Token>)
```

**Compiler generates:**
```c
/* Specialized for int */
typedef struct List_int {
    int64_t* data;
    int64_t length;
    int64_t capacity;
} List_int;

/* Specialized for Token */
typedef struct List_Token {
    Token* data;
    int64_t length;
    int64_t capacity;
} List_Token;
```

### Advantages
- ✅ Zero runtime overhead
- ✅ Full type safety
- ✅ Easy to debug (see actual types in C)
- ✅ No runtime type information needed
- ✅ Optimizable by C compiler

### Disadvantages
- ⚠️ Code bloat (multiple copies of same logic)
- ⚠️ Compile time increases with many instantiations
- ⚠️ No dynamic dispatch

**Assessment**: Advantages far outweigh disadvantages for our use case.

---

## Phase 1: Lexer Changes

### Add Generic Delimiters
Need to lex `<` and `>` as separate tokens when used for generics vs. comparison operators.

**Challenge**: `<` and `>` are already comparison operators!

**Solution**: Context-sensitive lexing or lookahead.

```nano
List<int>     /* < and > are generic delimiters */
(< x 10)      /* < is comparison operator */
```

**Approach**: 
- Lex `<` and `>` as normal
- Parser handles disambiguation based on context
- After identifier: likely generic (`List<int>`)
- After value/expression: likely comparison (`x < 10`)

**Token Types** (already exist):
- `TOKEN_LT` - Less than / Left angle bracket
- `TOKEN_GT` - Greater than / Right angle bracket

---

## Phase 2: Parser Changes

### Parse Generic Type Syntax

```nano
/* Type annotation with generic parameters */
type → IDENTIFIER '<' type_list '>'
     | IDENTIFIER
     
type_list → type
          | type ',' type_list
```

### AST Nodes

```c
/* Generic type reference */
struct ASTGenericType {
    char *base_name;           // e.g., "List"
    Type *type_params;         // e.g., [TYPE_INT]
    int type_param_count;
}

/* Generic struct definition */
struct ASTGenericStructDef {
    char *name;                // e.g., "List"
    char **type_param_names;   // e.g., ["T"]
    int type_param_count;
    /* ... fields ... */
}
```

---

## Phase 3: Type Checker Changes

### Track Generic Instantiations

```c
typedef struct GenericInstantiation {
    char *generic_name;        // "List"
    Type *type_args;           // [TYPE_INT]
    int type_arg_count;
    char *concrete_name;       // "List_int" (generated)
} GenericInstantiation;

/* Environment tracks instantiations */
typedef struct Environment {
    /* ... existing fields ... */
    GenericInstantiation *generic_instances;
    int generic_instance_count;
} Environment;
```

### Type Checking Algorithm

1. **Encounter generic type**: `List<int>`
2. **Check if already instantiated**: Look up ("List", [int])
3. **If new**:
   - Generate concrete name: "List_int"
   - Mark for code generation
   - Record instantiation
4. **Substitute**: Replace `List<int>` with concrete type

---

## Phase 4: Transpiler Changes

### Generate Concrete Types

```c
/* For each instantiation of List<T> */
for (each GenericInstantiation inst) {
    generate_concrete_struct(inst);
    generate_concrete_functions(inst);
}

/* Example output for List<int> */
typedef struct List_int {
    int64_t* data;
    int64_t length;
    int64_t capacity;
} List_int;

List_int* list_int_new() {
    List_int* list = malloc(sizeof(List_int));
    list->data = malloc(sizeof(int64_t) * 4);
    list->length = 0;
    list->capacity = 4;
    return list;
}

void list_int_push(List_int* list, int64_t value) {
    if (list->length >= list->capacity) {
        list->capacity *= 2;
        list->data = realloc(list->data, sizeof(int64_t) * list->capacity);
    }
    list->data[list->length++] = value;
}
```

### Name Mangling Strategy

```
List<int>           → List_int
List<Token>         → List_Token
Map<string, int>    → Map_string_int
Pair<List<int>, bool> → Pair_List_int_bool
```

**Algorithm**:
1. Start with base name: "List"
2. Append underscore: "List_"
3. For each type parameter:
   - If primitive: append type name ("int", "bool")
   - If compound: recursively mangle
   - Separate with underscore

---

## Phase 5: Standard Library

### Implement List<T> as Generic

```nano
/* Generic List definition (pseudocode - actual will be C template) */
struct List<T> {
    data: array<T>,
    length: int,
    capacity: int
}

/* Generic functions (extern - implemented in C) */
extern fn list_new<T>() -> List<T>
extern fn list_push<T>(list: List<T>, value: T) -> void
extern fn list_pop<T>(list: List<T>) -> T
extern fn list_get<T>(list: List<T>, index: int) -> T
extern fn list_length<T>(list: List<T>) -> int
```

### C Runtime Implementation

Create generic template in C:

```c
/* src/runtime/list_generic.h */
#define DEFINE_LIST_TYPE(T, TypeName) \
    typedef struct List_##TypeName { \
        T* data; \
        int64_t length; \
        int64_t capacity; \
    } List_##TypeName; \
    \
    List_##TypeName* list_##TypeName##_new() { \
        List_##TypeName* list = malloc(sizeof(List_##TypeName)); \
        list->data = malloc(sizeof(T) * 4); \
        list->length = 0; \
        list->capacity = 4; \
        return list; \
    } \
    \
    void list_##TypeName##_push(List_##TypeName* list, T value) { \
        if (list->length >= list->capacity) { \
            list->capacity *= 2; \
            list->data = realloc(list->data, sizeof(T) * list->capacity); \
        } \
        list->data[list->length++] = value; \
    } \
    /* ... more functions ... */

/* Instantiate for common types */
DEFINE_LIST_TYPE(int64_t, int)
DEFINE_LIST_TYPE(Token, Token)
DEFINE_LIST_TYPE(char*, string)
```

---

## Implementation Phases

### Phase 1: Minimal Generic Support (Week 1)
- [x] Design document (this file)
- [ ] Parse `List<T>` syntax
- [ ] Type checker tracks instantiations
- [ ] Transpiler generates concrete types
- [ ] Test with `List<int>` and `List<Token>`

### Phase 2: Generic Functions (Week 2)
- [ ] Parse generic function syntax
- [ ] Type checker handles generic functions
- [ ] Transpiler generates specialized functions
- [ ] Test with generic utility functions

### Phase 3: Multiple Type Parameters (Week 3)
- [ ] Support `Map<K, V>` style syntax
- [ ] Complex type parameter resolution
- [ ] Nested generics: `List<List<int>>`

### Phase 4: Generic Unions (Future)
- [ ] `Result<T>` union type
- [ ] `Option<T>` union type
- [ ] Pattern matching with generics

---

## Example Usage

### Before (Current)
```nano
/* Must use specific list types */
extern fn list_int_new() -> int
extern fn list_int_push(list: int, value: int) -> void

extern fn list_string_new() -> int  
extern fn list_string_push(list: int, value: string) -> void

/* Cannot have List<Token> at all! */
```

### After (With Generics)
```nano
/* Single generic type */
extern fn list_new<T>() -> List<T>
extern fn list_push<T>(list: List<T>, value: T) -> void

/* Use with any type */
let numbers: List<int> = (list_new<int>)
(list_push numbers 42)

let tokens: List<Token> = (list_new<Token>)
(list_push tokens my_token)

/* Works with user types */
struct Point { x: int, y: int }
let points: List<Point> = (list_new<Point>)
```

---

## Testing Strategy

### Unit Tests
1. Parse generic syntax correctly
2. Type checker creates instantiations
3. Generated C code compiles
4. Runtime behavior matches expectations

### Integration Tests
```nano
/* test_generics.nano */
struct Point { x: int, y: int }

fn test_generic_list() -> int {
    let numbers: List<int> = (list_new<int>)
    (list_push numbers 10)
    (list_push numbers 20)
    let len: int = (list_length numbers)
    assert (== len 2)
    
    let first: int = (list_get numbers 0)
    assert (== first 10)
    
    return 0
}

fn test_list_of_structs() -> int {
    let points: List<Point> = (list_new<Point>)
    let p1: Point = Point { x: 1, y: 2 }
    (list_push points p1)
    
    let retrieved: Point = (list_get points 0)
    assert (== retrieved.x 1)
    assert (== retrieved.y 2)
    
    return 0
}
```

---

## Known Limitations

### 1. No Generic Constraints
Cannot specify that `T` must have certain properties.

```nano
/* Cannot do this (yet): */
fn sort<T: Comparable>(list: List<T>) -> void { ... }
```

**Future**: Add trait system or interface constraints.

### 2. No Partial Specialization
Cannot provide specialized implementations for specific types.

```nano
/* Cannot do this (yet): */
impl List<int> {
    fn sum(self) -> int { ... }  /* Only for List<int> */
}
```

**Future**: Add specialization support.

### 3. Code Bloat
Each instantiation generates full copy of code.

**Mitigation**: Only instantiate types that are actually used.

---

## Timeline

### Week 1: Core Generics
- Days 1-2: Parser changes
- Days 3-4: Type checker changes
- Days 5: Transpiler changes
- Days 6-7: Testing and fixes

### Week 2: Generic Functions
- Days 8-9: Function generic syntax
- Days 10-11: Type checking generics
- Days 12-14: Testing and refinement

### Week 3: Polish
- Days 15-17: Complex types, nested generics
- Days 18-19: Documentation
- Days 20-21: Integration with self-hosting

---

## Success Criteria

✅ **Minimum Viable Generics**:
1. Can define `List<T>` generic type
2. Can instantiate `List<int>` and `List<Token>`
3. Generated C code compiles and runs
4. Type checker enforces type safety
5. Self-hosted lexer can use `List<Token>`

✅ **Full Feature**:
1. Generic functions work
2. Multiple type parameters work
3. Nested generics work
4. Comprehensive test suite
5. Documentation complete

---

## Next Immediate Steps

1. **Add generic parsing** to parser.c
2. **Create GenericInstantiation tracking** in environment
3. **Implement monomorphization** in transpiler
4. **Test with `List<Token>`**
5. **Update lexer_v2.nano** to use generics

Let's start with Step 1!

