# Simple Lists Design for nanolang

**Status:** ✅ Design Approved - Simple Specialized Approach  
**Priority:** #3 for Self-Hosting (after structs, enums)  
**Principles:** Simplicity, Minimalism, Pragmatism

## Philosophy: Start Simple

**Decision:** Implement only the 4 list types needed for the self-hosting compiler.

**No generics, no monomorphization, no complexity** - just straightforward specialized implementations.

**Rationale:**
- ✅ Faster to implement (2-3 weeks vs 5-6 weeks)
- ✅ No compiler complexity (no generic type system)
- ✅ Sufficient for self-hosting
- ✅ Aligns with nanolang minimalism
- ✅ Can add generic syntax later as sugar

---

## The 4 List Types

### 1. `list_int` - Basic integer lists
```nano
let mut numbers: list_int = (list_int_new)
(list_int_push numbers 42)
let num: int = (list_int_get numbers 0)
```

**Use case:** Numeric data, indices, counters

---

### 2. `list_string` - String lists
```nano
let mut words: list_string = (list_string_new)
(list_string_push words "hello")
let word: string = (list_string_get words 0)
```

**Use case:** Split strings, file paths, error messages

---

### 3. `list_token` - Token lists (requires structs)
```nano
struct Token {
    type: int,
    value: string,
    line: int
}

let mut tokens: list_token = (list_token_new)
(list_token_push tokens tok)
let tok: Token = (list_token_get tokens 0)
```

**Use case:** Lexer output

---

### 4. `list_astnode` - AST node lists (requires structs)
```nano
struct ASTNode {
    type: int,
    # ... other fields
}

let mut statements: list_astnode = (list_astnode_new)
(list_astnode_push statements stmt)
let stmt: ASTNode = (list_astnode_get statements 0)
```

**Use case:** Parser output, block statements

---

## API (Same for All Types)

Each list type has 12 functions with identical semantics:

### Creation
```nano
list_T_new() -> list_T
list_T_with_capacity(cap: int) -> list_T
```

### Mutation (require mut)
```nano
list_T_push(list: mut list_T, item: T) -> void
list_T_pop(list: mut list_T) -> T
list_T_insert(list: mut list_T, index: int, item: T) -> void
list_T_remove(list: mut list_T, index: int) -> T
list_T_set(list: mut list_T, index: int, item: T) -> void
list_T_clear(list: mut list_T) -> void
```

### Query (read-only)
```nano
list_T_get(list: list_T, index: int) -> T
list_T_length(list: list_T) -> int
list_T_capacity(list: list_T) -> int
list_T_is_empty(list: list_T) -> bool
```

**Total:** 12 functions × 4 types = **48 functions**

---

## Example: Using list_token

```nano
struct Token {
    type: int,
    value: string,
    line: int,
    column: int
}

fn tokenize(source: string) -> list_token {
    let mut tokens: list_token = (list_token_new)
    let mut pos: int = 0
    let len: int = (str_length source)
    
    while (< pos len) {
        let c: string = (str_char_at source pos)
        
        if (str_equals c "(") {
            let tok: Token = Token {
                type: TOKEN_LPAREN,
                value: "(",
                line: 1,
                column: pos
            }
            (list_token_push tokens tok)
            set pos (+ pos 1)
        } else {
            # Handle other tokens...
            set pos (+ pos 1)
        }
    }
    
    return tokens
}

shadow tokenize {
    let tokens: list_token = (tokenize "(+ 1 2)")
    assert (== (list_token_length tokens) 5)
    
    let first: Token = (list_token_get tokens 0)
    assert (== first.type TOKEN_LPAREN)
}
```

---

## C Implementation (Example: list_int)

```c
// runtime/list_int.h
#ifndef LIST_INT_H
#define LIST_INT_H

#include <stdint.h>
#include <stdbool.h>

typedef struct {
    int64_t *data;
    int length;
    int capacity;
} List_int;

// Creation
List_int* list_int_new(void);
List_int* list_int_with_capacity(int capacity);

// Mutation
void list_int_push(List_int *list, int64_t value);
int64_t list_int_pop(List_int *list);
void list_int_insert(List_int *list, int index, int64_t value);
int64_t list_int_remove(List_int *list, int index);
void list_int_set(List_int *list, int index, int64_t value);
void list_int_clear(List_int *list);

// Query
int64_t list_int_get(List_int *list, int index);
int list_int_length(List_int *list);
int list_int_capacity(List_int *list);
bool list_int_is_empty(List_int *list);

// Cleanup
void list_int_free(List_int *list);

#endif
```

```c
// runtime/list_int.c
#include "list_int.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define INITIAL_CAPACITY 8

List_int* list_int_new(void) {
    return list_int_with_capacity(INITIAL_CAPACITY);
}

List_int* list_int_with_capacity(int capacity) {
    List_int *list = malloc(sizeof(List_int));
    if (!list) {
        fprintf(stderr, "Error: Failed to allocate list\n");
        exit(1);
    }
    
    list->data = malloc(sizeof(int64_t) * capacity);
    if (!list->data) {
        fprintf(stderr, "Error: Failed to allocate list data\n");
        exit(1);
    }
    
    list->length = 0;
    list->capacity = capacity;
    return list;
}

void list_int_push(List_int *list, int64_t value) {
    // Grow if needed
    if (list->length == list->capacity) {
        int new_capacity = list->capacity * 2;
        int64_t *new_data = realloc(list->data, sizeof(int64_t) * new_capacity);
        if (!new_data) {
            fprintf(stderr, "Error: Failed to grow list\n");
            exit(1);
        }
        list->data = new_data;
        list->capacity = new_capacity;
    }
    
    list->data[list->length++] = value;
}

int64_t list_int_pop(List_int *list) {
    if (list->length == 0) {
        fprintf(stderr, "Error: Cannot pop from empty list\n");
        exit(1);
    }
    return list->data[--list->length];
}

void list_int_insert(List_int *list, int index, int64_t value) {
    if (index < 0 || index > list->length) {
        fprintf(stderr, "Error: Insert index %d out of range [0, %d]\n", 
                index, list->length);
        exit(1);
    }
    
    // Grow if needed
    if (list->length == list->capacity) {
        int new_capacity = list->capacity * 2;
        int64_t *new_data = realloc(list->data, sizeof(int64_t) * new_capacity);
        if (!new_data) {
            fprintf(stderr, "Error: Failed to grow list\n");
            exit(1);
        }
        list->data = new_data;
        list->capacity = new_capacity;
    }
    
    // Shift elements right
    memmove(&list->data[index + 1], &list->data[index], 
            sizeof(int64_t) * (list->length - index));
    
    list->data[index] = value;
    list->length++;
}

int64_t list_int_remove(List_int *list, int index) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Remove index %d out of bounds (length %d)\n", 
                index, list->length);
        exit(1);
    }
    
    int64_t value = list->data[index];
    
    // Shift elements left
    memmove(&list->data[index], &list->data[index + 1], 
            sizeof(int64_t) * (list->length - index - 1));
    
    list->length--;
    return value;
}

void list_int_set(List_int *list, int index, int64_t value) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Set index %d out of bounds (length %d)\n", 
                index, list->length);
        exit(1);
    }
    list->data[index] = value;
}

void list_int_clear(List_int *list) {
    list->length = 0;
}

int64_t list_int_get(List_int *list, int index) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Get index %d out of bounds (length %d)\n", 
                index, list->length);
        exit(1);
    }
    return list->data[index];
}

int list_int_length(List_int *list) {
    return list->length;
}

int list_int_capacity(List_int *list) {
    return list->capacity;
}

bool list_int_is_empty(List_int *list) {
    return list->length == 0;
}

void list_int_free(List_int *list) {
    if (list) {
        free(list->data);
        free(list);
    }
}
```

**Total:** ~150 lines per list type

**To create other list types:** Copy this file, replace `int64_t` with `Token`/`ASTNode`/`char*`, done!

---

## Implementation Plan

### Week 1: Foundation
**Day 1-2: list_int**
- [ ] Implement list_int.c (150 lines)
- [ ] Write C unit tests
- [ ] Add to nanolang type system (TYPE_LIST_INT)
- [ ] Test thoroughly

**Day 3-4: list_string**
- [ ] Implement list_string.c (150 lines)
- [ ] Handle string memory (strdup)
- [ ] Write C unit tests
- [ ] Add to type system (TYPE_LIST_STRING)

**Day 5: Transpiler basics**
- [ ] Add list types to transpiler
- [ ] Generate #include statements
- [ ] Link runtime library

### Week 2: Struct Lists (requires structs complete)
**Day 1-2: list_token**
- [ ] Implement list_token.c (150 lines)
- [ ] Handle Token struct copying
- [ ] Write C unit tests
- [ ] Add to type system (TYPE_LIST_TOKEN)

**Day 3-4: list_astnode**
- [ ] Implement list_astnode.c (150 lines)
- [ ] Handle ASTNode struct copying
- [ ] Write C unit tests
- [ ] Add to type system (TYPE_LIST_ASTNODE)

**Day 5: Integration**
- [ ] Test all 4 list types together
- [ ] Update transpiler

### Week 3: Polish & Documentation
**Day 1-2: nanolang examples**
- [ ] Write example: list_int usage
- [ ] Write example: list_string usage
- [ ] Write example: lexer with list_token

**Day 3-4: Testing**
- [ ] Comprehensive shadow tests
- [ ] Negative tests (bounds errors)
- [ ] Memory leak testing (valgrind)

**Day 5: Documentation**
- [ ] Update SPECIFICATION.md
- [ ] Update STDLIB.md
- [ ] Update QUICK_REFERENCE.md

---

## Type System Changes

### Add New Types
```c
// In nanolang.h
typedef enum {
    TYPE_INT,
    TYPE_FLOAT,
    TYPE_BOOL,
    TYPE_STRING,
    TYPE_VOID,
    TYPE_ARRAY,
    TYPE_STRUCT,
    TYPE_ENUM,
    TYPE_LIST_INT,      // NEW
    TYPE_LIST_STRING,   // NEW
    TYPE_LIST_TOKEN,    // NEW
    TYPE_LIST_ASTNODE,  // NEW
    TYPE_UNKNOWN
} Type;
```

### Parser Changes
```c
// In parse_type()
case TOKEN_IDENTIFIER:
    if (strcmp(tok->value, "list_int") == 0) {
        type = TYPE_LIST_INT;
    } else if (strcmp(tok->value, "list_string") == 0) {
        type = TYPE_LIST_STRING;
    } else if (strcmp(tok->value, "list_token") == 0) {
        type = TYPE_LIST_TOKEN;
    } else if (strcmp(tok->value, "list_astnode") == 0) {
        type = TYPE_LIST_ASTNODE;
    } else {
        // Struct type
        type = TYPE_STRUCT;
    }
    break;
```

---

## Transpiler Changes

### Include Headers
```c
// At top of generated C file
#include "runtime/list_int.h"
#include "runtime/list_string.h"
#include "runtime/list_token.h"
#include "runtime/list_astnode.h"
```

### Variable Declarations
```nano
# nanolang
let mut tokens: list_token = (list_token_new)
```

Generates:
```c
List_token *tokens = list_token_new();
```

### Function Calls
```nano
# nanolang
(list_int_push numbers 42)
```

Generates:
```c
list_int_push(numbers, 42);
```

**No special handling needed - just direct translation!**

---

## Memory Management

### For Value Types (int, float, bool)
```c
// Simple - just copy values
List_int *list = list_int_new();
list_int_push(list, 42);  // Copies value
```

### For Strings
```c
// Need to duplicate strings
void list_string_push(List_string *list, const char *value) {
    // ... grow logic ...
    list->data[list->length++] = strdup(value);
}

void list_string_free(List_string *list) {
    for (int i = 0; i < list->length; i++) {
        free(list->data[i]);
    }
    free(list->data);
    free(list);
}
```

### For Structs (Token, ASTNode)
```c
// Deep copy structs
void list_token_push(List_token *list, Token value) {
    // ... grow logic ...
    list->data[list->length] = value;  // Struct copy
    // If Token contains strings, need to strdup them
    list->data[list->length].value = strdup(value.value);
    list->length++;
}
```

---

## Safety Features

### ✅ Bounds Checking
All get/set/remove operations check bounds:
```c
if (index < 0 || index >= list->length) {
    fprintf(stderr, "Error: index out of bounds\n");
    exit(1);
}
```

### ✅ Automatic Growth
Push automatically grows capacity:
```c
if (list->length == list->capacity) {
    list->capacity *= 2;
    list->data = realloc(list->data, ...);
}
```

### ✅ Type Safety
Each list type only accepts its element type (enforced by C compiler).

---

## Performance

| Operation | Complexity |
|-----------|------------|
| push | O(1) amortized |
| pop | O(1) |
| get | O(1) |
| set | O(1) |
| insert | O(n) |
| remove | O(n) |
| length | O(1) |

**Growth strategy:** Double capacity (8 → 16 → 32 → 64 → ...)

---

## Testing Strategy

### Unit Tests (C)
```c
// tests/unit/test_list_int.c
void test_list_int_basic() {
    List_int *list = list_int_new();
    assert(list_int_length(list) == 0);
    
    list_int_push(list, 42);
    assert(list_int_length(list) == 1);
    assert(list_int_get(list, 0) == 42);
    
    list_int_free(list);
}

void test_list_int_growth() {
    List_int *list = list_int_new();
    
    // Add 100 elements - should grow automatically
    for (int i = 0; i < 100; i++) {
        list_int_push(list, i);
    }
    
    assert(list_int_length(list) == 100);
    assert(list_int_get(list, 50) == 50);
    
    list_int_free(list);
}
```

### Shadow Tests (nanolang)
```nano
fn test_list_int() -> int {
    let mut nums: list_int = (list_int_new)
    
    (list_int_push nums 10)
    (list_int_push nums 20)
    (list_int_push nums 30)
    
    assert (== (list_int_length nums) 3)
    assert (== (list_int_get nums 0) 10)
    assert (== (list_int_get nums 1) 20)
    assert (== (list_int_get nums 2) 30)
    
    return 0
}

shadow test_list_int {
    assert (== (test_list_int) 0)
}
```

---

## Future: Generic Syntax (Optional)

Once self-hosting works, add syntactic sugar:

```nano
# Generic syntax (sugar)
let mut tokens: list<Token> = (list_new)
(list_push tokens tok)

# Transpiler rewrites to:
List_token *tokens = list_token_new();
list_token_push(tokens, tok);
```

**Implementation:** Simple name substitution in transpiler - no complex type system!

---

## Success Criteria

✅ **All 4 list types implemented and tested**
- [ ] list_int works
- [ ] list_string works  
- [ ] list_token works
- [ ] list_astnode works

✅ **Can build token list in lexer**
- [ ] Lexer returns list_token
- [ ] Parser accepts list_token

✅ **All tests pass**
- [ ] C unit tests
- [ ] nanolang shadow tests
- [ ] No memory leaks (valgrind)

✅ **Documentation complete**

---

## Summary

**Approach:** Specialized list implementations (no generics)

**Why:** Simple, fast to implement, sufficient for self-hosting

**Timeline:** 2-3 weeks

**Complexity:** ⭐ Very Low (just C structs and functions)

**Code:** ~600 lines of C (4 × 150 lines)

**Benefit:** Gets us to self-hosting faster without language complexity

**Future:** Can add generic syntax later as syntactic sugar

---

**Status:** ✅ Approved - Ready to Implement  
**Next:** Implement list_int first (after structs complete)  
**Dependencies:** Structs (for list_token and list_astnode)


