# Lists Implementation Guide - Quick Start

**Approach:** Specialized lists only (no generics)  
**Timeline:** 2-3 weeks  
**Complexity:** ⭐ Very Low

## Quick Reference

**4 Types to Implement:**
1. `list_int` - integer lists
2. `list_string` - string lists  
3. `list_token` - Token struct lists
4. `list_astnode` - ASTNode struct lists

**Each type needs:** 12 functions + header file (~150 lines each)

---

## Step-by-Step Implementation

### Step 1: Create Runtime Directory (5 minutes)

```bash
mkdir -p src/runtime
```

### Step 2: Implement list_int (Day 1)

**File:** `src/runtime/list_int.h`
```c
#ifndef LIST_INT_H
#define LIST_INT_H

#include <stdint.h>
#include <stdbool.h>

typedef struct {
    int64_t *data;
    int length;
    int capacity;
} List_int;

List_int* list_int_new(void);
List_int* list_int_with_capacity(int capacity);
void list_int_push(List_int *list, int64_t value);
int64_t list_int_pop(List_int *list);
void list_int_insert(List_int *list, int index, int64_t value);
int64_t list_int_remove(List_int *list, int index);
void list_int_set(List_int *list, int index, int64_t value);
void list_int_clear(List_int *list);
int64_t list_int_get(List_int *list, int index);
int list_int_length(List_int *list);
int list_int_capacity(List_int *list);
bool list_int_is_empty(List_int *list);
void list_int_free(List_int *list);

#endif
```

**File:** `src/runtime/list_int.c` (see LISTS_DESIGN_SIMPLE.md for full code)

### Step 3: Add to Type System (Day 1)

**In `src/nanolang.h`:**
```c
typedef enum {
    // ... existing types ...
    TYPE_LIST_INT,      // ADD
    TYPE_LIST_STRING,   // ADD
    TYPE_LIST_TOKEN,    // ADD
    TYPE_LIST_ASTNODE,  // ADD
    TYPE_UNKNOWN
} Type;
```

**In `src/typechecker.c` (type_to_string):**
```c
case TYPE_LIST_INT: return "list_int";
case TYPE_LIST_STRING: return "list_string";
case TYPE_LIST_TOKEN: return "list_token";
case TYPE_LIST_ASTNODE: return "list_astnode";
```

### Step 4: Add to Parser (Day 2)

**In `src/parser.c` (parse_type):**
```c
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
        // Struct or unknown type
        type = TYPE_STRUCT;
    }
    advance(p);
    return type;
```

### Step 5: Add Built-in Functions (Day 2)

**In `src/env.c` (is_builtin_function):**
```c
// list_int operations
if (strcmp(name, "list_int_new") == 0) return true;
if (strcmp(name, "list_int_push") == 0) return true;
if (strcmp(name, "list_int_get") == 0) return true;
if (strcmp(name, "list_int_length") == 0) return true;
// ... add all 12 functions for each type
```

**In `src/typechecker.c` (check builtin signatures):**
```c
// list_int_new: () -> list_int
if (strcmp(name, "list_int_new") == 0) {
    if (arg_count != 0) {
        fprintf(stderr, "Error: list_int_new expects 0 arguments\n");
        return TYPE_UNKNOWN;
    }
    return TYPE_LIST_INT;
}

// list_int_push: (mut list_int, int) -> void
if (strcmp(name, "list_int_push") == 0) {
    if (arg_count != 2) {
        fprintf(stderr, "Error: list_int_push expects 2 arguments\n");
        return TYPE_UNKNOWN;
    }
    // TODO: Check arg types
    return TYPE_VOID;
}

// list_int_get: (list_int, int) -> int
if (strcmp(name, "list_int_get") == 0) {
    if (arg_count != 2) {
        fprintf(stderr, "Error: list_int_get expects 2 arguments\n");
        return TYPE_UNKNOWN;
    }
    return TYPE_INT;
}
```

### Step 6: Add to Transpiler (Day 3)

**In `src/transpiler.c` (at top of transpile_to_c):**
```c
// Add includes
fprintf(output, "#include \"runtime/list_int.h\"\n");
fprintf(output, "#include \"runtime/list_string.h\"\n");
fprintf(output, "#include \"runtime/list_token.h\"\n");
fprintf(output, "#include \"runtime/list_astnode.h\"\n");
```

**Handle list types in variable declarations:**
```c
case TYPE_LIST_INT:
    fprintf(output, "List_int *%s", name);
    break;
case TYPE_LIST_STRING:
    fprintf(output, "List_string *%s", name);
    break;
// ... etc
```

**Function calls transpile directly (no special handling needed!):**
```nano
(list_int_push nums 42)
```
→
```c
list_int_push(nums, 42);
```

### Step 7: Update Makefile (Day 3)

```makefile
# Add runtime objects
RUNTIME_SOURCES = src/runtime/list_int.c src/runtime/list_string.c \
                  src/runtime/list_token.c src/runtime/list_astnode.c
RUNTIME_OBJECTS = $(patsubst src/%.c,obj/%.o,$(RUNTIME_SOURCES))

# Update compiler dependencies
$(COMPILER): $(COMPILER_OBJECTS) $(RUNTIME_OBJECTS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(COMPILER) $(COMPILER_OBJECTS) $(RUNTIME_OBJECTS)

# Runtime object rule
obj/runtime/%.o: src/runtime/%.c | obj/runtime
	$(CC) $(CFLAGS) -c $< -o $@

obj/runtime:
	mkdir -p obj/runtime
```

### Step 8: Write Tests (Days 4-5)

**C Unit Test:**
```c
// tests/unit/test_list_int.c
#include "../../src/runtime/list_int.h"
#include <assert.h>
#include <stdio.h>

void test_basic() {
    List_int *list = list_int_new();
    assert(list_int_length(list) == 0);
    
    list_int_push(list, 42);
    assert(list_int_length(list) == 1);
    assert(list_int_get(list, 0) == 42);
    
    list_int_free(list);
    printf("✓ Basic test passed\n");
}

int main() {
    test_basic();
    return 0;
}
```

**nanolang Test:**
```nano
# examples/18_list_int_test.nano
fn test_list() -> int {
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

shadow test_list {
    assert (== (test_list) 0)
}

fn main() -> int {
    return (test_list)
}

shadow main {
    assert (== (main) 0)
}
```

### Step 9: Copy for Other Types (Week 2)

**list_string:**
1. Copy list_int.c → list_string.c
2. Replace `int64_t` with `char*`
3. Add `strdup()` in push
4. Add `free()` in list_string_free

**list_token (after structs work):**
1. Copy list_int.c → list_token.c
2. Replace `int64_t` with `Token`
3. Handle struct copying
4. Test with Token structs

**list_astnode:**
1. Copy list_int.c → list_astnode.c
2. Replace `int64_t` with `ASTNode`
3. Handle struct copying
4. Test with ASTNode structs

---

## Testing Checklist

### For Each List Type

- [ ] Create new list (empty)
- [ ] Push elements (single)
- [ ] Push elements (many - test growth)
- [ ] Get elements (valid indices)
- [ ] Get out of bounds (should error)
- [ ] Length is correct
- [ ] Pop elements
- [ ] Clear list
- [ ] Insert at various positions
- [ ] Remove from various positions
- [ ] Set element values
- [ ] Check is_empty
- [ ] Memory leak test (valgrind)

### Integration Tests

- [ ] Use in function parameters
- [ ] Use in return values
- [ ] Pass between functions
- [ ] Nested in structs (later)
- [ ] Multiple lists simultaneously

---

## Common Issues & Solutions

### Issue: Segfault on push
**Cause:** Forgot to allocate data array  
**Fix:** In list_new, ensure `malloc(sizeof(T) * capacity)`

### Issue: Type checker complains about list functions
**Cause:** Not added to is_builtin_function  
**Fix:** Add all 12 functions × 4 types to builtin check

### Issue: Linker error "undefined reference to list_int_new"
**Cause:** Runtime not linked  
**Fix:** Update Makefile to include runtime objects

### Issue: Out of bounds doesn't error
**Cause:** Forgot bounds check  
**Fix:** Add check in get/set/remove functions

---

## Code Size Estimate

| Component | Lines of Code |
|-----------|---------------|
| list_int.c | 150 |
| list_string.c | 150 |
| list_token.c | 150 |
| list_astnode.c | 150 |
| Header files (4) | 80 |
| Parser changes | 20 |
| Type system changes | 30 |
| Transpiler changes | 50 |
| **Total** | **~780 lines** |

**Existing codebase:** ~3,200 lines  
**After lists:** ~4,000 lines  
**Increase:** ~25%

---

## Timeline

### Week 1: Basic Lists
- **Monday:** Implement list_int (C code)
- **Tuesday:** Add to type system and parser
- **Wednesday:** Add to transpiler, test basic example
- **Thursday:** Implement list_string
- **Friday:** Testing and bug fixes

### Week 2: Struct Lists (requires structs done)
- **Monday:** Implement list_token
- **Tuesday:** Implement list_astnode
- **Wednesday:** Integration testing
- **Thursday:** Memory testing (valgrind)
- **Friday:** Bug fixes

### Week 3: Polish
- **Monday:** Write comprehensive examples
- **Tuesday:** Write shadow tests
- **Wednesday:** Negative tests
- **Thursday:** Documentation
- **Friday:** Final review and merge

---

## Success Metrics

✅ **All 4 list types work**  
✅ **Can compile example using each type**  
✅ **All tests pass**  
✅ **No memory leaks**  
✅ **Documentation complete**

---

## Next Steps After Lists

Once lists are done, you have everything needed for the compiler:

```nano
# Can now write lexer!
fn tokenize(source: string) -> list_token {
    let mut tokens: list_token = (list_token_new)
    // ... parse and create tokens ...
    return tokens
}

# Can now write parser!
fn parse(tokens: list_token) -> list_astnode {
    let mut statements: list_astnode = (list_astnode_new)
    // ... parse and create AST nodes ...
    return statements
}
```

**You're ready for Phase 2: Writing the compiler in nanolang!** 🚀

---

**See:** [LISTS_DESIGN_SIMPLE.md](docs/LISTS_DESIGN_SIMPLE.md) for detailed design  
**Status:** Ready to implement (after structs complete)  
**Priority:** #3 for self-hosting


