# Dynamic Lists Design for nanolang

**Status:** Design Phase  
**Priority:** #3 for Self-Hosting (after structs, enums)  
**Principles:** Safety, Bounds Checking, No Manual Memory Management

## Overview

Dynamic lists (`list<T>`) are resizable collections. Unlike fixed-size arrays, lists grow automatically. Essential for storing variable-length collections of tokens, AST nodes, etc.

## Design Philosophy

**Safety First:**
- Bounds checking on all access
- No buffer overflows
- No manual memory management
- Automatic growth

**Simple API:**
- Small set of operations
- Clear semantics
- No surprises

**Value Semantics:**
- Lists are values (like structs)
- Copying creates new list
- No shared references (no pointers!)

## Syntax & API

### Creating Lists

```nano
# Create empty list
let mut tokens: list<Token> = (list_new)

# Initial capacity (optional optimization, not required)
let mut numbers: list<int> = (list_with_capacity 100)
```

---

### Adding Elements

```nano
# Append to end (O(1) amortized)
(list_push my_list item)

# Insert at index (O(n))
(list_insert my_list index item)
```

**Rules:**
- List must be mutable
- `list_push` grows list automatically
- `list_insert` shifts elements right
- Index must be valid (0 <= index <= length)

---

### Accessing Elements

```nano
# Get element at index (O(1))
let item: T = (list_get my_list index)

# Get last element (O(1))
let last: T = (list_last my_list)

# Get first element (O(1))
let first: T = (list_first my_list)
```

**Rules:**
- Index must be in bounds (0 <= index < length)
- Runtime error if out of bounds
- Empty list error for first/last

---

### Modifying Elements

```nano
# Set element at index (O(1))
(list_set my_list index new_value)
```

**Rules:**
- List must be mutable
- Index must be in bounds
- Type checking ensures value matches element type

---

### Removing Elements

```nano
# Remove last element (O(1))
(list_pop my_list)

# Remove at index (O(n))
(list_remove my_list index)

# Remove all elements (O(1))
(list_clear my_list)
```

**Rules:**
- List must be mutable
- Index must be in bounds
- `pop` on empty list is error

---

### Query Operations

```nano
# Get length (O(1))
let len: int = (list_length my_list)

# Get capacity (O(1))
let cap: int = (list_capacity my_list)

# Check if empty (O(1))
let is_empty: bool = (list_is_empty my_list)
```

**Rules:**
- Works on both mutable and immutable lists
- Always safe (no bounds checking needed)

---

## Complete API Reference

| Function | Signature | Mutates | Description |
|----------|-----------|---------|-------------|
| `list_new` | `() -> list<T>` | - | Create empty list |
| `list_with_capacity` | `(int) -> list<T>` | - | Create with capacity |
| `list_push` | `(mut list<T>, T) -> void` | Yes | Append element |
| `list_pop` | `(mut list<T>) -> T` | Yes | Remove and return last |
| `list_insert` | `(mut list<T>, int, T) -> void` | Yes | Insert at index |
| `list_remove` | `(mut list<T>, int) -> T` | Yes | Remove at index |
| `list_get` | `(list<T>, int) -> T` | No | Get element |
| `list_set` | `(mut list<T>, int, T) -> void` | Yes | Set element |
| `list_first` | `(list<T>) -> T` | No | Get first element |
| `list_last` | `(list<T>) -> T` | No | Get last element |
| `list_length` | `(list<T>) -> int` | No | Get length |
| `list_capacity` | `(list<T>) -> int` | No | Get capacity |
| `list_is_empty` | `(list<T>) -> bool` | No | Check if empty |
| `list_clear` | `(mut list<T>) -> void` | Yes | Remove all elements |

---

## Examples

### Example 1: Storing Tokens

```nano
struct Token {
    type: int,
    value: string,
    line: int
}

fn tokenize(source: string) -> list<Token> {
    let mut tokens: list<Token> = (list_new)
    
    # Parse source and create tokens
    let tok1: Token = Token { type: 0, value: "42", line: 1 }
    let tok2: Token = Token { type: 1, value: "+", line: 1 }
    
    (list_push tokens tok1)
    (list_push tokens tok2)
    
    return tokens
}

shadow tokenize {
    let tokens: list<Token> = (tokenize "42 + 10")
    assert (== (list_length tokens) 2)
    
    let first: Token = (list_get tokens 0)
    assert (== first.type 0)
    assert (str_equals first.value "42")
}
```

---

### Example 2: Processing Numbers

```nano
fn sum_list(numbers: list<int>) -> int {
    let mut total: int = 0
    let mut i: int = 0
    let len: int = (list_length numbers)
    
    while (< i len) {
        let num: int = (list_get numbers i)
        set total (+ total num)
        set i (+ i 1)
    }
    
    return total
}

shadow sum_list {
    let mut nums: list<int> = (list_new)
    (list_push nums 1)
    (list_push nums 2)
    (list_push nums 3)
    (list_push nums 4)
    (list_push nums 5)
    
    assert (== (sum_list nums) 15)
    
    # Empty list
    let empty: list<int> = (list_new)
    assert (== (sum_list empty) 0)
}

fn filter_positive(numbers: list<int>) -> list<int> {
    let mut result: list<int> = (list_new)
    let mut i: int = 0
    let len: int = (list_length numbers)
    
    while (< i len) {
        let num: int = (list_get numbers i)
        if (> num 0) {
            (list_push result num)
        }
        set i (+ i 1)
    }
    
    return result
}

shadow filter_positive {
    let mut nums: list<int> = (list_new)
    (list_push nums -2)
    (list_push nums 3)
    (list_push nums -1)
    (list_push nums 5)
    
    let positive: list<int> = (filter_positive nums)
    assert (== (list_length positive) 2)
    assert (== (list_get positive 0) 3)
    assert (== (list_get positive 1) 5)
}
```

---

### Example 3: Building and Modifying

```nano
fn build_sequence(n: int) -> list<int> {
    let mut seq: list<int> = (list_new)
    let mut i: int = 0
    
    while (< i n) {
        (list_push seq i)
        set i (+ i 1)
    }
    
    return seq
}

shadow build_sequence {
    let seq: list<int> = (build_sequence 5)
    assert (== (list_length seq) 5)
    assert (== (list_get seq 0) 0)
    assert (== (list_get seq 4) 4)
}

fn double_all(numbers: mut list<int>) -> void {
    let mut i: int = 0
    let len: int = (list_length numbers)
    
    while (< i len) {
        let num: int = (list_get numbers i)
        (list_set numbers i (* num 2))
        set i (+ i 1)
    }
}

shadow double_all {
    let mut nums: list<int> = (build_sequence 5)
    (double_all nums)
    
    assert (== (list_get nums 0) 0)
    assert (== (list_get nums 1) 2)
    assert (== (list_get nums 2) 4)
    assert (== (list_get nums 3) 6)
    assert (== (list_get nums 4) 8)
}
```

---

## Implementation Strategy

### Approach: Generic vs Specialized

**Option 1: True Generics (Complex)**
- Single `list<T>` implementation
- Works with any type
- Requires full generics system
- Much more work

**Option 2: Specialized Versions (Simpler)**
- `list_int`, `list_float`, `list_string`, etc.
- Separate implementation for each type
- More code, but simpler
- No generics needed yet

**Option 3: Hybrid (Recommended)**
- Generic syntax: `list<Token>`
- Monomorphization: Generate specialized version for each concrete type
- Best of both worlds
- Can add true generics later

**Decision: Option 3 (Hybrid)**

---

## Implementation Plan

### Phase 1: Type System (Week 1)

**Add list type:**
```c
typedef enum {
    TYPE_INT,
    TYPE_FLOAT,
    // ... existing ...
    TYPE_LIST,  // NEW
} Type;

// Extended type info
typedef struct TypeInfo {
    Type base_type;
    struct TypeInfo *element_type;  // For list<T>
    char *struct_name;
} TypeInfo;
```

**Tasks:**
- [ ] Add TYPE_LIST to type system
- [ ] Track element type in TypeInfo
- [ ] Type checking for list operations
- [ ] Write type tests

---

### Phase 2: Runtime Implementation (Weeks 1-2)

**C implementation:**
```c
// Generic list structure (in generated C)
typedef struct {
    void *data;       // Array of elements
    int length;       // Current length
    int capacity;     // Allocated capacity
    size_t elem_size; // Size of each element
} List;

// Create new list
List* list_new_int(void);
List* list_new_Token(void);
// ... specialized for each type

// Operations
void list_push_int(List *list, int64_t value);
int64_t list_get_int(List *list, int index);
// ... specialized for each type
```

**Tasks:**
- [ ] Implement list data structure in C
- [ ] Implement growth algorithm (double capacity)
- [ ] Implement bounds checking
- [ ] Memory management (malloc/realloc/free)
- [ ] Write C unit tests

---

### Phase 3: Parser (Week 2)

**Parse list type syntax:**
```nano
let mut tokens: list<Token> = (list_new)
```

**Tasks:**
- [ ] Parse `list<Type>` syntax
- [ ] Handle in variable declarations
- [ ] Handle in function parameters
- [ ] Handle in return types
- [ ] Write parser tests

---

### Phase 4: Type Checker (Week 3)

**Tasks:**
- [ ] Check list type annotations
- [ ] Check list operations (push, get, set, etc.)
- [ ] Verify element types match
- [ ] Check mutability requirements
- [ ] Handle in function signatures
- [ ] Write type tests

**Error messages:**
```
Error: list_push requires mutable list
Error: Type mismatch: list<int> cannot store string
Error: list_get index out of bounds (got 10, length is 5)
Error: Cannot use list<Token> as list<int>
```

---

### Phase 5: Interpreter (Week 3)

**Value representation:**
```c
typedef struct {
    ValueType type;
    bool is_return;
    union {
        // ... existing ...
        List *list_val;  // NEW
    } as;
} Value;
```

**Tasks:**
- [ ] Evaluate list operations
- [ ] Track lists in interpreter
- [ ] Memory management
- [ ] Write interpreter tests

---

### Phase 6: C Transpiler (Weeks 4-5)

**Monomorphization:**

```nano
# nanolang
let mut tokens: list<Token> = (list_new)
(list_push tokens tok)
```

Generates:

```c
// Specialized list for Token
typedef struct {
    Token *data;
    int length;
    int capacity;
} List_Token;

List_Token* list_new_Token() {
    List_Token *list = malloc(sizeof(List_Token));
    list->data = malloc(sizeof(Token) * 8);  // Initial capacity
    list->length = 0;
    list->capacity = 8;
    return list;
}

void list_push_Token(List_Token *list, Token value) {
    if (list->length == list->capacity) {
        list->capacity *= 2;
        list->data = realloc(list->data, sizeof(Token) * list->capacity);
    }
    list->data[list->length++] = value;
}

// Usage
List_Token *tokens = list_new_Token();
list_push_Token(tokens, tok);
```

**Tasks:**
- [ ] Generate specialized list types
- [ ] Generate specialized functions
- [ ] Handle in variable declarations
- [ ] Handle list operations
- [ ] Memory management (free lists)
- [ ] Write transpiler tests
- [ ] Test generated C compiles and runs

---

### Phase 7: Testing & Documentation (Week 6)

**Test files:**
- [ ] `tests/unit/list_basic.nano` - Basic operations
- [ ] `tests/unit/list_growth.nano` - Automatic growth
- [ ] `tests/unit/list_bounds.nano` - Bounds checking
- [ ] `tests/unit/list_types.nano` - Type checking
- [ ] `tests/negative/list_errors/` - Error cases
- [ ] `examples/19_lists.nano` - Example program

**Documentation:**
- [ ] Update SPECIFICATION.md
- [ ] Update STDLIB.md (list operations)
- [ ] Update GETTING_STARTED.md
- [ ] Update IMPLEMENTATION_STATUS.md

---

## Safety Guarantees

### ✅ Bounds Checking

```nano
let mut nums: list<int> = (list_new)
(list_push nums 42)

let x: int = (list_get nums 0)   # OK
let y: int = (list_get nums 10)  # ERROR: index 10 out of bounds (length 1)
```

**Runtime check:**
```c
int64_t list_get_int(List_int *list, int index) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: list index %d out of bounds (length %d)\n", 
                index, list->length);
        exit(1);
    }
    return list->data[index];
}
```

---

### ✅ Type Safety

```nano
let mut nums: list<int> = (list_new)
(list_push nums 42)   # OK
# (list_push nums "hello")  # ERROR: type mismatch
```

**Compile-time check in type checker.**

---

### ✅ No Buffer Overflows

```nano
let mut nums: list<int> = (list_new)

# Add 1000 elements - automatic growth!
let mut i: int = 0
while (< i 1000) {
    (list_push nums i)
    set i (+ i 1)
}

# Never overflows, always safe
```

**Automatic realloc in list_push.**

---

### ✅ Memory Safety

**No manual memory management:**
- Lists allocated automatically
- Growth handled automatically
- Deallocation handled by C runtime (initially)
- Future: Add garbage collection or ref counting

---

## Value Semantics

### Copying Creates New List

```nano
let mut list1: list<int> = (list_new)
(list_push list1 42)

let list2: list<int> = list1  # DEEP COPY

# Modifying list2 doesn't affect list1
(list_push list2 100)

assert (== (list_length list1) 1)  # Still 1
assert (== (list_length list2) 2)  # Now 2
```

**Note:** Deep copying can be expensive. Document this clearly. Future optimization: copy-on-write.

---

## Performance Characteristics

| Operation | Time Complexity | Notes |
|-----------|----------------|-------|
| `list_new` | O(1) | Allocates small initial capacity |
| `list_push` | O(1) amortized | Doubles capacity when full |
| `list_pop` | O(1) | No reallocation |
| `list_get` | O(1) | Direct array access |
| `list_set` | O(1) | Direct array access |
| `list_insert` | O(n) | Shifts elements |
| `list_remove` | O(n) | Shifts elements |
| `list_length` | O(1) | Stored field |
| Copy | O(n) | Deep copy all elements |

**Growth strategy:** Double capacity when full (common approach)
- Initial capacity: 8 elements
- Growth: 8 → 16 → 32 → 64 → 128 → ...
- Amortized O(1) push

---

## Timeline

**Total Time:** 4-6 weeks (after structs + enums complete)

- **Week 1:** Type system + C runtime start
- **Week 2:** C runtime complete + Parser
- **Week 3:** Type checker + Interpreter
- **Weeks 4-5:** C Transpiler (monomorphization)
- **Week 6:** Testing + Documentation

---

## Success Criteria

✅ All tests pass:
- [ ] Create and use lists
- [ ] Automatic growth
- [ ] Bounds checking
- [ ] Type checking
- [ ] Various element types (int, struct, etc.)

✅ Can store tokens:
- [ ] `list<Token>` works
- [ ] Can build token list in lexer
- [ ] Ready for compiler implementation

✅ Documentation complete

---

## Dependencies

**Requires:**
- Structs (to store structs in lists)
- Enums (optional, but useful)

**Unlocks:**
- Lexer (store tokens)
- Parser (store AST nodes)
- Type checker (store symbols)

---

**Status:** Ready to implement (after structs + enums)  
**Priority:** #3  
**Estimated Time:** 4-6 weeks  
**Dependencies:** Structs, Enums

