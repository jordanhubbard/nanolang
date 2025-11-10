# Array Design for nanolang

## Syntax Design (Prefix-Notation Consistent)

### Array Type Declaration
```nano
# Fixed-size array type
array<int>      # Array of integers
array<float>    # Array of floats  
array<string>   # Array of strings
array<bool>     # Array of booleans
```

### Array Creation
```nano
# Array literal
let nums: array<int> = [1, 2, 3, 4, 5]

# Empty array with size
let arr: array<int> = (array_new 10 0)  # 10 elements, initialized to 0

# From existing values
let values: array<float> = [3.14, 2.71, 1.41]
```

### Array Operations (Prefix Notation)
```nano
# Access element (at = array access)
let x: int = (at nums 0)          # Get first element
let y: int = (at nums 2)          # Get third element

# Set element (array must be mutable)
let mut arr: array<int> = [1, 2, 3]
(array_set arr 0 42)               # Set first element to 42

# Get length
let len: int = (array_length nums)

# Note: Keeping prefix notation for consistency!
```

### Array in Functions
```nano
fn sum_array(arr: array<int>) -> int {
    let mut total: int = 0
    let mut i: int = 0
    let len: int = (array_length arr)
    
    while (< i len) {
        set total (+ total (at arr i))
        set i (+ i 1)
    }
    
    return total
}

shadow sum_array {
    assert (== (sum_array [1, 2, 3]) 6)
    assert (== (sum_array [0]) 0)
    assert (== (sum_array [10, 20, 30]) 60)
}
```

## Implementation Plan

### Phase 1: Core Array Support
1. Add `TOKEN_LBRACKET`, `TOKEN_RBRACKET`, `TOKEN_ARRAY` to lexer
2. Add `TOKEN_LT`, `TOKEN_GT` for generic syntax (or use existing)
3. Add `TYPE_ARRAY` to type system
4. Add `AST_ARRAY_LITERAL`, `AST_ARRAY_ACCESS` to AST
5. Parser support for array literals and types
6. Type checker for array operations

### Phase 2: Array Operations
1. `array_new(size, default)` - Create array
2. `at(array, index)` - Access element (prefix)
3. `array_set(array, index, value)` - Set element
4. `array_length(array)` - Get length

### Phase 3: Advanced (Future)
1. `array_append(array, value)` - Append element
2. `array_slice(array, start, end)` - Get sub-array
3. `array_concat(arr1, arr2)` - Concatenate arrays
4. `array_map(array, fn)` - Map function over array
5. `array_filter(array, fn)` - Filter array

## C Transpilation Strategy

```c
// Array type in C
typedef struct {
    int64_t length;
    int64_t capacity;
    void* data;
    size_t element_size;
} nl_array;

// Operations
nl_array* nl_array_new(int64_t size, int64_t element_size);
void* nl_array_at(nl_array* arr, int64_t index);
void nl_array_set(nl_array* arr, int64_t index, void* value);
int64_t nl_array_length(nl_array* arr);
```

## Type System Changes

```c
typedef enum {
    TYPE_INT,
    TYPE_FLOAT,
    TYPE_BOOL,
    TYPE_STRING,
    TYPE_VOID,
    TYPE_ARRAY,    // NEW
    TYPE_UNKNOWN
} Type;

typedef struct TypeInfo {
    Type base_type;
    struct TypeInfo* element_type;  // For arrays
} TypeInfo;
```

## Considerations

### Memory Management
- Arrays allocated on heap
- Need proper cleanup (ref counting or GC)
- For now: manual memory management (arrays live until end of scope)

### Bounds Checking
- Always check array bounds at runtime
- Error message with line/column on out-of-bounds access

### Immutability
- Array binding immutable by default
- Elements can be changed if array is `mut`
- Following existing mut/immut pattern

## Example Programs

### Array Statistics
```nano
fn array_min(arr: array<int>) -> int {
    let len: int = (array_length arr)
    let mut min_val: int = (at arr 0)
    let mut i: int = 1
    
    while (< i len) {
        let val: int = (at arr i)
        if (< val min_val) {
            set min_val val
        }
        set i (+ i 1)
    }
    
    return min_val
}

fn array_max(arr: array<int>) -> int {
    let len: int = (array_length arr)
    let mut max_val: int = (at arr 0)
    let mut i: int = 1
    
    while (< i len) {
        let val: int = (at arr i)
        if (> val max_val) {
            set max_val val
        }
        set i (+ i 1)
    }
    
    return max_val
}

fn array_average(arr: array<float>) -> float {
    let len: int = (array_length arr)
    let mut sum: float = 0.0
    let mut i: int = 0
    
    while (< i len) {
        set sum (+ sum (at arr i))
        set i (+ i 1)
    }
    
    return (/ sum len)
}
```

## Timeline

- **Phase 1 (Core):** ~2-3 hours implementation
- **Phase 2 (Operations):** ~1 hour
- **Phase 3 (Advanced):** Future enhancement

## Decision: Start with String & Math First

Arrays are complex and require significant type system changes. Let's implement:
1. **More math functions** (30 minutes - easy stdlib additions)
2. **String operations** (1 hour - stdlib additions)
3. **Then arrays** (2-3 hours - type system changes)

This gives us incremental progress and quick wins!

