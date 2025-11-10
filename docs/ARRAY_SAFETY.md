# Array Safety and Verifiability in nanolang

**Design Philosophy:** Arrays must be safe by construction and verifiable through shadow tests.

---

## Core Safety Principles

### 1. **Always Bounds-Checked** (Runtime Safety)

**Every array access is checked:**
```nano
let nums: array<int> = [1, 2, 3, 4, 5]

# Safe access - returns value
let x: int = (at nums 2)  # Returns 3

# Out of bounds - compile-time error OR runtime panic
let y: int = (at nums 10)  # ERROR: index 10 out of bounds [0..5)
```

**Implementation:**
```c
// In transpiled C code
int64_t nl_array_at(nl_array* arr, int64_t index) {
    if (index < 0 || index >= arr->length) {
        fprintf(stderr, "Runtime Error: Array index %lld out of bounds [0..%lld)\n", 
                index, arr->length);
        exit(1);  // Fail fast - no undefined behavior
    }
    return ((int64_t*)arr->data)[index];
}
```

**Why this is safe:**
- ✅ No buffer overflows possible
- ✅ No undefined behavior
- ✅ Fail fast with clear error messages
- ✅ Line/column numbers in error (from AST)

### 2. **Type-Safe by Construction** (Compile-Time Safety)

**Homogeneous arrays only:**
```nano
# Valid - all elements same type
let nums: array<int> = [1, 2, 3, 4, 5]
let names: array<string> = ["Alice", "Bob", "Carol"]

# INVALID - mixed types rejected at compile time
let mixed: array<int> = [1, "hello", 3.14]  # TYPE ERROR
```

**Type checker validates:**
1. Array literal elements match declared type
2. All operations preserve type safety
3. No implicit conversions

**Why this is safe:**
- ✅ No type confusion
- ✅ LLM can always know exact types
- ✅ Compiler catches type errors early

### 3. **Immutable by Default** (Memory Safety)

**Arrays are immutable unless marked `mut`:**
```nano
# Immutable array - cannot change after creation
let nums: array<int> = [1, 2, 3]
(array_set nums 0 99)  # COMPILE ERROR: nums is immutable

# Mutable array - can be modified
let mut counts: array<int> = [0, 0, 0]
(array_set counts 0 99)  # OK - counts is mutable
```

**Why this is safe:**
- ✅ No unexpected mutations
- ✅ Easier to reason about
- ✅ LLM can track state changes
- ✅ Functional programming benefits

### 4. **Fixed-Size Arrays** (Predictable Memory)

**Array size is known at creation:**
```nano
# Size known from literal
let nums: array<int> = [1, 2, 3, 4, 5]  # Length = 5

# Size specified explicitly
let zeros: array<int> = (array_new 10 0)  # Length = 10, all zeros

# Size cannot change after creation
let len: int = (array_length nums)  # Always returns 5
```

**Why this is safe:**
- ✅ No dynamic resizing surprises
- ✅ Memory usage predictable
- ✅ No reallocation bugs
- ✅ LLM can track size statically

**Future enhancement (Phase 4):**
```nano
# Dynamic arrays with explicit operations
let mut vec: vector<int> = (vector_new)
(vector_push vec 42)  # Explicit growth
(vector_pop vec)      # Explicit shrinkage
```

### 5. **Shadow-Tested Operations** (Verification)

**Every array function has mandatory tests:**
```nano
fn array_sum(arr: array<int>) -> int {
    let mut total: int = 0
    let mut i: int = 0
    let len: int = (array_length arr)
    
    while (< i len) {
        set total (+ total (at arr i))
        set i (+ i 1)
    }
    
    return total
}

# MANDATORY shadow test
shadow array_sum {
    # Test normal case
    assert (== (array_sum [1, 2, 3, 4, 5]) 15)
    
    # Test edge cases
    assert (== (array_sum [0]) 0)
    assert (== (array_sum [-5, 5]) 0)
    
    # Test empty array (if supported)
    assert (== (array_sum []) 0)
}
```

**Why this is verifiable:**
- ✅ Every function behavior is documented
- ✅ Edge cases are tested
- ✅ LLM can see expected behavior
- ✅ Tests execute at compile time

---

## Safety Guarantees

### What nanolang Arrays GUARANTEE:

1. **No Buffer Overflows**
   - Every access is bounds-checked
   - Out-of-bounds = immediate error with diagnostics
   - No silent corruption

2. **No Type Confusion**
   - Homogeneous types enforced
   - Type checker validates at compile time
   - No `void*` or `any` type arrays

3. **No Null/Undefined**
   - Arrays cannot be null
   - Elements cannot be undefined
   - Must provide default values

4. **No Memory Corruption**
   - Fixed-size prevents reallocation bugs
   - Immutability prevents race conditions
   - Clear ownership semantics

5. **Predictable Behavior**
   - No implicit conversions
   - No operator overloading
   - Explicit operations only

---

## Verifiability Through Shadow Tests

### Example: Verified Array Operations

```nano
fn array_max(arr: array<int>) -> int {
    # Requires: array length > 0
    let len: int = (array_length arr)
    assert (> len 0)  # Precondition check
    
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

shadow array_max {
    # Property: max is >= all elements
    assert (== (array_max [5, 2, 8, 1, 9, 3]) 9)
    assert (== (array_max [1]) 1)
    assert (== (array_max [-10, -5, -20]) -5)
    
    # Boundary values
    assert (== (array_max [2147483647]) 2147483647)  # INT_MAX
    assert (== (array_max [-2147483648]) -2147483648)  # INT_MIN
}
```

### LLM-Verifiable Properties

**The LLM can verify:**

1. **Preconditions** - What must be true before calling
   ```nano
   assert (> len 0)  # Array must not be empty
   ```

2. **Postconditions** - What is guaranteed after calling
   ```nano
   # Result is in the array
   # Result is >= all elements
   ```

3. **Invariants** - What remains true during execution
   ```nano
   # i is always in bounds [0..len)
   # max_val is always the max of arr[0..i]
   ```

4. **Edge Cases** - Boundary conditions tested
   ```nano
   # Single element
   # Negative numbers
   # Maximum/minimum values
   ```

---

## Advanced Safety Features

### Option 1: Compile-Time Length Tracking

**Track array lengths in type system:**
```nano
# Future enhancement
let nums: array<int, 5> = [1, 2, 3, 4, 5]  # Length known at compile time

fn get_third(arr: array<int, 5>) -> int {
    return (at arr 2)  # Compiler knows 2 < 5, no runtime check needed!
}
```

**Benefits:**
- ✅ Some bounds checks can be eliminated
- ✅ More errors caught at compile time
- ✅ Better optimization opportunities

**Drawbacks:**
- ⚠️ More complex type system
- ⚠️ Less flexible
- ⚠️ Harder to implement

**Verdict:** Phase 4 feature - keep simple for now

### Option 2: Range Types for Indices

**Restrict indices to valid range:**
```nano
# Future enhancement
type Index5 = range<0, 5>  # Valid indices: 0, 1, 2, 3, 4

fn safe_get(arr: array<int, 5>, i: Index5) -> int {
    return (at arr i)  # Type system guarantees i is in bounds!
}
```

**Benefits:**
- ✅ Compile-time bounds checking
- ✅ No runtime overhead for validated code
- ✅ Proof of correctness

**Drawbacks:**
- ⚠️ Complex type system
- ⚠️ Harder for LLM to generate
- ⚠️ Runtime checks still needed at API boundaries

**Verdict:** Interesting for future, overkill for v1

### Option 3: Contracts and Assertions

**Already supported through shadow tests!**
```nano
fn binary_search(arr: array<int>, target: int) -> int {
    # Contract: array must be sorted
    let len: int = (array_length arr)
    let mut i: int = 1
    while (< i len) {
        assert (<= (at arr (- i 1)) (at arr i))  # Verify sorted
        set i (+ i 1)
    }
    
    # ... binary search implementation
}

shadow binary_search {
    # Tests verify contract
    assert (== (binary_search [1, 3, 5, 7, 9] 5) 2)
    # Test will fail if array not sorted
}
```

---

## Comparison with Other Languages

### nanolang vs C Arrays

| Feature | C | nanolang |
|---------|---|----------|
| Bounds checking | ❌ None | ✅ Always |
| Type safety | ⚠️ Weak | ✅ Strong |
| Null safety | ❌ Pointers can be NULL | ✅ No null arrays |
| Memory safety | ❌ Manual | ✅ Managed |
| Verifiable | ❌ No | ✅ Shadow tests |

### nanolang vs Python Lists

| Feature | Python | nanolang |
|---------|--------|----------|
| Bounds checking | ✅ Yes | ✅ Yes |
| Type safety | ❌ Dynamic | ✅ Static |
| Immutability | ⚠️ Optional | ✅ Default |
| Performance | ⚠️ Interpreted | ✅ Native |
| Verifiable | ⚠️ Limited | ✅ Shadow tests |

### nanolang vs Rust Vec

| Feature | Rust | nanolang |
|---------|------|----------|
| Bounds checking | ✅ Yes | ✅ Yes |
| Type safety | ✅ Strong | ✅ Strong |
| Memory safety | ✅ Ownership | ✅ Managed |
| Complexity | ⚠️ High | ✅ Simple |
| LLM-friendly | ⚠️ Moderate | ✅ Very |

**nanolang goal:** Rust-level safety with Python-level simplicity!

---

## Implementation Strategy

### Phase 1: Basic Safe Arrays (Next)

**Must have:**
1. ✅ Runtime bounds checking (always)
2. ✅ Type-safe array<T> 
3. ✅ Immutable by default
4. ✅ Fixed size
5. ✅ Shadow tests for all operations

**Operations:**
```nano
# Creation
let arr: array<int> = [1, 2, 3]
let arr2: array<int> = (array_new 10 0)

# Access (bounds-checked)
let x: int = (at arr 0)

# Length
let len: int = (array_length arr)

# Mutation (if mutable)
let mut arr3: array<int> = [1, 2, 3]
(array_set arr3 0 99)
```

### Phase 2: Array Utilities (Later)

**Additional operations:**
```nano
# Copying
let arr2: array<int> = (array_copy arr)

# Slicing (creates new array)
let slice: array<int> = (array_slice arr 1 3)

# Searching
let found: bool = (array_contains arr 42)
let idx: int = (array_find arr 42)  # Returns -1 if not found

# Functional operations
let doubled: array<int> = (array_map arr (fn (x: int) -> int { return (* x 2) }))
let evens: array<int> = (array_filter arr (fn (x: int) -> bool { return (== (% x 2) 0) }))
```

### Phase 3: Advanced Features (Future)

**Dynamic arrays (vectors):**
```nano
let mut vec: vector<int> = (vector_new)
(vector_push vec 42)
(vector_push vec 43)
let x: int = (vector_pop vec)
```

**Multi-dimensional arrays:**
```nano
let matrix: array<array<int>> = [[1, 2], [3, 4]]
let cell: int = (at (at matrix 0) 1)  # matrix[0][1] = 2
```

---

## Safety Checklist

Before implementing arrays, verify:

- [ ] ✅ **Bounds checking implemented** in evaluator
- [ ] ✅ **Bounds checking implemented** in transpiler
- [ ] ✅ **Type safety enforced** by type checker
- [ ] ✅ **Immutability enforced** by type checker
- [ ] ✅ **Shadow tests** for all array operations
- [ ] ✅ **Error messages** include line/column numbers
- [ ] ✅ **Memory management** is safe (no leaks)
- [ ] ✅ **Edge cases tested** (empty arrays, single element, etc.)
- [ ] ✅ **Documentation** includes safety guarantees
- [ ] ✅ **LLM can understand** all array operations

---

## Example: Fully Verified Array Function

```nano
fn array_reverse(arr: array<int>) -> array<int> {
    let len: int = (array_length arr)
    let mut result: array<int> = (array_new len 0)
    let mut i: int = 0
    
    while (< i len) {
        (array_set result i (at arr (- (- len i) 1)))
        set i (+ i 1)
    }
    
    return result
}

shadow array_reverse {
    # Property: reverse(reverse(arr)) == arr
    let arr: array<int> = [1, 2, 3, 4, 5]
    let rev1: array<int> = (array_reverse arr)
    let rev2: array<int> = (array_reverse rev1)
    
    assert (== (array_length rev2) 5)
    assert (== (at rev2 0) 1)
    assert (== (at rev2 1) 2)
    assert (== (at rev2 2) 3)
    assert (== (at rev2 3) 4)
    assert (== (at rev2 4) 5)
    
    # Edge cases
    assert (== (array_length (array_reverse [])) 0)
    assert (== (at (array_reverse [42]) 0) 42)
    
    # Original unchanged (immutability)
    assert (== (at arr 0) 1)
}
```

**LLM can verify:**
- ✅ Input array is not modified (immutability)
- ✅ Output array has same length
- ✅ Elements are in reverse order
- ✅ Edge cases are handled
- ✅ No bounds violations possible
- ✅ Double reverse returns original

---

## Conclusion

**nanolang arrays will be:**

1. **Safe by Construction**
   - Always bounds-checked
   - Type-safe
   - Immutable by default
   - No null/undefined

2. **Verifiable Through Tests**
   - Mandatory shadow tests
   - Properties documented
   - Edge cases covered
   - Compile-time validation

3. **LLM-Friendly**
   - Clear semantics
   - No hidden behavior
   - Predictable operations
   - Complete introspection

4. **Production-Ready**
   - No undefined behavior
   - Clear error messages
   - Memory safe
   - Performance competitive

**Design Philosophy:** 
> "Make the right thing easy and the wrong thing impossible."

Arrays in nanolang will be **provably safe** and **trivially verifiable** by both humans and LLMs! 🛡️✅

---

**Next Steps:**
1. Implement basic safe arrays (Phase 1)
2. Add comprehensive shadow tests
3. Benchmark performance vs C/Python
4. Document safety guarantees
5. Get LLM to generate array code and verify correctness

**Status:** Design complete, ready for implementation

