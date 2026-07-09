# Array Safety and Verifiability

I ensure arrays are safe by construction and verifiable through shadow tests.

---

## Core Safety Principles

### 1. Always Bounds-Checked

I check every array access. You cannot reach outside the memory I allocated.

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

**Why I am safe:**
- No buffer overflows possible.
- No undefined behavior.
- I fail fast with clear error messages.
- I provide line and column numbers in errors from my AST.

### 2. Type-Safe by Construction

I only allow homogeneous arrays. Every element must be the same type.

```nano
# Valid - all elements same type
let nums: array<int> = [1, 2, 3, 4, 5]
let names: array<string> = ["Alice", "Bob", "Carol"]

# INVALID - mixed types rejected at compile time
let mixed: array<int> = [1, "hello", 3.14]  # TYPE ERROR
```

**My type checker validates:**
1. Array literal elements match the declared type.
2. All operations preserve type safety.
3. No implicit conversions.

**Why I am safe:**
- No type confusion.
- An LLM reading my code knows the exact types.
- I catch type errors early during compilation.

### 3. Immutable by Default

My arrays are immutable unless you explicitly mark them `mut`.

```nano
# Immutable array - cannot change after creation
let nums: array<int> = [1, 2, 3]
(array_set nums 0 99)  # COMPILE ERROR: nums is immutable

# Mutable array - can be modified
let mut counts: array<int> = [0, 0, 0]
(array_set counts 0 99)  # OK - counts is mutable
```

**Why I am safe:**
- No unexpected mutations.
- Easier to reason about state.
- An LLM can track my state changes reliably.
- I provide functional programming benefits.

### 4. Fixed-Size Arrays

I require array sizes to be known at creation.

```nano
# Size known from literal
let nums: array<int> = [1, 2, 3, 4, 5]  # Length = 5

# Size specified explicitly
let zeros: array<int> = (array_new 10 0)  # Length = 10, all zeros

# Size cannot change after creation
let len: int = (array_length nums)  # Always returns 5
```

**Why I am safe:**
- No dynamic resizing surprises.
- Memory usage is predictable.
- No reallocation bugs.
- An LLM can track my size statically.

**Future enhancement (Phase 4):**
```nano
# Dynamic arrays with explicit operations
let mut vec: vector<int> = (vector_new)
(vector_push vec 42)  # Explicit growth
(vector_pop vec)      # Explicit shrinkage
```

### 5. Shadow-Tested Operations

I refuse to compile an array function unless you provide a shadow test.

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

**Why I am verifiable:**
- Every behavior is documented.
- Edge cases are tested.
- An LLM can see my expected behavior.
- Tests execute during compilation.

---

## Safety Guarantees

### What I GUARANTEE:

1. **No Buffer Overflows**
   - I check every access against bounds.
   - Out-of-bounds access results in an immediate error with diagnostics.
   - I do not allow silent corruption.

2. **No Type Confusion**
   - I enforce homogeneous types.
   - My type checker validates types at compile time.
   - I have no `void*` or `any` type arrays.

3. **No Null or Undefined**
   - My arrays cannot be null.
   - My elements cannot be undefined.
   - You must provide default values.

4. **No Memory Corruption**
   - My fixed sizes prevent reallocation bugs.
   - My default immutability prevents race conditions.
   - I have clear ownership semantics.

5. **Predictable Behavior**
   - I perform no implicit conversions.
   - I do not use operator overloading.
   - I only allow explicit operations.

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

An LLM can verify my properties:

1. **Preconditions** - What must be true before calling.
   ```nano
   assert (> len 0)  # Array must not be empty
   ```

2. **Postconditions** - What I guarantee after calling.
   ```nano
   # Result is in the array
   # Result is >= all elements
   ```

3. **Invariants** - What remains true during execution.
   ```nano
   # i is always in bounds [0..len)
   # max_val is always the max of arr[0..i]
   ```

4. **Edge Cases** - Boundary conditions I have tested.
   ```nano
   # Single element
   # Negative numbers
   # Maximum and minimum values
   ```

---

## Advanced Safety Features

### Option 1: Compile-Time Length Tracking

I can track array lengths in my type system.

```nano
# Future enhancement
let nums: array<int, 5> = [1, 2, 3, 4, 5]  # Length known at compile time

fn get_third(arr: array<int, 5>) -> int {
    return (at arr 2)  # I know 2 < 5, so no runtime check is needed
}
```

**Benefits:**
- I can eliminate some bounds checks.
- I catch more errors at compile time.
- I find better optimization opportunities.

**Drawbacks:**
- My type system becomes more complex.
- I become less flexible.
- I am harder to implement.

**Verdict:** This is a Phase 4 feature. I will keep things simple for now.

### Option 2: Range Types for Indices

I can restrict indices to a valid range.

```nano
# Future enhancement
type Index5 = range<0, 5>  # Valid indices: 0, 1, 2, 3, 4

fn safe_get(arr: array<int, 5>, i: Index5) -> int {
    return (at arr i)  # My type system guarantees i is in bounds
}
```

**Benefits:**
- I provide compile-time bounds checking.
- I have no runtime overhead for validated code.
- I offer proof of correctness.

**Drawbacks:**
- My type system becomes complex.
- I am harder for an LLM to generate.
- I still need runtime checks at my API boundaries.

**Verdict:** This is interesting for later, but overkill for my first version.

### Option 3: Contracts and Assertions

I already support these through shadow tests.

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

### My Arrays vs C Arrays

| Feature | C | Me |
|---------|---|----------|
| Bounds checking | None | Always |
| Type safety | Weak | Strong |
| Null safety | Pointers can be NULL | No null arrays |
| Memory safety | Manual | Managed |
| Verifiable | No | Shadow tests |

### My Arrays vs Python Lists

| Feature | Python | Me |
|---------|--------|----------|
| Bounds checking | Yes | Yes |
| Type safety | Dynamic | Static |
| Immutability | Optional | Default |
| Performance | Interpreted | Native |
| Verifiable | Limited | Shadow tests |

### My Arrays vs Rust Vec

| Feature | Rust | Me |
|---------|------|----------|
| Bounds checking | Yes | Yes |
| Type safety | Strong | Strong |
| Memory safety | Ownership | Managed |
| Complexity | High | Simple |
| LLM-friendly | Moderate | Very |

**My goal:** Rust-level safety with Python-level simplicity.

---

## Implementation Strategy

### Phase 1: Basic Safe Arrays (Current)

**What I must have:**
1. Runtime bounds checking (always).
2. Type-safe `array<T>`.
3. Immutable by default.
4. Fixed size.
5. Shadow tests for all operations.

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

Before I implement arrays, I verify:

- [ ] Bounds checking is implemented in my evaluator.
- [ ] Bounds checking is implemented in my transpiler.
- [ ] My type checker enforces type safety.
- [ ] My type checker enforces immutability.
- [ ] I have shadow tests for all array operations.
- [ ] My error messages include line and column numbers.
- [ ] My memory management is safe and does not leak.
- [ ] I have tested edge cases like empty arrays and single elements.
- [ ] My documentation includes my safety guarantees.
- [ ] An LLM can understand all my array operations.

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

**What an LLM can verify:**
- My input array is not modified.
- My output array has the same length.
- My elements are in reverse order.
- I handle edge cases.
- No bounds violations are possible.
- My double reverse returns the original.

---

## Conclusion

**My arrays are:**

1. **Safe by Construction**
   - Always bounds-checked.
   - Type-safe.
   - Immutable by default.
   - Never null or undefined.

2. **Verifiable Through Tests**
   - Mandatory shadow tests.
   - Properties are documented.
   - Edge cases are covered.
   - Validated at compile time.

3. **LLM-Friendly**
   - Clear semantics.
   - No hidden behavior.
   - Predictable operations.
   - Complete introspection.

4. **Production-Ready**
   - No undefined behavior.
   - Clear error messages.
   - Memory safe.
   - Performance competitive.

**My Design Philosophy:**
> Make the right thing easy and the wrong thing impossible.

My arrays are provably safe and trivially verifiable by both humans and LLMs.

---

**Next Steps:**
1. Implement basic safe arrays.
2. Add comprehensive shadow tests.
3. Benchmark my performance against C and Python.
4. Document my safety guarantees.
5. Have an LLM generate array code and verify correctness.

**Status:** Design complete. I am ready for implementation.

