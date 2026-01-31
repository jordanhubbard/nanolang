# Chapter 6: Collections

**Work with arrays and dynamic lists in NanoLang.**

Collections let you group multiple values together. This chapter covers arrays (fixed-size) and lists (dynamic), and how to work with them effectively.

## 6.1 Arrays (Immutable Collections)

Arrays are fixed-size, indexed collections of elements. All elements must have the same type.

### Array Creation

**Array literal syntax:**

```nano
fn create_arrays() -> array<int> {
    let numbers: array<int> = [1, 2, 3, 4, 5]
    let empty: array<int> = []
    let floats: array<float> = [1.5, 2.5, 3.5]
    return numbers
}

shadow create_arrays {
    let arr: array<int> = (create_arrays)
    assert (== (array_length arr) 5)
}
```

**Using array_new:**

```nano
fn create_with_default(size: int, value: int) -> array<int> {
    return (array_new size value)
}

shadow create_with_default {
    let zeros: array<int> = (create_with_default 5 0)
    assert (== (array_length zeros) 5)
    assert (== (array_get zeros 0) 0)
}
```

### Array Syntax

**Type annotation:** `array<T>` where T is the element type

```nano
fn type_examples() -> void {
    let ints: array<int> = [1, 2, 3]
    let strings: array<string> = ["a", "b", "c"]
    let bools: array<bool> = [true, false, true]
}

shadow type_examples {
    (type_examples)
}
```

### Array Characteristics

**Fixed size:**

```nano
fn array_size_demo() -> int {
    let arr: array<int> = [1, 2, 3]
    return (array_length arr)  # Always 3
}

shadow array_size_demo {
    assert (== (array_size_demo) 3)
}
```

**Homogeneous (same type):**

```nano
✅ Valid:
let numbers: array<int> = [1, 2, 3, 4]

❌ Invalid (mixed types):
# let mixed = [1, "two", 3.0]  # Type error!
```

**Immutable by default:**

```nano
fn immutable_demo() -> array<int> {
    let arr: array<int> = [1, 2, 3]
    # Can't modify arr directly
    # Must create new array with changes
    return arr
}

shadow immutable_demo {
    assert (== (array_length (immutable_demo)) 3)
}
```

### Examples

**Array of strings:**

```nano
fn name_array() -> array<string> {
    return ["Alice", "Bob", "Charlie"]
}

shadow name_array {
    let names: array<string> = (name_array)
    assert (== (array_get names 0) "Alice")
    assert (== (array_get names 2) "Charlie")
}
```

**Empty arrays:**

```nano
fn empty_arrays() -> int {
    let empty: array<int> = []
    return (array_length empty)
}

shadow empty_arrays {
    assert (== (empty_arrays) 0)
}
```

## 6.2 Array Operations

Working with array elements using built-in functions.

### Accessing Elements

Use `array_get` to access elements by index (0-based):

```nano
fn access_example(arr: array<int>, index: int) -> int {
    return (array_get arr index)
}

shadow access_example {
    let nums: array<int> = [10, 20, 30, 40, 50]
    assert (== (access_example nums 0) 10)
    assert (== (access_example nums 2) 30)
    assert (== (access_example nums 4) 50)
}
```

⚠️ **Watch Out:** Index out of bounds causes runtime error.

### Array Functions

**Get length:**

```nano
fn get_length(arr: array<int>) -> int {
    return (array_length arr)
}

shadow get_length {
    assert (== (get_length [1, 2, 3]) 3)
    assert (== (get_length []) 0)
    assert (== (get_length [42]) 1)
}
```

**Set element (mutable arrays):**

```nano
fn modify_array() -> array<int> {
    let mut arr: array<int> = [1, 2, 3]
    (array_set arr 1 99)
    return arr
}

shadow modify_array {
    let result: array<int> = (modify_array)
    assert (== (array_get result 1) 99)
}
```

**Push element (creates new array):**

```nano
fn append_element(arr: array<int>, value: int) -> array<int> {
    return (array_push arr value)
}

shadow append_element {
    let original: array<int> = [1, 2, 3]
    let extended: array<int> = (append_element original 4)
    assert (== (array_length extended) 4)
    assert (== (array_get extended 3) 4)
}
```

### Iterating Over Arrays

**Using for loop:**

```nano
fn sum_array(arr: array<int>) -> int {
    let mut sum: int = 0
    for (let i: int = 0) (< i (array_length arr)) (set i (+ i 1)) {
        set sum (+ sum (array_get arr i))
    }
    return sum
}

shadow sum_array {
    assert (== (sum_array [1, 2, 3, 4, 5]) 15)
    assert (== (sum_array []) 0)
}
```

**Using while loop:**

```nano
fn product_array(arr: array<int>) -> int {
    let mut product: int = 1
    let mut i: int = 0
    while (< i (array_length arr)) {
        set product (* product (array_get arr i))
        set i (+ i 1)
    }
    return product
}

shadow product_array {
    assert (== (product_array [2, 3, 4]) 24)
    assert (== (product_array [1, 1, 1]) 1)
}
```

### Examples

**Find maximum:**

```nano
fn find_max(arr: array<int>) -> int {
    let mut max: int = (array_get arr 0)
    for (let i: int = 1) (< i (array_length arr)) (set i (+ i 1)) {
        let val: int = (array_get arr i)
        if (> val max) {
            set max val
        }
    }
    return max
}

shadow find_max {
    assert (== (find_max [1, 5, 3, 9, 2]) 9)
    assert (== (find_max [-5, -2, -10]) -2)
}
```

**Count occurrences:**

```nano
fn count_value(arr: array<int>, target: int) -> int {
    let mut count: int = 0
    for (let i: int = 0) (< i (array_length arr)) (set i (+ i 1)) {
        if (== (array_get arr i) target) {
            set count (+ count 1)
        }
    }
    return count
}

shadow count_value {
    assert (== (count_value [1, 2, 3, 2, 4, 2] 2) 3)
    assert (== (count_value [1, 2, 3] 9) 0)
}
```

**Check if contains:**

```nano
fn contains(arr: array<int>, target: int) -> bool {
    for (let i: int = 0) (< i (array_length arr)) (set i (+ i 1)) {
        if (== (array_get arr i) target) {
            return true
        }
    }
    return false
}

shadow contains {
    assert (contains [1, 2, 3, 4, 5] 3)
    assert (not (contains [1, 2, 3] 9))
}
```

**Reverse array:**

```nano
fn reverse(arr: array<int>) -> array<int> {
    let len: int = (array_length arr)
    let mut result: array<int> = (array_new len 0)
    for (let i: int = 0) (< i len) (set i (+ i 1)) {
        (array_set result (- (- len i) 1) (array_get arr i))
    }
    return result
}

shadow reverse {
    let reversed: array<int> = (reverse [1, 2, 3, 4, 5])
    assert (== (array_get reversed 0) 5)
    assert (== (array_get reversed 4) 1)
}
```

## 6.3 List<T> (Dynamic Collections)

Lists are dynamic arrays that can grow and shrink. They're implemented in the standard library.

### Dynamic Lists

**Note:** Lists are a higher-level abstraction built on arrays. For basic programs, arrays are sufficient.

```nano
# Lists allow adding elements without knowing size upfront
# Import from stdlib when needed:
# from "stdlib/list.nano" import List, list_new, list_push, list_get
```

### List Operations

**Creating lists:**

```nano
# Example (conceptual - requires stdlib import):
# let mut my_list: List<int> = (list_new)
# (list_push my_list 1)
# (list_push my_list 2)
# (list_push my_list 3)
```

### Growing and Shrinking Lists

Lists automatically handle capacity:
- Start with small capacity
- Grow when full (typically double capacity)
- Can remove elements

### Examples

For most use cases, arrays are sufficient. Use lists when:
- Size isn't known in advance
- Frequent additions/removals
- Building collections incrementally

## 6.4 Working with Collections

Common patterns and best practices.

### Common Patterns

**Map pattern (transform elements):**

```nano
fn double_all(arr: array<int>) -> array<int> {
    let len: int = (array_length arr)
    let mut result: array<int> = (array_new len 0)
    for (let i: int = 0) (< i len) (set i (+ i 1)) {
        (array_set result i (* (array_get arr i) 2))
    }
    return result
}

shadow double_all {
    let doubled: array<int> = (double_all [1, 2, 3])
    assert (== (array_get doubled 0) 2)
    assert (== (array_get doubled 2) 6)
}
```

**Filter pattern (select elements):**

```nano
fn count_positives(arr: array<int>) -> int {
    let mut count: int = 0
    for (let i: int = 0) (< i (array_length arr)) (set i (+ i 1)) {
        if (> (array_get arr i) 0) {
            set count (+ count 1)
        }
    }
    return count
}

shadow count_positives {
    assert (== (count_positives [1, -2, 3, -4, 5]) 3)
}
```

**Reduce pattern (accumulate):**

```nano
fn sum_with_initial(arr: array<int>, initial: int) -> int {
    let mut acc: int = initial
    for (let i: int = 0) (< i (array_length arr)) (set i (+ i 1)) {
        set acc (+ acc (array_get arr i))
    }
    return acc
}

shadow sum_with_initial {
    assert (== (sum_with_initial [1, 2, 3] 10) 16)
}
```

### Collection Comparison

**When to use arrays:**
- ✅ Fixed size known upfront
- ✅ Simple iteration
- ✅ Performance is critical
- ✅ Most common use case

**When to use lists:**
- ✅ Size unknown or variable
- ✅ Frequent additions
- ✅ Building collections incrementally

### When to Use Each Type

**Arrays are best for:**

```nano
fn process_fixed_data() -> array<int> {
    # Known data structure
    let days_in_week: array<string> = [
        "Monday", "Tuesday", "Wednesday", 
        "Thursday", "Friday", "Saturday", "Sunday"
    ]
    
    # Known computation results
    let fibonacci_10: array<int> = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    
    return fibonacci_10
}

shadow process_fixed_data {
    let fib: array<int> = (process_fixed_data)
    assert (== (array_length fib) 10)
}
```

**Lists are best for:**

```nano
# Parsing input where count is unknown
# Building result sets from filters
# Dynamic data structures
# (Requires stdlib import)
```

### Practical Examples

**Average of array:**

```nano
fn average(arr: array<int>) -> float {
    let sum: int = (sum_array arr)
    let count: int = (array_length arr)
    return (/ (int_to_float sum) (int_to_float count))
}

shadow average {
    let avg: float = (average [10, 20, 30])
    assert (and (> avg 19.9) (< avg 20.1))
}
```

**Find all indices of value:**

```nano
fn find_all_indices(arr: array<int>, target: int) -> array<int> {
    # Count occurrences first
    let count: int = (count_value arr target)
    let mut result: array<int> = (array_new count 0)
    let mut result_idx: int = 0
    
    for (let i: int = 0) (< i (array_length arr)) (set i (+ i 1)) {
        if (== (array_get arr i) target) {
            (array_set result result_idx i)
            set result_idx (+ result_idx 1)
        }
    }
    return result
}

shadow find_all_indices {
    let indices: array<int> = (find_all_indices [1, 2, 3, 2, 4, 2] 2)
    assert (== (array_length indices) 3)
    assert (== (array_get indices 0) 1)
    assert (== (array_get indices 1) 3)
    assert (== (array_get indices 2) 5)
}
```

**Check if sorted:**

```nano
fn is_sorted(arr: array<int>) -> bool {
    for (let i: int = 1) (< i (array_length arr)) (set i (+ i 1)) {
        if (< (array_get arr i) (array_get arr (- i 1))) {
            return false
        }
    }
    return true
}

shadow is_sorted {
    assert (is_sorted [1, 2, 3, 4, 5])
    assert (not (is_sorted [1, 3, 2, 4]))
    assert (is_sorted [])
    assert (is_sorted [42])
}
```

**Merge two sorted arrays:**

```nano
fn merge_sorted(a: array<int>, b: array<int>) -> array<int> {
    let len_a: int = (array_length a)
    let len_b: int = (array_length b)
    let mut result: array<int> = (array_new (+ len_a len_b) 0)
    
    let mut i: int = 0
    let mut j: int = 0
    let mut k: int = 0
    
    while (and (< i len_a) (< j len_b)) {
        if (<= (array_get a i) (array_get b j)) {
            (array_set result k (array_get a i))
            set i (+ i 1)
        } else {
            (array_set result k (array_get b j))
            set j (+ j 1)
        }
        set k (+ k 1)
    }
    
    while (< i len_a) {
        (array_set result k (array_get a i))
        set i (+ i 1)
        set k (+ k 1)
    }
    
    while (< j len_b) {
        (array_set result k (array_get b j))
        set j (+ j 1)
        set k (+ k 1)
    }
    
    return result
}

shadow merge_sorted {
    let merged: array<int> = (merge_sorted [1, 3, 5] [2, 4, 6])
    assert (== (array_length merged) 6)
    assert (== (array_get merged 0) 1)
    assert (== (array_get merged 5) 6)
    assert (is_sorted merged)
}
```

### Summary

In this chapter, you learned:
- ✅ Arrays are fixed-size, homogeneous collections
- ✅ Array operations: get, set, push, length
- ✅ Iterating over arrays with for/while loops
- ✅ Common patterns: map, filter, reduce
- ✅ When to use arrays vs lists
- ✅ Practical examples: find max, reverse, merge sorted

### Practice Exercises

```nano
# 1. Remove duplicates (return unique elements)
fn unique(arr: array<int>) -> array<int> {
    let len: int = (array_length arr)
    if (== len 0) { return [] }
    
    # Count unique elements
    let mut unique_count: int = 0
    for (let i: int = 0) (< i len) (set i (+ i 1)) {
        let val: int = (array_get arr i)
        let mut is_duplicate: bool = false
        for (let j: int = 0) (< j i) (set j (+ j 1)) {
            if (== (array_get arr j) val) {
                set is_duplicate true
            }
        }
        if (not is_duplicate) {
            set unique_count (+ unique_count 1)
        }
    }
    
    # Build result
    let mut result: array<int> = (array_new unique_count 0)
    let mut idx: int = 0
    for (let i: int = 0) (< i len) (set i (+ i 1)) {
        let val: int = (array_get arr i)
        let mut is_duplicate: bool = false
        for (let j: int = 0) (< j i) (set j (+ j 1)) {
            if (== (array_get arr j) val) {
                set is_duplicate true
            }
        }
        if (not is_duplicate) {
            (array_set result idx val)
            set idx (+ idx 1)
        }
    }
    return result
}

shadow unique {
    let uniq: array<int> = (unique [1, 2, 2, 3, 1, 4])
    assert (== (array_length uniq) 4)
}

# 2. Find second largest element
fn second_largest(arr: array<int>) -> int {
    let mut largest: int = (array_get arr 0)
    let mut second: int = (array_get arr 0)
    
    for (let i: int = 1) (< i (array_length arr)) (set i (+ i 1)) {
        let val: int = (array_get arr i)
        if (> val largest) {
            set second largest
            set largest val
        } else { if (and (!= val largest) (> val second)) {
            set second val
        }}
    }
    return second
}

shadow second_largest {
    assert (== (second_largest [1, 5, 3, 9, 2]) 5)
}

# 3. Rotate array right by n positions
fn rotate_right(arr: array<int>, n: int) -> array<int> {
    let len: int = (array_length arr)
    if (== len 0) { return arr }
    
    let actual_n: int = (% n len)
    let mut result: array<int> = (array_new len 0)
    
    for (let i: int = 0) (< i len) (set i (+ i 1)) {
        let new_pos: int = (% (+ i actual_n) len)
        (array_set result new_pos (array_get arr i))
    }
    return result
}

shadow rotate_right {
    let rotated: array<int> = (rotate_right [1, 2, 3, 4, 5] 2)
    assert (== (array_get rotated 0) 4)
    assert (== (array_get rotated 1) 5)
}
```

---

**Previous:** [Chapter 5: Control Flow](05_control_flow.html)  
**Next:** [Chapter 7: Data Structures](07_data_structures.html)
