# Chapter 10: Collections Library

**Work with arrays and hash maps efficiently.**

This chapter covers NanoLang's collection types: arrays for ordered sequences and hash maps for key-value storage. Both provide constant-time operations and type safety.

## 10.1 Arrays (Fixed-Size Collections)

Arrays are fixed-size, ordered collections of elements. All elements must have the same type.

### Creating Arrays

**Literal syntax:**

```nano
fn create_arrays() -> array<int> {
    let numbers: array<int> = [1, 2, 3, 4, 5]
    let empty: array<int> = []
    let strings: array<string> = ["a", "b", "c"]
    return numbers
}

shadow create_arrays {
    let arr: array<int> = (create_arrays)
    assert (== (array_length arr) 5)
}
```

**Using `array_new`:**

```nano
fn create_with_new() -> array<int> {
    let zeros: array<int> = (array_new 10 0)
    let spaces: array<string> = (array_new 5 " ")
    return zeros
}

shadow create_with_new {
    let arr: array<int> = (create_with_new)
    assert (== (array_length arr) 10)
    assert (== (at arr 0) 0)
}
```

**Syntax:** `array_new(size, default_value)`
- Size must be non-negative
- All elements initialized to default value

### Accessing Elements

```nano
fn access_example() -> int {
    let nums: array<int> = [10, 20, 30, 40, 50]
    
    let first: int = (at nums 0)   # 10
    let third: int = (at nums 2)   # 30
    let last: int = (at nums 4)    # 50
    
    return (+ first last)
}

shadow access_example {
    assert (== (access_example) 60)
}
```

⚠️ **Warning:** `at` performs bounds checking. Out-of-bounds access terminates the program.

### Modifying Arrays

Arrays must be declared `mut` to be modified:

```nano
fn modify_array() -> array<int> {
    let mut nums: array<int> = [1, 2, 3, 4, 5]
    
    (array_set nums 0 10)  # nums[0] = 10
    (array_set nums 2 30)  # nums[2] = 30
    
    return nums
}

shadow modify_array {
    let result: array<int> = (modify_array)
    assert (== (at result 0) 10)
    assert (== (at result 1) 2)  # Unchanged
    assert (== (at result 2) 30)
}
```

**Syntax:** `array_set(mut_array, index, value)`
- Requires mutable array
- Bounds-checked
- Type-checked (value must match array type)

### Array Length

```nano
fn length_example() -> bool {
    let nums: array<int> = [1, 2, 3, 4, 5]
    let empty: array<int> = []
    
    return (and 
        (== (array_length nums) 5)
        (== (array_length empty) 0)
    )
}

shadow length_example {
    assert (length_example)
}
```

### Iterating Over Arrays

**With for loop:**

```nano
fn sum_array(arr: array<int>) -> int {
    let mut sum: int = 0
    let len: int = (array_length arr)
    
    for i in (range 0 len) {
        set sum (+ sum (at arr i))
    }
    
    return sum
}

shadow sum_array {
    let nums: array<int> = [1, 2, 3, 4, 5]
    assert (== (sum_array nums) 15)
}
```

**With while loop:**

```nano
fn find_max(arr: array<int>) -> int {
    assert (> (array_length arr) 0)  # Non-empty
    
    let mut max: int = (at arr 0)
    let mut i: int = 1
    let len: int = (array_length arr)
    
    while (< i len) {
        let current: int = (at arr i)
        if (> current max) {
            set max current
        }
        set i (+ i 1)
    }
    
    return max
}

shadow find_max {
    assert (== (find_max [5, 2, 9, 1, 7]) 9)
    assert (== (find_max [42]) 42)
}
```

### Common Array Patterns

**Pattern 1: Fill array**

```nano
fn fill_array(size: int, value: int) -> array<int> {
    let mut arr: array<int> = (array_new size 0)
    
    for i in (range 0 size) {
        (array_set arr i value)
    }
    
    return arr
}

shadow fill_array {
    let arr: array<int> = (fill_array 5 42)
    assert (== (at arr 0) 42)
    assert (== (at arr 4) 42)
}
```

**Pattern 2: Map (transform elements)**

```nano
fn double_all(arr: array<int>) -> array<int> {
    let len: int = (array_length arr)
    let mut result: array<int> = (array_new len 0)
    
    for i in (range 0 len) {
        (array_set result i (* (at arr i) 2))
    }
    
    return result
}

shadow double_all {
    let input: array<int> = [1, 2, 3]
    let output: array<int> = (double_all input)
    assert (== (at output 0) 2)
    assert (== (at output 1) 4)
    assert (== (at output 2) 6)
}
```

**Pattern 3: Filter**

```nano
fn count_positive(arr: array<int>) -> int {
    let mut count: int = 0
    let len: int = (array_length arr)
    
    for i in (range 0 len) {
        if (> (at arr i) 0) {
            set count (+ count 1)
        }
    }
    
    return count
}

shadow count_positive {
    let nums: array<int> = [1, -2, 3, -4, 5]
    assert (== (count_positive nums) 3)
}
```

**Pattern 4: Reduce (accumulate)**

```nano
fn product(arr: array<int>) -> int {
    let mut result: int = 1
    let len: int = (array_length arr)
    
    for i in (range 0 len) {
        set result (* result (at arr i))
    }
    
    return result
}

shadow product {
    assert (== (product [2, 3, 4]) 24)
    assert (== (product [5, 5]) 25)
}
```

## 10.2 Hash Maps (Key-Value Storage)

Hash maps provide O(1) average-time lookups for key-value pairs.

### Creating Hash Maps

```nano
fn create_hashmap() -> HashMap<string, int> {
    let scores: HashMap<string, int> = (map_new)
    return scores
}

shadow create_hashmap {
    let hm: HashMap<string, int> = (create_hashmap)
    assert (== (map_size hm) 0)
    (map_free hm)
}
```

💡 **Pro Tip:** Type annotation is **required** for `map_new`. The compiler needs to know key and value types.

### Supported Types

**Keys:** `int` or `string`  
**Values:** `int` or `string`

```nano
fn all_combinations() -> void {
    let hm_si: HashMap<string, int> = (map_new)
    let hm_ss: HashMap<string, string> = (map_new)
    let hm_ii: HashMap<int, int> = (map_new)
    let hm_is: HashMap<int, string> = (map_new)
    
    (map_free hm_si)
    (map_free hm_ss)
    (map_free hm_ii)
    (map_free hm_is)
}

shadow all_combinations {
    (all_combinations)
}
```

### Adding Elements

```nano
fn add_scores() -> HashMap<string, int> {
    let scores: HashMap<string, int> = (map_new)
    
    (map_put scores "Alice" 100)
    (map_put scores "Bob" 85)
    (map_put scores "Carol" 92)
    
    return scores
}

shadow add_scores {
    let hm: HashMap<string, int> = (add_scores)
    assert (== (map_size hm) 3)
    (map_free hm)
}
```

**Syntax:** `map_put(hashmap, key, value)`
- Inserts new key
- Updates existing key

### Retrieving Values

```nano
fn get_score(scores: HashMap<string, int>, name: string) -> int {
    return (map_get scores name)
}

shadow get_score {
    let scores: HashMap<string, int> = (map_new)
    (map_put scores "Alice" 100)
    
    assert (== (get_score scores "Alice") 100)
    assert (== (get_score scores "Unknown") 0)  # Default
    
    (map_free scores)
}
```

**Default values:**
- `int` keys/values: returns `0`
- `string` keys/values: returns `""`

### Checking Existence

```nano
fn safe_get(hm: HashMap<string, int>, key: string) -> int {
    if (map_has hm key) {
        return (map_get hm key)
    }
    return -1  # Not found
}

shadow safe_get {
    let hm: HashMap<string, int> = (map_new)
    (map_put hm "key" 42)
    
    assert (== (safe_get hm "key") 42)
    assert (== (safe_get hm "missing") -1)
    
    (map_free hm)
}
```

### Removing Elements

```nano
fn remove_example() -> bool {
    let hm: HashMap<string, int> = (map_new)
    (map_put hm "a" 1)
    (map_put hm "b" 2)
    
    assert (== (map_size hm) 2)
    
    (map_remove hm "a")
    
    let result: bool = (and
        (== (map_size hm) 1)
        (not (map_has hm "a"))
    )
    
    (map_free hm)
    return result
}

shadow remove_example {
    assert (remove_example)
}
```

### Hash Map Size

```nano
fn size_example() -> bool {
    let hm: HashMap<string, int> = (map_new)
    
    assert (== (map_size hm) 0)
    
    (map_put hm "a" 1)
    assert (== (map_size hm) 1)
    
    (map_put hm "b" 2)
    assert (== (map_size hm) 2)
    
    (map_put hm "a" 3)  # Update existing
    assert (== (map_size hm) 2)  # Size unchanged
    
    (map_free hm)
    return true
}

shadow size_example {
    assert (size_example)
}
```

💡 **Note:** `map_length` is an alias for `map_size`.

### Clearing and Freeing

```nano
fn clear_vs_free() -> void {
    let hm: HashMap<string, int> = (map_new)
    (map_put hm "a" 1)
    (map_put hm "b" 2)
    
    # Clear removes all entries, but map still usable
    (map_clear hm)
    assert (== (map_size hm) 0)
    
    (map_put hm "c" 3)  # Can still add
    assert (== (map_size hm) 1)
    
    # Free releases all memory
    (map_free hm)
    # Cannot use hm after this!
}

shadow clear_vs_free {
    (clear_vs_free)
}
```

**Differences:**
- `map_clear` - Removes entries, keeps capacity
- `map_free` - Releases all memory

### Iterating Over Keys

```nano
fn sum_all_values(hm: HashMap<string, int>) -> int {
    let keys: array<string> = (map_keys hm)
    let mut sum: int = 0
    let len: int = (array_length keys)
    
    for i in (range 0 len) {
        let key: string = (at keys i)
        let value: int = (map_get hm key)
        set sum (+ sum value)
    }
    
    return sum
}

shadow sum_all_values {
    let hm: HashMap<string, int> = (map_new)
    (map_put hm "a" 10)
    (map_put hm "b" 20)
    (map_put hm "c" 30)
    
    assert (== (sum_all_values hm) 60)
    
    (map_free hm)
}
```

### Iterating Over Values

```nano
fn max_value(hm: HashMap<string, int>) -> int {
    let values: array<int> = (map_values hm)
    let len: int = (array_length values)
    
    if (== len 0) {
        return 0
    }
    
    let mut max: int = (at values 0)
    
    for i in (range 1 len) {
        let current: int = (at values i)
        if (> current max) {
            set max current
        }
    }
    
    return max
}

shadow max_value {
    let hm: HashMap<string, int> = (map_new)
    (map_put hm "a" 5)
    (map_put hm "b" 99)
    (map_put hm "c" 42)
    
    assert (== (max_value hm) 99)
    
    (map_free hm)
}
```

## 10.3 Practical Examples

### Example 1: Word Counter

```nano
fn count_words(words: array<string>) -> HashMap<string, int> {
    let counts: HashMap<string, int> = (map_new)
    let len: int = (array_length words)
    
    for i in (range 0 len) {
        let word: string = (at words i)
        if (map_has counts word) {
            let current: int = (map_get counts word)
            (map_put counts word (+ current 1))
        } else {
            (map_put counts word 1)
        }
    }
    
    return counts
}

shadow count_words {
    let words: array<string> = ["apple", "banana", "apple", "cherry", "banana", "apple"]
    let counts: HashMap<string, int> = (count_words words)
    
    assert (== (map_get counts "apple") 3)
    assert (== (map_get counts "banana") 2)
    assert (== (map_get counts "cherry") 1)
    
    (map_free counts)
}
```

### Example 2: Unique Elements

```nano
fn unique(arr: array<int>) -> array<int> {
    let seen: HashMap<int, int> = (map_new)
    let mut unique_list: array<int> = (array_new (array_length arr) 0)
    let mut count: int = 0
    
    for i in (range 0 (array_length arr)) {
        let value: int = (at arr i)
        if (not (map_has seen value)) {
            (map_put seen value 1)
            (array_set unique_list count value)
            set count (+ count 1)
        }
    }
    
    # Create result with exact size
    let mut result: array<int> = (array_new count 0)
    for i in (range 0 count) {
        (array_set result i (at unique_list i))
    }
    
    (map_free seen)
    return result
}

shadow unique {
    let input: array<int> = [1, 2, 3, 2, 1, 4, 5, 3]
    let output: array<int> = (unique input)
    
    assert (== (array_length output) 5)
    assert (== (at output 0) 1)
    assert (== (at output 1) 2)
    assert (== (at output 2) 3)
}
```

### Example 3: Leaderboard

```nano
struct Player {
    name: string,
    score: int
}

fn top_score(players: array<Player>) -> string {
    if (== (array_length players) 0) {
        return "No players"
    }
    
    let mut top_player: Player = (at players 0)
    
    for i in (range 1 (array_length players)) {
        let player: Player = (at players i)
        if (> player.score top_player.score) {
            set top_player player
        }
    }
    
    return top_player.name
}

shadow top_score {
    let mut players: array<Player> = (array_new 3 Player { name: "", score: 0 })
    
    (array_set players 0 Player { name: "Alice", score: 100 })
    (array_set players 1 Player { name: "Bob", score: 150 })
    (array_set players 2 Player { name: "Carol", score: 120 })
    
    assert (== (top_score players) "Bob")
}
```

### Example 4: Index Mapping

```nano
fn build_index(items: array<string>) -> HashMap<string, int> {
    let index: HashMap<string, int> = (map_new)
    
    for i in (range 0 (array_length items)) {
        let item: string = (at items i)
        (map_put index item i)
    }
    
    return index
}

fn find_position(items: array<string>, target: string) -> int {
    let index: HashMap<string, int> = (build_index items)
    
    if (map_has index target) {
        let pos: int = (map_get index target)
        (map_free index)
        return pos
    }
    
    (map_free index)
    return -1
}

shadow find_position {
    let items: array<string> = ["apple", "banana", "cherry", "date"]
    
    assert (== (find_position items "banana") 1)
    assert (== (find_position items "date") 3)
    assert (== (find_position items "grape") -1)
}
```

### Example 5: Group By

```nano
fn group_by_length(words: array<string>) -> HashMap<int, int> {
    let groups: HashMap<int, int> = (map_new)
    
    for i in (range 0 (array_length words)) {
        let word: string = (at words i)
        let len: int = (str_length word)
        
        if (map_has groups len) {
            let count: int = (map_get groups len)
            (map_put groups len (+ count 1))
        } else {
            (map_put groups len 1)
        }
    }
    
    return groups
}

shadow group_by_length {
    let words: array<string> = ["a", "bb", "ccc", "dd", "e", "fff"]
    let groups: HashMap<int, int> = (group_by_length words)
    
    assert (== (map_get groups 1) 2)  # "a", "e"
    assert (== (map_get groups 2) 2)  # "bb", "dd"
    assert (== (map_get groups 3) 2)  # "ccc", "fff"
    
    (map_free groups)
}
```

## 10.4 Performance & Best Practices

### Arrays

**✅ DO:**

```nano
# Pre-allocate known size
let mut arr: array<int> = (array_new 100 0)

# Cache length in loops
let len: int = (array_length arr)
for i in (range 0 len) {
    # Process (at arr i)
}

# Use for-range for clean iteration
for i in (range 0 (array_length arr)) {
    # Simpler than while
}
```

**❌ DON'T:**

```nano
# Don't call array_length repeatedly in loop condition
let mut i: int = 0
while (< i (array_length arr)) {  # Calls length every iteration!
    set i (+ i 1)
}

# Don't ignore bounds checking
# (at arr -1)  # Will terminate!
```

### Hash Maps

**✅ DO:**

```nano
# Always check before getting
if (map_has hm key) {
    let value: int = (map_get hm key)
    # Use value
}

# Always free when done
let hm: HashMap<string, int> = (map_new)
# ... use hm ...
(map_free hm)  # Prevent memory leak

# Use clear for reuse
(map_clear hm)
# Can add new entries
```

**❌ DON'T:**

```nano
# Don't forget type annotation
# let hm = (map_new)  # ERROR: can't infer type!

# Don't assume default means "not found"
let value: int = (map_get hm key)
# Is value 0 because missing or because stored 0?

# Better: Check with map_has first
```

### Memory Management

```nano
fn proper_cleanup() -> int {
    let hm: HashMap<string, int> = (map_new)
    (map_put hm "key" 42)
    
    # ... use hm ...
    
    # Always free before returning
    (map_free hm)
    return 0
}

shadow proper_cleanup {
    assert (== (proper_cleanup) 0)
}
```

## Summary

In this chapter, you learned:
- ✅ Arrays: fixed-size, ordered, type-safe collections
- ✅ Array operations: `array_new`, `at`, `array_set`, `array_length`
- ✅ Array patterns: map, filter, reduce, iteration
- ✅ Hash maps: O(1) key-value storage
- ✅ Hash map operations: `map_new`, `map_put`, `map_get`, `map_has`, `map_remove`
- ✅ Iteration with `map_keys` and `map_values`
- ✅ Memory management with `map_free`

### Quick Reference

| Operation | Array | HashMap |
|-----------|-------|---------|
| **Create** | `[1, 2, 3]` or `array_new` | `map_new` (needs type annotation) |
| **Size** | `array_length(arr)` | `map_size(hm)` |
| **Access** | `at(arr, i)` | `map_get(hm, key)` |
| **Modify** | `array_set(arr, i, val)` | `map_put(hm, key, val)` |
| **Check** | `< i (array_length arr)` | `map_has(hm, key)` |
| **Remove** | N/A (fixed size) | `map_remove(hm, key)` |
| **Iterate** | `for i in range` | `map_keys` / `map_values` |
| **Clear** | N/A | `map_clear(hm)` |
| **Free** | N/A (auto) | `map_free(hm)` required |

---

**Previous:** [Chapter 9: Core Utilities](09_core_utilities.html)  
**Next:** [Chapter 11: I/O & Filesystem](11_io_filesystem.html)
