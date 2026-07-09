# My Dynamic Arrays

## Overview

I provide built-in dynamic arrays that grow and shrink as needed. I manage their memory automatically and ensure they remain type-safe.

## Array Operations

### Creation

```nano
let arr: array<int> = []                    # Empty array
let arr2: array<int> = [1, 2, 3, 4]         # Array literal
```

### Adding Elements

```nano
let mut arr: array<int> = []
set arr (array_push arr 42)                 # Add to end
set arr (array_push arr 43)
# arr is now [42, 43]
```

### Removing Elements

```nano
let val: int = (array_pop arr)              # Remove and return last element
# val = 43, arr = [42]

set arr (array_remove_at arr 0)             # Remove element at index 0
# arr = []
```

### Accessing Elements

```nano
let first: int = (at arr 0)                 # Get element at index
(array_set arr 0 100)                       # Set element at index
let len: int = (array_length arr)           # Get length
```

## Complete API Reference

| Function | Signature | Description |
|----------|-----------|-------------|
| `array_push` | `(arr, value) -> array` | Append element to end |
| `array_pop` | `(arr) -> value` | Remove and return last element |
| `array_remove_at` | `(arr, index) -> array` | Remove element at index |
| `at` | `(arr, index) -> value` | Get element at index |
| `array_set` | `(arr, index, value)` | Set element at index |
| `array_length` | `(arr) -> int` | Get number of elements |

## Type-Specific Operations

I require arrays to be typed. I support all my primitive types.

```nano
# Integer arrays
let mut ints: array<int> = []
set ints (array_push ints 42)
let val: int = (array_pop ints)

# Float arrays  
let mut floats: array<float> = []
set floats (array_push floats 3.14)
let pi: float = (array_pop floats)

# String arrays
let mut names: array<string> = []
set names (array_push names "Alice")
let name: string = (array_pop names)

# Boolean arrays
let mut flags: array<bool> = []
set flags (array_push flags true)
let flag: bool = (array_pop flags)

# Nested arrays
let mut matrix: array<array<int>> = []
set matrix (array_push matrix [1, 2, 3])
let row: array<int> = (array_pop matrix)
```

## Examples

### Example 1: Dynamic Stack

```nano
fn stack_demo() -> int {
    let mut stack: array<int> = []
    
    # Push elements
    set stack (array_push stack 10)
    set stack (array_push stack 20)
    set stack (array_push stack 30)
    
    # Pop elements (LIFO)
    let top: int = (array_pop stack)       # 30
    let second: int = (array_pop stack)    # 20
    
    return (+ top second)  # Returns 50
}
```

### Example 2: Remove Elements

```nano
fn remove_demo() -> int {
    let mut arr: array<int> = [10, 20, 30, 40, 50]
    
    # Remove element at index 2 (30)
    set arr (array_remove_at arr 2)
    # arr is now [10, 20, 40, 50]
    
    # Remove first element
    set arr (array_remove_at arr 0)
    # arr is now [20, 40, 50]
    
    return (at arr 0)  # Returns 20
}
```

### Example 3: Processing Arrays

```nano
fn sum_array(arr: array<int>) -> int {
    let mut sum: int = 0
    let len: int = (array_length arr)
    
    for i in (range 0 len) {
        set sum (+ sum (at arr i))
    }
    
    return sum
}

fn process_data() -> int {
    let mut data: array<int> = []
    
    for i in (range 0 10) {
        set data (array_push data i)
    }
    
    return (sum_array data)  # Returns 45
}
```

### Example 4: Array Filtering

```nano
fn filter_positive(arr: array<int>) -> array<int> {
    let mut result: array<int> = []
    
    for i in (range 0 (array_length arr)) {
        let val: int = (at arr i)
        if (> val 0) {
            set result (array_push result val)
        }
    }
    
    return result
}

fn demo_filter() -> int {
    let data: array<int> = [-5, 3, -2, 8, 0, 12]
    let positive: array<int> = (filter_positive data)
    
    return (array_length positive)  # Returns 3
}
```

## Performance Characteristics

- **Append (array_push)**: Amortized O(1)
- **Pop (array_pop)**: O(1)
- **Remove at index (array_remove_at)**: O(n)
- **Access (at)**: O(1)
- **Set (array_set)**: O(1)
- **Length (array_length)**: O(1)

## Memory Management

I manage array memory automatically. When you no longer reference an array, I reclaim its memory.

```nano
fn create_temporary() -> int {
    let temp: array<int> = [1, 2, 3]
    return (array_length temp)
}  # I free temp after the function returns
```

## Bounds Checking

I perform bounds checking on all array operations at runtime.

```nano
let arr: array<int> = [1, 2, 3]
let val: int = (at arr 10)  # Runtime error: index out of bounds
```

## Mutability

You must declare an array as mutable if you intend to modify it.

```nano
let arr: array<int> = [1, 2, 3]          # Immutable
set arr (array_push arr 4)                # Error
```

```nano
let mut arr2: array<int> = [1, 2, 3]      # Mutable
set arr2 (array_push arr2 4)              # Allowed
```

## See Also

- [Generic Types](./GENERICS_DEEP_DIVE.md)
- [Language Specification](./SPECIFICATION.md)
- [spec.json](../spec.json)
- [Examples](../examples/)
