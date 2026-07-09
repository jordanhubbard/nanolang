# My Memory Management

I manage memory so you can focus on logic. I use automatic garbage collection with reference counting.

## Table of Contents

1. [Overview](#overview)
2. [My Memory Management Model](#my-memory-management-model)
3. [What I Collect](#what-i-collect)
4. [My Garbage Collector Details](#my-garbage-collector-details)
5. [Stack vs Heap Allocation](#stack-vs-heap-allocation)
6. [Lifetime and Ownership](#lifetime-and-ownership)
7. [Performance Considerations](#performance-considerations)
8. [Best Practices](#best-practices)
9. [Debugging Memory Issues](#debugging-memory-issues)
10. [Advanced Topics](#advanced-topics)

---

## Overview

I use automatic garbage collection with reference counting for my memory management. This provides:

- No manual memory management. I don't use malloc or free in my user code.
- Deterministic cleanup. I free objects when their last reference disappears.
- Cycle detection. I detect and collect circular references.
- Zero runtime pauses. My reference counting has no stop-the-world pauses.
- Native performance. I compile to C with minimal overhead.

---

## My Memory Management Model

I use a hybrid approach:

1. Stack Allocation for my primitives and small values.
2. Garbage Collection for my dynamic data structures.
3. Reference Counting as my primary mechanism.
4. Cycle Collection as my backup for circular references.

```
┌─────────────────┬──────────────────────┐
│  Stack          │  Heap (GC-managed)   │
├─────────────────┼──────────────────────┤
│ int             │ string               │
│ float           │ array<T>             │
│ bool            │ structs (if large)   │
│ function ptrs   │ List<T>              │
│                 │ closures             │
└─────────────────┴──────────────────────┘
```

---

## What I Collect

### My GC-Managed Types

I automatically collect these types:

| Type | Description | Example |
|------|-------------|---------|
| **string** | UTF-8 text | `"hello"` |
| **bstring** | Binary strings | `bstr_new("data")` |
| **array<T>** | Dynamic arrays | `[1, 2, 3]` |
| **List<T>** | Generic lists | `(List_int_new)` |
| **struct** (large) | Structs > 128 bytes | Heap-allocated |
| **closures** | Function closures | Captured variables |
| **HashMap<K,V>** | Hash maps | ARC-wrapped, auto-freed |
| **Regex** | Compiled patterns | ARC-wrapped, auto-freed |

### My Stack-Allocated Types

I do not collect these (they are stack-allocated):

| Type | Description | Lifetime |
|------|-------------|----------|
| **int** | 64-bit integer | Function scope |
| **float** | 64-bit float | Function scope |
| **bool** | Boolean | Function scope |
| **u8** | 8-bit unsigned | Function scope |
| **struct** (small) | Structs ≤ 128 bytes | Function scope |
| **enum** | Enumeration | Function scope |
| **fn(...)** | Function pointers | Function scope |

---

## My Garbage Collector Details

### Algorithm: Reference Counting + Cycle Detection

My Primary Mechanism is reference counting.
- I give each object a reference count.
- I increment the count on assignment or copy.
- I decrement the count when a reference goes out of scope.
- I free the object when its count reaches zero.

My Secondary Mechanism is mark-and-sweep cycle collection.
- I run this periodically or when requested.
- I detect and collect circular references.
- I don't require stop-the-world pauses.

### GC Operations

#### My Automatic Operations

```nano
fn example() -> int {
    # String allocated (ref_count = 1)
    let s1: string = "hello"
    
    # String retained (ref_count = 2)
    let s2: string = s1
    
    # s1 goes out of scope (ref_count = 1)
    # s2 goes out of scope (ref_count = 0)
    # String automatically freed here
    
    return 0
}
```

#### Manual Control (Advanced)

```nano
# Force cycle collection
extern fn gc_collect_cycles() -> void

# Get GC statistics
extern fn gc_get_stats() -> GCStats

# Print debug info
extern fn gc_print_stats() -> void

fn debug_memory() -> void {
    (gc_collect_cycles)
    (gc_print_stats)
}
```

### GC Statistics

```nano
struct GCStats {
    total_allocated: int,   # Total bytes allocated
    total_freed: int,       # Total bytes freed
    current_usage: int,     # Current memory usage
    num_objects: int,       # Number of live objects
    num_collections: int    # Number of GC cycles
}
```

### ARC Wrapping for Opaque Types

I automatically manage opaque types from C modules using ARC-style wrapping. When a C function returns an opaque pointer, my runtime wraps it with a GC-tracked envelope. I call the appropriate cleanup function when the object is no longer referenced.

- `gc_wrap_external(ptr, finalizer)` - I wrap a malloc'd pointer with a cleanup function.
- `gc_unwrap(ptr)` - I extract the original pointer for passing back to C functions.

This happens transparently at call boundaries. You don't see the wrapping.

**What I handle automatically:**
- **HashMap** - I call `map_free` automatically.
- **Regex** - I call `regex_free` automatically.
- **Strings from modules** - Functions like `path_join` or `file_read` return strings I manage.

**What you manage manually:**
- **Json** - You must call `json_free` manually. I exclude Json from ARC because functions like `get` and `get_index` return borrowed references into the parent object. I cannot distinguish these from owned allocations.

### Scope-Based Cleanup (Compiled Mode)

In my compiled mode, I track opaque variables and generate `gc_release()` calls at the end of each block scope:

```nano
fn example() -> void {
    let pattern: Regex = (compile "^[a-z]+$")
    let result: int = (match pattern "hello")
    # gc_release(pattern) generated automatically here
}
```

I ensure that opaque types created in loops and blocks are released without your intervention.

### When I Collect

1. My reference count reaches zero. This is immediate.
2. You call `gc_collect_cycles()` explicitly.
3. My heap threshold is reached. This is automatic.
4. My program exits. I free all remaining objects.

### Tuning My GC Behavior

Set environment variables:

```bash
# Set heap threshold for cycle collection (bytes)
export NANO_GC_THRESHOLD=10485760  # 10MB

# Enable GC debug output
export NANO_GC_DEBUG=1

# Run program
./myprogram
```

---

## Stack vs Heap Allocation

### Stack Allocation (Fast)

```nano
fn compute() -> int {
    # All stack-allocated (fast!)
    let x: int = 42
    let y: int = 17
    let z: bool = true
    
    return (+ x y)
}  # Automatic cleanup (no GC involved)
```

**Benefits:**
- Extremely fast. I only adjust the stack pointer.
- No GC overhead.
- Deterministic cleanup.
- Cache-friendly.

**Limitations:**
- Limited lifetime. Restricted to function scope.
- I can't return references to local variables.
- Stack size is limited (typically ~8MB).

### Heap Allocation (Flexible)

```nano
fn create_data() -> array<int> {
    # Heap-allocated (survives function return)
    let numbers: array<int> = [1, 2, 3, 4, 5]
    
    return numbers  # OK: heap-allocated, GC-managed
}

fn main() -> int {
    let data: array<int> = (create_data)
    # data is still valid here
    (println (array_length data))
    return 0
}  # GC frees data here
```

**Benefits:**
- Flexible lifetime. Survives function return.
- I grow these dynamically.
- Shared across functions.

**Costs:**
- Allocation overhead.
- GC overhead from reference counting.
- Less cache-friendly.

---

## Lifetime and Ownership

### Variable Lifetime

```nano
fn example() -> void {
    let x: int = 42        # Lifetime: rest of function
    
    {
        let y: int = 17    # Lifetime: this block only
    }  # y destroyed here
    
    # y is not accessible here
}  # x destroyed here
```

### Struct Field Lifetimes

```nano
struct Person {
    name: string,     # GC-managed, freed when Person freed
    age: int          # Stack value, no GC
}

fn create_person() -> Person {
    let p: Person = Person {
        name: "Alice",  # String allocated on heap
        age: 30
    }
    
    return p  # p copied, but name is retained (ref_count++)
}  # Original p's stack frame destroyed, but name survives
```

### Array Element Lifetimes

```nano
fn example() -> void {
    let strings: array<string> = ["hello", "world"]
    # Each string has ref_count = 1
    
    let s: string = (at strings 0)
    # "hello" now has ref_count = 2
    
}  # strings freed, all elements released
```

---

## Performance Considerations

### Memory Allocation Costs

| Operation | Cost | When It Happens |
|-----------|------|-----------------|
| `int` variable | ~0ns | Stack allocation |
| `string` literal | ~100ns | Heap allocation + GC header |
| `array<T>` creation | ~200ns | Heap + metadata |
| `List<T>` push | ~50ns | Amortized (array doubling) |
| String concatenation | ~150ns | New allocation |
| GC retain | ~5ns | Increment ref_count |
| GC release | ~10ns | Decrement + check |
| GC cycle collection | ~1-10ms | Periodic (depends on heap size) |

### My Optimization Tips

#### 1. Prefer Stack Allocation

I prefer stack allocation for speed:

```nano
fn process(x: int, y: int) -> int {
    let result: int = (+ x y)
    return result
}
```

Avoid unnecessary heap allocation:

```nano
fn process(x: int, y: int) -> array<int> {
    return [x, y]  # Heap allocation for 2 ints!
}
```

#### 2. Avoid Excessive String Concatenation

This is slow because I must allocate each time:

```nano
fn build_string(n: int) -> string {
    let mut s: string = ""
    let mut i: int = 0
    while (< i n) {
        set s (str_concat s "x")  # New allocation each time!
        set i (+ i 1)
    }
    return s
}
```

Use my StringBuilder API instead (this is a future feature):

```nano
# Future: StringBuilder API
fn build_string(n: int) -> string {
    let mut sb: StringBuilder = (sb_new)
    let mut i: int = 0
    while (< i n) {
        (sb_append sb "x")
        set i (+ i 1)
    }
    return (sb_to_string sb)
}
```

#### 3. Reuse My Arrays

I recommend reusing arrays where possible:

```nano
fn process_batches(data: array<int>) -> void {
    # Process in place - no new allocations
    let mut i: int = 0
    while (< i (array_length data)) {
        let val: int = (at data i)
        (process val)
        set i (+ i 1)
    }
}
```

Avoid creating a new array in each iteration:

```nano
fn process_batches(data: array<int>) -> void {
    let mut i: int = 0
    while (< i (array_length data)) {
        let subset: array<int> = [( at data i)]  # New allocation!
        (process subset)
        set i (+ i 1)
    }
}
```

---

## Best Practices

### 1. Minimize Heap Allocations in Hot Loops

I suggest allocating outside the loop:

```nano
# Good: allocation outside loop
fn process_data(n: int) -> void {
    let buffer: array<int> = (array_new 1000 0)
    
    let mut i: int = 0
    while (< i n) {
        # Work with pre-allocated buffer
        (array_set buffer 0 i)
        set i (+ i 1)
    }
}

# Bad: allocation inside loop
fn process_data(n: int) -> void {
    let mut i: int = 0
    while (< i n) {
        let buffer: array<int> = (array_new 1000 0)  # Allocates every iteration!
        set i (+ i 1)
    }
}
```

### 2. Be Aware of My Hidden Allocations

- My string concatenation allocates.
- My array literals allocate.
- My struct construction may allocate if the struct is larger than 128 bytes.

### 3. Use Primitives When Possible

I prefer stack-allocated integers over heap-allocated arrays:

```nano
# I prefer this (stack-allocated)
fn calculate(x: int, y: int) -> int {
    return (+ x y)
}

# Not this (heap-allocated)
fn calculate(coords: array<int>) -> int {
    return (+ (at coords 0) (at coords 1))
}
```

### 4. Don't Worry About Cycles

I handle cycles automatically. This is fine:

```nano
# I will detect and collect this cycle
struct Node {
    value: int,
    next: Option<Node>  # Can form cycle
}
```

### 5. Profile Before Optimizing

```bash
# Run with my GC stats enabled
NANO_GC_DEBUG=1 ./myprogram

# Check my memory usage
time ./myprogram
```

---

## Debugging Memory Issues

### Memory Leaks

If you notice memory usage growing, you can debug it this way:

```nano
fn main() -> int {
    # Enable my GC stats
    (gc_print_stats)
    
    # Run your code
    (my_function)
    
    # Force my collection
    (gc_collect_cycles)
    
    # Check my stats again
    (gc_print_stats)
    
    return 0
}
```

Common causes:
1. Circular references (I should collect these automatically).
2. Global variables. I never free these.
3. C FFI memory that was not freed. This requires your manual management.

### Memory Corruption

If I crash with a segfault or other random error, try these:

```bash
# Run with valgrind
valgrind --leak-check=full ./myprogram

# Run with address sanitizer
./bin/nanoc mycode.nano -o myprogram -DUSE_ASAN
./myprogram
```

Common causes:
1. Unsafe FFI calls.
2. Array out-of-bounds access. I should catch these.
3. Use-after-free. This shouldn't happen with my GC.

---

## Advanced Topics

### Affine Types (Resource Management)

For resources that you must free exactly once:

```nano
affine type FileHandle

fn open_file(path: string) -> FileHandle {
    # Opens file, returns affine handle
}

fn close_file(handle: FileHandle) -> void {
    # Consumes handle, can't use again
}
```

I explain this in [AFFINE_TYPES_GUIDE.md](AFFINE_TYPES_GUIDE.md).

### Manual Memory Management (FFI)

When you call C code that allocates:

```nano
# C function that allocates
extern fn c_malloc(size: int) -> opaque

# C function that frees
extern fn c_free(ptr: opaque) -> void

fn example() -> void {
    let ptr: opaque = (c_malloc 1024)
    
    # Use ptr...
    
    (c_free ptr)  # You must call this. I don't track it.
}
```

### My GC API Reference

```nano
# Force cycle collection
extern fn gc_collect_cycles() -> void

# Collect all unreachable objects
extern fn gc_collect_all() -> void

# Get statistics
extern fn gc_get_stats() -> GCStats

# Print stats to stderr
extern fn gc_print_stats() -> void

# Check if pointer is GC-managed
extern fn gc_is_managed(ptr: opaque) -> bool
```

---

## Related Documentation

- [AFFINE_TYPES_GUIDE.md](AFFINE_TYPES_GUIDE.md) - Resource management
- [EXTERN_FFI.md](EXTERN_FFI.md) - C interop and manual memory
- [SPECIFICATION.md](SPECIFICATION.md) - My language specification
- [DEBUGGING_GUIDE.md](DEBUGGING_GUIDE.md) - Debugging techniques

---

**Last Updated:** February 7, 2026
**Status:** Complete
**Version:** 0.2.0+
