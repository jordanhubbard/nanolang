# Memory Management in NanoLang

Complete guide to NanoLang's memory management model, garbage collection, and best practices.

## Table of Contents

1. [Overview](#overview)
2. [Memory Management Model](#memory-management-model)
3. [What is Garbage Collected?](#what-is-garbage-collected)
4. [Garbage Collector Details](#garbage-collector-details)
5. [Stack vs Heap Allocation](#stack-vs-heap-allocation)
6. [Lifetime and Ownership](#lifetime-and-ownership)
7. [Performance Considerations](#performance-considerations)
8. [Best Practices](#best-practices)
9. [Debugging Memory Issues](#debugging-memory-issues)
10. [Advanced Topics](#advanced-topics)

---

## Overview

NanoLang uses **automatic garbage collection** with **reference counting** for memory management. This provides:

✅ **No manual memory management** - No malloc/free in user code
✅ **Deterministic cleanup** - Objects freed when last reference goes away
✅ **Cycle detection** - Circular references are detected and collected
✅ **Zero runtime pauses** - Reference counting has no stop-the-world pauses
✅ **Native performance** - Compiles to C, minimal overhead

---

## Memory Management Model

NanoLang uses a **hybrid approach**:

1. **Stack Allocation** - Primitives and small values
2. **Garbage Collection** - Dynamic data structures
3. **Reference Counting** - Primary GC mechanism
4. **Cycle Collection** - Backup for circular references

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

## What is Garbage Collected?

### GC-Managed Types ✅

These types are automatically garbage collected:

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

### Stack-Allocated Types ❌

These are NOT garbage collected (stack-allocated):

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

## Garbage Collector Details

### Algorithm: Reference Counting + Cycle Detection

**Primary Mechanism**: Reference counting
- Each object has a reference count
- Count incremented on assignment/copy
- Count decremented when reference goes out of scope
- Object freed when count reaches zero

**Secondary Mechanism**: Mark-and-sweep cycle collection
- Runs periodically or on demand
- Detects and collects circular references
- Doesn't require stop-the-world pauses

### GC Operations

#### Automatic Operations

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

Opaque types (Regex, HashMap, etc.) from C modules are automatically managed via **ARC-style wrapping**. When a C function returns an opaque pointer, the runtime wraps it with a GC-tracked envelope that calls the appropriate cleanup function when the object is no longer referenced.

- `gc_wrap_external(ptr, finalizer)` — Wraps a malloc'd pointer with a cleanup function
- `gc_unwrap(ptr)` — Extracts the original pointer for passing back to C functions

This happens transparently at call boundaries — user code never sees the wrapping.

**What's automatic:**
- **HashMap** — `map_free` called automatically
- **Regex** — `regex_free` called automatically
- **Strings from modules** — `path_join`, `file_read`, etc. return GC-managed strings

**What requires manual management:**
- **Json** — Must call `json_free` manually. Json is excluded from ARC because functions like `get` and `get_index` return *borrowed references* into the parent object, which ARC cannot distinguish from owned allocations.

### Scope-Based Cleanup (Compiled Mode)

In compiled mode, the transpiler tracks opaque variables and generates `gc_release()` calls at the end of each block scope:

```nano
fn example() -> void {
    let pattern: Regex = (compile "^[a-z]+$")
    let result: int = (match pattern "hello")
    # gc_release(pattern) generated automatically here
}
```

This ensures opaque types created in loops and blocks are properly released without manual intervention.

### When Does Collection Happen?

1. **Reference count reaches zero** - Immediate
2. **Explicit gc_collect_cycles()** - On demand
3. **Heap threshold reached** - Automatic (configurable)
4. **Program exit** - All remaining objects freed

### Tuning GC Behavior

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
- ✅ Extremely fast (just stack pointer adjustment)
- ✅ No GC overhead
- ✅ Deterministic cleanup
- ✅ Cache-friendly

**Limitations:**
- ❌ Limited lifetime (function scope only)
- ❌ Can't return references to local variables
- ❌ Stack size limited (~8MB typical)

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
- ✅ Flexible lifetime (survives function return)
- ✅ Can grow dynamically
- ✅ Shared across functions

**Costs:**
- ⚠️ Allocation overhead
- ⚠️ GC overhead (reference counting)
- ⚠️ Less cache-friendly

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

### Optimization Tips

#### 1. Prefer Stack Allocation

✅ **Good** (stack):
```nano
fn process(x: int, y: int) -> int {
    let result: int = (+ x y)
    return result
}
```

❌ **Avoid** (unnecessary heap):
```nano
fn process(x: int, y: int) -> array<int> {
    return [x, y]  # Heap allocation for 2 ints!
}
```

#### 2. Avoid Excessive String Concatenation

❌ **Slow** (N allocations):
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

✅ **Fast** (use StringBuilder - future):
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

#### 3. Reuse Arrays

✅ **Good** (reuse):
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

❌ **Avoid** (new array each iteration):
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

```nano
# ✅ Good: allocation outside loop
fn process_data(n: int) -> void {
    let buffer: array<int> = (array_new 1000 0)
    
    let mut i: int = 0
    while (< i n) {
        # Work with pre-allocated buffer
        (array_set buffer 0 i)
        set i (+ i 1)
    }
}

# ❌ Bad: allocation inside loop
fn process_data(n: int) -> void {
    let mut i: int = 0
    while (< i n) {
        let buffer: array<int> = (array_new 1000 0)  # Allocates every iteration!
        set i (+ i 1)
    }
}
```

### 2. Be Aware of Hidden Allocations

```nano
# String concatenation allocates
let s: string = (str_concat "hello" "world")  # New allocation

# Array literals allocate
let arr: array<int> = [1, 2, 3]  # Heap allocation

# Struct construction may allocate (if > 128 bytes)
let p: BigStruct = BigStruct { ... }  # Heap if large
```

### 3. Use Primitives When Possible

```nano
# ✅ Prefer this (stack-allocated)
fn calculate(x: int, y: int) -> int {
    return (+ x y)
}

# ❌ Not this (heap-allocated)
fn calculate(coords: array<int>) -> int {
    return (+ (at coords 0) (at coords 1))
}
```

### 4. Don't Worry About Cycles (Usually)

The GC handles cycles automatically:

```nano
# This is fine - GC will detect and collect the cycle
struct Node {
    value: int,
    next: Option<Node>  # Can form cycle
}
```

### 5. Profile Before Optimizing

```bash
# Run with GC stats
NANO_GC_DEBUG=1 ./myprogram

# Check memory usage
time ./myprogram
```

---

## Debugging Memory Issues

### Memory Leaks

**Symptom**: Memory usage keeps growing

**Debug:**
```nano
fn main() -> int {
    # Enable GC stats
    (gc_print_stats)
    
    # Run your code
    (my_function)
    
    # Force collection
    (gc_collect_cycles)
    
    # Check stats again
    (gc_print_stats)
    
    return 0
}
```

**Common Causes**:
1. Circular references (should be collected automatically)
2. Global variables (never freed)
3. C FFI memory not freed (manual management required)

### Memory Corruption

**Symptom**: Segfaults, random crashes

**Debug:**
```bash
# Run with valgrind
valgrind --leak-check=full ./myprogram

# Run with address sanitizer
./bin/nanoc mycode.nano -o myprogram -DUSE_ASAN
./myprogram
```

**Common Causes**:
1. Unsafe FFI calls
2. Array out-of-bounds (should be caught)
3. Use-after-free (shouldn't happen with GC)

---

## Advanced Topics

### Affine Types (Resource Management)

For resources that MUST be freed exactly once:

```nano
affine type FileHandle

fn open_file(path: string) -> FileHandle {
    # Opens file, returns affine handle
}

fn close_file(handle: FileHandle) -> void {
    # Consumes handle, can't use again
}
```

See [AFFINE_TYPES_GUIDE.md](AFFINE_TYPES_GUIDE.md) for details.

### Manual Memory Management (FFI)

When calling C code that allocates:

```nano
# C function that allocates
extern fn c_malloc(size: int) -> opaque

# C function that frees
extern fn c_free(ptr: opaque) -> void

fn example() -> void {
    let ptr: opaque = (c_malloc 1024)
    
    # Use ptr...
    
    (c_free ptr)  # MUST call manually - GC doesn't track this!
}
```

### GC API Reference

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
- [SPECIFICATION.md](SPECIFICATION.md) - Language specification
- [DEBUGGING_GUIDE.md](DEBUGGING_GUIDE.md) - Debugging techniques

---

**Last Updated:** February 7, 2026
**Status:** Complete
**Version:** 0.2.0+
