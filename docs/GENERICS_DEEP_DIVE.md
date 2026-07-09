# Generics Deep Dive: Monomorphization in My Core

I implement generics through monomorphization. This document explains how I handle these types, the performance I achieve, and the trade-offs I accept.

## Table of Contents

1. [Overview](#overview)
2. [How Monomorphization Works](#how-monomorphization-works)
3. [Trade-offs](#trade-offs)
4. [Performance Characteristics](#performance-characteristics)
5. [Binary Size Impact](#binary-size-impact)
6. [Compilation Time](#compilation-time)
7. [Comparison with Other Approaches](#comparison-with-other-approaches)
8. [Best Practices](#best-practices)
9. [Advanced Topics](#advanced-topics)

---

## Overview

I use monomorphization for generics. This is the same approach used by Rust and C++. It means:

- I specialize generic types at compile time.
- I generate a separate implementation for each concrete type.
- I achieve zero runtime cost. I do not use type erasure or boxing.
- I produce larger binaries because each instantiation adds code.
- I execute quickly because I optimize every implementation for its specific type.

### What is Monomorphization?

```nano
# You write ONE generic definition:
let ints: List<int> = (List_int_new)
let strs: List<string> = (List_string_new)
let points: List<Point> = (List_Point_new)

# I generate THREE separate implementations:
# - List_int_new, List_int_push, List_int_get, ...
# - List_string_new, List_string_push, List_string_get, ...
# - List_Point_new, List_Point_push, List_Point_get, ...
```

Every instantiation I create is a completely separate function in my generated code.

---

## How Monomorphization Works

### Step 1: Type Discovery

I scan your code to find where you use generic types:

```nano
let nums: List<int> = (List_int_new)
let names: List<string> = (List_string_new)
```

**Types I discover:**
- `List<int>` → I need `List_int_*` functions.
- `List<string>` → I need `List_string_*` functions.

### Step 2: Code Generation

I generate specialized implementations based on these discoveries.

**Input (generic template):**
```nano
# Conceptual generic definition (not actual syntax yet)
fn List<T>_push(list: List<T>, item: T) -> void {
    # Generic implementation
}
```

**Output (generated C code):**
```c
// For List<int>
void List_int_push(List_int* list, int64_t item) {
    // Specialized implementation for int
}

// For List<string>
void List_string_push(List_string* list, char* item) {
    // Specialized implementation for string
}
```

### Step 3: Optimization

I optimize each generated function independently.

```c
// List<int> - I can use vectorization
void List_int_push(List_int* list, int64_t item) {
    // Optimized for integers
}

// List<string> - I use a different strategy
void List_string_push(List_string* list, char* item) {
    // Optimized for pointer management
}
```

---

## Trade-offs

### Advantages

| Benefit | Description | Impact |
|---------|-------------|--------|
| **Zero runtime cost** | I avoid boxing and virtual dispatch. | Fastest |
| **Full optimization** | I optimize each type independently. | Peak performance |
| **Type safety** | I catch errors at compile time. | Safe |
| **No runtime** | I do not need a GC for generic containers. | Predictable |
| **Inline-friendly** | I can inline code across types. | Speed |

### Disadvantages

| Cost | Description | Impact |
|------|-------------|--------|
| **Binary bloat** | Each type I add increases the code size. | Large binaries |
| **Compile time** | I have more code to generate and compile. | Slower builds |
| **Code duplication** | I repeat similar code structures. | Redundancy |
| **No runtime generics** | I cannot store `List<T>` with an unknown T. | Inflexible |
| **Longer link time** | I create more symbols for the linker. | Slower linking |

---

## Performance Characteristics

### Runtime Performance: Excellent

```nano
# List<int> operations are as fast as hand-written int code
let nums: List<int> = (List_int_new)
(List_int_push nums 42)        # Direct function call, no indirection
let x: int = (List_int_get nums 0)  # No type checking at runtime
```

**Benchmark (relative to C array):**

| Operation | List<int> | C array | Overhead |
|-----------|-----------|---------|----------|
| Push | 1.02x | 1.00x | 2% |
| Get | 1.00x | 1.00x | 0% |
| Iterate | 1.01x | 1.00x | 1% |

My generic code performs as fast as hand-written code.

### Memory Performance: Excellent

```c
// List<int> - compact memory layout
struct List_int {
    int64_t* data;    // 8 bytes
    size_t length;    // 8 bytes
    size_t capacity;  // 8 bytes
};                    // Total: 24 bytes + array data

// No boxing, no type tags, no vtables
```

My memory overhead is zero. It matches what you would write by hand.

---

## Binary Size Impact

### Example: Generic vs Non-Generic

**Code:**
```nano
# Use List with 5 types
let ints: List<int> = (List_int_new)
let floats: List<float> = (List_float_new)
let strings: List<string> = (List_string_new)
let points: List<Point> = (List_Point_new)
let configs: List<Config> = (List_Config_new)
```

**Generated code size:**

| Component | Code Size | Notes |
|-----------|-----------|-------|
| List<int> | ~2 KB | 4 functions × ~500 bytes |
| List<float> | ~2 KB | Similar to int |
| List<string> | ~2.5 KB | Pointer handling |
| List<Point> | ~2.5 KB | Struct copying |
| List<Config> | ~3 KB | Large struct |
| **Total** | **~12 KB** | For 5 instantiations |

### Scaling

```
1 type   → ~2 KB
5 types  → ~12 KB
10 types → ~25 KB
50 types → ~125 KB
```

I usually add 2 to 3 KB for each generic instantiation.

### Real-World Impact

**Small program (< 10 generic types):**
- Binary size: ~500 KB → ~520 KB.
- Impact: Negligible.

**Medium program (< 50 generic types):**
- Binary size: ~2 MB → ~2.1 MB.
- Impact: Acceptable.

**Large program (> 200 generic types):**
- Binary size: ~10 MB → ~12 MB.
- Impact: Noticeable.

### Measuring Binary Size

```bash
# Compile without generics
./bin/nanoc program.nano -o prog_no_generics
ls -lh prog_no_generics

# Compile with generics
./bin/nanoc program_with_generics.nano -o prog_generics
ls -lh prog_generics

# Compare
ls -lh prog_*
```

---

## Compilation Time

### Impact on Build Speed

**Without generics:**
```bash
$ time ./bin/nanoc simple.nano -o simple
real    0m0.152s
```

**With generics (5 instantiations):**
```bash
$ time ./bin/nanoc with_generics.nano -o with_generics
real    0m0.189s
```

**With generics (50 instantiations):**
```bash
$ time ./bin/nanoc heavy_generics.nano -o heavy_generics
real    0m0.431s
```

### Scaling

| Instantiations | Compile Time | Slowdown |
|----------------|--------------|----------|
| 1 | 150ms | baseline |
| 5 | 189ms | +26% |
| 10 | 253ms | +69% |
| 50 | 431ms | +187% |
| 100 | 712ms | +375% |

I typically add 5 to 8ms to the compile time for each instantiation.

### Strategies to Reduce Compile Time

1. Reuse types to minimize instantiations.
2. Use one module per generic type for separate compilation.
3. Use non-generic alternatives when you do not need peak performance.

---

## Comparison with Other Approaches

### Monomorphization (My approach, Rust, C++)

**Pros:**
- I achieve zero runtime cost.
- I provide full optimization.
- I ensure type safety.

**Cons:**
- I cause code bloat.
- I slow down compilation.
- I do not support runtime polymorphism.

### Type Erasure (Java, Go)

**Pros:**
- They produce small binaries.
- They compile quickly.
- They support runtime polymorphism.

**Cons:**
- They incur boxing overhead.
- They require runtime type checks.
- They offer less optimization.

### Comparison Table

| Feature | Monomorphization | Type Erasure | Tagged Unions |
|---------|------------------|--------------|---------------|
| **Runtime speed** | Fastest | Slower | Fast |
| **Binary size** | Large | Small | Small |
| **Compile time** | Slow | Fast | Fast |
| **Type safety** | Compile-time | Runtime | Compile-time |
| **Memory use** | Compact | Boxed | Compact |
| **Runtime generics** | No | Yes | Limited |

**Example:**

```nano
# My approach (monomorphization)
let nums: List<int> = (List_int_new)  # I generate List_int_* functions

// Java (type erasure)
List<Integer> nums = new ArrayList<>();  # Uses Object internally, boxes integers

// OCaml (tagged unions)
let nums = [1; 2; 3]  # Runtime type tags, no monomorphization
```

---

## Best Practices

### 1. Limit Generic Instantiations in Libraries

I recommend that library authors limit internal generic usage. A library that uses 20 different List types internally adds 40 KB to every program that imports it. I prefer libraries that use a maximum of 2 or 3 generic types and provide non-generic APIs for most operations.

### 2. Reuse Generic Types

Reuse types whenever possible.

```nano
# I reuse List<int> for all of these
let user_ids: List<int> = (List_int_new)
let product_ids: List<int> = (List_int_new)
let order_ids: List<int> = (List_int_new)
```

All three variables reuse the same `List<int>` instantiation I created.

### 3. Consider Non-Generic Alternatives

If you only use a generic type once in a non-critical path, consider an alternative.

```nano
# I add 2 to 3 KB for this single use
let temp: List<RareType> = (List_RareType_new)

# This hand-written array has no generic overhead
let temp: array<RareType> = (array_new 10 default_value)
```

### 4. Profile Before Optimizing

I provide tools to help you measure the impact.

```bash
# Check your binary size
ls -lh myprogram

# Identify what uses space
nm -S myprogram | grep List_ | sort -k2 -rn | head -20

# Profile my compile time
time ./bin/nanoc myprogram.nano -o myprogram --verbose
```

### 5. Document Generic Usage

I suggest documenting the instantiations you use in your modules. This helps you track the overhead, such as 7 KB for `List<int>`, `List<string>`, and `List<Config>`.

---

## Advanced Topics

### Generic Type Explosion

Nested generics can cause exponential growth.

```nano
# Each level multiplies my instantiations
let matrix: List<List<int>> = ...          # I need List_List_int and List_int
let tensor: List<List<List<int>>> = ...    # I need three types
```

I recommend avoiding deep nesting to keep the number of instantiations manageable.

### Cross-Module Generics

```nano
# module_a.nano
pub fn process_ints(list: List<int>) -> void { ... }

# module_b.nano
from "./module_a.nano" import process_ints
let nums: List<int> = (List_int_new)
(process_ints nums)
```

I ensure that both modules agree on the `List<int>` implementation to maintain consistency.

### Future: Generic Functions (Not Yet Implemented)

I plan to support generic functions in version 1.0.

```nano
# Generic function (I do not support this yet)
fn generic_swap<T>(a: T, b: T) -> (T, T) {
    return (b, a)
}

let (x, y) = (generic_swap 1 2)           # I will instantiate swap<int>
let (s1, s2) = (generic_swap "a" "b")    # I will instantiate swap<string>
```

---

## Tooling

### Viewing Generated Code

I allow you to inspect the C code I generate.

```bash
# I keep the generated C code
./bin/nanoc myprogram.nano -o myprogram --keep-c

# You can inspect my functions
cat myprogram.gen.c | grep "List_int"
```

### Measuring Impact

```bash
# Calculate size per instantiation
nm -S myprogram | grep "List_" | awk '{sum+=$2} END {print sum}'

# Count how many instantiations I created
nm myprogram | grep "List_" | cut -d_ -f2 | sort -u | wc -l
```

---

## Related Documentation

- [SPECIFICATION.md](SPECIFICATION.md) - I specify generic types here.
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - My generic syntax at a glance.
- [PERFORMANCE.md](PERFORMANCE.md) - How I tune performance.
- [DEBUGGING_GUIDE.md](DEBUGGING_GUIDE.md) - How to debug my generics.

---

**Last Updated:** February 20, 2026
**Status:** Complete
**Version:** 0.2.0+

