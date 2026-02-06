# Chapter 24: Performance & Optimization

**Understanding compilation, memory management, and optimization techniques.**

NanoLang compiles to C and achieves performance within 5-15% of hand-written C code. This chapter explains how to understand the compilation process, manage memory efficiently, profile your programs, and avoid common performance pitfalls.

## 24.1 Understanding Compilation

### From NanoLang to C

NanoLang is a **compiled language** that transpiles to C, then uses your system's C compiler (gcc or clang) to produce native binaries. This approach provides:

- **Native performance** - No interpreter overhead
- **Portability** - Runs anywhere C compiles
- **Interoperability** - Easy FFI with C libraries
- **Debugging** - Can inspect generated C code

### Compilation Stages

The NanoLang compiler (`nanoc`) processes your code through six stages:

```
┌─────────────────┐
│  Source Code    │  (.nano file)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 1. Lexical      │  Tokenizes source into tokens
│    Analysis     │  (lexer.c / lexer.nano)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. Parsing      │  Builds Abstract Syntax Tree (AST)
│                 │  (parser_iterative.c / parser.nano)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. Type         │  Static type checking and inference
│    Checking     │  (typechecker.c / typecheck.nano)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. Shadow       │  Runs compile-time tests
│    Tests        │  (validates function correctness)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 5. Transpile    │  Two-pass C code generation
│    to C         │  (transpiler_iterative_v3_twopass.c)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 6. C Compile    │  gcc/clang produces native binary
│                 │  (-std=c99 -Wall -Wextra -Werror)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Native Binary   │  Executable program
└─────────────────┘
```

### Viewing Generated C Code

Understanding the generated C code helps with debugging and optimization. NanoLang provides several ways to inspect it:

**Keep the generated C file:**
```bash
# Saves .c file alongside binary
./bin/nanoc program.nano -o bin/program --keep-c
# Creates: bin/program and bin/program.c
```

**Save to .genC file:**
```bash
# Creates program.nano.genC for inspection
./bin/nanoc program.nano -S
```

**Print to stdout:**
```bash
# Displays generated C without compiling
./bin/nanoc program.nano -fshow-intermediate-code
```

**Verbose compilation:**
```bash
# Shows all compilation steps
./bin/nanoc program.nano -o bin/program --verbose
```

### Examples

**NanoLang source:**
```nano
fn add(a: int, b: int) -> int {
    return (+ a b)
}

fn main() -> void {
    let result: int = (add 5 3)
    print result
}
```

**Generated C (simplified):**
```c
#include <stdio.h>
#include "nl_runtime.h"

int64_t nl_add(int64_t a, int64_t b) {
    return (a + b);
}

void nl_main(void) {
    int64_t result = nl_add(5, 3);
    printf("%lld\n", result);
}

int main(int argc, char** argv) {
    nl_gc_init();
    nl_main();
    nl_gc_cleanup();
    return 0;
}
```

Key observations:
- Functions are prefixed with `nl_` to avoid C name collisions
- `int` becomes `int64_t` (64-bit integers)
- GC initialization/cleanup wraps `main()`
- Prefix operators like `(+ a b)` become infix `(a + b)`

## 24.2 Memory Management

### Stack vs Heap

NanoLang uses a **hybrid memory model** where some values live on the stack and others on the heap:

**Stack-Allocated (Fast, No GC overhead):**
| Type | Size | Notes |
|------|------|-------|
| `int` | 8 bytes | 64-bit signed integer |
| `float` | 8 bytes | 64-bit double |
| `bool` | 1 byte | true/false |
| `u8` | 1 byte | Unsigned byte |
| Small structs | ≤128 bytes | Copied by value |
| Enums | Varies | Tag + payload |

**Heap-Allocated (GC-Managed):**
| Type | Overhead | Notes |
|------|----------|-------|
| `string` | ~24 bytes | UTF-8 text, immutable |
| `bstring` | ~24 bytes | Binary data |
| `array<T>` | ~32 bytes | Dynamic arrays |
| `List<T>` | ~32 bytes | Generic lists |
| `HashMap<K,V>` | ~48 bytes | Hash tables |
| Large structs | ~16 bytes | >128 bytes |

**Rule of thumb:** Primitives and small structs are stack-allocated; collections and strings are heap-allocated.

### Automatic Memory Management (ARC)

**NEW in v2.3.0:** NanoLang uses Automatic Reference Counting (ARC) for zero-overhead memory management. No manual free() calls needed!

**How ARC Works:**
- **Owned references** (from constructors like `parse`, `new_object`) are automatically freed when going out of scope
- **Borrowed references** (from accessors like `get`, `as_string`) have zero overhead - no ref counting
- **Cycle detection** - Circular references are automatically collected
- **Deterministic cleanup** - Objects freed immediately when last reference disappears

```nano
fn efficient_memory() -> void {
    # Stack allocated - no GC overhead
    let x: int = 42
    let point: Point = Point { x: 1.0, y: 2.0 }

    # Heap allocated - ARC managed (no manual free needed!)
    let name: string = "Hello"
    let numbers: array<int> = [1, 2, 3, 4, 5]

    # When function exits:
    # - Stack variables automatically freed
    # - Heap variables: refcount decremented by ARC
    # - If refcount reaches 0, memory freed automatically
    # No manual cleanup required!
}
```

**Passing data efficiently:**
```nano
# Primitives are copied (cheap for small types)
fn double_int(x: int) -> int {
    return (* x 2)
}

# Arrays/strings are passed by reference (cheap)
fn sum_array(arr: array<int>) -> int {
    # arr is a reference - no copy made
    let mut total: int = 0
    let mut i: int = 0
    while (< i (array_length arr)) {
        set total (+ total (array_get arr i))
        set i (+ i 1)
    }
    return total
}
```

### Memory Leaks and How to Avoid Them

NanoLang's garbage collector prevents most memory leaks, but cycles can cause issues:

**Reference cycles (potential leak):**
```nano
struct Node {
    value: int,
    next: Node?  # Optional reference
}

fn create_cycle() -> void {
    let a: Node = Node { value: 1, next: none }
    let b: Node = Node { value: 2, next: some(a) }
    # If we set a.next = some(b), we create a cycle
    # The cycle collector will eventually clean this up
}
```

**Best practices:**
1. **Avoid circular references** when possible
2. **Use weak references** for back-pointers (if available)
3. **Let values go out of scope** when no longer needed
4. **Don't hold references** longer than necessary

### Profiling Memory Usage

**Check GC statistics:**
```nano
extern fn gc_print_stats() -> void

fn main() -> void {
    # ... your code ...
    
    # Print GC statistics at end
    gc_print_stats()
}
```

**Environment variables for GC tuning:**
```bash
# Set heap threshold for cycle collection (default: 10MB)
export NANO_GC_THRESHOLD=10485760

# Enable GC debug output
export NANO_GC_DEBUG=1

# Run your program
./bin/myprogram
```

**Using Valgrind for detailed analysis:**
```bash
# Memory leak detection
valgrind --leak-check=full ./bin/myprogram

# Memory profiling over time
valgrind --tool=massif ./bin/myprogram
ms_print massif.out.*
```

## 24.3 Profiling Techniques

### Time Profiling

**Basic timing:**
```bash
# Simple wall-clock timing
time ./bin/myprogram

# Output:
# real    0m2.345s  (wall clock)
# user    0m2.100s  (CPU time in user mode)
# sys     0m0.123s  (CPU time in kernel)
```

**Built-in profiling with -pg:**
```bash
# Compile with profiling enabled
./bin/nanoc program.nano -o bin/program -pg

# Run program (generates profiling data)
./bin/program

# JSON profiling output appears on stderr
./bin/program 2> profile.json
```

See [Chapter 8: LLM-Powered Profiling](../08_profiling.md) for comprehensive profiling documentation.

### Memory Profiling

**Valgrind massif (heap profiler):**
```bash
# Profile heap usage over time
valgrind --tool=massif ./bin/myprogram

# Analyze results
ms_print massif.out.12345

# Key metrics:
# - Peak heap usage
# - Allocation sites
# - Memory timeline
```

**AddressSanitizer (compile-time):**
```bash
# Compile with ASan
./bin/nanoc program.nano -o bin/program -fsanitize=address

# Run - detects memory errors
./bin/program
```

### Hotspot Identification

When profiling shows a function consuming significant time, investigate:

1. **Algorithm complexity** - Is it O(n²) when O(n) is possible?
2. **Loop iterations** - Can iterations be reduced?
3. **Allocations** - Is memory being allocated inside hot loops?
4. **Function calls** - Can expensive calls be hoisted out of loops?

**Example analysis:**
```nano
# Profile shows this takes 80% of runtime
fn process_items(items: array<Item>) -> void {
    let count: int = (array_length items)
    let mut i: int = 0
    while (< i count) {
        let item: Item = (array_get items i)
        
        # Suspect: Is expensive_lookup O(n)?
        let data: Data = (expensive_lookup item.id)
        
        # Suspect: String concat in loop?
        set log (str_concat log item.name)
        
        set i (+ i 1)
    }
}
```

**Optimization opportunities:**
- Cache `expensive_lookup` results in a HashMap
- Build strings using array, then join once at end

### Tools and Techniques

| Tool | Purpose | Command |
|------|---------|---------|
| `time` | Wall-clock timing | `time ./program` |
| `-pg` flag | LLM-ready profiling | `nanoc -pg` then run |
| `gprof` | Call graph profiling | `gprof program gmon.out` |
| `perf` | Linux performance counters | `perf record ./program` |
| `valgrind` | Memory analysis | `valgrind --tool=massif` |
| `Instruments` | macOS profiling GUI | `open -a Instruments` |

## 24.4 Common Pitfalls

### Performance Anti-Patterns

**1. Premature optimization**
```nano
# ❌ Don't optimize without measuring
fn overly_clever() -> int {
    # Bit hacks that obscure intent
    return (& (>> x 31) 1)  # Just use (< x 0)!
}

# ✅ Write clear code first, optimize measured hotspots
fn clear_intent(x: int) -> bool {
    return (< x 0)
}
```

**2. Ignoring algorithm complexity**
```nano
# ❌ O(n²) - checking every pair
fn has_duplicates_slow(arr: array<int>) -> bool {
    let n: int = (array_length arr)
    let mut i: int = 0
    while (< i n) {
        let mut j: int = (+ i 1)
        while (< j n) {
            if (== (array_get arr i) (array_get arr j)) {
                return true
            }
            set j (+ j 1)
        }
        set i (+ i 1)
    }
    return false
}

# ✅ O(n) - use a set
fn has_duplicates_fast(arr: array<int>) -> bool {
    let seen: HashMap<int, bool> = (hashmap_new)
    let n: int = (array_length arr)
    let mut i: int = 0
    while (< i n) {
        let val: int = (array_get arr i)
        if (hashmap_contains seen val) {
            return true
        }
        hashmap_set seen val true
        set i (+ i 1)
    }
    return false
}
```

### String Building in Loops

This is the **most common performance mistake** in NanoLang:

```nano
# ❌ SLOW - O(n²) time complexity!
fn build_csv_slow(items: array<Item>) -> string {
    let mut result: string = ""
    let n: int = (array_length items)
    let mut i: int = 0
    while (< i n) {
        let item: Item = (array_get items i)
        # Each concat copies the ENTIRE string so far
        set result (str_concat result item.name)
        set result (str_concat result ",")
        set i (+ i 1)
    }
    return result
}
# For 10,000 items: ~50 million character copies!
```

**Why it's slow:** Each `str_concat` creates a new string and copies all previous characters. With n items, you copy approximately n²/2 characters total.

```nano
# ✅ FAST - O(n) time complexity
fn build_csv_fast(items: array<Item>) -> string {
    let mut parts: array<string> = []
    let n: int = (array_length items)
    let mut i: int = 0
    while (< i n) {
        let item: Item = (array_get items i)
        set parts (array_push parts item.name)
        set i (+ i 1)
    }
    # Single join at the end
    return (str_join parts ",")
}
# For 10,000 items: ~10,000 character copies (1000x faster!)
```

### Unnecessary Allocations

**Allocations in hot loops:**
```nano
# ❌ Creates new Point every iteration
fn move_particles_slow(particles: array<Particle>) -> void {
    let n: int = (array_length particles)
    let mut i: int = 0
    while (< i n) {
        let p: Particle = (array_get particles i)
        # New Vec2 allocation each iteration
        let velocity: Vec2 = Vec2 { x: p.vx, y: p.vy }
        # ... use velocity ...
        set i (+ i 1)
    }
}

# ✅ Reuse or avoid allocation
fn move_particles_fast(particles: array<Particle>) -> void {
    let n: int = (array_length particles)
    let mut i: int = 0
    while (< i n) {
        let p: Particle = (array_get particles i)
        # Use fields directly - no allocation
        let new_x: float = (+ p.x p.vx)
        let new_y: float = (+ p.y p.vy)
        # ... update particle ...
        set i (+ i 1)
    }
}
```

### Collection Choices

**Choose the right collection for your access pattern:**

| Pattern | Best Collection | Why |
|---------|----------------|-----|
| Sequential access | `array<T>` | Cache-friendly, O(1) index |
| Key-value lookup | `HashMap<K,V>` | O(1) average lookup |
| Membership test | `HashMap<T,bool>` | O(1) contains check |
| FIFO queue | `array<T>` with index | O(1) push/pop from end |
| Sorted data | `array<T>` + sort | Binary search O(log n) |

**Example - membership testing:**
```nano
# ❌ O(n) per lookup
fn is_valid_slow(word: string, valid_words: array<string>) -> bool {
    let n: int = (array_length valid_words)
    let mut i: int = 0
    while (< i n) {
        if (== word (array_get valid_words i)) {
            return true
        }
        set i (+ i 1)
    }
    return false
}

# ✅ O(1) per lookup
fn is_valid_fast(word: string, valid_set: HashMap<string, bool>) -> bool {
    return (hashmap_contains valid_set word)
}
```

## 24.5 Performance Checklist

Before optimizing, run through this checklist:

- [ ] **Measure first** - Is performance actually a problem?
- [ ] **Profile** - Where is time actually spent?
- [ ] **Check algorithms** - Is Big-O complexity optimal?
- [ ] **String concatenation** - Using array + join for loops?
- [ ] **Collection choice** - Right data structure for access pattern?
- [ ] **Allocations** - Avoiding allocations in hot loops?
- [ ] **C optimization** - Compiling generated C with `-O2`/`-O3`?

## 24.6 Summary

| Aspect | Recommendation |
|--------|---------------|
| **Compilation** | Use `--keep-c` to inspect generated code |
| **Memory** | Prefer stack types; let GC handle heap |
| **Profiling** | Always measure before optimizing |
| **Strings** | Never concatenate in loops - use array + join |
| **Collections** | Match collection to access pattern |
| **Algorithms** | Big-O matters more than micro-optimization |

**Performance target:** NanoLang programs typically run within 5-15% of equivalent hand-written C code. If your program is significantly slower, check for the common pitfalls described in this chapter.

---

**Previous:** [Chapter 23: Higher-Level Patterns](23_patterns.html)  
**Next:** [Chapter 25: Contributing & Extending](25_contributing.html)
