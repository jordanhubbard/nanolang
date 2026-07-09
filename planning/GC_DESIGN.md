# Garbage Collection and Dynamic Arrays Design

**Date**: 2025-11-16  
**Status**: Implementation Plan  
**Goal**: Add dynamic object management without exposing pointers

---

## Design Principles

1. **No Exposed Pointers**: Users never see memory addresses
2. **Automatic Memory Management**: GC handles allocation and deallocation
3. **Simple and Predictable**: Reference counting + periodic cycle detection
4. **Zero-Cost When Not Used**: Static arrays don't pay GC overhead
5. **Deterministic**: Predictable GC behavior for game loops

---

## GC Architecture

### Reference Counting + Cycle Detection

**Reference Counting** (primary mechanism):
- Each object tracks reference count
- Increment on assignment
- Decrement when variable goes out of scope
- Free immediately when count reaches 0
- Fast and deterministic

**Cycle Detection** (backup mechanism):
- Periodic mark-and-sweep for cycles
- Only needed for circular references
- Run during idle time or allocation pressure
- Optional: can be disabled for performance

### Object Types Managed by GC

1. **Dynamic Arrays** (new)
   - Variable-length arrays
   - Grow/shrink automatically
   - Support push/pop/insert/remove

2. **Strings** (already heap-allocated, now GC-managed)
   - Immutable strings on heap
   - Automatic deduplication (optional)

3. **Structs** (future)
   - Heap-allocated struct instances
   - Nested object references

---

## Implementation Strategy

### Phase 1: GC Runtime (src/runtime/gc.c)

```c
// GC Object Header (prepended to all GC objects)
typedef struct {
    uint32_t ref_count;
    uint32_t mark : 1;      // For mark-and-sweep
    uint32_t type : 7;      // Object type
    uint32_t size : 24;     // Object size in bytes
} GCHeader;

// GC Functions
void* gc_alloc(size_t size, uint8_t type);
void gc_retain(void* ptr);
void gc_release(void* ptr);
void gc_collect_cycles();  // Periodic cycle collection
```

### Phase 2: Dynamic Array Type

```c
// Dynamic array structure
typedef struct {
    GCHeader header;
    int64_t length;
    int64_t capacity;
    uint8_t element_type;  // VAL_INT, VAL_FLOAT, VAL_STRING, etc.
    void* data;            // Element storage
} DynamicArray;

// Dynamic array operations
DynamicArray* dyn_array_new(uint8_t element_type);
DynamicArray* dyn_array_push(DynamicArray* arr, Value val);
Value dyn_array_pop(DynamicArray* arr);
DynamicArray* dyn_array_remove_at(DynamicArray* arr, int64_t index);
DynamicArray* dyn_array_insert_at(DynamicArray* arr, int64_t index, Value val);
```

### Phase 3: Language Integration

**Syntax (no changes needed)**:
```nano
# Static arrays (existing, no GC)
let static_arr: array<int> = [1, 2, 3]

# Dynamic arrays (new, GC-managed)
let mut dynamic_arr: array<int> = []
set dynamic_arr (array_push dynamic_arr 42)
set dynamic_arr (array_push dynamic_arr 43)
let val: int = (array_pop dynamic_arr)
```

**Type System**:
- Arrays are already a first-class type
- No syntax changes needed
- Compiler determines static vs dynamic based on usage:
  - Literal `[1, 2, 3]` → static
  - Empty `[]` or result of `array_push` → dynamic

**Transpiler Changes**:
- Generate `gc_retain()` calls on assignment
- Generate `gc_release()` calls when variables go out of scope
- Wrap function returns with GC management

---

## Memory Management Rules

### Automatic Reference Counting

```nano
fn example() -> void {
    let mut arr: array<int> = []        # Alloc: ref_count = 1
    set arr (array_push arr 42)         # Old ref_count--, new ref_count++
    let arr2: array<int> = arr          # arr ref_count++
    # At end of function:
    # arr ref_count-- (may free if 0)
    # arr2 ref_count-- (may free if 0)
}
```

### Function Boundaries

```nano
fn create_array() -> array<int> {
    let mut arr: array<int> = []
    set arr (array_push arr 1)
    return arr                          # Transfer ownership, ref_count stays 1
}

fn use_array() -> void {
    let my_arr: array<int> = (create_array)  # Takes ownership, ref_count = 1
    # ... use array ...
}  # my_arr ref_count--, freed if 0
```

### Cycles (rare, handled by periodic GC)

```nano
# If we later add struct references, cycles possible:
struct Node {
    value: int,
    next: Node  # Could create cycle
}
# Periodic mark-and-sweep handles this
```

---

## Performance Characteristics

### Reference Counting Overhead

- **Cost per assignment**: ~1 instruction (inc/dec counter)
- **Cost per scope exit**: ~1 instruction per variable
- **Benefit**: Immediate deallocation, no GC pauses

### Cycle Collection Overhead

- **Frequency**: Only when needed (allocation pressure or manual trigger)
- **Cost**: O(live objects) mark + O(live objects) sweep
- **Typical**: 1-5ms per 10,000 objects
- **Avoidable**: Don't create cycles (rare in game code)

### Memory Overhead

- **Per dynamic array**: 24 bytes header + data
- **Per string**: 12 bytes header + string data
- **Static arrays**: 0 bytes overhead (no GC)

---

## Implementation Plan

### Week 1: Core GC Runtime

**Day 1-2**: GC Infrastructure
- [ ] `src/runtime/gc.c` - GC allocator
- [ ] `src/runtime/gc.h` - GC public API
- [ ] Reference counting implementation
- [ ] Free list management

**Day 3-4**: Dynamic Arrays
- [ ] `src/runtime/dyn_array.c` - Dynamic array implementation
- [ ] `array_push`, `array_pop`, `array_remove_at`
- [ ] `array_insert_at`, `array_clear`, `array_filter`
- [ ] Growth strategy (2x on overflow)

**Day 5**: String GC Integration
- [ ] Migrate strings to GC
- [ ] String deduplication (optional optimization)

### Week 2: Language Integration

**Day 1-2**: Transpiler Updates
- [ ] Generate `gc_retain()`/`gc_release()` calls
- [ ] Track variable lifetimes
- [ ] Function boundary handling

**Day 3**: Evaluator Updates
- [ ] Interpreter GC integration
- [ ] Shadow test execution with GC

**Day 4-5**: Testing & Documentation
- [ ] Unit tests for GC
- [ ] Example programs
- [ ] STDLIB.md updates
- [ ] Performance benchmarks

### Week 3: Cycle Detection (Optional)

**Day 1-2**: Mark-and-Sweep
- [ ] Mark phase implementation
- [ ] Sweep phase implementation
- [ ] Root set tracking

**Day 3-5**: Integration & Testing
- [ ] Periodic GC triggers
- [ ] Cycle test cases
- [ ] Performance tuning

---

## New Stdlib Functions

### Dynamic Array Operations

```nano
# Create empty dynamic array
fn array_new<T>() -> array<T>

# Add element to end (returns new array, old array invalid)
fn array_push<T>(arr: mut array<T>, value: T) -> array<T>

# Remove and return last element
fn array_pop<T>(arr: mut array<T>) -> T

# Remove element at index (returns new array)
fn array_remove_at<T>(arr: mut array<T>, index: int) -> array<T>

# Insert element at index
fn array_insert_at<T>(arr: mut array<T>, index: int, value: T) -> array<T>

# Clear all elements
fn array_clear<T>(arr: mut array<T>) -> array<T>

# Filter elements by predicate
fn array_filter<T>(arr: array<T>, pred: fn(T) -> bool) -> array<T>

# Map elements with function
fn array_map<T, U>(arr: array<T>, f: fn(T) -> U) -> array<U>

# Get capacity (how many elements before realloc)
fn array_capacity<T>(arr: array<T>) -> int

# Reserve capacity (pre-allocate)
fn array_reserve<T>(arr: mut array<T>, capacity: int) -> array<T>
```

---

## Safety Guarantees

1. **No Use-After-Free**: Reference counting prevents dangling references
2. **No Memory Leaks**: Cycle detection catches circular references
3. **No Double-Free**: Reference counting ensures single deallocation
4. **No Buffer Overflows**: Bounds checking on all array access
5. **Type Safety**: Arrays are homogeneous, type-checked at compile time

---

## Migration Path

### Backward Compatibility

**Existing code continues to work**:
```nano
# Static arrays (unchanged)
let nums: array<int> = [1, 2, 3, 4, 5]
let val: int = (at nums 2)
```

**New code uses dynamic arrays**:
```nano
# Dynamic arrays (new)
let mut nums: array<int> = []
set nums (array_push nums 1)
set nums (array_push nums 2)
```

### Opt-In Performance

**Static arrays remain zero-overhead**:
- No GC metadata
- Direct memory access
- Stack or data segment allocation
- Perfect for fixed-size data

**Dynamic arrays pay for flexibility**:
- GC metadata (24 bytes)
- Reference counting (~2 instructions per operation)
- Heap allocation
- Perfect for variable-size data

---

## Success Criteria

✅ **Asteroids game works** with dynamic entities  
✅ **No manual memory management** in user code  
✅ **Predictable performance** for game loops  
✅ **Zero cost for static arrays** (backward compat)  
✅ **<5% overhead** for dynamic arrays vs manual malloc  
✅ **Comprehensive tests** for GC correctness  
✅ **Clear documentation** for users

---

## Timeline

- **Week 1**: Core GC + Dynamic Arrays (40 hours)
- **Week 2**: Language Integration + Testing (40 hours)
- **Week 3**: Cycle Detection + Polish (40 hours)
- **Total**: ~120 hours (~3 weeks full-time)

---

## Next Steps

1. ✅ Create this design document
2. → Implement `src/runtime/gc.c` (reference counting)
3. → Implement `src/runtime/dyn_array.c` (dynamic arrays)
4. → Update transpiler for GC integration
5. → Test with asteroids example
6. → Document in STDLIB.md

**Status**: Ready to begin implementation 🚀

