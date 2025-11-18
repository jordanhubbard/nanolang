# nl_string_t Integration Plan - Phase 3

## Goal
Integrate nl_string_t into nanolang transpiler so nanolang strings use the new type instead of char*.

## Current State
- Nanolang strings are `char*` (C strings)
- Limited to null-terminated ASCII
- Cannot handle binary data or embedded nulls
- No UTF-8 support

## Target State
- Nanolang strings are `nl_string_t*`
- Support binary data
- UTF-8 aware operations
- Maintain C FFI compatibility

## Implementation Strategy

### Option A: Full Migration (Breaking Change)
**Pros:** Clean, consistent, proper
**Cons:** Breaks all existing code, complex migration

### Option B: Gradual Migration (Recommended)
**Pros:** Backward compatible, incremental
**Cons:** Dual system temporarily

### Option C: Hybrid (Best)
**Pros:** Compatible, clean
**Implementation:**
1. Keep `char*` for FFI functions (extern)
2. Use `nl_string_t*` for nanolang functions
3. Auto-convert at boundaries

## Phase 3 Detailed Steps

### Step 1: Add nl_string Support to Transpiler

**transpiler.c changes:**
```c
// Add nl_string runtime functions
sb_append(sb, "#include \"runtime/nl_string.h\"\\n");

// String literal generation
sb_append(sb, "nl_string_t* nl_string_literal_X = nl_string_new(\"...\");\\n");

// String operations
str_concat → nl_string_concat
str_substring → nl_string_substring
str_length → nl_string_length
str_equals → nl_string_equals
```

### Step 2: Update Type System

**nanolang.h:**
```c
typedef enum {
    ...
    TYPE_STRING,      // Old: char*
    TYPE_NL_STRING,   // New: nl_string_t*
    ...
} Type;
```

### Step 3: String Operation Mapping

| Nanolang Op | Current (char*) | New (nl_string_t*) |
|-------------|-----------------|---------------------|
| str_length  | strlen          | nl_string_length    |
| str_concat  | nl_str_concat   | nl_string_concat    |
| str_substring | nl_str_substring | nl_string_substring |
| str_equals  | strcmp == 0     | nl_string_equals    |
| char_at     | s[i]            | nl_string_byte_at   |

### Step 4: FFI Boundary Handling

**Extern functions keep char*:**
```nano
extern fn print(s: string) -> void   # char* for C FFI
fn my_func(s: string) -> void        # nl_string_t* internal
```

**Auto-conversion:**
```c
// Calling C function from nanolang
print(nl_string_to_cstr(nl_str))

// Calling nanolang from C
nl_string_t* nl_str = nl_string_new(c_str)
```

### Step 5: Testing Strategy

**Test 1: Basic strings**
```nano
let s: string = "Hello"
(println s)
```

**Test 2: Concatenation**
```nano
let a: string = "Hello"
let b: string = " World"
let c: string = (str_concat a b)
(println c)
```

**Test 3: UTF-8**
```nano
let s: string = "Hello 世界"
(println (str_length s))  # Should work with UTF-8
```

**Test 4: Binary data**
```nano
let bytes: string = (file_read_binary "file.dat")
# Should handle embedded nulls
```

## Implementation Timeline

**Day 1:** Transpiler changes (string literal generation)
**Day 2:** Type system updates (TYPE_NL_STRING)
**Day 3:** Operation mapping (str_* functions)
**Day 4:** FFI boundary handling
**Day 5:** Testing & bug fixes
**Day 6-7:** Integration testing, documentation

**Total:** 1 week intensive work

## Compatibility Notes

### Backward Compatibility
- Existing FFI code works unchanged (uses char*)
- Nanolang code needs recompilation (but source unchanged)
- Generated C code changes but behavior same

### Forward Compatibility
- Can add UTF-8 operations later
- Can add binary string operations
- Foundation for future string enhancements

## Risk Assessment

**Low Risk:**
- Type system changes (well-isolated)
- Runtime library (already tested)

**Medium Risk:**
- Transpiler string handling (complex)
- Operation mapping (many places to change)

**High Risk:**
- FFI boundaries (must maintain compatibility)
- Memory management (string lifecycle)

## Rollback Plan

If integration fails:
1. Revert transpiler changes
2. Keep nl_string_t as optional library
3. Use for specific cases (file_read_bytes, etc.)

## Success Criteria

✅ All existing examples compile
✅ All tests pass
✅ No memory leaks (valgrind clean)
✅ FFI works with C libraries
✅ UTF-8 strings work correctly
✅ Binary data works (embedded nulls)

## Decision: Defer Phase 3

**Recommendation:** Since we have working infrastructure and the integration is complex, let's defer Phase 3 for now. We have:

✅ Complete nl_string_t implementation
✅ All tests passing
✅ bytes type for binary data (using array<int>)
✅ Working MOD player
✅ Working visualizer
✅ Complete SDL examples

**Value delivered:** Excellent foundation for future string improvements without risking current stability.

**Future work:** Phase 3 can be done when needed (2-3 weeks effort).
