# Binary String Type Design

## Problem
Current strings are C strings (null-terminated):
- Cannot store binary data (embedded nulls)
- No UTF-8 length tracking
- Unsafe strlen operations

## Proposed Solution: String Structure

```c
typedef struct {
    char *data;      // Raw bytes (may contain nulls)
    size_t length;   // Byte length (NOT null-terminated)
    size_t capacity; // Allocated size
    bool is_utf8;    // UTF-8 validation flag
} nl_string_t;
```

## Operations Needed

**Creation:**
- nl_string_new(const char *cstr)
- nl_string_new_binary(const void *data, size_t len)
- nl_string_from_utf8(const char *utf8)

**Access:**
- nl_string_length(nl_string_t *s) → byte length
- nl_string_char_at(nl_string_t *s, size_t index) → byte
- nl_string_utf8_length(nl_string_t *s) → character count
- nl_string_utf8_char_at(nl_string_t *s, size_t char_index) → codepoint

**Manipulation:**
- nl_string_concat(nl_string_t *a, nl_string_t *b)
- nl_string_substring(nl_string_t *s, size_t start, size_t len)
- nl_string_equals(nl_string_t *a, nl_string_t *b)

**Conversion:**
- nl_string_to_cstr(nl_string_t *s) → const char* (adds \\0)
- nl_string_to_binary(nl_string_t *s, size_t *out_len) → void*

## Implementation Phases

### Phase 1: Structure & Core (Week 1)
- Define nl_string_t structure
- Implement creation/destruction
- Basic operations (length, concat, substring)

### Phase 2: UTF-8 Support (Week 2)
- UTF-8 validation
- Character counting
- Codepoint iteration
- UTF-8 substring

### Phase 3: Integration (Week 3)
- Update transpiler to use nl_string_t
- Convert all string operations
- Update nanolang type system
- Maintain C string compatibility for FFI

### Phase 4: Optimization (Week 4)
- String interning (deduplicate literals)
- Copy-on-write for efficiency
- Small string optimization
- Arena allocation

## Backward Compatibility

**Option A: Gradual Migration**
- Keep C strings for FFI
- nl_string_t for internal use
- Automatic conversion at boundaries

**Option B: All New Strings**
- Replace all string types
- FFI requires explicit conversion
- Breaking change

## Recommendation
Start with Phase 1 (core structure) as proof of concept.
Test with binary file operations.
Then decide on full integration strategy.

## Estimated Effort
- Phase 1: 2-3 days (core implementation)
- Phase 2: 2-3 days (UTF-8 support)
- Phase 3: 5-7 days (integration)
- Phase 4: 3-5 days (optimization)

**Total: 2-3 weeks of focused work**

## Alternative: Simpler Approach
For immediate needs (MOD files):
- Add `bytes` type = array<int> with 0-255 values
- Keep strings as-is
- Separate types for different use cases
- Easier to implement (2-3 days)

## Decision Point
Full string overhaul OR simpler bytes type?
