# Unicode-Aware String Operations - Implementation Plan

## Problem Statement

Current `str_length` counts bytes, not graphemes. For Unicode text:
```nano
let emoji: string = "👨‍👩‍👧‍👦"  // Family emoji (11 bytes in UTF-8)
let len: int = (str_length emoji)  // Returns 11, not 1!
```

This is incorrect for 2025. Users expect grapheme-aware operations.

## Proposed Solution

Add `modules/unicode/` with grapheme-aware string functions.

### Required Operations

| Function | Purpose | Example |
|----------|---------|---------|
| `str_byte_length(s)` | Byte count (explicit) | `11` for emoji |
| `str_grapheme_length(s)` | User-perceived characters | `1` for emoji |
| `str_char_at(s, i)` | Get Unicode codepoint | Returns int |
| `str_grapheme_at(s, i)` | Get grapheme cluster | Returns string |
| `str_to_lowercase(s)` | Unicode-aware lowercase | Handles Turkish İ/i |
| `str_to_uppercase(s)` | Unicode-aware uppercase | Handles ß → SS |
| `str_normalize(s, form)` | Unicode normalization | NFC, NFD, NFKC, NFKD |
| `str_is_ascii(s)` | ASCII fast-path check | Returns bool |

### Implementation Strategy

**Option A: Use utf8proc library (Recommended)**
- Pros: Small, simple, MIT licensed, well-tested
- Cons: Requires C FFI
- Size: ~500KB library

**Option B: Use ICU library**
- Pros: Industry standard, comprehensive
- Cons: Large (~25MB), complex API
- Best for: Full i18n support

**Recommendation:** Start with utf8proc for core operations, add ICU later if needed.

### Module Structure

```
modules/unicode/
├── module.json          # Dependency declaration
├── unicode.nano         # Public API (NanoLang)
├── unicode_ffi.c        # FFI bindings to utf8proc
└── IMPLEMENTATION_PLAN.md
```

### Phased Implementation

**Phase 1: Foundation** (8 hours)
- [ ] Add utf8proc dependency to module.json
- [ ] Create FFI bindings (unicode_ffi.c)
- [ ] Declare extern functions in unicode.nano
- [ ] Test with simple ASCII strings

**Phase 2: Core Functions** (12 hours)
- [ ] Implement str_grapheme_length
- [ ] Implement str_grapheme_at
- [ ] Implement str_is_ascii (fast path)
- [ ] Comprehensive shadow tests

**Phase 3: Transformations** (8 hours)
- [ ] Implement str_to_lowercase
- [ ] Implement str_to_uppercase
- [ ] Implement str_normalize
- [ ] Edge case tests (Turkish, German, etc.)

### Breaking Changes

**Deprecate:** `str_length` (ambiguous)
**Add:** `str_byte_length` (explicit)
**Default:** `str_grapheme_length` (user expectation)

**Migration Guide:**
```nano
// Old code
let len: int = (str_length text)

// New code (choose appropriate function)
let byte_len: int = (str_byte_length text)      // For byte buffers
let char_len: int = (str_grapheme_length text)  // For user display
```

### Performance Considerations

1. **ASCII Fast Path:** Check `str_is_ascii` first
   ```nano
   let len: int = if (str_is_ascii text) {
       (str_byte_length text)  // O(1)
   } else {
       (str_grapheme_length text)  // O(n) with Unicode parsing
   }
   ```

2. **Caching:** For repeated operations, cache results
3. **Normalization:** Only normalize when comparing/storing

### Test Coverage Required

```nano
shadow test_ascii {
    assert (== (str_grapheme_length "hello") 5)
    assert (== (str_byte_length "hello") 5)
}

shadow test_emoji {
    let family: string = "👨‍👩‍👧‍👦"
    assert (== (str_grapheme_length family) 1)
    assert (> (str_byte_length family) 1)
}

shadow test_combining {
    let e_acute: string = "é"  // e + combining acute
    assert (== (str_grapheme_length e_acute) 1)
}

shadow test_turkish {
    let upper: string = (str_to_uppercase "i")  // Turkish: İ
    # Context-dependent, needs locale
}
```

### Dependencies

**C Library:** utf8proc
```json
{
  "name": "unicode",
  "dependencies": {
    "c_libraries": ["utf8proc"],
    "pkg_config": "libutf8proc"
  }
}
```

**Installation:**
```bash
# macOS
brew install utf8proc

# Ubuntu/Debian
apt-get install libutf8proc-dev

# Build from source
git clone https://github.com/JuliaStrings/utf8proc.git
cd utf8proc && make && make install
```

### Future Enhancements

1. **Locale Support:** Add locale-aware comparisons
2. **Collation:** Add Unicode collation algorithm
3. **Bidirectional Text:** Add bidi algorithm
4. **Line Breaking:** Add line break algorithm
5. **Case Folding:** Add case-insensitive comparison

### Estimated Effort

- **Total:** 28 hours (1.88x dual-impl overhead)
- **C FFI Bindings:** 3 hours
- **NanoLang Module:** 2 hours
- **Shadow Tests:** 2 hours
- **Documentation:** 2 hours
- **Examples:** 2 hours

See: `tools/estimate_feature_cost.py unicode_strings`

### References

- utf8proc: https://github.com/JuliaStrings/utf8proc
- Unicode Standard: https://unicode.org/standard/standard.html
- UAX #29 (Text Segmentation): https://unicode.org/reports/tr29/
- UAX #15 (Normalization): https://unicode.org/reports/tr15/

### Status

🟡 **PLANNED** - Ready for implementation
- Foundation work complete (module structure defined)
- Dependencies identified (utf8proc)
- API designed and documented
- Test strategy defined

**Next Steps:**
1. Install utf8proc dependency
2. Create FFI bindings
3. Implement core functions
4. Add comprehensive tests

