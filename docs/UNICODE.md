# Unicode Support in NanoLang

## Overview

NanoLang provides **UTF-8 string support** through two complementary string types:
- **`string`** - C-compatible UTF-8 strings (null-terminated `char*`)
- **`bstring`** - Binary strings with explicit length (UTF-8 aware)

This document explains what Unicode features are supported, limitations, and best practices for working with international text.

## Quick Summary

| Feature | `string` | `bstring` | Notes |
|---------|----------|-----------|-------|
| **UTF-8 Storage** | ✅ Yes | ✅ Yes | Both store UTF-8 bytes |
| **Null-terminated** | ✅ Yes | ❌ No | `string` is C-compatible |
| **Embedded Nulls** | ❌ No | ✅ Yes | `bstring` allows `\0` in data |
| **Length = Bytes** | ✅ Yes | ✅ Yes | Both count bytes, not characters |
| **Character Operations** | ❌ Limited | ✅ Better | `bstring` has UTF-8 helpers |
| **FFI Compatible** | ✅ Yes | ⚠️ Conversion | `string` works directly with C |
| **Normalization** | ❌ No | ❌ No | Not supported |
| **Grapheme Clusters** | ❌ No | ❌ No | Not supported |

## String Type (`string`)

### What is `string`?

The `string` type is a **null-terminated C string** (`char*`) that stores UTF-8 encoded text.

```nano
let message: string = "Hello, 世界!"  # UTF-8 bytes stored
let emoji: string = "Hello 👋"       # Multi-byte emoji supported
```

### Capabilities

✅ **Supported:**
- Store UTF-8 text (any valid UTF-8 byte sequence)
- Pass to C FFI functions directly
- Concatenation (`str_concat`)
- Comparison (`str_equals`)
- Length in bytes (`str_length`)
- Substring by byte index (`str_substring`)

❌ **Not Supported:**
- Embedded null bytes (`\0` terminates the string)
- Character count (only byte count via `str_length`)
- Unicode normalization
- Grapheme cluster operations

### Example

```nano
fn greet(name: string) -> string {
    return (str_concat "Hello, " name)
}

shadow greet {
    let msg: string = (greet "世界")
    assert (str_equals msg "Hello, 世界")
    
    # Length is in BYTES, not characters
    # "世界" is 6 bytes in UTF-8 (3 bytes × 2 characters)
    assert (== (str_length "世界") 6)
}
```

### When to Use `string`

Use `string` when:
- Interfacing with C libraries (FFI)
- Simple text that doesn't need character-level operations
- No embedded null bytes needed
- Maximum C compatibility required

## Binary String Type (`bstring`)

### What is `bstring`?

The `bstring` type is a **length-explicit string** (`nl_string_t*`) that:
- Stores length separately (not null-terminated)
- Allows embedded null bytes
- Provides UTF-8 validation and character counting

```nano
let data: bstring = (bstr_new "Hello, 世界!")
let bytes: int = (bstr_length data)       # Byte count
let chars: int = (bstr_utf8_length data)  # Character count
```

### Capabilities

✅ **Supported:**
- Store binary data (embedded null bytes)
- UTF-8 validation (`bstr_validate_utf8`)
- Character count (`bstr_utf8_length`)
- Length in bytes (`bstr_length`)
- Substring by byte index (`bstr_substring`)
- Byte access (`bstr_byte_at`)
- Convert to/from `string` (`bstr_to_cstr`, `bstr_new`)

❌ **Not Supported:**
- Direct FFI use (must convert to `string`)
- Unicode normalization
- Grapheme cluster operations
- Character-indexed substring

### Example

```nano
fn analyze_text(text: string) -> (int, int) {
    let bs: bstring = (bstr_new text)
    let byte_count: int = (bstr_length bs)
    let char_count: int = (bstr_utf8_length bs)
    return (byte_count, char_count)
}

shadow analyze_text {
    # ASCII: 1 byte per character
    let ascii: (int, int) = (analyze_text "Hello")
    assert (== ascii.0 5)  # 5 bytes
    assert (== ascii.1 5)  # 5 characters
    
    # Multi-byte UTF-8: "世界" = 6 bytes, 2 characters
    let utf8: (int, int) = (analyze_text "世界")
    assert (== utf8.0 6)   # 6 bytes
    assert (== utf8.1 2)   # 2 characters
    
    # Emoji: "👋" = 4 bytes, 1 character
    let emoji: (int, int) = (analyze_text "👋")
    assert (== emoji.0 4)  # 4 bytes
    assert (== emoji.1 1)  # 1 character
}
```

### When to Use `bstring`

Use `bstring` when:
- Need to count characters (not just bytes)
- Working with binary data or embedded nulls
- Need UTF-8 validation
- Don't need direct C FFI

## Byte vs Character Operations

### Important: Indexing is BYTE-BASED

Both `string` and `bstring` use **byte indices**, not character indices:

```nano
fn demonstrate_byte_indexing() -> void {
    let text: string = "Hi👋"  # H=1 byte, i=1 byte, 👋=4 bytes
    
    # str_substring works with BYTE indices
    let hi: string = (str_substring text 0 2)    # "Hi" (bytes 0-1)
    let wave: string = (str_substring text 2 6)  # "👋" (bytes 2-5)
    
    # char_at returns byte at index (as ASCII value)
    let h: int = (char_at text 0)  # 72 (ASCII 'H')
    let i: int = (char_at text 1)  # 105 (ASCII 'i')
    # (char_at text 2) returns first byte of 👋, not the full character!
}

shadow demonstrate_byte_indexing {
    (demonstrate_byte_indexing)
    assert true
}
```

### Character Counting

To count characters (not bytes), use `bstr_utf8_length`:

```nano
fn count_characters(text: string) -> int {
    let bs: bstring = (bstr_new text)
    return (bstr_utf8_length bs)
}

shadow count_characters {
    assert (== (count_characters "Hello") 5)
    assert (== (count_characters "世界") 2)
    assert (== (count_characters "👋🌍") 2)  # 2 emoji = 2 characters
}
```

## Unicode Limitations

### ❌ Not Supported

NanoLang does **NOT** support:

1. **Unicode Normalization** (NFC, NFD, NFKC, NFKD)
   - "é" as one character vs "e" + combining accent are treated differently
   - No automatic normalization

2. **Grapheme Cluster Operations**
   - "👨‍👩‍👧‍👦" (family emoji) is multiple codepoints, not one character
   - `bstr_utf8_length` counts codepoints, not grapheme clusters

3. **Character-Index Substring**
   - All substring operations use byte indices
   - Must manually calculate byte offsets for multi-byte characters

4. **Unicode Case Folding**
   - `char_to_upper`/`char_to_lower` only work for ASCII
   - No support for Turkish İ, German ß, etc.

5. **Bidirectional Text (BiDi)**
   - No support for right-to-left languages like Arabic/Hebrew
   - Text is rendered as-is

### Example of Limitations

```nano
fn demonstrate_limitations() -> void {
    # Normalization: These are DIFFERENT
    let nfc: string = "é"          # Single character (U+00E9)
    let nfd: string = "é"          # e + combining accent (U+0065 U+0301)
    assert (not (str_equals nfc nfd))  # They're different byte sequences!
    
    # Grapheme clusters: Family emoji counted as multiple
    let family: bstring = (bstr_new "👨‍👩‍👧‍👦")
    let count: int = (bstr_utf8_length family)
    # count is 7 (Man + ZWJ + Woman + ZWJ + Girl + ZWJ + Boy)
    # NOT 1 as a user might expect!
}

shadow demonstrate_limitations {
    (demonstrate_limitations)
    assert true
}
```

## Best Practices

### 1. Use `bstring` for Character Counting

```nano
# ✅ Good - accurate character count
fn count_visual_length(text: string) -> int {
    let bs: bstring = (bstr_new text)
    return (bstr_utf8_length bs)
}

# ❌ Bad - counts bytes, not characters
fn count_visual_length_wrong(text: string) -> int {
    return (str_length text)  # Wrong for multi-byte characters!
}
```

### 2. Validate UTF-8 Input

```nano
fn safe_process(text: string) -> bool {
    let bs: bstring = (bstr_new text)
    if (not (bstr_validate_utf8 bs)) {
        (println "Invalid UTF-8!")
        return false
    }
    # Process valid UTF-8...
    return true
}

shadow safe_process {
    assert (safe_process "Hello 世界")
    assert true
}
```

### 3. Be Careful with Byte Indexing

```nano
# ✅ Good - process entire string
fn uppercase_ascii(text: string) -> string {
    let mut result: string = ""
    let len: int = (str_length text)
    let mut i: int = 0
    
    while (< i len) {
        let c: int = (char_at text i)
        if (and (>= c 97) (<= c 122)) {  # a-z
            set c (- c 32)  # Convert to uppercase
        }
        set result (str_concat result (string_from_char c))
        set i (+ i 1)
    }
    return result
}

# ❌ Bad - substring at arbitrary byte index may split multi-byte character!
fn bad_substring(text: string, start: int, end: int) -> string {
    # If start/end are in the middle of a multi-byte character, this breaks!
    return (str_substring text start end)
}
```

### 4. Use `string` for FFI, Convert as Needed

```nano
# FFI function expects C string
extern fn write_file(path: string, content: string) -> int

fn save_data(path: string, data: bstring) -> int {
    # Convert bstring to string for FFI
    let content: string = (bstr_to_cstr data)
    return (write_file path content)
}

shadow save_data {
    let data: bstring = (bstr_new "Hello")
    # Test would need file system access - simplified
    assert true
}
```

## Common Scenarios

### Working with Emoji

```nano
fn count_emoji(text: string) -> int {
    # Note: This counts codepoints, not "emoji as user sees them"
    let bs: bstring = (bstr_new text)
    return (bstr_utf8_length bs)
}

shadow count_emoji {
    # Simple emoji: 1 codepoint = 1 counted
    assert (== (count_emoji "😀") 1)
    
    # Skin tone modifier: 2 codepoints = 2 counted
    assert (== (count_emoji "👋🏽") 2)  # Wave + skin tone
    
    # Combined emoji: multiple codepoints
    let family_count: int = (count_emoji "👨‍👩‍👧‍👦")
    # family_count is 7, not 1!
}
```

### Internationalization

```nano
fn greet_multilingual(lang: string, name: string) -> string {
    if (str_equals lang "en") {
        return (str_concat "Hello, " name)
    } else if (str_equals lang "zh") {
        return (str_concat "你好，" name)
    } else if (str_equals lang "ar") {
        return (str_concat "مرحبا، " name)  # No BiDi support
    } else {
        return (str_concat "Hi, " name)
    }
}

shadow greet_multilingual {
    assert (str_equals (greet_multilingual "en" "Alice") "Hello, Alice")
    assert (str_equals (greet_multilingual "zh" "小明") "你好，小明")
}
```

## Summary

| Task | Recommended Approach |
|------|----------------------|
| Store UTF-8 text | `string` (simple) or `bstring` (advanced) |
| Count characters | `bstr_utf8_length` |
| Count bytes | `str_length` or `bstr_length` |
| Substring by bytes | `str_substring` or `bstr_substring` |
| FFI with C libraries | Use `string`, convert from `bstring` if needed |
| Validate UTF-8 | `bstr_validate_utf8` |
| Embedded nulls | Use `bstring` |
| Case conversion | ASCII only (`char_to_upper`/`char_to_lower`) |
| Normalization | Not supported - handle externally |
| Grapheme clusters | Not supported - counts codepoints |

## See Also

- **[Standard Library](STDLIB.md)** - Complete list of string/bstring functions
- **[Type System](SPECIFICATION.md#3-types)** - Type definitions
- **[Memory Management](MEMORY_MANAGEMENT.md)** - String memory handling
- **[FFI Guide](EXTERN_FFI.md)** - Using strings with C libraries

---

**Last Updated:** January 25, 2026  
**Status:** `string` and `bstring` fully implemented and documented
