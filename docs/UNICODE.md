# How I Handle Unicode

## Overview

I provide UTF-8 string support through two complementary string types. I don't try to hide the underlying byte representation from you.
- **`string`** - C-compatible UTF-8 strings. These are null-terminated `char*` pointers.
- **`bstring`** - Binary strings with explicit length. I use these when I need to be UTF-8 aware or handle binary data.

This document explains what Unicode features I support, where I have limitations, and how I expect you to work with international text.

## Quick Summary

| Feature | `string` | `bstring` | Notes |
|---------|----------|-----------|-------|
| **UTF-8 Storage** | Yes | Yes | Both store UTF-8 bytes |
| **Null-terminated** | Yes | No | `string` is C-compatible |
| **Embedded Nulls** | No | Yes | `bstring` allows `\0` in data |
| **Length = Bytes** | Yes | Yes | Both count bytes, not characters |
| **Character Operations** | Limited | Better | `bstring` has UTF-8 helpers |
| **FFI Compatible** | Yes | Partial | `string` works directly with C |
| **Normalization** | No | No | I do not support this |
| **Grapheme Clusters** | No | No | I do not support this |

## My String Type (`string`)

### What is `string`?

My `string` type is a null-terminated C string (`char*`). I use it to store UTF-8 encoded text.

```nano
let message: string = "Hello, 世界!"  # UTF-8 bytes stored
let emoji: string = "Hello 👋"       # Multi-byte emoji supported
```

### Capabilities

**What I support:**
- Storing UTF-8 text as a sequence of bytes.
- Passing strings to C FFI functions directly.
- Concatenation through `str_concat`.
- Comparison through `str_equals`.
- Byte-length retrieval through `str_length`.
- Substring creation by byte index through `str_substring`.

**What I do not support:**
- Embedded null bytes. A `\0` byte terminates the string.
- Character counting. I only count bytes in `str_length`.
- Unicode normalization.
- Grapheme cluster operations.

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

### When to use `string`

I recommend using `string` when you:
- Interface with C libraries.
- Work with simple text that doesn't need character-level operations.
- Do not need embedded null bytes.
- Require maximum C compatibility.

## My Binary String Type (`bstring`)

### What is `bstring`?

I use the `bstring` type (`nl_string_t*`) when I need more control.
- I store the length separately, so I don't rely on null termination.
- I allow embedded null bytes.
- I provide UTF-8 validation and character counting.

```nano
let data: bstring = (bstr_new "Hello, 世界!")
let bytes: int = (bstr_length data)       # Byte count
let chars: int = (bstr_utf8_length data)  # Character count
```

### Capabilities

**What I support:**
- Storing binary data with embedded null bytes.
- UTF-8 validation through `bstr_validate_utf8`.
- Character counting through `bstr_utf8_length`.
- Byte-length retrieval through `bstr_length`.
- Substring creation by byte index through `bstr_substring`.
- Byte access through `bstr_byte_at`.
- Conversion to and from `string` using `bstr_to_cstr` and `bstr_new`.

**What I do not support:**
- Direct FFI use. You must convert to `string` first.
- Unicode normalization.
- Grapheme cluster operations.
- Character-indexed substring operations.

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

### When to use `bstring`

I recommend using `bstring` when you:
- Need to count characters rather than just bytes.
- Work with binary data or embedded nulls.
- Need to validate UTF-8 sequences.
- Do not need direct C FFI compatibility.

## Bytes and Character Operations

### Indexing is byte-based

I use byte indices for both `string` and `bstring`. I do not use character indices.

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

### Character counting

If you want to count characters, use my `bstr_utf8_length` function.

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

### What I do not support

I am honest about what I cannot do. I do not support:

1. **Unicode Normalization** (NFC, NFD, NFKC, NFKD). I treat "é" as one character and "e" with a combining accent as two different byte sequences. I do not perform automatic normalization.
2. **Grapheme Cluster Operations**. A family emoji like "👨‍👩‍👧‍👦" consists of multiple codepoints. My `bstr_utf8_length` function counts these codepoints. It does not count grapheme clusters as a single unit.
3. **Character-indexed Substrings**. All my substring operations use byte indices. You must calculate byte offsets yourself if you work with multi-byte characters.
4. **Unicode Case Folding**. My `char_to_upper` and `char_to_lower` functions only work for ASCII characters. I do not support Turkish İ or German ß.
5. **Bidirectional Text (BiDi)**. I do not support right-to-left languages like Arabic or Hebrew. I render text exactly as it is stored.

### Examples of my limitations

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

### 1. Use `bstring` for character counting

```nano
# I recommend this for accurate character counts.
fn count_visual_length(text: string) -> int {
    let bs: bstring = (bstr_new text)
    return (bstr_utf8_length bs)
}

# I do not recommend this. It counts bytes, not characters.
fn count_visual_length_wrong(text: string) -> int {
    return (str_length text)  # Wrong for multi-byte characters!
}
```

### 2. Validate UTF-8 input

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

### 3. Be careful with byte indexing

```nano
# This is safe because it processes the entire string.
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

# This is dangerous. Substring at an arbitrary byte index might split a character.
fn bad_substring(text: string, start: int, end: int) -> string {
    return (str_substring text start end)
}
```

### 4. Use `string` for FFI and convert when necessary

```nano
# My FFI functions expect C strings.
extern fn write_file(path: string, content: string) -> int

fn save_data(path: string, data: bstring) -> int {
    # I convert bstring to string for FFI calls.
    let content: string = (bstr_to_cstr data)
    return (write_file path content)
}

shadow save_data {
    let data: bstring = (bstr_new "Hello")
    assert true
}
```

## Common Scenarios

### Working with emoji

```nano
fn count_emoji(text: string) -> int {
    # I count codepoints here, not visual characters.
    let bs: bstring = (bstr_new text)
    return (bstr_utf8_length bs)
}

shadow count_emoji {
    # Simple emoji: 1 codepoint
    assert (== (count_emoji "😀") 1)
    
    # Skin tone modifier: 2 codepoints
    assert (== (count_emoji "👋🏽") 2)  # Wave + skin tone
    
    # Combined emoji: multiple codepoints
    let family_count: int = (count_emoji "👨‍👩‍👧‍👦")
    # I count 7 codepoints here, not 1.
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
        return (str_concat "مرحبا، " name)  # I do not support BiDi
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

| Task | My Recommended Approach |
|------|-------------------------|
| Store UTF-8 text | Use `string` for simplicity or `bstring` for advanced needs |
| Count characters | Use `bstr_utf8_length` |
| Count bytes | Use `str_length` or `bstr_length` |
| Create substring by bytes | Use `str_substring` or `bstr_substring` |
| Use C libraries (FFI) | Use `string`. Convert from `bstring` if needed |
| Validate UTF-8 | Use `bstr_validate_utf8` |
| Handle embedded nulls | Use `bstring` |
| Case conversion | ASCII only using `char_to_upper` or `char_to_lower` |
| Normalization | I do not support this. Handle it externally |
| Grapheme clusters | I do not support this. I count codepoints |

## See Also

- **[Standard Library](STDLIB.md)** - My string and bstring functions.
- **[Type System](SPECIFICATION.md#3-types)** - My type definitions.
- **[Memory Management](MEMORY_MANAGEMENT.md)** - How I handle string memory.
- **[FFI Guide](EXTERN_FFI.md)** - How I use strings with C libraries.

---

**Last Updated:** February 20, 2026  
**Status:** I have fully implemented and documented `string` and `bstring`.

