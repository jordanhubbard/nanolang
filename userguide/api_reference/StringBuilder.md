# StringBuilder API Reference

The `StringBuilder` module provides efficient string building without O(n²) concatenation overhead. In NanoLang, the `+` operator on strings allocates a new string every time it is called. When building a string from many parts — especially in a loop — use `StringBuilder` to accumulate content and produce the final string in one step.

Import the module with:

```nano
import "stdlib/StringBuilder"
```

Because NanoLang structs are immutable, all mutating operations return a new `StringBuilder`. Chain calls by reassigning to the same variable.

---

## Types

### StringBuilder

```nano
struct StringBuilder {
    buffer: string,
    length: int,
    capacity: int
}
```

Holds the accumulated content, its current character count, and the allocated capacity. All fields are read-only from user code; use the provided functions to interact with a `StringBuilder`.

**Fields:**
- `buffer` — the current accumulated string content
- `length` — the number of characters currently stored
- `capacity` — the internal capacity hint (automatically grows as needed)

---

## Functions

### StringBuilder_new

```nano
fn StringBuilder_new() -> StringBuilder
```

Creates a new, empty `StringBuilder` with the default initial capacity (`STRINGBUILDER_INITIAL_CAPACITY = 256`).

**Returns:** An empty `StringBuilder`.

**Example:**
```nano
let sb: StringBuilder = (StringBuilder_new)
```

---

### StringBuilder_with_capacity

```nano
fn StringBuilder_with_capacity(capacity: int) -> StringBuilder
```

Creates a new, empty `StringBuilder` with a specified initial capacity. Use this when you know roughly how large the final string will be, to avoid intermediate reallocations.

**Parameters:**
- `capacity` — the initial capacity hint; must be greater than 0

**Returns:** An empty `StringBuilder` with the specified capacity.

**Precondition:** `capacity > 0`

**Example:**
```nano
let sb: StringBuilder = (StringBuilder_with_capacity 1024)
```

---

### StringBuilder_append

```nano
fn StringBuilder_append(sb: StringBuilder, text: string) -> StringBuilder
```

Appends a string to the `StringBuilder` and returns the updated `StringBuilder`. If `text` is empty, the original `StringBuilder` is returned unchanged.

**Parameters:**
- `sb` — the current `StringBuilder`
- `text` — the string to append

**Returns:** A new `StringBuilder` containing the appended text.

**Example:**
```nano
let sb1: StringBuilder = (StringBuilder_new)
let sb2: StringBuilder = (StringBuilder_append sb1 "Hello")
let sb3: StringBuilder = (StringBuilder_append sb2 " World")
let result: string = (StringBuilder_to_string sb3)  # "Hello World"
```

---

### StringBuilder_append_line

```nano
fn StringBuilder_append_line(sb: StringBuilder, text: string) -> StringBuilder
```

Appends a string followed by a newline character (`\n`). Equivalent to calling `StringBuilder_append` twice.

**Parameters:**
- `sb` — the current `StringBuilder`
- `text` — the string to append before the newline

**Returns:** A new `StringBuilder` with the text and newline appended.

**Example:**
```nano
let sb1: StringBuilder = (StringBuilder_new)
let sb2: StringBuilder = (StringBuilder_append_line sb1 "Line 1")
let sb3: StringBuilder = (StringBuilder_append_line sb2 "Line 2")
let result: string = (StringBuilder_to_string sb3)  # "Line 1\nLine 2\n"
```

---

### StringBuilder_append_int

```nano
fn StringBuilder_append_int(sb: StringBuilder, n: int) -> StringBuilder
```

Converts an integer to its string representation and appends it. Equivalent to `StringBuilder_append(sb, int_to_string(n))`.

**Parameters:**
- `sb` — the current `StringBuilder`
- `n` — the integer to append

**Returns:** A new `StringBuilder` with the integer appended as text.

**Example:**
```nano
let sb1: StringBuilder = (StringBuilder_new)
let sb2: StringBuilder = (StringBuilder_append sb1 "Count: ")
let sb3: StringBuilder = (StringBuilder_append_int sb2 42)
let result: string = (StringBuilder_to_string sb3)  # "Count: 42"
```

---

### StringBuilder_append_char

```nano
fn StringBuilder_append_char(sb: StringBuilder, c: int) -> StringBuilder
```

Appends a single character specified by its ASCII value. Uses `string_from_char` internally.

**Parameters:**
- `sb` — the current `StringBuilder`
- `c` — the ASCII code point of the character to append

**Returns:** A new `StringBuilder` with the character appended.

**Example:**
```nano
let sb1: StringBuilder = (StringBuilder_new)
let sb2: StringBuilder = (StringBuilder_append_char sb1 'H')
let sb3: StringBuilder = (StringBuilder_append_char sb2 'i')
let result: string = (StringBuilder_to_string sb3)  # "Hi"
```

---

### StringBuilder_to_string

```nano
fn StringBuilder_to_string(sb: StringBuilder) -> string
```

Extracts the accumulated string content from a `StringBuilder`. This is the final step in the builder pattern.

**Parameters:**
- `sb` — the `StringBuilder` to convert

**Returns:** The current content as a `string`.

**Example:**
```nano
let sb: StringBuilder = (StringBuilder_new)
let sb2: StringBuilder = (StringBuilder_append sb "done")
let result: string = (StringBuilder_to_string sb2)  # "done"
```

---

### StringBuilder_length

```nano
fn StringBuilder_length(sb: StringBuilder) -> int
```

Returns the number of characters currently stored in the `StringBuilder`.

**Parameters:**
- `sb` — the `StringBuilder` to query

**Returns:** The current character count. Always `>= 0`.

**Example:**
```nano
let sb1: StringBuilder = (StringBuilder_new)
let sb2: StringBuilder = (StringBuilder_append sb1 "Hello")
let len: int = (StringBuilder_length sb2)  # 5
```

---

### StringBuilder_clear

```nano
fn StringBuilder_clear(sb: StringBuilder) -> StringBuilder
```

Resets the `StringBuilder` to empty while preserving its current capacity allocation.

**Parameters:**
- `sb` — the `StringBuilder` to clear

**Returns:** A new, empty `StringBuilder` with the same capacity as `sb`.

**Example:**
```nano
let sb1: StringBuilder = (StringBuilder_new)
let sb2: StringBuilder = (StringBuilder_append sb1 "temporary")
let sb3: StringBuilder = (StringBuilder_clear sb2)
let result: string = (StringBuilder_to_string sb3)  # ""
```

---

### StringBuilder_is_empty

```nano
fn StringBuilder_is_empty(sb: StringBuilder) -> bool
```

Returns `true` if the `StringBuilder` contains no characters.

**Parameters:**
- `sb` — the `StringBuilder` to check

**Returns:** `true` if `length == 0`, otherwise `false`.

**Example:**
```nano
let sb: StringBuilder = (StringBuilder_new)
let empty: bool = (StringBuilder_is_empty sb)  # true
```

---

### StringBuilder_from_parts

```nano
fn StringBuilder_from_parts(parts: array<string>) -> string
```

Concatenates all strings in an array into a single string. This is a convenience function that creates an internal `StringBuilder`, appends each element, and returns the result.

**Parameters:**
- `parts` — an array of strings to concatenate

**Returns:** A single string containing all elements of `parts` concatenated in order.

**Example:**
```nano
let parts: array<string> = (array_new 3 "")
(array_set parts 0 "Hello")
(array_set parts 1 " ")
(array_set parts 2 "World")
let result: string = (StringBuilder_from_parts parts)  # "Hello World"
```

---

### StringBuilder_join

```nano
fn StringBuilder_join(parts: array<string>, separator: string) -> string
```

Joins all strings in an array with a separator between each element (but not at the beginning or end).

**Parameters:**
- `parts` — an array of strings to join
- `separator` — the string to insert between elements

**Returns:** A single string with elements separated by `separator`.

**Example:**
```nano
let parts: array<string> = (array_new 3 "")
(array_set parts 0 "one")
(array_set parts 1 "two")
(array_set parts 2 "three")
let result: string = (StringBuilder_join parts ", ")  # "one, two, three"
```

---

### StringBuilder_repeat

```nano
fn StringBuilder_repeat(text: string, n: int) -> string
```

Repeats a string `n` times and returns the result.

**Parameters:**
- `text` — the string to repeat
- `n` — the number of repetitions; must be `>= 0`

**Returns:** A string containing `text` repeated `n` times. Returns `""` when `n = 0`.

**Precondition:** `n >= 0`

**Example:**
```nano
let result: string = (StringBuilder_repeat "ab" 3)  # "ababab"
let divider: string = (StringBuilder_repeat "-" 40)  # "----------------------------------------"
```

---

### StringBuilder_indent

```nano
fn StringBuilder_indent(level: int, spaces_per_level: int) -> string
```

Generates an indentation string of spaces for use in code generation or formatted output.

**Parameters:**
- `level` — the indentation depth; must be `>= 0`
- `spaces_per_level` — the number of spaces per indentation level; must be `>= 0`

**Returns:** A string of `level * spaces_per_level` spaces.

**Preconditions:** `level >= 0` and `spaces_per_level >= 0`

**Example:**
```nano
let indent0: string = (StringBuilder_indent 0 4)  # ""
let indent1: string = (StringBuilder_indent 1 4)  # "    "
let indent2: string = (StringBuilder_indent 2 4)  # "        "
```

---

## Constants

### STRINGBUILDER_INITIAL_CAPACITY

| Name | Type | Value |
|------|------|-------|
| `STRINGBUILDER_INITIAL_CAPACITY` | `int` | `256` |

The default initial capacity (in characters) used by `StringBuilder_new`. A `StringBuilder` automatically grows beyond this limit when needed.

### STRINGBUILDER_GROWTH_FACTOR

| Name | Type | Value |
|------|------|-------|
| `STRINGBUILDER_GROWTH_FACTOR` | `int` | `2` |

The factor by which the internal capacity doubles each time the buffer needs to grow. This results in amortized O(1) append operations.
