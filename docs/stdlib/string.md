# String Standard Library

Built-in string manipulation functions.

> Auto-generated from source. Do not edit directly.

---

## Functions

- [`str_length(s: string) -> int`](#str_length)
- [`str_concat(a: string, b: string) -> string`](#str_concat)
- [`str_substring(s: string, start: int, length: int) -> string`](#str_substring)

---

### `str_length(s: string) -> int` { #str_length }

Returns the number of characters in a string.

**Parameters:**

- `s` — The string to measure.

**Returns:** The character count of s.

**Example:**

```nano
(str_length "hello")  # => 5
(str_length "")       # => 0
```

---

### `str_concat(a: string, b: string) -> string` { #str_concat }

Concatenates two strings and returns the combined result.

**Parameters:**

- `a` — The first string.
- `b` — The second string to append to a.

**Returns:** A new string containing a followed by b.

**Example:**

```nano
(str_concat "Hello, " "World!")  # => "Hello, World!"
(str_concat "" "abc")            # => "abc"
```

---

### `str_substring(s: string, start: int, length: int) -> string` { #str_substring }

Returns a substring of s beginning at start with the given length.

**Parameters:**

- `s` — The source string.
- `start` — Zero-based start index.
- `length` — Number of characters to extract.

**Returns:** The extracted substring.

**Example:**

```nano
(str_substring "Hello, World!" 0 5)  # => "Hello"
(str_substring "Hello, World!" 7 5)  # => "World"
```

---
