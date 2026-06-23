# 13.3 StringBuilder — Efficient String Building

**Amortized O(n) string assembly for loops, templates, and large output generation.**

The `modules/std/collections/stringbuilder.nano` module provides a `StringBuilder` type and a collection of functions for building strings incrementally without the quadratic cost of repeated concatenation. Each `sb_append` call mutates the builder in-place and returns the same `StringBuilder` — the underlying C buffer grows geometrically so that the total cost of N appends is O(N) rather than O(N²).

Use StringBuilder whenever you are assembling a string in a loop, generating a document or report, or concatenating more than a handful of fragments. For simple two-or-three piece joins, plain `+` is fine.

---

## Quick Start

```nano
from "std/collections/stringbuilder.nano" import sb_new, sb_append,
                                                   sb_to_string, StringBuilder

fn build_greeting(names: array<string>) -> string {
    let mut sb: StringBuilder = (sb_new)
    set sb (sb_append sb "Hello to: ")

    let mut i: int = 0
    while (< i (array_length names)) {
        if (!= i 0) {
            set sb (sb_append sb ", ")
        }
        set sb (sb_append sb (at names i))
        set i (+ i 1)
    }

    set sb (sb_append sb "!")
    return (sb_to_string sb)
}

shadow build_greeting {
    let names: array<string> = (array_new 3 "")
    (array_set names 0 "Alice")
    (array_set names 1 "Bob")
    (array_set names 2 "Carol")
    let result: string = (build_greeting names)
    assert (== result "Hello to: Alice, Bob, Carol!")
}
```

---

## Import

```nano
from "std/collections/stringbuilder.nano" import sb_new, sb_with_capacity,
                                                   sb_append, sb_append_line,
                                                   sb_append_int, sb_append_char,
                                                   sb_to_string, sb_length, sb_capacity,
                                                   sb_clear, sb_is_empty, sb_free,
                                                   sb_from_parts, sb_join,
                                                   sb_repeat, sb_indent,
                                                   StringBuilder
```

Import only the functions you need. `StringBuilder` is the struct type; import it to annotate variables.

---

## The Performance Case

String concatenation with `+` is immutable and copies the entire string on every call. Building a 1,000-character string by appending one character at a time:

```
iteration 1:  copy 1 byte
iteration 2:  copy 2 bytes
...
iteration N:  copy N bytes
total:        N*(N+1)/2 copies  →  O(N²)
```

StringBuilder avoids this by maintaining an internal buffer that grows geometrically (doubling when full). Each append copies only the new fragment, and the buffer is reallocated infrequently:

```
N appends:    ~N total bytes copied  →  O(N) amortized
```

**Rule of thumb:** If you are concatenating strings in a loop — even a short one — use StringBuilder. If you are joining two or three literals or variables outside a loop, `+` is simpler and perfectly fast.

---

## Key Concept: Mutable Handle Semantics

> **`StringBuilder` wraps an opaque C handle.** Unlike plain NanoLang structs, the underlying buffer is mutated in-place by every `sb_append` call. `sb_append` returns the same `StringBuilder` value (same handle) for chaining convenience, but the mutation has already occurred — there is no separate "before" and "after" value.
>
> This means two variables that hold the same `StringBuilder` (e.g. after an assignment without `sb_clear`) share the same underlying buffer. In practice, always use a single `mut` variable and `set` it:
>
> ```nano
> let mut sb: StringBuilder = (sb_new)
> set sb (sb_append sb "Hello")   # mutates buffer, returns same sb
> set sb (sb_append sb " World")
> let result: string = (sb_to_string sb)  # "Hello World"
> ```
>
> When you are done with a builder that you created, call `sb_free` to release the C buffer. Builders created by `sb_from_parts`, `sb_join`, `sb_repeat`, and `sb_indent` are freed internally — do not free them again.

---

## API Reference

### `sb_new`

```nano
fn sb_new() -> StringBuilder
```

Creates a new, empty `StringBuilder` with a default initial capacity. The capacity grows automatically as content is appended.

**Returns:** An empty `StringBuilder` with `length = 0`.

**Example:**

```nano
from "std/collections/stringbuilder.nano" import sb_new, sb_length, StringBuilder

fn example_new() -> bool {
    let sb: StringBuilder = (sb_new)
    return (== (sb_length sb) 0)
}

shadow example_new {
    assert (example_new)
}
```

---

### `sb_with_capacity`

```nano
fn sb_with_capacity(capacity: int) -> StringBuilder
```

Creates an empty `StringBuilder` pre-sized to `capacity` bytes. Use this when you know approximately how large the final string will be; it avoids intermediate reallocations.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `capacity` | `int` | Initial buffer size in bytes. |

**Example:**

```nano
from "std/collections/stringbuilder.nano" import sb_with_capacity, sb_append,
                                                   sb_to_string, sb_free, StringBuilder

fn build_large_report(line_count: int) -> string {
    # Pre-size for roughly 80 chars per line
    let estimated: int = (* line_count 80)
    let mut sb: StringBuilder = (sb_with_capacity estimated)
    let mut i: int = 0
    while (< i line_count) {
        set sb (sb_append sb (+ "Line " (+ (int_to_string i) "\n")))
        set i (+ i 1)
    }
    let result: string = (sb_to_string sb)
    (sb_free sb)
    return result
}
```

---

### `sb_append`

```nano
fn sb_append(sb: StringBuilder, text: string) -> StringBuilder
```

Appends `text` to the builder's internal buffer (mutating it in-place) and returns `sb`. The returned value is the same `StringBuilder` handle, not a new one.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `sb` | `StringBuilder` | The builder to append to |
| `text` | `string` | Text to append |

**Returns:** The same `StringBuilder`, for convenient chaining with `set`.

**Example:**

```nano
from "std/collections/stringbuilder.nano" import sb_new, sb_append,
                                                   sb_to_string, sb_free, StringBuilder

fn chain_append() -> string {
    let mut sb: StringBuilder = (sb_new)
    set sb (sb_append sb "one")
    set sb (sb_append sb ", ")
    set sb (sb_append sb "two")
    set sb (sb_append sb ", ")
    set sb (sb_append sb "three")
    let result: string = (sb_to_string sb)
    (sb_free sb)
    return result
}

shadow chain_append {
    assert (== (chain_append) "one, two, three")
}
```

---

### `sb_append_line`

```nano
fn sb_append_line(sb: StringBuilder, text: string) -> StringBuilder
```

Appends `text` followed by a newline character (`\n`). Equivalent to two `sb_append` calls but more convenient when building multi-line output.

**Example:**

```nano
from "std/collections/stringbuilder.nano" import sb_new, sb_append_line,
                                                   sb_to_string, sb_free, StringBuilder

fn build_poem() -> string {
    let mut sb: StringBuilder = (sb_new)
    set sb (sb_append_line sb "Roses are red,")
    set sb (sb_append_line sb "Violets are blue,")
    set sb (sb_append_line sb "NanoLang is fast,")
    set sb (sb_append_line sb "And memory-safe too.")
    let result: string = (sb_to_string sb)
    (sb_free sb)
    return result
}

shadow build_poem {
    let poem: string = (build_poem)
    assert (str_contains poem "NanoLang")
}
```

---

### `sb_append_int`

```nano
fn sb_append_int(sb: StringBuilder, n: int) -> StringBuilder
```

Converts `n` to its decimal string representation and appends it. Equivalent to `(sb_append sb (int_to_string n))` but reads more naturally in numeric formatting code.

**Example:**

```nano
from "std/collections/stringbuilder.nano" import sb_new, sb_append,
                                                   sb_append_int, sb_to_string,
                                                   sb_free, StringBuilder

fn format_coords(x: int, y: int) -> string {
    let mut sb: StringBuilder = (sb_new)
    set sb (sb_append sb "(")
    set sb (sb_append_int sb x)
    set sb (sb_append sb ", ")
    set sb (sb_append_int sb y)
    set sb (sb_append sb ")")
    let result: string = (sb_to_string sb)
    (sb_free sb)
    return result
}

shadow format_coords {
    assert (== (format_coords 10 20) "(10, 20)")
    assert (== (format_coords -5 0) "(-5, 0)")
}
```

---

### `sb_append_char`

```nano
fn sb_append_char(sb: StringBuilder, c: int) -> StringBuilder
```

Appends a single character given by its ASCII integer value. Use character literals (`'A'`, `'\n'`) for readability.

**Example:**

```nano
from "std/collections/stringbuilder.nano" import sb_new, sb_append,
                                                   sb_append_char, sb_to_string,
                                                   sb_free, StringBuilder

fn surround_with_quotes(s: string) -> string {
    let mut sb: StringBuilder = (sb_new)
    set sb (sb_append_char sb '"')
    set sb (sb_append sb s)
    set sb (sb_append_char sb '"')
    let result: string = (sb_to_string sb)
    (sb_free sb)
    return result
}

shadow surround_with_quotes {
    assert (== (surround_with_quotes "hello") "\"hello\"")
}
```

---

### `sb_to_string`

```nano
fn sb_to_string(sb: StringBuilder) -> string
```

Returns the accumulated content as a plain `string`. Because the buffer is mutable, the returned string reflects all appends made up to the point of the call. You can call `sb_to_string` multiple times; each call returns a snapshot of the current content.

**Example:**

```nano
from "std/collections/stringbuilder.nano" import sb_new, sb_append,
                                                   sb_to_string, sb_free, StringBuilder

fn snapshot_example() -> bool {
    let mut sb: StringBuilder = (sb_new)
    set sb (sb_append sb "Hello")
    let partial: string = (sb_to_string sb)  # "Hello"

    set sb (sb_append sb " World")
    let full: string = (sb_to_string sb)     # "Hello World"

    (sb_free sb)
    return (and (== partial "Hello") (== full "Hello World"))
}

shadow snapshot_example {
    assert (snapshot_example)
}
```

---

### `sb_length`

```nano
fn sb_length(sb: StringBuilder) -> int
```

Returns the current number of bytes in the builder.

**Example:**

```nano
from "std/collections/stringbuilder.nano" import sb_new, sb_append,
                                                   sb_length, sb_free, StringBuilder

fn length_demo() -> bool {
    let mut sb: StringBuilder = (sb_new)
    assert (== (sb_length sb) 0)
    set sb (sb_append sb "hello")
    assert (== (sb_length sb) 5)
    set sb (sb_append sb " world")
    assert (== (sb_length sb) 11)
    (sb_free sb)
    return true
}

shadow length_demo {
    assert (length_demo)
}
```

---

### `sb_capacity`

```nano
fn sb_capacity(sb: StringBuilder) -> int
```

Returns the current allocated buffer capacity in bytes. This is always at least as large as `sb_length`. Useful for diagnostics or pre-sizing decisions.

---

### `sb_clear`

```nano
fn sb_clear(sb: StringBuilder) -> StringBuilder
```

Resets the builder's content to empty while preserving the allocated capacity. The returned value is the same `StringBuilder` handle. Useful for reusing a builder across multiple formatting passes without paying for reallocation.

**Example:**

```nano
from "std/collections/stringbuilder.nano" import sb_new, sb_append, sb_clear,
                                                   sb_to_string, sb_length,
                                                   sb_free, StringBuilder

fn reuse_builder() -> bool {
    let mut sb: StringBuilder = (sb_new)
    set sb (sb_append sb "first pass")
    let first: string = (sb_to_string sb)

    set sb (sb_clear sb)
    assert (== (sb_length sb) 0)

    set sb (sb_append sb "second pass")
    let second: string = (sb_to_string sb)

    (sb_free sb)
    return (and (== first "first pass") (== second "second pass"))
}

shadow reuse_builder {
    assert (reuse_builder)
}
```

---

### `sb_is_empty`

```nano
fn sb_is_empty(sb: StringBuilder) -> bool
```

Returns `true` if the builder contains no characters. Equivalent to `(== (sb_length sb) 0)` but more expressive.

---

### `sb_free`

```nano
fn sb_free(sb: StringBuilder) -> void
```

Releases the underlying C buffer. Call this when you are finished with a `StringBuilder` that you created with `sb_new` or `sb_with_capacity`. After calling `sb_free`, do not use the builder again.

Builders created internally by `sb_from_parts`, `sb_join`, `sb_repeat`, and `sb_indent` are freed by those functions — do not free them yourself.

---

### `sb_from_parts`

```nano
fn sb_from_parts(parts: array<string>) -> string
```

Efficiently concatenates all strings in `parts` in order and returns the result as a plain `string`. This is a standalone utility — it creates and frees its own internal builder, so you do not need to manage one yourself.

**Example:**

```nano
from "std/collections/stringbuilder.nano" import sb_from_parts

fn join_words() -> string {
    let words: array<string> = (array_new 4 "")
    (array_set words 0 "The")
    (array_set words 1 " ")
    (array_set words 2 "quick")
    (array_set words 3 " fox")
    return (sb_from_parts words)
}

shadow join_words {
    assert (== (join_words) "The quick fox")
}
```

---

### `sb_join`

```nano
fn sb_join(parts: array<string>, separator: string) -> string
```

Concatenates all strings in `parts`, inserting `separator` between each adjacent pair. No separator is added before the first element or after the last.

**Example:**

```nano
from "std/collections/stringbuilder.nano" import sb_join

fn csv_row(fields: array<string>) -> string {
    return (sb_join fields ",")
}

shadow csv_row {
    let fields: array<string> = (array_new 3 "")
    (array_set fields 0 "alice")
    (array_set fields 1 "30")
    (array_set fields 2 "engineer")
    assert (== (csv_row fields) "alice,30,engineer")
}
```

---

### `sb_repeat`

```nano
fn sb_repeat(text: string, n: int) -> string
```

Returns a string consisting of `text` repeated `n` times. When `n` is zero, returns `""`.

**Example:**

```nano
from "std/collections/stringbuilder.nano" import sb_repeat

fn make_divider(width: int) -> string {
    return (sb_repeat "-" width)
}

shadow make_divider {
    assert (== (make_divider 5) "-----")
    assert (== (make_divider 0) "")
}
```

---

### `sb_indent`

```nano
fn sb_indent(level: int, spaces_per_level: int) -> string
```

Returns an indentation string of `level * spaces_per_level` spaces. Useful when generating code or structured text output.

**Example:**

```nano
from "std/collections/stringbuilder.nano" import sb_indent

fn indent_demo() -> bool {
    assert (== (sb_indent 0 4) "")
    assert (== (sb_indent 1 4) "    ")
    assert (== (sb_indent 2 4) "        ")
    assert (== (sb_indent 1 2) "  ")
    return true
}

shadow indent_demo {
    assert (indent_demo)
}
```

---

## Examples

### Example 1: HTML Generation

```nano
from "std/collections/stringbuilder.nano" import sb_new, sb_append, sb_append_line,
                                                   sb_to_string, sb_free, StringBuilder

fn build_html_page(title: string, items: array<string>) -> string {
    let mut sb: StringBuilder = (sb_new)

    set sb (sb_append_line sb "<!DOCTYPE html>")
    set sb (sb_append_line sb "<html>")
    set sb (sb_append_line sb "<head>")
    set sb (sb_append     sb "  <title>")
    set sb (sb_append     sb title)
    set sb (sb_append_line sb "</title>")
    set sb (sb_append_line sb "</head>")
    set sb (sb_append_line sb "<body>")
    set sb (sb_append_line sb "  <ul>")

    let mut i: int = 0
    while (< i (array_length items)) {
        set sb (sb_append sb "    <li>")
        set sb (sb_append sb (at items i))
        set sb (sb_append_line sb "</li>")
        set i (+ i 1)
    }

    set sb (sb_append_line sb "  </ul>")
    set sb (sb_append_line sb "</body>")
    set sb (sb_append sb "</html>")

    let result: string = (sb_to_string sb)
    (sb_free sb)
    return result
}

shadow build_html_page {
    let items: array<string> = (array_new 3 "")
    (array_set items 0 "Alpha")
    (array_set items 1 "Beta")
    (array_set items 2 "Gamma")
    let html: string = (build_html_page "My List" items)
    assert (str_contains html "<title>My List</title>")
    assert (str_contains html "<li>Alpha</li>")
    assert (str_contains html "<li>Gamma</li>")
}
```

### Example 2: CSV Generation

```nano
from "std/collections/stringbuilder.nano" import sb_new, sb_append, sb_append_line,
                                                   sb_append_int, sb_to_string,
                                                   sb_free, StringBuilder

struct Employee {
    name: string,
    age: int,
    department: string
}

fn employees_to_csv(employees: array<Employee>) -> string {
    let mut sb: StringBuilder = (sb_new)

    # Header row
    set sb (sb_append_line sb "name,age,department")

    # Data rows
    let mut i: int = 0
    while (< i (array_length employees)) {
        let emp: Employee = (at employees i)
        set sb (sb_append sb emp.name)
        set sb (sb_append sb ",")
        set sb (sb_append_int sb emp.age)
        set sb (sb_append sb ",")
        set sb (sb_append_line sb emp.department)
        set i (+ i 1)
    }

    let result: string = (sb_to_string sb)
    (sb_free sb)
    return result
}

shadow employees_to_csv {
    let emps: array<Employee> = (array_new 2 Employee { name: "", age: 0, department: "" })
    (array_set emps 0 Employee { name: "Alice", age: 30, department: "Engineering" })
    (array_set emps 1 Employee { name: "Bob", age: 25, department: "Design" })
    let csv: string = (employees_to_csv emps)
    assert (str_contains csv "name,age,department")
    assert (str_contains csv "Alice,30,Engineering")
    assert (str_contains csv "Bob,25,Design")
}
```

### Example 3: JSON Object Building

```nano
from "std/collections/stringbuilder.nano" import sb_new, sb_append, sb_append_int,
                                                   sb_to_string, sb_free, StringBuilder

fn build_json_object(name: string, age: int, active: bool) -> string {
    let mut sb: StringBuilder = (sb_new)
    set sb (sb_append sb "{")
    set sb (sb_append sb "\"name\":\"")
    set sb (sb_append sb name)
    set sb (sb_append sb "\",")
    set sb (sb_append sb "\"age\":")
    set sb (sb_append_int sb age)
    set sb (sb_append sb ",")
    set sb (sb_append sb "\"active\":")
    if active {
        set sb (sb_append sb "true")
    } else {
        set sb (sb_append sb "false")
    }
    set sb (sb_append sb "}")
    let result: string = (sb_to_string sb)
    (sb_free sb)
    return result
}

shadow build_json_object {
    let json: string = (build_json_object "Alice" 30 true)
    assert (== json "{\"name\":\"Alice\",\"age\":30,\"active\":true}")
    let json2: string = (build_json_object "Bob" 25 false)
    assert (str_contains json2 "\"active\":false")
}
```

### Example 4: Indented Code Generation

```nano
from "std/collections/stringbuilder.nano" import sb_new, sb_append, sb_append_line,
                                                   sb_indent, sb_to_string,
                                                   sb_free, StringBuilder

fn generate_function(fn_name: string, body_lines: array<string>) -> string {
    let mut sb: StringBuilder = (sb_new)

    # Function signature
    set sb (sb_append sb "fn ")
    set sb (sb_append sb fn_name)
    set sb (sb_append_line sb "() -> void {")

    # Body lines, indented by 4 spaces
    let indent: string = (sb_indent 1 4)
    let mut i: int = 0
    while (< i (array_length body_lines)) {
        set sb (sb_append sb indent)
        set sb (sb_append_line sb (at body_lines i))
        set i (+ i 1)
    }

    # Closing brace
    set sb (sb_append sb "}")

    let result: string = (sb_to_string sb)
    (sb_free sb)
    return result
}

shadow generate_function {
    let lines: array<string> = (array_new 2 "")
    (array_set lines 0 "let x: int = 42")
    (array_set lines 1 "(println x)")
    let code: string = (generate_function "example" lines)
    assert (str_contains code "fn example() -> void {")
    assert (str_contains code "    let x: int = 42")
    assert (str_contains code "    (println x)")
    assert (str_contains code "}")
}
```

### Example 5: Building a Delimited Report

This example shows combining `sb_join` and `sb_repeat` to produce a formatted tabular report.

```nano
from "std/collections/stringbuilder.nano" import sb_new, sb_append, sb_append_line,
                                                   sb_join, sb_repeat,
                                                   sb_to_string, sb_free, StringBuilder

fn make_table(headers: array<string>, rows: array<array<string>>) -> string {
    let mut sb: StringBuilder = (sb_new)
    let divider: string = (sb_repeat "-" 40)

    # Header
    set sb (sb_append_line sb divider)
    set sb (sb_append_line sb (sb_join headers " | "))
    set sb (sb_append_line sb divider)

    # Rows
    let mut i: int = 0
    while (< i (array_length rows)) {
        let row: array<string> = (at rows i)
        set sb (sb_append_line sb (sb_join row " | "))
        set i (+ i 1)
    }

    set sb (sb_append sb divider)
    let result: string = (sb_to_string sb)
    (sb_free sb)
    return result
}
```

---

## When to Use StringBuilder vs. Plain Concatenation

| Situation | Recommendation |
|-----------|---------------|
| Joining exactly 2 strings | `(+ a b)` — simpler |
| Joining 3–5 strings outside a loop | Either — use judgment |
| Concatenating in a loop of any size | StringBuilder |
| Building strings from 10+ fragments | StringBuilder |
| Generating HTML, CSV, JSON, code | StringBuilder |
| Building a simple error message | `(+ "Error: " msg)` |
| Pre-known set of parts | `sb_from_parts` or `sb_join` |

---

## Common Pitfalls

**Forgetting to call `sb_free`.** The `StringBuilder` wraps a C heap allocation. For every `sb_new` or `sb_with_capacity` call, there must be a corresponding `sb_free` when the builder is no longer needed.

**Passing the wrong type.** `sb_append` only accepts `string`. Use `sb_append_int` for integers or `sb_append_char` for character values. For floats, call `float_to_string` explicitly before appending.

```nano
from "std/collections/stringbuilder.nano" import sb_new, sb_append, sb_append_int,
                                                   sb_to_string, sb_free, StringBuilder

fn format_ratio(num: int, den: int) -> string {
    let mut sb: StringBuilder = (sb_new)
    set sb (sb_append_int sb num)
    set sb (sb_append sb "/")
    set sb (sb_append_int sb den)
    let result: string = (sb_to_string sb)
    (sb_free sb)
    return result
}

shadow format_ratio {
    assert (== (format_ratio 3 4) "3/4")
}
```

**Using `sb_clear` instead of creating a new builder when capacity should reset.** `sb_clear` preserves the allocated capacity, which is desirable for reuse. If you want a truly fresh builder with a smaller footprint, call `sb_free` then `sb_new`.

**Pre-sizing for large outputs.** If you know you will be building a multi-kilobyte string, seed the capacity upfront with `sb_with_capacity` to avoid repeated reallocations.

---

**Previous:** [13.2 log](log.html)
**Next:** [Chapter 14: Data Formats](../14_data_formats/index.html)
