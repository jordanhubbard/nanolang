# 13.3 StringBuilder — Efficient String Building

**Amortized O(n) string assembly for loops, templates, and large output generation.**

The `stdlib/StringBuilder.nano` module provides a `StringBuilder` struct and a collection of functions for building strings incrementally without the quadratic cost of repeated concatenation. Each `StringBuilder_append` call returns a new `StringBuilder` value with the text added — the underlying implementation tracks a buffer and length, expanding the buffer geometrically so that the total cost of N appends is O(N) rather than O(N²).

Use StringBuilder whenever you are assembling a string in a loop, generating a document or report, or concatenating more than a handful of fragments. For simple two-or-three piece joins, plain `+` is fine.

---

## Quick Start

```nano
from "stdlib/StringBuilder.nano" import StringBuilder_new, StringBuilder_append,
                                         StringBuilder_to_string, StringBuilder

fn build_greeting(names: array<string>) -> string {
    let mut sb: StringBuilder = (StringBuilder_new)
    set sb (StringBuilder_append sb "Hello to: ")

    let mut i: int = 0
    while (< i (array_length names)) {
        if (!= i 0) {
            set sb (StringBuilder_append sb ", ")
        }
        set sb (StringBuilder_append sb (array_get names i))
        set i (+ i 1)
    }

    set sb (StringBuilder_append sb "!")
    return (StringBuilder_to_string sb)
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
from "stdlib/StringBuilder.nano" import StringBuilder_new, StringBuilder_append,
                                         StringBuilder_append_line, StringBuilder_append_int,
                                         StringBuilder_append_char, StringBuilder_to_string,
                                         StringBuilder_length, StringBuilder_clear,
                                         StringBuilder_is_empty, StringBuilder_with_capacity,
                                         StringBuilder_from_parts, StringBuilder_join,
                                         StringBuilder_repeat, StringBuilder_indent,
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

## The Immutable Value Pattern

`StringBuilder` in `stdlib/StringBuilder.nano` is a plain struct (not an opaque handle). Each `StringBuilder_append` call returns a *new* `StringBuilder` value. You must capture the return value or the append has no effect.

```nano
from "stdlib/StringBuilder.nano" import StringBuilder_new, StringBuilder_append,
                                         StringBuilder_to_string, StringBuilder

fn wrong_usage() -> string {
    let sb: StringBuilder = (StringBuilder_new)
    (StringBuilder_append sb "Hello")   # Return value discarded — BUG!
    return (StringBuilder_to_string sb) # Returns ""
}

fn correct_usage() -> string {
    let mut sb: StringBuilder = (StringBuilder_new)
    set sb (StringBuilder_append sb "Hello")   # Capture the new value
    set sb (StringBuilder_append sb " World")
    return (StringBuilder_to_string sb)  # Returns "Hello World"
}

shadow correct_usage {
    assert (== (correct_usage) "Hello World")
}
```

---

## API Reference

### `StringBuilder_new`

```nano
fn StringBuilder_new() -> StringBuilder
```

Creates a new, empty `StringBuilder` with a default initial capacity of 256 bytes. The capacity grows automatically as content is appended.

**Returns:** An empty `StringBuilder` with `length = 0`.

**Example:**

```nano
from "stdlib/StringBuilder.nano" import StringBuilder_new, StringBuilder_length, StringBuilder

fn example_new() -> bool {
    let sb: StringBuilder = (StringBuilder_new)
    return (== (StringBuilder_length sb) 0)
}

shadow example_new {
    assert (example_new)
}
```

---

### `StringBuilder_with_capacity`

```nano
fn StringBuilder_with_capacity(capacity: int) -> StringBuilder
requires (> capacity 0)
```

Creates an empty `StringBuilder` pre-sized to `capacity` bytes. Use this when you know approximately how large the final string will be; it avoids intermediate reallocations.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `capacity` | `int` | Initial buffer size in bytes. Must be greater than zero. |

**Example:**

```nano
from "stdlib/StringBuilder.nano" import StringBuilder_with_capacity, StringBuilder_append,
                                         StringBuilder_to_string, StringBuilder

fn build_large_report(line_count: int) -> string {
    # Pre-size for roughly 80 chars per line
    let estimated: int = (* line_count 80)
    let mut sb: StringBuilder = (StringBuilder_with_capacity estimated)
    let mut i: int = 0
    while (< i line_count) {
        set sb (StringBuilder_append sb (+ "Line " (+ (int_to_string i) "\n")))
        set i (+ i 1)
    }
    return (StringBuilder_to_string sb)
}
```

---

### `StringBuilder_append`

```nano
fn StringBuilder_append(sb: StringBuilder, text: string) -> StringBuilder
```

Appends `text` to the current content of `sb` and returns the updated `StringBuilder`. If `text` is empty, returns `sb` unchanged.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `sb` | `StringBuilder` | The current builder value |
| `text` | `string` | Text to append |

**Returns:** A `StringBuilder` containing the original content plus `text`.

**Example:**

```nano
from "stdlib/StringBuilder.nano" import StringBuilder_new, StringBuilder_append,
                                         StringBuilder_to_string, StringBuilder

fn chain_append() -> string {
    let mut sb: StringBuilder = (StringBuilder_new)
    set sb (StringBuilder_append sb "one")
    set sb (StringBuilder_append sb ", ")
    set sb (StringBuilder_append sb "two")
    set sb (StringBuilder_append sb ", ")
    set sb (StringBuilder_append sb "three")
    return (StringBuilder_to_string sb)
}

shadow chain_append {
    assert (== (chain_append) "one, two, three")
}
```

---

### `StringBuilder_append_line`

```nano
fn StringBuilder_append_line(sb: StringBuilder, text: string) -> StringBuilder
```

Appends `text` followed by a newline character (`\n`). Equivalent to two `StringBuilder_append` calls but more convenient when building multi-line output.

**Example:**

```nano
from "stdlib/StringBuilder.nano" import StringBuilder_new, StringBuilder_append_line,
                                         StringBuilder_to_string, StringBuilder

fn build_poem() -> string {
    let mut sb: StringBuilder = (StringBuilder_new)
    set sb (StringBuilder_append_line sb "Roses are red,")
    set sb (StringBuilder_append_line sb "Violets are blue,")
    set sb (StringBuilder_append_line sb "NanoLang is fast,")
    set sb (StringBuilder_append_line sb "And memory-safe too.")
    return (StringBuilder_to_string sb)
}

shadow build_poem {
    let poem: string = (build_poem)
    assert (str_contains poem "NanoLang")
}
```

---

### `StringBuilder_append_int`

```nano
fn StringBuilder_append_int(sb: StringBuilder, n: int) -> StringBuilder
```

Converts `n` to its decimal string representation and appends it. Equivalent to `StringBuilder_append sb (int_to_string n)` but reads more naturally in numeric formatting code.

**Example:**

```nano
from "stdlib/StringBuilder.nano" import StringBuilder_new, StringBuilder_append,
                                         StringBuilder_append_int, StringBuilder_to_string,
                                         StringBuilder

fn format_coords(x: int, y: int) -> string {
    let mut sb: StringBuilder = (StringBuilder_new)
    set sb (StringBuilder_append sb "(")
    set sb (StringBuilder_append_int sb x)
    set sb (StringBuilder_append sb ", ")
    set sb (StringBuilder_append_int sb y)
    set sb (StringBuilder_append sb ")")
    return (StringBuilder_to_string sb)
}

shadow format_coords {
    assert (== (format_coords 10 20) "(10, 20)")
    assert (== (format_coords -5 0) "(-5, 0)")
}
```

---

### `StringBuilder_append_char`

```nano
fn StringBuilder_append_char(sb: StringBuilder, c: int) -> StringBuilder
```

Appends a single character given by its ASCII integer value. Use character literals (`'A'`, `'\n'`) for readability.

**Example:**

```nano
from "stdlib/StringBuilder.nano" import StringBuilder_new, StringBuilder_append,
                                         StringBuilder_append_char, StringBuilder_to_string,
                                         StringBuilder

fn surround_with_quotes(s: string) -> string {
    let mut sb: StringBuilder = (StringBuilder_new)
    set sb (StringBuilder_append_char sb '"')
    set sb (StringBuilder_append sb s)
    set sb (StringBuilder_append_char sb '"')
    return (StringBuilder_to_string sb)
}

shadow surround_with_quotes {
    assert (== (surround_with_quotes "hello") "\"hello\"")
}
```

---

### `StringBuilder_to_string`

```nano
fn StringBuilder_to_string(sb: StringBuilder) -> string
```

Returns the accumulated content as a plain `string`. Does not modify the `StringBuilder`; you can call `to_string` multiple times on the same value and append more afterward.

**Example:**

```nano
from "stdlib/StringBuilder.nano" import StringBuilder_new, StringBuilder_append,
                                         StringBuilder_to_string, StringBuilder

fn checkpoint_example() -> bool {
    let mut sb: StringBuilder = (StringBuilder_new)
    set sb (StringBuilder_append sb "Hello")
    let partial: string = (StringBuilder_to_string sb)  # "Hello"

    set sb (StringBuilder_append sb " World")
    let full: string = (StringBuilder_to_string sb)     # "Hello World"

    return (and (== partial "Hello") (== full "Hello World"))
}

shadow checkpoint_example {
    assert (checkpoint_example)
}
```

---

### `StringBuilder_length`

```nano
fn StringBuilder_length(sb: StringBuilder) -> int
ensures (>= result 0)
```

Returns the current number of characters (bytes) in the builder.

**Example:**

```nano
from "stdlib/StringBuilder.nano" import StringBuilder_new, StringBuilder_append,
                                         StringBuilder_length, StringBuilder

fn length_demo() -> bool {
    let mut sb: StringBuilder = (StringBuilder_new)
    assert (== (StringBuilder_length sb) 0)
    set sb (StringBuilder_append sb "hello")
    assert (== (StringBuilder_length sb) 5)
    set sb (StringBuilder_append sb " world")
    assert (== (StringBuilder_length sb) 11)
    return true
}

shadow length_demo {
    assert (length_demo)
}
```

---

### `StringBuilder_clear`

```nano
fn StringBuilder_clear(sb: StringBuilder) -> StringBuilder
```

Returns a new `StringBuilder` with the content reset to empty while preserving the allocated capacity. Useful for reusing a builder across multiple formatting passes without paying for reallocation.

**Example:**

```nano
from "stdlib/StringBuilder.nano" import StringBuilder_new, StringBuilder_append,
                                         StringBuilder_clear, StringBuilder_to_string,
                                         StringBuilder_length, StringBuilder

fn reuse_builder() -> bool {
    let mut sb: StringBuilder = (StringBuilder_new)
    set sb (StringBuilder_append sb "first pass")
    let first: string = (StringBuilder_to_string sb)

    set sb (StringBuilder_clear sb)
    assert (== (StringBuilder_length sb) 0)

    set sb (StringBuilder_append sb "second pass")
    let second: string = (StringBuilder_to_string sb)

    return (and (== first "first pass") (== second "second pass"))
}

shadow reuse_builder {
    assert (reuse_builder)
}
```

---

### `StringBuilder_is_empty`

```nano
fn StringBuilder_is_empty(sb: StringBuilder) -> bool
```

Returns `true` if the builder contains no characters. Equivalent to `(== (StringBuilder_length sb) 0)` but more expressive.

---

### `StringBuilder_from_parts`

```nano
fn StringBuilder_from_parts(parts: array<string>) -> string
```

Efficiently concatenates all strings in `parts` in order and returns the result. This is a standalone utility — it creates its own internal builder, so you do not need to manage one yourself.

**Example:**

```nano
from "stdlib/StringBuilder.nano" import StringBuilder_from_parts

fn join_words() -> string {
    let words: array<string> = (array_new 4 "")
    (array_set words 0 "The")
    (array_set words 1 " ")
    (array_set words 2 "quick")
    (array_set words 3 " fox")
    return (StringBuilder_from_parts words)
}

shadow join_words {
    assert (== (join_words) "The quick fox")
}
```

---

### `StringBuilder_join`

```nano
fn StringBuilder_join(parts: array<string>, separator: string) -> string
```

Concatenates all strings in `parts`, inserting `separator` between each adjacent pair. No separator is added before the first element or after the last.

**Example:**

```nano
from "stdlib/StringBuilder.nano" import StringBuilder_join

fn csv_row(fields: array<string>) -> string {
    return (StringBuilder_join fields ",")
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

### `StringBuilder_repeat`

```nano
fn StringBuilder_repeat(text: string, n: int) -> string
requires (>= n 0)
```

Returns a string consisting of `text` repeated `n` times. When `n` is zero, returns `""`.

**Example:**

```nano
from "stdlib/StringBuilder.nano" import StringBuilder_repeat

fn make_divider(width: int) -> string {
    return (StringBuilder_repeat "-" width)
}

shadow make_divider {
    assert (== (make_divider 5) "-----")
    assert (== (make_divider 0) "")
}
```

---

### `StringBuilder_indent`

```nano
fn StringBuilder_indent(level: int, spaces_per_level: int) -> string
requires (>= level 0)
requires (>= spaces_per_level 0)
```

Returns an indentation string of `level * spaces_per_level` spaces. Useful when generating code or structured text output.

**Example:**

```nano
from "stdlib/StringBuilder.nano" import StringBuilder_indent

fn indent_demo() -> bool {
    assert (== (StringBuilder_indent 0 4) "")
    assert (== (StringBuilder_indent 1 4) "    ")
    assert (== (StringBuilder_indent 2 4) "        ")
    assert (== (StringBuilder_indent 1 2) "  ")
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
from "stdlib/StringBuilder.nano" import StringBuilder_new, StringBuilder_append,
                                         StringBuilder_append_line, StringBuilder_to_string,
                                         StringBuilder

fn build_html_page(title: string, items: array<string>) -> string {
    let mut sb: StringBuilder = (StringBuilder_new)

    set sb (StringBuilder_append_line sb "<!DOCTYPE html>")
    set sb (StringBuilder_append_line sb "<html>")
    set sb (StringBuilder_append_line sb "<head>")
    set sb (StringBuilder_append sb   "  <title>")
    set sb (StringBuilder_append sb   title)
    set sb (StringBuilder_append_line sb "</title>")
    set sb (StringBuilder_append_line sb "</head>")
    set sb (StringBuilder_append_line sb "<body>")
    set sb (StringBuilder_append_line sb "  <ul>")

    let mut i: int = 0
    while (< i (array_length items)) {
        set sb (StringBuilder_append sb "    <li>")
        set sb (StringBuilder_append sb (array_get items i))
        set sb (StringBuilder_append_line sb "</li>")
        set i (+ i 1)
    }

    set sb (StringBuilder_append_line sb "  </ul>")
    set sb (StringBuilder_append_line sb "</body>")
    set sb (StringBuilder_append sb "</html>")

    return (StringBuilder_to_string sb)
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
from "stdlib/StringBuilder.nano" import StringBuilder_new, StringBuilder_append,
                                         StringBuilder_append_line, StringBuilder_append_int,
                                         StringBuilder_to_string, StringBuilder

struct Employee {
    name: string,
    age: int,
    department: string
}

fn employees_to_csv(employees: array<Employee>) -> string {
    let mut sb: StringBuilder = (StringBuilder_new)

    # Header row
    set sb (StringBuilder_append_line sb "name,age,department")

    # Data rows
    let mut i: int = 0
    while (< i (array_length employees)) {
        let emp: Employee = (array_get employees i)
        set sb (StringBuilder_append sb emp.name)
        set sb (StringBuilder_append sb ",")
        set sb (StringBuilder_append_int sb emp.age)
        set sb (StringBuilder_append sb ",")
        set sb (StringBuilder_append_line sb emp.department)
        set i (+ i 1)
    }

    return (StringBuilder_to_string sb)
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
from "stdlib/StringBuilder.nano" import StringBuilder_new, StringBuilder_append,
                                         StringBuilder_append_int, StringBuilder_to_string,
                                         StringBuilder

fn build_json_object(name: string, age: int, active: bool) -> string {
    let mut sb: StringBuilder = (StringBuilder_new)
    set sb (StringBuilder_append sb "{")
    set sb (StringBuilder_append sb "\"name\":\"")
    set sb (StringBuilder_append sb name)
    set sb (StringBuilder_append sb "\",")
    set sb (StringBuilder_append sb "\"age\":")
    set sb (StringBuilder_append_int sb age)
    set sb (StringBuilder_append sb ",")
    set sb (StringBuilder_append sb "\"active\":")
    if active {
        set sb (StringBuilder_append sb "true")
    } else {
        set sb (StringBuilder_append sb "false")
    }
    set sb (StringBuilder_append sb "}")
    return (StringBuilder_to_string sb)
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
from "stdlib/StringBuilder.nano" import StringBuilder_new, StringBuilder_append,
                                         StringBuilder_append_line, StringBuilder_indent,
                                         StringBuilder_to_string, StringBuilder

fn generate_function(fn_name: string, body_lines: array<string>) -> string {
    let mut sb: StringBuilder = (StringBuilder_new)

    # Function signature
    set sb (StringBuilder_append sb "fn ")
    set sb (StringBuilder_append sb fn_name)
    set sb (StringBuilder_append_line sb "() -> void {")

    # Body lines, indented by 4 spaces
    let indent: string = (StringBuilder_indent 1 4)
    let mut i: int = 0
    while (< i (array_length body_lines)) {
        set sb (StringBuilder_append sb indent)
        set sb (StringBuilder_append_line sb (array_get body_lines i))
        set i (+ i 1)
    }

    # Closing brace
    set sb (StringBuilder_append sb "}")

    return (StringBuilder_to_string sb)
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

This example shows combining `StringBuilder_join` and `StringBuilder_repeat` to produce a formatted tabular report.

```nano
from "stdlib/StringBuilder.nano" import StringBuilder_new, StringBuilder_append,
                                         StringBuilder_append_line, StringBuilder_join,
                                         StringBuilder_repeat, StringBuilder_to_string,
                                         StringBuilder

fn make_table(headers: array<string>, rows: array<array<string>>) -> string {
    let mut sb: StringBuilder = (StringBuilder_new)
    let divider: string = (StringBuilder_repeat "-" 40)

    # Header
    set sb (StringBuilder_append_line sb divider)
    set sb (StringBuilder_append_line sb (StringBuilder_join headers " | "))
    set sb (StringBuilder_append_line sb divider)

    # Rows
    let mut i: int = 0
    while (< i (array_length rows)) {
        let row: array<string> = (array_get rows i)
        set sb (StringBuilder_append_line sb (StringBuilder_join row " | "))
        set i (+ i 1)
    }

    set sb (StringBuilder_append sb divider)
    return (StringBuilder_to_string sb)
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
| Pre-known set of parts | `StringBuilder_from_parts` or `StringBuilder_join` |

---

## Common Pitfalls

**Not capturing the return value.** Every `StringBuilder_append` returns a new value. Discarding it silently produces an empty or stale result. Use `let mut` and `set`.

**Passing the wrong type.** `StringBuilder_append` only accepts `string`. Use `StringBuilder_append_int` for integers or call `int_to_string` / `float_to_string` explicitly before appending.

```nano
from "stdlib/StringBuilder.nano" import StringBuilder_new, StringBuilder_append,
                                         StringBuilder_append_int, StringBuilder_to_string,
                                         StringBuilder

fn format_ratio(num: int, den: int) -> string {
    let mut sb: StringBuilder = (StringBuilder_new)
    set sb (StringBuilder_append_int sb num)
    set sb (StringBuilder_append sb "/")
    set sb (StringBuilder_append_int sb den)
    return (StringBuilder_to_string sb)
}

shadow format_ratio {
    assert (== (format_ratio 3 4) "3/4")
}
```

**Forgetting `StringBuilder_with_capacity` for very large outputs.** If you know you will be building a multi-megabyte string, seed the capacity upfront to avoid repeated halving-then-doubling reallocations.

**Using `StringBuilder_clear` instead of creating a new builder when capacity should reset.** `StringBuilder_clear` preserves the allocated capacity, which is desirable for reuse. If you want a truly fresh builder, call `StringBuilder_new` again.

---

**Previous:** [13.2 log](log.html)
**Next:** [Chapter 14: Data Formats](../14_data_formats/index.html)
