# 13.1 regex — Regular Expressions

**Pattern matching, text extraction, and string transformation with POSIX regular expressions.**

The `stdlib/regex.nano` module provides a complete regular expression API built on POSIX extended regex. It covers everything from simple one-off checks to compiled patterns reused across thousands of inputs. The module exposes two tiers: a quick API that compiles and discards a pattern in one call, and a compiled API where you retain the `Regex` handle for repeated use.

All compiled patterns are garbage-collected — there is no manual memory management required in normal use, though a `free` function is available if you want explicit control.

---

## Quick Start

```nano
from "stdlib/regex.nano" import compile, matches, find_all, replace_all, split, Regex

fn main() -> int {
    # Validate an email address
    let email_re: Regex = (compile "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}")
    let valid: bool = (matches email_re "user@example.com")    # true
    let bad:   bool = (matches email_re "not-an-email")        # false

    # Redact all numbers in a string
    let digit_re: Regex = (compile "[0-9]+")
    let redacted: string = (replace_all digit_re "Order 42 costs $19.99" "X")
    # redacted = "Order X costs $X.X"

    # Split on any whitespace run
    let ws_re: Regex = (compile "[ \\t]+")
    let words: array<string> = (split ws_re "hello   world  nano")
    # words = ["hello", "world", "nano"]

    return 0
}

shadow main {
    assert (== (main) 0)
}
```

---

## Import

```nano
from "stdlib/regex.nano" import compile, matches, find, find_all, groups,
                                   replace, replace_all, split, Regex

# Quick (one-shot) variants — no Regex handle required:
from "stdlib/regex.nano" import quick_match, quick_find, quick_replace, quick_split
```

Import only what you need. `Regex` is the opaque type returned by `compile`; you must import it to annotate variables that hold compiled patterns.

---

## API Reference

### `compile`

```nano
fn compile(pattern: string) -> Regex
```

Compiles a POSIX extended regular expression and returns an opaque `Regex` handle. The handle is garbage-collected; you do not need to call `free` unless you want deterministic cleanup.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `pattern` | `string` | A POSIX extended regex pattern string |

**Returns:** A compiled `Regex` handle.

**Notes:**
- Backslashes must be doubled in NanoLang string literals. The pattern `\d+` is written `"\\d+"`.
- POSIX ERE does not support `\d`, `\w`, `\s` shorthand classes. Use character classes: `[0-9]`, `[a-zA-Z0-9_]`, `[ \t\n]`.
- Compile once and reuse whenever the same pattern is matched against multiple strings.

**Example:**

```nano
from "stdlib/regex.nano" import compile, Regex

fn make_email_validator() -> Regex {
    return (compile "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}")
}
```

---

### `matches`

```nano
fn matches(regex: Regex, text: string) -> bool
```

Tests whether `text` contains at least one match for the compiled pattern. The match may occur anywhere in the string — use `^` and `$` anchors to require a full-string match.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `regex` | `Regex` | A compiled pattern from `compile` |
| `text` | `string` | The string to test |

**Returns:** `true` if the pattern matches anywhere in `text`, `false` otherwise.

**Example:**

```nano
from "stdlib/regex.nano" import compile, matches, Regex

fn validate_hex_color(s: string) -> bool {
    let re: Regex = (compile "^#[0-9a-fA-F]{6}$")
    return (matches re s)
}

shadow validate_hex_color {
    assert (validate_hex_color "#ff00cc")
    assert (validate_hex_color "#AABBCC")
    assert (not (validate_hex_color "ff00cc"))      # missing #
    assert (not (validate_hex_color "#ff00c"))      # too short
}
```

---

### `find`

```nano
fn find(regex: Regex, text: string) -> int
```

Locates the first match of the pattern in `text` and returns its byte offset.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `regex` | `Regex` | A compiled pattern |
| `text` | `string` | The string to search |

**Returns:** The zero-based byte index of the start of the first match, or `-1` if there is no match.

**Example:**

```nano
from "stdlib/regex.nano" import compile, find, Regex

fn find_first_number(text: string) -> int {
    let re: Regex = (compile "[0-9]+")
    return (find re text)
}

shadow find_first_number {
    assert (== (find_first_number "abc 42 xyz") 4)
    assert (== (find_first_number "no digits here") -1)
}
```

---

### `find_all`

```nano
fn find_all(regex: Regex, text: string) -> array<int>
```

Finds every non-overlapping match in `text` and returns their starting byte offsets.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `regex` | `Regex` | A compiled pattern |
| `text` | `string` | The string to search |

**Returns:** An `array<int>` of zero-based byte offsets, one per match. Returns an empty array if there are no matches.

**Example:**

```nano
from "stdlib/regex.nano" import compile, find_all, Regex

fn count_vowels(text: string) -> int {
    let re: Regex = (compile "[aeiouAEIOU]")
    let positions: array<int> = (find_all re text)
    return (array_length positions)
}

shadow count_vowels {
    assert (== (count_vowels "hello world") 3)
    assert (== (count_vowels "rhythm") 0)
}
```

---

### `groups`

```nano
fn groups(regex: Regex, text: string) -> array<string>
```

Executes the pattern against `text` and extracts capture groups from the first match. Capture groups are delimited by parentheses in the pattern.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `regex` | `Regex` | A compiled pattern containing one or more capture groups |
| `text` | `string` | The string to match against |

**Returns:** An `array<string>` where index `0` is the full match text and indices `1`, `2`, … correspond to the first, second, … capture groups. Returns an empty array if there is no match.

**Example:**

```nano
from "stdlib/regex.nano" import compile, groups, Regex

fn parse_date(date_str: string) -> array<string> {
    # Matches "YYYY-MM-DD" and captures year, month, day separately
    let re: Regex = (compile "([0-9]{4})-([0-9]{2})-([0-9]{2})")
    return (groups re date_str)
}

shadow parse_date {
    let parts: array<string> = (parse_date "2024-03-15")
    assert (== (array_length parts) 4)        # full match + 3 groups
    assert (== (array_get parts 0) "2024-03-15")
    assert (== (array_get parts 1) "2024")
    assert (== (array_get parts 2) "03")
    assert (== (array_get parts 3) "15")
}
```

---

### `replace`

```nano
fn replace(regex: Regex, text: string, replacement: string) -> string
```

Returns a new string with the **first** occurrence of the pattern replaced by `replacement`.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `regex` | `Regex` | A compiled pattern |
| `text` | `string` | The original string |
| `replacement` | `string` | The string to substitute for the matched portion |

**Returns:** A new string with the first match replaced. The original `text` is not modified.

**Example:**

```nano
from "stdlib/regex.nano" import compile, replace, Regex

fn censor_first_swear(text: string) -> string {
    let re: Regex = (compile "darn|shoot|heck")
    return (replace re text "***")
}

shadow censor_first_swear {
    assert (== (censor_first_swear "oh darn, darn!") "oh ***, darn!")
}
```

---

### `replace_all`

```nano
fn replace_all(regex: Regex, text: string, replacement: string) -> string
```

Returns a new string with **every** occurrence of the pattern replaced by `replacement`.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `regex` | `Regex` | A compiled pattern |
| `text` | `string` | The original string |
| `replacement` | `string` | The string to substitute for each matched portion |

**Returns:** A new string with all matches replaced.

**Example:**

```nano
from "stdlib/regex.nano" import compile, replace_all, Regex

fn normalize_whitespace(text: string) -> string {
    let re: Regex = (compile "[ \\t]+")
    return (replace_all re text " ")
}

shadow normalize_whitespace {
    assert (== (normalize_whitespace "hello   world\tthere") "hello world there")
}
```

---

### `split`

```nano
fn split(regex: Regex, text: string) -> array<string>
```

Splits `text` into segments using the pattern as a delimiter. The matched delimiter text is not included in the results.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `regex` | `Regex` | A compiled pattern to use as the delimiter |
| `text` | `string` | The string to split |

**Returns:** An `array<string>` of substrings between the delimiters. If the pattern does not match, the array contains the original string as a single element.

**Example:**

```nano
from "stdlib/regex.nano" import compile, split, Regex

fn parse_csv_line(line: string) -> array<string> {
    # Split on comma with optional surrounding whitespace
    let re: Regex = (compile " *, *")
    return (split re line)
}

shadow parse_csv_line {
    let fields: array<string> = (parse_csv_line "alice , 30 , engineer")
    assert (== (array_length fields) 3)
    assert (== (array_get fields 0) "alice ")
    assert (== (array_get fields 1) "30 ")
    assert (== (array_get fields 2) "engineer")
}
```

---

### Quick (One-Shot) Functions

These functions compile the pattern, perform the operation, and discard the compiled pattern — all in a single call. Convenient for ad-hoc use, but less efficient when the same pattern is applied repeatedly.

```nano
fn quick_match(pattern: string, text: string) -> bool
fn quick_find(pattern: string, text: string) -> int
fn quick_replace(pattern: string, text: string, replacement: string) -> string
fn quick_split(pattern: string, text: string) -> array<string>
```

**Example:**

```nano
from "stdlib/regex.nano" import quick_match, quick_find, quick_replace, quick_split

fn demo_quick() -> bool {
    let is_zip: bool   = (quick_match "[0-9]{5}" "90210")
    let pos: int       = (quick_find "[0-9]+" "abc123")
    let cleaned: string = (quick_replace "[aeiou]" "hello" "*")
    let words: array<string> = (quick_split "[ ]+" "one two three")
    return is_zip
}

shadow demo_quick {
    assert (demo_quick)
}
```

---

### `free`

```nano
fn free(regex: Regex) -> void
```

Explicitly releases the memory associated with a compiled pattern. Under normal usage the garbage collector handles this automatically. Call `free` when you need deterministic cleanup or are working in a long-running loop that creates many short-lived patterns.

---

## Examples

### Example 1: Email Validation

```nano
from "stdlib/regex.nano" import compile, matches, Regex

fn is_valid_email(email: string) -> bool {
    let re: Regex = (compile "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}")
    return (matches re email)
}

shadow is_valid_email {
    assert (is_valid_email "user@example.com")
    assert (is_valid_email "first.last+tag@sub.domain.org")
    assert (not (is_valid_email "user@"))
    assert (not (is_valid_email "no-at-sign"))
    assert (not (is_valid_email "@missing-user.com"))
}
```

### Example 2: Extracting All Numbers from Text

```nano
from "stdlib/regex.nano" import compile, find_all, Regex

fn extract_numbers_as_positions(text: string) -> int {
    let re: Regex = (compile "[0-9]+")
    let positions: array<int> = (find_all re text)
    return (array_length positions)
}

shadow extract_numbers_as_positions {
    assert (== (extract_numbers_as_positions "There are 3 cats and 14 dogs among 2 owners") 3)
    assert (== (extract_numbers_as_positions "no numbers") 0)
}
```

### Example 3: Parsing Structured Text with Capture Groups

```nano
from "stdlib/regex.nano" import compile, groups, Regex

struct LogEntry {
    level: string,
    category: string,
    message: string
}

fn parse_log_line(line: string) -> LogEntry {
    # Matches: [LEVEL] category: message
    let re: Regex = (compile "\\[([A-Z]+)\\] ([a-z]+): (.+)")
    let parts: array<string> = (groups re line)

    if (< (array_length parts) 4) {
        return LogEntry { level: "UNKNOWN", category: "", message: line }
    }

    return LogEntry {
        level:    (array_get parts 1),
        category: (array_get parts 2),
        message:  (array_get parts 3)
    }
}

shadow parse_log_line {
    let entry: LogEntry = (parse_log_line "[ERROR] database: Connection refused")
    assert (== entry.level "ERROR")
    assert (== entry.category "database")
    assert (== entry.message "Connection refused")
}
```

### Example 4: Data Sanitization Pipeline

```nano
from "stdlib/regex.nano" import compile, replace_all, Regex

fn sanitize_identifier(raw: string) -> string {
    # Replace non-alphanumeric/underscore chars with underscore
    let bad_chars: Regex = (compile "[^a-zA-Z0-9_]")
    let underscored: string = (replace_all bad_chars raw "_")

    # Collapse multiple consecutive underscores into one
    let multi_us: Regex = (compile "__+")
    return (replace_all multi_us underscored "_")
}

shadow sanitize_identifier {
    assert (== (sanitize_identifier "hello world!")  "hello_world_")
    assert (== (sanitize_identifier "foo--bar")      "foo_bar")
    assert (== (sanitize_identifier "valid_name_123") "valid_name_123")
}
```

### Example 5: URL Extraction

```nano
from "stdlib/regex.nano" import compile, find, find_all, Regex

fn has_url(text: string) -> bool {
    let re: Regex = (compile "https?://[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}")
    return (!= (find re text) -1)
}

fn count_urls(text: string) -> int {
    let re: Regex = (compile "https?://[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}")
    let hits: array<int> = (find_all re text)
    return (array_length hits)
}

shadow has_url {
    assert (has_url "Visit https://example.com for more info")
    assert (not (has_url "No link here"))
}

shadow count_urls {
    assert (== (count_urls "See http://a.io and https://b.com/x") 2)
}
```

---

## Common Patterns Reference

| Purpose | Pattern |
|---------|---------|
| Email (simple) | `[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}` |
| US phone | `\(?[0-9]{3}\)?[-. ]?[0-9]{3}[-. ]?[0-9]{4}` |
| URL | `https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}` |
| Integer | `-?[0-9]+` |
| Float | `-?[0-9]+\.[0-9]+` |
| Hex color | `#[0-9a-fA-F]{6}` |
| ISO date | `[0-9]{4}-[0-9]{2}-[0-9]{2}` |
| Identifier | `[a-zA-Z_][a-zA-Z0-9_]*` |
| Whitespace run | `[ \t\n]+` |
| Any digit | `[0-9]` |

*Remember to double backslashes in NanoLang string literals: `\(` becomes `"\\("`.*

---

## Common Pitfalls

**Forgetting to double backslashes.** In NanoLang string literals, `\` must be written as `\\`. The regex metacharacter `\.` becomes `"\\."` in source code.

```nano
# Wrong — matches any character followed by "com"
let re1: Regex = (compile ".com")

# Correct — matches literal dot then "com"
let re2: Regex = (compile "\\.com")
```

**POSIX ERE lacks `\d`, `\w`, `\s`.** These are Perl/PCRE extensions and are not available in POSIX extended regex. Use character classes instead.

```nano
# Wrong — POSIX ERE does not recognize \d
let bad: Regex = (compile "\\d+")

# Correct
let good: Regex = (compile "[0-9]+")
```

**`matches` is substring, not full-string.** A pattern without anchors matches if it appears *anywhere* in the string.

```nano
from "stdlib/regex.nano" import compile, matches, Regex

fn example_anchor() -> void {
    let re: Regex = (compile "[0-9]+")
    let result: bool = (matches re "abc123")   # true! "123" is a substring match
    (print "")
}
```

Use `^` and `$` for full-string validation:

```nano
let digits_only: Regex = (compile "^[0-9]+$")
```

**Compiling inside a tight loop.** Compiling a regex is expensive compared to matching. If the same pattern is used repeatedly, compile it once outside the loop.

```nano
from "stdlib/regex.nano" import compile, matches, Regex

fn validate_batch(items: array<string>) -> int {
    let re: Regex = (compile "^[a-z]+$")    # compile ONCE
    let mut count: int = 0
    let mut i: int = 0
    while (< i (array_length items)) {
        if (matches re (array_get items i)) {
            set count (+ count 1)
        }
        set i (+ i 1)
    }
    return count
}
```

**`groups` returns empty array on no match.** Always check the array length before indexing.

```nano
from "stdlib/regex.nano" import compile, groups, Regex

fn safe_group_extract(text: string) -> string {
    let re: Regex = (compile "([0-9]{4})-([0-9]{2})")
    let parts: array<string> = (groups re text)
    if (< (array_length parts) 3) {
        return ""
    }
    return (array_get parts 1)
}
```

---

**Previous:** [Chapter 13 Overview](index.html)
**Next:** [13.2 log](log.html)
