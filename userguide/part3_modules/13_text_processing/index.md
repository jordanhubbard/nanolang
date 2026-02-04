# Chapter 13: Text Processing

**Pattern matching, logging, and efficient string building.**

This chapter covers three essential text processing modules: regular expressions for pattern matching, structured logging for debugging, and StringBuilder for efficient string concatenation.

## 13.1 Regular Expressions

The `stdlib/regex.nano` module provides POSIX regex pattern matching.

### Compiling Patterns

```nano
from "stdlib/regex.nano" import compile, matches, Regex

fn validate_email(email: string) -> bool {
    let pattern: Regex = (compile "[a-zA-Z0-9]+@[a-zA-Z0-9]+\\.[a-z]+")
    let result: bool = (matches pattern email)
    (free pattern)
    return result
}

shadow validate_email {
    assert (validate_email "user@example.com")
    assert (not (validate_email "invalid"))
}
```

**Key points:**
- `compile(pattern)` - Returns opaque `Regex` handle
- Always call `free(regex)` when done
- Backslashes must be escaped: `\\`

### Matching Patterns

```nano
from "stdlib/regex.nano" import compile, matches, Regex

fn test_patterns() -> bool {
    let pattern: Regex = (compile "hello.*world")
    
    let match1: bool = (matches pattern "hello beautiful world")  # true
    let match2: bool = (matches pattern "hello world")             # true
    let match3: bool = (matches pattern "goodbye world")           # false
    
    (free pattern)
    return (and match1 (and match2 (not match3)))
}

shadow test_patterns {
    assert (test_patterns)
}
```

### Finding Matches

```nano
from "stdlib/regex.nano" import compile, find, free, Regex

fn find_position(text: string, pattern_str: string) -> int {
    let pattern: Regex = (compile pattern_str)
    let pos: int = (find pattern text)
    (free pattern)
    return pos
}

shadow find_position {
    assert (== (find_position "hello world" "world") 6)
    assert (== (find_position "no match" "xyz") -1)
}
```

**Returns:** Index of first match, or `-1` if not found

### Finding All Matches

```nano
from "stdlib/regex.nano" import compile, find_all, free, Regex

fn count_matches(text: string, pattern_str: string) -> int {
    let pattern: Regex = (compile pattern_str)
    let positions: array<int> = (find_all pattern text)
    let count: int = (array_length positions)
    (free pattern)
    return count
}

shadow count_matches {
    assert (== (count_matches "aaa" "a") 3)
    assert (== (count_matches "hello world" "o") 2)
}
```

### Capture Groups

```nano
from "stdlib/regex.nano" import compile, groups, free, Regex

fn extract_parts(text: string) -> array<string> {
    let pattern: Regex = (compile "([a-z]+)@([a-z]+)\\.([a-z]+)")
    let captures: array<string> = (groups pattern text)
    (free pattern)
    return captures
}

shadow extract_parts {
    let parts: array<string> = (extract_parts "user@example.com")
    # parts[0] = full match, parts[1..3] = groups
    assert (> (array_length parts) 0)
}
```

### Replacing Text

```nano
from "stdlib/regex.nano" import compile, replace, replace_all, free, Regex

fn clean_text(text: string) -> string {
    let pattern: Regex = (compile "[0-9]+")
    let result: string = (replace_all pattern text "X")
    (free pattern)
    return result
}

shadow clean_text {
    assert (== (clean_text "abc123def456") "abcXdefX")
}
```

**Functions:**
- `replace(regex, text, replacement)` - Replace first match
- `replace_all(regex, text, replacement)` - Replace all matches

### Splitting Strings

```nano
from "stdlib/regex.nano" import compile, split, free, Regex

fn split_by_comma(text: string) -> array<string> {
    let pattern: Regex = (compile ",\\s*")
    let parts: array<string> = (split pattern text)
    (free pattern)
    return parts
}

shadow split_by_comma {
    let parts: array<string> = (split_by_comma "a, b, c")
    assert (== (array_length parts) 3)
}
```

### Common Patterns

**Email validation:**
```nano
let email_pattern: Regex = (compile "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}")
```

**Phone numbers:**
```nano
let phone_pattern: Regex = (compile "\\(?[0-9]{3}\\)?[-. ]?[0-9]{3}[-. ]?[0-9]{4}")
```

**URLs:**
```nano
let url_pattern: Regex = (compile "https?://[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}")
```

**Integers:**
```nano
let int_pattern: Regex = (compile "-?[0-9]+")
```

**Floats:**
```nano
let float_pattern: Regex = (compile "-?[0-9]+\\.[0-9]+")
```

## 13.2 Structured Logging

The `stdlib/log.nano` module provides hierarchical logging with categories.

### Log Levels

```nano
from "stdlib/log.nano" import log_trace, log_debug, log_info
from "stdlib/log.nano" import log_warn, log_error, log_fatal

fn demonstrate_levels() -> void {
    (log_trace "cat" "Detailed trace info")      # TRACE
    (log_debug "cat" "Debug information")        # DEBUG
    (log_info "cat" "Normal operation")          # INFO (default)
    (log_warn "cat" "Warning message")           # WARN
    (log_error "cat" "Error occurred")           # ERROR
    (log_fatal "cat" "Fatal error")              # FATAL
}

shadow demonstrate_levels {
    (demonstrate_levels)
}
```

**Log levels (least to most severe):**
1. **TRACE** - Verbose debugging
2. **DEBUG** - Development debugging
3. **INFO** - Normal operations (default threshold)
4. **WARN** - Potential problems
5. **ERROR** - Failures that don't halt
6. **FATAL** - Critical errors

### Using Categories

```nano
from "stdlib/log.nano" import log_info, log_error

fn process_user_request(user_id: int) -> bool {
    (log_info "auth" (+ "User login: " (int_to_string user_id)))
    
    if (< user_id 0) {
        (log_error "validation" "Invalid user ID")
        return false
    }
    
    (log_info "database" "Fetching user data")
    return true
}

shadow process_user_request {
    assert (process_user_request 123)
    assert (not (process_user_request -1))
}
```

**Categories help organize logs by:**
- Component: `"auth"`, `"database"`, `"network"`
- Module: `"parser"`, `"compiler"`, `"runtime"`
- Feature: `"payment"`, `"search"`, `"upload"`

### Logging Without Categories

```nano
from "stdlib/log.nano" import log, log_err

fn simple_logging() -> void {
    (log "Application started")
    (log_err "Something went wrong")
}

shadow simple_logging {
    (simple_logging)
}
```

**Convenience functions:**
- `log(message)` - Log at INFO level, no category
- `log_err(message)` - Log at ERROR level, no category

### Output Format

```
[LEVEL] category: message
```

**Examples:**
```
[INFO] app: Application started
[DEBUG] parser: Parsing token at position 42
[ERROR] database: Connection timeout
[WARN] cache: Memory usage at 90%
```

### Best Practices

**✅ DO:**

```nano
from "stdlib/log.nano" import log_info, log_error, log_debug

fn good_logging() -> bool {
    # Log important state changes
    (log_info "app" "Processing batch of 100 items")
    
    # Use appropriate levels
    (log_debug "details" "Processing item 42")
    
    # Include context
    (log_error "database" "Failed to connect: timeout after 30s")
    
    return true
}

shadow good_logging {
    assert (good_logging)
}
```

**❌ DON'T:**

```nano
# Don't log everything at ERROR
(log_error "app" "Normal operation")  # Wrong level!

# Don't include sensitive data
(log_info "auth" (+ "Password: " password))  # Security risk!

# Don't log in tight loops
for i in (range 0 1000000) {
    (log_debug "loop" "Iteration")  # Performance hit!
}
```

## 13.3 StringBuilder

The `stdlib/StringBuilder.nano` module provides efficient string building.

### Why Use StringBuilder?

**Problem:** Naive concatenation is O(n²)

```nano
# ❌ Slow: Creates new string each iteration
let mut result: string = ""
for i in (range 0 1000) {
    set result (+ result "x")  # O(n²) - copies entire string!
}
```

**Solution:** StringBuilder is O(n)

```nano
from "stdlib/StringBuilder.nano" import StringBuilder_new, StringBuilder_append
from "stdlib/StringBuilder.nano" import StringBuilder_to_string

# ✅ Fast: Amortized O(1) per append
let mut sb: StringBuilder = (StringBuilder_new)
for i in (range 0 1000) {
    set sb (StringBuilder_append sb "x")  # O(1) average
}
let result: string = (StringBuilder_to_string sb)
```

### Creating StringBuilders

```nano
from "stdlib/StringBuilder.nano" import StringBuilder_new, StringBuilder_with_capacity
from "stdlib/StringBuilder.nano" import StringBuilder

fn create_builders() -> StringBuilder {
    # Default capacity (256)
    let sb1: StringBuilder = (StringBuilder_new)
    
    # Custom capacity
    let sb2: StringBuilder = (StringBuilder_with_capacity 1024)
    
    return sb1
}

shadow create_builders {
    let sb: StringBuilder = (create_builders)
    assert (== sb.length 0)
}
```

### Appending Strings

```nano
from "stdlib/StringBuilder.nano" import StringBuilder_new, StringBuilder_append
from "stdlib/StringBuilder.nano" import StringBuilder_to_string

fn build_greeting(name: string) -> string {
    let mut sb: StringBuilder = (StringBuilder_new)
    set sb (StringBuilder_append sb "Hello, ")
    set sb (StringBuilder_append sb name)
    set sb (StringBuilder_append sb "!")
    return (StringBuilder_to_string sb)
}

shadow build_greeting {
    assert (== (build_greeting "Alice") "Hello, Alice!")
}
```

### Appending Other Types

```nano
from "stdlib/StringBuilder.nano" import StringBuilder_new, StringBuilder_append
from "stdlib/StringBuilder.nano" import StringBuilder_append_int, StringBuilder_append_float
from "stdlib/StringBuilder.nano" import StringBuilder_to_string

fn format_data(name: string, age: int, score: float) -> string {
    let mut sb: StringBuilder = (StringBuilder_new)
    set sb (StringBuilder_append sb "Name: ")
    set sb (StringBuilder_append sb name)
    set sb (StringBuilder_append sb ", Age: ")
    set sb (StringBuilder_append_int sb age)
    set sb (StringBuilder_append sb ", Score: ")
    set sb (StringBuilder_append_float sb score)
    return (StringBuilder_to_string sb)
}

shadow format_data {
    let result: string = (format_data "Alice" 30 95.5)
    assert (str_contains result "Alice")
}
```

### StringBuilder Operations

```nano
from "stdlib/StringBuilder.nano" import StringBuilder_new, StringBuilder_append
from "stdlib/StringBuilder.nano" import StringBuilder_length, StringBuilder_clear
from "stdlib/StringBuilder.nano" import StringBuilder_to_string

fn builder_operations() -> bool {
    let mut sb: StringBuilder = (StringBuilder_new)
    set sb (StringBuilder_append sb "hello")
    
    # Get length
    let len: int = (StringBuilder_length sb)
    assert (== len 5)
    
    # Clear contents
    set sb (StringBuilder_clear sb)
    assert (== (StringBuilder_length sb) 0)
    
    return true
}

shadow builder_operations {
    assert (builder_operations)
}
```

### Practical Example: HTML Generation

```nano
from "stdlib/StringBuilder.nano" import StringBuilder_new, StringBuilder_append
from "stdlib/StringBuilder.nano" import StringBuilder_to_string

fn build_html(title: string, body: string) -> string {
    let mut sb: StringBuilder = (StringBuilder_new)
    set sb (StringBuilder_append sb "<!DOCTYPE html>\n")
    set sb (StringBuilder_append sb "<html>\n")
    set sb (StringBuilder_append sb "<head><title>")
    set sb (StringBuilder_append sb title)
    set sb (StringBuilder_append sb "</title></head>\n")
    set sb (StringBuilder_append sb "<body>")
    set sb (StringBuilder_append sb body)
    set sb (StringBuilder_append sb "</body>\n")
    set sb (StringBuilder_append sb "</html>")
    return (StringBuilder_to_string sb)
}

shadow build_html {
    let html: string = (build_html "Test" "Content")
    assert (str_contains html "<title>Test</title>")
}
```

### Performance Guidelines

**Use StringBuilder when:**
- ✅ Building strings in loops
- ✅ Concatenating 10+ strings
- ✅ Generating templates/reports
- ✅ Building large outputs

**Use simple concatenation when:**
- ✅ Joining 2-3 strings once
- ✅ Simple formatting
- ✅ Readability is more important

## 13.4 Combined Example: Log Parser

```nano
from "stdlib/regex.nano" import compile, matches, groups, free, Regex
from "stdlib/log.nano" import log_info, log_error
from "stdlib/StringBuilder.nano" import StringBuilder_new, StringBuilder_append
from "stdlib/StringBuilder.nano" import StringBuilder_to_string

struct LogEntry {
    level: string,
    category: string,
    message: string
}

fn parse_log_line(line: string) -> LogEntry {
    # Pattern: [LEVEL] category: message
    let pattern: Regex = (compile "\\[([A-Z]+)\\] ([a-z]+): (.+)")
    let parts: array<string> = (groups pattern line)
    
    if (< (array_length parts) 4) {
        (log_error "parser" "Invalid log line format")
        (free pattern)
        return LogEntry { level: "", category: "", message: "" }
    }
    
    let entry: LogEntry = LogEntry {
        level: (at parts 1),
        category: (at parts 2),
        message: (at parts 3)
    }
    
    (free pattern)
    return entry
}

fn format_log_entry(entry: LogEntry) -> string {
    let mut sb: StringBuilder = (StringBuilder_new)
    set sb (StringBuilder_append sb "[")
    set sb (StringBuilder_append sb entry.level)
    set sb (StringBuilder_append sb "] ")
    set sb (StringBuilder_append sb entry.category)
    set sb (StringBuilder_append sb ": ")
    set sb (StringBuilder_append sb entry.message)
    return (StringBuilder_to_string sb)
}

shadow format_log_entry {
    let entry: LogEntry = LogEntry {
        level: "INFO",
        category: "app",
        message: "Started"
    }
    let formatted: string = (format_log_entry entry)
    assert (== formatted "[INFO] app: Started")
}
```

## Summary

In this chapter, you learned:
- ✅ Regular expressions: compile, match, find, replace, split
- ✅ Structured logging: 6 levels, categories, formatting
- ✅ StringBuilder: efficient string building, O(n) performance
- ✅ Combined usage for text processing tasks

### Quick Reference

| Module | Key Functions |
|--------|---------------|
| **regex** | `compile`, `matches`, `find`, `find_all`, `groups`, `replace`, `replace_all`, `split`, `free` |
| **log** | `log_trace`, `log_debug`, `log_info`, `log_warn`, `log_error`, `log_fatal` |
| **StringBuilder** | `StringBuilder_new`, `StringBuilder_append`, `StringBuilder_to_string`, `StringBuilder_length` |

---

**Previous:** [Chapter 12: System & Runtime](../../part2_stdlib/12_system_runtime.html)  
**Next:** [Chapter 14: Data Formats](../14_data_formats/index.html)
