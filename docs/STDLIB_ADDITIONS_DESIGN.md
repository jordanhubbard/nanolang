# Standard Library Additions for Self-Hosting

**Status:** Design Phase  
**Priority:** #4-6 for Self-Hosting  
**Principles:** Safety, Simple Error Handling, No Exceptions

## Overview

Three categories of stdlib additions needed for self-hosting:
1. **File I/O** - Read source files, write C output
2. **Advanced String Operations** - Character access, parsing, formatting
3. **System Execution** - Invoke gcc

All are relatively simple to implement (standard C library wrappers).

---

# Part 1: File I/O

## Required Functions

### `file_read(path: string) -> string`

Read entire file as string.

```nano
fn load_source(path: string) -> string {
    let content: string = (file_read path)
    if (== (str_length content) 0) {
        print "Warning: file is empty or couldn't be read"
    }
    return content
}

shadow load_source {
    # Test with known file
    let content: string = (load_source "examples/hello.nano")
    assert (> (str_length content) 0)
}
```

**Behavior:**
- Returns file contents as string
- Returns empty string `""` if file doesn't exist or can't be read
- No exception throwing (keeps language simple)
- Check result with `str_length` to detect errors

**C Implementation:**
```c
char* nano_file_read(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return strdup("");  // Return empty on error
    
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    char *content = malloc(size + 1);
    fread(content, 1, size, f);
    content[size] = '\0';
    fclose(f);
    
    return content;
}
```

---

### `file_write(path: string, content: string) -> void`

Write string to file (overwrites existing).

```nano
fn save_output(path: string, code: string) -> void {
    (file_write path code)
}

shadow save_output {
    let test_code: string = "int main() { return 0; }"
    (save_output "test_output.c" test_code)
    
    # Verify by reading back
    let read_back: string = (file_read "test_output.c")
    assert (str_equals read_back test_code)
}
```

**Behavior:**
- Creates file if it doesn't exist
- Overwrites file if it exists
- Silent failure (no return value)
- Future: Could return bool for success/failure

**C Implementation:**
```c
void nano_file_write(const char *path, const char *content) {
    FILE *f = fopen(path, "wb");
    if (!f) return;  // Silent failure for now
    
    fputs(content, f);
    fclose(f);
}
```

---

### `file_append(path: string, content: string) -> void`

Append string to file.

```nano
fn log_message(msg: string) -> void {
    (file_append "compiler.log" msg)
    (file_append "compiler.log" "\n")
}
```

**Behavior:**
- Creates file if it doesn't exist
- Appends to existing file
- Silent failure

**C Implementation:**
```c
void nano_file_append(const char *path, const char *content) {
    FILE *f = fopen(path, "ab");
    if (!f) return;
    
    fputs(content, f);
    fclose(f);
}
```

---

### `file_exists(path: string) -> bool`

Check if file exists.

```nano
fn compile_if_exists(path: string) -> int {
    if (file_exists path) {
        let source: string = (file_read path)
        # ... compile ...
        return 0
    } else {
        print "File not found"
        return 1
    }
}

shadow compile_if_exists {
    assert (== (file_exists "examples/hello.nano") true)
    assert (== (file_exists "nonexistent.nano") false)
}
```

**C Implementation:**
```c
bool nano_file_exists(const char *path) {
    FILE *f = fopen(path, "r");
    if (f) {
        fclose(f);
        return true;
    }
    return false;
}
```

---

## File I/O Summary

| Function | Signature | Description |
|----------|-----------|-------------|
| `file_read` | `(string) -> string` | Read entire file |
| `file_write` | `(string, string) -> void` | Write/overwrite file |
| `file_append` | `(string, string) -> void` | Append to file |
| `file_exists` | `(string) -> bool` | Check if file exists |

**Implementation Time:** 1-2 weeks
**Complexity:** Low (standard C library wrappers)

---

# Part 2: Advanced String Operations

## Required Functions

### `str_char_at(s: string, index: int) -> string`

Get single character at index as string.

```nano
fn first_char(s: string) -> string {
    if (> (str_length s) 0) {
        return (str_char_at s 0)
    } else {
        return ""
    }
}

shadow first_char {
    assert (str_equals (first_char "Hello") "H")
    assert (str_equals (first_char "X") "X")
    assert (str_equals (first_char "") "")
}
```

**Behavior:**
- Returns single-character string
- Bounds checked (returns "" if out of bounds)
- UTF-8 aware (returns first byte for simplicity)

**C Implementation:**
```c
char* nano_str_char_at(const char *s, int index) {
    int len = strlen(s);
    if (index < 0 || index >= len) {
        return strdup("");  // Out of bounds
    }
    
    char *result = malloc(2);
    result[0] = s[index];
    result[1] = '\0';
    return result;
}
```

---

### `str_char_code(s: string) -> int`

Get ASCII/UTF-8 code of first character.

```nano
fn is_digit_char(s: string) -> bool {
    let code: int = (str_char_code s)
    return (and (>= code 48) (<= code 57))  # '0' = 48, '9' = 57
}

shadow is_digit_char {
    assert (== (is_digit_char "5") true)
    assert (== (is_digit_char "a") false)
}
```

**Behavior:**
- Returns code of first character
- Returns 0 if string is empty
- ASCII/UTF-8 code

**C Implementation:**
```c
int64_t nano_str_char_code(const char *s) {
    if (s[0] == '\0') return 0;
    return (int64_t)(unsigned char)s[0];
}
```

---

### `str_from_code(code: int) -> string`

Create single-character string from code.

```nano
fn digit_to_char(n: int) -> string {
    return (str_from_code (+ 48 n))  # '0' + n
}

shadow digit_to_char {
    assert (str_equals (digit_to_char 0) "0")
    assert (str_equals (digit_to_char 5) "5")
    assert (str_equals (digit_to_char 9) "9")
}
```

**C Implementation:**
```c
char* nano_str_from_code(int64_t code) {
    char *result = malloc(2);
    result[0] = (char)code;
    result[1] = '\0';
    return result;
}
```

---

### `str_format(template: string, arg0: string, arg1: string, ...) -> string`

Simple string formatting with {0}, {1}, etc.

**Initial version (1-2 args):**

```nano
fn format_error(line: int, msg: string) -> string {
    let line_str: string = (int_to_string line)
    return (str_format "Error at line {0}: {1}" line_str msg)
    # Result: "Error at line 42: unexpected token"
}
```

**Alternative (simpler):** Start with just concatenation, add formatting later.

```nano
fn format_error(line: int, msg: string) -> string {
    let line_str: string = (int_to_string line)
    let s1: string = (str_concat "Error at line " line_str)
    let s2: string = (str_concat s1 ": ")
    let s3: string = (str_concat s2 msg)
    return s3
}
```

**Decision:** Start without `str_format`, add if needed. Concatenation works for now.

---

### `str_split(s: string, delimiter: string) -> array<string>`

Split string by delimiter.

```nano
fn parse_csv_line(line: string) -> array<string> {
    return (str_split line ",")
}

shadow parse_csv_line {
    let parts: array<string> = (parse_csv_line "a,b,c")
    assert (== (array_length parts) 3)
    assert (str_equals (at parts 0) "a")
    assert (str_equals (at parts 1) "b")
    assert (str_equals (at parts 2) "c")
}
```

**Behavior:**
- Returns array of substrings
- Empty delimiter: return array of single string
- Delimiter not found: return array with original string
- Max splits: unlimited (or configurable)

**C Implementation:** (using strtok or manual parsing)

```c
// Simplified - production version needs more work
Array* nano_str_split(const char *s, const char *delim) {
    // Count occurrences to allocate array
    // Split and store in array
    // Return array
}
```

**Alternative:** Could return `list<string>` instead of `array<string>` (after lists implemented).

---

### `str_to_int(s: string) -> int`

Parse string to integer.

```nano
fn parse_number(token: string) -> int {
    let num: int = (str_to_int token)
    if (== num 0) {
        # Could be 0 or parse error - check string equals "0"
        if (str_equals token "0") {
            return 0  # Actually zero
        } else {
            return -1  # Parse error (use -1 as sentinel)
        }
    }
    return num
}

shadow parse_number {
    assert (== (parse_number "42") 42)
    assert (== (parse_number "0") 0)
    assert (== (parse_number "-100") -100)
}
```

**Behavior:**
- Returns integer value
- Returns 0 on parse error (limitation)
- Can't distinguish "0" from error without checking

**C Implementation:**
```c
int64_t nano_str_to_int(const char *s) {
    return (int64_t)atoll(s);  // Returns 0 on error
}
```

---

### `str_to_float(s: string) -> float`

Parse string to float.

```nano
fn parse_float_literal(token: string) -> float {
    return (str_to_float token)
}

shadow parse_float_literal {
    assert (== (parse_float_literal "3.14") 3.14)
    assert (== (parse_float_literal "0.5") 0.5)
}
```

**C Implementation:**
```c
double nano_str_to_float(const char *s) {
    return atof(s);  // Returns 0.0 on error
}
```

---

### `int_to_string(n: int) -> string`

Convert integer to string.

```nano
fn format_line_number(line: int) -> string {
    let prefix: string = "Line "
    let num_str: string = (int_to_string line)
    return (str_concat prefix num_str)
}

shadow format_line_number {
    let s: string = (format_line_number 42)
    assert (str_contains s "42")
}
```

**C Implementation:**
```c
char* nano_int_to_string(int64_t n) {
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "%lld", (long long)n);
    return strdup(buffer);
}
```

---

### `float_to_string(f: float) -> string`

Convert float to string.

```nano
fn format_result(value: float) -> string {
    let prefix: string = "Result: "
    let value_str: string = (float_to_string value)
    return (str_concat prefix value_str)
}
```

**C Implementation:**
```c
char* nano_float_to_string(double f) {
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "%g", f);
    return strdup(buffer);
}
```

---

## String Operations Summary

| Function | Signature | Priority | Description |
|----------|-----------|----------|-------------|
| `str_char_at` | `(string, int) -> string` | High | Get character at index |
| `str_char_code` | `(string) -> int` | High | Get character code |
| `str_from_code` | `(int) -> string` | Medium | Create char from code |
| `str_split` | `(string, string) -> array<string>` | Medium | Split by delimiter |
| `str_to_int` | `(string) -> int` | High | Parse integer |
| `str_to_float` | `(string) -> float` | High | Parse float |
| `int_to_string` | `(int) -> string` | High | Format integer |
| `float_to_string` | `(float) -> string` | Medium | Format float |

**Implementation Time:** 2-3 weeks  
**Complexity:** Low (mostly wrappers)

---

# Part 3: System Execution

## Required Function

### `system(command: string) -> int`

Execute system command, return exit code.

```nano
fn compile_c_to_binary(c_file: string, output: string) -> int {
    # Build command
    let cmd1: string = (str_concat "gcc -o " output)
    let cmd2: string = (str_concat cmd1 " ")
    let cmd: string = (str_concat cmd2 c_file)
    
    # Execute: gcc -o output c_file
    let exit_code: int = (system cmd)
    
    if (== exit_code 0) {
        print "Compilation successful"
    } else {
        print "Compilation failed"
    }
    
    return exit_code
}

shadow compile_c_to_binary {
    # Write test C file
    (file_write "test.c" "int main() { return 0; }")
    
    # Compile it
    let result: int = (compile_c_to_binary "test.c" "test_exe")
    assert (== result 0)
    
    # Verify executable exists
    assert (== (file_exists "test_exe") true)
}
```

**Behavior:**
- Executes command in shell
- Returns exit code (0 = success, non-zero = error)
- Blocks until command completes
- stdout/stderr go to terminal (not captured)

**C Implementation:**
```c
int64_t nano_system(const char *command) {
    int result = system(command);
    return (int64_t)result;
}
```

---

### `system_output(command: string) -> string` (Optional)

Execute command and capture output.

**Initial version:** Not needed. Just use `system()` and let output go to terminal.

**Future:** If needed, add output capture:

```nano
let version: string = (system_output "gcc --version")
print version
```

**C Implementation:**
```c
char* nano_system_output(const char *command) {
    FILE *pipe = popen(command, "r");
    if (!pipe) return strdup("");
    
    // Read output into buffer
    char buffer[4096];
    size_t size = fread(buffer, 1, sizeof(buffer)-1, pipe);
    buffer[size] = '\0';
    
    pclose(pipe);
    return strdup(buffer);
}
```

**Decision:** Start without `system_output`, add later if needed.

---

## System Execution Summary

| Function | Signature | Priority | Description |
|----------|-----------|----------|-------------|
| `system` | `(string) -> int` | High | Execute command, return exit code |
| `system_output` | `(string) -> string` | Low | Execute and capture output (future) |

**Implementation Time:** 1-2 weeks  
**Complexity:** Very Low (single C function)

---

## Security Considerations

### Command Injection Risk

**Problem:**
```nano
# User input could inject commands
let file: string = (getenv "INPUT_FILE")
let cmd: string = (str_concat "gcc " file)
let result: int = (system cmd)

# If INPUT_FILE = "file.c; rm -rf /", bad things happen!
```

**Mitigation:**
1. **Document the risk** - Warn users in documentation
2. **Validate input** - Check for dangerous characters
3. **Use fixed commands** - Don't build commands from user input
4. **Future:** Add `system_argv` that takes array of args (safer)

**For self-hosting:** Not a concern (compiler controls all inputs).

---

## Implementation Roadmap

### Phase 1: File I/O (Week 1)

**Tasks:**
- [ ] Implement `file_read` in C
- [ ] Implement `file_write` in C
- [ ] Implement `file_append` in C
- [ ] Implement `file_exists` in C
- [ ] Add to transpiler (recognize builtins)
- [ ] Write tests
- [ ] Update documentation

---

### Phase 2: String Operations (Weeks 2-3)

**Week 2: Character operations**
- [ ] Implement `str_char_at` in C
- [ ] Implement `str_char_code` in C
- [ ] Implement `str_from_code` in C
- [ ] Write tests

**Week 3: Parsing/formatting**
- [ ] Implement `str_to_int` in C
- [ ] Implement `str_to_float` in C
- [ ] Implement `int_to_string` in C
- [ ] Implement `float_to_string` in C
- [ ] Implement `str_split` in C (optional)
- [ ] Write tests
- [ ] Update documentation

---

### Phase 3: System Execution (Week 4)

**Tasks:**
- [ ] Implement `system` in C
- [ ] Add to transpiler
- [ ] Write tests (careful - has side effects)
- [ ] Document security risks
- [ ] Update documentation

---

## Testing Strategy

### File I/O Tests

**Test with real files:**
```nano
# tests/unit/file_io_test.nano

fn test_read_write() -> void {
    let content: string = "Hello, nanolang!"
    (file_write "test.txt" content)
    
    let read_back: string = (file_read "test.txt")
    assert (str_equals read_back content)
}

shadow test_read_write {
    (test_read_write)
}
```

---

### String Operations Tests

```nano
# tests/unit/string_ops_test.nano

fn test_char_operations() -> void {
    let s: string = "Hello"
    assert (str_equals (str_char_at s 0) "H")
    assert (== (str_char_code "A") 65)
    assert (str_equals (str_from_code 65) "A")
}

shadow test_char_operations {
    (test_char_operations)
}

fn test_conversions() -> void {
    assert (== (str_to_int "42") 42)
    assert (str_equals (int_to_string 42) "42")
}

shadow test_conversions {
    (test_conversions)
}
```

---

### System Execution Tests

```nano
# tests/unit/system_test.nano

fn test_system_success() -> void {
    let result: int = (system "echo test")
    assert (== result 0)  # Success
}

shadow test_system_success {
    (test_system_success)
}

fn test_system_failure() -> void {
    let result: int = (system "false")  # Command that returns 1
    assert (== result 1)  # Failure
}

shadow test_system_failure {
    (test_system_failure)
}
```

---

## Timeline Summary

| Feature | Weeks | Complexity |
|---------|-------|------------|
| File I/O | 1 | Low |
| String Ops | 2-3 | Low-Medium |
| System Execution | 1 | Very Low |
| **Total** | **4-5** | **Low** |

---

## Success Criteria

✅ **File I/O:**
- [ ] Can read source files
- [ ] Can write C output
- [ ] Can check file existence
- [ ] All tests pass

✅ **String Operations:**
- [ ] Can access individual characters
- [ ] Can parse numbers
- [ ] Can convert numbers to strings
- [ ] Ready for lexer implementation

✅ **System Execution:**
- [ ] Can invoke gcc
- [ ] Exit codes work correctly
- [ ] Security risks documented

✅ **Documentation:**
- [ ] STDLIB.md updated
- [ ] Examples added
- [ ] Security warnings included

---

## Dependencies

**File I/O:**
- None (can implement immediately)

**String Operations:**
- Arrays (for `str_split` return type)
- Or Lists (better for `str_split`)

**System Execution:**
- None (can implement immediately)

**All three unlock:**
- Complete compiler implementation
- Self-hosting capability

---

**Status:** Ready to implement (after structs, enums, lists)  
**Priority:** #4-6  
**Estimated Time:** 4-5 weeks total  
**Dependencies:** Minimal (arrays for str_split)

