# nanolang Standard Library Reference

Complete reference for built-in functions in nanolang.

---

## Core I/O Functions (3)

### `print(value: any) -> void`
Prints a value without a trailing newline.

```nano
print "Hello"
print 42
print 3.14
```

### `println(value: any) -> void`
Prints a value with a trailing newline. Polymorphic - works with int, float, string, and bool.

```nano
(println "Hello, World!")
(println 42)
(println 3.14159)
(println true)
```

### `assert(condition: bool) -> void`
Runtime assertion. Terminates program if condition is false.

```nano
assert (== 2 2)           # Passes
assert (> 5 3)            # Passes
assert (== (add 2 2) 4)   # Passes
```

---

## Math Operations (11)

### Basic Math (3)

#### `abs(x: int|float) -> int|float`
Returns the absolute value. Type-preserving (int → int, float → float).

```nano
(abs -5)      # Returns 5
(abs 5)       # Returns 5
(abs -3.14)   # Returns 3.14
```

#### `min(a: int|float, b: int|float) -> int|float`
Returns the minimum of two values. Both arguments must be the same type.

```nano
(min 5 10)    # Returns 5
(min 3.14 2.71)  # Returns 2.71
```

#### `max(a: int|float, b: int|float) -> int|float`
Returns the maximum of two values. Both arguments must be the same type.

```nano
(max 5 10)    # Returns 10
(max 3.14 2.71)  # Returns 3.14
```

### Advanced Math (5)

#### `sqrt(x: int|float) -> float`
Returns the square root. Always returns float.

```nano
(sqrt 16.0)   # Returns 4.0
(sqrt 2.0)    # Returns 1.41421...
(sqrt 9)      # Returns 3.0 (int converted to float)
```

#### `pow(base: int|float, exponent: int|float) -> float`
Returns base raised to the power of exponent. Always returns float.

```nano
(pow 2.0 3.0)    # Returns 8.0 (2³)
(pow 5.0 2.0)    # Returns 25.0 (5²)
(pow 2.0 -1.0)   # Returns 0.5 (2⁻¹)
```

#### `floor(x: int|float) -> float`
Returns the largest integer ≤ x as a float.

```nano
(floor 3.7)     # Returns 3.0
(floor 3.2)     # Returns 3.0
(floor -2.3)    # Returns -3.0
```

#### `ceil(x: int|float) -> float`
Returns the smallest integer ≥ x as a float.

```nano
(ceil 3.2)      # Returns 4.0
(ceil 3.7)      # Returns 4.0
(ceil -2.7)     # Returns -2.0
```

#### `round(x: int|float) -> float`
Rounds to the nearest integer as a float.

```nano
(round 3.4)     # Returns 3.0
(round 3.6)     # Returns 4.0
(round 3.5)     # Returns 4.0 (rounds half to even)
```

### Trigonometric Functions (3)

All trig functions work in **radians** and always return float.

#### `sin(x: int|float) -> float`
Returns the sine of x (in radians).

```nano
(sin 0.0)       # Returns 0.0
(sin 1.5708)    # Returns ≈1.0 (π/2)
```

#### `cos(x: int|float) -> float`
Returns the cosine of x (in radians).

```nano
(cos 0.0)       # Returns 1.0
(cos 3.14159)   # Returns ≈-1.0 (π)
```

#### `tan(x: int|float) -> float`
Returns the tangent of x (in radians).

```nano
(tan 0.0)       # Returns 0.0
(tan 0.7854)    # Returns ≈1.0 (π/4)
```

---

## String Operations (18)

### `str_length(s: string) -> int`
Returns the length of a string in bytes.

```nano
let text: string = "Hello"
let len: int = (str_length text)  # Returns 5
```

### `str_concat(s1: string, s2: string) -> string`
Concatenates two strings, returning a new string.

```nano
let hello: string = "Hello"
let world: string = " World"
let result: string = (str_concat hello world)  # "Hello World"
```

### `str_substring(s: string, start: int, length: int) -> string`
Extracts a substring starting at `start` with the given `length`.
- `start` is 0-indexed
- If `start + length` exceeds string length, returns until end of string
- Returns empty string if start is out of bounds

```nano
let text: string = "Hello, World!"
let hello: string = (str_substring text 0 5)   # "Hello"
let world: string = (str_substring text 7 5)   # "World"
```

### `str_contains(s: string, substr: string) -> bool`
Returns true if string `s` contains substring `substr`.

```nano
let text: string = "The quick brown fox"
(str_contains text "quick")   # Returns true
(str_contains text "slow")    # Returns false
```

### `str_equals(s1: string, s2: string) -> bool`
Returns true if both strings are exactly equal.

```nano
let s1: string = "Hello"
let s2: string = "Hello"
let s3: string = "World"
(str_equals s1 s2)   # Returns true
(str_equals s1 s3)   # Returns false
```

---

## Advanced String Operations (13)

### Character Access (2)

#### `char_at(s: string, index: int) -> int`
Returns the ASCII value of the character at the specified index.
- Index is 0-based
- Returns integer ASCII value (0-255)
- **Bounds-checked** - terminates with error if index is out of bounds

```nano
let text: string = "Hello"
let h: int = (char_at text 0)  # Returns 72 (ASCII 'H')
let e: int = (char_at text 1)  # Returns 101 (ASCII 'e')
let o: int = (char_at text 4)  # Returns 111 (ASCII 'o')
```

**Use Case:** Essential for lexical analysis and character-by-character parsing.

#### `string_from_char(c: int) -> string`
Creates a single-character string from an ASCII value.

```nano
let a: string = (string_from_char 65)   # Returns "A"
let z: string = (string_from_char 90)   # Returns "Z"
let zero: string = (string_from_char 48)  # Returns "0"
let space: string = (string_from_char 32)  # Returns " "
```

**Use Case:** Building strings character-by-character, useful for code generation.

### Character Classification (6)

#### `is_digit(c: int) -> bool`
Returns true if the character is a digit ('0'-'9').

```nano
(is_digit 48)   # Returns true  ('0')
(is_digit 53)   # Returns true  ('5')
(is_digit 57)   # Returns true  ('9')
(is_digit 65)   # Returns false ('A')
(is_digit 97)   # Returns false ('a')
```

**Use Case:** Token classification in lexical analysis.

#### `is_alpha(c: int) -> bool`
Returns true if the character is a letter (a-z, A-Z).

```nano
(is_alpha 65)   # Returns true  ('A')
(is_alpha 90)   # Returns true  ('Z')
(is_alpha 97)   # Returns true  ('a')
(is_alpha 122)  # Returns true  ('z')
(is_alpha 48)   # Returns false ('0')
(is_alpha 32)   # Returns false (' ')
```

**Use Case:** Identifier validation in parsers.

#### `is_alnum(c: int) -> bool`
Returns true if the character is alphanumeric (digit or letter).

```nano
(is_alnum 48)   # Returns true  ('0')
(is_alnum 65)   # Returns true  ('A')
(is_alnum 97)   # Returns true  ('a')
(is_alnum 32)   # Returns false (' ')
(is_alnum 33)   # Returns false ('!')
```

**Use Case:** Checking if character is valid in an identifier.

#### `is_whitespace(c: int) -> bool`
Returns true if the character is whitespace (space, tab, newline, carriage return).

```nano
(is_whitespace 32)  # Returns true  (' ')
(is_whitespace 9)   # Returns true  ('\t')
(is_whitespace 10)  # Returns true  ('\n')
(is_whitespace 13)  # Returns true  ('\r')
(is_whitespace 65)  # Returns false ('A')
```

**Use Case:** Skipping whitespace during tokenization.

#### `is_upper(c: int) -> bool`
Returns true if the character is an uppercase letter (A-Z).

```nano
(is_upper 65)   # Returns true  ('A')
(is_upper 90)   # Returns true  ('Z')
(is_upper 77)   # Returns true  ('M')
(is_upper 97)   # Returns false ('a')
(is_upper 48)   # Returns false ('0')
```

#### `is_lower(c: int) -> bool`
Returns true if the character is a lowercase letter (a-z).

```nano
(is_lower 97)   # Returns true  ('a')
(is_lower 122)  # Returns true  ('z')
(is_lower 109)  # Returns true  ('m')
(is_lower 65)   # Returns false ('A')
(is_lower 48)   # Returns false ('0')
```

### Type Conversions (5)

#### `int_to_string(n: int) -> string`
Converts an integer to its string representation.

```nano
let s1: string = (int_to_string 42)    # Returns "42"
let s2: string = (int_to_string 0)     # Returns "0"
let s3: string = (int_to_string -100)  # Returns "-100"
let s4: string = (int_to_string 999)   # Returns "999"
```

**Use Case:** Formatting numbers for output, error messages, code generation.

#### `string_to_int(s: string) -> int`
Parses a string to an integer. Returns 0 if string cannot be parsed.

```nano
let n1: int = (string_to_int "42")     # Returns 42
let n2: int = (string_to_int "0")      # Returns 0
let n3: int = (string_to_int "-100")   # Returns -100
let n4: int = (string_to_int "12345")  # Returns 12345
```

**Use Case:** Parsing numeric literals during compilation.

#### `digit_value(c: int) -> int`
Converts a digit character to its numeric value. Returns -1 if not a digit.

```nano
(digit_value 48)  # Returns 0  ('0' -> 0)
(digit_value 49)  # Returns 1  ('1' -> 1)
(digit_value 53)  # Returns 5  ('5' -> 5)
(digit_value 57)  # Returns 9  ('9' -> 9)
(digit_value 65)  # Returns -1 ('A' is not a digit)
```

**Use Case:** Parsing multi-digit numbers character-by-character.

#### `char_to_lower(c: int) -> int`
Converts an uppercase letter to lowercase. Non-letters are unchanged.

```nano
(char_to_lower 65)   # Returns 97  ('A' -> 'a')
(char_to_lower 90)   # Returns 122 ('Z' -> 'z')
(char_to_lower 77)   # Returns 109 ('M' -> 'm')
(char_to_lower 97)   # Returns 97  ('a' -> 'a', already lowercase)
(char_to_lower 48)   # Returns 48  ('0' -> '0', not a letter)
```

**Use Case:** Case-insensitive comparisons, keyword normalization.

#### `char_to_upper(c: int) -> int`
Converts a lowercase letter to uppercase. Non-letters are unchanged.

```nano
(char_to_upper 97)   # Returns 65  ('a' -> 'A')
(char_to_upper 122)  # Returns 90  ('z' -> 'Z')
(char_to_upper 109)  # Returns 77  ('m' -> 'M')
(char_to_upper 65)   # Returns 65  ('A' -> 'A', already uppercase)
(char_to_upper 48)   # Returns 48  ('0' -> '0', not a letter)
```

**Use Case:** Normalizing identifiers, formatting output.

### Practical Example: Simple Lexer

```nano
fn classify_char(c: int) -> string {
    if (is_whitespace c) {
        return "WHITESPACE"
    }
    if (is_digit c) {
        return "DIGIT"
    }
    if (is_alpha c) {
        return "LETTER"
    }
    return "SYMBOL"
}

fn parse_number(source: string, start: int) -> int {
    let mut result: int = 0
    let mut pos: int = start
    let len: int = (str_length source)
    
    while (< pos len) {
        let c: int = (char_at source pos)
        if (is_digit c) {
            let digit: int = (digit_value c)
            set result (+ (* result 10) digit)
            set pos (+ pos 1)
        } else {
            return result
        }
    }
    
    return result
}
```

---

## Array Operations (4)

### `at(arr: array<T>, index: int) -> T`
Returns the element at the specified index. **Bounds-checked** - terminates with error if index is out of bounds.

```nano
let nums: array<int> = [1, 2, 3, 4, 5]
let first: int = (at nums 0)   # Returns 1
let last: int = (at nums 4)    # Returns 5
# (at nums 10)                  # ERROR: index out of bounds!
```

**Safety:** This function includes runtime bounds checking to prevent memory corruption and security vulnerabilities.

### `array_length(arr: array<T>) -> int`
Returns the length (number of elements) of an array.

```nano
let nums: array<int> = [10, 20, 30]
let len: int = (array_length nums)  # Returns 3

let empty: array<int> = []
let zero: int = (array_length empty)  # Returns 0
```

### `array_new(size: int, default: T) -> array<T>`
Creates a new array with the specified size, filled with the default value.

```nano
# Create array of 5 zeros
let zeros: array<int> = (array_new 5 0)
# [0, 0, 0, 0, 0]

# Create array of 3 empty strings
let strings: array<string> = (array_new 3 "")
# ["", "", ""]
```

**Note:** Size must be non-negative. Negative sizes will cause an error.

### `array_set(arr: mut array<T>, index: int, value: T) -> void`
Sets the element at the specified index. **Bounds-checked** - terminates with error if index is out of bounds. Requires a **mutable** array.

```nano
let mut nums: array<int> = [1, 2, 3]
(array_set nums 1 42)  # nums is now [1, 42, 3]

# Type checking enforced
# (array_set nums 0 "hello")  # ERROR: type mismatch!

# Bounds checking enforced
# (array_set nums 10 99)      # ERROR: index out of bounds!
```

---

## HashMap Operations (10)

**HashMap<K,V>** is a key-value collection with **O(1)** average lookup.

**Current constraints:**
- **Key type:** `int` or `string`
- **Value type:** `int` or `string`
- Requires explicit type annotation: `HashMap<K,V>`

### `map_new() -> HashMap<K,V>`
Creates a new hash map. **Requires a type annotation** on the variable.

```nano
let hm: HashMap<string, int> = (map_new)
```

### `map_put(map: HashMap<K,V>, key: K, value: V) -> void`
Inserts or updates a key/value pair.

```nano
(map_put hm "alice" 10)
(map_put hm "bob" 20)
```

### `map_get(map: HashMap<K,V>, key: K) -> V`
Returns the value for a key, or a default (`0` or `""`) if missing.

```nano
let score: int = (map_get hm "alice")
```

### `map_has(map: HashMap<K,V>, key: K) -> bool`
Checks if a key exists.

```nano
if (map_has hm "alice") { (println "found") }
```

### `map_remove(map: HashMap<K,V>, key: K) -> void`
Removes a key/value pair if present.

### `map_length(map: HashMap<K,V>) -> int`
Returns the number of entries. Alias: `map_size`.

### `map_clear(map: HashMap<K,V>) -> void`
Removes all entries without freeing the map. Alias: `map_free` frees memory.

### `map_free(map: HashMap<K,V>) -> void`
Frees the hash map and its internal storage.

### `map_keys(map: HashMap<K,V>) -> array<K>`
Returns all keys as an array.

### `map_values(map: HashMap<K,V>) -> array<V>`
Returns all values as an array.

**Safety:**
- Requires array to be declared `mut`
- Runtime bounds checking prevents buffer overflows
- Type checking ensures homogeneous arrays

---

## Result Type Operations (5)

The Result<T, E> type represents success (Ok) or failure (Err) values.

### `result_is_ok(r: Result<T, E>) -> bool`
Checks if Result is Ok variant.

```nano
let r: Result<int, string> = (divide 10 2)
if (result_is_ok r) {
    (println "Success!")
}
```

### `result_is_err(r: Result<T, E>) -> bool`
Checks if Result is Err variant.

```nano
let r: Result<int, string> = (divide 10 0)
if (result_is_err r) {
    (println "Error occurred")
}
```

### `result_unwrap(r: Result<T, E>) -> T`
Extracts Ok value, panics if Err.

```nano
let r: Result<int, string> = (divide 10 2)
let value: int = (result_unwrap r)  # 5
```

**Warning:** Panics if Result is Err. Use `result_is_ok` check first.

### `result_unwrap_or(r: Result<T, E>, default: T) -> T`
Extracts Ok value, or returns default if Err.

```nano
let r: Result<int, string> = (divide 10 0)
let value: int = (result_unwrap_or r 0)  # Returns 0 (default)
```

**Safe alternative to `result_unwrap`.**

### `result_unwrap_err(r: Result<T, E>) -> E`
Extracts Err value, panics if Ok.

```nano
let r: Result<int, string> = (divide 10 0)
if (result_is_err r) {
    let error: string = (result_unwrap_err r)
    (println error)
}
```

---

## Binary String Operations (8)

Binary strings (bstring) handle raw binary data.

### `bytes_from_string(s: string) -> bstring`
Converts string to binary string.

```nano
let bs: bstring = (bytes_from_string "Hello")
```

### `string_from_bytes(bs: bstring) -> string`
Converts binary string to regular string.

```nano
let s: string = (string_from_bytes bs)
```

### `bstring_length(bs: bstring) -> int`
Gets binary string length in bytes.

```nano
let len: int = (bstring_length bs)
```

### `bstring_at(bs: bstring, index: int) -> int`
Gets byte value at index.

```nano
let byte: int = (bstring_at bs 0)  # Returns 0-255
```

### `bstring_slice(bs: bstring, start: int, length: int) -> bstring`
Extracts portion of binary string.

```nano
let subset: bstring = (bstring_slice bs 0 10)
```

### `bstring_concat(bs1: bstring, bs2: bstring) -> bstring`
Concatenates two binary strings.

```nano
let combined: bstring = (bstring_concat bs1 bs2)
```

### `bstring_new(size: int) -> bstring`
Creates new binary string of specified size (zero-filled).

```nano
let bs: bstring = (bstring_new 1024)
```

### `bstring_set(bs: mut bstring, index: int, value: int) -> void`
Sets byte at index (value 0-255).

```nano
(bstring_set bs 0 65)  # Sets first byte to 'A'
```

**Safety:** Bounds checking prevents buffer overflows.

---

## OS/System Functions (3)

### `getcwd() -> string`
Returns the current working directory as an absolute path.

```nano
let cwd: string = (getcwd)
(println cwd)  # Prints something like "/Users/username/project"
```

### `getenv(name: string) -> string`
Gets an environment variable value. Returns empty string if not set.

```nano
let home: string = (getenv "HOME")
let path: string = (getenv "PATH")
```

### `range(start: int, end: int) -> iterator`
Special function used only in `for` loops. Creates an iterator from `start` (inclusive) to `end` (exclusive).

```nano
for i in (range 0 10) {
    (println i)  # Prints 0, 1, 2, ..., 9
}
```

**Note:** `range` can only be used in for-loop contexts, not as a regular function call.

---

## File Operations (10)

### `file_read(path: string) -> string`
Reads entire file contents as a string.

```nano
let content: string = (file_read "data.txt")
(println content)
```

**Returns:** File contents as string. Empty string on error.

### `file_read_bytes(path: string) -> bstring`
Reads file contents as binary string (bstring).

```nano
let data: bstring = (file_read_bytes "image.png")
let size: int = (bstring_length data)
```

**Use for:** Binary files, images, non-text data.

### `file_write(path: string, content: string) -> int`
Writes string content to file, overwriting if exists.

```nano
let status: int = (file_write "output.txt" "Hello, World!")
if (== status 0) {
    (println "Write successful")
} else {
    (println "Write failed")
}
```

**Returns:** 0 on success, 1 on failure.

### `file_append(path: string, content: string) -> int`
Appends string content to end of file.

```nano
let status: int = (file_append "log.txt" "New log entry\n")
```

**Returns:** 0 on success, 1 on failure.

### `file_exists(path: string) -> bool`
Checks if file exists and is accessible.

```nano
if (file_exists "config.json") {
    let config: string = (file_read "config.json")
} else {
    (println "Config file not found")
}
```

### `file_remove(path: string) -> int`
Deletes a file.

```nano
let status: int = (file_remove "temp.txt")
```

**Returns:** 0 on success, 1 on failure.

**Warning:** Permanent deletion. No confirmation or undo.

### `file_rename(old_path: string, new_path: string) -> int`
Renames or moves a file.

```nano
let status: int = (file_rename "old.txt" "new.txt")
```

**Returns:** 0 on success, 1 on failure.

### `file_size(path: string) -> int`
Gets file size in bytes.

```nano
let size: int = (file_size "data.bin")
(println (+ "File size: " (int_to_string size)))
```

**Returns:** File size in bytes, or -1 on error.

### `file_copy(src: string, dst: string) -> int`
Copies file from source to destination.

```nano
let status: int = (file_copy "original.txt" "backup.txt")
```

**Returns:** 0 on success, 1 on failure.

### `file_chmod(path: string, mode: int) -> int`
Changes file permissions (Unix mode bits).

```nano
# Make file executable: chmod +x
let status: int = (file_chmod "script.sh" 493)  # 0755 in octal
```

**Returns:** 0 on success, 1 on failure.

---

## Directory Operations (5)

### `dir_exists(path: string) -> bool`
Checks if directory exists.

```nano
if (not (dir_exists "output")) {
    (dir_create "output")
}
```

### `dir_create(path: string) -> int`
Creates a directory. Parent directories must exist.

```nano
let status: int = (dir_create "build/output")
```

**Returns:** 0 on success, 1 on failure.

**Note:** Does not create parent directories. Use `dir_create_all` for that.

### `dir_remove(path: string) -> int`
Removes an empty directory.

```nano
let status: int = (dir_remove "temp")
```

**Returns:** 0 on success, 1 on failure.

**Warning:** Directory must be empty. Use recursion for non-empty directories.

### `dir_list(path: string) -> array<string>`
Lists all entries in a directory.

```nano
let entries: array<string> = (dir_list ".")
for entry in entries {
    (println entry)
}
```

**Returns:** Array of filenames (not full paths). Empty array on error.

### `chdir(path: string) -> int`
Changes current working directory.

```nano
let status: int = (chdir "/tmp")
let cwd: string = (getcwd)
(println cwd)  # Prints "/tmp"
```

**Returns:** 0 on success, 1 on failure.

---

## Path Operations (6)

### `path_join(a: string, b: string) -> string`
Joins two path components with proper separator.

```nano
let full_path: string = (path_join "/home/user" "documents")
# Result: "/home/user/documents"
```

**Platform-aware:** Uses `/` on Unix, `\` on Windows.

### `path_basename(path: string) -> string`
Extracts filename from path.

```nano
let filename: string = (path_basename "/path/to/file.txt")
# Result: "file.txt"
```

### `path_dirname(path: string) -> string`
Extracts directory path.

```nano
let dir: string = (path_dirname "/path/to/file.txt")
# Result: "/path/to"
```

### `path_extension(path: string) -> string`
Extracts file extension.

```nano
let ext: string = (path_extension "document.pdf")
# Result: "pdf"
```

### `path_normalize(path: string) -> string`
Normalizes path (removes `.`, `..`, redundant separators).

```nano
let clean: string = (path_normalize "./foo/../bar/./baz")
# Result: "bar/baz"
```

### `path_absolute(path: string) -> string`
Converts relative path to absolute path.

```nano
let abs: string = (path_absolute "file.txt")
# Result: "/current/working/dir/file.txt"
```

---

## Process Operations (5)

### `system(command: string) -> int`
Executes shell command and waits for completion.

```nano
let status: int = (system "ls -la")
```

**Returns:** Exit code from command (0 typically means success).

**Warning:** Use with caution. Command injection risk with user input.

### `exit(code: int) -> void`
Terminates program with exit code.

```nano
if (not (file_exists "required.txt")) {
    (println "Error: required.txt not found")
    (exit 1)
}
```

**Note:** Does not return. Exits immediately.

### `getenv(name: string) -> string`
Gets environment variable value.

```nano
let home: string = (getenv "HOME")
let user: string = (getenv "USER")
```

**Returns:** Variable value, or empty string if not set.

### `setenv(name: string, value: string) -> int`
Sets environment variable for current process.

```nano
let status: int = (setenv "MY_VAR" "my_value")
```

**Returns:** 0 on success, 1 on failure.

**Scope:** Only affects current process and child processes.

### `unsetenv(name: string) -> int`
Removes environment variable.

```nano
let status: int = (unsetenv "MY_VAR")
```

**Returns:** 0 on success, 1 on failure.

---

## Advanced Math Functions (15)

### Exponential and Logarithmic

#### `exp(x: float) -> float`
Returns e^x (exponential function).

```nano
let result: float = (exp 2.0)  # e^2 ≈ 7.389
```

#### `log(x: float) -> float`
Natural logarithm (base e).

```nano
let result: float = (log 10.0)  # ln(10) ≈ 2.303
```

#### `log10(x: float) -> float`
Base-10 logarithm.

```nano
let result: float = (log10 100.0)  # Result: 2.0
```

#### `log2(x: float) -> float`
Base-2 logarithm.

```nano
let result: float = (log2 8.0)  # Result: 3.0
```

### Hyperbolic Functions

#### `sinh(x: float) -> float`
Hyperbolic sine.

```nano
let result: float = (sinh 1.0)
```

#### `cosh(x: float) -> float`
Hyperbolic cosine.

```nano
let result: float = (cosh 1.0)
```

#### `tanh(x: float) -> float`
Hyperbolic tangent.

```nano
let result: float = (tanh 0.5)
```

#### `asinh(x: float) -> float`
Inverse hyperbolic sine.

#### `acosh(x: float) -> float`
Inverse hyperbolic cosine.

#### `atanh(x: float) -> float`
Inverse hyperbolic tangent.

### Advanced Operations

#### `cbrt(x: float) -> float`
Cube root.

```nano
let result: float = (cbrt 27.0)  # Result: 3.0
```

#### `hypot(x: float, y: float) -> float`
Hypotenuse: sqrt(x² + y²).

```nano
let dist: float = (hypot 3.0 4.0)  # Result: 5.0
```

#### `fmod(x: float, y: float) -> float`
Floating-point remainder.

```nano
let remainder: float = (fmod 5.5 2.0)  # Result: 1.5
```

#### `copysign(x: float, y: float) -> float`
Returns x with sign of y.

```nano
let result: float = (copysign 3.0 -1.0)  # Result: -3.0
```

#### `fmax(x: float, y: float) -> float`
Maximum of two floats (handles NaN correctly).

```nano
let max_val: float = (fmax 3.5 2.1)  # Result: 3.5
```

#### `fmin(x: float, y: float) -> float`
Minimum of two floats (handles NaN correctly).

```nano
let min_val: float = (fmin 3.5 2.1)  # Result: 2.1
```

---

## Type Conversion (6)

### `int_to_string(n: int) -> string`
Converts integer to string.

```nano
let s: string = (int_to_string 42)  # "42"
```

### `float_to_string(f: float) -> string`
Converts float to string.

```nano
let s: string = (float_to_string 3.14)  # "3.14"
```

### `string_to_int(s: string) -> int`
Parses string to integer.

```nano
let n: int = (string_to_int "123")  # 123
```

**Returns:** Parsed integer, or 0 if parsing fails.

### `string_to_float(s: string) -> float`
Parses string to float.

```nano
let f: float = (string_to_float "3.14")  # 3.14
```

**Returns:** Parsed float, or 0.0 if parsing fails.

### `bool_to_string(b: bool) -> string`
Converts boolean to string.

```nano
let s: string = (bool_to_string true)  # "true"
```

### `char_to_string(c: int) -> string`
Converts character code to string.

```nano
let s: string = (char_to_string 65)  # "A"
```

---

## Character Operations (5)

### `char_at(s: string, index: int) -> int`
Gets character code at index.

```nano
let code: int = (char_at "Hello" 0)  # 72 ('H')
```

**Returns:** Character code (0-127 for ASCII), or 0 if index out of bounds.

### `char_to_lower(c: int) -> int`
Converts character code to lowercase.

```nano
let lower: int = (char_to_lower 65)  # 97 ('a')
```

### `char_to_upper(c: int) -> int`
Converts character code to uppercase.

```nano
let upper: int = (char_to_upper 97)  # 65 ('A')
```

### `digit_value(c: int) -> int`
Converts digit character to numeric value.

```nano
let val: int = (digit_value 53)  # 5 (from '5')
```

**Returns:** Digit value (0-9), or -1 if not a digit.

### `string_from_char(c: int) -> string`
Creates single-character string from code.

```nano
let s: string = (string_from_char 65)  # "A"
```

---

## Array Advanced Operations (5)

### `array_push(arr: mut array<T>, value: T) -> void`
Appends element to end of array.

```nano
let mut numbers: array<int> = [1, 2, 3]
(array_push numbers 4)
# numbers is now [1, 2, 3, 4]
```

**Requires:** Array must be declared `mut`.

### `array_pop(arr: mut array<int>) -> int`
Removes and returns last element.

```nano
let mut stack: array<int> = [1, 2, 3]
let last: int = (array_pop stack)  # 3
# stack is now [1, 2]
```

**Returns:** Last element, or 0 if array empty.

### `array_slice(arr: array<T>, start: int, length: int) -> array<T>`
Creates sub-array from portion of array.

```nano
let numbers: array<int> = [1, 2, 3, 4, 5]
let subset: array<int> = (array_slice numbers 1 3)
# subset is [2, 3, 4]
```

### `array_remove_at(arr: mut array<T>, index: int) -> void`
Removes element at index, shifting remaining elements.

```nano
let mut items: array<int> = [10, 20, 30, 40]
(array_remove_at items 1)
# items is now [10, 30, 40]
```

### `filter(arr: array<T>, predicate: fn(T) -> bool) -> array<T>`
Creates new array with elements matching predicate.

```nano
fn is_even(n: int) -> bool {
    return (== (% n 2) 0)
}

let numbers: array<int> = [1, 2, 3, 4, 5, 6]
let evens: array<int> = (filter numbers is_even)
# evens is [2, 4, 6]
```

---

## List Operations (Dynamic Lists)

Dynamic lists provide resizable, type-safe collections.

### List<int> Operations

#### `list_int_new() -> List<int>`
Creates new empty list of integers.

```nano
let numbers: List<int> = (list_int_new)
```

#### `list_int_with_capacity(capacity: int) -> List<int>`
Creates list with pre-allocated capacity.

```nano
let numbers: List<int> = (list_int_with_capacity 100)
```

**Benefit:** Avoids reallocations if you know the size in advance.

#### `list_int_push(list: mut List<int>, value: int) -> void`
Appends value to end of list.

```nano
let mut numbers: List<int> = (list_int_new)
(list_int_push numbers 10)
(list_int_push numbers 20)
(list_int_push numbers 30)
# list is now [10, 20, 30]
```

#### `list_int_pop(list: mut List<int>) -> int`
Removes and returns last element.

```nano
let last: int = (list_int_pop numbers)  # 30
# list is now [10, 20]
```

**Returns:** Last element, or 0 if list is empty.

#### `list_int_get(list: List<int>, index: int) -> int`
Gets element at index.

```nano
let value: int = (list_int_get numbers 0)  # 10
```

**Returns:** Element at index, or 0 if out of bounds.

#### `list_int_set(list: mut List<int>, index: int, value: int) -> void`
Sets element at index.

```nano
(list_int_set numbers 1 25)
# numbers[1] is now 25
```

**Note:** Index must be valid (< length).

#### `list_int_insert(list: mut List<int>, index: int, value: int) -> void`
Inserts value at index, shifting elements right.

```nano
(list_int_insert numbers 1 15)
# If list was [10, 20, 30], now [10, 15, 20, 30]
```

#### `list_int_remove(list: mut List<int>, index: int) -> void`
Removes element at index, shifting elements left.

```nano
(list_int_remove numbers 1)
# If list was [10, 20, 30], now [10, 30]
```

#### `list_int_length(list: List<int>) -> int`
Gets number of elements in list.

```nano
let len: int = (list_int_length numbers)
```

#### `list_int_capacity(list: List<int>) -> int`
Gets allocated capacity (may be > length).

```nano
let cap: int = (list_int_capacity numbers)
```

#### `list_int_is_empty(list: List<int>) -> bool`
Checks if list has no elements.

```nano
if (list_int_is_empty numbers) {
    (println "List is empty")
}
```

#### `list_int_clear(list: mut List<int>) -> void`
Removes all elements (keeps capacity).

```nano
(list_int_clear numbers)
# length is now 0, but capacity unchanged
```

#### `list_int_free(list: mut List<int>) -> void`
Frees list memory.

```nano
(list_int_free numbers)
```

**Note:** List cannot be used after freeing.

### List<string> Operations

All operations available for `List<string>`:
- `list_string_new()`, `list_string_with_capacity()`
- `list_string_push()`, `list_string_pop()`
- `list_string_get()`, `list_string_set()`
- `list_string_insert()`, `list_string_remove()`
- `list_string_length()`, `list_string_capacity()`
- `list_string_is_empty()`, `list_string_clear()`, `list_string_free()`

**Example:**
```nano
let mut names: List<string> = (list_string_new)
(list_string_push names "Alice")
(list_string_push names "Bob")
(list_string_push names "Charlie")

for i in (range 0 (list_string_length names)) {
    let name: string = (list_string_get names i)
    (println name)
}
```

---

## Higher-Order Functions

### `map(arr: array<T>, f: fn(T) -> U) -> array<U>`
Transforms each element using function.

```nano
fn square(x: int) -> int {
    return (* x x)
}

let numbers: array<int> = [1, 2, 3, 4]
let squares: array<int> = (map numbers square)
# squares is [1, 4, 9, 16]
```

### `reduce(arr: array<T>, init: U, f: fn(U, T) -> U) -> U`
Reduces array to single value.

```nano
fn add(acc: int, x: int) -> int {
    return (+ acc x)
}

let numbers: array<int> = [1, 2, 3, 4]
let sum: int = (reduce numbers 0 add)
# sum is 10
```

### `fold(arr: array<T>, init: U, f: fn(U, T) -> U) -> U`
Alias for `reduce`.

```nano
let product: int = (fold numbers 1 multiply)
```

---

## Input/Output

### `getchar() -> int`
Reads single character from stdin.

```nano
(println "Press any key...")
let ch: int = (getchar)
(println (+ "You pressed: " (string_from_char ch)))
```

**Returns:** Character code (0-255), or -1 on EOF/error.

### `print_int(n: int) -> void`
Prints integer without newline.

```nano
(print_int 42)
(print_int 100)
# Output: 42100
```

### `print_float(f: float) -> void`
Prints float without newline.

```nano
(print_float 3.14159)
```

### `print_bool(b: bool) -> void`
Prints boolean as "true" or "false".

```nano
(print_bool true)   # Prints: true
(print_bool false)  # Prints: false
```

---

## Time and Sleep

### `time_now() -> int`
Gets current Unix timestamp (seconds since epoch).

```nano
let timestamp: int = (time_now)
(println (int_to_string timestamp))
```

### `time_ms() -> int`
Gets current time in milliseconds.

```nano
let start: int = (time_ms)
# ... do work ...
let end: int = (time_ms)
let elapsed: int = (- end start)
(println (+ "Elapsed: " (+ (int_to_string elapsed) "ms")))
```

### `sleep(ms: int) -> void`
Sleeps for specified milliseconds.

```nano
(println "Waiting 2 seconds...")
(sleep 2000)
(println "Done!")
```

---

## Process Spawning

### `spawn(command: string, args: array<string>) -> int`
Spawns child process with arguments.

```nano
let args: array<string> = ["-la", "/tmp"]
let pid: int = (spawn "ls" args)
```

**Returns:** Process ID (pid), or -1 on error.

### `waitpid(pid: int) -> int`
Waits for child process to complete.

```nano
let exit_code: int = (waitpid pid)
if (== exit_code 0) {
    (println "Process succeeded")
}
```

**Returns:** Exit code of child process.

---

## Usage Examples

### Example 1: Mathematical Computation

```nano
fn pythagorean(a: float, b: float) -> float {
    let a_squared: float = (pow a 2.0)
    let b_squared: float = (pow b 2.0)
    let c_squared: float = (+ a_squared b_squared)
    return (sqrt c_squared)
}

shadow pythagorean {
    # 3-4-5 triangle
    assert (== (pythagorean 3.0 4.0) 5.0)
}
```

### Example 2: String Processing

```nano
fn process_name(full_name: string) -> string {
    let len: int = (str_length full_name)
    
    if (str_contains full_name " ") {
        (println "Full name detected")
        return full_name
    } else {
        (println "Single name")
        return (str_concat "Mr. " full_name)
    }
}

shadow process_name {
    assert (str_equals (process_name "John") "Mr. John")
    assert (str_equals (process_name "John Doe") "John Doe")
}
```

### Example 3: Array Processing

```nano
fn sum_array(arr: array<int>) -> int {
    let mut total: int = 0
    let mut i: int = 0
    let len: int = (array_length arr)
    
    while (< i len) {
        set total (+ total (at arr i))
        set i (+ i 1)
    }
    
    return total
}

shadow sum_array {
    let nums: array<int> = [1, 2, 3, 4, 5]
    assert (== (sum_array nums) 15)
    
    let empty: array<int> = []
    assert (== (sum_array empty) 0)
}

fn main() -> int {
    # Create and manipulate arrays
    let mut data: array<int> = (array_new 5 0)
    
    # Fill with values
    (array_set data 0 10)
    (array_set data 1 20)
    (array_set data 2 30)
    (array_set data 3 40)
    (array_set data 4 50)
    
    let total: int = (sum_array data)
    (println total)  # Prints 150
    
    return 0
}
```

### Example 4: Trigonometric Calculation

```nano
fn calculate_angle(opposite: float, adjacent: float) -> float {
    # Calculate angle in radians using arctan
    # (Not implemented yet, but shows how trig functions work)
    let ratio: float = (/ opposite adjacent)
    return ratio  # Simplified - would use atan in real code
}

fn demonstrate_trig() -> int {
    let pi: float = 3.14159265359
    let pi_over_2: float = (/ pi 2.0)
    
    (println "sin(π/2) should be 1:")
    (println (sin pi_over_2))
    
    (println "cos(0) should be 1:")
    (println (cos 0.0))
    
    return 0
}
```

---

## Type Compatibility

### Polymorphic Functions
These functions accept multiple types:
- `println(any)` - Accepts int, float, string, bool
- `print(any)` - Accepts int, float, string, bool
- `abs(int|float)` - Returns same type as input
- `min(int|float, int|float)` - Both args must be same type
- `max(int|float, int|float)` - Both args must be same type

### Type-Converting Functions
These always return float regardless of input:
- `sqrt(int|float) -> float`
- `pow(int|float, int|float) -> float`
- `floor(int|float) -> float`
- `ceil(int|float) -> float`
- `round(int|float) -> float`
- `sin(int|float) -> float`
- `cos(int|float) -> float`
- `tan(int|float) -> float`

### String-Only Functions
These only work with strings:
- `str_length(string) -> int`
- `str_concat(string, string) -> string`
- `str_substring(string, int, int) -> string`
- `str_contains(string, string) -> bool`
- `str_equals(string, string) -> bool`

---

## Performance Notes

### Optimizations
- Math functions use C standard library directly (fast)
- String operations are bounds-checked (safe)
- `str_length` is O(n) - uses `strlen`
- `str_contains` is O(n*m) - uses `strstr`
- Memory for new strings is allocated on heap

### Memory Management
- String results are allocated with `malloc`
- In transpiled C code, memory should be managed carefully
- Shadow tests help catch memory issues early

---

## Future Standard Library Additions

Planned for future releases:
- **String:** `str_uppercase`, `str_lowercase`, `str_split`, `str_join`
- **Math:** `log`, `exp`, `atan`, `atan2`, `asin`, `acos`
- **File I/O:** `file_read`, `file_write`, `file_exists`
- **Arrays:** `array_map`, `array_filter`, `array_reduce`, `array_slice`

---

## Notes

- All functions are documented with shadow tests in example files
- Every stdlib function is tested in at least one example program
- Error messages include line and column numbers
- Type checking happens at compile time
- Shadow tests execute during compilation to verify stdlib correctness

**Total Functions:** 72 across 9 categories (see spec.json for authoritative list)
- I/O: 3 functions (print, println, assert)
- Math: 11 functions (abs, min, max, sqrt, pow, floor, ceil, round, sin, cos, tan)
- String: 18 functions (char operations, classification, conversions)
- Binary String: 12 functions (bstring operations for binary data)
- Array: 10 functions (at, length, new, set, push, pop, remove_at, filter, map, reduce)
- OS: 3 functions (getcwd, getenv, range)
- Generics: 4 functions (List<T> operations)
- Checked Math: 5 functions (overflow-safe arithmetic)

**Documentation Status:** 151 of 166 builtin functions documented (91% coverage)

**Categories:**
- Core I/O: 3 functions
- Input/Output: 4 functions (getchar, print_int, print_float, print_bool)
- Math Operations: 26 functions (basic, advanced, trigonometric, hyperbolic)
- String Operations: 18 functions
- Character Operations: 11 functions
- Type Conversion: 6 functions
- Array Operations: 9 functions
- Array Advanced: 5 functions (push, pop, slice, remove_at, filter)
- HashMap Operations: 10 functions (interpreter-only)
- List<int> Operations: 14 functions
- List<string> Operations: 14 functions
- Higher-Order Functions: 3 functions (map, reduce, fold)
- File Operations: 10 functions
- Directory Operations: 5 functions
- Path Operations: 6 functions
- Process Operations: 5 functions
- Process Spawning: 2 functions (spawn, waitpid)
- Time and Sleep: 3 functions (time_now, time_ms, sleep)
- Binary String Operations: 8 functions
- Result Type Operations: 5 functions

**Remaining:** 15 functions need documentation

**HashMap Note:** HashMap operations are interpreter-only and not available in compiled code.

**Test Coverage:** All functions have shadow tests in their implementation

