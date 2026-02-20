# My Standard Library Reference

I provide these built-in functions. They are my core capabilities, available to you without external modules.

---

## Core I/O Functions (3)

### `print(value: any) -> void`
I print a value without a trailing newline.

```nano
print "Hello"
print 42
print 3.14
```

### `println(value: any) -> void`
I print a value with a trailing newline. This is polymorphic - it works with int, float, string, and bool.

```nano
(println "Hello, World!")
(println 42)
(println 3.14159)
(println true)
```

### `assert(condition: bool) -> void`
My runtime assertion. I terminate the program if the condition is false.

```nano
assert (== 2 2)           # Passes
assert (> 5 3)            # Passes
assert (== (add 2 2) 4)   # Passes
```

---

## Math Operations (11)

### Basic Math (3)

#### `abs(x: int|float) -> int|float`
I return the absolute value. I preserve the type: int becomes int, float becomes float.

```nano
(abs -5)      # Returns 5
(abs 5)       # Returns 5
(abs -3.14)   # Returns 3.14
```

#### `min(a: int|float, b: int|float) -> int|float`
I return the minimum of two values. Both arguments must be the same type.

```nano
(min 5 10)    # Returns 5
(min 3.14 2.71)  # Returns 2.71
```

#### `max(a: int|float, b: int|float) -> int|float`
I return the maximum of two values. Both arguments must be the same type.

```nano
(max 5 10)    # Returns 10
(max 3.14 2.71)  # Returns 3.14
```

### Advanced Math (5)

#### `sqrt(x: int|float) -> float`
I return the square root. I always return a float.

```nano
(sqrt 16.0)   # Returns 4.0
(sqrt 2.0)    # Returns 1.41421...
(sqrt 9)      # Returns 3.0 (int converted to float)
```

#### `pow(base: int|float, exponent: int|float) -> float`
I return the base raised to the power of the exponent. I always return a float.

```nano
(pow 2.0 3.0)    # Returns 8.0 (2³)
(pow 5.0 2.0)    # Returns 25.0 (5²)
(pow 2.0 -1.0)   # Returns 0.5 (2⁻¹)
```

#### `floor(x: int|float) -> float`
I return the largest integer ≤ x as a float.

```nano
(floor 3.7)     # Returns 3.0
(floor 3.2)     # Returns 3.0
(floor -2.3)    # Returns -3.0
```

#### `ceil(x: int|float) -> float`
I return the smallest integer ≥ x as a float.

```nano
(ceil 3.2)      # Returns 4.0
(ceil 3.7)      # Returns 4.0
(ceil -2.7)     # Returns -2.0
```

#### `round(x: int|float) -> float`
I round to the nearest integer as a float.

```nano
(round 3.4)     # Returns 3.0
(round 3.6)     # Returns 4.0
(round 3.5)     # Returns 4.0 (rounds half to even)
```

### Trigonometric Functions (3)

I evaluate all trig functions in **radians**. They always return a float.

#### `sin(x: int|float) -> float`
I return the sine of x (in radians).

```nano
(sin 0.0)       # Returns 0.0
(sin 1.5708)    # Returns ≈1.0 (π/2)
```

#### `cos(x: int|float) -> float`
I return the cosine of x (in radians).

```nano
(cos 0.0)       # Returns 1.0
(cos 3.14159)   # Returns ≈-1.0 (π)
```

#### `tan(x: int|float) -> float`
I return the tangent of x (in radians).

```nano
(tan 0.0)       # Returns 0.0
(tan 0.7854)    # Returns ≈1.0 (π/4)
```

#### `asin(x: float) -> float`
I return the arcsine of x (inverse sine) in radians.

```nano
(asin 1.0)      # Returns ≈1.5708 (π/2)
(asin 0.5)      # Returns ≈0.5236 (π/6)
```

**Domain:** -1.0 to 1.0
**Range:** -π/2 to π/2

#### `acos(x: float) -> float`
I return the arccosine of x (inverse cosine) in radians.

```nano
(acos 1.0)      # Returns 0.0
(acos 0.0)      # Returns ≈1.5708 (π/2)
```

**Domain:** -1.0 to 1.0
**Range:** 0 to π

#### `atan(x: float) -> float`
I return the arctangent of x (inverse tangent) in radians.

```nano
(atan 1.0)      # Returns ≈0.7854 (π/4)
(atan 0.0)      # Returns 0.0
```

**Range:** -π/2 to π/2

---

## String Operations (18)

### `str_length(s: string) -> int`
I return the length of a string in bytes.

```nano
let text: string = "Hello"
let len: int = (str_length text)  # Returns 5
```

### `str_concat(s1: string, s2: string) -> string`
I concatenate two strings and return a new string.

```nano
let hello: string = "Hello"
let world: string = " World"
let result: string = (str_concat hello world)  # "Hello World"
```

### `str_substring(s: string, start: int, length: int) -> string`
I extract a substring starting at `start` with the given `length`.
- `start` is 0-indexed
- If `start + length` exceeds string length, I return until the end of the string
- I return an empty string if start is out of bounds

```nano
let text: string = "Hello, World!"
let hello: string = (str_substring text 0 5)   # "Hello"
let world: string = (str_substring text 7 5)   # "World"
```

### `str_contains(s: string, substr: string) -> bool`
I return true if string `s` contains substring `substr`.

```nano
let text: string = "The quick brown fox"
(str_contains text "quick")   # Returns true
(str_contains text "slow")    # Returns false
```

### `str_equals(s1: string, s2: string) -> bool`
I return true if both strings are exactly equal.

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
I return the ASCII value of the character at the specified index.
- Index is 0-based
- I return the integer ASCII value (0-255)
- **Bounds-checked** - I terminate with an error if the index is out of bounds

```nano
let text: string = "Hello"
let h: int = (char_at text 0)  # Returns 72 (ASCII 'H')
let e: int = (char_at text 1)  # Returns 101 (ASCII 'e')
let o: int = (char_at text 4)  # Returns 111 (ASCII 'o')
```

I use this for lexical analysis and character-by-character parsing.

#### `string_from_char(c: int) -> string`
I create a single-character string from an ASCII value.

```nano
let a: string = (string_from_char 65)   # Returns "A"
let z: string = (string_from_char 90)   # Returns "Z"
let zero: string = (string_from_char 48)  # Returns "0"
let space: string = (string_from_char 32)  # Returns " "
```

I use this when building strings character-by-character, such as during code generation.

### Character Classification (6)

#### `is_digit(c: int) -> bool`
I return true if the character is a digit ('0'-'9').

```nano
(is_digit 48)   # Returns true  ('0')
(is_digit 53)   # Returns true  ('5')
(is_digit 57)   # Returns true  ('9')
(is_digit 65)   # Returns false ('A')
(is_digit 97)   # Returns false ('a')
```

I use this for token classification in lexical analysis.

#### `is_alpha(c: int) -> bool`
I return true if the character is a letter (a-z, A-Z).

```nano
(is_alpha 65)   # Returns true  ('A')
(is_alpha 90)   # Returns true  ('Z')
(is_alpha 97)   # Returns true  ('a')
(is_alpha 122)  # Returns true  ('z')
(is_alpha 48)   # Returns false ('0')
(is_alpha 32)   # Returns false (' ')
```

I use this for identifier validation in parsers.

#### `is_alnum(c: int) -> bool`
I return true if the character is alphanumeric (digit or letter).

```nano
(is_alnum 48)   # Returns true  ('0')
(is_alnum 65)   # Returns true  ('A')
(is_alnum 97)   # Returns true  ('a')
(is_alnum 32)   # Returns false (' ')
(is_alnum 33)   # Returns false ('!')
```

I use this to check if a character is valid in an identifier.

#### `is_whitespace(c: int) -> bool`
I return true if the character is whitespace (space, tab, newline, carriage return).

```nano
(is_whitespace 32)  # Returns true  (' ')
(is_whitespace 9)   # Returns true  ('\t')
(is_whitespace 10)  # Returns true  ('\n')
(is_whitespace 13)  # Returns true  ('\r')
(is_whitespace 65)  # Returns false ('A')
```

I use this to skip whitespace during tokenization.

#### `is_upper(c: int) -> bool`
I return true if the character is an uppercase letter (A-Z).

```nano
(is_upper 65)   # Returns true  ('A')
(is_upper 90)   # Returns true  ('Z')
(is_upper 77)   # Returns true  ('M')
(is_upper 97)   # Returns false ('a')
(is_upper 48)   # Returns false ('0')
```

#### `is_lower(c: int) -> bool`
I return true if the character is a lowercase letter (a-z).

```nano
(is_lower 97)   # Returns true  ('a')
(is_lower 122)  # Returns true  ('z')
(is_lower 109)  # Returns true  ('m')
(is_lower 65)   # Returns false ('A')
(is_lower 48)   # Returns false ('0')
```

I provide these C-style aliases for compatibility and familiarity, though I prefer the `is_*` style in new code:
- `isdigit` (alias for `is_digit`)
- `isalpha` (alias for `is_alpha`)
- `isalnum` (alias for `is_alnum`)
- `isspace` (alias for `is_whitespace`)
- `isupper` (alias for `is_upper`)
- `islower` (alias for `is_lower`)
- `isprint` (returns true for printable characters)
- `ispunct` (returns true for punctuation characters)
- `tolower` (alias for `char_to_lower`)
- `toupper` (alias for `char_to_upper`)

### Type Conversions (5)

#### `int_to_string(n: int) -> string`
I convert an integer to its string representation.

```nano
let s1: string = (int_to_string 42)    # Returns "42"
let s2: string = (int_to_string 0)     # Returns "0"
let s3: string = (int_to_string -100)  # Returns "-100"
let s4: string = (int_to_string 999)   # Returns "999"
```

I use this for formatting numbers for output, error messages, and code generation.

#### `string_to_int(s: string) -> int`
I parse a string to an integer. I return 0 if the string cannot be parsed.

```nano
let n1: int = (string_to_int "42")     # Returns 42
let n2: int = (string_to_int "0")      # Returns 0
let n3: int = (string_to_int "-100")   # Returns -100
let n4: int = (string_to_int "12345")  # Returns 12345
```

I use this to parse numeric literals during compilation.

#### `digit_value(c: int) -> int`
I convert a digit character to its numeric value. I return -1 if it is not a digit.

```nano
(digit_value 48)  # Returns 0  ('0' -> 0)
(digit_value 49)  # Returns 1  ('1' -> 1)
(digit_value 53)  # Returns 5  ('5' -> 5)
(digit_value 57)  # Returns 9  ('9' -> 9)
(digit_value 65)  # Returns -1 ('A' is not a digit)
```

I use this when parsing multi-digit numbers character-by-character.

#### `char_to_lower(c: int) -> int`
I convert an uppercase letter to lowercase. I leave non-letters unchanged.

```nano
(char_to_lower 65)   # Returns 97  ('A' -> 'a')
(char_to_lower 90)   # Returns 122 ('Z' -> 'z')
(char_to_lower 77)   # Returns 109 ('M' -> 'm')
(char_to_lower 97)   # Returns 97  ('a' -> 'a', already lowercase)
(char_to_lower 48)   # Returns 48  ('0' -> '0', not a letter)
```

I use this for case-insensitive comparisons and keyword normalization.

#### `char_to_upper(c: int) -> int`
I convert a lowercase letter to uppercase. I leave non-letters unchanged.

```nano
(char_to_upper 97)   # Returns 65  ('a' -> 'A')
(char_to_upper 122)  # Returns 90  ('z' -> 'Z')
(char_to_upper 109)  # Returns 77  ('m' -> 'M')
(char_to_upper 65)   # Returns 65  ('A' -> 'A', already uppercase)
(char_to_upper 48)   # Returns 48  ('0' -> '0', not a letter)
```

I use this to normalize identifiers and format output.

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
I return the element at the specified index. I perform **bounds-checking** and terminate with an error if the index is out of bounds.

```nano
let nums: array<int> = [1, 2, 3, 4, 5]
let first: int = (at nums 0)   # Returns 1
let last: int = (at nums 4)    # Returns 5
# (at nums 10)                  # ERROR: index out of bounds!
```

I include runtime bounds checking to prevent memory corruption and security vulnerabilities.

### `array_length(arr: array<T>) -> int`
I return the length (number of elements) of an array.

```nano
let nums: array<int> = [10, 20, 30]
let len: int = (array_length nums)  # Returns 3

let empty: array<int> = []
let zero: int = (array_length empty)  # Returns 0
```

### `array_new(size: int, default: T) -> array<T>`
I create a new array with the specified size, filled with the default value.

```nano
# Create array of 5 zeros
let zeros: array<int> = (array_new 5 0)
# [0, 0, 0, 0, 0]

# Create array of 3 empty strings
let strings: array<string> = (array_new 3 "")
# ["", "", ""]
```

The size must be non-negative. I will cause an error if you provide a negative size.

### `array_set(arr: mut array<T>, index: int, value: T) -> void`
I set the element at the specified index. I perform **bounds-checking** and terminate with an error if the index is out of bounds. I require a **mutable** array.

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

My **HashMap<K,V>** is a key-value collection with **O(1)** average lookup.

My current constraints:
- **Key type:** `int` or `string`
- **Value type:** `int` or `string`
- I require an explicit type annotation: `HashMap<K,V>`

### `map_new() -> HashMap<K,V>`
I create a new hash map. I require a type annotation on the variable.

```nano
let hm: HashMap<string, int> = (map_new)
```

### `map_put(map: HashMap<K,V>, key: K, value: V) -> void`
I insert or update a key/value pair.

```nano
(map_put hm "alice" 10)
(map_put hm "bob" 20)
```

### `map_get(map: HashMap<K,V>, key: K) -> V`
I return the value for a key, or a default (`0` or `""`) if it is missing.

```nano
let score: int = (map_get hm "alice")
```

### `map_has(map: HashMap<K,V>, key: K) -> bool`
I check if a key exists.

```nano
if (map_has hm "alice") { (println "found") }
```

### `map_remove(map: HashMap<K,V>, key: K) -> void`
I remove a key/value pair if it is present.

### `map_length(map: HashMap<K,V>) -> int`
I return the number of entries. My alias is `map_size`.

### `map_clear(map: HashMap<K,V>) -> void`
I remove all entries without freeing the map.

I use ARC to manage my HashMaps. They are automatically freed when they go out of scope, so `map_free` is no longer needed.

### `map_keys(map: HashMap<K,V>) -> array<K>`
I return all keys as an array.

### `map_values(map: HashMap<K,V>) -> array<V>`
I return all values as an array.

I enforce these safety rules:
- I require the array to be declared `mut`
- My runtime bounds checking prevents buffer overflows
- My type checking ensures homogeneous arrays

---

## Result Type Operations (5)

My Result<T, E> type represents success (Ok) or failure (Err) values.

### `result_is_ok(r: Result<T, E>) -> bool`
I check if the Result is an Ok variant.

```nano
let r: Result<int, string> = (divide 10 2)
if (result_is_ok r) {
    (println "Success!")
}
```

### `result_is_err(r: Result<T, E>) -> bool`
I check if the Result is an Err variant.

```nano
let r: Result<int, string> = (divide 10 0)
if (result_is_err r) {
    (println "Error occurred")
}
```

### `result_unwrap(r: Result<T, E>) -> T`
I extract the Ok value. I panic if it is an Err.

```nano
let r: Result<int, string> = (divide 10 2)
let value: int = (result_unwrap r)  # 5
```

I warn you that I panic if the Result is an Err. Use `result_is_ok` to check first.

### `result_unwrap_or(r: Result<T, E>, default: T) -> T`
I extract the Ok value, or I return the default if it is an Err.

```nano
let r: Result<int, string> = (divide 10 0)
let value: int = (result_unwrap_or r 0)  # Returns 0 (default)
```

I offer this as a safe alternative to `result_unwrap`.

### `result_unwrap_err(r: Result<T, E>) -> E`
I extract the Err value. I panic if it is an Ok.

```nano
let r: Result<int, string> = (divide 10 0)
if (result_is_err r) {
    let error: string = (result_unwrap_err r)
    (println error)
}
```

---

## Binary String Operations (8)

I use binary strings (bstring) to handle raw binary data.

### `bytes_from_string(s: string) -> bstring`
I convert a string to a binary string.

```nano
let bs: bstring = (bytes_from_string "Hello")
```

### `string_from_bytes(bs: bstring) -> string`
I convert a binary string to a regular string.

```nano
let s: string = (string_from_bytes bs)
```

### `bstring_length(bs: bstring) -> int`
I return the binary string length in bytes.

```nano
let len: int = (bstring_length bs)
```

### `bstring_at(bs: bstring, index: int) -> int`
I return the byte value at the index.

```nano
let byte: int = (bstring_at bs 0)  # Returns 0-255
```

### `bstring_slice(bs: bstring, start: int, length: int) -> bstring`
I extract a portion of a binary string.

```nano
let subset: bstring = (bstring_slice bs 0 10)
```

### `bstring_concat(bs1: bstring, bs2: bstring) -> bstring`
I concatenate two binary strings.

```nano
let combined: bstring = (bstring_concat bs1 bs2)
```

### `bstring_new(size: int) -> bstring`
I create a new binary string of the specified size, which I zero-fill.

```nano
let bs: bstring = (bstring_new 1024)
```

### `bstring_set(bs: mut bstring, index: int, value: int) -> void`
I set the byte at the index (value 0-255).

```nano
(bstring_set bs 0 65)  # Sets first byte to 'A'
```

My bounds checking prevents buffer overflows.

---

## OS/System Functions (3)

### `getcwd() -> string`
I return the current working directory as an absolute path.

```nano
let cwd: string = (getcwd)
(println cwd)  # Prints something like "/Users/username/project"
```

### `getenv(name: string) -> string`
I get an environment variable value. I return an empty string if it is not set.

```nano
let home: string = (getenv "HOME")
let path: string = (getenv "PATH")
```

### `range(start: int, end: int) -> iterator`
I provide this special function for use only in `for` loops. I create an iterator from `start` (inclusive) to `end` (exclusive).

```nano
for i in (range 0 10) {
    (println i)  # Prints 0, 1, 2, ..., 9
}
```

I only allow `range` to be used in for-loop contexts.

---

## File Operations (10)

### `file_read(path: string) -> string`
I read entire file contents as a string.

```nano
let content: string = (file_read "data.txt")
(println content)
```

I return file contents as a string, or an empty string on error.

### `file_read_bytes(path: string) -> bstring`
I read file contents as a binary string (bstring).

```nano
let data: bstring = (file_read_bytes "image.png")
let size: int = (bstring_length data)
```

I recommend this for binary files, images, and non-text data.

### `file_write(path: string, content: string) -> int`
I write string content to a file. I overwrite it if it exists.

```nano
let status: int = (file_write "output.txt" "Hello, World!")
if (== status 0) {
    (println "Write successful")
} else {
    (println "Write failed")
}
```

I return 0 on success and 1 on failure.

### `file_append(path: string, content: string) -> int`
I append string content to the end of a file.

```nano
let status: int = (file_append "log.txt" "New log entry\n")
```

I return 0 on success and 1 on failure.

### `file_exists(path: string) -> bool`
I check if a file exists and is accessible.

```nano
if (file_exists "config.json") {
    let config: string = (file_read "config.json")
} else {
    (println "Config file not found")
}
```

### `file_remove(path: string) -> int`
I delete a file.

```nano
let status: int = (file_remove "temp.txt")
```

I return 0 on success and 1 on failure. I warn you that this is a permanent deletion with no confirmation.

### `file_rename(old_path: string, new_path: string) -> int`
I rename or move a file.

```nano
let status: int = (file_rename "old.txt" "new.txt")
```

I return 0 on success and 1 on failure.

### `file_size(path: string) -> int`
I return the file size in bytes.

```nano
let size: int = (file_size "data.bin")
(println (+ "File size: " (int_to_string size)))
```

I return the file size in bytes, or -1 on error.

### `file_copy(src: string, dst: string) -> int`
I copy a file from a source to a destination.

```nano
let status: int = (file_copy "original.txt" "backup.txt")
```

I return 0 on success and 1 on failure.

### `file_chmod(path: string, mode: int) -> int`
I change file permissions using Unix mode bits.

```nano
# Make file executable: chmod +x
let status: int = (file_chmod "script.sh" 493)  # 0755 in octal
```

I return 0 on success and 1 on failure.

---

## Directory Operations (5)

### `dir_exists(path: string) -> bool`
I check if a directory exists.

```nano
if (not (dir_exists "output")) {
    (dir_create "output")
}
```

### `dir_create(path: string) -> int`
I create a directory. Parent directories must exist.

```nano
let status: int = (dir_create "build/output")
```

I return 0 on success and 1 on failure. I do not create parent directories; use `dir_create_all` for that.

### `dir_remove(path: string) -> int`
I remove an empty directory.

```nano
let status: int = (dir_remove "temp")
```

I return 0 on success and 1 on failure. I require the directory to be empty.

### `dir_list(path: string) -> array<string>`
I list all entries in a directory.

```nano
let entries: array<string> = (dir_list ".")
for entry in entries {
    (println entry)
}
```

I return an array of filenames, not full paths. I return an empty array on error.

### `chdir(path: string) -> int`
I change the current working directory.

```nano
let status: int = (chdir "/tmp")
let cwd: string = (getcwd)
(println cwd)  # Prints "/tmp"
```

I return 0 on success and 1 on failure.

### `fs_walkdir(path: string, callback: fn(string) -> void) -> void`
I recursively walk a directory tree and call the callback for each file.

```nano
fn print_file(filepath: string) -> void {
    (println filepath)
}

# Print all files in current directory recursively
(fs_walkdir "." print_file)
```

I provide this to help you find files matching a pattern or to calculate total sizes.

---

## Path Operations (6)

### `path_join(a: string, b: string) -> string`
I join two path components with the proper separator.

```nano
let full_path: string = (path_join "/home/user" "documents")
# Result: "/home/user/documents"
```

I am platform-aware: I use `/` on Unix and `\` on Windows.

### `path_basename(path: string) -> string`
I extract the filename from a path.

```nano
let filename: string = (path_basename "/path/to/file.txt")
# Result: "file.txt"
```

### `path_dirname(path: string) -> string`
I extract the directory path.

```nano
let dir: string = (path_dirname "/path/to/file.txt")
# Result: "/path/to"
```

### `path_extension(path: string) -> string`
I extract the file extension.

```nano
let ext: string = (path_extension "document.pdf")
# Result: "pdf"
```

### `path_normalize(path: string) -> string`
I normalize a path by removing `.`, `..`, and redundant separators.

```nano
let clean: string = (path_normalize "./foo/../bar/./baz")
# Result: "bar/baz"
```

### `path_absolute(path: string) -> string`
I convert a relative path to an absolute path.

```nano
let abs: string = (path_absolute "file.txt")
# Result: "/current/working/dir/file.txt"
```

---

## Process Operations (5)

### `system(command: string) -> int`
I execute a shell command and wait for it to complete.

```nano
let status: int = (system "ls -la")
```

I return the exit code from the command. I advise caution regarding command injection risks with user input.

### `exit(code: int) -> void`
I terminate the program with an exit code.

```nano
if (not (file_exists "required.txt")) {
    (println "Error: required.txt not found")
    (exit 1)
}
```

I do not return; I exit immediately.

### `getenv(name: string) -> string`
I get an environment variable value.

```nano
let home: string = (getenv "HOME")
let user: string = (getenv "USER")
```

I return the variable value, or an empty string if it is not set.

### `setenv(name: string, value: string) -> int`
I set an environment variable for the current process.

```nano
let status: int = (setenv "MY_VAR" "my_value")
```

I return 0 on success and 1 on failure. This only affects the current process and its children.

### `unsetenv(name: string) -> int`
I remove an environment variable.

```nano
let status: int = (unsetenv "MY_VAR")
```

I return 0 on success and 1 on failure.

---

## Advanced Math Functions (15)

### Exponential and Logarithmic

#### `exp(x: float) -> float`
I return e^x (the exponential function).

```nano
let result: float = (exp 2.0)  # e^2 ≈ 7.389
```

#### `log(x: float) -> float`
My natural logarithm (base e).

```nano
let result: float = (log 10.0)  # ln(10) ≈ 2.303
```

#### `log10(x: float) -> float`
My base-10 logarithm.

```nano
let result: float = (log10 100.0)  # Result: 2.0
```

#### `log2(x: float) -> float`
My base-2 logarithm.

```nano
let result: float = (log2 8.0)  # Result: 3.0
```

### Hyperbolic Functions

#### `sinh(x: float) -> float`
I return the hyperbolic sine.

```nano
let result: float = (sinh 1.0)
```

#### `cosh(x: float) -> float`
I return the hyperbolic cosine.

```nano
let result: float = (cosh 1.0)
```

#### `tanh(x: float) -> float`
I return the hyperbolic tangent.

```nano
let result: float = (tanh 0.5)
```

#### `asinh(x: float) -> float`
I return the inverse hyperbolic sine.

#### `acosh(x: float) -> float`
I return the inverse hyperbolic cosine.

#### `atanh(x: float) -> float`
I return the inverse hyperbolic tangent.

### Advanced Operations

#### `cbrt(x: float) -> float`
I return the cube root.

```nano
let result: float = (cbrt 27.0)  # Result: 3.0
```

#### `hypot(x: float, y: float) -> float`
I return the hypotenuse: sqrt(x² + y²).

```nano
let dist: float = (hypot 3.0 4.0)  # Result: 5.0
```

#### `fmod(x: float, y: float) -> float`
I return the floating-point remainder.

```nano
let remainder: float = (fmod 5.5 2.0)  # Result: 1.5
```

#### `copysign(x: float, y: float) -> float`
I return x with the sign of y.

```nano
let result: float = (copysign 3.0 -1.0)  # Result: -3.0
```

#### `fmax(x: float, y: float) -> float`
I return the maximum of two floats. I handle NaN correctly.

```nano
let max_val: float = (fmax 3.5 2.1)  # Result: 3.5
```

#### `fmin(x: float, y: float) -> float`
I return the minimum of two floats. I handle NaN correctly.

```nano
let min_val: float = (fmin 3.5 2.1)  # Result: 2.1
```

#### `fabs(x: float) -> float`
I return the absolute value for floats.

```nano
let val: float = (fabs -3.5)  # Result: 3.5
```

I remind you to use `abs` for integers.

---

## Type Conversion (10)

### `int_to_string(n: int) -> string`
I convert an integer to a string.

```nano
let s: string = (int_to_string 42)  # "42"
```

### `float_to_string(f: float) -> string`
I convert a float to a string.

```nano
let s: string = (float_to_string 3.14)  # "3.14"
```

### `string_to_int(s: string) -> int`
I parse a string to an integer.

```nano
let n: int = (string_to_int "123")  # 123
```

I return the parsed integer, or 0 if parsing fails.

### `string_to_float(s: string) -> float`
I parse a string to a float.

```nano
let f: float = (string_to_float "3.14")  # 3.14
```

I return the parsed float, or 0.0 if parsing fails.

### `bool_to_string(b: bool) -> string`
I convert a boolean to a string.

```nano
let s: string = (bool_to_string true)  # "true"
```

### `char_to_string(c: int) -> string`
I convert a character code to a string.

```nano
let s: string = (char_to_string 65)  # "A"
```

### `cast_int(value: any) -> int`
I cast a value to an integer.

```nano
let i: int = (cast_int 3.14)  # 3
let i2: int = (cast_int "42")  # 42
```

I truncate floats and parse strings during this operation.

### `cast_float(value: any) -> float`
I cast a value to a float.

```nano
let f: float = (cast_float 42)  # 42.0
let f2: float = (cast_float "3.14")  # 3.14
```

### `cast_string(value: any) -> string`
I cast a value to a string.

```nano
let s: string = (cast_string 42)  # "42"
let s2: string = (cast_string 3.14)  # "3.14"
let s3: string = (cast_string true)  # "true"
```

### `cast_bool(value: any) -> bool`
I cast a value to a boolean.

```nano
let b: bool = (cast_bool 1)  # true
let b2: bool = (cast_bool 0)  # false
let b3: bool = (cast_bool "")  # false
let b4: bool = (cast_bool "hello")  # true
```

I evaluate 0, an empty string, or null as false; everything else becomes true.

---

## Character Operations (5)

### `char_at(s: string, index: int) -> int`
I return the character code at the index.

```nano
let code: int = (char_at "Hello" 0)  # 72 ('H')
```

I return the character code (0-127 for ASCII), or 0 if the index is out of bounds.

### `char_to_lower(c: int) -> int`
I convert a character code to lowercase.

```nano
let lower: int = (char_to_lower 65)  # 97 ('a')
```

### `char_to_upper(c: int) -> int`
I convert a character code to uppercase.

```nano
let upper: int = (char_to_upper 97)  # 65 ('A')
```

### `digit_value(c: int) -> int`
I convert a digit character to its numeric value.

```nano
let val: int = (digit_value 53)  # 5 (from '5')
```

I return the digit value (0-9), or -1 if it is not a digit.

### `string_from_char(c: int) -> string`
I create a single-character string from a code.

```nano
let s: string = (string_from_char 65)  # "A"
```

---

## Array Advanced Operations (5)

### `array_push(arr: mut array<T>, value: T) -> void`
I append an element to the end of an array.

```nano
let mut numbers: array<int> = [1, 2, 3]
(array_push numbers 4)
# numbers is now [1, 2, 3, 4]
```

I require the array to be declared `mut`.

### `array_pop(arr: mut array<int>) -> int`
I remove and return the last element.

```nano
let mut stack: array<int> = [1, 2, 3]
let last: int = (array_pop stack)  # 3
# stack is now [1, 2]
```

I return the last element, or 0 if the array is empty.

### `array_slice(arr: array<T>, start: int, length: int) -> array<T>`
I create a sub-array from a portion of an array.

```nano
let numbers: array<int> = [1, 2, 3, 4, 5]
let subset: array<int> = (array_slice numbers 1 3)
# subset is [2, 3, 4]
```

### `array_remove_at(arr: mut array<T>, index: int) -> void`
I remove the element at the index and shift remaining elements.

```nano
let mut items: array<int> = [10, 20, 30, 40]
(array_remove_at items 1)
# items is now [10, 30, 40]
```

### `filter(arr: array<T>, predicate: fn(T) -> bool) -> array<T>`
I create a new array with elements that match the predicate.

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

I provide dynamic lists for resizable, type-safe collections.

### List<int> Operations

#### `list_int_new() -> List<int>`
I create a new empty list of integers.

```nano
let numbers: List<int> = (list_int_new)
```

#### `list_int_with_capacity(capacity: int) -> List<int>`
I create a list with pre-allocated capacity.

```nano
let numbers: List<int> = (list_int_with_capacity 100)
```

I use this to avoid reallocations when you know the size in advance.

#### `list_int_push(list: mut List<int>, value: int) -> void`
I append a value to the end of a list.

```nano
let mut numbers: List<int> = (list_int_new)
(list_int_push numbers 10)
(list_int_push numbers 20)
(list_int_push numbers 30)
# list is now [10, 20, 30]
```

#### `list_int_pop(list: mut List<int>) -> int`
I remove and return the last element.

```nano
let last: int = (list_int_pop numbers)  # 30
# list is now [10, 20]
```

I return the last element, or 0 if the list is empty.

#### `list_int_get(list: List<int>, index: int) -> int`
I return the element at the index.

```nano
let value: int = (list_int_get numbers 0)  # 10
```

I return the element at the index, or 0 if it is out of bounds.

#### `list_int_set(list: mut List<int>, index: int, value: int) -> void`
I set the element at the index.

```nano
(list_int_set numbers 1 25)
# numbers[1] is now 25
```

I require the index to be valid.

#### `list_int_insert(list: mut List<int>, index: int, value: int) -> void`
I insert a value at the index and shift elements to the right.

```nano
(list_int_insert numbers 1 15)
# If list was [10, 20, 30], now [10, 15, 20, 30]
```

#### `list_int_remove(list: mut List<int>, index: int) -> void`
I remove the element at the index and shift elements to the left.

```nano
(list_int_remove numbers 1)
# If list was [10, 20, 30], now [10, 30]
```

#### `list_int_length(list: List<int>) -> int`
I return the number of elements in the list.

```nano
let len: int = (list_int_length numbers)
```

#### `list_int_capacity(list: List<int>) -> int`
I return the allocated capacity.

```nano
let cap: int = (list_int_capacity numbers)
```

#### `list_int_is_empty(list: List<int>) -> bool`
I check if the list has no elements.

```nano
if (list_int_is_empty numbers) {
    (println "List is empty")
}
```

#### `list_int_clear(list: mut List<int>) -> void`
I remove all elements but keep the capacity.

```nano
(list_int_clear numbers)
# length is now 0, but capacity unchanged
```

#### `list_int_free(list: mut List<int>) -> void`
I free the list memory.

```nano
(list_int_free numbers)
```

I will not allow you to use the list after it is freed.

### List<string> Operations

I provide all these operations for `List<string>` as well:
- `list_string_new()`, `list_string_with_capacity()`
- `list_string_push()`, `list_string_pop()`
- `list_string_get()`, `list_string_set()`
- `list_string_insert()`, `list_string_remove()`
- `list_string_length()`, `list_string_capacity()`
- `list_string_is_empty()`, `list_string_clear()`, `list_string_free()`

Example:
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
I transform each element using the provided function.

```nano
fn square(x: int) -> int {
    return (* x x)
}

let numbers: array<int> = [1, 2, 3, 4]
let squares: array<int> = (map numbers square)
# squares is [1, 4, 9, 16]
```

### `reduce(arr: array<T>, init: U, f: fn(U, T) -> U) -> U`
I reduce an array to a single value.

```nano
fn add(acc: int, x: int) -> int {
    return (+ acc x)
}

let numbers: array<int> = [1, 2, 3, 4]
let sum: int = (reduce numbers 0 add)
# sum is 10
```

### `fold(arr: array<T>, init: U, f: fn(U, T) -> U) -> U`
My alias for `reduce`.

```nano
let product: int = (fold numbers 1 multiply)
```

---

## Input/Output

### `getchar() -> int`
I read a single character from stdin.

```nano
(println "Press any key...")
let ch: int = (getchar)
(println (+ "You pressed: " (string_from_char ch)))
```

I return the character code (0-255), or -1 on EOF or error.

### `print_int(n: int) -> void`
I print an integer without a newline.

```nano
(print_int 42)
(print_int 100)
# Output: 42100
```

### `print_float(f: float) -> void`
I print a float without a newline.

```nano
(print_float 3.14159)
```

### `print_bool(b: bool) -> void`
I print a boolean as "true" or "false".

```nano
(print_bool true)   # Prints: true
(print_bool false)  # Prints: false
```

---

## Time and Sleep

### `time_now() -> int`
I return the current Unix timestamp (seconds since the epoch).

```nano
let timestamp: int = (time_now)
(println (int_to_string timestamp))
```

### `time_ms() -> int`
I return the current time in milliseconds.

```nano
let start: int = (time_ms)
# ... do work ...
let end: int = (time_ms)
let elapsed: int = (- end start)
(println (+ "Elapsed: " (+ (int_to_string elapsed) "ms")))
```

### `sleep(ms: int) -> void`
I sleep for the specified number of milliseconds.

```nano
(println "Waiting 2 seconds...")
(sleep 2000)
(println "Done!")
```

---

## Process Spawning

### `spawn(command: string, args: array<string>) -> int`
I spawn a child process with arguments.

```nano
let args: array<string> = ["-la", "/tmp"]
let pid: int = (spawn "ls" args)
```

I return the process ID (pid), or -1 on error.

### `waitpid(pid: int) -> int`
I wait for a child process to complete.

```nano
let exit_code: int = (waitpid pid)
if (== exit_code 0) {
    (println "Process succeeded")
}
```

I return the exit code of the child process.

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
I allow these functions to accept multiple types:
- `println(any)` - I accept int, float, string, and bool
- `print(any)` - I accept int, float, string, and bool
- `abs(int|float)` - I return the same type as the input
- `min(int|float, int|float)` - Both arguments must be the same type
- `max(int|float, int|float)` - Both arguments must be the same type

### Type-Converting Functions
I always return a float from these, regardless of input:
- `sqrt(int|float) -> float`
- `pow(int|float, int|float) -> float`
- `floor(int|float) -> float`
- `ceil(int|float) -> float`
- `round(int|float) -> float`
- `sin(int|float) -> float`
- `cos(int|float) -> float`
- `tan(int|float) -> float`

### String-Only Functions
I only allow these to work with strings:
- `str_length(string) -> int`
- `str_concat(string, string) -> string`
- `str_substring(string, int, int) -> string`
- `str_contains(string, string) -> bool`
- `str_equals(string, string) -> bool`

---

## Performance Notes

I am honest about my performance characteristics:
- My math functions use the C standard library directly. They are fast.
- I bounds-check my string operations to ensure safety.
- My `str_length` is O(n) because I use `strlen`.
- My `str_contains` is O(n*m) because I use `strstr`.
- I allocate memory for new strings on the heap.
- In my transpiled C code, you must manage memory with care.
- I use shadow tests to help you catch memory issues early.

---

## My Plans for Standard Library Additions

I plan to add these in future releases:
- **String:** `str_uppercase`, `str_lowercase`, `str_split`, `str_join`
- **Math:** `log`, `exp`, `atan`, `atan2`, `asin`, `acos`
- **File I/O:** `file_read`, `file_write`, `file_exists`
- **Arrays:** `array_map`, `array_filter`, `array_reduce`, `array_slice`

---

## Notes

- I document all functions with shadow tests in example files.
- I test every stdlib function in at least one example program.
- I include line and column numbers in my error messages.
- I perform type checking at compile time.
- I execute shadow tests during compilation to verify my own correctness.

**Total Functions:** 72 across 9 categories (see spec.json for the authoritative list)
- I/O: 3 functions (print, println, assert)
- Math: 11 functions (abs, min, max, sqrt, pow, floor, ceil, round, sin, cos, tan)
- String: 18 functions (char operations, classification, conversions)
- Binary String: 12 functions (bstring operations for binary data)
- Array: 10 functions (at, length, new, set, push, pop, remove_at, filter, map, reduce)
- OS: 3 functions (getcwd, getenv, range)
- Generics: 4 functions (List<T> operations)
- Checked Math: 5 functions (overflow-safe arithmetic)

**Documentation Status:** I have documented over 160 of my 166 builtin functions (96%+ coverage).

**My Categories:**
- Core I/O: 3 functions
- Input/Output: 4 functions (getchar, print_int, print_float, print_bool)
- Math Operations: 26 functions (basic, advanced, trigonometric, hyperbolic)
- String Operations: 18 functions
- Character Operations: 11 functions
- Type Conversion: 6 functions
- Array Operations: 9 functions
- Array Advanced: 5 functions (push, pop, slice, remove_at, filter)
- HashMap Operations: 10 functions (I only support these in my interpreter)
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

I have 15 functions remaining that need documentation. My HashMap operations are interpreter-only and are not available in my compiled code. Every function has shadow tests in its implementation.


