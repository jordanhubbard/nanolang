# My Standard Library Reference

I provide these built-in functions. They are my core capabilities, available to you without external modules.

---

## Core I/O (3)

### `print(value: any) -> void`
I print a value without a trailing newline.

```nano
(print "Hello")
(print 42)
(print 3.14)
```

### `println(value: any) -> void`
I print a value with a trailing newline. I am polymorphic — I work with int, float, string, and bool.

```nano
(println "Hello, World!")
(println 42)
(println true)
```

### `range(start: int, end: int) -> iterator`
I provide this special function for use only in `for` loops. I create an iterator from `start` (inclusive) to `end` (exclusive).

```nano
for i in (range 0 10) {
    (println i)  # Prints 0, 1, 2, ..., 9
}
for i in (range 5 8) {
    (println i)  # Prints 5, 6, 7
}
```

I only allow `range` to be used in for-loop contexts.

---

## Math (20)

### `abs(x: int) -> int`
I return the absolute value of an integer.

```nano
(abs -5)    # Returns 5
(abs 5)     # Returns 5
(abs 0)     # Returns 0
```

### `min(a: int, b: int) -> int`
I return the minimum of two integers.

```nano
(min 5 10)   # Returns 5
(min -3 0)   # Returns -3
(min 7 7)    # Returns 7
```

### `max(a: int, b: int) -> int`
I return the maximum of two integers.

```nano
(max 5 10)   # Returns 10
(max -3 0)   # Returns 0
(max 7 7)    # Returns 7
```

### `sqrt(x: float) -> float`
I return the square root of x.

```nano
(sqrt 16.0)   # Returns 4.0
(sqrt 2.0)    # Returns 1.41421...
(sqrt 9.0)    # Returns 3.0
```

### `pow(base: float, exponent: float) -> float`
I return base raised to the power of exponent.

```nano
(pow 2.0 3.0)    # Returns 8.0
(pow 5.0 2.0)    # Returns 25.0
(pow 2.0 -1.0)   # Returns 0.5
```

### `floor(x: float) -> float`
I return the largest integer ≤ x as a float.

```nano
(floor 3.7)    # Returns 3.0
(floor 3.2)    # Returns 3.0
(floor -2.3)   # Returns -3.0
```

### `ceil(x: float) -> float`
I return the smallest integer ≥ x as a float.

```nano
(ceil 3.2)    # Returns 4.0
(ceil 3.7)    # Returns 4.0
(ceil -2.7)   # Returns -2.0
```

### `round(x: float) -> float`
I round to the nearest integer as a float.

```nano
(round 3.4)   # Returns 3.0
(round 3.6)   # Returns 4.0
(round 3.5)   # Returns 4.0
```

### `sin(x: float) -> float`
I return the sine of x in radians.

```nano
(sin 0.0)      # Returns 0.0
(sin 1.5708)   # Returns ≈1.0 (π/2)
(sin 3.14159)  # Returns ≈0.0 (π)
```

### `cos(x: float) -> float`
I return the cosine of x in radians.

```nano
(cos 0.0)      # Returns 1.0
(cos 3.14159)  # Returns ≈-1.0 (π)
(cos 1.5708)   # Returns ≈0.0 (π/2)
```

### `tan(x: float) -> float`
I return the tangent of x in radians.

```nano
(tan 0.0)     # Returns 0.0
(tan 0.7854)  # Returns ≈1.0 (π/4)
(tan 1.0)     # Returns ≈1.5574
```

### `atan2(y: float, x: float) -> float`
I return the angle in radians between the positive x-axis and the point (x, y). I handle the quadrant correctly, unlike `atan`.

```nano
(atan2 1.0 1.0)    # Returns ≈0.7854 (π/4, first quadrant)
(atan2 1.0 -1.0)   # Returns ≈2.3562 (3π/4, second quadrant)
(atan2 0.0 1.0)    # Returns 0.0
```

### `asin(x: float) -> float`
I return the arcsine of x (inverse sine) in radians. My domain is -1.0 to 1.0 and I return values in -π/2 to π/2.

```nano
(asin 1.0)    # Returns ≈1.5708 (π/2)
(asin 0.5)    # Returns ≈0.5236 (π/6)
(asin 0.0)    # Returns 0.0
```

### `acos(x: float) -> float`
I return the arccosine of x (inverse cosine) in radians. My domain is -1.0 to 1.0 and I return values in 0 to π.

```nano
(acos 1.0)    # Returns 0.0
(acos 0.0)    # Returns ≈1.5708 (π/2)
(acos -1.0)   # Returns ≈3.14159 (π)
```

### `atan(x: float) -> float`
I return the arctangent of x (inverse tangent) in radians. I return values in -π/2 to π/2.

```nano
(atan 1.0)    # Returns ≈0.7854 (π/4)
(atan 0.0)    # Returns 0.0
(atan -1.0)   # Returns ≈-0.7854 (-π/4)
```

### `log(x: float) -> float`
I return the natural logarithm (base e) of x.

```nano
(log 1.0)    # Returns 0.0
(log 2.718)  # Returns ≈1.0
(log 10.0)   # Returns ≈2.303
```

### `log2(x: float) -> float`
I return the base-2 logarithm of x.

```nano
(log2 1.0)   # Returns 0.0
(log2 2.0)   # Returns 1.0
(log2 8.0)   # Returns 3.0
```

### `log10(x: float) -> float`
I return the base-10 logarithm of x.

```nano
(log10 1.0)    # Returns 0.0
(log10 10.0)   # Returns 1.0
(log10 100.0)  # Returns 2.0
```

### `exp(x: float) -> float`
I return e raised to the power of x.

```nano
(exp 0.0)   # Returns 1.0
(exp 1.0)   # Returns ≈2.71828
(exp 2.0)   # Returns ≈7.389
```

### `fmod(x: float, y: float) -> float`
I return the floating-point remainder of x divided by y.

```nano
(fmod 5.5 2.0)    # Returns 1.5
(fmod 10.0 3.0)   # Returns 1.0
(fmod -5.5 2.0)   # Returns -1.5
```

---

## Type Casting and Conversion (11)

### `cast_int(value: any) -> int`
I cast any value to an integer. I truncate floats and parse strings.

```nano
(cast_int 3.14)   # Returns 3
(cast_int "42")   # Returns 42
(cast_int true)   # Returns 1
```

### `cast_float(value: any) -> float`
I cast any value to a float. I parse strings and convert integers.

```nano
(cast_float 42)      # Returns 42.0
(cast_float "3.14")  # Returns 3.14
(cast_float false)   # Returns 0.0
```

### `cast_bool(value: any) -> bool`
I cast any value to a boolean. I treat 0, empty string, and null as false; everything else becomes true.

```nano
(cast_bool 1)       # Returns true
(cast_bool 0)       # Returns false
(cast_bool "hello") # Returns true
(cast_bool "")      # Returns false
```

### `cast_string(value: any) -> string`
I cast any value to its string representation.

```nano
(cast_string 42)    # Returns "42"
(cast_string 3.14)  # Returns "3.14"
(cast_string true)  # Returns "true"
```

### `to_string(value: any) -> string`
I convert any value to its string representation. I am an alias for `cast_string`.

```nano
(to_string 99)      # Returns "99"
(to_string false)   # Returns "false"
(to_string 1.5)     # Returns "1.5"
```

### `int_to_string(n: int) -> string`
I convert an integer to its string representation.

```nano
(int_to_string 42)    # Returns "42"
(int_to_string 0)     # Returns "0"
(int_to_string -100)  # Returns "-100"
```

### `float_to_string(f: float) -> string`
I convert a float to its string representation.

```nano
(float_to_string 3.14)   # Returns "3.14"
(float_to_string 0.0)    # Returns "0.0"
(float_to_string -1.5)   # Returns "-1.5"
```

### `bool_to_string(b: bool) -> string`
I convert a boolean to its string representation.

```nano
(bool_to_string true)   # Returns "true"
(bool_to_string false)  # Returns "false"
```

### `string_to_int(s: string) -> int`
I parse a string to an integer. I return 0 if the string cannot be parsed.

```nano
(string_to_int "42")     # Returns 42
(string_to_int "-100")   # Returns -100
(string_to_int "abc")    # Returns 0
```

### `string_to_float(s: string) -> float`
I parse a string to a float. I return 0.0 if the string cannot be parsed.

```nano
(string_to_float "3.14")  # Returns 3.14
(string_to_float "1e-3")  # Returns 0.001
(string_to_float "bad")   # Returns 0.0
```

### `null_opaque() -> opaque`
I return a null opaque handle. I am useful as a sentinel value when working with foreign function interfaces.

```nano
let handle: opaque = (null_opaque)
```

---

## String Operations (7)

### `str_length(s: string) -> int`
I return the length of a string in bytes.

```nano
(str_length "Hello")   # Returns 5
(str_length "")        # Returns 0
(str_length "abc")     # Returns 3
```

### `str_concat(s1: string, s2: string) -> string`
I concatenate two strings and return a new string.

```nano
(str_concat "Hello" " World")   # Returns "Hello World"
(str_concat "foo" "bar")        # Returns "foobar"
(str_concat "" "test")          # Returns "test"
```

### `str_substring(s: string, start: int, length: int) -> string`
I extract a substring starting at `start` with the given `length`. The index is 0-based. I return until the end of the string if `start + length` exceeds the string length. I return an empty string if `start` is out of bounds.

```nano
(str_substring "Hello, World!" 0 5)   # Returns "Hello"
(str_substring "Hello, World!" 7 5)   # Returns "World"
(str_substring "Hello" 2 100)         # Returns "llo"
```

### `str_contains(s: string, substr: string) -> bool`
I return true if string `s` contains substring `substr`.

```nano
(str_contains "The quick brown fox" "quick")   # Returns true
(str_contains "The quick brown fox" "slow")    # Returns false
(str_contains "hello" "")                      # Returns true
```

### `str_equals(s1: string, s2: string) -> bool`
I return true if both strings are exactly equal.

```nano
(str_equals "Hello" "Hello")   # Returns true
(str_equals "Hello" "World")   # Returns false
(str_equals "" "")             # Returns true
```

### `char_at(s: string, index: int) -> int`
I return the ASCII value of the character at the specified 0-based index. I terminate with an error if the index is out of bounds.

```nano
(char_at "Hello" 0)   # Returns 72 ('H')
(char_at "Hello" 1)   # Returns 101 ('e')
(char_at "Hello" 4)   # Returns 111 ('o')
```

### `string_from_char(c: int) -> string`
I create a single-character string from an ASCII value.

```nano
(string_from_char 65)   # Returns "A"
(string_from_char 90)   # Returns "Z"
(string_from_char 48)   # Returns "0"
```

---

## Character Classification (10)

### `is_digit(c: int) -> bool`
I return true if the character code represents a decimal digit ('0'–'9').

```nano
(is_digit 48)   # Returns true  ('0')
(is_digit 53)   # Returns true  ('5')
(is_digit 65)   # Returns false ('A')
```

### `is_alpha(c: int) -> bool`
I return true if the character code represents a letter (a–z, A–Z).

```nano
(is_alpha 65)    # Returns true  ('A')
(is_alpha 97)    # Returns true  ('a')
(is_alpha 48)    # Returns false ('0')
```

### `is_alnum(c: int) -> bool`
I return true if the character code represents an alphanumeric character (digit or letter).

```nano
(is_alnum 48)   # Returns true  ('0')
(is_alnum 65)   # Returns true  ('A')
(is_alnum 32)   # Returns false (' ')
```

### `is_space(c: int) -> bool`
I return true if the character code is a space character (ASCII 32).

```nano
(is_space 32)   # Returns true  (' ')
(is_space 65)   # Returns false ('A')
(is_space 9)    # Returns false ('\t')
```

### `is_whitespace(c: int) -> bool`
I return true if the character code represents any whitespace: space, tab, newline, or carriage return.

```nano
(is_whitespace 32)   # Returns true  (' ')
(is_whitespace 9)    # Returns true  ('\t')
(is_whitespace 10)   # Returns true  ('\n')
(is_whitespace 65)   # Returns false ('A')
```

### `is_upper(c: int) -> bool`
I return true if the character code represents an uppercase letter (A–Z).

```nano
(is_upper 65)   # Returns true  ('A')
(is_upper 90)   # Returns true  ('Z')
(is_upper 97)   # Returns false ('a')
```

### `is_lower(c: int) -> bool`
I return true if the character code represents a lowercase letter (a–z).

```nano
(is_lower 97)    # Returns true  ('a')
(is_lower 122)   # Returns true  ('z')
(is_lower 65)    # Returns false ('A')
```

### `digit_value(c: int) -> int`
I convert a digit character code to its numeric value. I return -1 if it is not a digit.

```nano
(digit_value 48)   # Returns 0  ('0' -> 0)
(digit_value 53)   # Returns 5  ('5' -> 5)
(digit_value 57)   # Returns 9  ('9' -> 9)
(digit_value 65)   # Returns -1 ('A' is not a digit)
```

### `char_to_lower(c: int) -> int`
I convert an uppercase letter code to lowercase. I leave non-letters unchanged.

```nano
(char_to_lower 65)   # Returns 97  ('A' -> 'a')
(char_to_lower 90)   # Returns 122 ('Z' -> 'z')
(char_to_lower 48)   # Returns 48  ('0' -> '0', unchanged)
```

### `char_to_upper(c: int) -> int`
I convert a lowercase letter code to uppercase. I leave non-letters unchanged.

```nano
(char_to_upper 97)    # Returns 65  ('a' -> 'A')
(char_to_upper 122)   # Returns 90  ('z' -> 'Z')
(char_to_upper 48)    # Returns 48  ('0' -> '0', unchanged)
```

---

## Array Operations (13)

### `at(arr: array<T>, index: int) -> T`
I return the element at the specified 0-based index. I perform bounds-checking and terminate with an error if the index is out of bounds.

```nano
let nums: array<int> = [1, 2, 3, 4, 5]
(at nums 0)   # Returns 1
(at nums 4)   # Returns 5
```

### `array_get(arr: array<T>, index: int) -> T`
I return the element at the specified 0-based index. I am an alias for `at`.

```nano
let nums: array<int> = [10, 20, 30]
(array_get nums 0)   # Returns 10
(array_get nums 2)   # Returns 30
```

### `array_length(arr: array<T>) -> int`
I return the number of elements in an array.

```nano
let nums: array<int> = [10, 20, 30]
(array_length nums)    # Returns 3
(array_length [])      # Returns 0
```

### `array_new(size: int, default: T) -> array<T>`
I create a new array of the specified size, filled with the default value.

```nano
let zeros: array<int> = (array_new 5 0)
# [0, 0, 0, 0, 0]
let strs: array<string> = (array_new 3 "")
# ["", "", ""]
```

### `array_set(arr: array<T>, index: int, value: T) -> void`
I set the element at the specified 0-based index. I perform bounds-checking and terminate with an error if the index is out of bounds. I require a mutable array.

```nano
let mut nums: array<int> = [1, 2, 3]
(array_set nums 1 42)
# nums is now [1, 42, 3]
```

### `array_push(arr: array<T>, value: T) -> array<T>`
I append an element to the end of an array and return the updated array.

```nano
let mut numbers: array<int> = [1, 2, 3]
(array_push numbers 4)
# numbers is now [1, 2, 3, 4]
```

### `array_pop(arr: array<T>) -> T`
I remove and return the last element of the array.

```nano
let mut stack: array<int> = [1, 2, 3]
let last: int = (array_pop stack)   # Returns 3
# stack is now [1, 2]
```

### `array_remove_at(arr: array<T>, index: int) -> array<T>`
I remove the element at the specified index, shifting remaining elements left, and return the updated array.

```nano
let mut items: array<int> = [10, 20, 30, 40]
(array_remove_at items 1)
# items is now [10, 30, 40]
```

### `array_slice(arr: array<T>, start: int, length: int) -> array<T>`
I create a new array from a portion of the original, starting at `start` with the given `length`.

```nano
let numbers: array<int> = [1, 2, 3, 4, 5]
let subset: array<int> = (array_slice numbers 1 3)
# subset is [2, 3, 4]
```

### `array_concat(arr1: array<T>, arr2: array<T>) -> array<T>`
I concatenate two arrays and return a new array containing all elements.

```nano
let a: array<int> = [1, 2, 3]
let b: array<int> = [4, 5, 6]
let c: array<int> = (array_concat a b)
# c is [1, 2, 3, 4, 5, 6]
```

### `array_map(arr: array<T>, f: fn(T) -> U) -> array<U>`
I apply a function to each element of an array and return a new array of the results.

```nano
fn square(x: int) -> int { return (* x x) }

let nums: array<int> = [1, 2, 3, 4]
let squares: array<int> = (array_map nums square)
# squares is [1, 4, 9, 16]
```

### `array_filter(arr: array<T>, pred: fn(T) -> bool) -> array<T>`
I return a new array containing only the elements for which the predicate returns true.

```nano
fn is_even(n: int) -> bool { return (== (% n 2) 0) }

let nums: array<int> = [1, 2, 3, 4, 5, 6]
let evens: array<int> = (array_filter nums is_even)
# evens is [2, 4, 6]
```

### `array_fold(arr: array<T>, init: U, f: fn(U, T) -> U) -> U`
I reduce an array to a single value by applying a function cumulatively, starting with `init`.

```nano
fn add(acc: int, x: int) -> int { return (+ acc x) }

let nums: array<int> = [1, 2, 3, 4]
let sum: int = (array_fold nums 0 add)
# sum is 10
```

---

## Higher-Order Functions (3)

### `filter(arr: array<T>, predicate: fn(T) -> bool) -> array<T>`
I create a new array with elements that satisfy the predicate. I am equivalent to `array_filter`.

```nano
fn is_even(n: int) -> bool { return (== (% n 2) 0) }

let numbers: array<int> = [1, 2, 3, 4, 5, 6]
let evens: array<int> = (filter numbers is_even)
# evens is [2, 4, 6]
```

### `map(arr: array<T>, f: fn(T) -> U) -> array<U>`
I transform each element using the provided function. I am equivalent to `array_map`.

```nano
fn square(x: int) -> int { return (* x x) }

let numbers: array<int> = [1, 2, 3, 4]
let squares: array<int> = (map numbers square)
# squares is [1, 4, 9, 16]
```

### `reduce(arr: array<T>, init: U, f: fn(U, T) -> U) -> U`
I reduce an array to a single value. I am equivalent to `array_fold`.

```nano
fn add(acc: int, x: int) -> int { return (+ acc x) }

let numbers: array<int> = [1, 2, 3, 4]
let sum: int = (reduce numbers 0 add)
# sum is 10
```

---

## HashMap Operations (16)

I provide HashMap as a key-value collection with O(1) average lookup.

### `hashmap_new() -> hashmap`
I create a new empty hashmap.

```nano
let hm: hashmap = (hashmap_new)
```

### `hashmap_get(hm: hashmap, key: string) -> any`
I return the value associated with `key`, or null if the key does not exist.

```nano
(hashmap_set hm "name" "Alice")
let val: string = (hashmap_get hm "name")   # Returns "Alice"
```

### `hashmap_set(hm: hashmap, key: string, value: any) -> void`
I insert or update the value for `key`.

```nano
let hm: hashmap = (hashmap_new)
(hashmap_set hm "score" 100)
(hashmap_set hm "name" "Bob")
```

### `hashmap_has(hm: hashmap, key: string) -> bool`
I return true if `key` exists in the hashmap.

```nano
(hashmap_set hm "x" 42)
(hashmap_has hm "x")       # Returns true
(hashmap_has hm "missing")  # Returns false
```

### `hashmap_delete(hm: hashmap, key: string) -> void`
I remove the key-value pair for `key`. I do nothing if the key does not exist.

```nano
(hashmap_set hm "temp" 99)
(hashmap_delete hm "temp")
(hashmap_has hm "temp")   # Returns false
```

### `hashmap_keys(hm: hashmap) -> array<string>`
I return all keys in the hashmap as an array of strings.

```nano
(hashmap_set hm "a" 1)
(hashmap_set hm "b" 2)
let keys: array<string> = (hashmap_keys hm)
# keys contains ["a", "b"] (order may vary)
```

### `hashmap_values(hm: hashmap) -> array<any>`
I return all values in the hashmap as an array.

```nano
(hashmap_set hm "x" 10)
(hashmap_set hm "y" 20)
let vals: array<int> = (hashmap_values hm)
# vals contains [10, 20] (order may vary)
```

### `hashmap_length(hm: hashmap) -> int`
I return the number of key-value pairs in the hashmap.

```nano
let hm: hashmap = (hashmap_new)
(hashmap_set hm "a" 1)
(hashmap_set hm "b" 2)
(hashmap_length hm)   # Returns 2
```

### `map_new() -> hashmap`
I create a new empty hashmap. I am an alias for `hashmap_new`.

```nano
let m: hashmap = (map_new)
```

### `map_get(hm: hashmap, key: string) -> any`
I return the value for `key`. I am an alias for `hashmap_get`.

```nano
let score: int = (map_get hm "alice")
```

### `map_set(hm: hashmap, key: string, value: any) -> void`
I insert or update a key-value pair. I am an alias for `hashmap_set`.

```nano
(map_set hm "alice" 10)
(map_set hm "bob" 20)
```

### `map_has(hm: hashmap, key: string) -> bool`
I check if a key exists. I am an alias for `hashmap_has`.

```nano
if (map_has hm "alice") { (println "found") }
```

### `map_delete(hm: hashmap, key: string) -> void`
I remove a key-value pair. I am an alias for `hashmap_delete`.

```nano
(map_delete hm "temp")
```

### `map_keys(hm: hashmap) -> array<string>`
I return all keys as an array. I am an alias for `hashmap_keys`.

```nano
let keys: array<string> = (map_keys hm)
```

### `map_values(hm: hashmap) -> array<any>`
I return all values as an array. I am an alias for `hashmap_values`.

```nano
let vals: array<int> = (map_values hm)
```

### `map_length(hm: hashmap) -> int`
I return the number of entries. I am an alias for `hashmap_length`.

```nano
let count: int = (map_length hm)
```

---

## Result Type Operations (7)

My `Result<T, E>` type represents either success (`Ok`) or failure (`Err`).

### `result_is_ok(r: Result<T, E>) -> bool`
I return true if the result is an `Ok` value.

```nano
let r: Result<int, string> = (divide 10 2)
if (result_is_ok r) { (println "Success!") }
```

### `result_is_err(r: Result<T, E>) -> bool`
I return true if the result is an `Err` value.

```nano
let r: Result<int, string> = (divide 10 0)
if (result_is_err r) { (println "Error occurred") }
```

### `result_unwrap(r: Result<T, E>) -> T`
I extract the `Ok` value. I panic if the result is `Err` — use `result_is_ok` to check first.

```nano
let r: Result<int, string> = (divide 10 2)
let value: int = (result_unwrap r)   # Returns 5
```

### `result_unwrap_err(r: Result<T, E>) -> E`
I extract the `Err` value. I panic if the result is `Ok`.

```nano
let r: Result<int, string> = (divide 10 0)
if (result_is_err r) {
    let msg: string = (result_unwrap_err r)
    (println msg)
}
```

### `result_unwrap_or(r: Result<T, E>, default: T) -> T`
I extract the `Ok` value, or return `default` if the result is `Err`.

```nano
let r: Result<int, string> = (divide 10 0)
let value: int = (result_unwrap_or r 0)   # Returns 0 (the default)
```

### `result_map(r: Result<T, E>, f: fn(T) -> U) -> Result<U, E>`
I apply a function to the `Ok` value and return a new result. I pass `Err` values through unchanged.

```nano
fn double(x: int) -> int { return (* x 2) }

let r: Result<int, string> = (divide 10 2)
let r2: Result<int, string> = (result_map r double)
# r2 is Ok(10)
```

### `result_and_then(r: Result<T, E>, f: fn(T) -> Result<U, E>) -> Result<U, E>`
I apply a function that itself returns a Result, and flatten the result. I pass `Err` values through unchanged. I use this to chain fallible operations.

```nano
fn safe_sqrt(x: int) -> Result<float, string> {
    if (< x 0) { return (Err "negative input") }
    return (Ok (sqrt (cast_float x)))
}

let r: Result<int, string> = (divide 16 1)
let r2: Result<float, string> = (result_and_then r safe_sqrt)
# r2 is Ok(4.0)
```

---

## File I/O (8)

### `file_read(path: string) -> string`
I read the entire contents of a file and return them as a string. I return an empty string on error.

```nano
let content: string = (file_read "data.txt")
(println content)
```

### `file_read_bytes(path: string) -> array<int>`
I read file contents as an array of byte values (0–255). I recommend this for binary files.

```nano
let data: array<int> = (file_read_bytes "image.png")
let size: int = (array_length data)
```

### `file_write(path: string, content: string) -> int`
I write string content to a file, overwriting it if it already exists. I return 0 on success and 1 on failure.

```nano
let status: int = (file_write "output.txt" "Hello, World!")
if (== status 0) { (println "Write successful") }
```

### `file_append(path: string, content: string) -> int`
I append string content to the end of a file, creating it if it does not exist. I return 0 on success and 1 on failure.

```nano
let status: int = (file_append "log.txt" "New log entry\n")
```

### `file_remove(path: string) -> int`
I delete a file permanently. I return 0 on success and 1 on failure.

```nano
let status: int = (file_remove "temp.txt")
```

### `file_rename(old_path: string, new_path: string) -> int`
I rename or move a file. I return 0 on success and 1 on failure.

```nano
let status: int = (file_rename "old.txt" "new.txt")
```

### `file_exists(path: string) -> bool`
I return true if the file exists and is accessible.

```nano
if (file_exists "config.json") {
    let config: string = (file_read "config.json")
} else {
    (println "Config file not found")
}
```

### `file_size(path: string) -> int`
I return the file size in bytes, or -1 on error.

```nano
let size: int = (file_size "data.bin")
(println (str_concat "File size: " (int_to_string size)))
```

---

## Directory and Navigation (10)

### `dir_exists(path: string) -> bool`
I return true if the path exists and is a directory.

```nano
if (not (dir_exists "output")) {
    (dir_create "output")
}
```

### `dir_create(path: string) -> int`
I create a directory. Parent directories must already exist. I return 0 on success and 1 on failure.

```nano
let status: int = (dir_create "build/output")
```

### `dir_remove(path: string) -> int`
I remove an empty directory. I return 0 on success and 1 on failure.

```nano
let status: int = (dir_remove "temp")
```

### `dir_list(path: string) -> array<string>`
I list all entries in a directory and return an array of filenames (not full paths). I return an empty array on error.

```nano
let entries: array<string> = (dir_list ".")
for entry in entries {
    (println entry)
}
```

### `getcwd() -> string`
I return the current working directory as an absolute path.

```nano
let cwd: string = (getcwd)
(println cwd)   # Prints e.g. "/home/user/project"
```

### `chdir(path: string) -> int`
I change the current working directory. I return 0 on success and 1 on failure.

```nano
let status: int = (chdir "/tmp")
let cwd: string = (getcwd)
(println cwd)   # Prints "/tmp"
```

### `fs_walkdir(path: string) -> array<string>`
I recursively walk a directory tree and return an array of all file paths found.

```nano
let files: array<string> = (fs_walkdir ".")
for f in files {
    (println f)
}
```

### `tmp_dir() -> string`
I return the system's temporary directory path.

```nano
let tmp: string = (tmp_dir)
(println tmp)   # Prints e.g. "/tmp"
```

### `mktemp(prefix: string) -> string`
I create a new temporary file with the given prefix and return its path. The file is created and left open for writing.

```nano
let path: string = (mktemp "nano_work_")
(file_write path "some data")
```

### `mktemp_dir(prefix: string) -> string`
I create a new temporary directory with the given prefix and return its path.

```nano
let dir: string = (mktemp_dir "nano_build_")
let out: string = (path_join dir "output.txt")
```

---

## Path Operations (6)

### `path_isfile(path: string) -> bool`
I return true if the path exists and is a regular file.

```nano
if (path_isfile "config.json") {
    (println "Found config file")
}
```

### `path_isdir(path: string) -> bool`
I return true if the path exists and is a directory.

```nano
if (path_isdir "src") {
    (println "src directory exists")
}
```

### `path_join(a: string, b: string) -> string`
I join two path components with the appropriate separator. I use `/` on Unix.

```nano
let full: string = (path_join "/home/user" "documents")
# Returns "/home/user/documents"
let nested: string = (path_join "src" "main.nano")
# Returns "src/main.nano"
```

### `path_basename(path: string) -> string`
I extract the filename component from a path.

```nano
(path_basename "/path/to/file.txt")   # Returns "file.txt"
(path_basename "src/main.nano")       # Returns "main.nano"
```

### `path_dirname(path: string) -> string`
I extract the directory component from a path.

```nano
(path_dirname "/path/to/file.txt")   # Returns "/path/to"
(path_dirname "src/main.nano")       # Returns "src"
```

### `path_normalize(path: string) -> string`
I normalize a path by resolving `.`, `..`, and redundant separators.

```nano
(path_normalize "./foo/../bar/./baz")   # Returns "bar/baz"
(path_normalize "/a//b/./c")           # Returns "/a/b/c"
```

---

## Process and Environment (5)

### `system(command: string) -> int`
I execute a shell command and wait for it to complete. I return the exit code.

```nano
let status: int = (system "ls -la")
if (!= status 0) { (println "Command failed") }
```

### `exit(code: int) -> void`
I terminate the program immediately with the given exit code.

```nano
if (not (file_exists "required.txt")) {
    (println "Error: required.txt not found")
    (exit 1)
}
```

### `getenv(name: string) -> string`
I return the value of an environment variable. I return an empty string if it is not set.

```nano
let home: string = (getenv "HOME")
let path: string = (getenv "PATH")
```

### `setenv(name: string, value: string) -> int`
I set an environment variable for the current process and its children. I return 0 on success and 1 on failure.

```nano
let status: int = (setenv "MY_VAR" "my_value")
```

### `process_run(command: string) -> array<string>`
I execute a command and return its output as an array of strings. The first element is the exit code as a string, and subsequent elements are lines of stdout.

```nano
let result: array<string> = (process_run "echo hello")
let code: string = (at result 0)     # "0" (exit code)
let line: string = (at result 1)     # "hello"
```

---

## Binary String and UTF-8 (5)

### `bytes_from_string(s: string) -> array<int>`
I convert a string to an array of byte values (0–255), one per character.

```nano
let bytes: array<int> = (bytes_from_string "Hello")
(at bytes 0)   # Returns 72 ('H')
```

### `string_from_bytes(bytes: array<int>) -> string`
I construct a string from an array of byte values.

```nano
let bytes: array<int> = [72, 101, 108, 108, 111]
let s: string = (string_from_bytes bytes)   # Returns "Hello"
```

### `bstr_utf8_length(s: string) -> int`
I return the number of Unicode code points in a UTF-8 encoded string, which may differ from its byte length for multi-byte characters.

```nano
(bstr_utf8_length "Hello")   # Returns 5
(bstr_utf8_length "café")    # Returns 4 (4 code points, 5 bytes)
```

### `bstr_utf8_char_at(s: string, index: int) -> int`
I return the Unicode code point at the given character index (not byte index) in a UTF-8 string.

```nano
(bstr_utf8_char_at "Hello" 0)   # Returns 72 ('H')
(bstr_utf8_char_at "café" 3)    # Returns the code point for 'é'
```

### `bstr_validate_utf8(s: string) -> bool`
I return true if the string contains valid UTF-8 encoded text.

```nano
if (bstr_validate_utf8 content) {
    (println "Valid UTF-8")
} else {
    (println "Invalid encoding")
}
```
