# NanoLang Feature Inventory

**Last Updated:** Auto-generated from spec.json and user guide audit

This document tracks all NanoLang language features and their documentation status in the user guide.

---

## Types

### Primitive Types

| Feature | Documented | Location |
|---------|------------|----------|
| `int` (64-bit signed) | ✅ Yes | userguide/03_basic_types.md, appendices/b_quick_reference.md |
| `float` (64-bit IEEE 754) | ✅ Yes | userguide/03_basic_types.md, appendices/b_quick_reference.md |
| `bool` (true/false) | ✅ Yes | userguide/03_basic_types.md, appendices/b_quick_reference.md |
| `string` (UTF-8, null-terminated) | ✅ Yes | userguide/03_basic_types.md, appendices/b_quick_reference.md |
| `bstring` (binary string, UTF-8 aware) | ✅ Yes | appendices/b_quick_reference.md |
| `void` (no return value) | ✅ Yes | appendices/b_quick_reference.md |

### Composite Types

| Feature | Documented | Location |
|---------|------------|----------|
| `array<T>` (dynamic arrays) | ✅ Yes | userguide/03_basic_types.md, part1_fundamentals/06_collections.md |
| `struct` (product types) | ✅ Yes | userguide/03_basic_types.md, part1_fundamentals/07_data_structures.md |
| `enum` (named constants) | ✅ Yes | userguide/03_basic_types.md, part1_fundamentals/07_data_structures.md |
| `union` (tagged unions) | ✅ Yes | part1_fundamentals/07_data_structures.md |
| `tuple` (fixed-size heterogeneous) | ✅ Yes | userguide/03_basic_types.md |
| `List<T>` (generic list) | ✅ Yes | userguide/03_basic_types.md, appendices/b_quick_reference.md |
| `HashMap<K,V>` (hash map) | ✅ Yes | userguide/03_basic_types.md, appendices/b_quick_reference.md |
| `fn(...) -> T` (function types) | ✅ Yes | userguide/04_higher_level_patterns.md |
| `opaque type` (FFI types) | ✅ Yes | part1_fundamentals/07_data_structures.md |

---

## Syntax

### Notation

| Feature | Documented | Location |
|---------|------------|----------|
| Prefix notation `(f x y)` for function calls | ✅ Yes | userguide/06_canonical_syntax.md, docs/CANONICAL_STYLE.md |
| Infix notation `a + b` for binary operators | ✅ Yes | userguide/06_canonical_syntax.md |
| S-expression function calls | ✅ Yes | userguide/06_canonical_syntax.md |

---

## Operations

### Arithmetic

| Operator | Documented | Location |
|----------|------------|----------|
| `+` (addition) | ✅ Yes | appendices/b_quick_reference.md |
| `-` (subtraction) | ✅ Yes | appendices/b_quick_reference.md |
| `*` (multiplication) | ✅ Yes | appendices/b_quick_reference.md |
| `/` (division) | ✅ Yes | appendices/b_quick_reference.md |
| `%` (modulo) | ✅ Yes | appendices/b_quick_reference.md |

### Comparison

| Operator | Documented | Location |
|----------|------------|----------|
| `==` (equal) | ✅ Yes | appendices/b_quick_reference.md |
| `!=` (not equal) | ✅ Yes | appendices/b_quick_reference.md |
| `<` (less than) | ✅ Yes | appendices/b_quick_reference.md |
| `<=` (less than or equal) | ✅ Yes | appendices/b_quick_reference.md |
| `>` (greater than) | ✅ Yes | appendices/b_quick_reference.md |
| `>=` (greater than or equal) | ✅ Yes | appendices/b_quick_reference.md |

### Logical

| Operator | Documented | Location |
|----------|------------|----------|
| `and` (logical AND, short-circuit) | ✅ Yes | appendices/b_quick_reference.md |
| `or` (logical OR, short-circuit) | ✅ Yes | appendices/b_quick_reference.md |
| `not` (logical NOT) | ✅ Yes | appendices/b_quick_reference.md |

---

## Statements

| Statement | Documented | Location |
|-----------|------------|----------|
| `let` (immutable declaration) | ✅ Yes | userguide/06_canonical_syntax.md, part1_fundamentals/03_variables.md |
| `let mut` (mutable declaration) | ✅ Yes | userguide/06_canonical_syntax.md, part1_fundamentals/03_variables.md |
| `set` (assignment) | ✅ Yes | userguide/06_canonical_syntax.md |
| `if/else` (conditional statement) | ✅ Yes | userguide/02_control_flow.md, part1_fundamentals/05_control_flow.md |
| `while` (loop) | ✅ Yes | userguide/02_control_flow.md, part1_fundamentals/05_control_flow.md |
| `for` (range loop) | ✅ Yes | userguide/02_control_flow.md, part1_fundamentals/05_control_flow.md |
| `return` (return statement) | ✅ Yes | part1_fundamentals/04_functions.md |
| `match` (pattern matching) | ✅ Yes | part1_fundamentals/07_data_structures.md |
| `unsafe` (FFI block) | ✅ Yes | userguide/04_higher_level_patterns.md, appendices/b_quick_reference.md |

---

## Expressions

| Expression | Documented | Location |
|------------|------------|----------|
| `cond` (multi-branch conditional) | ✅ Yes | userguide/02_control_flow.md, part1_fundamentals/05_control_flow.md |

---

## Standard Library Functions

### I/O (3 functions)

| Function | Documented | Location |
|----------|------------|----------|
| `print(any) -> void` | ✅ Yes | appendices/b_quick_reference.md |
| `println(any) -> void` | ✅ Yes | appendices/b_quick_reference.md |
| `assert(bool) -> void` | ✅ Yes | appendices/b_quick_reference.md |

### Math (11 functions)

| Function | Documented | Location |
|----------|------------|----------|
| `abs(int\|float)` | ✅ Yes | appendices/b_quick_reference.md |
| `min(a, b)` | ✅ Yes | appendices/b_quick_reference.md |
| `max(a, b)` | ✅ Yes | appendices/b_quick_reference.md |
| `sqrt(float)` | ✅ Yes | appendices/b_quick_reference.md |
| `pow(base, exp)` | ✅ Yes | appendices/b_quick_reference.md |
| `floor(float)` | ✅ Yes | appendices/b_quick_reference.md |
| `ceil(float)` | ✅ Yes | appendices/b_quick_reference.md |
| `round(float)` | ✅ Yes | appendices/b_quick_reference.md |
| `sin(float)` | ✅ Yes | appendices/b_quick_reference.md |
| `cos(float)` | ✅ Yes | appendices/b_quick_reference.md |
| `tan(float)` | ✅ Yes | appendices/b_quick_reference.md |

### String (18 functions)

| Function | Documented | Location |
|----------|------------|----------|
| `str_length(string) -> int` | ✅ Yes | appendices/b_quick_reference.md |
| `str_concat(string, string) -> string` | ✅ Yes | appendices/b_quick_reference.md (deprecated) |
| `str_substring(string, int, int) -> string` | ✅ Yes | appendices/b_quick_reference.md |
| `str_contains(string, string) -> bool` | ✅ Yes | appendices/b_quick_reference.md |
| `str_equals(string, string) -> bool` | ✅ Yes | appendices/b_quick_reference.md |
| `char_at(string, int) -> int` | ✅ Yes | appendices/b_quick_reference.md |
| `string_from_char(int) -> string` | ✅ Yes | appendices/b_quick_reference.md |
| `is_digit(int) -> bool` | ✅ Yes | appendices/b_quick_reference.md |
| `is_alpha(int) -> bool` | ✅ Yes | appendices/b_quick_reference.md |
| `is_alnum(int) -> bool` | ✅ Yes | appendices/b_quick_reference.md |
| `is_whitespace(int) -> bool` | ✅ Yes | appendices/b_quick_reference.md |
| `is_upper(int) -> bool` | ✅ Yes | appendices/b_quick_reference.md |
| `is_lower(int) -> bool` | ✅ Yes | appendices/b_quick_reference.md |
| `int_to_string(int) -> string` | ✅ Yes | appendices/b_quick_reference.md |
| `string_to_int(string) -> int` | ✅ Yes | appendices/b_quick_reference.md |
| `digit_value(int) -> int` | ✅ Yes | appendices/b_quick_reference.md |
| `char_to_lower(int) -> int` | ✅ Yes | appendices/b_quick_reference.md |
| `char_to_upper(int) -> int` | ✅ Yes | appendices/b_quick_reference.md |

### Binary String (12 functions)

| Function | Documented | Location |
|----------|------------|----------|
| `bstr_new(string) -> bstring` | ✅ Yes | appendices/b_quick_reference.md |
| `bstr_new_binary(string, int) -> bstring` | ✅ Yes | appendices/b_quick_reference.md |
| `bstr_length(bstring) -> int` | ✅ Yes | appendices/b_quick_reference.md |
| `bstr_concat(bstring, bstring) -> bstring` | ✅ Yes | appendices/b_quick_reference.md |
| `bstr_substring(bstring, int, int) -> bstring` | ✅ Yes | appendices/b_quick_reference.md |
| `bstr_equals(bstring, bstring) -> bool` | ✅ Yes | appendices/b_quick_reference.md |
| `bstr_byte_at(bstring, int) -> int` | ✅ Yes | appendices/b_quick_reference.md |
| `bstr_to_cstr(bstring) -> string` | ✅ Yes | appendices/b_quick_reference.md |
| `bstr_validate_utf8(bstring) -> bool` | ✅ Yes | appendices/b_quick_reference.md |
| `bstr_utf8_length(bstring) -> int` | ✅ Yes | appendices/b_quick_reference.md |
| `bstr_utf8_char_at(bstring, int) -> int` | ✅ Yes | appendices/b_quick_reference.md |
| `bstr_free(bstring) -> void` | ✅ Yes | appendices/b_quick_reference.md |

### Array (10 functions)

| Function | Documented | Location |
|----------|------------|----------|
| `at(array<T>, int) -> T` | ✅ Yes | appendices/b_quick_reference.md |
| `array_length(array<T>) -> int` | ✅ Yes | appendices/b_quick_reference.md |
| `array_new(int, T) -> array<T>` | ✅ Yes | appendices/b_quick_reference.md |
| `array_set(array<T>, int, T) -> array<T>` | ✅ Yes | appendices/b_quick_reference.md |
| `array_push(array<T>, T) -> array<T>` | ✅ Yes | appendices/b_quick_reference.md |
| `array_pop(array<T>) -> T` | ✅ Yes | appendices/b_quick_reference.md |
| `array_remove_at(array<T>, int) -> array<T>` | ✅ Yes | appendices/b_quick_reference.md |
| `filter(array<T>, fn(T) -> bool) -> array<T>` | ✅ Yes | appendices/b_quick_reference.md |
| `map(array<T>, fn(T) -> T) -> array<T>` | ✅ Yes | appendices/b_quick_reference.md |
| `reduce(array<T>, A, fn(A, T) -> A) -> A` | ✅ Yes | appendices/b_quick_reference.md |

### OS (3 functions)

| Function | Documented | Location |
|----------|------------|----------|
| `getcwd() -> string` | ✅ Yes | appendices/b_quick_reference.md |
| `getenv(string) -> string` | ✅ Yes | appendices/b_quick_reference.md |
| `range(int, int) -> iterator` | ✅ Yes | appendices/b_quick_reference.md |

### Generic List (4 functions)

| Function | Documented | Location |
|----------|------------|----------|
| `list_T_new() -> List<T>` | ✅ Yes | appendices/b_quick_reference.md |
| `list_T_push(List<T>, T) -> void` | ✅ Yes | appendices/b_quick_reference.md |
| `list_T_get(List<T>, int) -> T` | ✅ Yes | appendices/b_quick_reference.md |
| `list_T_length(List<T>) -> int` | ✅ Yes | appendices/b_quick_reference.md |

### Checked Math (5 functions)

| Function | Documented | Location |
|----------|------------|----------|
| `checked_add(int, int) -> Result<int, string>` | ✅ Yes | appendices/b_quick_reference.md |
| `checked_sub(int, int) -> Result<int, string>` | ✅ Yes | appendices/b_quick_reference.md |
| `checked_mul(int, int) -> Result<int, string>` | ✅ Yes | appendices/b_quick_reference.md |
| `checked_div(int, int) -> Result<int, string>` | ✅ Yes | appendices/b_quick_reference.md |
| `checked_mod(int, int) -> Result<int, string>` | ✅ Yes | appendices/b_quick_reference.md |

### HashMap (6 functions)

| Function | Documented | Location |
|----------|------------|----------|
| `map_new() -> HashMap<K,V>` | ✅ Yes | appendices/b_quick_reference.md |
| `map_put(HashMap<K,V>, K, V) -> void` | ✅ Yes | appendices/b_quick_reference.md |
| `map_get(HashMap<K,V>, K) -> V` | ✅ Yes | appendices/b_quick_reference.md |
| `map_has(HashMap<K,V>, K) -> bool` | ✅ Yes | appendices/b_quick_reference.md |
| `map_size(HashMap<K,V>) -> int` | ✅ Yes | appendices/b_quick_reference.md |
| `map_free(HashMap<K,V>) -> void` | ✅ Yes | appendices/b_quick_reference.md |

---

## Language Features

| Feature | Documented | Location |
|---------|------------|----------|
| Module imports | ✅ Yes | userguide/05_modules.md, part1_fundamentals/08_modules.md |
| Import aliases | ✅ Yes | part1_fundamentals/08_modules.md |
| `extern fn` (FFI declarations) | ✅ Yes | userguide/04_higher_level_patterns.md |
| `pub` exports | ✅ Yes | part1_fundamentals/08_modules.md |
| Shadow tests (mandatory) | ✅ Yes | userguide/06_canonical_syntax.md, all examples |
| First-class functions | ✅ Yes | userguide/04_higher_level_patterns.md |
| Generics (monomorphization) | ✅ Yes | userguide/03_basic_types.md |
| Garbage collection | ✅ Yes | (mentioned in spec.json, brief in userguide) |
| `unsafe` blocks | ✅ Yes | userguide/04_higher_level_patterns.md, appendices/b_quick_reference.md |

---

## Summary Statistics

| Category | Total | Documented | Coverage |
|----------|-------|------------|----------|
| Primitive Types | 6 | 6 | 100% |
| Composite Types | 9 | 9 | 100% |
| Operations | 14 | 14 | 100% |
| Statements | 9 | 9 | 100% |
| Stdlib Functions | 72 | 72 | 100% |
| Language Features | 9 | 9 | 100% |
| **Total** | **119** | **119** | **100%** |

---

## Gap Analysis

All features from `spec.json` are now documented in the user guide. The quick reference (`appendices/b_quick_reference.md`) has been updated to include:

- ✅ Binary string (bstring) functions
- ✅ HashMap functions  
- ✅ Checked math functions
- ✅ Generic List functions
- ✅ OS functions (getcwd, getenv, range)
- ✅ Unsafe blocks and FFI

## Cross-References Added

Brief overview pages now link to detailed documentation:

- `01_getting_started.md` → `part1_fundamentals/01_getting_started.md`
- `02_control_flow.md` → `part1_fundamentals/05_control_flow.md`
- `03_basic_types.md` → `part1_fundamentals/02_syntax_types.md`, `07_data_structures.md`
- `04_higher_level_patterns.md` → `part1_fundamentals/06_collections.md`, `07_data_structures.md`
- `05_modules.md` → `part1_fundamentals/08_modules.md`
- `06_canonical_syntax.md` → `docs/CANONICAL_STYLE.md`, `docs/LLM_CORE_SUBSET.md`

---

*Generated from spec.json version 0.3.0*
