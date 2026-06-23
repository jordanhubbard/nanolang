# 14.1 JSON - Parsing & Generation

**Parse and generate JSON data for APIs and configuration.**

The `std_json` module provides full JSON support backed by cJSON. Use it to parse JSON responses from HTTP APIs, read configuration files, build JSON payloads for outbound requests, and traverse nested data structures. All `Json` values are heap-allocated opaque objects; call `free` when done to release memory.

## Installation

The module lives in the standard library. Import only the symbols you need:

```nano
from "modules/std/json/json.nano" import parse, free, stringify, Json
```

You can import additional helpers in the same line:

```nano
from "modules/std/json/json.nano" import parse, free, stringify, Json,
    get_string, get_int, get_bool, get_float,
    object_has, keys, get, get_index,
    as_string, as_int, as_float, as_bool,
    is_object, is_array, is_string, is_number, is_bool, is_null,
    array_size, new_object, new_array,
    new_string, new_int, new_bool, new_null,
    object_set, json_array_push,
    decode_int_array, decode_string_array
```

## Quick Start

```nano
from "modules/std/json/json.nano" import parse, free, get_string, get_int, Json

fn greet_user(json_text: string) -> string {
    let obj: Json = (parse json_text)
    let name: string = (get_string obj "name")
    let age: int = (get_int obj "age")
    (free obj)
    return (+ "Hello, " (+ name (+ "! You are " (+ (int_to_string age) " years old."))))
}

shadow greet_user {
    let msg: string = (greet_user "{\"name\": \"Alice\", \"age\": 30}")
    assert (str_contains msg "Alice")
    assert (str_contains msg "30")
}
```

---

## API Reference

### Parsing and Serialization

#### `fn parse(text: string) -> Json`

Parse a JSON string into a `Json` value. Returns a non-zero opaque handle on success. If the input is malformed, behavior is undefined — validate input before parsing when possible.

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `string` | A well-formed JSON string |

**Returns:** `Json` — opaque handle; must be freed with `free` when no longer needed.

```nano
let obj: Json = (parse "{\"x\": 1}")
```

#### `fn free(json: Json) -> void`

Release memory held by a `Json` value. Every value returned by `parse`, `new_object`, `new_array`, `new_string`, `new_int`, `new_bool`, or `new_null` must eventually be freed. Values returned by `get` and `get_index` are **borrowed references** into the parent — do not free them independently.

```nano
let obj: Json = (parse "{\"x\": 1}")
# ... use obj ...
(free obj)
```

#### `fn stringify(json: Json) -> string`

Serialize a `Json` value back to a JSON string.

| Parameter | Type | Description |
|-----------|------|-------------|
| `json` | `Json` | Any valid `Json` value |

**Returns:** `string` — the serialized JSON text.

```nano
let obj: Json = (new_object)
(object_set obj "answer" (new_int 42))
let text: string = (stringify obj)
# text = "{\"answer\":42}"
(free obj)
```

---

### Type Checking

All type-check functions take a `Json` and return `bool`.

| Function | Returns `true` when |
|----------|---------------------|
| `is_null(json)` | Value is JSON `null` |
| `is_bool(json)` | Value is `true` or `false` |
| `is_number(json)` | Value is a number (int or float) |
| `is_string(json)` | Value is a JSON string |
| `is_array(json)` | Value is a JSON array `[...]` |
| `is_object(json)` | Value is a JSON object `{...}` |

```nano
from "modules/std/json/json.nano" import parse, free, is_object, is_array, is_string, Json

fn describe(json_text: string) -> string {
    let j: Json = (parse json_text)
    let kind: string = (cond
        ((is_object j) "object")
        ((is_array j)  "array")
        ((is_string j) "string")
        (else          "other")
    )
    (free j)
    return kind
}

shadow describe {
    assert (== (describe "{}") "object")
    assert (== (describe "[]") "array")
    assert (== (describe "\"hi\"") "string")
    assert (== (describe "42") "other")
}
```

---

### Extracting Primitive Values

These functions convert a `Json` node that holds a primitive value into a NanoLang native type.

#### `fn as_int(json: Json) -> int`
#### `fn as_float(json: Json) -> float`
#### `fn as_bool(json: Json) -> bool`
#### `fn as_string(json: Json) -> string`

Call these after retrieving a node with `get` or `get_index`. Do **not** free the node returned by `get`/`get_index` — it is a borrowed reference.

```nano
from "modules/std/json/json.nano" import parse, free, get, as_int, as_string, Json

fn read_fields(text: string) -> int {
    let obj: Json = (parse text)
    let name_node: Json = (get obj "name")
    let count_node: Json = (get obj "count")
    let name: string = (as_string name_node)
    let count: int = (as_int count_node)
    (free obj)
    return count
}

shadow read_fields {
    assert (== (read_fields "{\"name\": \"x\", \"count\": 7}") 7)
}
```

---

### Convenience Field Getters

These combine `get` + `as_*` and return a safe default if the key is missing.

| Function | Signature | Default |
|----------|-----------|---------|
| `get_string` | `(obj, key) -> string` | `""` |
| `get_int` | `(obj, key) -> int` | `0` |
| `get_float` | `(obj, key) -> float` | `0.0` |
| `get_bool` | `(obj, key) -> bool` | `false` |

```nano
from "modules/std/json/json.nano" import parse, free, get_string, get_int, get_bool, Json

fn load_settings(text: string) -> int {
    let obj: Json = (parse text)
    let host: string = (get_string obj "host")
    let port: int = (get_int obj "port")
    let debug: bool = (get_bool obj "debug")
    (free obj)
    return port
}

shadow load_settings {
    assert (== (load_settings "{\"host\": \"localhost\", \"port\": 9000, \"debug\": true}") 9000)
    assert (== (load_settings "{}") 0)
}
```

---

### Working with JSON Objects

#### `fn object_has(obj: Json, key: string) -> bool`

Returns `true` if the object contains the given key.

#### `fn get(obj: Json, key: string) -> Json`

Returns the value associated with `key` as a borrowed `Json` reference. Returns `0` (null handle) if the key does not exist. Do not free the returned value.

#### `fn keys(obj: Json) -> array<string>`

Returns all keys of a JSON object as an `array<string>`.

#### `fn object_set(obj: Json, key: string, val: Json) -> bool`

Sets a key on a JSON object. `val` is **consumed** by the object — do not free it separately. Returns `true` on success.

```nano
from "modules/std/json/json.nano" import parse, free, object_has, get_string, keys, Json

fn print_keys(text: string) -> int {
    let obj: Json = (parse text)
    let key_list: array<string> = (keys obj)
    let count: int = (array_length key_list)
    for i in (range 0 count) {
        let k: string = (at key_list i)
        if (object_has obj k) {
            let v: string = (get_string obj k)
            (println (+ k (+ " = " v)))
        } else {
            (print "")
        }
    }
    (free obj)
    return count
}

shadow print_keys {
    assert (== (print_keys "{\"a\": \"1\", \"b\": \"2\"}") 2)
}
```

---

### Working with JSON Arrays

#### `fn array_size(arr: Json) -> int`

Returns the number of elements in a JSON array.

#### `fn get_index(arr: Json, idx: int) -> Json`

Returns the element at zero-based index `idx` as a borrowed reference. Do not free the returned value.

#### `fn json_array_push(arr: Json, val: Json) -> bool`

Appends `val` to the end of a JSON array. `val` is consumed — do not free it separately.

#### `fn decode_int_array(arr: Json) -> array<int>`

Converts a JSON array of numbers into a native `array<int>`.

#### `fn decode_string_array(arr: Json) -> array<string>`

Converts a JSON array of strings into a native `array<string>`.

```nano
from "modules/std/json/json.nano" import parse, free, array_size, get_index, as_int, decode_int_array, Json

fn sum_array(text: string) -> int {
    let arr: Json = (parse text)
    let nums: array<int> = (decode_int_array arr)
    let mut total: int = 0
    for i in (range 0 (array_length nums)) {
        set total (+ total (at nums i))
    }
    (free arr)
    return total
}

shadow sum_array {
    assert (== (sum_array "[10, 20, 30]") 60)
    assert (== (sum_array "[]") 0)
}
```

---

### Creating JSON Values

| Function | Signature | Creates |
|----------|-----------|---------|
| `new_object` | `() -> Json` | Empty JSON object `{}` |
| `new_array` | `() -> Json` | Empty JSON array `[]` |
| `new_string` | `(s: string) -> Json` | JSON string |
| `new_int` | `(v: int) -> Json` | JSON number from int |
| `new_bool` | `(v: bool) -> Json` | JSON boolean |
| `new_null` | `() -> Json` | JSON `null` |

Each constructor allocates a new `Json` value that you own and must eventually `free` (unless you pass it to `object_set` or `json_array_push`, which take ownership).

```nano
from "modules/std/json/json.nano" import new_object, new_array, new_string, new_int, new_bool, new_null,
    object_set, json_array_push, stringify, free, Json

fn build_payload() -> string {
    let obj: Json = (new_object)
    (object_set obj "name" (new_string "Alice"))
    (object_set obj "score" (new_int 99))
    (object_set obj "active" (new_bool true))
    (object_set obj "notes" (new_null))

    let tags: Json = (new_array)
    (json_array_push tags (new_string "admin"))
    (json_array_push tags (new_string "user"))
    (object_set obj "tags" tags)

    let result: string = (stringify obj)
    (free obj)
    return result
}

shadow build_payload {
    let s: string = (build_payload)
    assert (str_contains s "Alice")
    assert (str_contains s "admin")
}
```

---

## Examples

### Example 1: Parsing an API Response

A typical pattern when consuming a REST API that returns JSON:

```nano
from "modules/std/json/json.nano" import parse, free, get_string, get_int, object_has, Json
from "modules/curl/curl.nano" import nl_curl_simple_get

struct GithubUser {
    login: string,
    id: int,
    public_repos: int
}

fn fetch_github_user(username: string) -> GithubUser {
    let url: string = (+ "https://api.github.com/users/" username)
    unsafe {
        let body: string = (nl_curl_simple_get url)
        let obj: Json = (parse body)

        let user: GithubUser = GithubUser {
            login:        (get_string obj "login"),
            id:           (get_int obj "id"),
            public_repos: (get_int obj "public_repos")
        }

        (free obj)
        return user
    }
}

shadow fetch_github_user {
    assert true
}
```

### Example 2: Reading a Configuration File

```nano
from "modules/std/json/json.nano" import parse, free, get_string, get_int, get_bool, object_has, Json
from "modules/std/fs.nano" import read

struct AppConfig {
    host: string,
    port: int,
    debug: bool,
    log_level: string
}

fn load_config(path: string) -> AppConfig {
    let text: string = (read path)
    let obj: Json = (parse text)

    let cfg: AppConfig = AppConfig {
        host:      (get_string obj "host"),
        port:      (get_int obj "port"),
        debug:     (get_bool obj "debug"),
        log_level: (get_string obj "log_level")
    }

    (free obj)
    return cfg
}

shadow load_config { assert true }
```

### Example 3: Building a JSON POST Body

```nano
from "modules/std/json/json.nano" import new_object, new_string, new_int, object_set, stringify, free, Json

fn make_create_user_body(name: string, email: string, age: int) -> string {
    let obj: Json = (new_object)
    (object_set obj "name" (new_string name))
    (object_set obj "email" (new_string email))
    (object_set obj "age" (new_int age))
    let body: string = (stringify obj)
    (free obj)
    return body
}

shadow make_create_user_body {
    let body: string = (make_create_user_body "Bob" "bob@example.com" 25)
    assert (str_contains body "Bob")
    assert (str_contains body "bob@example.com")
}
```

### Example 4: Traversing Nested Structures

```nano
from "modules/std/json/json.nano" import parse, free, get, get_string, array_size, get_index, as_string, Json

fn get_first_tag(text: string) -> string {
    let obj: Json = (parse text)
    let tags_node: Json = (get obj "tags")
    if (== tags_node 0) {
        (free obj)
        return ""
    } else {
        let count: int = (array_size tags_node)
        if (== count 0) {
            (free obj)
            return ""
        } else {
            let first: Json = (get_index tags_node 0)
            let tag: string = (as_string first)
            (free obj)
            return tag
        }
    }
}

shadow get_first_tag {
    assert (== (get_first_tag "{\"tags\": [\"nano\", \"lang\"]}") "nano")
    assert (== (get_first_tag "{\"tags\": []}") "")
    assert (== (get_first_tag "{}") "")
}
```

### Example 5: Converting Between JSON Arrays and Native Arrays

```nano
from "modules/std/json/json.nano" import parse, free, get, decode_string_array, new_array, new_string, json_array_push, stringify, Json

fn filter_tags(text: string, prefix: string) -> string {
    let obj: Json = (parse text)
    let tags_node: Json = (get obj "tags")
    let all_tags: array<string> = (decode_string_array tags_node)

    let result: Json = (new_array)
    for i in (range 0 (array_length all_tags)) {
        let tag: string = (at all_tags i)
        if (str_starts_with tag prefix) {
            (json_array_push result (new_string tag))
        } else {
            (print "")
        }
    }

    let out: string = (stringify result)
    (free result)
    (free obj)
    return out
}

shadow filter_tags {
    let result: string = (filter_tags "{\"tags\": [\"api_v1\", \"web\", \"api_v2\"]}" "api_")
    assert (str_contains result "api_v1")
    assert (str_contains result "api_v2")
}
```

---

## Common Pitfalls

### Pitfall 1: Freeing borrowed references

`get` and `get_index` return references **into** the parent object — they share the same memory. Freeing them separately will corrupt the heap.

```nano
# WRONG
let obj: Json = (parse "{\"x\": 1}")
let x: Json = (get obj "x")
(free x)     # Do not do this — x is borrowed from obj
(free obj)   # Crash or corruption

# CORRECT
let obj: Json = (parse "{\"x\": 1}")
let x: Json = (get obj "x")
let val: int = (as_int x)
(free obj)   # Frees everything including x
```

### Pitfall 2: Forgetting to free top-level values

Every call to `parse`, `new_object`, `new_array`, `new_string`, `new_int`, `new_bool`, or `new_null` allocates memory. Forgetting `free` leaks memory.

```nano
# WRONG — leaks the parsed object
fn get_name(text: string) -> string {
    let obj: Json = (parse text)
    return (get_string obj "name")
}

# CORRECT
fn get_name(text: string) -> string {
    let obj: Json = (parse text)
    let name: string = (get_string obj "name")
    (free obj)
    return name
}
```

### Pitfall 3: Passing values to object_set or json_array_push after use

`object_set` and `json_array_push` take **ownership** of the value passed to them. Do not use or free that value afterwards.

```nano
# WRONG
let s: Json = (new_string "hello")
(object_set obj "key" s)
(free s)     # Already owned by obj — double free

# CORRECT
(object_set obj "key" (new_string "hello"))
# No separate free needed
```

### Pitfall 4: Not checking for missing keys

`get` returns `0` when the key does not exist. Calling `as_string` or `as_int` on a null handle will crash.

```nano
# WRONG
let val_node: Json = (get obj "maybe_missing")
let val: int = (as_int val_node)   # Crash if key absent

# CORRECT — use convenience helpers
let val: int = (get_int obj "maybe_missing")   # Returns 0 if missing

# OR check manually
let val_node: Json = (get obj "maybe_missing")
if (!= val_node 0) {
    let val: int = (as_int val_node)
    (println (int_to_string val))
} else {
    (println "key not found")
}
```

### Pitfall 5: Using the wrong import path

The module is at `modules/std/json/json.nano`, not `modules/std/json.nano`.

```nano
# WRONG
from "modules/std/json.nano" import parse

# CORRECT
from "modules/std/json/json.nano" import parse
```

---

## Best Practices

- Always call `free` on top-level `Json` values before the function returns, including early-return paths.
- Prefer the `get_string`, `get_int`, `get_float`, `get_bool` convenience helpers over manually calling `get` + `as_*` — they handle missing keys gracefully.
- Use `object_has` before calling `get` when the key is truly optional and the default-returning helpers are not appropriate.
- For large arrays of numbers or strings, use `decode_int_array` / `decode_string_array` to convert into native arrays, which are safer to iterate over.
- When building JSON to send over HTTP, always call `stringify` and then `free` the object before sending the string.

---

**Previous:** [Chapter 14 Overview](index.html)
**Next:** [14.2 SQLite - Embedded Database](sqlite.html)
