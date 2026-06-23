# nanolang Standard Library

`stdlib/` contains the nanolang standard library modules.

## Usage

```nano
import option
import result
import list
import map
import set
import iter
import string
```

## Modules

| Module | Description | Key Types |
|--------|-------------|-----------|
| `option` | Optional values — present or absent | `Option { has_value, value }` |
| `result` | Explicit error handling | `Result { is_ok, value, err_msg }` |
| `list`   | Persistent singly-linked list | `int` (opaque handle) |
| `map`    | Persistent sorted map (string→int) | `int` (opaque handle) |
| `set`    | Persistent sorted set of ints | `int` (opaque handle) |
| `iter`   | Lazy iterator combinators | `int` (iterator handle) |
| `string` | String utility functions | `string` |

## Design Philosophy

1. **Persistent / immutable** — all operations return new values; no mutation
2. **Opaque handles** — collections are `int` indices into interpreter storage
3. **Functional API** — map/filter/fold on every collection
4. **Option-returning** — unsafe operations (head, nth) have safe `_opt` variants
5. **Built-in bridge** — `__builtin_*` calls delegate to the runtime for O(1) ops

## Examples

```nano
import list
import option
import result
import iter

-- List operations
let xs    = list.of(1, 2, 3)
let ys    = list.map(xs, fn(x) { x * 2 })       -- [2, 4, 6]
let evens = list.filter(xs, fn(x) { x % 2 == 0 }) -- [2]
let sum   = list.fold(xs, 0, fn(a, x) { a + x }) -- 6

-- Option
let found = list.nth(xs, 1)   -- Some(2)
let v     = option.unwrap_or(found, 0)

-- Result
fn safe_div(a: int, b: int) -> result.Result {
  if b == 0 { result.err("division by zero") }
  else { result.ok(a / b) }
}
let r = safe_div(10, 2)
print(result.unwrap(r))  -- 5

-- Iterator
let sum2 = iter.sum(iter.filter(iter.range(1, 100), fn(x) { x % 2 == 0 }))

-- Map
let m  = map.empty()
let m2 = map.insert(m, "score", 42)
print(option.unwrap(map.get(m2, "score")))  -- 42
```

## Runtime Support

The collection modules rely on built-in functions (prefixed `__builtin_*`)
provided by `src/builtins_registry.c`.  The interpreter implements:

- `__builtin_list_cons`, `__builtin_list_head`, `__builtin_list_tail`
- `__builtin_map_new`, `__builtin_map_insert`, `__builtin_map_get`, etc.
- `__builtin_iter_range`, `__builtin_iter_map`, `__builtin_iter_collect`, etc.
- `__builtin_strlen`, `__builtin_string_slice`, `__builtin_string_split`, etc.

See `src/builtins_registry.c` and `src/runtime/` for implementation details.
