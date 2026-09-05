# My Known Limitations

## Function Variables (First-Class Functions)

**Status:** FULLY SUPPORTED (I fixed this in commits d5dceb2 and c12704f)

### What I Support
- Functions as parameters: `fn apply(f: fn(int)->int, x: int) -> int`
- Passing functions to other functions: `(apply increment 5)`
- Functions returning function references: `fn get_func() -> fn(int)->int`
- **Function variables**: Storing functions in variables and calling them
  ```nano
  let f: fn(int) -> int = increment  # Variable assignment works
  let result: int = (f 10)            # Calling through variable works
  ```
- Calling function-typed parameters: `fn filter(predicate: fn(int)->bool, n: int) -> bool { return (predicate n) }`
- Conditional function selection: `fn select(use_add: bool) -> fn(int,int)->int { if use_add { return add } else { return mul } }`

### What I Fixed
1. **Memory Management (d5dceb2)**: I fixed function value cleanup in `env_set_var()`, `free_environment()`, and `eval_function()`. I now use `free_function_signature()` instead of the incomplete manual cleanup I used before.
2. **Type Checking (c12704f)**: I added the `fn_sig` field to `TypeInfo` and now store function signatures when I add function parameters to the environment. This allows me to correctly infer return types when you call function parameters.

### Technical Details
My fixes addressed:
- Proper cleanup of function_name and signature in all paths. I no longer double-free memory.
- Complete signature cleanup including param_struct_names and nested signatures.
- Storage of function signatures in TypeInfo for function-typed parameters.
- Return type retrieval from stored signatures during my type checking phase.

I have verified that all first-class function features are functional and tested.

## Top-Level Constants with Uppercase Names in Conditionals

**Status:** FIXED in my parser

This was causing issues, but I resolved it by improving my struct literal detection heuristic.

---

## Tree-Walking Interpreter (`bin/nano`) — Known Limitations

The `bin/nano` interpreter runs NanoLang programs directly without compiling to C.
Its performance and host-module boundaries differ from compiled C output.

`bin/nano` sets `NANO_INTERPRETER=1` unless that variable is already present.
Language examples that used to time out or abort now either finish a smaller
workload or print `SKIP:` and exit 0.

### Performance: Computationally Intensive Programs

The tree-walking interpreter is slower than compiled C output. Those examples
take a smaller default when `NANO_INTERPRETER` is set:

- `nl_primes_sieve.nano` — 10,000 under the interpreter, 1,000,000 when compiled.
  Override with `NANO_PRIME_LIMIT`.
- `nl_game_of_life.nano` — 8×8 and 2 generations under the interpreter, 40×20
  and 10 generations when compiled. Override with `NANO_LIFE_WIDTH`,
  `NANO_LIFE_HEIGHT`, and `NANO_LIFE_GENERATIONS`.

`nl_primes.nano` prints primes up to 50. It is not a million-element trial.

### Platform: libdispatch (Grand Central Dispatch)

Three examples use GCD for concurrent task dispatch:

- `nl_dispatch_counter.nano`
- `nl_dispatch_pipeline.nano`
- `nl_dispatch_stats.nano`

These require the `dispatch` module and libdispatch. On macOS that is the
system GCD. The C stubs on other hosts return `dispatch_available() = false`.
The examples print `SKIP: libdispatch is not available on this platform` and
exit 0 instead of aborting on `group_wait`. Under `bin/nano` they also print
`SKIP:` because the tree-walker cannot run libdispatch callbacks.

### Runtime: Dictionary file

- `nl_random_sentence.nano` — Reads `NANO_DICT_PATH` if set, then
  `examples/language/data/words.txt`, then `/usr/share/dict/words`, then a
  built-in word list. Missing system dictionaries are not a hard failure.
