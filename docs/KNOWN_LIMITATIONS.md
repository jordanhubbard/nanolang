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

The `bin/nano` interpreter runs nanolang programs directly without compiling to C.
63 of 69 language examples pass. The remaining 6 have the following limitations:

### Performance: Computationally Intensive Programs

The tree-walking interpreter is significantly slower than compiled C output.
Programs that iterate millions of times will time out:

- `nl_primes_sieve.nano` — Sieve of Eratosthenes over 1,000,000 elements
- `nl_primes_trial_division.nano` — Trial division over 1,000,000 numbers
- `nl_game_of_life.nano` — 40×20 grid Conway's Game of Life (10 generations)

These programs work correctly when compiled with `nanoc` and run natively.
For the interpreter, the benchmark limit should be reduced (e.g. to 10,000) to
complete in a reasonable time.

### Platform: libdispatch (Grand Central Dispatch)

Three examples use GCD for concurrent task dispatch:

- `nl_dispatch_counter.nano`
- `nl_dispatch_pipeline.nano`
- `nl_dispatch_stats.nano`

These require the `dispatch` module shared library and libdispatch. On macOS
this works natively. On Linux, libdispatch must be installed separately
(`libdispatch-dev`) and the module must be compiled. The `group_wait` call
fails when the native library is unavailable.

### Runtime: Missing Data File

- `nl_random_sentence.nano` — Requires a dictionary file at a runtime path
  that is not included in the repository. This is a data dependency, not an
  interpreter bug.
