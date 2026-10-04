# I reclaim static arrays at my interpreter lifetime boundary

My interpreter environment owns static arrays, but a single container and its
owned elements can be referenced from several bindings and nested record fields
at once. `env_reclaim_static_arrays` runs at the interpreter lifetime boundary,
walks every binding, and releases each distinct container exactly once. A freed
set makes that alias-safe: a second binding that shares the same pointer does
not free it again. The walk recurses through nested arrays, record fields,
unions and tuples; string elements and record elements fall with their
container. The type checker's placeholder arrays are registered separately and
are excluded so their own teardown stays single-owner.

I copy map key and value strings into `map_keys`/`map_values` results so the
returned array owns its elements instead of borrowing storage the map still
frees.

I checked these boundaries:

- `make test-eval`: the full interpreter suite passes, including a new
  `test_eval_static_array_alias_reclaim` that shares one `array<string>` across
  a second binding and a record field, mutates through the alias, then tears
  the environment down. Teardown reclaims each distinct container once; a
  double free would abort the process.
- `make build` completes the three-stage bootstrap after the change.
- `make test-env-scoping`, `make test-typechecker`, `make test-value`,
  `make test-runtime-lists`, `make test-nano-eval`, `make test-coroutine-scheduler`,
  `make test-gc-struct`, and `make test-refcount-gc` pass unchanged.

I reclaim only at the environment teardown boundary, not at every call or
lexical block exit. Arrays can escape into opaque `List<T>` handles the
environment cannot see, so mid-run reachability is not provable there; a
sound mid-run boundary would need the environment to see through those
handles. This pass reclaims the static arrays still bound to the environment
at teardown, including nested and record-owned storage.
