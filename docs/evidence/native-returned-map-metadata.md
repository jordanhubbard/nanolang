# Native returned-map metadata and evaluation order

I retain declared map key/value types when native map operations receive an
ordinary function call or checked callback result. This is MAC
`task_e018b78bc20a47d18619fce55a20e567`, PR #496.

## Observed failure and repair

A direct `map_get` on a function returning `HashMap<string,string>` previously
selected an integer fallback instead of the map specialization. The ordinary
fixture at `/tmp/nanolang-map-return-expression.nano` fails native compilation
with the preceding validated compiler; its baseline log is retained beside it.
An explicit typed map local worked, isolating this defect from lexical cleanup.

My native suffix resolver now reads checked callback result TypeInfo first,
then a visible function-value signature or the named function declaration's
complete return annotation. It preserves the existing identifier and checked
field paths. Each operation captures its receiver, key and value once, in
source order, before invoking the selected native helper. Distinct generated
names preserve nesting and avoid registered source bindings.

I do not change cleanup, map allocation, returned-value retention or lifetime
semantics. Those remain separate from selecting the correct native operation.

## Validation

At source checkpoint `f190d173`, I passed a fresh normal-budget bootstrap and
all 24 methods across these suites (53.062 seconds):

- `tests.test_native_returned_maps`
- `tests.test_map_field_operations`
- `tests.test_map_constructor_contexts`
- `tests.test_native_effect_execution`

The earlier focused map gate passed 13 methods in 37.869 seconds. My dedicated
suite covers all four int/string key/value pairs through native and NanoVirt,
including get, set/put, has, remove, length/size, clear, keys and values. Trace
counters require receiver/key/value order and single evaluation. Nested receiver
calls retain the same order. A return expression uses a newly produced map
exactly once, and a checked callback result retains its map annotation.
Twenty-four wrong key/value/arity decisions across C seed and NanoVirt reject
before publication and preserve an existing artifact.

The native explicit-free order case is deliberately native-only. NanoVirt
currently rejects `map_free` while compiling mandatory shadows. I preserve that
first failing gate in `/tmp/nanolang-returned-map-focused.log` and track the
missing portable lifetime/alias contract and lowering in
`task_2f848b73acf847a79df68418b9213637`. This is not a claim that map_clear and
map_free have equivalent lifetime behavior.

The reproducible dedicated gate is `make test-native-returned-maps`. Passing
logs are `/tmp/nanolang-returned-map-focused-r2.log`,
`/tmp/nanolang-returned-map-bootstrap.log` and
`/tmp/nanolang-returned-map-integrated.log`. General early-return/borrowed-result
cleanup remains open in `task_195cac35e7704e56805932977512ae02`.
