# Concrete union literal context

I retain the declared payload annotation while emitting a nested union literal.
My native self-hosted emitter previously passed an empty expected type to every
payload, so an inline `array<Plain>` chose integer storage. I now substitute
concrete arguments using the same lexical helper as my checker and pass the
result into recursive expression emission. I preserve complete type-name tokens,
including qualified names.

On September 17, 2026, my fresh three-stage bootstrap passed. My three positive
programs execute fixed, generic and nested generic record-array payloads. Those
programs pass with my C seed and both self-hosted stages. Both self-hosted stages
also reject fixed and generic wrong-record payloads without replacing a prior
artifact. Four initial methods across both stages passed in 22.251 seconds; the
additional generic negative passed across both stages in 3.306 seconds. My 49
adjacent generic, payload, affine-boundary and collection methods passed in
105.541 seconds.

My complete new paired gate is **not yet passing**: my C seed accepts both
wrong-record payloads at this checkpoint. The fixed contract is under separate
review; concrete generic constructor context is tracked as
`task_dd2be49bc494483f9bb18646a0013055`. I retain both failing regressions in
`tests/test_union_literal_context.py`. This checkpoint does not establish
selected-variant ownership transfer or NanoISA ownership support.

My local logs are `/tmp/nanolang-union-context-bootstrap.log`,
`/tmp/nanolang-union-context-selfhost-tests.log`,
`/tmp/nanolang-union-context-generic-wrong-selfhost.log`, and
`/tmp/nanolang-union-context-adjacent.log`. The failing C checks are preserved in
`/tmp/nanolang-union-context-c-tests.log` and
`/tmp/nanolang-union-context-generic-wrong-c.log`.
