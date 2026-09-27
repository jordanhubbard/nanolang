# My C-seed owned foreign strings

My C seed can retain a raw foreign string until explicit release or process
exit. I opt into this behavior only when the source declares both a
string-returning extern and its `<name>__nano_string_release_v1` companion.
The companion must be an extern returning `void` with one `string` parameter,
and both declarations must carry the same source-file identity. I currently
apply this adapter to `int`, `u8`, `float`, `bool` and `string` parameters.
Missing or differently shaped companions do not authorize cleanup.

I emit a typed wrapper around the linked function and register each non-null
result with my shared process owner. A typed finalizer calls the declared
companion; I do not guess that an arbitrary string can be freed. The wrapper
preserves a null result. If owner registration fails, the owner releases the
new result and I exit with a diagnostic rather than publish a dangling pointer.

I map function values as well as direct calls to the wrapper. The explicit
release wrapper first forgets the process registration, then calls the original
companion. The caller must stop using every alias after explicit release.
Otherwise aliases and returned strings remain valid until process exit. This
is process-lifetime retention, not bounded lexical ownership.

My standard filesystem module declares its six existing path companions.
Four C-seed path operations already select GC-managed runtime facades; they
keep that mapping and require GC shutdown separately. The provider ABI stays
unchanged. My VM and canonical native adapters retain
their existing copy-and-release rule; these C-seed wrappers do not change it.
My native entry points register checked GC shutdown before entering the program,
so managed facade results also receive real exit cleanup. Wasm does not receive
these C-seed process-exit hooks.

`tests.test_cseed_process_ownership` checks real path results, escaping aliases,
all six path helpers, and a counted foreign provider with function-value calls
and explicit release. The shared owner tests exercise allocation and exit-hook
failures, concurrent adoption, explicit cleanup and process exit. Instrumented
product checks disable conservative leak roots so retained globals cannot
substitute for actual cleanup.
