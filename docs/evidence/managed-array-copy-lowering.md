# My counted literal and fresh slice lowering

I qualify managed child b702 after VM slice cleanup PR715, private runtime
PR718 and non-admitting shape PR719. Production checkpoint `7aefa397` adds
ARR_LITERAL/SLICE only to the managed profile, with the existing whole-module
shape predicate. Other profiles and nested/cyclic/nominal/host exclusions stay
unchanged. My exact contract is `NANOISA_MANAGED_ARRAY_LITERAL_SLICE.md`.

I read literal inputs into entry-allocated scalar payload/tag scratch without
removing their original stack roots. Failed private preparation reaches frame
cleanup before pushing a result. Successful preparation consumes every input
exactly once through a counted loop, outside FrameOutput's three-owner limit.
A later release failure discards the prepared result. My slice helper consumes
receiver and both bounds after their final use, including type refusal and
non-integer fallback. Existing frame status checks own and clean its result.

My nine focused methods pass in 7.068 seconds. Ordinary values run on NanoVM,
ASan-instrumented native LLVM, import-free Node and Wasmtime, including the
nvm2wasm published artifact. I exercise all finite packed pairs, counts
0/1/9/17, boxed leaf source order, duplicate dynamic string children with NUL,
copy identity and independent mutation, mixed packed origins, prepared split
copies, endpoint wrapping/defaults, borrowed helper calls and child survival.
Native/Node public-entry controls establish global persistence, initializer
order, repeated entry versus fresh instances, type-error cleanup, deterministic
native allocation failure and finite-memory Wasm copy failure. I do not claim
those custom repeated-entry harnesses ran in Wasmtime.

My native allocation budgets include both failure and success and require zero
live objects after complete frame cleanup and a successful subsequent entry.
My Wasm pressure case keeps a 32,000-element packed source and two independent
copies in globals; the next copy returns MEMORY under a 1 MiB limit. Three
committed array roots survive repeated failed entries and terminal disposal
reclaims them. Unsupported packed/nested shapes preserve prior output.

I retain the first seven-method pass in
`/tmp/nanolang-copy-lowering-focused.log`, the first expanded run in
`/tmp/nanolang-copy-lowering-expanded.log`, and corrected nine-method evidence
in `/tmp/nanolang-copy-lowering-corrected.log`. The expanded run exposed two
fixture assumptions: the VM's intended result 2 is a nonzero process exit, and
a 40,000-element source could exhaust the Wasm allocator during capacity growth
before the intended slice pressure. I corrected the exit expectation and used
a bounded 32,000-element source with two committed copies to isolate slice
failure. I retained exact root-count and cleanup assertions.

Full aggregate488, managed51da and release acceptance remain open. Source
array_slice start/length convention efa11 remains separate. I do not claim a
new source compiler bootstrap, Darwin sanitizer gate or historical incident
resolution from this runtime-only change.

My frozen combined gate passes all 61 managed instruction methods in 82.487
seconds, all 18 shape methods in 2.716 seconds, the shared verifier profile gate,
and their runtime/package/core prerequisites. I retain
`/tmp/nanolang-copy-lowering-full.log`. Root independently reviewed production
`7aefa397` and its frame-cleanup surroundings without a scoped blocker.
