# My selected owned-union runtime checkpoint

I admit ownership format 4 through the existing bounded standalone owned profile
only after declaration validation and selected affine dataflow succeed. I implement
`OWN_UNPACK_VARIANT` in VM switch/threaded dispatch and native emission, using its
selected field count. I check exact carrier identity and unique ownership before
detaching payloads. VM moves, destinations, calls and returns resolve union ordinals
to retained layout indices. Native union arguments and returns retain tag, layout,
variant and count checks.

`make test-owned-union-runtime` passes 1,443 C fixture checks and 81 VM heap
allocation-failure checks. Eleven paired VM/native cases cover owned, ordinary,
empty, nested and STRING payloads, aliases, arguments/returns and assertion traps.
Five public refusals preserve previous native output. Native execution runs under
ASan/UBSan with leak detection, failing each allocation in turn and requiring zero
live allocations after both failure and success. I select Homebrew LLVM on this
Darwin host because Apple ASan does not support leak detection.

`instrument_runtime.py` instruments VM dispatch, heap, cycle collector, native
emitter and both fixtures; other linked dependencies remain ordinary objects.
Both default threaded dispatch and `--switch` pass the same 1,443/81 checks with
ASan/UBSan and `detect_leaks=1`. The fixture alternates fused and unfused profiles.
This is scoped instrumentation, not a fully instrumented product build.

My existing owned runtime passes 1,521 fixture checks and 32 VM allocation checks.
Its case 4 exposed an existing native `AGG_GET` guard that refused STRUCT carriers.
I retain that first failing log and correct the guard to accept either checked
record or union carriers. The unchanged result-84 case now passes. The final
native harness also requires allocation failures to return the allocation status.

Regression gates pass: 33 schema tests, 98 verifier tests, 1,500 shape assertions,
2,428 translator assertions, affine state 425/457 and bytecode 793/1,155 normal/fault
checks, scalar-union runtime and ownership transport. The transport-only format-4
fixture still refuses because it contains no owned transfer; its diagnostic now
states that boundary rather than claiming all format-4 execution is unsupported.

I have not qualified canonical resource-union source lowering, a fresh bootstrap,
final fixed points or the complete hosted matrix. Those remain required for #522.
`logs.json` records uncompressed log hashes; compressed files preserve the terminals.
