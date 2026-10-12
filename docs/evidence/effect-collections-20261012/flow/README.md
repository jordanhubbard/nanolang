# My typed array instruction analysis

I connect retained element identities to affine stack joins, local loads/stores,
helper arguments/results and handler parameters/resumption. Equivalent table rows
compare structurally; distinct nominal leaves remain different. Resource arrays
move explicitly, while copyable arrays support ordinary local initialization.

I check nonempty literals against every element in source order. Empty construction
requires one unambiguous retained element type for its opcode tag. Array pushes
preserve exact element ownership, including nested resource construction. Copyable
arrays support get, pop, set, length, slice and remove. Resource-array length uses
a live local observation; consuming an owned array merely to obtain length is
refused. Destructive resource-element extraction/overwrite remains unsupported.
Array borrowing and ambiguous empty nominal construction require further work.

I moved the blanket typed-array refusal from instruction analysis to executable
ownership verification. Every new raw case still fails VM verification and native
emission. These instruction facts do not grant runtime authority. Producer wiring,
actual VM/native operations, cleanup, resource payload layouts and full collection
semantics remain open.

The final suite passes 2,059 normal and 2,996 allocation-fault bytecode checks.
Every failed new analysis preserves instruction-fact buffers. The array/effect/call
fault sweep checks every injected analyzer allocation and a successful recovery.
The same 2,996 checks pass with analyzer, affine state, ownership reader, verifier
and test harness instrumented by ASan/UBSan/leak detection; other linked objects
retain their existing instrumentation. The 563/634 state checks and ownership
contract harness pass. Existing shared C-source effects and 26 VM/native runtime
fault/cleanup fixtures pass; this is regression coverage, not array execution.

I retain the first obsolete contract assertion expecting blanket analysis refusal,
then the raw-fixture parser failure from symbolic tags in numeric u8 operands.
Corrected assertions require analysis success and runtime refusal; corrected raw
fixtures use numeric operands. I do not claim a fresh Nano compiler product.
