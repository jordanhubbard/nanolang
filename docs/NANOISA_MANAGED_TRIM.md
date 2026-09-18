# My managed byte-trimming contract

I implement full target-coverage child `task_bc2bd84520124b17aa4dfad09fb663dd`
after the managed predicate and substring prerequisites. I admit STR_TRIM only
in the managed profile; scalar/literal-only decisions remain unchanged.

I match the current VM handler: remove only leading/trailing ASCII space, tab,
newline and carriage return from the complete stored byte view. Vertical tab,
form feed, NUL, non-ASCII and interior whitespace remain ordinary retained
bytes. Empty, all-whitespace and already-trimmed inputs still allocate a fresh
result, as the VM does. I do not silently turn unchanged output into an alias.

My runtime helper consumes one input owner on every path. I validate its view,
scan indices within the stored length, and delegate the checked offset/length
to my existing consuming substring helper. Copying/allocation completes before
the input owner is released. Empty spans still use the same allocator. A failed
allocation releases that transferred owner, preserves unrelated aliases and
leaves the output handle unchanged. I add no new object representation or host
imports, and I preserve first-error status and module teardown.

LLVM lowering transfers exactly the source operand to this helper. Wrong tags
are reported through existing managed status and release that operand once;
frame cleanup remains responsible for other locals/stack values. A successful
result carries one string owner. Calls/globals/reentry follow current rules.

I require paired VM/native LLVM/import-free Node/Wasmtime byte controls on
literal and computed inputs, exact whitespace/NUL/high-byte behavior, same-value
freshness in direct runtime tests, retained aliases, calls/globals/reentry and
complete disposal. Native/Wasm runtime fault injection must cover allocation
failure with unchanged output and surviving aliases; native emitted failure
and recovery must use the real packaged runtime. Shared profile and unsupported
operation previous-output controls remain required.

During static audit I separately recorded missing release of STR_CHAR_AT's
popped index operand as task `task_62caf894db9649cd904ce6faf3c37ffb`. That operation
stays unadmitted here. Root owns canonical prefix-conversion task1c2a. Neither
this slice nor that companion closes full runtime51da, Darwin managed7ba,
historical evaluator791a, aggregate/cycle488 or host-linkage2d2 obligations.
