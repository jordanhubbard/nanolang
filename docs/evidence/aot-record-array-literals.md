# Record-array literals and unresolved local fields

I classify explicitly tagged record-array literals as record arrays, including
empty literals. I unify their element shape constraints and reject scalar
elements or conflicting record field representations. Generated C constructs
the array and inserts record values in source order. I retain the existing
record-array runtime capacity; this change does not remove that bound.

Adding literal support alone did not clear the compiler's function 264 field
conflict. Further tracing found that unresolved values loaded through locals
inherited an integer field vector. I changed that fallback to unknown facts.
Unresolved information must not establish an integer representation.

## Verification

`make -j1 test-nvm2c` passes 1,080 AOT checks and 965 graph checks. Tests cover
empty record literals, subsequent insertion, mixed-field literal elements,
element order, scalar-element rejection, conflicting-field rejection, and a
nested record routed through a local after a forward call. `git diff --check`
passes.
`make -j1 test-nvm2c-sanitizers` passes the same 1,080 AOT and 965 graph checks
with fresh ASan/UBSan object instrumentation verified.

Compiler acceptance clears function 264's metadata merge conflict. It now stops
at function 267 (`build_field_metadata_index`) with operand-stack underflow.
That function's hashmap opcode requirements need investigation and implementation;
this is not a successful full compiler acceptance run.

Parent MAC task: `task_9c850e94e5a74b6f8941622e2872af23`. Claiming still returns
`agent_status_unavailable`; I retain repository evidence and keep the parent
task open.
