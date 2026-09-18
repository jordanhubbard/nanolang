# My consuming-call runtime evidence

I implement `task_4ef48e1066534e639e577c9622445697` under the bounded
[consuming-call contract](../NANOISA_CONSUMING_CALL.md). Production is
`d6c8bd56`; I keep source admission and full ownership parents open.

I test seven ordinary modules: leaf and nested arguments, repeated calls,
helper and entry assertion failures, and exact int/bool/u8 results. The helper
borrows its incoming owner and a separately constructed local owner before
explicit destruction. The caller retains a distinct owner throughout the call.
All four VM entry APIs repeat the cases and check empty terminal stacks/frames,
inactive reference contexts, increasing invocation generations and baseline
heap object counts. I compare generated native artifacts and verify each
module before execution. Native tests check exact scalar results, repeated
calls, each owned allocation failure and zero remaining allocations under
ASan/UBSan/LSan with both GCC and Clang18.

My static refusal controls cover held and moved sources, observations, a
same-shaped different nominal owner, unresolved callee owners, deeper calls
and a borrowed formal supplied to consuming CALL. I never execute those
refused modules. CALL now has a connected bounded contract, so the previous
reachable CALL0 control expects its exact entry-to-helper refusal. My separate
unreachable unsupported-opcode control uses still-unconnected TAIL_CALL.

## Measured failure paths

- I pass 3,091 focused lifecycle/refusal checks and 104 heap allocation checks.
- I separately pass 90 real-VM preflight checks across all four entry APIs:
  failed stack reallocation and an injected NULL helper-contract creation
  result, each followed by successful reuse. No helper generation activates
  before frame reservation. These do not claim exhaustive affine allocator
  suballocation injection.
- I preserve the ordinary standalone VM convention: only INT return values
  become process status. I check BOOL/U8 values through the VM APIs and native
  result harness rather than equating their standalone process statuses.

## Retained qualification corrections

My first test compilation found misleading indentation in the new fixture;
I corrected the test formatting. The first BOOL/U8 CLI assertion assumed the
INT-only VM process-status convention applied to all scalar results; I corrected
that harness assumption without changing runtime behavior. Initial `clang`
selected a host installation with a GCC include-directory preference warning
under `-Werror`; installed `clang-18` passes without warning suppression.

My first adjacent affine test expected CALL to remain wholly unconnected; I
updated the precise contract checks described above. The older-base native
suite passes 2,421 checks and fails one stale unsupported-ROT3 control already
repaired by canonical commit `248fd407`; I retain that failure and integrate
the existing correction before final qualification.

An optional build instrumenting every VM/NanoISA source at `-O0` exceeds the
per-command 90-second harness limit without sanitizer diagnostics. I preserve
`/tmp/nanolang-consuming-call-sanitizer-final.log` and measure that unchanged
binary once with a separate 240-second bound; permanent test deadlines remain
unchanged. My first instrumentation command mechanically mapped the facade
object to a nonexistent same-name source; its corrected source list retains
that unrelated object and instruments actual VM/NanoISA source files.

I retain the focused, Clang18, preflight and adjacent logs under
`/tmp/nanolang-consuming-call-*.log`. This evidence concerns corrected ordinary
modules and defensive failure paths, not historical product incidents.
