# My scalar-global runtime checkpoint

I admit exact scalar globals in the standalone owned value graph under my
[contract](../../../NANOISA_OWNED_SCALAR_GLOBALS.md). My verifier establishes
entry initialization and exact scalar stores before public execution. My VM
checks tags using its existing global storage; my native entry passes a local
global context to helpers and releases it on success or a trap.

Nine paired raw modules cover helper counters, all five scalar tags, retained
and overwritten strings, both initializer branches, a counter in an owned-union
producer, loops, helper string mutation, and a trap with live globals and owners.
Nine refusals cover absent/incomplete initialization, wrong slots/tags, owners
and observations stored as scalars, early calls, absent declarations and wrong
helper stores. Refused native publication preserves the previous output.
My VM fixture repeats each accepted entry twenty times with both dispatch
profiles and refuses direct helper invocation. It passes 1,289 assertions.
Generated native cases pass ASan/UBSan with leak detection and exhaustive
single-allocation failure injection with zero live allocations after entry.

My VM allocation fixture covers initialization and invocation in both dispatch
profiles, successful and trapping entries, single failures and sustained
failures. It passes 660 assertions. The first version incorrectly required a
memory error for every failed allocation: initial stack allocation can recover
in stack reservation, and initial intern-table allocation can recover during
insertion. I retain that failing terminal, explicitly check the two recovery
cases against the original result/trap, and require memory errors for all
subsequent required allocations and for sustained failures. Stack/frame cleanup
and heap object counts remain checked.

Both computed-goto and switch builds pass scoped ASan/UBSan with leak detection.
`instrument_runtime.py` rebuilds VM, heap, cycle collector, native emitter,
fixtures and allocation-injected VM/heap objects with Homebrew LLVM; other
linked dependencies remain ordinary objects. These checks do not establish
whole-program instrumentation coverage.

Adjacent checks pass: scalar-global contracts (105,066 ordinary and 131,242
allocation-injection assertions), affine bytecode (837 and 1,199), owned-union
runtime (1,443 assertions and 81 allocation checks), shape constraints (1,500),
and the complete native translator regression (2,428). `provenance.json` pins
source hashes; `logs.json` pins compressed terminal contents and exit status.

Source-global initialization/read/write lowering remains unfinished in both
owned producers. The original exactly-once source counter, fresh stages and
complete platform/release gates remain required. I keep PR #522 draft.
