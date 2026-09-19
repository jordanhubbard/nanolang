# I retain prepared mutation roots through later failures

I own runtime acceptance child `task_5652d22de64799e0d9734a9d01958917`
under parent430220 and the reviewed [mutation source contract](NANOISA_OWNED_FLOAT_ARRAY_MUTATION_SOURCE.md).
Public activation842 is merged at8953110f4. Source18731 and mutation bba622 remain
with their producer owner. This checkpoint proposes fixtures and qualification,
not runtime or producer changes. I seek review before constructing or executing
new modules.

## I distinguish existing evidence from missing combinations

My qualified runtime corpus exercises shared aliases, mutation and growth,
reverse unpack, nested factories/relay results, STRING cleanup, optional FLOAT
comparisons, heap allocation failures and subsequent recovery. Direct growth
controls preserve storage/content/length and accounting after failed realloc.
The source contract additionally requires these ordered combinations:

| Obligation | Current evidence limit | Required new observation |
| --- | --- | --- |
| Prepared ARRAY receiver below a later consuming INT-helper call | Existing calls do not leave that receiver below the consuming arguments | The receiver is evaluated once before the call; call failure cleans it and the moved argument without losing prior effects |
| Failed owner pack after successful append | Existing packing precedes later append/growth | Earlier append is visible before a precisely attributed shell/field allocation failure; no duplicate release or leaked alias follows |
| Append result used as a receiver/field and setter discards its internal ARRAY result | Primitive stack behavior is qualified separately | Nested ordering, alias identity and exact final stack balance agree in one admitted graph |

These are acceptance gaps, not demonstrated runtime defects. I keep all existing
refusals and public authority intact. No bare ARRAY signature, managed binding
reassignment, borrowed managed root, ordinary-record/owner profile composition,
FLOAT helper result or optional FLOAT write is newly admitted.

## I use ordinary admitted value graphs

I construct fresh exact FLOAT-array modules using existing literal, ARR_PUSH,
ARR_SET/POP, ARR_LEN, owner pack/move/unpack, direct acyclic call and INT print
operations. Exact finite FLOAT comparison observers use the already qualified
six comparisons and their optional operand checks. I obtain fresh public complete
authority before executing each valid case. Invalid controls only query refusal.
All descriptors, function/local/stack limits, unique shell consumption and exact
FLOAT origins remain within the merged profile.

The first graph retains an alias, appends through a nested receiver exactly once,
then leaves that ARRAY root below an owner argument consumed by an INT-returning
index helper. The helper consumes the shell and returns a valid index; the caller
stores an exact FLOAT and discards ARR_SET's receiver result. Integer length,
finite comparisons and distinct INT output markers observe argument order,
once-only mutation and alias identity. Normal completion leaves one INT result,
no operand residue, and no live owner/reference activation.

The second graph creates its child Handle before mutation, appends through an
outside alias, prints the resulting length, then packs a Handle+ARRAY owner.
The pack failure must occur after the append marker. Successful controls unpack
and consume the child once while the alias remains usable. A separate append-as-
constructor-field path observes source-order staging without changing field-order
packing or introducing new source syntax. If the current builders need a two-field
owner layout, I add only a fresh supported fixture descriptor, not production
classification or ABI changes.

## I force an actual call preflight allocation

My existing VM fault hooks instrument heap.c positive allocations. They do not
cover vm.c's stack_reserve_frame realloc. VM_STACK_INITIAL is4096, larger than
these bounded fixtures naturally need; I do not claim a call-allocation failure
from a heap-only loop.

For a separate VM stress control, after normal vm_init and before invocation I
replace the empty stack with a genuinely allocated eight-cell buffer, free the
old empty buffer and set its matching capacity. I retain all module constants
and leave stack_size/frame_count/reference state zero. A root with four locals
fits together with its prepared receiver and one owner argument, while an
otherwise supported callee with eight locals requires real growth at CALL.
This changes only valid fixture capacity, not bytecode semantics or public APIs.

I compile the fixture VM translation unit with a realloc interposer. Its armed
condition matches the current stack pointer and a real requested growth; it
records the requested size, current function/IP, stack depth and frame count,
then returns NULL at exactly that allocation. Other reallocations delegate to
the real allocator. The fixture verifies that failure is the consuming CALL
preflight, after the receiver and owner argument have been evaluated and before
the helper marker or new activation. I do not inject a synthetic VM error or
change production limits. Disarming the fault permits an independent ordinary
recovery invocation with the same admitted module.

I keep exact output conventions: preadmission/refused continuations preserve
caller sentinels; post-entry vm_invoke/callable failures retain their existing
VOID convention, while stack-result APIs do not invent a result. Exact visible
prefixes, VM_ERR_MEMORY, no remaining frames/operand/owner/ARRAY roots and the
existing module-root graph plus post-collection byte baseline remain required.
The retained buffered zero-reference candidates are not counted as leaked roots.

## I attribute pack and growth faults to their actual allocation sites

I retain the existing primitive growth failure checks and public terminal cleanup
controls. For new post-append packing, I first inspect the generated/native and
VM allocation sequence. Fault records distinguish shell allocation, field-buffer
allocation, managed carrier reservation and call preflight; I do not infer which
site failed merely from a later passing run. If ordinary allocator-prefix counts
cannot establish the exact site, I propose a fixture-only allocation observer
before implementing it. I do not add a production hook without a separate review.

Native generated C uses the existing allocator wrapper and checked entry/output
convention. Failed allocations preserve earlier printed mutation observations,
return the established memory status and leave the output sentinel plus zero
tracked live allocations. Successful recovery observes every alias and consumes
each unique shell once. Native helper carrier preflight is qualified at its own
actual allocation site; I do not equate it with the VM stack allocator.

## I freeze the bounded acceptance

1. I review the static module/admission and allocation-site plan before fixtures.
2. I freeze the new harness and reviewed production, then qualify normal controls,
   exact faults, cleanup and recovery across switch/computed-goto and fused/unfused
   VM dispatch plus native O0/O2. GCC and Clang strict/sanitizer configurations
   retain exact instrumentation scope; pending source and Darwin acceptance remain
   separate, not silently completed by Linux runtime checks.
3. I prepare ordinary providers/CLIs once and run fixture commands directly from
   exact Make target environments. Inventoried provider/CLI/compiler hashes match
   around every phase. An external time bound and first-terminal seals remain.
4. I send evidence to the producer owner for the later paired source mutation
   matrix. I close only this bounded runtime child after reviewed canonical merge;
   source18731, bba622 and parent430220 require their own complete acceptance.

## I pin the actual allocation observers before implementation

My static native audit corrects one promise above: nvm2c_owned.h emits fixed
`t[256]`/`l[256]` arrays and shares the existing NmsRuntime across consuming
helpers. Parameter validation and transfer in that path allocate no heap carrier.
I therefore qualify native call success/order/alias behavior and actual pack/growth
faults; I do not invent a native call-preflight allocation failure. The VM's real
stack realloc obligation is unchanged.

For post-append packing, I prepare the child Handle and STRING field before the
append. The returned ARRAY remains on the operand stack after a duplicated
length observation prints its marker; loading the previously prepared STRING
performs only a retain, then OWN_PACK constructs the Bundle. VM observers match
current function and the decoded next instruction PC for that exact OWN_PACK,
then distinguish heap.c's malloc(sizeof(VmStruct)) shell allocation and
calloc(field_count,sizeof(NanoValue)) field allocation. They record exact
arguments and independently assert the retained operand tags before returning
NULL. They do not wrap layout-decoder allocations or synthesize an opcode error.

Native's existing NOWN_ALLOC macro wraps both strings and shells. I use a
fixture-only replacement that delegates normal allocations and fails the first
NOWN_ALLOC after the exact append-length marker. In this fixed graph the STRING
and Handle allocations precede that marker and the next NOWN_ALLOC is the Bundle
shell/inline-field allocation emitted for OWN_PACK. I verify the generated
sequence statically and assert the count/size supplied to the observer. I retain
separate allocator accounting and exact output prefix/recovery checks; no
production hook or emitted source rewrite follows.

## My first complete fixture checkpoint

I prepare two fresh graphs using the existing five-layout builder, with no new
layout or production changes. The call graph has root locals ARRAY, Handle, INT,
INT and an eight-local consuming helper. Its stack before CALL is four locals,
the append-result ARRAY receiver and one moved Handle. The callee preflight needs
13 slots while the real capacity is eight. The helper prints its consumed7 and
returns index0; the caller sets3.5, observes it through the original alias and
prints length1. Normal output is `1\n7\n1\n`; the targeted preflight failure must
retain exactly `1\n` and no helper activation.

The pack graph prepares Handle7 and STRING first, then stages Handle, append-result
ARRAY, and retained STRING for the existing three-field Bundle layout. The array
length marker is1. The exact OWN_PACK next-PC identifies heap shell and three-field
buffer failures; both leave the complete staged values for normal failure cleanup.
Normal reverse unpack consumes the child and observes the same alias, producing
`1\n7\n1\n`. Native's first NOWN_ALLOC after that marker must have one item of
`sizeof(nown_record)+3*sizeof(nown_value)`; its injected refusal produces only
`1\n`, status1 and an unchanged result sentinel. VM memory status and per-entry
result conventions remain exact.

I qualify four synchronous public APIs and both fusion settings with fresh VMs,
separate true-switch/computed-goto compilation, native O0/O2 and recovery after
each targeted fault. The static VM preflight observer records exact stack types,
size and frame count; no pending module executes before full public admission.
Existing growth failure/accounting controls remain the separately qualified
primitive/whole-runtime evidence and an adjacent unchanged target; this checkpoint
does not relabel them as new allocation sites. The only shared-fixture change is
a configurable name for its existing main so the new fixture can reuse exact
root/byte accounting without executing or altering the original corpus.
