# My shared mutable lexical captures

I track this work in task_af8091f571a842bc90656e2c7f19b68e, within the full
computed-byte and NanoISA-only 5.1 requirements. My retained70516 source gate
observes a closure update returning2 while its outer byte binding remains1;
the unrelated global remains900. I preserve that assertion and both platform
terminals. This document specifies the repair before implementation.

## Existing mechanism and required meaning

My C NanoISA producer loads parent local/upvalue values before CLOSURE_NEW.
The VM transfers those values into a closure capture array. STORE_UPVALUE
replaces one capture slot. That mechanism retains values but does not establish
shared mutable binding identity. A global-name fallback cannot repair it.

Each mutable lexical binding must have one identity for its dynamic lifetime.
Outer code and every capturing sibling or descendant observe the same updates.
A distinct declaration, recursive activation or new loop-body declaration gets
a distinct identity even when it reuses a physical local slot or source name.
An escaping closure retains its binding after the defining activation exits.
Checked byte conversion occurs once at the value's declared destination; the
cell mechanism must not add a second source evaluation or change that type.

## Owned cells and binding operations

I propose an owned heap cell containing one ordinary language value. It never
points into a VM stack, native automatic variable or borrowed activation.
Capturing a mutable local lazily establishes that cell; later captures reuse
it. Ordinary reads and assignments access its value, while a new declaration
replaces the local binding rather than mutating a cell retained by an earlier
closure. Lexical cleanup drops the local reference without overwriting the
escaped value. Immutable captures retain their existing value semantics.
Capture mode follows the declaring binding's mutability, not whether a
particular closure writes it: a read-only sibling must still share a mutable
binding's cell. I preserve existing affine/resource capture refusals; an owned
cell does not make a resource owner an ordinary shareable value.

I need explicit internal operations for capturing a local binding, forwarding
an existing captured binding, initializing a new binding and clearing a local
reference. These operations must remain distinct from ordinary assignment.
Before changing production, I will specify their exact wire encodings, stack
effects, capture-mode metadata, limits and compatibility/refusal behavior in
the shared ISA schema. No opcode number or serialized layout is assigned by
this preliminary design. Old modules retain their existing interpretation.

The verifier must keep internal cell references separate from ordinary source
values. A capture transfer must match the exact target function's declared
capture mode and slot; arbitrary arrays, foreign calls or unrelated operations
must not consume internal cell references. Unsupported backends must refuse
before producing runnable output until their complete lowering is qualified.
Refusal is an implementation-stage boundary, not final 5.1 acceptance.
Direct, indirect, tail and foreign-callback entry paths must all check the
target's required capture shape. A raw function entry cannot fabricate the
environment expected by a capturing function.

## Lifetime, failure and execution audit

I will enumerate every local initialization, assignment and cleanup emitter,
including parameters, loop variables, match bindings, anonymous staging,
nested functions and effect-handler exits. An assignment must not accidentally
rebind a cell, and cleanup must not assign VOID into an escaped binding.

Cells and their values participate in retain/release, root traversal, cycle
collection and heap accounting. Closure-to-cell-to-closure cycles require
actual reclamation checks. Allocation failure must not publish a partial cell,
lose the previous local value or mutate a prior closure. Return, tail call,
trap, handler unwind, cancellation and destruction retain their existing
ownership obligations. I require an explicit allocation/lifetime table and
independent source review before executing changed heap code.
That table must state parameter-value copying and tail-argument retains, plus
the exact multi-capture staging/rollback sequence when an allocation fails
after earlier captures were prepared. Handler continuation and retry must
preserve the prior binding values and existing alias identities. A successful
single-cell allocation alone does not establish whole-closure failure behavior.

Both source producers and C/LLVM/Wasm consumers must implement the same
identity and conversion rules without embedding an opcode interpreter in AOT
products. Native/source paired behavior and existing copied-capture modules
remain separate regression obligations. I do not call a VM-only fix complete.

## Ordered acceptance

1. I finalize the versioned wire/metadata contract, verifier transfer rules,
   binding classification and allocation/lifetime audit before production code.
2. I implement and review shared schema/verifier/heap/VM support, preserving
   original operation decoding and explicit unsupported-consumer refusal.
3. I lower exact lexical bindings in both producers and all required generated
   backends, including reentrant activations and per-iteration declarations.
4. I retain the original failing byte assertion and test outer writes, sibling
   and nested aliases, escaped closures, shadowed names, globals, repeated loop
   declarations, recursion, managed payload replacement and cyclic reclamation.
5. I qualify failure boundaries and unchanged neighbors, both VM dispatch modes,
   native optimization configurations and all required hosts/backends. I then
   resume the entire computed-byte source gate and compiler bootstrap gates.

My [wire and ownership contract](NANOISA_CAPTURE_BINDING_CONTRACT.md) proposes
atomic descriptor-based environment construction, explicit binding init/clear,
required container metadata and an allocation/rollback table. It refines the
separate capture/forward operation sketch above without exposing cells on the
ordinary operand stack. Independent review precedes schema and heap changes.

This is a design checkpoint. No shared-cell production code or qualification
is claimed here, and the full task remains open.
