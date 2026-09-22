# I retain byte-array identity in native C lowering

I track this boundary under `task_046cd53fee0443aca7d87a26bfe4a9fd`.
My isolated source starts at root `3a7665809`; I do not modify the root or
native-list worktrees. My original 17-method computed-U8 fixture stays intact.
This is a source contract, not a qualification result.

## I preserve the actual first evidence

The retained producer-0 input has `ARR_LITERAL 1 1` in `receiver` at line41,
not a byte-tagged literal. Its generated C builds `narr_lit` and carries array
kind3. A later explicitly converted byte operand has runtime value kind2;
`nvalue_require_int` correctly refuses it. The failed output is not rerun.
The source fixture declares `let a: array<u8> = [1]`. Therefore I must retain
the checked destination through the actual C-bytecode literal producer as
well as implement native byte-array storage. A new backend kind alone cannot
repair this existing integer-tagged input. I do not infer byte identity from
a later inserted value or a shared pointer representation.

## I keep exact element identity separate from physical storage

I add private `NVM2C_VK_UARR` alongside ARR/BARR/FARR. It shares the existing
checked word-array allocation, roots and cleanup, but retains its own kind in
locals, parameters, results, records, globals and control-flow joins. I add
an exact `NVM_SHAPE_U8` array-element leaf. Array<INT> and Array<U8> do not
unify or convert merely because their C pointer/storage layout agrees.

The actual opcode tag establishes constructor identity: ARR_NEW U8 and
ARR_LITERAL U8 create UARR, including empty arrays. Untyped legacy INT-array
construction keeps its existing inference policy; it cannot claim byte
identity without a byte constructor or an exact propagated fact. Reads carry
VOID|U8 and preserve the existing out-of-range result. Set and push retain the
receiver's exact element identity and return it, with unchanged bounds,
allocation, evaluation order and once-only operand behavior.

I audit all classifier and emitter sites together: constructor classification
and module-needs discovery; local/call/result fixed points; joins and record
fields; shape resolution; stack storage/prefix/type selection; native roots;
ARR_LEN/GET/SET/PUSH; global boxing; tagged array helper dispatch; and runtime
helper admission. No path may fall through to ARR or string-array storage for
an unrecognized byte kind.

## I preserve scalar byte proof through the graph

Scalar U8 currently uses VALUE storage and a tag mask. `shape_carrier_box`
currently maps the accepted INT|U8 mask to NUMERIC, whose documented graph
members are INT|FLOAT. I do not widen NUMERIC or exact INT constraints.
I distinguish exact U8 payloads and an explicit private INT|U8 payload set
from the existing INT|FLOAT set. Exact byte pushes/casts, declared byte
parameters/results and optional array reads seed their actual payload shape;
copies, calls and joins preserve their masks and use the corresponding graph
constraints. The finite mixed set does not prove an exact U8 array element.
The runtime byte extractor accepts only tagged U8, with its byte-range
invariant, rather than accepting tagged INT or weakening nvalue_require_int.
Conversions remain actual CAST_U8 operations at checked source destinations.

I keep existing enum behavior and finite scalar/INT-array variant admission
separate. Merely adding UARR to shared word storage must not admit it into the
existing VARIANT_INT_ARRAY set. Any additional unsupported aggregate route
must refuse explicitly and remain a recorded full-scope dependency.

## I repair the producer at its checked destination boundary

Before implementation I trace the checked array-literal annotation through
C-bytecode let and return checking, including any repeated inference that
replaces the element type. I preserve an authoritative checked U8 destination,
compile every literal member once with the existing expected-tag conversion,
and emit the actual U8 constructor tag. I retain direct literal range refusal
and existing array<int>-to-array<u8> refusal. The backend does not repair a
mistagged producer by guessing from a function's generic ARRAY result tag.
Root owns the shared source scalar policy; I coordinate this producer delta
before editing overlapping checker or codegen functions.

## I require additive controls and the original corpus

I retain the original source fixture byte-for-byte. Added opcode-level cases
cover empty/nonempty constructors, exact byte read tags, push/set, parameter
and result propagation, record/global carriers, same-kind joins and explicit
INT/U8 conversion before a write. Shape controls reject exact INT/U8 array
unification and keep the existing numeric/variant constraints. Refusal cases
cover wrong tags and mismatched array joins without executing rejected output.
Runtime fail-fast controls use the existing supervised fixture policy only
after source review; they cannot pass on sanitizer diagnostics.

After source review I run corrected-only focused shape/generated-C controls,
then the unchanged actual producer/backend O0/O2 matrix under its original
bounds. I preserve the first failed artifact, command/source/tool identities,
and full original 17-method acceptance. This contract does not claim LLVM or
Wasm parity from a native C result.
