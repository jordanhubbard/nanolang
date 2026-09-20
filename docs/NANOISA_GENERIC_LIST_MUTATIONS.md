# My checked generic-list mutation prerequisite

I record a design checkpoint for MAC
`task_7b805000dfda4da386b55d4691e8c647`, based on canonical PR922
`4b2422ffbee8ed3a621846b868ed9982407f3457`. No implementation or execution is
part of this checkpoint. My full verifier corpus currently verifies174 of176
programs on both hosts; `tests/token_value_bytes.nano` stops while compiling
its shadows at `list_LexerToken_insert`. I retain that source and every existing
insert/set/remove/pop and64-bit `value_bytes` assertion unchanged.

## Existing contracts I must reconcile

My C runtime implements insert by growing once and shifting elements right;
remove returns the removed element after shifting left; pop returns the last
element. Index errors and empty pop terminate before mutation. My canonical
producer currently lowers only generic-list new/push/get/length/set.

My checker treats generic remove as void, despite its C return type. My proposed
correction makes remove element-returning, like get/pop, with the exact record
or enum identity. A statement discards that result using the existing expression
statement rule. This choice requires review; I will not silently infer a return
contract from the C spelling alone. I retain the original statement-form remove
and add a separate returned-element control. The checker must retain nominal
return metadata for remove and pop as well as get.

Raw `ARR_GET` returns VOID for a missing index; raw empty `ARR_POP` returns VOID.
`ARR_REMOVE` returns the array, not the removed element. Those raw contracts do
not supply my checked-list bounds or return semantics and will remain unchanged.

## Proposed producer boundary

I add no opcode or profile. I recognize only an existing checked list operation,
its exact arity and declared element identity. A lexical function value or real
function declaration of the same spelling retains precedence. I do not use a
`list_` prefix as permission to accept an unknown type, arbitrary operation or
resource-containing element graph. Existing complete module/backend authorities
remain responsible for their admitted shapes; this work does not admit affine
owners, references, services or a new aggregate target.

For these new operations I validate receiver identity, an INT index, exact
inserted element compatibility, and the result type. I use current checked
ordinary record/enum list identities, including type names containing underscores.
I preserve the existing scalar list declarations. Known mismatches, bad arities,
unknown element names and owner-containing shapes refuse without publishing a
new output. This does not silently relax unrelated list operations.

| Operation | Arguments | Result | Required bounds before mutation |
| --- | --- | --- | --- |
| insert | list<T>, INT index, T value | VOID | 0 <= index <= original length |
| remove | list<T>, INT index | exact T | 0 <= index < original length |
| pop | list<T> | exact T | original length > 0 |

I evaluate arguments once, left to right, and stage all of them before inspecting
length or mutating storage. The staged list reference identifies the original
receiver even if a later argument reassigns its source binding. The length is
read after argument evaluation, so side effects on that same list are observed.

I propose explicit boolean bounds followed by the existing ASSERT instruction.
Invalid list operations therefore terminate in the canonical assertion category
before any element or length changes. Native C's existing bounds helper still
terminates with its index diagnostic. I require equal refusal and no publication,
not identical error text or VM error numbers across those two routes. This
choice, especially ASSERT's ordinary host boundary, is reviewed before code.
Raw array error conventions do not change.

Insert saves the original length, appends the staged value once, then shifts
existing elements right from the former last index toward the insertion index,
and finally replaces the insertion slot. Descending order preserves every old
value; the temporary appended value keeps the new value rooted. Inserting at
length is just the append. The only runtime allocation is the existing append
capacity growth; failure occurs before any successful shift. Element moves use
existing GET/SET retain/release behavior and do not allocate another collection.

Remove retains the selected element before ARR_REMOVE, discards the returned
array reference, and publishes the retained element. Pop uses ARR_POP only after
the nonempty guard. Both preserve exact element tags and nominal source metadata.
I release synthetic local roots on successful completion after preserving the
result on the operand stack. Existing frame unwind owns them on failure. Local
slot counts and generated branch offsets remain checked; no stale temporary may
extend a successful operation's lifetime beyond its expression.

## Checkpoint and qualification order

1. Review this source contract, especially remove's result and bounds category.
2. Implement checker/producer changes only, with exact typed scratch layout and
   all error/unwind paths, then obtain independent source review.
3. Add meaningful ordinary-record and enum controls plus original unchanged
   token-value-byte source. Cover head/middle/tail/empty insertion; growth; retained
   aliases; remove/pop returned records and field observations above32 bits;
   discarded results; argument order and once-only effects; source binding
   reassignment; underscore type names; shadowed operation names; and mismatches.
4. Cover negative/length/INT64 extrema bounds and empty pop before mutation, actual
   append allocation failure with roots retained and fresh recovery, generated
   instruction/stack validation, direct VM and strict native behavior. Helpers
   added in Nano source carry meaningful shadows. No test is removed or weakened.
5. Qualify fresh C-seed and applicable Stage1/Stage2 producer paths, the unchanged
   token fixture, ordinary list/record neighbors, and the full verifier corpus.
   Self-hosted producer support is a separately measured requirement, not inferred
   from C-seed success. Preserve all first terminals, copied output identities and
   explicit compiler/provider scope on Linux and Darwin.

This prerequisite does not close the U8 conversion task or full5.1 release.

## My first production checkpoint

I implement the reviewed C checker and canonical NanoISA producer boundary in
`src/typechecker.c` and `src/nanovirt/codegen.c`. I have not built or executed it.
The legacy parser retains `List<T>` in declaration names rather than a complete
`TypeInfo`; I preserve that nominal evidence in local/global symbols and both
function-parameter registration paths. I do not change the serialized AST or
infer nominal identity from the runtime array tag. Missing identity refuses.

My new checker path covers insert/remove/pop only, validates the exact declared
record/enum and receiver identity, and retains remove/pop nominal results. Real
functions and lexical callables keep their existing precedence. My generated
loop has fixed instruction size; checked scratch capacity precedes argument
lowering. Insert stages ARRAY, INT, element and INT length; remove stages ARRAY,
INT and length; pop stages ARRAY and length. Every successful path clears those
locals after preserving any returned value on the stack. Every failing path
retains them for ordinary frame unwind. No runtime or authority changes occur.

I keep scalar list declarations on their existing declaration/extern route;
this checkpoint does not silently reinterpret a scalar extern as a new generic
operation. My existing native C call builder already stages arguments in order.
I still require the later fixture checkpoint and all producer/backend controls.

During this audit I also found a separate legacy evaluator limitation:
`src/eval.c` returns a raw INT from generic remove and distinguishes non-scalar
get/pop by spelling rather than the actual enum/record declaration. That route
is not measured by this producer checkpoint. I keep its ownership and nominal
result repair as an explicit follow-up under the list task, before claiming
complete evaluator parity; I do not execute the faulty value interpretation.

My static review also found a prerequisite before this draft may qualify:
legacy generic-list binding, assignment and call compatibility does not always
compare T. Declared receiver metadata alone cannot prove a mismatched initializer
or actual argument was rejected. I must extend the existing collection boundary
checks to List<T> bindings, assignments, record fields, direct/indirect arguments
and returns, preserving exact signature metadata and refusing missing identity.
I must inspect all such paths, including both module passes and inferred lets,
before treating this draft as an accepted-flow implementation. No qualification
is authorized from this draft, and no test expectation is weakened.
