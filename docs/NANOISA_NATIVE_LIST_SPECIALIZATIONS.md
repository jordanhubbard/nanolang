# My complete native ordinary-list specializations

I record this design under task_2752a051dce443d0ada447c46b667561 before changing
native production. The strict checker checkpoint 0e72a33a6 does not complete
native lists. My source audit finds these independent existing gaps:

* In src/transpiler.c, generate_list_implementations emits only new/push/get/set/
  length. Its get is unchecked and set silently ignores an invalid index.
* In src_nano/transpiler.nano, generate_list_for_type emits only new/push/get/
  length. Its push publishes doubled capacity before realloc succeeds; get
  reads without checking bounds. Its discovery walks annotated local lets and
  selected statement blocks, not every accepted list-producing expression.
* nb_rewrite canonicalizes colliding record/List annotations, while native call
  emission still spells raw list_Item_* names. The C iterative emitter likewise
  treats a generic-list spelling as a runtime symbol before ordinary mapping.
* The selfhost generic call path emits ordinary C arguments without establishing
  left-to-right receiver/index/value ordering. The C-seed iterative emitter
  already uses build_ordered_call_args; I preserve that actual staged route.

I have not executed an invalid generated program to reproduce these findings.
The original imported/mutation corpus remains required. I do not start it just
to rediscover a missing operation already demonstrated by source inspection.
Fresh checker/bootstrap work at 0e72 is independent and stays attributed there.

## My shared semantic boundary

I implement the already checked ordinary-record catalog: new, push, get, set,
insert, remove, pop, length, capacity, is_empty, clear and free. Remove/pop/get
return the exact record by value. The existing int/string providers remain
separate and retain their additional with_capacity operation. I do not invent
record with_capacity, enum-list ABI, owning-resource support or new NanoISA
admission. The required enum parity task remains open.

My generated generic storage retains its existing data/count/capacity layout.
Record values retain the native record representation: nested record fields
copy by value; string and collection fields follow their existing native
reference representation. Clear/free retire list slots/storage, not arbitrary
pointees of those fields. This change does not establish a new deep-copy or
reference ownership ABI. I must inspect the actual native managed-string/root
and record publication paths before changing any element cleanup policy; I do
not infer evaluator arena behavior for native C.

I use the existing src/runtime/list_capacity.h checked capacity/length helpers
for both producers, with its real include dependency recorded. Before storage
mutation I check a nonnull receiver, representable count/capacity, and the full
source-width INT index. Get/set/remove require 0 <= index < count; insert permits
index == count; pop requires count > 0. I compare before narrowing to the
existing int storage width. For scalar provider calls I likewise validate an
index or with_capacity argument at source width before its existing int ABI
narrows it; I do not silently change the provider ABI. New and growth must check
size multiplication and
allocation before committing a pointer, capacity or count. A failed constructor
frees its partial storage. A failed growth retains every previous byte and
metadata field. Shifts use the established record representation and checked
sizes; remove stages its result before shifting; pop stages before reducing
length; vacated slots are cleared without inventing a destructor.

I propose the existing checked list runtime's nonzero fail-fast policy for
bounds, invalid storage and OOM: a precise diagnostic and exit(1), before any
successful mutation. This deliberately replaces generic inline silent failure
or NULL-on-OOM behavior; source review must approve that observable failure
contract. It is not recoverable allocation acceptance. The evaluator's checked
clone/arena allocation sweeps remain separate and unchanged. Free(NULL) retains
its existing harmless cleanup behavior; no other operation accepts NULL.

## My exact call and identity boundary

I distinguish lexical callable values, declared functions and externs before
selecting a list intrinsic. A spelling alone never overrides any of them.
Qualified calls use the actual bound declaration, never a bare suffix fallback.
For a selected intrinsic I resolve its encoded element using the call owner's
nominal binding and one actual record declaration. I preserve the established
unique extern-record namespace, reject competing definitions and retain the
same resource/unknown refusals as the checked source contract.

The generated specialization key is the exact canonical declaration, not a
raw spelling, receiver layout coincidence or current global name. For the
self-hosted producer I use the existing #type bindings/nb_type result only after
actual declaration validation. For the C producer I audit existing checked
NominalIdentity and registered generic-instantiation origin facts before choosing
its corresponding canonical emitter key. I do not repurpose callable generic
specialization metadata or return-only annotations as unchecked list authority.
Any new retained per-call metadata needs its constructor/copy/destructor audit
in that source checkpoint. This C metadata choice is an explicit prerequisite,
not an already implemented or settled ABI.

I retain C-seed build_ordered_call_args and give the selfhost selected list path
the same ordering contract. I stage the selected callee/receiver, index and value
once, left to right, in
collision-free expression-local temporaries. The runtime observes length after
all argument effects. A later argument may mutate the same list, and the staged
receiver still denotes the original object if a variable is rebound. Captured
record results remain valid by their established representation after movement.
I preserve VOID versus record/scalar expression results and existing enclosing
control flow. Declaration/callback routes retain their own signatures and do
not accidentally call a generated intrinsic helper.

I discover every accepted specialization from actual declarations and checked
expressions, including parameters/returns/fields, inferred locals, nested block/
branch/loop/call contexts and discarded constructors. I emit forward list types
before record fields that refer to them, then implementations after complete
record definitions. Exact keys deduplicate definitions. Runtime-provided schema
lists are identified by their actual known declaration/provider identity, not
an arbitrary AST or Compiler name prefix. Unsupported names, unrepresentable
identifiers or specialization collisions produce a checked failure before a
new output is published. I do not silently truncate a discovery list.

## My review and acceptance sequence

1. Finish the C identity/metadata and native element-lifetime audit above; submit
   the concrete selected representation before implementation if it adds fields.
2. Implement complete helper discovery/emission and exact call selection for
   both producers, with meaningful Nano shadows and all ownership/error paths.
   Submit the whole production checkpoint before builds or new execution.
3. Add focused generated-C controls for every operation, exact returned records,
   underscore names, colliding imported owners/aliases, schema-looking ordinary
   record names, declarations/callbacks, inferred/discarded constructors and
   nested record/List fields. Retain all original eighteen source methods.
4. Test corrected-only bounds and allocation failures in bounded subprocesses,
   including INT64 extrema before narrowing, zero/first/growth storage and
   unchanged prior state under injected allocation failure. Prove staged order
   with the existing trace and receiver-rebinding controls, not C argument order.
5. Run fresh C-seed/Stage1/Stage2 native outputs, evaluator and already supported
   NanoISA routes with exact provider maps on both hosts. The complete source
   matrix, original LexerToken program and unchanged make test remain required.
   Preserve every first terminal; no deadline or assertion is weakened.

I keep full list, enum parity, bootstrap/fixed-point, whole-Make and release
criteria open. This design is not source qualification or backend admission.
