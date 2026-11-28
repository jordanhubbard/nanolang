# My representation control-flow joins

I replace the complete private `sim_join` classifier operation in NanoLang after
my representation facts, shapes and operand-stack helpers. I do not replace a
subset selected for an easier comparison. My reference is the actual unchanged
helper in `src/nanoisa/nvm2c.c`, from `static int sim_join(` to the following
`static int jump_target(`. At source78eb700c06 the helper contains200 lines and
its SHA256 is `81432e73c0f4be78b2c104fcf1d5fdadefb9af604cd0dba5d85e74ffd6a019af`. I extract its verbatim body and exact dependencies
for ordinary and instrumented references. A minimal reference owner models the
fields read by these functions; it does not claim the production binary layout.

## My state and caller boundary

I reuse `representation_stack.Slot` and `representation_shapes.State`. My join
state retains whether a stack is established, its active height and complete
slots. My classifier state retains the ordered scalar-plan list, changed/final
flags, `has_maps`, the current graph and the first diagnostic. Plans retain
function identity, target and complete uint16 masks. I return all changes that
occur before success or refusal, including a newly allocated initial stack that
has not yet become established. I distinguish malformed wrapper input from a
valid product refusal.

The classifier supplies checked uint32 function/target/offset values, uint8
kinds and fields, uint16 masks and signed32 origins. Slot/field buffers must
exist for every operation that dereferences them in C. I check those transport
preconditions before invoking either implementation. They do not admit a
module, instruction, payload or executable. I keep uint32 frontiers, the
zero-height case, complete65535-field and65535-slot boundary reports, and
partial tails in the framing controls. I preserve the transport's explicit
size/count bounds without truncating a valid report.

My caller owns the active slot vector. Retained branch slots and published
field vectors have the same copy/borrow boundaries as the C classifier:
initial stack copying is shallow for immutable fields, and a field merge copies
before mutation. Returned graph operations retain the existing graph API's
mutation model. I do not claim allocator equivalence between NanoLang and C;
reference allocation failures remain separately observed C boundaries, and no
passing ordinary case establishes a NanoLang out-of-memory guarantee.

## My operation order

1. I find the first plan with the requested function and target. A mismatching
   plan height refuses before establishing or changing the retained stack.
2. On the first incoming edge I copy every active slot, apply existing nonzero
   planned tags in slot order, and create destination storage when shape
   tracking requires it. A nonfinal optional plan may be cleared only at the
   exact unknown-plus-finite discovery condition in C. Unknown provenance
   never becomes finite proof. Record and scalar destination storage does not
   rewrite an exact producer. I establish the join height/set flag only after
   all slots succeed; zero height establishes an empty join.
3. Later edges require the same height. For each slot I first intersect array
   payload identity, union masks and check proved finite provenance. I retain
   plan clearing, creation, head insertion, mask updates and `changed` timing.
   A final plan may change only for the exact unchanged VALUE-carrier exception.
   Scalar joins set VALUE storage and `has_maps` at the same point as C.
4. An unproved optional widening on a backward edge refuses. A permitted
   forward widening first establishes the OPTIONAL wrapper. I preserve every
   original scalar/optional/finite distinction and exact diagnostic spelling.
5. Record joins clone fields, merge them and constrain directed record returns
   before publishing that vector. Other shape joins convert for optional,
   scalar or unresolved carriers and unify otherwise. A shape refusal keeps
   the exact earlier slot/plan/graph mutations and first diagnostic.
6. I publish combined tags, reconcile origins and adopt an unknown carrier's
   incoming slot while preserving the existing destination shape, reconciled
   payload and origin. This whole-slot adoption also preserves the C helper's
   actual incoming-tag assignment; I do not silently normalize it.
7. I retain the later aggregate field-copy/merge pass in its original order,
   including its published partial vector on refusal. I do not conflate this
   pass with the earlier unpublished record-return vector.

## My differential qualification

I require both actual-C references and all five existing routes: C-seed native,
both C/NanoLang NanoISA producers in NanoVM, and both producers through
sanitizer-instrumented C AOT. A selected genuine qualified Stage2 supplies the
NanoLang producer. I retain its actual image/receipt/host closure and before/
after guards. A prepared root or a bootstrap receipt from the observer whose
shadow policy was refused supplies no such admission.

My permanent matrix retains all256 raw kind identities and all65536 masks,
then exercises first/later edges, plans present/absent/unrelated/height-mismatched,
nonfinal/final operation, mixed carrier kinds, optional/boxed finite masks and
unknown contamination. Named expectations separately cover discovery reset,
final instability, the unchanged-carrier exception, finite storage refusal,
backward widening, origin loss and payload loss. Merely matching two wrappers
is insufficient.

Tracked graph sequences cover fresh destination storage, conflicting kinds,
records and variant fields, unresolved producers, arrays/maps, cycles, shared
children and solve-after-join. Every report retains the complete graph nodes,
edges, tags, directed facts, error, slots, fields, plans and flags. Multi-slot
failures must identify what changed before the failing slot and what remained
untouched afterward. Input framing, declared lengths and count frontiers are
checked before allocation or semantic work. Deliberately invalid programs live
in separate packets so they do not prevent valid boundary cases from running.

I preserve original600-second compiler builds,60-second executions and ordinary
10-second dependency shadows. I retain the existing fact/shape/stack/solver
regressions and add a useful shadow for every new named NanoLang function.
I record actual commands, return codes, complete streams and generated products.
An independent reader must bind source/tool/image and qualified-producer guards,
replay every original method against captured data with products disabled, and
reject a changed manifest before component acceptance.

This document records my implementation contract. No join implementation or
executable qualification is accepted by this checkpoint. Full classification,
module/CFG/body/runtime/profile/ownership and host emission, paired platforms,
installed pure-translator cutover, both C repairs and every original5.1 release
gate remain required.

## My implementation checkpoint

I write the complete first/later-edge join operation in
`src_nano/nvm2c/representation_joins.nano`, with28 named functions and28
corresponding shadows. Review preserves the old partial slot buffer when an
unestablished join subsequently receives an empty edge; I retain that named
shadow alongside height/origin/plan/field/error controls.

The selected actual sourcef8 C seed accepts the module and dependency shadow
types through `--reflect`: exit0 in0.059131seconds, seven source hashes and the
compiler hash unchanged. I use the supported NANO_MODULE_PATH to resolve this
separate branch and request no native build. Reflection precedes module
compilation and shadow execution. This is syntax/type-check evidence only.
It does not execute those28 shadows or grant component qualification. I retain
the first reflection and the subsequent reviewed-source result under my volume
namespace `tmp/representation-joins-preflight-20261128/`.

The permanent whole-state C reference, packet driver, independent expectations
and five-route regression still need implementation and execution. All three
ordered roadmap items remain unchecked. I change no existing test, C source,
production selection or qualification deadline at this checkpoint.

## My permanent executable comparison setup

I add `test-nanolang-translator-representation-joins` to `test-units`. Its four
methods use26 verbatim C helpers, including my unchanged complete sim_join,
and both actual-C ordinary/sanitized references plus all five existing routes.
The new transport retains the original stack operations and full graph report,
then adds raw join inputs, scalar-plan metadata and graph operations. I preserve
both original stack drivers and all old tests. The reference tracks each
observer-owned branch buffer for eventual cleanup, including a failed first
edge, while leaving the helper body unchanged.

My matrix covers every256-by256 kind pair and every65536 mask at a later-edge
join, plus discovery/final/first-error plan combinations. Tracked graph cases
cover records, arrays, aliases, cyclic edges and deferred solving. I require
complete65535-slot and65535-field output and retain malformed packets separately
from valid frontiers. My named expectations assert actual error spelling and
partial state, independent of the implementations.

The first helper extraction refuses before compilation because my global end
marker incorrectly counts both jump_target declaration and definition. The
external static observer initially also lacks the explicit repository import
path. I retain both setup errors, correct those paths, and retain the subsequent
strict C refusal for an omitted optional_scalar_tags dependency. The corrected
26-helper reference passes C11/Wall/Wextra/Werror syntax checking; the complete
NanoLang driver passes module and dependency-shadow type checking. Neither
static command builds or executes an executable. Full executable qualification,
reference allocation-failure controls, retained evidence and independent review
still remain required before component acceptance.
