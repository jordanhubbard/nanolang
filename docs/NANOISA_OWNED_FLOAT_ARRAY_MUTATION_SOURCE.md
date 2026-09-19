# I mutate shared FLOAT arrays through owned source graphs

I make the required mutation extension of parent430220 explicit as
`task_bba6228369c04b6c900f628d501e5553`, after
[source child18731](NANOISA_OWNED_FLOAT_ARRAY_SOURCE.md), qualified private runtime
and separately reviewed public owner-ARRAY activation. This is a static contract
at5f988ed79, not production, fixture execution or admission evidence. I also
record the prerequisite builtin-identity task
`task_13d1c53953674d279a849306c9eee26d` below. Source18731 alone does not close my
parent's mutation requirement.

## I retain the actual operations and results

| Checked operation in this slice | Evaluation order | Canonical instructions | Result |
| --- | --- | --- | --- |
| `array_set(array<float>, int, float)` | receiver, index, replacement | `ARR_SET`, then `POP` | void, no operand left |
| `array_push(array<float>, float)` | receiver, appended value | `ARR_PUSH` | the same shared array identity |
| `array_length(array<float>)` | receiver once | `ARR_LEN` | exact INT length |

My existing ordinary C canonical lowering in `src/nanovirt/codegen.c` emits
those sequences. Selfhost `nisa_emit_set_call` and `nisa_emit_push_call` agree;
`nisa_emit_call_args` traverses source arguments in order. The interpreter's
argument loop is also ordered; `builtin_array_push` returns its original array
value after in-place append, and `builtin_array_set` returns void. I do not turn
append into a functional copy or expose the setter's internal returned receiver
as a source value. Existing builtin result/arity registration and exact lowerer
checks are distinct from full element checking; I do not claim the general
checker already enforces every rule below.

A receiver is a mutable ARRAY object. An immutable `let` handle does not prove
deep immutability; the existing purity analyses state this explicitly. I require
no new `let mut` flag merely to mutate shared array contents. This is distinct
from reassigning the binding with `set`: this slice retains the specialized
producer's existing refusal of managed-binding reassignment, including
`set values (array_push values 1.5)`. The returned append value may instead be
used directly, bound under a new name, passed as an exact constructor field or
discarded as a call statement. Those uses do not require rebinding the old name.

## I establish builtin identity before choosing mutation

C's ordinary canonical path checks declared functions and locals before builtin
emission. The specialized paired source route must likewise check lexical/formal
bindings and exact declared functions before matching `array_set`/`array_push`.
A supported declared call retains normal call checking/lowering; an unsupported
bound callable or noncallable refuses. Neither may fall through to mutation.
Initializers resolve against the preceding environment, then bind their new
name; scope exit restores the outer binding. Global/module-qualified/indirect
forms remain outside this value-graph slice.

My static checker audit finds a concrete prerequisite: selfhost
`check_expr_node`'s two-argument `array_push` inference returns an array result
before ordinary `env_lookup`. Its generic `check_builtin_function` result also
recognizes `array_set`/`array_push` before that lookup, without the existing
scoped guards used for `string_to_float`/`reduce`. `tc_bound_name` preserves a
lexical nonfunction symbol's spelling. Thus spelling-based result selection can
mask that lexical identity. Declaration owner/mangling paths require explicit
controls rather than an assumption that every declaration keeps the same name.
This is static path evidence; I have not executed a counterexample.

Task13d1 first audits meaningful declared/local/formal and initializer controls
against the checked source pipeline. If the guard correction is required, I
send a separate source checkpoint that routes only these bound mutation names
back to existing ordinary symbol checking, with mandatory shadows. No global
builtin-resolution rewrite or unknown-type fallback follows. C already reserves
`array_set` as a declared function name; I preserve that checked refusal instead
of promising a positive declaration that its frontend rejects. Nonreserved
`array_push` and lexical variables provide meaningful precedence controls.
Rejected syntax is not evidence of correct binding/type behavior.

## I prove exact FLOAT values without changing optional reads

Only the distinct owner-ARRAY value graph gains these calls. Samples/STRING and
owner-free routes keep their prior policies. Receivers must have an exact flat
FLOAT-array fact carried from an admitted literal, declared owner field,
complete unpack, rooted projection, alias or preceding append result. A bare
ARRAY descriptor or unknown alternative is insufficient. Every replacement or
append expression must have exact FLOAT type; the index must be exact INT.
Arity is exactly three or two, including calls whose result is discarded.

I preserve18731's contextual-empty boundary: an empty literal can acquire FLOAT
from its exact owner constructor field, then be projected/unpacked and appended.
This extension does not newly infer an untyped `[]` receiver from the first push,
even though broader existing language routes have that inference. Wrong-element,
nested/owner arrays and implicit integer-to-FLOAT conversion remain refused.

ARR_GET remains FLOAT|VOID. I cannot use `(at values index)` as a replacement or
append operand merely because the source element type is FLOAT: the final
origin/authority query must reject that unresolved optional typed use. The six
qualified optional comparisons remain separate; they are not a generic
FLOAT-extraction primitive. I add no FLOAT-returning helper signature, cast or
bounds/default semantics to manufacture a replacement value.

## I preserve ordered roots and shared identity

Each call evaluates every argument once, left to right, before the operation.
The receiver is already a rooted ordinary ARRAY value when the index or value
expression runs; it is not reloaded later from a potentially moved owner or
changed binding. Completing an owner field projection finishes its temporary
observation and retains the ARRAY independently before a later argument can
consume that owner. A nested helper call must retain all previously evaluated
caller operands through frame allocation, owned transfers and failure cleanup.

ARR_SET mutates the shared object, then the explicit POP releases only the
operation's receiver root. ARR_PUSH mutates and returns that same root; a call
statement drops it through the existing nonvoid POP, while a binding/constructor
keeps its own root. No array operation changes unique shell liveness or releases
an unrelated alias. Multiple fields/shells, outside aliases and returned owners
continue to observe identical contents and length after growth.

Complete owner destructuring transfers fields once into hidden exact slots.
Named ARRAY bindings retain aliases to those slots; owner children still move.
Mutation before/after pack, observation, reverse unpack, owner move, helper
consumption and nested owner return must preserve this distinction. I retain
explicit child consumption, live-owner joins and source lifetime checks. No
implicit owner disposal or borrowing of managed roots follows.

Bounds and allocation follow the qualified runtime: ARR_SET rejects negative
or out-of-range indices without replacing content. ARR_PUSH checks growth before
publishing changed length/storage and preserves all aliases on failure. Both
backends release receiver roots, staged earlier arguments and nested owners on
terminal failure. Earlier effects from evaluating arguments remain observable;
failed mutation does not roll back already completed argument effects. Error
status and exact output prefixes remain part of qualification, not an assumption
that a late failure means nothing happened.

## I require an ordered acceptance matrix

1. I qualify task13d1's checked identity boundary before dependent mutation:
   meaningful declared/local/formal names, initializer-before-binding, nested
   scopes, exact arity and prior artifact preservation. I retain C's reserved
   declaration policy and distinguish stage-specific frontend constraints.
2. After runtime public alignment and18731, I review the paired small mutation
   delta before execution: builtin guards, exact facts, ordered expressions,
   setter POP/void return and append ARRAY result. New helpers have meaningful
   mandatory shadows. This does not touch native runtime or public selection.
3. Fresh source controls use an owner containing two ARRAY fields and a nested
   owner, with the same array retained outside and through a returned owner.
   They set through one alias, append through another and inspect both original
   and appended indices through every surviving alias before/after reverse
   unpack, call/return and explicit shell consumption. Append result aliases
   must observe later writes; call-statement append must not leave stack values.
   Empty contextual arrays grow, and repeated append crosses storage capacity.
4. Source-order controls use nested `array_push` as a receiver and a consuming
   INT-returning index helper, with checked mutations observable through a
   retained alias. This distinguishes receiver-before-index and once-only
   evaluation without adding globals, callbacks or FLOAT helper results.
   Constructor fields containing append expressions preserve source order before
   declaration-order pack. Existing strict slot/function budgets stay checked.
5. Refusals cover wrong arity/receiver/index/value, optional FLOAT replacement,
   unsupported empty inference, shadowed builtin misuse, owner field assignment,
   copied/partially moved owner, unconsumed child, mixed ordinary-record profile
   intersection and unsupported borrowed roots. A setter cannot initialize a
   FLOAT/ARRAY local or become an owner field. False selected shadows fail
   publication; every rejected compile preserves its previous artifact.
6. I run unchanged Bundle/PREFIX and every original shadow alongside these
   controls through Cseed/Stage1/Stage2/NanoVirt and both canonical emitters,
   requiring exact byte/dump parity and VM/native behavior. Linux/Darwin, Samples,
   STRING, borrowed and owner-free adjacency remain explicit. Valid dynamic
   bounds/failure modules execute only after complete public admission review.
7. Runtime fault controls exercise failed append growth, failed owner shell
   packing after a successful append, later consuming-call preflight failures
   with a prepared receiver, exact prior effects, root accounting before terminal
   disposal and subsequent recovery. I distinguish actual injected allocation
   sites from static cleanup review. I seal first terminals, sources/tools/SDKs
   and results; no failed artifact is replayed.

This explicitly supplies parent430220's source mutation requirement. The parent
still closes only after its complete runtime/public/source acceptance, including
this extension. Full managed/ownership targets, product/fixed-point gates and
release publication remain independently open.

## My independent builtin-identity checkpoint

Task13d1 now has a narrow production proposal on canonical2456abd64 plus the
reviewed contracts. `env_lookup` searches backwards, so the last lexical/formal
symbol is authoritative. `tc_bound_name` preserves a nonfunction lexical name
and otherwise resolves its declaration owner. I guard the direct array_push
result special case with absence of that resolved symbol, and suppress only
array_set/array_push builtin result selection when an exact symbol exists.
Qualified lookup uses its exact resolved qualified identity, matching the
existing string_to_float guard; I do not match an unrelated short-name symbol.

All bound cases continue into existing ordinary symbol checking. I do not change
its general noncallable, qualified argument or callback rules, and do not claim
this guard repairs unrelated limitations there. No C checker policy changes:
array_set declarations remain reserved. The modified check_expr_node shadow
checks both unbound mutation result types, declared INT results, a later lexical
callable BOOL result, restoration of the outer declared result, ordinary arity
diagnostics and exact qualified declaration results. Its synthetic Symbol rows
exercise checker resolution; they do not claim C accepts a reserved declaration.

No qualification has run at this checkpoint. After independent production
review I will freeze fresh bootstrap tools and qualify actual declared/local/
formal/initializer source cases and expected refusal phases with output
sentinels. Unbound typed mutation controls and conversion/callback checking stay
adjacent. This independent checker correction grants no owner-ARRAY mutation,
source profile or public runtime admission;18731/bba622 remain held.

## I measure mutation through a bounded length operation

At canonical2cb8728a3, neither specialized paired producer has an ARR_LEN branch.
I add this explicit requirement to mutation task_bba6228369c04b6c900f628d501e5553
before implementation. Ordinary C canonical lowering already maps the one-argument
builtin to ARR_LEN; selfhost ordinary canonical lowering does the same. Those
existing paths do not authorize a specialized owner-ARRAY call by spelling alone.

In this extension only, an unbound direct array_length takes exactly one proved
array<float> receiver, evaluates it exactly once and emits ARR_LEN, returning INT.
I retain receiver alias identity and current length; I do not copy, mutate or
consume its containing owner. The instruction consumes its temporary stack root;
other local, constructor, unpacked-field and owner-field aliases stay rooted under
the already qualified runtime. Empty contextual arrays report zero, and length
through every surviving alias observes later successful append growth. A nested
receiver such as `(array_length (array_push values 2.0))` must append exactly once.

Lexical/formal and exact declaration lookup precede this builtin branch. A bound
name follows an already supported declared-call route or receives the existing
checked refusal; it never silently becomes ARR_LEN. I preserve initializer-before-
binding and nested scope restoration. C reserves array_length declarations; I
retain that frontend refusal. I do not extend13d1's completed push/set checker
claim to length or promise that all existing selfhost noncallable/qualified call
checking is repaired. No global builtin result dispatch rewrite follows.

Mandatory helper shadows and paired source controls cover exact arity, exact FLOAT
origin, empty/nonempty length, aliases across owner pack/unpack and call/return,
length before/after capacity growth and once-only nested receiver evaluation.
Wrong arity, scalar/non-FLOAT/unknown receiver, bare untyped empty literal, bound
unsupported callable and owner-as-receiver controls preserve prior output and
must fail in the intended checked phase. Bare ARRAY parameters/results, managed
binding reassignment, other collection builtins and ordinary-plus-owner-ARRAY
profile composition remain excluded. Initial source18731 does not depend on this
read operation; the required mutation/growth acceptance does.

The [held lowering checkpoint plan](NANOISA_OWNED_ARRAY_SOURCE_CHECKPOINT_PLAN.md)
separates preparation, independent source review and execution after qualified
public activation. No source or runtime production changes accompany this addendum.

## I prepare the dependent paired lowering checkpoint

Actual PR846 merge3660228d qualifies source18731 and fixtured27a; their live MAC
states are COMPLETED after verified merge reconciliation. I start this mutation
checkpoint in a fresh canonical tree, preserving both qualified1c4e trees. The
approved runtime supplement5652 is separate and does not authorize source tests
before this complete paired delta receives review.

I select mutation/length only in the existing owner-array profile, after exact
local and declared-function identity checks. I validate arity before evaluating
any argument; then emit receiver, optional setter index and exact FLOAT value
in source order. ARR_SET followed by POP returns VOID, ARR_PUSH returns ARRAY,
and ARR_LEN returns INT. No new allocation, runtime, descriptor, optional-value,
ordinary Samples, signature or managed-binding-assignment rule changes. I add
meaningful arity and whole-source lowering shadows before qualification.

## I freeze the source fixture proposal before execution

After root review of production1de22, I prepare six mutation methods plus the
six unchanged owner-array source groups. I retain PREFIX and every original
shadow. Positive programs observe shared fields/outside aliases/append results,
capacity growth, contextual-empty independence, returned owners and reverse
unpack. A nested receiver append is observed inside a consuming INT index helper,
proving receiver-before-helper and once-only evaluation. A constructor's first
append changes the length observed by its later owner factory before a second
append; declaration-order pack cannot replace source evaluation order.

I check builtin-name initializers against prior scope, inner-scope restoration,
a declared array_push owner helper (no ARR_PUSH), exact length and call-statement
results. Negative fixtures preserve prior outputs for arity/receiver/index/value,
optional writes at final assembly, setter VOID locals/owner field, lexical misuse,
managed rebinding and owner field assignment. Dynamic bounds and a consuming
index helper failure retain the prepared receiver and use verified raw modules
for VM/native cleanup controls. False selected shadows remain publication errors.
No fixture, new source module or bootstrap has run at this preparation checkpoint.
