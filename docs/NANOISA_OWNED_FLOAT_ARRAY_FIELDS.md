# I retain mutable FLOAT arrays inside unique owners

I track `task_430220ce190946518d404088533531b6` under my managed lifetime
`task_51da49b39230468784da3481b893563b`, aggregate identity
`task_488a05eb5e2a417caf83a8353363a30d`, affine equivalence
`task_28f2fb4b1f3c8a5ce93df628bb569d76` and installed-product
`task_e8d860a16da0464891dd32e91c42bef1` parents. This is a contract, not
implementation or admission evidence. I start from canonical `bfdde227`.

## My unchanged acceptance and prerequisites

I preserve `test_ordinary_array_field_keeps_element_type` in
`tests/test_owned_record_patterns.py`, including its complete PREFIX and selected
shadows. Main constructs `Bundle { file: Handle { fd: 7 }, samples: [1.5, 2.5] }`,
destructures `samples` before `file`, checks element zero against 1.5 and consumes
the Handle through close. I retain the inline constructor and original close
and main shadows. STRING fields and ordinary Samples do not satisfy this case.

I depend on qualified mixed ordinary FLOAT arrays (`task_4be28fef163f42069064357639b3b5cc`)
and retained STRING owner lifecycle (`task_793523b9e4c04e389f9d63652758991a`).
The current private mixed view, shape proof and composed analysis grant no
execution permission. I do not implement dependent admission until their
runtime/native/source prerequisites qualify. The existing inline staging,
nested exact owner results and FLOAT-local behavior remain intact.

## My exact identities and proof obligations

The Bundle shell and Handle child remain unique owners. The FLOAT array is a
shared mutable managed value, including when several owner fields or an ordinary
local retain the same array. Consuming an owner does not consume independently
retained array aliases. I keep source struct ordinals, global nominal layouts,
compact managed layouts and allocation origins distinct.

TAG_ARRAY with no nominal layout index does not describe an element type. I
preserve current wire meanings and construct closed FLOAT origin facts from the
immutable module. I do not clear RESOURCE, invent VOID descriptors, filter out
owner bodies, or accept a caller-supplied proof as runtime authority. Any shared
descriptor widening requires an audit of every validator, selector, converter,
borrowed-root query and VM/native consumer before production.

I extend provenance through owner packing, retained field observation,
destructive unpack, moves, exact owner calls and returns. A field fact identifies
every possible array origin after control-flow joins. Mutable writes through any
alias must preserve exact FLOAT elements. Unknown origins, incompatible element
tags, arrays of owners, nested arrays and record arrays remain outside this first
slice. The all-body check includes unused helpers and every selected shadow.

I keep the current finite acyclic call/layout bounds. The production proposal
must state additional storage/worklist bounds before code. Calls and returned
owners require explicit finite field provenance summaries and instantiation;
the callee's nominal owner type alone cannot establish its managed field facts.
If a private checkpoint covers only local Bundle construction, I label it
partial. Calls, nested returned owners and their retained array aliases remain
required before this child closes. STRING-bearing siblings retain their own
qualified semantics; they cannot bypass array provenance.

I refuse borrowing a root containing a managed array, including a path ending
at its scalar sibling, until a separate reference contract qualifies that case.
The existing ordinary, closed, linked and LLVM/Wasm selectors keep their exact
qualified boundaries. Wider target acceptance remains in the parent roadmap.

## My operations and lifetimes

I reuse the qualified mixed-array carrier and checked operations. I do not make
an immutable STRING carrier stand in for a mutable array. Retains publish only
after overflow checks. Allocation, capacity growth and byte-size arithmetic are
checked before mutation; failed growth preserves the old array and its aliases.

Constructor expressions run once in source order and pack in declaration order.
Each staged array root has exactly one cleanup owner. Packing transfers the
staged root into the unique shell without duplicating or losing its retain.
Projection retains a managed array before dropping the temporary shell
observation. Unpack transfers each shell field into an independently rooted
value. Frame cleanup, assertion failure, failed calls and pending owner returns
release every remaining root once. A copied array alias survives both shell
consumption and reassignment of a different local.

ARR_NEW/LITERAL create FLOAT origins; ARR_PUSH/SET accept FLOAT only. Mutations
remain visible through all aliases, including aliases retained outside an owner.
ARR_LEN observes that same object. I preserve raw ARR_GET's FLOAT-or-VOID result
for absent, negative and large indices. Typed FLOAT uses retain explicit runtime
checks; I do not silently turn a missing element into zero or invent a bounds
proof. Generic equality and typed comparisons keep their existing distinct
semantics, including NaN and signed zero. Heap/owner comparison is not admitted
by this contract. I retain ARR_SET's checked out-of-bounds failure.

## My ordered acceptance

1. I review the complete private field-provenance and authority proposal before
   executable admission. I qualify conflicting joins, initialization, all-body
   checks, exact calls/results and failure-atomic proof publication. Public
   verification stays unchanged at that checkpoint.
2. I review runtime and native changes against the qualified mixed carrier and
   STRING cleanup implementation. Before paired source changes I qualify actual
   VM APIs and native execution, with exact unchanged public refusals for other
   profiles. No historical failed artifact is replayed.
3. I exercise multiple aliases before and after pack, observation, unpack, move,
   call and return. Writes and append growth through one alias are observed by
   another; empty/nonempty arrays, two fields sharing one array, nested owners,
   overwrites, assertions and repeated invocations retain exact cleanup.
4. I inject allocation failures at every newly reachable acquisition/growth/
   publication boundary. I require unchanged output sentinels, correct surviving
   aliases, no leaked roots and successful subsequent recovery. I test signed
   and extreme indices and exact FLOAT bit behavior with native sanitizers.
5. I review paired source lowering separately, then freeze fresh bootstrap tools
   and qualify the unchanged original Bundle/PREFIX plus all shadows through
   both canonical producers, VM/native execution and ordinary installed routes.
   I retain source evaluation order, inferred element types, false-shadow and
   output-preservation refusals. Linux and Darwin evidence stays explicit.
6. I integrate and seal source, actual tools, commands, statuses and artifacts.
   This child closes only its stated FLOAT-array owner slice. Complete managed
   targets, real service handles, whole-product and fixed-point acceptance and
   release publication remain governed by their existing parents.
