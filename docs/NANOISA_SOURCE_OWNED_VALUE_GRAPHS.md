# My owned value-call graph source contract

I track `task_46122ab40a234b6787d0de2927fd5bf2`. I follow the unchanged affine-example blocker
`task_c4351c720aee424ea9b90187e51a08f2`, runtime graph PR725 and exact
owned/void result PR738 (canonical 460db419). I incorporate the still-open
multiple consuming parameter acceptance of
`task_d54033e921f24e279e53e8ed4cf50d17`; its earlier entry0/helper1 contract
remains historical evidence, not the new graph limit. This contract precedes
source implementation. String/PRINT runtime work belongs to the peer's separate
prerequisite; I neither duplicate it nor claim the unchanged example passes yet.

## My static source boundary

At 460db419, `src/nanovirt/codegen.c:codegen_compile_internal` and
`src_nano/compiler/nanoisa_borrows.nano:nanoisa_has_borrows` route declared
resource or borrowed parameters into specialized lowering. Resource results
alone do not select it. `borrow_codegen.inc:codegen_borrow_compile` and
`nanoisa_borrow_emit` require exactly main plus one helper. Their expression
emitters allow that call only outside the helper, use CALL index1 and one named
owner argument for consuming mode. Function construction and ownership records
hardcode scalar one-result signatures and two functions. This source contract
must change those facts together; merely relaxing the function count is not
sufficient. I inspect source only, without replaying the failed product artifact.

## My bounded graph and identity

I admit a standalone, closed, acyclic graph of at most eight emitted functions,
including entry0; every direct target is a resolved declaration identity.
Production main is entry0, has no parameters and returns int. In shadow mode
one fresh synthetic entry0 executes every selected shadow in original order;
ordinary main is a separate callable function. I preserve the supplied
first-shadow boundary and lexical scopes. The synthetic name cannot collide
with any source declaration. Source order supplies deterministic indices for
all remaining declarations. Thus a shadow module admits at most seven ordinary
functions. I validate the complete emitted graph, including unreachable code,
and refuse cycles, calls to production entry0, absent targets and excess counts.
I do not use inlining or omission to fit these limits.

The first slice retains existing no-import/global/extern/callback/indirect-call
restrictions. Local names or formal names shadowing a function cannot resolve
back to that declaration by spelling alone. Resource parameter, result and
local/construction uses select checked ownership lowering; unsupported ownership
uses refuse explicitly rather than falling through to ordinary aggregate code.
Borrowed-only CALL_REF remains its existing separate profile. Reference/value
mixtures remain refused. Actual transfer must be established over the admitted
module, not assumed from signatures or required independently in a scalar-only
wrapper or synthetic entry.

## My signatures, arguments and results

Each non-entry function has 0..8 mode-zero parameters: int, bool or an exact
plain finite resource record. Zero-argument factories and scalar-only wrappers
are admitted within a graph containing actual ownership transfer. Existing
finite nested resource parameter/local layouts retain their qualified bounds;
returned resource layouts remain scalar-leaf int/bool only. I do not expand
source U8, float, generic-resource, nested-result or reference result support.
String parameters/literals and PRINT are conditional on the separate qualified
runtime contract and require a later reviewed source checkpoint.

I inspect actuals exactly once, left to right. Scalar actuals must yield their
exact declared tag; each resource actual initially remains a named live whole
owner, mode zero, exact nominal layout, unborrowed and not pending disposal.
I emit OWN_MOVE_LOCAL at that argument position and mark the source moved before
preparing later actuals. Later observation or duplicate transfer therefore
refuses; an earlier scalar observation before a later move remains valid.
Earlier prepared owners remain runtime caller-stack roots until full call
preflight succeeds. Constructed/projected/nested computed owner actuals remain
refused in this slice. Mixed scalar/owner positional metadata is exact; the
runtime still validates every position and reserves the frame before activation.

Helper results are int/bool, VOID with zero operands, or one exact declared
scalar-leaf owner. Shared result descriptors, function tags/counts and parameter
metadata must agree. No wire-format change is needed. Scalar and VOID layouts
use NO_INDEX; resource results use their exact retained nominal layout ID.
Resource-valued calls bind a new local through OWN_STORE_LOCAL or forward an
exact owned result in a return. Resource returns may move a named live owner,
construct an exact declared record, or forward an exact owned call result;
each form transfers once, and every other source owner must already be consumed.
VOID calls emit no POP or fabricated value; implicit VOID fallthrough is admitted
only after exact exit obligations, while a bare explicit return follows the
same rules. Discarded owner results, implicit owner drops and live overwrites
remain refusals. Cleanup drains only existing proven disposal holders.

## My control, metadata and publication

I retain exact reaching-arm owner states, loop/backedge and break/continue
contracts, lexical shadowing, name intervals, source diagnostics and
initializer-before-new-binding order. Each function resets its local and
reference state; mode-zero resource parameters begin live at their real
positions. Scalar parameters retain their tags and normal local semantics.
Function indices, contracts, layouts and optional local names agree in C-seed
and selfhost output, including factory results and shadow main calls. Every
selected shadow is emitted or the entire publication is refused. Any failed
signature, graph or body check stops lowering and preserves the previous output.
No successful metadata is published from a partial graph.

## My qualification order

1. I review the production graph/signature checkpoint before final gates.
2. I compare C-seed and both selfhost stages for zero-argument factories,
   scalar wrappers, multiple helpers, sibling calls and three/eight-frame paths;
   two distinct owners in both orders and eight interleaved parameters; exact
   owned forwarding and VOID consumption; caller observations before/after
   disjoint child calls; and shadow entries invoking ordinary main.
3. I verify canonical metadata and binary/text roundtrips, execute admitted
   modules on VM and strict/sanitized native targets, and retain all existing
   source-borrow, consuming, result and runtime authority gates. Mandatory
   shadows include ordinary pass and deliberate assertion-failure publication
   controls. Refused modules are never executed.
4. I require semantic/output-preserving refusals for wrong nominal identity,
   duplicate or late use after move, unconsumed parameters, discarded results,
   wrong return count/type, function-name shadowing, recursion, graph/arity
   overflow, mixed references, constructed/projected owner actuals and missing
   selected-shadow support. Old valid multi-owner refusal fixtures become
   positive acceptance rather than being deleted.
5. Only after the peer's string/PRINT prerequisite and paired source checkpoint
   qualify do I compile the unchanged affine example, execute all mandatory
   shadows and compare output/results, then return it to full product gates.

I leave full normative ownership, mixed reference graphs, owner entry results,
new nested returned layouts and the full c435 blocker open until their exact
acceptance is measured. This document records a proposal, not implementation
or admission evidence.
