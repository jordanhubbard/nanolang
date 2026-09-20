# My precise mixed record-array global flow

I retain task_d0208a61082f4a08b407768b1a8b4252 and the stopped88ae fixture.
Puck reached the first linked instrumented VM program, but preparation refused
before private execution: `I require exact known field-write tags.` Linux88ae
instead stopped in setup with an explicit temporary-file ENOSPC error. I do
not conflate those outcomes or modify the original graph assertions.

## My observed boundary

In `managed_array_shapes.c`, globals begin as VOID. STORE_GLOBAL weakly merges
into one program-wide summary; LOAD_GLOBAL reads that summary. Thus even a
successful dominating store of a sliced ARRAY leaves VOID in a subsequent
load. The first VM graph stores its slice in global2, then installs the loaded
value into an exact ARRAY record field. Existing field/call/global qualification
loads a global receiver for GET, but does not establish this field-installation
path. The refusal is conservative; it is not evidence of a VM handler fault.

I also retain the later heap-result fixture unchanged. Its TYPE_CHECK VOID
branch initializes a global before returning it, and a later invocation mutates
that same retained global/result alias before assertion failure. Precise stores
alone do not discharge this case: my branch transfer must retain the exact
checked global snapshot relation and must not reuse a stale check after a
store or callee effect. I review that relation with the global flow rather than
repeatedly discovering a known conservative join as a separate runtime failure.

## My boundary and immutable output

I change only the private same-original-module record-array analysis mode
selected by `a->structure`. Old leaf/graph/record wrappers keep their existing
transfer choices and refusal behavior. Public admission, runtime APIs and the
copied plan/report ABI do not change. The original93-opcode closed profile,
256 functions/globals/locals/operand slots,64 allocation origins,1048576 abstract
cells and existing byte/work ceilings remain. I do not remove exact field-write
checks, treat unknown as a concrete tag, or trust a module-side verified bit.

My retained states gain a global vector after their fixed local/operand area.
Each global remains a finite tag/origin/unknown value. The per-instruction
vector is separate from the monotone program-wide committed-value summary.
STORE_GLOBAL replaces the selected path-state binding with the consumed value;
LOAD_GLOBAL reads that path state. All branch/backedge joins union values;
initial VOID remains on paths that have not established a value. A store in an
unexecuted function or branch never initializes another path by itself.

I determine the complete global count before allocating any function state,
including references in later functions and dead instructions. Every state's
stride is checked `locals + maximum_stack + globals`, with at least one cell.
I copy/join the full fixed global vector independently of current operand depth.
The existing cell ceiling includes the added retained global exit summaries.
All products, totals and indices use checked arithmetic before allocation.

## My calls, initialization and repeated entry

I compute a conservative direct/transitive may-write set for each function
from every STORE_GLOBAL and closed CALL edge. Unreachable stores may overstate
this set; they do not fabricate a value or a successful return. Unsupported
indirect/host transfers remain refused. The finite bitset propagation charges
all inspected words/edges and stops at the existing work limit.

A call seeds its callee with actual argument values and caller path globals,
joining different call contexts conservatively. Normal explicit and implicit
returns publish a monotone global-exit vector and an explicit normal-return
reachable bit, alongside the existing result summary. A caller does not invent
normal continuation before a callee return exists. At a returning call it
replaces may-written global slots with the callee's conservative exit values;
slots outside the transitive may-write set retain the caller's more precise
values. If a callee conditionally writes, its exit join includes the unchanged
entry alternative. Recursive summaries use the same finite monotone fixed
point; an unresolved/nonreturning cycle is not a successful initializer.

The committed-value summary still includes initial VOID and every reachable
store value, including stores before assertion/type/allocation failure. This
summary conservatively seeds possible future invocation entry state because
my runtime retains committed globals after failure. When it grows, I reseed
root entry and the selected initializer rather than keeping stale entry facts.
An initializer's normal exit, when present, seeds main; main is not seeded as
though the initializer had succeeded before its normal exit is known. A failed
initializer may contribute committed values to a later invocation but cannot
make the current invocation enter main. Unused functions still receive the
existing conservative unknown-argument validation; they are not treated as
executed initialization. Their conservative summaries may cause refusal.

## My checked global snapshot relation

I attach non-public analysis provenance to values: a current global-load
identity, or a TYPE_CHECK predicate on that exact loaded global/tag. This is
analysis metadata, not a runtime pointer, stamp, opcode or reusable certificate.
A LOAD_GLOBAL records its binding index; TYPE_CHECK can produce a predicate
only from that current identity. Copying a value within one function preserves
the relation. A join preserves it only when all incoming alternatives agree;
otherwise the relation becomes unknown. A STORE_GLOBAL invalidates every
outstanding load/predicate relation concerning that slot in locals, operands
and path-global values, before publishing the replacement. A CALL invalidates
relations for every transitive may-written slot. No relation crosses function
entry/exit summaries or survives an unknown writer as proof.

JMP_TRUE/JMP_FALSE may refine only this retained TYPE_CHECK relation. The true
edge intersects the corresponding global tag set with the checked tag; the
false edge removes it. An empty known alternative has no successor. Origins
are filtered consistently with the remaining ARRAY/STRUCT kinds; scalar tags
carry no nominal heap-origin alternative. Unknown input remains unknown and
cannot justify a concrete field value. Other boolean/comparison operations
conservatively discard this relation unless independently specified later.
This suffices for the original VOID-initialization branch without guessing
truth from constant values or weakening joins. Work includes invalidation scans,
origin filtering and both edge-state copies.

## My allocation, work and termination accounting

I preallocate new-mode global-flow storage once: checked function-by-global
exit values, reachable flags, may-write bitsets and any reverse call/work queue.
The same allocator/error channel owns all allocations; no per-transfer heap
allocation is introduced. Preparation accounts retained state, exits, bitsets,
queue storage and transient local/operand/global copies at their simultaneous
peak. A private full-state scratch has at most768 values; explicit additional
branch scratch is included, not hidden as free stack storage. I keep old-mode
scratch/control paths separate where preserving their behavior requires it.

Every state copy, global join, committed-summary merge, provenance invalidation,
call seed/exit application, may-write propagation, return and root reseed charges
the existing work counter. Limits refuse with output untouched and full cleanup;
allocation failure remains exact MEMORY. The lattice has finite tags,64 origin
bits, unknown flags, finite predicate identities and a top/no-proof state.
Joins only lose precision or add alternatives. Strong stores are monotone
transfer functions over that lattice; they do not mutate an already retained
predecessor state. Finite work bounds apply even if a conservative fixed point
would otherwise take many iterations.

## My required acceptance before resuming the VM corpus

I preserve the original stopped global-slice/field and retained heap-result
programs byte-for-byte as positive requirements. Query-only controls first
cover dominating stores, no-store paths, conditional incompatible stores,
initial VOID, unused writers, call/no-call joins, callee overwrites and untouched
slots, recursion, successful/failed initialization, repeated invocation seeds
and stores before failure. Snapshot controls include a stale predicate after a
direct store, a may-writing call and a branch join; those cases must not borrow
an earlier guard to establish an exact type. Matching guards without an
intervening writer must establish the original valid VOID/ARRAY branch.

I retain old-wrapper outcomes, all independent plan rows/getters, source-input
immutability, exact byte/work/cell boundaries and complete one-shot/persistent
allocation-prefix cleanup/recovery. Only after root source/fixture review and
fresh query qualification do I resume the complete original VM matrix. I may
run separately selected independent VM cases with transparent attribution, but
that does not turn the stopped corpus or full mixed release green. Native C,
LLVM/Wasm, nested/cyclic graphs and paired source remain full parent obligations.

## My production checkpoint mapping

The source checkpoint stays in `managed_array_shapes.c`. `GlobalFlow` owns
function-by-global exits,256 return bits and256-by4 may-write words. The private
prepass examines the retained structural decoder before any function decoder
moves ownership or any function-state stride is allocated. Exit cells join the
same1048576-cell total as function states; exact allocation widths join the
existing peak-byte budget. The prepass charges each decoded row, each propagated
word and zeroed control byte. It uses no reverse-graph allocation.

`join` merges the fixed global vector separately from live operands. `seed`
strips relations at function boundaries. `global_return` publishes normal exits.
`global_invalidates` scans the bounded state before a strong store or may-writing
call. `global_refine` filters only a current checked global tag/origin snapshot;
tags outside the16-bit abstract domain cannot trigger an unchecked shift.
Relations use a16-bit index/kind and an8-bit checked tag in the old Value padding
on the selected ABIs. Ordinary `merge` still exports only tags/origins/unknown,
so heap summaries and function results never retain snapshot proof metadata.

The private scratch reservation covers three simultaneous768-value vectors:
walk state, branch copy and nested seed. Old-mode logical state remains512
values. Existing per-transfer charges increase from2048 to4096 only in the new
mode, with separately charged joins, invalidations, branch copies, seeds and
return summaries. Cleanup frees the optional exit vector/control owner after
all transferred function states and before the final analysis owner. No source
fixture or stopped VM program changes in this checkpoint.
