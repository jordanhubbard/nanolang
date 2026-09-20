# I check File frames and calls inside the counted arena

This is the preimplementation contract for
task_bb3381ff39934b379067adaad2785ced under runtime82ff. I build on
[my carrier contract](NANOISA_FILE_PRIVATE_RUNTIME.md) and its
[qualified first milestone](evidence/file-runtime-carrier.md), PR858.
I integrate its eventual canonical merge without changing qualified trees.
This contract grants no production or execution approval by itself.

## I reuse the actual bounds

file_hosted_bounds already derives locals L, operand peak P, staging count T,
frame count and VM/native value bounds from the complete acyclic body query.
T is at least one and at least parameters+1 at every reachable call.
Wire max_stack zero still means derived storage; nonzero declarations must
cover the actual peak. I preserve this convention and all descriptor identities.

The carrier has one value, reference, region and concrete frame arena. I use
these allocations rather than allocate a second frame arena. I persist its
selected VM/native mode explicitly: equal slot counts cannot identify it.
Additional context/frame fields enter the owning sizeof arithmetic before
allocation. I retain the64MiB bound including hosted preparation and the nested
File-values/service/capability storage once. There is no per-call heap
allocation after begin or uncounted argument/result workspace.

For absolute frame base B I fix the layout:

| Segment | First slot | Count |
|---|---:|---:|
| Locals | B | L |
| Argument/result staging | B+L | T |
| Operands | B+L+T | P |

At a call with operand count S and value argument count
A = parameters - borrowed_inputs, the callee starts at B+L+T+S-A in VM mode
and B+L+T+P in native mode. These are the existing hosted recurrences.
I check A<=S, every widened sum/subtraction, the callee aggregate requirement
and arena end before indexing. The VM suffix can overlap consumed operands
and unused tail; caller locals/staging/live prefix remain outside it.
Native frames are disjoint. Depth stays within the prepared count (at most64);
each depth has its own checked256-reference slice.

## I define the private adapter responsibilities

The source checkpoint introduces a private header/include alongside
file_runtime.c, with these concrete operations:

1. Frame start chooses only the prepared initializer, or entry if absent, at
   base zero after successful carrier begin.
2. Frame view copies current function/instruction, depth, segment bases,
   operand count, reference base and region floor. It exposes no mutable
   context pointer or owner handle, and preserves output on failure.
3. Local/operand/reference location queries validate current-frame bounds;
   an operand reservation requires an empty destination. Local stores compare
   the exact tag, mode, catalog and global nominal identity before consumption.
4. Stack completion checks physical occupancy and the current body's input/
   output counts. A removed slot cannot hide a live owner/formal alias in its
   unused tail. Staging must be empty at ordinary instruction boundaries.
5. Frame call accepts only the current decoded CALL/CALL_REF and its prepared
   obligation. It stages arguments, then activates that exact callee.
6. Frame return accepts only the current checked RET, stages the exact result,
   clears the callee and resumes the saved caller continuation or hosted root.
7. Continuation follows only decoded successors after count validation. This
   operation does not evaluate arithmetic, assertions or branch predicates;
   the later matched dispatcher must derive branch decisions from actual
   typed values and Result arms, rather than a supplied guess.

Mutators check phase, first error, current function/site and frame bounds before
root changes. Low-level manual carrier primitives remain private; supplying
a matching site is not a certificate for an external caller. I introduce no
public frame handle or wire representation. Full instruction semantics and
pending-runtime-mask coverage remain later dispatcher work.

## I stage all arguments before reusing a VM suffix

I preflight decoded target, exact callee signature, input count, argument types,
result room, staging emptiness, reference origins, frame depth and all ranges
before moving an owner. Parameters follow declaration order; value parameters
consume the existing operand suffix in that order. The first parameters staging
positions are indexed by formal number; the last position holds the return
result. Borrowed parameter positions stay empty during value staging.

I immediately move each value argument to its parent staging root before
repurposing callee-overlapped slots. Only after ALL value arguments have reached parent staging do I install ANY
callee local. I then install each staged value in its exact parameter local;
nonparameter locals begin uninitialized. Scalar/passive values retain
their copy semantics; File/OpenResult move. A formal alias or passive view
cannot satisfy an owning declaration.

Current CALL_REF encodes one reference operand. The body query maps it to
borrowed formals and refuses repeated origins. I add no multi-reference
encoding and accept no previously refused call. A formal alias occupies
callee_reference_base+parameter_index, matching the logical state constructor.
I resolve the caller reference, retaining its ancestor origin, and install a
non-owning callee alias. Nested forwarding can share an origin; distinct
parameters of one call cannot share it. Parent/child reference slices are
disjoint. I derive mappings from immutable declarations/instructions rather
than allocate another argument map.

I promise rooted failure cleanup, not rollback. A generation-limited core move
can fail after earlier arguments have staged. I retain the first call-site
error and every owner at its actual source, staged or installed root, then use
terminal cleanup. I neither invent reverse moves nor execute a partly entered
callee. Structural preflight failures before any move preserve input roots.
Successful intermediate moves always leave exactly one tracked owner root.

## I stage the result before clearing a callee

RET must match the exact result count/type/identity and actual operand count.
No other callee owner, local borrow or region may remain live. Formal aliases
are not owning results; removing them does not end caller borrow epochs.

For one result, I first move/copy it into the parent's reserved result staging
slot. Only then can I clear callee locals/staging/operands and pop its frame.
The caller's result position may equal the callee's first local in VM mode.
After clearance, I publish the staged result there, restore the saved count/
continuation and empty staging. Zero-result returns still enforce exact exit
cleanup. Failures retain roots for terminal cleanup and publish no success.

The failing call/RET retains its first diagnostic site; secondary close errors
remain separate. Normal mutators refuse after first error, so exceptional
unwind uses the qualified terminal drain, not retries of ordinary mutators.

An initializer has zero results. I pass the existing clean root-completion
boundary before activating entry at base zero. Its cleanup-error snapshot
still suppresses entry even after consuming the Error Result. Both roots share
one invocation; entry exposes only exact INT/BOOL after clean terminal finish.
Helper File/Result returns remain internal.

## I bound references and regions by their frame

I save the global region-stack floor on entry. A callee cannot end an ancestor
region. Local reference numbers translate only inside the current256-slot
slice. FILE_END_BORROW cannot release a formal alias as a local origin; internal
return cleanup removes those aliases. Origins refuse end while aliases live.
Terminal unwind removes aliases, ends origins, drains owners, then disposes
the service, retaining the qualified order.

External serialization remains required. Busy checks cover synchronous
reentry, not concurrent exclusion. Copied query outputs preserve sentinels
on failure and remain disjoint from context/input/other outputs.

## I qualify transfers before dispatch

I send complete source plus allocation/layout tables for independent review,
then send the complete fixture checkpoint before any new service executes.
Focused qualification covers:

- VM suffix overlap with caller locals/prefix intact; native disjoint frames;
  equal-count modes, nested calls, zero/scalar/owner/helper-Result returns,
  permuted nominal maps and wrong/uninitialized local refusals.
- All arguments rooted before callee installation and results before callee
  clearance; injected failure at each owner staging boundary with exactly-once
  cleanup and output preservation. Successful calls/returns allocate nothing.
- Nested alias forwarding, same-call duplicate-origin refusal within existing
  encoding, origin survival after return, region-floor/formal-end refusal and
  full cleanup after callee failure.
- Exact accounted storage/caps, maximum supported frame depth, repeated acyclic
  calls reusing empty slices, initializer order and cleanup suppression,
  first/secondary errors and synchronous reentry.
- Old manual carrier and public verifier/VM/native/FFI/CLI refusal neighbors.

Fixtures may drive this private frame API manually over fresh prepared bodies;
that qualifies transfers, not an opcode interpreter or generated-native
program. The subsequent matched VM dispatcher and direct native C lowering
must separately execute complete bodies and discharge every pending obligation
before public selection. Loops, indirect calls, richer multi-borrow encoding,
full source/shadows and installed File acceptance remain concrete required
parent work; an acyclic milestone does not remove them from release scope.

## I implement the first private API checkpoint

I add `file_runtime_frames.h` and its owning include in `file_runtime.c`.
I persist the selected mode and active depth in the context. Each counted frame
holds its function, instruction, local/staging/operand bases, logical operand
count, reference base, region floor and suspended-call flag. I retain the
call instruction while suspended and recover the exact continuation/result
count from my immutable hosted plan; I allocate no second continuation table.

| Storage | Count and accounting | Lifetime/ownership |
|---|---|---|
| Context | One `sizeof(NvmFileRuntime)`, including mode and active depth | Create through destroy |
| Frame arena | Prepared frame count times actual `sizeof(FileRuntimeFrame)`, including suspended-call flag | Allocated once; active prefix, cleared on return/root completion |
| Value arena | Selected hosted VM/native bound times actual `sizeof(FileRuntimeValue)` | Every source, staging and installed owner remains a tracked root until moved or drained |
| Reference arena | Prepared reference slots times actual `sizeof(FileRuntimeReference)` | One256-slot slice per depth; aliases retain ancestor origin without owning its epoch |
| Region arena | Prepared region slots times `sizeof(uint64_t)` | One global stack; each frame saves its entry floor |
| File core | One nested File-values/service/capability size query | Begin through terminal drain/disposal; counted once |
| Call/return temporaries | Fixed scalar indices and copied immutable facts on the C stack | No owner handles, argument vectors, heap allocation or second root arena |

I check each allocation product and sum in the existing owning create path
before allocation, retaining the64MiB limit. Frame layout uses widened sums,
checks the selected recursive extent, own operand end and reference slice, then
narrows to arena indices. Every call stages all value arguments before local
installation. No return clears its child until its result is in parent staging.
A failed generation-limited move stops immediately with actual roots retained;
my qualified terminal drain does not require the frame stack to describe every
partly installed root.

My local, operand and reference getters return absolute arena indices, never
mutable owner handles. `frame_store` handles only exact decoded stores.
`frame_next` checks the body's completed output occupancy and a decoded
successor; it does not evaluate the instruction or choose a branch predicate.
Call and return have separate transfer paths. Region wrappers require the exact
current opcode and prevent ending an ancestor region or a formal alias.
The later matched dispatchers must call these operations in their actual opcode
semantics; private C callers can still misuse raw carrier operations, and this
checkpoint does not advertise them as a public execution authority.

I have performed a strict C11 syntax-only check and a whitespace check of this
checkpoint. I have not built or executed a new service fixture. The earlier
carrier qualification remains pinned to its original source; this new API still
requires reviewed fixtures and fresh qualification before any dispatcher work.

## I prepare focused manual transfer qualification

My new `test_file_runtime_frames.c` first calls the complete unchanged carrier
suite through an optional main-name macro. I retain every original assertion,
allocation-prefix/transient control and attempted public loader/fork refusal.
The standalone carrier fixture keeps its original main by default.

I add explicit manual sequences, not an opcode-dispatch loop:

- A borrowed first parameter precedes two value arguments. In VM mode installing
  its first value local overlaps the later argument's old slot. I assert both
  values, caller locals/live prefix and nested alias origins survive. On return
  the caller result occupies the old callee first-local slot. I repeat the call
  with different values and compare native disjoint placement.
- I return File, affine OpenResult, scalar Result, INT and zero results through
  exact prepared helper signatures, including permuted nominal maps. I check
  nested borrowed-formal forwarding and ancestor use after return.
- I check equal-bound mode persistence,64-frame depth, empty/out-of-range copied
  getters, wrong/uninitialized arguments, exact local-store refusal, hidden
  owner tails, leftover return owners, mismatched results, formal END refusal
  and an ancestor-region floor. Repeated-origin CALL_REF remains a preparation
  refusal before context publication or host acquisition.
- I set an actual File-values slot and its valid handle near generation
  exhaustion in an instrumented owning TU. I exercise both argument staging
  moves, both local-installation moves, return staging and return publication.
  I retain real move/destructor code, inspect each surviving physical owner root,
  require exactly-once close and preserve the first call/RET failure plus a
  separately modeled close-report error. I do not introduce a production hook.
- I pass an initializer through its actual RET/frame boundary after a handled
  close Error. The cleanup snapshot still refuses entry; clean initializers
  reuse the arena for entry. Successful checked calls/returns run with a zero
  allocation budget, so any allocation attempt is an assertion failure.

My fixture host hooks and counters retain their earlier scope: linked mode
uses actual separately compiled cores, while instrumented mode observes real
File descriptors and injects error reports after actual close. I keep the
`ferror`/partial-progress model label and do not infer arbitrary libc close
behavior from it. The only new owning-TU fixture helper changes a valid internal
generation pair; the old carrier remains included verbatim apart from its
optional main-name macro.

Before execution I require review of the complete fixture and Python driver.
The new explicit normal/sanitizer Make targets prepare the same provider closure;
I do not add an unqualified target to `test-units`. For frozen evidence I prepare
providers once, then invoke the two new unittest methods directly with exact
object/link lists. I retain every provider, fixture object/binary, command,
terminal and source/tool inventory before and after each phase.

My ordered gate plan is Linux ordinary GCC, GCC and Clang ASan/UBSan/LSan, then
Darwin ordinary Apple Clang and Homebrew Clang sanitizers on isolated puck.local.
I select compilers explicitly. Native Linux Clang alone receives the known GCC13
installation flag. Darwin records its actual SDK, pkg-config binary, resolved
libffi headers and OpenSSL library; wrapper providers are prepared before the
freeze. Both modes execute the complete old carrier suite and new manual frame
controls. I then run unchanged hosted/body/flow/value/opcode refusal and wrapper
neighbors using the same frozen provider inventory; no phony rebuild is silently
attributed to a prior inventory. I preserve the first terminal and stop its
remaining dependent phases on failure. No service bytecode dispatcher, native
emitter or public File selection is qualified by these manual sequences.

I retain the prefreeze syntax-only finding: my first synthetic builder named a
v2-only max_stack member on the in-memory function entry. Both syntax checks
refused before execution. I removed that assignment; the service bridge supplies
the intended derived wire bound. Corrected linked/instrumented strict C11
syntax checks and Python parsing pass. These checks execute no fixture.

## I retain exact refusal precedence

My first56bb Linux normal frame run stops at the fixture expectation for a
CALL_REF callee with two borrowed formals. The retained log establishes that
`INVALID` was not returned, but does not print the actual enum. Static inspection
establishes the earlier boundary: `file_code_operands` in `file_code.inc`
requires exactly one borrowed formal for CALL_REF and returns `UNRESOLVED`
otherwise. Only admitted calls can reach the logical flow's repeated-origin
`INVALID` check. My current encoding is still one-reference only.

I correct this fixture to require exactly `UNRESOLVED`, preserving the original
output pointer sentinel and unchanged host-attempt counter. I do not accept a
set of possible statuses, change production, execute preserved artifacts, or
claim that the original log measured the replacement status. Fresh corrected
qualification must establish that result. The initial native Clang setup
failure and its corrected11.405s setup remain separate retained terminals.
