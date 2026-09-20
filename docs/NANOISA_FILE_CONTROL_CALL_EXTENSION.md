# I extend File control and calls only through matched proofs

I record design child `task_243a9a5809bf422caab0bddafa447098` under
`task_72556b6bf6c6d793e83b9cb427a8613a` and
`task_6931ec89b210421e9827fecdbb459dbb`, after the bounded
[private runtime](NANOISA_FILE_PRIVATE_RUNTIME_RECONCILIATION.md).
This is a review proposal, not implementation or execution approval. The public
acyclic activation child dfa149 remains independently owned. I add no public
route, wire feature, host grant, source opcode or runtime behavior here.

## I preserve the measured boundary

`file_code.inc:file_code_cfg` currently topologically orders every instruction
and returns UNRESOLVED for a cycle. `file_body.inc` consumes that order once;
`file_flow.c` allocates fresh symbolic owner and region identities during
transfers. Merely permitting a backward edge would invalidate both convergence
and ownership reasoning. A byte offset less than its predecessor is not itself
a cycle; exact decoded successor edges determine the graph.

`file_code.inc` currently accepts CALL_REF only with exactly one borrowed
formal. The lower flow primitive can compare several reference inputs and
reject duplicate underlying owners, but that is not encoded/runtime authority
for a multi-reference call. Direct calls retain exact signatures and the closed
acyclic call graph. No existing generic function tag proves a File callable.

I retain64 functions,256 locals/stack/references/owners and per-function decoded
instructions,4096 total instructions,64KiB CODE and64 runtime frames, plus the
existing checked query/hosted/runtime memory ceilings. Any proposed numerical
limit change needs its own rationale and review; no dynamic growing owner arena
or unaccounted work queue follows from this design.

## I first prove cyclic intraprocedural flow without admission

My smallest implementation checkpoint is a separate non-admitting checked
query, leaving the old acyclic plan and selectors unchanged. It decodes all
functions and unused instructions with existing byte/operand rules, computes
strongly connected components, and checks all edges including loop exits.
Calls remain direct and nonrecursive. I do not use a runtime fuel counter as
proof that owner flow is sound.

I define a finite abstract state before writing the worklist: exact initialized
local/stack categories and nominal identities; Result-arm alternatives; owner
locations and equality classes; reference origins and region nesting; pending
cleanup obligations. Unknown or uninitialized alternatives never become
initialized by a loop backedge. Header joins include the entry/zero-iteration
edge and every backedge. Distinct live roots cannot collapse to one abstract
owner, and a consumed owner cannot reappear because an allocation site repeats.

I propose canonical symbolic naming by live owner location/equality relation at
an edge, with bijective renaming only when every observation, held epoch,
pending transfer and cleanup obligation maps consistently. Runtime generations
remain distinct and are never canonicalized. A create/use/close cycle may
return to an owner-empty header; an unchanged File may remain live across a
loop. A replacement owner at the same local needs an explicit invariant proving
that the old obligation was consumed and no observation survives. Unsupported
renaming or alias alternatives remain UNRESOLVED, not guessed equality.

The query must have finite state/alternative and worklist-pop bounds checked
before allocation and publication. The production checkpoint must specify
exact constants and a termination argument; exhaustion yields LIMIT atomically.
Repeated transfer visits must not mint unbounded symbolic IDs or append duplicate
obligations. Obligations are keyed by exact function/instruction and preserve
all owner/Result alternatives; peak storage is a simultaneous-state maximum,
not an accumulated per-iteration count. Failed or incomplete fixed points never
publish a certificate. Query success remains non-executable.

## I then match both private targets

After query review and qualification, a separate checkpoint consumes its exact
certificate in the hosted plan, private VM and real generated native functions.
Every loop edge retains category, lifetime and cleanup checks. Both targets use
one invocation-wide finite step budget with the same instruction accounting,
including initializer and helpers; budget exhaustion records its site, drains
all roots and preserves prior scalar output. Exact API/default/maximum budget
and ABI revision are reviewed before code. The existing UINT64 overflow guard
is not advertised as a practical execution bound.

I require zero/one/many iterations, nested loops and multiple exits; owner live
across a backedge; repeated acquire/extract/use/close with bounded live slots;
Result alternatives, continue/break/early return, assertion and fuel failure;
borrow regions opened/ended per iteration; rejected observation-after-move and
unbalanced-region edges. Live stream/root counts must return to their expected
per-iteration level before terminal disposal, including runs exceeding64 total
acquisitions. Allocation prefixes, transient failures, host errors and secondary
cleanup failures retain sentinel isolation and recovery. The same serialized
modules execute privately in VM and native O0/O2 on Linux/Darwin with the original
acyclic corpus retained. Public selection is a later reviewed conjunction.

## I add closed indirect calls as a distinct dependency

I first inventory the actual ISA callable representation and source callable
identity before choosing any encoding. A finite target set must come only from
checked same-module function identities, with exact parameter modes, nominal
arguments/results and compatible cleanup contracts for every alternative.
Arbitrary integers, foreign pointers, callbacks, captured environments and
unresolved imports provide no target authority. Recursive target-set edges
remain refused until a separately reviewed recursive resource bound exists.

A private query propagates target alternatives through locals, branches and
loops and validates every candidate body, including unused candidates. Its
certificate retains original function/global identities and the complete set;
there is no first-candidate shortcut. Runtime selection checks membership before
argument transfer; generated C uses a bounded checked dispatch to real generated
functions, not generic FFI or a bytecode interpreter. Failure before/after a
partial transfer cleans exactly caller suffix and staged prefix. VM and native
checks, ABI/transport preservation, forgery/unknown-target refusals and O0/O2
fault acceptance precede any public or paired-source activation.

## I add richer borrowed calls without inventing owners

I next specify exact transport for multiple distinct exclusive File references
and mixed owned/scalar/reference parameters. Existing CALL_REF encoding does
not silently gain extra operands. Any new encoding or sidecar needs a separate
codec/bridge/roundtrip/old-consumer refusal checkpoint before runtime work.
The query compares underlying owner identity across all references and owned
arguments, including forwarded formals; two aliases of one owner remain refused.

All actual arguments are evaluated and checked in source order before transfer.
Formal aliases never own the caller epoch. Callee unwind clears aliases before
originating borrows end, and every failure preserves or drains exactly its
acquired roots. I require nested forwarding, reordered arguments, duplicate
aliases, held-owner moves/close, later-argument failure, loops with balanced
borrow regions, and first/secondary error preservation across both targets.
Borrowed return escape, stored references, shared/nonexclusive references,
callbacks, async work and recursive borrows remain separate required design
questions; this proposal does not claim their acceptance or erase parent scope.

## I order publication and source acceptance

For each extension I review: exact abstract domain/encoding contract; private
query production; meaningful query fixtures and seals; matched VM/native/ABI
production; complete lifecycle fixtures and platform seals; then public gate
integration with dfa149's grant, locking, packaging and atomic output semantics.
Old acyclic behavior must remain identical and new failed preparation cannot
fall back to it. Plan/fact mismatches refuse before host acquisition.

Paired C-seed/Stage1/Stage2 producers and checked NSI binding generation follow
reviewed runtime authority. Full original programs/imports/helpers and every
mandatory shadow remain selected. I qualify actual loops, callable alternatives
and richer borrow source routes, retaining lexical identity and output guards.
Neither query success nor private bytecode execution closes this source work,
LLVM/Wasm/linked support, installed product acceptance or full72556/6931.
