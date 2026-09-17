# My scalar passive eligibility record

I carry version-1 and version-2 scalar eligibility records in v2 section `0x0c`, guarded by
feature bit `0x40`. Readers that do not know that feature reject the module.
Legacy output refuses records rather than dropping them. This is a bounded IR
foundation for `PASSIVE_PARALLELISM_DESIGN.md`; broader frontend eligibility,
resource metadata and the full conformance matrix remain unfinished.

Every field is a little-endian `u32`. The section starts with `version = 1` or `version = 2`
and a positive block count. Blocks follow function index and entry offset order,
without overlapping ranges. Offsets are absolute positions in the code section;
exit offsets are exclusive. Each block contains:

- kind (`1 = par`, `2 = flow`), function index, entry, exit, node count;
- one node per source binding, in source order; its array index is its dense ID;
- each node: entry, exit, result local, dependency count, external read count,
  observable effect mask (zero), resource count (zero), sorted dependency IDs,
  then sorted external parameter local indices.

No trailing bytes, duplicate IDs, unknown version, flags, effects or resource
claims are accepted. A node ID also defines its source-order index; there is no
second field that can disagree with it. I recompute the lowest-source-index
ready node at every step. Physical instruction ranges follow that stable
**topological** order, including forward references in `flow`. `par` admits no
internal edges. Ranges exactly partition the block, with no gaps or overlaps.
The node's final instruction stores its result, which serves as the serial
binding commit; the next node begins with an empty stack.

The version-1 verifier permits straight-line scalar stack operations and arithmetic.
Each node may read declared completed node results and may write only its
distinct result local. External read counts must be zero in this version: the
ordinary verifier does not yet prove caller values from parameter annotations. It cannot access
an incoming stack value. Every recorded read/dependency must occur in code,
and every code read must be recorded. Result locals must not alias parameters.
External input support is tracked by `task_bf571298c10d4cc5a387b9f233ff3c40`.
Calls, captured values, aggregates, host operations and
handler control flow are refused in this slice. Outside node ranges I also
allow ordinary branches, return/halt, print and assert; a branch cannot enter
the middle of a block. These restrictions are explicit prerequisites, not a
claim that scalar-only functions exhaust the language contract.

Ordinary verification still applies. Serial VM and native translators execute
the unchanged instruction stream. Loading, verifying, or serializing malformed
eligibility claims fails; metadata-free modules retain their existing behavior.
The bridge copies the exact payload. Code transforms that change offsets must
rebuild valid records; they cannot retain stale claims or silently discard them.

## Canonical text

I write exact eligibility bytes as ordered `.passive "hexadecimal"` chunks in
canonical disassembly. Each emitted chunk carries at most 32 bytes. The
assembler appends chunks outside functions and verifies the complete record
against the assembled code; invalid hex, incomplete records, stale offsets and
false graph claims reject assembly. Reassembling canonical text preserves the
full canonical v2 bytes, including this section and its feature bit.

This is lossless textual transport. Editing instructions still requires updating
and revalidating their claims; this does not establish transformation equivalence.

## Version 2 guarded scalar inputs

I use record version `2` for a bounded extension with the same field layout.
Version `1` keeps its zero-external-read rule. Older readers reject version `2`;
I do not erase the version or the claim to obtain compatibility.

An external parameter read in version `2` requires an executable guard in a
straight-line prefix at the owning function entry. Each guard is exactly
`LOAD_LOCAL parameter; TYPE_CHECK tag; ASSERT`. Guards use strictly increasing
parameter indices and declared tags from `int`, `bool`, `string`, or `float`.
Every external read must have its matching guard before the passive block.
Annotations alone are not evidence of the runtime value. Calls enter at the
function start, and a failed guard stops execution before the block.

I forbid writes to each recorded external parameter throughout the function.
This preserves the checked value even when ordinary branches revisit the block.
Node operations and dependency restrictions stay unchanged. `TYPE_CHECK` is
permitted outside node ranges in version `2`; it does not itself make a node
eligible. I retain ordinary stack and instruction verification.

The guarded-input acceptance excludes `u8`, aggregates, captures and foreign
purity summaries. Closed local calls have the additional checks below. Frontend
`par` emission has the separate bounded contract below; `flow` remains open.
The complete external-input task remains open until its broader
acceptance is met. Guarded scalar validation and paired execution evidence are
in [my acceptance record](evidence/passive-guarded-inputs.md).

## Structured producer markers

A textual producer may bracket a function-local independent block with
`.par_begin` and `.par_end`. Before each binding's instructions it writes
`.par_node result-local [external-parameter ...]`, with sorted external indices.
The assembler resolves current function and instruction offsets into the
existing version-2 record. It does not invent guards, purity facts, dependencies,
or resource permissions. Ordinary passive verification remains authoritative.

Markers must be complete, nonnested, and inside one function. A block needs at
least one node. Raw `.passive` chunks and producer markers cannot be mixed in
one input. Canonical disassembly continues to emit exact `.passive` hexadecimal
chunks; producer convenience syntax is not a second binary format.

## Version 2 closed scalar calls

I accept a concrete local `CALL` inside a node only after checking its reachable
callee instructions. Node inputs already come from executable scalar parameter
guards, constants, scalar operations, or verified completed-node results. A call
consumes the target's actual arity and pushes its checked result count; declared
parameter and result tags do not establish scalar values.

Under those scalar actual arguments, I derive scalar results from instructions.
I intersect definitely initialized locals at control-flow joins and check every
reachable local read after convergence. Every admitted producer returns a scalar
or traps. Local reassignment and loops are permitted. Nested direct calls and
tail calls require the same check; every successful return has the exact result
count. I do not accept a declaration as a substitute for this body check.

I refuse recursion, foreign or indirect calls, captures, global access,
aggregate operations, shared mutation, printing, and other effects in a callee.
Ordinary assertion failure remains possible, as with scalar arithmetic traps;
this is a purity boundary, not a termination theorem. Unreachable instructions
are outside the callee summary. Ordinary module verification still checks them.
Calls elsewhere in the owning function remain outside this extension.

The check is bounded: at most 64 active call frames, 65,536 instructions per
callee, 2,097,152 local-state words per callee, and 1,048,576 worklist steps per
callee. Exceeding a bound refuses the claim. Allocation failure also refuses it.
Version 1 remains unchanged. Version 2 additionally admits typed integer/float/
boolean arithmetic and string concatenation with the same scalar provenance.

## Bounded source emission

I retain `par` identity in both frontends and require distinct immutable `let`
bindings with scalar initializers. I inspect all initializers against the
pre-block environment before introducing their names. Sibling references,
mutable inputs, aggregates, and unsupported effects are refused.

Closed scalar source calls require resolved body inspection; scalar local loops
and reassignment are permitted. My separate `pure fn` rules are unchanged. The
NanoISA emitters require guarded parameter inputs and retain every admitted
binding in version-2 records, which undergo independent bytecode verification.
Native source compilation retains the same lexical binding behavior.

I test the original square/cube fixture and calculator on all three native
compiler stages, and compare the unchanged calculator scalar closure across
both NanoISA emitters and VM/native execution. The full raw calculator still
refuses its `strlen` ABI. Foreign identity, broader external inputs, and `flow`
extraction remain open. [My frontend evidence](evidence/passive-par-frontends.md)
states the exact boundary.
