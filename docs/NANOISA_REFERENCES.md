# My call-scoped reference contract

I require real references for `&T` and `&mut T`. This document orders the
NanoISA implementation of the already supported native cases; it does not
claim that my current VM executes borrowed calls. My C and self-hosted
NanoISA producers still refuse them. My full affine contract remains in
[AFFINE_TYPES_DESIGN.md](AFFINE_TYPES_DESIGN.md).

## Representation

I identify a place by an owning invocation, a frame-local root, its nominal
layout index, and a sequence of numeric record-field indices. A field name
is resolved before emission. Identical printed names do not equate layouts.
Within one module, a layout index is nominal identity; linking must remap
indices while preserving that identity, including distinct same-shaped
records. A runtime invocation identity is not a serialized pointer or a
function index: recursion creates distinct owners.

An empty path denotes the whole local. Each projection must select a record
field whose retained nested layout agrees with its type. I initially admit
only fixed record referents with int, uint8, float or bool fields, reached
through record-only paths. Tuple, union, collection, generic and unknown
referents remain refused until their independent contracts are implemented.
A descriptor never substitutes for the owning local's authoritative type.

My v2 bridge now preserves complete layouts through owned canonical bytes,
required feature bit 7 and canonical `.layouts` reconstruction. Count-only
legacy modules still produce field-less placeholders. Those placeholders do
not establish authoritative record shapes. My producers must populate actual
layouts and resource/function contracts before reference verification can
use them. My [transport evidence](evidence/nanoisa-retained-layouts.md)
distinguishes retention from producer completeness and ownership verification.

My first implementation slice supplies descriptor validation and pure overlap
queries against v2 layouts. It does not install a new section or opcode.
The later executable representation must carry:

- Function parameter mode (owned value, shared reference, exclusive reference)
  and exact referent layout. Reference results are forbidden.
- Authoritative root-local layouts and resource classification, including
  nominal nested layout edges. Existing tag-only signatures are insufficient.
- Reference creation, checked field access and call-region termination in code.
  Every creation names a checked root/path and mode; it cannot claim an
  unrelated layout to make overlap disappear.
- Versioned required ownership metadata. Old readers must reject the required
  feature. Serializers, linkers, reconstruction and translators must preserve
  it or refuse the module; converting to v1 must not erase it.

I will allocate wire codes only with their codec and verification changes.
The existing extended instruction plane provides room without renumbering
legacy operations. An unimplemented declaration is not an executable feature.

## Verification before execution

I compare places by owning invocation and root, then by field-index prefixes.
Equal paths and ancestor/descendant paths overlap. Different field indices
are disjoint, including fields with similar printed names. Two shared holds
may overlap; an exclusive hold may not overlap another hold. A held whole
owner may not be moved, replaced or consumed. Reads overlapping an exclusive
hold and writes overlapping any hold are rejected unless performed through
the authorized reference itself.

I derive holds from actual code and checked signatures, not producer claims.
Argument evaluation begins a hold when the reference is formed, so a later
argument cannot consume or mutate the held place. A direct call checks every
mode and exact referent identity. Forwarding creates a subordinate reborrow;
it preserves provenance, cannot strengthen shared to exclusive, and suspends
incompatible access through the parent until that nested call returns.

Every reachable join must agree on live references, provenance, ownership
and modes. Unknown type information cannot widen a reference into an ordinary
value. Loop back edges must restore entry state. Returns, tail calls and
unwinding must end the current call region without escaping a caller-local
reference. Tail calls with live local-root references are initially refused.
References cannot be stored in globals, heap aggregates, arrays, captures,
returned values or foreign/callback arguments. Ordinary stack duplication
and local loads must not silently copy exclusive authority. Imported calls
remain refused until their contracts can be checked across modules.

My existing ordinary-value type verifier allows unknown values at joins.
That policy is not sufficient for reference verification. I require a
separate precise reference/provenance state and conservative refusal wherever
that state cannot be established. Whole-program ownership verification also
requires explicit moves and resource obligations; place overlap alone does
not prove cleanup.

## Runtime and translator path

I keep reference identity outside ordinary copyable aggregate values. NanoVM
will resolve an invocation handle, root-local slot and checked field path
against the live owning frame. I do not retain a raw pointer into a locals
array that can move when a nested call grows the stack. Invocation generations
must distinguish a returned frame from a later frame reusing its slot.
Reference field reads/writes access the owner's actual scalar field; passing
a copied record and writing it back is not equivalent.

My native translator must preserve the same checked identity and call-region
lifetime. Stable root storage and checked projections may lower to native
addresses only after verification; it must neither box a detached record
copy nor allow a pointer to survive its frame. VM and translated execution
must demonstrate mutation visible through the caller's original nested place.
Unhandled reference operations must fail before output publication.

## Dependency order and acceptance

1. I establish checked place identity, field-path resolution, scalar referent
   bounds and overlap/access queries (task_83d7ced8a4e34d93b7193fdf2b841137).
2. I retain authoritative resource/layout and function-mode contracts through
   v2 codec, bridge, linking and reconstruction, with exact round-trip tests
   and explicit refusal by consumers not yet supporting them.
3. I add reference instructions and precise lifetime/provenance verification,
   including argument order, nested reborrows, joins and non-escape cases.
4. I implement actual NanoVM and native reference access from those verified
   instructions, with recursion, stack growth and caller-visible mutation.
5. I enable both frontends together and compare decisions and retained facts
   for the existing shared/exclusive/nested corpus. I run one pinned artifact
   through VM and native translation before lifting the current refusals.
6. I retain full move/cleanup verification, real service-handle migration and
   the release equivalence matrix as the broader ed702/d03c/28f tasks.

The first slice is a tested prerequisite, not completion of any later row.
My publication hold remains. The old equivalence task's references to explicit
`discard` or replacing borrows do not override my accepted design: I support
real call-scoped borrows and reject `drop`/`discard` syntax.

## Function and root metadata prerequisite

My required declaration section is OWNERSHIP (section 13, feature bit 8), version 1.
It requires retained layouts and does not permit reference execution by itself.
All words are little-endian. I encode the version and retained layout count as
u32, one u8 flag per layout (bit 0 complete, bit 1 resource), zero padding to
four-byte alignment, and a u32 function count. For every function in table
order I encode u16 local count, u16 parameter count, one result descriptor,
and one descriptor per local. The parameter descriptors are the first locals.
A descriptor is u8 value tag, u8 mode (0 value, 1 shared, 2 exclusive), two
zero reserved bytes, and u32 retained layout index or `0xffffffff`.

The counts must match my authoritative tables. A record descriptor with a
layout uses that exact complete record layout. Reference modes require the
currently supported fixed scalar-field resource referent and may appear
only among parameters. Results are values, never references. A void tag
without a layout denotes unknown ordinary local information, not permission
to forget reference provenance. Record fields referring to resource layouts
must propagate resource classification to their complete containing layout.
The metadata declares types; it does not prove a live owner or a correct
instruction trace. I continue to refuse execution of resource/reference
contracts until the instruction verifier and runtime implement them.

My [declaration evidence](evidence/nanoisa-ownership-contracts.md) records
codec and canonical seed checks, ordinary execution controls and explicit
refusal by verified assembly, direct VM APIs and native translation.

Actual float-record lowering remains an independent prerequisite
(`task_93574cf9d200459aa16e959baf68201d`). Retaining a float field tag does not
establish its VM/native implementation. I keep that reference case refused
until its runtime gates pass with the other reference semantics.

## Local instruction-state prerequisite

I first check local-normalized transitions before mapping them to bytecode.
I obtain exact local tags/layouts and parameter modes from my validated
OWNERSHIP section. Scalar definition cannot manufacture an owned record.
Record construction names every field in order and moves resource fields;
a whole-record move invalidates its source. Whole-record unpack consumes
its source atomically and gives each resource field its own obligation.
A scalar observation never consumes a containing owner. I reject replacement
of a live resource and every failed transition preserves the prior state.

I form references in a nested call region, before evaluating later arguments.
Each reference retains the root and numeric projection; a subordinate
reborrow preserves that place and cannot strengthen its parent's mode.
Parent access is suspended when incompatible with a live child. Region end
invalidates its references; exit requires no regions and no untransferred
owned locals. I compare joins exactly, including local liveness, reference
slots, provenance, modes and region depth. An uninitialized or consumed local
cannot be accessed. I do not merge disagreement into an unknown value.

This transition API is not an executable bytecode verifier. I still require
operand-stack provenance, instruction decoding, reachable CFG propagation,
loop/back-edge checks and callee argument/result transfer. I keep all current
resource/reference execution refusals until those checks and genuine VM/native
reference lowering pass. No wire opcode is allocated by this prerequisite.

My transition engine uses symbolic invocation `1` within one function analysis.
Exact join comparison accepts clones sharing that analysis's immutable facts;
separately created analyses do not become equal by matching printed types.
Entry borrowed parameters represent obligations supplied by a future checked
caller. Distinct parameter slots are not proof that actual caller places are
disjoint. My call-boundary verifier must substitute caller provenance and
validate argument overlap before it can use those entry assumptions.

I currently support scalar locals and complete finite record trees in this
API. Reference slots and nested regions are explicit verifier inputs; they
are not heap-storable values. No API operation copies a reference into an
ordinary local, packs it into a record or returns it. I retain uninitialized
and consumed locals as equally unavailable; loop joins require the same live
obligations and reference provenance, not an identical history of moves.

## Bounded bytecode dataflow contract

My next analysis consumes a decoded function, not a producer-supplied list
of transitions. I initially admit exact numeric/bool stack values, scalar
locals and read-only observations of record parameters. `LOAD_LOCAL` of a
record produces an observation tied to its checked root, never an owned copy.
Only a checked scalar `AGG_GET`/`STRUCT_GET` may turn that observation into a
scalar value. I refuse observation duplication, stores, returns, calls,
aggregate construction and mutation until their distinct transfer/reference
instruction contracts are connected. An observation does not consume a live
resource obligation.

I propagate cloned local state and stack tags/provenance along reachable
branch edges. Each join and loop back edge must match exactly; I do not widen
missing or conflicting facts. I require explicit returns with the exact
scalar result tag and no remaining owned obligations. I reject falling off
the code end. Instructions outside this slice are refused even in dead code,
so dead branches cannot hide an unimplemented transfer operation.

This analysis initially bounds a function at 4096 decoded instructions,
256 locals and 256 stack values; exceeding a bound is explicit refusal.
Those bounds limit analysis storage and do not alter general NanoISA limits.
Entry references still assume a separately checked caller contract. Analysis
success alone does not satisfy `nvm_verify` or install runtime semantics.
Executable eligibility additionally requires the standalone contract below.
Source borrow producers remain disabled.

### Concrete transfer connection

My transfer instructions are `OWN_MOVE_LOCAL U16` (invalidate the named
local and push its unique owner), `OWN_STORE_LOCAL U16` (consume that owner
into an exact vacant local), `OWN_PACK U32` (use a retained layout index,
consume its ordered fields and create an owner), and `OWN_UNPACK_LOCAL U16`
(invalidate the whole record and push every ordered field and obligation
atomically). I initially proposed the extended plane. I now allocate verified vacant
primary bytes 0x0b through 0x0e in that order, without changing any existing
opcode. This does not implement extended decoding. Their
codec, assembly/reconstruction, decoded stack effects and strict provenance
transitions must land together; plain `LOAD_LOCAL` remains an observation.
An owner temporarily on the operand stack must neither disappear at a branch
nor duplicate through `DUP`, storage or a call.

My non-admitting entry point is `nvm_verify_affine_function` in `verifier.c`.
It checks structural declarations before the affine pass. Ordinary structure
verification consults this pass before its runtime refusal; the normal
function verifier also refuses explicit transfer instructions, including
instructions without ownership metadata. I keep independent direct
VM/native guards outside the standalone subset below. General admission through
`nvm_verify_function`, `nvm_verify_function_max_stack` and linked verification
still requires reference creation/access/end-region instructions, exact
caller-place alias substitution at direct calls, safe imported contracts and
actual VM/native transfer and reference semantics. Passing this analysis alone
does not authorize removing any of those guards.


## Explicit owned transfer wire and verifier contract

My four transfer operands use existing little-endian codecs: `OWN_MOVE_LOCAL`
(0x0b, u16 source), `OWN_STORE_LOCAL` (0x0c, u16 destination), `OWN_PACK`
(0x0d, u32 retained-layout index), and `OWN_UNPACK_LOCAL` (0x0e, u16 source).
Pack consumes exactly the layout's ordered fields, with the last field on top;
unpack pushes fields in declaration order. Only checked complete record
layouts can carry ownership. A resource field must arrive as an owned token,
not an observation; scalar fields must have exact tags. Whole-record unpack
invalidates the source and creates every field obligation atomically.

I expose structural plus affine verification separately from runtime admission.
Normal verification consults this dataflow before deciding executable
eligibility. The standalone subset below has paired VM/native semantics; other
owned functions remain non-executable. The canonical assembler publishes only
that admitted subset. Its non-executing reconstruction API can retain other
contracts for codec/verifier tests.
I require OWNERSHIP metadata and exact resource declarations; ordinary loads,
stores and aggregate operations do not become implicit transfers.

## Standalone owned-transfer execution contract

I admit one function containing an explicit owned-transfer instruction at entry zero with zero parameters and captures, no
imports or initializer, and exactly one int, bool or u8 result. I require ownership metadata,
complete finite record layouts with only those scalar leaves or earlier record
layouts, and successful structural plus affine analysis. All locals have value
mode. I reject floats, calls, reference parameter modes, globals, collections,
aggregate results and linked ownership contracts. The same eligibility query
must govern normal verification, VM execution and native translation; there is
no separate bypass API. My supported instruction set is the non-floating affine
subset, including the four explicit transfers, scalar field observations and
branches/loops whose joins preserve exact obligations. My subsequent
[same-frame reference contract](NANOISA_SAME_FRAME_REFERENCES.md) adds root-only
shared/exclusive slots and int/bool/u8 field access within that standalone frame.
My [nested extension](NANOISA_NESTED_REFERENCES.md) adds bounded numeric paths
and checked reborrows within that frame. Caller references remain refused.

A move transfers a record pointer and clears the source slot. A store consumes
its stack owner into its exact declared slot. Pack allocates one record shell
and transfers each field in declaration order. Unpack transfers every field,
clears the source and shell fields, then releases the empty shell. I neither
copy a resource nor implement ownership as copy/writeback. Allocation failure
must leave remaining live roots reclaimable; loop execution must reclaim
consumed shells instead of accumulating them. VM and emitted native execution
must agree on scalar results, including nested transfers and balanced branches.
I require sanitizer-backed cleanup tests and unchanged prior outputs for
excluded modules before enabling this subset. Caller alias substitution,
shared/exclusive references, owned call/results and source producer admission
remain separate acceptance obligations.
