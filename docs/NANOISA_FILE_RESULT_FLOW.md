# My private File and Result transfer-state foundation

I prepare task6739329b1fcde420706bb5a10f481919 under72556/6931 from canonical
PR833 at1b54ed96d. My qualified private File lifetime core and exact nominal
transport establish different facts. Neither checks a function's owner flow.
This contract adds the first private transfer-state foundation, not an executable
profile. Production review precedes fixtures. Full host dispatch, bytecode
refinement, paired source/shadows and File/Socket/GPU acceptance remain open.

## My current boundary

| Current code | Established fact and missing dependency |
|---|---|
| `service_file_nominal_plan.c` | Exact eight catalog identities, global/per-kind maps, five imports, RESOURCE flags and syntactic function/local descriptors; no initialized local or function-flow fact. |
| `nsi_file_values.c/.h` | Checked invocation/slot generation, moves, exclusive borrow epochs, affine OpenResult arm extraction, scalar Results and cleanup; no bytecode authority. |
| `affine_state.c`, `affine_bytecode.c` | Existing record/reference states do not establish opaque File or affine UNION semantics. I do not relax their tags. |
| `isa.h` UNION_TAG/FIELD and AGG_TAG/GET | Generic consuming value operations do not specify affine Result observation/refinement and payload transfer. Ordinary integer equality does not itself refine an owned Result. |
| `service_bindings_module.c`, verifier/VM/native guards | Non-executing metadata retention only; service claims still refuse before ordinary/mixed selection. |

My private API consumes a valid immutable in-memory NvmModule and resolves facts
itself. It does not accept a caller-owned certificate, arbitrary catalog, boolean
permission flag, substitute layout or trusted list of owner declarations. I call
the existing exact nominal query, then copy the bounded function/local descriptors
from the validated ownership payload. A caller cannot certify a module by making
some successful state API calls. These calls do not decode or validate CODE.

## My exact private states

I own an opaque declarations object and function states. They retain no caller
module pointers. Descriptors preserve function indices, physical local indices,
exact tag, catalog/global/per-kind identity and parameter mode. Bare STRUCT or
UNION with NO_INDEX, unknown nominal rows, other resource families, arrays,
strings, references with noncatalog referents, globals/upvalues and callbacks
are unresolved in this first File state family. I do not silently turn an
unknown declaration into VOID, ordinary storage or File.

I distinguish these value facts:

- Exact scalar INT, BOOL and VOID, with initialization separate from type.
- Copyable catalog FileError and ReadByte records, and the four exact scalar
  Result types. Their fields have the immutable catalog shape. Copyable means
  no affine obligation; later runtime allocation/root cleanup is still required.
- A live File owner, with a unique symbolic ownership identity.
- A live affine OpenResult, whose arm is unrefined, Ok, or Error. Both arms
  retain a linear Result obligation until take/drop; Error is not a copy license.
- A live exclusive File reference bound to an exact owner local or a borrowed
  formal, region and reference identity. It is not an ordinary STRUCT value.

The state has a value stack, exact local declarations/initialization, owner
locations, reference slots and region state. An owner appears at exactly one
owned location. A reference observes that owner without acquiring ownership.
Symbolic identity is verifier bookkeeping, not an invocation token or host right.
I check ID exhaustion before publication and never wrap/reuse a live identity.

Entry parameters receive facts only from their exact validated declarations:
mode0 File/OpenResult parameters are owned formals; mode2 File parameters are
exclusive borrowed formals whose owner remains outside the callee. Other borrow
modes are unresolved. Nonparameters begin uninitialized. Scalar nominal records
and scalar Results are distinct from File/OpenResult despite shared wire tags.

I bound this private API to64 function declarations,256 physical locals per
function,256 stack values and256 symbolic owned locations per state. I retain
the existing65,536 layout limit. Before allocation I check every multiplication
and total requested owned storage against16MiB per declarations object or state,
including copied declarations, nominal maps, references and scratch. LIMIT is
distinct from allocator MEMORY failure. These are private analysis limits, not
new ISA limits. The runtime's64-slot File capacity remains a separately checked
failure condition; static state success does not promise available host slots.

## My transfer operations

Private logical operations have named C API semantics below. I assign no new
wire opcode or assembler spelling in this checkpoint.

1. Exact scalar/copyable pushes, loads, stores and drops preserve declared type
   and initialization. Generic DUP/LOAD/STORE of File/OpenResult is refused.
   Generic record/union construction, casts, integer conversion and field access
   never create or expose File. Catalog error/read records and scalar Result
   construction need exact fields/variants; affine constructors remain refused.
2. Owned move/store transfers one exact File/OpenResult. Source becomes empty;
   destination must be uninitialized/empty and match the exact nominal type.
   Overwriting a live owner is refused. POP is not an implicit affine destructor;
   the private owner-drop operation records one cleanup obligation explicitly.
3. Exclusive borrow requires a live File in an eligible local and an available
   reference slot/region. While held, owner move, drop, close and conflicting
   borrow refuse. Ending a reference invalidates that reference only. A borrowed
   formal cannot be moved, dropped, closed or returned as an owned File.
4. A service transfer resolves the exact mapped import and immutable method.
   Temp has no owner input and publishes an unrefined affine OpenResult. Write,
   rewind and read require an exact live exclusive File reference and preserve
   its owner on both Ok and Error, returning the exact scalar Result. Close takes
   an unborrowed File owner once and returns CloseResult; both outcomes consume
   the owner. API refusal changes no state. Accepted-call runtime failure and
   Result.Error remain different eventual runtime paths.
5. Service transfer records method rights, byte-domain and checked-runtime
   obligations. It does not infer rights from a STRUCT tag or grant a host
   context. Write's INT input still needs the0..255 check before stream access;
   a rejected range preserves File and produces the specified Error. Every later
   service call must check actual binding, invocation, liveness, rights and borrow
   epoch. This state API neither performs nor discharges those host checks.
6. Explicit Result refinement operates on an initialized exact Result local and
   creates separate Ok/Error successor states. Observation retains the Result;
   it never duplicates the affine payload. A known incompatible arm is marked
   unreachable, not relabeled. Take requires the matching refined local: Ok from
   OpenResult consumes that Result and transfers a File; Error consumes it and
   produces FileError. Each scalar Result produces its exact catalog payload.
   A refinement cannot survive replacement/move of the source local. Generic
   equality, UNION_FIELD or AGG_GET cannot authorize an affine take.
7. An internal-call transfer resolves the actual callee descriptor and exact
   argument order/modes. Owner arguments leave the caller once; borrowed arguments
   keep their caller roots and cannot escape. It records a pending call obligation
   and exact declared result fact. A synthetic result is not evidence that the
   callee body returns correctly: later whole-module validation must check every
   callee and complete exit. A returned File/OpenResult has one owned destination.
   Signature mismatch or failed staging preserves the entire caller state.
8. Complete function exit requires the declared exact result, no leftover stack
   values, no owned local/temporary beyond the one transferred result, and no
   unclosed local borrow/region. Borrowed formals remain caller-owned and cannot
   be consumed by callee cleanup. Explicit drop, consuming service call or checked
   internal transfer must account for each owner; an early return cannot leak it.

The first API does not model reborrows, reference-valued results, recursive call
summaries, tail calls, ordinary heap fields or public owner escapes. Those remain
explicit unresolved dependencies, not implicit successful transfers.

## My joins, transactions and output lifetime

Clone/refine stage their complete allocations before publishing outputs. Both
refinement outputs remain untouched if either clone fails. Every mutating
transfer validates operands, bounds and space before consuming anything; failure
preserves the state and all output facts. No error path invents a missing owner
or rolls back a host operation: this checkpoint performs no host operation.

Join requires the same declarations/function, stack height, initialized types,
owner locations and identities, and exact active reference/region relationships.
It can conservatively widen an otherwise identical Result arm to unrefined; it
cannot merge a live owner with an empty slot or discard one alternative's owner.
Different owner identities are unresolved in this first API rather than silently
aliased. This permits loop states that conserve the same owner and balance local
borrows; it does not claim a complete fixed-point module analyzer. Unknown,
unreachable and uninitialized remain distinct. Getter failure preserves outputs.

I distinguish OK, INVALID, UNRESOLVED, LIMIT and MEMORY. INVALID covers malformed
metadata or internally contradictory state requests; UNRESOLVED covers operations
outside this bounded model. Allocation failure and declared limits never publish
partial declarations, states, clones, refinements or summaries. Free is safe on
NULL and releases each allocation once. A successful state and its getters remain
valid after the input module is destroyed.

## My ordered implementation and acceptance

1. Review the declaration/state API and immutable fact ownership, then implement
   exact checked construction, scalar/owner moves, references and exit checks.
   Send production before preparing or executing fixtures.
2. Review service transfer, explicit arm refinement/take, internal-call obligations
   and conservative join/clone transactions. Shared validators, selectors, codec,
   VM/native execution and source producers remain unchanged.
3. Freeze ordinary private C controls on Linux/Darwin, with selected actual
   compiler/library hashes, strict GCC/Clang sanitizer qualification as applicable,
   retained allocation-prefix failures and complete output/state atomicity checks.
   Exercise exact nominal permutations, missing identities, all method outcomes,
   moves/aliases/held references, both Result arms, wrong-arm take, empty/live
   joins, zero-iteration loop-shaped joins, helper argument/result obligations,
   early-return leaks, limits, clone/refinement partial allocation rollback and
   input lifetime independence. No service bytecode or File host action executes.
4. Preserve the qualified private core and nominal query/transport regressions.
   Independently review the later bytecode/refinement encoding and bounded CFG
   driver, including complete all-function/call-graph checks and public entry
   restrictions, before it may issue a module flow certificate. Public entry
   means every host invocation/export route, not merely a function named main.
5. Only a later reviewed combined verifier plus VM/native host-granted runtime
   may select execution. It must discharge every pending call/runtime obligation,
   preserve all uncommitted operand/result roots, bind the exact invocation context,
   retain first and secondary cleanup errors, and enforce scalar-only public
   inputs/results with no escaping File/owned Result. Paired generated binding
   publishers and complete original source/shadows follow that acceptance.

I do not close72556/6931, d03c, ed702 or the full release from this private state
checkpoint. Existing non-service profiles retain their current behavior; every
service module remains rejected by public consumers throughout this work.
