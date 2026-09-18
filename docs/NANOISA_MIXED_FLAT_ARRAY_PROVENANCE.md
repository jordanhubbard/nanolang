# I prove closed array origins before mixed execution

I record task_a2e46977551b4f628c81f719dbf30777 under mixed parent
`task_4be28fef163f42069064357639b3b5cc`, from canonical
`4fc20434c99181a9471a79c34ecd329c19f3e693`. PR802 completed descriptive child20d4.
My [descriptor contract](NANOISA_MIXED_DESCRIPTOR_VIEW.md) remains unchanged.
This is a preimplementation contract; I have not added or executed the proof.

## My result proves shape, not execution

I add a private query, provisionally nvm_analyze_mixed_float_origins, that creates
its own immutable descriptor view from the supplied immutable module. I do not
accept a caller-forged mutable view as authority. A successful owned certificate
records exact closed allocation origins, possible value tags and record-field
alternatives. It grants neither affine authority nor runtime/collector admission.
I change no public verifier, shared ownership validator, existing managed selector,
native emitter, VM dispatch or source producer in this checkpoint.

My next independent integration must conjunct ordinary structural/instruction
verification, exact affine obligations and this positive provenance certificate.
The descriptive reader continues returning PENDING_ARRAY_PROOF; it never mutates
its declaration classes or silently promotes pending records. A separate certified
mapping may later include positively proved ordinary rows while retaining every
source ordinal/global index. Resource rows never enter that mapping.

The old managed_array_shapes entry calls nvm_verify and
nvm_verify_function_max_stack. I cannot invoke that entry for this query: future
mixed verification would create a circular proof. I may reuse independent decoded
instruction/transfer utilities only after verifying they call no public verifier
or selector. Any extraction from existing code must preserve old acceptance and
outputs; my first implementation favors a separate bounded walker over changing
shared dispatch. I do not treat an old verifier refusal as permission to bypass it.

## My structural envelope is checked first

I inspect every declared function and instruction, including unused functions and
all selected shadow bodies represented in the synthetic entry graph. I require
the existing standalone, closed, acyclic value-call topology with entry0 and at
most eight functions. Imports, callbacks, captures, globals, initializers, indirect
calls, CALL_REF and external injection are unresolved in this first proof. I retain
existing scalar/owned call signature restrictions; ordinary arrays/records cannot
enter through parameters or escape as call results in this checkpoint.

I validate code ranges, instruction decoding, operands, local/constant indices,
complete branch targets, call indices and actual arity/result stack effects before
materializing per-instruction state. I independently require compatible stack
heights at joins, sufficient operands and exact return counts. Structural failures
are INVALID, unsupported well-formed instructions/shapes are UNRESOLVED, bounded
work exhaustion is LIMIT, and allocation failure is MEMORY. I cannot publish a
positive result merely because part of the graph was unreachable or unvisited.
Disconnected functions receive checked declared input states; every syntactically
present instruction must belong to the supported transfer inventory.

I bound the first implementation to eight functions, 4,096 decoded instructions
across the module, 256 locals and 256 operand slots per function, 64 total array/
ordinary-record allocation origins, 65,536 field-summary cells, and 1,048,576 total
stored value cells. Products and additions use checked wide arithmetic before
allocation. A deduplicated worklist has an explicit 262,144-pop budget; exhaustion
returns LIMIT even for a potentially valid program. These are private proof
budgets, not a new wire-format rule or a claim to decide every valid program.

## My abstract values preserve alternatives

Each value carries possible runtime tags, allocation-origin bits, an explicit
unknown flag and a separate owner/observation category with exact global nominal
identity. Origin IDs are keyed by function and decoded byte offset. Ordinary
record origins additionally retain source struct ordinal and original global
layout index. I do not substitute compact ordinal for either identity.

Scalars carry exact tags. Array origins prove declared packed FLOAT storage and
only approved scalar writes. Empty FLOAT arrays have the same declared storage
proof without inventing an element value. Record origins carry one summary per
field in declaration order; each summary retains every possible tag and child
origin. A generic ARRAY or STRUCT tag without exact origin is unresolved.

A join unions ordinary tags, origin sets and unknown facts from all reaching
predecessors. Repeated allocation at one site represents every such allocation;
I weakly union its field/write facts rather than overwrite evidence from a prior
iteration. Copies and aliases preserve origin identity. Record construction cannot
hide a child alternative by choosing one predecessor or the last field write.
Missing initialized facts remain explicit VOID/unknown alternatives, never FLOAT.

Owner tokens are not ordinary origins. OWN operations consume/produce exact opaque
owner tokens according to their checked descriptor stack effects; ordinary DUP,
field insertion and array writes cannot copy or contain them. Owner observations
stay distinct and cannot become array/record origins. Unsupported owner-adjacent
operations return UNRESOLVED instead of dropping their effects. This bookkeeping
is not a replacement for the later full affine verifier: it does not establish
borrow lifetimes, reference generations or source consumption authority.

## My first transfer inventory is narrow

I support the instructions needed by unchanged Samples/PREFIX and bounded controls:
scalar literals and copies; local load/store; direct calls/returns; branch/loop
control; assertions; existing scalar comparison/arithmetic transfers; existing
closed scalar-leaf owner pack/move/store/unpack effects; typed FLOAT ARR_NEW and
ARR_LITERAL; ARR_PUSH, ARR_SET, ARR_LEN, ARR_GET; and ordinary record AGG_PACK and
exact field AGG_GET. Stack shuffles preserve complete values, not just tags.
Dynamic pack/call effects come from exact operands and retained descriptors.
I enumerate the concrete opcodes in the implementation before qualification.
Every other opcode remains unresolved; I do not inherit a permissive default.

For first writes, array literals, append and set require exact FLOAT values and
exact FLOAT-array receiver origins, with INT indices where needed. I deliberately
refuse even the VM's supported INT-to-packed-FLOAT write coercion until separately
contracted; I do not label that well-formed case invalid. Wrong or uncertain tags,
owned children, ordinary record children and unknown element origins do not acquire
positive flat-array proof. The report records each checked write and its receiver
origin alternatives. Alias mutation updates the same abstract allocation origins.

Ordinary AGG_PACK resolves the bytecode's per-kind struct operand through the
complete source-to-global map, requires the exact field count, and checks every
field's possible tags/origins against that global declaration. A STRUCT child must
have that field's exact global nominal identity; an ARRAY child must have positive
flat FLOAT origins. Incomplete/UNKNOWN declarations do not become ordinary through
an observed constructor. Pending parents require all child alternatives proved.
Nested ordinary fields remain finite and owner-free. Managed fields inside an
affine owner, including both Bundle examples, stay unresolved: proving a child's
FLOAT content never clears the containing RESOURCE authority.

First ordinary record mutation, generic/boxed arrays, slice/pop, partial record
construction and managed aggregate call arguments/results remain unresolved.
AGG_GET preserves full field origin/tag alternatives from every receiver origin.
Owner AGG_GET/OWN_UNPACK use a separate exact owner path; they cannot use an ordinary
field proof as a transfer permission. No unmodeled destructive operation is skipped.

## My array reads keep missing values visible

Raw ARR_GET returns VOID for negative or out-of-range indices. I perform no bounds
proof in this checkpoint: a read from positively proved packed FLOAT origins has
possible tags FLOAT|VOID, including reads immediately after a known-size literal.
The result has no heap-child origins because elements are flat scalars. I never
replace missing values with zero, invent an index trap, or erase VOID based on an
assertion, comparison, or subsequent desired type.

The certificate records this actual result union. A following typed F64 operation
still requires the future admission/runtime-check contract; shape success alone
does not prove its operand exactness or make it safe to read FLOAT storage blindly.
Comparisons follow actual possible tags without pruning the VOID alternative.
Any result requirement that depends on stronger scalar typing stays unresolved or
is explicitly recorded as an unsatisfied admission obligation, never certified as
exact FLOAT. The current query certifies managed shape only, not absence of ordinary
scalar runtime errors. Existing VM error and cleanup semantics remain untouched.

## My fixed point is closed and bounded

I process the complete acyclic call graph in dependency order, then conservatively
reschedule affected functions when a shared allocation/field summary grows.
Function summaries retain all result alternatives and exact permitted signature
identities. Source-order stack preparation retains earlier arguments; a later
factory cannot overwrite an earlier origin or owner token. Prepared call values
are consumed according to exact positional transfer; a different same-shaped
nominal declaration cannot satisfy an owned signature.

Loop backedges and zero-iteration paths participate in the same monotone union.
I seed each first edge explicitly; I do not confuse an unseen state with a known
empty origin set. I require fixed-point completion before validating all observed
field/write alternatives and producing the certificate. Work/allocation failure
frees temporary states/decoded functions/view and leaves *out unchanged. The
certificate owns its report storage, but is valid only for its analyzed immutable
module invocation. I introduce no persistent module trust or cross-call cache.

## My query-only qualification precedes integration

I test packed FLOAT literals and empty arrays; local/stack aliases; field projection;
multiple exact record alternatives; construction field order; append/set through
aliases; lower-index callees and multiple helpers; and diamond/loop origins in both
predecessor orders. ARR_GET reports FLOAT|VOID for in-range, negative, out-of-range
and empty controls. No test obtains an exact FLOAT fact merely from a desired use.

I require output-preserving refusal for unknown origin, wrong declared element,
integer writes outside this proof subset, wrong nominal/field count, uninitialized
managed local, incomplete child, owner-containing ordinary container, managed field
inside RESOURCE, unmodeled owner operation, managed call signature, unsupported
instruction, cycle, invalid branch and limit exhaustion. Malformed unused rows and
unused functions cannot escape inspection. Allocation sweeps reach a successful
terminal with unchanged input/output on every failed budget.

I verify that existing shared validators/selectors retain their decisions before
and after queries. I do not execute any pending or refused module to qualify this
proof. Later verifier/runtime/native integration must separately qualify roots,
reference survival, failures and all four APIs before paired source admission of
the complete unchanged Samples/PREFIX graph. I retain historical950f evidence and
leave mixed parent4be, ownership28f2 and managed51da/488 open.

## My first production opcode inventory

My separate `mixed_float_proof.c` query admits only the following transfers for
shape analysis. This list grants no executable profile admission:

| Family | Concrete opcodes |
| --- | --- |
| Literals | PUSH_I64, PUSH_F64, PUSH_U8, PUSH_BOOL, PUSH_VOID |
| Stack/local | NOP, DUP, POP, SWAP, ROT3, LOAD_LOCAL, STORE_LOCAL |
| Scalar arithmetic | I64_ADD, I64_SUB, I64_MUL, I64_DIV_S, I64_REM_S, I64_NEG; F64_ADD, F64_SUB, F64_MUL, F64_DIV, F64_NEG |
| Scalar predicates | I64_EQ, I64_NE, I64_LT_S, I64_LE_S, I64_GT_S, I64_GE_S; F64_EQ, F64_NE, F64_LT, F64_LE, F64_GT, F64_GE; BOOL_AND, BOOL_OR, BOOL_NOT; EQ, NE, LT, LE, GT, GE |
| Flow | JMP, JMP_TRUE, JMP_FALSE, CALL, RET, ASSERT |
| Opaque owners | OWN_MOVE_LOCAL, OWN_STORE_LOCAL, OWN_PACK, OWN_UNPACK_LOCAL; LOAD_LOCAL/AGG_GET observation path |
| Ordinary managed values | AGG_PACK (STRUCT kind only), AGG_GET; ARR_NEW/ARR_LITERAL (FLOAT only), ARR_PUSH, ARR_SET, ARR_GET, ARR_LEN |

I refuse all other opcodes, even in syntactically unreachable code. This first
checkpoint additionally requires checked state for every instruction; unreachable
instructions conservatively produce UNRESOLVED. I require explicit RET/JMP
termination and refuse branches to implicit end-of-code, overlapping function
ranges and named initializers. Every declared function is seeded independently
from its concrete scalar/owned signature; ordinary managed signatures remain
refused, so no ordinary origin crosses a call boundary. Callee bodies still all
undergo analysis and exact return checking. I do not claim caller-sensitive
scalar summaries.

My owner effects currently require scalar-leaf resource layouts, including unpack
and owner results. Larger resource trees remain unresolved in this proof even
though independently qualified runtime subsets support them. Owner observations
cannot be duplicated, stored or consumed as ordinary fields; moving/unpacking an
observed local refuses. Remaining affine exit/lifetime obligations are explicit
in every successful report rather than silently considered proved.

My query first computes monotone field/state facts, then checks every reached
instruction against the completed facts. Typed scalar instructions record actual
input tag alternatives versus their required tag; their output tag describes
successful continuation only. In particular, a FLOAT|VOID read remains in its
ARR_GET obligation, and feeding it to F64 records an unsatisfied exact-FLOAT input
obligation. Generic comparisons retain actual scalar alternatives for later
runtime checking. Report fields never establish scalar-check discharge. I perform
no runtime checks or execution in this phase.
