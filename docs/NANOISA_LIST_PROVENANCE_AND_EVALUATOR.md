# My generic-list provenance and evaluator checkpoint

I record this design before extending the unqualified `7d97f824d` source draft.
My owner is `task_7b805000dfda4da386b55d4691e8c647`. I have not built or executed
that draft. My original LexerToken corpus, bounds checks and shadow deadline
remain unchanged. This checkpoint proposes implementation boundaries for review;
it does not grant a backend admission or close the full 5.1 list requirement.

## My declaration identity

I distinguish a declaration from its spelling. `env_get_struct` and
`env_get_enum` can fall back to another module after a local lookup; that fallback
is not sufficient evidence for a list element. `env_get_struct_owned` already
matches a record's owner and accepts its canonical or original name. Record
registration appends declarations and skips an already registered owner/name;
reallocation can move the table. I will therefore retain an environment-local
identity consisting of declaration kind and checked table ordinal, with zero
reserved for unknown, rather than retain a table element pointer. I will audit
every registration/reset path before relying on ordinal stability. Identities
never cross environments or enter serialized AST/NanoISA data.

I resolve an annotation with an explicit declaring owner. A qualified alias must
belong to that owner and export the requested declaration; its actual module
must match exactly, including the global-owner case. An unqualified annotation
uses the declaring module and the existing legal import binding. A canonical
mangled name may identify its existing declaration, but I do not strip a prefix
and search globally. An ambiguous or absent binding refuses; I do not select the
first same-spelled declaration. I preserve existing import visibility and aliases.
The resolver returns an identity, not a temporary change to `current_module`.

| Evidence source | Owner used for annotation resolution |
| --- | --- |
| Explicit local/global or parameter annotation | Its declaring module, captured when the symbol is installed |
| Direct function formal and return | The resolved Function's `module_name` |
| Record field annotation | The resolved containing StructDef's `module_name` |
| Explicit function-value annotation | The annotation's declaring module |
| Inferred function value | The actual callable's signature owner, preserved through copies |
| Returned function value | The declaring function's return-signature owner |
| Qualified reference | The namespace owned by the reference's lexical module |

I add narrowly scoped symbol metadata for the resolved nominal identity and
callable-signature owner, initialize it on every symbol installation, and copy
owner strings through my existing environment-owned metadata registry. Symbol
truncation/redefinition must not inherit an old identity. Signature comparisons
receive an owner for each side, including nested function signatures. Branch
results must agree on resolved identity and full callable signature; no first-arm
shortcut. Fields and direct calls recover their owners from definitions; they do
not borrow a caller-relative raw name. I do not add a general AST-address cache,
which would outlive transient checker metadata or an AST allocation.

## My value provenance before a collection boundary

An exact list receiver cannot repair an earlier mislabeled element. I apply
ordinary record declaration identity at explicit/inferred local and global
initialization, reassignment, field construction/update, direct/indirect argument
passing, and return. A value-producing block, conditional or match carries the
identity shared by its reachable result paths. I check the actual expression
before publishing destination metadata. I reject assignment between different
record declarations even if their fields happen to match; native C struct
assignment does not make those declarations interchangeable.

I keep numeric enum conversion separate. My existing `types_match` permits
numeric values at enum destinations. I do not replace that policy with universal
enum incompatibility or reject a previously valid scalar conversion merely to
simplify list checking. A destination conversion must perform the existing
numeric conversion and then carry the destination declaration. Relabeling a U8
value as an enum while retaining a U8 VM tag is insufficient.

My source evidence is incomplete for the full-width case: `type_to_tag` in
`src/nanovirt/codegen.c` maps source enums to TAG_INT, while
`generate_enum_definitions` in `src/transpiler.c` emits ordinary C `typedef enum`
with int variant values. That does not establish identical I64 conversion for
all values or target enum representations. Raw ENUM_VAL/tag-9 scalar transport
in `docs/NANOISA_LLVM_ENUM_SCALARS.md` is a different contract. I record the
I64/C-enum destination representation gap under
`task_b62f81577990c1d4b5b07809bb1cf3c2` as a separate required 5.1 acceptance:
inspect the existing enum specification, preserve its numeric policy, then
measure exact destination behavior across evaluator, C, NanoISA and reconstructed
source before proposing any representation correction. I do not invent a new
enum ABI, truncate I64 by assumption, or claim CAST_INT alone resolves that gap.

My first independently qualifiable list increment is ordinary record elements,
including LexerToken. The new implicit enum-list route must explicitly refuse in
checker, NanoISA lowering and evaluator until the enum destination contract and
parity are implemented. This is an intermediate boundary, not removal of enum
lists from the roadmap. Existing declared scalar/extern routes retain their
existing behavior and are not evidence for the new implicit enum-list route.

## My evaluator storage and lifetime proposal

`eval_call_impl` currently encodes a List_int pointer as VAL_INT for generic
lists. Push allocates a record copy, get/pop infer record status from a spelling,
set/insert store the Value integer union member, and remove returns an INT.
Those paths do not establish record parity or complete slot ownership. I will
replace only the implicit ordinary-record route with checked, typed storage.
Existing scalar lists and Token's explicit native wrappers remain separate.

I propose an environment-owned registry of record-list handles. Each entry owns
its List_int storage and resolved element identity. The existing integer handle
is an evaluator implementation detail, not a source INT conversion. Before any
operation I find the entry by handle, check live state and exact identity, then
check arity, argument kinds and bounds. I do not dereference an arbitrary integer
as a record-list header. The List_int backing shape does not establish iteration parity: my checker
already accepts TYPE_LIST_GENERIC iteration, but my evaluator has only explicit
INT/STRING list branches. I must add the typed record iteration consumer, recover
its receiver identity before installing the loop symbol, and copy each element
through the result snapshot path. I preserve existing loop mutation semantics
only after comparing C and NanoISA lowering; I do not assume fixed-length versus
live-length iteration. Loop symbols must retain exact element provenance. Aliases share one entry. Explicit free releases elements and vector
storage but retains a closed handle identity until environment teardown, avoiding
accidental identity reuse. This does not promise safe use after free in compiled C.

Each occupied slot owns a record snapshot: recursively cloned record fields and
copied/retained strings, with arrays and other reference fields retaining the
existing `create_struct` borrowing semantics. I reuse one checked clone/discard
pair; `discard_literal_record` documents that ownership boundary. I reject a
resource-containing element through the existing ownership checks. I do not
claim deep ownership of arbitrary arrays, lists, functions or foreign resources.
All allocation sizes and growth arithmetic are checked. Clone/growth failure
cleans staged allocations and leaves length and occupied slots unchanged before
the existing evaluator failure path. I do not silently substitute VOID success.

| Operation | Ownership transition after argument evaluation and validation |
| --- | --- |
| new | Publish the registry entry only after its storage is complete |
| push/insert | Stage a complete owned snapshot before growing or shifting slots |
| set | Stage a new snapshot, replace the slot, then discard the old snapshot |
| get | Publish an independent result snapshot; the slot remains owned |
| remove/pop | Stage the result snapshot before mutation, then discard the removed slot |
| clear | Discard each occupied snapshot once and set length to zero |
| free | Clear slots, release vector storage, mark the registry entry closed |

I propose retaining list-produced result snapshots in a separate environment-owned
result arena. They are borrowed Values until the environment is destroyed;
binding installation and function return copy them through the existing value
copy boundary. This keeps an earlier argument/result valid when a later argument
mutates, clears or frees the source list. The arena must never also register a
symbol-owned clone. Its teardown uses the same recursive snapshot discard and
runs after symbols no longer refer to its results. I will audit both function-call
entry paths, return copying, `repl_eval_node`, field access and every temporary
cleanup before implementing that ownership rule. A consumer that frees a borrowed
result would require an explicit ownership correction before qualification.

This proposal retains memory proportional to cumulative list-result snapshots
until environment teardown; it does not claim expression-bounded interpreter
memory. Compiled NanoISA scratch roots still clear after each expression. My REPL
keeps its Environment alive across commands and shows each result before teardown;
a returned borrowed evaluator result may not outlive that Environment. I will not
change the public Value/StructValue representation or retrofit an uninitialized
ownership flag into every historical constructor. If the audit finds an existing
API requiring independently owned escaping results, that boundary must clone
explicitly or receive a separately reviewed lifetime design.

## My source and acceptance order

I first submit the owner-aware resolver, symbol/callable provenance and complete
record boundary checks, including explicit refusal of the new enum-list route.
I then submit the evaluator registry/clone/result-lifetime implementation and its
complete consumer audit. No qualification starts from either partial draft.
Fixtures follow source review and retain the original token_value_bytes program.
They cover same-spelled declarations in two modules, qualified aliases, nested
callable signatures, both module passes, every boundary above, ordinary record
copies and string lifetime, old new/get/push/set plus new insert/remove/pop,
argument order, independent results after set/remove/clear/free, allocation-prefix
cleanup, and environment teardown. Accepted record programs must agree across
evaluator/C-seed/Stage1/Stage2/NanoVM/native routes; unsupported target profiles
must continue explicit refusal. Enum-list parity and I64 enum destination behavior
remain required subsequent checkpoints, not inferred from LexerToken acceptance.

### My first consumer-audit findings

I verified that ordinary field access already copies a string field with
`create_string`; returning an arena-backed record therefore does not lend its
owned string directly to a releasing local. Record binding and reassignment use
`create_struct` copies, including nested records. I must not register those
symbol-owned copies in the proposed result arena. `repl_eval_node` delegates to
`eval_statement`; the REPL displays a result while its persistent Environment is
live. Those facts support, but do not complete, the lifetime audit.

My two function call implementations differ: `eval_call_impl` copies returned
records (currently only direct string fields), whereas `call_function_at` copies
only returned strings. Before claiming nested-record list result parity I must
route both through the reviewed record snapshot ownership boundary, including
normal returns, enclosing-handler returns and parameter truncation. The existing
record cleanup paths are not a blanket leak-free interpreter claim. I also found
the accepted generic-list iteration gap above; it stays in this task and cannot
be hidden by an insert-only fixture. No old faulty path was executed.

## My first owner-aware source checkpoint

I add environment-local declaration identity queries and environment-owned
nominal/callable owner strings on Symbol. The ordinal is a temporary comparison
result; I recompute it from retained annotation context rather than cache it in
an AST or persist it across environments. Normal symbol insertion initializes all
fields. Header constants already zero their entire Symbol. Both lowerers preserve
these contexts when replaying checked match bindings. Inferred local bindings
retain their origin owner across repeated checks of the same source declaration.

My expression query follows identifiers, ordinary fields, direct/indirect results,
effect results, implicit list results and agreeing branch results. Exact record
checks now precede destination publication at the existing local/global/field/
call/assignment/return boundaries. Callable comparison carries a separate owner
for each signature, including nested signatures and branch alternatives. Generic
record iteration derives its element declaration from the receiver expression.
Unknown expression evidence refuses instead of inheriting an unrelated name.

Generated uppercase list declarations previously had a global module with a raw
T signature. I now resolve their concrete record first and retain its annotation
owner in their generated Function. These generated extern declarations have no
private user body; they are marked public for the existing callable access check.
This particular choice needs source review alongside same-spelled imported
records and actual user declaration precedence. I do not claim that generated
specialization spelling alone establishes an owner. Existing specialization
registration is keyed by concrete spelling; ambiguous or colliding declarations
must not become accepted through that legacy key. Qualification must exercise
this boundary before any module compatibility claim.

I leave scalar enum numeric compatibility unchanged and refuse the new implicit
enum-list route in the checker, mutation lowerer and evaluator fallback. The only
evaluator changes in this checkpoint are that explicit refusal and preservation
of actual function declaration precedence. Typed record-list storage, both return
copy paths and generic evaluator iteration remain the next reviewed source
checkpoint; I have not implemented the arena yet.

This is an unqualified source checkpoint. I checked patch whitespace and inspected
new symbol initialization and copied-binding sites; I have not compiled or run a
fixture. Full consumer audit, legal import/alias controls, generated declaration
collision controls, inferred metadata replay and every accepted ordinary record
expression shape remain required fixture/source review boundaries. Missing
metadata refusals are not a permanent reduction of my full list scope.

### My source-review corrections before the next checkpoint

I retain `0ec5b0b32` as an unqualified draft. Review found that two unresolved
TYPE_STRUCT signature names could fall through to raw spelling equality; I will
instead classify real record, enum, union or opaque declarations explicitly, and
refuse an unresolved formal/concrete name in a runtime callable signature. Enum
scalar destination conversion remains unchanged. I will also preserve the prior
indirect argument compatibility result when adding nominal checks; a legacy
exemption cannot overwrite a failed base-type check.

I will compare complete signatures for function-valued IF/COND/MATCH branches
before selecting an annotation owner, retaining a representative signature and
its owner together. Mere same-owner agreement is insufficient. A signature
borrowed from a different arm must not be paired with that representative owner.

I withdraw the draft's generated `is_pub=true` change. Generated list functions
retain their old global/private lookup behavior. A side record on their existing
GenericInstantiation will identify their exact function-table ordinals and the
resolved element identity; signature-owner queries use that association without
changing source visibility or exported module lists. Concrete specialization name
collisions between distinct identities refuse before adding a conflicting row.
Actual user declarations remain selected over generated helper declarations.
These corrections require source review before any fixture execution.

My corrected source now classifies runtime callable nominal signatures as an
exact record, enum or union identity, or the same existing global opaque entry.
Unresolved generic/formal spellings refuse here; scalar enum assignment conversion
is unchanged. Complete TypeInfo facts still compare after canonicalizing the
resolved declaration name in temporary views. I preserve the original indirect
argument match with conjunction.

My callable view owns a signature copy and keeps its annotation owner beside it.
IF/COND/MATCH compare every branch's complete view; inferred binding publication
and direct invocation of a function-valued expression consume that paired view.
Every temporary copy is released after comparison or transferred into an owned
inferred TypeInfo. No borrowed signature is paired with another branch's owner.

My GenericInstantiation side record contains the exact record identity and four
one-based generated function indices. All three production instantiation
constructors already zero their structures; the new scalar fields need no
separate destructor. Generated helper visibility is again `is_pub=false` with
`module_name=NULL`, and my signature-owner query consults only exact associated
function slots. Lookup prefers a real declaration over an associated generated
helper. Registration now reports success/failure, and every production caller
uses the checked wrapper. A same-spelling instance with a different declaration
identity, or an overlong generated name, refuses before publishing a new row.
Ordinary record renaming/import compatibility still needs the retained source
and fixture acceptance; this correction does not rename an ABI to hide a collision.

### My recursive annotation correction before evaluator work

Independent review of `94bbeded8` found that nested TypeInfo leaves still reached
ownerless equality: arrays, flattened tuples/rows and concrete generic arguments
could compare identical unresolved spellings or distinct same-spelled module
records. I record this as an accepted-flow blocker without executing a malformed
program. I will replace both legacy fallback calls in the owner-aware signature
checker with one bounded recursive annotation comparison. It will check every
existing TypeInfo field and resolve every nominal leaf under its side's owner,
including arrays, generic arguments, tuple/row fields and nested callable
signatures. Missing nominal declaration evidence refuses even when both pointers
or strings are identical. Scalar enum destination conversion remains unchanged.

The recursive correction now uses no ownerless equality call from the checked
signature path. It compares every TypeInfo child, negative/count mismatches,
flattened tuple/row leaf type and name, row openness/variable/field labels,
quantified-variable labels, opaque registry identity and nested callable
signature. The same non-null node pointer is still traversed and resolved; cycles
or excessive depth refuse at the existing bounded comparison limit. Parameter
and return tags must agree with their complete annotations, and parallel legacy
nominal names may not contradict those annotations.

I normalize only the two existing List<T> encodings and named declaration aliases
in temporary views. Explicit element subtrees remain fully compared. Concrete
union argument counts must match the resolved declaration, and every argument is
recursively checked in its original owner's context. An unresolved nominal or
uninstantiated formal leaf refuses. Quantifier/row labels are structural metadata,
not evidence that a concrete nominal declaration exists. Legacy flattened metadata
that cannot represent a nested callable/array annotation does not become proof by
matching another equally incomplete shape; completing such source transport
remains required rather than silently admitting it. No enum ABI, opcode, evaluator
storage or test expectation changes accompany this correction.

### My evaluator consumer correction before qualification

My storage audit found two additional consumers of the same lifetime contract.
Both uppercase generated-list call paths currently bypass the lowercase storage;
I will use their exact registered generated-function association and the same
registry, preserving actual user declarations. My C and NanoISA loops both
capture the initial length, so my record evaluator loop will do likewise and
check each current index rather than adopt the scalar evaluator's live length.

Borrowed record-field assignment currently stores an incoming Value directly.
That can install a borrowed list-result record/string inside an owning record.
Before adding recursive cleanup I must stage a record/string copy, replace only
after success, and discard the old owned field. This is part of the record
consumer audit; it does not add ownership of array/function reference fields.
My block-result path must preserve returned nested records before releasing
owned local records. My public call boundary must copy any arena-backed result
that could otherwise escape its Environment. I have not executed these old paths.

### My binding retirement rule

Before changing truncation, I record a single ownership transfer. I allocate a
retirement entry while the symbol still owns its record. Failure leaves that
symbol and output unchanged and follows the evaluator's fatal allocation path.
After successful registration I clear the symbol Value, then release its name
and truncate its slot. I reject duplicate registration. Environment teardown
releases each retired record through the same recursive discard exactly once.
This retains pointers already borrowed by a tuple or pending callback within the
Environment's lifetime. It does not make those values safe after teardown.

My cumulative storage now includes retired binding records as well as result
snapshots and live list slots. I do not claim expression-bounded reclamation or
constant memory per loop. I preserve old function/reference-field borrowing and
do not release function values at scopes that did not previously release them.

I discovered two distinct escape requirements and filed them before implementing
either: `task_c3e8419dec9b8ccbcd6e1f99e5dbe5ec` for tuple/record result ownership,
and `task_f8b5ecacb7e4bcc1542712b4652308ad` for deferred Environment leases. Their
proposal is in `NANOISA_EVALUATOR_ESCAPE_LIFETIME.md`. My list storage source is
reviewable independently; it is not yet a qualification candidate without these
accepted-flow lifetime boundaries.

### My evaluator storage source checkpoint

My implementation lives in `src/env_record_lists.inc`, included by `env.c` so
standalone Environment users gain no new dependency on `eval.o`. `env.o` has an
explicit include prerequisite in addition to the existing generated dependencies.
Environment zero-initialization starts both lists empty; teardown discards my
registry after symbols and before definitions/checker metadata.

| Source API | Ownership and failure |
| --- | --- |
| `env_clone_record` / `env_discard_record` | Checked recursive record/string snapshot; output unchanged on failure; other reference fields borrowed |
| `env_record_snapshot` | Staged clone becomes an Environment-owned result only after complete success |
| `env_retire_record` | Unique owned record transfers after node allocation succeeds; caller clears the symbol only after true |
| `env_record_list_identity` | Lookup by integer equality before dereference; live entry and output required |
| `env_record_list_apply` | Exact record identity/live state/arity/kinds/bounds; stage before mutation; unchanged output on failure |
| `env_generated_list_element` | Existing generated-function ordinal association, now shared with evaluator dispatch |

My slots use embedded List_int headers in nonmoving registry entries. Vector
reallocation uses a temporary pointer and checked INT_MAX/size bounds. `free`
clears records/vector and retains the header as a tombstone; no arbitrary integer
becomes a dereferenceable handle. `with_capacity` and `is_empty` are matched as
whole suffixes, not split at their final underscore. Both generated uppercase
paths use the same registry; actual declared functions keep precedence. The
ordinary literal path copies the selected canonical name before evaluating
fields, which may move definition tables.

`get`, `remove`, and `pop` publish independent snapshots; mutation discards only
slot-owned records. Both function return paths copy records into the result arena
before scope truncation, preserving enclosing-handler return markers. Public
`call_function` copies a top-level arena record into caller-owned storage.
Tuple-containing escape remains the explicitly unimplemented dependency above.
Scope truncation retires owned record bindings exactly once. Field replacement
clones records/strings before installing them and discards the prior owned field.
No symbol-owned clone is also registered while that symbol owns it.

My static array/dynamic-array record insertion, slicing and reading paths already
copy via `create_struct`; they now use the same checked clone. Array/list/function
reference fields remain borrowed. Tuple construction and deferred spawn were the
exceptions found in this audit, recorded in separate preimplementation tasks.
I preserve existing function-value cleanup scope; this is not a blanket fix for
all interpreter allocations. The current clone bound is 128 record nodes along
a path, with cumulative list results and retired bindings retained until teardown.
No build, source execution, fixture qualification or historical reproduction was
performed for this checkpoint. Only source review and whitespace checks apply.

### My synchronous argument staging correction

Independent review of `6e8042182` found a pending-argument ownership gap before
execution: FIELD_ACCESS may lend a nested record, and a later argument's field
replacement can release it before parameter cloning. I record the correction
under my existing list task before implementing it. Direct, callable-expression
and module-qualified calls must snapshot each by-value record/tuple argument
immediately after evaluation, before the next argument. Explicit borrowed formal
parameters retain their actual source identity and are not silently copied.
Aggregate fields/elements need the same left-to-right staging. This is a source
correction, not a reproduction or an assertion change.
