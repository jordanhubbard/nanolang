# My complete native ordinary-list specializations

I record this design under task_2752a051dce443d0ada447c46b667561 before changing
native production. The strict checker checkpoint 0e72a33a6 does not complete
native lists. My source audit finds these independent existing gaps:

* In src/transpiler.c, generate_list_implementations emits only new/push/get/set/
  length. Its get is unchecked and set silently ignores an invalid index.
* In src_nano/transpiler.nano, generate_list_for_type emits only new/push/get/
  length. Its push publishes doubled capacity before realloc succeeds; get
  reads without checking bounds. Its discovery walks annotated local lets and
  selected statement blocks, not every accepted list-producing expression.
* nb_rewrite canonicalizes colliding record/List annotations, while native call
  emission still spells raw list_Item_* names. The C iterative emitter likewise
  treats a generic-list spelling as a runtime symbol before ordinary mapping.
* The selfhost generic call path emits ordinary C arguments without establishing
  left-to-right receiver/index/value ordering. The C-seed iterative emitter
  already uses build_ordered_call_args; I preserve that actual staged route.

I have not executed an invalid generated program to reproduce these findings.
The original imported/mutation corpus remains required. I do not start it just
to rediscover a missing operation already demonstrated by source inspection.
Fresh checker/bootstrap work at 0e72 is independent and stays attributed there.

## My shared semantic boundary

I implement the already checked ordinary-record catalog: new, push, get, set,
insert, remove, pop, length, capacity, is_empty, clear and free. Remove/pop/get
return the exact record by value. The existing int/string providers remain
separate and retain their additional with_capacity operation. I do not invent
record with_capacity, enum-list ABI, owning-resource support or new NanoISA
admission. The required enum parity task remains open.

My generated generic storage retains its existing data/count/capacity layout.
Record values retain the native record representation: nested record fields
copy by value; string and collection fields follow their existing native
reference representation. Clear/free retire list slots/storage, not arbitrary
pointees of those fields. This change does not establish a new deep-copy or
reference ownership ABI. I must inspect the actual native managed-string/root
and record publication paths before changing any element cleanup policy; I do
not infer evaluator arena behavior for native C.

I use the existing src/runtime/list_capacity.h checked capacity/length helpers
for both producers, with its real include dependency recorded. Before storage
mutation I check a nonnull receiver, representable count/capacity, and the full
source-width INT index. Get/set/remove require 0 <= index < count; insert permits
index == count; pop requires count > 0. I compare before narrowing to the
existing int storage width. For scalar provider calls I likewise validate an
index or with_capacity argument at source width before its existing int ABI
narrows it; I do not silently change the provider ABI. New and growth must check
size multiplication and
allocation before committing a pointer, capacity or count. A failed constructor
frees its partial storage. A failed growth retains every previous byte and
metadata field. Shifts use the established record representation and checked
sizes; remove stages its result before shifting; pop stages before reducing
length; vacated slots are cleared without inventing a destructor.

I propose the existing checked list runtime's nonzero fail-fast policy for
bounds, invalid storage and OOM: a precise diagnostic and exit(1), before any
successful mutation. This deliberately replaces generic inline silent failure
or NULL-on-OOM behavior; source review must approve that observable failure
contract. It is not recoverable allocation acceptance. The evaluator's checked
clone/arena allocation sweeps remain separate and unchanged. Free(NULL) retains
its existing harmless cleanup behavior; no other operation accepts NULL.

## My exact call and identity boundary

I distinguish lexical callable values, declared functions and externs before
selecting a list intrinsic. A spelling alone never overrides any of them.
Qualified calls use the actual bound declaration, never a bare suffix fallback.
For a selected intrinsic I resolve its encoded element using the call owner's
nominal binding and one actual record declaration. I preserve the established
unique extern-record namespace, reject competing definitions and retain the
same resource/unknown refusals as the checked source contract.

The generated specialization key is the exact canonical declaration, not a
raw spelling, receiver layout coincidence or current global name. For the
self-hosted producer I use the existing #type bindings/nb_type result only after
actual declaration validation. For the C producer I audit existing checked
NominalIdentity and registered generic-instantiation origin facts before choosing
its corresponding canonical emitter key. I do not repurpose callable generic
specialization metadata or return-only annotations as unchecked list authority.
Any new retained per-call metadata needs its constructor/copy/destructor audit
in that source checkpoint. This C metadata choice is an explicit prerequisite,
not an already implemented or settled ABI.

I retain C-seed build_ordered_call_args and give the selfhost selected list path
the same ordering contract. I stage the selected callee/receiver, index and value
once, left to right, in
collision-free expression-local temporaries. The runtime observes length after
all argument effects. A later argument may mutate the same list, and the staged
receiver still denotes the original object if a variable is rebound. Captured
record results remain valid by their established representation after movement.
I preserve VOID versus record/scalar expression results and existing enclosing
control flow. Declaration/callback routes retain their own signatures and do
not accidentally call a generated intrinsic helper.

I discover every accepted specialization from actual declarations and checked
expressions, including parameters/returns/fields, inferred locals, nested block/
branch/loop/call contexts and discarded constructors. I emit forward list types
before record fields that refer to them, then implementations after complete
record definitions. Exact keys deduplicate definitions. Runtime-provided schema
lists are identified by their actual known declaration/provider identity, not
an arbitrary AST or Compiler name prefix. Unsupported names, unrepresentable
identifiers or specialization collisions produce a checked failure before a
new output is published. I do not silently truncate a discovery list.

## My review and acceptance sequence

1. Finish the C identity/metadata and native element-lifetime audit above; submit
   the concrete selected representation before implementation if it adds fields.
2. Implement complete helper discovery/emission and exact call selection for
   both producers, with meaningful Nano shadows and all ownership/error paths.
   Submit the whole production checkpoint before builds or new execution.
3. Add focused generated-C controls for every operation, exact returned records,
   underscore names, colliding imported owners/aliases, schema-looking ordinary
   record names, declarations/callbacks, inferred/discarded constructors and
   nested record/List fields. Retain all original eighteen source methods.
4. Test corrected-only bounds and allocation failures in bounded subprocesses,
   including INT64 extrema before narrowing, zero/first/growth storage and
   unchanged prior state under injected allocation failure. Prove staged order
   with the existing trace and receiver-rebinding controls, not C argument order.
5. Run fresh C-seed/Stage1/Stage2 native outputs, evaluator and already supported
   NanoISA routes with exact provider maps on both hosts. The complete source
   matrix, original LexerToken program and unchanged make test remain required.
   Preserve every first terminal; no deadline or assertion is weakened.

I keep full list, enum parity, bootstrap/fixed-point, whole-Make and release
criteria open. This design is not source qualification or backend admission.

## My concrete C identity and lifetime audit

My C binder already writes canonical record names into StructDef.name while
retaining original_name and module_name. env_nominal_identity selects the exact
owner/declaration; env_nominal_name returns that declaration's canonical name.
GenericInstantiation.list_element already stores the checked kind/ordinal, and
env_register_list_instantiation refuses a same-spelled different identity.
I will reuse these facts rather than add an Environment-borrowed AST pointer.

For a successfully selected ordinary record-list call I will transactionally
canonicalize the existing owned AST_CALL.name to list_<canonical>_<operation>,
and retain canonical return_struct_type_name for element-returning operations.
Every selected operation must register/validate its exact specialization, not
only new. I allocate replacement strings before publication, preserve the old
call on failure, and never rewrite a declared/extern/callback route. The
emitter checks the actual registered list_element and canonical declaration,
not a current-owner guess from the raw source name. It rejects a real callable
or generated-C symbol collision rather than silently redirecting it. Existing
compiler metadata allocation limitations remain separate from checked runtime
growth; this is not an all-checker OOM-recovery claim.

This uses existing call-name ownership: create_node zero-initializes it,
parser clone_ast_node duplicates it, free_ast frees it, and the canonical
main pipeline finishes typechecking/shadows before its optional PGO transform.
It adds no C AST field, borrowed Environment lifetime or portable schema field.
All downstream consumers keep the same selected operation and exact canonical
element: evaluator and NanoISA resolve canonical declarations, while native
emission uses the existing exact specialization registry. I must add source
controls for repeated checking and all those consumers before qualification.
The generic C dispatcher also needs whole-suffix parsing for is_empty; the
evaluator already has it. Task_bff42f7451f243b39149c04408df05f8 records that
specific source mismatch. Unsupported NanoISA operations remain refused.

The C-seed scope tracker explicitly does not release TYPE_STRING locals. Native
managed strings start with ref_count=1; plain C record assignment does not
invoke gc_struct_set_field or gc_struct_free. The selfhost emitter likewise
uses the native record layout, not evaluator Value/arena graphs. Thus I preserve
record-by-value and existing pointer-field representations without adding
recursive frees/retains to slot movement or free. List clear/free reclaim only
their own storage. This is not expression-bounded managed-memory reclamation,
foreign-handle ownership or a deep-copy claim. Opaque/provider ownership still
follows its existing contract; no list operation becomes an owner of arbitrary
foreign pointees. The copied record and string controls remain required.

I also found an independent existing PGO issue: clone_node bitwise-copies the
owned AST_CALL.name and calls it a shared constant, while free_ast frees it.
Task_ec4cdac029ef4d20a3a54f046ca10a79 retains that static clone/destruction audit;
no old memory-faulting program was run. This list repair adds no new field to
that ownership graph, and it does not claim optional PGO graph qualification.
The full compiler/release requirement still includes that repair.

For the native bodies I propose one shared typed C macro/header containing the
checked ordinary-list operations. Both producers emit the exact forward typedef,
complete record definition, then the macro specialization; this avoids two
independent copies of growth/bounds logic. Existing schema/int/string runtime
providers keep their own definitions. Provider selection must use the actual
known catalog and declaration, not the old broad AST/Compiler prefix heuristic.
The selfhost call path gets expression-local argument staging and exact owner
mapping; C-seed keeps its existing ordered argument machinery.

## My selfhost emission collection order

I retain an emission-local list of canonical record keys, reset at each
transpile_parser_mode entry, like my existing function-type emission state.
It is a code-generation worklist, not checked runtime authority or a cache
across compiler invocations. A selected intrinsic registers only after lexical
callable/declaration precedence and exact owner binding select it. Declared
parameter/local/return/field annotations contribute their already bound types.
I collect function bodies, selected shadow bodies and global initializers
before emitting list forward declarations, complete record definitions and
specializations. This includes inferred/discarded constructors without selecting
an intrinsic merely because an unrelated call has a list-like spelling.

Direct helper shadows reset this worklist when they inspect it. Failed emission
returns no product through my existing emission-error latch; the next invocation
resets the worklist. The full source checkpoint must demonstrate this order and
include consecutive-parser, selected-shadow and declaration-shadow controls.

My scalar C list providers are registered by register_builtin_functions, not
by env_function_is_builtin's separate cache. Their actual selected registration
has a null body/params/shadow/module/alias and is_extern=false. Every parsed
ordinary function has a body and every parsed foreign declaration has
is_extern=true. I use that complete registration shape only after exact scalar
list catalog selection; I do not classify all body-less Functions as intrinsics
or change the common builtin-cache predicate. The source checkpoint must retain
ordinary definition, foreign declaration and callback controls for this route.

## My first complete native production checkpoint

I now draft the shared checked native_record_list.h provider and both producers,
with full catalog matching, canonical C call publication, exact selected
selfhost calls and left-to-right argument staging. I validate full-width indices
before narrowing both installed-provider and generated-provider calls. Native
provider discovery consumes exact registered C records and selfhost declaration
annotations plus actual selected emission calls; union field discovery uses the
same concrete instances and substitutions as existing native union emission.
I emit selected shadow bodies and global initializers before final provider
publication, and reset the worklist for the next parser.

The complete operation family reserves its generated C symbols. An actual
callable collision produces a compiler refusal, rather than silently replacing
that declaration with an intrinsic. Existing installed schema providers remain
separate and require the exact catalog name plus foreign declaration. Native
record payload ownership is unchanged. My complete source fixture matrix and
runtime allocation/bounds controls are still pending; this checkpoint has had
text-only delimiter/whitespace inspection, no build or execution.

I preserve the old generate_list_specializations shadow's positive body and
List_Point assertion, adding its missing Point declaration now that provider
selection validates actual definitions. Other original checker/source methods
remain unchanged. New emission shadows cover staged argument names, exact
catalog boundaries, declaration precedence, inferred constructor/removal facts,
selected shadows and consecutive parser isolation. These assertions are source
only until the reviewed qualification runs.

## My corrected-only qualification fixture plan

1. I preserve all eighteen existing source methods and their assertions, including
   actual LexerToken mutation and all evaluator/borrowed/scheduler controls. The
   native extension adds cases; it does not substitute a reduced corpus.
2. I instantiate the actual shared header in one bounded C fixture with plain
   record and nested record/string fields. I exercise all twelve operations,
   first/last/empty positions, repeated growth, full INT64 index extremes,
   copied removed values, vacated slots and clear/free ownership. Test-only
   allocator/exit hooks observe constructor partial cleanup and every measured
   constructor/growth allocation prefix, unchanged storage on refusal and fresh
   recovery. Hook scope is this header and list_capacity.h, not the compiler.
   Separate real subprocess cases verify the actual exit(1) diagnostic rather
   than treating the hook's nonlocal return as the production failure policy.
3. I add source programs for both native producers: inferred and discarded
   constructors, signature/field/nested annotation discovery, concrete union
   fields, imported same-spelled record owners, receiver/index/value trace order,
   callbacks/declared/foreign list-like names, selected shadows, and consecutive
   parser products. I compile the corrected longer supported canonical name to
   exercise the actual wrapper scanner. I retain declaration-collision refusal
   as an emitter refusal, not checker evidence. No old faulting revision runs.
4. I compile actual C-seed/Stage1/Stage2 outputs with strict warnings and ordinary
   O0/O2, then selected supported sanitizer controls on both hosts. Full-width
   scalar/schema indices must be refused by the emitted guard before any legacy
   provider narrows them. I preserve compile/output sentinels and exact command
   terminals before assertions; failure cleanup remains bounded by process group.
5. Only after full source/fixture review do I run fresh bootstrap and the complete
   eighteen-method corpus plus added controls, using the existing ten-second
   shadow policy and explicit current source/tool/provider maps. Successful 0e72
   gates remain their original evidence, not qualification of this new native
   implementation. Whole make test, enum parity, integration and fixed-point
   acceptance remain separate required continuations.

The wrapper-boundary MAC retry succeeded as
`task_1df19951939ca5ac58ad89d10efcabf9`. Its first authentication failure remains
recorded; no shared tunnel was changed.

My additive fixture checkpoint contains eight NativeRecordLists methods plus
the unchanged eighteen GenericRecordLists methods. The native runner forwards
actual selected compiler commands, preserves their arguments, appends an
explicit O0/O2 selection, and retains every final argv. Its C storage fixture
instruments only this shared header and list_capacity.h. It measures all actual
allocation positions, checks full retained capacity bytes and metadata on
refusal, and separately exercises real exit diagnostics without allocated
fixture roots. Ordinary native source routes and the instrumented C-header
matrix have separate compiler/flag selectors. I do not label intentionally
terminating source programs or the existing native string lifetime as leak-free
whole-compiler sanitizer acceptance.

The source extension preserves the original LexerToken file at its actual path
and checks its exact nine-line output on all three native producers at O0/O2.
It adds concrete union/field/signature discovery, inferred results, imported
same-spelled owners, long canonical names, declared/callback/foreign precedence,
explicit provider-collision refusal, full-width scalar/schema indices and scalar
capacity refusal. The compiler must successfully publish corrected guard code
before any expected runtime refusal is executed. A compiler refusal instead is
not accepted as runtime-bound evidence. No fixture has run at this checkpoint.

My first 1b8 Linux build stopped at five GCC format-truncation diagnostics before bootstrap or fixtures. I retain `/tmp/nanolang-record-lists-1b8-linux-prepare`, its unchanged source/tool endpoints and build-log SHA256 `bfc9082a8efe175215562cc11b4ac750cfe5fd657948cf86afaf174c692124f9`. The existing length check enforces the legacy 63-byte limit, but the widened 256-byte temporary obscures that fact from strict compilation. Before corrected execution I will keep the wide name only for inline-provider recognition and copy the checked legacy name including its terminator into a 64-byte array. I preserve the accepted name set, rejection diagnostics and all strict compiler flags. MAC `task_083bf0873b46fcd620174ed46e38abac` tracks this prerequisite.

### Direct generator worklist initialization prerequisite

I retain original 1b8 Darwin bootstrap status 2 at287.651 s, no timeout and equal source/tools. Actual OS crash report shadows-2026-09-21-015545.ips identifies __nano_shadow_688 -> gen_c_program_with_modules -> generate_list_forward_declarations -> detect_list_types -> dyn_array_length(NULL). Static source confirms the new native_list_keys complex global is emitted as zero by the existing global literal fallback, while initialization was only explicit at transpile_parser_mode and selected shadows; the older direct generator shadow reaches discovery first. I will give this emission worklist a scalar initialized latch and checked existing array-construction initialization before both actual readers/publishers, retaining explicit reset for every complete parser product. I will add cold direct generator, idempotent ensure, registered-key retention and reset controls. I do not broaden general complex-global initialization or change provider selection. No failed workload rerun: Linux corrected strict build passed74.828 s, dependent bootstrap externally stopped -15 at51.820 s, not counted as Linux product failure. Native18+8/full bootstrap acceptance remains open; review source before corrected-only execution. Durable evidence /Users/jkh/nanolang-qualification/record-lists-1b8 and /home/jkh/nanolang-qualification/record-lists-1b8/puck.

MAC `task_b4e7e3d0b5a57a4b0865d8d3e6356350` owns this bounded correction. I preserve the existing general complex-global policy; the emission-local worklist explicitly initializes through its own API.
