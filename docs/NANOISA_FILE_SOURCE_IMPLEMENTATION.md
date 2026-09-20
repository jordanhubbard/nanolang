# My generated File source implementation contract

I base this design on actual token911 merge
`cf238dfd1c1f3967cc84d3e0596b595f076b24d3`, publisher908 merge
`0061feed573ad7e4cac2a9e61d5ed38f0fb62440`, and the existing
[paired source contract](NANOISA_FILE_PAIRED_SOURCE.md). I keep ownership under
`task_8bbc1cf5295b4b59b314640ef57c725f` and parents6fc/72556/6931/d03c.
This is a proposed implementation contract, not measured source acceptance.
I change the actual compiler pipeline, not a second private parser that cannot
compile the publisher's output. Review precedes each production checkpoint;
fixture review precedes new execution.

## My actual input and parser

I compile the unmodified publisher directory: `binding.nano` and its sibling
`interface.nsi.json`. My root imports it using existing syntax:

```nano
module "binding.nano" as files
```

Its declaration stays exactly:

```nano
service "nsi:nanolang/filesystem" catalog 1 from "interface.nsi.json"
```

I preserve the five generated behavior shadows and their current calls,
`Ok()`/`Ok(file)`/`Error(error)` patterns and `&mut owned` operands. I do not
replace them with a reduced test source. `rewind` returns `PositionResult`, as
already specified by the catalog and renderer; I correct the older contract's
RewindResult spelling without inventing another type.

In `src/parser.c::parse_program` and
`src_nano/parser.nano::parse_definition/parse_program`, I recognize `service`
only at a declaration boundary. I require a STRING, contextual `catalog`, the
exact numeric token text `1`, existing `from`, and a STRING. `01`, `1.0`, signs,
unsupported versions, missing operands and extra same-line tokens refuse. I
allow the ordinary end-of-line/end-of-file boundary and comments. A subsequent
declaration on the following line is parsed normally. The published declaration
is implicitly module-public; I reject an additional `pub service` rather than
silently ignoring its modifier. These words remain ordinary identifiers outside
this production. I add no new global lexer keywords.

I use the existing literal decoder and qualified `value_bytes`, not a second
escape grammar. Both decoded strings must have their complete counted extent,
contain no NUL and satisfy the interface/path policy before resolution. I test
escaped NUL with a suffix, malformed escapes, non-ASCII and partial UTF-8 counts.
The token runtime's inability to represent embedded NUL is a refusal condition,
not permission to repair the count. The parser does not open the document.

I add `ASTServiceDecl` to `schema/compiler_schema.json`, with interface/path
strings and decoded counts, version, token line/column and origin index. I append
the dedicated C AST tag after inventorying current tags; I do not renumber old
tags or assume C and Nano tags already share numbers. Nano gets a generated
service list/table/count in Parser. Every parser constructor/store function
retains it, including historical direct constructors in nanoc_integrated.nano.
I regenerate all four canonical generated files and compare complete output
from Python and the independently compiled Nano generator.

C owns copied node strings and frees every failed prefix. Nano owns immutable
record values through its existing runtime; I do not claim recoverable OOM
parity. AST clone/free, reflection/export, module traversal and nonparticipating
consumers either retain complete service facts or return an explicit unsupported
result before effects. A service node cannot disappear through an opaque-type
or extern-function case.

## My origins, snapshots and resolved identity

I attach C module origins before import flattening in `src/module.c`. In the
real Nano driver `nanoc_v06.nano::merge_with_imports`, I use its existing
`files/file_starts/source_files/source_lines` mapping. I resolve a service node's
merged line to the original canonical module and line; root-line thresholds and
textually mangled aliases are not identities. I preserve the service line and
origin while applying import aliases. I test transitive imports, reordered
imports, same basenames in different directories, and multiple aliases of one
module through the actual driver.

I propose a shared data-only snapshot facility with an opaque owning handle:
open relative to retained origin, bounded read, strict `nl_file_binding_prepare`,
then counted immutable catalog/document getters and one destroy operation.
The facility cannot inspect source ASTs, resolve names, assign wire layouts or
emit instructions. Nano constructs and checks its own request records and calls
its own descriptive-plan implementation. C calls `nl_file_source_plan_build`.
I retain the existing 16 requests, 13 bindings per request, 64 aliases, 256
ordinary declarations and 1MiB copied-text bounds; strict document bounds apply
independently. I check aggregate snapshot storage products before acquisition
and cap simultaneous document/strict-plan storage at 64MiB. The implementation
checkpoint must show exact peak accounting including old/new buffers and copied
plan text, without counting a nested allocation bound twice.

I reject final-component symlinks, nonregular inputs, incomplete reads and
oversized documents. I retain the publisher's stable-ancestor precondition; I
do not claim hostile ancestor replacement protection. I close descriptors once,
retain the first failure and free all acquired snapshots. Successful snapshots
remain immutable through checking/lowering; I never reopen the companion to
change facts midway. Missing files and malformed or mismatched complete catalogs
fail before service-host or generic-loader attempts. Reading a declared source
companion is separately reported compiler input I/O, not a File service grant.

I first collect the complete ordinary/service/import namespace, then allocate
unique declaration IDs and pass complete requests/aliases to the existing plan.
I never invoke its checked-namespace precondition on a partial namespace. Each
resolved reference carries module origin, declaration ID, kind and catalog
ordinal; visible spelling is diagnostic only. I preserve original identities
through aliases. A duplicate declaration or cross-module lookalike cannot acquire
File authority. Distinct modules' declarations remain distinct nominal identities
unless they resolve to the same original imported declaration.

I represent File nominal identity explicitly in C TypeInfo/resolved expression
facts and corresponding Nano type/checker tables, with an invalid sentinel for
ordinary types. I do not turn every TYPE_OPAQUE or ordinary union into a service
type. Field and variant lookup use the exact catalog row and owning identity.
The production checkpoint inventories every type copy/equality/signature path;
old ordinary type behavior is unchanged. Declared variables/parameters/returns,
record fields and shadow targets retain these facts through resolution.

## My two independent checking and lowering paths

I extend the real C checker and `src/nanovirt/codegen.c` with a dedicated File
source selection before ordinary borrow/codegen/FFI fallback. In Nano I extend
the real checker and the `nanoc_v06.nano::compile_program` byte-emission path,
with independent typed expression, statement, layout and instruction producers.
A shared serializer may encode already-decided fields; it cannot implement Nano
name resolution or lowering. I retain per-producer normalized identity maps and
compare complete decoded output, not just runtime exit codes.

My source lowering handles the renderer's whole expression vocabulary: typed
locals, mutable File owners, exact scalar record field reads, prefix arithmetic
and comparisons, assertions, direct calls, if/else, returns and Result matches.
I verify every body, including uncalled helpers. Unsupported constructs are
explicit pre-execution refusals. I map File/OpenResult moves and refinements to
FILE_SERVICE, FILE_RESULT_BRANCH, FILE_RESULT_TAKE, FILE_DROP_LOCAL/STACK and
FILE_END_BORROW, with the qualified direct-call/reference forms. A unit arm has
no invented INT payload. I stage moves exactly once and preserve borrow origins
on both outcomes. Close consumes the owner even on a reported close failure;
cleanup errors are not converted into successful scalar publication.

I independently produce full v2 service120 metadata, exact import/catalog/global
layout maps, flags, signatures, local/reference slots, relative branches, real
startup functions and truthful max_stack. The consumer's nominal/body/hosted
validation is mandatory after serialization. No generic CALL_EXTERN, ordinary
union payload extraction, or an assembler lacking service metadata substitutes
for this route. FileError's seven fields, ReadByte's two fields, and scalar versus
affine Result distinctions remain exact.

Ordinary shared v3 union/array envelopes remain canonical, not a competing
schema. The first acyclic File source module may contain only combinations that
the existing File v2 transport and consumer actually retain and validate. A graph
requiring unsupported simultaneous v3 ownership facts refuses explicitly before
output. I require a coexistence test proving retention or that refusal; general
mixed execution remains mandatory later work, not silently accepted.

## My driver and authority contract

I propose `--allow-temporary-files` in actual C seed, nano_virt and Nano driver
options. For a service-bearing source graph, default native compilation selects
the qualified File native emitter, not ordinary C transpilation. `--emit-nvm`
selects File v2 bytes; I add that explicit mode to C seed for paired parity.
`nano_virt --run` uses the public byte API. `--target c`, GPU, LLVM/Wasm, daemon,
COP and legacy interpreter routes refuse the service graph until separately
qualified. Unsupported option combinations refuse before shadows/output.
Ordinary source behavior and option defaults stay unchanged.

The source flag authorizes this compilation's required service shadows only;
it is not a persisted runtime grant. I propose generated standalone native
launchers require their own runtime `--allow-temporary-files` argument, create
and revoke/destroy one grant for that invocation, and refuse without it before
service effects. Library emission exports the validated unique entry identifier
and requires the caller's explicit grant, with no hidden main/global grant.
Runtime VM invocation likewise uses its existing explicit grant opt-in. I test
separate generated programs linked to one installed grant/gate owner and native
symbol extraction without a VM engine dependency.

Both source producers run `nvm_file_execute_bytes` for selected VM shadows and
use `nvm2c_emit_file_bytes` for native emission. Nano may call a data/byte execution
bridge only after independently producing and validating the module; that bridge
cannot lower source. Native acceptance separately executes actual compiled
functions/labels with the same grant/cleanup contract. Scalar INT/BOOL is the
only clean entry escape; APIs retain full INT bits and CLI uses its documented
low-byte status. Initializers are strictly VOID/no-result, run in exact selected
order, and finish cleanly before entry. A failure suppresses later initialization
and entry, preserving first and secondary cleanup reports.

## My selected shadows and staged publication

I resolve generated shadows to exact service declaration IDs. None receives the
ordinary extern exemption. Each selected shadow gets a fresh invocation/context
and explicit grant, including imported generated modules selected by default.
I retain actual selected name-plus-origin multisets, start/completion records
and assertions for every helper/module/generated behavior shadow. Explicit
root-only opt-out remains separately tested and is not full-product acceptance.
I keep the existing ten-second whole-suite deadline, durable file-backed output
and bounded process-group cleanup. I do not optimize away missing shadows.

The outer compiler owns a private staging directory and all provisional byte,
C, object and executable outputs. It publishes the requested single final
artifact only after complete checking, selected shadows, consumer validation and
clean terminal cleanup. Failure, timeout or Nano process OOM leaves no new final
artifact and preserves an existing output. I avoid source/companion path aliases;
I track failed cleanup explicitly rather than claiming rollback success. Binding
directory publication remains the separate qualified publisher transaction.

## My ordered implementation and acceptance checkpoints

1. I add schema/node retention and contextual parsing directly to both actual
   parsers, with all constructor/clone/export refusals and complete generator
   equality. I review source and fixtures before fresh paired parser gates.
2. I add bounded snapshot ownership, original-module association, complete
   namespace resolution and explicit nominal type propagation in both checkers.
   I test every allocation prefix on C, lifetimes, mutation-after-snapshot,
   import/alias/collision cases, counted malformed text and all catalog fields.
3. I add both independent source lowerers and byte validation. I compare exact
   normalized declarations/layouts/imports/instructions/startup facts, all-arm
   ownership behavior and rejection of unsupported bodies, without executing
   unreviewed generated service fixtures.
4. I add actual source-driver shadows, per-invocation grants, native/VM routing
   and staged publication, then review the complete product fixture corpus before
   execution. Real publisher output is an input, not copied replacement text.
5. I freshly bootstrap C seed, Stage1 and Stage2 on Linux and Darwin with complete
   ABI closure. All three compile the same generated bindings, roots, transitive
   imports and all selected helper/module shadows. I retain full ordinary build,
   compiler/module/parser/wrapper regressions, selected-provider sanitizer scope,
   every first terminal and actual source/tool/product inventories.
6. I install the actual headers/archive/compiler/publisher, compile outside the
   source tree, and execute actual VM and native products. I prove missing-grant
   and unsupported-backend refusal with attempted host/loader counters; real
   temporary write/rewind/read/close, error arms, initialization suppression,
   descriptor cleanup, scalar exit and no-publication failures must all pass.

My first source profile remains acyclic, direct-call and the currently qualified
single-reference boundary. Cyclic source selection needs the separately reviewed
explicit fuel/public conjunction; indirect source calls need finite target and
matched runtime admission; richer borrows need their own transport and lifetime
qualification. I retain these and complete mixed ownership/source acceptance as
required 5.1 parent obligations. Passing the first profile does not close them.

## My first source checkpoint audit

I append C AST_SERVICE_DECL and schema PNODE_SERVICE_DECL; I retain all previous
numeric values. ASTServiceDecl and Parser's service list/count change the internal
compiler ABI, so I require a fresh compiler/runtime/module closure, never old
Parser values or generated list objects. Python regeneration writes all four
canonical outputs; compiler_contracts.nano has no changed bytes. Independent
C-seed/Stage1/Stage2 generator equality remains a later gate.

My node owns two decoded strings, complete counts, catalog1 and origin_index=-1.
I do not fabricate an origin without the later module map. C parse/clone/free
owns these strings exactly once. All39 current-schema Nano Parser copies retain
services/count, and the service store changes only that count and normal node
bookkeeping. The historical nanoc_integrated.nano owns a separate older Parser
schema; it explicitly refuses service syntax rather than mixing the new ABI.

I reject unresolved service graphs before C import processing, ordinary type
checking/shadows, reflection/typed export, C/NanoISA/C-backend emission and Nano
checking/transpilation/NanoISA (including direct borrow emission). Direct C
statement evaluation terminates with the existing invariant-failure mechanism.
C fold/DCE/CPS walkers preserve the node as an owning leaf; clone and free handle
its strings. HM/LSP inference and NanoCore refuse it. The data-only parsers do
not read companions, resolve authority or execute generated shadows.

I reuse the shared C literal decoder and Nano's existing import-path decoder,
checking complete raw-decoded counts before decoding and resulting extent after.
Unknown escape spellings retain the current decoder behavior; no new Unicode
escape grammar is introduced. Both declaration strings have a1MiB counted limit,
interface spelling is exact, document path is nonempty and nonabsolute. Other
path/filesystem policy remains the later snapshot resolver's responsibility.

My added C declaration has two decode allocations and one zeroed node; failure
frees both strings. Its clone owns a fresh node and two copies. Program entry
checks both initial buffers, checked table growth keeps the old pointer on
failure, and final root allocation frees retained declarations on failure.
This is not whole-parser recoverable-OOM acceptance: older unrelated expression,
lambda-hoist and AST-constructor paths still contain unchecked allocations. I
record that prerequisite rather than claiming the new service checks repair it.
The forthcoming fixture checkpoint must bound measured allocation claims to the
actual paths and retain the whole-source publication failure obligation.

I have not run a compiler, fixture, bootstrap or service gate at this source
checkpoint. Mandatory new helper/parser shadows belong to the next reviewed
fixture checkpoint; none may be omitted from eventual full selected shadows.

### My focused source-review corrections

I add explicit strict UTF-8 checking after complete decoded extent equality. C
uses `nl_utf8_validate`; Nano independently checks the same RFC3629 lead-byte,
continuation, overlong, surrogate, truncation and U+10FFFF boundaries. Its
`char_at` is an unsigned byte operation in the current interpreter, generated C
and nl_cstr runtime, not a code-point iterator. This does not add Unicode escape
syntax. The forthcoming paired fixture must include malformed direct token
bytes as well as actual source input, because the C CLI already validates input
UTF-8 and cannot alone prove the reusable parser's refusal.

I also bound final lambda-hoist table growth. A failed allocation keeps the old
items pointer; cleanup frees all items (including previously moved lambdas),
then only the still-unmoved lambda suffix, then both pointer tables. Size overflow
uses the same cleanup. Root allocation failure occurs after all lambda roots
have moved, so it frees items once. Earlier lambda creation allocations remain
part of the separately recorded whole-parser audit.

I audit the declaration-presence helper against actual parser-produced ASTs:
service construction is reachable only from parse_program; function/block
parsers never construct this tag. AST_MODULE_DECL contains a name and no nested
body. Imported modules are separate cached AST_PROGRAM roots; load_module calls
process_imports on that root before type checking/cache publication. Root and
imported service declarations therefore meet the same refusal. The helper is
not a structural validator for caller-forged service nodes hidden in arbitrary
expression trees, and it makes no such claim. The later resolver must preserve
this invariant while binding module origins.

My clone's checked calloc has exactly create_node's successful initialization:
zero the complete ASTNode, then set type, line and column. The service case copies
scalar fields and duplicates both strings; it never shares their ownership.
There are no additional hidden create_node initializers. I retain no execution
or whole-parser failure-recovery claim from this source inspection.

## My unexecuted parser fixture checkpoint

I add five meaningful mandatory shadows to the actual parser's new UTF-8,
store, declaration and getter helpers. I retain existing shadows unchanged. The
paired product fixture imports this real parser and parses freshly published
complete binding text, all five behavior-shadow ASTs, malformed grammar/counts/
NUL/UTF-8, and exact EOF/comment/later-definition retention. It does not execute
the parsed File shadows or claim resolved source authority.

My C fixture includes the actual parser with four allocation hooks plus strdup,
links freshly compiled lexer/UTF-8, and retains the exact ordinary common/runtime
provider list. It frees original tokens before reading retained declarations,
checks clone lifetime after source destruction, sweeps every measured allocation
in a33-declaration program with transient and persistent refusal plus recovery,
and targets final lambda-hoist item growth with a real lambda-bearing program.
It distinguishes moved roots from the remaining suffix and requires zero tracked
live allocations. This does not inject every older expression-construction site.

I run actual C-seed/Stage1/Stage2 generated parser fixtures and compare their
canonical retained summary. I reuse the complete token/generator gate, including
all four generated files from all three compiled generators plus Python, with
isolated output directories and exact selected-shadow multisets. I also invoke
all three actual compilers and nano_virt on the complete publisher file and
require unresolved-service refusal and no executable output. Root helper shadows
and imported parser/helper shadows are all selected; their names and source
hashes are retained, without count-only or substring selection acceptance.

The Make target requires bootstrap3/nano_virt; the outer orchestrator must start
fresh with the changed Parser ABI. Supported sanitizer phases rebuild the included
parser and selected lexer/UTF-8 TUs with strict leak settings; common/runtime
objects remain ordinary and explicitly inventoried. Actual publisher setup
freshly builds its seven-provider recipe under selected flags. The reused runner
retains file-backed output, first status, launch failures and bounded process
group cleanup, clearing inherited LSAN_OPTIONS. No fixture or new compiler has
been executed at this checkpoint; Linux/Darwin ordinary and supported sanitizer
configs await review, with source/tool/product seals and first terminals required.

### My prelaunch runtime-list closure correction

I found the missing provider statically before any40a build or fixture run.
`is_schema_list_type` classifies every AST-prefixed element as a dedicated runtime
list, so generic fallback cannot provide ASTServiceDecl. I specialize the existing
ASTOpaqueType source/header by identifier substitution, trailing-whitespace cleanup and first-person diagnostics only. Its checked capacity,
process-OOM behavior, shallow element copying, index checks and free behavior are
unchanged. It includes the canonical generated compiler_schema.h element layout;
I add no independent struct definition or bespoke list ABI.

I add the provider to Make RUNTIME_SOURCES, the C driver's explicit runtime source
array, the actual nanoc_v06 runtime source command, the Nano transpiler's schema
header list, and the normal/daemon wrapper's runtime object list. C emission's
existing AST-prefix header rule now resolves the real file. The common Make object
rule already depends on the canonical schema header and emits dependency files.
New parser fixtures inherit the complete RUNTIME_OBJECTS through their explicit
provider list, so the new object is inventoried rather than hidden.

I checked the current install target: it installs compiler binaries and the
separate explicit File byte-runtime archive/header package; it does not install
ordinary compiler runtime source files. I do not add an internal AST list to the
public File runtime ABI or pretend this is standalone relocatable source support.
The existing C/Nano driver resolves ordinary runtime sources from its selected
project root. Full outside-tree source support/install acceptance remains the
later parent gate; this checkpoint proves fresh in-tree ABI closure only.

### My measured zero-binding match prerequisite

Both fresh8baca bootstraps pass. My3936 fixture first compiles and exercises
counted grammar, service allocation prefixes, clone lifetime and hoist cleanup,
then aborts while parsing the unchanged complete publisher binding: CloseResult
uses `Ok()`, but both real match parsers required an identifier. I retain both
Linux/puck terminals and all five generated shadows. I do not repair a parser
acceptance gap by changing or reducing my qualified publisher output.

My existing ordinary union representation already has exact per-variant field
counts; `None {}` denotes zero fields. I now retain `None()` as an owned empty
binding string in C and the corresponding empty string in Nano's arm table.
I leave `None(n)` accepted as before. `Some(_)` retains its existing explicit
discard and ownership rules. Empty binding is not a wildcard or implicit payload
discard: the paired typecheckers require a known selected variant with exactly
zero fields. Named unknown variants and nonzero payload omissions are refused.
This adds no schema field, token kind, catalog identity or service authority.

| Consumer | My retained/checking behavior |
| --- | --- |
| C/Nano real parsers | I accept only identifier or immediate closing parenthesis, retain the empty/named distinction, then require the closing delimiter/arrow/body as before. |
| C match-domain checker, both expression/statement paths | I resolve the actual union and require an exact zero-field selected variant before accepting an empty binder; I create no empty environment symbol. |
| Nano match coverage/checking | I retain matching field counts beside exact variant names, diagnose empty/nonzero or unknown selection, and retain existing no-empty-symbol handling. |
| C evaluator | I require actual selected union field_count zero for an empty binder, and create no local. |
| C native and direct C/NanoISA emitters | I check exact selected declaration/count before emitting an empty binding; the native helper covers all four expression/statement branch forms. Existing refusal mechanisms remain in place. |
| Nano native emitter and NanoISA coverage | I refuse unknown/nonzero empty selection through the existing emitter error/coverage path. NanoISA's internal empty-name slot remains its existing inaccessible local sentinel. |
| Ownership, purity, HM scope and PGO/TCO | I introduce no empty source name. Ownership still refuses discarded resource payloads. HM remains an opaque auxiliary inference pass, not exact union admission; purity already skips empty names. |
| NanoCore export | I refuse empty binding because my current formal match export requires a binder; I do not claim a new formal correspondence. |
| Free/retention | The ordinary match table already owns/frees each binding string; the new C empty strdup is checked and uses the same cleanup path. Nano parser copies preserve the existing match table. |

The bounded change introduces no new allocation product beyond the existing
one binding string per named arm (now one byte for empty). Earlier unchecked
ordinary match table/create_node allocation paths remain within my already
recorded whole-parser OOM prerequisite; I do not extend the service-specific
allocation acceptance claim to all expressions. The copied ASTServiceDecl and
public unresolved-service refusals remain unchanged.

Before qualification I require ordinary, service-free zero-field acceptance,
nonzero omission refusal, named zero-field compatibility, explicit discard
compatibility and unknown-variant refusal in both frontends. I will inspect the
retained binding distinction and direct backend/core refusal boundaries, then
rerun the complete publisher corpus and all existing allocation assertions.
This semantic correction needs complete production review before fixture edits
and fresh ABI/compiler qualification; no old failed binary is replayed.

### My zero-binding fixture supplement

I preserve every original assertion. The included C parser fixture now also
parses five service-free ordinary cases: empty zero-field, old named-zero plus
underscore discard, omitted nonzero payload, unknown empty variant and malformed
punctuation. I compare exact retained binder strings, run the actual checker and
ordinary evaluator, emit both C paths and NanoISA for the accepted cases, and
require explicit NanoCore refusal for empty binding. A separately mutated,
previously checked payload arm forces direct C/NanoISA arity refusal without
relying on the main typechecker; the direct C output keeps its nine-byte sentinel.
I link and inventory the actual ordinary NanoVirt/NanoVM/NanoISA providers in
addition to the prior common/runtime closure. Sanitizers still instrument only
the parser fixture and selected lexer/UTF8 TUs, not these ordinary providers.

My paired Nano fixture imports the actual checker and both emitters. It checks
the same five cases, exact binding strings, complete checking diagnostics,
NanoISA coverage and native emission error/recovery. Every imported shadow remains
mandatory in the exact selection multiset; I add no root-only opt-out. I replace
the real parse_match helper's old tautological shadow with a retained two-arm
syntax check. All three actual compiler CLIs also build and run both positive
ordinary cases (exit7), and refuse each negative without an output executable.
These cases contain no service declaration, so unresolved-service guards cannot
substitute for ordinary arity checks. Full publisher bytes, all five service
shadows, seven C configurations and independent four-file generator equality
remain required. No File service handler is executed by this checkpoint.

### My actual metadata destructor observation boundary

Both18334 fresh bootstraps pass. Each first ordinary parser fixture reaches its
zero-live assertion after the first union case, then aborts. The old terminal
prints neither a retained pointer nor a count; I do not invent either. Static
tracing establishes one missing observer boundary: parser.c allocates payload
TypeInfo, while the actual recursive destructor in env.c releases it outside
the included-parser allocation tracker.

I compile the actual env.c afresh with a scoped free-only hook header. The hook
forwards every free to the existing tracker removal and real libc free. I do not
copy or simulate its recursive ownership rules, hook env allocations into the
parser's failure-prefix domain, reset live_count, or remove the final assertion.
The selected ordinary env.o is excluded from the fixture link, and the new env
object is inventoried beside lexer/UTF8. Sanitizer scope now explicitly covers
the included parser plus fresh env, lexer and UTF8; other common/runtime and
NanoISA providers remain ordinary. This is a fixture/provider correction only;
actual compiler/runtime sources and retained18334 bootstrap inputs are unchanged.
Fresh corrected fixtures and module products still require source/product
identity proof before reusing those exact bootstrap binaries.

### My generated local keyword correction

The8134 fixtures on both hosts reach the full publisher after their ordinary
match and cleanup controls, then reject `Ok(byte)` because byte is an existing
reserved type token. I retain the first diagnostics and do not claim either
whole fixture passed. I rename that renderer-local binding to `octet`, updating
the independent complete golden at the same checkpoint. The catalog's value/eof
fields, function names, constants, success/error branches and all five shadow
predicates are unchanged; I do not rewrite actual publication inside a test.

I require fresh strict document/publisher qualification with each of the same
seven selected compiler/sanitizer configurations, in addition to the complete
parser corpus. The existing three-method publisher suite builds all seven
actual Make providers, exercises API/CLI faults and exact source bytes, then
runs the five-provider linked/instrumented strict binding and legacy neighbors.
Its allocation/roundtrip/budget/golden assertions remain intact. Those fresh
selected providers have their own instrumentation scope and inventories; they
are not evidence that ordinary compiler providers were sanitizer rebuilt.

The changed renderer is absent from the retained18334 compiler/NanoVirt object
dependency files and from their selected provider source lists. My real parser,
all src_nano compiler inputs and runtime sources remain unchanged by this
correction. I may therefore reuse exact18334 bootstrap products after full
source/product identity proof, while always freshly building this changed
publisher and its fixture/module outputs. Original bootstrap and sanitizer pins
remain distinct; no fresh-bootstrap or File service execution claim follows.
