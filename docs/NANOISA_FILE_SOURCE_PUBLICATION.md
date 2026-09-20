# I bind paired source declarations to my File catalog

I record `task_8bbc1cf5295b4b59b314640ef57c725f` under hosted6fc, execution72556, NSI6931 and real-handle d03c. My starting point is actual PR895 merge `97546742af995329967df630a20c4c72dd3f6dd4`. That merge qualifies explicit granted byte execution and an installed native package. It does not make my source producers, generic NSI generator or ordinary opaque types into verified File bindings.

This is a preimplementation contract. I change no parser, schema, emitter, selector or service handler here. Every implementation checkpoint and its fixtures require source review before execution. I retain the full paired-source/shadow, cyclic/indirect/richer-borrow and real-handle roadmap requirements.

## My current source routes

| Actual route | Required connection |
| --- | --- |
| `src/nsi_gen.c:nano_type`, `nl_nsi_gen_nanolang` | RESOURCE/VARIANT become demonstration integers. I need a distinct checked File generation mode, not a change in meaning of this legacy output. |
| `src/nsi_file_plan.c`, `nsi_file_catalog.h` | Existing `nl_file_plan_build` already checks the exact current-schema NlNsi against one immutable catalog. I reuse this authority description; I do not introduce a second permissive descriptor query. |
| `src/nanolang.h:TypeInfo`, `AST_OPAQUE_TYPE` | Opaque source declarations carry a name; their ordinary runtime type can become TAG_OPAQUE. Neither implies an affine File. |
| `src/nanovirt/codegen.c:codegen_compile_internal`, `borrow_codegen.inc` | Current owner selection is driven by actual resource/borrow facts and delegates to bounded owned profiles. Defined functions and records are not service declarations. File selection must precede an inappropriate ordinary/owned fallback and use exact resolved declaration identity. |
| `src_nano/compiler/nanoisa_codegen.nano:nanoisa_emit_program_nasm`, `nanoisa_emit_shadows_nasm`, `nanoisa_borrows.nano:nanoisa_borrow_emit` | The selfhost publisher emits assembly for existing profiles. Merely adding File spellings to this text does not attach the required service maps and ownership facts. |
| `src_nano/nanoc_v06.nano:compile_program` | Selected shadows currently assemble through `Nanoisa.assemble_text_save` and invoke `nano_vm --check-shadows`. PR895 deliberately refuses that flag with File execution. A new checked File shadow path is required before publication. |
| `src/nanovirt/main.c`, `shadow_runner.h:check_shadows` | The C-seed path checks selected dependency/root shadows and may bind generic foreign imports. File must use the explicit granted runner, never generic FFI/COP binding or evaluator fallback. |
| `src/nanoisa/file_hosted.inc`, public File API | Exact serialized facts, all bodies, startup and VM/native storage must still pass the current acyclic conjunction. A compiler plan is neither a grant nor a cacheable execution certificate. |

I audit the actual installed C-seed entry and fresh Stage1/Stage2 driver route before implementing their option plumbing; alternate historical driver files are not acceptance substitutes. All emitted source identities survive module merging, cloning and schema serialization.

## My concrete declaration and generated binding proposal

I propose one new **declaration**, not an extern body or a raw token constructor:

```text
service "nsi:nanolang/filesystem" catalog 1 from "temporary_file.nsi.json"
```

This spelling is proposed grammar, not accepted current syntax. It appears at top level in a binding module. The generated `.nano` module contains this declaration plus useful behavioral shadows for the five declared operations. Existing module syntax imports it under a caller-chosen alias, for example `module "temporary_file.nano" as files`. The service declaration contributes the eight exact public nominal types and five public callable declarations to that module's namespace. It has no host side effect and creates no ordinary callable implementation body.

A new generator mode named `file-binding` first validates the existing NSI v0 document through `nl_file_plan_build`. Its concrete output option is `--file-binding-dir DIR`: DIR must not exist. I stage a sibling directory containing canonical current-schema `interface.nsi.json` and `binding.nano`, whose declaration uses `from "interface.nsi.json"`. I check both writes/closes before publishing the whole directory with a no-replacement operation; any failure removes only owned staging and preserves an existing destination. A later replacement/update interface needs its own review. This avoids pretending that two independent file renames form a transaction. No new manifest or ignored NSI/module.json key is introduced. The first descriptive checkpoint publishes neither file.

The source declaration resolves its NSI path relative to the declaring module through the existing canonical module resolver. It reads a bounded snapshot once; later phases use owned validated facts rather than reopening a potentially changed document. I compare full IDs and contents against catalog1; filenames, comments, hashes, sanitized symbols and generated provenance are never authority. A handwritten identical declaration may request the same interface, but still needs all checks and a host grant. Unknown keys remain rejected by the unchanged NSI v0 parser. Unsupported generator languages refuse the File mode rather than returning demonstration integers.

The declaration stores interface ID, catalog version, source module identity, source span and validated document identity. Its immutable resolved plan assigns declaration identities to all eight catalog types and five methods. Aliases point to these identities; they do not create new service types. A second module importing the same exact interface/catalog has compatible service nominals only after the same complete validation; a user record named File or an extern named temp never does. Lexical shadowing either resolves to its actual ordinary declaration or produces an ambiguity error; lowering never re-recognizes the service from its final string spelling.

I keep explicit maps from source declaration identity to catalog ordinal, original global layout, STRUCT/UNION table identity, function/import index and retained runtime category. Eight distinct type mappings and five exact SERVICE bindings are required. Declaration ordering, unrelated record insertion and module aliases cannot change semantics or silently renumber an existing descriptor. Duplicate/conflicting service declarations, malformed source schemas, truncated IDs, embedded NULs, generated collisions and unknown catalog versions refuse before publication.

## My source type and ownership rules

The exact source surface is derived from the already checked catalog:

| Operation | Source-level signature and ownership |
| --- | --- |
| temp | zero inputs, OpenResult; Ok owns one File, Error owns none |
| write_byte | exclusive call-scoped `&mut File`, int -> WriteResult; File preserved on both arms; int must be0..255 at runtime before host access |
| rewind | exclusive call-scoped `&mut File` -> PositionResult; File preserved on both arms |
| read_byte | exclusive call-scoped `&mut File` -> ReadResult; value0..255 or canonical0 at EOF |
| close | consumed File -> CloseResult; accepted close consumes on both Ok and Error |

I reuse the existing syntactic exclusive borrow form, but classify it by the service declaration. Shared File borrows do not satisfy exclusive operations. The current single-borrow CALL_REF and root-only reference limits remain exact. I do not invent a general mutation/borrow guarantee for ordinary opaque types.

File is opaque and affine: no record literal, integer/pointer conversion, null construction, field projection, copy, serialization or generic extern transport. OpenResult is affine; matching inspects its discriminant then transfers the selected payload exactly once. An unmatched live Ok remains an owned cleanup obligation. Error has the exact FileError record fields `status, host_errno, cleanup_errno, bytes, eof, consumed, cleanup_failed`. ReadByte has `value, eof`. WriteResult.Ok carries int; PositionResult.Ok and CloseResult.Ok carry unit, with no fabricated integer payload. The four non-owning Results retain ordinary value semantics.

My checker resolves exact nominal arm identity and arity, including empty Ok payloads. An affine match cannot duplicate the result through repeated patterns, leave a moved source readable, hide an owner under wildcard handling, or return a borrowed File. Every early return, branch merge and lexical exit accounts for actual roots. Internal helpers may transfer File/OpenResult through the qualified call/result staging; public entry and shadows return only their approved scalar result. Scalars never carry disguised host tokens.

I lower service calls and Result refinement through existing FILE_SERVICE, FILE_RESULT_BRANCH, FILE_RESULT_TAKE, FILE_DROP_LOCAL/STACK and FILE_END_BORROW semantics, using checked physical slots and declaration maps. Arguments evaluate once in source order; partial result/argument owners stay rooted until publication. Explicit drops and implicit lexical cleanup have the same retained first/secondary-error behavior as the qualified runtime. I do not implement these transitions with generic CALL_EXTERN, AGG_GET or ordinary union projection.

## My schema coexistence prerequisite

Peer PR893 `fb2167a6239da798022724a596029833a5023442` owns the active scalar-union instance contract. `29ecd8e3f` proposes ordinary record/array authority. No common version agreement is established at this checkpoint. I reserve no presumed v3 grammar, codec field or conflicting AST enum number.

File Result/catalog authority stays distinct from ordinary-union authority. Before paired AST/schema changes, I enumerate every primary schema, generated C/Nano node definition, clone/copy/serializer and consumer. I obtain one reviewed version/required-field policy that preserves both routes or explicitly refuses the unsupported combination. Old consumers must reject new execution-significant declaration facts rather than ignore them. No ordinary resource-UNION validator is broadened to accommodate File. A mixed source containing unsupported ordinary unions/arrays beside File must refuse with its full input retained; silently dropping the ordinary declarations is not eligibility.

## My nonexecuting source eligibility and publication boundary

I propose paired owned `FileSourcePlan` descriptions: C construction over checked AST/environment/module graph; Nano construction over checked Parser/type/import facts. Exact public names and storage structs belong to their source checkpoint. Plans own all copied identities and selected-function/shadow/initializer tables; no pointers into destroyed input graphs remain. All products, counts, path strings and allocations are checked before indexing. Failure preserves output and earlier destination files, and frees every staged plan element.

Before the new grammar exists, checkpoint1 takes an explicit immutable request record containing module identity, source span, interface ID, catalog version and the already validated document-plan view, plus existing checked declaration/import facts. Fixtures construct those requests directly. C and Nano use equivalent typed records and compare a canonical descriptive report; this is not a claim that the current Parser can read `service`. The later paired parser checkpoint becomes the only compiler constructor of these request records, and must validate its complete source/schema before calling the plan. No side file silently injects a request into an ordinary compile.

The first plan is descriptive only: it exposes the exact catalog/declaration maps and explicit eligibility reasons. It neither emits executable source nor asserts successful bytecode verification. I reuse the existing immutable catalog through a bounded read-only interface; I do not let the selfhost path ask a hidden C backend to lower its AST or supply arbitrary postconditions. Both publishers must perform their own source lowering and produce equivalent semantic facts. Any shared catalog bridge is data-only and separately audited in bootstrap/module closure.

The later dedicated File publisher constructs one complete v2 module with exact signatures, locals, code, ownership, service120-byte metadata and required bits1/9. It uses existing checked transport APIs and runs the complete current hosted query on the final bytes. Generic assembler/module-facade refusal remains unchanged. Output publication is staged after successful complete validation and all selected shadows. I do not treat the generic text assembler's acceptance of six opcode spellings as module authority.

Eligibility covers the whole imported program and all function bodies, including uncalled helpers, rather than only the entry's reachable happy path. Initial limits are the actual acyclic File code/flow/hosted budgets, not the unrelated eight-function owned-source emitter limit. Wire max_stack and local/ref-slot counts are truthful and checked before emission. Globals, captures, callbacks, foreign imports, arrays/strings or scalar instructions absent from the matched File handlers refuse. No generic backend fallback may accept a rejected File source by erasing ownership.

## My exact shadows and startup

Default selection retains all dependency and root shadows in the same deterministic order as the compiler's existing import policy. Explicit root-only selection remains an observable option, never the default acceptance configuration. I record selected source identities and spans before synthetic entry construction. Each selected shadow is lowered against the complete checked module graph and all required helper bodies; compiling a separate entry is not permission to partition away inconvenient declarations or shadows.

Each shadow invocation gets a fresh explicit temporary-file grant and invocation context. A synthesized zero-argument scalar entry executes precisely that selected shadow, returning canonical success only after its assertions and cleanup. Compiler-generated names are collision-checked against every source/imported declaration; a user spelling cannot acquire synthetic entry authority. Failed assertion, timeout, error Result mishandling, initialization/cleanup failure or missing grant stops publication and preserves the prior destination. No effectful shadow is skipped or replaced with a stub.

The first source profile has no globals. If ordinary module initialization needs unsupported global storage/effects, I refuse rather than dropping it. Eligible compiler-owned initializers are zero-argument VOID functions with no escaping owner; their dependency order is recorded. I synthesize exactly one aggregate `__init__` in the function table, calling each selected initializer once in that order; the public runtime still selects its first table-order `__init__`. Other bodies remain checked. User collisions/duplicate reserved initializer identities refuse. A scalar/owner initializer result is not ignored. Each shadow's fresh invocation reruns the same eligible startup, as does main; startup cleanup must succeed before entry.

Both source drivers require the explicit `--allow-temporary-files` opt-in to execute File shadows or a produced File program. Emission is not itself a grant. I do not weaken PR895's deliberate rejection of `--check-shadows` on the File CLI: the new supervised compiler path invokes scalar shadow-entry modules through the granted byte API/opt-in command. The C evaluator, generic loader/COP and ordinary shadow runner cannot substitute for this path. Existing timeout/process-group/output ownership applies, with attempted host access observed before readiness failures.

For native source output, both producers emit checked File bytecode then use the qualified direct native emitter and installed package, with unique entry identifiers. The selfhost producer still owns its AST-to-byte lowering. Generated native execution has no VM interpreter fallback. Compiler publication remains behind all selected shadow success, whether the requested final artifact is bytecode, C or a native executable.

## My dependency-ordered checkpoints

1. Review this grammar/identity/publication proposal and shared schema coexistence before any code. First implement only paired descriptive source plans and catalog data access, with explicit non-admitting results. Independently review complete ownership/allocation tables before fixtures.
2. Review paired service declaration parsing, type/name resolution, clone/schema transport and the distinct NSI File generator mode. Freeze exact companion publication semantics first. Qualify malformed/collision/source-lifetime/atomic-output controls without service execution.
3. Review paired acyclic File lowering, precise match/borrow/cleanup, full module metadata, whole-program eligibility and complete final-byte conjunction. Qualify nonexecuting paired semantic equivalence and refusal controls first.
4. Review supervised per-shadow grants/startup, explicit compiler options and installed publication, then execute the full retained source corpus on Linux/Darwin with C-seed and fresh Stage1/Stage2. No private fixture or integer extern stub closes this milestone.
5. Retain concrete parent continuations: matched cyclic VM/native/fuel and a later public conjunction; finite indirect target checking; separately encoded richer/multiple borrows; paired source support for those profiles. Query-only cyclic facts never select a runtime or authorize a source loop.

My full acceptance imports the real generated binding and retains every helper and mandatory shadow. It acquires a temp, matches both arms, transfers File through helpers, writes0/255, rewinds, reads exact bytes/EOF, changes direction through rewind and consumes close. It also covers unused Ok, early return, owner joins, byte-range failure, selected initializer suppression and exact cleanup/sentinel isolation. Zero-iteration/loop cases remain mandatory parent acceptance after the matched cyclic prerequisite, not silently deleted from the complete corpus.

I compare C-seed/fresh Stage1/fresh Stage2 selected declarations, catalog maps, function/local facts, selected shadows and actual VM/native results. I run unmodified full conversion/core/package/bootstrap/product gates and actual installed outside-tree binding consumers on Linux/Darwin; selected supported strict sanitizers retain clear provider scope. I preserve first terminals and exact source/tool/artifact identities. Passing only an isolated plan, reduced source, one backend, hand-written wire or handwritten C binding cannot close6931/72556/6fc/d03c.

## My first implementation ABI and allocation boundary

I pin a non-admitting plan over at most16 explicit module requests,13 exact catalog bindings per request,64 already-resolved import aliases and256 ordinary namespace declarations. Declaration IDs are nonzero31-bit integers, unique across the complete supplied namespace; module identities are bounded nonempty byte strings without NUL, and names are bounded ASCII identifiers. Each request carries the full canonical descriptive catalog view, not a secret or authority token. The caller must have validated the actual NSI document and supplied complete checked namespace facts; this checkpoint does not read source/JSON or prove those preconditions. C and Nano compare every catalog field through the existing immutable catalog's read-only data bridge. They resolve aliases and collisions themselves. A successful plan remains explicitly nonexecuting, with future layout/import indices unset.

My C opaque plan owns one checked allocation containing rows and copied strings, publishes only after complete validation, and reports MEMORY without changing caller output. The Nano plan is returned as an owned value after complete preflight; ordinary language array/string allocation has no recoverable per-allocation status API. I therefore do not claim paired recoverable-OOM equivalence. No file is emitted by this checkpoint. Recoverable publication/process failure containment, including staging cleanup, remains a required later driver prerequisite; a Nano allocation failure cannot be relabeled a successful plan. Logical source storage is bounded to1MiB; C reports its exact owning allocation separately, while Nano's GC/runtime overhead is not falsely equated to that C sizeof.

The catalog module exposes only bounded string/integer getters over existing static catalog data. It accepts no AST, code, service operation, authority callback or alternate catalog. Both plan builders independently compare requests and construct source identity mappings; the Nano builder never calls the C plan builder or delegates lowering. This module is added only as an explicit descriptive provider; existing parser/compiler selectors and public runtime archives remain unchanged.

| First checkpoint storage | Ownership and failure boundary |
| --- | --- |
| Immutable catalog getter strings/numbers | Existing process-lifetime catalog; invalid field/member/index queries return empty/-1; no allocation, mutable globals or document-validation claim |
| Canonical descriptive view | C stages at most32768 bytes in automatic storage and copies only after complete formatting/capacity checks; Nano independently formats the same length-prefixed fields from read-only getters |
| Explicit request arrays | Borrowed immutable valid storage during C construction; typed Nano input values; at most16x13 bindings,64 resolved aliases,256 ordinary names |
| Temporary C maps | Fixed automatic arrays for272 result rows and528 namespace entries; allocation-free complete validation before publication |
| Published C plan | One malloc of header + exact row product + terminated module/name copies; checked input-plus-output logical text budget1MiB and SIZE_MAX products; one free, getters borrow live plan strings |
| Nano result | Fresh typed value with bounded row/namespace counts and the same logical text budget; GC/runtime allocation overhead is additional, and allocator exhaustion is not paired recoverable MEMORY |

I inspected `gc_alloc`, `dyn_array_new` and `dyn_array_push_struct`: allocation can return NULL, and array growth can abort. I do not assert a universal recoverable or graceful process failure behavior. No current file publication occurs; later supervised driver cleanup must contain unsuccessful child processes before publishing anything. Struct array widths in this plan remain below the current255-byte native bound; qualification must check both actual producer layouts rather than assume that from the C API's distinct struct sizes.

### I retain explicit Nano request byte extents

Before preparing fixtures I inspected `builtins_registry.c:str_length` and `nl_cstr_length`: native/evaluator string length uses strlen. The initial6c22 bare-string request could silently truncate a counted source identity containing NUL. I therefore mirror the C request span as Nano `FileSourceText { data: string, size: int }`. The caller preserves the complete decoded source byte count; validation requires it to equal the available string length and rejects NUL-containing/full-extent mismatches before identity comparison. Passing a shorter truthful span describes only that prefix on either API; neither API proves the caller supplied a complete source token. No runtime string representation changes or execution claim follows.
