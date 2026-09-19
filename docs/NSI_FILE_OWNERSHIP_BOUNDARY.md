# I connect file service identity to verified ownership

I record `task_6931ec89b210421e9827fecdbb459dbb` under d03c, with ed702
transport obligations, from canonical `9821e390`. PR811 qualifies only my private
real-file adapter. This is a source audit and preimplementation contract. I do
not change code, generate fixtures, execute services or admit imports here.

## My actual starting boundary

| Current source | Established behavior and missing connection |
| --- | --- |
| `src/nsi.h`, `src/nsi.c:parse_type` | I retain parameter direction, ownership, lifetime and mutability; resource types and variant cases exist. I have no per-method owner-state relation for each Result alternative. Strict allowed-key validation means adding such facts needs a reviewed schema extension, not ignored JSON keys. |
| `schema/nsi/modules/filesystem.nsi.json` | My existing `open(path)` declares transferred File and an interface error name. It does not describe Result cases, close, rights effects or actual path authority. I leave this existing pathname contract unchanged in the first temporary-file binding. |
| `src/nsi_gen.c:nano_type`, `nl_nsi_gen_nanolang` | RESOURCE and VARIANT map to int; every generated Nano function returns int and has a demonstration body. These are not verified resource service bindings. Other generators must preserve the new contract or explicitly refuse it; their demonstration output cannot silently stand in for it. |
| `src/nsi_runtime.c:dispatch` filesystem branch | I mint a descriptive handle and return JSON. That path does not call the PR811 real-file adapter. No current JSON response establishes affine ownership. |
| `src/nanovirt/borrow_codegen.inc:borrow_collect`, `src_nano/compiler/nanoisa_borrows.nano` | My selected owned source profiles require import-free defined functions and bounded resource records; the selfhost path rejects unions. Existing normal extern lowering does not supply service authority. |
| `src/nanoisa/nvm_v2_imports.c`, `nvm_format.h:NvmImportEntry` | Imports carry names, signature and kind, not service ownership, rights or per-outcome transfer facts. Ownership declarations describe layouts/functions/locals, not this imported-call state transition. |
| `src/nanoisa/ownership_contracts.c:check_layout_facts` | COMPLETE ownership layouts are STRUCT; there is no admitted resource-bearing Result variant here. A source Result needs new checked affine variant facts rather than an ordinary shared union containing an owner. |
| `src/nanoisa/verifier.c:nvm_verify_owned_module`, `affine_bytecode.c:nvm_affine_value_call_graph` | I reject import-bearing owned modules and CALL_EXTERN; the owned whitelist lacks Result construction/extraction and service operations. Ordinary import signature verification is not lifetime verification. |
| `src/nanovm/vm.c` CALL_EXTERN/trap path, `vm_ffi.c` | Arguments leave the stack for generic FFI then are released. There is no exact service-owned return/borrow/consume transition or rollback policy. I must not route owner payloads through raw integer/pointer FFI conversion. |
| `src/nanoisa/nvm2c_owned.h` | My native owner carrier recursively frees record storage. It has no real file finalizer or opaque service payload. Adding a token integer field would not make cleanup close its file. |

These are known unmet integration obligations, not newly executed failures. I
preserve both the private adapter and all existing ordinary/owned admission.
Current descriptor work for mixed arrays and STRING fields remains independent;
this first file path does not depend on their managed-field expansion.

## My first exact descriptor plan is non-admitting

I first add a private checked descriptor/query and generator plan, separately
reviewed before implementation. It accepts only this closed local-file family.
Its published plan owns its storage; on any invalid input, unresolved binding,
allocation failure or limit, caller outputs and generated destination remain
unchanged. Generation preflights the entire plan before output publication.
I compare full identifiers, reject duplicates and generated-name collisions, and
never infer method authority from a spelling prefix or sanitized name alone.

The version-one logical contract contains exactly:

- Interface ID `nsi:nanolang/filesystem`, resource ID
  `nsi:nanolang/filesystem#File`, method ID and an explicit binding ABI version.
- Ordered parameter and result type IDs; parameter mode is copy, call-scoped
  exclusive borrow, or consume. Unknown enum values, optional/streaming/async
  parameters and unsupported combinations refuse this finite plan.
- Exact required rights per method, operation identity, host binding identity,
  and whether a successful call creates, preserves or consumes the File owner.
- Result nominal type identity, ordered variant IDs and exact payload identities.
  Each alternative lists owned payload positions and their resource IDs. No
  ordinary numeric field is treated as an ownership slot.
- Every input owner's postcondition for every alternative: still live outside
  the result, moved to a specific result payload, or consumed. Borrowed calls
  cannot return the borrow or destroy its root. Acquisition creates an owner
  only in its success alternative. Close returns no owner on either alternative.
- Error payload type identity and exact scalar field order below; capability
  context, generation, secret and host pointer are absent from source payloads.

The first plan supports at most five methods, one service resource, two variants
per Result and eight fields per ordinary payload. Counts and allocation products
are checked before allocation. These are implementation bounds for this child,
not a reduction of d03c. I reject unsupported documents instead of truncating.

I do not store these authoritative facts in optional debug text. A later checked
NanoISA transport must make their presence execution-significant, bind them to
exact import/signature/layout indices, and reject unsupported consumers. Its
numeric section/kind assignments and wire bytes need a separate codec review
before implementation. The first private query publishes no executable .nvm
extension and leaves all current selectors unchanged. I do not guess an unused
opcode or claim an opaque metadata string is sufficient transport.

## My representative binding keeps the host boundary small

I add distinct temporary-file method IDs; I do not redefine existing `#open`.
The following is a signature/ownership table, not proposed new source syntax.

| Exact suffix under filesystem | Inputs | Success | Error | Rights |
| --- | --- | --- | --- | --- |
| `#temp` | none | `OpenResult.Ok(File)` | `OpenResult.Error(FileError)` | Creates READ/WRITE/TRANSFER owner |
| `#write_byte` | exclusive call borrow File, copied U8 | `WriteResult.Ok(progress:int)` | `WriteResult.Error(FileError)` | WRITE |
| `#rewind` | exclusive call borrow File | `PositionResult.Ok(unit)` | `PositionResult.Error(FileError)` | READ |
| `#read_byte` | exclusive call borrow File | `ReadResult.Ok(ReadByte)` | `ReadResult.Error(FileError)` | READ |
| `#close` | consumed File | `CloseResult.Ok(unit)` | `CloseResult.Error(FileError)` | Current owner, no additional right |

`ReadByte` is `{value:U8, eof:bool}`: EOF has zero progress and canonical value0;
a successful byte has progress1 and eof=false. `FileError` has exact ordered
fields `{status:int, host_errno:int, cleanup_errno:int, bytes:int, eof:bool,
consumed:bool, cleanup_failed:bool}`. Unit has no payload. Status refers to the
versioned private adapter statuses; host errno is target-specific and is not
promised numerically equal across operating systems. Transfer of language
ownership through a local/helper does not duplicate or remint a host token.
Explicit service token rotation remains separately required by the broader
capability migration; the private adapter already tests its own rotation.

The byte wrapper uses a fixed local byte buffer and the existing checked length
API, preserving NUL without needing managed arrays or returning caller memory.
It never advertises arbitrary-buffer, pathname, socket or GPU authority. Read,
write and seek mutate stream position, so their call borrows are exclusive even
when the byte buffer is immutable. Both update-stream direction restrictions
and partial progress/error semantics remain exactly PR811. I do not hide a
positioning call in a wrapper. A failed borrow operation retains its File root.

Close moves the owner into the adapter boundary before invocation. A valid
accepted close attempt consumes it even when fclose reports error. A rejected
raw/stale token closes nothing; a verified source cannot manufacture such a
File. Host context disposal invalidates outstanding tokens: subsequent close
reports rejection and consumes only its language shell, never an unrelated
stream. The error's consumed field retains the host-adapter fact; the language
postcondition is separately always consumed for this close API. This distinction
must be explicit in verifier facts and tests, not inferred from that bool.

## My opaque owner and Result prerequisites

File must be a distinct opaque service owner, constructible only by an approved
service result, not a public resource record with packable integer fields. I
require exact nominal/service provenance in verifier state and runtime values.
DUP, scalar casts, generic field projection, structural reconstruction from
numbers, ordinary union insertion and a mismatched service call refuse it.
A binding name or matching descriptor alone does not grant host authority: the
execution host must explicitly register the exact supported local service ABI.
No arbitrary dlsym/library lookup resolves this service contract.

An owned Result is affine if any alternative owns a File. Moving it transfers
its selected payload once; matching consumes that Result and moves only the
selected payload into its arm. Error arms do not fabricate an empty owner.
Joins preserve exact alternative/owner state, including zero-iteration branches,
helper returns and early exits. No owner may be copied out with ordinary union
field operations. Non-resource Results use their exact declared scalar payloads.
This requires separately reviewed variant descriptors, flow and runtime cleanup;
I do not relax the old complete-STRUCT rule or import refusal in isolation.

The runtime invocation owns the file-service context. Values and pending results
retain its lifetime until their cleanup completes. Each File shell has a unique
live/consumed state and context-qualified token. Normal close invalidates the
shell before releasing it. Unhandled error, failed result allocation, VM trap,
AOT error, and abandoned Result must close each still-live shell once; cleanup
preserves the first execution error and records secondary close failure. All
shell/Result allocation must be prepared before acquisition where possible; if
publication after acquisition fails, the private adapter consumes the new token.
Context disposal is the final fallback after value-root cleanup, not a substitute
for releasing individual files during repeated bounded-live execution.

## My ordered review and acceptance checkpoints

1. I implement only the private logical descriptor/query and representative
   generator plan after this contract is reviewed. Existing v0 generation stays
   distinguishable; the new path refuses unsupported documents atomically.
   Tests cover exact five-method relation, identities, per-alternative ownership,
   invalid modes/counts, duplicate names, output preservation and allocation faults.
2. I separately contract versioned NSI schema and NanoISA transport, with exact
   bytes/index remapping, before coding them. C-seed and selfhost emit identical
   facts; assemblers, codec round trips, linker and disassembler preserve them.
   Reconstruction and every shipped translator preserve/consume or explicitly
   refuse this boundary without partial output. Unknown versions never silently
   become ordinary imports. This is a bounded contribution to ed702, not closure.
3. I review opaque File and affine Result state/cleanup implementation privately.
   Tests distinguish observation from transfer, each selected result arm, call
   failure before/after ownership acquisition, constructor allocation rollback,
   repeated invocation, context disposal and stale/duplicate isolation. VM and
   native carriers need matched semantics; LLVM/Wasm consumers retain explicit
   refusal until their separately applicable host-service contract exists.
4. I review shared service-call verification together with VM and native AOT
   lowering/dispatch. Only the exact registered descriptor family is admitted.
   Rights, runtime tag/context/generation checks and owner-root cleanup accompany
   static facts. No public admission-only checkpoint and no generic extern escape.
5. I generate the representative binding and unchanged normal-supervision shadows
   through both frontends, including C-seed and fresh Stage1/2 producer pins.
   Execute one pinned .nvm corpus in VM/native AOT against real temporary files:
   acquire, match, write byte0, rewind, read byte0/EOF, move through a helper, close;
   cover both I/O directions, checked errors and fault cleanup. Both platforms
   retain actual tool/input inventories, close counts and sentinel isolation.
   I do not weaken shadows to avoid effects; each gets an isolated service
   context and deterministic adapter fault controls, with ordinary successes
   exercising actual files. First failures remain sealed before correction.

The smallest next implementation is checkpoint1, not general foreign execution.
No contract alone qualifies a generated binding. I keep File path acquisition,
Socket/GPU adapters, capability integration, representative standard library,
full d03c, ed702 and 5.1 release acceptance open. I retain existing task owners
and the separate mixed-array/STRING work without duplicating their source edits.
