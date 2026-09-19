# I execute File services only with complete owner and Result facts

I record `task_72556b6bf6c6d793e83b9cb427a8613a` under
`task_6931ec89b210421e9827fecdbb459dbb`, after merged PR821 at0955dca62.
This is a preimplementation contract. No source, codec, public selector or
service operation changes accompany it. The first requested implementation
checkpoint below is private File/Result lifetime only; each later production
checkpoint requires separate review before execution.

## My current boundary and exact missing dependencies

| Current implementation | What I must add before public File execution |
| --- | --- |
| PR811 `nsi_file.c`, `nsi_cap_private.h` | Real temporary streams, generation/context-qualified tokens, rights, explicit direction changes and consuming close are qualified privately. Language values need their own unique lifetime and invocation ownership. |
| PR814 `nsi_file_plan.c` and immutable catalog | Exact five methods/eight types and per-outcome owner facts exist. They are not source declarations or runtime authority. |
| PR821 `service_bindings.c`, `service_bindings_module.c` | Required bits1/9, section14 and import kind3 retain five exact import indices. Version1 has no File/Result layout mapping and remains non-executable. |
| `nvm_v2_layouts.c` and retained layouts | Global STRUCT/UNION descriptors and prior-index nesting can describe this finite family. Source per-kind ordinals are not global layout indices; neither is a runtime service slot. |
| `ownership_contracts.c`, `affine_state.c`, `affine_bytecode.c` | The existing complete-STRUCT and bounded record/reference flows do not establish opaque File or owned Result alternatives. Broadening their tags alone would lose obligations. |
| `nvm2c_owned.h`, VM heap/value/reference paths | Existing owner records/STRING/mixed arrays have no File finalizer or selected-owned-union payload. Ordinary union/record construction and shared aggregate access cannot produce or copy File ownership. |
| `isa.h:TAG_OPAQUE`, VM proxy instructions | OPAQUE already means a COP proxy. I do not reinterpret this tag, raw INT, ordinary STRUCT payload or `as.proxy_id` as a File token. |
| C `borrow_codegen.inc`, Nano `nanoisa_borrows.nano`, `nsi_gen.c` | Imported calls/unions are outside the current owned source path; the v0 demonstration generator erases resource/variant types to int. Neither path becomes a verified binding by changing a name. |

I preserve ordinary, affine STRING and mixed Samples profiles. This File family
gets a separate exact conjunction; I do not admit general resource fields,
callbacks, arrays of handles, arbitrary unions or foreign libraries.

## My first private value checkpoint

I introduce a private File-value runtime, separate from public NanoValue/native
carrier selection. Its opaque invocation owns one qualified `NlFileService` and
its live owner bookkeeping. Callers cannot supply a different service catalog or
cleanup callback. Calls are serialized, as required by the current adapter;
parallel/asynchronous entry is refused rather than assuming counter safety.

A live File identity includes invocation identity plus a checked slot/generation
and exactly one language owner. The host token remains private. Moves transfer
that owner, clear the source and preserve the host token; they do not acquire or
remint a host capability. An accepted explicit close invalidates the language
owner even when the host close reports an error. A stale/duplicate/wrong-context
value never closes another stream. I check liveness before touching host state;
slot/context reuse and integer exhaustion cannot make old values valid again.
C use after destruction of the context pointer remains outside its API contract;
checked stale values against a live context are inside the acceptance controls.

My private operations are acquire, move, exclusive call-scoped access, close,
Result inspection/extraction, value cleanup and invocation finish. Exact public
C names and storage layout are reviewed with production. Transfer/extraction
failure leaves source and destination unchanged; destination must be empty.
Cleanup invalidates an owner once and records its close outcome. An outstanding
exclusive access prevents move, close or a second access until the call ends.
No public raw token getter, integer conversion, generic field projection or
caller-supplied finalizer is added. The private checkpoint is bounded by the
adapter's64 live capability slots; result storage/products and generation increments
are checked. It admits only File and these catalog Results, without nested owned
Results or cycles. Later bytecode analysis retains checked existing function,
local, stack and reference bounds rather than relying on post-write validation.

OpenResult is affine because its Ok alternative owns File. Error owns only the
exact FileError scalar payload. Reading its discriminant does not duplicate or
release it. Matching consumes the result and moves exactly its selected payload;
there is no owner in Error and no second extraction. Dropping an unhandled Ok
closes the File once. The other four Results carry only scalar/ordinary-record
payloads and retain value semantics; I do not invent extra ownership restrictions
on them merely to simplify the implementation. Error fields retain the catalog
order/status, saved host errno, secondary cleanup errno, bytes, eof, consumed
and cleanup_failed. Exact unit cases have no fabricated integer owner.

I use the immutable five-method relation: temp creates an owner only in Ok;
write_byte/rewind/read_byte preserve their exclusive-borrowed File on both
alternatives; close consumes on every accepted close outcome. Invalid host-token
or wrong-rights inputs do not accidentally consume another owner. A verifier
violation is an execution error, not a fabricated successful Result.

Before a host acquisition I prepare every fallible owner/Result storage step
possible. If later publication fails, I consume the acquired token and preserve
its cleanup failure separately. Ordinary adapter failures publish the exact
Error alternative. Runtime allocation failure is a checked execution failure
unless the complete Error representation can be produced without allocation;
I never claim a Result was returned when no value was published. Borrowed I/O
failure keeps its File root live. Checked write_byte rejects int outside0..255
before narrowing or host access and returns Argument while preserving File.
read_byte returns0..255, or canonical0 at EOF. Direction switches require a
successful explicit rewind in both directions; failed positioning does not
reset the tracked direction.

Finish first cleans all owned temporaries/results/locals, then disposes and
destroys the invocation context. It preserves the first execution error and
records secondary cleanup failure; a previously successful computation cannot
publish success when required close/disposal failed. Repeated bounded-live
acquire/drop must reclaim individual streams before final disposal, not hide a
per-iteration leak behind eventual context destruction. No File/owned Result
escapes this private context's lifetime.

Private acceptance uses actual temporary streams on Linux/Darwin and both normal
and strict sanitizer builds. It covers byte0/255, EOF, both direction changes,
checked range/rights, move through several empty slots, selected-arm extraction,
unhandled Ok cleanup, all allocation prefixes, host acquisition/I/O/seek/close
faults, first-error/secondary-error retention, duplicate/stale tokens, reused
slots/contexts, exhaustion and an unrelated live sentinel. I observe actual
close and temporary-file cleanup; I do not introduce a user pathname authority.
`tmpfile`'s removal-on-close semantics remain the qualified adapter's contract.
This checkpoint admits no bytecode and generates no callable source.

## My exact nominal binding prerequisite

I propose service payload version2 for the later execution plan, while keeping
version1's exact56 bytes and non-executing behavior unchanged. This proposal must
receive its own rawcodec/module review before implementation. Required bits1/9,
section14 and SERVICE import kind remain; an old version1-only reader rejects
version2 by version/length rather than silently dropping the extra authority.

The finite proposed version2 layout is120 little-endian bytes: u16 version2,
u16 catalog1, u32 method_count5, u32 type_count8, u32 reserved0; five existing
(u32 method_ordinal,u32 import_index) pairs; then eight
(u32 type_ordinal,u32 global_layout_index) pairs. The type ordinals are the
immutable plan's order, not inferred spelling or declaration order. All eight
indices are distinct/in range and never NO_INDEX. They identify exactly File,
FileError, ReadByte and the five Result types; validators compare the complete
catalog mapping, layout/case/field order, nested nominal identities and exact
scalar tags. Index products/offsets are checked before decode/allocation; output
publication and bridge rollback remain staged. Unsupported versions, malformed
counts, missing/duplicate maps and mismatched catalog identities refuse.

File has a dedicated opaque semantic identity in the execution plan. Its
retained empty resource descriptor is not a zero-field constructible record.
FileError and ReadByte have exact scalar fields. The five union descriptors
have exact Ok/Error order and payload identity, with prior-order nested edges.
OpenResult's ownership comes from the checked catalog relation, not from a
shared ordinary UNION tag or a newly permissive global ownership validator.
I retain explicit maps among catalog type ordinal, global layout index, source
STRUCT/UNION ordinal and runtime category. A later publisher may choose different
global indices; every map must be checked without renumbering existing modules.

Version2 metadata describes the requested interface; it grants no host authority
by itself. Hand-written bytecode may request the same exact contract only if it
passes the full verifier and receives the explicit host grant. There is no
compiler-secret signature or trusted spelling shortcut. Source and bytecode
cannot construct a File via record literals, OWN_PACK, generic AGG_PACK, casts,
constant pools, integers or copied foreign memory. They also cannot insert an
owner into an ordinary shared union or extract it through generic AGG_GET.

## My shared flow and matched VM/native execution checkpoint

Before selector admission I define exact File/Result local, stack, helper and
branch facts. Each CFG join conserves owner obligations and exact nominal
identity. The discriminant test and selected payload transfer must expose a
checked refinement; generic scalar equality is not assumed to refine a Result.
The instruction encoding/decoding/source lowering for that refinement gets a
separate reviewed checkpoint; no unassigned opcode is reserved by this document.
If CALL_EXTERN is retained for service calls, kind3 must route exclusively to
the certified service plan, never to generic FFI/COP/dlsym. Old selectors keep
refusing until the complete new verifier and both runtime paths are present.

Each call checks exact catalog method, declared argument/Result identity,
copy/exclusive/consume mode and current rights. Static flow and runtime tag,
invocation, generation and liveness checks are both required. CALL/RET retain
pending argument and result roots until ownership commits; a failure before or
after commit cleans exactly the owners it actually acquired. No scalar/ref path
may observe a File payload. Ordinary scalar helper calls remain exact; internal
File/OpenResult helpers transfer once and have checked complete exit obligations.

The host explicitly enables only the exact temporary-file catalog for an
invocation. No existing default VM/native entry silently gains filesystem
permission. A service-aware host entry owns the context, isolates nested/reentrant
invocations, and refuses absent/mismatched host bindings before side effects.
Public incoming owners, File/OpenResult returns, globals, closures, callbacks,
asynchronous work and serialization of live owners remain refused. Internal
helpers may return File/owned Result. All public entry/export routes must enforce
this restriction, not merely a function named main.

VM and native AOT share the qualified adapter and private lifetime core. Native
code must use the same implementation through a checked generated package or
explicit fixed runtime linkage, not a copied handwritten cleanup implementation.
The exact packaging choice and full source/object/provider lists are reviewed
before qualification. Ordinary standalone scalar emission remains unchanged.
LLVM/Wasm, reconstruction and other consumers continue explicit refusal until
they receive their own applicable host-service contract. This does not turn
public C host execution into a universal backend completion requirement.

## My paired source and full-shadow acceptance

After the runtime conjunction qualifies, I add a distinct executable File binding
generator mode from the exact checked NlFilePlan. V0 demonstration generation
remains visibly separate. New ownership JSON keys are still rejected; immutable
catalog relations are not falsely advertised as serialized NSI v0 fields.
Unsupported generators refuse this new mode atomically rather than returning
int demonstrations. Generated destination publication occurs only after full
plan/type/name validation.

The binding presents opaque File, the exact Results and five methods to both
frontends through checked declaration identities. User declarations, aliases,
imports or sanitized-name collisions cannot acquire builtin service authority.
Ordinary shadowing follows actual language binding rules and cannot silently
resolve to the host adapter. The source representation/parser/checker changes
for opaque service declarations and owned match arms get their own paired
checkpoint; this contract does not assert existing opaque extern types already
satisfy affine ownership. Both C and Nano publishers emit the same method/type
maps, truthful signatures and exact local/Result facts, with checked function,
local, stack, nesting and layout limits before indexing.

My acceptance source retains its complete imported binding, every helper and
mandatory shadow. The program acquires a temp, matches Ok/Error, moves File
through an internal helper, writes0 and255, rewinds, reads exact bytes and EOF,
rewinds to change direction again and consumes close. Error arms exercise exact
ownership without invented success values. Additional controls cover unused Ok,
early return, branches/zero iterations, nested helpers, failed publication,
wrong byte domain, missing host authority and duplicate/escaping-owner refusal.

C-seed, fresh Stage1 and Stage2 compile the same source and all shadows under
normal supervision. I execute their pinned modules in the ordinary verified VM
and generated native AOT on Linux/Darwin, with fresh actual tool/input inventories.
Every shadow invocation gets its own explicit context; deterministic fault
controls select only test adapter faults, while ordinary success paths operate
on actual files. I do not skip effectful shadows, substitute imported stubs,
partition the source, or mistake an integer extern wrapper for ownership proof.
I preserve all first terminals before corrections and check actual cleanup,
sentinel isolation and output preservation. No private query/carrier checkpoint
closes the full6931/d03c/ed702 source/execution obligations.

### My private API details before implementation

For checkpoint1 I choose a fully serialized C API precondition, including context
creation and use alongside the underlying private adapter. I add no unsynchronized
busy flag and make no thread-safe or concurrent-entry refusal claim. Later public
entry synchronization remains a separately reviewed requirement.

An opaque `NlFileValues` invocation owns64 slots. Checked value handles carry an
invocation identity, slot and monotone generation; they expose no host token.
Moving a File/OpenResult or taking Ok advances only the language generation, so
old copied handles become stale while the adapter token stays unchanged. Empty
source/destination handles are explicit; self/overlapping transfer is refused
unchanged. Slots at exhausted value generation or borrow epoch retire after cleanup. OpenResult.Error
uses the same bounded affine result slot until extraction/drop, without a File.
The other Result payloads are returned by value and require no owner slot.

Borrow handles additionally carry a monotone call epoch. Finish invalidates any
remaining borrow during terminal cleanup; it does not race a running call under
this serialized contract. Checked operation failures preserve owner/output state.
Successful service Error publication is distinct from execution-status refusal.
Result storage is embedded in the preallocated slots: no fallible allocation
occurs after host acquisition in this checkpoint, and underlying adapter mint
failure retains its qualified rollback behavior. Capacity exhaustion before a
Result slot is available is a checked execution limit, not a fabricated Error.

Explicit close returns its exact scalar CloseResult. Any close/rollback cleanup
failure is also retained in the invocation finish report, preventing a later
caller from silently publishing a clean overall completion. Read/write/rewind
Errors retain their File and do not themselves become cleanup errors. Finish
accepts the first pending execution status from its caller, retains the first
cleanup event plus the first later event and a checked count, cleans slots before
service disposal/destruction, and caches its terminal report for idempotence.
All output pointers must be valid and disjoint from context storage; differently
typed result/handle output objects must not overlap. No use after context
storage destruction is supported.
