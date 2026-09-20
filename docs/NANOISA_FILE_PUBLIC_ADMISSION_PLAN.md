# I join explicit File grants with matched public byte execution

I record child `task_dfa149b7e14e4d0592eb00ecb778b3ea` under6fc/72556/82ff,
on actual grant PR887 merge `dc83a0c92432534226d50b422a9d0b8e78657f79`.
This is the concrete implementation/test inventory for my
[approved conjunction](NANOISA_FILE_PUBLIC_CONJUNCTION.md), not new execution
acceptance. Private native9a remains owned by vm_effects. Its platform gates are
reported passing, but its final seal, independent review and canonical merge
remain prerequisites. I change no source or run a service in this document.

## My exact public and internal boundaries

I add `src/nanoisa/file_public.h` with the existing opaque grant header,
`NvmFileScalar { uint8_t tag; int64_t value; }`, and declarations for:

- `NvmFileRuntimeReport nvm_file_execute_bytes(NvmFileHostGrant *,
  const uint8_t *, size_t, NvmFileScalar *)`.
- `NvmFileRuntimeStatus nvm2c_emit_file_bytes(const uint8_t *, size_t,
  const char *entry_identifier, char **out, char *diagnostic, size_t)`.

The runtime report/status are the actual existing `file_runtime.h` types. I do
not duplicate enum numbers, drop cleanup fields, or claim this native C ABI is a
wire encoding. My first installed header package preserves the exact transitive
header closure rather than moving existing report types during activation.
Public scalar tags are the existing INT=1 and BOOL=4; BOOL values are only0/1.
Inputs and all outputs retain the existing valid, disjoint C storage precondition.

The VM public wrapper in proposed `src/nanovm/file_public_vm.c` enters the grant
once, before any plan allocation, then calls the same exact serialized engine
used by the qualified private VM. It uses a local passive RuntimeView and publishes
NvmFileScalar only after successful terminal cleanup and exact scalar validation.
Grant statuses map explicitly: INVALID, MEMORY, STATE, UNRESOLVED and BUSY to the
same-named runtime statuses. A pre-acquisition report has acquired=false, no
invented site and empty cleanup, following the existing refused-report convention.
An impossible malformed successful scalar becomes TYPE with the original cleanup
report retained, and never touches caller output. All exits release only the gate
actually acquired by this wrapper, after context destruction. A failing/reentrant
inner call neither releases nor finishes its caller's active invocation.

I factor the existing private VM body into proposed `file_vm_engine.inc` as one
static serialized engine implementation. The existing macro-only qualification
entry remains a thin wrapper with its unchanged signature. The public TU includes
the same engine but exports only the granted byte entry. Default builds do not
export the ungranted private qualification symbol. I review the exact extraction
and rerun its original corpus; I do not hand-copy a second evolving interpreter.

Native emission uses a reviewed internal surface parameter at emission time,
not text replacement over generated C. Private emission retains its historical
entry under its private macro. The public emitter validates the approved1..63
ASCII identifier and enters the query gate without creating a host grant. Every
exit frees its temporary plan/buffer before leaving. Invalid arguments/names
preserve `*out`; diagnostic storage may be updated within its documented bounds.

Public generated C exports only `nvm_file_program_<identifier>`, with signature
`NvmFileRuntimeReport (NvmFileHostGrant *, NvmFileScalar *)`. Its exact existing
functions/labels, embedded bytes, fact agreement and NATIVE frame operations stay
static. The exported wrapper enters the grant before ABI checks/preparation,
stages the private passive result, completes cleanup and publishes only the exact
scalar. It cannot call the VM engine. The owning carrier's semantic ABI check
remains real; public header agreement alone is insufficient. Any needed macro
factoring of that check is reviewed without changing its semantic revision or
turning the private ungranted callable into a default exported API.

Generated units remain C99-compatible and contain no atomic types/gate definition.
The linked `file_host_grant.c` alone is compiled with the explicit C11 recipe.
Two differently named generated programs use exactly one owning runtime and grant;
duplicate names fail the actual link. Wrong ABI/facts and unavailable runtime
objects fail before host effects or at the explicit link boundary, respectively.

## My actual build and installed package

Current `nvm2c-runtime` supplies only `nano_aot_runtime.o` for array/GC imports.
It is not my File package. Current `install` installs nano_vm but omits nvm2c;
private generated C includes repository-relative headers and assumes test macros.
I must correct these concrete product gaps as part of this public milestone.

I add a distinct `file-public-runtime` Make target producing
`lib/libnano_file_runtime.a`, plus an explicit header manifest. The archive has
one grant object, carrier/frame object, `nsi_cap`, `nsi_file`, `nsi_file_values`,
public VM wrapper/engine, public native emitter and the following existing query
provider closure (names below are source stems, with their existing directories):

```
nanoisa/affine_bytecode nanoisa/affine_state nanoisa/file_flow nanoisa/isa
nanoisa/managed_array_shapes nanoisa/mixed_float_proof nanoisa/nvm_format
nanoisa/nvm_format_v2 nanoisa/nvm_v2_constants nanoisa/nvm_v2_convert
nanoisa/nvm_v2_cursor nanoisa/nvm_v2_functions nanoisa/nvm_v2_imports
nanoisa/nvm_v2_layouts nanoisa/nvm_v2_module nanoisa/nvm_v2_signatures
nanoisa/ownership_contracts nanoisa/passive nanoisa/reference_places
nanoisa/retained_layouts nanoisa/service_bindings nanoisa/service_bindings_module
nanoisa/service_file_nominal nanoisa/service_file_nominal_plan
nanoisa/verifier nanoisa/verifier_types nanovm/vm_decode nsi_file_plan
```

I derived this candidate closure from current sources and a read-only symbol walk
of retained private VM428 objects. That is a static planning aid, not current
link qualification. The source checkpoint must verify the exact final native
provider references, archive member names and unresolved system symbols. It may
not replace a missing provider with a stub, weak symbol or full compiler-object
bundle. No compiler evaluator, generic VM interpreter, loader, COP or module
facade is needed by the observed query closure. Public native link acceptance
must prove the final executable does not extract a VM engine/interpreter object.
The public VM and emitter objects occupy separate archive members, allowing a
native-only caller to leave them unextracted.

The proposed installed root is `$(PREFIX)/include/nanolang/file/`. I preserve
`nanoisa/` and sibling `nsi_*.h` relative structure beneath it. Installed generated
C uses `<nanolang/file/nanoisa/file_native_public.h>`; host code uses
`<nanolang/file/nanoisa/file_public.h>`. The explicit baseline transitive manifest
is file_body.h, file_code.h, file_flow.h, file_hosted.h, file_runtime.h,
file_runtime_frames.h, generated_schema.h, isa.h, nvm_format.h,
nvm_format_v2.h, nvm_v2_sections.h,
service_bindings.h, service_file_nominal.h under nanoisa, and nsi_cap.h,
nsi_file.h, nsi_file_values.h at the root. I add the public/native/grant headers
and any exact new native-fact header dependencies found during source factoring.
No `.c` or `.inc` lookup occurs from generated code or an installed invocation.
The native detail header exposes only the checked internal declarations needed
by generated functions, retaining their trusted-C preconditions. Arbitrary native
C is still not a security sandbox.

`install` depends on the public package and nvm2c, installs the archive under
`$(PREFIX)/lib` and both commands under `$(PREFIX)/bin`. `uninstall` removes only
these explicitly owned package files. Qualification uses a fresh temporary PREFIX,
then compiles/executes from an unrelated directory using only installed headers,
archive and selected system compiler. No repository include path, environment
search fallback, unpublished object or private macro can make this check pass.
I retain the actual archive/indexer tools and installed bytes. I do not fold the
File gate into NanoISA/Forth module manifests or wrapper object lists, because
those consumers remain non-admitting. Ordinary CLI links use the single archive
owner; no separate grant object may duplicate its identity/gate.

## My explicit installed command routing

`nano_vm --allow-temporary-files input.nvm` opts into exactly this bounded File
profile. Without the flag the existing public pending-service guards remain
first. The opt-in route reads bounded raw serialized bytes directly, creates a
fresh grant, invokes the new byte API and destroys its grant. It neither calls
vm_init nor preloads FFI imports. An input that is not the exact File profile is
refused on this explicit route; ordinary invocation without the flag is unchanged.

The first File CLI route rejects combinations with daemon, check-shadows,
verify-only, repeat, profiling, COP/isolate-ffi, debug and guest arguments before
service/grant creation. These modes require separate reviewed ownership/startup
semantics; none can silently fall through to the old VM. The CLI returns1 with
a diagnostic on failed execution/cleanup; successful INT returns its low8 bits,
BOOL returns its canonical0/1. Full64-bit scalar equality is tested through the
public API, not inferred from an eight-bit process exit. There is no new stdout
result format. Later paired-source shadow integration remains required work.

`nvm2c --file-temporary --entry-name IDENT input.nvm [-o out.c]` selects the
nonexecuting File emitter. Both options are required together; neither an ordinary
module nor missing opt-in falls back into File emission. The flag is an explicit
emission request, not a grant for the subsequently compiled program. The host
must still create and pass a live grant for every invocation. Default nvm2c
continues through its existing module emitter/refusal path.

The explicit File input reader checks open/seek/length/read/close errors and the
existing16MiB hosted input bound before allocation/narrowing. Reading the input
file and writing a requested C output are CLI host I/O, distinct from bytecode
service acquisition or dynamic-loader attempts. Tests count those separately.
The File `-o` path stages the complete emitted text and writes/checks/closes a
same-directory temporary before rename; earlier failure preserves an existing
output file and cleans temporary files. Stdout is written only after successful
emission, but a downstream I/O failure can expose partial stdout: I promise no
rollback of bytes already observed by a pipe. Existing ordinary emission behavior
is not silently changed by this scoped path.

## My complete review and qualification inventory

| Boundary | Required controls before acceptance |
| --- | --- |
| Public bytes and policy | Exact full native/VM corpus through real byte APIs; NULL/revoked/incompatible grant, malformed/version1/truncated/forged catalog, unknown bits/dead opcode/mask and ABI mismatch; unchanged scalar/C-output sentinels and zero attempted service acquisition/loading before readiness |
| Exact startup/ownership | VOID-only selected initializer and scalar-only entry, initializer cleanup suppresses entry; internal File/OpenResult returns, all partial call/return/service roots, formal-before-origin cleanup, both service arms, byte/direction errors and first/secondary failures |
| Shared gate | VM/native, two named native programs and nonexecuting emitter share one grant/gate; deterministic contention and synchronous reentry; BUSY cannot release outer gate, mutate its report or destroy its roots; revoked grant and post-failure fresh invocation |
| Native identity/package | All1/63 identifier bounds and invalid64/characters, duplicate-name actual link failure, distinct names coexist; exact current ABI/facts at invocation, C99 generated units, C11 owning gate; no VM interpreter extraction in native-only executable |
| Allocation/lifetime | Full query/carrier/host-grant/emitter allocation prefixes and transient failures, no post-begin project allocation, staged outputs unchanged, zero tracked leaks and real FD cleanup; modeled fclose/I/O reporting failures labeled separately from arbitrary libc failure |
| Installed CLI | Fresh PREFIX, actual installed command hashes, outside-repository compilation/link/run, explicit flags and incompatible combinations, missing-grant/opt-in counters, input-reader limits/failures, existing-file preservation, clean temp removal and checked stdout failure |
| Old consumer boundaries | Direct old vm_execute/call/invoke/callable, generic verify/linked/mixed/owned, ordinary nvm2c, wrappers/daemon, supplied-module/actual-blob mismatches, LLVM/Wasm/reconstruction retain facts/refusal; ordinary neighbors stay accepted |
| Platforms/evidence | Fresh Linux GCC/Clang and puck Apple/Homebrew ordinary/O0/O2 native plus strict scoped ASan/UBSan/LSan, actual SDK/compiler/archive/link maps, real source/provider before/after identities, bounded process-group logs/first terminals and all retained binaries; every advertised report is a committed Git blob |

I reuse complete qualified fixtures through explicit public wrappers with the
same assertions; I do not drop unused helpers, selected shadows or hard fault
cases. Test-only mutation of copied facts or opaque fields stays labeled as such.
Coverage of the expanded public preamble, scalar conversion and gate release is
new acceptance, not inherited merely because private execution passed.

My implementation order is: first review this exact API/package/CLI amendment;
then integrate the actual qualified native merge in a new tree; then send the
complete joint production delta (including engine extraction, gate routing,
archive/install and CLI changes) for source review; then send complete fixtures
for independent review; only then execute new public service gates. The existing
private full acceptance and grant evidence remain pinned and untouched. No
admission-only intermediate commit or test executes a half-implemented target.

A passing result here closes only this bounded public byte/installed-package
child. All loops/backedges, checked indirect callee sets, needed richer-borrow
transport, executable generated NSI bindings, paired C-seed/Stage1/Stage2 source
publishers and complete selected shadows remain concrete full5.1 work under
72556/6931. File/Socket/GPU and full One-IR obligations under d03c/ed702 remain
open. No parent closure follows from this first acyclic public profile.

## My actual implementation base

I begin the reviewed joint source checkpoint only after actual private-native
PR888 merge `f1606e2c84e67491e9652a5bf71944d235216d95`. Root independently
reviewed its final evidence and approved this plan. My fresh branch integrates
that canonical merge plus the two reviewed precode commits; qualified native,
VM and grant source/report trees stay immutable. Engine factoring and the native
surface selector must preserve the private generated bytes and old fixture APIs.
No new public fixture executes until complete source and fixture reviews.


## My complete first production checkpoint

I factor the qualified VM body into one shared internal include and keep its
private macro wrapper. My public byte entry holds the shared grant gate around
that same body, terminal destruction and scalar publication. The extracted VM
body is text-identical to the actual888 body apart from its static function name.
My native emitter selects either the existing private surface or an installed
public surface with one validated exported identifier. The generated public
entry holds the same gate around direct compiled functions and labels; it does
not call the VM engine. Private generated-byte equality remains a qualification
gate, not a conclusion from this source comparison.

| Storage or resource | Owner, bound and terminal action |
| --- | --- |
| Host grant | Existing owning C11 object; explicit create/revoke/destroy, retained across invocations; a successful enter alone owns one matching leave |
| Hosted plan, carrier, frames and NSI context | Existing runtime create/begin/destroy; unchanged conservative64MiB project-owned bound and no new post-begin allocation; destruction precedes public scalar publication and gate release |
| Public adapter scalar/view/report | Fixed automatic storage; no new heap root, mutable-module cache or escaping owner; rejected or unclean execution preserves the caller scalar |
| Native emission | Existing hosted plan plus checked output builder bounded at128MiB; old builder freed on failure and plan freed on every terminal; malloc-owned C text publishes once after complete generation |
| CLI input | One checked positive input allocation at most16MiB, freed after execution/emission; checked close before byte publication; host input I/O is separate from guest service acquisition |
| CLI C output | One checked pathname allocation and one same-directory staged file; write/flush/close precede rename, failed publication attempts unlink and preserves the earlier output; primary error is captured before cleanup |
| Installed archive | One freshly staged archive assembled from35 explicit provider objects; checked shell failures prevent partial archive publication; no compiler, loader, COP or generic VM interpreter object belongs to this archive |

The installed header closure has22 exact transitive files. The generated native
unit references carrier/query/grant members only; the separate public VM archive
member is extracted only by its byte API. Actual linker extraction and an
outside-tree installed executable remain required measured gates. Archive and
header-install recipes stop on the first command failure. Header dependencies
cover the public providers, and the owning grant object retains its explicit
C11 recipe while generated and host consumers remain C99.

I add only explicit CLI branches: `nano_vm --allow-temporary-files` and paired
`nvm2c --file-temporary --entry-name IDENT`. My default verifier, VM, mixed/owned
selectors, generic converter/emitter and wrapper guards are unchanged. Their
ordinary/refusal neighbors remain in the qualification inventory. Public input
and output objects obey ordinary C validity and disjointness preconditions;
trusted generated-detail headers are not a security boundary. The shared gate
does not make unrelated direct private query calls safe under concurrent entry.

At this checkpoint I have performed static source comparison, transitive-header
closure inspection and whitespace checks only. I have not compiled this new
source, prepared public fixtures or executed a service. Complete production
review precedes fixture preparation; complete fixture review precedes gates.

## My first complete public fixture checkpoint

I retain every original private VM/native corpus assertion. Small defaulted
fixture selectors route the new public suite through the actual public scalar
API or exact-byte registered generated native functions; old suites keep their
original paths. The bridge checks the public scalar sentinel before reconstructing
the passive view expected by inherited assertions. Manual carrier/frame controls
remain labeled as manual. The old grant-less route checks remain refusals.

My new controls include null/revoked grants, incompatible internal ABI/catalog,
malformed/truncated/version1/required-feature/catalog bytes, unsupported startup
signatures, output preservation, all identifier length/character boundaries,
16 deterministic contending threads and synchronous reentry during real service
operations. A held gate is shared by actual VM, native program and nonexecuting
emitter calls; failed entrants cannot release it. Existing allocation-prefix,
transient, generation, partial-service, initializer suppression and first/secondary
cleanup assertions run through the public adapters. The grant is created outside
the per-invocation allocation ledger; its complete allocation/lifecycle controls
remain the separately qualified grant neighbor, and sanitizer leak detection
covers its actual lifetime here.

I build the real `make -B -j2 install` package once per selected configuration,
with C99 consumer flags and the explicit C11 grant recipe checked in the actual
command log. My package controls use the installed commands, all22 installed
headers and the actual installed archive from a directory outside the repository.
Two named native programs share one grant and gate, execute the real temporary
file/write/rewind/read/close path, retain scalar sentinels on refusal, and leave
no additional observed descriptors in the scanned0..1023 range. Native-only
symbol inspection rejects extraction of either File VM entry or generic VM
execution. Duplicate names must fail an actual link with a duplicate-symbol
message. Separate generated ABI/fact mutations refuse before acquisition and
then permit an unchanged second program with the same grant.

Installed CLI controls cover absent opt-in, incompatible modes/guest arguments,
invalid names, bounded/empty inputs, earlier-output preservation, failed rename
with no staged file, and an actual read-only stdout descriptor producing a checked
write failure. A separate C99 CLI fixture performs real I/O while modeling
allocation, partial-write, flush, close, rename and fdopen errors; it checks saved
primary errors, exact pointer/length sentinels, staged-file cleanup and real close
observations. Modeled post-close/post-I/O reporting remains explicitly distinct
from arbitrary host libc failure behavior.

For private generated-byte parity I freshly compile the exact emitter source
from actual888 `f1606e2c84e67491e9652a5bf71944d235216d95`, under a distinct test
symbol. I compare and retain its C output against the factored private emitter
for every accepted linked capture case. I do not execute any old retained binary.
The new public generated corpus compiles as C99 at both O0 and O2. The fixtures
retain source, commands, stdout/stderr, status and bounded process-group cleanup
records, and clear inherited LSAN_OPTIONS while requiring leak detection1.

After independent fixture review my frozen configuration order is Linux GCC
ordinary/strict sanitizer and Clang ordinary/strict sanitizer, then isolated
puck AppleClang ordinary and HomebrewClang ordinary/strict sanitizer. Every
configuration has fresh provider objects and a fresh installed archive; compiler,
SDK, libffi/crypto, source and tool identities and all intermediate binaries are
retained. Sanitizer scope is the actual compiled command closure, not a claim
about uninstrumented external libraries. The old host-grant, File opcode/service
refusal and ordinary wrapper/ISA neighbors remain required adjacent gates.
Source/provider before/after maps distinguish the fresh install rebuild from
previously prepared fixture providers. All advertised seal reports must be
committed Git blobs. I preserve the first terminal and require a reviewed
correction before any demonstrated failure is retried.

No C build or fixture has run at this checkpoint. Python AST parsing and diff
whitespace inspection are static preparation only. The initially reported
missing-C11 review finding was retracted: the inherited explicit grant rule is
present, and a standalone Make dry run confirmed that its recipe-bearing source
prerequisite remains `$<`. No production correction or build-defect claim follows
from that disproven suspicion.
