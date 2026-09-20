# I make cyclic File execution an explicit public choice

I design the next dependency of
`task_15a92c930af7433e9e25b41c7c5c761f` from PR907 head
`92b9fd67705bb058369870975d95990bfc66f86d`. Its qualified private
[dispatch](NANOISA_FILE_CYCLIC_DISPATCH_PLAN.md) uses the checked cyclic carrier,
physical witnesses and shared fuel; the public acyclic APIs are already canonical
from PR895. This document proposes their explicit conjunction. It changes no
production source, installs no new API and authorizes no execution. Actual PR907
merge, design review, complete source review and fixture review precede gates.

I preserve the original cyclic execution, closed-indirect, richer-borrow and
paired-source goals. This milestone exposes only the already qualified private
cyclic profile through a live grant. My existing acyclic API remains useful and
unchanged; failing it never triggers a cyclic fallback.

## My exact proposed API and native C ABI

I add installed `nanoisa/file_cyclic_public.h`, including `file_public.h` and a
small `file_cyclic_report.h`. The latter factors these existing private value
types without changing their names, field order, field types or revision:

```c
typedef struct {
    uint32_t revision;
    uint64_t instruction_limit;
} NvmFileCyclicOptions;
typedef struct {
    uint32_t revision;
    NvmFileRuntimeReport runtime;
    uint64_t instruction_limit, instructions_started;
    bool fuel_exhausted;
} NvmFileCyclicExecutionReport;
```

I retain `NVM_FILE_CYCLIC_RUNTIME_REVISION == 1` and
`NVM_FILE_CYCLIC_FUEL_MAX == 1000000`. The historical private default constant
is not a public default: no public entry substitutes a limit for NULL, zero or
an omitted command flag. NULL options is INVALID; revision1 and any limit in
0..1000000 are valid. Zero means zero admitted opcode starts, not unlimited.
The installed value header remains C99/C11/C++ compatible and contains no atomic
state or new opaque owning context. All data retains valid, immutable-during-call
and disjoint C storage preconditions. These structs are a native C ABI, not a
serialized wire encoding; I do not compare their padding.

The public header declares, inside `extern "C"` for C++:

```c
NvmFileCyclicExecutionReport nvm_file_execute_cyclic_bytes(
    NvmFileHostGrant *grant, const uint8_t *bytes, size_t size,
    const NvmFileCyclicOptions *options, NvmFileScalar *out);
NvmFileRuntimeStatus nvm2c_emit_file_cyclic_bytes(
    const uint8_t *bytes, size_t size, const char *entry_identifier,
    char **out, char *diagnostic, size_t diagnostic_size);
```

The emitter remains nonexecuting and has no fuel argument. Generated C exports
exactly `nvm_file_cyclic_program_<identifier>` with this signature:

```c
NvmFileCyclicExecutionReport nvm_file_cyclic_program_example(
    NvmFileHostGrant *grant, const NvmFileCyclicOptions *options,
    NvmFileScalar *out);
```

Each invocation supplies its own explicit limit. Emission does not create a
grant, bake in an execution budget or authorize a future host invocation. The
identifier retains the current1..63 ASCII letter/digit/underscore rule, beginning
with a letter. The distinct prefix permits acyclic and cyclic programs with the
same identifier to coexist without changing either signature. Duplicate cyclic
identifiers fail the actual link; there is no weak-symbol override.

I freeze a distinct `NVM_FILE_CYCLIC_PUBLIC_ABI == 1` for this wrapper contract.
An installed native-detail helper
`bool nvm_file_runtime_cyclic_public_abi(uint32_t, size_t, size_t, size_t)` checks
that revision and the exact sizes of options, cyclic report and public scalar.
Generated code also retains the existing semantic cyclic-native ABI guard and
its exact carrier/frame/value sizes before preparation/begin. Same-sized but
semantically incompatible changes require a revision change; size equality is
not semantic agreement. Old host/catalog/native ABI1 functions and the existing
acyclic RuntimeReport remain unchanged. Missing new symbols fail linking; a
mismatched runtime guard returns UNRESOLVED before File-core acquisition.

## I select one complete authority before effects

The new explicit API accepts the qualified cyclic profile, including an acyclic
member of that profile. Existing `nvm_file_execute_bytes`,
`nvm2c_emit_file_bytes` and `nvm_file_program_<identifier>` remain acyclic-only.
Generic VM/core/module/FFI and ordinary nvm2c entry guards continue refusing File
service claims. No successful nominal descriptor, query-only report, grant or
opcode tag independently authorizes execution. Query accessors retain
`runtime_admitted=false`; I do not flip descriptive evidence into a certificate.

Every invocation prepares a fresh owned cyclic hosted plan from the exact wire
bytes, validates the complete profile and uses the already qualified adapter
coverage conjunction before `begin`. I retain all original identities: global
nominals, original import mappings, functions, signatures, locals, initializer,
entry, byte offsets, decoded operands and callee order. All functions and decoded
instructions are checked, including unused helpers and unreachable opcodes.
All reachable variants, input/output stacks, owner/reference/region relations,
edge masks and exact destination variant ordinals must agree. The entry seed
ordinal stays0. Empty/unreachable native labels remain explicit state refusals.

Native execution additionally compares every defined field of its emitted facts
against this fresh plan, including non-first alternatives and both branch edges,
and compares embedded wire spans. No caller supplies a trusted report, altered
layout view or inferred ordinary value. I neither scan alternatives until one
matches nor normalize away absent or unsupported authority. Any disagreement
refuses before acquisition; allocation/query/coverage failure has no fallback.

The VM wrapper consumes the same fresh plan/coverage path as the qualified VM.
Native generation uses real C functions, labels, direct calls and checked frame
operations. It cannot invoke the VM dispatcher or embed a bytecode interpreter.
The distinct indirect-composition query remains descriptive and unselected:
FUNCTION targets, indirect calls, recursive calls, richer borrow profiles and
unqualified opcodes remain refused here. Current finite function/stack/local/
variant/storage limits, direct-call graph and scalar startup/result restrictions
remain exactly those of the private hosted/dispatch conjunction.

## I preserve grant lifetime and error precedence

Both execution wrappers first call the existing owning
`nvm_file_host_enter(grant, NVM_FILE_HOST_ABI, NVM_FILE_HOST_CATALOG)`.
Only its successful caller owns the gate and may leave it, once, after all
staging/context cleanup. The same archive owns the grant identity and single
atomic gate used by acyclic and cyclic entries. I add no second gate or implicit
grant. Revocation/destruction during an invocation return BUSY; a nested refusal
cannot drain its caller, end a borrow, consume an owner or release its gate.
The grant must remain allocated for the complete call; a stale freed C pointer
is not a supported input or a sandbox boundary.

BUSY precedes argument inspection. Its report has revision1, statusBUSY,
acquired=false, sitesNO_INDEX, empty cleanup, zero instructions and
instruction_limit=0; I cannot read the supplied limit while promising this
precedence. It preserves scalar output and diagnostics. Other grant refusals
map through the existing explicit INVALID/MEMORY/STATE/UNRESOLVED mapping, with
no File acquisition. With otherwise valid caller storage their cyclic report
retains the supplied limit when options is non-NULL, or0 when NULL. Invalid
options likewise preserve the supplied numeric limit while reporting INVALID;
an invalid revision never changes the returned report's own revision1.

After successfully entering the gate I validate/copy options once, validate
outputs and exact byte bounds, then prepare the engine. Options/input/output
storage may not alias or mutate during the call. Successful gate entry is not
File-core acquisition: `runtime.acquired` keeps its existing core meaning.
All refused pre-begin reports have no invented instruction site, no fuel used
and no fuel-exhausted claim. Invalid arguments and preparation failures preserve
the entire output scalar. Existing ambiguous query allocation statuses retain
their qualified classification; I do not promise MEMORY for every legacy
conflated failure.

Emission uses `nvm_file_host_enter_query`, not a host grant. BUSY returns before
reading caller arguments or modifying diagnostics/output. Once acquired it
validates identifier/spans, performs the complete query/coverage operation and
builds bounded C text. All failure paths free staged plans/buffers before
leaving the gate and preserve `*out`; diagnostics may change only within the
provided valid span. Success publishes malloc-owned text exactly once. No host
stream, File owner or runtime invocation is created by emission.

## My fuel, cleanup and publication semantics are deterministic

I retain the existing invocation-wide unsigned limit. Initializer, entry and all
direct callees share one counter; CALL, RET, branch and each other actual opcode
are charged once by `cyclic_enter` before their effects. Frame switches, repeated
allocation sites and generation recycling never reset fuel. If the next start
would exceed the limit, I return LIMIT at that exact pending function/PC with
`instructions_started == instruction_limit` and `fuel_exhausted=true`. A terminal
RET that consumes the last permitted unit may complete successfully; I do not
charge a phantom next instruction. Zero may prepare/begin and clean up a core,
but starts no opcode and performs no bytecode File service effect.

Limits from preparation, capacity, generation retirement or staging remain LIMIT
with `fuel_exhausted=false`. Assertion, type/state, borrow, stale identity, host
and cleanup failures retain the private engine's first-error precedence and
complete cleanup fields. Cleanup is not fuel-charged and cannot be skipped by
exhaustion. A failed invocation still drains owners, references, regions and
staging, then finishes/disposes its own core before leaving the gate. Root
counts are checked before disposal in qualification, not inferred from a final
sweep. No failed path publishes a scalar or leaves a transferable File value.

I stage the passive result, run the existing exact public scalar validation after
terminal destruction, then copy INT or canonical BOOL to `NvmFileScalar` only on
clean success. A malformed successful view becomes TYPE (or the existing
acquisition/cleanup status) while retaining the original cleanup report and
leaving output unchanged. The cyclic wrapper retains all outer fuel fields;
it does not reconstruct or discard the report when validating a scalar.

Fuel bounds admitted opcode starts; it is not a deadline for query preparation,
a blocking host call, native C or cleanup. External harness deadlines remain
necessary. These ordinary trusted-C/runtime policies do not establish process
isolation or a security sandbox.

## I factor the qualified engines once

The production checkpoint first extracts the qualified VM body into one static
serialized cyclic engine include. The existing macro-only private entry becomes
a thin unchanged-signature wrapper; the new public translation unit calls that
same engine after grant entry. The default library exports only its granted
entry, not the ungranted private qualification symbol. I review the extraction
against production4a12 line-for-line and rerun the unchanged private corpus.

I add a checked surface parameter to cyclic native emission. Private mode keeps
its qualified include path/export. Public mode emits the installed native-detail
header, static engine and granted wrapper directly; no generated text rewriting
selects authority. Opcode operators, labels, actual call sequence, full agreement,
preallocated witnesses and failure cleanup remain common. Public factoring may
not silently broaden query modes or change the private semantic ABI revision.

## I extend one installed runtime package

I extend the existing `libnano_file_runtime.a`; I do not ship another owner of
host identity/gate or link a private fixture object. Public cyclic VM and emitter
occupy separate archive members, so generated native callers do not extract the
VM engine, generic interpreter, loader, compiler or FFI/COP dispatch. The carrier,
core, query and grant providers already have their qualified roles. Any newly
required member is enumerated from actual symbol dependencies, not supplied by
a full compiler-object bundle or stub.

The explicit installed manifest gains `file_cyclic_public.h`,
`file_cyclic_report.h`, a `file_cyclic_native_public.h` detail header and the exact
transitive cyclic fact/carrier headers that generated C requires. The source
checkpoint enumerates these from real includes, retaining the existing
`include/nanolang/file/nanoisa/` and sibling structure. It installs no private
VM/emitter entry header, source `.c`/`.inc`, test macro, fixture or source-path
fallback. A native-detail protocol header remains trusted generated-C machinery,
not a public unchecked execution certificate; its old source-private wording
must be updated accurately when it enters this installed closure.

The grant provider keeps its explicit owning C11 recipe/source path. Generated
units and public host clients retain C99/C11/C++ header compatibility; actual
system library requirements stay explicit. Make dependencies include every
new header/engine include so changing them rebuilds the correct archive member
and commands. Install/uninstall use only the explicit owned manifest and preserve
unrelated prefix files. Qualification inventories archive/indexer/compiler/SDK,
installed headers/library/commands and actual interpreter bytes before/after.
The historical PR907 Linux interpreter-hash omission is not repeated.

## I require explicit CLI choices

I propose these exact additional forms:

```text
nano_vm --allow-temporary-files --file-cyclic --file-instruction-limit N input.nvm
nvm2c --file-temporary --file-cyclic --entry-name IDENT input.nvm [-o out.c]
```

For nano_vm, `--file-cyclic` and `--file-instruction-limit` are required together
and require the existing grant opt-in. A limit alone, cyclic flag without limit,
duplicate new flag, negative/signed/blank/junk number, overflow or value above
1000000 is a usage error before grant creation or service execution. The parser
accepts only nonempty ASCII decimal digits;0 is explicit and valid. There is no
environment default, inferred budget, retry or implicit fallback. Without these
new flags the existing `--allow-temporary-files` route remains acyclic-only.
Existing incompatible daemon/shadow/verify/repeat/profile/debug/COP/guest modes
remain refused on the new route before grants. Ordinary invocations are unchanged.

nvm2c requires the existing emission opt-in and validated entry name with the
new cyclic flag. It does not accept a runtime instruction-limit flag: fuel is
supplied to each generated host invocation. Omitted cyclic selection uses the
unchanged acyclic emitter. Malformed combinations fail before publication.
The current bounded byte reader and atomic file writer retain open/seek/read/
close/output error behavior and the16MiB input bound; CLI input/output I/O is
counted separately from bytecode File service acquisition.

The VM CLI creates/destroys one explicit grant, reports the exact cyclic status,
site, limit, started count, exhaustion flag and cleanup failure count on failure,
and returns1. Clean INT returns its low8 bits; clean BOOL returns0/1, matching the
old scalar CLI convention. Full signed64-bit equality is an API assertion, not
an exit-status claim. No output artifact or scalar is published on refusal.

## My ordered acceptance before public activation

1. I review this contract and the source/interface checklist, then implement only
   the exact report/header factoring, common engine extraction, public wrappers,
   installed package and explicit CLI paths. Complete source and fixture reviews
   precede new public cyclic execution.
2. I retain the full qualified private dispatch corpus and original acyclic
   API/CLI/installed controls. New public API, generated native O0/O2 and installed
   VM CLI agree on original modules, catalog permutations, initializer/helper
   charges, loops beyond257 reuses, held-reference exhaustion, both arms, host
   denial, assertion, staged failure and cleanup outcomes. C API comparisons
   include every defined report/scalar field and ordered host effects.
3. I cover exact N/N-1/0/MAX options, invalid/null/overflow options, zero-fuel
   unsupported input precedence, early report fields and non-fuel LIMIT causes.
   BUSY/revoked/wrong grant, nested mixed acyclic/cyclic calls, repeated calls
   and grant destruction preserve first reports, caller roots and output sentinels.
4. I exercise all startup/call/unused-helper/edge variants and forced dead-label
   refusals. ABI revision/size and non-first variant/reference/edge discrepancies
   refuse before acquisition. Malformed/service identity/linked/indirect/richer
   borrow/unsupported metadata keeps its existing refusal with no second route.
5. I qualify measured allocation-prefix/transient faults with independent recovery,
   emission byte limits, no per-instruction project allocations, zero roots before
   disposal, and exact cleanup after limit/assertion/host failures. Instrumentation
   scope and unmeasured peak-memory claims remain explicit.
6. I install into a fresh unrelated PREFIX and compile/run from an unrelated cwd
   using only installed headers/archive/system tools. Two cyclic programs plus
   an acyclic program share one gate/grant; native symbols prove no VM extraction.
   Duplicate exports and missing/wrong ABI runtime fail as specified. Strict C99,
   C11 and C++ public-header forms, O0/O2, ordinary and supported ASan/UBSan/LSan
   routes run on Linux/Darwin. Uninstall preserves an unrelated sentinel.
7. I test every new CLI flag combination and every publication/error path, then
   seal actual source/compiler/interpreter/provider/archive identities and first
   terminals. Independent review and actual canonical merge close only this
   bounded public milestone. Paired File source/shadows, indirect execution,
   richer references and full15a92/72556/6931 remain open and required.
