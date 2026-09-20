# My first portable host linkage checkpoint

I scope task2d2 after merged managed-string acceptance896/898 at462c98e2d.
The parent requires exact host signatures, result ownership and real native LLVM
and Wasm adapters. Filesystem/compiler operations remain applicable requirements.
This design neither reuses an arbitrary C symbol as authority nor duplicates
my public temporary-File service, cyclic File lane or peer union implementation.
No production or qualification is authorized by this proposal alone.

## My concrete existing operation and gap

The first operation is the existing ordinary `file_read` family: empty-namespace
FFI imports `file_read`, `vm_file_read`, `nl_os_file_read`, exactly one STRING
parameter and one STRING result. `src/nanoisa/nvm2c.c` already maps these exact
names to nhost_file_read. `src/nanovm/vm_builtins.c:vm_file_read` delegates to
`src/runtime/file_text.h` / `file_bytes.h`: read and close a stream, discard
partial bytes on read/close error, return empty text for missing files or file
content containing NUL. Allocation failure is distinct from an empty result.
The argument's C-string prefix before the first NUL is the current path meaning.
I must preserve that behavior rather than silently choose byte-file semantics.

`nvm_verify_profile` currently refuses imports in the closed LLVM profiles;
`nvm2llvm.c` emits no reviewed CALL_EXTERN host contract. `scripts/nvm2wasm.py`
links without arbitrary unresolved symbols. `NvmV2Import` already retains module,
symbol, kind and signature index; the common signature table carries exact tags.
These bytes provide declaration identity, not permission to call host functions.
I require an explicit supplied host context in addition to exact catalog match.
No ownership-envelope version is needed for this STRING-only checkpoint.

## My first production proposal: non-admitting binding query

I propose `portable_host_plan.h/.c`, a private owned plan for one immutable
module. It validates the existing complete module/import/signature envelope,
then resolves only the three exact empty-namespace FFI spellings above to one
READ_TEXT revision1 catalog entry. It copies original import indices, namespace/
symbol bytes and exact parameter/result tags; it never uses dlsym or path names
as trusted artifact contracts. Embedded NUL in import identifiers, wrong kind,
namespace, signature, unknown aliases or unavailable typed parameter facts refuse.
A missing explicit executable entry, invalid unused function or call operand
still refuses through normal structure checks. Presence of pending File service
operations/metadata refuses before host selection. CALL_MODULE, callbacks,
artifact imports, reference/passive and owned nominal routes remain unsupported
in this first plan. A checked read declaration does not discharge their work.

Proposed API is `nvm_portable_read_plan(const NvmModule *, NvmPortableReadPlan **)`
returning a result with status PREPARED/NOT_SELECTED/INVALID/UNSUPPORTED/LIMIT/
MEMORY, original function/pc/import or NO_INDEX, and static diagnostic. Only
PREPARED publishes. Report getters copy original import and catalog facts; all
failure outputs stay unchanged. Free(NULL) is valid. Plan storage owns all facts,
remains valid after input destruction and exposes no executable-trust boolean.
Independent limits are64 imports,256 functions,65536 decoded instructions,
16MiB module bytes and16MiB total retained/transient query allocations, with
checked count arithmetic and no recursion through public profile selection.
No shared profile or translator accepts the plan until later reviewed composition.

## My concrete adapter ABI proposal

I separate host text bytes from NmsHandle, VmString and legacy char-pointer
ownership. The host owns no managed handle. Proposed native revision1 callback:

```c
typedef int32_t (*NvmReadTextHostV1)(
    void *context, const uint8_t *path, uint32_t path_length,
    uint8_t *destination, uint32_t capacity, uint32_t *length_out);
```

Status values are OK0, DENIED1, LIMIT2, MEMORY3, INVALID4. A null context or missing
callback is unavailable, never ambient permission. The context contains a fixed
application-supplied allowlist of exact path bytes and explicitly selected real
reader; it is immutable for an invocation, serial and not thread-safe. This is
an authority boundary, not a filesystem sandbox or a race-free file identity
claim. No path normalization, symlink containment or process-global permission
is inferred. The callback borrows input and destination synchronously and may
not retain them or reenter the same instance. It preserves length_out on error;
on success it writes exactly the returned number of initialized bytes, no NUL
terminator required. Failed destination bytes are scratch, never language-visible.

The wrapper computes the effective path prefix ending before the first NUL;
the exact allowlist comparison and both host adapters use that same prefix.
The native production adapter checks the allowlist before opening a file, uses
actual bounded stream reads, checks read and close status and closes exactly once.
Missing/open/read/close failure yields successful empty text to match existing
file_read; permission refusal, bounds and allocation failure remain terminal host
statuses, not fabricated empty strings. Embedded-NUL content also becomes empty.
A bounded diagnostic records the host failure category separately if required by
the surrounding checked interface; it does not change the source return type.
The fixed first implementation capacity is1MiB, path bytes4096; an input exceeding
these limits refuses explicitly. Reading one excess byte detects a growing file
instead of accepting a truncated prefix. No writes, compiler launch or deletion
is permitted by this READ_TEXT context.

For wasm32 I propose the exact import `nanolang_host_v1.read_text` with core type
`(i32 path_offset, i32 path_length, i32 destination_offset, i32 capacity,
i32 length_offset) -> i32`. The instance-bound host object supplies its own
allowlist/context; there is no native pointer or global integer grant in Wasm.
The host validates all unsigned offset/length ranges against the current memory
using subtraction-safe bounds, the four-byte little-endian output length and
nonoverlap with input/output regions before host I/O. It reacquires the current
memory view for the call, does not retain views and cannot grow/reenter guest
memory while writing. Both Node and Wasmtime adapters must use real file reads
with byte-path behavior, exact NUL/text rules and actual close handling. Test
stubs alone do not qualify this adapter. A browser without this declared host
capability must refuse instantiation/execution; that is not a blanket claim that
portable file/compiler features are inapplicable to Wasm.

The compiler-owned wrapper borrows the rooted argument through the host call,
keeps caller/result/global roots live, and uses a separately allocated bounded
scratch destination plus disjoint length cell. Scratch allocation occurs before
host effects. It validates returned status/length and calls nms_create to make
one instance-owned string root; it always releases scratch. It publishes only
that new managed root after successful creation. Allocation failure after the
read discards scratch and unwinds with the existing managed failure protocol;
other aliases and previously committed globals survive. Generated CALL_EXTERN
consumes its argument root exactly once on success or failure after the wrapper
borrow ends. No borrowed C snapshot escapes; returned aliases cannot outlive a
host buffer because the managed result is copied. Empty success is still the
runtime's valid owned empty-string representation. No NmsRuntime layout crosses
the native/Wasm external ABI.

## My ordered implementation and acceptance boundaries

1. Review exact binding query/API/status/limits, then implement only the private
   query. Qualify malformed and mismatched signatures, original import identity,
   refused extra imports, output atomicity and
   allocation prefixes. No file is opened by query preparation.
2. Review the native/Wasm adapter and managed wrapper source before tests.
   Qualify absent/denied host capabilities and explicit host invocation with real files in isolated temporary trees
   on Linux/Darwin, and Node plus Wasmtime. Check empty/multibyte/NUL content,
   missing paths, exact capacity/oversize, aliases, repeated reads, deny-before-
   open counters, partial/error close handling, every allocation boundary and
   no live handle/byte drift after failure and disposal. Modeled I/O errors are
   labeled separately from real successful filesystem effects.
3. Review an explicit opt-in LLVM host profile and emitted callback context,
   real native linked adapter and exact allowlisted Wasm import packaging. The
   old closed profile remains unchanged. No `--allow-undefined` blanket escape:
   inspect actual module imports against the exact reviewed ABI. Complete fresh
   preparation precedes each execution/emission; failure never falls back to a
   looser profile. Emission and CLI output publication remain atomic.
4. Qualify actual ordinary source producers' selected extern declarations and
   full helper/main shadows, VM/native LLVM/Wasm result equivalence, before/after
   optimization, sanitizer scopes, default refusal and installed real host links.
   No adapter-only check closes this source or packaging obligation.

I retain the full2d2 continuation: filesystem write/metadata/directory and byte
array results; process/environment/arguments; compiler_support's actual
`nlc_module_artifact` build-and-snapshot lifetime; exact artifact ABI and foreign
release contracts; linked NanoISA module graphs with original CALL_MODULE
identity and shared instance cleanup; aggregate/recursive/callback values and
full compiler capabilities. I do not flatten dependency modules as an implicit
link solution. Required shared array/record authority15f/488 and peer union work
remain dependencies for aggregate-returning adapters. Full compiler and
NanoISA-only bootstrap/fixed-point gates are unchanged and unfulfilled by this
single read-text capability. Parent2d2 remains open throughout this checkpoint.

## My query implementation boundary

I report exact declarations and the common structural/stack envelope, not a
complete operand type, managed lifetime or target execution proof. My shared
`nvm_verify` type analysis is advisory: it can skip deep stacks or allocation
failures. I infer no type fact from its success. Reported shared allocation
failures remain INVALID; only my own directly classified allocation reports
MEMORY. There is no host context in this query and no capability-denial claim.

I predecode without allocation and conservatively charge the common decoder's
geometric instruction allocations, byte boundary/index arrays, stack worklists
and at most256 type-state slots per instruction before invoking the verifier.
The16MiB query budget counts the sum of these conservative per-function bounds
plus my retained report; it therefore also bounds simultaneous live requests.
Allocator bookkeeping and the caller-owned input are outside that allocation
budget. The separate16MiB module budget counts represented table/payload bytes,
not unused caller capacities or an original serialized container's padding.

I reject retained layout/ownership/passive declarations and aggregate type
counts in this first query, as well as linked modules, callbacks, closures,
indirect/reference calls and reference/owned-transfer instructions. Ordinary
scalar and STRING function signatures require explicit parameter tags; VOID
is permitted only for zero-result functions. These are private description
limits, not changes to existing public admission. Header and advisory metadata
are checked in the in-memory module; original wire offsets/checksum are not
reinterpreted as a fresh serialized image. Existing decoder, stack and operand
checks cover every function, including unreachable helpers. My API header is
the exact status/getter/output contract for source review before fixtures.

My first source checkpoint adds only `src/nanoisa/portable_host_plan.h/.c`.
No default provider list, shared verifier, profile, CLI or runtime changes.
My report uses one fixed allocation for at most64 copied import rows. Counts
and row getters copy complete values and preserve outputs on invalid indices.
The row distinguishes catalog ownership obligations (borrow a rooted argument;
copy an owned managed result) from behavior a future adapter must establish.
My call graph is allocation-free envelope/predecode/budget checks, then existing
`nvm_verify` → ordinary per-function decoder/stack/advisory types, then report
allocation. Preflight excludes every metadata condition selecting owner/mixed
delegation; no public entry calls this private query. All query output remains
unchanged on reported failure. The only retained pointers are static diagnostics;
the plan borrows no input storage. No fixtures, build or host operation has run
for this source checkpoint.

## My first query fixture checkpoint (not executed)

I prepare `tests/nanoisa/test_portable_host_plan.c` and its retained Python gate.
A normal linked query and an allocator-observed query use the same malformed,
limit, copied-fact and closed-profile controls. The explicit Make target remains
outside default providers and test-units. No module body executes. My positive
fixture has all three exact aliases and an unused STRING identity helper; rows
are still checked after destroying the entire module. Invalid getter calls and
all query refusals preserve sentinels. Closed scalar/literal/managed profiles
still reject the import-bearing module before and independently of any plan.

I cover missing entry, unknown/extra import, kind/namespace/NUL/name/signature
mismatches, missing parameter facts, malformed unused body/local/call operands,
owned/passive/layout/linked/callback envelopes, File metadata and bare File
instructions in either entry or unused helper. Distinct limit controls stop at
function/import/string counts, represented CODE/string bytes, decoded instruction
count and conservative allocation budget before the first query allocation.
A no-import complete scalar module reports NOT_SELECTED.

My allocation observer wraps all four allocation functions in exactly four
translation units: portable_host_plan.c, vm_decode.c, verifier.c (including its
internal implementation includes) and verifier_types.c. Other prepared linked
providers remain ordinary objects. Query preflight excludes their allocating
nominal paths. I distinguish QUERY/DECODE/STACK/TYPES sites, preserve realloc's
old allocation on failure, and require zero tracked objects/bytes after every
attempt. Every persistent prefix and single transient failure through a fresh
successful query's measured request count is inspected; later successful
recovery and exact input header/table/CODE/string/parameter bytes are required.
I print every actual index/domain/status. INVALID requires observed decoder or
stack allocation failure, MEMORY requires report allocation failure, and a
successful injected query requires only advisory type-allocation failures. I
retain those successes as an explicit limitation, never as typed-operand proof.

The driver builds linked and observed variants separately with strict warnings;
selected sanitizer flags instrument those four translation units plus the
fixture, not all prepared providers. It retains command JSON, file-backed output,
status and each newly built object/binary before assertions. Commands have a
120-second bound followed by bounded five-second TERM and KILL waits; unresolved
cleanup cannot pass. A content-addressed archive retains exact participating
C/header/include source files, selected compiler/Python executables and named
provider objects, with fresh before/after phase hashes. This endpoint comparison
is not a claim about intermediate immutability or every transitive system tool.
The external qualification still prepares frozen providers, records actual host,
compiler/runtime identities and seals first terminals. Linux/Darwin ordinary
and explicit sanitizer configurations await review; no gate has run here.

My first external Linux setup stopped before building providers or fixtures:
Clang23's `-print-file-name=libclang_rt.asan-aarch64.so` returned the unresolved
basename. I retain `/tmp/nanolang-portable-read-e9-linux` and the exact first
external driver. This installation uses its target-specific runtime directory
`/usr/local/lib/clang/23/lib/aarch64-unknown-linux-gnu`, with unqualified
`libclang_rt.asan.so` and static ASan/preinit/UBSan archives. My corrected driver
queries and requires those actual paths, preserving strict existence checks;
there is no source/fixture change or sanitizer suppression. Darwin keeps its
explicit Homebrew `libclang_rt.asan_osx_dynamic.dylib` selection. This setup
failure establishes no query result; corrected gates use fresh evidence roots.
