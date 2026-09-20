# My copied read-text host boundary

I continue task_b7ef4d216a4948c58f7a710feacc4251 under parent2d2 from
canonical `f42201461`, the actual merge of my private declaration query899.
This is the next preimplementation checkpoint. I do not turn that query's
PREPARED result into operand-type, lifetime or execution authority. I propose
private adapter and managed-wrapper source first, source review before fixtures,
and frozen fixture review before any host effects. Public LLVM/Wasm selection,
source lowering and installed host linkage remain separate later checkpoints.

## My existing lifetime boundary

`managed_strings.h` exposes instance-local NmsHandle, nms_view, nms_create,
nms_retain/release and nms_begin/finish/dispose. A view borrows its handle's
lifetime. nms_create copies bytes and publishes one owned root only on success;
its allocation can fail after bytes are allocated but before slot publication.
I use those APIs without changing the runtime layout or handle ABI. I neither
export managed pointers nor ask a host to manufacture a handle.

The real legacy text reader in `runtime/file_text.h` consumes a FILE through
file_bytes.h, discards failed partial reads and content containing NUL, and
returns empty text for ordinary open/read/close errors. Allocation failure is
separate. I preserve those observable rules and the path prefix before NUL.
My explicit capability/size failures are terminal outcomes of the new opt-in
host boundary. They do not change the old ambient file_read implementation.

## My native revision1 types and ownership

I propose new private `portable_read_host.h/.c` and
`portable_read_managed.h/.c`, outside default provider lists:

```c
typedef enum {
    NPR_OK=0, NPR_DENIED=1, NPR_LIMIT=2, NPR_MEMORY=3, NPR_INVALID=4
} NprStatus;
typedef struct { const uint8_t *data; uint32_t length; } NprPath;
typedef struct NprFileHost NprFileHost;
typedef int32_t (*NprReadCallback)(void *context,
    const uint8_t *path, uint32_t path_length,
    uint8_t *destination, uint32_t capacity, uint32_t *length_out);
typedef struct {
    NprReadCallback read;
    void *context;
} NprHostBinding;
typedef struct {
    NprStatus host_status;
    NmsStatus managed_status;
    NmsHandle value;
} NprManagedResult;
NprStatus npr_file_host_create(const NprPath *, uint32_t, NprFileHost **);
NprStatus npr_file_host_destroy(NprFileHost *);
int32_t npr_file_read(void *, const uint8_t *, uint32_t,
                      uint8_t *, uint32_t, uint32_t *);
NprManagedResult npr_read_managed(NmsRuntime *, NmsHandle,
                                 const NprHostBinding *);
```

Status values and argument order match the preceding linkage contract. Creation
copies at most64 exact, nonempty, NUL-free paths of at most4096 bytes. It uses
one fixed bounded allocation for its immutable allowlist plus an active flag;
checked size is at most sizeof(context), including all fixed rows. Empty
allowlists are valid deny-all contexts. Duplicate paths are harmless and need
no extra permission interpretation. A nonzero count requires a valid rows pointer;
all rows are validated before allocation. Failed creation leaves output unchanged.
Destroy(NULL) succeeds; destroy while active refuses without freeing. Callers
serialize access and keep the context alive through calls. This is not a thread-
safety, hostile-native-pointer or filesystem-sandbox guarantee.

The callback borrows all storage synchronously. It cannot retain pointers,
reenter the binding or mutate the runtime through another route. Null callback
or context is DENIED. Unknown callback status becomes INVALID. A callback that
returns OK must have initialized exactly length_out bytes within capacity and
must not have retained scratch. This is a trusted embedding contract: a length
check cannot prove initialization or contain an arbitrary malicious C callback.
Failed callbacks may dirty scratch but cannot publish it as a language value.
The wrapper checks returned status and length, and rejects returned embedded NUL
as INVALID; the supplied real reader itself converts embedded-NUL content to
successful empty text before returning. Unrecognized status never means empty.

The concrete real adapter takes only its own context. Before opening anything,
it validates path/capacity/output pointers, path length and exact allowlist match;
validates uintptr_t ranges with checked addition; and rejects overlap among
input bytes, destination bytes and the four-byte output cell, and overlap of
any of those three ranges with the complete known opaque context allocation.
I check its sizeof(context) range before writing the active flag, preserving the
copied allowlist against aliased buffers. Zero-length ranges
are empty. C pointer validity remains a caller precondition. This overlap check
is defined integer arithmetic, not ordering unrelated C pointers. The output
length cell changes only on success. Output bytes remain unpublished scratch.
Only the effective prefix supplied by the wrapper is authorized; no normalization,
symlink containment or durable inode identity is inferred from exact path bytes.

The adapter copies a validated path to bounded NUL-terminated local storage,
opens in binary-read mode, and reads at most capacity plus one probe byte.
Capacity is at most1048576. A byte beyond capacity produces LIMIT, never a
truncated successful result. It checks read error and fclose separately, closes
exactly once and never retries close. A failed read/close returns OK with zero
length only when no terminal host error has already been selected. In particular,
a detected excess byte selects LIMIT immediately; later fclose failure cannot
replace LIMIT with empty success. A read error before any excess byte instead
selects legacy empty success, discarding partial bytes. Cleanup closes once in
each case. The first non-success host status remains first if cleanup also fails.
No project heap allocation is needed inside the real callback. libc/OS buffering
is outside that claim. No writes, process launch or deletion are part of this API.

## My managed call and cleanup

The wrapper requires a non-disposed active NmsRuntime and an existing STRING
handle in that runtime. Invalid runtime/handle returns managed failure, zero
value and no callback. The caller owns the argument throughout; the wrapper
borrows it and does not consume it on any path. The generated CALL_EXTERN layer,
when later admitted, will separately consume its argument after the borrow ends.
No owner movement is hidden inside the adapter API.

After nms_view, I find the first NUL within the bounded view using at most4097
examined bytes. A prefix longer than4096 returns LIMIT. A short prefix permits
ignored suffix bytes as the existing C-string path semantics require. Empty
prefix is denied by the concrete nonempty allowlist. The wrapper allocates one
fixed scratch block containing a copied path plus terminator, a disjoint1MiB
result buffer and an aligned length cell. Checked size includes alignment.
All scratch allocation and path copy precede host effects. Holding the original
argument root keeps its view alive; copying also isolates the callback's path
from subsequent managed slot-array relocation. A local copied binding is used
throughout this synchronous call.

I initialize length to an invalid sentinel, call exactly once, validate known
status and length<=capacity, then copy via nms_create. I free scratch on every
path. Only host_status==NPR_OK and managed_status==NMS_OK returns a nonzero owned
value. Host failure returns managed_status==NMS_OK with zero value; managed
validation/allocation failure returns host_status==NPR_OK, its exact NmsStatus
and zero value. NPR_OK in that pair does not assert that a callback ran. Every
return initializes all three fields. The wrapper explicitly checks disposed and
active before nms_view; nms_view alone does not enforce invocation state.
Scratch allocation reports NPR_MEMORY. A failed nms_create reports NMS_MEMORY,
not an empty string. A successful empty result is still a valid owned STRING.
The result struct returns by value, avoiding aliasing an output handle slot with
caller storage. Existing globals, aliases and argument counts remain unchanged.
The caller releases a successful result once. The wrapper does not call begin,
finish, disposal or collection, nor drain caller roots after a rejected call.

## My exact wasm32 host boundary

The private Wasm wrapper uses the same checked managed logic and scratch layout,
with a small adapter thunk to exactly one import:
`nanolang_host_v1.read_text : (i32,i32,i32,i32,i32)->i32`.
Its fields are path offset/length, destination offset/capacity, length offset.
No callback pointer or native context crosses Wasm; the host binding belongs to
one instantiated memory and one copied exact-path allowlist. The public translator
continues to reject this import until its separate reviewed opt-in profile.

I propose real Node and Wasmtime embedding adapters, not CLI WASI ambient access.
Node uses byte Buffer paths and synchronous open/read/close. The Wasmtime host
uses its supported embedding API and byte-path filesystem calls. Before source
implementation I pin the exact available Wasmtime binding/tool version and its
memory/read/write APIs; a generic CLI invocation alone cannot register this
custom import. Any private dependency setup is recorded and hash-attributed.
Both hosts interpret incoming i32 values as unsigned bit patterns, validate all
ranges against the current instance memory with subtraction-safe arithmetic,
validate the four-byte little-endian length cell and reject pairwise overlap
before I/O. Memory offsets are not passed to a native FILE API as pointers.

The host acquires the current memory view for each call. It cannot grow/reenter
memory or retain the view; the Wasmtime adapter copies only checked ranges via
its embedding API. Only a successful operation writes a valid length cell.
Both real adapters implement the same capacity probe, NUL-content, error and
close-once rules. Oversize, unknown status and bad output lengths remain terminal.
No blanket linker allow-undefined option is an acceptance substitute: the private
fixture inspects every actual import and requires only this exact function plus
explicitly documented memory exports. Node/Wasmtime use the same byte vectors.

## My staged source and acceptance checkpoints

1. I implement only the native opaque context/real callback and borrowed managed
   wrapper above, including exact allocation bounds and status types. Source
   review precedes fixtures and any read. No verifier, emitter or CLI changes.
2. After source review I prepare real Linux/Darwin direct C and direct LLVM-linked
   wrapper fixtures. The LLVM fixture calls the real typed ABI; it is not an
   emitted NanoISA program or public admission. I inspect exact declarations and
   native symbol links. A copied return survives callback scratch destruction;
   argument/global aliases survive success, denial and post-read nms_create failure.
3. I separately review the wasm32 thunk and real Node/Wasmtime host source and
   pinned APIs before Wasm fixtures execute. Direct C-to-Wasm wrapper compilation
   and real embedding reads establish adapter parity, not NanoISA translation.
4. Acceptance covers missing/denied context, deny-before-open counters, exact
   NUL-prefix and multibyte paths, missing/empty/NUL/multibyte files, capacity and
   excess-byte boundaries, overlapping ranges and unsigned overflow, callback
   bad status/length, repeated calls, real successful close and separately modeled
   read/close errors, every project allocation boundary, retained roots/bytes,
   exact first-error cleanup and later recovery. I separate normal providers and
   fully rebuilt instrumented wrapper/runtime TUs. All first terminals persist.
5. Fresh complete typed/lifetime authority, opt-in native LLVM/Wasm emission,
   failure-atomic outputs, default refusal, installed adapters and actual source
   helper/main shadows remain required subsequent work. The declaration query
   alone cannot supply this authority. Linked-module/compiler/array/recursive
   capabilities and all original2d2/bootstrap/fixed-point gates stay open.

## My first native source checkpoint

I add only four private files: portable_read_host.h/.c and
portable_read_managed.h/.c. There are no Make/default-provider, profile, emitter,
CLI, managed runtime or query changes. No new fixture, compile or host read has
run at this checkpoint. Wasm thunk and embedding source are still pending.

My real context's one calloc request is sizeof(NprFileHost), containing count,
active flag and64 inline NprStoredPath rows. Each row contains a uint32 length
and4097 bytes; there are no hidden per-path allocations. My wrapper's one malloc
request is sizeof(NprScratch):4097 path bytes,1048576 destination bytes, and an
aligned uint32 length cell including compiler padding. The real callback uses
4097 stack bytes for a terminated path and four small checked address ranges;
its project allocation count is zero. libc buffering and nms_create's existing
managed byte/slot/workspace allocations are separate domains, not omitted from
future scoped failure and lifetime controls. Query allocations do not occur in
a direct adapter call, and no query report is treated as a runtime permit.

I validate the complete fixed allowlist before its allocation and publish its
pointer after every copy. The callback validates four pairwise-disjoint regions
before changing its active flag. Ordinary fopen/read/close failure yields empty
only while host status remains OK; a probe byte sets LIMIT before fclose and
retains it. All admitted opens reach exactly one fclose. The managed wrapper
initializes all result fields, explicitly checks disposed/active, copies the
bounded effective path before callback, validates status/length/content and
frees scratch after nms_create regardless of its result. It never changes the
caller's argument ownership. Native callers remain responsible for valid C
storage and serialized context/runtime use.

## My native fixture checkpoint before execution

My new C fixture builds real files in a retained caller-supplied directory. It
checks empty, one-byte, multibyte and partial-UTF8 content, embedded NUL,
exact1MiB and1MiB+1, effective NUL-prefix paths, missing files and deny-all.
It rejects pairwise input/destination/length/context overlaps before observed
open attempts, checks actual closed descriptors against an unrelated live FILE,
and distinguishes real read/close calls from modeled progress-plus-error or
post-real-close errors. It explicitly retains LIMIT when modeled close fails.

My allocator observer rebuilds all three new/managed runtime C translation
units with SDK-safe malloc/calloc/free hooks. Existing binary64 parser helpers
are header-only. The real-reader TU alone receives fread/ferror/fopen/fclose
hooks; those normally forward to libc. Fixture setup allocations/files and
libc/system allocations are outside project tracking. The unhooked comparison
rebuilds all three TUs separately and retains the ordinary semantic controls.
No prebuilt compiler/provider object participates in these direct adapter tests.

I fill all eight initial managed slots, retain three owners of the path (caller,
an outside alias and a simulated global root), and prepare collection storage.
The measured successful call must use exactly four project allocations: scratch,
result bytes, enlarged slot table and enlarged collection workspace. Every
persistent-prefix and single-transient failure preserves the three path owners,
all eight existing roots and tracked byte/object baseline; a fresh successful
call follows each failure. Context allocation failure separately preserves its
output sentinel. Simulated global roots are explicit reference-count controls,
not generated-global/source-program acceptance. Every final disposal requires
zero managed roots/bytes and zero tracked allocations.

The same fixture runs through direct C and separately compiled LLVM O0/O2
forwarders of the exact native callback signature. The wrapper calls that
forwarder as its real callback, exercising its typed native link. This is not
NanoISA translation. All production TUs and the fixture use the selected strict
compiler/sanitizer flags; the LLVM forwarder's independently selected compiler,
flags and target triple are recorded. Its IR only forwards arguments and has no
memory access to claim as a separately instrumented allocator.

The runner records commands, file-backed combined stdout/stderr, return/timeout/
process-group state and content-addressed products before assertions. Every
command snapshots products before and after; paths are never reused across
routes, and fixture file vectors use distinct names. Phase inputs include all
current source C/header/include files, exact fixtures, Make/contract, selected
compilers/Python and explicit additional tool/runtime files. Fresh phase hashes
are compared at exit; this is endpoint evidence, not intermediate or complete
transitive closure. TERM/KILL group cleanup has bounded waits and cannot pass
if cleanup was needed or left descendants. No gate has run at this checkpoint.
Real Linux/Darwin ordinary and supported strict-sanitizer qualification awaits
review of this full fixture/runner. Wasm hosts remain subsequent source work.
