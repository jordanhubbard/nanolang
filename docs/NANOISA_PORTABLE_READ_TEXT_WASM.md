# My real wasm32 read-text embedding boundary

I continue task_b7ef4d216a4948c58f7a710feacc4251 under
task_2d2e9eb552394f6e84e90f5aa08484e2 from canonical
`0a488a1040412746566290bd93110ff76e8720dd`, the actual merge of native PR903.
My native evidence stays at its recorded pins. This document proposes the next
private Wasm adapter and real embedding checkpoint. It is design only: I have
not built a guest, initialized a Wasmtime engine, installed a binding or read a
file through this new route. I downloaded official wheels for source inspection.

I preserve [my full linkage contract](NANOISA_PORTABLE_READ_TEXT_LINKAGE.md)
and [qualified native adapter boundary](NANOISA_PORTABLE_READ_TEXT_ADAPTERS.md).
The declaration query's PREPARED status supplies neither typed call authority
nor managed lifetime authority. My public translators, selectors, source
producers and CLI routes remain unchanged in this checkpoint. Direct compiled
C/LLVM-to-Wasm adapter linkage is not an emitted NanoISA CALL_EXTERN claim.

## My exact guest API and workspace amendment

I propose private `src/nanoisa/portable_read_wasm.h/.c`, outside default providers:

```c
NprManagedResult npr_wasm_read_managed(NmsRuntime *, NmsHandle argument);
/* Exactly one external Wasm function, not an indirect native callback. */
int32_t npr_wasm_host_read_text(uint32_t path_offset, uint32_t path_length,
    uint32_t destination_offset, uint32_t capacity, uint32_t length_offset);
```

The declaration uses Clang's import_module/import_name attributes for
`nanolang_host_v1.read_text`, with exact Wasm type
`(i32,i32,i32,i32,i32)->i32`. I reuse NprManagedResult, NPR_OK/DENIED/LIMIT/MEMORY/
INVALID values0..4 and unchanged NmsStatus/NmsHandle. The result is a private C
ABI struct, not a cross-host handle representation. No host receives NmsRuntime
or creates a managed handle. Guest exports for fixtures report numeric statuses
and observed bytes; no host import returns a C pointer or a borrowed string.

My native wrapper uses libc malloc/free for scratch. My existing freestanding
Wasm core deliberately has no malloc/free imports. I therefore amend the earlier
phrase “same scratch layout” to mean the same disjoint fields and bounds, with
**different storage lifetime**: one private static workspace per module instance,
not a per-call allocation. It contains4097 path bytes,1048576 destination bytes,
an aligned uint32 length and a busy flag. Source asserts its exact wasm32 size
and a conservative total<=1052692 bytes. Static data precedes __heap_base and
cannot overlap the unchanged managed allocator's pool. I reserve an explicit
64KiB stack, initial32 memory pages and maximum1024 pages (2MiB/64MiB).
There is no scratch allocation-failure call site to claim in this profile;
instantiation can fail before any file effect. Host scratch and managed result
allocation can still fail independently.

Only one synchronous call may use an instance at a time, including across
multiple NmsRuntime objects. Reentry returns NPR_INVALID/NMS_OK/zero before
changing workspace. A normal returned path clears busy, including every error.
An engine trap terminally invalidates the embedding instance; I do not reuse
its potentially busy workspace or assert that nms_finish ran after a trap.

I require active, non-disposed runtime and a valid STRING handle before the
import. I borrow argument ownership throughout; failures never consume it.
I inspect at most4097 bytes for the first NUL, reject prefix>4096, copy the
prefix to workspace and initialize length to UINT32_MAX. I pass fixed disjoint
workspace offsets and capacity1048576. I use explicit byte loops so optimized
freestanding compilation cannot introduce memcpy/memchr imports. Known status,
length<=capacity and no embedded NUL are checked after the import. Only then
nms_create copies a new owned result. Its allocation may grow linear memory;
no host view is live then. Managed validation/allocation failure returns
NPR_OK/exact NmsStatus/zero; host failure returns exact NprStatus/NMS_OK/zero.
All fields initialize on every path. Two OK statuses alone publish a nonzero
owned STRING, including empty. Caller roots/aliases remain; caller releases
success once. No implicit begin/finish/disposal or caller-root drain occurs.

## My instance and module envelope

I propose `src/runtime/portable_read_node.mjs` exporting synchronous
`createReadTextInstance(moduleBytes, paths)` and
`src/runtime/portable_read_wasmtime.py` exporting
`create_read_text_instance(module_bytes, paths)`. Both return one private
embedding object owning its compiled module, instance, copied allowlist and
export-active/callback-active/terminal state; Python also owns Engine/Store/Linker. Its checked call
method invokes a named export and terminally invalidates the object on an engine
trap. Close refuses during an active call, otherwise drops instance/context
references. Node GC does not promise immediate OS memory reclamation; Python
releases supported binding owners in dependency order. Neither keeps an fd
between callbacks. These hosts are serialized embedding libraries, not CLIs,
WASI grants, path sandboxes or general untrusted-module execution services.

Before engine instantiation I require a bounded structural module envelope:
magic/version, <=16MiB bytes, <=64 sections, <=4096 function types, <=65536
functions, names<=4096 bytes, checked u32 LEBs, exact section extents, unique
standard sections, no start section, and exactly one import: the five-i32/
one-i32 function above. I permit no imported memory, table, global, WASI or
other function. I require one defined unshared memory32 with explicit initial32
and maximum1024 pages and exactly one memory export named `memory` referring to
index0. Function exports remain fixture-specific and bounded by the envelope;
the engine validates remaining code/type/section-order rules before readiness.
Malformed, unsupported or duplicate structural declarations refuse before host
setup can open a path. This small parser does not prove guest body ownership.
I specify independent JS/Python implementations of the same bounded envelope,
with common byte-vector acceptance; no generic parser package is required.

Node's Module.imports lists names/kinds but not the full function signature;
it is not sufficient alone. My bounded type/import parser checks the exact
signature and I cross-check the engine's import/export inventory. Wasmtime's
Module.imports/FuncType supplies an additional exact signature cross-check.
The binding begins not-ready until instantiation and memory export checks finish;
the callback refuses if not-ready, callback-active or terminal. An export-active
flag guards the outer call separately: the first callback during that call is
allowed, while nested exports and nested callbacks are refused. No imported start callback
can cause effects. I never attach one context to multiple memories or stores.

## My callback validation and file semantics

At every invocation I reinterpret each i32 as unsigned32. I validate the current
memory length, path<=4096, capacity<=1048576, input/destination/four-byte output
ranges using offset<=size and length<=size-offset, and pairwise disjoint spans
before reading paths, mutating output or opening. Empty spans do not overlap.
The length cell may be unaligned; I encode little-endian bytes without native
unaligned casts. The host-owned context/allowlist is outside guest memory.
I copy path bytes, reject NUL/empty and compare exact bytes against an immutable
copied allowlist of<=64 nonempty NUL-free paths each<=4096. No normalization,
symlink containment, stable inode identity or ambient fallback is implied.

I acquire the actual current memory for each callback and retain no guest view
between calls. Neither host grows memory, calls guest code nor permits concurrent
access during this synchronous import. Shared memory and threads are excluded.
Memory growth between calls is supported; a stale ArrayBuffer/view is never
reused. Before publication I reacquire/recheck the memory size and expected
ranges. Unexpected memory change selects INVALID and publishes no length;
any prior file read is not rolled back. Host callbacks cannot access managed
roots, invoke another export or recycle the instance while active.

I allocate a host result buffer of capacity+1 and any small length/path storage
before open where the host API permits; allocation failure selects MEMORY.
I read real bytes until EOF or one excess byte. Excess selects LIMIT immediately.
Content containing NUL becomes successful empty. Open/read/close errors yield
successful empty only absent an earlier terminal status. A selected LIMIT or
MEMORY survives a later close failure. I close an opened descriptor exactly once
in finally, never retry close, and keep close outcome diagnostics separate from
first status. Unknown programming/engine faults terminate the instance rather
than becoming empty success. Known host errors return numeric statuses, not
engine traps. I distinguish actual I/O from injected progress/error reports.

Only OK copies payload into guest scratch, then writes the four-byte length
last. Failed callbacks preserve the old length cell; output bytes are unpublished
scratch and need not be restored after a failed publication. A host publication
allocation failure returns MEMORY with unchanged length. The wrapper rejects
unset/oversize lengths and unexpected status from test bindings. Host language
and runtime allocations are not claimed allocation-free, even where preallocated
read buffers avoid an extra allocation per read.

Node uses Buffer byte paths and fs.openSync/readSync/closeSync; readSync fills
preallocated storage. Python uses os.open with bytes, os.readv into preallocated
bytearray views and os.close. Interrupted reads are handled by the supported
runtime or bounded supervised call; I do not retry close. Nontermination from
external files is controlled by the outer process deadline, not falsely called
a per-file finite-time guarantee. No writes, process launch or deletion belong
to this read callback. Host libraries are trusted embedding code, not containment
for malicious JS/Python replacements.

## My pinned real providers

I inspected official [Wasmtime43.0.0 distribution metadata](https://pypi.org/project/wasmtime/43.0.0/)
and its shipped Python sources, not just current online API documentation.
The copied [provenance manifest](design/portable-read-wasm/wasmtime-43-provenance.json)
contains official download URLs, exact wheel size/hash and member hashes.

| Platform | Wheel SHA256 | Embedded engine SHA256 |
|---|---|---|
| Linux aarch64 | 30b042fd4a05d0f8a320baed53fcb971aff8a3789ed6967f4521f87931ace717 | 80e2b31ed365e66763d64e0829bf90209695ea37d85f1d14215fcdecaf5779ec |
| Darwin arm64 | 5a03c7aa03519df58fed5115ad8093d6deac46386115add715e725448e89ab25 | 1f1fe5ae2e60d9271c1d6e1db9d1048b9802478a0e30dee64103b5b6db91ff64 |

The exact APIs are Linker.define_func(module,name,FuncType,callback,
access_caller=True), Linker.instantiate, Caller.get("memory"),
Memory.data_len(caller), Memory.read(caller,start,stop) and
Memory.write(caller,bytearray,start). Shipped _memory.py uses Python slice
normalization, so read/write APIs do not replace my explicit unsigned bounds
checks. Config disables wasm_threads, wasm_multi_memory and wasm_memory64;
the envelope independently rejects incompatible memory declarations. A Wasmtime
CLI alone cannot register this import. Neither host currently has this Python
binding installed; future reviewed setup uses a private extraction/venv with
these exact wheels, no global install. Linux Python3.14.5 and puck Python3.14.7
identities and all actually loaded native libraries will be inventoried then.

I inspected the supported [Node26 filesystem APIs](https://nodejs.org/download/release/v26.0.0/docs/api/fs.html)
and [Python os.readv byte-buffer API](https://docs.python.org/3/library/os.html#os.readv).
Current Linux Node26.0.0 is `/home/linuxbrew/.linuxbrew/bin/node`, SHA256
`e09477b9d377ead4cdd6cde98242d7bdebf931751cfe6fed36cc277cdae9631a`;
puck Node26.9.0 is `/opt/homebrew/bin/node`, SHA256
`28f0fc07e2b86fc0eae5f9751f34ab14d63bd0132882f89cc4a5b4a44965592e`.
Guest compilation uses Linux `/usr/local/bin/clang` LLVM23git and wasm-ld,
SHA256 `3679c26d7a60727b11acba72366337f0e35853c9cd3cd342a463aa2f8fe10c9e`
and `26aef56c3a0344597409c0a90c74906157513c8ff44a5c56c3af11f56ba77221`;
puck Homebrew23.1.1 clang and `/opt/homebrew/opt/lld/bin/wasm-ld`, SHA256
`570c488e53383b198796e706e91b5ce5ec45bb730683a5af5e822d56a2eb1888`
and `1f80841d925f7b7d875d88e593e6e12dd496ab9227fc60f6ce9b07f73df6c81e`.
These are design preflight observations, not a qualified frozen toolchain.

## My ordered source and acceptance checkpoints

1. I review this design, then implement the private guest thunk and two real
   embedding libraries plus bounded envelope readers. I freeze complete API,
   exact workspace sizeof, cleanup and numeric limits for full source review.
   Native adapters, managed allocator, public translation and default providers
   remain byte unchanged. No host execution precedes source and fixture reviews.
2. I prepare strict freestanding O0/O2 guest compilation through retained LLVM
   IR and wasm-ld with a one-symbol allow-undefined file, never blanket
   allow-undefined. I inspect every actual import/signature/memory/export and
   absence of malloc/free/WASI/VM interpreter. Both hosts build fresh guests;
   both Node and actual private Wasmtime run the resulting real artifacts.
3. I cover native byte vectors: empty/exact1MiB/+1, multibyte filename/content,
   partial UTF8, embedded content NUL, path prefix NUL,4096/4097,64/65 copied
   allowlist rows and denied-before-open. I observe close-once and first-error
   precedence separately for actual filesystem outcomes and injected faults.
4. I exercise unsigned high-bit offsets, subtraction boundaries, all pairwise
   overlaps, unaligned length, wrong/missing/extra imports and signature/memory
   envelopes. Refusals precede effects and preserve length. I separately test
   real growth between calls, growth during result creation, exhausted memory,
   callback status/length refusals, reentry/active close, trap terminal state and
   two independent instances. All managed argument/outside/global-simulated
   roots survive; copied results survive scratch reuse; final explicit normal
   cleanup reaches baseline. No true source-global claim follows from simulation.
5. I retain output/log/status before fixture assertions with bounded process-group
   TERM/KILL cleanup, source/tool/provider maps, private package/member hashes,
   guests/IR and all first terminals. Native sanitizers cannot instrument guest
   Wasm or prebuilt engines; guest allocation hooks and actual bounded memory
   refusal are separately attributed. Existing native adapter/query and relevant
   managed Wasm/core controls remain adjacent evidence, not substituted gates.
6. After this private milestone, I still require separately reviewed complete
   typed/lifetime call authority, actual NanoISA-to-LLVM/Wasm emission and source
   lowering, exact host registration/installed packaging and full relevant
   compiler/bootstrap/application gates. b7ef/2d2 remain open throughout.

## My first production source checkpoint

I add only the two guest files and two embedding libraries named above. The
workspace is exactly1052684 wasm32 bytes, including its latch and alignment;
static assertions pin it. I leave native adapters, managed runtime, query,
translators, selectors and Make/provider lists byte unchanged. My import symbol
exists only in the private guest source. No build or host call has run.

Both envelope readers accept plain function types with numeric scalar tags
i32/i64/f32/f64 and at most64 parameters and64 results per type. They reject
GC/recursive/reference/vector type encodings. Up to65535 defined functions plus
the one imported function meet the65536 total bound. Export rows are capped
at65536. Standard section IDs above12 and any start section refuse; standard
section contents not inspected by the envelope still pass through the engine's
complete binary validation before readiness. These are private module-envelope
limits, not new language refusals. I retain exact UTF8 names and checked LEB
limits independently in each implementation.

Each factory returns a private object with call(name,...args), close() and
report(). I expose no instance/memory/export object. Python names the factory
create_read_text_instance and Node createReadTextInstance. Invalid factory
inputs raise before file effects; failed engine setup releases partial Python
owners or drops Node references. Explicit close is required after normal calls
or traps, and repeated close succeeds. Active close returns NPR_INVALID with
all owners retained. Python tears down Linker, Store, Module and Engine in
order, attempting remaining closes even if one unexpectedly raises. Node drops
references without claiming immediate engine/GC reclamation. An outer invalid
call selection refuses unchanged; an exception from a selected guest call
terminally disables further calls. A report remains available after close.

report() copies last completed callback fields status, opened, closeAttempted,
closeError and bytesRead, plus ready, terminal, exportActive and callbackActive.
A nested callback refusal leaves the enclosing callback's report intact. Fields
are diagnostics, not proof of OS close success; bytesRead counts observed read
progress even when the language result is empty/refused. The callback updates
existing report fields in its finally block without allocating a new report
after publishing the length cell. report() itself may allocate outside I/O.

Catchable Python MemoryError and Node RangeError from bounded allocation paths
return MEMORY with length unchanged. Node fatal OOM is not recoverable and is
not a claimed status path. Node filesystem error classification requires an
Error with numeric errno/string code; Python catches OSError. Unknown host
programming faults terminally invalidate the instance. For real APIs, the
checked lengths make Buffer/view range errors impossible except allocation
failure. Arbitrary monkeypatches are trusted fixture hooks, not containment.

The Python host checks installed distribution version43.0.0 and uses only
supported binding methods. Its shipped Memory.write constructs ctypes views
before the final memmove: a catchable publication allocation failure therefore
cannot partly publish the four-byte length. Payload writes can already have
happened and remain unpublished scratch. I keep all small Node publication
views and the length Buffer allocated before its final byte copy. A later
engine/host trap is terminal, not a synthetic successful cleanup report.

## My complete fixture checkpoint before execution

I add `tests/nanoisa/test_portable_read_wasm.c` and independent Node/Python
fixture drivers. The C fixture links the exact production thunk and managed
core through retained LLVM IR. Its exported controls set bounded input bytes,
initialize one or eight STRING roots, hold three references to the argument,
call the production wrapper into either of two result slots, inspect length and
FNV byte hash, and release/finish/dispose explicitly. These exports are private
fixture machinery. A separate bounded raw byte area exercises the actual host
import offsets without exposing any new production export or opcode.

The fixture requires exact argument bytes/reference count, object/byte totals,
and, in NMS_TESTING builds, unchanged tracked allocation count immediately after
failed calls and zero tracked allocations after normal disposal. Filling eight
slots and preparing collection forces the three result-publication allocations
(payload, expanded slots, expanded workspace). I measure that exact count and
exercise every persistent failure prefix plus fresh recovery. This is not a
single-transient sweep or host allocator census. The normal build omits
NMS_TESTING; both builds still execute real files, memory bounds and cleanup.

| Controls | What I assert |
|---|---|
| Eight real file vectors | Empty, copied bytes, actual multibyte filename/content, partial UTF8, content NUL, exact1MiB, one excess byte, missing file; exact status/length/hash and close attempt. |
| Copied allowlist/return |64 rows accepted,65 rejected; mutable original path bytes and row list changed after creation; first result remains `copied` after a real second read returns `second`; separate instances deny independently. |
| Managed refusal | Null/inactive/disposed runtime, invalid/array handle, empty/4096/4097 path prefix and embedded-NUL suffix; roots preserved and host entry absent where required. |
| Real memory behavior | Growth between calls; result creation grows memory on exact1MiB; actual maximum-memory exhaustion reports NMS_MEMORY with argument roots retained. |
| Raw import | High-bit/out-of-bounds offsets, all three overlap pairs, oversized capacity/path, zero-capacity EOF/excess probe and unaligned length cell; no open before rejection and unchanged sentinel length. |
| Actual engine envelope | Duplicate types, wrong section order, start, shared/memory64/wrong limits, wrong namespace/signature/import count, missing memory export, invalid body, count/section/LEB bounds. Order/body cases specifically require the engine's own compile-error class. |
| Host I/O hooks | Real first-byte progress followed by modeled EIO; actual close followed by modeled EIO/MemoryError; first LIMIT retained; fd actually closed, no retry. |
| Host storage hooks | Three direct host copy/buffer allocations independently refused, and length publication/view allocation refused; length stays91, recovery succeeds. Python additionally observes already-copied payload before refused length publication. |
| Host lifecycle hooks | Active close/nested export refused; an explicitly trusted hook grows actual memory during callback and publication refuses INVALID; a real engine trap disables all subsequent calls and explicit close succeeds. |
| Separate replacement imports | Unknown status, untouched sentinel, excessive length, embedded NUL and guest scratch-latch reentry; no second import and exact roots/normal cleanup. These are modeled callback obligations, not filesystem acceptance. |

Node temporarily intercepts the real Instance constructor only to capture the
fixture instance for forbidden-memory-growth and length-view allocation probes;
normal calls still traverse the production object. Python uses supported
Memory.read/grow/write methods and actual Caller, with scoped wrappers restored
in finally. Neither fixture changes generated guest bodies, substitutes a VM
interpreter, weakens engine memory checks, or claims containment of trusted hooks.
The trap fixture deliberately starts with live managed roots; it drops the
terminal engine instance and does not claim generated nms_finish ran. Node GC
reclamation timing remains unmeasured.

`tests/test_portable_read_wasm.py` builds four fresh guests per host: O0/O2,
each normal and NMS_TESTING, then runs each under actual Node and Wasmtime43.
All three translation units rebuild consistently in each guest. The linker
allowlist file contains only npr_wasm_host_read_text. Both production envelope
readers check the resulting actual import signature and memory before execution;
there is no blanket unresolved-symbol linker option. Eight primary real vectors
and16 modeled groups per normal engine run (19 for observed managed builds)
are counted separately; additional compound real calls are asserted in those
groups. The count is not a claim of exhaustive engine/allocator coverage.

The runner requires explicit PORTABLE_WASM_CLANG, PORTABLE_WASM_LD,
PORTABLE_WASM_NODE, PORTABLE_WASM_PYTHON and PORTABLE_WASM_WHEEL. It privately
extracts only the platform's exact checksummed43.0.0 wheel, rejects unsafe zip
paths/symlinks, verifies every pinned member, and sets child PYTHONPATH to that
private directory. It does not use pip or modify global packages. Child bytecode
cache writes and Python optimized assertions are disabled. Optional
PORTABLE_WASM_FLAGS and PORTABLE_WASM_EXTRA_TOOLS remain recorded selections.
Native ASan/UBSan options do not instrument guest Wasm or the prebuilt engines;
NMS_TESTING and explicit host hooks have the narrow domains above.

I reuse the reviewed native-adapter runner's command supervision and content
retention by module import, exposing only the new unittest class to discovery.
Each command writes argv/log/status and archives products before assertions;
process groups receive bounded TERM/KILL cleanup. Inputs and selected provider
bytes are freshly hashed at phase endpoints. Source/tool endpoint equality is
not intermediate immutability or a complete transitive toolchain claim.
The private engine, wheel, Python sources, selected executables, C/header/fixture
inputs, generated LLVM/object/Wasm and malformed module bytes remain retained.
Both hosts use isolated trees and external supervised phase/source/tool maps;
first failures stop that gate and remain recorded before correction.

I retain ordinary native adapter/query and relevant managed runtime/package
neighbors as separately scoped follow-up phases. Full compiler/bootstrap and
actual public host-linked source emission remain later b7ef/2d2 obligations.
This checkpoint contains source only; no build, wheel installation, engine
initialization or fixture filesystem call has run.

## My first qualification terminal and narrow fixture correction

Both first4ee gates stop at linker version discovery before compiling any guest.
The fixture resolves the selected wasm-ld symlink and invokes generic lld, which
refuses that mode on Linux and puck. I retain the actual first argv/log/status,
product archives and equal source/tool endpoints in
/tmp/nanolang-read-wasm-4ee-linux and puck's
/private/tmp/nanolang-read-wasm-4ee-puck. The supervised phase durations are
1.268s and0.450s; both return1 with reaped leaders and vanished process groups.
These are fixture preflight failures, not demonstrated guest or host defects.

Before changing the fixture I record this correction: selected() keeps the
absolute executable spelling returned by PATH lookup, without resolving the
symlink for argv. The existing retain() still resolves and hashes its actual
bytes. This preserves wasm-ld's required invocation mode and keeps byte identity
separate from command spelling. I change no assertion, warning policy, production
source or selected tool. Corrected fresh gates require independent review first.

## My puck adjacency tool-discovery correction

The corrected9856 engine matrix passes all eight routes on both hosts before
adjacency. Puck's first neighbor driver stops during its own tool inventory,
before any test/build, because non-login SSH supplies only system PATH entries
and shutil.which(pkg-config) returns no path. The installed tool is
/opt/homebrew/bin/pkg-config. I retain its0.048s return1 terminal at
/private/tmp/nanolang-read-wasm-9856-puck-neighbors with equal outer endpoints.

Before correcting this external driver I record the exact change: Darwin selects
that absolute pkg-config path and prepends /opt/homebrew/bin to child PATH, which
Make uses for its existing python3/pkg-config recipes. The driver records PATH
and still hashes all selected tool bytes. No fixture, compiler flag, production
source, assertion or engine route changes. Fresh puck adjacency requires review;
completed engine gates are not repeated or attributed to this later driver.
My neighbors remain native core/module lifecycle with GCC/Homebrew unsuppressed
ASan/UBSan/LSan, ordinary native adapter and declaration query. Full installed
CLI/package acceptance remains later under2d2.
