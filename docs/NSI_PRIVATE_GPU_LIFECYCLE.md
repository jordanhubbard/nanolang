# I bind real device buffers to private GPU owners

I record `task_3c92beec2dc94f618d7471e46f003443` under d03c
(`task_d03c232dc067e75cbc2fb2b7fb84ee46`), related to ed702, from canonical
`d8caf70ff4a06c2586003e797a629561075c4e4e`. This is my preimplementation
contract. I require independent review of this contract and then production
before fixtures, driver initialization or GPU operations. I have only read source,
installed headers and filesystem metadata. I have not loaded a driver, queried a
live device, allocated device storage or reproduced a legacy defect.

## My source and host evidence

My selected module manifest `modules/gpu/module.json` builds
`opencl_runtime.c`. Its CUDA-first/OpenCL selector can select a CPU OpenCL
platform (`opencl_runtime.c:315-332`); success is not necessarily GPU execution.
Its allocation/copy/release entrypoints expose integer identities. The separate
`cuda_runtime.c:38-65,120-163,267-298` shows existing dynamically loaded Driver
API types, global context creation and real device allocation/copies. Neither
wrapper establishes the private affine capability boundary proposed here.

`schema/nsi/modules/gpu.nsi.json` declares submit(INT) returning resource Device;
`src/nsi_runtime.c:305-332` produces descriptive handles. It does not implement
this buffer/context lifecycle. I preserve both existing integer module ABIs and
that schema. My private Buffer is not silently treated as public Device.

On this audit's Linux aarch64 host `sparky`, filesystem metadata shows
`/dev/nvidia0`, the control/UVM nodes, `libcuda.so.595.84`, and CUDA13.0 headers.
The OpenCL loader is installed too. These files indicate a candidate host, not
working hardware or driver acceptance. I do not use nvidia-smi, clinfo, CUDA
initialization or an existing GPU example during this contract audit.

I read the installed authoritative Driver API declarations in
`/usr/local/cuda-13.0/include/cuda.h`: versioned memory/copy macros at93-105,
context push/pop/destroy at137-139 and6280-6393, memory release at8675-8708,
copy declarations at9673/9709, and the explicit three-argument
`cuCtxCreate_v2` declaration at25633. I do not bind the new four-argument
CUDA13 `cuCtxCreate` spelling to an older function-pointer signature.
My audit JSON records those file hashes; implementation must verify its actual
ABI declarations against these pinned inputs rather than assume symbol names.

## My bounded backend and identity

I first implement only a private C CUDA Driver API adapter, proposed
`src/nsi_gpu.c/.h`. I dynamically bind exact versioned symbols without using the
legacy singleton or positional OpenCL map. I require a selected CUDA device,
checked driver version, device ordinal, name and UUID in copied context metadata.
I expose backend CUDA and device class GPU explicitly. Missing driver/symbols,
no device, or initialization failure returns unavailable/error with no published
context. I never select an OpenCL CPU, managed host allocation, mock device or
malloc-backed substitute to satisfy the actual GPU gate.

Linux CUDA is this child's first representative actual GPU route. A Darwin build
may qualify an explicit unavailable result and driver-independent capability
controls; that is not Darwin GPU acceptance. Real Darwin/OpenCL GPU ownership,
public Device/Buffer/Result integration, both source frontends and verified
VM/native service calls remain required parent work. I neither implement kernels
nor require kernel execution to measure buffer allocation/copy/release here.

An opaque serialized context owns a new CUDA context, a checked private
capability table and at most64 stable registry slots. Tokens contain context
identity and existing NlCap, never a device pointer. My exact private identities
are `nsi:nanolang/gpu#Buffer` and `nsi:nanolang/gpu/private-cuda-buffer`.
READ, WRITE and TRANSFER are the only accepted rights. Every operation checks
context, exact resource/service identity, live generation and required rights
before any driver operation. Close requires a live owner, not WRITE. I reuse
private mint/transfer/consume without restarting generations or exposing the
table. Generation/capacity exhaustion preserves other owners and outputs.
A registry slot stays bound to its own allocation; I never compact live owners.

Each allocation has1..1,048,576 bytes; at most64 live allocations therefore
consume at most64MiB through this adapter. All offsets/counts are size_t and
checked by `offset <= size` and `count <= size-offset`, before pointer arithmetic.
I make no general device-memory allocator, subbuffer, shared-context, asynchronous
stream, kernel/module, remote service or concurrent caller admission.

## My context and publication discipline

I use an explicit per-call current-context bracket. Creation obtains its own
context, then restores the caller's prior current context before publication.
Operations save the prior context, push the adapter context, perform the bounded
operation, then pop and verify the popped identity and restored prior context.
I use checked versioned push/pop plus current-context query; I do not assume the
host thread has no existing CUDA user. Calls on a context are serialized; no
foreign caller can access its raw context or race disposal. I do not use the
primary-context singleton. Each operation can run on a serialized caller thread
only after the bracket establishes exact ownership; no raw handle escapes.

A failed bracket restoration is a terminal context fault: I publish no new
owner or read result, mark the context unusable for further data operations and
record `context_restore_unknown`. I never report that a foreign prior context
was restored unless checked. Cleanup uses explicit context identity, not an
assumption about whichever context is current. No blind retry of pop, release
or context destruction is allowed. If a driver fault makes cleanup impossible,
I report that limit and retain the corresponding driver library lifetime.

Acquisition reserves an unpublished capability, obtains checked device storage,
then completes the bracket before publishing the token. Any failure retires the
reserved capability, attempts cleanup of acquired storage under the exact
context and leaves the output token unchanged. I report the first primary and
first cleanup error independently. Context creation follows the same staging:
all required symbols, device identity, table allocation and host context setup
must complete before the caller receives it. Failure cleans each acquired host
resource once and records any unproved context release.

## My operation outcomes

| Operation | Owner and observable outcome |
| --- | --- |
| allocate | Success publishes one Buffer; refusal/failure preserves output and existing owners. Storage is uninitialized and must be written before its contents are used by a fixture. |
| write | I borrow Buffer and caller bytes once; exact rights/range checks precede CUDA. Synchronous copy plus synchronization establishes success. On driver failure I retain ownership but mark buffer contents unknown; data operations then refuse and close remains available. I claim no partial byte count. |
| read | I borrow Buffer and copy into bounded private host staging. Only after successful copy, synchronization and context restoration do I copy into caller output. Failure preserves caller bytes, even when the driver modified staging. |
| transfer | I publish a fresh generation-qualified token only after the existing private transfer succeeds, then retire the old identity. Input/output token alias is supported through a saved input. Capacity/exhaustion failure preserves the owner and output. No device copy occurs. |
| close | I consume an accepted capability before one device release attempt. Duplicate/stale close does no driver work. Failed release never restores authority or claims storage is free; context-level cleanup remains responsible. |
| dispose | I retire remaining owners, attempt each release once when its context can be established, then attempt explicit context destruction. I retain first primary/secondary errors and distinguish individual successful frees, attempted releases, successful context destruction and unknown cleanup. |
| destroy | I release disposed adapter C storage; unresolved driver/context lifetime is reported, not hidden as successful cleanup. Calls after destruction are outside the C API. |

A device-free error retires the token but leaves `release_unknown` in context
history. A later successful context destruction reclaims that context's ordinary
cuMemAlloc storage according to the pinned Driver API contract; I report that
separate success without rewriting the earlier free result. Failed destruction
has unknown resource status and is not retried. I keep library code loaded while
an unresolved context may depend on it. I report retained-library/context counts
instead of claiming leak-free driver failure recovery. Injected failures that
perform the real release first are explicitly distinct from failures that do not.

Caller buffers must be valid for the supplied byte count. I reject output ranges
that overlap token storage before I/O, using defined uintptr_t range checks,
including overflow refusal. Internal context/table storage is opaque. API outputs
are staged, and result records are returned by value. Zero-byte copies validate
owner/rights/range but perform no driver copy; a zero-byte allocation refuses.
Host staging allocation failure performs no driver copy and preserves output.
Host memory sanitizers cover adapter/staging ownership, not driver internals or
physical device memory correctness.

## My ordered checkpoints and acceptance

1. I review this contract, exact dynamic ABI list and context/error policy. I
   keep the separate static OpenCL identity task87ca untouched; no legacy
   identity failure is reproduced or minimized.
2. I implement only the private adapter and send the full production delta for
   independent review before preparing/executing real GPU fixtures.
3. I freeze source, harness and compiler/driver identity. Driver-independent
   controls qualify exact capability/range/publication/alias checks and every
   acquisition, bracket, allocation, copy, synchronization and cleanup failure
   point. Fault reports distinguish invoked real operations from simulated
   returns; first outcomes remain sealed.
4. I qualify an ordinary real-device lifecycle: device name/UUID/backend/driver
   recorded; allocate two independent buffers; exact varied bytes including NUL
   written/read; bounded subranges and zero-length behavior; transfer; release
   one while the other remains usable; stale/duplicate/cross-context refusals;
   repeated slot reuse and capacity rollback; final context disposal. No source
   fixture or raw legacy integer handle is used as authority. Driver errors do
   not become skipped hardware passes.
5. I qualify caller-context preservation with an unrelated live CUDA context,
   nested host context brackets, and a surviving independent allocation. Fault
   controls preserve release outcomes and report any harness-owned recovery
   separately. I do not call released raw device handles to test staleness;
   stale rejection is observed at the private token boundary before driver I/O.
6. I run strict GCC/Clang C builds, host ASan/UBSan/leak checks, instrumented and
   separately linked adapter controls, and unchanged capability/File/Socket
   adjacency. A linked real-driver pass is required in addition to mocks. I
   qualify available platform refusals honestly and retain actual GPU coverage
   gaps. I publish sealed source/tool/driver/log identities for review/merge.
7. I reconcile only this private child after canonical merge. Public GPU/Result,
   paired source/VM/native integration, real other-platform GPU acceptance,
   kernel/service submission and task87ca remain separate required obligations
   under d03c/ed702. This contract creates no executable admission by itself.
