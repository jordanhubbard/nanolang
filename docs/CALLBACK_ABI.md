# My retained callback ABI

## Status

I require this work before 5.0. My retained VM bridge and Apple dispatch
adapters are implemented and tested below. My SDL_mixer post-mix lifecycle
passes focused tests, but its wider audio-lock/error-reporting audit and
complete release acceptance remain unfinished on my roadmap.

My handle runtime in `src/runtime/callback_runtime.c` passes
`make test-callback-runtime`: 40,000 cross-thread invocations, scalar signature
checks, same-owner nesting, failed allocation/publication, owner-only cleanup,
deterministic queued cancellation, and late calls after shutdown. Both test
binaries pass AddressSanitizer/UndefinedBehaviorSanitizer and ThreadSanitizer
on Darwin arm64. LeakSanitizer is unsupported on that host; I ran ASan with
`detect_leaks=0`. None of this establishes VM activation, dispatch, or COP
integration.

I now preserve declared ordinary function parameter tags in the execution
module and through v2 serialization. Legacy producers still leave unknown
tags; an unknown tag is not permission to publish a callback. My v2 bridge
passes 295 checks, including mixed parameter order, distinct-signature
interning, and table growth. Allocation-failure tests check transactional
growth and updates, including a source entry borrowed from the growing table.
They pass ASan/UBSan with leak detection disabled on Darwin.

The 64-check NanoVirt suite includes reused parameter names across nested
functions, distinct nested result types, a void capturing closure, and a
tail-position call that must preserve captures. Typechecker tests reject a
nested result of the wrong type and a nested `break` targeting the outer
function's loop. The 93-check verifier suite passes. This is declaration
preservation and selected execution evidence, not full callback support.

My callback metadata tests now check serialized contracts, malformed input,
canonical assembly round trips, and allocation failure. The FFI suite passes
24 tests, including rejection of contracted imports before legacy dispatch or
co-process launch. My bytecode compiler now binds manifest contracts to loaded
declarations, and my VM host dispatches their explicit adapters. Metadata alone does
not authorize the old calling convention.

## Module manifests

I select native adapters explicitly in `module.json`:

```json
{
  "name": "queue_support",
  "c_sources": ["queue.c"],
  "callback_adapters": {
    "submit": {
      "symbol": "retained_submit",
      "abi": "retained_v1",
      "execution": "worker"
    },
    "wait": {
      "symbol": "retained_wait",
      "abi": "retained_v1",
      "execution": "worker"
    }
  }
}
```

Keys name source extern functions. `symbol` names their retained-ABI native
adapter in the selected artifact; both names must be C identifiers. I require
all three fields, accept only `retained_v1` and `owner` or `worker`, and reject
duplicate entries, duplicate fields, and unknown fields within a contract.
I reject NUL-bearing manifest text and strings before JSON decoding can lose
their length. A literal escaped backslash followed by `u0000` is still text.

My `nano_virt` CLI retains the parsed metadata alongside the native generation
returned by the builder. Adapter fields participate in its build-context
fingerprint. Both shadow and production binding use that retained metadata
and the loaded source declaration, not another read of `module.json`. A test
changes the manifest during shadow execution and checks the original contract
in production bytecode. This is a snapshot-selection check, not protection
against hostile native code modifying compiler memory.

I derive scalar argument and result tags from each declared `fn(...) -> ...`
parameter. An import without callback parameters gets a policy-only record.
I reject a callback-bearing import without an adapter contract before bytecode
publication, including an unused declaration. Unsupported shapes also fail
before publication. The C-native emitter still uses its original C ABI;
these adapters are the bytecode runtime boundary, not a C function-pointer
compatibility layer.

## Serialized import contracts

I carry contracts in v2 section `0x0B`, with feature bit `1 << 5`. I require
the bit exactly when the section contains contracts. Older readers reject the
unknown feature; legacy serialization rejects contract-bearing modules rather
than dropping their ABI facts.

The section starts with a little-endian `u32` count followed by exactly that
many 16-byte records. I reject trailing bytes and truncated records.

| Offset | Field | Encoding |
| --- | --- | --- |
| 0 | Import index | u32 |
| 4 | Parameter index | u16 |
| 6 | Handle ABI version | u8, currently 1 |
| 7 | Execution policy | u8: 0 owner, 1 worker |
| 8 | Callback signature index | u32 into SIGNATURES |
| 12 | Adapter symbol index | u32 into string constants |

I require records sorted uniquely by import and parameter index. For each
contracted import, every function or closure parameter has one record. All
records for that import agree on its nonempty, NUL-free adapter symbol and
execution policy. Callback signatures have at most 16 scalar parameters and
zero or one scalar result. An unknown tag does not satisfy the contract.

Parameter index `65535` describes a policy-only import, such as a blocking
wait. That import has no callback parameters, and its signature index is
`0xffffffff`. I do not invent a callback signature for a wait operation.

My canonical assembler preserves these facts explicitly:

```text
.callback 0 1 "retained_submit" 1 worker bool int float
.callback 1 65535 "retained_wait" 1 worker void
.import_kind 0 artifact
.parameters 2 int float
```

The referenced imports and functions must exist; `.parameters` supplies the
function's exact arity. These directives preserve metadata, not native code.

## Boundary

I do not turn a bytecode function index into a machine address. A callback-
aware native adapter receives a versioned handle with an immutable signature
and `retain`, `release`, and `invoke` operations. An ordinary C function-pointer
parameter is a different ABI. I require an explicit adapter contract at the
import boundary, checked against the function's declared signature. I do not
infer that contract from names such as `nl_queue_async`.

My initial handle ABI carries signed 64-bit integers, doubles, booleans,
bytes, borrowed opaque pointers, and void results. I reject unsupported tags
and mismatched argument or result types. Strings and aggregates need explicit
marshalling and ownership contracts before crossing this boundary.

## Ownership and shutdown

I lend a callback handle for the duration of a foreign call. A native adapter
retains it before publishing asynchronous work and releases it after its last
possible invocation, including cancelled work. Every invocation borrows a
live reference for its entire duration. Retaining an already released pointer
is invalid C, not something a reference counter can repair.

I root the callable and captures while the handle is live. Only the VM's
owning thread creates or destroys those roots. Releasing the last native
reference queues reclamation for that thread. Failed publication returns
ownership to the submitter instead of leaking the root.

Shutdown first closes admission and cancels queued invocations. On the owner
thread I detach every VM root before destroying the heap. Retained native
handles remain valid tombstones: `invoke` returns cancellation without reading
the former VM. They free themselves after the final native release. I neither
wait indefinitely for a distant timer nor free a trampoline still held by C.
Shutdown during an executing callback is rejected; the host retries after
unwinding. Cancellation is an explicit result, not successful execution.

## Execution

I keep callable identity within one VM: module ID `1` is its root, IDs `2`
and above select its append-only linked-module registry, and `0` is invalid.
Function values retain this ID in alignment space without growing my 16-byte
value representation. Closures retain the same owner alongside their captures.
`FUNCREF`, `CLOSURE_NEW`, and indirect calls use that identity. Equal function
indices in different modules do not compare equal. Host `val_function(index)`
constructs a root-module value, not a value relative to the currently executing
module. These IDs are not portable across VMs or processes.

`vm_invoke_callable` enters a callable at a suspended owner-thread host boundary.
Its activation floor prevents `RET` from resuming the paused caller. I snapshot
borrowed arguments before stack growth, retain the callable while executing,
and unwind only the new frames and stack values. I restore the caller's module,
function, instruction pointer, activation floor, and pending halt state. Normal
return preserves the caller's prior error state; failure records the callback's
error. `HALT` is not a successful callback return. This API requires the owner
thread and a suspended core; it is not a cross-thread entry point.

My tests pause at a real print trap with one or two caller frames, invoke a
capturing function from another module, and resume after normal return,
assertion failure, or halt. Allocation-failure tests check rejection and actual
stack relocation while borrowing caller stack values. Linked-module tests
exercise returned functions, returned closures, and a root callable passed into
a dependency. This establishes activation and target identity for those cases;
retained-handle execution and automatic foreign-call pumping use this activation
mechanism. Dispatch-module adapters remain unfinished.
The 272359-check VM suite and stack-allocation failure tests pass ASan/UBSan
on Darwin with leak detection disabled. I disable inlining in the sanitized
VM harness: its optimized `main` otherwise inlines enough large stack-based
VM fixtures to overflow before tests begin. The ordinary optimized suite
passes separately. This is not evidence of concurrent VM entry safety.

Native threads enqueue typed requests and wait for their result. They never
read or mutate VM frames, globals, reference counts, or intern tables. The
owner services requests at safe points. A same-owner invocation may nest;
each nested activation must restore the suspended activation on every exit.

Foreign calls that can wait for callbacks must run outside the owner while
the owner pumps requests. Marshalling and result construction remain owner-
thread work. Moving the existing whole FFI function to a worker would race
heap allocation and is not an implementation of this contract. Native code
that requires a particular OS thread needs an explicit execution policy.

I serialize VM execution, not native work. Native concurrent queues still
schedule asynchronously; callbacks share the original globals and heap. I do
not simulate completion by running every submission immediately.

## VM-owned handles

`vm_callback_create` resolves the VM-local callable and compares its recorded
parameter tags, result tag/count, and capture shape with the import contract.
Unknown parameters are rejected. I map only the six scalar/void ABI tags; I
do not infer native signatures from a function address. Publication and root
cleanup belong to the thread that initialized the VM. Foreign threads use
only the handle operations, not VM publication, pumping, or shutdown.

I retain the callable before publication and release it on the owner after
the final native reference disappears. Failed root allocation or publication
leaves the caller's reference intact. `vm_callback_pump` services a request
through `vm_invoke_callable`; it also collects released roots. VM execution
failures return `NANO_CALLBACK_EXECUTION_ERROR` and preserve the first failure
in `callback_error` and `callback_error_msg`, even if native code ignores the
status. My host checks this failure before resuming normal execution and keeps
the original VM error code when a native adapter ignores a failed callback.

`vm_callback_shutdown` cancels admission and detaches roots before heap
destruction. I reject publication after shutdown. A busy callback prevents
shutdown, and `vm_destroy` then leaves the heap intact; the owner must finish
the activation and retry. Native references can outlive successful VM teardown
as cancelled handles, with no access to the old VM or captures.

The VM suite passes 272379 checks. My retained-closure test performs 256
foreign-thread calls through a captured array and shared global state, then
starts another producer, executes 128 requests, destroys the VM, and checks
128 cancellations. I also test late invocation, unknown and mismatched
signatures, wrong-thread host operations, isolated-FFI refusal, VM assertion
failure, final-release collection, and failed root allocation/publication.
The VM bridge passes ASan/UBSan and TSan on Darwin; allocation-failure tests
pass ASan/UBSan. These use the sanitizer harness settings described above.

## Native-call scheduling

`vm_ffi_call_vm` validates contracted imports and marshals their arguments on
the owner thread. I resolve only the declared adapter in the selected module,
not a same-named global symbol. Scalar arguments use private native storage;
callable arguments use rooted retained handles. String arguments use private
NUL-terminated copies, valid only until the native call returns. I reject null
string objects and embedded NUL bytes; opaque integer-zero nulls remain separate.
Native code must not retain or free these borrowed argument copies. Aggregates
and callable return values require additional contracts and are rejected here.

For a declared string result, I copy the borrowed native bytes immediately on
the executing thread, before a worker's thread-local destructors run. Only the
owner allocates the resulting VM string after joining the worker. A native null
string result remains `void`, matching my existing direct FFI convention. I do
not free the library's returned pointer; ownership stays with the adapter.
Allocation failure stops the call without publishing a VM string. This supports
outer native-call strings, not string parameters in the retained callback ABI.

For `worker` policy, a native thread executes `ffi_call` and copies any borrowed
string result into ordinary native memory, then publishes its
completion under a mutex, and wakes the callback runtime. The owner pumps
requests while waiting and joins that worker before releasing argument storage
or constructing the VM result. A policy-only wait uses the same mechanism.
For `owner` policy, the adapter executes directly and may invoke callbacks on
that thread. It must not wait for a foreign-thread callback: that would deadlock.
Native adapter declarations remain trusted; this is not native-code isolation.

I retain one additional loader reference for each native image used by a
retained adapter. That reference survives VM and loader-registry shutdown and
lasts until process exit. Repeated use of the same loaded image shares the
reference. Final callback release is not an unload boundary: native code can
still be returning through the adapter after releasing its last handle.
This policy deliberately retains image memory and static state, including old
generations in a long-lived host. Hot-unloading those images requires a separate
native quiescence contract; I do not claim to support it.

Once a callback runtime exists, my core yields within 1,024 instruction
dispatches. The host pumps one request at each resumed trap boundary. I avoid
draining queued requests at the entry to a new callback activation, before its
first instruction. Nested blocking native calls pump explicitly. No native
worker reads VM frames, globals, or heap reference counts.

The FFI fixture tests owner and worker calls, nested foreign waits, retained
asynchronous work, policy-only joins, mixed scalar arguments, repeated yields
from a non-terminating instruction stream, VM progress while native work is
pending, and latched callback failure. The 39-test bytecode-shadow suite now
compiles and executes a real foreign-thread callback during a shadow test.
These scheduler fixtures do not replace the dispatch and release gates below.
The string fixture returns worker-local storage freed at thread exit, invokes
a VM callback before returning it, and tests aliasing argument/result storage
under both execution policies. It rejects embedded NUL and malformed arguments.
The native fixture also cancels outstanding work at VM teardown, shuts down
the loader registry, then calls the still-resident native join function and
checks cancellation. Handle tombstones alone would not keep that code live.
I test failed image-reference allocation and reopen, recovery, and deduplication
across loader shutdown. The failure fixture passes ASan/UBSan; the scheduler
and loader bridge pass ASan/UBSan and TSan on Darwin with the settings above.

## Co-process transport

Pointers and handle vtables cannot cross a process boundary. Isolated calls
need registered callback IDs, typed request/result messages, cancellation,
and retain/release accounting. Until that transport is implemented, a
callback-bearing isolated import must fail explicitly before publication;
I must not silently fall back to in-process execution. The final release
acceptance must state and test the supported policy.

## Dispatch adapters

My Apple dispatch module now declares retained adapters for async, sync,
barrier, delayed, group-async and group-notify callbacks. Submission runs on
the VM owner; sync, group waits and queue destruction use worker policy so
the owner can service callbacks. Each asynchronous publication retains its
handle and releases it after invocation, including cancellation after VM
shutdown. A queue tracks accepted timers and notifications as pending work;
destruction waits for them as well as already-enqueued tasks.

I require callers to stop submitting before destroying a queue. A callback
must not synchronously wait on its own serial queue, destroy its own queue,
or wait for a group that includes itself. Moving VM execution to an owner
thread does not remove those dependency cycles. Queue/group pointers remain
trusted opaque native handles, not checked affine resources.

I support these callbacks in-process in NanoVM. `--isolate-ffi` rejects the
contracted import explicitly; I do not fall back to in-process execution.
The non-Apple implementation reports unavailable and aborts dispatch operations.
It is not a concurrency simulator or a tested Linux libdispatch backend.

`make test-dispatch-callbacks` checks shared captured arrays, nested native
waits on distinct queues, concurrent groups, notifications, barriers, delayed
work, real dependency shadows and isolated-call refusal. The native lifecycle
fixture executes 133 callbacks, checks timer draining, and cancels a timer
after destroying its callback runtime. It passes ASan/UBSan and TSan on Darwin;
wrapper-allocation failure/recovery checks pass ASan/UBSan.

The NanoVirt suite includes transitive named and anonymous captures. I retain
the parser's non-owning link from a lambda expression to its hoisted declaration
and compile it in lexical context. Intermediate capture tables preserve entries
discovered by nested closures. These changes do not establish C-native capture
parity. The six original dispatch example gates and newly exposed SDL_mixer
callback contracts remain part of the full release acceptance.

## SDL_mixer post-mix lifecycle

I replace the invalid `Mix_SetPostMix(fn(void, int), void)` declaration with
`nl_mix_set_post_mix(fn(Mix_UserData, Mix_AudioBuffer, int) -> void, Mix_UserData)
-> int`. It returns zero after publication and minus one on allocation or
signature failure. `nl_mix_clear_post_mix()` removes the hook explicitly.
The callback borrows userdata and the writable audio buffer until it returns;
neither pointer gains ownership or may outlive its native owner.

My adapter retains a callback before publication. SDL_mixer replaces post-mix
hooks under its audio lock; after replacement returns, I can release the old
registration. A separate registration mutex serializes replacement, clearing
and close. The callback does not take that mutex. Worker policies let the VM
owner service the old callback while a setter waits for audio quiescence.
My native-C adapter widens SDL's byte count to the language's 64-bit integer.
My module header routes native audio close through the same hook cleanup.

These are lifetime rules, not a real-time latency guarantee. The audio thread
waits for VM execution. A post-mix callback must not acquire SDL's audio lock
or replace/clear itself. This restriction also excludes mixer operations from
inside the post-mix callback: a worker waiting for the audio lock cannot finish
while that callback is waiting for the worker. My unsafe boundary does not
detect or break such application dependency cycles.

I route every other declared mixer operation through a typed worker adapter;
error lookup and clearing run on the owner. This lets ordinary VM playback
changes wait for the audio lock while the owner pumps the active hook. I widen
SDK integer results to 64 bits and reject integer arguments outside the SDK
range. `Mix_FreeChunk` and `Mix_RewindMusic` preserve their source-level integer
result by explicitly returning zero after the SDK's void call. Native-C header
aliases use the same conversions and error handling without a VM scheduler.
`Mix_GetNumChannels` queries `Mix_AllocateChannels(-1)` rather than assuming a
nonexistent SDK symbol.

My adapter error snapshot belongs to SDL's process-global mixer. Each operation
clears its executing thread's SDL error first and copies the resulting message
before returning. `Mix_GetError` returns the last completed operation's snapshot,
at most 1023 bytes; `Mix_ClearError` clears it and returns zero. A successful
operation without an SDL error clears an older failure. Read the message before
starting another operation. Concurrent users share this snapshot, not per-VM
error state; mixer resource access still requires application coordination.
The getter lends a per-thread copy until its next call, which my VM string
boundary copies immediately. This intentionally differs from SDL's thread-local,
sticky-error convention. Errors from unrelated raw SDL calls are not included.

I test playback changes with the hook active and worker error transport.
General C-native language closure capture parity remains open release work;
the native C callback fixture does not establish it.

`make test-mixer-callbacks` checks allocation failure preserving the old hook,
replacement, clearing while an old callback holds the modeled audio lock,
buffer mutation, native callback arguments, and cancellation after runtime
destruction. The lifecycle fixture passes ASan/UBSan and TSan. The integration
fixture uses a real SDL dummy audio device and a captured VM array, executes
dependency/root shadows, receives at least eight callbacks per run, clears and
closes audio, and checks isolated-call refusal. This is not a hardware latency
or sound-quality test.
It also loads and frees chunks/music, plays, pauses, resumes, rewinds, fades,
changes channels and volumes, checks missing-file errors across workers and
rejects oversized integers while leaving the hook registered. A separate
native-C fixture checks the same adapter headers, cross-thread error snapshots
and callback quiescence at close.

`make test-mixer-sanitizers` runs that native-C fixture with ASan/UBSan and
TSan. On Darwin with SDL2 compatibility, I obtain SDL3's library directory from
pkg-config and set `DYLD_LIBRARY_PATH` only for the test children. An executable
with the same sanitizer first loads SDL3 without linking SDL2. It tests a missing
library as well, so loader failures appear in stderr before SDL2's startup code
can open its fatal-error dialog. This checks library discovery, not every future
SDL ABI mismatch. Darwin LeakSanitizer is disabled because it is unsupported;
these runs do not establish leak freedom. The SDK libraries themselves are not
rebuilt with instrumentation by this target.

I accept integer zero, including my current `null_opaque()` value, as a null
opaque foreign argument. I reject nonzero integer addresses at the retained
boundary. VM equality recognizes opaque null versus integer zero in either
order; floating-point zero is not a pointer.

## Evidence required

I test native retention beyond the submitting call; concurrent invocation;
owner-only execution and destruction; nested invocation; exact signatures;
callback errors; final release during execution; cancelled queued calls;
late invocation after shutdown; and allocation/publication failures. VM
integration adds closure capture collection, shared globals, nested foreign
waits, instruction safe points, and trap unwinding. Dispatch acceptance keeps
all dependency shadows and delayed queue/group work enabled.
