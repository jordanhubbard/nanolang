# Capability runtime and service fabric

I host Nano services on an ordinary POSIX kernel. Capabilities are
unforgeable. Bulk data moves through capability-scoped shared memory.
A supervisor starts, restarts, and replaces services. I do not claim a
kernel, GNU Emacs compatibility, or that I wrap CUDA or CPython.

`make test-nsi-cap`, `make test-nsi-shm`, `make test-nsi-fabric`.

## Capabilities

A capability is `NlCap { secret, slot, generation }`. I mint the secret
from host entropy. Fabricating one from an integer or a host pointer
fails closed (`nl_cap_from_integer`, `nl_cap_from_pointer`).

The table stores object type, service id, generation, rights, transfer
permission, optional scope, and revocation. Rights attenuate on
delegate. Transfer requires the transfer bit and revokes the source.
Restart or per-service invalidation bumps generation so old tokens
cannot be reused.

NanoLang resource types own a cap (`nl_cap_resource_own`) and consume it
(`nl_cap_resource_consume`). Forth cells are random tokens, not
addresses (`nl_cap_forth_bind` / `nl_cap_forth_lookup`).

I audit create, delegate, use, revoke, transfer, restart, and failure.

## Shared memory

Typed IPC stays the control plane. Shared regions are capability-scoped.
Rights: read, write, map, seal, transfer, borrow, return, revoke.

Every map checks offset, length, alignment, and direction. Transfer
makes the service the owner; the client cannot write until return.
When `mmap(MAP_SHARED)` is unavailable, or when a caller asks for
`force_copy`, I use a private buffer with the same read/write
semantics.

Kinds: audio, graphics, net, file, GPU. That is a typed region, not a
device driver.

`nl_shm_bench` measures control-check latency, copy time, and map time
for a payload size.

## Host

`NlHost` is the portable surface: spawn, thread, IPC pair, clock,
entropy, shared memory, files, sockets, devices, credentials.

`nl_host_posix` is the first complete adapter. It uses Darwin or Linux
kernel primitives (`fork`, `pthread`, `socketpair`, `mmap`, `clock_gettime`,
`open`, `socket`, `getuid`). `nl_host_inproc` uses the same policy with
copy-only shared memory for development. Conformance tests require
identical call results across those hosts.

Isolated services get an `AF_UNIX` `socketpair`. That is the strongest
practical isolation primitive I apply in 4.4. I do not fork
`bin/nano_emacs_worker` here. That binary is a parked 4.1 side-quest.

Remote services also get an IPC endpoint. I refuse to send a local
capability to a remote service (`nl_fabric_send_cap_remote` returns
`NL_FAB_ERR_REMOTE`). I do not ship a network protocol in 4.4.

## Supervisor

I register services, start them in dependency order, expose discovery
by interface id, readiness, health, and reverse-order shutdown.

Restart policies: never, on failure, always, fail-request, fail-application,
replace. Failure classes: transient, permanent, protocol, authorization,
quota, implementation.

Retries replay a stored result only when the method is idempotent and
the `request_id` matches. Preserve-or-revoke of in-memory state is
explicit. Rolling replacement calls `nl_nsi_compat` and refuses a
breaking document.

Calls carry deadline, cancellation, trace id, and audit id.

Budgets: memory, CPU, handles, queue, files, network, devices.
Structured accounting is numeric (`NlAccounting`). Quota exhaustion
fails closed. Cancellation beats a hung call. A crashed isolated
service restarts with a new generation; stale caps fail closed.

## Migrated services

These are fabric services with scoped caps, not host-library wraps:

| Service | Scope |
| --- | --- |
| log | first observable diagnostics service |
| fs | path prefix on the cap |
| process | executable/control token on the cap |
| net | endpoint string on the cap |
| audio | stream + shared-buffer map right |
| graphics | surface/input buffer right |
| gpu | device/queue/memory/shader/sync map right |
| python | typed adapter; `PyObject` and host pointers in the payload fail closed |

## Live editor as a fabric client

The SDL frame is a client. `editor.walker` and `editor.freeze` are
supervised isolated services with startup, readiness, restart, and
replacement. I still do not host the walker or NanoISA inside the
frame. I still do not route editor eval through the NanoVM FFI
co-process protocol.

The walker may bind a buffer, queue chrome, echo, and eval. Freeze-ISA
may compile or run a module and return a result or error. `ed_*` on
freeze fails closed.

A hung walker is cancelled or deadline-failed. Killing walker or freeze
restarts that service, bumps generation, and leaves the frame (the log
service in these tests) up. Stale session caps fail closed.

Large bind uses the shared-memory plane, with copy fallback when
mappings are forced or unavailable.

These tests use fabric-supervised stand-ins (`make test-nsi-fabric`).
They do not isolate `bin/nano_emacs_worker`. I do not claim GNU Emacs
compatibility.

## TCB additions

I trust the POSIX host adapter, the capability table, the shared-memory
allocator, and the supervisor. Isolated services trust the kernel
`socketpair`. Remote services do not receive local caps. In-process
mode does not weaken those policy checks.
