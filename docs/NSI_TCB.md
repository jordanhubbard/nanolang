# NSI trusted computing base

I document who I trust in each deployment mode. 4.3 covers NSI invocation.
4.4 adds the capability table, shared-memory allocator, and POSIX fabric
supervisor. See [NSI_FABRIC.md](NSI_FABRIC.md).

## In-process (`adapter: inproc`)

I trust the calling process: the NSI loader (`src/nsi.c`), the session
runtime (`src/nsi_runtime.c`), the loaded document, and the in-process
handler. There is no address-space boundary. A buggy handler can touch
caller memory. Auth and capability checks still run.

## Mock (`adapter: mock`)

I trust the test or development process only. The handler is the same
typed dispatch; it does not open host `FILE*` objects or Python objects.
Use this to prove a client against the contract without a service child.

## Local-process (`adapter: local`)

I trust the parent process, the child process that loads the same NSI
document, the `socketpair` transport, and the host kernel's process and
IPC primitives. Frames are length-prefixed JSON. A malformed frame
returns `nsi:core/malformed` and does not dispatch. I do not trust the
child with host pointers: resource handles carry generation, rights,
type id, and service id.

## Remote

Remote services in 4.4 get an IPC endpoint and cannot receive a local
capability (`nl_fabric_send_cap_remote`). I do not ship a network
protocol. See [NSI_FABRIC.md](NSI_FABRIC.md).

## Shared across modes

- UTF-8 JSON documents with `nsi_version` 0
- Stable interface and method ids
- `nl_nsi_session_hello` compatibility check before calls
- Caller auth token and granted capabilities

I do not claim capability unforgeability (Phase 17) or a supervisor
(Phase 18).
