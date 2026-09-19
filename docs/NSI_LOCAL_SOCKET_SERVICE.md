# I bind local socket endpoints to private service owners

I record `task_372aa1e18e45e605d09f0e611495f948` under
`task_d03c232dc067e75cbc2fb2b7fb84ee46`, related to ed702, from canonical
`1cbe8c6f4c8dd2130ca44eb89e2d1f62594d32f4`. This is my preimplementation
contract. I require independent review before production, fixtures or actual
socket operations. I preserve the qualified File adapter and separate File
ownership-descriptor task6931; I do not duplicate its source or schema work.

## My current boundary

`src/nsi_runtime.c:305-332` returns descriptive handles for `net#connect`.
`src/nsi_fabric.c:55-57,125-127` creates real POSIX sockets, but these host hooks
are not an affine Socket service. `modules/websocket/websocket_helpers.c` uses
real socket I/O and integer-encoded context pointers. These existing paths do
not establish opaque, generation-qualified Socket ownership. My source resource
syntax and simulated integer handles are not operating-system cleanup evidence.

I first add only a private C local socket service. I do not modify the existing
NSI `net#connect` method, WebSocket wrapper, generated bindings, source producers,
NanoISA imports, VM dispatch or native selection. I neither expose arbitrary
endpoint/network authority nor treat a successful private adapter as d03c
completion. Real network connect, WebSocket integration and the public
Socket/Result/verified-service boundary remain required subsequent work.

## My private identity and finite capacity

I propose `src/nsi_socket.c/.h`, with an opaque, serialized service context that
owns every descriptor and a private capability table. A token contains context
identity and the existing `NlCap`; descriptors and host pointers never become
caller tokens. I use exact private resource identity `nsi:nanolang/net#Socket`
and service identity `nsi:nanolang/net/local-socket`. This is distinct from the
existing public `net#Conn` declaration; a later binding must resolve that
relationship explicitly instead of matching spelling prefixes.

I reuse PR811's checked private capability mint/transfer/consume operations.
Token validation checks context, service/resource identity, live generation and
required rights before host access. READ, WRITE and TRANSFER are the only
accepted rights. Current owners can close without WRITE. Unknown rights refuse.
A context owns at most64 live capability slots; pair acquisition requires two
available slots and transfer may need one spare slot. Exhaustion preserves
existing owners and outputs. Generation exhaustion refuses before wrap; slot
reuse never resets issued identity. I do not restart a capability table to
recover capacity while other owners remain live.

All calls on one context are serialized. I admit neither concurrent disposal
nor external access/revocation of its capability table. This is a local lifetime
contract, not a process-isolation claim. Disposed contexts remain queryable until
separate destruction releases their C storage; use after destruction is outside
the API. A context from another invocation cannot validate its tokens.

## My pair and byte operations

I acquire a real `AF_UNIX` stream socket pair. I expose the two tokens in one
private pair-result output so publication is atomic. Each endpoint has its own
checked requested rights. Both descriptors, required configuration and capability
associations must succeed before publication. A partial failure closes acquired
descriptors once, retires unpublished token associations and leaves the caller's
pair output unchanged. Existing live endpoints and their identity are unaffected.

I require nonblocking I/O and close-on-exec configuration before publication.
Platform-specific setup is reviewed before code and qualified on Linux/Darwin;
I do not claim arbitrary concurrent process-spawn atomicity from a fallback
configuration sequence. No DNS, external connection, listener, pathname or peer
credential policy is introduced by this local pair.

| Private operation | Authority and owner outcome |
| --- | --- |
| Create context / acquire pair | Publish only complete new state; acquisition failure publishes neither endpoint. |
| Send one copied U8 | Require WRITE; retain the endpoint on success, interruption, would-block or error; report exact0-or1 progress. |
| Receive one byte | Require READ; retain the endpoint; report byte with progress1, EOF with progress0/canonical value0, or an explicit non-success status. |
| Transfer endpoint | Require TRANSFER; publish a fresh token and retire the old identity only after success, preserving endpoint/stream state. Failure preserves source and destination. |
| Consume close | Validate current owner, detach and retire before one host close attempt; accepted attempts consume language authority even on reported host error. Rejected tokens close nothing. |
| Dispose / destroy | Invalidate all tokens and attempt every remaining close once, preserving the first error; destruction also frees context storage. |

My private result states distinguish OK, EOF, would-block, interrupted, host I/O,
invalid argument/token/context/rights, disposed, capacity/generation limit and
memory failure. They retain host errno, exact byte progress, accepted consumption
and secondary cleanup errors. Host errno values are target-specific. Non-success
leaves byte/pair output sentinels unchanged; I do not promise rollback of bytes
already sent or received. EOF is a successful observation with canonical output.
These C records are not a claim of language Result ABI support.

I use fixed local byte storage, preserving NUL without a managed-array or borrowed
buffer return. I perform a bounded operation rather than an unbounded retry loop;
would-block and interruption are explicit results. I prevent peer-close writes
from terminating the process using reviewed per-call/per-socket platform controls,
without changing the process-wide SIGPIPE disposition. The production checkpoint
must specify those platform choices.

I never retry a raw descriptor close after ambiguous failure, since a numeric
descriptor is not stable ownership. The production checkpoint must document the
supported platforms' close/error handling and distinguish an accepted close
attempt from proved host closure. Fault controls must not fabricate a successful
host release; real descriptor cleanup evidence is measured separately. Context
disposal remains a final cleanup path, not a replacement for bounded-live
per-endpoint retirement.

## My review and qualification order

1. I review the private API, capability association, exact error/close policy and
   platform configuration before production. I preserve existing public
   capability and File semantics. Shared changes, if necessary, receive a
   separate review rather than silently widening those APIs.
2. I review production before executing fresh ordinary fixtures. I test complete
   pair acquisition, real byte0/ordinary byte traffic in both directions, empty
   nonblocking reads, peer closure/EOF, rights checks, transfer and disposal.
   Stable token identity is independent of descriptor-number reuse.
3. I qualify stale/duplicate/cross-context rejection through the checked adapter
   without dereferencing fabricated pointers. Unrelated live endpoint sentinels
   survive refusals and cleanup. Repeated bounded-live operations reclaim slots;
   generation exhaustion and unavailable capacity preserve outputs/owners.
4. I inject failures at adapter allocation, each descriptor/configuration step,
   capability publication and transfer. Every acquired host descriptor is either
   published once or included in rollback. I retain the first error, secondary
   cleanup status, exact progress and successful subsequent recovery. I measure
   real close attempts and descriptor state separately from simulated failures.
5. I freeze source/tools and run Linux/Darwin strict compiler/sanitizer controls
   and adjacent File/capability suites. First failures remain sealed before any
   correction. Qualification identifies actual host operations and does not
   substitute in-memory byte queues for Socket acceptance.
6. I separately reuse/extend the reviewed File ownership-boundary plan for opaque
   Socket, owned pair/Result alternatives, per-outcome transfers, rights, context
   lifetime and execution-significant transport. Both frontends, selected shadows,
   VM and native AOT must then exercise actual cleanup. No raw integer FFI path
   or private descriptor query bypasses these required integration checkpoints.

Only this private local lifecycle can close this child. Public network connect,
WebSocket/service integration, full capability migration, GPU, d03c, ed702 and
release acceptance remain open. No socket, test or production code has run under
this contract-only checkpoint.

## My reviewed-platform proposal before production

For Linux and Darwin I call socketpair without relying on optional combined
creation flags, then check F_GETFL/F_SETFL(O_NONBLOCK) and
F_GETFD/F_SETFD(FD_CLOEXEC) on each endpoint before publishing either token.
The context is serialized; acquisition must not race a process fork/exec. I
make no atomic inheritance claim for this explicit configuration interval.
Linux sends use MSG_NOSIGNAL. Darwin acquisition additionally checks
SO_NOSIGPIPE on both endpoints and sends with flags0. I change no global signal
handler. Unsupported platforms fail at compilation rather than quietly omit
this policy. Each send/receive performs one syscall; EINTR and EAGAIN/EWOULDBLOCK
are distinct non-success outcomes without retry. EOF is distinct from would-block
and publishes canonical byte0/progress0; successful data publishes progress1.

For both platforms I detach the descriptor and retire its capability before one
close call. Result metadata counts attempted closes and calls returning success.
Only return0 establishes successful host closure in this adapter's evidence;
any error, including EINTR, records closure_unknown. I do not retry, probe and
close a reused descriptor number, or claim the OS kept/released it based solely
on that error. Language authority is consumed after accepted retirement even
when closure is unknown. Rollback/disposal aggregate those counts, continue
through independent live endpoints, and preserve the primary error plus the
first secondary cleanup error. An ambiguous close is an explicit cleanup limit,
not reported as leak-free recovery. Fault injection must identify whether it
performed the real close before reporting a simulated error. A platform-specific
stronger closure claim needs separate evidence; the normal real-close path must
qualify on both supported hosts.

Pair rights are copied values and the complete pair output is written once after
both endpoints are ready. A transfer snapshots its input token and registry entry
before any output write, so out==input is supported and failure preserves it.
Receive rejects overlap between its output byte and the input token's storage
before host I/O; it otherwise leaves the output byte untouched on non-success.
Output storage must be a valid caller-owned C object; overwriting an unrelated
live token is not implicit disposal. The private registry is opaque. No raw
caller buffer or pointer is retained after an operation.

My first production checkpoint keeps unknown-close state on the context after
individual tokens retire. Later disposal/destruction reports that uncertainty
instead of treating the now-empty registry as proof of complete host closure.
The unchanged private capability implementation remains the authority provider;
no NSI dispatcher, producer, verifier, native emitter or GPU file changes here.
Production remains unexecuted until independent review.
