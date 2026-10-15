# I own pending TCP connections

Under [#990](https://github.com/jordanhubbard/nanolang/issues/990), I extend my
private Socket context with numeric IPv4/IPv6 TCP acquisition. My host caller
already holds network authority; this API does not grant source programs network
access. I retain the local-pair contract in `NSI_LOCAL_SOCKET_SERVICE.md`.

I accept copied network-order address bytes, a nonzero host-order port and an
IPv6 scope identifier. IPv4 requires zero trailing bytes and zero scope. I do
not retain the address or resolve names. The existing public `net#Conn` remains
a distinct declaration: my private `net#Socket` token is not a public Conn.
TCP entries use exact service identity `nsi:nanolang/net/tcp-socket`; local pairs
retain `nsi:nanolang/net/local-socket`. Validation checks the entry's kind and
exact capability identity before touching the host.

I mint one unpublished capability, acquire one TCP descriptor, then configure
nonblocking, close-on-exec and the existing per-platform SIGPIPE protection before
calling connect once. Immediate success publishes a ready owner. EINPROGRESS
publishes a pending owner with status OK and `connect_pending=true`: OK means
acquisition succeeded, not that a peer accepted. Every other connect error,
including EINTR, rolls back without publication. I never reconnect a failed
descriptor. I preserve the primary error and sticky unknown-close reporting.

My explicit `finish_connect` requires a current token but no data rights. On a
pending owner I poll once with timeout zero. No readiness leaves the owner
pending. Readiness permits one SO_ERROR read; a nonzero socket error becomes a
sticky failed state. Poll/getsockopt interruption or would-block retains pending
state. Other observation errors fail the connection. Invalid descriptors fail
with EBADF; a malformed SO_ERROR size fails with EIO. Zero error without writable
readiness, or with an error/hangup event, fails with ECONNABORTED. I use this
conservative failure policy instead of publishing ambiguous readiness.

I follow the [Linux connect contract](https://man7.org/linux/man-pages/man2/connect.2.html)
for writable polling followed by SO_ERROR, and the
[Darwin connect contract](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/connect.2.html)
for nonblocking initiation. I keep retries and deadlines with the caller;
one adapter call never waits for network progress.

Pending send/receive returns WOULD_BLOCK without data I/O; failed send/receive
returns the latched error without host access. Byte outputs remain unchanged.
Transfer preserves pending/ready/failed state and invalidates the old token.
Close/disposal accepts every state and attempts each owned descriptor once.
The serialized context, finite capacity, generation limits, valid C storage,
and no-racing-fork/exec preconditions remain unchanged.

I require real IPv4 and IPv6 loopback connections in included-source and linked
tests, bounded test waits, bidirectional NUL/byte traffic, EOF, refusal, transfer
and cleanup. Fault tests cover every setup step, connection/completion errors,
unpublished outputs, stale/cross-context/rights rejection and exact descriptor
accounting. I expose checked single-catalog source execution through my
[explicit TCP host API](SOCKET_HOST_API.md). Full Linux/Darwin release
qualification, mixed File/TCP execution, DNS policy and WebSocket integration
remain required.
