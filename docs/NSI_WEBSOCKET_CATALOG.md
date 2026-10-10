# My owned WebSocket catalog

I define revision 1 of `nsi:nanolang/websocket` in
`tests/fixtures/nsi_websocket_plan.json`. My immutable query implementation is
`src/nsi_websocket_plan.c`. It validates the complete descriptor before allocating
a plan, preserves output on refusal, and returns process-lifetime facts independent
of the input document. A plan is not a source-admission or execution token.

## Values and operations

| Method | Inputs | Result | Owner outcome |
| --- | --- | --- | --- |
| `connect` | `url: string`, `timeout_ms: int` | `ConnectResult` | `Ok` acquires one `Connection`; `Error` acquires none |
| `send` | `connection: &mut Connection`, `message: Message`, `timeout_ms: int` | `SendResult` | Both arms preserve the borrowed owner |
| `receive` | `connection: &mut Connection`, `timeout_ms: int` | `ReceiveResult` | Both arms preserve the borrowed owner |
| `close` | `connection: Connection`, `timeout_ms: int` | `CloseResult` | Both arms consume the owner |

`Connection` is a resource. I acquire read/write/transfer rights on successful
connect, require write for send and read for receive, and allow consuming close
without read/write rights. An operation that makes the transport terminal does
not erase its caller's ownership obligation: the caller must still close the
connection or let invocation cleanup consume it. Close must retire authority
before attempting transport cleanup, including its Error outcome.

`Message` is a passive record with ordered fields `binary: bool` and
`data: string`. Data is a counted immutable byte sequence, including embedded zero
bytes and empty messages. Text messages must pass UTF-8 validation; binary
messages need no UTF-8 interpretation. Both directions are bounded by the
existing 1 MiB protocol limit. `SendResult.Ok` reports the complete payload byte
count, and `ReceiveResult.Ok` contains a copied `Message`, never a borrowed decoder
buffer. Receive timeout and peer close are Error outcomes, not empty messages.

My four Result types retain ordered `Ok`/`Error` variants. Connect's Ok contains
`Connection`; send's Ok contains int; receive's Ok contains `Message`; close's Ok
has no payload. Each Error contains `WebSocketError` with ordered fields:
`status`, `host_errno`, `resolver_error`, `supervisor_status`, `close_code`,
`cleanup_errno`, `cleanup_failed`, `closure_unknown`, `terminal`. The first six
are int, and the last three are bool. These retain separate host/resolver/
supervisor/cleanup domains. My shared host transport fixes status values as
0 OK, 1 argument, 2 rights, 3 memory, 4 limit, 5 timeout, 6 host I/O,
7 protocol, 8 cryptographic randomness and 9 closed. I use close code 0 when
no peer close code was received. These values do not themselves admit execution.

My private `nsi_websocket_transport.h` API now owns the shared transport used by
the legacy module. Its caller must close each successful connection exactly
once, including after terminal I/O failure. Receive publishes independently
allocated counted bytes; they survive further receives and connection closure.
Allocation refusal preserves the pending message for retry. I retain partial
input on timeout and terminate a stream after failed frame transmission.
Close destroys the connection even when its deadline is invalid or expires.
The API assumes serialized calls and valid, disjoint C storage; its pointer is
not an affine source value or an unforgeable runtime token.

## Deadline and authority requirements

Every timeout parameter has descriptive domain `NL_SERVICE_DOMAIN_TIMEOUT_MS`:
0..60000 milliseconds. Zero permits only immediately available progress and may
not initiate DNS. Positive connect time covers DNS, socket connect and HTTP
upgrade together. Send/receive cover control-frame processing as well as payload
progress. Close consumes authority even when its handshake cannot finish before
the deadline. A partial frame send cannot leave a usable connection with an
ambiguous stream boundary; its error must mark the transport terminal. Partial
receive input remains owned across nonterminal timeout.

I declare distinct capabilities, in order:

1. `cap:nanolang/websocket.connect` for network connection authority.
2. `cap:nanolang/net.lookup` for hostname lookup authority.

Declarations do not grant authority. The eventual per-instance trusted host
policy must authorize connection separately from DNS, cap operation deadlines,
and select the supervised resolver executable. Numeric addresses need connection
permission but no DNS permission. I require the reviewed URL validation, plain
`ws://` transport, fresh masking/nonces and upgrade/framing checks. `wss://`
remains refused until a TLS contract is implemented.

## Private nominal transport

I now validate a private nominal map with transport version 2, catalog ID 3,
four import mappings and seven type mappings. Its exact extent is 104 bytes:
a 16-byte header, four 8-byte method rows and seven 8-byte type rows. Integer
fields use little-endian encoding. The header records version/catalog as u16,
method/type counts as u32 and one zero reserved u32. Each row retains its exact
catalog ordinal followed by a distinct, non-reserved module index. File and TCP
maps retain their existing wire bytes.

My private checker matches import identities and exact parameter tags, ordered
layout names/members and prior-layout references. It recognizes core string as a
scalar with no nested layout. Connection and ConnectResult carry complete plus
resource ownership flags; Message and the remaining results remain passive.
String parameters cannot be borrowed as resource references. Same-shaped ordinary
records gain no catalog identity. I permit valid permutations of global import
and layout indices while preserving these contracts.

The checked map owns its copied rows and survives destruction of source metadata.
It describes types and imports only: it does not validate function bodies or
execute methods. The public service validator still refuses catalog 3, and
execution remains classified as pending. `test-websocket-nominal-boundary`
checks that separation directly against the actual public validator.

## Admission boundary

The descriptor and its queries are implemented here. The behaviors above are the
runtime acceptance contract, not claims that service execution already exists.
I have not registered this catalog in source selection or the executable binding
registry. My remaining dependencies are string-bearing runtime fields/results,
owned WebSocket transport values and cleanup, per-instance policy, verified
VM/native dispatch and paired frontend lowering. I must exercise real protocol,
refusal, fault, cleanup and installed Linux/Darwin paths before public admission.

My legacy WebSocket integer API stays separate. It cannot fabricate a verified
`Connection`, and its successful tests do not establish this ownership contract.
