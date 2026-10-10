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
supervisor/cleanup domains. Exact service status values and absent-close-code
representation must be fixed with the runtime adapter before execution admission.

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

## Admission boundary

The descriptor and its queries are implemented here. The behaviors above are the
runtime acceptance contract, not claims that service execution already exists.
I have not registered this catalog in source selection or the executable binding
registry. My remaining dependencies are string-bearing nominal fields/results,
owned WebSocket transport values and cleanup, per-instance policy, verified
VM/native dispatch and paired frontend lowering. I must exercise real protocol,
refusal, fault, cleanup and installed Linux/Darwin paths before public admission.

My legacy WebSocket integer API stays separate. It cannot fabricate a verified
`Connection`, and its successful tests do not establish this ownership contract.
