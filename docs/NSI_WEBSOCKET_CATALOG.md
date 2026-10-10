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

## Private logical flow

My `websocket_flow` checker now uses the actual four-method nominal map. Shared
service transfer reads ordered arguments from the immutable method descriptor:
an exclusive receiver comes from a reference; a consuming receiver precedes its
copied arguments on the value stack. This supports Message plus timeout for send
and Connection plus timeout for close without changing File/TCP signatures.
Message construction and projection recognize counted strings as scalar fields.

Every WebSocket call records a distinct pending timeout-domain obligation
(`CHECK_TIMEOUT`, bit 1024), alongside binding, invocation, liveness, rights,
result and cleanup checks. Send/receive also require an exclusive borrow.
Close consumes its owner on both arms; failed logical transitions preserve state.
My logical checker neither validates timeout values nor executes instructions.

My private CODE/body checkers now apply these transfers to decoded instructions,
result branches, Message construction/projection and complete function exits.
The acyclic checker refuses loops; the separate cyclic checker computes bounded
ownership states across backedges. The indirect checker combines exact
same-module callable targets with ownership analysis, including strings returned
through helpers. Reports own their retained facts and do not borrow input CODE
or metadata. Their runtime-admitted flags remain false.

I test complete connection lifecycles in these profiles, permuted nominal maps,
malformed operands, input destruction and failed allocation prefixes. Timeout,
rights, liveness and cleanup obligations remain pending. Runtime obligation
discharge and matched dispatch must follow before public admission.

## Private wire preparation

My `websocket_codec` entrypoints now transport the exact 104-byte nominal map
through the ordinary v2 section and bridge machinery with explicitly selected
WebSocket metadata validators. I keep that selection in trusted native code;
serialized input cannot choose it. Ordinary codec entrypoints retain their
original validators and refuse this same catalog. I preserve section checks,
layout indices, signatures and ownership flags in both directions.

My WebSocket hosted plans retain copied wire and complete checked body facts,
match each pending obligation against the immutable catalog, and compute frame,
operand, reference and region bounds. The indirect plan retains literal strings
and exact same-module call targets. Its preparation tests survive destruction of
the original inputs and reject missing timeout obligations. These plans still
have `runtime_admitted=false`: they neither grant host policy nor execute a call.

## Private runtime carrier

My `websocket_runtime` carrier now accepts ordered arguments and a deadline for
all four methods. I copy explicit host policy before begin. Message records and
ReceiveResult values retain counted strings through invocation-local identities;
I validate those identities before host use. Close consumes its Connection on
both Result arms, including an invalid deadline. Failed receive publication
frees temporary bytes and invocation cleanup drains the remaining owner.

I reserve 32 MiB inside the existing 64 MiB invocation bound for the value context
and live transport peaks, including decoder growth and one temporary message or
frame per transport. I check the reservation before attempting connection.
Allocator, kernel, resolver-process and crypto-provider overhead are outside
this requested-heap bound. Copied runtime strings are charged separately within
the same invocation limit. I test these direct carrier operations against real
peers. My matched execution paths build on these primitives.

## Matched private execution

My explicit private VM and generated-C entrypoints now execute complete checked
WebSocket bytecode. Both take host policy explicitly, match catalog-derived
borrow and timeout obligations, consume ordered stack arguments and publish
only after clean invocation teardown. I retain the existing fuel and exact
instruction/variant checks. Generated code agrees with the retained wire facts
before beginning and calls its own emitted functions rather than a VM runner.

My real-peer parity tests cover direct/indirect string helpers, loops, binary NUL
payloads, permuted catalog maps, protocol errors, denied/missing policy, fuel
exhaustion and failed close cleanup. Altered import/argument calls refuse before
socket acquisition. These private paths do not register public source selection
or grant authority from service declarations.

## Immutable source companions

My `nl_websocket_binding_prepare` validates the complete counted document,
compares the exact catalog and retains canonical interface bytes plus a source
service declaration. Canonical output preserves both connection and lookup
capabilities. The shared parser rejects malformed/extra data before publication;
failed preparation preserves output. Successful plans own their bytes.

My shared source-snapshot API accepts explicit catalog identity 3, reads a
companion relative to an absolute source origin and keeps copied bytes after the
file changes or disappears. The `file_source_inputs` module carries these views
through C-seed, NanoVirt, Stage 1 and Stage 2 consumer programs, including VM and
native products. This supplies compiler input storage. My paired source checks now resolve
WebSocket names and signatures; executable lowering remains required.

## Paired source checking

My C and Nano parsers accept the exact WebSocket service declaration. Both
source plans retain all seven types and four methods, original declaration
identity, aliases and the catalog selector. Their complete rendered type/method
views agree. Wrong catalogs, incomplete bindings and out-of-range ordinals refuse.

Both type systems retain `Message.data` as a string and check the three inputs
of `send`: exclusive connection borrow, Message and timeout. Body checking
requires each Message field exactly once, accepts either field order and checks
field expressions in source order. Result arms retain their nominal payloads.
Source ownership checking remains independent of type checking and execution.

My product lowerers still refuse WebSocket catalogs before output publication.
These source checks do not yet produce executable WebSocket bytecode. I must
extend both lowerers and connect the resulting products to explicit host grants.

## Explicit public providers

I expose `nvm_websocket_execute_indirect_bytes` and
`nvm2c_emit_websocket_indirect_bytes` through an explicit catalog-3 provider.
`make -f Makefile.gnu install-websocket-public-runtime` installs my C99 headers
under `include/nanolang/websocket` and `lib/libnano_websocket_runtime.a`.
Consumers link the archive with their OpenSSL crypto and math libraries.
Generated C takes a host grant, revision-1 fuel options and scalar output.
It executes emitted functions without linking my VM or emitter entrypoints.

My host grant copies separate connection and lookup permissions, a 0..60000 ms
ceiling and an optional absolute resolver-helper path. Lookup permission requires
that path. Creation performs no I/O. A denied lookup does not affect numeric
addresses. Revocation prevents later invocations; destruction frees the grant.
Callers retain ordinary C pointer-lifetime and disjoint-storage obligations.

I serialize grants, public execution and emission through the same process-local
gate as File/TCP/mixed providers. Concurrent or reentrant calls return BUSY before
reading their arguments. Execution copies policy into the invocation and publishes
its scalar only after clean teardown. Direct internal APIs still require trusted
callers and external serialization.

My installed-provider checks cover denial, separate lookup authority, revocation,
immutable copied policy, malformed bytes and generated-native isolation. My
public real-peer suite is implemented but remains unqualified in the current
sandbox, which refuses localhost bind. This does not enable source execution or satisfy release/platform acceptance.

## Admission boundary

My private invocation carrier in `nsi_websocket_values.[ch]` now owns up to 64
Connection or ConnectResult values. It copies trusted policy and resolver path,
checks invocation/slot/generation identities, invalidates old copies on moves
and extraction, and requires one exact exclusive borrow for send/receive.
Borrow epochs cannot be reused, and exhausted counters refuse further minting.
Close clears the value before host cleanup, including invalid-deadline outcomes.
Invocation finish aborts transport I/O and destroys outstanding connections
without waiting for peer handshakes, including borrowed and unhandled owners.
I retain cleanup failure details separately from the invocation's first execution
status. Returned message bytes have independent caller ownership.

This C API assumes serialized callers and valid native storage. Its identities
reject stale/cross-invocation values; they do not protect against arbitrary native
memory access or establish verified source authority. It has no public binding.

My descriptor, owned runtime and explicit checked execution providers are
implemented. My descriptive source catalog and paired checkers now recognize
WebSocket; executable binding selection remains pending. Paired frontend lowering and public real-peer,
fault/cleanup and installed Linux/Darwin qualification remain required before
source admission and release acceptance.

My legacy WebSocket integer API stays separate. It cannot fabricate a verified
`Connection`, and its successful tests do not establish this ownership contract.
