# I describe TCP declarations with explicit ownership

Under [#990](https://github.com/jordanhubbard/nanolang/issues/990), I define the
exact TCP catalog in `src/nsi_socket_plan.c` and its parsed-document fixture in
`tests/fixtures/nsi_socket_plan.json`. This is descriptive compiler input. I do
not select a compiler path, NanoISA import, host grant or runtime dispatch merely
because a document matches it.

I share declaration shapes and exact document comparison with the File catalog.
File retains its five methods, eight types, enum values, field layout and strict
document behavior. Each plan selects its own immutable descriptor in production
code; no caller can replace that descriptor or reinterpret one catalog as another.
All document validation precedes the single plan allocation. Queries borrow
process-lifetime data and remain valid after freeing the input document.

## I distinguish Conn from the private Socket token

My nominal resource is exactly `nsi:nanolang/net#Conn`, matching the existing
public declaration. Its future checked TCP binding represents one Conn with one
`NlSocketValue` connection. That value privately retains the adapter's
`nsi:nanolang/net#Socket` token. This is an explicit implementation relationship,
not a nominal alias: the catalog rejects Socket spellings, private token shapes
and substituted resource identifiers. No descriptor or adapter token appears in
the source resource's fields.

My legacy `schema/nsi/modules/net.nsi.json` describes `connect(string) -> Conn`.
I preserve that separate contract. This catalog instead declares `begin_connect`
with a numeric Endpoint and an owned Result, plus explicit completion. I do not
reinterpret a legacy connect call as this operation. DNS, the public string
endpoint path and WebSocket integration remain release requirements.

| Ordinal | Method | Input ownership | Result |
| --- | --- | --- | --- |
| 0 | begin_connect(Endpoint) | Copied endpoint | ConnectResult.Ok owns Conn; Error owns none |
| 1 | send_byte(&mut Conn, int) | Exclusive borrow; WRITE right | SendResult with progress or error |
| 2 | finish_connect(&mut Conn) | Exclusive borrow; no data right | ConnectStatus with unit or error |
| 3 | receive_byte(&mut Conn) | Exclusive borrow; READ right | ReceiveResult with ReadByte or error |
| 4 | close(Conn) | Consumes on both outcomes | CloseResult with unit or error |

Acquisition gives READ, WRITE and TRANSFER rights. Borrowed calls preserve the
owner on both outcomes; accepted close consumes it even if host closure is
unknown. All calls are effectful. The binding identifiers are exact
`nsi.tcp.v1.<method>` values, with ABI version1 and the existing
`cap:nanolang/net.connect` capability declaration. The plan describes that grant;
it does not issue it.

My nine types, in order, are Conn, SocketError, ReadByte, ConnectResult,
SendResult, ConnectStatus, ReceiveResult, CloseResult and Endpoint. Every Result
has ordered Ok/Error cases. Only ConnectResult carries an affine payload.
SocketError retains status, host errno, cleanup errno, byte progress, attempted
and successful close counts, EOF, consumption, cleanup failure, unknown closure
and pending-connect state. ReadByte carries an integer byte and EOF flag.

## I check numeric Endpoint domains before host access

Endpoint contains seven copied integers in this order: family, address0,
address1, address2, address3, port and scope_id. Family is4 or6. Address words and
scope are unsigned32-bit values represented in my signed64-bit integer type;
port is1..65535. Each word encodes four address bytes, most significant byte
first. IPv4 uses address0 and requires all other address words and scope to be0.
IPv6 uses all four words and permits an interface scope.

For example, IPv4 loopback is `(4, 2130706433, 0, 0, 0, port, 0)`; IPv6 loopback
is `(6, 0, 0, 0, 1, port, 0)`. `nl_socket_endpoint_decode` checks all integer
domains and address relationships before narrowing or shifting. It preserves
output on failure and rejects overlapping input/output storage.
`nl_socket_values_begin_connect` passes valid addresses to the qualified value
adapter. Invalid endpoint values produce ConnectResult.Error without acquiring a
host socket. Invalid C storage or exhausted value capacity publishes nothing.

I require exact catalog mutation/refusal checks, File binding byte compatibility,
integer boundary and byte-order checks, and real TCP traffic through this
Endpoint-to-value path. Paired source resolution/lowering, nominal wire metadata,
selected shadows, verified VM/native dispatch and release-platform qualification
remain required after these prerequisites.
