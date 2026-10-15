# My WebSocket protocol replacement

Under #990 I replace my legacy framing/upgrade implementation with
`src/nsi_websocket_protocol.[ch]` and move the existing module onto my owned
Socket adapter. This is executable production integration, not source-level
affine service admission.

## Implementation

I stop decoding at complete events and retain partial headers, payloads and
fragmented messages. I preserve interleaved control payloads and reject masked
server frames, reserved bits/opcodes, nonminimal lengths, oversized frames or
messages, invalid UTF-8, invalid close payloads and continuation misuse. My
client encoder reserves the complete fourteen-byte maximum header and masks
every frame. OpenSSL supplies fresh nonce and mask bytes. I validate status,
upgrade/connection headers and the expected accept value, and refuse unsolicited
extensions/subprotocols. I refuse WSS rather than select plaintext transport.

My module stores sockets as owned adapter tokens, uses monotone registry IDs
instead of pointer casts, and retires transport on failures. Timeouts preserve
partial receive state. Explicit close sends one close frame, waits for the peer
within the remaining I/O deadline, then retires the handle even on failure.
I send no duplicate close when the peer responds.

## Executed evidence

- `nl51-ws-protocol-sanitized-final.log`: separately linked and included-source
  protocol tests pass with ASan, UBSan and leak detection. I exercise allocation
  failure, all splits of a fragmented/control/binary/close stream, UTF-8 spanning
  fragments, malformed headers/payloads, key/accept vectors and masked frames at
  lengths 0, 1, 125, 126, 65535, 65536 and 1048576.
- `nl51-ws-final-sanitized.log`: six real-peer methods pass with the C transport
  compiled under ASan/UBSan, plus one independently packaged NanoLang binding
  test. I test IPv4, IPv6 and hostname connections, 70 KB frames, distinct masks,
  nonempty ping/pong, fragmented text, timeout/resume, forged upgrades, protocol
  refusal, close handshake, and stale/forged identities. The binding method uses
  ordinary repository compilers; it is not a sanitized compiler claim.
- The final GCC logs pass the same six real-peer methods and both protocol
  linkage modes with allocation-failure controls.
- `nl51-ws-bindings.log`: a cold-cache module build and dependency shadows pass
  through the ordinary C-seed native and bytecode/VM routes.
- The retained initial module capture fails because POSIX feature flags hide
  Darwin `SO_NOSIGPIPE`. My metadata now enables `_DARWIN_C_SOURCE`.
- The retained initial native link omits providers classified as shared-only.
  Protocol/socket/capability providers now belong to the ordinary object closure;
  the common UTF-8 provider remains shared-only.

My initial protocol test incorrectly treated close code 1007 as reserved. I
corrected the fixture to the reserved 1015; I did not change valid-code behavior.
These results are Darwin results. I have not run exact-candidate Linux gates or
another full bootstrap for this batch.

## Open implementation

`nl51-ws-stage1.log` and `nl51-ws-stage2.log` retain the installed self-hosted
compilers' refusal: `unsupported extern result or symbol nl_ws_is_connected`.
The ordinary wrapper's module integration is not paired compiler acceptance.
I still require manifest-backed extern lowering, service string transport,
public DNS authority, supervised hostname resolution and affine WebSocket
source/Result bindings. DNS inside the legacy wrapper remains synchronous.
I also recorded the separately discovered Linux capability entropy fallback;
checked WebSocket randomness does not fix that capability defect.

My protocol follows the client requirements in
[RFC 6455](https://www.rfc-editor.org/rfc/rfc6455.html). The old failure baseline
remains at `../websocket-baseline-20261010/`; it describes the implementation
before this replacement.
