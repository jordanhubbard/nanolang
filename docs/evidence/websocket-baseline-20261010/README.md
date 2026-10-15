# My WebSocket baseline

I audit the legacy adapter at `f5696fdc6` under #990 while the mixed compiler
bootstrap is running. I preserve its production source unchanged.

My included-source fixture substitutes only send/recv, so the actual URL parser,
base64 encoder, handshake validator and frame writer execute deterministically.
LLVM clang compiles it with `-std=c11 -O1 -g -fsanitize=address,undefined
-fno-sanitize-recover=all`. The default run demonstrates:

- I accept `wss://localhost/private` and select plaintext port 80.
- I accept HTTP 403 containing an unrelated `X-Trace: 101` field as an upgrade.
- I encode sixteen zero bytes without the required two padding characters.

The `long-frame` run sends 65,536 payload bytes and ASan reports a stack buffer
overflow at `websocket_helpers.c:303`: I append four mask bytes after a ten-byte
header inside a ten-byte array. The process aborts; no external server is used.
My JSON preserves both terminals. These failures are not corrected here.

My migration must satisfy the client handshake, masking, control-frame and
fragmentation requirements in [RFC 6455](https://www.rfc-editor.org/rfc/rfc6455.html),
alongside verified service ownership, explicit authority, bounded execution and
cleanup. Repairing the existing integer/pointer wrapper alone would not satisfy
my 5.1 release contract.

## Framing and handle lifetime

My second included-source fixture substitutes only send/recv/close and executes
production frame and context functions. I use the same Clang ASan/UBSan flags.
All input bytes and original outputs are retained in its C/JSON files.

| Input | Observed legacy behavior |
| --- | --- |
| Ping payload `abc` | I send `8a00`: unmasked pong with no echoed payload |
| Nonfinal text fragment `abc` | I publish it immediately as a complete message |
| Text frame with RSV1 set | I accept it without a negotiated extension |
| Text bytes `c0af` | I publish invalid UTF-8 |
| Masked server frame | I accept and decode it |
| Close | I emit `888200000000`, declaring two absent payload bytes |
| Two text sends | I repeat the fixed masking key `37fa213d` |
| Forged handle `1` | UBSan aborts on a misaligned context member access |
| Query after close | ASan aborts on heap use-after-free |

These protocol failures violate the linked RFC's client framing requirements.
The fixture observes real context allocation/destruction, but its mocked I/O
is not a real-network qualification. No adapter or public binding changes here;
all listed failures remain required migration regressions.

## Subsequent implementation

I retain these historical failures from before my [protocol replacement](../websocket-protocol-20261010/README.md). The replacement fixes the reproduced framing, upgrade and pointer-handle defects and integrates owned sockets into the legacy module. Affine source service bindings and paired self-hosted acceptance remain open.
