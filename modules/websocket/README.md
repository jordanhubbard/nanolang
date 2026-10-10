# My WebSocket client

I accept `ws://` URLs, including hostname and bracketed IPv6 authorities. I
refuse `wss://` because I have not implemented TLS. I use OpenSSL through
`pkg-config openssl` for fresh nonce/mask randomness and the upgrade digest.

I validate my HTTP upgrade response and expected accept key, reassemble text
fragments, process ping/pong and close control frames, and reject malformed
frames and invalid UTF-8. My message limit is 1 MiB. I send masked client frames
and preserve partial input across receive timeouts.

My existing NanoLang API still returns integer registry identities. These are
not pointers, and stale or forged identities are refused. They are also not
verified affine source resources: that migration remains required under #990.
Internally, I own sockets through `NlSocketService` and close them on protocol
or transport failure. I serialize wrapper calls and refuse concurrent entry.
Callers must close every successful connection to reclaim its registry slot.

I bound connect/upgrade, send and explicit close I/O to ten seconds. Default
receive uses ten seconds; `receive_timeout` accepts 0–60000 milliseconds. Zero
polls available input. Hostname resolution remains synchronous and can outlast
that I/O budget. Public DNS capability and supervised resolution remain work.

I return complete text through the legacy C-string API. I refuse embedded NUL
text rather than truncate it; I skip binary messages. My internal protocol
engine preserves counted binary and text bytes for future service bindings.
Received strings borrow storage until the next receive or close.

I test the protocol with `make -f Makefile.gnu test-nsi-websocket-protocol`, real
local peers with `test-websocket-client`, and module packaging and dependency
shadows with `test-websocket-bindings`. I implement the relevant client framing
and upgrade requirements from [RFC 6455](https://www.rfc-editor.org/rfc/rfc6455.html).
