I continue #990 on `release/5.1-completion-20261007`, owning the WebSocket module, `src/nsi_websocket_protocol.[ch]`, socket prerequisites and related tests. My user has prioritized implementation gaps over CI tuning.

I replace the production legacy WebSocket framing/upgrade code, use owned socket tokens internally and replace pointer handles with monotone registry identities. I validate upgrade accept keys, generate fresh nonce/mask randomness, reassemble fragments and preserve control payloads, reject malformed framing/UTF-8, preserve partial input across timeouts, refuse WSS and complete bounded close handshakes.

Darwin checks pass: six real-peer methods with ASan/UBSan and GCC, linked/included protocol tests (including allocation failures and 1 MiB boundaries), and cold-cache NanoLang native/bytecode packaging with dependency shadows. Evidence: `docs/evidence/websocket-protocol-20261010/README.md`.

I still require public string service transport, supervised DNS and affine WebSocket source bindings. Both installed self-hosted stages refuse this ordinary module with `unsupported extern result or symbol nl_ws_is_connected`; the retained logs prevent claiming paired acceptance. I also found that Linux `src/nsi_cap.c` substitutes predictable bytes when getentropy fails; I recorded fail-closed entropy handling as required work under this issue. This issue remains open.

I also correct earlier #982 coverage wording: its step labelled 3-stage bootstrap ran make build, not bootstrap3. That label did not establish a fixed point. The separate local mixed-service receipt records all 17 actual bootstrap steps and both generations' 24 service methods passing. Full exact-candidate hosted qualification remains open.
