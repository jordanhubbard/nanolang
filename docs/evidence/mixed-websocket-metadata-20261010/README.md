# My mixed WebSocket metadata checkpoint

I extend the version-3 mixed catalog table under #990 from `a8c163a8d`.
I validate File/TCP/WebSocket and repeated WebSocket instances, four-method
imports, seven-type layouts, exact string tags, canonical unused slots and
same-instance nominal edges. I preserve metadata through both module conversions
and serialized transport. I derive the wire adapter's minima from the checked
catalog table; a single four-import/seven-layout WebSocket instance is covered.

I retain explicit mixed executable refusal until the flow and runtime support
WebSocket strings, deadlines and copied per-instance policy. Five WebSocket
instances exercise the case where twenty imports would pass the old five-method
count divisibility check. Existing ordinary VM/native consumers still refuse.

## Verification

I run with `/opt/homebrew/opt/llvm/bin/clang` on Darwin:

- `make -f Makefile.gnu test-multi-nominal test-services-flow CC=/opt/homebrew/opt/llvm/bin/clang`:
  metadata linked 13,530 checks; allocation-instrumented 13,841 checks;
  existing File/TCP flow linked 11,123 checks; instrumented 12,184 checks.
  These fixtures use ASan/UBSan/LSan. Metadata allocation injection includes
  five WebSocket instances through plan allocation and both wire conversions.
- I compile the original `a8c163a8d` File/TCP fixture against current providers
  and compare its serialized module with the updated fixture: byte-identical,
  SHA-256 `69299eacbefc3dad8f88731665e7f0fdb15ad48203f29522cfed4b3846b723a9`.
- I rebuild all four public runtime archives. The common nominal validator's
  WebSocket catalog dependency is included in their shared query closure.
- `make -f Makefile.gnu test-websocket-service-drivers CC=/opt/homebrew/opt/llvm/bin/clang`:
  five methods pass through C-seed/NanoVirt drivers, VM/native products and
  relocated installed publication (8.027 seconds). I do not select optional
  self-hosted compiler generations in this run.
- After removing a duplicate catalog object from the WebSocket archive list,
  relocated installed publication and repeated File/TCP permissions pass again
  (two methods, 6.853 seconds). The archive contains one catalog object.

## Retained corrections and limits

My first test link lacks the newly exercised mixed flow providers. I add those
to the test's linked and allocation-instrumented source closure. The next run
exposes the ordinary wire adapter's File-only minimum; the checked catalog
counts correct that refusal. I retain both failures alongside passing commands.

Mixed WebSocket flow, per-instance host policy/runtime, source lowering and
VM/native execution remain open. Live-network source qualification and the
Linux/Darwin exact-candidate release gates are not established here. The prior
full native text-reader failure remains open. GitHub API access prevents an
issue update; this checkpoint does not claim that #990 is complete.
