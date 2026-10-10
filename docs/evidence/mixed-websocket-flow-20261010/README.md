# My mixed WebSocket flow checkpoint

I continue #990 after `bd7a3a9ac`. My mixed nominal/flow/CODE/body/cyclic/indirect
queries now recognize WebSocket's four methods, seven types and counted strings
alongside File and TCP. I retain fixed logical method slots with the unused
fifth WebSocket slot refused. All WebSocket operations retain pending timeout
obligations, separate from byte and Endpoint obligations.

I enable string literal/equality/length opcodes for direct mixed queries as well
as indirect queries. Legacy single-catalog decoder selection is unchanged. My
first test run passed indirect analysis but exposed the direct decoder's string
restriction; `initial-direct-string-refusal.log` retains that failure.

I do not yet admit mixed WebSocket execution. Runtime type loading explicitly
refuses its seven-type layout, host authorization refuses a File grant for it,
and product policy refuses mixed WebSocket instead of reusing TCP permission.
Native emission preserves prior output on refusal. My unused-WebSocket fixture
checks the same boundary with no service instruction at all.

## Verification

I use `/opt/homebrew/opt/llvm/bin/clang` on Darwin. `commands.json` records exact
fixture commands; source hashes and complete Make logs accompany the results.

- `make -f Makefile.gnu test-services-flow test-websocket-service-drivers CC=/opt/homebrew/opt/llvm/bin/clang`
  passes the final query and driver corpus. The mixed WebSocket fixture passes
  30,958 linked checks and 32,518 allocation-instrumented checks. Existing
  File/TCP flow passes 11,123 linked and 12,184 instrumented checks. All five
  standalone WebSocket driver methods pass, including relocated installation
  and separate invocation/lookup permissions (50.495 seconds).
- I check 24 valid graphs: single WebSocket, File/TCP/repeated-WebSocket, and
  five WebSockets, each with direct/indirect calls, loops/no loops, and ordered/
  permuted metadata. I retain Message bytes containing NUL after overwriting
  and freeing both original module and serialized input. Ten invalid graphs
  cover wrong-instance Message/borrowed helper/close, wrong timeout tag and
  missing close timeout through direct and indirect calls.
- The mixed query fixtures use ASan/UBSan/LSan. Allocation injection exercises
  179 failing flow prefixes and 611 failing hosted prefixes before success,
  preserving caller outputs on failure. Runtime-refusal calls link the ordinary
  production archive; they do not claim sanitizer coverage of every archive
  object or any successful mixed WebSocket host execution.
- `make -f Makefile.gnu test-services-flow test-multi-nominal CC=/opt/homebrew/opt/llvm/bin/clang`
  also passes metadata's 13,530 linked and 13,841 instrumented checks.
- `make -f Makefile.gnu test-services-flow test-websocket-body test-websocket-hosted CC=/opt/homebrew/opt/llvm/bin/clang`
  preserves standalone WebSocket body (2,271 linked/4,814 instrumented) and
  codec/hosted (504 linked/4,554 instrumented) results. These standalone Make
  recipes use their allocation hooks without the mixed wrapper's sanitizer flags.
- Repeated File/TCP compiler permissions pass after the final rebuild:
  `NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang python3 -m unittest -v tests.test_mixed_service_drivers.MixedServiceDrivers.test_repeated_single_catalog_permissions`.

I leave per-instance copied WebSocket policy/value storage, VM/native dispatch,
paired source lowering, real-peer qualification and exact release/platform
acceptance open. I do not clear the previously retained full native text-reader
failure or declare release completion from these focused checks.
