# My private WebSocket wire and hosted preparation

I qualify the explicit native WebSocket codec and hosted plans under #990.
Ordinary codec entrypoints retain their original validators and refuse catalog
3. My private selection establishes exact nominal ownership metadata and all
shared container checks; it grants no execution authority.

I pass linked/instrumented hosted checks (504/4554), including both nominal
permutations, copied wire and literal retention, exact catalog/ownership refusal,
missing timeout obligations, malformed containers and 61 failed flow/hosted
allocation prefixes per permutation. Those prefixes do not instrument the
shared codec or nominal allocator; the service-module and codec regression
suites separately exercise bridge allocation recovery.

I retain these successful commands and results:

- `clang.log`: `make -f Makefile.gnu test-websocket-hosted test-websocket-body test-websocket-flow test-websocket-nominal CC=/opt/homebrew/opt/llvm/bin/clang`.
- `regression.log`: v2 module/conversion/end-to-end, File indirect hosted and mixed-service flow targets. Shared codec allocation recovery passes.
- `gcc.log` and `sanitized.log`: exact commands rebuilding the changed shared codec sources as well as WebSocket sources. `qualify.py` records the driver; GCC and LLVM ASan/UBSan/LSan pass both hosted modes.
- `public-llvm.log`: `NANO_SERVICE_MODULE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang make -f Makefile.gnu test-service-module test-websocket-nominal-boundary CC=/opt/homebrew/opt/llvm/bin/clang NMS_RUNTIME_CLANG=/opt/homebrew/opt/llvm/bin/clang NMS_RUNTIME_OPT=/opt/homebrew/opt/llvm/bin/opt`. All three service-module methods and 11719 public-boundary checks pass.

My initial fixture attempts exposed nested main macros, duplicate string-pool
entries and the existing service-section extent filter. I correct the fixture
and select the exact private extent; I retain the failed logs. String interning
reduces fixture bookkeeping assertions without removing product assertions.
My first public invocation omitted `NMS_RUNTIME_OPT`; the next selected Apple's
compiler inside the Python test despite Make's LLVM `CC`, causing unsupported
leak detection. The final command explicitly selects LLVM for that harness and
passes with leak detection enabled. I retain both failed invocations.

My input hashes cover the changed implementation and test files. This evidence
is Darwin-only. Runtime obligation discharge, string-bearing record execution,
WebSocket dispatch, paired source integration and complete release qualification
remain required. My plans still report `runtime_admitted=false`.
