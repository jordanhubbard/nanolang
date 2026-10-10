# My checked mixed-service flow checkpoint

I build this batch from `6147284e0` under #990. I instantiate my existing logical
ownership, CODE, body, cyclic and indirect/hosted engines with per-instance
catalog queries. I preserve their separate File and TCP configurations. My
[transport and flow contract](../../SERVICE_MULTI_TRANSPORT.md) distinguishes
wire identities, private flow identifiers and pending runtime obligations.

I resolve each Result member, constructor, service parameter and owner inside
its originating instance. Repeated File catalogs do not become interchangeable.
Direct and indirect calls retain exact nominal declarations. Begin-connect
retains Endpoint checking, byte writes retain byte checking, and host rights,
liveness, cleanup and result obligations remain pending. I do not admit mixed
runtime execution or publish a mixed source product in this checkpoint.

## Qualification

I run `make -f Makefile.gnu CC=/opt/homebrew/opt/llvm/bin/clang
test-services-flow`. Its final corpus passes 12,184 checks with allocation
instrumentation and 11,123 without allocation instrumentation in 10.169 seconds.
Both builds use ASan/UBSan and leak detection for the new engines and rebuilt
transport dependencies; other linked objects retain ordinary build flags.
I do not claim whole-program sanitizer instrumentation.

My independent bytecode fixture combines File/TCP/File instances and covers:

- All five methods per instance, Ok/Error branches and exact result types.
- Direct and indirect owned calls and exclusive borrowed helper calls.
- Loops, permuted maps, acyclic/CODE/body, cyclic and indirect query paths.
- Acyclic, cyclic and indirect serialized hosted preparations.
- Wrong-instance borrowed and owned calls, close operands, result storage and
  error constructors; failed logical transitions preserve their prior state.
- Copied hosted facts surviving destruction of source buffers/modules.
- A full 64-instance map with 320 imports, including bridge-array growth.
- Allocation failure before every successful preparation allocation: 121 flow
  prefixes and 440 hosted prefixes, with untouched output pointers on failure.

My synthetic Endpoints do not establish network-domain validity. That remains
a pending runtime obligation. The test does not execute mixed host operations.

I retain passing regressions for `test-socket-flow`, `test-socket-code`,
`test-socket-body`, `test-socket-hosted`, and all three
`test-file-indirect-queries` suites. I set their selected compiler to LLVM clang
and retain LLVM's binary directory on PATH. These fourteen test methods pass.

The real TCP VM/native dispatch corpus passes in 29.200 seconds. Both File
indirect VM/native dispatch methods pass under their sanitizer configuration
in 162.338 seconds. GCC 16 strict syntax checks pass for the new nominal adapter,
shared mixed engine instantiation and test fixture. These execution regressions
remain single-catalog; they do not establish mixed execution.

## Evidence and remaining work

I retain exact commands and logs in the archives, with original top-level logs
in `raw-logs.tar.gz`; adjacent text logs normalize trailing whitespace.
`source-hashes.json` pins the implementation and the unchanged user guide
fixture. Earlier passing logs record the corpus before I added the maximum
instance, standalone query, all-method and constructor controls; `complete.log`
is the final mixed corpus.

I add `test-services-flow` to `test-units`. The mixed entry points remain private
queries and do not change compiler/runtime admission. Mixed runtime values,
VM/native dispatch, per-catalog grants, paired source lowering and supervised
publication are next. DNS/WebSocket, exact release/platform checks and the full
5.1 contract remain open. I keep #990 and the release umbrella open.
