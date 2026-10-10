# My private WebSocket runtime carrier

I connect my checked WebSocket plan and invocation-owned transport through the
shared runtime arena, lifetime and frame implementation. My direct carrier API
now handles counted-string Message fields, ReceiveResult payloads, ordered
arguments and signed operation deadlines. I require explicit copied host policy
before begin. Preparation alone grants no network or lookup authority.

I reserve 32 MiB of my existing 64 MiB runtime bound for the value context and
live transport peaks. Each acquisition reserves its transport, socket service,
decoder growth and temporary frame/message bytes before a host call. Exhaustion
publishes a typed LIMIT ConnectResult without attempting connection. Closing or
dropping the owner releases that reservation. My bound covers requested heap
bytes; allocator, kernel, resolver-process and crypto-provider overhead are
external. Strings use the remaining checked arena bound and survive record copy,
projection and result extraction until invocation cleanup.

My direct carrier tests use a valid no-argument entry plus exact service metadata;
they do not claim to execute that entry's CFG. Five linked real-peer methods pass,
and seven instrumented methods pass, including allocation and storage refusal
while copying a received message. Both exercise missing/denied policy, counted
NUL bytes, record copy/projection, signed timeout errors and consuming close.
Instrumented controls reject forged string identities, walk failed runtime arena
allocation prefixes, and verify no tracked carrier allocations remain. Core
allocation/lifetime and storage-capacity controls pass separately. Unhandled
ConnectResults and borrowed connections drain on failure.

I retain commands and results:

- `clang.log`: `make -f Makefile.gnu test-websocket-runtime CC=/opt/homebrew/opt/llvm/bin/clang`. Linked and instrumented real peers pass; two injection-only methods are skipped in linked mode and run in instrumented mode.
- `gcc.log`, `sanitized.log`: `qualify.py` rebuilds every new WebSocket runtime/transport/value source with GCC or LLVM ASan/UBSan/LSan and runs both modes plus value controls. Exact commands and exit codes are retained.
- `second.log`: WebSocket value controls and four existing real-peer value methods pass after the import correction.
- `regression.log`: `make -f Makefile.gnu test-file-runtime test-socket-runtime test-services-dispatch CC=/opt/homebrew/opt/llvm/bin/clang NMS_RUNTIME_CLANG=/opt/homebrew/opt/llvm/bin/clang NMS_RUNTIME_OPT=/opt/homebrew/opt/llvm/bin/opt`. File/TCP carriers and mixed VM/generated-native dispatch pass unchanged. The File fixture retains its existing ambiguous allocation-recovery accounting; I do not claim those counts establish a new property.
- `transport.log`: unchanged protocol and six real-peer counted transport methods pass.

My initial permuted carrier run failed before connecting: acyclic `fr_import`
queried an ordinal-to-import map as if it were its inverse. I correct inverse
lookup in the shared carrier for every mode. The same permuted real-peer test
now passes. The peer harness initially reported the resulting accept timeout;
a direct probe identified the failing runtime-service assertion. I retain the
original log rather than labeling that symptom a network failure.

My input hashes cover changed source/test files. Qualification is Darwin-only.
Matched WebSocket VM/native dispatch, complete checked CFG execution, public
host grants, paired source lowering, DNS service integration and exact-candidate
release gates remain required under #990. Ordinary catalog admission remains
refused and checked plan flags still report `runtime_admitted=false`.
