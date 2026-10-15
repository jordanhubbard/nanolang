# My TCP nominal wire checkpoint

I retain TCP catalog2 as a distinct version2, 128-byte map containing five
import identities and nine nominal layout identities. File keeps its version2,
catalog1, 120-byte map. I instantiate one bounded codec and nominal validator
per immutable catalog; I do not infer catalog identity from matching shapes.

I require exact import names, directions, tags, canonical layout fields,
prior-only nested layouts and ownership descriptors. Endpoint remains a
seven-integer record, SocketError an eleven-field record, Conn an owner and
ConnectResult an affine Result. I retain global and per-kind source indices
independently, including reordered layouts and imports. Successful plans own
their maps independently of module lifetime; rejected calls preserve outputs.

I retain the final Make command and results in `make-checks.log`:

- File golden codec/nominal checks: 19,745 linked and 19,770 allocation-
  instrumented checks, both ordinary and LLVM ASan/UBSan/leak-detection runs.
- TCP codec/nominal checks: 23,069 linked and 23,082 allocation-instrumented
  checks under LLVM ASan/UBSan/leak detection.
- Adjacent File flow checks: 3,332 linked and 3,477 allocation-instrumented.

My raw corpus includes exact golden bytes, every truncation/capacity,
byte mutations, duplicate/reserved indices, overlapping input/output and
unchanged failure outputs. My module corpus mutates every retained metadata
byte, import signatures and names, tests reordered tables and allocation
failure, and retains queries after module destruction. The source hashes and
per-mode compiler commands/logs accompany this record. `compiler-matrix.log`
records the TCP corpus compiled and executed with Apple Clang and GCC16 too.
The ordinary linked dependency objects are not all sanitizer-instrumented;
the changed codec/validator and TCP catalog are.

I have not connected this private map to paired source lowering, CODE
verification, VM/native dispatch, grants or network shadows. Current File
and generic execution consumers still refuse the TCP fixture. A valid map
alone admits no network execution. Mixed File/TCP execution, public connect,
DNS/WebSocket and exact-candidate platform/release qualification remain open
under #990/#976. I do not claim a fresh full compiler bootstrap here.
