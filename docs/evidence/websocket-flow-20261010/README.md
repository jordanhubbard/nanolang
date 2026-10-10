# My WebSocket logical transfer checks

I replace shared transfer assumptions about one-byte send and owner-only close
with exact ordered arguments from the immutable method catalog. I recognize
string record fields, use each nominal catalog's method count, and retain a
separate timeout-domain obligation for WebSocket calls. My private WebSocket flow
instantiation follows `81f45823b` under #990.

I test connect/send/receive/close, both connection-result arms, exclusive borrow
requirements, copied Message and timeout arguments, receiver preservation and
consumption, Message construction/projection and ReceiveResult extraction,
permuted import/layout maps, input metadata destruction, allocation refusal and
ordinary same-shaped record refusal. Invalid transitions retain owner/stack state.

| Gate | Result |
| --- | --- |
| WebSocket flow, linked/instrumented | 268/285 checks with Clang, GCC and LLVM ASan/UBSan/LSan |
| Unchanged private nominal tests | 12127/11722 checks with Clang and GCC |
| Unchanged File flow | 3477/3332 checks |
| Unchanged TCP flow | 4369/4188 checks |
| Mixed flow/CODE/cyclic/indirect/hosted | 12184/11123 checks |
| Actual public validator refusal | 11728 checks |

I run `make -f Makefile.gnu test-websocket-flow test-websocket-nominal` with
`CC=/opt/homebrew/opt/llvm/bin/clang` or `CC=/opt/homebrew/bin/gcc-16` (separate
OBJ_DIR for GCC). My sanitizer flow run uses the LLVM compiler with
`CFLAGS='-D_GNU_SOURCE -g -fsanitize=address,undefined -fno-omit-frame-pointer'`
and `ASAN_OPTIONS=detect_leaks=1`. Final flow logs are `final.log`,
`gcc-final.log` and `sanitized-final.log`; initial combined logs retain nominal
results. Adjacent targets are `test-file-flow test-socket-flow`, followed by
`test-services-flow test-websocket-nominal-boundary`, using LLVM Clang.

My first refusal assertion expected UNRESOLVED for an ordinary Message declaration;
the nominal validator rejected it earlier with INVALID and preserved output.
I retained that stronger refusal and corrected the expectation. My final test
places the ordinary record specifically in the Message local.

These are Darwin logical transfer checks, not executable WebSocket bytecode.
Timeout obligations are pending domain checks, not validated deadlines. Checked
CODE/body integration, runtime strings in records/results, storage/policy
integration, matched VM/native dispatch and paired source lowering remain open.
Public WebSocket execution remains refused.
