# My mixed WebSocket value evidence

I extend the private mixed value carrier to File, TCP and repeated WebSocket
instances. I copy explicit per-instance policy into bounded WebSocket cores,
preserve catalog/instance identities through ownership and borrowing, and
aggregate exactly-once terminal cleanup. My legacy catalog-only constructor and
scalar File/TCP operations still refuse WebSocket.

My LLVM ASan/UBSan/leak-checked controlled transport fixture passes 510 linked
and 12,484 instrumented checks, including 151 partial-creation allocation
prefixes. I check independent resolver paths/deadline ceilings/storage limits,
wrong-instance and forged/stale owners, embedded-NUL message retention after
context destruction, consuming close errors, cleanup failures and cached finish.
I separately pass real-transport connection denial and a relocated installed
archive/header consumer. The installed consumer uses only installed headers and
the static archive plus libcrypto.

I preserve File/TCP boundary behavior: a focused invocation of the existing
boundary fixture passes 803 checks, including all 193 failed-creation prefixes.
This does not execute that fixture's live TCP lifecycles. My four mixed flow
methods, five standalone WebSocket driver methods and repeated File/TCP product
permission method pass. I add WebSocket transport dependencies to the mixed
archive and private/public test build closures, and link libcrypto for mixed
native products.

My required real-peer mixed test fails at local listener bind with EPERM. I do
not skip that test or count this carrier as fully network-qualified. GCC's
sanitized controlled and denial probes time out at 90 seconds. A minimal GCC
sanitizer program with no NanoLang code prints from main, then times out at
LeakSanitizer's exit check. With leak detection disabled, that control and the
same two product binaries exit successfully; this is diagnostic evidence only,
not a passing GCC leak gate. LLVM's leak-checked controls pass.

I retain initial warning/discovery corrections in my roadmap: an unread receive
counter, misleading indentation, and accidental discovery of an imported peer
TestCase. None changes the ownership assertions. My checked mixed VM/native
runtime, per-instance public WebSocket grants and paired source integration are
still required under #990. These results qualify the private carrier only.
