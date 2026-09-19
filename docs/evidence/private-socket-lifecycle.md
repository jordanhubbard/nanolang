# I qualify private local Socket owners

I freeze production `724076809d050362555e62c1b175ff7ce17b29f9` and harness
`d8144114b` after root and independent static review. I run real local socket
pairs only; no external network connection or GPU operation is part of this
qualification. Both platforms pass on the first attempt with unchanged source,
head and tool inventories. Their2,132-file source inventories are identical.
My [seal](private-socket-lifecycle.json) retains exact commands, outcomes, source
and compiler identities, logs and executable artifacts.

| Actual route | Observed result |
| --- | --- |
| Linux ordinary GCC13.3 target | Socket instrumented/linked, File instrumented/linked and7 capability checks pass |
| Linux GCC13.3 ASan/UBSan/LSan | Same five binaries pass |
| Linux Clang18.1.3 ASan/UBSan/LSan | Same five binaries pass |
| Darwin `puck`, Apple Clang17.0 ordinary target | Same five binaries pass |
| Darwin Homebrew LLVM23.1.1 ASan/UBSan/LSan | Same five binaries pass |

Linux has23 passing runner steps,6.907seconds total measured command time;
Darwin has12,9.000seconds. Darwin is arm64 macOS26.6.2 build25G83 with SDK26.2;
its actual hostname/compiler/SDK query was retained before the gate. These are
puck measurements, not the distinct release-peer host. Linux Clang is18, not23.

Each Linux instrumented execution passes4,673 checks with322 real descriptors
opened/closed. Each Darwin execution passes5,013 with334 descriptors: its two
additional SO_NOSIGPIPE setup-failure positions are included. Both test every
endpoint's fcntl configuration step, second-mint generation exhaustion, complete
pair rollback, preserved output sentinels, live independent endpoints and fresh
recovery. Capacity/aliased-transfer failure leaves the source usable; generation
and bounded-live reuse preserve identity. Byte0, bidirectional data, EOF versus
would-block, interruption, exact rights, both token-edge overlap checks,
context/stale/duplicate refusals and closed-peer writes all pass. The harness
sets SIGPIPE to its default disposition and checks that the adapter leaves it
unchanged. Real closed-descriptor probes and an unrelated live-file sentinel
are checked without assuming descriptor numbers are identities.

Close evidence deliberately distinguishes three cases: a real successful close;
a real close followed by an injected error report; and one injected pre-close
error which deliberately leaves its descriptor open. The last case requires one
explicit harness-owned recovery close per execution. I do not attribute that
recovery to the adapter. Accepted authority is consumed in both error cases;
unknown host closure remains visible through later disposal/destruction. The
harness separately checks a historical unknown close, a later disposal error,
primary/secondary errors and continued cleanup of independent endpoints. I do
not claim every failed close proves kernel closure or leak-free recovery.

The separately linked Socket binary uses the public private-adapter header and
real implementation without syscall hooks. File/capability adjacency uses its
existing unchanged fixtures. Sanitizer builds instrument the complete relevant
adapter/capability sources; driver injection exists only in the instrumented
fixture. Ordinary Make binaries and all sanitizer binaries are retained. No
source bootstrap, public binding, NanoISA service import, VM/native service-call
admission, pathname/endpoint connect, WebSocket, GPU or full d03c/ed702 completion
is claimed. Those remain required work in their own contracts and ledger rows.
