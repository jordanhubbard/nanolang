# My checked WebSocket bytecode bodies

I instantiate shared CODE, acyclic body, cyclic ownership and indirect target/body
analysis for the actual WebSocket nominal catalog after `acbde5e20` under #990.
I add core string members to the shared indirect target lookup, matching logical
flow and nominal metadata. I do not register public execution.

My actual instruction fixtures exercise connect, Message construction, borrowed
send/receive, connection and receive-result branches, string projection and
consuming close. I test each profile with ordinary and permuted import/layout
maps. The acyclic entry accepts a string argument; the indirect entry obtains
strings from a checked same-module callable. A backedge repeats borrowed send
without losing the owner or borrow. Old acyclic preparation still refuses loops.

I inspect retained service facts and indirect candidate conjunction, preserve
pending timeout checks, destroy original input storage, and reject malformed
references, variants and string indices. My included-source allocation probe
refuses every prefix before the first successful indirect analysis: 59 failed
prefixes preserve output and release tracked storage. Nominal allocation is
covered separately by my preceding nominal suite.

| Gate | Result |
| --- | --- |
| WebSocket CODE/body linked/instrumented | 2298/4844 checks with Clang, GCC and LLVM ASan/UBSan/LSan |
| Adjacent WebSocket logical flow | 268/285 checks under all three configurations |
| File indirect targets | 2726/2067 checks |
| File indirect ownership | 10277/3724 checks |
| Mixed flow/CODE/cyclic/indirect/hosted | 12184/11123 checks |

Final commands use `make -f Makefile.gnu test-websocket-body test-websocket-flow`
with `CC=/opt/homebrew/opt/llvm/bin/clang` or `CC=/opt/homebrew/bin/gcc-16`.
The LLVM sanitizer run supplies `ASAN_OPTIONS=detect_leaks=1` and
`CFLAGS='-D_GNU_SOURCE -g -fsanitize=address,undefined -fno-omit-frame-pointer'`.
Changed checker sources and probes are rebuilt; preexisting common objects use
their recorded Make build flags. `final-*.log` retain these commands and terminals.
The LLVM adjacent command uses `test-file-indirect-targets test-file-indirect-flow
test-services-flow`; its six methods pass in `adjacent.log`.

My initial negative test writes 0xff into a reference byte already equal to
0xff, so valid input remains valid and the refusal assertion fails. I preserve
that failure in `initial.log` and change the mutation to XOR 0x80, guaranteeing
changed operands. Product assertions and public admission were not weakened.

These Darwin checks establish private retained bytecode facts, not execution.
Runtime records/results, allocation accounting and trusted policy, obligation
discharge, matched VM/native dispatch, paired frontend lowering and final
Linux/Darwin release qualification remain required.
