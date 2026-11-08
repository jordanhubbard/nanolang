# My checked call-storage checkpoint

I retain fresh instrumented GCC/Clang run 42450237952 at 7c8b9f7d08da97cd319a1e1cd24630384f88311b. Both source trees match the recorded Git commit and provider hashes remain unchanged.

Clang passes all 21 selected methods. GCC fails four native computed-call subcases: once-only left-to-right argument order at 16/17/64 arguments and a later side effect after a handler return. The same fixtures pass actual interpreter and seed-shadow execution before their native invocation. I preserve these failures under #1086 and retain every original assertion. Both jobs independently pass the complete original interpreter unit suite and original array alias routes.

The checked carrier harness executes my actual private header under ASan/UBSan, retains the allocation-free small path, checks large carrier contents and zero tracked storage after release, and requires exact fail-closed denial/negative-size diagnostics. A terminated denial process does not establish recoverable allocation cleanup. My full source tests retain valid larger calls, direct/lexical/computed identity, returned strings, handler returns and prior-output shadow refusal.

I preserve complete logs, compiler/worker information, exact providers and compressed complete source trees here. The follow-up emitter correction, fresh stages and integrated Linux/Darwin acceptance remain required; #1085 and #1086 stay open.
