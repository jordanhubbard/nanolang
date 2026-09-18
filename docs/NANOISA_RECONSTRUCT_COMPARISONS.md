# My exact INT/BOOL generic comparison reconstruction

I record `task_4db38e272b3146f1a7a855d3094ec101` before implementation.
For EQ/NE/LT/LE/GT/GE, I admit only exact INT/BOOL operands. I preserve
`src/nanovm/value.c` equality separately from ordering: same INT compares
numerically; same BOOL has false before true; mixed INT/BOOL equality is
false and ordering follows tags INT1 before BOOL4, regardless of payload.
I use canonical bool-to-int for all same-BOOL source comparisons. Existing
typed comparator rules stay unchanged. No float, byte, enum, void or heap
values are admitted.

I preserve ordinary evaluated operand snapshots even when a mixed-tag result
is constant. Pure loop conditions remain free of calls/stores. I require
fresh small same-module VM/native/reconstructed-C/Nano checks for every
operation, both mixed orders, Boolean truth tables, integer endpoints,
call/local snapshots and loops. Other-tag refusal preserves previous output
without executing rejected operations. Historical carry679 remains excluded.

I use the isolated successful compiler copies at detached `f38b6409`, with
original/copy host-library hash checks. These preserve absolute original
cache paths; this is not hermetic relocation or a new bootstrap.
Full reconstruction parent `task_4bd034f6029b7458201db74e2c3aeb32` stays open.

My first three focused methods pass, but the pure-loop fixture reaches the
existing C-seed nested-comparison warning boundary: copied f38b `nanoc_c`
refuses generated `while (== (< local 3) true)` under `-Werror=parentheses`.
I retain `/tmp/nanolang-reconstruct-comparison-gcc.log`, exact compiler argv
and copied failure artifacts, and attach fresh evidence to existing
`task_de7d1397f86940f8b759ae07eb46820f`. This is a checked compilation
refusal, not a crash or current-main compiler conclusion.

Before changing production, I extend the reconstruction contract to use
canonical bool-to-int for equality as well as ordering when both operands
are BOOL. Equality of canonical 0/1 values is exactly Boolean equality.
The immutable snapshot and pure-loop restrictions remain; the fixture and
typed comparator rules stay unchanged. This separates nested comparisons
through existing helper calls without changing their evaluated semantics.
