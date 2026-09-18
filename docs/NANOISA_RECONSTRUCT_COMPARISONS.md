# My exact INT/BOOL generic comparison reconstruction

I record `task_4db38e272b3146f1a7a855d3094ec101` before implementation.
For EQ/NE/LT/LE/GT/GE, I admit only exact INT/BOOL operands. I preserve
`src/nanovm/value.c` equality separately from ordering: same INT compares
numerically; same BOOL has false before true; mixed INT/BOOL equality is
false and ordering follows tags INT1 before BOOL4, regardless of payload.
I use canonical bool-to-int only for same-BOOL source ordering. Existing
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
