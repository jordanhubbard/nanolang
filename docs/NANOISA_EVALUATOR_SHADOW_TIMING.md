# My evaluator shadow timing boundary

I retain the first cc606 build reports at
/tmp/nanolang-record-lists-cc606-linux-prepare and Puck
/private/tmp/nanolang-record-lists-cc606-puck-prepare. Both stopped at the original
ten-second parser shadow deadline, before bootstrap or the nine-method fixtures.
I track task_2deaad56f65c497f80546220aa1ca9d0.

I use a separate macro-only diagnostic source checkpoint. I rebuild env.c and
eval.c with that macro and link a fresh C seed against hash-verified unchanged
cc606 providers. I retain exact source, provider, tool, argv, output and binary
identities. I run the original parser_driver.nano compiler command once with its
complete imported shadows and unchanged ten-second supervision, inside a bounded
120-second external process group. No reduced graph or timing acceptance follows.

My stderr markers identify each shadow start/end, monotonic time, cumulative
checked snapshot allocation attempts, clone nodes, retained roots, borrowed-root
lookup calls and visited entries. I cap markers at 8192 and check counter overflow.
I measure cumulative clone and borrowed-root lookup nanoseconds separately; clone
timing is inclusive and not additive with overall shadow timing. Only successful
clock samples count; a diagnostic clock/overflow failure exits explicitly.
Allocator counters cover only the record/tuple/string graph include, not all
compiler allocations. Retired storage remains alive exactly as in production.

My counters and clocks add overhead. Completed marker intervals show work before
the last marker; the killed interval has no end sample. Cumulative arena cost is
a hypothesis until measured. I do not optimize or alter ownership in this build.
