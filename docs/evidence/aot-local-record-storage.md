# My local record storage joins

I register a directed storage conversion when I assign a record to a local.
I do not equate the producer's field nodes with the local's joined storage.
Other local kinds retain their existing equality constraints. My conversion
solver still rejects incompatible payload kinds.

My compiler acceptance tests exercise nested present/missing string records
in both assignment and function orders, with sequential assignments and with
the second assignment's branch taken or skipped. They execute in NanoVM and
native C built with warnings as errors. The producer still returns the original
string after the destination joins. Six corresponding integer-versus-string
payload cases must fail native translation.

My normal and fresh ASan/UBSan suites each pass 1,572 AOT and 1,073 shape
checks. Leak detection is disabled; I do not infer leak freedom. The focused local
assignment test passes all twelve combinations. Full compiler acceptance has
three passing tests and one failure: I pass the former `check_match_expr`
local assignment but stop at `TAIL_CALL` in `check_expr_node` (323), offset 397,
with a string/optional shape conflict. I track that remaining failure in MAC
`task_5c9df665e6024af3a0cd243dfa6fe8ca`; I have not passed the full compiler gate.

MAC refuses my claim for `task_0557eb72ec60494983314ecb11a31a36` with
`agent_status_unavailable`. I retain the evidence without forcing a closure.
