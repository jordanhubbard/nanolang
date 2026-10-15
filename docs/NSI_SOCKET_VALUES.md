# I retain TCP owners through Results and borrows

Under [#990](https://github.com/jordanhubbard/nanolang/issues/990), I add the
private runtime value layer needed by my future checked Socket bindings.
`NlSocketValue` contains invocation, generation and slot identity. It exposes
neither the adapter token nor a descriptor. Copying its C bytes does not duplicate
ownership. I retain the public `net#Conn` declaration separately from the adapter's
private `net#Socket`; this value layer does not equate their nominal types or
grant compiler, NSI dispatcher, VM or foreign-call admission.

I prepare one of64 value slots before TCP acquisition. A successful value call
publishes either ConnectResult.Ok, containing one pending/ready owner, or
ConnectResult.Error, containing copied adapter error metadata. The result view
reports the initial pending flag; it does not query the network. Host acquisition
failure publishes Error, while invalid C arguments, allocation/counter exhaustion
and exhausted value slots publish nothing. No publication allocates after the
host acquires a descriptor. Dropping an unhandled Ok closes its owner; dropping
Error closes nothing. Taking Ok changes that same slot to a connection and
increments its generation. Taking Error retires its slot and copies the error.

Moving any live value increments its generation, clears the source and invalidates
old C copies. Source/output storage must be disjoint and the destination empty.
The adapter token stays private in the same slot; a value move does not need a
second adapter capability slot. I refuse generation wrap rather than resurrect a
stale value. Retired counters survive slot reuse.

One exclusive borrow at a time may call finish-connect, send-byte or receive-byte.
I validate invocation, slot, generation and monotonically increasing borrow epoch
before host access. A held borrow prevents move, drop and explicit close. Ending
it clears the borrow record; copied old borrows cannot become valid again. I
never let a ConnectResult borrow its hidden owner before taking Ok.

Data calls publish canonical scalar Results: kind, success, copied adapter detail,
integer value and EOF. Successful send carries progress1; successful receive
carries0..255 or canonical0/EOF. Completion/close carry unit0. Errors carry value0
and EOF=false, including pending/interrupted/error outcomes. Receive EOF counts
as success. Host error Results retain the connection; checked value refusals
leave outputs untouched. I reject output/input overlap before mutation or I/O.

Accepted close consumes the value even if host closure is unknown. I retain the
first two cleanup failure reports and a saturating total. The adapter's terminal
unknown-close report can repeat a failure already observed during close; this
total counts reports, not distinct failed descriptors. Finish is terminal: I close
every remaining owner, including borrowed or unhandled Ok values, dispose the
adapter and cache the first execution status plus cleanup report. Finish may
invalidate outstanding borrows because no execution follows it. Repeated finish
does no host work. Invalid execution status on a live context refuses cleanup
and destruction, so the caller can supply a valid status afterward. Unknown host
closure remains an explicit failure, never a clean publication.

All operations and invocation creation require serialization. Caller C objects
must be valid and disjoint from opaque context storage; no use follows destruction.
I expose a conservative storage bound and pure kind/borrow/live-slot queries for
future execution planning. A query creates no authority.

I require real IPv4/IPv6 traffic through these values, both result arms, pending
and failed owner moves, selected fault paths and exact descriptor accounting in
included-source and separately linked tests. Full nominal catalog/source/shadow,
VM/native and WebSocket integration remain required after this lifetime layer.
