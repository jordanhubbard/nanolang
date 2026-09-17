# I append returned records in source order

My native array append uses the existing array element resolver for locals,
record fields and producer calls. I capture the receiver and value once in
source order. A record-returning expression therefore has addressable storage,
and a record-array field selects record storage rather than an integer helper.

The focused record-array suite passes four methods. Typed-local, direct `at`
and `array_get` forms of the original nested fixture now execute in native C
and the VM with shadows retained. A counter fixture checks receiver-before-value
order, once-only calls, alias mutation and record contents. Its source also
contains a legal compiler-temporary-like name. The six existing native call
order methods pass with their interpreter prerequisite built.

I retain the separate interpreter empty-array alias discrepancy as
`task_adb9b837ce4144a68a9d91ea00749ae0`. My order fixture uses a nonempty receiver
so it tests this repair without claiming that separate representation fix.

Evidence: /tmp/nanolang-array-append-repro.log,
/tmp/nanolang-array-append-focused.log (one missing interpreter executable in
an adjacent invocation), /tmp/nanolang-array-append-order-final.log (correct
Make target passes), and /tmp/nanolang-array-append-bootstrap.log.

Fresh three-stage bootstrap and the transpiler gate pass. An independent
review found no blocker in element selection or snapshot storage.
