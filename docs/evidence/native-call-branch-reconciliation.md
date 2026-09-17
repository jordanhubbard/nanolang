# Native call branch reconciliation

I reconcile PR #309 at `663ff9d59c16ebcf6509682e0ae434e557c90a09`
with my integration checkpoint `74438201`.

The incoming branch checks every `snprintf` result and increases its fixed
call buffer and operand stack to support 256 arguments. I already check the
name, each argument append and closing delimiter. My formatter supports
1,024 parameters, uses the shared storage-kind selector, and my operand
stack is dynamically sized. I retain that implementation and self-tail
restart handling; my production translator is unchanged by this merge.

I import the incoming regression that calls a 256-argument void function
and tail-calls a 256-argument integer function. My existing tests additionally
exercise ordinary and tail calls with 1,024 mixed integer/string parameters.

`make -j4 test-nvm2c` passes 1,723 translator checks and 1,073 shape checks.
`make test-nvm2c-sanitizers` passes the same checks with fresh ASan/UBSan
objects and verifies instrumentation in both translator and shape objects.
This focused branch reconciliation does not establish release acceptance.
I still need the remaining branches, lifetime work and full release gates.

I also record a separate PR #303 lifetime defect in
`task_d3310bef8bd541ba9e1e267ee213eb9e`: its `emit_map_collection` scans only
the current function's map/tagged locals and stack, while `nmap_collect`
sweeps process-wide owners. Caller frames and unboxed string aliases are
not included. This is source evidence, not yet a reproduced runtime failure.
I have not imported that collector.
