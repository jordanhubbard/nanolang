# Ordinary and tagged string arguments

I widen a function parameter from ordinary string storage to tagged storage
when it receives a lookup value. This is a monotonic inference step: once a
parameter is optional, another ordinary-string caller does not narrow it.
I do not apply this widening to aggregate fields.

An ordinary string argument constrains the optional parameter's present-value
edge, not the optional node itself. Its caller keeps ordinary string storage
and boxes the value at the call boundary. A tagged caller passes its tag and
payload unchanged. Missing values are not converted or checked by the call.
The callee checks or converts them when it consumes them.

I defer `TYPE_CHECK` representation validation until the final classification
pass. A function defined before its callers must not fail merely because its
parameter is still unresolved. Unsupported final representations still fail.

My inference bound allows the additional string-to-optional transition.
Unrelated kind conflicts remain errors, and the shape graph rejects a
string caller combined with an optional integer payload in this representation.
General scalar-tag inference remains unfinished.

## Verification

`make -j1 test-nvm2c` and `make -j1 test-nvm2c-sanitizers` each pass 1,137
AOT checks and 990 shape checks. Fresh ASan/UBSan translator and graph objects
are verified; opcode case-parity and driver tests pass. `git diff --check`
passes.

My regression matrix changes both function definition order and caller order.
It passes ordinary, present and missing strings through direct calls, a
forwarding tail call and self-tail recursion. The callee inspects the void tag
before converting a present value. A negative regression rejects incompatible
optional payload shapes after parameter widening.

Full compiler acceptance advances from the mixed-parameter conflict in
function 280 to unsupported tagged record packing in function 282
(`type_from_string`). That function stores its input in a `Type` record field;
I must preserve the tag through record packing, extraction, result inference
and compatible field joins. I have not completed that work or the release gate.

MAC still refuses my claim with `agent_status_unavailable`. I attach evidence
without closing the unfinished parent task.
