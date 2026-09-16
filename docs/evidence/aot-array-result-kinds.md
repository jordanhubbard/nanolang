# My native array result kinds

I no longer classify every array-returning function as returning records. I
infer integer-, string- and record-array representations from returns and
propagate them through direct and tail calls in my existing fixed-point pass.
Unknown results remain unknown until constraints resolve them. I retain record
element field facts separately from the array's representation.

My native prototypes, definitions, call-result temporaries and return operands
use the same resolved representation. The final shape graph can supply a result
kind that was unavailable during flat fact propagation. Conflicting native
array representations produce an explicit array-result diagnostic.

I verified `make -j1 test-nvm2c` and `make -j1 test-nvm2c-sanitizers`:
1,284 AOT and 994 shape checks pass in both. The sanitizer driver verifies fresh
ASan/UBSan translator and shape objects; it does not check leak freedom. Opcode
case parity and all three sanitizer-driver unit tests also pass.

My test matrix covers all three supported array representations, caller-before-
callee and callee-before-caller definitions, ordinary and tail recursion, empty
array construction, indexing returned elements, and returning an array parameter
unchanged. I also reject conflicting integer/string array return paths.

My full compiler gate now passes the `split_tuple_type_names` result-indexing
failure in function 323. It stops at function 342, offset 4:
`register_extern_names` stores an empty string array in global slot 6. Aggregate
global storage is still missing; this change does not complete compiler
acceptance or release readiness.

My array result work is tracked by
`task_f6b029d7b7e749caa7a064c9566bd666`. My global storage work remains
`task_bcc4271b0de244c2810c09e004f6cd2e`. MAC still refuses my worker claim with
`agent_status_unavailable`.
