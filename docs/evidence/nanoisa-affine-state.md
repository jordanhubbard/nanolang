# My local affine transition evidence

I implement `task_379b04ef0b8d4a7985ad72b2ab244d5b` after PR543. My
`affine_state.c` API reads validated OWNERSHIP declarations, owns a decoded
layout copy and retains exact function-local types. This is a prerequisite
under ed702 and my full borrow task718, not completion of either parent.

I test explicit construction from fields, whole-local movement and atomic
whole-record unpacking. Resource fields move into their containing record;
unpacking creates independent field obligations. A scalar observation does
not consume its owner. Failed transitions preserve the prior state, including
when destination ownership, nominal identity or duplicate fields prevent a
transfer. A declared owned result discharges only its exact local obligation.
Other live resources still prevent exit.

My borrow cases cover repeated shared references, disjoint nested exclusive
places, held owners during later argument evaluation, shared-to-exclusive
rejection, subordinate reborrows and parent suspension. Ending an inner region
restores compatible parent access; ended references cannot be accessed. Exact
joins reject differing ownership or regions and accept a balanced move cycle.
State clones own their paths, share immutable declarations and survive source
module destruction.

I run 157 ordinary checks and 182 checks with allocation-failure injection in
this API, including constructor, clone and projected-reference allocation.
The 182-check variant also passes ASan/UBSan with the transition engine,
ownership declarations and place queries instrumented and leak detection
active. Other linked objects are ordinary builds; this is not a whole-VM
sanitizer claim. Existing 75 place checks and the declaration artifact test
also pass; ordinary artifacts still execute and resource contracts still
refuse VM/native publication.

I preserve every current executable ownership refusal. This API neither
traverses bytecode nor checks an operand stack or CFG. Its symbolic invocation
and exact joins are scoped to clones of one function analysis. Caller-place
alias substitution, callee transfers, instruction encoding, stack provenance,
whole-module verification, real VM/native access and paired producers remain
required. Float-record lowering remains task93574. My release hold remains.

I also audit all five explicit source/object consumers. The main build,
NanoISA host manifest, Forth SEE host manifest, separate examples library
rule and NanoVirt wrapper object list all include the new transition source.
My two real Forth host build/load methods and five ordinary/daemon wrapper
link tests pass. The C seed builds the actual canonical compiler; that seed
runs `--help` and emits an ordinary hello artifact which verifies and runs.
These checks establish dependency closure, not compiler bytecode fixed-point
or reference execution acceptance.

Local logs are `/tmp/nanolang-affine-state-final.log`,
`/tmp/nanolang-affine-state-sanitizers-final.log`,
`/tmp/nanolang-affine-state-host-final.log`,
`/tmp/nanolang-affine-state-seed.log` and
`/tmp/nanolang-affine-state-hello.log`.
