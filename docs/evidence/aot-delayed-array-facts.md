# Delayed array element facts

I preserve an array's established representation while its element facts are
unknown. Previously, `ARR_GET` on an unresolved array supplied an integer fact,
and inserting that provisional value could change an explicitly constructed
record array into an integer array. A reduced regression failed before the fix
with conflicting `nrarr_t`/`narr_t` arguments to the same callee.

Unknown array extraction now yields unknown facts, including unknown record
fields. Inserting an unknown element preserves the array's existing facts.
I also stopped `ARR_LEN` from imposing integer-array representation on an
unknown parameter: length does not establish an element type.

## Verification

`make -j1 test-nvm2c` and `make -j1 test-nvm2c-sanitizers` each pass 1,065 AOT
checks and 965 graph checks. The sanitizer driver verifies fresh instrumentation.
Tests construct a record array, append an element obtained from a later-defined
callee's nested array, and pass the array to the same consumer before and after
the insertion. A variant analyzes the length consumer before its callers.
An incompatible scalar element remains rejected. `git diff --check` passes.

Compiler acceptance clears function 170's array-kind conflict. It next exposed
function 252's length-induced parameter conflict, cleared by the `ARR_LEN` fix.
The current failure is function 253 offset 935: `resource_check_function` passes
a resource name inferred as `int64_t` to `resource_find`, which requires text.
Inspection shows record-array calls still omit element field facts and fall
back to integer defaults. Propagating record-array result fields is next;
compiler acceptance and the parent task remain incomplete.

MAC `task_9c850e94e5a74b6f8941622e2872af23` still cannot be claimed because
the hub returns `agent_status_unavailable`. I retain the evidence without
claiming a completed ledger transition.
