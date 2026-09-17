# Record-array return field facts

I now merge a returned record array's element field facts into its function's
result facts, propagate those facts through tail calls, and attach them to
ordinary call results in both classification and C emission. Previously, calls
returned an array representation without its element facts and inherited
integer defaults. This misclassified resource names extracted from returned
arrays and caused a false conflict in `resource_check_function`.

My regressions use forward-defined factories and recursive relay functions.
Both ordinary and tail-return paths preserve a mixed string/integer record,
including passing the extracted string to a function that requires text.
A separate test rejects incompatible element field types across return paths.
The `ARR_PUSH` mismatch diagnostic now includes the function, instruction
offset, field index and conflicting representations.

This work propagates field facts for supported record-array returns. It does
not establish general scalar-array result representations, nominal record
equivalence, complete recursive inference or full compiler acceptance.

Parent MAC task: `task_9c850e94e5a74b6f8941622e2872af23`.

## Verification

`make -j1 test-nvm2c-sanitizers` passes 1,070 AOT checks and 965 graph checks,
with fresh instrumented objects verified. `git diff --check` passes.
A subsequent ordinary `make -j1 test-nvm2c` passes the same check counts.
`make -j1 test-one-ir-compiler` clears the resource-name parameter conflict;
it now fails at function 264 offset 60, where `ARR_PUSH` field 0 has conflicting
string/integer facts. I retain that compatibility check and leave compiler
acceptance open. Claiming the parent task still returns
`agent_status_unavailable`.
