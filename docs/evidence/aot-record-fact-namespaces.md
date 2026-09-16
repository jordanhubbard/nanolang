# Independent record field-fact namespaces

My C emitter allocates independent indices for record temporaries (`r`) and
record-array temporaries (`ra`). They previously indexed one shared field-kind
table. Creating an integer record at the same index as a live string-record
array replaced the array's field-kind facts. Valid bytecode then failed C
translation with `STR_LEN: expected string value`.

I split those tables. Record-array creation, loading, duplication and branch
merges use array-element facts. Array append transfers record facts into the
array namespace; array get transfers them back. Snapshots own both tables and
restore them independently. Cleanup releases both on success and failure.

## Verification

Before the fix, all three new array-first cases failed translation: direct,
true branch and false branch. After the fix, `make -j1 test-nvm2c` passes
1,010 checks on Darwin. Six generated executables cover both collision
directions and all three paths, including duplication and branch restoration.
They preserve the string field and return its length, nine. Existing tests
for 300 temporaries and legacy array representations continue passing.
`git diff --check` passes.

`make -j1 test-one-ir-compiler` still fails at function 20 with
`AGG_PACK has too many fields`. This fixes a field-fact prerequisite, not the
75-field aggregate or array-valued-field implementation. Full compiler
acceptance and release remain unfinished.

MAC `task_3673443775f2477c94688b1021d6102a` tracks the fix. The hub refuses my
claim with `agent_status_unavailable`; repository verification does not imply
successful ledger closure.
