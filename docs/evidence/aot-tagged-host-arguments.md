# My tagged host-call boundary

I preserve tagged caller storage when a builtin host adapter consumes an
integer or string. My existing emitted consumer checks reject missing values,
booleans and wrong scalar tags before calling the adapter. I do not reinterpret
tagged storage as an untagged C argument.

I defer unknown parameter inference until caller facts are available. Otherwise
a wrapper visited before its caller can acquire an integer storage constraint
that conflicts with the caller's tagged value. My final classification still
checks the exact host signature. Invalid raw argument kinds remain rejected.

My regression matrix exercises `vm_file_exists` and `vm_string_from_char`,
valid, uninitialized, wrong-scalar and boolean values, and both function orders.
Successful wrappers inspect their original parameter after the host call.

I run `make -j1 test-nvm2c` and `make test-nvm2c-sanitizers` for normal and fresh
ASan/UBSan coverage: both pass 1,476 AOT and 994 shape checks. I also rerun
`make -j1 test-one-ir-compiler`: the focused
empty-array source fixture passes, but full compiler translation remains
incomplete. It now reaches function 599, `collect_files_dfs`, whose packed
result includes a hashmap field rejected by `AGG_PACK` classification.

This is boundary regression evidence, not full compiler or release acceptance.
