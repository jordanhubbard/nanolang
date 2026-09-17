# My map fields

I retain the full declared `HashMap<K,V>` annotation when checking a record
field. Inference can borrow that declaration before the field is checked; the
checked field owns a copy for later compilation. Native emission uses that
checked annotation to select the concrete map helper.

I also record map-call failures through structured diagnostics. My previous
plain stderr messages allowed standalone wrong-key `map_get` and `map_has`
expressions to publish bytecode despite the error. Invalid calls now stop
publication and preserve an existing output artifact.

My three field-operation tests execute all four supported scalar pairs, direct
and nested receivers, and an imported record through C-native and NanoVM. They
check insertion, retrieval, membership, length and removal. Eight invalid forms
run through both drivers and require a type diagnostic and preserved output.
The adjacent constructor/diagnostic and map-boundary suite passes eight methods;
the component build gate passes. Forty parser/checker/teardown iterations pass
ASan and UBSan; leak detection remains disabled for the separately tracked
legacy metadata leaks. Independent source review found no scoped blocker.

Logs: `/tmp/nanolang-map-fields-adjacent.log`,
`/tmp/nanolang-map-fields-bootstrap.log`, and
`/tmp/nanolang-map-fields-asan.log`.

This closes `task_160826784e8a4aa4ac9d5e589a54c814`. It does not establish complete
map ownership, arbitrary generic substitution, or bytecode bootstrap equality.
Native lexical map cleanup remains `task_1edadd5eb33a445d9bf6516744bc405e`.

The subsequent integrated `make bootstrap` check, including generated
`nanoc_stage1` and `nanoc_stage2`, also passes with the selected-array context
repair. Its log is `/tmp/nanolang-selected-array-full-bootstrap.log`. This is
stronger evidence than the earlier `make build` component check.
